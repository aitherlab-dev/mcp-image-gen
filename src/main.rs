mod config;
mod mcp;
mod tools;

use config::Config;
use mcp::{JsonRpcRequest, JsonRpcResponse};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use tracing::{error, info, warn};

enum Event {
    StdinLine(String),
    ToolResult(JsonRpcResponse),
    StdinClosed,
}

/// Ensure HF token is available under HF_HOME.
/// Priority: HF_TOKEN env → copy ~/.cache/huggingface/token → models_path/token
fn ensure_hf_token(models_path: &Path) {
    let dest = models_path.join("token");

    // 1. If HF_TOKEN env is set, write it to models_path/token
    if let Ok(token) = std::env::var("HF_TOKEN") {
        let token = token.trim().to_string();
        if !token.is_empty() {
            if !dest.exists() {
                if let Err(e) = std::fs::write(&dest, &token) {
                    warn!("Failed to write HF token to {}: {e}", dest.display());
                }
            }
            return;
        }
    }

    // 2. If models_path/token already exists, nothing to do
    if dest.exists() {
        return;
    }

    // 3. Copy from default HF cache location
    if let Some(home) = dirs::home_dir() {
        let src = home.join(".cache/huggingface/token");
        if src.exists() {
            if let Ok(token) = std::fs::read_to_string(&src) {
                let token = token.trim();
                if !token.is_empty() {
                    if let Err(e) = std::fs::write(&dest, token) {
                        warn!("Failed to write HF token to {}: {e}", dest.display());
                    } else {
                        info!("HF token copied to {}", dest.display());
                    }
                }
            }
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mcp_image_gen=info".parse().unwrap()),
        )
        .init();

    info!("mcp-image-gen server starting");

    let config = match Config::load() {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to load config: {e}");
            std::process::exit(1);
        }
    };
    // Validate paths before creating directories
    if let Err(e) = tools::validate_path_safe(&config.models_path) {
        error!("Invalid models_path: {e}");
        std::process::exit(1);
    }
    if let Err(e) = tools::validate_path_safe(&config.images_path) {
        error!("Invalid images_path: {e}");
        std::process::exit(1);
    }
    config.ensure_dirs();
    // SAFETY: called before spawning any threads
    unsafe { std::env::set_var("HF_HOME", &config.models_path) };

    // Copy HF token into models_path/token so hf-hub finds it under HF_HOME
    ensure_hf_token(&config.models_path);

    // Pass token to diffusion-rs internal API (it ignores HF_HOME/token and HF_TOKEN env)
    if let Ok(token) = std::env::var("HF_TOKEN") {
        let token = token.trim().to_string();
        if !token.is_empty() {
            diffusion_rs::util::set_hf_token(&token);
            info!("HF token set for diffusion-rs from HF_TOKEN env");
        }
    } else if let Ok(token) = std::fs::read_to_string(config.models_path.join("token")) {
        let token = token.trim().to_string();
        if !token.is_empty() {
            diffusion_rs::util::set_hf_token(&token);
            info!("HF token set for diffusion-rs from models_path/token");
        }
    }

    // Suppress diffusion-rs C++ log/progress output before any gen_img() calls
    tools::suppress_diffusion_output();

    info!(
        models_path = %config.models_path.display(),
        images_path = %config.images_path.display(),
        selected_model = %config.selected_model,
        "Config loaded"
    );

    let (event_tx, event_rx) = mpsc::channel::<Event>();

    // Stdin reader thread
    let stdin_tx = event_tx.clone();
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) => {
                    let l = l.trim().to_string();
                    if !l.is_empty() && stdin_tx.send(Event::StdinLine(l)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    error!("Failed to read stdin: {e}");
                    break;
                }
            }
        }
        let _ = stdin_tx.send(Event::StdinClosed);
    });

    let mut stdout = io::stdout();
    let mut pending_tools: usize = 0;
    let mut stdin_closed = false;

    for event in &event_rx {
        match event {
            Event::StdinClosed => {
                stdin_closed = true;
                if pending_tools == 0 {
                    break;
                }
                info!("Stdin closed, waiting for {pending_tools} pending tool call(s)");
            }
            Event::ToolResult(response) => {
                pending_tools = pending_tools.saturating_sub(1);
                write_response(&mut stdout, &response);
                if stdin_closed && pending_tools == 0 {
                    break;
                }
            }
            Event::StdinLine(line) => {
                let request: JsonRpcRequest = match serde_json::from_str(&line) {
                    Ok(r) => r,
                    Err(e) => {
                        error!("Failed to parse request: {e}");
                        let resp = JsonRpcResponse::error(
                            Value::Null,
                            -32700,
                            format!("Parse error: {e}"),
                        );
                        write_response(&mut stdout, &resp);
                        continue;
                    }
                };

                // Notifications have no id — don't respond
                let id = match request.id {
                    Some(id) => id,
                    None => {
                        if request.method == "notifications/initialized" {
                            info!("Client initialized notification received");
                        }
                        continue;
                    }
                };

                let response = match request.method.as_str() {
                    "initialize" => handle_initialize(id),
                    "tools/list" => handle_tools_list(id),
                    "tools/call" => {
                        pending_tools += 1;
                        handle_tools_call(id, &request.params, &config, &event_tx);
                        continue;
                    }
                    method => {
                        info!("Unknown method: {method}");
                        JsonRpcResponse::error(
                            id,
                            -32601,
                            format!("Method not found: {method}"),
                        )
                    }
                };

                write_response(&mut stdout, &response);
            }
        }
    }

    info!("mcp-image-gen server shutting down");
}

fn panic_message(err: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = err.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

fn write_response(stdout: &mut io::Stdout, response: &JsonRpcResponse) {
    match serde_json::to_string(response) {
        Ok(json) => {
            if let Err(e) = writeln!(stdout, "{json}") {
                error!("Failed to write response: {e}");
            }
            if let Err(e) = stdout.flush() {
                error!("Failed to flush stdout: {e}");
            }
        }
        Err(e) => {
            error!("Failed to serialize response: {e}");
        }
    }
}

fn handle_initialize(id: Value) -> JsonRpcResponse {
    info!("Client connected (initialize)");

    JsonRpcResponse::success(
        id,
        json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "mcp-image-gen",
                "version": env!("CARGO_PKG_VERSION")
            }
        }),
    )
}

fn handle_tools_list(id: Value) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        json!({
            "tools": [
                {
                    "name": "generate_image",
                    "description": "Generate an image from a text prompt using a local Stable Diffusion / FLUX model. Returns the file path of the generated image.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Text description of the image to generate"
                            },
                            "negative_prompt": {
                                "type": "string",
                                "description": "What to avoid in the generated image"
                            },
                            "width": {
                                "type": "integer",
                                "description": "Image width in pixels (default from config)"
                            },
                            "height": {
                                "type": "integer",
                                "description": "Image height in pixels (default from config)"
                            },
                            "steps": {
                                "type": "integer",
                                "description": "Number of diffusion steps (default from config)"
                            },
                            "seed": {
                                "type": "integer",
                                "description": "Random seed (-1 for random)"
                            }
                        },
                        "required": ["prompt"]
                    }
                },
                {
                    "name": "download_model",
                    "description": "Download a model from HuggingFace. The model will be stored in the configured models directory.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "model_id": {
                                "type": "string",
                                "description": "Model identifier (e.g. FLUX.2-klein-4B, FLUX.1-schnell, SDXL-turbo)"
                            }
                        },
                        "required": ["model_id"]
                    }
                },
                {
                    "name": "list_models",
                    "description": "List available (downloaded) image generation models",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }),
    )
}

fn handle_tools_call(
    id: Value,
    params: &Value,
    config: &Config,
    event_tx: &mpsc::Sender<Event>,
) {
    let tool_name = params
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("");

    match tool_name {
        "generate_image" => {
            let params = params.clone();
            let config = config.clone();
            let tx = event_tx.clone();
            thread::spawn(move || {
                let response = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    tools::generate_image(&params, &config)
                })) {
                    Ok(Ok(msg)) => JsonRpcResponse::tool_result(id, msg),
                    Ok(Err(e)) => JsonRpcResponse::tool_error(id, e),
                    Err(panic_err) => {
                        let msg = panic_message(&panic_err);
                        JsonRpcResponse::tool_error(id, format!("generate_image panicked: {msg}"))
                    }
                };
                let _ = tx.send(Event::ToolResult(response));
            });
        }
        "download_model" => {
            let params = params.clone();
            let config = config.clone();
            let tx = event_tx.clone();
            thread::spawn(move || {
                let response = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    tools::download_model(&params, &config)
                })) {
                    Ok(Ok(msg)) => JsonRpcResponse::tool_result(id, msg),
                    Ok(Err(e)) => JsonRpcResponse::tool_error(id, e),
                    Err(panic_err) => {
                        let msg = panic_message(&panic_err);
                        JsonRpcResponse::tool_error(id, format!("download_model panicked: {msg}"))
                    }
                };
                let _ = tx.send(Event::ToolResult(response));
            });
        }
        "list_models" => {
            let response = match tools::list_models(config) {
                Ok(msg) => JsonRpcResponse::tool_result(id, msg),
                Err(e) => JsonRpcResponse::tool_error(id, e),
            };
            let _ = event_tx.send(Event::ToolResult(response));
        }
        _ => {
            let response = JsonRpcResponse::error(
                id,
                -32602,
                format!("Unknown tool: {tool_name}"),
            );
            let _ = event_tx.send(Event::ToolResult(response));
        }
    }
}
