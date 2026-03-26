use crate::config::Config;
use chrono::Utc;
use diffusion_rs::api::{gen_img, ConfigBuilder, ModelConfigBuilder};
use hf_hub::api::sync::ApiBuilder;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Component, Path, PathBuf};
use tracing::{error, info, warn};

/// Redirect diffusion-rs C++ log and progress output to stderr
/// (instead of stdout which would corrupt MCP JSON-RPC).
/// Uses the stable-diffusion.cpp callback API.
/// Call once at startup before any gen_img() calls.
pub fn suppress_diffusion_output() {
    unsafe extern "C" fn stderr_log(
        _level: diffusion_rs_sys::sd_log_level_t,
        text: *const std::os::raw::c_char,
        _data: *mut std::ffi::c_void,
    ) {
        if !text.is_null() {
            let msg = std::ffi::CStr::from_ptr(text).to_string_lossy();
            eprint!("[sdcpp] {msg}");
        }
    }
    unsafe extern "C" fn noop_progress(
        _step: std::os::raw::c_int,
        _steps: std::os::raw::c_int,
        _time: f32,
        _data: *mut std::ffi::c_void,
    ) {
    }

    unsafe {
        diffusion_rs_sys::sd_set_log_callback(Some(stderr_log), std::ptr::null_mut());
        diffusion_rs_sys::sd_set_progress_callback(Some(noop_progress), std::ptr::null_mut());
    }
}

/// Validate that a path is safe: no traversal components, must be absolute.
pub fn validate_path_safe(path: &Path) -> Result<(), String> {
    if !path.is_absolute() {
        return Err(format!("Path must be absolute: {}", path.display()));
    }
    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            return Err(format!(
                "Path traversal detected (..): {}",
                path.display()
            ));
        }
    }
    Ok(())
}

/// Download a file from HuggingFace Hub into models_path (used as cache dir).
fn download_hf_file(models_path: &Path, repo: &str, file: &str) -> Result<PathBuf, String> {
    let mut builder = ApiBuilder::new()
        .with_cache_dir(models_path.to_path_buf())
        .with_progress(false);

    // Try HF_TOKEN env first, then models_path/token file
    let token = std::env::var("HF_TOKEN")
        .ok()
        .filter(|t| !t.trim().is_empty())
        .or_else(|| {
            std::fs::read_to_string(models_path.join("token"))
                .ok()
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
        });

    if let Some(t) = token {
        builder = builder.with_token(Some(t));
    }

    let api = builder.build().map_err(|e| format!("Failed to build HF API: {e}"))?;
    let repo_api = api.model(repo.to_string());
    repo_api.get(file).map_err(|e| {
        let msg = format!("Failed to download {repo}/{file}: {e}");
        let err_lower = e.to_string().to_lowercase();
        if err_lower.contains("401")
            || err_lower.contains("403")
            || err_lower.contains("unauthorized")
            || err_lower.contains("forbidden")
        {
            format!(
                "{msg}. This model requires a HuggingFace token and license acceptance. \
                 Set HF_TOKEN env variable and accept the license at huggingface.co"
            )
        } else {
            msg
        }
    })
}

/// Repo+file pair for a HuggingFace component.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RepoFile {
    repo: String,
    file: String,
}

/// Model definition as stored in models.json.
/// Each entry fully describes how to download and configure a model.
// Keep in sync with src-tauri/src/image_gen.rs::ModelDefinition
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelDefinition {
    id: String,
    name: String,
    diffusion: RepoFile,
    #[serde(default)]
    vae: Option<RepoFile>,
    #[serde(default)]
    llm: Option<RepoFile>,
    #[serde(default)]
    clip_l: Option<RepoFile>,
    #[serde(default)]
    t5xxl: Option<RepoFile>,
    #[serde(default)]
    single_file: bool,
    #[serde(default = "default_steps")]
    steps: i32,
    #[serde(default = "default_cfg_scale")]
    cfg_scale: f32,
    #[serde(default = "default_size")]
    width: i32,
    #[serde(default = "default_size")]
    height: i32,
    #[serde(default)]
    offload_cpu: bool,
    #[serde(default)]
    flash_attn: bool,
    #[serde(default)]
    vae_tiling: bool,
    #[serde(default)]
    size_mb: u64,
    #[serde(default)]
    lora: Option<String>,
    #[serde(default = "default_lora_strength")]
    lora_strength: f32,
    #[serde(default = "default_true")]
    lora_enabled: bool,
}

fn default_steps() -> i32 { 4 }
fn default_cfg_scale() -> f32 { 1.0 }
fn default_size() -> i32 { 1024 }
fn default_lora_strength() -> f32 { 1.0 }
fn default_true() -> bool { true }

/// Components needed for a model: (repo, file) pairs for each role.
#[derive(Debug)]
struct ModelComponents {
    diffusion: (String, String),
    vae: Option<(String, String)>,
    llm: Option<(String, String)>,
    clip_l: Option<(String, String)>,
    t5xxl: Option<(String, String)>,
    single_file: bool,
    steps: i32,
    cfg_scale: f32,
    width: i32,
    height: i32,
    offload_cpu: bool,
    flash_attn: bool,
    vae_tiling: bool,
    lora: Option<String>,
    lora_strength: f32,
    lora_enabled: bool,
}

/// Default model used as fallback when models.json doesn't exist.
fn default_model_definition() -> ModelDefinition {
    ModelDefinition {
        id: "flux2-klein-4b".into(),
        name: "FLUX.2 Klein 4B".into(),
        diffusion: RepoFile {
            repo: "leejet/FLUX.2-klein-4B-GGUF".into(),
            file: "flux-2-klein-4b-Q8_0.gguf".into(),
        },
        vae: Some(RepoFile {
            repo: "black-forest-labs/FLUX.2-dev".into(),
            file: "vae/diffusion_pytorch_model.safetensors".into(),
        }),
        llm: Some(RepoFile {
            repo: "unsloth/Qwen3-4B-GGUF".into(),
            file: "Qwen3-4B-Q8_0.gguf".into(),
        }),
        clip_l: None,
        t5xxl: None,
        single_file: false,
        steps: 4,
        cfg_scale: 1.0,
        width: 1024,
        height: 1024,
        offload_cpu: true,
        flash_attn: true,
        vae_tiling: true,
        size_mb: 4403,
        lora: None,
        lora_strength: 1.0,
        lora_enabled: true,
    }
}

/// Path to the shared models.json config file.
/// Format: ~/.config/mcp-image-gen/models.json
fn models_json_path() -> Result<PathBuf, String> {
    let config_dir = dirs::config_dir()
        .ok_or("Cannot determine config directory (XDG_CONFIG_HOME)")?;
    Ok(config_dir.join("mcp-image-gen").join("models.json"))
}

/// Load model definitions from models.json.
/// Creates the file with the default model if it doesn't exist.
fn load_model_definitions() -> Result<Vec<ModelDefinition>, String> {
    let path = models_json_path()?;

    if path.exists() {
        let content = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
        let models: Vec<ModelDefinition> = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;
        info!("Loaded {} model(s) from {}", models.len(), path.display());
        Ok(models)
    } else {
        info!("No models.json found, creating default at {}", path.display());
        let models = vec![default_model_definition()];
        if let Err(e) = save_model_definitions(&path, &models) {
            warn!("Failed to save default models.json: {e}");
        }
        Ok(models)
    }
}

/// Write model definitions to models.json atomically.
fn save_model_definitions(path: &Path, models: &[ModelDefinition]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create dir {}: {e}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(models)
        .map_err(|e| format!("Failed to serialize models: {e}"))?;
    // Atomic write: temp file + rename
    let parent = path.parent().ok_or("models.json path has no parent")?;
    let tmp_path = parent.join(format!(".tmp_models_{}", std::process::id()));
    fs::write(&tmp_path, json.as_bytes())
        .map_err(|e| format!("Failed to write temp file: {e}"))?;
    fs::rename(&tmp_path, path).inspect_err(|_| {
        let _ = fs::remove_file(&tmp_path);
    }).map_err(|e| format!("Failed to rename temp to {}: {e}", path.display()))
}

impl ModelDefinition {
    fn to_components(&self) -> ModelComponents {
        let rf_to_tuple = |rf: &RepoFile| (rf.repo.clone(), rf.file.clone());
        ModelComponents {
            diffusion: rf_to_tuple(&self.diffusion),
            vae: self.vae.as_ref().map(rf_to_tuple),
            llm: self.llm.as_ref().map(rf_to_tuple),
            clip_l: self.clip_l.as_ref().map(rf_to_tuple),
            t5xxl: self.t5xxl.as_ref().map(rf_to_tuple),
            single_file: self.single_file,
            steps: self.steps,
            cfg_scale: self.cfg_scale,
            width: self.width,
            height: self.height,
            offload_cpu: self.offload_cpu,
            flash_attn: self.flash_attn,
            vae_tiling: self.vae_tiling,
            lora: self.lora.clone(),
            lora_strength: self.lora_strength,
            lora_enabled: self.lora_enabled,
        }
    }
}

fn get_model_components(model_id: &str) -> Result<ModelComponents, String> {
    let models = load_model_definitions()?;
    let id_lower = model_id.to_lowercase().replace('.', "");

    // Search by id (case-insensitive, dots stripped)
    if let Some(def) = models.iter().find(|m| {
        let m_id = m.id.to_lowercase().replace('.', "");
        m_id == id_lower || m.id == model_id
    }) {
        return Ok(def.to_components());
    }

    // Fallback: try the hardcoded default if it matches
    let default = default_model_definition();
    let default_id = default.id.to_lowercase().replace('.', "");
    if default_id == id_lower {
        return Ok(default.to_components());
    }

    let available: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    Err(format!(
        "Unknown model: {model_id}. Available: {}",
        available.join(", ")
    ))
}

/// Build ConfigBuilder + ModelConfigBuilder from components, downloading files as needed.
fn build_model_config(
    components: &ModelComponents,
    models_path: &Path,
    diffusion_override: Option<PathBuf>,
) -> Result<(ConfigBuilder, ModelConfigBuilder), String> {
    let mut config = ConfigBuilder::default();
    let mut model_config = ModelConfigBuilder::default();

    // Download/resolve diffusion model
    let diffusion_path = match diffusion_override {
        Some(p) => p,
        None => download_hf_file(models_path, &components.diffusion.0, &components.diffusion.1)?,
    };

    if components.single_file {
        model_config.model(diffusion_path);
    } else {
        model_config.diffusion_model(diffusion_path);
    }

    // VAE
    if let Some((ref repo, ref file)) = components.vae {
        let vae_path = download_hf_file(models_path, repo, file)?;
        model_config.vae(vae_path);
    }

    // LLM text encoder
    if let Some((ref repo, ref file)) = components.llm {
        let llm_path = download_hf_file(models_path, repo, file)?;
        model_config.llm(llm_path);
    }

    // CLIP-L text encoder
    if let Some((ref repo, ref file)) = components.clip_l {
        let clip_l_path = download_hf_file(models_path, repo, file)?;
        model_config.clip_l(clip_l_path);
    }

    // T5-XXL text encoder
    if let Some((ref repo, ref file)) = components.t5xxl {
        let t5_path = download_hf_file(models_path, repo, file)?;
        model_config.t5xxl(t5_path);
    }

    // Model-specific flags
    if components.offload_cpu {
        model_config.offload_params_to_cpu(true);
    }
    if components.flash_attn {
        model_config.flash_attention(true);
    }
    if components.vae_tiling {
        model_config.vae_tiling(true);
    }

    // LoRA adapter (only if enabled)
    if components.lora_enabled {
        if let Some(ref lora_path) = components.lora {
            let path = PathBuf::from(lora_path);
            if path.exists() {
                if let Some(parent) = path.parent() {
                    let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
                    let spec = diffusion_rs::api::LoraSpec {
                        file_name: stem,
                        multiplier: components.lora_strength,
                        is_high_noise: false,
                    };
                    model_config.lora_models(parent, vec![spec]);
                }
            } else {
                warn!("LoRA file not found: {lora_path}");
            }
        }
    }

    config
        .cfg_scale(components.cfg_scale)
        .steps(components.steps)
        .width(components.width)
        .height(components.height);

    Ok((config, model_config))
}

pub fn generate_image(params: &Value, config: &Config) -> Result<String, String> {
    let args = params.get("arguments").ok_or("Missing arguments")?;

    let prompt = args
        .get("prompt")
        .and_then(|v| v.as_str())
        .ok_or("Missing required parameter: prompt")?
        .to_string();

    let negative_prompt = args
        .get("negative_prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let width = args
        .get("width")
        .and_then(|v| v.as_i64())
        .and_then(|v| i32::try_from(v).ok())
        .unwrap_or(config.width);

    let height = args
        .get("height")
        .and_then(|v| v.as_i64())
        .and_then(|v| i32::try_from(v).ok())
        .unwrap_or(config.height);

    let steps = args
        .get("steps")
        .and_then(|v| v.as_i64())
        .and_then(|v| i32::try_from(v).ok())
        .unwrap_or(config.steps);

    let seed = args
        .get("seed")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1);

    // Validate output path
    validate_path_safe(&config.images_path)?;

    // Generate output filename: timestamp + short hash of prompt
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_bytes());
    let hash = format!("{:x}", hasher.finalize());
    let short_hash = &hash[..8];
    let filename = format!("{timestamp}_{short_hash}.png");
    let output_path = config.images_path.join(&filename);

    // Validate the full output path too
    validate_path_safe(&output_path)?;

    info!(
        prompt = %prompt,
        width = width,
        height = height,
        steps = steps,
        seed = seed,
        output = %output_path.display(),
        "Generating image"
    );

    // Resolve model: either a known id or a file path
    let (components, diffusion_override) = resolve_model(&config.selected_model)?;

    // Build configs with direct paths (no PresetBuilder)
    let (mut img_config_builder, model_config_builder) =
        build_model_config(&components, &config.models_path, diffusion_override)?;

    // Apply user overrides
    let out = output_path.clone();
    let neg = negative_prompt.clone();
    img_config_builder
        .width(width)
        .height(height)
        .steps(steps)
        .seed(seed)
        .output(out)
        .prompt(prompt.clone());
    if !neg.is_empty() {
        img_config_builder.negative_prompt(neg);
    }

    let img_config = img_config_builder
        .build()
        .map_err(|e| format!("Failed to build image config: {e}"))?;
    let mut model_config = model_config_builder
        .build()
        .map_err(|e| format!("Failed to build model config: {e}"))?;

    gen_img(&img_config, &mut model_config).map_err(|e| {
        error!("Image generation failed: {e}");
        format!("Image generation failed: {e}")
    })?;

    info!("Image saved to {}", output_path.display());

    let result = serde_json::json!({
        "path": output_path.to_string_lossy(),
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "prompt": prompt
    });

    Ok(result.to_string())
}

pub fn list_models(config: &Config) -> Result<String, String> {
    let models_dir = &config.models_path;

    if !models_dir.exists() {
        return Ok(
            "No models directory found. Models will be downloaded automatically on first use."
                .into(),
        );
    }

    let entries = fs::read_dir(models_dir)
        .map_err(|e| format!("Failed to read models directory: {e}"))?;

    let mut models = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
        let ft = entry
            .file_type()
            .map_err(|e| format!("Failed to get file type: {e}"))?;

        if ft.is_dir() || ft.is_file() {
            let name = entry.file_name().to_string_lossy().to_string();
            let kind = if ft.is_dir() { "directory" } else { "file" };
            models.push(format!("  - {} ({})", name, kind));
        }
    }

    if models.is_empty() {
        return Ok(
            "No models found. Models will be downloaded automatically on first use.".into(),
        );
    }

    let mut result = format!(
        "Models directory: {}\n\nAvailable models:\n",
        models_dir.display()
    );
    result.push_str(&models.join("\n"));
    result.push_str(&format!("\n\nDefault model: {}", config.selected_model));

    Ok(result)
}

pub fn download_model(params: &Value, config: &Config) -> Result<String, String> {
    let args = params.get("arguments").ok_or("Missing arguments")?;

    let model_id = args
        .get("model_id")
        .and_then(|v| v.as_str())
        .ok_or("Missing required parameter: model_id")?;

    info!(model_id = model_id, "Downloading model");

    let components = get_model_components(model_id)?;

    // Download all components
    download_hf_file(&config.models_path, &components.diffusion.0, &components.diffusion.1)?;
    if let Some((ref repo, ref file)) = components.vae {
        download_hf_file(&config.models_path, repo, file)?;
    }
    if let Some((ref repo, ref file)) = components.llm {
        download_hf_file(&config.models_path, repo, file)?;
    }
    if let Some((ref repo, ref file)) = components.clip_l {
        download_hf_file(&config.models_path, repo, file)?;
    }
    if let Some((ref repo, ref file)) = components.t5xxl {
        download_hf_file(&config.models_path, repo, file)?;
    }

    info!(model_id = model_id, "Model downloaded successfully");

    let result = serde_json::json!({
        "status": "ok",
        "model": model_id,
        "path": config.models_path.to_string_lossy()
    });

    Ok(result.to_string())
}

/// Check if selected_model is a file path or a model id.
/// Returns (ModelComponents, Option<PathBuf>) — components for config scaffolding,
/// and optional path override for diffusion_model.
fn resolve_model(selected: &str) -> Result<(ModelComponents, Option<PathBuf>), String> {
    // If it looks like a file path — extract components from filename
    if selected.contains('/') || selected.ends_with(".gguf") || selected.ends_with(".safetensors") {
        let path = PathBuf::from(selected);
        if !path.exists() {
            return Err(format!("Model file not found: {selected}"));
        }
        let filename = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();
        let model_id = model_id_from_filename(&filename);
        let components = get_model_components(model_id)?;
        info!(path = selected, "Using model file directly");
        Ok((components, Some(path)))
    } else {
        let components = get_model_components(selected)?;
        Ok((components, None))
    }
}

/// Match a filename to the closest model id (for config scaffolding: vae, clip, steps, etc.)
fn model_id_from_filename(filename: &str) -> &'static str {
    if filename.contains("klein") && filename.contains("4b") {
        "flux2-klein-4b"
    } else if filename.contains("klein") && filename.contains("9b") {
        "flux2-klein-9b"
    } else if filename.contains("flux") && filename.contains("2") && filename.contains("dev") {
        "flux2-dev"
    } else if filename.contains("flux") && filename.contains("schnell") {
        "flux1-schnell"
    } else if filename.contains("flux") && filename.contains("mini") {
        "flux1-mini"
    } else if filename.contains("flux") && filename.contains("1") && filename.contains("dev") {
        "flux1-dev"
    } else if filename.contains("sdxl") || filename.contains("sd_xl")
        || (filename.contains("sd") && filename.contains("turbo"))
    {
        "sdxl-turbo"
    } else {
        warn!("Cannot determine model type from filename '{filename}', defaulting to FLUX.2 Klein 4B");
        "flux2-klein-4b"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ── default_model_definition ──

    #[test]
    fn test_default_model_definition() {
        let def = default_model_definition();
        assert_eq!(def.id, "flux2-klein-4b");
        assert_eq!(def.name, "FLUX.2 Klein 4B");
        assert_eq!(def.steps, 4);
        assert!(def.llm.is_some());
        assert!(def.vae.is_some());
        assert!(def.clip_l.is_none());
        assert!(def.t5xxl.is_none());
        assert!(!def.single_file);
        assert!(def.offload_cpu);
        assert!(def.flash_attn);
        assert!(def.vae_tiling);
    }

    #[test]
    fn test_default_model_vae_is_flux2() {
        let def = default_model_definition();
        let vae = def.vae.unwrap();
        assert!(vae.repo.contains("FLUX.2"), "Default model should use FLUX.2 VAE");
        assert!(vae.file.contains("diffusion_pytorch_model"));
    }

    // ── ModelDefinition serde ──

    #[test]
    fn test_model_definition_serde_roundtrip() {
        let def = default_model_definition();
        let json = serde_json::to_string_pretty(&def).unwrap();
        let restored: ModelDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, def.id);
        assert_eq!(restored.steps, def.steps);
        assert_eq!(restored.diffusion.repo, def.diffusion.repo);
    }

    #[test]
    fn test_model_definitions_json_array() {
        let models = vec![default_model_definition()];
        let json = serde_json::to_string_pretty(&models).unwrap();
        let restored: Vec<ModelDefinition> = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].id, "flux2-klein-4b");
    }

    // ── save/load models.json ──

    #[test]
    fn test_save_and_load_models_json() {
        let dir = std::env::temp_dir().join("mcp-image-gen-test-json");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("models.json");

        let models = vec![default_model_definition()];
        save_model_definitions(&path, &models).unwrap();
        assert!(path.exists());

        let content = fs::read_to_string(&path).unwrap();
        let loaded: Vec<ModelDefinition> = serde_json::from_str(&content).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "flux2-klein-4b");

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn test_load_custom_models_from_json() {
        let dir = std::env::temp_dir().join("mcp-image-gen-test-custom");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("models.json");

        let custom = ModelDefinition {
            id: "my-custom-model".into(),
            name: "My Custom Model".into(),
            diffusion: RepoFile { repo: "org/repo".into(), file: "model.gguf".into() },
            vae: None,
            llm: None,
            clip_l: None,
            t5xxl: None,
            single_file: true,
            steps: 10,
            cfg_scale: 2.0,
            width: 512,
            height: 512,
            offload_cpu: false,
            flash_attn: false,
            vae_tiling: false,
            size_mb: 1000,
            lora: None,
            lora_strength: 1.0,
            lora_enabled: true,
        };
        let models = vec![default_model_definition(), custom];
        save_model_definitions(&path, &models).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        let loaded: Vec<ModelDefinition> = serde_json::from_str(&content).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[1].id, "my-custom-model");
        assert!(loaded[1].single_file);
        assert_eq!(loaded[1].steps, 10);

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }

    // ── to_components ──

    #[test]
    fn test_model_definition_to_components() {
        let def = default_model_definition();
        let comp = def.to_components();
        assert_eq!(comp.diffusion.0, "leejet/FLUX.2-klein-4B-GGUF");
        assert_eq!(comp.diffusion.1, "flux-2-klein-4b-Q8_0.gguf");
        assert_eq!(comp.steps, 4);
        assert!(comp.llm.is_some());
        assert!(comp.vae.is_some());
        assert!(comp.offload_cpu);
    }

    // ── get_model_components (reads from JSON or fallback) ──

    #[test]
    fn test_get_model_components_default() {
        // The default model should always resolve (from JSON or fallback)
        let result = get_model_components("flux2-klein-4b");
        assert!(result.is_ok(), "Default model should always resolve: {:?}", result.err());
        let c = result.unwrap();
        assert_eq!(c.steps, 4);
        assert!(c.llm.is_some());
    }

    #[test]
    fn test_get_model_components_unknown() {
        let result = get_model_components("nonexistent-model");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown model"));
    }

    // ── model_id_from_filename ──

    #[test]
    fn test_model_id_from_filename_klein_4b() {
        assert_eq!(model_id_from_filename("flux-2-klein-4b-Q8_0.gguf"), "flux2-klein-4b");
    }

    #[test]
    fn test_model_id_from_filename_klein_9b() {
        assert_eq!(model_id_from_filename("flux-2-klein-9b-Q8_0.gguf"), "flux2-klein-9b");
    }

    #[test]
    fn test_model_id_from_filename_flux2_dev() {
        assert_eq!(model_id_from_filename("flux2-dev-q2_k.gguf"), "flux2-dev");
    }

    #[test]
    fn test_model_id_from_filename_schnell() {
        assert_eq!(model_id_from_filename("flux1-schnell-q8_0.gguf"), "flux1-schnell");
    }

    #[test]
    fn test_model_id_from_filename_mini() {
        assert_eq!(model_id_from_filename("flux.1-mini-q8_0.gguf"), "flux1-mini");
    }

    #[test]
    fn test_model_id_from_filename_flux1_dev() {
        assert_eq!(model_id_from_filename("flux1-dev-q8_0.gguf"), "flux1-dev");
    }

    #[test]
    fn test_model_id_from_filename_sdxl() {
        assert_eq!(model_id_from_filename("sd_xl_turbo_1.0_fp16.safetensors"), "sdxl-turbo");
        assert_eq!(model_id_from_filename("sdxl-base.safetensors"), "sdxl-turbo");
    }

    #[test]
    fn test_model_id_from_filename_fallback() {
        assert_eq!(model_id_from_filename("totally-unknown-model.gguf"), "flux2-klein-4b");
    }

    // ── resolve_model ──

    #[test]
    fn test_resolve_model_by_id() {
        let (components, path) = resolve_model("flux2-klein-4b").unwrap();
        assert!(path.is_none());
        assert_eq!(components.steps, 4);
        assert!(components.llm.is_some());
    }

    #[test]
    fn test_resolve_model_by_path() {
        let dir = std::env::temp_dir().join("mcp-image-gen-test");
        if let Err(e) = fs::create_dir_all(&dir) {
            warn!("Failed to create test dir: {e}");
        }
        let file = dir.join("flux-2-klein-4b-Q8_0.gguf");
        let mut f = std::fs::File::create(&file).unwrap();
        f.write_all(b"fake").unwrap();

        let (components, model_path) = resolve_model(file.to_str().unwrap()).unwrap();
        assert!(model_path.is_some());
        assert_eq!(model_path.unwrap(), file);
        assert!(components.llm.is_some());

        if let Err(e) = fs::remove_file(&file) {
            warn!("Failed to clean up test file: {e}");
        }
        if let Err(e) = fs::remove_dir(&dir) {
            warn!("Failed to clean up test dir: {e}");
        }
    }

    #[test]
    fn test_resolve_model_missing_file() {
        let result = resolve_model("/nonexistent/path/to/model.gguf");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    // ── validate_path_safe ──

    #[test]
    fn test_validate_path_safe_absolute() {
        assert!(validate_path_safe(Path::new("/home/user/models")).is_ok());
        assert!(validate_path_safe(Path::new("/tmp")).is_ok());
    }

    #[test]
    fn test_validate_path_safe_relative() {
        assert!(validate_path_safe(Path::new("relative/path")).is_err());
        assert!(validate_path_safe(Path::new("./here")).is_err());
    }

    #[test]
    fn test_validate_path_safe_traversal() {
        assert!(validate_path_safe(Path::new("/home/../etc/passwd")).is_err());
        assert!(validate_path_safe(Path::new("/tmp/models/../../etc")).is_err());
    }

    // ── set_hf_token ──

    #[test]
    fn test_hf_token_api_accessible() {
        diffusion_rs::util::set_hf_token("");
        diffusion_rs::util::set_hf_token("test-token");
    }

    // ── config ──

    #[test]
    fn test_config_loads_models_path() {
        let config = Config::new().unwrap();
        assert!(!config.models_path.as_os_str().is_empty());
        assert!(!config.images_path.as_os_str().is_empty());
        assert!(config.width > 0);
        assert!(config.height > 0);
        assert!(config.steps > 0);
    }
}
