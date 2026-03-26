use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

#[test]
#[ignore] // запускать вручную: cargo test -p mcp-image-gen --test integration -- --ignored
fn test_mcp_server_generate_image() {
    let binary = env!("CARGO_BIN_EXE_mcp-image-gen");
    let images_dir = tempfile::tempdir().expect("Failed to create temp dir for images");

    let mut child = Command::new(binary)
        .env("AITHERFLOW_SELECTED_MODEL", "FLUX.2-klein-4B")
        .env("AITHERFLOW_MODELS_PATH", "/mnt/SOURCE2/local-model")
        .env("AITHERFLOW_IMAGES_PATH", images_dir.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start mcp-image-gen");

    let stdin = child.stdin.as_mut().unwrap();
    let stdout = BufReader::new(child.stdout.take().unwrap());

    // Initialize
    writeln!(stdin, r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2024-11-05","capabilities":{{}},"clientInfo":{{"name":"test","version":"1.0"}}}}}}"#).unwrap();
    stdin.flush().unwrap();

    // Read init response
    let mut lines = stdout.lines();
    let init_resp = lines.next().unwrap().unwrap();
    assert!(
        init_resp.contains("mcp-image-gen"),
        "Init response: {init_resp}"
    );

    // Send initialized notification
    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#
    )
    .unwrap();
    stdin.flush().unwrap();

    // Send generate_image (small, 1 step)
    writeln!(stdin, r#"{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"generate_image","arguments":{{"prompt":"test cat","width":256,"height":256,"steps":1}}}}}}"#).unwrap();
    stdin.flush().unwrap();

    // Wait for response (up to 10 minutes — model loading is slow)
    let handle = std::thread::spawn(move || lines.next().map(|r| r.unwrap_or_default()));

    let result = handle
        .join()
        .unwrap();

    match result {
        Some(resp) => {
            // Truncate base64 for readable output
            let display_resp = if resp.len() > 500 {
                format!("{}... ({} bytes total)", &resp[..500], resp.len())
            } else {
                resp.clone()
            };
            println!("Generate response: {display_resp}");

            // Check for error
            if resp.contains("\"isError\":true") || resp.contains("\"isError\": true") {
                let output = child.wait_with_output().unwrap();
                let stderr_str = String::from_utf8_lossy(&output.stderr);
                panic!("Generation failed!\nResponse: {display_resp}\nStderr: {stderr_str}");
            }

            // Verify response contains image content block with base64 data
            assert!(
                resp.contains("\"type\":\"image\"") || resp.contains("\"type\": \"image\""),
                "Response should contain image content block: {display_resp}"
            );
            assert!(
                resp.contains("\"mimeType\":\"image/png\"") || resp.contains("\"mimeType\": \"image/png\""),
                "Response should contain image/png mimeType: {display_resp}"
            );
            assert!(
                resp.contains("\"data\":") || resp.contains("\"data\" :"),
                "Response should contain base64 data: {display_resp}"
            );

            // Verify text metadata block is also present (path is JSON-escaped inside text field)
            assert!(
                resp.contains("path"),
                "Response should contain path in text metadata: {display_resp}"
            );

            // Verify image file was actually created on disk
            let files: Vec<_> = std::fs::read_dir(images_dir.path())
                .expect("Failed to read images dir")
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map_or(false, |ext| ext == "png" || ext == "jpg" || ext == "jpeg")
                })
                .collect();
            assert!(
                !files.is_empty(),
                "No image files found in {}",
                images_dir.path().display()
            );
            println!(
                "Image created: {}",
                files[0].path().display()
            );
        }
        None => {
            let output = child.wait_with_output().unwrap();
            let stderr_str = String::from_utf8_lossy(&output.stderr);
            panic!("No response from server!\nStderr: {stderr_str}");
        }
    }

    // Cleanup: kill server if still running
    let _ = child.kill();
}
