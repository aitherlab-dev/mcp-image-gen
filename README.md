# mcp-image-gen

MCP server for **local image generation** using [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) via [diffusion-rs](https://github.com/newfla/diffusion-rs). Runs entirely on your machine — no API keys, no cloud, no data leaves your computer.

Works with any MCP client: **Claude Code CLI**, Claude Desktop, or any other MCP-compatible tool.

## What it does

Gives your AI assistant three tools:

| Tool | Description |
|------|-------------|
| `generate_image` | Generate an image from a text prompt. Returns the file path |
| `download_model` | Download a model from HuggingFace |
| `list_models` | List available (downloaded) models |

Supported architectures: **FLUX.2**, **FLUX.1**, **SDXL**, and compatible models in GGUF/safetensors format.

## Features

- **LoRA support** — apply `.safetensors` LoRA adapters with configurable strength
- **Multiple models** — switch between models via `models.json` config
- **Auto-download** — models are downloaded from HuggingFace on first use
- **CUDA acceleration** — optional, via feature flag
- **HuggingFace token** — supports gated models (FLUX.2-dev, etc.)
- **CPU offloading** — run large models on limited VRAM
- **Flash attention & VAE tiling** — per-model optimization flags

## Quick start

### 1. Build

```bash
git clone https://github.com/aitherlab-dev/mcp-image-gen.git
cd mcp-image-gen
cargo build --release
```

With CUDA support:

```bash
cargo build --release --features cuda
```

The binary will be at `target/release/mcp-image-gen`.

Optionally, copy it somewhere on your `$PATH`:

```bash
cp target/release/mcp-image-gen ~/.local/bin/
```

### 2. Add to Claude Code

Edit `~/.claude.json` (or create it) and add the server to `mcpServers`:

```json
{
  "mcpServers": {
    "image-gen": {
      "command": "/home/you/.local/bin/mcp-image-gen",
      "env": {}
    }
  }
}
```

That's it. The server will use default paths and auto-download the default model on first use.

#### Custom paths (optional)

If you want to control where models and images are stored:

```json
{
  "mcpServers": {
    "image-gen": {
      "command": "/home/you/.local/bin/mcp-image-gen",
      "env": {
        "MCP_IMAGE_GEN_MODELS_PATH": "/home/you/ai-models",
        "MCP_IMAGE_GEN_IMAGES_PATH": "/home/you/ai-images",
        "HF_TOKEN": "hf_your_token_here"
      }
    }
  }
}
```

#### Claude Desktop

Same config format — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "image-gen": {
      "command": "/home/you/.local/bin/mcp-image-gen",
      "env": {}
    }
  }
}
```

### 3. Use it

Start Claude Code and ask it to generate an image:

```
> Generate an image of a sunset over mountains
```

On first run, the model will be downloaded automatically (~4.4 GB for the default FLUX.2 Klein 4B). Subsequent runs use the cached model.

The server returns the file path of the generated image. In Claude Code CLI you'll see something like:

```
Image saved: /home/you/.local/share/mcp-image-gen/images/20250326_142030_abc12345.png
```

## Configuration

### Settings file

```
~/.config/mcp-image-gen/settings.json
```

```json
{
  "modelsPath": "/home/you/.local/share/mcp-image-gen/models",
  "imagesPath": "/home/you/.local/share/mcp-image-gen/images",
  "selectedModel": "flux2-klein-4b",
  "width": 1024,
  "height": 1024,
  "steps": 20
}
```

Created automatically with defaults on first run. You can edit it manually or let the server manage it.

### Environment variables

Override any config value via environment:

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_IMAGE_GEN_MODELS_PATH` | Where to store/cache downloaded models | `~/.local/share/mcp-image-gen/models` |
| `MCP_IMAGE_GEN_IMAGES_PATH` | Where generated images are saved | `~/.local/share/mcp-image-gen/images` |
| `MCP_IMAGE_GEN_SELECTED_MODEL` | Default model ID | `flux2-klein-4b` |
| `HF_TOKEN` | HuggingFace API token for gated models | — |
| `RUST_LOG` | Log level (e.g. `mcp_image_gen=debug`) | `mcp_image_gen=info` |

### Models config

Models are defined in `~/.config/mcp-image-gen/models.json`. Created automatically with a default model on first run.

Each model entry specifies HuggingFace repos/files for all components:

```json
[
  {
    "id": "flux2-klein-4b",
    "name": "FLUX.2 Klein 4B",
    "diffusion": { "repo": "leejet/FLUX.2-klein-4B-GGUF", "file": "flux-2-klein-4b-Q8_0.gguf" },
    "vae": { "repo": "black-forest-labs/FLUX.2-dev", "file": "vae/diffusion_pytorch_model.safetensors" },
    "llm": { "repo": "unsloth/Qwen3-4B-GGUF", "file": "Qwen3-4B-Q8_0.gguf" },
    "steps": 4,
    "cfg_scale": 1.0,
    "width": 1024,
    "height": 1024,
    "offload_cpu": true,
    "flash_attn": true,
    "vae_tiling": true
  }
]
```

### Adding custom models

Add entries to `models.json` with the HuggingFace repo and filename for each component:

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | Unique identifier |
| `name` | yes | Display name |
| `diffusion` | yes | Main model file (GGUF or safetensors) — `{"repo": "...", "file": "..."}` |
| `vae` | no | VAE decoder |
| `llm` | no | Text encoder (FLUX models) |
| `clip_l` | no | CLIP-L text encoder (SDXL) |
| `t5xxl` | no | T5-XXL encoder (FLUX) |
| `single_file` | no | `true` for single-file models (SDXL safetensors) |
| `steps` | no | Default inference steps (default: 4) |
| `cfg_scale` | no | Classifier-free guidance scale (default: 1.0) |
| `width` / `height` | no | Default resolution (default: 1024) |
| `offload_cpu` | no | Offload model params to CPU (saves VRAM) |
| `flash_attn` | no | Enable flash attention |
| `vae_tiling` | no | Enable VAE tiling (saves VRAM for large images) |

### LoRA adapters

Add LoRA fields to any model entry in `models.json`:

```json
{
  "id": "flux2-klein-4b",
  "lora": "/home/you/loras/anime-style.safetensors",
  "lora_strength": 0.8,
  "lora_enabled": true
}
```

| Field | Description |
|-------|-------------|
| `lora` | Full path to a `.safetensors` LoRA file |
| `lora_strength` | Multiplier, 0.0–2.0 (default: 1.0) |
| `lora_enabled` | Toggle on/off without removing the path (default: true) |

## Default model

**FLUX.2 Klein 4B** (Q8_0 GGUF) — ~4.4 GB total:
- Diffusion model: `leejet/FLUX.2-klein-4B-GGUF`
- VAE: `black-forest-labs/FLUX.2-dev`
- Text encoder: `unsloth/Qwen3-4B-GGUF`

Fast (4 steps), good quality at 1024×1024. Runs on 8 GB VRAM with CPU offloading.

## Requirements

- **Rust 1.75+**
- **CMake** (for building stable-diffusion.cpp)
- **CUDA toolkit** (optional, for GPU acceleration)
- ~4.4 GB disk space for the default model
- 8 GB+ VRAM recommended (CPU offloading available for less)

### Linux (Ubuntu/Debian)

```bash
sudo apt install cmake build-essential
```

### Linux (Arch)

```bash
sudo pacman -S cmake base-devel
```

### macOS

```bash
brew install cmake
```

## How it works

```
Claude Code ←→ mcp-image-gen (stdio, JSON-RPC) ←→ diffusion-rs ←→ stable-diffusion.cpp
```

The server communicates via stdin/stdout using the [MCP protocol](https://modelcontextprotocol.io/) (JSON-RPC over stdio). Image generation runs in a separate thread to not block the protocol handler.

Generated images are saved to disk and the file path is returned as text. Your MCP client (Claude Code, Claude Desktop, etc.) receives the path and can display or reference the image.

## Troubleshooting

**"Failed to build HF API"** — check your internet connection. Models are downloaded from HuggingFace on first use.

**"This model requires a HuggingFace token"** — set `HF_TOKEN` in your env or `~/.claude.json`. Some models (like FLUX.2-dev) require accepting a license on huggingface.co first.

**Out of VRAM** — enable `offload_cpu: true` and `vae_tiling: true` in your model config. Or use a smaller model.

**Slow generation** — build with `--features cuda` for GPU acceleration. Without it, inference runs on CPU.

**Logs** — stderr output goes to Claude Code's MCP server logs. Set `RUST_LOG=mcp_image_gen=debug` for verbose output.

## License

MIT

## Credits

Built by [aitherlab](https://github.com/aitherlab-dev). Part of the [aitherflow](https://github.com/aitherlab-dev/aitherflow) project.

Powered by:
- [diffusion-rs](https://github.com/newfla/diffusion-rs) — Rust bindings for stable-diffusion.cpp
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — C++ inference engine
