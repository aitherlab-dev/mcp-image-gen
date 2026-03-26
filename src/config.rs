use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::{fs, io};
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    pub models_path: PathBuf,
    pub images_path: PathBuf,
    pub selected_model: String,
    pub width: i32,
    pub height: i32,
    pub steps: i32,
}

impl Config {
    pub fn new() -> Result<Self, String> {
        let data_dir = dirs::data_dir()
            .ok_or("Cannot determine data directory (XDG_DATA_HOME)")?;
        let data_dir = data_dir.join("mcp-image-gen");

        Ok(Self {
            models_path: data_dir.join("models"),
            images_path: data_dir.join("images"),
            selected_model: "FLUX.2-klein-4B".into(),
            width: 1024,
            height: 1024,
            steps: 20,
        })
    }

    pub fn config_path() -> Result<PathBuf, String> {
        let config_dir = dirs::config_dir()
            .ok_or("Cannot determine config directory (XDG_CONFIG_HOME)")?;
        Ok(config_dir
            .join("mcp-image-gen")
            .join("settings.json"))
    }

    pub fn load() -> Result<Self, String> {
        let path = Self::config_path()?;

        let mut config = if path.exists() {
            match fs::read_to_string(&path) {
                Ok(content) => match serde_json::from_str::<Config>(&content) {
                    Ok(cfg) => {
                        info!("Config loaded from {}", path.display());
                        cfg
                    }
                    Err(e) => {
                        warn!("Failed to parse config: {e}, using defaults");
                        let cfg = Config::new()?;
                        cfg.save();
                        cfg
                    }
                },
                Err(e) => {
                    warn!("Failed to read config: {e}, using defaults");
                    let cfg = Config::new()?;
                    cfg.save();
                    cfg
                }
            }
        } else {
            info!("No config found at {}, creating defaults", path.display());
            let cfg = Config::new()?;
            cfg.save();
            cfg
        };

        // Env-переменные имеют приоритет над конфиг-файлом
        if let Ok(val) = std::env::var("MCP_IMAGE_GEN_MODELS_PATH") {
            info!("Overriding models_path from env: {val}");
            config.models_path = PathBuf::from(val);
        }
        if let Ok(val) = std::env::var("MCP_IMAGE_GEN_IMAGES_PATH") {
            info!("Overriding images_path from env: {val}");
            config.images_path = PathBuf::from(val);
        }
        if let Ok(val) = std::env::var("MCP_IMAGE_GEN_SELECTED_MODEL") {
            info!("Overriding selected_model from env: {val}");
            config.selected_model = val;
        }

        Ok(config)
    }

    pub fn save(&self) {
        let path = match Self::config_path() {
            Ok(p) => p,
            Err(e) => {
                warn!("Cannot determine config path: {e}");
                return;
            }
        };

        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                warn!("Failed to create config dir: {e}");
                return;
            }
        }

        match serde_json::to_string_pretty(self) {
            Ok(json) => {
                if let Err(e) = atomic_write(&path, json.as_bytes()) {
                    warn!("Failed to write config: {e}");
                }
            }
            Err(e) => {
                warn!("Failed to serialize config: {e}");
            }
        }
    }

    pub fn ensure_dirs(&self) {
        if let Err(e) = fs::create_dir_all(&self.models_path) {
            warn!("Failed to create models dir: {e}");
        }
        if let Err(e) = fs::create_dir_all(&self.images_path) {
            warn!("Failed to create images dir: {e}");
        }
    }
}

/// Write to a temp file in the same directory, then atomically rename.
fn atomic_write(path: &PathBuf, data: &[u8]) -> io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "path has no parent directory")
    })?;
    let tmp_path = parent.join(format!(".tmp_{}", std::process::id()));
    fs::write(&tmp_path, data)?;
    fs::rename(&tmp_path, path).inspect_err(|_| {
        let _ = fs::remove_file(&tmp_path);
    })
}
