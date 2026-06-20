//! Default paths, persistent configuration, and path validation.
//!
//! Resolves the platform configuration directory (via the `dirs` crate), persists the
//! user's default dictionary/backend/format choices across sessions, and validates
//! user-supplied paths before they are opened. Shared by [`crate::cli::commands`] so the
//! CLI and REPL agree on where configuration and default dictionaries live.

use super::args::SerializationFormat;
use crate::repl::state::DictionaryBackend;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Get the configuration directory for liblevenshtein
pub fn config_dir() -> Result<PathBuf> {
    let base = dirs::data_local_dir().context("Could not determine local data directory")?;
    Ok(base.join("liblevenshtein"))
}

/// Get the default dictionary path for a given backend and format
pub fn default_dict_path(
    backend: DictionaryBackend,
    format: SerializationFormat,
) -> Result<PathBuf> {
    let dir = config_dir()?;
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create directory: {}", dir.display()))?;

    let filename = format!(
        "default-{}.{}",
        backend.to_string().to_lowercase(),
        file_extension(format)
    );

    Ok(dir.join(filename))
}

/// Get the application config file path (never changes)
pub fn app_config_path() -> Result<PathBuf> {
    let dir = config_dir()?;
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create directory: {}", dir.display()))?;
    Ok(dir.join("app-config.json"))
}

/// Get the default user config path
fn default_user_config_path() -> Result<PathBuf> {
    let dir = config_dir()?;
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create directory: {}", dir.display()))?;
    Ok(dir.join("config.json"))
}

/// Get the persistent config path (reads from app config)
pub fn config_file_path() -> Result<PathBuf> {
    let app_config = AppConfig::load()?;
    Ok(app_config.user_config_path)
}

/// Get the config file path with optional override
pub fn config_file_path_with_override(custom_path: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = custom_path {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
        Ok(path)
    } else {
        config_file_path()
    }
}

/// Get file extension for a serialization format
pub fn file_extension(format: SerializationFormat) -> &'static str {
    match format {
        SerializationFormat::Text => "txt",
        SerializationFormat::Bincode => "bin",
        SerializationFormat::Json => "json",
        #[cfg(feature = "protobuf")]
        SerializationFormat::Protobuf => "pb",
        #[cfg(feature = "compression")]
        SerializationFormat::BincodeGzip => "bin.gz",
        #[cfg(feature = "compression")]
        SerializationFormat::JsonGzip => "json.gz",
        #[cfg(all(feature = "protobuf", feature = "compression"))]
        SerializationFormat::ProtobufGzip => "pb.gz",
        SerializationFormat::PathsNative => "paths",
    }
}

/// Validate that a config file path has .json extension
pub fn validate_config_path(path: &Path) -> Result<()> {
    match path.extension().and_then(|s| s.to_str()) {
        Some("json") => Ok(()),
        Some(ext) => Err(anyhow::anyhow!(
            "Config file must have .json extension, got .{}. Please use a .json file.",
            ext
        )),
        None => Err(anyhow::anyhow!(
            "Config file must have .json extension. Please add .json to the filename."
        )),
    }
}

/// Validate that a dictionary path has the correct extension for its format
pub fn validate_dict_path(path: &Path, format: SerializationFormat) -> Result<()> {
    let expected_ext = file_extension(format);
    match path.extension().and_then(|s| s.to_str()) {
        Some(ext) if ext == expected_ext => Ok(()),
        Some(ext) => Err(anyhow::anyhow!(
            "Dictionary file extension .{} doesn't match serialization format {} (expected .{})",
            ext,
            format,
            expected_ext
        )),
        None => Err(anyhow::anyhow!(
            "Dictionary file must have .{} extension for {} format",
            expected_ext,
            format
        )),
    }
}

/// Change the extension of a path to match the serialization format
pub fn change_extension(path: &Path, format: SerializationFormat) -> PathBuf {
    let mut new_path = path.to_path_buf();
    new_path.set_extension(file_extension(format));
    new_path
}

/// Application-level configuration (points to user config)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AppConfig {
    /// Path to the user's configuration file
    pub user_config_path: PathBuf,
}

impl AppConfig {
    /// Load application config
    pub fn load() -> Result<Self> {
        let path = app_config_path()?;
        if !path.exists() {
            // Create default app config
            let default = Self::default();
            default.save()?;
            return Ok(default);
        }

        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read app config: {}", path.display()))?;

        serde_json::from_str(&contents)
            .with_context(|| format!("Failed to parse app config: {}", path.display()))
    }

    /// Save application config
    pub fn save(&self) -> Result<()> {
        let path = app_config_path()?;
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, contents)
            .with_context(|| format!("Failed to write app config: {}", path.display()))
    }

    /// Switch to a different user config file
    pub fn switch_user_config(&mut self, new_path: PathBuf) -> Result<()> {
        // Validate config file extension
        validate_config_path(&new_path)?;

        // Ensure parent directory exists
        if let Some(parent) = new_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
        self.user_config_path = new_path;
        self.save()
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            user_config_path: default_user_config_path()
                .unwrap_or_else(|_| PathBuf::from("config.json")),
        }
    }
}

/// User configuration stored in config file
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersistentConfig {
    /// Default dictionary path (if set by user)
    pub dict_path: Option<PathBuf>,
    /// Default backend
    pub backend: Option<DictionaryBackend>,
    /// Default format
    pub format: Option<SerializationFormat>,
    /// Default algorithm
    pub algorithm: Option<crate::transducer::Algorithm>,
    /// Default max distance
    pub max_distance: Option<usize>,
    /// Default prefix mode
    pub prefix_mode: Option<bool>,
    /// Default show distances
    pub show_distances: Option<bool>,
    /// Default result limit
    pub result_limit: Option<Option<usize>>,
    /// Auto-sync enabled
    pub auto_sync: Option<bool>,
}

impl PersistentConfig {
    /// Load configuration from file
    pub fn load() -> Result<Self> {
        Self::load_from(None)
    }

    /// Load configuration from custom path
    pub fn load_from(custom_path: Option<PathBuf>) -> Result<Self> {
        let path = config_file_path_with_override(custom_path)?;
        if !path.exists() {
            return Ok(Self::default());
        }

        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        serde_json::from_str(&contents)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))
    }

    /// Save configuration to file
    pub fn save(&self) -> Result<()> {
        self.save_to(None)
    }

    /// Save configuration to custom path
    pub fn save_to(&self, custom_path: Option<PathBuf>) -> Result<()> {
        let path = config_file_path_with_override(custom_path)?;
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, contents)
            .with_context(|| format!("Failed to write config file: {}", path.display()))
    }

    /// Copy configuration to a new location
    pub fn copy_to(&self, dest: &Path) -> Result<()> {
        // Create parent directory if needed
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }

        // Save to destination
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(dest, contents)
            .with_context(|| format!("Failed to write config file: {}", dest.display()))
    }

    /// Merge with command-line options (CLI options take precedence)
    ///
    /// This method accepts a partial config where CLI options override the stored config.
    pub fn merge_with_cli(&self, cli_overrides: &Self) -> Self {
        Self {
            dict_path: cli_overrides
                .dict_path
                .clone()
                .or_else(|| self.dict_path.clone()),
            backend: cli_overrides.backend.or(self.backend),
            format: cli_overrides.format.or(self.format),
            algorithm: cli_overrides.algorithm.or(self.algorithm),
            max_distance: cli_overrides.max_distance.or(self.max_distance),
            prefix_mode: cli_overrides.prefix_mode.or(self.prefix_mode),
            show_distances: cli_overrides.show_distances.or(self.show_distances),
            result_limit: cli_overrides.result_limit.or(self.result_limit),
            auto_sync: cli_overrides.auto_sync.or(self.auto_sync),
        }
    }
}

impl Default for PersistentConfig {
    fn default() -> Self {
        // Default to Protobuf if available, otherwise Bincode
        #[cfg(feature = "protobuf")]
        let default_format = Some(SerializationFormat::Protobuf);
        #[cfg(not(feature = "protobuf"))]
        let default_format = Some(SerializationFormat::Bincode);

        Self {
            dict_path: None,
            backend: None,
            format: default_format,
            algorithm: Some(crate::transducer::Algorithm::Standard),
            max_distance: Some(2),
            prefix_mode: Some(false),
            show_distances: Some(false),
            result_limit: Some(None),
            auto_sync: Some(false),
        }
    }
}
