//! Serialization-format auto-detection.
//!
//! Determines the [`SerializationFormat`] of a dictionary file using a three-stage
//! cascade: (1) **file extension**, which distinguishes bincode from protobuf;
//! (2) a binary-content fallback to bincode when no extension is available.
//! Used by [`crate::cli::commands`] so both the CLI and REPL load files without an
//! explicit `--format` flag.

use super::args::SerializationFormat;
use crate::repl::state::DictionaryBackend;
use anyhow::{bail, Context, Result};
use std::io::Read;
use std::path::Path;

const CONTENT_DETECTION_PROBE_LIMIT: usize = 1024;

#[inline]
fn content_detection_probe_len(file_len: u64) -> usize {
    usize::try_from(file_len).map_or(CONTENT_DETECTION_PROBE_LIMIT, |len| {
        len.min(CONTENT_DETECTION_PROBE_LIMIT)
    })
}

/// Dictionary format detection result
#[derive(Debug, Clone, Copy)]
pub struct DictFormat {
    /// Detected or specified backend
    pub backend: DictionaryBackend,
    /// Detected or specified serialization format
    pub format: SerializationFormat,
}

/// Detected format with confidence level
#[derive(Debug)]
pub struct FormatDetection {
    /// Detected format
    pub format: DictFormat,
    /// Detection method used
    pub method: DetectionMethod,
}

/// Method used to detect dictionary format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMethod {
    /// Heuristic detection via file extension
    Extension,
    /// Heuristic detection via file content analysis
    Content,
    /// User explicitly specified
    UserSpecified,
}

impl std::fmt::Display for DetectionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Extension => write!(f, "heuristic (file extension)"),
            Self::Content => write!(f, "heuristic (content analysis)"),
            Self::UserSpecified => write!(f, "user specified"),
        }
    }
}

/// Detect dictionary format from file
pub fn detect_format(
    path: &Path,
    user_backend: Option<DictionaryBackend>,
    user_format: Option<SerializationFormat>,
) -> Result<FormatDetection> {
    // If user specified both, use those
    if let (Some(backend), Some(format)) = (user_backend, user_format) {
        return Ok(FormatDetection {
            format: DictFormat { backend, format },
            method: DetectionMethod::UserSpecified,
        });
    }

    // Try extension-based heuristics
    if let Ok(detection) = detect_by_extension(path) {
        let mut format = detection.format;

        // Override with user specifications
        if let Some(backend) = user_backend {
            format.backend = backend;
        }
        if let Some(fmt) = user_format {
            format.format = fmt;
        }

        return Ok(FormatDetection {
            format,
            method: detection.method,
        });
    }

    // Try content-based heuristics
    if let Ok(detection) = detect_by_content(path) {
        let mut format = detection.format;

        // Override with user specifications
        if let Some(backend) = user_backend {
            format.backend = backend;
        }
        if let Some(fmt) = user_format {
            format.format = fmt;
        }

        return Ok(FormatDetection {
            format,
            method: detection.method,
        });
    }

    // Fallback to defaults with user overrides
    Ok(FormatDetection {
        format: DictFormat {
            backend: user_backend.unwrap_or(DictionaryBackend::PathMap),
            format: user_format.unwrap_or(SerializationFormat::Bincode),
        },
        method: if user_backend.is_some() || user_format.is_some() {
            DetectionMethod::UserSpecified
        } else {
            DetectionMethod::Extension
        },
    })
}

/// Detect format by file extension
fn detect_by_extension(path: &Path) -> Result<FormatDetection> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .context("No file extension")?;

    let format = match ext.to_lowercase().as_str() {
        "bin" | "bincode" => SerializationFormat::Bincode,
        #[cfg(feature = "protobuf")]
        "pb" | "protobuf" => SerializationFormat::Protobuf,
        _ => bail!("Unknown file extension: {}", ext),
    };

    // Try to detect backend from filename
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .context("Invalid filename")?
        .to_lowercase();

    let backend =
        if filename.contains("dawg") || filename.contains("dynamic") || filename.contains("dyn") {
            DictionaryBackend::DynamicDawg
        } else {
            DictionaryBackend::PathMap
        };

    Ok(FormatDetection {
        format: DictFormat { backend, format },
        method: DetectionMethod::Extension,
    })
}

/// Detect format by analyzing file content
fn detect_by_content(path: &Path) -> Result<FormatDetection> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;

    let mut buffer = vec![0u8; content_detection_probe_len(file.metadata()?.len())];
    file.read_exact(&mut buffer)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;

    // Both supported formats are binary. Without an extension, bincode is the
    // conservative local default; protobuf callers should specify the format.
    Ok(FormatDetection {
        format: DictFormat {
            backend: DictionaryBackend::PathMap,
            format: SerializationFormat::Bincode,
        },
        method: DetectionMethod::Content,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_detection_probe_len_bounds_before_narrowing() {
        let probe_limit =
            u64::try_from(CONTENT_DETECTION_PROBE_LIMIT).expect("probe cap fits in u64");

        assert_eq!(content_detection_probe_len(0), 0);
        assert_eq!(content_detection_probe_len(17), 17);
        assert_eq!(
            content_detection_probe_len(probe_limit),
            CONTENT_DETECTION_PROBE_LIMIT
        );
        assert_eq!(
            content_detection_probe_len(probe_limit + 1),
            CONTENT_DETECTION_PROBE_LIMIT
        );
        assert_eq!(
            content_detection_probe_len(u64::MAX),
            CONTENT_DETECTION_PROBE_LIMIT
        );
    }
}
