//! Serialization-format auto-detection.
//!
//! Determines the [`SerializationFormat`] of a dictionary file using a three-stage
//! cascade: (1) **magic bytes** — an exact signature at the start of the file
//! (e.g. the gzip header, the bincode/protobuf markers); (2) **file extension** —
//! a heuristic when no signature matches; (3) **content analysis** — a fallback that
//! inspects the bytes (e.g. valid UTF-8 / JSON shape) to guess plain-text vs binary.
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
    /// Exact detection via magic bytes or header
    Exact,
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
            Self::Exact => write!(f, "exact (magic bytes)"),
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

    // Try exact detection first (magic bytes)
    if let Ok(detection) = detect_exact(path) {
        // Validate against user specifications if provided
        if let Some(backend) = user_backend {
            if backend != detection.format.backend {
                bail!(
                    "Backend mismatch: file contains {} but --backend specified {}",
                    detection.format.backend,
                    backend
                );
            }
        }
        if let Some(format) = user_format {
            if format != detection.format.format {
                bail!(
                    "Format mismatch: file contains {} but --format specified {}",
                    detection.format.format,
                    format
                );
            }
        }
        return Ok(detection);
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
            format: user_format.unwrap_or(SerializationFormat::Text),
        },
        method: if user_backend.is_some() || user_format.is_some() {
            DetectionMethod::UserSpecified
        } else {
            DetectionMethod::Extension
        },
    })
}

/// Detect format by exact matching (magic bytes)
fn detect_exact(path: &Path) -> Result<FormatDetection> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;

    let mut header = [0u8; 16];
    let bytes_read = file
        .read(&mut header)
        .with_context(|| format!("Failed to read file header: {}", path.display()))?;

    if bytes_read < 4 {
        bail!("File too small for magic byte detection");
    }

    // Check for JSON format
    if header[0] == b'{' || header[0] == b'[' {
        // Could be JSON - try to determine backend from content
        return Ok(FormatDetection {
            format: DictFormat {
                backend: DictionaryBackend::PathMap, // Default for JSON
                format: SerializationFormat::Json,
            },
            method: DetectionMethod::Exact,
        });
    }

    // Check for plain text (all printable ASCII or starts with letter)
    if bytes_read > 0 && header[0].is_ascii_alphabetic() {
        let is_text = header[..bytes_read]
            .iter()
            .all(|&b| b.is_ascii() && (b.is_ascii_graphic() || b.is_ascii_whitespace()));

        if is_text {
            return Ok(FormatDetection {
                format: DictFormat {
                    backend: DictionaryBackend::PathMap,
                    format: SerializationFormat::Text,
                },
                method: DetectionMethod::Content,
            });
        }
    }

    bail!("Could not detect format from magic bytes")
}

/// Detect format by file extension
fn detect_by_extension(path: &Path) -> Result<FormatDetection> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .context("No file extension")?;

    let format = match ext.to_lowercase().as_str() {
        "txt" | "text" | "dict" => SerializationFormat::Text,
        "bin" | "bincode" => SerializationFormat::Bincode,
        "json" => SerializationFormat::Json,
        #[cfg(feature = "protobuf")]
        "pb" | "protobuf" => SerializationFormat::Protobuf,
        "paths" => SerializationFormat::PathsNative,
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

    // Check if it's valid UTF-8 text
    if let Ok(text) = std::str::from_utf8(&buffer) {
        // Plain text dictionary: one word per line
        if text.lines().take(10).all(|line| {
            let trimmed = line.trim();
            trimmed.is_empty()
                || trimmed.starts_with('#')
                || trimmed
                    .chars()
                    .all(|c| c.is_alphanumeric() || c.is_whitespace() || "-_'".contains(c))
        }) {
            return Ok(FormatDetection {
                format: DictFormat {
                    backend: DictionaryBackend::PathMap,
                    format: SerializationFormat::Text,
                },
                method: DetectionMethod::Content,
            });
        }
    }

    // Assume binary format if not text
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
