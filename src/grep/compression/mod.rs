//! Compression format detection and streaming decompression.
//!
//! This module provides a unified interface for decompressing various formats:
//! - Gzip (.gz)
//! - Zstandard (.zst)
//! - XZ/LZMA (.xz, .lzma)
//! - Bzip2 (.bz2)
//!
//! All decompressors implement streaming via the standard `Read` trait,
//! allowing chunk-by-chunk processing without loading entire files into memory.

#[cfg(feature = "flate2")]
pub mod gzip;

#[cfg(feature = "zstd")]
pub mod zstd;

#[cfg(feature = "xz2")]
pub mod lzma;

#[cfg(feature = "bzip2")]
pub mod bzip2;

pub mod detection;

use std::io::{self, Read};
use std::path::Path;

use crate::grep::error::{GrepError, GrepResult};

/// Supported compression formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionFormat {
    /// No compression (plain text).
    None,
    /// Gzip compression (.gz).
    Gzip,
    /// Zstandard compression (.zst, .zstd).
    Zstd,
    /// XZ/LZMA compression (.xz, .lzma).
    Xz,
    /// Bzip2 compression (.bz2).
    Bzip2,
}

impl CompressionFormat {
    /// Detect compression format from file extension.
    ///
    /// # Example
    ///
    /// ```
    /// use std::path::Path;
    /// use liblevenshtein::grep::CompressionFormat;
    ///
    /// assert_eq!(
    ///     CompressionFormat::from_extension(Path::new("file.txt.gz")),
    ///     CompressionFormat::Gzip
    /// );
    /// assert_eq!(
    ///     CompressionFormat::from_extension(Path::new("data.zst")),
    ///     CompressionFormat::Zstd
    /// );
    /// ```
    pub fn from_extension(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("gz") | Some("gzip") => Self::Gzip,
            Some("zst") | Some("zstd") => Self::Zstd,
            Some("xz") | Some("lzma") => Self::Xz,
            Some("bz2") | Some("bzip2") => Self::Bzip2,
            _ => Self::None,
        }
    }

    /// Get the typical file extension for this format.
    pub fn extension(&self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Gzip => Some("gz"),
            Self::Zstd => Some("zst"),
            Self::Xz => Some("xz"),
            Self::Bzip2 => Some("bz2"),
        }
    }

    /// Get a human-readable name for this format.
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "plain",
            Self::Gzip => "gzip",
            Self::Zstd => "zstd",
            Self::Xz => "xz",
            Self::Bzip2 => "bzip2",
        }
    }

    /// Check if the required feature is enabled for this format.
    pub fn is_available(&self) -> bool {
        match self {
            Self::None => true,
            Self::Gzip => cfg!(feature = "flate2"),
            Self::Zstd => cfg!(feature = "zstd"),
            Self::Xz => cfg!(feature = "xz2"),
            Self::Bzip2 => cfg!(feature = "bzip2"),
        }
    }

    /// Get the feature flag name required for this format.
    pub fn required_feature(&self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Gzip => Some("flate2"),
            Self::Zstd => Some("zstd"),
            Self::Xz => Some("xz2"),
            Self::Bzip2 => Some("bzip2"),
        }
    }
}

impl std::fmt::Display for CompressionFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A boxed reader that implements Send for thread-safety.
pub type BoxedReader = Box<dyn Read + Send>;

/// Passthrough "decompressor" for uncompressed data.
pub struct PassthroughReader<R> {
    inner: R,
}

impl<R: Read> PassthroughReader<R> {
    /// Create a new passthrough reader.
    pub fn new(reader: R) -> Self {
        Self { inner: reader }
    }
}

impl<R: Read> Read for PassthroughReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

/// Create a streaming decompressor for the given format.
///
/// Returns a boxed `Read` implementation that streams decompressed data.
///
/// # Errors
///
/// Returns an error if the compression format is not available (feature not enabled)
/// or if there's an error initializing the decompressor.
///
/// # Example
///
/// ```ignore
/// use std::fs::File;
/// use std::io::Read;
/// use liblevenshtein::grep::compression::{CompressionFormat, create_decompressor};
///
/// let file = File::open("data.gz")?;
/// let mut reader = create_decompressor(file, CompressionFormat::Gzip)?;
///
/// let mut content = String::new();
/// reader.read_to_string(&mut content)?;
/// ```
pub fn create_decompressor<R: Read + Send + 'static>(
    reader: R,
    format: CompressionFormat,
) -> GrepResult<BoxedReader> {
    match format {
        CompressionFormat::None => Ok(Box::new(PassthroughReader::new(reader))),

        #[cfg(feature = "flate2")]
        CompressionFormat::Gzip => Ok(Box::new(gzip::GzipDecompressor::new(reader))),

        #[cfg(not(feature = "flate2"))]
        CompressionFormat::Gzip => Err(GrepError::FeatureNotEnabled {
            feature: "flate2".to_string(),
        }),

        #[cfg(feature = "zstd")]
        CompressionFormat::Zstd => {
            let decoder = zstd::ZstdDecompressor::new(reader)
                .map_err(|e| GrepError::decompression("", e.to_string()))?;
            Ok(Box::new(decoder))
        }

        #[cfg(not(feature = "zstd"))]
        CompressionFormat::Zstd => Err(GrepError::FeatureNotEnabled {
            feature: "zstd".to_string(),
        }),

        #[cfg(feature = "xz2")]
        CompressionFormat::Xz => Ok(Box::new(lzma::XzDecompressor::new(reader))),

        #[cfg(not(feature = "xz2"))]
        CompressionFormat::Xz => Err(GrepError::FeatureNotEnabled {
            feature: "xz2".to_string(),
        }),

        #[cfg(feature = "bzip2")]
        CompressionFormat::Bzip2 => Ok(Box::new(bzip2::Bzip2Decompressor::new(reader))),

        #[cfg(not(feature = "bzip2"))]
        CompressionFormat::Bzip2 => Err(GrepError::FeatureNotEnabled {
            feature: "bzip2".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_from_extension() {
        assert_eq!(
            CompressionFormat::from_extension(Path::new("file.gz")),
            CompressionFormat::Gzip
        );
        assert_eq!(
            CompressionFormat::from_extension(Path::new("file.zst")),
            CompressionFormat::Zstd
        );
        assert_eq!(
            CompressionFormat::from_extension(Path::new("file.xz")),
            CompressionFormat::Xz
        );
        assert_eq!(
            CompressionFormat::from_extension(Path::new("file.bz2")),
            CompressionFormat::Bzip2
        );
        assert_eq!(
            CompressionFormat::from_extension(Path::new("file.txt")),
            CompressionFormat::None
        );
    }

    #[test]
    fn test_format_display() {
        assert_eq!(CompressionFormat::Gzip.to_string(), "gzip");
        assert_eq!(CompressionFormat::None.to_string(), "plain");
    }

    #[test]
    fn test_passthrough_reader() {
        let data = b"hello world";
        let mut reader = PassthroughReader::new(&data[..]);
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).expect("read should succeed");
        assert_eq!(buf, data);
    }
}
