//! Archive format support for grep operations.
//!
//! This module provides support for searching within archives:
//! - Tar archives (.tar, .tar.gz, .tar.zst, .tar.xz, .tar.bz2)
//! - Zip archives (.zip, .jar, .war)
//!
//! Archives are processed entry-by-entry in streaming fashion,
//! without extracting to disk.

pub mod entry;
pub mod filter;

#[cfg(feature = "tar")]
pub mod tar;

#[cfg(feature = "zip")]
pub mod zip;

use std::path::Path;

use crate::grep::compression::CompressionFormat;

/// Supported archive formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchiveFormat {
    /// Not an archive (plain file, possibly compressed).
    None,

    /// Tar archive, possibly with outer compression.
    Tar {
        /// Compression applied to the tar archive.
        compression: CompressionFormat,
    },

    /// Zip archive (handles per-entry compression internally).
    Zip,
}

impl ArchiveFormat {
    /// Detect archive format from file path.
    ///
    /// Recognizes common extensions and compound extensions like `.tar.gz`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::path::Path;
    /// use liblevenshtein::grep::ArchiveFormat;
    /// use liblevenshtein::grep::CompressionFormat;
    ///
    /// assert!(matches!(
    ///     ArchiveFormat::from_path(Path::new("archive.tar.gz")),
    ///     ArchiveFormat::Tar { compression: CompressionFormat::Gzip }
    /// ));
    ///
    /// assert!(matches!(
    ///     ArchiveFormat::from_path(Path::new("archive.zip")),
    ///     ArchiveFormat::Zip
    /// ));
    /// ```
    pub fn from_path(path: &Path) -> Self {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Check for compound tar extensions first (order matters)
        if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
            return Self::Tar {
                compression: CompressionFormat::Gzip,
            };
        }
        if name.ends_with(".tar.zst") || name.ends_with(".tar.zstd") {
            return Self::Tar {
                compression: CompressionFormat::Zstd,
            };
        }
        if name.ends_with(".tar.xz") || name.ends_with(".txz") {
            return Self::Tar {
                compression: CompressionFormat::Xz,
            };
        }
        if name.ends_with(".tar.bz2") || name.ends_with(".tbz2") || name.ends_with(".tbz") {
            return Self::Tar {
                compression: CompressionFormat::Bzip2,
            };
        }
        if name.ends_with(".tar.lzma") {
            return Self::Tar {
                compression: CompressionFormat::Xz,
            };
        }
        if name.ends_with(".tar") {
            return Self::Tar {
                compression: CompressionFormat::None,
            };
        }

        // Check simple extensions
        match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => match ext.to_lowercase().as_str() {
                "zip" | "jar" | "war" | "ear" | "apk" | "aar" => Self::Zip,
                _ => Self::None,
            },
            None => Self::None,
        }
    }

    /// Get a human-readable name for this format.
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "plain",
            Self::Tar { compression } => match compression {
                CompressionFormat::None => "tar",
                CompressionFormat::Gzip => "tar.gz",
                CompressionFormat::Zstd => "tar.zst",
                CompressionFormat::Xz => "tar.xz",
                CompressionFormat::Bzip2 => "tar.bz2",
            },
            Self::Zip => "zip",
        }
    }

    /// Check if the required features are enabled for this format.
    pub fn is_available(&self) -> bool {
        match self {
            Self::None => true,
            Self::Tar { compression } => {
                cfg!(feature = "tar") && compression.is_available()
            }
            Self::Zip => cfg!(feature = "zip"),
        }
    }

    /// Get the feature flags required for this format.
    pub fn required_features(&self) -> Vec<&'static str> {
        match self {
            Self::None => vec![],
            Self::Tar { compression } => {
                let mut features = vec!["tar"];
                if let Some(comp_feature) = compression.required_feature() {
                    features.push(comp_feature);
                }
                features
            }
            Self::Zip => vec!["zip"],
        }
    }
}

impl std::fmt::Display for ArchiveFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tar_detection() {
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("archive.tar")),
            ArchiveFormat::Tar { compression: CompressionFormat::None }
        ));
    }

    #[test]
    fn test_tar_gz_detection() {
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("archive.tar.gz")),
            ArchiveFormat::Tar { compression: CompressionFormat::Gzip }
        ));
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("archive.tgz")),
            ArchiveFormat::Tar { compression: CompressionFormat::Gzip }
        ));
    }

    #[test]
    fn test_tar_zst_detection() {
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("archive.tar.zst")),
            ArchiveFormat::Tar { compression: CompressionFormat::Zstd }
        ));
    }

    #[test]
    fn test_tar_xz_detection() {
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("archive.tar.xz")),
            ArchiveFormat::Tar { compression: CompressionFormat::Xz }
        ));
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("archive.txz")),
            ArchiveFormat::Tar { compression: CompressionFormat::Xz }
        ));
    }

    #[test]
    fn test_tar_bz2_detection() {
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("archive.tar.bz2")),
            ArchiveFormat::Tar { compression: CompressionFormat::Bzip2 }
        ));
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("archive.tbz2")),
            ArchiveFormat::Tar { compression: CompressionFormat::Bzip2 }
        ));
    }

    #[test]
    fn test_zip_detection() {
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("archive.zip")),
            ArchiveFormat::Zip
        ));
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("app.jar")),
            ArchiveFormat::Zip
        ));
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("app.war")),
            ArchiveFormat::Zip
        ));
    }

    #[test]
    fn test_plain_file() {
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("file.txt")),
            ArchiveFormat::None
        ));
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("file.txt.gz")),
            ArchiveFormat::None
        ));
    }

    #[test]
    fn test_case_insensitive() {
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("ARCHIVE.TAR.GZ")),
            ArchiveFormat::Tar { compression: CompressionFormat::Gzip }
        ));
        assert!(matches!(
            ArchiveFormat::from_path(Path::new("Archive.ZIP")),
            ArchiveFormat::Zip
        ));
    }

    #[test]
    fn test_format_name() {
        assert_eq!(ArchiveFormat::None.name(), "plain");
        assert_eq!(
            ArchiveFormat::Tar { compression: CompressionFormat::None }.name(),
            "tar"
        );
        assert_eq!(
            ArchiveFormat::Tar { compression: CompressionFormat::Gzip }.name(),
            "tar.gz"
        );
        assert_eq!(ArchiveFormat::Zip.name(), "zip");
    }
}
