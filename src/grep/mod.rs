//! Support for grepping through compressed and archived files.
//!
//! This module provides streaming support for searching through:
//! - Compressed files: `.gz`, `.zst`, `.xz`, `.lzma`, `.bz2`
//! - Archive files: `.tar`, `.tar.gz`, `.tar.zst`, `.tar.xz`, `.tar.bz2`, `.zip`
//!
//! All operations are streaming (chunk-by-chunk) for memory efficiency.
//!
//! # Features
//!
//! This module requires one or more optional features:
//! - `grep-compression`: Enables compression support (gzip, zstd, xz, bzip2)
//! - `grep-archives`: Enables archive support (tar, zip) - implies `grep-compression`
//! - `grep-full`: Enables all grep features including phonetic rules
//!
//! # Usage
//!
//! ## Searching a compressed file
//!
//! ```ignore
//! use liblevenshtein::grep::{GrepSource, GrepPath, GrepConfig};
//!
//! let path = GrepPath::parse("logs.gz")?;
//! let source = GrepSource::new(GrepConfig::default());
//!
//! for entry in source.open(&path)? {
//!     let entry = entry?;
//!     // Process entry.content() with your grep matcher
//! }
//! ```
//!
//! ## Searching within archives
//!
//! ```ignore
//! use liblevenshtein::grep::{GrepSource, GrepPath, GrepConfig};
//!
//! // Search all files in archive
//! let path = GrepPath::parse("source.tar.gz")?;
//!
//! // Or filter to specific entries
//! let path = GrepPath::parse("logs.tar.gz:*.log")?;
//!
//! let source = GrepSource::new(GrepConfig::default());
//! for entry in source.open(&path)? {
//!     let entry = entry?;
//!     println!("Searching: {}", entry.source_id());
//!     // Process entry.content()
//! }
//! ```

pub mod compression;
pub mod error;
mod limited_read;
pub mod position_tracker;
pub mod result;
pub mod source;

#[cfg(any(feature = "tar", feature = "zip"))]
pub mod archive;

#[cfg(any(
    feature = "grep-pdf",
    feature = "grep-docx",
    feature = "grep-xlsx",
    feature = "grep-epub",
    feature = "grep-odt"
))]
pub mod document;

// Re-export main types at module level for convenient access
pub use compression::CompressionFormat;
pub use error::{GrepError, GrepResult};
pub use position_tracker::PositionTracker;
pub use result::{GrepMatchResult, MatchLocation, SourceId};
pub use source::GrepPath;

#[cfg(any(feature = "tar", feature = "zip"))]
pub use archive::ArchiveFormat;

#[cfg(any(
    feature = "grep-pdf",
    feature = "grep-docx",
    feature = "grep-xlsx",
    feature = "grep-epub",
    feature = "grep-odt"
))]
pub use document::{DocumentExtractor, DocumentExtractorConfig, DocumentFormat};

use std::fs::File;
use std::io::{self, BufReader};
use std::path::Path;

use compression::create_decompressor;

#[cfg(feature = "tar")]
use archive::tar::TarArchiveReader;

#[cfg(feature = "zip")]
use archive::zip::ZipArchiveReader;

fn initial_plain_read_capacity(
    file_len: u64,
    max_file_size: Option<u64>,
    compression: CompressionFormat,
) -> usize {
    if compression != CompressionFormat::None {
        return 0;
    }

    let bounded_len = max_file_size.map_or(file_len, |max_size| file_len.min(max_size));
    limited_read::initial_read_capacity(Some(bounded_len))
}

/// Configuration for grep source operations.
#[derive(Debug, Clone)]
pub struct GrepConfig {
    /// Whether to auto-detect compression format.
    /// If false, files are treated as plain text.
    pub auto_decompress: bool,

    /// Maximum file size to read (bytes). Files larger than this are skipped.
    /// `None` means no limit.
    pub max_file_size: Option<u64>,

    /// Skip files that appear to be binary (contain NUL bytes).
    pub skip_binary: bool,

    /// Whether to include hidden files (starting with `.`).
    pub include_hidden: bool,

    /// Default glob filter for archives when none is specified.
    /// `None` means search all entries.
    pub default_archive_filter: Option<String>,

    /// Whether to extract text from document files (PDF, DOCX, etc.).
    pub extract_documents: bool,

    /// Enable OCR for image-based PDFs.
    pub enable_ocr: bool,

    /// OCR language code (Tesseract format, e.g., "eng").
    pub ocr_language: String,
}

impl Default for GrepConfig {
    fn default() -> Self {
        Self {
            auto_decompress: true,
            max_file_size: Some(100 * 1024 * 1024), // 100 MB default limit
            skip_binary: true,
            include_hidden: false,
            default_archive_filter: None,
            // Auto-enable document extraction when document features are available
            #[cfg(any(
                feature = "grep-pdf",
                feature = "grep-docx",
                feature = "grep-xlsx",
                feature = "grep-epub",
                feature = "grep-odt"
            ))]
            extract_documents: true,
            #[cfg(not(any(
                feature = "grep-pdf",
                feature = "grep-docx",
                feature = "grep-xlsx",
                feature = "grep-epub",
                feature = "grep-odt"
            )))]
            extract_documents: false,
            enable_ocr: false,
            ocr_language: "eng".to_string(),
        }
    }
}

impl GrepConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Disable auto-decompression (treat compressed files as plain text).
    pub fn no_decompress(mut self) -> Self {
        self.auto_decompress = false;
        self
    }

    /// Set maximum file size limit.
    pub fn max_size(mut self, bytes: u64) -> Self {
        self.max_file_size = Some(bytes);
        self
    }

    /// Remove file size limit.
    pub fn no_size_limit(mut self) -> Self {
        self.max_file_size = None;
        self
    }

    /// Skip binary files.
    pub fn skip_binary_files(mut self, skip: bool) -> Self {
        self.skip_binary = skip;
        self
    }

    /// Include hidden files.
    pub fn include_hidden_files(mut self, include: bool) -> Self {
        self.include_hidden = include;
        self
    }

    /// Set default archive filter pattern.
    pub fn default_filter(mut self, pattern: impl Into<String>) -> Self {
        self.default_archive_filter = Some(pattern.into());
        self
    }

    /// Enable document text extraction (PDF, DOCX, etc.).
    pub fn extract_documents(mut self, extract: bool) -> Self {
        self.extract_documents = extract;
        self
    }

    /// Enable OCR for image-based PDFs.
    pub fn enable_ocr(mut self, enable: bool) -> Self {
        self.enable_ocr = enable;
        self
    }

    /// Set OCR language code (Tesseract format, e.g., "eng", "deu").
    pub fn ocr_language(mut self, lang: impl Into<String>) -> Self {
        self.ocr_language = lang.into();
        self
    }

    /// Enable document extraction with OCR.
    pub fn with_ocr(self, language: impl Into<String>) -> Self {
        self.extract_documents(true)
            .enable_ocr(true)
            .ocr_language(language)
    }
}

/// A source for grep operations supporting compressed and archived files.
///
/// # Example
///
/// ```ignore
/// use liblevenshtein::grep::{GrepSource, GrepPath, GrepConfig};
///
/// let source = GrepSource::new(GrepConfig::default());
/// let path = GrepPath::parse("data.tar.gz:*.log")?;
///
/// for entry in source.open(&path)? {
///     let entry = entry?;
///     // entry.source_id() identifies the file/archive entry
///     // entry.content() provides the decompressed content
/// }
/// ```
#[derive(Debug, Clone)]
pub struct GrepSource {
    config: GrepConfig,
}

impl GrepSource {
    /// Create a new grep source with the given configuration.
    pub fn new(config: GrepConfig) -> Self {
        Self { config }
    }

    /// Create a grep source with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(GrepConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &GrepConfig {
        &self.config
    }

    /// Open a path and return an iterator over its entries.
    ///
    /// For plain files, returns a single entry.
    /// For archives, returns an entry for each matching file within.
    pub fn open(&self, path: &GrepPath) -> GrepResult<GrepEntryIterator> {
        let fs_path = &path.filesystem_path;

        // Check file exists
        if !fs_path.exists() {
            return Err(GrepError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("file not found: {}", fs_path.display()),
            )));
        }

        let metadata = fs_path.metadata()?;

        // Check file size limit
        if let Some(max_size) = self.config.max_file_size {
            if metadata.len() > max_size {
                return Err(GrepError::Io(limited_read::file_too_large_error(
                    "file",
                    metadata.len(),
                    max_size,
                )));
            }
        }

        // Determine how to handle based on format
        let compression = if self.config.auto_decompress {
            path.compression
        } else {
            CompressionFormat::None
        };

        #[cfg(any(feature = "tar", feature = "zip"))]
        let archive = if self.config.auto_decompress {
            path.archive
        } else {
            ArchiveFormat::None
        };

        #[cfg(not(any(feature = "tar", feature = "zip")))]
        let archive = ();

        // Get filter pattern
        let filter = path
            .archive_filter
            .clone()
            .or_else(|| self.config.default_archive_filter.clone());

        // Create appropriate iterator
        #[cfg(any(feature = "tar", feature = "zip"))]
        match archive {
            ArchiveFormat::None => {
                // Plain file (possibly compressed)
                self.open_plain_file(fs_path, metadata.len(), compression)
            }
            #[cfg(feature = "tar")]
            ArchiveFormat::Tar {
                compression: tar_compression,
            } => self.open_tar_archive(fs_path, tar_compression, filter),
            #[cfg(not(feature = "tar"))]
            ArchiveFormat::Tar { .. } => Err(GrepError::FeatureNotEnabled {
                feature: "tar".to_string(),
            }),
            #[cfg(feature = "zip")]
            ArchiveFormat::Zip => self.open_zip_archive(fs_path, filter),
            #[cfg(not(feature = "zip"))]
            ArchiveFormat::Zip => Err(GrepError::FeatureNotEnabled {
                feature: "zip".to_string(),
            }),
        }

        #[cfg(not(any(feature = "tar", feature = "zip")))]
        {
            let _ = (archive, filter);
            self.open_plain_file(fs_path, metadata.len(), compression)
        }
    }

    /// Open a plain (non-archive) file.
    fn open_plain_file(
        &self,
        path: &Path,
        file_len: u64,
        compression: CompressionFormat,
    ) -> GrepResult<GrepEntryIterator> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let decompressed = create_decompressor(reader, compression)?;

        // Read all content (streaming would require different architecture)
        let mut content = Vec::with_capacity(initial_plain_read_capacity(
            file_len,
            self.config.max_file_size,
            compression,
        ));
        let mut reader = decompressed;
        limited_read::read_to_end_limited(
            &mut reader,
            &mut content,
            self.config.max_file_size,
            "file",
        )?;

        // Try document extraction if enabled
        #[cfg(any(
            feature = "grep-pdf",
            feature = "grep-docx",
            feature = "grep-xlsx",
            feature = "grep-epub",
            feature = "grep-odt"
        ))]
        if self.config.extract_documents {
            let doc_format = document::detect_document_format(&content, Some(path));
            if doc_format != document::DocumentFormat::None {
                let extractor_config = document::DocumentExtractorConfig {
                    enable_ocr: self.config.enable_ocr,
                    ocr_language: self.config.ocr_language.clone(),
                    max_size: self.config.max_file_size,
                    skip_on_error: false,
                };
                let extractor = document::DocumentExtractor::with_config(extractor_config);
                let text = extractor.extract_format(&content, doc_format)?;
                content = text.into_bytes();
            }
        }

        // Check for binary content (skip if document extraction was performed)
        if self.config.skip_binary && compression::detection::is_binary(&content) {
            return Err(GrepError::BinaryFile(path.to_path_buf()));
        }

        let source_id = SourceId::file(path);
        let entry = GrepEntry { source_id, content };

        Ok(GrepEntryIterator {
            inner: GrepEntryIteratorInner::Single(Some(entry).into_iter()),
        })
    }

    /// Open a tar archive.
    #[cfg(feature = "tar")]
    fn open_tar_archive(
        &self,
        path: &Path,
        compression: CompressionFormat,
        filter: Option<String>,
    ) -> GrepResult<GrepEntryIterator> {
        let mut reader = TarArchiveReader::open(path, compression)?
            .with_max_entry_size(self.config.max_file_size);

        if let Some(pattern) = filter {
            reader = reader.with_pattern(&pattern)?;
        }

        let entries = reader.entries()?.map(|result| {
            result.map(|(source_id, _meta, content)| GrepEntry { source_id, content })
        });

        Ok(GrepEntryIterator {
            inner: GrepEntryIteratorInner::Archive(Box::new(entries)),
        })
    }

    /// Open a zip archive.
    #[cfg(feature = "zip")]
    fn open_zip_archive(
        &self,
        path: &Path,
        filter: Option<String>,
    ) -> GrepResult<GrepEntryIterator> {
        let mut reader =
            ZipArchiveReader::open(path)?.with_max_entry_size(self.config.max_file_size);

        if let Some(pattern) = filter {
            reader = reader.with_pattern(&pattern)?;
        }

        let entries = reader.entries()?.map(|result| {
            result.map(|(source_id, _meta, content)| GrepEntry { source_id, content })
        });

        Ok(GrepEntryIterator {
            inner: GrepEntryIteratorInner::Archive(Box::new(entries)),
        })
    }

    /// Open a path string (convenience method that parses the path).
    pub fn open_str(&self, path: &str) -> GrepResult<GrepEntryIterator> {
        let grep_path = GrepPath::parse(path)?;
        self.open(&grep_path)
    }
}

impl Default for GrepSource {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// An entry from a grep source (a file or archive entry).
#[derive(Debug, Clone)]
pub struct GrepEntry {
    /// Identifies the source (file path, or archive + entry path).
    source_id: SourceId,

    /// The decompressed content.
    content: Vec<u8>,
}

impl GrepEntry {
    /// Get the source identifier.
    pub fn source_id(&self) -> &SourceId {
        &self.source_id
    }

    /// Get the content as bytes.
    pub fn content(&self) -> &[u8] {
        &self.content
    }

    /// Get the content as a UTF-8 string.
    ///
    /// Returns an error if the content is not valid UTF-8.
    pub fn content_str(&self) -> GrepResult<&str> {
        std::str::from_utf8(&self.content).map_err(|e| GrepError::InvalidUtf8 {
            file_path: self.source_id.display(),
            message: e.to_string(),
        })
    }

    /// Consume the entry and return the content.
    pub fn into_content(self) -> Vec<u8> {
        self.content
    }

    /// Check if this entry is from an archive.
    pub fn is_archive_entry(&self) -> bool {
        self.source_id.is_archive_entry()
    }
}

/// Iterator over grep entries from a source.
pub struct GrepEntryIterator {
    inner: GrepEntryIteratorInner,
}

enum GrepEntryIteratorInner {
    /// Single plain file entry.
    Single(std::option::IntoIter<GrepEntry>),
    /// Archive entries.
    #[cfg(any(feature = "tar", feature = "zip"))]
    Archive(Box<dyn Iterator<Item = GrepResult<GrepEntry>>>),
}

impl Iterator for GrepEntryIterator {
    type Item = GrepResult<GrepEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            GrepEntryIteratorInner::Single(iter) => iter.next().map(Ok),
            #[cfg(any(feature = "tar", feature = "zip"))]
            GrepEntryIteratorInner::Archive(iter) => iter.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "zip")]
    use std::fs::File;
    use std::io::Cursor;
    #[cfg(feature = "zip")]
    use std::io::Write;

    #[test]
    fn test_grep_config_default() {
        let config = GrepConfig::default();
        assert!(config.auto_decompress);
        assert!(config.skip_binary);
        assert_eq!(config.max_file_size, Some(100 * 1024 * 1024));
    }

    #[test]
    fn test_grep_config_builder() {
        let config = GrepConfig::new()
            .no_decompress()
            .no_size_limit()
            .skip_binary_files(false)
            .include_hidden_files(true);

        assert!(!config.auto_decompress);
        assert!(config.max_file_size.is_none());
        assert!(!config.skip_binary);
        assert!(config.include_hidden);
    }

    #[test]
    fn test_initial_plain_read_capacity_uses_plain_file_size() {
        assert_eq!(
            initial_plain_read_capacity(4096, Some(10 * 1024), CompressionFormat::None),
            4096
        );
        assert_eq!(
            initial_plain_read_capacity(4096, Some(1024), CompressionFormat::None),
            1024
        );
    }

    #[test]
    fn test_initial_plain_read_capacity_is_capped_without_size_limit() {
        assert_eq!(
            initial_plain_read_capacity(
                limited_read::usize_to_u64_saturating(limited_read::MAX_INITIAL_READ_CAPACITY) + 1,
                None,
                CompressionFormat::None
            ),
            limited_read::MAX_INITIAL_READ_CAPACITY
        );
    }

    #[test]
    fn test_initial_plain_read_capacity_ignores_compressed_size() {
        assert_eq!(
            initial_plain_read_capacity(4096, Some(10 * 1024), CompressionFormat::Gzip),
            0
        );
    }

    #[test]
    fn test_read_to_end_limited_accepts_exact_limit() {
        let mut reader = Cursor::new(b"hello");
        let mut content = Vec::new();

        limited_read::read_to_end_limited(&mut reader, &mut content, Some(5), "file")
            .expect("read within limit");

        assert_eq!(content, b"hello");
    }

    #[test]
    fn test_read_to_end_limited_rejects_expanded_content() {
        let mut reader = Cursor::new(b"hello");
        let mut content = Vec::new();

        let err = limited_read::read_to_end_limited(&mut reader, &mut content, Some(4), "file")
            .expect_err("limit exceeded");

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(content.is_empty());
    }

    #[test]
    fn test_grep_entry() {
        let entry = GrepEntry {
            source_id: SourceId::file("test.txt"),
            content: b"Hello, World!".to_vec(),
        };

        assert_eq!(entry.content(), b"Hello, World!");
        assert_eq!(entry.content_str().expect("valid utf8"), "Hello, World!");
        assert!(!entry.is_archive_entry());
    }

    #[test]
    fn test_grep_entry_archive() {
        let entry = GrepEntry {
            source_id: SourceId::archive_entry("archive.tar.gz", "dir/file.txt"),
            content: b"test content".to_vec(),
        };

        assert!(entry.is_archive_entry());
        assert_eq!(entry.source_id().display(), "archive.tar.gz:dir/file.txt");
    }

    #[cfg(feature = "zip")]
    #[test]
    fn test_grep_source_applies_max_size_to_zip_entries() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let zip_path = dir.path().join("test.zip");

        let file = File::create(&zip_path).expect("create zip file");
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        let content = vec![b'a'; 1024 * 1024];
        zip.start_file("large.txt", options)
            .expect("start zip entry");
        zip.write_all(&content).expect("write zip entry");
        zip.finish().expect("finish zip");

        let max_size = zip_path.metadata().expect("stat zip").len() + 1;
        assert!(limited_read::len_exceeds_limit(content.len(), max_size));

        let source = GrepSource::new(GrepConfig::new().max_size(max_size));
        let path = GrepPath::parse(zip_path.to_string_lossy().as_ref()).expect("parse zip path");
        let mut entries = source.open(&path).expect("open zip source");
        let err = entries
            .next()
            .expect("zip should yield one entry")
            .expect_err("zip entry should exceed max size");

        assert!(
            matches!(err, GrepError::Io(ref io_err) if io_err.kind() == io::ErrorKind::InvalidData)
        );
    }
}
