//! Document text extraction for grep operations.
//!
//! This module provides text extraction from binary document formats:
//! - PDF files (with optional OCR support)
//! - Microsoft Office files (DOCX, XLSX)
//! - E-book formats (EPUB)
//! - Open Document formats (ODT)
//! - DjVu files (placeholder)
//!
//! # Features
//!
//! - `grep-pdf`: Enable PDF text extraction
//! - `grep-pdf-ocr`: Enable OCR for image-based PDFs
//! - `grep-docx`: Enable DOCX text extraction
//! - `grep-xlsx`: Enable XLSX text extraction
//! - `grep-epub`: Enable EPUB text extraction
//! - `grep-odt`: Enable ODT text extraction
//! - `grep-documents`: Enable all document formats

pub mod detection;

#[cfg(feature = "grep-pdf")]
pub mod pdf;

#[cfg(feature = "grep-pdf-ocr")]
pub mod pdf_ocr;

#[cfg(feature = "grep-docx")]
pub mod docx;

#[cfg(feature = "grep-xlsx")]
pub mod xlsx;

#[cfg(feature = "grep-epub")]
pub mod epub;

#[cfg(feature = "grep-odt")]
pub mod odt;

pub mod djvu;

use std::path::Path;

use crate::grep::error::{GrepError, GrepResult};

pub use detection::detect_document_format;

/// Document format enum representing supported file types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DocumentFormat {
    /// Not a recognized document format (plain text, binary, etc.)
    None,
    /// PDF document
    Pdf,
    /// Microsoft Word document (.docx)
    Docx,
    /// Microsoft Excel spreadsheet (.xlsx)
    Xlsx,
    /// EPUB e-book
    Epub,
    /// OpenDocument Text (.odt)
    Odt,
    /// DjVu document
    DjVu,
}

impl DocumentFormat {
    /// Detect document format from file extension.
    pub fn from_extension(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => Self::from_extension_str(ext),
            None => Self::None,
        }
    }

    /// Detect document format from extension string.
    pub fn from_extension_str(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "pdf" => Self::Pdf,
            "docx" => Self::Docx,
            "xlsx" | "xls" => Self::Xlsx,
            "epub" => Self::Epub,
            "odt" => Self::Odt,
            "djvu" | "djv" => Self::DjVu,
            _ => Self::None,
        }
    }

    /// Check if this format is a recognized document type.
    pub fn is_document(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Check if extraction is supported for this format (feature enabled).
    #[allow(unreachable_code)]
    pub fn is_supported(&self) -> bool {
        match self {
            Self::None => false,
            Self::Pdf => {
                #[cfg(feature = "grep-pdf")]
                return true;
                false
            }
            Self::Docx => {
                #[cfg(feature = "grep-docx")]
                return true;
                false
            }
            Self::Xlsx => {
                #[cfg(feature = "grep-xlsx")]
                return true;
                false
            }
            Self::Epub => {
                #[cfg(feature = "grep-epub")]
                return true;
                false
            }
            Self::Odt => {
                #[cfg(feature = "grep-odt")]
                return true;
                false
            }
            Self::DjVu => false, // Placeholder, not yet implemented
        }
    }

    /// Get the file extension for this format.
    pub fn extension(&self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Pdf => Some("pdf"),
            Self::Docx => Some("docx"),
            Self::Xlsx => Some("xlsx"),
            Self::Epub => Some("epub"),
            Self::Odt => Some("odt"),
            Self::DjVu => Some("djvu"),
        }
    }

    /// Get human-readable format name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "Unknown",
            Self::Pdf => "PDF",
            Self::Docx => "Microsoft Word",
            Self::Xlsx => "Microsoft Excel",
            Self::Epub => "EPUB",
            Self::Odt => "OpenDocument Text",
            Self::DjVu => "DjVu",
        }
    }
}

impl std::fmt::Display for DocumentFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Trait for document text extractors.
///
/// Implement this trait to add support for a new document format.
pub trait TextExtractor: Send + Sync {
    /// Extract text content from document bytes.
    ///
    /// # Arguments
    ///
    /// * `data` - The raw document bytes
    ///
    /// # Returns
    ///
    /// The extracted text content, or an error if extraction fails.
    fn extract(&self, data: &[u8]) -> GrepResult<String>;

    /// Get the document format this extractor handles.
    fn format(&self) -> DocumentFormat;

    /// Check if this extractor supports the given format.
    fn supports(&self, format: DocumentFormat) -> bool {
        self.format() == format
    }
}

/// Configuration for document extraction.
#[derive(Debug, Clone)]
pub struct DocumentExtractorConfig {
    /// Enable OCR for image-based PDFs.
    pub enable_ocr: bool,

    /// OCR language code (Tesseract format, e.g., "eng").
    pub ocr_language: String,

    /// Maximum document size to process (bytes).
    pub max_size: Option<u64>,

    /// Skip documents that fail to extract (return empty string instead of error).
    pub skip_on_error: bool,
}

impl Default for DocumentExtractorConfig {
    fn default() -> Self {
        Self {
            enable_ocr: false,
            ocr_language: "eng".to_string(),
            max_size: Some(100 * 1024 * 1024), // 100 MB default
            skip_on_error: false,
        }
    }
}

impl DocumentExtractorConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable OCR with the specified language.
    pub fn with_ocr(mut self, language: &str) -> Self {
        self.enable_ocr = true;
        self.ocr_language = language.to_string();
        self
    }

    /// Set maximum document size.
    pub fn with_max_size(mut self, bytes: u64) -> Self {
        self.max_size = Some(bytes);
        self
    }

    /// Disable size limit.
    pub fn no_size_limit(mut self) -> Self {
        self.max_size = None;
        self
    }

    /// Skip documents that fail to extract.
    pub fn skip_on_error(mut self, skip: bool) -> Self {
        self.skip_on_error = skip;
        self
    }
}

/// Main document extractor that routes to format-specific extractors.
#[derive(Debug, Clone)]
pub struct DocumentExtractor {
    config: DocumentExtractorConfig,
}

impl DocumentExtractor {
    /// Create a new document extractor with default config.
    pub fn new() -> Self {
        Self {
            config: DocumentExtractorConfig::default(),
        }
    }

    /// Create a document extractor with the given config.
    pub fn with_config(config: DocumentExtractorConfig) -> Self {
        Self { config }
    }

    /// Create a document extractor with OCR enabled.
    pub fn with_ocr(language: &str) -> Self {
        Self {
            config: DocumentExtractorConfig::default().with_ocr(language),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &DocumentExtractorConfig {
        &self.config
    }

    /// Extract text from document bytes, auto-detecting the format.
    ///
    /// # Arguments
    ///
    /// * `data` - The raw document bytes
    /// * `path` - Optional path for extension-based format detection
    ///
    /// # Returns
    ///
    /// The extracted text content, or an error.
    pub fn extract(&self, data: &[u8], path: Option<&Path>) -> GrepResult<String> {
        let format = detect_document_format(data, path);
        self.extract_format(data, format)
    }

    /// Extract text from document bytes with an explicit format.
    ///
    /// # Arguments
    ///
    /// * `data` - The raw document bytes
    /// * `format` - The document format
    ///
    /// # Returns
    ///
    /// The extracted text content, or an error.
    pub fn extract_format(&self, data: &[u8], format: DocumentFormat) -> GrepResult<String> {
        // Check size limit
        if let Some(max_size) = self.config.max_size {
            if data.len() as u64 > max_size {
                return Err(GrepError::DocumentExtraction {
                    file_path: std::path::PathBuf::from("<memory>"),
                    message: format!(
                        "Document too large: {} bytes (limit: {} bytes)",
                        data.len(),
                        max_size
                    ),
                });
            }
        }

        let result = match format {
            DocumentFormat::None => {
                return Err(GrepError::UnsupportedDocument(
                    "Not a recognized document format".to_string(),
                ));
            }
            DocumentFormat::Pdf => self.extract_pdf(data),
            DocumentFormat::Docx => self.extract_docx(data),
            DocumentFormat::Xlsx => self.extract_xlsx(data),
            DocumentFormat::Epub => self.extract_epub(data),
            DocumentFormat::Odt => self.extract_odt(data),
            DocumentFormat::DjVu => self.extract_djvu(data),
        };

        // Handle errors based on config
        match result {
            Ok(text) => Ok(text),
            Err(e) if self.config.skip_on_error => {
                // Log warning and return empty string
                eprintln!("Warning: Failed to extract document: {}", e);
                Ok(String::new())
            }
            Err(e) => Err(e),
        }
    }

    /// Extract text from PDF.
    #[allow(unused_variables)]
    fn extract_pdf(&self, data: &[u8]) -> GrepResult<String> {
        #[cfg(feature = "grep-pdf")]
        {
            let text = pdf::extract_text(data)?;

            // If no text extracted and OCR is enabled, try OCR
            #[cfg(feature = "grep-pdf-ocr")]
            if text.trim().is_empty() && self.config.enable_ocr {
                return pdf_ocr::extract_text_ocr(data, &self.config.ocr_language);
            }

            Ok(text)
        }

        #[cfg(not(feature = "grep-pdf"))]
        Err(GrepError::FeatureNotEnabled {
            feature: "grep-pdf".to_string(),
        })
    }

    /// Extract text from DOCX.
    #[allow(unused_variables)]
    fn extract_docx(&self, data: &[u8]) -> GrepResult<String> {
        #[cfg(feature = "grep-docx")]
        {
            docx::extract_text(data)
        }

        #[cfg(not(feature = "grep-docx"))]
        Err(GrepError::FeatureNotEnabled {
            feature: "grep-docx".to_string(),
        })
    }

    /// Extract text from XLSX.
    #[allow(unused_variables)]
    fn extract_xlsx(&self, data: &[u8]) -> GrepResult<String> {
        #[cfg(feature = "grep-xlsx")]
        {
            xlsx::extract_text(data)
        }

        #[cfg(not(feature = "grep-xlsx"))]
        Err(GrepError::FeatureNotEnabled {
            feature: "grep-xlsx".to_string(),
        })
    }

    /// Extract text from EPUB.
    #[allow(unused_variables)]
    fn extract_epub(&self, data: &[u8]) -> GrepResult<String> {
        #[cfg(feature = "grep-epub")]
        {
            epub::extract_text(data)
        }

        #[cfg(not(feature = "grep-epub"))]
        Err(GrepError::FeatureNotEnabled {
            feature: "grep-epub".to_string(),
        })
    }

    /// Extract text from ODT.
    #[allow(unused_variables)]
    fn extract_odt(&self, data: &[u8]) -> GrepResult<String> {
        #[cfg(feature = "grep-odt")]
        {
            odt::extract_text(data)
        }

        #[cfg(not(feature = "grep-odt"))]
        Err(GrepError::FeatureNotEnabled {
            feature: "grep-odt".to_string(),
        })
    }

    /// Extract text from DjVu.
    #[allow(unused_variables)]
    fn extract_djvu(&self, data: &[u8]) -> GrepResult<String> {
        djvu::extract_text(data)
    }
}

impl Default for DocumentExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_format_from_extension() {
        assert_eq!(
            DocumentFormat::from_extension(Path::new("test.pdf")),
            DocumentFormat::Pdf
        );
        assert_eq!(
            DocumentFormat::from_extension(Path::new("test.docx")),
            DocumentFormat::Docx
        );
        assert_eq!(
            DocumentFormat::from_extension(Path::new("test.xlsx")),
            DocumentFormat::Xlsx
        );
        assert_eq!(
            DocumentFormat::from_extension(Path::new("test.epub")),
            DocumentFormat::Epub
        );
        assert_eq!(
            DocumentFormat::from_extension(Path::new("test.odt")),
            DocumentFormat::Odt
        );
        assert_eq!(
            DocumentFormat::from_extension(Path::new("test.djvu")),
            DocumentFormat::DjVu
        );
        assert_eq!(
            DocumentFormat::from_extension(Path::new("test.txt")),
            DocumentFormat::None
        );
    }

    #[test]
    fn test_document_format_case_insensitive() {
        assert_eq!(DocumentFormat::from_extension_str("PDF"), DocumentFormat::Pdf);
        assert_eq!(DocumentFormat::from_extension_str("DOCX"), DocumentFormat::Docx);
        assert_eq!(DocumentFormat::from_extension_str("Epub"), DocumentFormat::Epub);
    }

    #[test]
    fn test_document_format_is_document() {
        assert!(DocumentFormat::Pdf.is_document());
        assert!(DocumentFormat::Docx.is_document());
        assert!(!DocumentFormat::None.is_document());
    }

    #[test]
    fn test_document_format_name() {
        assert_eq!(DocumentFormat::Pdf.name(), "PDF");
        assert_eq!(DocumentFormat::Docx.name(), "Microsoft Word");
        assert_eq!(DocumentFormat::Xlsx.name(), "Microsoft Excel");
    }

    #[test]
    fn test_document_extractor_config() {
        let config = DocumentExtractorConfig::new()
            .with_ocr("deu")
            .with_max_size(50 * 1024 * 1024)
            .skip_on_error(true);

        assert!(config.enable_ocr);
        assert_eq!(config.ocr_language, "deu");
        assert_eq!(config.max_size, Some(50 * 1024 * 1024));
        assert!(config.skip_on_error);
    }

    #[test]
    fn test_document_extractor_size_limit() {
        let extractor = DocumentExtractor::with_config(
            DocumentExtractorConfig::new().with_max_size(10),
        );

        // Large data should fail
        let large_data = vec![0u8; 100];
        let result = extractor.extract_format(&large_data, DocumentFormat::Pdf);
        assert!(result.is_err());

        if let Err(GrepError::DocumentExtraction { message, .. }) = result {
            assert!(message.contains("too large"));
        } else {
            panic!("Expected DocumentExtraction error");
        }
    }
}
