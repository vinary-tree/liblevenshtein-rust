//! Magic byte detection for document formats.
//!
//! This module provides detection of document formats by examining
//! file magic bytes (file signatures) at the start of the data.

use std::path::Path;

use super::DocumentFormat;

/// PDF magic bytes: %PDF
const PDF_MAGIC: &[u8] = b"%PDF";

/// ZIP magic bytes (DOCX, XLSX, ODT, EPUB are ZIP-based)
const ZIP_MAGIC: &[u8] = &[0x50, 0x4B, 0x03, 0x04];

/// DjVu magic bytes: AT&TFORM
const DJVU_MAGIC: &[u8] = b"AT&TFORM";

/// Minimum header size needed for reliable detection.
pub const MIN_HEADER_SIZE: usize = 8;

/// Detect document format from magic bytes and optional file path.
///
/// Uses a combination of magic byte detection and file extension
/// to determine the document format. Magic bytes take precedence
/// for formats that have unique signatures (PDF, DjVu), while
/// ZIP-based formats (DOCX, XLSX, EPUB, ODT) require extension
/// disambiguation since they all share the same ZIP header.
///
/// # Arguments
///
/// * `header` - The first few bytes of the file (at least 8 bytes recommended)
/// * `path` - Optional file path for extension-based disambiguation
///
/// # Returns
///
/// The detected `DocumentFormat`, or `DocumentFormat::None` if not recognized.
///
/// # Example
///
/// ```ignore
/// use liblevenshtein::grep::document::detection::detect_document_format;
/// use liblevenshtein::grep::document::DocumentFormat;
/// use std::path::Path;
///
/// // PDF detection by magic bytes
/// let pdf_header = b"%PDF-1.4";
/// assert_eq!(
///     detect_document_format(pdf_header, None),
///     DocumentFormat::Pdf
/// );
///
/// // DOCX detection (ZIP + extension)
/// let zip_header = &[0x50, 0x4B, 0x03, 0x04];
/// assert_eq!(
///     detect_document_format(zip_header, Some(Path::new("doc.docx"))),
///     DocumentFormat::Docx
/// );
/// ```
pub fn detect_document_format(header: &[u8], path: Option<&Path>) -> DocumentFormat {
    // Check magic bytes first (for formats with unique signatures)

    // PDF: %PDF
    if header.len() >= 4 && header.starts_with(PDF_MAGIC) {
        return DocumentFormat::Pdf;
    }

    // DjVu: AT&TFORM
    if header.len() >= 8 && header.starts_with(DJVU_MAGIC) {
        return DocumentFormat::DjVu;
    }

    // ZIP-based formats need extension disambiguation
    if header.len() >= 4 && header.starts_with(ZIP_MAGIC) {
        if let Some(path) = path {
            return match path.extension().and_then(|e| e.to_str()) {
                Some(ext) => match ext.to_lowercase().as_str() {
                    "docx" => DocumentFormat::Docx,
                    "xlsx" | "xls" => DocumentFormat::Xlsx,
                    "epub" => DocumentFormat::Epub,
                    "odt" => DocumentFormat::Odt,
                    _ => DocumentFormat::None,
                },
                None => DocumentFormat::None,
            };
        }
        // ZIP file but no path to disambiguate
        return DocumentFormat::None;
    }

    // Fallback to extension-only detection
    if let Some(path) = path {
        return DocumentFormat::from_extension(path);
    }

    DocumentFormat::None
}

/// Check if data appears to be a recognized document format.
///
/// Quick check without full format detection.
pub fn is_document(header: &[u8], path: Option<&Path>) -> bool {
    detect_document_format(header, path) != DocumentFormat::None
}

/// Get the list of document file extensions.
pub fn document_extensions() -> &'static [&'static str] {
    &["pdf", "docx", "xlsx", "xls", "epub", "odt", "djvu", "djv"]
}

/// Check if a file extension indicates a document format.
pub fn is_document_extension(ext: &str) -> bool {
    document_extensions().contains(&ext.to_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_pdf() {
        let header = b"%PDF-1.4\n%something";
        assert_eq!(detect_document_format(header, None), DocumentFormat::Pdf);
    }

    #[test]
    fn test_detect_djvu() {
        let header = b"AT&TFORMxxxxxxx";
        assert_eq!(detect_document_format(header, None), DocumentFormat::DjVu);
    }

    #[test]
    fn test_detect_docx() {
        let header = &[0x50, 0x4B, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(
            detect_document_format(header, Some(Path::new("document.docx"))),
            DocumentFormat::Docx
        );
    }

    #[test]
    fn test_detect_xlsx() {
        let header = &[0x50, 0x4B, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(
            detect_document_format(header, Some(Path::new("spreadsheet.xlsx"))),
            DocumentFormat::Xlsx
        );
    }

    #[test]
    fn test_detect_epub() {
        let header = &[0x50, 0x4B, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(
            detect_document_format(header, Some(Path::new("book.epub"))),
            DocumentFormat::Epub
        );
    }

    #[test]
    fn test_detect_odt() {
        let header = &[0x50, 0x4B, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(
            detect_document_format(header, Some(Path::new("document.odt"))),
            DocumentFormat::Odt
        );
    }

    #[test]
    fn test_zip_without_path() {
        // ZIP header without path should return None (can't disambiguate)
        let header = &[0x50, 0x4B, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(detect_document_format(header, None), DocumentFormat::None);
    }

    #[test]
    fn test_extension_only_pdf() {
        // No magic bytes, but PDF extension
        let header = b"garbage";
        assert_eq!(
            detect_document_format(header, Some(Path::new("test.pdf"))),
            DocumentFormat::Pdf
        );
    }

    #[test]
    fn test_plain_text() {
        let header = b"Hello, World!";
        assert_eq!(
            detect_document_format(header, Some(Path::new("test.txt"))),
            DocumentFormat::None
        );
    }

    #[test]
    fn test_empty_header() {
        let header: &[u8] = &[];
        assert_eq!(detect_document_format(header, None), DocumentFormat::None);
    }

    #[test]
    fn test_is_document() {
        let pdf = b"%PDF-1.4";
        assert!(is_document(pdf, None));

        let text = b"Hello";
        assert!(!is_document(text, Some(Path::new("test.txt"))));
    }

    #[test]
    fn test_is_document_extension() {
        assert!(is_document_extension("pdf"));
        assert!(is_document_extension("PDF"));
        assert!(is_document_extension("docx"));
        assert!(is_document_extension("XLSX"));
        assert!(!is_document_extension("txt"));
        assert!(!is_document_extension("rs"));
    }

    #[test]
    fn test_document_extensions() {
        let exts = document_extensions();
        assert!(exts.contains(&"pdf"));
        assert!(exts.contains(&"docx"));
        assert!(exts.contains(&"xlsx"));
        assert!(exts.contains(&"epub"));
        assert!(exts.contains(&"odt"));
        assert!(exts.contains(&"djvu"));
    }
}
