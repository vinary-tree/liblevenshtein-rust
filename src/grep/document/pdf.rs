//! PDF text extraction using pdf-extract.
//!
//! This module provides text extraction from PDF documents.
//! Stdout is suppressed during extraction to hide pdf-extract's font warnings.

use crate::grep::error::{GrepError, GrepResult};
use gag::Gag;

/// Extract text from PDF document bytes.
///
/// # Arguments
///
/// * `data` - The raw PDF file bytes
///
/// # Returns
///
/// The extracted text content, or an error if extraction fails.
///
/// # Behavior
///
/// Stdout is suppressed during extraction to hide pdf-extract's font mapping warnings
/// (e.g., "missing char 81 in unicode map..."). These warnings are informational and
/// do not affect extraction quality.
///
/// # Example
///
/// ```ignore
/// use liblevenshtein::grep::document::pdf::extract_text;
///
/// let pdf_bytes = std::fs::read("document.pdf")?;
/// let text = extract_text(&pdf_bytes)?;
/// println!("Extracted text: {}", text);
/// ```
pub fn extract_text(data: &[u8]) -> GrepResult<String> {
    // Suppress pdf-extract's stdout warnings during extraction
    let _gag = Gag::stdout().ok();

    pdf_extract::extract_text_from_mem(data).map_err(|e| GrepError::DocumentExtraction {
        file_path: std::path::PathBuf::from("<memory>"),
        message: format!("PDF extraction failed: {}", e),
    })
}

/// Extract text from a PDF file.
///
/// # Arguments
///
/// * `path` - Path to the PDF file
///
/// # Returns
///
/// The extracted text content, or an error if extraction fails.
///
/// # Behavior
///
/// Stdout is suppressed during extraction to hide pdf-extract's font mapping warnings.
pub fn extract_text_from_file(path: &std::path::Path) -> GrepResult<String> {
    // Suppress pdf-extract's stdout warnings during extraction
    let _gag = Gag::stdout().ok();

    pdf_extract::extract_text(path).map_err(|e| GrepError::DocumentExtraction {
        file_path: path.to_path_buf(),
        message: format!("PDF extraction failed: {}", e),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_invalid_pdf() {
        let invalid_data = b"This is not a PDF file";
        let result = extract_text(invalid_data);
        assert!(result.is_err());
    }

    // Integration tests with real PDF files belong in tests/.
}
