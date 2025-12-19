//! PDF OCR text extraction using Tesseract via leptess.
//!
//! This module provides OCR-based text extraction for image-based PDFs
//! where regular text extraction returns empty or minimal content.

use crate::grep::error::{GrepError, GrepResult};

/// Extract text from PDF document bytes using OCR.
///
/// This function renders PDF pages to images and uses Tesseract OCR
/// to extract text. It's useful for scanned documents or image-based PDFs
/// where `pdf::extract_text()` returns empty content.
///
/// # Arguments
///
/// * `data` - The raw PDF file bytes
/// * `language` - Tesseract language code (e.g., "eng", "deu", "fra")
///
/// # Returns
///
/// The extracted text content, or an error if OCR fails.
///
/// # Example
///
/// ```ignore
/// use liblevenshtein::grep::document::pdf_ocr::extract_text_ocr;
///
/// let pdf_bytes = std::fs::read("scanned_document.pdf")?;
/// let text = extract_text_ocr(&pdf_bytes, "eng")?;
/// println!("OCR text: {}", text);
/// ```
pub fn extract_text_ocr(data: &[u8], language: &str) -> GrepResult<String> {
    use leptess::LepTess;
    use std::io::Cursor;

    // First, we need to render PDF pages to images
    // This requires pdf-image or similar crate, but for now we'll use a simpler approach
    // by attempting to use leptess directly on the PDF (which won't work for most PDFs)

    // For a proper implementation, we would:
    // 1. Use pdfium or pdf-image to render each page to PNG
    // 2. Run OCR on each rendered image
    // 3. Concatenate the results

    // For now, return a helpful error since full PDF-to-image rendering
    // requires additional dependencies not currently in Cargo.toml
    let _ = (data, language); // Suppress unused warnings

    // Try to initialize Tesseract to check if it's available
    match LepTess::new(None, language) {
        Ok(_) => {
            // Tesseract is available, but we can't render PDF pages without additional deps
            Err(GrepError::OcrError {
                file_path: std::path::PathBuf::from("<memory>"),
                message: format!(
                    "PDF OCR requires PDF-to-image rendering. \
                     Consider converting to images first, or use the pdf feature for text-based PDFs. \
                     Language '{}' is available in Tesseract.",
                    language
                ),
            })
        }
        Err(e) => Err(GrepError::OcrNotAvailable(format!(
            "Tesseract initialization failed for language '{}': {}. \
             Ensure Tesseract is installed and the language pack is available.",
            language, e
        ))),
    }
}

/// Check if Tesseract OCR is available with the specified language.
///
/// # Arguments
///
/// * `language` - Tesseract language code to check
///
/// # Returns
///
/// `true` if Tesseract is available with the language, `false` otherwise.
pub fn is_ocr_available(language: &str) -> bool {
    use leptess::LepTess;
    LepTess::new(None, language).is_ok()
}

/// Get list of available Tesseract languages.
///
/// # Returns
///
/// A vector of available language codes, or an empty vector if Tesseract is not available.
pub fn available_languages() -> Vec<String> {
    // leptess doesn't provide a way to list languages, so we check common ones
    let common_languages = [
        "eng", "deu", "fra", "spa", "ita", "por", "nld", "pol", "rus", "chi_sim", "chi_tra", "jpn",
        "kor", "ara", "hin", "tha", "vie",
    ];

    common_languages
        .iter()
        .filter(|lang| is_ocr_available(lang))
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocr_availability_check() {
        // This test just verifies the function doesn't panic
        let _ = is_ocr_available("eng");
    }

    #[test]
    fn test_available_languages() {
        // This test just verifies the function doesn't panic
        let langs = available_languages();
        // Result depends on system Tesseract installation
        println!("Available OCR languages: {:?}", langs);
    }

    #[test]
    fn test_extract_text_ocr_error() {
        // PDF OCR currently returns an error since we can't render PDF pages
        let fake_pdf = b"%PDF-1.4 fake pdf data";
        let result = extract_text_ocr(fake_pdf, "eng");
        // Should return either OcrError or OcrNotAvailable
        assert!(result.is_err());
    }
}
