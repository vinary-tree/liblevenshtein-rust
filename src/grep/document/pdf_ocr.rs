//! PDF OCR text extraction using Poppler and Tesseract.
//!
//! This module provides OCR-based text extraction for image-based PDFs
//! where regular text extraction returns empty or minimal content.

use crate::grep::error::{GrepError, GrepResult};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const PDF_RENDERER: &str = "pdftoppm";
const TESSERACT_CLI: &str = "tesseract";
const RENDER_DPI: i32 = 200;

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
/// # Runtime Requirements
///
/// This path requires Poppler's `pdftoppm` executable for page rendering and
/// Tesseract language data for the requested `language`.
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

    let mut tesseract = LepTess::new(None, language).map_err(|e| {
        GrepError::OcrNotAvailable(format!(
            "Tesseract initialization failed for language '{}': {}. \
             Ensure Tesseract is installed and the language pack is available.",
            language, e
        ))
    })?;

    let temp_dir = tempfile::tempdir()?;
    let pdf_path = temp_dir.path().join("input.pdf");
    fs::write(&pdf_path, data)?;

    let image_prefix = temp_dir.path().join("page");
    render_pdf_pages(&pdf_path, &image_prefix)?;

    let page_images = rendered_page_images(temp_dir.path())?;
    if page_images.is_empty() {
        return Err(GrepError::DocumentEmpty(PathBuf::from("<memory>")));
    }

    let mut text = String::new();
    for (page_index, image_path) in page_images.iter().enumerate() {
        tesseract
            .set_image(image_path)
            .map_err(|e| GrepError::OcrError {
                file_path: image_path.clone(),
                message: format!("failed to load rendered page image for OCR: {e}"),
            })?;
        tesseract.set_source_resolution(RENDER_DPI);

        let page_text = tesseract.get_utf8_text().map_err(|e| GrepError::OcrError {
            file_path: image_path.clone(),
            message: format!("Tesseract returned invalid UTF-8: {e}"),
        })?;

        if page_index > 0 && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&page_text);
    }

    if text.trim().is_empty() {
        Err(GrepError::DocumentEmpty(PathBuf::from("<memory>")))
    } else {
        Ok(text)
    }
}

fn render_pdf_pages(pdf_path: &Path, image_prefix: &Path) -> GrepResult<()> {
    let output = Command::new(PDF_RENDERER)
        .arg("-q")
        .arg("-r")
        .arg(RENDER_DPI.to_string())
        .arg("-png")
        .arg("-forcenum")
        .arg(pdf_path)
        .arg(image_prefix)
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                GrepError::OcrNotAvailable(format!(
                    "PDF OCR requires Poppler's `{PDF_RENDERER}` executable to render pages"
                ))
            } else {
                GrepError::Io(e)
            }
        })?;

    if output.status.success() {
        Ok(())
    } else {
        Err(renderer_error(pdf_path, output))
    }
}

fn renderer_error(pdf_path: &Path, output: Output) -> GrepError {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let message = if stderr.trim().is_empty() {
        stdout.trim()
    } else {
        stderr.trim()
    };

    GrepError::OcrError {
        file_path: pdf_path.to_path_buf(),
        message: if message.is_empty() {
            format!("{PDF_RENDERER} failed with status {}", output.status)
        } else {
            format!(
                "{PDF_RENDERER} failed with status {}: {message}",
                output.status
            )
        },
    }
}

fn rendered_page_images(dir: &Path) -> GrepResult<Vec<PathBuf>> {
    let mut pages = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension() == Some(OsStr::new("png"))
            && path
                .file_stem()
                .and_then(OsStr::to_str)
                .is_some_and(|stem| stem.starts_with("page-"))
        {
            pages.push(path);
        }
    }

    pages.sort_by_key(|path| {
        path.file_stem()
            .and_then(OsStr::to_str)
            .and_then(|stem| stem.strip_prefix("page-"))
            .and_then(|num| num.parse::<usize>().ok())
            .unwrap_or(usize::MAX)
    });
    Ok(pages)
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

/// Check if Poppler's PDF page renderer is available.
///
/// # Returns
///
/// `true` if `pdftoppm` can be executed, `false` otherwise.
pub fn is_pdf_renderer_available() -> bool {
    Command::new(PDF_RENDERER)
        .arg("-v")
        .output()
        .is_ok_and(|output| output.status.success())
}

/// Get list of available Tesseract languages.
///
/// # Returns
///
/// A vector of available language codes, or an empty vector if Tesseract is not available.
pub fn available_languages() -> Vec<String> {
    let output = match Command::new(TESSERACT_CLI).arg("--list-langs").output() {
        Ok(output) if output.status.success() => output,
        _ => return Vec::new(),
    };

    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));

    combined
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("List of available languages"))
        .map(ToOwned::to_owned)
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
    fn test_pdf_renderer_availability_check() {
        let _ = is_pdf_renderer_available();
    }

    #[test]
    fn test_extract_text_ocr_error() {
        let fake_pdf = b"%PDF-1.4 fake pdf data";
        let result = extract_text_ocr(fake_pdf, "eng");
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_text_ocr_simple_pdf_when_tools_available() {
        if !is_pdf_renderer_available() || !is_ocr_available("eng") {
            return;
        }

        let pdf = simple_text_pdf("HELLO OCR");
        let text = extract_text_ocr(&pdf, "eng").expect("OCR should extract rendered PDF text");
        let normalized = text.to_uppercase();

        assert!(
            normalized.contains("HELLO") && normalized.contains("OCR"),
            "OCR output did not contain expected text: {text:?}"
        );
    }

    fn simple_text_pdf(text: &str) -> Vec<u8> {
        fn escape_pdf_text(text: &str) -> String {
            text.chars().fold(String::new(), |mut escaped, ch| {
                match ch {
                    '(' | ')' | '\\' => {
                        escaped.push('\\');
                        escaped.push(ch);
                    }
                    _ => escaped.push(ch),
                }
                escaped
            })
        }

        let stream = format!(
            "BT\n/F1 72 Tf\n72 500 Td\n({}) Tj\nET\n",
            escape_pdf_text(text)
        );
        let objects = [
            "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n".to_string(),
            "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n".to_string(),
            concat!(
                "3 0 obj\n",
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] ",
                "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\n",
                "endobj\n"
            )
            .to_string(),
            "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".to_string(),
            format!(
                "5 0 obj\n<< /Length {} >>\nstream\n{}endstream\nendobj\n",
                stream.len(),
                stream
            ),
        ];

        let mut pdf = Vec::from("%PDF-1.4\n".as_bytes());
        let mut offsets = vec![0usize];
        for object in objects {
            offsets.push(pdf.len());
            pdf.extend_from_slice(object.as_bytes());
        }

        let xref_start = pdf.len();
        pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for offset in offsets.iter().skip(1) {
            pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n",
                offsets.len()
            )
            .as_bytes(),
        );
        pdf
    }
}
