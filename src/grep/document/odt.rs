//! ODT text extraction using quick-xml and zip.
//!
//! This module provides text extraction from OpenDocument Text (.odt) files.
//! ODT files are ZIP archives containing an XML file (content.xml) with the document content.

use crate::grep::error::{GrepError, GrepResult};

/// Extract text from ODT document bytes.
///
/// # Arguments
///
/// * `data` - The raw ODT file bytes
///
/// # Returns
///
/// The extracted text content, or an error if extraction fails.
///
/// # Example
///
/// ```ignore
/// use liblevenshtein::grep::document::odt::extract_text;
///
/// let odt_bytes = std::fs::read("document.odt")?;
/// let text = extract_text(&odt_bytes)?;
/// println!("Extracted text: {}", text);
/// ```
pub fn extract_text(data: &[u8]) -> GrepResult<String> {
    use quick_xml::events::Event;
    use quick_xml::Reader;
    use std::io::{Cursor, Read};
    use zip::ZipArchive;

    // Open the ODT as a ZIP archive
    let cursor = Cursor::new(data);
    let mut archive = ZipArchive::new(cursor).map_err(|e| GrepError::DocumentExtraction {
        file_path: std::path::PathBuf::from("<memory>"),
        message: format!("ODT archive is invalid: {}", e),
    })?;

    // Find and read content.xml
    let mut content_xml = String::new();
    {
        let mut content_file =
            archive
                .by_name("content.xml")
                .map_err(|e| GrepError::DocumentExtraction {
                    file_path: std::path::PathBuf::from("<memory>"),
                    message: format!("content.xml not found in ODT: {}", e),
                })?;
        content_file.read_to_string(&mut content_xml).map_err(|e| {
            GrepError::DocumentExtraction {
                file_path: std::path::PathBuf::from("<memory>"),
                message: format!("Failed to read content.xml: {}", e),
            }
        })?;
    }

    // Parse the XML and extract text
    let mut reader = Reader::from_str(&content_xml);
    reader.config_mut().trim_text(true);

    let mut text_parts = Vec::new();
    let mut current_text = String::new();
    let mut in_text_element = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let local_name = e.local_name();
                let name = std::str::from_utf8(local_name.as_ref()).unwrap_or("");

                // Track when we're in text elements
                // ODT uses text:p for paragraphs, text:h for headings, text:span for spans
                if name == "p" || name == "h" || name == "span" {
                    in_text_element = true;
                }

                // Handle line breaks and tabs
                if name == "line-break" {
                    current_text.push('\n');
                } else if name == "tab" {
                    current_text.push('\t');
                } else if name == "s" {
                    // text:s is a space element, count attribute gives number of spaces
                    let count = get_space_count(e);
                    current_text.push_str(&" ".repeat(count));
                }
            }
            Ok(Event::Text(ref e)) => {
                if in_text_element {
                    if let Ok(text) = e.unescape() {
                        current_text.push_str(&text);
                    }
                }
            }
            Ok(Event::End(ref e)) => {
                let local_name = e.local_name();
                let name = std::str::from_utf8(local_name.as_ref()).unwrap_or("");

                // End of paragraph or heading - add to results
                if name == "p" || name == "h" {
                    if !current_text.is_empty() {
                        text_parts.push(std::mem::take(&mut current_text));
                    }
                    in_text_element = false;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(GrepError::DocumentExtraction {
                    file_path: std::path::PathBuf::from("<memory>"),
                    message: format!("XML parsing error: {}", e),
                });
            }
            _ => {}
        }
    }

    // Add any remaining text
    if !current_text.is_empty() {
        text_parts.push(current_text);
    }

    if text_parts.is_empty() {
        return Err(GrepError::DocumentEmpty(std::path::PathBuf::from(
            "<memory>",
        )));
    }

    Ok(text_parts.join("\n"))
}

/// Get the space count from a text:s element's c attribute.
fn get_space_count(e: &quick_xml::events::BytesStart) -> usize {
    for attr in e.attributes().flatten() {
        let local_name = attr.key.local_name();
        let key = std::str::from_utf8(local_name.as_ref()).unwrap_or("");
        if key == "c" {
            if let Ok(val) = std::str::from_utf8(&attr.value) {
                if let Ok(count) = val.parse::<usize>() {
                    return count;
                }
            }
        }
    }
    1 // Default to 1 space if no count specified
}

/// Extract text from an ODT file.
///
/// # Arguments
///
/// * `path` - Path to the ODT file
///
/// # Returns
///
/// The extracted text content, or an error if extraction fails.
pub fn extract_text_from_file(path: &std::path::Path) -> GrepResult<String> {
    let data = std::fs::read(path).map_err(GrepError::Io)?;
    extract_text(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_invalid_odt() {
        let invalid_data = b"This is not an ODT file";
        let result = extract_text(invalid_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_empty_zip() {
        // A minimal ZIP file (empty) - should fail as content.xml is missing
        let empty_zip = [
            0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let result = extract_text(&empty_zip);
        assert!(result.is_err());
    }
}
