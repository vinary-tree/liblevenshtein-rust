//! ODT text extraction using quick-xml and zip.
//!
//! This module provides text extraction from OpenDocument Text (.odt) files.
//! ODT files are ZIP archives containing an XML file (content.xml) with the document content.

use crate::grep::error::{GrepError, GrepResult};
use crate::grep::limited_read::read_to_vec_limited;

use super::text_output::push_line;

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
    extract_text_limited(data, None)
}

/// Extract text from ODT document bytes, enforcing a limit on `content.xml`.
pub fn extract_text_limited(data: &[u8], max_content_xml_size: Option<u64>) -> GrepResult<String> {
    use quick_xml::events::Event;
    use quick_xml::Reader;
    use std::io::Cursor;
    use zip::ZipArchive;

    // Open the ODT as a ZIP archive
    let cursor = Cursor::new(data);
    let mut archive = ZipArchive::new(cursor).map_err(|e| GrepError::DocumentExtraction {
        file_path: std::path::PathBuf::from("<memory>"),
        message: format!("ODT archive is invalid: {}", e),
    })?;

    // Find and read content.xml
    let content_xml = {
        let mut content_file =
            archive
                .by_name("content.xml")
                .map_err(|e| GrepError::DocumentExtraction {
                    file_path: std::path::PathBuf::from("<memory>"),
                    message: format!("content.xml not found in ODT: {}", e),
                })?;
        let content_size = content_file.size();
        let content_bytes = read_to_vec_limited(
            &mut content_file,
            Some(content_size),
            max_content_xml_size,
            "content.xml",
        )
        .map_err(|e| GrepError::DocumentExtraction {
            file_path: std::path::PathBuf::from("<memory>"),
            message: format!("Failed to read content.xml: {}", e),
        })?;

        String::from_utf8(content_bytes).map_err(|e| GrepError::DocumentExtraction {
            file_path: std::path::PathBuf::from("<memory>"),
            message: format!("content.xml is not valid UTF-8: {}", e),
        })?
    };

    // Parse the XML and extract text
    let mut reader = Reader::from_str(&content_xml);
    reader.config_mut().trim_text(true);

    let mut output = String::new();
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
                    // quick-xml 0.41 removed `BytesText::unescape`; the reader now
                    // yields raw (escaped) bytes. Decode the charset, then resolve
                    // XML entities via the free `escape::unescape` to preserve the
                    // previous behavior.
                    if let Ok(decoded) = e.decode() {
                        if let Ok(text) = quick_xml::escape::unescape(&decoded) {
                            current_text.push_str(&text);
                        }
                    }
                }
            }
            Ok(Event::End(ref e)) => {
                let local_name = e.local_name();
                let name = std::str::from_utf8(local_name.as_ref()).unwrap_or("");

                // End of paragraph or heading - add to results
                if name == "p" || name == "h" {
                    if !current_text.is_empty() {
                        push_line(&mut output, &current_text);
                        current_text.clear();
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
        push_line(&mut output, &current_text);
    }

    if output.is_empty() {
        return Err(GrepError::DocumentEmpty(std::path::PathBuf::from(
            "<memory>",
        )));
    }

    Ok(output)
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
    use std::io::{Cursor, Write};

    fn odt_content_xml_body(body_xml: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text>{body_xml}</office:text></office:body></office:document-content>"#
        )
    }

    fn odt_content_xml(body: &str) -> String {
        odt_content_xml_body(&format!(r#"<text:p>{body}</text:p>"#))
    }

    fn build_odt(content_xml: &str) -> Vec<u8> {
        let cursor = Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);

        zip.start_file("content.xml", options)
            .expect("start content.xml");
        zip.write_all(content_xml.as_bytes())
            .expect("write content.xml");
        zip.finish().expect("finish odt").into_inner()
    }

    #[test]
    fn test_extract_valid_odt() {
        let data = build_odt(&odt_content_xml("Hello World"));

        let text = extract_text(&data).expect("extract ODT text");

        assert_eq!(text, "Hello World");
    }

    #[test]
    fn test_extract_multiple_odt_paragraphs() {
        let data = build_odt(&odt_content_xml_body(
            "<text:p>First paragraph</text:p><text:p>Second paragraph</text:p>",
        ));

        let text = extract_text(&data).expect("extract ODT paragraphs");

        assert_eq!(text, "First paragraph\nSecond paragraph");
    }

    #[test]
    fn test_extract_text_limited_rejects_oversized_content_xml() {
        let body = "a".repeat(4096);
        let content_xml = odt_content_xml(&body);
        let data = build_odt(&content_xml);

        let err = extract_text_limited(&data, Some(128))
            .expect_err("content.xml should exceed the configured limit");

        assert!(
            matches!(err, GrepError::DocumentExtraction { ref message, .. } if message.contains("content.xml") && message.contains("too large"))
        );
    }

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
