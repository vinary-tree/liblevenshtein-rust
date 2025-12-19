//! EPUB text extraction using epub crate.
//!
//! This module provides text extraction from EPUB e-books.
//! EPUB files are ZIP archives containing XHTML content files.

use crate::grep::error::{GrepError, GrepResult};

/// Extract text from EPUB document bytes.
///
/// # Arguments
///
/// * `data` - The raw EPUB file bytes
///
/// # Returns
///
/// The extracted text content (all chapters concatenated), or an error if extraction fails.
///
/// # Example
///
/// ```ignore
/// use liblevenshtein::grep::document::epub::extract_text;
///
/// let epub_bytes = std::fs::read("book.epub")?;
/// let text = extract_text(&epub_bytes)?;
/// println!("Extracted text: {}", text);
/// ```
pub fn extract_text(data: &[u8]) -> GrepResult<String> {
    use epub::doc::EpubDoc;
    use std::io::Cursor;

    let cursor = Cursor::new(data.to_vec());
    let mut doc = EpubDoc::from_reader(cursor).map_err(|e| GrepError::DocumentExtraction {
        file_path: std::path::PathBuf::from("<memory>"),
        message: format!("EPUB parsing failed: {}", e),
    })?;

    let mut all_text = Vec::new();

    // Get metadata for context - handle MetadataItem type
    if let Some(title) = doc.mdata("title") {
        all_text.push(format!("Title: {:?}", title));
    }
    if let Some(author) = doc.mdata("creator") {
        all_text.push(format!("Author: {:?}", author));
    }
    if !all_text.is_empty() {
        all_text.push(String::new()); // Separator after metadata
    }

    // Iterate through spine (reading order)
    // The epub crate requires iterating through resources
    while doc.go_next() {
        if let Some((content, _mime)) = doc.get_current_str() {
            // Content is HTML, strip tags to get plain text
            let plain_text = strip_html_tags(&content);
            if !plain_text.trim().is_empty() {
                all_text.push(plain_text);
            }
        }
    }

    if all_text.is_empty() {
        return Err(GrepError::DocumentEmpty(std::path::PathBuf::from(
            "<memory>",
        )));
    }

    Ok(all_text.join("\n\n"))
}

/// Strip HTML tags from content to get plain text.
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut last_was_space = true;

    let chars: Vec<char> = html.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if c == '<' {
            // Check for script/style tags
            let remaining: String = chars[i..].iter().take(10).collect();
            let remaining_lower = remaining.to_lowercase();

            if remaining_lower.starts_with("<script") {
                in_script = true;
            } else if remaining_lower.starts_with("</script") {
                in_script = false;
            } else if remaining_lower.starts_with("<style") {
                in_style = true;
            } else if remaining_lower.starts_with("</style") {
                in_style = false;
            }

            in_tag = true;
        } else if c == '>' {
            in_tag = false;
        } else if !in_tag && !in_script && !in_style {
            // Handle HTML entities
            if c == '&' {
                let entity_end = chars[i..].iter().position(|&x| x == ';');
                if let Some(end) = entity_end {
                    let entity: String = chars[i..=i + end].iter().collect();
                    let decoded = decode_html_entity(&entity);
                    if decoded == " " || decoded == "\n" {
                        if !last_was_space {
                            result.push(' ');
                            last_was_space = true;
                        }
                    } else {
                        result.push_str(&decoded);
                        last_was_space = false;
                    }
                    i += end;
                } else {
                    result.push(c);
                    last_was_space = false;
                }
            } else if c.is_whitespace() {
                if !last_was_space {
                    result.push(' ');
                    last_was_space = true;
                }
            } else {
                result.push(c);
                last_was_space = false;
            }
        }

        i += 1;
    }

    result.trim().to_string()
}

/// Decode common HTML entities.
fn decode_html_entity(entity: &str) -> String {
    match entity {
        "&nbsp;" | "&#160;" => " ".to_string(),
        "&lt;" | "&#60;" => "<".to_string(),
        "&gt;" | "&#62;" => ">".to_string(),
        "&amp;" | "&#38;" => "&".to_string(),
        "&quot;" | "&#34;" => "\"".to_string(),
        "&apos;" | "&#39;" => "'".to_string(),
        "&mdash;" | "&#8212;" => "\u{2014}".to_string(),
        "&ndash;" | "&#8211;" => "\u{2013}".to_string(),
        "&hellip;" | "&#8230;" => "\u{2026}".to_string(),
        "&ldquo;" | "&#8220;" => "\u{201C}".to_string(),
        "&rdquo;" | "&#8221;" => "\u{201D}".to_string(),
        "&lsquo;" | "&#8216;" => "\u{2018}".to_string(),
        "&rsquo;" | "&#8217;" => "\u{2019}".to_string(),
        "&copy;" | "&#169;" => "\u{00A9}".to_string(),
        "&reg;" | "&#174;" => "\u{00AE}".to_string(),
        _ => {
            // Try to decode numeric entities
            if entity.starts_with("&#x") && entity.ends_with(';') {
                let hex = &entity[3..entity.len() - 1];
                if let Ok(code) = u32::from_str_radix(hex, 16) {
                    if let Some(c) = char::from_u32(code) {
                        return c.to_string();
                    }
                }
            } else if entity.starts_with("&#") && entity.ends_with(';') {
                let num = &entity[2..entity.len() - 1];
                if let Ok(code) = num.parse::<u32>() {
                    if let Some(c) = char::from_u32(code) {
                        return c.to_string();
                    }
                }
            }
            entity.to_string()
        }
    }
}

/// Extract text from an EPUB file.
///
/// # Arguments
///
/// * `path` - Path to the EPUB file
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
    fn test_strip_html_tags() {
        let html = "<html><body><p>Hello <b>World</b>!</p></body></html>";
        let text = strip_html_tags(html);
        assert_eq!(text, "Hello World!");
    }

    #[test]
    fn test_strip_html_with_entities() {
        let html = "<p>Hello&nbsp;World &amp; Friends</p>";
        let text = strip_html_tags(html);
        assert_eq!(text, "Hello World & Friends");
    }

    #[test]
    fn test_decode_html_entity() {
        assert_eq!(decode_html_entity("&amp;"), "&");
        assert_eq!(decode_html_entity("&lt;"), "<");
        assert_eq!(decode_html_entity("&#65;"), "A");
        assert_eq!(decode_html_entity("&#x41;"), "A");
    }

    #[test]
    fn test_extract_invalid_epub() {
        let invalid_data = b"This is not an EPUB file";
        let result = extract_text(invalid_data);
        assert!(result.is_err());
    }
}
