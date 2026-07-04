//! EPUB text extraction using epub crate.
//!
//! This module provides text extraction from EPUB e-books.
//! EPUB files are ZIP archives containing XHTML content files.

use super::text_output::{push_section, write_section};
use crate::grep::error::{GrepError, GrepResult};
use std::borrow::Cow;

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

    let cursor = Cursor::new(data);
    let mut doc = EpubDoc::from_reader(cursor).map_err(|e| GrepError::DocumentExtraction {
        file_path: std::path::PathBuf::from("<memory>"),
        message: format!("EPUB parsing failed: {}", e),
    })?;

    let mut all_text = String::new();

    // Get metadata for context - handle MetadataItem type
    if let Some(title) = doc.mdata("title") {
        write_section(&mut all_text, format_args!("Title: {title:?}"));
    }
    if let Some(author) = doc.mdata("creator") {
        write_section(&mut all_text, format_args!("Author: {author:?}"));
    }
    if !all_text.is_empty() {
        push_section(&mut all_text, ""); // Separator after metadata
    }

    // Iterate through spine (reading order)
    // The epub crate requires iterating through resources
    while doc.go_next() {
        if let Some((content, _mime)) = doc.get_current_str() {
            // Content is HTML, strip tags to get plain text
            let plain_text = strip_html_tags(&content);
            if !plain_text.trim().is_empty() {
                push_section(&mut all_text, &plain_text);
            }
        }
    }

    if all_text.is_empty() {
        return Err(GrepError::DocumentEmpty(std::path::PathBuf::from(
            "<memory>",
        )));
    }

    Ok(all_text)
}

/// Strip HTML tags from content to get plain text.
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut last_was_space = true;

    let mut i = 0;

    while i < html.len() {
        let remaining = &html[i..];
        let c = remaining
            .chars()
            .next()
            .expect("index is advanced only along UTF-8 character boundaries");

        if c == '<' {
            // Check for script/style tags
            if starts_with_ignore_ascii_case(remaining, "<script") {
                in_script = true;
            } else if starts_with_ignore_ascii_case(remaining, "</script") {
                in_script = false;
            } else if starts_with_ignore_ascii_case(remaining, "<style") {
                in_style = true;
            } else if starts_with_ignore_ascii_case(remaining, "</style") {
                in_style = false;
            }

            in_tag = true;
        } else if c == '>' {
            in_tag = false;
        } else if !in_tag && !in_script && !in_style {
            // Handle HTML entities
            if c == '&' {
                let entity_end = remaining.find(';');
                if let Some(end) = entity_end {
                    let entity = &remaining[..=end];
                    let decoded = decode_html_entity(entity);
                    if decoded == " " || decoded == "\n" {
                        if !last_was_space {
                            result.push(' ');
                            last_was_space = true;
                        }
                    } else {
                        result.push_str(&decoded);
                        last_was_space = false;
                    }
                    i += end + 1;
                    continue;
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

        i += c.len_utf8();
    }

    if result.ends_with(' ') {
        result.pop();
    }

    result
}

fn starts_with_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    haystack
        .as_bytes()
        .get(..needle.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(needle.as_bytes()))
}

/// Decode common HTML entities.
fn decode_html_entity(entity: &str) -> Cow<'static, str> {
    match entity {
        "&nbsp;" | "&#160;" => Cow::Borrowed(" "),
        "&lt;" | "&#60;" => Cow::Borrowed("<"),
        "&gt;" | "&#62;" => Cow::Borrowed(">"),
        "&amp;" | "&#38;" => Cow::Borrowed("&"),
        "&quot;" | "&#34;" => Cow::Borrowed("\""),
        "&apos;" | "&#39;" => Cow::Borrowed("'"),
        "&mdash;" | "&#8212;" => Cow::Borrowed("\u{2014}"),
        "&ndash;" | "&#8211;" => Cow::Borrowed("\u{2013}"),
        "&hellip;" | "&#8230;" => Cow::Borrowed("\u{2026}"),
        "&ldquo;" | "&#8220;" => Cow::Borrowed("\u{201C}"),
        "&rdquo;" | "&#8221;" => Cow::Borrowed("\u{201D}"),
        "&lsquo;" | "&#8216;" => Cow::Borrowed("\u{2018}"),
        "&rsquo;" | "&#8217;" => Cow::Borrowed("\u{2019}"),
        "&copy;" | "&#169;" => Cow::Borrowed("\u{00A9}"),
        "&reg;" | "&#174;" => Cow::Borrowed("\u{00AE}"),
        _ => {
            // Try to decode numeric entities
            if entity.starts_with("&#x") && entity.ends_with(';') {
                let hex = &entity[3..entity.len() - 1];
                if let Ok(code) = u32::from_str_radix(hex, 16) {
                    if let Some(c) = char::from_u32(code) {
                        return Cow::Owned(c.to_string());
                    }
                }
            } else if entity.starts_with("&#") && entity.ends_with(';') {
                let num = &entity[2..entity.len() - 1];
                if let Ok(code) = num.parse::<u32>() {
                    if let Some(c) = char::from_u32(code) {
                        return Cow::Owned(c.to_string());
                    }
                }
            }
            Cow::Owned(entity.to_string())
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
