//! EPUB text extraction built on `zip` + `quick-xml`.
//!
//! An EPUB is a ZIP archive of XHTML documents plus a package (OPF) file that
//! declares the reading order. Everything needed to walk that structure is in the
//! spec, so this module implements it directly rather than depending on the `epub`
//! crate, which is **GPL-3.0** and would impose copyleft obligations on any binary
//! built with the `grep-epub` feature -- this crate ships Apache-2.0.
//!
//! The traversal is:
//!
//! 1. `META-INF/container.xml` names the package document via
//!    `<rootfile full-path="...">`.
//! 2. The package document's `<manifest>` maps item ids to hrefs, and its `<spine>`
//!    lists `<itemref idref="...">` in reading order.
//! 3. Each spine entry is read from the archive and stripped to plain text by
//!    `strip_html_tags`, which along with `decode_html_entity` is unchanged --
//!    those were always this crate's own code, not the dependency's.
//!
//! Hrefs in the manifest are relative to the package document, so they are resolved
//! against its directory before lookup.

use super::text_output::{push_section, write_section};
use crate::grep::error::{GrepError, GrepResult};
use crate::grep::limited_read::read_to_vec_limited;
use std::borrow::Cow;

fn extraction_error(message: impl Into<String>) -> GrepError {
    GrepError::DocumentExtraction {
        file_path: std::path::PathBuf::from("<memory>"),
        message: message.into(),
    }
}

/// Read one archive member fully, honouring the shared size guard.
fn read_member<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
) -> GrepResult<String> {
    let mut file = archive
        .by_name(name)
        .map_err(|e| extraction_error(format!("EPUB entry `{name}` not found: {e}")))?;
    let size = file.size();
    let bytes = read_to_vec_limited(&mut file, Some(size), None, name)
        .map_err(|e| extraction_error(format!("Failed to read `{name}`: {e}")))?;
    String::from_utf8(bytes)
        .map_err(|e| extraction_error(format!("`{name}` is not valid UTF-8: {e}")))
}

/// Value of the first matching attribute on the first matching element.
fn first_attr(xml: &str, element: &str, attr: &str) -> Option<String> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                if local_name(e.name().as_ref()) == element.as_bytes() {
                    for a in e.attributes().flatten() {
                        if local_name(a.key.as_ref()) == attr.as_bytes() {
                            return String::from_utf8(a.value.into_owned()).ok();
                        }
                    }
                }
            }
            Ok(Event::Eof) | Err(_) => return None,
            _ => {}
        }
        buf.clear();
    }
}

/// Strip any XML namespace prefix, so `opf:item` and `item` both match.
fn local_name(name: &[u8]) -> &[u8] {
    match name.iter().rposition(|&b| b == b':') {
        Some(i) => &name[i + 1..],
        None => name,
    }
}

/// Resolve an href that is relative to the package document's directory.
fn resolve_href(package_path: &str, href: &str) -> String {
    let base = match package_path.rfind('/') {
        Some(i) => &package_path[..=i],
        None => "",
    };
    let joined = format!("{base}{href}");
    // Normalise `a/../b` segments, which EPUBs in the wild do use.
    let mut parts: Vec<&str> = Vec::new();
    for seg in joined.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

/// The package document: reading-order hrefs plus title and author, if declared.
struct Package {
    spine_hrefs: Vec<String>,
    title: Option<String>,
    creator: Option<String>,
}

/// Parse the OPF package document: manifest (id -> href), spine order, metadata.
fn parse_package(package_path: &str, xml: &str) -> GrepResult<Package> {
    use quick_xml::events::Event;
    use quick_xml::Reader;
    use std::collections::HashMap;

    let mut manifest: HashMap<String, String> = HashMap::new();
    let mut spine_ids: Vec<String> = Vec::new();
    let mut title = None;
    let mut creator = None;

    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    // Which Dublin Core element we are inside, so its text can be captured.
    let mut pending_meta: Option<&'static str> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                let name = local_name(e.name().as_ref()).to_vec();
                match name.as_slice() {
                    b"item" => {
                        let mut id = None;
                        let mut href = None;
                        for a in e.attributes().flatten() {
                            match local_name(a.key.as_ref()) {
                                b"id" => id = String::from_utf8(a.value.into_owned()).ok(),
                                b"href" => href = String::from_utf8(a.value.into_owned()).ok(),
                                _ => {}
                            }
                        }
                        if let (Some(id), Some(href)) = (id, href) {
                            manifest.insert(id, href);
                        }
                    }
                    b"itemref" => {
                        for a in e.attributes().flatten() {
                            if local_name(a.key.as_ref()) == b"idref" {
                                if let Ok(v) = String::from_utf8(a.value.into_owned()) {
                                    spine_ids.push(v);
                                }
                            }
                        }
                    }
                    b"title" => pending_meta = Some("title"),
                    b"creator" => pending_meta = Some("creator"),
                    _ => {}
                }
            }
            Ok(Event::Text(t)) => {
                if let Some(which) = pending_meta.take() {
                    if let Ok(text) = t.xml10_content() {
                        let text = text.trim().to_string();
                        if !text.is_empty() {
                            match which {
                                "title" => title.get_or_insert(text),
                                _ => creator.get_or_insert(text),
                            };
                        }
                    }
                }
            }
            Ok(Event::End(_)) => pending_meta = None,
            Ok(Event::Eof) => break,
            Err(e) => return Err(extraction_error(format!("Malformed package document: {e}"))),
            _ => {}
        }
        buf.clear();
    }

    let spine_hrefs = spine_ids
        .iter()
        .filter_map(|id| manifest.get(id))
        .map(|href| resolve_href(package_path, href))
        .collect();

    Ok(Package {
        spine_hrefs,
        title,
        creator,
    })
}

/// Extract text from EPUB document bytes.
///
/// # Arguments
///
/// * `data` - The raw EPUB file bytes
///
/// # Returns
///
/// The extracted text content (all chapters concatenated in reading order), or an
/// error if extraction fails.
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
    use std::io::Cursor;
    use zip::ZipArchive;

    let mut archive = ZipArchive::new(Cursor::new(data))
        .map_err(|e| extraction_error(format!("EPUB archive is invalid: {e}")))?;

    // (1) container.xml points at the package document.
    let container = read_member(&mut archive, "META-INF/container.xml")?;
    let package_path = first_attr(&container, "rootfile", "full-path").ok_or_else(|| {
        extraction_error("EPUB container.xml declares no <rootfile full-path=...>")
    })?;

    // (2) The package document gives reading order and metadata.
    let package_xml = read_member(&mut archive, &package_path)?;
    let package = parse_package(&package_path, &package_xml)?;

    let mut all_text = String::new();
    if let Some(title) = &package.title {
        write_section(&mut all_text, format_args!("Title: {title}"));
    }
    if let Some(creator) = &package.creator {
        write_section(&mut all_text, format_args!("Author: {creator}"));
    }
    if !all_text.is_empty() {
        push_section(&mut all_text, ""); // Separator after metadata
    }

    // (3) Each spine document, stripped to plain text. A single unreadable chapter
    // is skipped rather than failing the whole book -- partial text is more useful
    // than none, and the empty-output check below still catches a total failure.
    for href in &package.spine_hrefs {
        let Ok(content) = read_member(&mut archive, href) else {
            continue;
        };
        let plain_text = strip_html_tags(&content);
        if !plain_text.trim().is_empty() {
            push_section(&mut all_text, &plain_text);
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

    /// Build a minimal but spec-shaped EPUB in memory: mimetype, container.xml
    /// pointing at `OEBPS/content.opf`, a manifest/spine naming two chapters in
    /// reading order, and Dublin Core title/creator.
    fn build_epub() -> Vec<u8> {
        use std::io::Write;
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            let opts: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);

            let add = |zip: &mut zip::ZipWriter<_>, name: &str, body: &str| {
                zip.start_file(name, opts).expect("start_file");
                zip.write_all(body.as_bytes()).expect("write");
            };

            add(&mut zip, "mimetype", "application/epub+zip");
            add(
                &mut zip,
                "META-INF/container.xml",
                r#"<?xml version="1.0"?>
                   <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
                     <rootfiles><rootfile full-path="OEBPS/content.opf"
                       media-type="application/oebps-package+xml"/></rootfiles>
                   </container>"#,
            );
            add(
                &mut zip,
                "OEBPS/content.opf",
                r#"<?xml version="1.0"?>
                   <package xmlns="http://www.idpf.org/2007/opf" version="3.0">
                     <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                       <dc:title>Test Book</dc:title>
                       <dc:creator>A. Author</dc:creator>
                     </metadata>
                     <manifest>
                       <item id="c2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
                       <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
                     </manifest>
                     <spine><itemref idref="c1"/><itemref idref="c2"/></spine>
                   </package>"#,
            );
            add(
                &mut zip,
                "OEBPS/ch1.xhtml",
                "<html><body><p>First chapter &amp; friends</p></body></html>",
            );
            add(
                &mut zip,
                "OEBPS/ch2.xhtml",
                "<html><body><p>Second chapter</p><script>ignored()</script></body></html>",
            );
            zip.finish().expect("finish");
        }
        buf.into_inner()
    }

    #[test]
    fn extracts_metadata_and_chapters_in_spine_order() {
        let text = extract_text(&build_epub()).expect("extract");
        assert!(text.contains("Title: Test Book"), "got: {text}");
        assert!(text.contains("Author: A. Author"), "got: {text}");
        assert!(
            text.contains("First chapter & friends"),
            "entities decoded: {text}"
        );
        assert!(text.contains("Second chapter"));
        assert!(
            !text.contains("ignored()"),
            "script bodies must be dropped: {text}"
        );

        // Spine order wins over manifest order: ch2 is declared first in the
        // manifest but second in the spine.
        let first = text.find("First chapter").expect("ch1 present");
        let second = text.find("Second chapter").expect("ch2 present");
        assert!(
            first < second,
            "chapters must follow the spine, not the manifest"
        );
    }

    #[test]
    fn resolves_hrefs_relative_to_the_package_document() {
        // The package lives in OEBPS/, so `ch1.xhtml` must resolve to OEBPS/ch1.xhtml.
        assert_eq!(
            resolve_href("OEBPS/content.opf", "ch1.xhtml"),
            "OEBPS/ch1.xhtml"
        );
        assert_eq!(resolve_href("content.opf", "ch1.xhtml"), "ch1.xhtml");
        assert_eq!(
            resolve_href("OEBPS/sub/content.opf", "../text/ch1.xhtml"),
            "OEBPS/text/ch1.xhtml"
        );
    }

    #[test]
    fn namespace_prefixes_are_ignored_when_matching_elements() {
        assert_eq!(local_name(b"opf:item"), b"item");
        assert_eq!(local_name(b"item"), b"item");
        assert_eq!(local_name(b"dc:title"), b"title");
    }

    #[test]
    fn missing_container_is_an_error_not_a_panic() {
        use std::io::Write;
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            let opts: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default();
            zip.start_file("mimetype", opts).expect("start_file");
            zip.write_all(b"application/epub+zip").expect("write");
            zip.finish().expect("finish");
        }
        assert!(extract_text(&buf.into_inner()).is_err());
    }
}
