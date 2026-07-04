//! DOCX text extraction using docx-rs.
//!
//! This module provides text extraction from Microsoft Word (.docx) documents.
//! DOCX files are ZIP archives containing XML files with the document content.

use super::text_output::begin_line;
use crate::grep::error::{GrepError, GrepResult};

/// Extract text from DOCX document bytes.
///
/// # Arguments
///
/// * `data` - The raw DOCX file bytes
///
/// # Returns
///
/// The extracted text content, or an error if extraction fails.
///
/// # Example
///
/// ```ignore
/// use liblevenshtein::grep::document::docx::extract_text;
///
/// let docx_bytes = std::fs::read("document.docx")?;
/// let text = extract_text(&docx_bytes)?;
/// println!("Extracted text: {}", text);
/// ```
pub fn extract_text(data: &[u8]) -> GrepResult<String> {
    use docx_rs::read_docx;

    let docx = read_docx(data).map_err(|e| GrepError::DocumentExtraction {
        file_path: std::path::PathBuf::from("<memory>"),
        message: format!("DOCX parsing failed: {:?}", e),
    })?;

    let mut text = String::new();

    // Extract text from document body
    for child in &docx.document.children {
        extract_text_from_content(child, &mut text);
    }

    Ok(text)
}

/// Recursively extract text from DOCX content elements.
fn extract_text_from_content(content: &docx_rs::DocumentChild, output: &mut String) {
    match content {
        docx_rs::DocumentChild::Paragraph(para) => {
            append_paragraph_text(&para.children, output);
        }
        docx_rs::DocumentChild::Table(table) => {
            for row in &table.rows {
                extract_text_from_table_row(row, output);
            }
        }
        _ => {}
    }
}

/// Extract text from a table row.
fn extract_text_from_table_row(row: &docx_rs::TableChild, output: &mut String) {
    // `docx_rs::TableChild` currently has only the `TableRow` variant; the
    // `if let` is defensive against upstream additions.
    #[allow(irrefutable_let_patterns)]
    if let docx_rs::TableChild::TableRow(tr) = row {
        for cell in &tr.cells {
            extract_text_from_table_cell(cell, output);
        }
    }
}

/// Extract text from a table cell.
fn extract_text_from_table_cell(cell: &docx_rs::TableRowChild, output: &mut String) {
    // `docx_rs::TableRowChild` currently has only the `TableCell` variant; the
    // `if let` is defensive against upstream additions.
    #[allow(irrefutable_let_patterns)]
    if let docx_rs::TableRowChild::TableCell(tc) = cell {
        for child in &tc.children {
            if let docx_rs::TableCellContent::Paragraph(para) = child {
                append_paragraph_text(&para.children, output);
            }
        }
    }
}

fn append_paragraph_text(children: &[docx_rs::ParagraphChild], output: &mut String) {
    let line_start = output.len();
    begin_line(output);
    let text_start = output.len();

    for child in children {
        extract_text_from_paragraph_child(child, output);
    }

    if output.len() == text_start {
        output.truncate(line_start);
    }
}

/// Extract text from paragraph children.
fn extract_text_from_paragraph_child(child: &docx_rs::ParagraphChild, text: &mut String) {
    match child {
        docx_rs::ParagraphChild::Run(run) => {
            for run_child in &run.children {
                if let docx_rs::RunChild::Text(t) = run_child {
                    text.push_str(&t.text);
                }
            }
        }
        docx_rs::ParagraphChild::Hyperlink(link) => {
            for run in &link.children {
                if let docx_rs::ParagraphChild::Run(r) = run {
                    for run_child in &r.children {
                        if let docx_rs::RunChild::Text(t) = run_child {
                            text.push_str(&t.text);
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// Extract text from a DOCX file.
///
/// # Arguments
///
/// * `path` - Path to the DOCX file
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

    fn paragraph_child_text(text: &str) -> docx_rs::ParagraphChild {
        let mut run = docx_rs::Run::default();
        run.children.push(docx_rs::RunChild::Text(docx_rs::Text {
            text: text.to_string(),
            preserve_space: true,
        }));
        docx_rs::ParagraphChild::Run(Box::new(run))
    }

    #[test]
    fn test_extract_invalid_docx() {
        let invalid_data = b"This is not a DOCX file";
        let result = extract_text(invalid_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_empty_zip() {
        // A minimal ZIP file (empty) - should fail as invalid DOCX
        let empty_zip = [
            0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let result = extract_text(&empty_zip);
        assert!(result.is_err());
    }

    #[test]
    fn test_append_paragraph_text_appends_directly() {
        let mut output = String::new();
        append_paragraph_text(&[paragraph_child_text("alpha")], &mut output);
        append_paragraph_text(&[paragraph_child_text("beta")], &mut output);

        assert_eq!(output, "alpha\nbeta");
    }

    #[test]
    fn test_append_paragraph_text_rolls_back_empty_paragraph() {
        let mut output = String::from("alpha");
        append_paragraph_text(&[], &mut output);

        assert_eq!(output, "alpha");
    }
}
