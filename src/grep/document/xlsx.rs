//! XLSX text extraction using calamine.
//!
//! This module provides text extraction from Microsoft Excel (.xlsx, .xls) spreadsheets.
//! It extracts cell values from all sheets, converting them to text.

use crate::grep::error::{GrepError, GrepResult};

/// Extract text from XLSX/XLS document bytes.
///
/// # Arguments
///
/// * `data` - The raw spreadsheet file bytes
///
/// # Returns
///
/// The extracted text content (all cells, tab-separated within rows, newline-separated between rows),
/// or an error if extraction fails.
///
/// # Example
///
/// ```ignore
/// use liblevenshtein::grep::document::xlsx::extract_text;
///
/// let xlsx_bytes = std::fs::read("spreadsheet.xlsx")?;
/// let text = extract_text(&xlsx_bytes)?;
/// println!("Extracted text: {}", text);
/// ```
pub fn extract_text(data: &[u8]) -> GrepResult<String> {
    use calamine::{open_workbook_auto_from_rs, Reader};
    use std::io::Cursor;

    let cursor = Cursor::new(data);
    let mut workbook =
        open_workbook_auto_from_rs(cursor).map_err(|e| GrepError::DocumentExtraction {
            file_path: std::path::PathBuf::from("<memory>"),
            message: format!("Spreadsheet parsing failed: {}", e),
        })?;

    let mut all_text = Vec::new();

    // Get all sheet names first
    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();

    for sheet_name in sheet_names {
        if let Ok(range) = workbook.worksheet_range(&sheet_name) {
            // Add sheet name as header
            all_text.push(format!("=== {} ===", sheet_name));

            for row in range.rows() {
                let row_text: Vec<String> = row.iter().map(|cell| cell.to_string()).collect();

                // Skip completely empty rows
                if row_text.iter().all(|s| s.is_empty()) {
                    continue;
                }

                all_text.push(row_text.join("\t"));
            }

            all_text.push(String::new()); // Empty line between sheets
        }
    }

    Ok(all_text.join("\n"))
}

/// Extract text from a spreadsheet file.
///
/// # Arguments
///
/// * `path` - Path to the spreadsheet file
///
/// # Returns
///
/// The extracted text content, or an error if extraction fails.
pub fn extract_text_from_file(path: &std::path::Path) -> GrepResult<String> {
    let data = std::fs::read(path).map_err(GrepError::Io)?;
    extract_text(&data)
}

/// Extract text from a specific sheet.
///
/// # Arguments
///
/// * `data` - The raw spreadsheet file bytes
/// * `sheet_name` - Name of the sheet to extract
///
/// # Returns
///
/// The extracted text content from the specified sheet, or an error if extraction fails.
pub fn extract_text_from_sheet(data: &[u8], sheet_name: &str) -> GrepResult<String> {
    use calamine::{open_workbook_auto_from_rs, Reader};
    use std::io::Cursor;

    let cursor = Cursor::new(data);
    let mut workbook =
        open_workbook_auto_from_rs(cursor).map_err(|e| GrepError::DocumentExtraction {
            file_path: std::path::PathBuf::from("<memory>"),
            message: format!("Spreadsheet parsing failed: {}", e),
        })?;

    let range = workbook
        .worksheet_range(sheet_name)
        .map_err(|e| GrepError::DocumentExtraction {
            file_path: std::path::PathBuf::from("<memory>"),
            message: format!("Sheet '{}' not found or invalid: {}", sheet_name, e),
        })?;

    let mut all_text = Vec::new();

    for row in range.rows() {
        let row_text: Vec<String> = row.iter().map(|cell| cell.to_string()).collect();

        // Skip completely empty rows
        if row_text.iter().all(|s| s.is_empty()) {
            continue;
        }

        all_text.push(row_text.join("\t"));
    }

    Ok(all_text.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_invalid_xlsx() {
        let invalid_data = b"This is not an XLSX file";
        let result = extract_text(invalid_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_empty_data() {
        let result = extract_text(&[]);
        assert!(result.is_err());
    }
}
