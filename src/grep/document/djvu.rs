//! DjVu text extraction (placeholder).
//!
//! DjVu support is not yet implemented. This module provides a placeholder
//! that returns an appropriate error message.

use crate::grep::error::{GrepError, GrepResult};

/// Extract text from DjVu document bytes.
///
/// **Note:** DjVu extraction is not yet implemented.
///
/// # Arguments
///
/// * `_data` - The raw DjVu file bytes (unused)
///
/// # Returns
///
/// Always returns an error indicating DjVu is not supported.
pub fn extract_text(_data: &[u8]) -> GrepResult<String> {
    Err(GrepError::UnsupportedDocument(
        "DjVu extraction is not yet implemented. Consider converting to PDF.".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_djvu_not_supported() {
        let data = b"AT&TFORM....";
        let result = extract_text(data);
        assert!(result.is_err());

        if let Err(GrepError::UnsupportedDocument(msg)) = result {
            assert!(msg.contains("not yet implemented"));
        } else {
            panic!("Expected UnsupportedDocument error");
        }
    }
}
