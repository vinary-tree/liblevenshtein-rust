//! Magic byte detection for compression formats.
//!
//! This module provides detection of compression formats by examining
//! file magic bytes (file signatures) at the start of the data.

use super::CompressionFormat;

/// Magic bytes for gzip format (RFC 1952).
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Magic bytes for zstd format.
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

/// Magic bytes for XZ format.
const XZ_MAGIC: [u8; 6] = [0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00];

/// Magic bytes for LZMA format (standalone .lzma files).
/// LZMA files start with properties byte, typically 0x5d for default settings.
const LZMA_MAGIC: [u8; 2] = [0x5d, 0x00];

/// Magic bytes for bzip2 format.
const BZIP2_MAGIC: [u8; 3] = [0x42, 0x5a, 0x68]; // "BZh"

/// Minimum header size needed for detection.
pub const MIN_HEADER_SIZE: usize = 6;

/// Detect compression format from magic bytes.
///
/// Examines the first few bytes of data to determine the compression format.
/// Returns `CompressionFormat::None` if no known format is detected.
///
/// # Arguments
///
/// * `header` - The first few bytes of the file (at least 6 bytes recommended)
///
/// # Example
///
/// ```
/// use liblevenshtein::grep::compression::detection::detect_from_magic;
/// use liblevenshtein::grep::CompressionFormat;
///
/// // Gzip magic bytes
/// let gzip_header = [0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00];
/// assert_eq!(detect_from_magic(&gzip_header), CompressionFormat::Gzip);
///
/// // Plain text
/// let text_header = b"Hello, World!";
/// assert_eq!(detect_from_magic(text_header), CompressionFormat::None);
/// ```
pub fn detect_from_magic(header: &[u8]) -> CompressionFormat {
    // Check in order of magic byte length (longer first for specificity)

    // XZ: 6 bytes
    if header.len() >= 6 && header[..6] == XZ_MAGIC {
        return CompressionFormat::Xz;
    }

    // Zstd: 4 bytes
    if header.len() >= 4 && header[..4] == ZSTD_MAGIC {
        return CompressionFormat::Zstd;
    }

    // Bzip2: 3 bytes
    if header.len() >= 3 && header[..3] == BZIP2_MAGIC {
        return CompressionFormat::Bzip2;
    }

    // Gzip: 2 bytes
    if header.len() >= 2 && header[..2] == GZIP_MAGIC {
        return CompressionFormat::Gzip;
    }

    // LZMA: 2 bytes (less reliable, check last)
    if header.len() >= 2 && header[..2] == LZMA_MAGIC {
        return CompressionFormat::Xz; // LZMA uses same decompressor
    }

    CompressionFormat::None
}

/// Check if data appears to be binary (non-text).
///
/// Uses heuristics to detect binary content:
/// - Presence of NUL bytes
/// - High proportion of non-printable characters
///
/// # Arguments
///
/// * `data` - A sample of the data to check (first 512-8192 bytes recommended)
///
/// # Returns
///
/// `true` if the data appears to be binary, `false` if it appears to be text.
pub fn is_binary(data: &[u8]) -> bool {
    // NUL byte is a strong indicator of binary
    if data.contains(&0) {
        return true;
    }

    // Check for high proportion of non-printable ASCII
    let non_text_count = data
        .iter()
        .filter(|&&b| {
            // Non-printable and not common whitespace
            b < 0x20 && b != b'\n' && b != b'\r' && b != b'\t'
        })
        .count();

    // If more than 10% non-text bytes, likely binary
    if data.len() > 10 && non_text_count * 10 > data.len() {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gzip() {
        let header = [0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(detect_from_magic(&header), CompressionFormat::Gzip);
    }

    #[test]
    fn test_detect_zstd() {
        let header = [0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x00];
        assert_eq!(detect_from_magic(&header), CompressionFormat::Zstd);
    }

    #[test]
    fn test_detect_xz() {
        let header = [0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00];
        assert_eq!(detect_from_magic(&header), CompressionFormat::Xz);
    }

    #[test]
    fn test_detect_bzip2() {
        let header = [0x42, 0x5a, 0x68, 0x39, 0x31, 0x41];
        assert_eq!(detect_from_magic(&header), CompressionFormat::Bzip2);
    }

    #[test]
    fn test_detect_plain_text() {
        let header = b"Hello, World!";
        assert_eq!(detect_from_magic(header), CompressionFormat::None);
    }

    #[test]
    fn test_detect_empty() {
        let header: &[u8] = &[];
        assert_eq!(detect_from_magic(header), CompressionFormat::None);
    }

    #[test]
    fn test_is_binary_with_nul() {
        let data = b"hello\x00world";
        assert!(is_binary(data));
    }

    #[test]
    fn test_is_binary_plain_text() {
        let data = b"Hello, World!\nThis is plain text.\n";
        assert!(!is_binary(data));
    }

    #[test]
    fn test_is_binary_with_tabs() {
        let data = b"column1\tcolumn2\tcolumn3\n";
        assert!(!is_binary(data));
    }

    #[test]
    fn test_is_binary_high_non_printable() {
        // More than 10% non-printable (need > 10 bytes for this check to trigger)
        let data = [
            0x01, 0x02, 0x03, b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h',
        ];
        assert!(is_binary(&data));
    }
}
