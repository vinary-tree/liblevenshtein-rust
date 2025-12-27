//! Character unit abstraction for dictionary edges.
//!
//! This module provides the [`CharUnit`] trait, which abstracts over byte-level
//! (u8) and character-level (char) operations. This allows dictionaries to operate
//! at either granularity, trading performance for Unicode correctness.

/// Trait abstracting character unit types for dictionary edges.
///
/// This trait allows dictionaries to operate at byte-level ([`u8`]) for maximum
/// performance with ASCII/Latin-1 text, or character-level ([`char`]) for proper
/// Unicode support.
///
/// # Performance Trade-offs
///
/// - **Byte-level (u8)**: 1 byte per edge, fastest, but treats multi-byte UTF-8
///   sequences as multiple characters. Distance of 1 from "a" won't reach "é" (2 bytes).
///
/// - **Character-level (char)**: 4 bytes per edge, ~15% slower, but correct Unicode
///   semantics. Distance of 1 from "a" correctly reaches "é" (1 character).
///
/// # Example
///
/// ```rust,ignore
/// // Byte-level dictionary (existing behavior)
/// let dict_bytes: DoubleArrayTrie = DoubleArrayTrie::from_terms(vec!["café"]);
///
/// // Character-level dictionary (proper Unicode)
/// let dict_chars: DoubleArrayTrieChar = DoubleArrayTrieChar::from_terms(vec!["café"]);
/// ```
pub trait CharUnit:
    Copy + Clone + Eq + PartialEq + std::hash::Hash + std::fmt::Debug + Send + Sync + 'static
{
    /// Convert from a string slice to a vector of units.
    ///
    /// For `u8`, this extracts the UTF-8 bytes.
    /// For `char`, this extracts the Unicode scalar values.
    fn from_str(s: &str) -> Vec<Self>;

    /// Convert from a slice of units back to a string.
    ///
    /// For `u8`, this uses lossy UTF-8 decoding (invalid sequences become �).
    /// For `char`, this is lossless.
    fn to_string(units: &[Self]) -> String;

    /// Create an iterator over the units in a string.
    ///
    /// For `u8`, iterates over bytes.
    /// For `char`, iterates over Unicode scalar values.
    fn iter_str(s: &str) -> Box<dyn Iterator<Item = Self> + '_>;
}

/// Byte-level implementation (existing behavior).
///
/// This is the default and recommended for ASCII/Latin-1 content.
/// Provides best performance but treats multi-byte UTF-8 sequences as
/// multiple units.
impl CharUnit for u8 {
    #[inline]
    fn from_str(s: &str) -> Vec<Self> {
        s.as_bytes().to_vec()
    }

    #[inline]
    fn to_string(units: &[Self]) -> String {
        String::from_utf8_lossy(units).into_owned()
    }

    #[inline]
    fn iter_str(s: &str) -> Box<dyn Iterator<Item = Self> + '_> {
        Box::new(s.bytes())
    }
}

/// Character-level implementation (Unicode-aware).
///
/// This provides proper Unicode semantics where edit distance is measured in
/// characters rather than bytes. Use when working with non-ASCII text.
impl CharUnit for char {
    #[inline]
    fn from_str(s: &str) -> Vec<Self> {
        s.chars().collect()
    }

    #[inline]
    fn to_string(units: &[Self]) -> String {
        units.iter().collect()
    }

    #[inline]
    fn iter_str(s: &str) -> Box<dyn Iterator<Item = Self> + '_> {
        Box::new(s.chars())
    }
}

/// 64-bit unit implementation (8 bytes per edge).
///
/// This is primarily intended for use with u64 token sequences (vocabulary IDs,
/// hash values) or f64 bit patterns for time series indexing. The string
/// conversion methods pack/unpack bytes in little-endian order for trait
/// compatibility, but the primary API for `DynamicDawgU64` uses direct
/// sequence operations (`insert_sequence`, `contains_sequence`, etc.).
///
/// # Use Cases
///
/// - **Token sequences**: Vocabulary IDs, hash-based tokens
/// - **Time series**: f64 values encoded via `f64::to_bits()` / `f64::from_bits()`
/// - **Binary data**: Any 8-byte aligned data
///
/// # String Encoding
///
/// When used with strings (for trait compatibility), bytes are packed into u64s
/// in little-endian order, 8 bytes per u64. Trailing bytes in the last u64 are
/// zero-padded. This is a secondary use case; prefer `insert_sequence` for u64 data.
impl CharUnit for u64 {
    #[inline]
    fn from_str(s: &str) -> Vec<Self> {
        let bytes = s.as_bytes();
        if bytes.is_empty() {
            return Vec::new();
        }
        bytes
            .chunks(8)
            .map(|chunk| {
                let mut arr = [0u8; 8];
                arr[..chunk.len()].copy_from_slice(chunk);
                u64::from_le_bytes(arr)
            })
            .collect()
    }

    #[inline]
    fn to_string(units: &[Self]) -> String {
        if units.is_empty() {
            return String::new();
        }
        let bytes: Vec<u8> = units.iter().flat_map(|&u| u.to_le_bytes()).collect();
        // Trim trailing zeros (padding from from_str)
        let end = bytes
            .iter()
            .rposition(|&b| b != 0)
            .map(|i| i + 1)
            .unwrap_or(0);
        String::from_utf8_lossy(&bytes[..end]).into_owned()
    }

    #[inline]
    fn iter_str(s: &str) -> Box<dyn Iterator<Item = Self> + '_> {
        Box::new(Self::from_str(s).into_iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u8_ascii() {
        let s = "hello";
        let units = u8::from_str(s);
        assert_eq!(units, vec![b'h', b'e', b'l', b'l', b'o']);
        assert_eq!(<u8 as CharUnit>::to_string(&units), s);
    }

    #[test]
    fn test_u8_unicode() {
        let s = "café";
        let units = u8::from_str(s);
        // 'é' is 2 bytes in UTF-8: 0xC3 0xA9
        assert_eq!(units.len(), 5); // c, a, f, é (2 bytes)
        assert_eq!(<u8 as CharUnit>::to_string(&units), s);
    }

    #[test]
    fn test_char_ascii() {
        let s = "hello";
        let units = char::from_str(s);
        assert_eq!(units, vec!['h', 'e', 'l', 'l', 'o']);
        assert_eq!(<char as CharUnit>::to_string(&units), s);
    }

    #[test]
    fn test_char_unicode() {
        let s = "café";
        let units = char::from_str(s);
        // Proper character-level: 4 characters
        assert_eq!(units, vec!['c', 'a', 'f', 'é']);
        assert_eq!(units.len(), 4);
        assert_eq!(<char as CharUnit>::to_string(&units), s);
    }

    #[test]
    fn test_char_emoji() {
        let s = "hello 🎉 world";
        let units = char::from_str(s);
        assert_eq!(units.len(), 13); // 13 characters including emoji
        assert!(units.contains(&'🎉'));
        assert_eq!(<char as CharUnit>::to_string(&units), s);
    }

    #[test]
    fn test_char_cjk() {
        let s = "中文";
        let units = char::from_str(s);
        assert_eq!(units, vec!['中', '文']);
        assert_eq!(units.len(), 2);
        assert_eq!(<char as CharUnit>::to_string(&units), s);
    }

    #[test]
    fn test_iter_u8() {
        let s = "hi";
        let collected: Vec<u8> = u8::iter_str(s).collect();
        assert_eq!(collected, vec![b'h', b'i']);
    }

    #[test]
    fn test_iter_char() {
        let s = "café";
        let collected: Vec<char> = <char as CharUnit>::iter_str(s).collect();
        assert_eq!(collected, vec!['c', 'a', 'f', 'é']);
    }

    #[test]
    fn test_u64_short_string() {
        let s = "hello";
        let units = u64::from_str(s);
        // "hello" is 5 bytes, fits in one u64 (padded with zeros)
        assert_eq!(units.len(), 1);
        assert_eq!(<u64 as CharUnit>::to_string(&units), s);
    }

    #[test]
    fn test_u64_exact_8_bytes() {
        let s = "12345678";
        let units = u64::from_str(s);
        assert_eq!(units.len(), 1);
        assert_eq!(<u64 as CharUnit>::to_string(&units), s);
    }

    #[test]
    fn test_u64_multi_unit() {
        let s = "hello world!"; // 12 bytes -> 2 u64s
        let units = u64::from_str(s);
        assert_eq!(units.len(), 2);
        assert_eq!(<u64 as CharUnit>::to_string(&units), s);
    }

    #[test]
    fn test_u64_empty() {
        let s = "";
        let units = u64::from_str(s);
        assert!(units.is_empty());
        assert_eq!(<u64 as CharUnit>::to_string(&units), s);
    }

    #[test]
    fn test_u64_unicode() {
        let s = "café"; // 5 bytes in UTF-8
        let units = u64::from_str(s);
        assert_eq!(units.len(), 1);
        assert_eq!(<u64 as CharUnit>::to_string(&units), s);
    }

    #[test]
    fn test_iter_u64() {
        let s = "hello world!"; // 12 bytes -> 2 u64s
        let collected: Vec<u64> = u64::iter_str(s).collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(<u64 as CharUnit>::to_string(&collected), s);
    }
}
