//! Shared flags infrastructure for phonetic pattern parsing.
//!
//! This module defines `ParsedFlags`, a shared structure used by both the LLRE
//! (Levenshtein Regex) and LLev (Levenshtein Level) parsers to represent
//! modifier flags that affect pattern matching behavior.
//!
//! # LLev vs LLRE Defaults
//!
//! - **LLev**: Case-insensitive by default (phonetic rules are about sound, not spelling)
//! - **LLRE**: Case-sensitive by default (regex semantics)
//!
//! # Flag Syntax
//!
//! Both parsers support scoped flag groups:
//! - LLRE: `(?flags:pattern)` - all flags supported
//! - LLev: `(?c:pattern)` or `(?-i:pattern)` - case-sensitive opt-out only

/// Parsed flags from a flag group like `(?i:...)` or `(?c:...)`.
///
/// Fields are `Option<bool>` where:
/// - `None` = flag not specified (use default)
/// - `Some(true)` = flag explicitly enabled
/// - `Some(false)` = flag explicitly disabled
///
/// # LLRE-Specific Fields
///
/// Some fields are only used by the LLRE parser:
/// - `unicode_normalization` - Unicode normalization form
/// - `feature_based` - Feature-based matching
/// - `multiline` - Multiline mode for `^` and `$`
/// - `dotall` - Allow `.` to match newlines
/// - `levenshtein_distance` - Local distance limit
///
/// These fields exist in the shared struct to avoid duplication, but LLev
/// parsers typically ignore them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParsedFlags {
    /// Case-insensitive flag (`i` or `-i`, or `c` for case-sensitive in LLev).
    ///
    /// - LLev: Defaults to `true` (case-insensitive), use `(?c:...)` to opt out
    /// - LLRE: Defaults to `false` (case-sensitive), use `(?i:...)` to enable
    pub case_insensitive: Option<bool>,

    /// Unicode normalization form (`u:NFC`, `u:NFD`, `u:NFKC`, `u:NFKD`).
    ///
    /// LLRE-only. Specifies how Unicode text should be normalized before matching.
    pub unicode_normalization: Option<String>,

    /// Feature-based matching flag (`f` or `-f`).
    ///
    /// LLRE-only. Enables matching based on phonetic feature bundles.
    pub feature_based: Option<bool>,

    /// Accent-insensitive flag (`a` or `-a`).
    ///
    /// When enabled, characters with diacritical marks match their base forms.
    /// For example, `é` matches `e`.
    pub accent_insensitive: Option<bool>,

    /// Multiline flag (`m` or `-m`).
    ///
    /// LLRE-only. When enabled, `^` and `$` match line boundaries, not just
    /// the start and end of the input.
    pub multiline: Option<bool>,

    /// Dotall flag (`s` or `-s`).
    ///
    /// LLRE-only. When enabled, `.` matches newline characters.
    pub dotall: Option<bool>,

    /// Local Levenshtein distance limit (`(?;N)` or `(?flags;N:pattern)`).
    ///
    /// LLRE-only. Limits the allowed edit distance for a specific pattern scope.
    pub levenshtein_distance: Option<u8>,
}

impl ParsedFlags {
    /// Create a new empty `ParsedFlags` with no flags set.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if any flags are set.
    ///
    /// Returns `true` if all fields are `None`.
    pub fn is_empty(&self) -> bool {
        self.case_insensitive.is_none()
            && self.unicode_normalization.is_none()
            && self.feature_based.is_none()
            && self.accent_insensitive.is_none()
            && self.multiline.is_none()
            && self.dotall.is_none()
            && self.levenshtein_distance.is_none()
    }

    /// Create flags with case-insensitive explicitly disabled.
    ///
    /// This is used by LLev for `(?c:...)` and `(?-i:...)` groups.
    #[inline]
    pub fn case_sensitive() -> Self {
        Self {
            case_insensitive: Some(false),
            ..Self::default()
        }
    }

    /// Create flags with case-insensitive explicitly enabled.
    ///
    /// This is used by LLRE for `(?i:...)` groups.
    #[inline]
    pub fn case_insensitive_enabled() -> Self {
        Self {
            case_insensitive: Some(true),
            ..Self::default()
        }
    }

    /// Merge with another set of flags, with `other` taking precedence.
    ///
    /// For each field, if `other` has a value, use it; otherwise keep `self`'s value.
    pub fn merge_with(&self, other: &ParsedFlags) -> ParsedFlags {
        ParsedFlags {
            case_insensitive: other.case_insensitive.or(self.case_insensitive),
            unicode_normalization: other
                .unicode_normalization
                .clone()
                .or_else(|| self.unicode_normalization.clone()),
            feature_based: other.feature_based.or(self.feature_based),
            accent_insensitive: other.accent_insensitive.or(self.accent_insensitive),
            multiline: other.multiline.or(self.multiline),
            dotall: other.dotall.or(self.dotall),
            levenshtein_distance: other.levenshtein_distance.or(self.levenshtein_distance),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_empty() {
        let flags = ParsedFlags::default();
        assert!(flags.is_empty());
    }

    #[test]
    fn test_case_sensitive_factory() {
        let flags = ParsedFlags::case_sensitive();
        assert!(!flags.is_empty());
        assert_eq!(flags.case_insensitive, Some(false));
    }

    #[test]
    fn test_case_insensitive_factory() {
        let flags = ParsedFlags::case_insensitive_enabled();
        assert!(!flags.is_empty());
        assert_eq!(flags.case_insensitive, Some(true));
    }

    #[test]
    fn test_merge_with() {
        let base = ParsedFlags {
            case_insensitive: Some(true),
            accent_insensitive: Some(false),
            ..Default::default()
        };

        let override_flags = ParsedFlags {
            case_insensitive: Some(false),
            multiline: Some(true),
            ..Default::default()
        };

        let merged = base.merge_with(&override_flags);

        // override takes precedence
        assert_eq!(merged.case_insensitive, Some(false));
        // base value preserved when not overridden
        assert_eq!(merged.accent_insensitive, Some(false));
        // new value from override
        assert_eq!(merged.multiline, Some(true));
        // unset in both remains None
        assert_eq!(merged.dotall, None);
    }

    #[test]
    fn test_is_empty_with_flags() {
        let mut flags = ParsedFlags::default();
        assert!(flags.is_empty());

        flags.case_insensitive = Some(true);
        assert!(!flags.is_empty());

        flags = ParsedFlags::default();
        flags.levenshtein_distance = Some(2);
        assert!(!flags.is_empty());
    }
}
