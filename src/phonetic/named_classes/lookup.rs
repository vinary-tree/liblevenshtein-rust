//! Name-resolution helpers for named phonetic classes.
//!
//! Resolves a (case-insensitive) class name to its [`NamedClass`] in the
//! [`registry`](super::registry), normalising the name with a stack-allocated lowercase
//! buffer for current built-ins and a heap fallback for longer ASCII names.

// ============================================================================
// Lookup Functions
// ============================================================================

use super::registry::{NamedClass, NAMED_CLASSES};

/// Maximum length of any built-in class name.
/// Used for the stack-allocated lowercase fast path.
const MAX_CLASS_NAME_LEN: usize = 20; // "nasalized_diacritic" = 20 chars

enum NormalizedClassName {
    Inline {
        buf: [u8; MAX_CLASS_NAME_LEN],
        len: usize,
    },
    Owned(String),
}

impl NormalizedClassName {
    #[inline]
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::Inline { buf, len } => std::str::from_utf8(&buf[..*len]).ok(),
            Self::Owned(name) => Some(name.as_str()),
        }
    }
}

/// Normalize a class name to lowercase using stack allocation.
///
/// Returns None if the name contains non-ASCII characters. Current built-in
/// names use the inline buffer; longer ASCII names fall back to a heap string
/// so future registry additions are not silently unreachable.
#[inline]
fn normalize_class_name(name: &str) -> Option<NormalizedClassName> {
    let len = name.len();
    if !name.is_ascii() {
        return None;
    }

    if len > MAX_CLASS_NAME_LEN {
        return Some(NormalizedClassName::Owned(name.to_ascii_lowercase()));
    }

    let mut buf = [0u8; MAX_CLASS_NAME_LEN];
    let bytes = name.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        buf[i] = b.to_ascii_lowercase();
    }

    Some(NormalizedClassName::Inline { buf, len })
}

/// Look up a named class by name (case-insensitive).
///
/// Returns the named class definition if found.
///
/// This function uses a stack-allocated buffer for known built-in names and a
/// heap fallback for longer ASCII inputs.
///
/// # Example
///
/// ```
/// use liblevenshtein::phonetic::named_classes::get_named_class;
///
/// let vowels = get_named_class("vowel").expect("vowel class exists");
/// assert!(vowels.patterns.len() > 0);
///
/// // Case-insensitive
/// let vowels2 = get_named_class("VOWEL").expect("case insensitive");
/// assert_eq!(vowels.patterns.len(), vowels2.patterns.len());
///
/// // Full word aliases (e.g., "plosive" for "stop", "semivowel" for "glide")
/// let stops = get_named_class("plosive").expect("alias works");
/// assert_eq!(get_named_class("stop").unwrap().patterns.len(), stops.patterns.len());
/// ```
pub fn get_named_class(name: &str) -> Option<&'static NamedClass> {
    // Fast path: exact match (common for lowercase names)
    if let Some(class) = NAMED_CLASSES.get(name) {
        return Some(class);
    }

    let normalized = normalize_class_name(name)?;
    let lowered = normalized.as_str()?;
    NAMED_CLASSES.get(lowered)
}

/// Check if a name is a built-in class (case-insensitive).
///
/// This is useful for detecting conflicts when users try to define
/// symbols with the same name as built-in classes.
///
/// Uses stack-allocated case conversion for known built-in names and a heap
/// fallback for longer ASCII inputs.
pub fn is_builtin_class(name: &str) -> bool {
    // Fast path: exact match
    if NAMED_CLASSES.contains_key(name) {
        return true;
    }

    let Some(normalized) = normalize_class_name(name) else {
        return false;
    };

    let Some(lowered) = normalized.as_str() else {
        return false;
    };
    NAMED_CLASSES.contains_key(lowered)
}

/// Get all built-in class names (for error messages and documentation).
pub fn all_builtin_class_names() -> Vec<&'static str> {
    let mut names: Vec<_> = NAMED_CLASSES.keys().copied().collect();
    names.sort_unstable();
    names
}

/// Get only the single-character patterns from a named class.
///
/// Useful when you need to build a simple character class without digraphs.
pub fn get_chars_only(name: &str) -> Option<Vec<char>> {
    get_named_class(name).map(|class| {
        let mut chars = Vec::with_capacity(class.patterns.len());
        chars.extend(class.patterns.iter().filter_map(|p| p.as_char()));
        chars
    })
}

/// Get only the digraph patterns from a named class.
pub fn get_digraphs_only(name: &str) -> Option<Vec<(char, char)>> {
    get_named_class(name).map(|class| {
        let mut digraphs = Vec::with_capacity(class.patterns.len());
        digraphs.extend(class.patterns.iter().filter_map(|p| p.as_digraph()));
        digraphs
    })
}

/// Get only the trigraph patterns from a named class.
///
/// Returns None if the class doesn't exist, or a Vec of (char, char, char) tuples
/// for trigraph patterns like ejective affricates.
///
/// # Example
/// ```ignore
/// let ejective_trigraphs = get_trigraphs_only("ejective");
/// // Returns Some(vec![('t', 's', 'ʼ'), ('t', 'ʃ', 'ʼ'), ...])
/// ```
pub fn get_trigraphs_only(name: &str) -> Option<Vec<(char, char, char)>> {
    get_named_class(name).map(|class| {
        let mut trigraphs = Vec::with_capacity(class.patterns.len());
        trigraphs.extend(class.patterns.iter().filter_map(|p| p.as_trigraph()));
        trigraphs
    })
}

/// Get only the tetragraph patterns from a named class.
///
/// Returns None if the class doesn't exist.
/// Returns an empty vec if the class exists but has no tetragraph patterns.
///
/// # Examples
///
/// ```ignore
/// let click_tetragraphs = get_tetragraphs_only("prenasalized_click");
/// // Returns Some(vec![('ŋ', 'ɡ', 'ǀ', 'ʰ'), ...])
/// ```
pub fn get_tetragraphs_only(name: &str) -> Option<Vec<(char, char, char, char)>> {
    get_named_class(name).map(|class| {
        let mut tetragraphs = Vec::with_capacity(class.patterns.len());
        tetragraphs.extend(class.patterns.iter().filter_map(|p| p.as_tetragraph()));
        tetragraphs
    })
}

/// Get only the sequence patterns from a named class.
///
/// Returns None if the class doesn't exist.
/// Returns an empty vec if the class exists but has no sequence patterns.
///
/// # Examples
///
/// ```ignore
/// let long_patterns = get_sequences_only("complex_clusters");
/// // Returns Some(vec![[...], [...], ...])
/// ```
pub fn get_sequences_only(name: &str) -> Option<Vec<Vec<char>>> {
    get_named_class(name).map(|class| {
        let mut sequences = Vec::with_capacity(class.patterns.len());
        sequences.extend(
            class
                .patterns
                .iter()
                .filter_map(|p| p.as_sequence().map(|s| s.to_vec())),
        );
        sequences
    })
}
