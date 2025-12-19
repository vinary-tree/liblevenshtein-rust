//! Built-in named character classes for phonetic grammars.
//!
//! This module provides standard POSIX classes and phonetic-specific classes
//! for use in LLev grammar files and regex patterns.
//!
//! # Class Categories
//!
//! - **POSIX classes**: `[:alpha:]`, `[:digit:]`, `[:space:]`, etc.
//! - **Phonetic classes**: `[:vowel:]`, `[:consonant:]`, `[:fricative:]`, etc.
//!
//! # Syntax
//!
//! - Standalone: `[:vowel:]` - matches any vowel
//! - Mixed: `[[:vowel:]y]` - matches vowels plus 'y'
//!
//! # Examples
//!
//! ```text
//! # LLev grammar rules
//! c -> s / _[:front_vowel:];    # c -> s before front vowel
//! [:voiced:] -> [:voiceless:];  # devoicing rule
//! [:nasal:] -> m / _[:stop:];   # nasal assimilation
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

// ============================================================================
// PhonePattern - Single char or digraph
// ============================================================================

/// A phonetic class element - either a single character or a digraph.
///
/// Phonetic classes can contain both single Unicode characters (including IPA)
/// and ASCII digraphs (two-character sequences representing single sounds).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhonePattern {
    /// Single character (e.g., 'a', 'ʃ', 'θ')
    Char(char),
    /// Two-character digraph (e.g., ('s', 'h') for "sh")
    Digraph(char, char),
}

impl PhonePattern {
    /// Create a single character pattern.
    pub const fn char(c: char) -> Self {
        PhonePattern::Char(c)
    }

    /// Create a digraph pattern.
    pub const fn digraph(c1: char, c2: char) -> Self {
        PhonePattern::Digraph(c1, c2)
    }

    /// Check if this pattern matches a single character.
    pub fn matches_char(&self, c: char) -> bool {
        matches!(self, PhonePattern::Char(pc) if *pc == c)
    }

    /// Check if this pattern matches a two-character sequence.
    pub fn matches_digraph(&self, c1: char, c2: char) -> bool {
        matches!(self, PhonePattern::Digraph(d1, d2) if *d1 == c1 && *d2 == c2)
    }

    /// Returns true if this is a single character.
    pub fn is_char(&self) -> bool {
        matches!(self, PhonePattern::Char(_))
    }

    /// Returns true if this is a digraph.
    pub fn is_digraph(&self) -> bool {
        matches!(self, PhonePattern::Digraph(_, _))
    }

    /// Get the character if this is a single char pattern.
    pub fn as_char(&self) -> Option<char> {
        match self {
            PhonePattern::Char(c) => Some(*c),
            PhonePattern::Digraph(_, _) => None,
        }
    }

    /// Get the digraph characters if this is a digraph pattern.
    pub fn as_digraph(&self) -> Option<(char, char)> {
        match self {
            PhonePattern::Char(_) => None,
            PhonePattern::Digraph(c1, c2) => Some((*c1, *c2)),
        }
    }
}

impl std::fmt::Display for PhonePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhonePattern::Char(c) => write!(f, "{}", c),
            PhonePattern::Digraph(c1, c2) => write!(f, "{}{}", c1, c2),
        }
    }
}

// ============================================================================
// Named Classes Registry
// ============================================================================

/// A named character class definition.
#[derive(Debug, Clone)]
pub struct NamedClass {
    /// The canonical name (lowercase)
    pub name: &'static str,
    /// Alternative names/aliases
    pub aliases: &'static [&'static str],
    /// The patterns in this class
    pub patterns: Vec<PhonePattern>,
    /// Description for documentation
    pub description: &'static str,
}

/// Built-in named phonetic classes.
///
/// Access classes via [`get_named_class`] which handles case-insensitive lookup.
pub static NAMED_CLASSES: LazyLock<HashMap<&'static str, NamedClass>> = LazyLock::new(|| {
    use PhonePattern::{Char, Digraph};

    let mut m = HashMap::new();

    // Helper to add a class with aliases
    let mut add_class = |class: NamedClass| {
        for &alias in class.aliases {
            m.insert(alias, class.clone());
        }
        m.insert(class.name, class);
    };

    // ========================================================================
    // POSIX Classes (single characters only)
    // ========================================================================

    add_class(NamedClass {
        name: "alpha",
        aliases: &[],
        patterns: ('a'..='z')
            .chain('A'..='Z')
            .map(Char)
            .collect(),
        description: "Alphabetic characters (a-z, A-Z)",
    });

    add_class(NamedClass {
        name: "lower",
        aliases: &[],
        patterns: ('a'..='z').map(Char).collect(),
        description: "Lowercase letters (a-z)",
    });

    add_class(NamedClass {
        name: "upper",
        aliases: &[],
        patterns: ('A'..='Z').map(Char).collect(),
        description: "Uppercase letters (A-Z)",
    });

    add_class(NamedClass {
        name: "digit",
        aliases: &[],
        patterns: ('0'..='9').map(Char).collect(),
        description: "Digits (0-9)",
    });

    add_class(NamedClass {
        name: "alnum",
        aliases: &[],
        patterns: ('a'..='z')
            .chain('A'..='Z')
            .chain('0'..='9')
            .map(Char)
            .collect(),
        description: "Alphanumeric characters (a-z, A-Z, 0-9)",
    });

    add_class(NamedClass {
        name: "word",
        aliases: &[],
        patterns: ('a'..='z')
            .chain('A'..='Z')
            .chain('0'..='9')
            .map(Char)
            .chain(std::iter::once(Char('_')))
            .collect(),
        description: "Word characters (a-z, A-Z, 0-9, _)",
    });

    add_class(NamedClass {
        name: "space",
        aliases: &[],
        patterns: vec![Char(' '), Char('\t'), Char('\n'), Char('\r')],
        description: "Whitespace characters",
    });

    add_class(NamedClass {
        name: "punct",
        aliases: &[],
        patterns: "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
            .chars()
            .map(Char)
            .collect(),
        description: "Punctuation characters",
    });

    // ========================================================================
    // Phonetic Vowel Classes (inclusive - ASCII + IPA + digraphs)
    // ========================================================================

    // All vowels (ASCII + IPA)
    add_class(NamedClass {
        name: "vowel",
        aliases: &[],
        patterns: vec![
            // ASCII vowels
            Char('a'), Char('e'), Char('i'), Char('o'), Char('u'),
            Char('A'), Char('E'), Char('I'), Char('O'), Char('U'),
            // IPA vowels
            Char('ə'), // schwa
            Char('ɪ'), // near-close near-front unrounded (bit)
            Char('ʊ'), // near-close near-back rounded (foot)
            Char('ɛ'), // open-mid front unrounded (bed)
            Char('ɔ'), // open-mid back rounded (thought)
            Char('æ'), // near-open front unrounded (cat)
            Char('ʌ'), // open-mid back unrounded (strut)
            Char('ɑ'), // open back unrounded (father)
            Char('ɒ'), // open back rounded (lot - British)
            Char('ɜ'), // open-mid central unrounded (bird)
            Char('ɐ'), // near-open central
        ],
        description: "All vowels (ASCII + IPA)",
    });

    // Front vowels
    add_class(NamedClass {
        name: "front_vowel",
        aliases: &[],
        patterns: vec![
            Char('e'), Char('i'), Char('E'), Char('I'),
            Char('ɪ'), Char('ɛ'), Char('æ'),
        ],
        description: "Front vowels (e, i and IPA equivalents)",
    });

    // Back vowels
    add_class(NamedClass {
        name: "back_vowel",
        aliases: &[],
        patterns: vec![
            Char('a'), Char('o'), Char('u'), Char('A'), Char('O'), Char('U'),
            Char('ʊ'), Char('ɔ'), Char('ɑ'), Char('ɒ'),
        ],
        description: "Back vowels (a, o, u and IPA equivalents)",
    });

    // High vowels
    add_class(NamedClass {
        name: "high_vowel",
        aliases: &[],
        patterns: vec![
            Char('i'), Char('u'), Char('I'), Char('U'),
            Char('ɪ'), Char('ʊ'),
        ],
        description: "High vowels (i, u and IPA equivalents)",
    });

    // Low vowels
    add_class(NamedClass {
        name: "low_vowel",
        aliases: &[],
        patterns: vec![
            Char('a'), Char('A'),
            Char('æ'), Char('ɑ'), Char('ɒ'), Char('ɐ'),
        ],
        description: "Low vowels (a and IPA equivalents)",
    });

    // Mid vowels
    add_class(NamedClass {
        name: "mid_vowel",
        aliases: &[],
        patterns: vec![
            Char('e'), Char('o'), Char('E'), Char('O'),
            Char('ə'), Char('ɛ'), Char('ɔ'), Char('ʌ'), Char('ɜ'),
        ],
        description: "Mid vowels (e, o and IPA equivalents)",
    });

    // Central vowels
    add_class(NamedClass {
        name: "central_vowel",
        aliases: &[],
        patterns: vec![Char('ə'), Char('ʌ'), Char('ɜ'), Char('ɐ')],
        description: "Central vowels (schwa family)",
    });

    // Schwa only
    add_class(NamedClass {
        name: "schwa",
        aliases: &[],
        patterns: vec![Char('ə')],
        description: "Schwa only",
    });

    // ========================================================================
    // Phonetic Consonant Classes (inclusive - ASCII + IPA + digraphs)
    // ========================================================================

    // All consonants
    add_class(NamedClass {
        name: "consonant",
        aliases: &[],
        patterns: vec![
            // ASCII consonants
            Char('b'), Char('c'), Char('d'), Char('f'), Char('g'),
            Char('h'), Char('j'), Char('k'), Char('l'), Char('m'),
            Char('n'), Char('p'), Char('q'), Char('r'), Char('s'),
            Char('t'), Char('v'), Char('w'), Char('x'), Char('y'), Char('z'),
            Char('B'), Char('C'), Char('D'), Char('F'), Char('G'),
            Char('H'), Char('J'), Char('K'), Char('L'), Char('M'),
            Char('N'), Char('P'), Char('Q'), Char('R'), Char('S'),
            Char('T'), Char('V'), Char('W'), Char('X'), Char('Y'), Char('Z'),
            // IPA consonants
            Char('ŋ'), // velar nasal (sing)
            Char('θ'), // voiceless dental fricative (thin)
            Char('ð'), // voiced dental fricative (this)
            Char('ʃ'), // voiceless postalveolar fricative (ship)
            Char('ʒ'), // voiced postalveolar fricative (measure)
            // Note: tʃ and dʒ are affricates, handled in affricate class with digraphs
            Char('ɹ'), // alveolar approximant (red - IPA)
            Char('ɾ'), // alveolar tap
            Char('ʔ'), // glottal stop
            Char('ɬ'), // voiceless lateral fricative
        ],
        description: "All consonants (ASCII + IPA)",
    });

    // Stop consonants (plosives)
    add_class(NamedClass {
        name: "stop",
        aliases: &["plosive"],
        patterns: vec![
            Char('p'), Char('b'), Char('t'), Char('d'), Char('k'), Char('g'),
            Char('P'), Char('B'), Char('T'), Char('D'), Char('K'), Char('G'),
            Char('ʔ'), // glottal stop
        ],
        description: "Stop consonants (plosives)",
    });

    // Fricatives (with digraphs)
    add_class(NamedClass {
        name: "fricative",
        aliases: &[],
        patterns: vec![
            // Single chars
            Char('f'), Char('v'), Char('s'), Char('z'), Char('h'),
            Char('F'), Char('V'), Char('S'), Char('Z'), Char('H'),
            Char('ʃ'), Char('ʒ'), Char('θ'), Char('ð'), Char('ɬ'),
            // Digraphs
            Digraph('s', 'h'), // sh -> ʃ
            Digraph('S', 'h'), Digraph('S', 'H'),
            Digraph('z', 'h'), // zh -> ʒ
            Digraph('Z', 'h'), Digraph('Z', 'H'),
            Digraph('t', 'h'), // th -> θ/ð
            Digraph('T', 'h'), Digraph('T', 'H'),
        ],
        description: "Fricative consonants (includes sh, th, zh digraphs)",
    });

    // Nasal consonants (with digraphs)
    add_class(NamedClass {
        name: "nasal",
        aliases: &[],
        patterns: vec![
            Char('m'), Char('n'), Char('M'), Char('N'),
            Char('ŋ'),
            Digraph('n', 'g'), // ng -> ŋ
            Digraph('N', 'g'), Digraph('N', 'G'),
        ],
        description: "Nasal consonants (includes ng digraph)",
    });

    // Liquid consonants
    add_class(NamedClass {
        name: "liquid",
        aliases: &[],
        patterns: vec![
            Char('l'), Char('r'), Char('L'), Char('R'),
            Char('ɹ'), Char('ɾ'),
        ],
        description: "Liquid consonants (l, r)",
    });

    // Glides (semivowels)
    add_class(NamedClass {
        name: "glide",
        aliases: &["semivowel"],
        patterns: vec![
            Char('w'), Char('y'), Char('W'), Char('Y'),
            Char('j'), Char('J'), // IPA j = English y
        ],
        description: "Glides/semivowels (w, y)",
    });

    // Affricates (with digraphs)
    add_class(NamedClass {
        name: "affricate",
        aliases: &[],
        patterns: vec![
            Digraph('c', 'h'), // ch -> tʃ
            Digraph('C', 'h'), Digraph('C', 'H'),
            Digraph('t', 's'), // ts
            Digraph('d', 'z'), // dz
            Digraph('d', 'j'), // dj (sometimes dʒ)
        ],
        description: "Affricate consonants (ch, ts, dz)",
    });

    // Voiced consonants
    add_class(NamedClass {
        name: "voiced",
        aliases: &[],
        patterns: vec![
            Char('b'), Char('d'), Char('g'), Char('v'), Char('z'),
            Char('B'), Char('D'), Char('G'), Char('V'), Char('Z'),
            Char('l'), Char('m'), Char('n'), Char('r'), Char('w'),
            Char('L'), Char('M'), Char('N'), Char('R'), Char('W'),
            Char('ð'), Char('ʒ'), Char('ŋ'), Char('ɹ'),
            Digraph('z', 'h'),
            Digraph('n', 'g'),
        ],
        description: "Voiced consonants",
    });

    // Voiceless consonants
    add_class(NamedClass {
        name: "voiceless",
        aliases: &[],
        patterns: vec![
            Char('p'), Char('t'), Char('k'), Char('f'), Char('s'), Char('h'),
            Char('P'), Char('T'), Char('K'), Char('F'), Char('S'), Char('H'),
            Char('c'), Char('q'), Char('x'),
            Char('C'), Char('Q'), Char('X'),
            Char('θ'), Char('ʃ'), Char('ʔ'),
            Digraph('s', 'h'),
            Digraph('t', 'h'),
            Digraph('c', 'h'),
        ],
        description: "Voiceless consonants",
    });

    // ========================================================================
    // ASCII-Only Subsets
    // ========================================================================

    add_class(NamedClass {
        name: "ascii_vowel",
        aliases: &[],
        patterns: vec![
            Char('a'), Char('e'), Char('i'), Char('o'), Char('u'),
            Char('A'), Char('E'), Char('I'), Char('O'), Char('U'),
        ],
        description: "ASCII vowels only (a, e, i, o, u)",
    });

    add_class(NamedClass {
        name: "ascii_consonant",
        aliases: &[],
        patterns: vec![
            Char('b'), Char('c'), Char('d'), Char('f'), Char('g'),
            Char('h'), Char('j'), Char('k'), Char('l'), Char('m'),
            Char('n'), Char('p'), Char('q'), Char('r'), Char('s'),
            Char('t'), Char('v'), Char('w'), Char('x'), Char('y'), Char('z'),
            Char('B'), Char('C'), Char('D'), Char('F'), Char('G'),
            Char('H'), Char('J'), Char('K'), Char('L'), Char('M'),
            Char('N'), Char('P'), Char('Q'), Char('R'), Char('S'),
            Char('T'), Char('V'), Char('W'), Char('X'), Char('Y'), Char('Z'),
        ],
        description: "ASCII consonants only",
    });

    add_class(NamedClass {
        name: "ascii_front",
        aliases: &[],
        patterns: vec![
            Char('e'), Char('i'), Char('E'), Char('I'),
        ],
        description: "ASCII front vowels only (e, i)",
    });

    add_class(NamedClass {
        name: "ascii_back",
        aliases: &[],
        patterns: vec![
            Char('a'), Char('o'), Char('u'), Char('A'), Char('O'), Char('U'),
        ],
        description: "ASCII back vowels only (a, o, u)",
    });

    // ========================================================================
    // IPA-Only Subsets (non-ASCII)
    // ========================================================================

    add_class(NamedClass {
        name: "ipa_vowel",
        aliases: &[],
        patterns: vec![
            Char('ə'), Char('ɪ'), Char('ʊ'), Char('ɛ'), Char('ɔ'),
            Char('æ'), Char('ʌ'), Char('ɑ'), Char('ɒ'), Char('ɜ'), Char('ɐ'),
        ],
        description: "IPA vowels only (non-ASCII)",
    });

    add_class(NamedClass {
        name: "ipa_consonant",
        aliases: &[],
        patterns: vec![
            Char('ŋ'), Char('θ'), Char('ð'), Char('ʃ'), Char('ʒ'),
            Char('ɹ'), Char('ɾ'), Char('ʔ'), Char('ɬ'),
        ],
        description: "IPA consonants only (non-ASCII)",
    });

    add_class(NamedClass {
        name: "ipa_front",
        aliases: &[],
        patterns: vec![Char('ɪ'), Char('ɛ'), Char('æ')],
        description: "IPA front vowels only",
    });

    add_class(NamedClass {
        name: "ipa_back",
        aliases: &[],
        patterns: vec![Char('ʊ'), Char('ɔ'), Char('ɑ'), Char('ɒ')],
        description: "IPA back vowels only",
    });

    m
});

// ============================================================================
// Lookup Functions
// ============================================================================

/// Maximum length of any built-in class name.
/// Used for stack-allocated lowercase buffer.
const MAX_CLASS_NAME_LEN: usize = 16; // "ascii_consonant" = 15 chars

/// Normalize a class name to lowercase using stack allocation.
///
/// Returns None if the name is too long or contains non-ASCII characters.
/// This is an internal helper for zero-allocation case-insensitive lookup.
#[inline]
fn normalize_class_name(name: &str) -> Option<([u8; MAX_CLASS_NAME_LEN], usize)> {
    let len = name.len();
    if len > MAX_CLASS_NAME_LEN || !name.is_ascii() {
        return None;
    }

    let mut buf = [0u8; MAX_CLASS_NAME_LEN];
    let bytes = name.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        buf[i] = b.to_ascii_lowercase();
    }

    Some((buf, len))
}

/// Look up a named class by name (case-insensitive).
///
/// Returns the named class definition if found.
///
/// This function uses a stack-allocated buffer for case conversion,
/// avoiding heap allocation on every lookup.
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

    // Normalize to lowercase in stack buffer
    let (buf, len) = normalize_class_name(name)?;

    // SAFETY: Input verified as ASCII, ASCII lowercase is valid UTF-8
    let lowered = unsafe { std::str::from_utf8_unchecked(&buf[..len]) };
    NAMED_CLASSES.get(lowered)
}

/// Check if a name is a built-in class (case-insensitive).
///
/// This is useful for detecting conflicts when users try to define
/// symbols with the same name as built-in classes.
///
/// Uses stack-allocated buffer for case conversion (no heap allocation).
pub fn is_builtin_class(name: &str) -> bool {
    // Fast path: exact match
    if NAMED_CLASSES.contains_key(name) {
        return true;
    }

    // Normalize to lowercase in stack buffer
    let Some((buf, len)) = normalize_class_name(name) else {
        return false;
    };

    // SAFETY: Input verified as ASCII
    let lowered = unsafe { std::str::from_utf8_unchecked(&buf[..len]) };
    NAMED_CLASSES.contains_key(lowered)
}

/// Get all built-in class names (for error messages and documentation).
pub fn all_builtin_class_names() -> Vec<&'static str> {
    let mut names: Vec<_> = NAMED_CLASSES.keys().copied().collect();
    names.sort();
    names.dedup();
    names
}

/// Get only the single-character patterns from a named class.
///
/// Useful when you need to build a simple character class without digraphs.
pub fn get_chars_only(name: &str) -> Option<Vec<char>> {
    get_named_class(name).map(|class| {
        class
            .patterns
            .iter()
            .filter_map(|p| p.as_char())
            .collect()
    })
}

/// Get only the digraph patterns from a named class.
pub fn get_digraphs_only(name: &str) -> Option<Vec<(char, char)>> {
    get_named_class(name).map(|class| {
        class
            .patterns
            .iter()
            .filter_map(|p| p.as_digraph())
            .collect()
    })
}

// ============================================================================
// Feature Bundle Helpers (for intersection semantics)
// ============================================================================

/// Get all phonetic characters (union of all vowels and consonants).
///
/// This is used as the universe for computing negation of character sets.
/// Returns both ASCII and IPA characters.
pub fn get_all_phonetic_chars() -> Vec<char> {
    let mut chars: HashSet<char> = HashSet::new();
    if let Some(v) = get_chars_only("vowel") {
        chars.extend(v);
    }
    if let Some(c) = get_chars_only("consonant") {
        chars.extend(c);
    }
    chars.into_iter().collect()
}

/// Compute the intersection of multiple character sets.
///
/// Returns characters that appear in ALL of the provided sets.
/// An empty input returns an empty result.
///
/// # Example
///
/// ```
/// use liblevenshtein::phonetic::named_classes::{get_chars_only, intersect_char_sets};
///
/// let voiced = get_chars_only("voiced").unwrap();
/// let stop = get_chars_only("stop").unwrap();
/// let result = intersect_char_sets(&[voiced, stop]);
/// // result contains only voiced stops: b, d, g
/// assert!(result.contains(&'b'));
/// assert!(result.contains(&'d'));
/// assert!(result.contains(&'g'));
/// assert!(!result.contains(&'p')); // voiceless
/// ```
pub fn intersect_char_sets(sets: &[Vec<char>]) -> Vec<char> {
    if sets.is_empty() {
        return Vec::new();
    }
    let mut result: HashSet<char> = sets[0].iter().copied().collect();
    for set in &sets[1..] {
        let other: HashSet<char> = set.iter().copied().collect();
        result = result.intersection(&other).copied().collect();
    }
    result.into_iter().collect()
}

/// Negate a character set (relative to all phonetic characters).
///
/// Returns all phonetic characters that are NOT in the provided set.
///
/// # Example
///
/// ```
/// use liblevenshtein::phonetic::named_classes::{get_chars_only, negate_char_set};
///
/// let nasal = get_chars_only("nasal").unwrap();
/// let not_nasal = negate_char_set(&nasal);
/// // not_nasal contains everything except m, n, ŋ
/// assert!(!not_nasal.contains(&'m'));
/// assert!(!not_nasal.contains(&'n'));
/// assert!(not_nasal.contains(&'p'));
/// assert!(not_nasal.contains(&'a'));
/// ```
pub fn negate_char_set(chars: &[char]) -> Vec<char> {
    let all = get_all_phonetic_chars();
    let excluded: HashSet<char> = chars.iter().copied().collect();
    all.into_iter().filter(|c| !excluded.contains(c)).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phone_pattern_char() {
        let p = PhonePattern::Char('a');
        assert!(p.is_char());
        assert!(!p.is_digraph());
        assert_eq!(p.as_char(), Some('a'));
        assert_eq!(p.as_digraph(), None);
        assert!(p.matches_char('a'));
        assert!(!p.matches_char('b'));
    }

    #[test]
    fn test_phone_pattern_digraph() {
        let p = PhonePattern::Digraph('s', 'h');
        assert!(!p.is_char());
        assert!(p.is_digraph());
        assert_eq!(p.as_char(), None);
        assert_eq!(p.as_digraph(), Some(('s', 'h')));
        assert!(p.matches_digraph('s', 'h'));
        assert!(!p.matches_digraph('s', 'z'));
    }

    #[test]
    fn test_get_named_class_vowel() {
        let vowels = get_named_class("vowel").expect("vowel class should exist");
        assert!(!vowels.patterns.is_empty());

        // Should contain ASCII vowels
        assert!(vowels.patterns.contains(&PhonePattern::Char('a')));
        assert!(vowels.patterns.contains(&PhonePattern::Char('e')));
        assert!(vowels.patterns.contains(&PhonePattern::Char('i')));
        assert!(vowels.patterns.contains(&PhonePattern::Char('o')));
        assert!(vowels.patterns.contains(&PhonePattern::Char('u')));

        // Should contain IPA vowels
        assert!(vowels.patterns.contains(&PhonePattern::Char('ə')));
        assert!(vowels.patterns.contains(&PhonePattern::Char('ɪ')));
    }

    #[test]
    fn test_case_insensitive_lookup() {
        let v1 = get_named_class("vowel").expect("lowercase");
        let v2 = get_named_class("VOWEL").expect("uppercase");
        let v3 = get_named_class("Vowel").expect("mixed case");

        assert_eq!(v1.patterns.len(), v2.patterns.len());
        assert_eq!(v2.patterns.len(), v3.patterns.len());
    }

    #[test]
    fn test_full_word_alias() {
        // "plosive" is an alias for "stop"
        let s1 = get_named_class("stop").expect("full name");
        let s2 = get_named_class("plosive").expect("alias");
        assert_eq!(s1.patterns.len(), s2.patterns.len());

        // "semivowel" is an alias for "glide"
        let g1 = get_named_class("glide").expect("full name");
        let g2 = get_named_class("semivowel").expect("alias");
        assert_eq!(g1.patterns.len(), g2.patterns.len());
    }

    #[test]
    fn test_fricative_has_digraphs() {
        let fric = get_named_class("fricative").expect("fricative class");

        // Should have IPA chars
        assert!(fric.patterns.contains(&PhonePattern::Char('ʃ')));
        assert!(fric.patterns.contains(&PhonePattern::Char('θ')));

        // Should have digraphs
        assert!(fric.patterns.contains(&PhonePattern::Digraph('s', 'h')));
        assert!(fric.patterns.contains(&PhonePattern::Digraph('t', 'h')));
    }

    #[test]
    fn test_nasal_has_ng_digraph() {
        let nasal = get_named_class("nasal").expect("nasal class");

        assert!(nasal.patterns.contains(&PhonePattern::Char('ŋ')));
        assert!(nasal.patterns.contains(&PhonePattern::Digraph('n', 'g')));
    }

    #[test]
    fn test_is_builtin_class() {
        assert!(is_builtin_class("vowel"));
        assert!(is_builtin_class("VOWEL"));
        assert!(is_builtin_class("plosive")); // alias for stop
        assert!(is_builtin_class("fricative"));
        assert!(is_builtin_class("alpha"));

        // Single-letter names are NOT built-in (reserved for user symbols)
        assert!(!is_builtin_class("V"));
        assert!(!is_builtin_class("C"));
        assert!(!is_builtin_class("not_a_class"));
        assert!(!is_builtin_class("CUSTOM"));
    }

    #[test]
    fn test_get_chars_only() {
        let chars = get_chars_only("fricative").expect("fricative exists");

        // Should have single chars
        assert!(chars.contains(&'f'));
        assert!(chars.contains(&'ʃ'));

        // Should NOT have digraph components as separate chars
        // (digraphs are filtered out)
    }

    #[test]
    fn test_get_digraphs_only() {
        let digraphs = get_digraphs_only("fricative").expect("fricative exists");

        assert!(digraphs.contains(&('s', 'h')));
        assert!(digraphs.contains(&('t', 'h')));
    }

    #[test]
    fn test_posix_classes() {
        let alpha = get_named_class("alpha").expect("alpha");
        let digit = get_named_class("digit").expect("digit");
        let alnum = get_named_class("alnum").expect("alnum");

        // Alpha should have letters only
        assert!(alpha.patterns.contains(&PhonePattern::Char('a')));
        assert!(alpha.patterns.contains(&PhonePattern::Char('Z')));
        assert!(!alpha.patterns.contains(&PhonePattern::Char('0')));

        // Digit should have digits only
        assert!(digit.patterns.contains(&PhonePattern::Char('0')));
        assert!(digit.patterns.contains(&PhonePattern::Char('9')));
        assert!(!digit.patterns.contains(&PhonePattern::Char('a')));

        // Alnum should have both
        assert!(alnum.patterns.contains(&PhonePattern::Char('a')));
        assert!(alnum.patterns.contains(&PhonePattern::Char('0')));
    }

    #[test]
    fn test_ascii_vs_ipa_subsets() {
        let ascii_v = get_named_class("ascii_vowel").expect("ascii_vowel");
        let ipa_v = get_named_class("ipa_vowel").expect("ipa_vowel");

        // ASCII should only have basic vowels
        assert!(ascii_v.patterns.contains(&PhonePattern::Char('a')));
        assert!(!ascii_v.patterns.contains(&PhonePattern::Char('ə')));

        // IPA should only have IPA vowels
        assert!(ipa_v.patterns.contains(&PhonePattern::Char('ə')));
        assert!(!ipa_v.patterns.contains(&PhonePattern::Char('a')));
    }

    #[test]
    fn test_all_builtin_class_names() {
        let names = all_builtin_class_names();

        assert!(names.contains(&"vowel"));
        assert!(names.contains(&"consonant"));
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"fricative"));
    }

    // =========================================================================
    // Feature Bundle Helper Tests
    // =========================================================================

    #[test]
    fn test_get_all_phonetic_chars() {
        let all = get_all_phonetic_chars();

        // Should contain vowels
        assert!(all.contains(&'a'));
        assert!(all.contains(&'e'));
        assert!(all.contains(&'ə')); // IPA schwa

        // Should contain consonants
        assert!(all.contains(&'b'));
        assert!(all.contains(&'p'));
        assert!(all.contains(&'ŋ')); // IPA eng

        // Should NOT contain digits or punctuation
        assert!(!all.contains(&'0'));
        assert!(!all.contains(&'.'));
    }

    #[test]
    fn test_intersect_char_sets_voiced_stop() {
        let voiced = get_chars_only("voiced").expect("voiced class");
        let stop = get_chars_only("stop").expect("stop class");
        let result = intersect_char_sets(&[voiced, stop]);

        // Voiced stops: b, d, g
        assert!(result.contains(&'b'));
        assert!(result.contains(&'d'));
        assert!(result.contains(&'g'));

        // Voiceless stops should NOT be in result
        assert!(!result.contains(&'p'));
        assert!(!result.contains(&'t'));
        assert!(!result.contains(&'k'));

        // Non-stops should NOT be in result
        assert!(!result.contains(&'v')); // voiced fricative
        assert!(!result.contains(&'z')); // voiced fricative
    }

    #[test]
    fn test_intersect_char_sets_empty() {
        let result = intersect_char_sets(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_intersect_char_sets_single() {
        let stop = get_chars_only("stop").expect("stop class");
        let result = intersect_char_sets(&[stop.clone()]);

        // Single set intersection should return the same set
        assert_eq!(result.len(), stop.len());
        for c in &stop {
            assert!(result.contains(c));
        }
    }

    #[test]
    fn test_intersect_char_sets_three_features() {
        // high + front + vowel should give high front vowels (i, ɪ, y, ʏ, etc.)
        let high = get_chars_only("high_vowel").expect("high_vowel class");
        let front = get_chars_only("front_vowel").expect("front_vowel class");
        let vowel = get_chars_only("vowel").expect("vowel class");
        let result = intersect_char_sets(&[high, front, vowel]);

        // i should be in high front vowels
        assert!(result.contains(&'i'));
        assert!(result.contains(&'I'));

        // a should NOT be in result (low, not high)
        assert!(!result.contains(&'a'));

        // u should NOT be in result (back, not front)
        assert!(!result.contains(&'u'));
    }

    #[test]
    fn test_negate_char_set_nasal() {
        let nasal = get_chars_only("nasal").expect("nasal class");
        let not_nasal = negate_char_set(&nasal);

        // Nasals should NOT be in result
        assert!(!not_nasal.contains(&'m'));
        assert!(!not_nasal.contains(&'n'));
        assert!(!not_nasal.contains(&'ŋ'));

        // Other consonants should be in result
        assert!(not_nasal.contains(&'p'));
        assert!(not_nasal.contains(&'b'));
        assert!(not_nasal.contains(&'t'));

        // Vowels should be in result
        assert!(not_nasal.contains(&'a'));
        assert!(not_nasal.contains(&'e'));
    }

    #[test]
    fn test_negate_char_set_empty() {
        let not_empty = negate_char_set(&[]);
        let all = get_all_phonetic_chars();

        // Negating empty set should give all chars
        assert_eq!(not_empty.len(), all.len());
    }

    #[test]
    fn test_intersect_with_negation() {
        // Test the pattern: [:!nasal stop:] = oral stops (p, t, k, b, d, g)
        let nasal = get_chars_only("nasal").expect("nasal class");
        let not_nasal = negate_char_set(&nasal);
        let stop = get_chars_only("stop").expect("stop class");
        let result = intersect_char_sets(&[not_nasal, stop]);

        // All non-nasal stops
        assert!(result.contains(&'p'));
        assert!(result.contains(&'t'));
        assert!(result.contains(&'k'));
        assert!(result.contains(&'b'));
        assert!(result.contains(&'d'));
        assert!(result.contains(&'g'));

        // Nasal consonants should NOT be in result (m, n are not stops anyway)
        assert!(!result.contains(&'m'));
        assert!(!result.contains(&'n'));

        // Other consonants should NOT be in result
        assert!(!result.contains(&'f'));
        assert!(!result.contains(&'s'));
    }
}
