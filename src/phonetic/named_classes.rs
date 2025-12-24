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

    // ========================================================================
    // Extended IPA Consonants - Place of Articulation
    // ========================================================================

    // Retroflex consonants (tongue curled back)
    add_class(NamedClass {
        name: "retroflex",
        aliases: &[],
        patterns: vec![
            Char('ʈ'), // voiceless retroflex stop (U+0288)
            Char('ɖ'), // voiced retroflex stop (U+0256)
            Char('ɳ'), // retroflex nasal (U+0273)
            Char('ɽ'), // retroflex flap (U+027D)
            Char('ʂ'), // voiceless retroflex fricative (U+0282)
            Char('ʐ'), // voiced retroflex fricative (U+0290)
            Char('ɻ'), // retroflex approximant (U+027B)
            Char('ɭ'), // retroflex lateral approximant (U+026D)
        ],
        description: "Retroflex consonants (tongue curled back)",
    });

    // Uvular consonants (back of throat)
    add_class(NamedClass {
        name: "uvular",
        aliases: &[],
        patterns: vec![
            Char('q'), // voiceless uvular stop
            Char('ɢ'), // voiced uvular stop (U+0262)
            Char('ɴ'), // uvular nasal (U+0274)
            Char('ʀ'), // uvular trill (U+0280)
            Char('χ'), // voiceless uvular fricative (U+03C7)
            Char('ʁ'), // voiced uvular fricative/approximant (U+0281)
        ],
        description: "Uvular consonants (articulated at uvula)",
    });

    // Pharyngeal consonants
    add_class(NamedClass {
        name: "pharyngeal",
        aliases: &[],
        patterns: vec![
            Char('ħ'), // voiceless pharyngeal fricative (U+0127)
            Char('ʕ'), // voiced pharyngeal fricative (U+0295)
        ],
        description: "Pharyngeal consonants (articulated at pharynx)",
    });

    // Glottal consonants
    add_class(NamedClass {
        name: "glottal",
        aliases: &[],
        patterns: vec![
            Char('ʔ'), // glottal stop (U+0294)
            Char('h'), // voiceless glottal fricative
            Char('ɦ'), // voiced glottal fricative (U+0266)
        ],
        description: "Glottal consonants",
    });

    // Palatal consonants
    add_class(NamedClass {
        name: "palatal",
        aliases: &[],
        patterns: vec![
            Char('c'), // voiceless palatal stop
            Char('ɟ'), // voiced palatal stop (U+025F)
            Char('ɲ'), // palatal nasal (U+0272)
            Char('ç'), // voiceless palatal fricative (U+00E7)
            Char('ʝ'), // voiced palatal fricative (U+029D)
            Char('j'), // palatal approximant
            Char('ʎ'), // palatal lateral approximant (U+028E)
        ],
        description: "Palatal consonants",
    });

    // Velar consonants
    add_class(NamedClass {
        name: "velar",
        aliases: &[],
        patterns: vec![
            Char('k'), Char('K'),
            Char('g'), Char('G'),
            Char('ŋ'), // velar nasal (U+014B)
            Char('x'), // voiceless velar fricative
            Char('ɣ'), // voiced velar fricative (U+0263)
            Char('ɰ'), // velar approximant (U+0270)
            Char('ʟ'), // velar lateral approximant (U+029F)
        ],
        description: "Velar consonants (articulated at soft palate)",
    });

    // Alveolar consonants
    add_class(NamedClass {
        name: "alveolar",
        aliases: &[],
        patterns: vec![
            Char('t'), Char('T'), Char('d'), Char('D'),
            Char('n'), Char('N'),
            Char('s'), Char('S'), Char('z'), Char('Z'),
            Char('ɹ'), // alveolar approximant (U+0279)
            Char('ɾ'), // alveolar tap (U+027E)
            Char('r'), Char('R'), // alveolar trill
            Char('l'), Char('L'),
            Char('ɬ'), // voiceless alveolar lateral fricative (U+026C)
            Char('ɮ'), // voiced alveolar lateral fricative (U+026E)
        ],
        description: "Alveolar consonants (articulated at alveolar ridge)",
    });

    // Bilabial consonants
    add_class(NamedClass {
        name: "bilabial",
        aliases: &[],
        patterns: vec![
            Char('p'), Char('P'), Char('b'), Char('B'),
            Char('m'), Char('M'),
            Char('ɸ'), // voiceless bilabial fricative (U+0278)
            Char('β'), // voiced bilabial fricative (U+03B2)
            Char('ʙ'), // bilabial trill (U+0299)
        ],
        description: "Bilabial consonants (articulated with both lips)",
    });

    // Labiodental consonants
    add_class(NamedClass {
        name: "labiodental",
        aliases: &[],
        patterns: vec![
            Char('f'), Char('F'), Char('v'), Char('V'),
            Char('ɱ'), // labiodental nasal (U+0271)
            Char('ⱱ'), // labiodental flap (U+2C71)
            Char('ʋ'), // labiodental approximant (U+028B)
        ],
        description: "Labiodental consonants (lower lip to upper teeth)",
    });

    // Dental consonants
    add_class(NamedClass {
        name: "dental",
        aliases: &[],
        patterns: vec![
            Char('θ'), // voiceless dental fricative (U+03B8)
            Char('ð'), // voiced dental fricative (U+00F0)
            Digraph('t', 'h'),
            Digraph('T', 'H'),
        ],
        description: "Dental consonants (articulated at teeth)",
    });

    // Postalveolar consonants
    add_class(NamedClass {
        name: "postalveolar",
        aliases: &[],
        patterns: vec![
            Char('ʃ'), // voiceless postalveolar fricative (U+0283)
            Char('ʒ'), // voiced postalveolar fricative (U+0292)
            Digraph('s', 'h'),
            Digraph('z', 'h'),
        ],
        description: "Postalveolar consonants (just behind alveolar ridge)",
    });

    // ========================================================================
    // Extended IPA Consonants - Manner of Articulation
    // ========================================================================

    // Trill consonants
    add_class(NamedClass {
        name: "trill",
        aliases: &[],
        patterns: vec![
            Char('r'), Char('R'), // alveolar trill
            Char('ʙ'), // bilabial trill (U+0299)
            Char('ʀ'), // uvular trill (U+0280)
        ],
        description: "Trill consonants (rapid vibration)",
    });

    // Tap/flap consonants
    add_class(NamedClass {
        name: "tap",
        aliases: &["flap"],
        patterns: vec![
            Char('ɾ'), // alveolar tap (U+027E)
            Char('ɽ'), // retroflex flap (U+027D)
            Char('ⱱ'), // labiodental flap (U+2C71)
        ],
        description: "Tap/flap consonants (single rapid contact)",
    });

    // Approximant consonants
    add_class(NamedClass {
        name: "approximant",
        aliases: &[],
        patterns: vec![
            Char('ɹ'), // alveolar approximant (U+0279)
            Char('ɻ'), // retroflex approximant (U+027B)
            Char('j'), // palatal approximant
            Char('w'), Char('W'), // labio-velar approximant
            Char('ɰ'), // velar approximant (U+0270)
            Char('ʋ'), // labiodental approximant (U+028B)
        ],
        description: "Approximant consonants (narrowing without turbulence)",
    });

    // Lateral approximant consonants
    add_class(NamedClass {
        name: "lateral",
        aliases: &[],
        patterns: vec![
            Char('l'), Char('L'),
            Char('ɭ'), // retroflex lateral (U+026D)
            Char('ʎ'), // palatal lateral (U+028E)
            Char('ʟ'), // velar lateral (U+029F)
            Char('ɬ'), // voiceless lateral fricative (U+026C)
            Char('ɮ'), // voiced lateral fricative (U+026E)
        ],
        description: "Lateral consonants (airflow around sides of tongue)",
    });

    // ========================================================================
    // Non-Pulmonic Consonants
    // ========================================================================

    // Click consonants
    add_class(NamedClass {
        name: "click",
        aliases: &[],
        patterns: vec![
            Char('ʘ'), // bilabial click (U+0298)
            Char('ǀ'), // dental click (U+01C0)
            Char('ǃ'), // alveolar click (U+01C3)
            Char('ǂ'), // palatal click (U+01C2)
            Char('ǁ'), // lateral click (U+01C1)
        ],
        description: "Click consonants (ingressive velaric airstream)",
    });

    // Implosive consonants
    add_class(NamedClass {
        name: "implosive",
        aliases: &[],
        patterns: vec![
            Char('ɓ'), // voiced bilabial implosive (U+0253)
            Char('ɗ'), // voiced alveolar implosive (U+0257)
            Char('ʄ'), // voiced palatal implosive (U+0284)
            Char('ɠ'), // voiced velar implosive (U+0260)
            Char('ʛ'), // voiced uvular implosive (U+029B)
        ],
        description: "Implosive consonants (ingressive glottalic airstream)",
    });

    // ========================================================================
    // Extended IPA Vowels - Rounded/Unrounded
    // ========================================================================

    // Rounded vowels
    add_class(NamedClass {
        name: "rounded",
        aliases: &["rounded_vowel"],
        patterns: vec![
            Char('o'), Char('O'), Char('u'), Char('U'),
            Char('ʊ'), // near-close back rounded (U+028A)
            Char('ɔ'), // open-mid back rounded (U+0254)
            Char('ɒ'), // open back rounded (U+0252)
            Char('y'), // close front rounded (U+0079)
            Char('ʏ'), // near-close front rounded (U+028F)
            Char('ø'), // close-mid front rounded (U+00F8)
            Char('œ'), // open-mid front rounded (U+0153)
            Char('ɶ'), // open front rounded (U+0276)
            Char('ɵ'), // close-mid central rounded (U+0275)
            Char('ɞ'), // open-mid central rounded (U+025E)
        ],
        description: "Rounded vowels (with lip rounding)",
    });

    // Unrounded vowels
    add_class(NamedClass {
        name: "unrounded",
        aliases: &["unrounded_vowel"],
        patterns: vec![
            Char('i'), Char('I'), Char('e'), Char('E'),
            Char('a'), Char('A'),
            Char('ɪ'), // near-close front unrounded (U+026A)
            Char('ɛ'), // open-mid front unrounded (U+025B)
            Char('æ'), // near-open front unrounded (U+00E6)
            Char('ɑ'), // open back unrounded (U+0251)
            Char('ə'), // mid central (schwa) (U+0259)
            Char('ɜ'), // open-mid central unrounded (U+025C)
            Char('ʌ'), // open-mid back unrounded (U+028C)
            Char('ɯ'), // close back unrounded (U+026F)
            Char('ɤ'), // close-mid back unrounded (U+0264)
            Char('ɨ'), // close central unrounded (U+0268)
            Char('ɘ'), // close-mid central unrounded (U+0258)
            Char('ɐ'), // near-open central (U+0250)
        ],
        description: "Unrounded vowels (without lip rounding)",
    });

    // Front rounded vowels (like German ü, ö)
    add_class(NamedClass {
        name: "front_rounded",
        aliases: &[],
        patterns: vec![
            Char('y'), // close front rounded (U+0079)
            Char('ʏ'), // near-close front rounded (U+028F)
            Char('ø'), // close-mid front rounded (U+00F8)
            Char('œ'), // open-mid front rounded (U+0153)
            Char('ɶ'), // open front rounded (U+0276)
        ],
        description: "Front rounded vowels (German ü, ö sounds)",
    });

    // Rhotic vowels (r-colored)
    add_class(NamedClass {
        name: "rhotic",
        aliases: &["rhotic_vowel", "r_colored"],
        patterns: vec![
            Char('ɚ'), // schwa with hook (mid central) (U+025A)
            Char('ɝ'), // open-mid central with hook (U+025D)
            // Note: syllabic r (ɹ + combining vertical line) would need digraph support
        ],
        description: "Rhotic/r-colored vowels (American English 'er')",
    });

    // ========================================================================
    // IPA Diacritics (combining characters)
    // ========================================================================

    // Common diacritics as standalone class
    add_class(NamedClass {
        name: "diacritic",
        aliases: &["diacritics"],
        patterns: vec![
            Char('\u{0325}'), // combining ring below (voiceless)
            Char('\u{032C}'), // combining caron below (voiced)
            Char('\u{0324}'), // combining diaeresis below (breathy)
            Char('\u{0330}'), // combining tilde below (creaky)
            Char('\u{0329}'), // combining vertical line below (syllabic)
            Char('\u{032A}'), // combining bridge below (dental)
            Char('\u{033A}'), // combining inverted bridge below (apical)
            Char('\u{033C}'), // combining seagull below (linguolabial)
            Char('\u{031F}'), // combining plus sign below (advanced)
            Char('\u{0320}'), // combining minus sign below (retracted)
            Char('\u{0308}'), // combining diaeresis (centralized)
            Char('\u{033D}'), // combining x above (mid-centralized)
            Char('\u{0319}'), // combining right tack below (retracted tongue root)
            Char('\u{031A}'), // combining left angle above (no audible release)
            Char('\u{0303}'), // combining tilde (nasalized)
            Char('\u{02B0}'), // modifier letter small h (aspirated)
            Char('\u{02B7}'), // modifier letter small w (labialized)
            Char('\u{02B2}'), // modifier letter small j (palatalized)
            Char('\u{02E0}'), // modifier letter small gamma (velarized)
            Char('\u{02E4}'), // modifier letter small reversed glottal stop (pharyngealized)
            Char('\u{02BC}'), // modifier letter apostrophe (ejective)
            Char('\u{0334}'), // combining tilde overlay (velarized/pharyngealized)
            Char('\u{031C}'), // combining left half ring below (less rounded)
            Char('\u{0339}'), // combining right half ring below (more rounded)
        ],
        description: "IPA diacritics (combining characters for modification)",
    });

    // Voiceless diacritic specifically
    add_class(NamedClass {
        name: "voiceless_diacritic",
        aliases: &[],
        patterns: vec![Char('\u{0325}')], // combining ring below
        description: "Voiceless diacritic (ring below)",
    });

    // Syllabic diacritic
    add_class(NamedClass {
        name: "syllabic_diacritic",
        aliases: &[],
        patterns: vec![Char('\u{0329}')], // combining vertical line below
        description: "Syllabic diacritic (vertical line below)",
    });

    // Nasalization diacritic
    add_class(NamedClass {
        name: "nasal_diacritic",
        aliases: &["nasalized_diacritic"],
        patterns: vec![Char('\u{0303}')], // combining tilde
        description: "Nasalization diacritic (tilde above)",
    });

    // Aspiration marker
    add_class(NamedClass {
        name: "aspirated_diacritic",
        aliases: &["aspiration"],
        patterns: vec![Char('\u{02B0}')], // modifier letter small h
        description: "Aspiration marker (superscript h)",
    });

    // ========================================================================
    // Suprasegmentals (stress, length, tone)
    // ========================================================================

    // Stress markers
    add_class(NamedClass {
        name: "stress",
        aliases: &["stress_mark"],
        patterns: vec![
            Char('ˈ'), // primary stress (U+02C8)
            Char('ˌ'), // secondary stress (U+02CC)
        ],
        description: "Stress markers (primary and secondary)",
    });

    // Length markers
    add_class(NamedClass {
        name: "length",
        aliases: &["length_mark"],
        patterns: vec![
            Char('ː'), // long (U+02D0)
            Char('ˑ'), // half-long (U+02D1)
            Char('˘'), // extra-short (U+02D8)
        ],
        description: "Length markers (long, half-long, extra-short)",
    });

    // Tone markers (contour and level)
    add_class(NamedClass {
        name: "tone",
        aliases: &["tone_mark"],
        patterns: vec![
            // Level tones (Chao tone letters)
            Char('˥'), // extra high (U+02E5)
            Char('˦'), // high (U+02E6)
            Char('˧'), // mid (U+02E7)
            Char('˨'), // low (U+02E8)
            Char('˩'), // extra low (U+02E9)
            // Combining tone marks
            Char('\u{030B}'), // combining double acute accent (extra high)
            Char('\u{0301}'), // combining acute accent (high)
            Char('\u{0304}'), // combining macron (mid)
            Char('\u{0300}'), // combining grave accent (low)
            Char('\u{030F}'), // combining double grave accent (extra low)
            Char('\u{030C}'), // combining caron (rising)
            Char('\u{0302}'), // combining circumflex (falling)
            Char('\u{1DC4}'), // combining macron-acute (high rising)
            Char('\u{1DC5}'), // combining grave-macron (low rising)
            Char('\u{1DC8}'), // combining grave-acute-grave (rising-falling)
        ],
        description: "Tone markers (level and contour tones)",
    });

    // ========================================================================
    // Affricates with Tie Bars
    // ========================================================================

    // Extended affricate class with IPA tie bar forms
    add_class(NamedClass {
        name: "ipa_affricate",
        aliases: &[],
        patterns: vec![
            Char('ʧ'), // voiceless postalveolar affricate (U+02A7)
            Char('ʤ'), // voiced postalveolar affricate (U+02A4)
            Char('ʦ'), // voiceless alveolar affricate (U+02A6)
            Char('ʣ'), // voiced alveolar affricate (U+02A3)
            Char('ʨ'), // voiceless alveolo-palatal affricate (U+02A8)
            Char('ʥ'), // voiced alveolo-palatal affricate (U+02A5)
            // Also include the digraph representations
            Digraph('t', 'ʃ'),
            Digraph('d', 'ʒ'),
            Digraph('t', 's'),
            Digraph('d', 'z'),
        ],
        description: "IPA affricates (precomposed and with digraphs)",
    });

    // ========================================================================
    // All IPA Characters (comprehensive set)
    // ========================================================================

    add_class(NamedClass {
        name: "ipa",
        aliases: &["ipa_all"],
        patterns: vec![
            // IPA vowels
            Char('ə'), Char('ɪ'), Char('ʊ'), Char('ɛ'), Char('ɔ'),
            Char('æ'), Char('ʌ'), Char('ɑ'), Char('ɒ'), Char('ɜ'),
            Char('ɐ'), Char('ɚ'), Char('ɝ'), Char('ʏ'), Char('ø'),
            Char('œ'), Char('ɶ'), Char('ɵ'), Char('ɞ'), Char('ɯ'),
            Char('ɤ'), Char('ɨ'), Char('ɘ'),
            // IPA consonants
            Char('ŋ'), Char('θ'), Char('ð'), Char('ʃ'), Char('ʒ'),
            Char('ɹ'), Char('ɾ'), Char('ʔ'), Char('ɬ'), Char('ɮ'),
            Char('ʈ'), Char('ɖ'), Char('ɳ'), Char('ɽ'), Char('ʂ'),
            Char('ʐ'), Char('ɻ'), Char('ɭ'), Char('ɢ'), Char('ɴ'),
            Char('ʀ'), Char('χ'), Char('ʁ'), Char('ħ'), Char('ʕ'),
            Char('ɦ'), Char('ɟ'), Char('ɲ'), Char('ç'), Char('ʝ'),
            Char('ʎ'), Char('ɣ'), Char('ɰ'), Char('ʟ'), Char('ɸ'),
            Char('β'), Char('ʙ'), Char('ɱ'), Char('ⱱ'), Char('ʋ'),
            // Click consonants
            Char('ʘ'), Char('ǀ'), Char('ǃ'), Char('ǂ'), Char('ǁ'),
            // Implosives
            Char('ɓ'), Char('ɗ'), Char('ʄ'), Char('ɠ'), Char('ʛ'),
            // Affricates
            Char('ʧ'), Char('ʤ'), Char('ʦ'), Char('ʣ'), Char('ʨ'), Char('ʥ'),
            // Suprasegmentals
            Char('ˈ'), Char('ˌ'), Char('ː'), Char('ˑ'),
            Char('˥'), Char('˦'), Char('˧'), Char('˨'), Char('˩'),
        ],
        description: "All IPA characters (comprehensive phonetic alphabet)",
    });

    m
});

// ============================================================================
// Lookup Functions
// ============================================================================

/// Maximum length of any built-in class name.
/// Used for stack-allocated lowercase buffer.
const MAX_CLASS_NAME_LEN: usize = 20; // "nasalized_diacritic" = 19 chars

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

    // =========================================================================
    // Extended IPA Class Tests
    // =========================================================================

    #[test]
    fn test_retroflex_class() {
        let retroflex = get_named_class("retroflex").expect("retroflex class");

        // Should have retroflex consonants
        assert!(retroflex.patterns.contains(&PhonePattern::Char('ʈ'))); // U+0288
        assert!(retroflex.patterns.contains(&PhonePattern::Char('ɖ'))); // U+0256
        assert!(retroflex.patterns.contains(&PhonePattern::Char('ɻ'))); // U+027B
        assert!(retroflex.patterns.contains(&PhonePattern::Char('ʂ'))); // U+0282
    }

    #[test]
    fn test_uvular_class() {
        let uvular = get_named_class("uvular").expect("uvular class");

        assert!(uvular.patterns.contains(&PhonePattern::Char('ʁ'))); // U+0281
        assert!(uvular.patterns.contains(&PhonePattern::Char('ɢ'))); // U+0262
        assert!(uvular.patterns.contains(&PhonePattern::Char('ʀ'))); // U+0280
    }

    #[test]
    fn test_click_class() {
        let click = get_named_class("click").expect("click class");

        assert!(click.patterns.contains(&PhonePattern::Char('ʘ'))); // bilabial
        assert!(click.patterns.contains(&PhonePattern::Char('ǀ'))); // dental
        assert!(click.patterns.contains(&PhonePattern::Char('ǃ'))); // alveolar
    }

    #[test]
    fn test_implosive_class() {
        let implosive = get_named_class("implosive").expect("implosive class");

        assert!(implosive.patterns.contains(&PhonePattern::Char('ɓ'))); // U+0253
        assert!(implosive.patterns.contains(&PhonePattern::Char('ɗ'))); // U+0257
        assert!(implosive.patterns.contains(&PhonePattern::Char('ɠ'))); // U+0260
    }

    #[test]
    fn test_front_rounded_class() {
        let front_rounded = get_named_class("front_rounded").expect("front_rounded class");

        // German ü, ö sounds
        assert!(front_rounded.patterns.contains(&PhonePattern::Char('y'))); // close front rounded
        assert!(front_rounded.patterns.contains(&PhonePattern::Char('ø'))); // close-mid front rounded
        assert!(front_rounded.patterns.contains(&PhonePattern::Char('œ'))); // open-mid front rounded
    }

    #[test]
    fn test_stress_class() {
        let stress = get_named_class("stress").expect("stress class");

        assert!(stress.patterns.contains(&PhonePattern::Char('ˈ'))); // primary
        assert!(stress.patterns.contains(&PhonePattern::Char('ˌ'))); // secondary
    }

    #[test]
    fn test_length_class() {
        let length = get_named_class("length").expect("length class");

        assert!(length.patterns.contains(&PhonePattern::Char('ː'))); // long
        assert!(length.patterns.contains(&PhonePattern::Char('ˑ'))); // half-long
    }

    #[test]
    fn test_tone_class() {
        let tone = get_named_class("tone").expect("tone class");

        // Chao tone letters
        assert!(tone.patterns.contains(&PhonePattern::Char('˥'))); // extra high
        assert!(tone.patterns.contains(&PhonePattern::Char('˩'))); // extra low
    }

    #[test]
    fn test_diacritic_class() {
        let diacritic = get_named_class("diacritic").expect("diacritic class");

        // Should have common diacritics
        assert!(diacritic.patterns.contains(&PhonePattern::Char('\u{0325}'))); // voiceless
        assert!(diacritic.patterns.contains(&PhonePattern::Char('\u{0303}'))); // nasalized
        assert!(diacritic.patterns.contains(&PhonePattern::Char('\u{02B0}'))); // aspirated
    }

    #[test]
    fn test_ipa_affricate_class() {
        let affricate = get_named_class("ipa_affricate").expect("ipa_affricate class");

        // Precomposed affricates
        assert!(affricate.patterns.contains(&PhonePattern::Char('ʧ'))); // tʃ
        assert!(affricate.patterns.contains(&PhonePattern::Char('ʤ'))); // dʒ
        assert!(affricate.patterns.contains(&PhonePattern::Char('ʦ'))); // ts
    }

    #[test]
    fn test_comprehensive_ipa_class() {
        let ipa = get_named_class("ipa").expect("ipa class");

        // Should have all major IPA categories
        // Vowels
        assert!(ipa.patterns.contains(&PhonePattern::Char('ə'))); // schwa
        assert!(ipa.patterns.contains(&PhonePattern::Char('ɚ'))); // rhotic schwa

        // Consonants
        assert!(ipa.patterns.contains(&PhonePattern::Char('ŋ'))); // velar nasal
        assert!(ipa.patterns.contains(&PhonePattern::Char('ʃ'))); // sh sound
        assert!(ipa.patterns.contains(&PhonePattern::Char('ʁ'))); // uvular fricative

        // Clicks
        assert!(ipa.patterns.contains(&PhonePattern::Char('ǃ'))); // alveolar click

        // Implosives
        assert!(ipa.patterns.contains(&PhonePattern::Char('ɓ'))); // bilabial implosive

        // Suprasegmentals
        assert!(ipa.patterns.contains(&PhonePattern::Char('ˈ'))); // primary stress
        assert!(ipa.patterns.contains(&PhonePattern::Char('ː'))); // long
    }

    #[test]
    fn test_tap_flap_alias() {
        let tap = get_named_class("tap").expect("tap class");
        let flap = get_named_class("flap").expect("flap alias");

        assert_eq!(tap.patterns.len(), flap.patterns.len());
        assert!(tap.patterns.contains(&PhonePattern::Char('ɾ'))); // alveolar tap
    }

    #[test]
    fn test_place_of_articulation_classes() {
        // Test various place of articulation classes
        let bilabial = get_named_class("bilabial").expect("bilabial");
        let alveolar = get_named_class("alveolar").expect("alveolar");
        let velar = get_named_class("velar").expect("velar");
        let palatal = get_named_class("palatal").expect("palatal");

        assert!(bilabial.patterns.contains(&PhonePattern::Char('p')));
        assert!(bilabial.patterns.contains(&PhonePattern::Char('m')));

        assert!(alveolar.patterns.contains(&PhonePattern::Char('t')));
        assert!(alveolar.patterns.contains(&PhonePattern::Char('n')));

        assert!(velar.patterns.contains(&PhonePattern::Char('k')));
        assert!(velar.patterns.contains(&PhonePattern::Char('ŋ')));

        assert!(palatal.patterns.contains(&PhonePattern::Char('ç'))); // voiceless palatal fricative
        assert!(palatal.patterns.contains(&PhonePattern::Char('j'))); // palatal approximant
    }

    #[test]
    fn test_manner_of_articulation_classes() {
        let trill = get_named_class("trill").expect("trill");
        let approximant = get_named_class("approximant").expect("approximant");
        let lateral = get_named_class("lateral").expect("lateral");

        assert!(trill.patterns.contains(&PhonePattern::Char('r')));
        assert!(trill.patterns.contains(&PhonePattern::Char('ʀ'))); // uvular trill

        assert!(approximant.patterns.contains(&PhonePattern::Char('ɹ'))); // alveolar approx
        assert!(approximant.patterns.contains(&PhonePattern::Char('j'))); // palatal approx

        assert!(lateral.patterns.contains(&PhonePattern::Char('l')));
        assert!(lateral.patterns.contains(&PhonePattern::Char('ɬ'))); // lateral fricative
    }
}
