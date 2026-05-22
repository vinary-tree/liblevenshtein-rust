// ============================================================================
// Named Classes Registry
// ============================================================================

use std::collections::HashMap;
use std::sync::LazyLock;

use super::phone_pattern::PhonePattern;

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
    use PhonePattern::{Char, Digraph, Hexagraph, Pentagraph, Tetragraph, Trigraph};

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
        patterns: ('a'..='z').chain('A'..='Z').map(Char).collect(),
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
            Char('a'),
            Char('e'),
            Char('i'),
            Char('o'),
            Char('u'),
            Char('A'),
            Char('E'),
            Char('I'),
            Char('O'),
            Char('U'),
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
            Char('e'),
            Char('i'),
            Char('E'),
            Char('I'),
            Char('ɪ'),
            Char('ɛ'),
            Char('æ'),
        ],
        description: "Front vowels (e, i and IPA equivalents)",
    });

    // Back vowels
    add_class(NamedClass {
        name: "back_vowel",
        aliases: &[],
        patterns: vec![
            Char('a'),
            Char('o'),
            Char('u'),
            Char('A'),
            Char('O'),
            Char('U'),
            Char('ʊ'),
            Char('ɔ'),
            Char('ɑ'),
            Char('ɒ'),
        ],
        description: "Back vowels (a, o, u and IPA equivalents)",
    });

    // High vowels
    add_class(NamedClass {
        name: "high_vowel",
        aliases: &[],
        patterns: vec![
            Char('i'),
            Char('u'),
            Char('I'),
            Char('U'),
            Char('ɪ'),
            Char('ʊ'),
        ],
        description: "High vowels (i, u and IPA equivalents)",
    });

    // Low vowels
    add_class(NamedClass {
        name: "low_vowel",
        aliases: &[],
        patterns: vec![
            Char('a'),
            Char('A'),
            Char('æ'),
            Char('ɑ'),
            Char('ɒ'),
            Char('ɐ'),
        ],
        description: "Low vowels (a and IPA equivalents)",
    });

    // Mid vowels
    add_class(NamedClass {
        name: "mid_vowel",
        aliases: &[],
        patterns: vec![
            Char('e'),
            Char('o'),
            Char('E'),
            Char('O'),
            Char('ə'),
            Char('ɛ'),
            Char('ɔ'),
            Char('ʌ'),
            Char('ɜ'),
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
            Char('b'),
            Char('c'),
            Char('d'),
            Char('f'),
            Char('g'),
            Char('h'),
            Char('j'),
            Char('k'),
            Char('l'),
            Char('m'),
            Char('n'),
            Char('p'),
            Char('q'),
            Char('r'),
            Char('s'),
            Char('t'),
            Char('v'),
            Char('w'),
            Char('x'),
            Char('y'),
            Char('z'),
            Char('B'),
            Char('C'),
            Char('D'),
            Char('F'),
            Char('G'),
            Char('H'),
            Char('J'),
            Char('K'),
            Char('L'),
            Char('M'),
            Char('N'),
            Char('P'),
            Char('Q'),
            Char('R'),
            Char('S'),
            Char('T'),
            Char('V'),
            Char('W'),
            Char('X'),
            Char('Y'),
            Char('Z'),
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
            Char('ʍ'), // voiceless labialized velar approximant (which vs witch)
        ],
        description: "All consonants (ASCII + IPA)",
    });

    // Stop consonants (plosives)
    add_class(NamedClass {
        name: "stop",
        aliases: &["plosive"],
        patterns: vec![
            Char('p'),
            Char('b'),
            Char('t'),
            Char('d'),
            Char('k'),
            Char('g'),
            Char('P'),
            Char('B'),
            Char('T'),
            Char('D'),
            Char('K'),
            Char('G'),
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
            Char('f'),
            Char('v'),
            Char('s'),
            Char('z'),
            Char('h'),
            Char('F'),
            Char('V'),
            Char('S'),
            Char('Z'),
            Char('H'),
            Char('ʃ'),
            Char('ʒ'),
            Char('θ'),
            Char('ð'),
            Char('ɬ'),
            // Digraphs
            Digraph('s', 'h'), // sh -> ʃ
            Digraph('S', 'h'),
            Digraph('S', 'H'),
            Digraph('z', 'h'), // zh -> ʒ
            Digraph('Z', 'h'),
            Digraph('Z', 'H'),
            Digraph('t', 'h'), // th -> θ/ð
            Digraph('T', 'h'),
            Digraph('T', 'H'),
        ],
        description: "Fricative consonants (includes sh, th, zh digraphs)",
    });

    // Nasal consonants (with digraphs)
    add_class(NamedClass {
        name: "nasal",
        aliases: &[],
        patterns: vec![
            Char('m'),
            Char('n'),
            Char('M'),
            Char('N'),
            Char('ŋ'),
            Digraph('n', 'g'), // ng -> ŋ
            Digraph('N', 'g'),
            Digraph('N', 'G'),
        ],
        description: "Nasal consonants (includes ng digraph)",
    });

    // Liquid consonants
    add_class(NamedClass {
        name: "liquid",
        aliases: &[],
        patterns: vec![
            Char('l'),
            Char('r'),
            Char('L'),
            Char('R'),
            Char('ɹ'),
            Char('ɾ'),
        ],
        description: "Liquid consonants (l, r)",
    });

    // Glides (semivowels)
    add_class(NamedClass {
        name: "glide",
        aliases: &["semivowel"],
        patterns: vec![
            Char('w'),
            Char('y'),
            Char('W'),
            Char('Y'),
            Char('j'),
            Char('J'), // IPA j = English y
            Char('ʍ'), // voiceless labialized velar approximant (U+028D) - "which" vs "witch"
        ],
        description: "Glides/semivowels (w, y, ʍ)",
    });

    // Affricates (with digraphs)
    add_class(NamedClass {
        name: "affricate",
        aliases: &[],
        patterns: vec![
            Digraph('c', 'h'), // ch -> tʃ
            Digraph('C', 'h'),
            Digraph('C', 'H'),
            Digraph('t', 's'), // ts
            Digraph('d', 'z'), // dz
            Digraph('d', 'j'), // dj (sometimes dʒ)
        ],
        description: "Affricate consonants (ch, ts, dz)",
    });

    // Aspirated affricates (Khmer, Quechua, Mandarin)
    add_class(NamedClass {
        name: "aspirated_affricate",
        aliases: &[],
        patterns: vec![
            // Aspirated affricates - 3 chars each (base + fricative + aspiration)
            Trigraph('t', 's', 'ʰ'), // t͡sʰ - aspirated alveolar affricate
            Trigraph('t', 'ʃ', 'ʰ'), // t͡ʃʰ - aspirated postalveolar affricate
            Trigraph('t', 'ɕ', 'ʰ'), // t͡ɕʰ - aspirated alveo-palatal affricate (Khmer, Mandarin)
            Trigraph('t', 'ʂ', 'ʰ'), // t͡ʂʰ - aspirated retroflex affricate (Mandarin)
        ],
        description: "Aspirated affricates (Khmer, Quechua, Mandarin)",
    });

    // Voiced consonants
    add_class(NamedClass {
        name: "voiced",
        aliases: &[],
        patterns: vec![
            Char('b'),
            Char('d'),
            Char('g'),
            Char('v'),
            Char('z'),
            Char('B'),
            Char('D'),
            Char('G'),
            Char('V'),
            Char('Z'),
            Char('l'),
            Char('m'),
            Char('n'),
            Char('r'),
            Char('w'),
            Char('L'),
            Char('M'),
            Char('N'),
            Char('R'),
            Char('W'),
            Char('ð'),
            Char('ʒ'),
            Char('ŋ'),
            Char('ɹ'),
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
            Char('p'),
            Char('t'),
            Char('k'),
            Char('f'),
            Char('s'),
            Char('h'),
            Char('P'),
            Char('T'),
            Char('K'),
            Char('F'),
            Char('S'),
            Char('H'),
            Char('c'),
            Char('q'),
            Char('x'),
            Char('C'),
            Char('Q'),
            Char('X'),
            Char('θ'),
            Char('ʃ'),
            Char('ʔ'),
            Digraph('s', 'h'),
            Digraph('t', 'h'),
            Digraph('c', 'h'),
        ],
        description: "Voiceless consonants",
    });

    // ========================================================================
    // Natural Classes (Phonological Feature Combinations)
    // ========================================================================
    // These classes represent major phonological natural classes used in
    // phonological rules and constraints across languages.

    // Obstruents: sounds that obstruct airflow (stops + fricatives + affricates)
    add_class(NamedClass {
        name: "obstruent",
        aliases: &[],
        patterns: vec![
            // Stops
            Char('p'),
            Char('b'),
            Char('t'),
            Char('d'),
            Char('k'),
            Char('g'),
            Char('P'),
            Char('B'),
            Char('T'),
            Char('D'),
            Char('K'),
            Char('G'),
            Char('ʔ'), // glottal stop
            Char('q'),
            Char('Q'), // uvular stop
            Char('ʈ'),
            Char('ɖ'), // retroflex stops
            Char('c'),
            Char('ɟ'), // palatal stops
            // Fricatives
            Char('f'),
            Char('v'),
            Char('s'),
            Char('z'),
            Char('h'),
            Char('F'),
            Char('V'),
            Char('S'),
            Char('Z'),
            Char('H'),
            Char('ʃ'),
            Char('ʒ'),
            Char('θ'),
            Char('ð'),
            Char('ɬ'),
            Char('ɮ'),
            Char('x'),
            Char('ɣ'),
            Char('χ'),
            Char('ʁ'), // velar/uvular fricatives
            Char('ɕ'),
            Char('ʑ'), // alveo-palatal fricatives
            Char('ʂ'),
            Char('ʐ'), // retroflex fricatives
            Char('ç'),
            Char('ʝ'), // palatal fricatives
            Char('ħ'),
            Char('ʕ'), // pharyngeal fricatives
            Char('ɸ'),
            Char('β'), // bilabial fricatives
            // Affricates (as digraphs)
            Digraph('t', 'ʃ'),
            Digraph('d', 'ʒ'), // postalveolar affricates
            Digraph('t', 's'),
            Digraph('d', 'z'), // alveolar affricates
            Digraph('t', 'ɕ'),
            Digraph('d', 'ʑ'), // alveo-palatal affricates
            Digraph('c', 'h'),
            Digraph('C', 'h'),
        ],
        description: "Obstruents: stops + fricatives + affricates (obstruct airflow)",
    });

    // Sonorants: sounds with spontaneous voicing (nasals + approximants + vowels)
    add_class(NamedClass {
        name: "sonorant",
        aliases: &[],
        patterns: vec![
            // Nasals
            Char('m'),
            Char('n'),
            Char('M'),
            Char('N'),
            Char('ŋ'),
            Char('ɲ'),
            Char('ɴ'), // velar, palatal, uvular nasals
            Char('ɱ'), // labiodental nasal
            Char('ɳ'), // retroflex nasal
            // Liquids (laterals + rhotics)
            Char('l'),
            Char('r'),
            Char('L'),
            Char('R'),
            Char('ɹ'),
            Char('ɾ'),
            Char('ɻ'), // approximant and tap rhotics
            Char('ʀ'),
            Char('ʙ'), // uvular and bilabial trills
            Char('ɭ'),
            Char('ɽ'), // retroflex lateral and flap
            Char('ʎ'), // palatal lateral
            Char('ɫ'), // velarized lateral (dark L)
            // Glides
            Char('w'),
            Char('y'),
            Char('W'),
            Char('Y'),
            Char('j'),
            Char('J'),
            Char('ʍ'),
            Char('ɥ'), // labial-palatal approximant
            // Vowels (all vowels are sonorants)
            Char('a'),
            Char('e'),
            Char('i'),
            Char('o'),
            Char('u'),
            Char('A'),
            Char('E'),
            Char('I'),
            Char('O'),
            Char('U'),
            Char('ə'),
            Char('ɪ'),
            Char('ʊ'),
            Char('ɛ'),
            Char('ɔ'),
            Char('æ'),
            Char('ʌ'),
            Char('ɑ'),
            Char('ɒ'),
            Char('ɜ'),
            Char('ɐ'),
            Char('y'),
            Char('ø'),
            Char('œ'),
            Char('ɯ'),
            Char('ɨ'),
            Char('ʉ'),
        ],
        description: "Sonorants: nasals + approximants + vowels (spontaneous voicing)",
    });

    // Continuants: sounds where airflow continues (fricatives + approximants + vowels)
    // Excludes: stops, affricates (which have complete closure), nasals (debated)
    add_class(NamedClass {
        name: "continuant",
        aliases: &[],
        patterns: vec![
            // Fricatives
            Char('f'),
            Char('v'),
            Char('s'),
            Char('z'),
            Char('h'),
            Char('F'),
            Char('V'),
            Char('S'),
            Char('Z'),
            Char('H'),
            Char('ʃ'),
            Char('ʒ'),
            Char('θ'),
            Char('ð'),
            Char('ɬ'),
            Char('ɮ'),
            Char('x'),
            Char('ɣ'),
            Char('χ'),
            Char('ʁ'),
            Char('ɕ'),
            Char('ʑ'),
            Char('ʂ'),
            Char('ʐ'),
            Char('ç'),
            Char('ʝ'),
            Char('ħ'),
            Char('ʕ'),
            Char('ɸ'),
            Char('β'),
            // Approximants (liquids + glides)
            Char('l'),
            Char('r'),
            Char('L'),
            Char('R'),
            Char('ɹ'),
            Char('ɾ'),
            Char('ɻ'),
            Char('ɭ'),
            Char('ʎ'),
            Char('ɫ'),
            Char('w'),
            Char('y'),
            Char('W'),
            Char('Y'),
            Char('j'),
            Char('J'),
            Char('ʍ'),
            Char('ɥ'),
            // Vowels
            Char('a'),
            Char('e'),
            Char('i'),
            Char('o'),
            Char('u'),
            Char('A'),
            Char('E'),
            Char('I'),
            Char('O'),
            Char('U'),
            Char('ə'),
            Char('ɪ'),
            Char('ʊ'),
            Char('ɛ'),
            Char('ɔ'),
            Char('æ'),
            Char('ʌ'),
            Char('ɑ'),
            Char('ɒ'),
            Char('ɜ'),
            Char('ɐ'),
            Char('y'),
            Char('ø'),
            Char('œ'),
            Char('ɯ'),
            Char('ɨ'),
            Char('ʉ'),
        ],
        description: "Continuants: fricatives + approximants + vowels (continuous airflow)",
    });

    // Voiceless fricatives (for voicing assimilation rules)
    add_class(NamedClass {
        name: "voiceless_fricative",
        aliases: &[],
        patterns: vec![
            Char('f'),
            Char('F'),
            Char('s'),
            Char('S'),
            Char('h'),
            Char('H'),
            Char('θ'), // voiceless dental fricative
            Char('ʃ'), // voiceless postalveolar fricative
            Char('ɬ'), // voiceless lateral fricative
            Char('x'), // voiceless velar fricative
            Char('χ'), // voiceless uvular fricative
            Char('ɕ'), // voiceless alveo-palatal fricative
            Char('ʂ'), // voiceless retroflex fricative
            Char('ç'), // voiceless palatal fricative
            Char('ħ'), // voiceless pharyngeal fricative
            Char('ɸ'), // voiceless bilabial fricative
            Digraph('s', 'h'),
            Digraph('S', 'h'),
            Digraph('S', 'H'),
            Digraph('t', 'h'),
            Digraph('T', 'h'),
            Digraph('T', 'H'),
        ],
        description: "Voiceless fricatives (for voicing assimilation)",
    });

    // Voiced fricatives (for voicing assimilation rules)
    add_class(NamedClass {
        name: "voiced_fricative",
        aliases: &[],
        patterns: vec![
            Char('v'),
            Char('V'),
            Char('z'),
            Char('Z'),
            Char('ð'), // voiced dental fricative
            Char('ʒ'), // voiced postalveolar fricative
            Char('ɮ'), // voiced lateral fricative
            Char('ɣ'), // voiced velar fricative
            Char('ʁ'), // voiced uvular fricative
            Char('ʑ'), // voiced alveo-palatal fricative
            Char('ʐ'), // voiced retroflex fricative
            Char('ʝ'), // voiced palatal fricative
            Char('ʕ'), // voiced pharyngeal fricative
            Char('β'), // voiced bilabial fricative
            Digraph('z', 'h'),
            Digraph('Z', 'h'),
            Digraph('Z', 'H'),
        ],
        description: "Voiced fricatives (for voicing assimilation)",
    });

    // Sibilant fricatives (for sibilant harmony rules)
    add_class(NamedClass {
        name: "sibilant_fricative",
        aliases: &["sibilant"],
        patterns: vec![
            // Alveolar sibilants
            Char('s'),
            Char('z'),
            Char('S'),
            Char('Z'),
            // Postalveolar sibilants
            Char('ʃ'),
            Char('ʒ'),
            // Retroflex sibilants
            Char('ʂ'),
            Char('ʐ'),
            // Alveo-palatal sibilants
            Char('ɕ'),
            Char('ʑ'),
            // Digraphs
            Digraph('s', 'h'),
            Digraph('S', 'h'),
            Digraph('z', 'h'),
            Digraph('Z', 'h'),
        ],
        description: "Sibilant fricatives (for sibilant harmony)",
    });

    // ========================================================================
    // ASCII-Only Subsets
    // ========================================================================

    add_class(NamedClass {
        name: "ascii_vowel",
        aliases: &[],
        patterns: vec![
            Char('a'),
            Char('e'),
            Char('i'),
            Char('o'),
            Char('u'),
            Char('A'),
            Char('E'),
            Char('I'),
            Char('O'),
            Char('U'),
        ],
        description: "ASCII vowels only (a, e, i, o, u)",
    });

    add_class(NamedClass {
        name: "ascii_consonant",
        aliases: &[],
        patterns: vec![
            Char('b'),
            Char('c'),
            Char('d'),
            Char('f'),
            Char('g'),
            Char('h'),
            Char('j'),
            Char('k'),
            Char('l'),
            Char('m'),
            Char('n'),
            Char('p'),
            Char('q'),
            Char('r'),
            Char('s'),
            Char('t'),
            Char('v'),
            Char('w'),
            Char('x'),
            Char('y'),
            Char('z'),
            Char('B'),
            Char('C'),
            Char('D'),
            Char('F'),
            Char('G'),
            Char('H'),
            Char('J'),
            Char('K'),
            Char('L'),
            Char('M'),
            Char('N'),
            Char('P'),
            Char('Q'),
            Char('R'),
            Char('S'),
            Char('T'),
            Char('V'),
            Char('W'),
            Char('X'),
            Char('Y'),
            Char('Z'),
        ],
        description: "ASCII consonants only",
    });

    add_class(NamedClass {
        name: "ascii_front",
        aliases: &[],
        patterns: vec![Char('e'), Char('i'), Char('E'), Char('I')],
        description: "ASCII front vowels only (e, i)",
    });

    add_class(NamedClass {
        name: "ascii_back",
        aliases: &[],
        patterns: vec![
            Char('a'),
            Char('o'),
            Char('u'),
            Char('A'),
            Char('O'),
            Char('U'),
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
            Char('ə'),
            Char('ɪ'),
            Char('ʊ'),
            Char('ɛ'),
            Char('ɔ'),
            Char('æ'),
            Char('ʌ'),
            Char('ɑ'),
            Char('ɒ'),
            Char('ɜ'),
            Char('ɐ'),
        ],
        description: "IPA vowels only (non-ASCII)",
    });

    add_class(NamedClass {
        name: "ipa_consonant",
        aliases: &[],
        patterns: vec![
            Char('ŋ'),
            Char('θ'),
            Char('ð'),
            Char('ʃ'),
            Char('ʒ'),
            Char('ɹ'),
            Char('ɾ'),
            Char('ʔ'),
            Char('ɬ'),
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
            Char('k'),
            Char('K'),
            Char('g'),
            Char('G'),
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
            Char('t'),
            Char('T'),
            Char('d'),
            Char('D'),
            Char('n'),
            Char('N'),
            Char('s'),
            Char('S'),
            Char('z'),
            Char('Z'),
            Char('ɹ'), // alveolar approximant (U+0279)
            Char('ɾ'), // alveolar tap (U+027E)
            Char('r'),
            Char('R'), // alveolar trill
            Char('l'),
            Char('L'),
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
            Char('p'),
            Char('P'),
            Char('b'),
            Char('B'),
            Char('m'),
            Char('M'),
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
            Char('f'),
            Char('F'),
            Char('v'),
            Char('V'),
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
            Char('r'),
            Char('R'), // alveolar trill
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
            Char('w'),
            Char('W'), // labio-velar approximant
            Char('ɰ'), // velar approximant (U+0270)
            Char('ʋ'), // labiodental approximant (U+028B)
            Char('ʍ'), // voiceless labialized velar approximant (U+028D)
        ],
        description: "Approximant consonants (narrowing without turbulence)",
    });

    // Lateral approximant consonants
    add_class(NamedClass {
        name: "lateral",
        aliases: &[],
        patterns: vec![
            Char('l'),
            Char('L'),
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

    // Prenasalized clicks (Xhosa, Zulu, other Bantu languages)
    add_class(NamedClass {
        name: "prenasalized_click",
        aliases: &["nasal_click"],
        patterns: vec![
            // Prenasalized aspirated clicks (Xhosa) - 3 chars each
            Trigraph('ŋ', 'ǀ', 'ʰ'), // ŋǀʰ - prenasalized dental click + aspiration
            Trigraph('ŋ', 'ǃ', 'ʰ'), // ŋǃʰ - prenasalized postalveolar click + aspiration
            Trigraph('ŋ', 'ǁ', 'ʰ'), // ŋǁʰ - prenasalized lateral click + aspiration
            // Prenasalized voiced clicks (Zulu) - 3 chars each
            Trigraph('ŋ', 'ɡ', 'ǀ'), // ŋɡǀ - prenasalized voiced dental click
            Trigraph('ŋ', 'ɡ', 'ǃ'), // ŋɡǃ - prenasalized voiced postalveolar click
            Trigraph('ŋ', 'ɡ', 'ǁ'), // ŋɡǁ - prenasalized voiced lateral click
            // Prenasalized voiced aspirated clicks (rare, 4 chars) - TETRAGRAPHS
            Tetragraph('ŋ', 'ɡ', 'ǀ', 'ʰ'), // ŋɡǀʰ - prenasalized voiced dental + aspiration
            Tetragraph('ŋ', 'ɡ', 'ǃ', 'ʰ'), // ŋɡǃʰ - prenasalized voiced postalveolar + aspiration
            Tetragraph('ŋ', 'ɡ', 'ǁ', 'ʰ'), // ŋɡǁʰ - prenasalized voiced lateral + aspiration
            // Prenasalized voiced aspirated labialized clicks (5 chars) - PENTAGRAPHS
            // These are attested in Khoisan languages (!Xóõ, Taa)
            Pentagraph('ŋ', 'ɡ', 'ǀ', 'ʰ', 'ʷ'), // ŋɡǀʰʷ - prenasalized voiced dental + aspiration + labialization
            Pentagraph('ŋ', 'ɡ', 'ǃ', 'ʰ', 'ʷ'), // ŋɡǃʰʷ - prenasalized voiced postalveolar + aspiration + labialization
            Pentagraph('ŋ', 'ɡ', 'ǁ', 'ʰ', 'ʷ'), // ŋɡǁʰʷ - prenasalized voiced lateral + aspiration + labialization
            // Glottalized prenasalized voiced aspirated labialized clicks (6 chars) - HEXAGRAPHS
            // These represent the most complex click consonants, attested in Khoisan languages (Taa, !Xóõ)
            Hexagraph('ŋ', 'ɡ', 'ǀ', 'ʰ', 'ʷ', 'ʼ'), // ŋɡǀʰʷʼ - glottalized prenasalized voiced dental + aspiration + labialization
            Hexagraph('ŋ', 'ɡ', 'ǃ', 'ʰ', 'ʷ', 'ʼ'), // ŋɡǃʰʷʼ - glottalized prenasalized voiced postalveolar + aspiration + labialization
            Hexagraph('ŋ', 'ɡ', 'ǁ', 'ʰ', 'ʷ', 'ʼ'), // ŋɡǁʰʷʼ - glottalized prenasalized voiced lateral + aspiration + labialization
            // Prenasalized labialized clicks without aspiration (4 chars) - TETRAGRAPHS
            Tetragraph('ŋ', 'ɡ', 'ǀ', 'ʷ'), // ŋɡǀʷ - prenasalized voiced dental + labialization
            Tetragraph('ŋ', 'ɡ', 'ǃ', 'ʷ'), // ŋɡǃʷ - prenasalized voiced postalveolar + labialization
            Tetragraph('ŋ', 'ɡ', 'ǁ', 'ʷ'), // ŋɡǁʷ - prenasalized voiced lateral + labialization
        ],
        description:
            "Prenasalized clicks (Xhosa aspirated, Zulu voiced, Khoisan labialized/glottalized)",
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
            Char('o'),
            Char('O'),
            Char('u'),
            Char('U'),
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
            Char('i'),
            Char('I'),
            Char('e'),
            Char('E'),
            Char('a'),
            Char('A'),
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
            Char('ə'),
            Char('ɪ'),
            Char('ʊ'),
            Char('ɛ'),
            Char('ɔ'),
            Char('æ'),
            Char('ʌ'),
            Char('ɑ'),
            Char('ɒ'),
            Char('ɜ'),
            Char('ɐ'),
            Char('ɚ'),
            Char('ɝ'),
            Char('ʏ'),
            Char('ø'),
            Char('œ'),
            Char('ɶ'),
            Char('ɵ'),
            Char('ɞ'),
            Char('ɯ'),
            Char('ɤ'),
            Char('ɨ'),
            Char('ɘ'),
            // IPA consonants
            Char('ŋ'),
            Char('θ'),
            Char('ð'),
            Char('ʃ'),
            Char('ʒ'),
            Char('ɹ'),
            Char('ɾ'),
            Char('ʔ'),
            Char('ɬ'),
            Char('ɮ'),
            Char('ʍ'),
            Char('ʈ'),
            Char('ɖ'),
            Char('ɳ'),
            Char('ɽ'),
            Char('ʂ'),
            Char('ʐ'),
            Char('ɻ'),
            Char('ɭ'),
            Char('ɢ'),
            Char('ɴ'),
            Char('ʀ'),
            Char('χ'),
            Char('ʁ'),
            Char('ħ'),
            Char('ʕ'),
            Char('ɦ'),
            Char('ɟ'),
            Char('ɲ'),
            Char('ç'),
            Char('ʝ'),
            Char('ʎ'),
            Char('ɣ'),
            Char('ɰ'),
            Char('ʟ'),
            Char('ɸ'),
            Char('β'),
            Char('ʙ'),
            Char('ɱ'),
            Char('ⱱ'),
            Char('ʋ'),
            // Click consonants
            Char('ʘ'),
            Char('ǀ'),
            Char('ǃ'),
            Char('ǂ'),
            Char('ǁ'),
            // Implosives
            Char('ɓ'),
            Char('ɗ'),
            Char('ʄ'),
            Char('ɠ'),
            Char('ʛ'),
            // Affricates
            Char('ʧ'),
            Char('ʤ'),
            Char('ʦ'),
            Char('ʣ'),
            Char('ʨ'),
            Char('ʥ'),
            // Suprasegmentals
            Char('ˈ'),
            Char('ˌ'),
            Char('ː'),
            Char('ˑ'),
            Char('˥'),
            Char('˦'),
            Char('˧'),
            Char('˨'),
            Char('˩'),
        ],
        description: "All IPA characters (comprehensive phonetic alphabet)",
    });

    // ========================================================================
    // Aspiration & Tenseness Classes (Korean, Hindi, Thai, Mandarin)
    // ========================================================================

    add_class(NamedClass {
        name: "aspirated",
        aliases: &["aspirated_stop"],
        patterns: vec![
            // Aspirated stops (with superscript h modifier)
            Digraph('p', 'ʰ'),
            Digraph('t', 'ʰ'),
            Digraph('k', 'ʰ'),
            Digraph('ʈ', 'ʰ'), // retroflex aspirated
            Digraph('c', 'ʰ'), // palatal aspirated
            Digraph('q', 'ʰ'), // uvular aspirated
            // Aspirated affricates
            Digraph('ʧ', 'ʰ'), // postalveolar affricate aspirated
            Digraph('ʨ', 'ʰ'), // alveo-palatal affricate aspirated
            Digraph('ʦ', 'ʰ'), // alveolar affricate aspirated
            // Modifier letter small h (aspiration marker standalone)
            Char('\u{02B0}'), // ʰ
        ],
        description: "Aspirated consonants (Korean, Hindi, Thai, Mandarin)",
    });

    add_class(NamedClass {
        name: "tense",
        aliases: &["fortis"],
        patterns: vec![
            // Korean tense consonants (with tenseness marker ͈ U+0348)
            Digraph('p', '\u{0348}'),
            Digraph('t', '\u{0348}'),
            Digraph('k', '\u{0348}'),
            Digraph('s', '\u{0348}'),
            Digraph('ʧ', '\u{0348}'),
            // Combining double vertical line below (tenseness marker standalone)
            Char('\u{0348}'), // ͈
        ],
        description: "Tense/fortis consonants (Korean ssang consonants)",
    });

    // ========================================================================
    // Secondary Articulation Classes (Arabic, Slavic)
    // ========================================================================

    add_class(NamedClass {
        name: "pharyngealized",
        aliases: &["emphatic"],
        patterns: vec![
            // Arabic emphatic consonants (with pharyngealization marker ˤ U+02E4)
            Digraph('s', 'ˤ'),
            Digraph('d', 'ˤ'),
            Digraph('t', 'ˤ'),
            Digraph('ð', 'ˤ'),
            Digraph('z', 'ˤ'),
            Digraph('l', 'ˤ'),
            Digraph('r', 'ˤ'),
            // Modifier letter small reversed glottal stop (pharyngealization)
            Char('\u{02E4}'), // ˤ
        ],
        description: "Pharyngealized/emphatic consonants (Arabic emphatics)",
    });

    add_class(NamedClass {
        name: "labialized",
        aliases: &[],
        patterns: vec![
            // Labialized consonants (with superscript w)
            Digraph('k', 'ʷ'),
            Digraph('g', 'ʷ'),
            Digraph('x', 'ʷ'),
            Digraph('ɣ', 'ʷ'),
            Digraph('q', 'ʷ'),
            Digraph('ʁ', 'ʷ'),
            // Modifier letter small w (labialization marker)
            Char('\u{02B7}'), // ʷ
        ],
        description: "Labialized consonants (with lip rounding)",
    });

    add_class(NamedClass {
        name: "velarized",
        aliases: &["dark"],
        patterns: vec![
            // Velarized L (dark L) - most common
            Char('ɫ'), // U+026B velarized L
            // Modifier letter small gamma (velarization marker)
            Char('\u{02E0}'), // ˠ
            // Combining tilde overlay (velarization diacritic)
            Char('\u{0334}'), // ̴
            // Common velarized consonants
            Digraph('l', '\u{0334}'), // l with tilde overlay
            Digraph('n', '\u{0334}'), // n with tilde overlay
        ],
        description: "Velarized consonants (dark L, etc.)",
    });

    // ========================================================================
    // Alveo-Palatal Class (Japanese, Korean, Polish)
    // ========================================================================

    add_class(NamedClass {
        name: "alveopalatal",
        aliases: &["alveolo_palatal"],
        patterns: vec![
            // Alveo-palatal fricatives
            Char('ɕ'), // voiceless alveo-palatal fricative (Mandarin x, Japanese sh)
            Char('ʑ'), // voiced alveo-palatal fricative
            // Alveo-palatal affricates (precomposed)
            Char('ʨ'), // voiceless alveo-palatal affricate (U+02A8)
            Char('ʥ'), // voiced alveo-palatal affricate (U+02A5)
            // Alveo-palatal nasal
            Char('ȵ'), // U+0235 alveo-palatal nasal
            // Tie bar representations
            Digraph('t', 'ɕ'),
            Digraph('d', 'ʑ'),
        ],
        description: "Alveo-palatal consonants (Japanese し/じ, Korean ㅈ/ㅊ, Polish ś/ź)",
    });

    // ========================================================================
    // Lateral Fricative Class (Welsh, Zulu)
    // ========================================================================

    add_class(NamedClass {
        name: "lateral_fricative",
        aliases: &[],
        patterns: vec![
            Char('ɬ'), // voiceless lateral fricative (Welsh ll)
            Char('ɮ'), // voiced lateral fricative
        ],
        description: "Lateral fricative consonants (Welsh ll, Zulu hl)",
    });

    // ========================================================================
    // Nasal Vowels (Portuguese, French, Polish)
    // ========================================================================

    add_class(NamedClass {
        name: "nasal_vowel",
        aliases: &["nasalized_vowel"],
        patterns: vec![
            // Precomposed nasal vowels (common in orthography)
            Char('ã'), // nasal a
            Char('ẽ'), // nasal e
            Char('ĩ'), // nasal i
            Char('õ'), // nasal o
            Char('ũ'), // nasal u
            // Polish nasal vowels
            Char('ą'), // nasal o (Polish)
            Char('ę'), // nasal e (Polish)
            // IPA nasal vowels (vowel + combining tilde)
            Digraph('a', '\u{0303}'), // ã
            Digraph('e', '\u{0303}'), // ẽ
            Digraph('i', '\u{0303}'), // ĩ
            Digraph('o', '\u{0303}'), // õ
            Digraph('u', '\u{0303}'), // ũ
            Digraph('ɐ', '\u{0303}'), // nasal near-open central
            Digraph('ɔ', '\u{0303}'), // nasal open-mid back rounded
            Digraph('ɛ', '\u{0303}'), // nasal open-mid front
            Digraph('œ', '\u{0303}'), // nasal open-mid front rounded
            // Combining tilde (nasalization diacritic standalone)
            Char('\u{0303}'), // ̃
        ],
        description: "Nasal vowels (Portuguese ã, French on/an, Polish ę/ą)",
    });

    // ========================================================================
    // Diphthongs (English, German, Dutch)
    // ========================================================================

    add_class(NamedClass {
        name: "diphthong",
        aliases: &[],
        patterns: vec![
            // English diphthongs
            Digraph('a', 'ɪ'), // PRICE
            Digraph('a', 'ʊ'), // MOUTH
            Digraph('e', 'ɪ'), // FACE
            Digraph('o', 'ʊ'), // GOAT (American)
            Digraph('ə', 'ʊ'), // GOAT (British)
            Digraph('ɔ', 'ɪ'), // CHOICE
            // Centering diphthongs
            Digraph('i', 'ə'), // NEAR
            Digraph('ɪ', 'ə'), // NEAR variant
            Digraph('e', 'ə'), // SQUARE
            Digraph('ɛ', 'ə'), // SQUARE variant
            Digraph('ʊ', 'ə'), // CURE
            Digraph('u', 'ə'), // CURE variant
            // German diphthongs
            Digraph('a', 'ʏ'), // German eu/äu
            Digraph('ɔ', 'ʏ'), // German eu variant
            // Additional diphthongs
            Digraph('ɛ', 'ɪ'), // FACE variant
            Digraph('ɑ', 'ɪ'), // PRICE variant (RP)
            Digraph('ɑ', 'ʊ'), // MOUTH variant (RP)
        ],
        description: "Common diphthongs (English, German, Dutch)",
    });

    // ========================================================================
    // Syllabic Consonants (English, Czech, Hindi)
    // ========================================================================

    add_class(NamedClass {
        name: "syllabic",
        aliases: &["syllabic_consonant"],
        patterns: vec![
            // Syllabic consonants (with combining vertical line below U+0329)
            Digraph('l', '\u{0329}'), // syllabic l (bottle)
            Digraph('n', '\u{0329}'), // syllabic n (button)
            Digraph('m', '\u{0329}'), // syllabic m (rhythm)
            Digraph('r', '\u{0329}'), // syllabic r (butter in rhotic accents)
            Digraph('ɹ', '\u{0329}'), // syllabic approximant r
            Digraph('ɾ', '\u{0329}'), // syllabic tap
            Digraph('ŋ', '\u{0329}'), // syllabic velar nasal
            // Combining vertical line below (syllabic marker standalone)
            Char('\u{0329}'), // ̩
        ],
        description: "Syllabic consonants (English 'bottle', Czech 'vlk')",
    });

    // ========================================================================
    // Rhotic Subtypes (Spanish, Portuguese, Hindi)
    // ========================================================================

    add_class(NamedClass {
        name: "approximant_r",
        aliases: &["rhotic_approximant"],
        patterns: vec![
            Char('ɹ'), // alveolar approximant (English r)
            Char('ɻ'), // retroflex approximant
            Char('ɰ'), // velar approximant (sometimes rhotic)
        ],
        description: "Rhotic approximants (English r)",
    });

    // Note: [:tap:] and [:trill:] already exist in the file

    // ========================================================================
    // Ejective Consonants (Georgian, Amharic, Quechua, Native American)
    // ========================================================================

    add_class(NamedClass {
        name: "ejective",
        aliases: &["glottalic_egressive"],
        patterns: vec![
            // Ejective stops
            Digraph('p', 'ʼ'),
            Digraph('t', 'ʼ'),
            Digraph('k', 'ʼ'),
            Digraph('q', 'ʼ'),
            Digraph('c', 'ʼ'),
            // Ejective affricates (using Trigraph - ignoring tie bar)
            Trigraph('t', 's', 'ʼ'), // t͡sʼ - alveolar ejective affricate
            Trigraph('t', 'ʃ', 'ʼ'), // t͡ʃʼ - postalveolar ejective affricate
            Trigraph('t', 'ɕ', 'ʼ'), // t͡ɕʼ - alveo-palatal ejective affricate
            Trigraph('t', 'ɬ', 'ʼ'), // t͡ɬʼ - lateral ejective affricate (Navajo)
            Trigraph('k', 'x', 'ʼ'), // k͡xʼ - velar ejective affricate (rare)
            // Ejective fricatives
            Digraph('s', 'ʼ'),
            Digraph('ʃ', 'ʼ'),
            Digraph('x', 'ʼ'),
            Digraph('f', 'ʼ'),
            // Labialized ejectives (Trigraph - 3 chars)
            Trigraph('k', 'ʷ', 'ʼ'), // kʷʼ - ejective labialized velar (Tlingit, Amharic, Hausa, Tigrinya)
            Trigraph('q', 'ʷ', 'ʼ'), // qʷʼ - ejective labialized uvular (Caucasian languages)
            Trigraph('p', 'ʷ', 'ʼ'), // pʷʼ - ejective labialized bilabial (rare)
            Trigraph('t', 'ʷ', 'ʼ'), // tʷʼ - ejective labialized alveolar
            // Labialized ejective affricates (Tetragraph - 4 chars)
            Tetragraph('t', 's', 'ʷ', 'ʼ'), // t͡sʷʼ - labialized alveolar ejective affricate
            Tetragraph('t', 'ʃ', 'ʷ', 'ʼ'), // t͡ʃʷʼ - labialized postalveolar ejective affricate
            // Ejective marker alone
            Char('ʼ'), // modifier letter apostrophe U+02BC
        ],
        description: "Ejective consonants (Georgian, Amharic, Quechua, Navajo, Tlingit)",
    });

    // Labialized ejective consonants (subset of ejectives with labialization)
    add_class(NamedClass {
        name: "labialized_ejective",
        aliases: &["ejective_labialized"],
        patterns: vec![
            // Labialized ejective stops (Trigraph - 3 chars)
            Trigraph('k', 'ʷ', 'ʼ'), // kʷʼ - labialized ejective velar (Tlingit, Amharic, Hausa, Tigrinya)
            Trigraph('q', 'ʷ', 'ʼ'), // qʷʼ - labialized ejective uvular (Caucasian languages)
            Trigraph('p', 'ʷ', 'ʼ'), // pʷʼ - labialized ejective bilabial (rare)
            Trigraph('t', 'ʷ', 'ʼ'), // tʷʼ - labialized ejective alveolar
            // Labialized ejective affricates (Tetragraph - 4 chars)
            Tetragraph('t', 's', 'ʷ', 'ʼ'), // t͡sʷʼ - labialized alveolar ejective affricate
            Tetragraph('t', 'ʃ', 'ʷ', 'ʼ'), // t͡ʃʷʼ - labialized postalveolar ejective affricate
            Tetragraph('t', 'ɬ', 'ʷ', 'ʼ'), // t͡ɬʷʼ - labialized lateral ejective affricate
            Tetragraph('k', 'x', 'ʷ', 'ʼ'), // k͡xʷʼ - labialized velar ejective affricate
        ],
        description: "Labialized ejective consonants (Caucasian, Tlingit, Amharic)",
    });

    // ========================================================================
    // Long Vowels (Japanese, Finnish, Arabic, Hungarian)
    // ========================================================================

    add_class(NamedClass {
        name: "long_vowel",
        aliases: &["geminate_vowel"],
        patterns: vec![
            // Basic long vowels
            Digraph('a', 'ː'),
            Digraph('e', 'ː'),
            Digraph('i', 'ː'),
            Digraph('o', 'ː'),
            Digraph('u', 'ː'),
            // IPA vowels with length
            Digraph('ə', 'ː'),
            Digraph('ɛ', 'ː'),
            Digraph('ɔ', 'ː'),
            Digraph('æ', 'ː'),
            Digraph('ɑ', 'ː'),
            Digraph('ɪ', 'ː'),
            Digraph('ʊ', 'ː'),
            Digraph('ʌ', 'ː'),
            Digraph('ɯ', 'ː'), // Japanese long u
            // Front rounded long vowels
            Digraph('y', 'ː'),
            Digraph('ø', 'ː'),
            Digraph('œ', 'ː'),
        ],
        description: "Long vowels with length marker (Japanese, Finnish, Arabic, Hungarian)",
    });

    // ========================================================================
    // Breathy Voice (Hindi, Gujarati - 4-way stop contrast)
    // ========================================================================

    add_class(NamedClass {
        name: "breathy",
        aliases: &["murmured", "breathy_voiced"],
        patterns: vec![
            // Breathy voiced stops (Hindi/Gujarati aspiration)
            Digraph('b', 'ʱ'),
            Digraph('d', 'ʱ'),
            Digraph('g', 'ʱ'),
            Digraph('ɖ', 'ʱ'), // retroflex
            Digraph('ɟ', 'ʱ'), // palatal
            Digraph('ɢ', 'ʱ'), // uvular
            // Breathy nasals
            Digraph('m', 'ʱ'),
            Digraph('n', 'ʱ'),
            Digraph('ɳ', 'ʱ'), // retroflex nasal
            // Breathy approximants
            Digraph('l', 'ʱ'),
            Digraph('r', 'ʱ'),
            // Breathy voice modifier
            Char('ʱ'),        // modifier letter small h with hook U+02B1
            Char('\u{0324}'), // combining diaeresis below (breathy marker)
        ],
        description: "Breathy voiced consonants (Hindi, Gujarati murmured stops)",
    });

    // ========================================================================
    // Creaky Voice (Vietnamese, Zapotec, Burmese)
    // ========================================================================

    add_class(NamedClass {
        name: "creaky",
        aliases: &["laryngealized", "glottalized_vowel"],
        patterns: vec![
            // Creaky vowels (combining tilde below U+0330)
            Digraph('a', '\u{0330}'),
            Digraph('e', '\u{0330}'),
            Digraph('i', '\u{0330}'),
            Digraph('o', '\u{0330}'),
            Digraph('u', '\u{0330}'),
            Digraph('ə', '\u{0330}'),
            // Creaky voice marker alone
            Char('\u{0330}'), // combining tilde below
        ],
        description: "Creaky voice/laryngealized segments (Vietnamese, Zapotec)",
    });

    // ========================================================================
    // Natural Classes (Phonological Groupings)
    // ========================================================================
    // These classes represent fundamental phonological categories used in
    // phonological rules across languages. They are essential for expressing
    // sonority hierarchy, syllable structure, and assimilation patterns.

    // Continuant: sounds produced with continuous airflow (fricatives + approximants + vowels)
    // Used in dissimilation rules, consonant clustering, and lenition
    add_class(NamedClass {
        name: "continuant",
        aliases: &[],
        patterns: vec![
            // Fricatives (continuous turbulent airflow)
            Char('f'),
            Char('v'),
            Char('s'),
            Char('z'),
            Char('h'),
            Char('F'),
            Char('V'),
            Char('S'),
            Char('Z'),
            Char('H'),
            Char('θ'),
            Char('ð'),
            Char('ʃ'),
            Char('ʒ'),
            Char('ɬ'),
            Char('ɮ'),
            Char('x'),
            Char('ɣ'),
            Char('χ'),
            Char('ʁ'),
            Char('ħ'),
            Char('ʕ'),
            Char('ç'),
            Char('ʝ'),
            Char('ɸ'),
            Char('β'),
            Char('ɦ'),
            Char('ʂ'),
            Char('ʐ'),
            Char('ɕ'),
            Char('ʑ'), // retroflex & alveopalatal
            // Approximants (continuous non-turbulent airflow)
            Char('w'),
            Char('W'),
            Char('j'),
            Char('J'),
            Char('y'),
            Char('Y'),
            Char('ɹ'),
            Char('ɻ'),
            Char('ɰ'),
            Char('ʋ'),
            Char('l'),
            Char('L'),
            Char('ɭ'),
            Char('ʎ'),
            Char('ʟ'),
            Char('r'),
            Char('R'),
            Char('ɾ'),
            Char('ɽ'), // rhotics
            // Vowels (maximal continuancy)
            Char('a'),
            Char('e'),
            Char('i'),
            Char('o'),
            Char('u'),
            Char('A'),
            Char('E'),
            Char('I'),
            Char('O'),
            Char('U'),
            Char('ə'),
            Char('ɪ'),
            Char('ʊ'),
            Char('ɛ'),
            Char('ɔ'),
            Char('æ'),
            Char('ʌ'),
            Char('ɑ'),
            Char('ɒ'),
            Char('ɜ'),
            Char('ɐ'),
            Char('y'),
            Char('ʏ'),
            Char('ø'),
            Char('œ'),
            Char('ɯ'),
            Char('ɤ'),
            // Labialized velar approximant (voiceless)
            Char('ʍ'), // U+028D - "which" vs "witch" distinction
        ],
        description:
            "Continuants: sounds with continuous airflow (fricatives + approximants + vowels)",
    });

    // Obstruent: sounds with complete or partial oral obstruction (stops + fricatives + affricates)
    // Used in voicing assimilation, final devoicing, and syllable coda constraints
    add_class(NamedClass {
        name: "obstruent",
        aliases: &[],
        patterns: vec![
            // Stops (complete obstruction)
            Char('p'),
            Char('b'),
            Char('t'),
            Char('d'),
            Char('k'),
            Char('g'),
            Char('P'),
            Char('B'),
            Char('T'),
            Char('D'),
            Char('K'),
            Char('G'),
            Char('ʔ'), // glottal stop
            Char('c'),
            Char('ɟ'), // palatal stops
            Char('q'),
            Char('ɢ'), // uvular stops
            Char('ʈ'),
            Char('ɖ'), // retroflex stops
            // Fricatives (partial obstruction with turbulence)
            Char('f'),
            Char('v'),
            Char('s'),
            Char('z'),
            Char('h'),
            Char('F'),
            Char('V'),
            Char('S'),
            Char('Z'),
            Char('H'),
            Char('θ'),
            Char('ð'),
            Char('ʃ'),
            Char('ʒ'),
            Char('ɬ'),
            Char('ɮ'),
            Char('x'),
            Char('ɣ'),
            Char('χ'),
            Char('ʁ'),
            Char('ħ'),
            Char('ʕ'),
            Char('ç'),
            Char('ʝ'),
            Char('ɸ'),
            Char('β'),
            Char('ɦ'),
            Char('ʂ'),
            Char('ʐ'),
            Char('ɕ'),
            Char('ʑ'), // retroflex & alveopalatal
            // Affricates (stop + fricative)
            Char('ʧ'),
            Char('ʤ'),
            Char('ʦ'),
            Char('ʣ'),
            Char('ʨ'),
            Char('ʥ'),
        ],
        description: "Obstruents: sounds with oral obstruction (stops + fricatives + affricates)",
    });

    // Sonorant: sounds produced with continuous resonance (nasals + approximants + vowels)
    // Used in sonority sequencing, syllable nuclei, and nasal assimilation
    add_class(NamedClass {
        name: "sonorant",
        aliases: &["resonant"],
        patterns: vec![
            // Nasals (resonant with nasal airflow)
            Char('m'),
            Char('n'),
            Char('M'),
            Char('N'),
            Char('ŋ'),
            Char('ɲ'),
            Char('ɳ'),
            Char('ɴ'),
            Char('ɱ'),
            // Approximants/liquids (oral resonance)
            Char('w'),
            Char('W'),
            Char('j'),
            Char('J'),
            Char('y'),
            Char('Y'),
            Char('l'),
            Char('L'),
            Char('r'),
            Char('R'),
            Char('ɹ'),
            Char('ɻ'),
            Char('ɾ'),
            Char('ɽ'),
            Char('ɰ'),
            Char('ʋ'),
            Char('ɭ'),
            Char('ʎ'),
            Char('ʟ'),
            Char('ʀ'),
            Char('ʙ'), // trills
            // Vowels (maximal sonority)
            Char('a'),
            Char('e'),
            Char('i'),
            Char('o'),
            Char('u'),
            Char('A'),
            Char('E'),
            Char('I'),
            Char('O'),
            Char('U'),
            Char('ə'),
            Char('ɪ'),
            Char('ʊ'),
            Char('ɛ'),
            Char('ɔ'),
            Char('æ'),
            Char('ʌ'),
            Char('ɑ'),
            Char('ɒ'),
            Char('ɜ'),
            Char('ɐ'),
            Char('y'),
            Char('ʏ'),
            Char('ø'),
            Char('œ'),
            Char('ɯ'),
            Char('ɤ'),
            Char('ɨ'),
            Char('ʉ'),
            Char('ɘ'),
            Char('ɵ'),
            Char('ɜ'),
            Char('ɞ'),
        ],
        description: "Sonorants: sounds with continuous resonance (nasals + approximants + vowels)",
    });

    // ========================================================================
    // Fricative Voicing Subclasses
    // ========================================================================
    // Essential for voicing assimilation rules across languages

    // Voiceless fricatives
    add_class(NamedClass {
        name: "voiceless_fricative",
        aliases: &[],
        patterns: vec![
            Char('f'),
            Char('F'),
            Char('s'),
            Char('S'),
            Char('h'),
            Char('H'),
            Char('θ'), // voiceless dental
            Char('ʃ'), // voiceless postalveolar
            Char('ɬ'), // voiceless lateral
            Char('x'), // voiceless velar
            Char('χ'), // voiceless uvular
            Char('ħ'), // voiceless pharyngeal
            Char('ç'), // voiceless palatal
            Char('ɸ'), // voiceless bilabial
            Char('ʂ'), // voiceless retroflex
            Char('ɕ'), // voiceless alveopalatal
            Char('ʍ'), // voiceless labialized velar approximant (often classified with fricatives)
            Digraph('s', 'h'),
            Digraph('t', 'h'),
        ],
        description: "Voiceless fricatives (f, s, θ, ʃ, x, h, etc.)",
    });

    // Voiced fricatives
    add_class(NamedClass {
        name: "voiced_fricative",
        aliases: &[],
        patterns: vec![
            Char('v'),
            Char('V'),
            Char('z'),
            Char('Z'),
            Char('ð'), // voiced dental
            Char('ʒ'), // voiced postalveolar
            Char('ɮ'), // voiced lateral
            Char('ɣ'), // voiced velar
            Char('ʁ'), // voiced uvular
            Char('ʕ'), // voiced pharyngeal
            Char('ʝ'), // voiced palatal
            Char('β'), // voiced bilabial
            Char('ɦ'), // voiced glottal
            Char('ʐ'), // voiced retroflex
            Char('ʑ'), // voiced alveopalatal
            Digraph('z', 'h'),
        ],
        description: "Voiced fricatives (v, z, ð, ʒ, ɣ, etc.)",
    });

    // Sibilant fricatives (high-frequency turbulence)
    // Used in sibilant harmony rules (e.g., Turkish, Navajo)
    add_class(NamedClass {
        name: "sibilant",
        aliases: &["sibilant_fricative"],
        patterns: vec![
            // Voiceless sibilants
            Char('s'),
            Char('S'),
            Char('ʃ'), // postalveolar
            Char('ʂ'), // retroflex
            Char('ɕ'), // alveopalatal
            // Voiced sibilants
            Char('z'),
            Char('Z'),
            Char('ʒ'), // postalveolar
            Char('ʐ'), // retroflex
            Char('ʑ'), // alveopalatal
            // Sibilant affricates
            Char('ʦ'),
            Char('ʣ'), // alveolar
            Char('ʧ'),
            Char('ʤ'), // postalveolar
            Char('ʨ'),
            Char('ʥ'), // alveopalatal
            Digraph('s', 'h'),
            Digraph('z', 'h'),
            Digraph('t', 's'),
            Digraph('d', 'z'),
        ],
        description: "Sibilant fricatives/affricates (s, z, ʃ, ʒ, ɕ, ʑ, etc.)",
    });

    // ========================================================================
    // Vowel Quality Combination Classes
    // ========================================================================

    // High front rounded vowels (German ü, French u)
    add_class(NamedClass {
        name: "high_front_rounded",
        aliases: &[],
        patterns: vec![
            Char('y'), // close front rounded (IPA for German ü)
            Char('ʏ'), // near-close front rounded
        ],
        description: "High front rounded vowels (German ü, French u)",
    });

    // High back unrounded vowels (Japanese u, Turkish ı)
    add_class(NamedClass {
        name: "high_back_unrounded",
        aliases: &[],
        patterns: vec![
            Char('ɯ'), // close back unrounded (Japanese u)
            Char('ɨ'), // close central unrounded (sometimes classified here)
        ],
        description: "High back unrounded vowels (Japanese u, Turkish ı)",
    });

    // Mid central vowels (schwa family)
    add_class(NamedClass {
        name: "mid_central",
        aliases: &[],
        patterns: vec![
            Char('ə'), // schwa (mid central)
            Char('ɘ'), // close-mid central unrounded
            Char('ɵ'), // close-mid central rounded
            Char('ɜ'), // open-mid central unrounded
            Char('ɞ'), // open-mid central rounded
        ],
        description: "Mid central vowels (schwa and related)",
    });

    // Low back vowels
    add_class(NamedClass {
        name: "low_back",
        aliases: &[],
        patterns: vec![
            Char('ɑ'), // open back unrounded (father)
            Char('ɒ'), // open back rounded (British lot)
        ],
        description: "Low back vowels (father, lot)",
    });

    // Low front vowels
    add_class(NamedClass {
        name: "low_front",
        aliases: &[],
        patterns: vec![
            Char('a'),
            Char('A'), // open front (varies by language)
            Char('æ'), // near-open front unrounded (cat)
        ],
        description: "Low front vowels (cat, trap)",
    });

    // ========================================================================
    // Voiceless Labialized Velar Approximant
    // ========================================================================
    // Add ʍ to the glide/approximant classes that were missing it

    add_class(NamedClass {
        name: "voiceless_glide",
        aliases: &["voiceless_semivowel"],
        patterns: vec![
            Char('ʍ'), // voiceless labialized velar approximant (U+028D)
                       // Note: ʍ represents "wh" in "which" vs "witch" distinction
        ],
        description: "Voiceless glides/semivowels (wh as in 'which')",
    });

    m
});
