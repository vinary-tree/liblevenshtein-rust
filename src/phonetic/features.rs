//! Phonetic feature definitions for the `(?f)` regex flag.
//!
//! This module provides phonetic feature tables for feature-based matching.
//! When the `(?f)` flag is set, characters are matched based on phonetic
//! similarity rather than exact equality.
//!
//! # Phonetic Features
//!
//! Characters are classified by their phonetic properties:
//!
//! | Feature | Description | Examples |
//! |---------|-------------|----------|
//! | Voiced | Vocal cords vibrate | b, d, g, v, z |
//! | Voiceless | No vocal cord vibration | p, t, k, f, s |
//! | Nasal | Air flows through nose | m, n, ŋ |
//! | Stop | Complete airflow closure | p, b, t, d, k, g |
//! | Fricative | Turbulent airflow | f, v, s, z, h |
//! | Approximant | Smooth airflow | l, r, w, y |
//! | Vowel | Open vocal tract | a, e, i, o, u |
//! | Consonant | Constricted vocal tract | all non-vowels |
//!
//! # Usage
//!
//! ```ignore
//! use liblevenshtein::phonetic::features::{PhoneticFeature, get_features, chars_with_features};
//!
//! // Get features for a character
//! let features = get_features('b');
//! assert!(features.contains(&PhoneticFeature::Voiced));
//! assert!(features.contains(&PhoneticFeature::Stop));
//!
//! // Get all characters with specific features
//! let voiced_stops = chars_with_features(&[PhoneticFeature::Voiced, PhoneticFeature::Stop]);
//! assert!(voiced_stops.contains(&'b'));
//! assert!(voiced_stops.contains(&'d'));
//! assert!(voiced_stops.contains(&'g'));
//! ```

use rustc_hash::FxHashSet;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Phonetic features for classifying sounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhoneticFeature {
    // Voicing
    /// Vocal cords vibrate during production (b, d, g, v, z, etc.)
    Voiced,
    /// No vocal cord vibration (p, t, k, f, s, etc.)
    Voiceless,

    // Manner of articulation
    /// Complete closure of the vocal tract (p, b, t, d, k, g)
    Stop,
    /// Turbulent airflow through narrow constriction (f, v, s, z, h)
    Fricative,
    /// Combined stop + fricative (ch, j)
    Affricate,
    /// Air flows through the nose (m, n, ŋ)
    Nasal,
    /// Smooth airflow approximation (l, r, w, y)
    Approximant,
    /// Lateral airflow (l)
    Lateral,
    /// Trill or tap (r in some languages)
    Rhotic,

    // Place of articulation
    /// Lips together (p, b, m)
    Bilabial,
    /// Lower lip to upper teeth (f, v)
    Labiodental,
    /// Tongue between teeth (th)
    Dental,
    /// Tongue tip to alveolar ridge (t, d, n, s, z, l)
    Alveolar,
    /// Tongue blade to post-alveolar region (sh, ch, j)
    PostAlveolar,
    /// Tongue body to palate (y)
    Palatal,
    /// Tongue back to velum (k, g, ŋ)
    Velar,
    /// Glottal constriction (h)
    Glottal,

    // Vowel/Consonant
    /// Open vocal tract (a, e, i, o, u)
    Vowel,
    /// Constricted vocal tract
    Consonant,

    // Vowel height
    /// High/close vowel (i, u)
    High,
    /// Mid vowel (e, o)
    Mid,
    /// Low/open vowel (a)
    Low,

    // Vowel backness
    /// Front vowel (i, e)
    Front,
    /// Central vowel
    Central,
    /// Back vowel (u, o)
    Back,

    // Lip rounding
    /// Rounded lips (o, u)
    Rounded,
    /// Unrounded lips (a, e, i)
    Unrounded,

    // Sibilance
    /// High-frequency fricative (s, z, sh, zh)
    Sibilant,

    // Secondary articulation
    /// Aspirated consonant (Korean, Hindi, Thai, Mandarin - pʰ, tʰ, kʰ)
    Aspirated,
    /// Tense/fortis consonant (Korean ssang consonants)
    Tense,
    /// Pharyngealized/emphatic consonant (Arabic emphatics - sˤ, dˤ, tˤ)
    Pharyngealized,
    /// Labialized consonant (with lip rounding - kʷ, gʷ)
    Labialized,
    /// Velarized consonant (dark L - ɫ)
    Velarized,

    // Extended place of articulation (IPA completeness)
    /// Retroflex articulation - tongue tip curled back (Hindi ट, ड, ण, ळ)
    Retroflex,
    /// Uvular articulation - back of tongue to uvula (Arabic ق, French R)
    Uvular,
    /// Pharyngeal articulation - root of tongue to pharynx (Arabic ح, ع)
    Pharyngeal,
    /// Epiglottal articulation - epiglottis constriction (rare)
    Epiglottal,

    // Additional manner features
    /// Tap/flap consonant - single brief contact (Spanish ɾ)
    Tap,
    /// Trill consonant - multiple vibrations (Spanish rr, French R)
    Trill,
    /// Ejective consonant - glottalic egressive airstream (Georgian, Amharic)
    Ejective,
    /// Implosive consonant - glottalic ingressive airstream (Swahili, Sindhi)
    Implosive,
    /// Click consonant - velaric ingressive airstream (Zulu, Xhosa)
    Click,
}

/// Feature table for common phonetic characters.
static FEATURE_TABLE: OnceLock<HashMap<char, FxHashSet<PhoneticFeature>>> = OnceLock::new();

fn feature_table() -> &'static HashMap<char, FxHashSet<PhoneticFeature>> {
    FEATURE_TABLE.get_or_init(|| {
        use PhoneticFeature::*;

        let mut table: HashMap<char, FxHashSet<PhoneticFeature>> = HashMap::new();

        // Vowels
        table.insert('a', [Vowel, Low, Central, Unrounded].into_iter().collect());
        table.insert('e', [Vowel, Mid, Front, Unrounded].into_iter().collect());
        table.insert('i', [Vowel, High, Front, Unrounded].into_iter().collect());
        table.insert('o', [Vowel, Mid, Back, Rounded].into_iter().collect());
        table.insert('u', [Vowel, High, Back, Rounded].into_iter().collect());

        // Uppercase vowels
        table.insert('A', [Vowel, Low, Central, Unrounded].into_iter().collect());
        table.insert('E', [Vowel, Mid, Front, Unrounded].into_iter().collect());
        table.insert('I', [Vowel, High, Front, Unrounded].into_iter().collect());
        table.insert('O', [Vowel, Mid, Back, Rounded].into_iter().collect());
        table.insert('U', [Vowel, High, Back, Rounded].into_iter().collect());

        // Accented vowels (same features as base)
        for (accented, base) in [
            ('à', 'a'),
            ('á', 'a'),
            ('â', 'a'),
            ('ã', 'a'),
            ('ä', 'a'),
            ('å', 'a'),
            ('è', 'e'),
            ('é', 'e'),
            ('ê', 'e'),
            ('ë', 'e'),
            ('ì', 'i'),
            ('í', 'i'),
            ('î', 'i'),
            ('ï', 'i'),
            ('ò', 'o'),
            ('ó', 'o'),
            ('ô', 'o'),
            ('õ', 'o'),
            ('ö', 'o'),
            ('ù', 'u'),
            ('ú', 'u'),
            ('û', 'u'),
            ('ü', 'u'),
            ('À', 'A'),
            ('Á', 'A'),
            ('Â', 'A'),
            ('Ã', 'A'),
            ('Ä', 'A'),
            ('Å', 'A'),
            ('È', 'E'),
            ('É', 'E'),
            ('Ê', 'E'),
            ('Ë', 'E'),
            ('Ì', 'I'),
            ('Í', 'I'),
            ('Î', 'I'),
            ('Ï', 'I'),
            ('Ò', 'O'),
            ('Ó', 'O'),
            ('Ô', 'O'),
            ('Õ', 'O'),
            ('Ö', 'O'),
            ('Ù', 'U'),
            ('Ú', 'U'),
            ('Û', 'U'),
            ('Ü', 'U'),
        ] {
            if let Some(features) = table.get(&base) {
                table.insert(accented, features.clone());
            }
        }

        // Bilabial stops
        table.insert(
            'p',
            [Consonant, Stop, Bilabial, Voiceless].into_iter().collect(),
        );
        table.insert(
            'b',
            [Consonant, Stop, Bilabial, Voiced].into_iter().collect(),
        );
        table.insert(
            'P',
            [Consonant, Stop, Bilabial, Voiceless].into_iter().collect(),
        );
        table.insert(
            'B',
            [Consonant, Stop, Bilabial, Voiced].into_iter().collect(),
        );

        // Alveolar stops
        table.insert(
            't',
            [Consonant, Stop, Alveolar, Voiceless].into_iter().collect(),
        );
        table.insert(
            'd',
            [Consonant, Stop, Alveolar, Voiced].into_iter().collect(),
        );
        table.insert(
            'T',
            [Consonant, Stop, Alveolar, Voiceless].into_iter().collect(),
        );
        table.insert(
            'D',
            [Consonant, Stop, Alveolar, Voiced].into_iter().collect(),
        );

        // Velar stops
        table.insert(
            'k',
            [Consonant, Stop, Velar, Voiceless].into_iter().collect(),
        );
        table.insert('g', [Consonant, Stop, Velar, Voiced].into_iter().collect());
        table.insert(
            'K',
            [Consonant, Stop, Velar, Voiceless].into_iter().collect(),
        );
        table.insert('G', [Consonant, Stop, Velar, Voiced].into_iter().collect());
        table.insert(
            'c',
            [Consonant, Stop, Velar, Voiceless].into_iter().collect(),
        ); // hard c
        table.insert(
            'C',
            [Consonant, Stop, Velar, Voiceless].into_iter().collect(),
        );
        table.insert(
            'q',
            [Consonant, Stop, Velar, Voiceless].into_iter().collect(),
        );
        table.insert(
            'Q',
            [Consonant, Stop, Velar, Voiceless].into_iter().collect(),
        );

        // Labiodental fricatives
        table.insert(
            'f',
            [Consonant, Fricative, Labiodental, Voiceless]
                .into_iter()
                .collect(),
        );
        table.insert(
            'v',
            [Consonant, Fricative, Labiodental, Voiced]
                .into_iter()
                .collect(),
        );
        table.insert(
            'F',
            [Consonant, Fricative, Labiodental, Voiceless]
                .into_iter()
                .collect(),
        );
        table.insert(
            'V',
            [Consonant, Fricative, Labiodental, Voiced]
                .into_iter()
                .collect(),
        );

        // Alveolar fricatives (sibilants)
        table.insert(
            's',
            [Consonant, Fricative, Alveolar, Voiceless, Sibilant]
                .into_iter()
                .collect(),
        );
        table.insert(
            'z',
            [Consonant, Fricative, Alveolar, Voiced, Sibilant]
                .into_iter()
                .collect(),
        );
        table.insert(
            'S',
            [Consonant, Fricative, Alveolar, Voiceless, Sibilant]
                .into_iter()
                .collect(),
        );
        table.insert(
            'Z',
            [Consonant, Fricative, Alveolar, Voiced, Sibilant]
                .into_iter()
                .collect(),
        );

        // Glottal fricative
        table.insert(
            'h',
            [Consonant, Fricative, Glottal, Voiceless]
                .into_iter()
                .collect(),
        );
        table.insert(
            'H',
            [Consonant, Fricative, Glottal, Voiceless]
                .into_iter()
                .collect(),
        );

        // Nasals
        table.insert(
            'm',
            [Consonant, Nasal, Bilabial, Voiced].into_iter().collect(),
        );
        table.insert(
            'n',
            [Consonant, Nasal, Alveolar, Voiced].into_iter().collect(),
        );
        table.insert(
            'M',
            [Consonant, Nasal, Bilabial, Voiced].into_iter().collect(),
        );
        table.insert(
            'N',
            [Consonant, Nasal, Alveolar, Voiced].into_iter().collect(),
        );
        table.insert(
            'ñ',
            [Consonant, Nasal, Palatal, Voiced].into_iter().collect(),
        );
        table.insert(
            'Ñ',
            [Consonant, Nasal, Palatal, Voiced].into_iter().collect(),
        );
        table.insert('ŋ', [Consonant, Nasal, Velar, Voiced].into_iter().collect()); // ng

        // Approximants
        table.insert(
            'l',
            [Consonant, Approximant, Lateral, Alveolar, Voiced]
                .into_iter()
                .collect(),
        );
        table.insert(
            'r',
            [Consonant, Approximant, Rhotic, Alveolar, Voiced]
                .into_iter()
                .collect(),
        );
        table.insert(
            'w',
            [Consonant, Approximant, Bilabial, Velar, Voiced]
                .into_iter()
                .collect(),
        );
        table.insert(
            'y',
            [Consonant, Approximant, Palatal, Voiced]
                .into_iter()
                .collect(),
        );
        table.insert(
            'L',
            [Consonant, Approximant, Lateral, Alveolar, Voiced]
                .into_iter()
                .collect(),
        );
        table.insert(
            'R',
            [Consonant, Approximant, Rhotic, Alveolar, Voiced]
                .into_iter()
                .collect(),
        );
        table.insert(
            'W',
            [Consonant, Approximant, Bilabial, Velar, Voiced]
                .into_iter()
                .collect(),
        );
        table.insert(
            'Y',
            [Consonant, Approximant, Palatal, Voiced]
                .into_iter()
                .collect(),
        );

        // Special consonants
        table.insert(
            'x',
            [Consonant, Fricative, Velar, Voiceless]
                .into_iter()
                .collect(),
        );
        table.insert(
            'X',
            [Consonant, Fricative, Velar, Voiceless]
                .into_iter()
                .collect(),
        );
        table.insert(
            'j',
            [Consonant, Affricate, PostAlveolar, Voiced]
                .into_iter()
                .collect(),
        );
        table.insert(
            'J',
            [Consonant, Affricate, PostAlveolar, Voiced]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ç',
            [Consonant, Fricative, Alveolar, Voiceless, Sibilant]
                .into_iter()
                .collect(),
        ); // cedilla c like s

        // ========================================================================
        // Secondary Articulation Features
        // ========================================================================

        // Velarized L (dark L)
        table.insert(
            'ɫ',
            [Consonant, Approximant, Lateral, Alveolar, Voiced, Velarized]
                .into_iter()
                .collect(),
        );

        // Aspiration marker (modifier letter)
        table.insert('\u{02B0}', [Aspirated].into_iter().collect()); // ʰ

        // Pharyngealization marker
        table.insert('\u{02E4}', [Pharyngealized].into_iter().collect()); // ˤ

        // Labialization marker
        table.insert('\u{02B7}', [Labialized].into_iter().collect()); // ʷ

        // Velarization marker
        table.insert('\u{02E0}', [Velarized].into_iter().collect()); // ˠ

        // Tenseness marker (Korean)
        table.insert('\u{0348}', [Tense].into_iter().collect()); // ͈

        // ========================================================================
        // Retroflex Consonants (Hindi, Tamil, Telugu, Sanskrit)
        // ========================================================================

        // Retroflex stops
        table.insert(
            'ʈ',
            [Consonant, Stop, Retroflex, Voiceless]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ɖ',
            [Consonant, Stop, Retroflex, Voiced].into_iter().collect(),
        );

        // Retroflex nasal
        table.insert(
            'ɳ',
            [Consonant, Nasal, Retroflex, Voiced].into_iter().collect(),
        );

        // Retroflex tap/flap
        table.insert(
            'ɽ',
            [Consonant, Tap, Retroflex, Voiced].into_iter().collect(),
        );

        // Retroflex lateral
        table.insert(
            'ɭ',
            [Consonant, Approximant, Lateral, Retroflex, Voiced]
                .into_iter()
                .collect(),
        );

        // Retroflex fricatives
        table.insert(
            'ʂ',
            [Consonant, Fricative, Retroflex, Voiceless, Sibilant]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ʐ',
            [Consonant, Fricative, Retroflex, Voiced, Sibilant]
                .into_iter()
                .collect(),
        );

        // Retroflex approximant
        table.insert(
            'ɻ',
            [Consonant, Approximant, Retroflex, Voiced]
                .into_iter()
                .collect(),
        );

        // ========================================================================
        // Uvular Consonants (Arabic, French, German)
        // ========================================================================

        // Uvular stops
        table.insert(
            'q',
            [Consonant, Stop, Uvular, Voiceless].into_iter().collect(),
        );
        table.insert('ɢ', [Consonant, Stop, Uvular, Voiced].into_iter().collect());

        // Uvular nasal
        table.insert(
            'ɴ',
            [Consonant, Nasal, Uvular, Voiced].into_iter().collect(),
        );

        // Uvular fricatives
        table.insert(
            'χ',
            [Consonant, Fricative, Uvular, Voiceless]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ʁ',
            [Consonant, Fricative, Uvular, Voiced].into_iter().collect(),
        );

        // Uvular trill
        table.insert(
            'ʀ',
            [Consonant, Trill, Uvular, Voiced].into_iter().collect(),
        );

        // ========================================================================
        // Pharyngeal Consonants (Arabic ح, ع)
        // ========================================================================

        // Pharyngeal fricatives
        table.insert(
            'ħ',
            [Consonant, Fricative, Pharyngeal, Voiceless]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ʕ',
            [Consonant, Fricative, Pharyngeal, Voiced]
                .into_iter()
                .collect(),
        );

        // ========================================================================
        // Epiglottal Consonants (rare, but IPA complete)
        // ========================================================================

        // Epiglottal fricatives
        table.insert(
            'ʜ',
            [Consonant, Fricative, Epiglottal, Voiceless]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ʢ',
            [Consonant, Fricative, Epiglottal, Voiced]
                .into_iter()
                .collect(),
        );

        // Epiglottal stop
        table.insert(
            'ʡ',
            [Consonant, Stop, Epiglottal, Voiceless]
                .into_iter()
                .collect(),
        );

        // ========================================================================
        // Additional IPA Consonants (Tap, Trill, Ejective, Implosive, Click)
        // ========================================================================

        // Alveolar tap
        table.insert(
            'ɾ',
            [Consonant, Tap, Alveolar, Voiced].into_iter().collect(),
        );

        // Bilabial trill
        table.insert(
            'ʙ',
            [Consonant, Trill, Bilabial, Voiced].into_iter().collect(),
        );

        // Alveolar trill (Spanish rr)
        // Note: 'r' is already in table as approximant; some dialects use trill
        // table.insert('r', [Consonant, Trill, Alveolar, Voiced].into_iter().collect());

        // Ejective marker
        table.insert('ʼ', [Ejective].into_iter().collect());

        // Implosives
        table.insert(
            'ɓ',
            [Consonant, Stop, Bilabial, Voiced, Implosive]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ɗ',
            [Consonant, Stop, Alveolar, Voiced, Implosive]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ɠ',
            [Consonant, Stop, Velar, Voiced, Implosive]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ʄ',
            [Consonant, Stop, Palatal, Voiced, Implosive]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ʛ',
            [Consonant, Stop, Uvular, Voiced, Implosive]
                .into_iter()
                .collect(),
        );

        // Clicks (bilabial, dental, alveolar, palatal, lateral)
        table.insert('ʘ', [Consonant, Click, Bilabial].into_iter().collect());
        table.insert('ǀ', [Consonant, Click, Dental].into_iter().collect());
        table.insert('ǃ', [Consonant, Click, Alveolar].into_iter().collect());
        table.insert('ǂ', [Consonant, Click, Palatal].into_iter().collect());
        table.insert('ǁ', [Consonant, Click, Lateral].into_iter().collect());

        // ========================================================================
        // Additional IPA Vowels (completeness)
        // ========================================================================

        // Schwa and near-open central
        table.insert('ə', [Vowel, Mid, Central, Unrounded].into_iter().collect());
        table.insert('ɐ', [Vowel, Low, Central, Unrounded].into_iter().collect());

        // Near-close vowels
        table.insert('ɪ', [Vowel, High, Front, Unrounded].into_iter().collect());
        table.insert('ʏ', [Vowel, High, Front, Rounded].into_iter().collect());
        table.insert('ʊ', [Vowel, High, Back, Rounded].into_iter().collect());

        // Open-mid vowels
        table.insert('ɛ', [Vowel, Mid, Front, Unrounded].into_iter().collect());
        table.insert('œ', [Vowel, Mid, Front, Rounded].into_iter().collect());
        table.insert('ɔ', [Vowel, Mid, Back, Rounded].into_iter().collect());
        table.insert('ʌ', [Vowel, Mid, Back, Unrounded].into_iter().collect());

        // Open vowels
        table.insert('æ', [Vowel, Low, Front, Unrounded].into_iter().collect());
        table.insert('ɑ', [Vowel, Low, Back, Unrounded].into_iter().collect());
        table.insert('ɒ', [Vowel, Low, Back, Rounded].into_iter().collect());

        // Close-mid vowels
        table.insert('ø', [Vowel, Mid, Front, Rounded].into_iter().collect());
        table.insert('ɤ', [Vowel, Mid, Back, Unrounded].into_iter().collect());

        // Close vowels
        table.insert('y', [Vowel, High, Front, Rounded].into_iter().collect());
        table.insert('ɯ', [Vowel, High, Back, Unrounded].into_iter().collect());
        table.insert('ɨ', [Vowel, High, Central, Unrounded].into_iter().collect());
        table.insert('ʉ', [Vowel, High, Central, Rounded].into_iter().collect());

        // Mid central vowels
        table.insert('ɘ', [Vowel, Mid, Central, Unrounded].into_iter().collect());
        table.insert('ɵ', [Vowel, Mid, Central, Rounded].into_iter().collect());

        // Open-mid central
        table.insert('ɜ', [Vowel, Mid, Central, Unrounded].into_iter().collect());
        table.insert('ɞ', [Vowel, Mid, Central, Rounded].into_iter().collect());

        // Near-open front
        table.insert('ɶ', [Vowel, Low, Front, Rounded].into_iter().collect());

        // ========================================================================
        // Palatal Consonants (IPA completeness)
        // ========================================================================

        // Palatal stops
        table.insert(
            'c',
            [Consonant, Stop, Palatal, Voiceless].into_iter().collect(),
        );
        table.insert(
            'ɟ',
            [Consonant, Stop, Palatal, Voiced].into_iter().collect(),
        );

        // Palatal nasal
        table.insert(
            'ɲ',
            [Consonant, Nasal, Palatal, Voiced].into_iter().collect(),
        );

        // Palatal fricatives
        table.insert(
            'ç',
            [Consonant, Fricative, Palatal, Voiceless]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ʝ',
            [Consonant, Fricative, Palatal, Voiced]
                .into_iter()
                .collect(),
        );

        // Palatal lateral
        table.insert(
            'ʎ',
            [Consonant, Approximant, Lateral, Palatal, Voiced]
                .into_iter()
                .collect(),
        );

        // Palatal approximant
        table.insert(
            'j',
            [Consonant, Approximant, Palatal, Voiced]
                .into_iter()
                .collect(),
        );

        // ========================================================================
        // PostAlveolar/Alveopalatal Fricatives
        // ========================================================================

        // PostAlveolar fricatives (English sh, zh)
        table.insert(
            'ʃ',
            [Consonant, Fricative, PostAlveolar, Voiceless, Sibilant]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ʒ',
            [Consonant, Fricative, PostAlveolar, Voiced, Sibilant]
                .into_iter()
                .collect(),
        );

        // Alveopalatal fricatives (Mandarin x, Japanese sh)
        table.insert(
            'ɕ',
            [Consonant, Fricative, PostAlveolar, Voiceless, Sibilant]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ʑ',
            [Consonant, Fricative, PostAlveolar, Voiced, Sibilant]
                .into_iter()
                .collect(),
        );

        // ========================================================================
        // Lateral Fricatives (Welsh, Zulu)
        // ========================================================================

        table.insert(
            'ɬ',
            [Consonant, Fricative, Lateral, Alveolar, Voiceless]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ɮ',
            [Consonant, Fricative, Lateral, Alveolar, Voiced]
                .into_iter()
                .collect(),
        );

        // ========================================================================
        // Glottal Stop
        // ========================================================================

        table.insert(
            'ʔ',
            [Consonant, Stop, Glottal, Voiceless].into_iter().collect(),
        );

        // ========================================================================
        // Dental Fricatives (English th)
        // ========================================================================

        table.insert(
            'θ',
            [Consonant, Fricative, Dental, Voiceless]
                .into_iter()
                .collect(),
        );
        table.insert(
            'ð',
            [Consonant, Fricative, Dental, Voiced].into_iter().collect(),
        );

        // ========================================================================
        // Labiodental Approximant
        // ========================================================================

        table.insert(
            'ʋ',
            [Consonant, Approximant, Labiodental, Voiced]
                .into_iter()
                .collect(),
        );

        // ========================================================================
        // Bilabial Fricatives
        // ========================================================================

        table.insert(
            'ɸ',
            [Consonant, Fricative, Bilabial, Voiceless]
                .into_iter()
                .collect(),
        );
        table.insert(
            'β',
            [Consonant, Fricative, Bilabial, Voiced]
                .into_iter()
                .collect(),
        );

        table
    })
}

/// Reverse index from features to characters.
static FEATURE_INDEX: OnceLock<HashMap<PhoneticFeature, FxHashSet<char>>> = OnceLock::new();

fn feature_index() -> &'static HashMap<PhoneticFeature, FxHashSet<char>> {
    FEATURE_INDEX.get_or_init(|| {
        let mut index: HashMap<PhoneticFeature, FxHashSet<char>> = HashMap::new();

        for (c, features) in feature_table().iter() {
            for feature in features {
                index.entry(*feature).or_default().insert(*c);
            }
        }

        index
    })
}

/// Get phonetic features for a character.
///
/// Returns an empty set for unknown characters.
pub fn get_features(c: char) -> FxHashSet<PhoneticFeature> {
    feature_table().get(&c).cloned().unwrap_or_default()
}

/// Get all characters that have ALL of the specified features.
///
/// Returns characters that share every feature in the list.
pub fn chars_with_features(features: &[PhoneticFeature]) -> FxHashSet<char> {
    if features.is_empty() {
        return FxHashSet::default();
    }

    // Start with chars having the first feature
    let mut result: FxHashSet<char> = feature_index()
        .get(&features[0])
        .cloned()
        .unwrap_or_default();

    // Intersect with chars having each subsequent feature
    for feature in &features[1..] {
        if let Some(chars) = feature_index().get(feature) {
            result.retain(|c| chars.contains(c));
        } else {
            return FxHashSet::default();
        }
    }

    result
}

/// Get all characters that have ANY of the specified features.
///
/// Returns the union of characters with each feature.
pub fn chars_with_any_feature(features: &[PhoneticFeature]) -> FxHashSet<char> {
    let mut result = FxHashSet::default();

    for feature in features {
        if let Some(chars) = feature_index().get(feature) {
            result.extend(chars.iter());
        }
    }

    result
}

/// Check if two characters are phonetically similar.
///
/// Characters are similar if they share at least one phonetic feature.
pub fn are_similar(c1: char, c2: char) -> bool {
    let f1 = get_features(c1);
    let f2 = get_features(c2);
    f1.intersection(&f2).next().is_some()
}

/// Get phonetically similar characters to the given character.
///
/// Returns all characters that share at least one feature with the input.
pub fn get_similar_chars(c: char) -> FxHashSet<char> {
    let features = get_features(c);
    let mut result = FxHashSet::default();

    for feature in features {
        if let Some(chars) = feature_index().get(&feature) {
            result.extend(chars.iter());
        }
    }

    // Remove the original character
    result.remove(&c);
    result
}

/// Get characters that share the same voicing as the input.
///
/// Useful for voicing-sensitive matching (b~p, d~t, g~k).
pub fn get_voicing_pair(c: char) -> Option<char> {
    let features = get_features(c);

    if features.is_empty() {
        return None;
    }

    // Find place and manner features
    let place = features.iter().find(|f| {
        matches!(
            f,
            PhoneticFeature::Bilabial
                | PhoneticFeature::Labiodental
                | PhoneticFeature::Dental
                | PhoneticFeature::Alveolar
                | PhoneticFeature::PostAlveolar
                | PhoneticFeature::Palatal
                | PhoneticFeature::Velar
                | PhoneticFeature::Glottal
                | PhoneticFeature::Retroflex
                | PhoneticFeature::Uvular
                | PhoneticFeature::Pharyngeal
                | PhoneticFeature::Epiglottal
        )
    })?;

    let manner = features.iter().find(|f| {
        matches!(
            f,
            PhoneticFeature::Stop
                | PhoneticFeature::Fricative
                | PhoneticFeature::Affricate
                | PhoneticFeature::Nasal
                | PhoneticFeature::Approximant
                | PhoneticFeature::Tap
                | PhoneticFeature::Trill
        )
    })?;

    // Find opposite voicing
    let target_voicing = if features.contains(&PhoneticFeature::Voiced) {
        PhoneticFeature::Voiceless
    } else if features.contains(&PhoneticFeature::Voiceless) {
        PhoneticFeature::Voiced
    } else {
        return None;
    };

    // Collect additional distinguishing features (Sibilant, Lateral, etc.)
    // These help distinguish between consonants with the same place/manner
    let additional_features: Vec<PhoneticFeature> = features
        .iter()
        .filter(|f| matches!(f, PhoneticFeature::Sibilant | PhoneticFeature::Lateral))
        .copied()
        .collect();

    // Find matching character with opposite voicing
    // Prefer pulmonic consonants over implosives/clicks/ejectives
    let candidates = chars_with_features(&[*place, *manner, target_voicing]);
    let is_pulmonic = |ch: &char| {
        let f = get_features(*ch);
        !f.contains(&PhoneticFeature::Implosive) && !f.contains(&PhoneticFeature::Click)
    };

    // Score candidates by how many additional features they share with input
    let score_candidate = |ch: &char| -> i32 {
        let ch_features = get_features(*ch);
        let mut score: i32 = 0;
        // Positive score for matching additional features
        for f in &additional_features {
            if ch_features.contains(f) {
                score += 10;
            }
        }
        // Negative score for having features the input doesn't have
        if ch_features.contains(&PhoneticFeature::Lateral)
            && !features.contains(&PhoneticFeature::Lateral)
        {
            score -= 5;
        }
        if ch_features.contains(&PhoneticFeature::Sibilant)
            && !features.contains(&PhoneticFeature::Sibilant)
        {
            score -= 5;
        }
        score
    };

    // Find the best matching pulmonic consonant
    let mut best_candidate: Option<char> = None;
    let mut best_score: i32 = i32::MIN;

    for &ch in candidates.iter() {
        if ch.is_lowercase() == c.is_lowercase() && is_pulmonic(&ch) {
            let score = score_candidate(&ch);
            if best_candidate.is_none() || score > best_score {
                best_candidate = Some(ch);
                best_score = score;
            }
        }
    }

    // Fall back to any matching consonant if no pulmonic found
    best_candidate.or_else(|| {
        candidates
            .into_iter()
            .find(|ch| ch.is_lowercase() == c.is_lowercase())
    })
}

/// Expand a character to all phonetically similar characters for feature-based matching.
///
/// This is used when the `(?f)` flag is set to expand patterns to match
/// phonetically similar strings.
pub fn expand_feature_based(c: char) -> Vec<char> {
    let features = get_features(c);

    if features.is_empty() {
        return vec![c];
    }

    // Get chars with same place and manner (ignoring voicing)
    let place = features.iter().find(|f| {
        matches!(
            f,
            PhoneticFeature::Bilabial
                | PhoneticFeature::Labiodental
                | PhoneticFeature::Dental
                | PhoneticFeature::Alveolar
                | PhoneticFeature::PostAlveolar
                | PhoneticFeature::Palatal
                | PhoneticFeature::Velar
                | PhoneticFeature::Glottal
                | PhoneticFeature::Retroflex
                | PhoneticFeature::Uvular
                | PhoneticFeature::Pharyngeal
                | PhoneticFeature::Epiglottal
        )
    });

    let manner = features.iter().find(|f| {
        matches!(
            f,
            PhoneticFeature::Stop
                | PhoneticFeature::Fricative
                | PhoneticFeature::Affricate
                | PhoneticFeature::Nasal
                | PhoneticFeature::Approximant
                | PhoneticFeature::Tap
                | PhoneticFeature::Trill
        )
    });

    let mut result = vec![c];

    if let (Some(place), Some(manner)) = (place, manner) {
        // Add voiced and voiceless variants at same place/manner
        let variants = chars_with_features(&[*place, *manner]);
        for variant in variants {
            if variant != c && variant.is_lowercase() == c.is_lowercase() {
                result.push(variant);
            }
        }
    }

    result.sort();
    result.dedup();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_features_vowels() {
        let features = get_features('a');
        assert!(features.contains(&PhoneticFeature::Vowel));
        assert!(features.contains(&PhoneticFeature::Low));
        assert!(!features.contains(&PhoneticFeature::Consonant));
    }

    #[test]
    fn test_get_features_consonants() {
        let features = get_features('p');
        assert!(features.contains(&PhoneticFeature::Consonant));
        assert!(features.contains(&PhoneticFeature::Stop));
        assert!(features.contains(&PhoneticFeature::Bilabial));
        assert!(features.contains(&PhoneticFeature::Voiceless));
    }

    #[test]
    fn test_get_features_unknown() {
        let features = get_features('£');
        assert!(features.is_empty());
    }

    #[test]
    fn test_chars_with_features_voiced_stops() {
        let chars = chars_with_features(&[PhoneticFeature::Voiced, PhoneticFeature::Stop]);
        assert!(chars.contains(&'b'));
        assert!(chars.contains(&'d'));
        assert!(chars.contains(&'g'));
        assert!(!chars.contains(&'p')); // voiceless
        assert!(!chars.contains(&'m')); // nasal, not stop
    }

    #[test]
    fn test_chars_with_any_feature() {
        let chars = chars_with_any_feature(&[PhoneticFeature::Nasal]);
        assert!(chars.contains(&'m'));
        assert!(chars.contains(&'n'));
        assert!(chars.contains(&'ŋ'));
    }

    #[test]
    fn test_are_similar() {
        // b and p share: Consonant, Stop, Bilabial
        assert!(are_similar('b', 'p'));
        // b and d share: Consonant, Stop, Voiced
        assert!(are_similar('b', 'd'));
        // a and b only share nothing meaningful
        assert!(are_similar('a', 'e')); // both vowels
    }

    #[test]
    fn test_get_similar_chars() {
        let similar = get_similar_chars('p');
        // Should include b (voiced bilabial stop), t/k (voiceless stops), etc.
        assert!(similar.contains(&'b'));
        assert!(similar.contains(&'t'));
        assert!(!similar.contains(&'p')); // not itself
    }

    #[test]
    fn test_get_voicing_pair() {
        assert_eq!(get_voicing_pair('p'), Some('b'));
        assert_eq!(get_voicing_pair('b'), Some('p'));
        assert_eq!(get_voicing_pair('t'), Some('d'));
        assert_eq!(get_voicing_pair('d'), Some('t'));
        assert_eq!(get_voicing_pair('k'), Some('g'));
        // 'g' may return 'k', 'c', or 'q' as these are all velar voiceless stops
        let g_pair = get_voicing_pair('g');
        assert!(
            g_pair == Some('k') || g_pair == Some('c') || g_pair == Some('q'),
            "Expected 'k', 'c', or 'q' for 'g' voicing pair, got {:?}",
            g_pair
        );
        assert_eq!(get_voicing_pair('f'), Some('v'));
        assert_eq!(get_voicing_pair('v'), Some('f'));
        assert_eq!(get_voicing_pair('s'), Some('z'));
        assert_eq!(get_voicing_pair('z'), Some('s'));
    }

    #[test]
    fn test_expand_feature_based() {
        let expanded = expand_feature_based('p');
        assert!(expanded.contains(&'p'));
        assert!(expanded.contains(&'b')); // voiced counterpart
    }

    #[test]
    fn test_accented_vowels_have_same_features() {
        let a_features = get_features('a');
        let acute_features = get_features('á');
        assert_eq!(a_features, acute_features);
    }

    #[test]
    fn test_uppercase_has_same_features() {
        let p_lower = get_features('p');
        let p_upper = get_features('P');
        assert_eq!(p_lower, p_upper);
    }
}
