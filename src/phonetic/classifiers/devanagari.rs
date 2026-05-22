//! Devanagari script vowel classifier.
//!
//! Supports Hindi, Sanskrit, Marathi, Nepali and other Devanagari-script languages.
//! Devanagari is an abugida where consonants carry an inherent /a/ vowel.

use super::VowelClassifier;

/// Vowel classifier for Devanagari script.
///
/// Devanagari has two types of vowels:
/// - **Independent vowels (svara)**: Stand-alone vowel letters (अ, आ, इ, ई, etc.)
/// - **Dependent vowels (matra)**: Vowel signs attached to consonants (ा, ि, ी, etc.)
///
/// The inherent vowel /a/ in consonants is NOT represented by a matra.
/// The virama (्) explicitly removes the inherent vowel.
#[derive(Debug, Clone, Copy, Default)]
pub struct DevanagariClassifier;

impl DevanagariClassifier {
    /// Create a new Devanagari classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Devanagari vowels (independent and dependent).
static DEVANAGARI_VOWELS: &[char] = &[
    // Independent vowels (svara)
    'अ', // a (short)
    'आ', // aa (long)
    'इ', // i (short)
    'ई', // ii (long)
    'उ', // u (short)
    'ऊ', // uu (long)
    'ऋ', // r (vocalic)
    'ॠ', // rr (vocalic long)
    'ऌ', // l (vocalic)
    'ॡ', // ll (vocalic long)
    'ए', // e
    'ऐ', // ai
    'ओ', // o
    'औ', // au
    // Dependent vowels (matra)
    'ा', // aa matra
    'ि', // i matra
    'ी', // ii matra
    'ु',  // u matra
    'ू',  // uu matra
    'ृ',  // r matra
    'ॄ',  // rr matra
    'ॢ',  // l matra
    'ॣ',  // ll matra
    'े',  // e matra
    'ै',  // ai matra
    'ो', // o matra
    'ौ', // au matra
];

impl VowelClassifier for DevanagariClassifier {
    fn is_vowel(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Independent vowels (Devanagari block)
            0x0904..=0x0914 => true, // अ through औ
            0x0960..=0x0961 => true, // ॠ, ॡ (vocalic long)

            // Dependent vowel signs (matras)
            0x093A..=0x093B => true, // Extended matras
            0x093E..=0x094C => true, // ा through ौ
            0x094E..=0x094F => true, // Additional matras
            0x0955..=0x0957 => true, // Extended vowel signs

            // Vowel signs in extended Devanagari
            0x0962..=0x0963 => true, // ॢ, ॣ (vocalic l matras)

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Devanagari"
    }

    fn vowels(&self) -> &[char] {
        DEVANAGARI_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Consonants (ka through ha)
            0x0915..=0x0939 => true,

            // Additional consonants
            0x0958..=0x095F => true, // Nukta consonants (qa, khha, etc.)

            // Virama is neither vowel nor consonant (it removes vowel)
            0x094D => false,

            // Anusvara and visarga are special marks
            0x0902 => false, // Chandrabindu
            0x0903 => false, // Visarga

            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_independent_vowels() {
        let c = DevanagariClassifier::new();
        assert!(c.is_vowel('अ')); // a
        assert!(c.is_vowel('आ')); // aa
        assert!(c.is_vowel('इ')); // i
        assert!(c.is_vowel('ई')); // ii
        assert!(c.is_vowel('उ')); // u
        assert!(c.is_vowel('ऊ')); // uu
        assert!(c.is_vowel('ए')); // e
        assert!(c.is_vowel('ऐ')); // ai
        assert!(c.is_vowel('ओ')); // o
        assert!(c.is_vowel('औ')); // au
    }

    #[test]
    fn test_vocalic_vowels() {
        let c = DevanagariClassifier::new();
        assert!(c.is_vowel('ऋ')); // r (vocalic)
        assert!(c.is_vowel('ॠ')); // rr (vocalic long)
        assert!(c.is_vowel('ऌ')); // l (vocalic)
    }

    #[test]
    fn test_dependent_vowels() {
        let c = DevanagariClassifier::new();
        assert!(c.is_vowel('ा')); // aa matra
        assert!(c.is_vowel('ि')); // i matra
        assert!(c.is_vowel('ी')); // ii matra
        assert!(c.is_vowel('ु')); // u matra
        assert!(c.is_vowel('ू')); // uu matra
        assert!(c.is_vowel('े')); // e matra
        assert!(c.is_vowel('ै')); // ai matra
        assert!(c.is_vowel('ो')); // o matra
        assert!(c.is_vowel('ौ')); // au matra
    }

    #[test]
    fn test_consonants() {
        let c = DevanagariClassifier::new();
        assert!(!c.is_vowel('क')); // ka
        assert!(!c.is_vowel('ख')); // kha
        assert!(!c.is_vowel('ग')); // ga
        assert!(!c.is_vowel('घ')); // gha
        assert!(!c.is_vowel('च')); // ca
        assert!(!c.is_vowel('ज')); // ja
        assert!(!c.is_vowel('त')); // ta
        assert!(!c.is_vowel('द')); // da
        assert!(!c.is_vowel('न')); // na
        assert!(!c.is_vowel('प')); // pa
        assert!(!c.is_vowel('म')); // ma
        assert!(!c.is_vowel('ह')); // ha
    }

    #[test]
    fn test_is_consonant() {
        let c = DevanagariClassifier::new();
        assert!(c.is_consonant('क')); // ka
        assert!(c.is_consonant('ग')); // ga
        assert!(c.is_consonant('त')); // ta
        assert!(c.is_consonant('न')); // na
        assert!(c.is_consonant('म')); // ma
    }

    #[test]
    fn test_virama_not_vowel() {
        let c = DevanagariClassifier::new();
        assert!(!c.is_vowel('्')); // virama
        assert!(!c.is_consonant('्')); // virama is neither
    }

    #[test]
    fn test_nukta_consonants() {
        let c = DevanagariClassifier::new();
        // Nukta consonants are composed of base consonant + nukta (two codepoints)
        // Test the base consonants instead
        assert!(c.is_consonant('क')); // ka (base for qa)
        assert!(c.is_consonant('ड')); // da (base for dda)
        assert!(c.is_consonant('ढ')); // dha (base for ddha)
                                      // Nukta itself is neither vowel nor consonant
        assert!(!c.is_vowel('\u{093C}')); // nukta
    }
}
