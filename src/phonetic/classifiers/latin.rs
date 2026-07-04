//! Latin script vowel classifier.
//!
//! Supports English, German, French, Spanish, Portuguese, Italian, Dutch,
//! Polish, Turkish, Tagalog, and other Latin-script languages.

use super::VowelClassifier;

/// Vowel classifier for Latin-script languages.
///
/// Recognizes vowels including:
/// - Basic vowels: a, e, i, o, u (and uppercase)
/// - Accented vowels: á, é, í, ó, ú, à, è, ì, ò, ù, â, ê, î, ô, û, etc.
/// - Umlauts: ä, ë, ï, ö, ü
/// - Special vowels: ã, õ (Portuguese), ø, æ (Scandinavian), y (sometimes)
#[derive(Debug, Clone, Copy, Default)]
pub struct LatinClassifier {
    /// Whether to treat 'y' as a vowel (varies by language)
    pub y_is_vowel: bool,
}

impl LatinClassifier {
    /// Create a new Latin classifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a classifier that treats 'y' as a vowel.
    ///
    /// Used for languages like Welsh where 'y' is always a vowel.
    pub fn with_y_as_vowel() -> Self {
        Self { y_is_vowel: true }
    }
}

/// Static list of Latin vowels for the `vowels()` method.
static LATIN_VOWELS: &[char] = &[
    // Basic vowels
    'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U', // Acute accents
    'á', 'é', 'í', 'ó', 'ú', 'Á', 'É', 'Í', 'Ó', 'Ú', // Grave accents
    'à', 'è', 'ì', 'ò', 'ù', 'À', 'È', 'Ì', 'Ò', 'Ù', // Circumflex
    'â', 'ê', 'î', 'ô', 'û', 'Â', 'Ê', 'Î', 'Ô', 'Û', // Umlauts/diaeresis
    'ä', 'ë', 'ï', 'ö', 'ü', 'Ä', 'Ë', 'Ï', 'Ö', 'Ü', // Tilde (Portuguese, Spanish)
    'ã', 'õ', 'ñ', 'Ã', 'Õ', 'Ñ', // Note: ñ is consonant in Spanish
    // Scandinavian
    'æ', 'ø', 'å', 'Æ', 'Ø', 'Å', // Other
    'œ', 'Œ', // French ligature
];

impl VowelClassifier for LatinClassifier {
    fn is_vowel(&self, c: char) -> bool {
        match c {
            // Basic vowels (most common, check first)
            'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U' => true,

            // Y as vowel (optional)
            'y' | 'Y' if self.y_is_vowel => true,

            // Acute accents
            'á' | 'é' | 'í' | 'ó' | 'ú' | 'Á' | 'É' | 'Í' | 'Ó' | 'Ú' => true,

            // Grave accents
            'à' | 'è' | 'ì' | 'ò' | 'ù' | 'À' | 'È' | 'Ì' | 'Ò' | 'Ù' => true,

            // Circumflex
            'â' | 'ê' | 'î' | 'ô' | 'û' | 'Â' | 'Ê' | 'Î' | 'Ô' | 'Û' => true,

            // Umlauts/diaeresis
            'ä' | 'ë' | 'ï' | 'ö' | 'ü' | 'Ä' | 'Ë' | 'Ï' | 'Ö' | 'Ü' => true,

            // Tilde vowels (Portuguese)
            'ã' | 'õ' | 'Ã' | 'Õ' => true,

            // Scandinavian vowels
            'æ' | 'ø' | 'å' | 'Æ' | 'Ø' | 'Å' => true,

            // French/Latin ligatures
            'œ' | 'Œ' => true,

            // Turkish dotless i
            'ı' | 'İ' => true,

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Latin"
    }

    fn vowels(&self) -> &[char] {
        LATIN_VOWELS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_vowels() {
        let c = LatinClassifier::new();
        assert!(c.is_vowel('a'));
        assert!(c.is_vowel('e'));
        assert!(c.is_vowel('i'));
        assert!(c.is_vowel('o'));
        assert!(c.is_vowel('u'));
        assert!(c.is_vowel('A'));
        assert!(c.is_vowel('E'));
        assert!(c.is_vowel('I'));
        assert!(c.is_vowel('O'));
        assert!(c.is_vowel('U'));
    }

    #[test]
    fn test_accented_vowels() {
        let c = LatinClassifier::new();
        assert!(c.is_vowel('á'));
        assert!(c.is_vowel('é'));
        assert!(!c.is_vowel('ñ')); // ñ is consonant
        assert!(c.is_vowel('ö'));
        assert!(c.is_vowel('ü'));
        assert!(c.is_vowel('ã'));
        assert!(c.is_vowel('æ'));
    }

    #[test]
    fn test_consonants() {
        let c = LatinClassifier::new();
        assert!(!c.is_vowel('b'));
        assert!(!c.is_vowel('c'));
        assert!(!c.is_vowel('d'));
        assert!(!c.is_vowel('z'));
        assert!(!c.is_vowel('ñ'));
        assert!(!c.is_vowel('ß'));
    }

    #[test]
    fn test_y_handling() {
        let default = LatinClassifier::new();
        assert!(!default.is_vowel('y'));

        let with_y = LatinClassifier::with_y_as_vowel();
        assert!(with_y.is_vowel('y'));
        assert!(with_y.is_vowel('Y'));
    }

    #[test]
    fn test_turkish_dotless_i() {
        let c = LatinClassifier::new();
        assert!(c.is_vowel('ı')); // dotless i
        assert!(c.is_vowel('İ')); // dotted I
    }
}
