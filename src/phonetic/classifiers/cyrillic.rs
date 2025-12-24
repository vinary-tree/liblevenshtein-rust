//! Cyrillic script vowel classifier.
//!
//! Supports Russian, Ukrainian, Bulgarian, and other Cyrillic-script languages.

use super::VowelClassifier;

/// Vowel classifier for Cyrillic-script languages.
///
/// Recognizes Russian/Slavic vowels:
/// - а, е, ё, и, о, у, ы, э, ю, я (and uppercase)
/// - Ukrainian specific: і, ї, є
/// - Bulgarian/other: ъ (sometimes treated as vowel)
#[derive(Debug, Clone, Copy, Default)]
pub struct CyrillicClassifier {
    /// Include Ukrainian-specific vowels (і, ї, є)
    pub include_ukrainian: bool,
}

impl CyrillicClassifier {
    /// Create a new Cyrillic classifier with standard Russian vowels.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a classifier that includes Ukrainian vowels.
    pub fn ukrainian() -> Self {
        Self {
            include_ukrainian: true,
        }
    }
}

/// Static list of Cyrillic vowels.
static CYRILLIC_VOWELS: &[char] = &[
    // Russian vowels (lowercase)
    'а', 'е', 'ё', 'и', 'о', 'у', 'ы', 'э', 'ю', 'я',
    // Russian vowels (uppercase)
    'А', 'Е', 'Ё', 'И', 'О', 'У', 'Ы', 'Э', 'Ю', 'Я',
    // Ukrainian specific
    'і', 'ї', 'є', 'І', 'Ї', 'Є',
];

impl VowelClassifier for CyrillicClassifier {
    fn is_vowel(&self, c: char) -> bool {
        match c {
            // Russian vowels (most common)
            'а' | 'е' | 'ё' | 'и' | 'о' | 'у' | 'ы' | 'э' | 'ю' | 'я' => true,
            'А' | 'Е' | 'Ё' | 'И' | 'О' | 'У' | 'Ы' | 'Э' | 'Ю' | 'Я' => true,

            // Ukrainian vowels (optional)
            'і' | 'ї' | 'є' | 'І' | 'Ї' | 'Є' if self.include_ukrainian => true,
            // Also include them by default since they're clearly vowels
            'і' | 'ї' | 'є' | 'І' | 'Ї' | 'Є' => true,

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Cyrillic"
    }

    fn vowels(&self) -> &[char] {
        CYRILLIC_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        // Cyrillic consonants
        match c {
            'б' | 'в' | 'г' | 'д' | 'ж' | 'з' | 'й' | 'к' | 'л' | 'м' | 'н' | 'п' | 'р' | 'с'
            | 'т' | 'ф' | 'х' | 'ц' | 'ч' | 'ш' | 'щ' => true,
            'Б' | 'В' | 'Г' | 'Д' | 'Ж' | 'З' | 'Й' | 'К' | 'Л' | 'М' | 'Н' | 'П' | 'Р' | 'С'
            | 'Т' | 'Ф' | 'Х' | 'Ц' | 'Ч' | 'Ш' | 'Щ' => true,
            // Soft and hard signs are neither vowel nor consonant
            'ь' | 'ъ' | 'Ь' | 'Ъ' => false,
            // Ukrainian consonant
            'ґ' | 'Ґ' => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_russian_vowels() {
        let c = CyrillicClassifier::new();
        assert!(c.is_vowel('а'));
        assert!(c.is_vowel('е'));
        assert!(c.is_vowel('ё'));
        assert!(c.is_vowel('и'));
        assert!(c.is_vowel('о'));
        assert!(c.is_vowel('у'));
        assert!(c.is_vowel('ы'));
        assert!(c.is_vowel('э'));
        assert!(c.is_vowel('ю'));
        assert!(c.is_vowel('я'));
    }

    #[test]
    fn test_russian_consonants() {
        let c = CyrillicClassifier::new();
        assert!(!c.is_vowel('б'));
        assert!(!c.is_vowel('в'));
        assert!(!c.is_vowel('г'));
        assert!(!c.is_vowel('д'));
        assert!(!c.is_vowel('к'));
        assert!(!c.is_vowel('л'));
        assert!(!c.is_vowel('м'));
        assert!(!c.is_vowel('н'));
    }

    #[test]
    fn test_soft_hard_signs() {
        let c = CyrillicClassifier::new();
        // Soft and hard signs are neither vowel nor consonant
        assert!(!c.is_vowel('ь'));
        assert!(!c.is_vowel('ъ'));
        assert!(!c.is_consonant('ь'));
        assert!(!c.is_consonant('ъ'));
    }

    #[test]
    fn test_ukrainian_vowels() {
        let c = CyrillicClassifier::ukrainian();
        assert!(c.is_vowel('і')); // Ukrainian i
        assert!(c.is_vowel('ї')); // Ukrainian yi
        assert!(c.is_vowel('є')); // Ukrainian ye
    }

    #[test]
    fn test_uppercase() {
        let c = CyrillicClassifier::new();
        assert!(c.is_vowel('А'));
        assert!(c.is_vowel('Е'));
        assert!(c.is_vowel('О'));
        assert!(!c.is_vowel('Б'));
        assert!(!c.is_vowel('К'));
    }
}
