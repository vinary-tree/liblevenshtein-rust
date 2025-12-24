//! Greek script vowel classifier.
//!
//! Greek uses an alphabet with distinct vowel and consonant letters.
//! This classifier identifies Greek vowels including accented variants.

use super::VowelClassifier;

/// Vowel classifier for Greek script.
///
/// Greek has 7 vowel letters (with 2 pairs that sound the same in Modern Greek):
/// - **α** (alpha): /a/
/// - **ε** (epsilon): /e/
/// - **η** (eta): /i/ (Modern Greek)
/// - **ι** (iota): /i/
/// - **ο** (omicron): /o/
/// - **υ** (upsilon): /i/ (Modern Greek, historically /y/)
/// - **ω** (omega): /o/
///
/// All vowels can have accent marks (tonos): ά, έ, ή, ί, ό, ύ, ώ
/// Some can have diaeresis: ϊ, ϋ (indicates separate syllable)
/// Combined: ΐ, ΰ (accent + diaeresis)
#[derive(Debug, Clone, Copy, Default)]
pub struct GreekClassifier;

impl GreekClassifier {
    /// Create a new Greek classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Greek vowels (base forms, lowercase and uppercase).
static GREEK_VOWELS: &[char] = &[
    // Lowercase vowels
    'α', 'ε', 'η', 'ι', 'ο', 'υ', 'ω',
    // Uppercase vowels
    'Α', 'Ε', 'Η', 'Ι', 'Ο', 'Υ', 'Ω',
    // Accented lowercase (tonos)
    'ά', 'έ', 'ή', 'ί', 'ό', 'ύ', 'ώ',
    // Accented uppercase (tonos)
    'Ά', 'Έ', 'Ή', 'Ί', 'Ό', 'Ύ', 'Ώ',
    // With diaeresis
    'ϊ', 'ϋ', 'Ϊ', 'Ϋ',
    // With both tonos and diaeresis
    'ΐ', 'ΰ',
];

impl VowelClassifier for GreekClassifier {
    fn is_vowel(&self, c: char) -> bool {
        match c {
            // Lowercase base vowels
            'α' | 'ε' | 'η' | 'ι' | 'ο' | 'υ' | 'ω' => true,
            // Uppercase base vowels
            'Α' | 'Ε' | 'Η' | 'Ι' | 'Ο' | 'Υ' | 'Ω' => true,
            // Lowercase with tonos (accent)
            'ά' | 'έ' | 'ή' | 'ί' | 'ό' | 'ύ' | 'ώ' => true,
            // Uppercase with tonos
            'Ά' | 'Έ' | 'Ή' | 'Ί' | 'Ό' | 'Ύ' | 'Ώ' => true,
            // With diaeresis
            'ϊ' | 'ϋ' | 'Ϊ' | 'Ϋ' => true,
            // With both tonos and diaeresis
            'ΐ' | 'ΰ' => true,
            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Greek"
    }

    fn vowels(&self) -> &[char] {
        GREEK_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        match c {
            // Lowercase consonants
            'β' | 'γ' | 'δ' | 'ζ' | 'θ' | 'κ' | 'λ' | 'μ' |
            'ν' | 'ξ' | 'π' | 'ρ' | 'σ' | 'ς' | 'τ' | 'φ' |
            'χ' | 'ψ' => true,
            // Uppercase consonants
            'Β' | 'Γ' | 'Δ' | 'Ζ' | 'Θ' | 'Κ' | 'Λ' | 'Μ' |
            'Ν' | 'Ξ' | 'Π' | 'Ρ' | 'Σ' | 'Τ' | 'Φ' | 'Χ' |
            'Ψ' => true,
            _ => false,
        }
    }

    fn normalize(&self, input: &str) -> String {
        // NFD decomposition to separate base letters from combining marks
        use unicode_normalization::UnicodeNormalization;
        input.nfd().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_vowels_lowercase() {
        let c = GreekClassifier::new();
        assert!(c.is_vowel('α')); // alpha
        assert!(c.is_vowel('ε')); // epsilon
        assert!(c.is_vowel('η')); // eta
        assert!(c.is_vowel('ι')); // iota
        assert!(c.is_vowel('ο')); // omicron
        assert!(c.is_vowel('υ')); // upsilon
        assert!(c.is_vowel('ω')); // omega
    }

    #[test]
    fn test_base_vowels_uppercase() {
        let c = GreekClassifier::new();
        assert!(c.is_vowel('Α')); // Alpha
        assert!(c.is_vowel('Ε')); // Epsilon
        assert!(c.is_vowel('Η')); // Eta
        assert!(c.is_vowel('Ι')); // Iota
        assert!(c.is_vowel('Ο')); // Omicron
        assert!(c.is_vowel('Υ')); // Upsilon
        assert!(c.is_vowel('Ω')); // Omega
    }

    #[test]
    fn test_accented_vowels() {
        let c = GreekClassifier::new();
        assert!(c.is_vowel('ά')); // alpha with tonos
        assert!(c.is_vowel('έ')); // epsilon with tonos
        assert!(c.is_vowel('ή')); // eta with tonos
        assert!(c.is_vowel('ί')); // iota with tonos
        assert!(c.is_vowel('ό')); // omicron with tonos
        assert!(c.is_vowel('ύ')); // upsilon with tonos
        assert!(c.is_vowel('ώ')); // omega with tonos
    }

    #[test]
    fn test_diaeresis_vowels() {
        let c = GreekClassifier::new();
        assert!(c.is_vowel('ϊ')); // iota with diaeresis
        assert!(c.is_vowel('ϋ')); // upsilon with diaeresis
        assert!(c.is_vowel('ΐ')); // iota with tonos and diaeresis
        assert!(c.is_vowel('ΰ')); // upsilon with tonos and diaeresis
    }

    #[test]
    fn test_consonants_lowercase() {
        let c = GreekClassifier::new();
        assert!(c.is_consonant('β')); // beta
        assert!(c.is_consonant('γ')); // gamma
        assert!(c.is_consonant('δ')); // delta
        assert!(c.is_consonant('ζ')); // zeta
        assert!(c.is_consonant('θ')); // theta
        assert!(c.is_consonant('κ')); // kappa
        assert!(c.is_consonant('λ')); // lambda
        assert!(c.is_consonant('μ')); // mu
        assert!(c.is_consonant('ν')); // nu
        assert!(c.is_consonant('ξ')); // xi
        assert!(c.is_consonant('π')); // pi
        assert!(c.is_consonant('ρ')); // rho
        assert!(c.is_consonant('σ')); // sigma
        assert!(c.is_consonant('ς')); // final sigma
        assert!(c.is_consonant('τ')); // tau
        assert!(c.is_consonant('φ')); // phi
        assert!(c.is_consonant('χ')); // chi
        assert!(c.is_consonant('ψ')); // psi
    }

    #[test]
    fn test_consonants_not_vowels() {
        let c = GreekClassifier::new();
        assert!(!c.is_vowel('β'));
        assert!(!c.is_vowel('γ'));
        assert!(!c.is_vowel('δ'));
        assert!(!c.is_vowel('κ'));
        assert!(!c.is_vowel('π'));
        assert!(!c.is_vowel('σ'));
        assert!(!c.is_vowel('τ'));
    }

    #[test]
    fn test_vowels_not_consonants() {
        let c = GreekClassifier::new();
        assert!(!c.is_consonant('α'));
        assert!(!c.is_consonant('ε'));
        assert!(!c.is_consonant('ι'));
        assert!(!c.is_consonant('ο'));
        assert!(!c.is_consonant('υ'));
    }
}
