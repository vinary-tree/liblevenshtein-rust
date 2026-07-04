//! Vowel classification traits and implementations for multi-script support.
//!
//! This module provides a trait-based vowel classification system that enables
//! language-specific phonetic rules to correctly identify vowels in different
//! writing systems.
//!
//! # Problem
//!
//! The original vowel classification was hardcoded to English/Latin vowels:
//! ```rust,ignore
//! fn is_vowel_char(c: char) -> bool {
//!     matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U')
//! }
//! ```
//!
//! This breaks context rules (`BeforeVowel`, `AfterVowel`) for non-Latin scripts.
//!
//! # Solution
//!
//! The `VowelClassifier` trait provides a standardized interface for vowel
//! classification across different writing systems:
//!
//! - **Latin**: English, German, Spanish, French, Portuguese, Italian, etc.
//! - **Cyrillic**: Russian, Ukrainian, Bulgarian
//! - **Hangul**: Korean jamo vowels
//! - **Hebrew**: Vowel points (niqqud) and matres lectionis
//! - **Arabic**: Long vowels and diacritical marks
//! - **Devanagari**: Hindi/Sanskrit vowels and matras
//!
//! # Usage
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::classifiers::{VowelClassifier, LatinClassifier, CyrillicClassifier};
//!
//! let latin = LatinClassifier::new();
//! assert!(latin.is_vowel('a'));
//! assert!(!latin.is_vowel('к')); // Cyrillic, not Latin
//!
//! let cyrillic = CyrillicClassifier::new();
//! assert!(cyrillic.is_vowel('а')); // Cyrillic 'a'
//! assert!(cyrillic.is_vowel('о')); // Cyrillic 'o'
//! ```

use std::fmt::Debug;
use std::sync::Arc;

pub mod arabic;
pub mod armenian;
pub mod bengali;
pub mod cyrillic;
pub mod devanagari;
pub mod georgian;
pub mod greek;
pub mod gujarati;
pub mod gurmukhi;
pub mod hangul;
pub mod hanzi;
pub mod hebrew;
pub mod latin;
pub mod tamil;
pub mod telugu;
pub mod thai;

// Re-export classifier implementations
pub use arabic::ArabicClassifier;
pub use armenian::ArmenianClassifier;
pub use bengali::BengaliClassifier;
pub use cyrillic::CyrillicClassifier;
pub use devanagari::DevanagariClassifier;
pub use georgian::GeorgianClassifier;
pub use greek::GreekClassifier;
pub use gujarati::GujaratiClassifier;
pub use gurmukhi::GurmukhiClassifier;
pub use hangul::HangulClassifier;
pub use hanzi::HanziClassifier;
pub use hebrew::HebrewClassifier;
pub use latin::LatinClassifier;
pub use tamil::TamilClassifier;
pub use telugu::TeluguClassifier;
pub use thai::ThaiClassifier;

// ============================================================================
// Vowel Classifier Trait
// ============================================================================

/// Trait for classifying characters as vowels or consonants in a specific script.
///
/// Implementations of this trait define what constitutes a vowel in a particular
/// writing system, enabling phonetic rules to correctly apply context-dependent
/// transformations.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to allow sharing across threads.
/// This is required because rule sets are often shared across multiple threads
/// in concurrent applications.
///
/// # Examples
///
/// Implementing for a custom script:
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::classifiers::VowelClassifier;
///
/// #[derive(Debug, Clone)]
/// struct MyScriptClassifier;
///
/// impl VowelClassifier for MyScriptClassifier {
///     fn is_vowel(&self, c: char) -> bool {
///         // Custom vowel detection logic
///         matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
///     }
///
///     fn script_name(&self) -> &'static str {
///         "MyScript"
///     }
/// }
/// ```
pub trait VowelClassifier: Debug + Send + Sync {
    /// Check if a character is a vowel in this script.
    ///
    /// # Arguments
    ///
    /// * `c` - The character to classify
    ///
    /// # Returns
    ///
    /// `true` if the character is a vowel, `false` otherwise.
    fn is_vowel(&self, c: char) -> bool;

    /// Get the name of the script this classifier handles.
    ///
    /// Used for debugging and error messages.
    fn script_name(&self) -> &'static str;

    /// Optional normalization step before vowel classification.
    ///
    /// Some scripts require normalization (e.g., NFD decomposition for Arabic
    /// diacritics) before vowels can be correctly identified.
    ///
    /// # Default Implementation
    ///
    /// Returns the input unchanged. Override for scripts that need normalization.
    fn normalize(&self, input: &str) -> String {
        input.to_string()
    }

    /// Check if a character is a consonant in this script.
    ///
    /// # Default Implementation
    ///
    /// Returns the opposite of `is_vowel()`. Override for scripts where
    /// some characters are neither vowels nor consonants (e.g., punctuation,
    /// diacritics that modify adjacent characters).
    fn is_consonant(&self, c: char) -> bool {
        !self.is_vowel(c) && c.is_alphabetic()
    }

    /// Get all vowel characters in this script.
    ///
    /// Returns a slice of all characters considered vowels. This is useful
    /// for generating character classes in phonetic rules.
    ///
    /// # Default Implementation
    ///
    /// Returns an empty slice. Override to provide the vowel inventory.
    fn vowels(&self) -> &[char] {
        &[]
    }
}

// ============================================================================
// Dynamic Dispatch Wrapper
// ============================================================================

/// A thread-safe, dynamically-dispatched vowel classifier.
///
/// This wrapper enables storing different classifier types in the same
/// rule set structure without generics.
#[derive(Clone)]
pub struct DynClassifier(Arc<dyn VowelClassifier>);

impl DynClassifier {
    /// Create a new dynamic classifier from any VowelClassifier implementation.
    pub fn new<C: VowelClassifier + 'static>(classifier: C) -> Self {
        Self(Arc::new(classifier))
    }

    /// Create a dynamic classifier for the default Latin script.
    pub fn latin() -> Self {
        Self::new(LatinClassifier::default())
    }

    /// Create a dynamic classifier for Cyrillic script.
    pub fn cyrillic() -> Self {
        Self::new(CyrillicClassifier::default())
    }

    /// Create a dynamic classifier for Korean Hangul.
    pub fn hangul() -> Self {
        Self::new(HangulClassifier)
    }

    /// Create a dynamic classifier for Hebrew script.
    pub fn hebrew() -> Self {
        Self::new(HebrewClassifier)
    }

    /// Create a dynamic classifier for Arabic script.
    pub fn arabic() -> Self {
        Self::new(ArabicClassifier::default())
    }

    /// Create a dynamic classifier for Devanagari script (Hindi).
    pub fn devanagari() -> Self {
        Self::new(DevanagariClassifier)
    }

    /// Create a dynamic classifier for Chinese Hanzi.
    pub fn hanzi() -> Self {
        Self::new(HanziClassifier)
    }

    /// Create a dynamic classifier for Greek script.
    pub fn greek() -> Self {
        Self::new(GreekClassifier)
    }

    /// Create a dynamic classifier for Bengali script.
    pub fn bengali() -> Self {
        Self::new(BengaliClassifier)
    }

    /// Create a dynamic classifier for Gujarati script.
    pub fn gujarati() -> Self {
        Self::new(GujaratiClassifier)
    }

    /// Create a dynamic classifier for Telugu script.
    pub fn telugu() -> Self {
        Self::new(TeluguClassifier)
    }

    /// Create a dynamic classifier for Tamil script.
    pub fn tamil() -> Self {
        Self::new(TamilClassifier)
    }

    /// Create a dynamic classifier for Gurmukhi script (Punjabi - India).
    pub fn gurmukhi() -> Self {
        Self::new(GurmukhiClassifier)
    }

    /// Create a dynamic classifier for Thai script.
    pub fn thai() -> Self {
        Self::new(ThaiClassifier)
    }

    /// Create a dynamic classifier for Georgian Mkhedruli script.
    pub fn georgian() -> Self {
        Self::new(GeorgianClassifier)
    }

    /// Create a dynamic classifier for Armenian script.
    pub fn armenian() -> Self {
        Self::new(ArmenianClassifier)
    }
}

impl VowelClassifier for DynClassifier {
    fn is_vowel(&self, c: char) -> bool {
        self.0.is_vowel(c)
    }

    fn script_name(&self) -> &'static str {
        self.0.script_name()
    }

    fn normalize(&self, input: &str) -> String {
        self.0.normalize(input)
    }

    fn is_consonant(&self, c: char) -> bool {
        self.0.is_consonant(c)
    }

    fn vowels(&self) -> &[char] {
        self.0.vowels()
    }
}

impl Debug for DynClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DynClassifier({})", self.script_name())
    }
}

impl Default for DynClassifier {
    fn default() -> Self {
        Self::latin()
    }
}

// ============================================================================
// Classifier Selection by Language Code
// ============================================================================

/// Get the appropriate classifier for a language code.
///
/// # Arguments
///
/// * `lang` - ISO 639-1 language code (e.g., "en", "ru", "ko")
///
/// # Returns
///
/// A dynamic classifier appropriate for the language's script.
///
/// # Examples
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::classifiers::classifier_for_language;
///
/// let english = classifier_for_language("en");
/// assert_eq!(english.script_name(), "Latin");
///
/// let russian = classifier_for_language("ru");
/// assert_eq!(russian.script_name(), "Cyrillic");
/// ```
pub fn classifier_for_language(lang: &str) -> DynClassifier {
    let lang_lower = lang.to_lowercase();
    match lang_lower.as_str() {
        // Latin script languages
        "en" | "english" | "de" | "german" | "fr" | "french" | "es" | "spanish" | "pt"
        | "portuguese" | "it" | "italian" | "nl" | "dutch" | "pl" | "polish" | "tr" | "turkish"
        | "tl" | "tagalog" | "filipino" | "ja-latn" | "romaji" | "zh-latn" | "pinyin" => {
            DynClassifier::latin()
        }

        // Cyrillic script languages
        "ru" | "russian" | "uk" | "ukrainian" | "bg" | "bulgarian" | "be" | "belarusian" | "sr"
        | "serbian" => DynClassifier::cyrillic(),

        // Greek
        "el" | "greek" => DynClassifier::greek(),

        // Korean (Hangul)
        "ko" | "korean" => DynClassifier::hangul(),

        // Hebrew
        "he" | "hebrew" | "iw" => DynClassifier::hebrew(),

        // Arabic script languages
        "ar" | "arabic" | "ur" | "urdu" | "fa" | "persian" | "farsi" => DynClassifier::arabic(),

        // Devanagari script languages
        "hi" | "hindi" | "sa" | "sanskrit" | "mr" | "marathi" | "ne" | "nepali" => {
            DynClassifier::devanagari()
        }

        // Bengali script languages
        "bn" | "bengali" | "as" | "assamese" => DynClassifier::bengali(),

        // Gujarati script
        "gu" | "gujarati" => DynClassifier::gujarati(),

        // Telugu script
        "te" | "telugu" => DynClassifier::telugu(),

        // Tamil script
        "ta" | "tamil" => DynClassifier::tamil(),

        // Gurmukhi script (Punjabi - India)
        "pa" | "punjabi" => DynClassifier::gurmukhi(),

        // Shahmukhi script (Punjabi - Pakistan) uses Arabic script
        "pa-arab" | "punjabi-shahmukhi" => DynClassifier::arabic(),

        // Thai script
        "th" | "thai" => DynClassifier::thai(),

        // Georgian script
        "ka" | "georgian" => DynClassifier::georgian(),

        // Armenian script
        "hy" | "armenian" => DynClassifier::armenian(),

        // Chinese (native Hanzi)
        "zh" | "chinese" | "hanzi" => DynClassifier::hanzi(),

        // Default to Latin for unknown languages
        _ => DynClassifier::latin(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latin_classifier_basic() {
        let classifier = LatinClassifier::default();
        assert!(classifier.is_vowel('a'));
        assert!(classifier.is_vowel('e'));
        assert!(classifier.is_vowel('i'));
        assert!(classifier.is_vowel('o'));
        assert!(classifier.is_vowel('u'));
        assert!(classifier.is_vowel('A'));
        assert!(!classifier.is_vowel('b'));
        assert!(!classifier.is_vowel('z'));
    }

    #[test]
    fn test_dyn_classifier_dispatch() {
        let latin = DynClassifier::latin();
        assert!(latin.is_vowel('a'));
        assert_eq!(latin.script_name(), "Latin");

        let cyrillic = DynClassifier::cyrillic();
        assert!(cyrillic.is_vowel('а')); // Cyrillic a
        assert_eq!(cyrillic.script_name(), "Cyrillic");
    }

    #[test]
    fn test_classifier_for_language() {
        assert_eq!(classifier_for_language("en").script_name(), "Latin");
        assert_eq!(classifier_for_language("ru").script_name(), "Cyrillic");
        assert_eq!(classifier_for_language("ko").script_name(), "Hangul");
        assert_eq!(classifier_for_language("he").script_name(), "Hebrew");
        assert_eq!(classifier_for_language("ar").script_name(), "Arabic");
        assert_eq!(classifier_for_language("hi").script_name(), "Devanagari");
        assert_eq!(classifier_for_language("zh").script_name(), "Hanzi");
        assert_eq!(classifier_for_language("el").script_name(), "Greek");
        assert_eq!(classifier_for_language("bn").script_name(), "Bengali");
        assert_eq!(classifier_for_language("gu").script_name(), "Gujarati");
        assert_eq!(classifier_for_language("te").script_name(), "Telugu");
        assert_eq!(classifier_for_language("ta").script_name(), "Tamil");
        assert_eq!(classifier_for_language("pa").script_name(), "Gurmukhi");
        assert_eq!(classifier_for_language("pa-arab").script_name(), "Arabic");
        assert_eq!(classifier_for_language("th").script_name(), "Thai");
        assert_eq!(classifier_for_language("ka").script_name(), "Georgian");
        assert_eq!(classifier_for_language("hy").script_name(), "Armenian");
    }

    #[test]
    fn test_default_classifier_is_latin() {
        let default = DynClassifier::default();
        assert_eq!(default.script_name(), "Latin");
    }
}
