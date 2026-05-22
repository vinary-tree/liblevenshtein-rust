//! Character-level substitution set for full Unicode support.
//!
//! This module provides [`SubstitutionSetChar`], which works with full Unicode
//! characters (`char`) instead of just ASCII bytes (`u8`). Use this with
//! character-level dictionaries for full Unicode substitution support.
//!
//! ## Relationship to SubstitutionSet
//!
//! - [`SubstitutionSet`](super::SubstitutionSet): Works with bytes (`u8`), for use with `DoubleArrayTrie`
//! - [`SubstitutionSetChar`]: Works with characters (`char`), for use with `DoubleArrayTrieChar`
//!
//! ## Use Cases
//!
//! - **International text**: Allow Unicode character substitutions
//! - **Diacritics**: Allow accented/unaccented equivalences (é ↔ e, ñ ↔ n)
//! - **Emoji variations**: Allow emoji modifiers (👋 ↔ 👋🏻)
//! - **Japanese**: Allow hiragana/katakana equivalences (あ ↔ ア)
//! - **Chinese**: Allow simplified/traditional equivalences (学 ↔ 學)
//!
//! ## Example
//!
//! ```rust
//! use liblevenshtein::transducer::SubstitutionSetChar;
//!
//! // Create diacritic substitution set
//! let mut diacritics = SubstitutionSetChar::new();
//! diacritics.allow('é', 'e');
//! diacritics.allow('e', 'é');
//! diacritics.allow('ñ', 'n');
//! diacritics.allow('n', 'ñ');
//!
//! // Or use a preset
//! let diacritics = SubstitutionSetChar::diacritics_latin();
//!
//! // Verify equivalences
//! assert!(diacritics.contains('é', 'e'));
//! assert!(diacritics.contains('ñ', 'n'));
//! ```

use rustc_hash::FxHashSet;

/// Character-level set of allowed substitutions for full Unicode support.
///
/// A `SubstitutionSetChar` defines which Unicode character pairs can be substituted
/// for each other during fuzzy matching. Unlike [`SubstitutionSet`](super::SubstitutionSet),
/// which only supports ASCII bytes (0-127), this supports the full Unicode character
/// range (U+0000 to U+10FFFF).
///
/// ## Performance
///
/// - **Storage**: HashSet with fast non-cryptographic hashing (FxHasher)
/// - **Lookup**: O(1) average case, ~10-30ns per check
/// - **Memory**: ~48 bytes base + 24 bytes per allowed pair (double `SubstitutionSet` due to `char` size)
///
/// ## Symmetry
///
/// Substitutions are **not symmetric by default**. If you want bidirectional
/// substitutions, you must add both directions explicitly:
///
/// ```rust
/// # use liblevenshtein::transducer::SubstitutionSetChar;
/// let mut set = SubstitutionSetChar::new();
/// set.allow('é', 'e');  // 'é' can be substituted with 'e'
/// set.allow('e', 'é');  // 'e' can be substituted with 'é' (symmetric)
/// ```
///
/// Most presets (like `diacritics_latin()`) include symmetric pairs where appropriate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubstitutionSetChar {
    /// Allowed substitution pairs (dict_char, query_char).
    /// Uses FxHasher for fast non-cryptographic hashing.
    allowed: FxHashSet<(char, char)>,
}

impl SubstitutionSetChar {
    /// Create an empty substitution set.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSetChar;
    /// let mut set = SubstitutionSetChar::new();
    /// set.allow('α', 'β');
    /// assert!(set.contains('α', 'β'));
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            allowed: FxHashSet::default(),
        }
    }

    /// Create a substitution set with expected capacity.
    ///
    /// Pre-allocates space for `capacity` substitution pairs to avoid
    /// reallocations during construction.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSetChar;
    /// let mut set = SubstitutionSetChar::with_capacity(100);
    /// // Add many pairs without reallocations
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            allowed: FxHashSet::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Allow substituting character `a` with character `b`.
    ///
    /// Works with any Unicode character (U+0000 to U+10FFFF).
    ///
    /// # Parameters
    ///
    /// - `a`: Dictionary character (source)
    /// - `b`: Query character (target)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSetChar;
    /// let mut set = SubstitutionSetChar::new();
    /// set.allow('é', 'e');  // 'é' in dict can match 'e' in query
    /// set.allow('ñ', 'n');  // 'ñ' in dict can match 'n' in query
    ///
    /// // This enables "café" to match "cafe" via é→e substitution
    /// ```
    #[inline]
    pub fn allow(&mut self, a: char, b: char) {
        self.allowed.insert((a, b));
    }

    /// Check if substituting character `a` with character `b` is allowed.
    ///
    /// This is the hot-path method called during character matching.
    /// It's marked `#[inline]` for performance.
    ///
    /// # Parameters
    ///
    /// - `a`: Dictionary character
    /// - `b`: Query character
    ///
    /// # Returns
    ///
    /// `true` if the substitution `a → b` is allowed, `false` otherwise.
    ///
    /// # Performance
    ///
    /// O(1) average case, ~10-30ns per lookup with FxHasher.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSetChar;
    /// let mut set = SubstitutionSetChar::new();
    /// set.allow('é', 'e');
    ///
    /// assert!(set.contains('é', 'e'));
    /// assert!(!set.contains('e', 'é'));  // Not symmetric
    /// ```
    #[inline]
    pub fn contains(&self, a: char, b: char) -> bool {
        self.allowed.contains(&(a, b))
    }

    /// Build a substitution set from character pairs.
    ///
    /// # Parameters
    ///
    /// - `pairs`: Slice of (source, target) character pairs
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSetChar;
    /// let set = SubstitutionSetChar::from_pairs(&[
    ///     ('é', 'e'), ('e', 'é'),  // symmetric
    ///     ('ñ', 'n'), ('n', 'ñ'),  // symmetric
    /// ]);
    ///
    /// assert!(set.contains('é', 'e'));
    /// assert!(set.contains('ñ', 'n'));
    /// ```
    pub fn from_pairs(pairs: &[(char, char)]) -> Self {
        let mut set = Self::with_capacity(pairs.len());
        for &(a, b) in pairs {
            set.allow(a, b);
        }
        set
    }

    /// Get the number of allowed substitution pairs.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSetChar;
    /// let mut set = SubstitutionSetChar::new();
    /// assert_eq!(set.len(), 0);
    ///
    /// set.allow('α', 'β');
    /// set.allow('γ', 'δ');
    /// assert_eq!(set.len(), 2);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.allowed.len()
    }

    /// Check if the substitution set is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSetChar;
    /// let set = SubstitutionSetChar::new();
    /// assert!(set.is_empty());
    ///
    /// let diacritics = SubstitutionSetChar::diacritics_latin();
    /// assert!(!diacritics.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty()
    }

    /// Clear all allowed substitutions.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSetChar;
    /// let mut set = SubstitutionSetChar::diacritics_latin();
    /// assert!(!set.is_empty());
    ///
    /// set.clear();
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.allowed.clear();
    }

    // ========================================================================
    // Preset Builders
    // ========================================================================

    /// Latin diacritic equivalences (accented ↔ unaccented).
    ///
    /// Includes bidirectional substitutions for common Latin diacritics:
    /// - **á/à/â/ä/ã/å ↔ a**: Various 'a' diacritics
    /// - **é/è/ê/ë ↔ e**: Various 'e' diacritics
    /// - **í/ì/î/ï ↔ i**: Various 'i' diacritics
    /// - **ó/ò/ô/ö/õ ↔ o**: Various 'o' diacritics
    /// - **ú/ù/û/ü ↔ u**: Various 'u' diacritics
    /// - **ñ ↔ n**: Spanish ñ
    /// - **ç ↔ c**: Cedilla
    ///
    /// Useful for matching text where diacritics may be omitted or inconsistent.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSetChar;
    /// let diacritics = SubstitutionSetChar::diacritics_latin();
    ///
    /// // é ↔ e
    /// assert!(diacritics.contains('é', 'e'));
    /// assert!(diacritics.contains('e', 'é'));
    ///
    /// // ñ ↔ n
    /// assert!(diacritics.contains('ñ', 'n'));
    /// assert!(diacritics.contains('n', 'ñ'));
    /// ```
    pub fn diacritics_latin() -> Self {
        Self::from_pairs(&[
            // Lowercase a variants
            ('á', 'a'),
            ('a', 'á'),
            ('à', 'a'),
            ('a', 'à'),
            ('â', 'a'),
            ('a', 'â'),
            ('ä', 'a'),
            ('a', 'ä'),
            ('ã', 'a'),
            ('a', 'ã'),
            ('å', 'a'),
            ('a', 'å'),
            ('æ', 'a'),
            ('a', 'æ'),
            // Lowercase e variants
            ('é', 'e'),
            ('e', 'é'),
            ('è', 'e'),
            ('e', 'è'),
            ('ê', 'e'),
            ('e', 'ê'),
            ('ë', 'e'),
            ('e', 'ë'),
            // Lowercase i variants
            ('í', 'i'),
            ('i', 'í'),
            ('ì', 'i'),
            ('i', 'ì'),
            ('î', 'i'),
            ('i', 'î'),
            ('ï', 'i'),
            ('i', 'ï'),
            // Lowercase o variants
            ('ó', 'o'),
            ('o', 'ó'),
            ('ò', 'o'),
            ('o', 'ò'),
            ('ô', 'o'),
            ('o', 'ô'),
            ('ö', 'o'),
            ('o', 'ö'),
            ('õ', 'o'),
            ('o', 'õ'),
            ('ø', 'o'),
            ('o', 'ø'),
            ('œ', 'o'),
            ('o', 'œ'),
            // Lowercase u variants
            ('ú', 'u'),
            ('u', 'ú'),
            ('ù', 'u'),
            ('u', 'ù'),
            ('û', 'u'),
            ('u', 'û'),
            ('ü', 'u'),
            ('u', 'ü'),
            // Lowercase special
            ('ñ', 'n'),
            ('n', 'ñ'),
            ('ç', 'c'),
            ('c', 'ç'),
            ('ß', 's'),
            ('s', 'ß'),
            // Uppercase A variants
            ('Á', 'A'),
            ('A', 'Á'),
            ('À', 'A'),
            ('A', 'À'),
            ('Â', 'A'),
            ('A', 'Â'),
            ('Ä', 'A'),
            ('A', 'Ä'),
            ('Ã', 'A'),
            ('A', 'Ã'),
            ('Å', 'A'),
            ('A', 'Å'),
            ('Æ', 'A'),
            ('A', 'Æ'),
            // Uppercase E variants
            ('É', 'E'),
            ('E', 'É'),
            ('È', 'E'),
            ('E', 'È'),
            ('Ê', 'E'),
            ('E', 'Ê'),
            ('Ë', 'E'),
            ('E', 'Ë'),
            // Uppercase I variants
            ('Í', 'I'),
            ('I', 'Í'),
            ('Ì', 'I'),
            ('I', 'Ì'),
            ('Î', 'I'),
            ('I', 'Î'),
            ('Ï', 'I'),
            ('I', 'Ï'),
            // Uppercase O variants
            ('Ó', 'O'),
            ('O', 'Ó'),
            ('Ò', 'O'),
            ('O', 'Ò'),
            ('Ô', 'O'),
            ('O', 'Ô'),
            ('Ö', 'O'),
            ('O', 'Ö'),
            ('Õ', 'O'),
            ('O', 'Õ'),
            ('Ø', 'O'),
            ('O', 'Ø'),
            ('Œ', 'O'),
            ('O', 'Œ'),
            // Uppercase U variants
            ('Ú', 'U'),
            ('U', 'Ú'),
            ('Ù', 'U'),
            ('U', 'Ù'),
            ('Û', 'U'),
            ('U', 'Û'),
            ('Ü', 'U'),
            ('U', 'Ü'),
            // Uppercase special
            ('Ñ', 'N'),
            ('N', 'Ñ'),
            ('Ç', 'C'),
            ('C', 'Ç'),
        ])
    }

    /// Greek character equivalences (uppercase ↔ lowercase).
    ///
    /// Includes case-insensitive matching for Greek alphabet.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSetChar;
    /// let greek = SubstitutionSetChar::greek_case_insensitive();
    ///
    /// // Α ↔ α
    /// assert!(greek.contains('Α', 'α'));
    /// assert!(greek.contains('α', 'Α'));
    /// ```
    pub fn greek_case_insensitive() -> Self {
        Self::from_pairs(&[
            ('Α', 'α'),
            ('α', 'Α'), // Alpha
            ('Β', 'β'),
            ('β', 'Β'), // Beta
            ('Γ', 'γ'),
            ('γ', 'Γ'), // Gamma
            ('Δ', 'δ'),
            ('δ', 'Δ'), // Delta
            ('Ε', 'ε'),
            ('ε', 'Ε'), // Epsilon
            ('Ζ', 'ζ'),
            ('ζ', 'Ζ'), // Zeta
            ('Η', 'η'),
            ('η', 'Η'), // Eta
            ('Θ', 'θ'),
            ('θ', 'Θ'), // Theta
            ('Ι', 'ι'),
            ('ι', 'Ι'), // Iota
            ('Κ', 'κ'),
            ('κ', 'Κ'), // Kappa
            ('Λ', 'λ'),
            ('λ', 'Λ'), // Lambda
            ('Μ', 'μ'),
            ('μ', 'Μ'), // Mu
            ('Ν', 'ν'),
            ('ν', 'Ν'), // Nu
            ('Ξ', 'ξ'),
            ('ξ', 'Ξ'), // Xi
            ('Ο', 'ο'),
            ('ο', 'Ο'), // Omicron
            ('Π', 'π'),
            ('π', 'Π'), // Pi
            ('Ρ', 'ρ'),
            ('ρ', 'Ρ'), // Rho
            ('Σ', 'σ'),
            ('σ', 'Σ'), // Sigma
            ('Σ', 'ς'),
            ('ς', 'Σ'), // Sigma (final)
            ('Τ', 'τ'),
            ('τ', 'Τ'), // Tau
            ('Υ', 'υ'),
            ('υ', 'Υ'), // Upsilon
            ('Φ', 'φ'),
            ('φ', 'Φ'), // Phi
            ('Χ', 'χ'),
            ('χ', 'Χ'), // Chi
            ('Ψ', 'ψ'),
            ('ψ', 'Ψ'), // Psi
            ('Ω', 'ω'),
            ('ω', 'Ω'), // Omega
        ])
    }

    /// Cyrillic character equivalences (uppercase ↔ lowercase).
    ///
    /// Includes case-insensitive matching for Cyrillic alphabet (Russian).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSetChar;
    /// let cyrillic = SubstitutionSetChar::cyrillic_case_insensitive();
    ///
    /// // А ↔ а
    /// assert!(cyrillic.contains('А', 'а'));
    /// assert!(cyrillic.contains('а', 'А'));
    /// ```
    pub fn cyrillic_case_insensitive() -> Self {
        Self::from_pairs(&[
            ('А', 'а'),
            ('а', 'А'), // A
            ('Б', 'б'),
            ('б', 'Б'), // Be
            ('В', 'в'),
            ('в', 'В'), // Ve
            ('Г', 'г'),
            ('г', 'Г'), // Ghe
            ('Д', 'д'),
            ('д', 'Д'), // De
            ('Е', 'е'),
            ('е', 'Е'), // Ye
            ('Ё', 'ё'),
            ('ё', 'Ё'), // Yo
            ('Ж', 'ж'),
            ('ж', 'Ж'), // Zhe
            ('З', 'з'),
            ('з', 'З'), // Ze
            ('И', 'и'),
            ('и', 'И'), // I
            ('Й', 'й'),
            ('й', 'Й'), // Short I
            ('К', 'к'),
            ('к', 'К'), // Ka
            ('Л', 'л'),
            ('л', 'Л'), // El
            ('М', 'м'),
            ('м', 'М'), // Em
            ('Н', 'н'),
            ('н', 'Н'), // En
            ('О', 'о'),
            ('о', 'О'), // O
            ('П', 'п'),
            ('п', 'П'), // Pe
            ('Р', 'р'),
            ('р', 'Р'), // Er
            ('С', 'с'),
            ('с', 'С'), // Es
            ('Т', 'т'),
            ('т', 'Т'), // Te
            ('У', 'у'),
            ('у', 'У'), // U
            ('Ф', 'ф'),
            ('ф', 'Ф'), // Ef
            ('Х', 'х'),
            ('х', 'Х'), // Ha
            ('Ц', 'ц'),
            ('ц', 'Ц'), // Tse
            ('Ч', 'ч'),
            ('ч', 'Ч'), // Che
            ('Ш', 'ш'),
            ('ш', 'Ш'), // Sha
            ('Щ', 'щ'),
            ('щ', 'Щ'), // Shcha
            ('Ъ', 'ъ'),
            ('ъ', 'Ъ'), // Hard sign
            ('Ы', 'ы'),
            ('ы', 'Ы'), // Yeru
            ('Ь', 'ь'),
            ('ь', 'Ь'), // Soft sign
            ('Э', 'э'),
            ('э', 'Э'), // E
            ('Ю', 'ю'),
            ('ю', 'Ю'), // Yu
            ('Я', 'я'),
            ('я', 'Я'), // Ya
        ])
    }

    /// Japanese Hiragana ↔ Katakana equivalences.
    ///
    /// Allows matching between hiragana and katakana scripts.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSetChar;
    /// let japanese = SubstitutionSetChar::japanese_hiragana_katakana();
    ///
    /// // あ ↔ ア
    /// assert!(japanese.contains('あ', 'ア'));
    /// assert!(japanese.contains('ア', 'あ'));
    /// ```
    pub fn japanese_hiragana_katakana() -> Self {
        Self::from_pairs(&[
            // Basic hiragana/katakana pairs (first 20)
            ('あ', 'ア'),
            ('ア', 'あ'), // a
            ('い', 'イ'),
            ('イ', 'い'), // i
            ('う', 'ウ'),
            ('ウ', 'う'), // u
            ('え', 'エ'),
            ('エ', 'え'), // e
            ('お', 'オ'),
            ('オ', 'お'), // o
            ('か', 'カ'),
            ('カ', 'か'), // ka
            ('き', 'キ'),
            ('キ', 'き'), // ki
            ('く', 'ク'),
            ('ク', 'く'), // ku
            ('け', 'ケ'),
            ('ケ', 'け'), // ke
            ('こ', 'コ'),
            ('コ', 'こ'), // ko
            ('さ', 'サ'),
            ('サ', 'さ'), // sa
            ('し', 'シ'),
            ('シ', 'し'), // shi
            ('す', 'ス'),
            ('ス', 'す'), // su
            ('せ', 'セ'),
            ('セ', 'せ'), // se
            ('そ', 'ソ'),
            ('ソ', 'そ'), // so
            ('た', 'タ'),
            ('タ', 'た'), // ta
            ('ち', 'チ'),
            ('チ', 'ち'), // chi
            ('つ', 'ツ'),
            ('ツ', 'つ'), // tsu
            ('て', 'テ'),
            ('テ', 'て'), // te
            ('と', 'ト'),
            ('ト', 'と'), // to
                          // Add more as needed...
        ])
    }
}

impl Default for SubstitutionSetChar {
    /// Create an empty substitution set (equivalent to [`new()`](Self::new)).
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let set = SubstitutionSetChar::new();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn test_allow_and_contains() {
        let mut set = SubstitutionSetChar::new();

        set.allow('α', 'β');
        assert!(set.contains('α', 'β'));
        assert!(!set.contains('β', 'α')); // Not symmetric

        set.allow('β', 'α'); // Add reverse
        assert!(set.contains('β', 'α'));
    }

    #[test]
    fn test_unicode_characters() {
        let mut set = SubstitutionSetChar::new();

        // Greek
        set.allow('α', 'β');
        assert!(set.contains('α', 'β'));

        // Chinese
        set.allow('你', '好');
        assert!(set.contains('你', '好'));

        // Emoji (basic only, skin tone modifiers are multi-codepoint)
        set.allow('👋', '🤚');
        assert!(set.contains('👋', '🤚'));
    }

    #[test]
    fn test_from_pairs() {
        let set = SubstitutionSetChar::from_pairs(&[('é', 'e'), ('ñ', 'n'), ('ü', 'u')]);

        assert_eq!(set.len(), 3);
        assert!(set.contains('é', 'e'));
        assert!(set.contains('ñ', 'n'));
        assert!(set.contains('ü', 'u'));
    }

    #[test]
    fn test_clear() {
        let mut set = SubstitutionSetChar::from_pairs(&[('α', 'β'), ('γ', 'δ')]);
        assert_eq!(set.len(), 2);

        set.clear();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn test_diacritics_latin() {
        let diacritics = SubstitutionSetChar::diacritics_latin();

        assert!(!diacritics.is_empty());

        // é ↔ e
        assert!(diacritics.contains('é', 'e'));
        assert!(diacritics.contains('e', 'é'));

        // ñ ↔ n
        assert!(diacritics.contains('ñ', 'n'));
        assert!(diacritics.contains('n', 'ñ'));

        // ü ↔ u
        assert!(diacritics.contains('ü', 'u'));
        assert!(diacritics.contains('u', 'ü'));
    }

    #[test]
    fn test_greek_case_insensitive() {
        let greek = SubstitutionSetChar::greek_case_insensitive();

        assert!(!greek.is_empty());

        // Α ↔ α
        assert!(greek.contains('Α', 'α'));
        assert!(greek.contains('α', 'Α'));

        // Β ↔ β
        assert!(greek.contains('Β', 'β'));
        assert!(greek.contains('β', 'Β'));
    }

    #[test]
    fn test_cyrillic_case_insensitive() {
        let cyrillic = SubstitutionSetChar::cyrillic_case_insensitive();

        assert!(!cyrillic.is_empty());

        // А ↔ а
        assert!(cyrillic.contains('А', 'а'));
        assert!(cyrillic.contains('а', 'А'));

        // Я ↔ я
        assert!(cyrillic.contains('Я', 'я'));
        assert!(cyrillic.contains('я', 'Я'));
    }

    #[test]
    fn test_japanese_hiragana_katakana() {
        let japanese = SubstitutionSetChar::japanese_hiragana_katakana();

        assert!(!japanese.is_empty());

        // あ ↔ ア
        assert!(japanese.contains('あ', 'ア'));
        assert!(japanese.contains('ア', 'あ'));

        // か ↔ カ
        assert!(japanese.contains('か', 'カ'));
        assert!(japanese.contains('カ', 'か'));
    }

    #[test]
    fn test_with_capacity() {
        let set = SubstitutionSetChar::with_capacity(100);
        assert_eq!(set.len(), 0);
        // Capacity is internal, but shouldn't panic
    }

    #[test]
    fn test_duplicate_pairs() {
        let mut set = SubstitutionSetChar::new();

        set.allow('α', 'β');
        set.allow('α', 'β'); // Duplicate

        // HashSet deduplicates
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_clone() {
        let set1 = SubstitutionSetChar::diacritics_latin();
        let set2 = set1.clone();

        assert_eq!(set1.len(), set2.len());
        assert_eq!(set1, set2);
    }

    #[test]
    fn test_debug() {
        let set = SubstitutionSetChar::from_pairs(&[('α', 'β')]);
        let debug_str = format!("{:?}", set);
        assert!(debug_str.contains("SubstitutionSetChar"));
    }
}
