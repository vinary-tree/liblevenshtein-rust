//! Set of allowed character substitutions for fuzzy matching.
//!
//! This module provides [`SubstitutionSet`], which defines which character
//! substitutions are permitted during Levenshtein automaton matching. It's
//! used with the [`Restricted`](super::substitution_policy::Restricted)
//! policy to implement custom character similarity.
//!
//! ## Use Cases
//!
//! - **Phonetic matching**: Allow phonetically similar substitutions (f/ph, c/k)
//! - **Keyboard typos**: Allow adjacent key substitutions
//! - **OCR errors**: Allow visually similar character substitutions
//! - **Leetspeak**: Allow common leetspeak substitutions (3/e, @/a, 0/o)
//!
//! ## Example
//!
//! ```rust,ignore
//! use liblevenshtein::transducer::{SubstitutionSet, Transducer, Algorithm};
//! use liblevenshtein::prelude::*;
//!
//! // Create phonetic substitution set
//! let mut phonetic = SubstitutionSet::new();
//! phonetic.allow('f', 'p');  // For "ph" → "f" phonetic similarity
//! phonetic.allow('c', 'k');
//! phonetic.allow('s', 'z');
//!
//! // Or use a preset
//! let phonetic = SubstitutionSet::phonetic_basic();
//!
//! // Use with dictionary
//! let dict = DoubleArrayTrie::from_terms(vec!["phone", "cat", "dogs"]);
//! let transducer = Transducer::with_substitutions(
//!     dict,
//!     Algorithm::Standard,
//!     phonetic
//! );
//!
//! // Now "fone" matches "phone" with distance 1 (via f/ph substitution)
//! let results: Vec<_> = transducer.query("fone", 1).collect();
//! ```

use rustc_hash::{FxHashMap, FxHashSet};

/// Internal representation of SubstitutionSet storage.
///
/// Uses a hybrid approach (H3 optimization):
/// - **Small sets (≤4 pairs)**: Linear scan in Vec (~200-350ns, 1.2-1.9× faster than hash)
/// - **Large sets (>4 pairs)**: Hash lookup in FxHashSet (~370ns constant time)
///
/// Crossover analysis shows linear scan wins for tiny sets due to:
/// - No hashing overhead (~3-5ns saved)
/// - Better cache behavior (sequential access)
/// - Predictable branches (linear loop)
/// - Smaller memory footprint (2× less than hash)
///
/// See: docs/optimization/substitution-set/05-crossover-analysis.md
#[derive(Clone, Debug, PartialEq, Eq)]
enum SubstitutionSetImpl {
    /// Small set using linear scan (≤4 pairs).
    /// Faster for tiny sets due to no hash overhead.
    Small(Vec<(u8, u8)>),

    /// Large set using hash lookup (>4 pairs).
    /// Scales better for larger sets with O(1) lookup.
    Large(FxHashSet<(u8, u8)>),
}

/// Internal storage for multi-character substitution pairs.
///
/// Uses the same hybrid approach as single-character storage for consistency
/// and performance:
/// - **Small sets (≤4 pairs)**: Linear scan in Vec
/// - **Large sets (>4 pairs)**: Hash map for O(1) lookup
///
/// Strings are stored as `Box<str>` for memory efficiency:
/// - Immutable (no excess capacity like String)
/// - Minimal overhead (size of pointer + inline length)
/// - Cache-friendly for small strings
///
/// The large variant uses `FxHashMap<Box<str>, FxHashSet<Box<str>>>` to
/// efficiently handle one-to-many mappings (one source can map to multiple targets).
#[derive(Clone, Debug, PartialEq, Eq)]
enum MultiCharSubstitutionImpl {
    /// Small set using linear scan (≤4 pairs).
    /// Box<str> is used instead of String for lower memory overhead.
    Small(Vec<(Box<str>, Box<str>)>),

    /// Large set using hash lookup (>4 pairs).
    /// Maps source string to set of valid target strings for efficient one-to-many lookups.
    Large(FxHashMap<Box<str>, FxHashSet<Box<str>>>),
}

/// Set of allowed character substitutions.
///
/// A `SubstitutionSet` defines which character or string pairs can be
/// substituted for each other during fuzzy matching. Single-byte ASCII pairs
/// use compact byte storage; multi-byte UTF-8 or multi-character pairs use
/// string storage.
///
/// ## Performance
///
/// Uses a hybrid approach (H3 optimization):
/// - **Small sets (≤4 pairs)**: Linear scan (~200-350ns, 15-48% faster than hash)
/// - **Large sets (>4 pairs)**: Hash lookup (~370ns constant time)
///
/// Automatically upgrades from Vec to FxHashSet when exceeding threshold.
///
/// ## Memory
///
/// - **Small sets**: ~24 bytes + 2 bytes per pair (50-75% less than hash)
/// - **Large sets**: ~48 bytes + 24 bytes per pair (same as before)
///
/// ## Symmetry
///
/// Substitutions are **not symmetric by default**. If you want bidirectional
/// substitutions, you must add both directions explicitly:
///
/// ```rust
/// # use liblevenshtein::transducer::SubstitutionSet;
/// let mut set = SubstitutionSet::new();
/// set.allow('a', 'b');  // 'a' can be substituted with 'b'
/// set.allow('b', 'a');  // 'b' can be substituted with 'a' (symmetric)
/// ```
///
/// Most presets (like `phonetic_basic()`) include symmetric pairs where appropriate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubstitutionSet {
    /// Internal storage for single-character pairs using hybrid Vec/HashSet approach.
    inner: SubstitutionSetImpl,

    /// Internal storage for multi-character pairs using hybrid Vec/HashMap approach.
    /// This is separate from single-char storage to maintain performance for the common case.
    multi_char: MultiCharSubstitutionImpl,
}

impl SubstitutionSet {
    /// Threshold for upgrading from Vec to FxHashSet.
    ///
    /// Based on empirical crossover analysis showing linear scan wins for ≤4 pairs.
    /// See: docs/optimization/substitution-set/05-crossover-analysis.md
    const SMALL_SET_THRESHOLD: usize = 4;

    /// Create an empty substitution set.
    ///
    /// Starts with a small Vec-based representation for optimal performance
    /// with tiny sets (automatically upgrades to hash for larger sets).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let mut set = SubstitutionSet::new();
    /// set.allow('a', 'b');
    /// assert!(set.contains(b'a', b'b'));
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: SubstitutionSetImpl::Small(Vec::new()),
            multi_char: MultiCharSubstitutionImpl::Small(Vec::new()),
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
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let mut set = SubstitutionSet::with_capacity(50);
    /// // Add many pairs without reallocations
    /// for i in 0..50 {
    ///     set.allow_byte(b'a' + i % 26, b'A' + i % 26);
    /// }
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        // Choose optimal storage based on expected capacity for single-char pairs
        // Multi-char storage starts empty and grows as needed
        if capacity <= Self::SMALL_SET_THRESHOLD {
            Self {
                inner: SubstitutionSetImpl::Small(Vec::with_capacity(capacity)),
                multi_char: MultiCharSubstitutionImpl::Small(Vec::new()),
            }
        } else {
            Self {
                inner: SubstitutionSetImpl::Large(FxHashSet::with_capacity_and_hasher(
                    capacity,
                    Default::default(),
                )),
                multi_char: MultiCharSubstitutionImpl::Small(Vec::new()),
            }
        }
    }

    /// Allow substituting character `a` with character `b`.
    ///
    /// This single-character convenience API stores only ASCII scalar values in
    /// the compact byte table. Use [`allow_str()`](Self::allow_str) for
    /// multi-byte UTF-8 substitutions.
    ///
    /// # Parameters
    ///
    /// - `a`: Dictionary character (source)
    /// - `b`: Query character (target)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let mut set = SubstitutionSet::new();
    /// set.allow('f', 'p');  // 'f' in dict can match 'p' in query
    /// set.allow('p', 'h');  // 'p' in dict can match 'h' in query
    ///
    /// // This enables "phone" to match "fone" via f→p substitution
    /// ```
    #[inline]
    pub fn allow(&mut self, a: char, b: char) {
        if a.is_ascii() && b.is_ascii() {
            self.allow_byte(a as u8, b as u8);
        }
    }

    /// Allow substituting byte `a` with byte `b` (low-level API).
    ///
    /// This is the low-level API that works directly with raw bytes. It's
    /// slightly faster than [`allow()`](Self::allow) since it skips scalar-value
    /// checks. Use [`allow_str()`](Self::allow_str) for UTF-8 strings.
    ///
    /// # Parameters
    ///
    /// - `a`: Dictionary byte (source)
    /// - `b`: Query byte (target)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let mut set = SubstitutionSet::new();
    /// set.allow_byte(b'a', b'b');
    /// assert!(set.contains(b'a', b'b'));
    /// ```
    #[inline]
    pub fn allow_byte(&mut self, a: u8, b: u8) {
        match &mut self.inner {
            SubstitutionSetImpl::Small(vec) if vec.len() < Self::SMALL_SET_THRESHOLD => {
                // Still small - add to Vec if not duplicate
                if !vec.contains(&(a, b)) {
                    vec.push((a, b));
                }
            }
            SubstitutionSetImpl::Small(vec) => {
                // Exceeded threshold - upgrade to hash set
                let mut set =
                    FxHashSet::with_capacity_and_hasher(vec.len() + 1, Default::default());
                for &pair in vec.iter() {
                    set.insert(pair);
                }
                set.insert((a, b));
                self.inner = SubstitutionSetImpl::Large(set);
            }
            SubstitutionSetImpl::Large(set) => {
                // Already large - use hash set
                set.insert((a, b));
            }
        }
    }

    /// Check if substituting byte `a` with byte `b` is allowed.
    ///
    /// This is the hot-path method called during character matching.
    /// It's marked `#[inline]` for performance.
    ///
    /// # Parameters
    ///
    /// - `a`: Dictionary byte
    /// - `b`: Query byte
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
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let mut set = SubstitutionSet::new();
    /// set.allow('x', 'y');
    ///
    /// assert!(set.contains(b'x', b'y'));
    /// assert!(!set.contains(b'y', b'x'));  // Not symmetric
    /// ```
    #[inline]
    pub fn contains(&self, a: u8, b: u8) -> bool {
        match &self.inner {
            SubstitutionSetImpl::Small(vec) => {
                // Linear scan for small sets (faster for ≤4 pairs)
                vec.iter().any(|&(x, y)| x == a && y == b)
            }
            SubstitutionSetImpl::Large(set) => {
                // Hash lookup for large sets (faster for >4 pairs)
                set.contains(&(a, b))
            }
        }
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
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let set = SubstitutionSet::from_pairs(&[
    ///     ('f', 'p'), ('p', 'h'),  // phonetic: ph → f
    ///     ('c', 'k'), ('k', 'c'),  // symmetric
    /// ]);
    ///
    /// assert!(set.contains(b'f', b'p'));
    /// assert!(set.contains(b'c', b'k'));
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
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let mut set = SubstitutionSet::new();
    /// assert_eq!(set.len(), 0);
    ///
    /// set.allow('a', 'b');
    /// set.allow('c', 'd');
    /// assert_eq!(set.len(), 2);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        match &self.inner {
            SubstitutionSetImpl::Small(vec) => vec.len(),
            SubstitutionSetImpl::Large(set) => set.len(),
        }
    }

    /// Check if the substitution set is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let set = SubstitutionSet::new();
    /// assert!(set.is_empty());
    ///
    /// let phonetic = SubstitutionSet::phonetic_basic();
    /// assert!(!phonetic.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        match &self.inner {
            SubstitutionSetImpl::Small(vec) => vec.is_empty(),
            SubstitutionSetImpl::Large(set) => set.is_empty(),
        }
    }

    /// Clear all allowed substitutions.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let mut set = SubstitutionSet::phonetic_basic();
    /// assert!(!set.is_empty());
    ///
    /// set.clear();
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        // Clear single-char storage
        match &mut self.inner {
            SubstitutionSetImpl::Small(vec) => vec.clear(),
            SubstitutionSetImpl::Large(set) => set.clear(),
        }

        // Clear multi-char storage
        match &mut self.multi_char {
            MultiCharSubstitutionImpl::Small(vec) => vec.clear(),
            MultiCharSubstitutionImpl::Large(map) => map.clear(),
        }
    }

    // ========================================================================
    // Preset Builders
    // ========================================================================

    // Const arrays for optimized preset initialization (15-28% faster)
    // See: docs/optimization/substitution-set/02-hypothesis1-const-arrays.md

    /// Phonetic substitutions as a const array.
    const PHONETIC_PAIRS: &[(u8, u8)] = &[
        // f/ph equivalence (bidirectional)
        (b'f', b'p'),
        (b'p', b'f'),
        // c/k equivalence (bidirectional)
        (b'c', b'k'),
        (b'k', b'c'),
        // c/s equivalence (bidirectional)
        (b'c', b's'),
        (b's', b'c'),
        // s/z equivalence (bidirectional)
        (b's', b'z'),
        (b'z', b's'),
        // Common vowel confusions
        (b'a', b'e'),
        (b'e', b'a'),
        (b'i', b'y'),
        (b'y', b'i'),
        // Silent letters (allow omission)
        (b'h', b'\0'),
        (b'k', b'\0'),
    ];

    /// QWERTY keyboard substitutions as a const array.
    const KEYBOARD_PAIRS: &[(u8, u8)] = &[
        // Top row
        (b'q', b'w'),
        (b'w', b'q'),
        (b'w', b'e'),
        (b'e', b'w'),
        (b'e', b'r'),
        (b'r', b'e'),
        (b'r', b't'),
        (b't', b'r'),
        (b't', b'y'),
        (b'y', b't'),
        (b'y', b'u'),
        (b'u', b'y'),
        (b'u', b'i'),
        (b'i', b'u'),
        (b'i', b'o'),
        (b'o', b'i'),
        (b'o', b'p'),
        (b'p', b'o'),
        // Middle row
        (b'a', b's'),
        (b's', b'a'),
        (b's', b'd'),
        (b'd', b's'),
        (b'd', b'f'),
        (b'f', b'd'),
        (b'f', b'g'),
        (b'g', b'f'),
        (b'g', b'h'),
        (b'h', b'g'),
        (b'h', b'j'),
        (b'j', b'h'),
        (b'j', b'k'),
        (b'k', b'j'),
        (b'k', b'l'),
        (b'l', b'k'),
        // Bottom row
        (b'z', b'x'),
        (b'x', b'z'),
        (b'x', b'c'),
        (b'c', b'x'),
        (b'c', b'v'),
        (b'v', b'c'),
        (b'v', b'b'),
        (b'b', b'v'),
        (b'b', b'n'),
        (b'n', b'b'),
        (b'n', b'm'),
        (b'm', b'n'),
        // Vertical adjacencies (selected)
        (b'q', b'a'),
        (b'a', b'q'),
        (b'w', b's'),
        (b's', b'w'),
        (b'e', b'd'),
        (b'd', b'e'),
        (b'r', b'f'),
        (b'f', b'r'),
        (b't', b'g'),
        (b'g', b't'),
        (b'y', b'h'),
        (b'h', b'y'),
        (b'u', b'j'),
        (b'j', b'u'),
        (b'i', b'k'),
        (b'k', b'i'),
        (b'o', b'l'),
        (b'l', b'o'),
    ];

    /// Leetspeak substitutions as a const array.
    const LEET_PAIRS: &[(u8, u8)] = &[
        (b'e', b'3'),
        (b'3', b'e'),
        (b'a', b'@'),
        (b'@', b'a'),
        (b'a', b'4'),
        (b'4', b'a'),
        (b'o', b'0'),
        (b'0', b'o'),
        (b'i', b'1'),
        (b'1', b'i'),
        (b'l', b'1'),
        (b'1', b'l'),
        (b's', b'$'),
        (b'$', b's'),
        (b's', b'5'),
        (b'5', b's'),
        (b't', b'7'),
        (b'7', b't'),
        (b'b', b'8'),
        (b'8', b'b'),
        (b'g', b'9'),
        (b'9', b'g'),
    ];

    /// OCR-friendly substitutions as a const array.
    const OCR_PAIRS: &[(u8, u8)] = &[
        // 0/O confusion
        (b'0', b'O'),
        (b'O', b'0'),
        (b'0', b'o'),
        (b'o', b'0'),
        // 1/I/l confusion
        (b'1', b'I'),
        (b'I', b'1'),
        (b'1', b'l'),
        (b'l', b'1'),
        (b'I', b'l'),
        (b'l', b'I'),
        // 8/B confusion
        (b'8', b'B'),
        (b'B', b'8'),
        // 5/S confusion
        (b'5', b'S'),
        (b'S', b'5'),
        // 6/G confusion
        (b'6', b'G'),
        (b'G', b'6'),
        // 2/Z confusion
        (b'2', b'Z'),
        (b'Z', b'2'),
    ];

    /// Common phonetic equivalences for English.
    ///
    /// Includes bidirectional substitutions for phonetically similar sounds:
    /// - **f/ph**: phone ↔ fone
    /// - **c/k**: cat ↔ kat
    /// - **c/s**: cent ↔ sent
    /// - **s/z**: dogs ↔ dogz
    /// - **tion/shun**: nation ↔ nashun
    ///
    /// This is a basic set suitable for casual phonetic matching. For more
    /// sophisticated phonetic matching, consider implementing Soundex or
    /// Metaphone-inspired rules.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use liblevenshtein::transducer::{SubstitutionSet, Transducer, Algorithm};
    /// # use liblevenshtein::prelude::*;
    /// let phonetic = SubstitutionSet::phonetic_basic();
    ///
    /// let dict = DoubleArrayTrie::from_terms(vec!["phone", "cat"]);
    /// let transducer = Transducer::with_substitutions(
    ///     dict,
    ///     Algorithm::Standard,
    ///     phonetic
    /// );
    ///
    /// // "fone" matches "phone" (f ↔ ph)
    /// let results: Vec<_> = transducer.query("fone", 1).collect();
    /// assert!(results.contains(&"phone"));
    /// ```
    pub fn phonetic_basic() -> Self {
        let mut set = Self::with_capacity(Self::PHONETIC_PAIRS.len());
        for &(a, b) in Self::PHONETIC_PAIRS {
            set.allow_byte(a, b);
        }
        set
    }

    /// QWERTY keyboard proximity substitutions.
    ///
    /// Allows substitutions between adjacent keys on a QWERTY keyboard,
    /// useful for catching typos caused by hitting nearby keys.
    ///
    /// Includes horizontal and vertical adjacencies for common keys.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let keyboard = SubstitutionSet::keyboard_qwerty();
    ///
    /// // 'q' and 'w' are adjacent
    /// assert!(keyboard.contains(b'q', b'w') || keyboard.contains(b'w', b'q'));
    /// ```
    pub fn keyboard_qwerty() -> Self {
        let mut set = Self::with_capacity(Self::KEYBOARD_PAIRS.len());
        for &(a, b) in Self::KEYBOARD_PAIRS {
            set.allow_byte(a, b);
        }
        set
    }

    /// Common leetspeak substitutions.
    ///
    /// Allows common character-to-number substitutions used in leetspeak:
    /// - **3 ↔ e**: leet ↔ l33t
    /// - **@ ↔ a**: at ↔ @t
    /// - **0 ↔ o**: root ↔ r00t
    /// - **1 ↔ i/l**: lit ↔ l1t
    /// - **$ ↔ s**: cash ↔ ca$h
    /// - **7 ↔ t**: test ↔ 7es7
    ///
    /// Useful for matching leetspeak variations in usernames or informal text.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let leet = SubstitutionSet::leet_speak();
    ///
    /// // '3' can substitute for 'e'
    /// assert!(leet.contains(b'e', b'3'));
    /// assert!(leet.contains(b'3', b'e'));
    /// ```
    pub fn leet_speak() -> Self {
        let mut set = Self::with_capacity(Self::LEET_PAIRS.len());
        for &(a, b) in Self::LEET_PAIRS {
            set.allow_byte(a, b);
        }
        set
    }

    /// OCR-friendly substitutions for commonly confused characters.
    ///
    /// Includes substitutions for characters that are visually similar
    /// and often confused by OCR systems:
    /// - **0/O**: Zero vs letter O
    /// - **1/I/l**: One vs capital I vs lowercase L
    /// - **8/B**: Eight vs capital B
    /// - **5/S**: Five vs capital S
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let ocr = SubstitutionSet::ocr_friendly();
    ///
    /// // '0' and 'O' are visually similar
    /// assert!(ocr.contains(b'0', b'O') || ocr.contains(b'O', b'0'));
    /// ```
    pub fn ocr_friendly() -> Self {
        let mut set = Self::with_capacity(Self::OCR_PAIRS.len());
        for &(a, b) in Self::OCR_PAIRS {
            set.allow_byte(a, b);
        }
        set
    }

    /// Allow substituting string `a` with string `b`.
    ///
    /// This method supports multi-byte UTF-8 and multi-character substitutions
    /// for generalized operations.
    ///
    /// # Parameters
    ///
    /// - `a`: Dictionary string (source)
    /// - `b`: Query string (target)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let mut set = SubstitutionSet::new();
    /// set.allow_str("ph", "f");  // "ph" in dict can match "f" in query
    /// set.allow_str("ch", "k");  // "ch" in dict can match "k" in query
    ///
    /// assert!(set.contains_str(b"ph", b"f"));
    /// assert!(set.contains_str(b"ch", b"k"));
    /// ```
    ///
    /// # Note
    ///
    /// Single-byte ASCII pairs use optimized byte storage. Multi-byte UTF-8
    /// or multi-character strings use separate string storage.
    pub fn allow_str(&mut self, a: &str, b: &str) {
        // Reject empty strings - they're not valid substitution pairs
        if a.is_empty() || b.is_empty() {
            return;
        }

        // Fast path: Single-char pairs use optimized single-byte storage
        if a.len() == 1 && b.len() == 1 {
            if let (Some(a_char), Some(b_char)) = (a.chars().next(), b.chars().next()) {
                if a_char.is_ascii() && b_char.is_ascii() {
                    self.allow_byte(a_char as u8, b_char as u8);
                    return;
                }
            }
        }

        // Convert to Box<str> for memory efficiency
        let a_boxed: Box<str> = a.into();
        let b_boxed: Box<str> = b.into();

        match &mut self.multi_char {
            MultiCharSubstitutionImpl::Small(vec) if vec.len() < Self::SMALL_SET_THRESHOLD => {
                // Still small - add to Vec if not duplicate
                if !vec.iter().any(|(x, y)| x.as_ref() == a && y.as_ref() == b) {
                    vec.push((a_boxed, b_boxed));
                }
            }
            MultiCharSubstitutionImpl::Small(vec) => {
                // Exceeded threshold - upgrade to hash map
                let mut map: FxHashMap<Box<str>, FxHashSet<Box<str>>> =
                    FxHashMap::with_capacity_and_hasher(vec.len() + 1, Default::default());

                for (src, tgt) in vec.drain(..) {
                    map.entry(src)
                        .or_insert_with(|| {
                            FxHashSet::with_capacity_and_hasher(1, Default::default())
                        })
                        .insert(tgt);
                }

                // Insert new pair
                map.entry(a_boxed)
                    .or_insert_with(|| FxHashSet::with_capacity_and_hasher(1, Default::default()))
                    .insert(b_boxed);

                self.multi_char = MultiCharSubstitutionImpl::Large(map);
            }
            MultiCharSubstitutionImpl::Large(map) => {
                // Already large - use hash map
                map.entry(a_boxed)
                    .or_insert_with(|| FxHashSet::with_capacity_and_hasher(1, Default::default()))
                    .insert(b_boxed);
            }
        }
    }

    /// Check if substituting string `a` with string `b` is allowed.
    ///
    /// This method supports multi-character substitution checking.
    ///
    /// # Parameters
    ///
    /// - `a`: Dictionary bytes
    /// - `b`: Query bytes
    ///
    /// # Returns
    ///
    /// `true` if the substitution `a → b` is allowed, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let mut set = SubstitutionSet::new();
    /// set.allow_str("ph", "f");
    ///
    /// assert!(set.contains_str(b"ph", b"f"));
    /// assert!(!set.contains_str(b"ph", b"g"));
    /// ```
    #[inline]
    pub fn contains_str(&self, a: &[u8], b: &[u8]) -> bool {
        // Fast path: Single-byte pairs use optimized lookup
        if a.len() == 1 && b.len() == 1 {
            return self.contains(a[0], b[0]);
        }

        // Slow path: Multi-byte and multi-character pairs query string storage.
        let a_str = match std::str::from_utf8(a) {
            Ok(s) => s,
            _ => return false,
        };
        let b_str = match std::str::from_utf8(b) {
            Ok(s) => s,
            _ => return false,
        };

        match &self.multi_char {
            MultiCharSubstitutionImpl::Small(vec) => {
                // Linear scan for small sets
                vec.iter()
                    .any(|(x, y)| x.as_ref() == a_str && y.as_ref() == b_str)
            }
            MultiCharSubstitutionImpl::Large(map) => {
                // Hash lookup for large sets
                map.get(a_str)
                    .map(|targets| targets.contains(b_str))
                    .unwrap_or(false)
            }
        }
    }

    /// Check if a source exists in any substitution pair, regardless of target.
    ///
    /// This method is useful for checking if an operation COULD apply to a source
    /// character before knowing the target character (e.g., when entering a split state).
    ///
    /// # Parameters
    ///
    /// - `source`: Source bytes to check
    ///
    /// # Returns
    ///
    /// `true` if `source` appears as a source in ANY substitution pair, `false` otherwise.
    ///
    /// # Performance
    ///
    /// - Single-byte: O(|set|) linear scan for both Small and Large variants
    /// - Multi-byte: O(1) for Large variant (HashMap key lookup), O(|set|) for Small
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let mut set = SubstitutionSet::new();
    /// set.allow_str("k", "ch");
    /// set.allow_str("t", "th");
    ///
    /// assert!(set.has_source(b"k"));   // k→ch exists
    /// assert!(set.has_source(b"t"));   // t→th exists
    /// assert!(!set.has_source(b"a"));  // a→?? doesn't exist
    /// ```
    #[inline]
    pub fn has_source(&self, source: &[u8]) -> bool {
        // Convert source to string for multi-char storage check
        let src_str = match std::str::from_utf8(source) {
            Ok(s) => s,
            _ => return false,
        };

        // Check single-byte storage if source is single character
        if source.len() == 1 {
            let src_byte = source[0];
            let found_in_single = match &self.inner {
                SubstitutionSetImpl::Small(vec) => vec.iter().any(|(s, _)| *s == src_byte),
                SubstitutionSetImpl::Large(set) => {
                    // FxHashSet doesn't support partial key lookup, so iterate
                    set.iter().any(|(s, _)| *s == src_byte)
                }
            };

            // If found in single-byte storage, return true immediately
            if found_in_single {
                return true;
            }

            // IMPORTANT: Also check multi-char storage!
            // Pairs like ("k", "ch") are stored in multi-char (not single-byte)
            // because target "ch" has length > 1
        }

        // Check multi-char storage (for all sources, regardless of length)
        match &self.multi_char {
            MultiCharSubstitutionImpl::Small(vec) => vec.iter().any(|(s, _)| s.as_ref() == src_str),
            MultiCharSubstitutionImpl::Large(map) => {
                // HashMap has sources as keys - O(1) lookup!
                map.contains_key(src_str)
            }
        }
    }

    /// Check if there exists a target for the given source that starts with the specified character.
    ///
    /// This is used for phonetic split operations to validate that the current input character
    /// matches the first character of a potential split target.
    ///
    /// # Parameters
    ///
    /// - `source`: The source bytes to look up
    /// - `first_char`: The character to check against the first character of targets
    ///
    /// # Returns
    ///
    /// `true` if at least one target for this source starts with `first_char`, `false` otherwise.
    #[inline]
    pub fn has_target_starting_with(&self, source: &[u8], first_char: char) -> bool {
        // Convert source to string for multi-char storage check
        let src_str = match std::str::from_utf8(source) {
            Ok(s) => s,
            _ => return false,
        };

        // Check single-byte storage if source is single character
        if source.len() == 1 {
            let src_byte = source[0];

            // Check single-byte storage
            // Note: Each entry is (source_byte, target_byte) - single byte pairs
            let found_in_single = match &self.inner {
                SubstitutionSetImpl::Small(vec) => {
                    vec.iter()
                        .filter(|(s, _)| *s == src_byte)
                        .any(|(_, target)| {
                            // Single-byte target - convert to char and check
                            let target_char = *target as char;
                            target_char == first_char
                        })
                }
                SubstitutionSetImpl::Large(set) => {
                    set.iter()
                        .filter(|(s, _)| *s == src_byte)
                        .any(|(_, target)| {
                            let target_char = *target as char;
                            target_char == first_char
                        })
                }
            };

            if found_in_single {
                return true;
            }

            // IMPORTANT: Also check multi-char storage!
            // Pairs like ("k", "ch") are stored in multi-char storage
        }

        // Check multi-char storage
        match &self.multi_char {
            MultiCharSubstitutionImpl::Small(vec) => {
                // Small variant stores individual (source, target) pairs
                vec.iter()
                    .filter(|(s, _)| s.as_ref() == src_str)
                    .any(|(_, target)| {
                        // target is a Box<str>, check its first char
                        target.chars().next() == Some(first_char)
                    })
            }
            MultiCharSubstitutionImpl::Large(map) => {
                // Large variant stores source → Set<target> mappings
                map.get(src_str)
                    .map(|target_set| {
                        // target_set is FxHashSet<Box<str>>
                        target_set
                            .iter()
                            .any(|target| target.chars().next() == Some(first_char))
                    })
                    .unwrap_or(false)
            }
        }
    }

    /// Build a substitution set from string pairs.
    ///
    /// This supports multi-character substitution pairs.
    ///
    /// # Parameters
    ///
    /// - `pairs`: Slice of (source, target) string pairs
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let set = SubstitutionSet::from_str_pairs(&[
    ///     ("ph", "f"),   // phonetic: ph → f
    ///     ("ch", "k"),   // phonetic: ch → k
    ///     ("ght", "t"),  // silent letters: ght → t
    /// ]);
    ///
    /// assert!(set.contains_str(b"ph", b"f"));
    /// assert!(set.contains_str(b"ch", b"k"));
    /// ```
    pub fn from_str_pairs(pairs: &[(&str, &str)]) -> Self {
        let mut set = Self::with_capacity(pairs.len());
        for &(a, b) in pairs {
            set.allow_str(a, b);
        }
        set
    }
}

impl Default for SubstitutionSet {
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
        let set = SubstitutionSet::new();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn test_storage_layout_size_budget() {
        assert!(std::mem::size_of::<SubstitutionSet>() <= 128);
    }

    #[test]
    fn test_allow_and_contains() {
        let mut set = SubstitutionSet::new();

        set.allow('a', 'b');
        assert!(set.contains(b'a', b'b'));
        assert!(!set.contains(b'b', b'a')); // Not symmetric

        set.allow('b', 'a'); // Add reverse
        assert!(set.contains(b'b', b'a'));
    }

    #[test]
    fn test_allow_byte() {
        let mut set = SubstitutionSet::new();

        set.allow_byte(b'x', b'y');
        assert!(set.contains(b'x', b'y'));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_from_pairs() {
        let set = SubstitutionSet::from_pairs(&[('a', 'b'), ('c', 'd'), ('e', 'f')]);

        assert_eq!(set.len(), 3);
        assert!(set.contains(b'a', b'b'));
        assert!(set.contains(b'c', b'd'));
        assert!(set.contains(b'e', b'f'));
    }

    #[test]
    fn test_clear() {
        let mut set = SubstitutionSet::from_pairs(&[('a', 'b'), ('c', 'd')]);
        assert_eq!(set.len(), 2);

        set.clear();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn test_phonetic_basic() {
        let phonetic = SubstitutionSet::phonetic_basic();

        assert!(!phonetic.is_empty());

        // f/ph equivalence
        assert!(phonetic.contains(b'f', b'p'));
        assert!(phonetic.contains(b'p', b'f'));

        // c/k equivalence
        assert!(phonetic.contains(b'c', b'k'));
        assert!(phonetic.contains(b'k', b'c'));

        // s/z equivalence
        assert!(phonetic.contains(b's', b'z'));
        assert!(phonetic.contains(b'z', b's'));
    }

    #[test]
    fn test_keyboard_qwerty() {
        let keyboard = SubstitutionSet::keyboard_qwerty();

        assert!(!keyboard.is_empty());

        // Adjacent keys on top row
        assert!(keyboard.contains(b'q', b'w'));
        assert!(keyboard.contains(b'w', b'q'));

        // Adjacent keys on middle row
        assert!(keyboard.contains(b'a', b's'));
        assert!(keyboard.contains(b's', b'a'));

        // Vertical adjacency
        assert!(keyboard.contains(b'q', b'a'));
        assert!(keyboard.contains(b'a', b'q'));
    }

    #[test]
    fn test_leet_speak() {
        let leet = SubstitutionSet::leet_speak();

        assert!(!leet.is_empty());

        // 3/e
        assert!(leet.contains(b'e', b'3'));
        assert!(leet.contains(b'3', b'e'));

        // @/a
        assert!(leet.contains(b'a', b'@'));
        assert!(leet.contains(b'@', b'a'));

        // 0/o
        assert!(leet.contains(b'o', b'0'));
        assert!(leet.contains(b'0', b'o'));
    }

    #[test]
    fn test_ocr_friendly() {
        let ocr = SubstitutionSet::ocr_friendly();

        assert!(!ocr.is_empty());

        // 0/O confusion
        assert!(ocr.contains(b'0', b'O'));
        assert!(ocr.contains(b'O', b'0'));

        // 1/I/l confusion
        assert!(ocr.contains(b'1', b'I'));
        assert!(ocr.contains(b'1', b'l'));
        assert!(ocr.contains(b'I', b'l'));
    }

    #[test]
    fn test_with_capacity() {
        let set = SubstitutionSet::with_capacity(100);
        assert_eq!(set.len(), 0);
        // Capacity is internal, but shouldn't panic
    }

    #[test]
    fn test_non_ascii_ignored() {
        let mut set = SubstitutionSet::new();

        // Non-ASCII characters should be ignored
        set.allow('α', 'β'); // Greek letters
        set.allow('你', '好'); // Chinese characters

        assert_eq!(set.len(), 0); // Nothing added
    }

    #[test]
    fn test_duplicate_pairs() {
        let mut set = SubstitutionSet::new();

        set.allow('a', 'b');
        set.allow('a', 'b'); // Duplicate

        // HashSet deduplicates
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_clone() {
        let set1 = SubstitutionSet::phonetic_basic();
        let set2 = set1.clone();

        assert_eq!(set1.len(), set2.len());
        assert_eq!(set1, set2);
    }

    #[test]
    fn test_debug() {
        let set = SubstitutionSet::from_pairs(&[('a', 'b')]);
        let debug_str = format!("{:?}", set);
        assert!(debug_str.contains("SubstitutionSet"));
    }

    // ========================================================================
    // Multi-Character Tests
    // ========================================================================

    #[test]
    fn test_multi_char_basic() {
        let mut set = SubstitutionSet::new();

        // Add multi-character substitutions
        set.allow_str("ph", "f");
        set.allow_str("ch", "k");

        // Test contains_str
        assert!(set.contains_str(b"ph", b"f"));
        assert!(set.contains_str(b"ch", b"k"));

        // Not symmetric
        assert!(!set.contains_str(b"f", b"ph"));
        assert!(!set.contains_str(b"k", b"ch"));

        // Test from_str_pairs
        let set2 = SubstitutionSet::from_str_pairs(&[("ph", "f"), ("ch", "k")]);

        assert!(set2.contains_str(b"ph", b"f"));
        assert!(set2.contains_str(b"ch", b"k"));
    }

    #[test]
    fn test_multi_char_digraphs() {
        let mut set = SubstitutionSet::new();

        // English phonetic digraphs (ASCII only)
        set.allow_str("ph", "f");
        set.allow_str("ch", "k"); // Use ASCII instead of ç
        set.allow_str("sh", "$");
        set.allow_str("th", "+");

        assert!(set.contains_str(b"ph", b"f"));
        assert!(set.contains_str(b"ch", b"k"));
        assert!(set.contains_str(b"sh", b"$"));
        assert!(set.contains_str(b"th", b"+"));

        // Test non-matches
        assert!(!set.contains_str(b"ph", b"g"));
        assert!(!set.contains_str(b"xy", b"z"));
    }

    #[test]
    fn test_multi_char_trigraphs() {
        let mut set = SubstitutionSet::new();

        // Trigraphs (ASCII only)
        set.allow_str("eau", "o"); // Use ASCII instead of ö
        set.allow_str("ght", "t");

        assert!(set.contains_str(b"eau", b"o"));
        assert!(set.contains_str(b"ght", b"t"));
    }

    #[test]
    fn test_multi_char_one_to_many() {
        let mut set = SubstitutionSet::new();

        // One source can map to multiple targets
        set.allow_str("c", "k");
        set.allow_str("c", "s");

        assert!(set.contains_str(b"c", b"k"));
        assert!(set.contains_str(b"c", b"s"));
    }

    #[test]
    fn test_multi_char_mixed_with_single() {
        let mut set = SubstitutionSet::new();

        // Mix single and multi-character substitutions
        set.allow('f', 'p'); // Single-char
        set.allow_str("ph", "f"); // Multi-char

        // Both should work
        assert!(set.contains(b'f', b'p'));
        assert!(set.contains_str(b"ph", b"f"));

        // Single-char should use fast path
        assert!(set.contains_str(b"f", b"p"));
    }

    #[test]
    fn test_multi_char_upgrade_to_hashmap() {
        let mut set = SubstitutionSet::new();

        // Add more than SMALL_SET_THRESHOLD (4) multi-char pairs
        set.allow_str("ph", "f");
        set.allow_str("ch", "k");
        set.allow_str("sh", "$");
        set.allow_str("th", "+");
        set.allow_str("wh", "w"); // 5th pair - should trigger upgrade

        // All should still be accessible
        assert!(set.contains_str(b"ph", b"f"));
        assert!(set.contains_str(b"ch", b"k"));
        assert!(set.contains_str(b"sh", b"$"));
        assert!(set.contains_str(b"th", b"+"));
        assert!(set.contains_str(b"wh", b"w"));
    }

    #[test]
    fn test_multi_char_duplicates() {
        let mut set = SubstitutionSet::new();

        set.allow_str("ph", "f");
        set.allow_str("ph", "f"); // Duplicate

        // Should be deduplicated (implementation uses FxHashSet for targets)
        assert!(set.contains_str(b"ph", b"f"));
    }

    #[test]
    fn test_multi_char_clear() {
        let mut set = SubstitutionSet::new();

        set.allow_str("ph", "f");
        set.allow_str("ch", "k");
        assert!(set.contains_str(b"ph", b"f"));

        set.clear();

        // Multi-char should be cleared
        assert!(!set.contains_str(b"ph", b"f"));
        assert!(!set.contains_str(b"ch", b"k"));
    }

    #[test]
    fn test_multi_char_utf8_substitutions() {
        let mut set = SubstitutionSet::new();

        set.allow_str("α", "β");
        set.allow_str("你", "好");
        set.allow_str("é", "e");
        set.allow_str("sch", "š");

        assert!(set.contains_str("α".as_bytes(), "β".as_bytes()));
        assert!(set.contains_str("你".as_bytes(), "好".as_bytes()));
        assert!(set.contains_str("é".as_bytes(), b"e"));
        assert!(set.contains_str(b"sch", "š".as_bytes()));

        assert!(set.has_source("α".as_bytes()));
        assert!(set.has_source("你".as_bytes()));
        assert!(set.has_source("é".as_bytes()));
        assert!(set.has_target_starting_with("sch".as_bytes(), 'š'));

        assert!(!set.contains_str("α".as_bytes(), "γ".as_bytes()));
        assert!(!set.has_target_starting_with("sch".as_bytes(), 's'));
    }

    #[test]
    fn test_multi_char_empty_strings() {
        let mut set = SubstitutionSet::new();

        // Empty strings should not cause panics and are handled gracefully
        set.allow_str("", ""); // Neither stored in single-char nor multi-char
        set.allow_str("a", ""); // Only 'a' is single-char, empty target is not stored
        set.allow_str("", "b"); // Empty source means nothing to match

        // All lookups should return false for empty strings
        // (they're not valid substitution targets)
        assert!(!set.contains_str(b"", b""));
        assert!(!set.contains_str(b"a", b""));
        assert!(!set.contains_str(b"", b"b"));

        // Verify the set is still empty (no valid pairs stored)
        assert!(set.is_empty());
    }

    #[test]
    fn test_has_source_single_char() {
        let mut set = SubstitutionSet::new();

        // Add single-character substitutions
        set.allow('f', 'p');
        set.allow('k', 'c');

        // Test has_source for single chars
        assert!(set.has_source(b"f"));
        assert!(set.has_source(b"k"));
        assert!(!set.has_source(b"a"));
        assert!(!set.has_source(b"p")); // 'p' is a target, not a source
    }

    #[test]
    fn test_has_source_multi_char() {
        let mut set = SubstitutionSet::new();

        // Add multi-character substitutions
        set.allow_str("ph", "f");
        set.allow_str("ch", "k");
        set.allow_str("k", "ch");

        // Test has_source for multi-char
        assert!(set.has_source(b"ph"));
        assert!(set.has_source(b"ch"));
        assert!(set.has_source(b"k"));
        assert!(!set.has_source(b"f")); // 'f' is a target, not a source
        assert!(!set.has_source(b"th")); // 'th' doesn't exist as source
    }

    #[test]
    fn test_has_source_mixed() {
        let mut set = SubstitutionSet::new();

        // Mix single and multi-character substitutions
        set.allow('f', 'p');
        set.allow_str("ph", "f");
        set.allow_str("k", "ch");

        // Test both single and multi-char sources
        assert!(set.has_source(b"f")); // Single-char source
        assert!(set.has_source(b"ph")); // Multi-char source
        assert!(set.has_source(b"k")); // Single-char source
        assert!(!set.has_source(b"p")); // Target, not source
        assert!(!set.has_source(b"ch")); // Target, not source
    }

    #[test]
    fn test_has_source_upgrade_to_large() {
        let mut set = SubstitutionSet::new();

        // Add enough pairs to trigger upgrade to Large variant (> 4 pairs)
        set.allow_str("a", "b");
        set.allow_str("c", "d");
        set.allow_str("e", "f");
        set.allow_str("g", "h");
        set.allow_str("i", "j"); // 5th pair - triggers upgrade

        // All sources should still be found after upgrade
        assert!(set.has_source(b"a"));
        assert!(set.has_source(b"c"));
        assert!(set.has_source(b"e"));
        assert!(set.has_source(b"g"));
        assert!(set.has_source(b"i"));
        assert!(!set.has_source(b"b")); // Target, not source
    }

    #[test]
    fn test_has_source_empty() {
        let set = SubstitutionSet::new();

        // Empty set should return false for any source
        assert!(!set.has_source(b"a"));
        assert!(!set.has_source(b"ph"));
    }

    #[test]
    fn test_has_source_phonetic_example() {
        let mut set = SubstitutionSet::new();

        // Realistic phonetic split operations (⟨1,2⟩)
        set.allow_str("k", "ch");
        set.allow_str("s", "sh");
        set.allow_str("f", "ph");
        set.allow_str("t", "th");

        // Test which single characters can be split
        assert!(set.has_source(b"k")); // k→ch
        assert!(set.has_source(b"s")); // s→sh
        assert!(set.has_source(b"f")); // f→ph
        assert!(set.has_source(b"t")); // t→th
        assert!(!set.has_source(b"a")); // a cannot be split
        assert!(!set.has_source(b"e")); // e cannot be split
    }

    #[test]
    fn test_has_target_starting_with_single_char_to_multi_char() {
        let mut set = SubstitutionSet::new();

        // Phonetic split operations: single char → multi char
        set.allow_str("k", "ch");
        set.allow_str("s", "sh");
        set.allow_str("t", "th");

        // Test k→ch (should start with 'c')
        assert!(set.has_target_starting_with(b"k", 'c'));
        assert!(!set.has_target_starting_with(b"k", 's'));
        assert!(!set.has_target_starting_with(b"k", 't'));

        // Test s→sh (should start with 's')
        assert!(set.has_target_starting_with(b"s", 's'));
        assert!(!set.has_target_starting_with(b"s", 'c'));

        // Test t→th (should start with 't')
        assert!(set.has_target_starting_with(b"t", 't'));
        assert!(!set.has_target_starting_with(b"t", 'c'));
    }

    #[test]
    fn test_has_target_starting_with_reverse_operations() {
        let mut set = SubstitutionSet::new();

        // Reverse phonetic operations: multi char → single char
        set.allow_str("ch", "k");
        set.allow_str("sh", "s");
        set.allow_str("th", "t");

        // Test ch→k (should start with 'k')
        assert!(set.has_target_starting_with(b"ch", 'k'));
        assert!(!set.has_target_starting_with(b"ch", 's'));

        // Test sh→s (should start with 's')
        assert!(set.has_target_starting_with(b"sh", 's'));
        assert!(!set.has_target_starting_with(b"sh", 'k'));
    }

    #[test]
    fn test_has_target_starting_with_multiple_targets() {
        let mut set = SubstitutionSet::new();

        // One source with multiple targets
        set.allow_str("a", "b");
        set.allow_str("a", "c");
        set.allow_str("a", "d");

        // Should match any of the first characters
        assert!(set.has_target_starting_with(b"a", 'b'));
        assert!(set.has_target_starting_with(b"a", 'c'));
        assert!(set.has_target_starting_with(b"a", 'd'));
        assert!(!set.has_target_starting_with(b"a", 'e'));
    }

    #[test]
    fn test_has_target_starting_with_nonexistent_source() {
        let mut set = SubstitutionSet::new();
        set.allow_str("k", "ch");

        // Source doesn't exist
        assert!(!set.has_target_starting_with(b"x", 'c'));
        assert!(!set.has_target_starting_with(b"y", 'y'));
    }

    #[test]
    fn test_has_target_starting_with_empty_set() {
        let set = SubstitutionSet::new();

        // Empty set
        assert!(!set.has_target_starting_with(b"k", 'c'));
        assert!(!set.has_target_starting_with(b"a", 'b'));
    }

    #[test]
    fn test_has_target_starting_with_realistic_phonetic() {
        let mut set = SubstitutionSet::new();

        // Realistic bidirectional phonetic operations
        // Splits: k→ch, s→sh, f→ph, t→th
        set.allow_str("k", "ch");
        set.allow_str("s", "sh");
        set.allow_str("f", "ph");
        set.allow_str("t", "th");

        // Merges: ch→k, sh→s, ph→f, th→t
        set.allow_str("ch", "k");
        set.allow_str("sh", "s");
        set.allow_str("ph", "f");
        set.allow_str("th", "t");

        // Test split targets (input char should match first char of target)
        assert!(set.has_target_starting_with(b"k", 'c')); // k→ch, need 'c' then 'h'
        assert!(set.has_target_starting_with(b"t", 't')); // t→th, need 't' then 'h'
        assert!(!set.has_target_starting_with(b"t", 'a')); // t→th, 'a' not valid first char
        assert!(!set.has_target_starting_with(b"k", 'a')); // k→ch, 'a' not valid first char

        // Test merge targets
        assert!(set.has_target_starting_with(b"ch", 'k')); // ch→k
        assert!(set.has_target_starting_with(b"th", 't')); // th→t
    }
}
