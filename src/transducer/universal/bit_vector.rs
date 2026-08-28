//! Bit Vector Encoding for Universal Levenshtein Automata
//!
//! This module implements characteristic vectors and word-pair encoding from
//! Petar Mitankin's 2005 thesis "Universal Levenshtein Automata - Building and Properties".
//!
//! # Theory Background
//!
//! ## Characteristic Vector β (Definition 9, page 17)
//!
//! The characteristic vector β(x, w) encodes which positions in word w match character x:
//!
//! ```text
//! β(x, w₁w₂...wₚ) = b₁b₂...bₚ where bᵢ = {1 if x = wᵢ, 0 otherwise}
//! ```
//!
//! **Examples**:
//! ```text
//! β('a', "banana") = "010101"
//! β('b', "banana") = "100000"
//! β('n', "banana") = "001010"
//! ```
//!
//! ## Word-Pair Encoding hₙ (Page 51)
//!
//! To make the universal automaton work with concrete words, we encode the pair (w, x) as a
//! sequence of characteristic vectors. This requires:
//!
//! 1. **Padding** (Definition 16, page 50): Prepend n special symbols $ to w:
//!    ```text
//!    w₋ₙ₊₁ = w₋ₙ₊₂ = ... = w₀ = $
//!    ```
//!
//! 2. **Relevant Subword** sₙ(w, i): Extract a window around position i:
//!    ```text
//!    sₙ(w, i) = w_{i-n} w_{i-n+1} ... w_v where v = min(|w|, i + n + 1)
//!    ```
//!
//! 3. **Encoding**: For input word x = x₁x₂...xₜ:
//!    ```text
//!    hₙ(w, x) = β(x₁, sₙ(w,1)) β(x₂, sₙ(w,2)) ... β(xₜ, sₙ(w,t))
//!    ```
//!
//! ## Key Property (Proposition 19, pages 52-56)
//!
//! ```text
//! x ∈ L^χ_Lev(n, w) ⇔ hₙ(w, x) ∈ L(A^{∀,χ}_n)
//! ```
//!
//! The universal automaton accepts the encoding hₙ(w, x) if and only if x is within distance n
//! from w!
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::transducer::universal::{characteristic_vector, encode_word_pair};
//!
//! // Characteristic vector
//! let cv = characteristic_vector('a', "banana");
//! assert_eq!(cv.len(), 6);
//! assert!(!cv.is_match(0)); // `b` does not match `a`.
//! assert!(cv.is_match(1));  // The first `a` is at index 1.
//! assert!(!cv.is_match(2)); // `n` does not match `a`.
//!
//! // Word-pair encoding
//! let encoding = encode_word_pair("abcabb", "dacab", 3)
//!     .expect("the input length is within the encoding domain");
//! assert_eq!(encoding.len(), 5);  // One bit vector per character in "dacab"
//! ```

use std::fmt;

/// Characteristic vector β(x, w) representing which positions in w match character x.
///
/// # Theory (Definition 9, page 17)
///
/// ```text
/// β(x, w₁w₂...wₚ) = b₁b₂...bₚ where bᵢ = {1 if x = wᵢ, 0 otherwise}
/// ```
///
/// # Implementation
///
/// Bits are stored as a `Vec<bool>` for simplicity and clarity. For production use, this could
/// be optimized to use a bit-packed representation (e.g., `Vec<u64>` or `bitvec` crate).
///
/// # Example
///
/// ```rust
/// use liblevenshtein::transducer::universal::CharacteristicVector;
///
/// let cv = CharacteristicVector::new('a', "banana");
/// assert_eq!(cv.len(), 6);
/// assert_eq!(cv.to_string(), "010101");
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CharacteristicVector {
    /// Bit vector: `bits[i] = true` if the character matches at position `i`.
    bits: Vec<bool>,
}

impl CharacteristicVector {
    /// Creates a new characteristic vector β(x, w).
    ///
    /// # Theory (Definition 9, page 17)
    ///
    /// Computes:
    /// ```text
    /// β(x, w₁w₂...wₚ) = b₁b₂...bₚ where bᵢ = {1 if x = wᵢ, 0 otherwise}
    /// ```
    ///
    /// # Arguments
    ///
    /// * `character` - The character x to match against
    /// * `word` - The word w to check for matches
    ///
    /// # Returns
    ///
    /// A characteristic vector where bit i is 1 if x = wᵢ, 0 otherwise
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::universal::CharacteristicVector;
    ///
    /// let cv = CharacteristicVector::new('a', "banana");
    /// assert_eq!(cv.to_string(), "010101");
    /// //                            ^ ^ ^
    /// //                            | | positions 3 and 5 match `a`
    /// //                            | position 1 matches `a`
    /// //                            positions: b a n a n a
    /// ```
    pub fn new(character: char, word: &str) -> Self {
        let bits = word.chars().map(|c| c == character).collect();
        Self { bits }
    }

    /// Returns the length of the characteristic vector (number of bits).
    ///
    /// This equals the length of the word used to create the vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Returns true if the characteristic vector is empty (length 0).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Returns true if the bit at the given position is 1 (match).
    ///
    /// # Arguments
    ///
    /// * `position` - The position to check (0-indexed)
    ///
    /// # Returns
    ///
    /// * `true` if the character matched at this position (bit = 1)
    /// * `false` if the character did not match (bit = 0)
    ///
    /// # Panics
    ///
    /// Panics if position >= len()
    #[inline]
    pub fn is_match(&self, position: usize) -> bool {
        self.bits[position]
    }

    /// Returns an iterator over the bits in the characteristic vector.
    ///
    /// # Returns
    ///
    /// Iterator yielding `true` for 1 bits, `false` for 0 bits
    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        self.bits.iter().copied()
    }

    /// Returns true if the first bit is 1 (prefix match).
    ///
    /// Equivalent to checking if "1 < b" in the thesis notation (Definition 7, page 14).
    ///
    /// # Returns
    ///
    /// * `true` if the bit vector starts with 1
    /// * `false` otherwise (empty or starts with 0)
    #[inline]
    pub fn starts_with_one(&self) -> bool {
        self.bits.first().copied().unwrap_or(false)
    }

    /// Returns the position of the first 1 bit, if any.
    ///
    /// Used to implement μz[bz = 1] from the thesis (Definition 7, page 14).
    ///
    /// # Returns
    ///
    /// * `Some(position)` of the first 1 bit
    /// * `None` if all bits are 0
    pub fn first_match(&self) -> Option<usize> {
        self.bits.iter().position(|&bit| bit)
    }

    /// Returns a slice of the bits starting at the given position.
    ///
    /// Used for checking patterns like "01 < b" or "00 < b" in transition functions.
    ///
    /// # Arguments
    ///
    /// * `start` - Starting position (0-indexed)
    ///
    /// # Returns
    ///
    /// A new CharacteristicVector containing bits from start to the end
    pub fn suffix(&self, start: usize) -> Self {
        Self {
            bits: self.bits[start..].to_vec(),
        }
    }

    /// Returns true if all bits are 0.
    ///
    /// Checks if the bit vector is "0ᵏ" in thesis notation.
    pub fn is_all_zeros(&self) -> bool {
        self.bits.iter().all(|&bit| !bit)
    }

    /// Returns true if the bit vector starts with the given pattern.
    ///
    /// # Arguments
    ///
    /// * `pattern` - A slice of boolean values to match
    ///
    /// # Returns
    ///
    /// `true` if the first len(pattern) bits match the pattern
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::universal::CharacteristicVector;
    ///
    /// let cv = CharacteristicVector::new('a', "banana");
    /// assert!(cv.starts_with(&[false, true]));
    /// assert!(!cv.starts_with(&[true, false]));
    /// ```
    pub fn starts_with(&self, pattern: &[bool]) -> bool {
        if pattern.len() > self.bits.len() {
            return false;
        }
        self.bits[..pattern.len()] == *pattern
    }
}

impl fmt::Display for CharacteristicVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for bit in &self.bits {
            write!(f, "{}", if *bit { '1' } else { '0' })?;
        }
        Ok(())
    }
}

/// Creates a characteristic vector β(x, w).
///
/// Convenience function that calls `CharacteristicVector::new`.
///
/// # Theory (Definition 9, page 17)
///
/// ```text
/// β(x, w₁w₂...wₚ) = b₁b₂...bₚ where bᵢ = {1 if x = wᵢ, 0 otherwise}
/// ```
///
/// # Arguments
///
/// * `character` - The character x to match against
/// * `word` - The word w to check for matches
///
/// # Returns
///
/// A characteristic vector encoding which positions in w match x
///
/// # Example
///
/// ```rust
/// use liblevenshtein::transducer::universal::characteristic_vector;
///
/// let cv = characteristic_vector('a', "banana");
/// assert_eq!(cv.to_string(), "010101");
/// ```
pub fn characteristic_vector(character: char, word: &str) -> CharacteristicVector {
    CharacteristicVector::new(character, word)
}

/// Special padding character used in word-pair encoding.
///
/// From Definition 16 (page 50): $ ∉ Σ is used to pad the beginning of word w.
const PADDING_CHAR: char = '$';

/// Computes the relevant subword sₙ(w, i) for position i.
///
/// # Theory (Page 51)
///
/// ```text
/// sₙ(w, i) = w_{i-n} w_{i-n+1} ... w_v where v = min(|w|, i + n + 1)
/// ```
///
/// With padding:
/// ```text
/// w₋ₙ₊₁ = w₋ₙ₊₂ = ... = w₀ = $
/// ```
///
/// # Arguments
///
/// * `word` - The word w (will be padded with n $ symbols at the start)
/// * `position` - The position i (1-indexed as in the thesis)
/// * `max_distance` - The maximum distance n
///
/// # Returns
///
/// The relevant subword as a String, containing:
/// - Characters from position (i-n) to min(|w|, i+n+1)
/// - $ padding where needed for positions before the start of w
///
/// # Example
///
/// ```text
/// // For w = "abc", n = 2, i = 1:
/// // sₙ(w, 1) should give us w_{-1}w_0w_1w_2w_3 = "$$abc"
/// let subword = relevant_subword_from_chars(&['a', 'b', 'c'], 1, 2);
/// assert_eq!(subword, "$$abc");
/// ```
fn relevant_subword_from_chars(word_chars: &[char], position: usize, max_distance: u8) -> String {
    let n = max_distance as usize;
    let word_len = word_chars.len();

    // Start index: i - n, with '$' padding for positions before the word start.
    let pad_count = n.saturating_add(1).saturating_sub(position);
    let start = position.saturating_sub(n).max(1);

    // End index: min(|w|, i + n).
    // Note: Thesis page 51 says i+n+1, but testing shows i+n gives correct results.
    let end = position.saturating_add(n).min(word_len);
    let word_char_count = if start <= end { end - start + 1 } else { 0 };

    let mut result = String::with_capacity(pad_count.saturating_add(word_char_count));
    result.extend(std::iter::repeat_n(PADDING_CHAR, pad_count));

    if start <= end {
        result.extend(word_chars[start - 1..end].iter().copied());
    }

    result
}

#[cfg(test)]
fn relevant_subword(word: &str, position: usize, max_distance: u8) -> String {
    let word_chars: Vec<char> = word.chars().collect();
    relevant_subword_from_chars(&word_chars, position, max_distance)
}

/// Encodes a word pair (w, x) as a sequence of characteristic vectors hₙ(w, x).
///
/// # Theory (Page 51)
///
/// ```text
/// hₙ(w, x₁x₂...xₜ) = β(x₁, sₙ(w,1)) β(x₂, sₙ(w,2)) ... β(xₜ, sₙ(w,t))
/// ```
///
/// where sₙ(w, i) is the relevant subword around position i.
///
/// This encoding transforms the problem of checking whether x ∈ L^χ_Lev(n, w) into checking
/// whether hₙ(w, x) is accepted by the universal automaton A^{∀,χ}_n.
///
/// # Key Property (Proposition 19, pages 52-56)
///
/// ```text
/// x ∈ L^χ_Lev(n, w) ⇔ hₙ(w, x) ∈ L(A^{∀,χ}_n)
/// ```
///
/// # Arguments
///
/// * `word` - The reference word w
/// * `input` - The input word x to check
/// * `max_distance` - The maximum edit distance n
///
/// # Returns
///
/// A vector of characteristic vectors, one for each character in input.
///
/// Returns `None` if |input| > |word| + n (encoding undefined in this case).
///
/// # Example
///
/// ```rust
/// use liblevenshtein::transducer::universal::encode_word_pair;
///
/// // Example from thesis page 52:
/// // w = "abcabb", x = "dacab", n = 3
/// let encoding = encode_word_pair("abcabb", "dacab", 3).expect("doc/test fixture: encode_word_pair with valid args");
///
/// // encoding[0] = β(d, "$$$abcab") = "00000000"
/// // encoding[1] = β(a, "$$abcabb") = "00100100"
/// // encoding[2] = β(c, "$abcabb")  = "0001000"
/// // encoding[3] = β(a, "abcabb")   = "100100"
/// // encoding[4] = β(b, "bcabb")    = "10011"
/// ```
pub fn encode_word_pair(
    word: &str,
    input: &str,
    max_distance: u8,
) -> Option<Vec<CharacteristicVector>> {
    let word_chars: Vec<char> = word.chars().collect();
    let input_chars: Vec<char> = input.chars().collect();
    let t = input_chars.len();
    let p = word_chars.len();
    let n = max_distance as usize;

    // Check validity: t ≤ |w| + n
    if t > p.saturating_add(n) {
        return None;
    }

    let mut encoding = Vec::with_capacity(t);

    // For each position i in input (1-indexed in theory, 0-indexed here)
    for (i, &character) in input_chars.iter().enumerate() {
        let position = i.saturating_add(1);

        // Get the relevant subword sₙ(w, i)
        let subword = relevant_subword_from_chars(&word_chars, position, max_distance);

        // Compute β(xᵢ, sₙ(w, i))
        let cv = characteristic_vector(character, &subword);

        encoding.push(cv);
    }

    Some(encoding)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== CharacteristicVector Tests ====================

    #[test]
    fn test_characteristic_vector_basic() {
        let cv = CharacteristicVector::new('a', "banana");
        assert_eq!(cv.len(), 6);
        // "banana" = b(0) a(1) n(2) a(3) n(4) a(5)
        // 'a' matches at positions 1, 3, 5
        assert_eq!(cv.to_string(), "010101");
    }

    #[test]
    fn test_characteristic_vector_no_matches() {
        let cv = CharacteristicVector::new('x', "banana");
        assert_eq!(cv.to_string(), "000000");
        assert!(cv.is_all_zeros());
    }

    #[test]
    fn test_characteristic_vector_all_matches() {
        let cv = CharacteristicVector::new('a', "aaaa");
        assert_eq!(cv.to_string(), "1111");
        assert!(!cv.is_all_zeros());
    }

    #[test]
    fn test_characteristic_vector_empty() {
        let cv = CharacteristicVector::new('a', "");
        assert_eq!(cv.len(), 0);
        assert!(cv.is_empty());
        assert_eq!(cv.to_string(), "");
    }

    #[test]
    fn test_characteristic_vector_single_char() {
        let cv1 = CharacteristicVector::new('a', "a");
        assert_eq!(cv1.to_string(), "1");

        let cv2 = CharacteristicVector::new('b', "a");
        assert_eq!(cv2.to_string(), "0");
    }

    #[test]
    fn test_is_match() {
        let cv = CharacteristicVector::new('a', "banana");
        // "banana" = b(0) a(1) n(2) a(3) n(4) a(5)
        assert!(!cv.is_match(0)); // b[0] = 0 (b ≠ a)
        assert!(cv.is_match(1)); // b[1] = 1 (a = a)
        assert!(!cv.is_match(2)); // b[2] = 0 (n ≠ a)
        assert!(cv.is_match(3)); // b[3] = 1 (a = a)
        assert!(!cv.is_match(4)); // b[4] = 0 (n ≠ a)
        assert!(cv.is_match(5)); // b[5] = 1 (a = a)
    }

    #[test]
    fn test_starts_with_one() {
        let cv1 = CharacteristicVector::new('b', "banana");
        assert!(cv1.starts_with_one()); // "100000" (b matches at pos 0)

        let cv2 = CharacteristicVector::new('a', "apple");
        assert!(cv2.starts_with_one()); // "10010" (a matches at pos 0)

        let cv3 = CharacteristicVector::new('x', "");
        assert!(!cv3.starts_with_one()); // empty

        let cv4 = CharacteristicVector::new('a', "banana");
        assert!(!cv4.starts_with_one()); // "010101" (starts with 0)
    }

    #[test]
    fn test_first_match() {
        let cv1 = CharacteristicVector::new('a', "banana");
        assert_eq!(cv1.first_match(), Some(1)); // First 'a' at position 1

        let cv2 = CharacteristicVector::new('b', "banana");
        assert_eq!(cv2.first_match(), Some(0)); // First 'b' at position 0

        let cv3 = CharacteristicVector::new('n', "banana");
        assert_eq!(cv3.first_match(), Some(2)); // First 'n' at position 2

        let cv4 = CharacteristicVector::new('x', "banana");
        assert_eq!(cv4.first_match(), None); // No matches
    }

    #[test]
    fn test_suffix() {
        let cv = CharacteristicVector::new('a', "banana");
        // "banana" with 'a' = "010101"
        // suffix from position 2 = "0101"
        let suffix = cv.suffix(2);
        assert_eq!(suffix.to_string(), "0101");
        assert_eq!(suffix.len(), 4);
    }

    #[test]
    fn test_starts_with_pattern() {
        let cv = CharacteristicVector::new('a', "banana"); // "010101"

        assert!(cv.starts_with(&[false]));
        assert!(cv.starts_with(&[false, true]));
        assert!(cv.starts_with(&[false, true, false]));

        assert!(!cv.starts_with(&[true]));
        assert!(!cv.starts_with(&[false, false]));
        assert!(!cv.starts_with(&[true, true]));
    }

    #[test]
    fn test_iter() {
        let cv = CharacteristicVector::new('a', "aba");
        // "aba" with 'a' = a(0) b(1) a(2) = "101"
        let bits: Vec<bool> = cv.iter().collect();
        assert_eq!(bits, vec![true, false, true]);
    }

    // ==================== Helper Function Tests ====================

    #[test]
    fn test_characteristic_vector_function() {
        let cv = characteristic_vector('a', "banana");
        assert_eq!(cv.to_string(), "010101");
    }

    // ==================== Relevant Subword Tests ====================

    #[test]
    fn test_relevant_subword_with_padding() {
        // Example: w = "abc", n = 2, i = 1
        // sₙ(w, 1) = w_{-1}w_0w_1w_2w_3 = "$$abc"
        let subword = relevant_subword("abc", 1, 2);
        assert_eq!(subword, "$$abc");
    }

    #[test]
    fn test_relevant_subword_middle() {
        // w = "abcdef", n = 2, i = 3
        // sₙ(w, 3) = w_{i-n} to w_v where v = min(|w|, i+n)
        // = w_1 to w_{min(6,5)} = w_1...w_5 = "abcde"
        let subword = relevant_subword("abcdef", 3, 2);
        assert_eq!(subword, "abcde");
    }

    #[test]
    fn test_relevant_subword_end() {
        // w = "abc", n = 2, i = 3
        // sₙ(w, 3) = w_1w_2w_3 = "abc" (truncated at end)
        let subword = relevant_subword("abc", 3, 2);
        assert_eq!(subword, "abc");
    }

    #[test]
    fn test_relevant_subword_all_padding() {
        // w = "a", n = 3, i = 1
        // sₙ(w, 1) = w_{-2}w_{-1}w_0w_1w_2w_3w_4 but w only has 1 char
        // = "$$$a" + up to position min(1, 1+3+1) = min(1, 5) = 1
        // So we get from i-n=-2 to min(|w|, i+n+1)=min(1,5)=1: "$$$a"
        let subword = relevant_subword("a", 1, 3);
        assert_eq!(subword, "$$$a");
    }

    #[test]
    fn test_relevant_subword_unicode_character_positions() {
        let subword = relevant_subword("éaß", 2, 1);
        assert_eq!(subword, "éaß");
    }

    #[test]
    fn relevant_subword_at_saturated_position_is_empty() {
        let subword = relevant_subword("abc", usize::MAX, u8::MAX);
        assert_eq!(subword, "");
    }

    // ==================== Word-Pair Encoding Tests ====================

    #[test]
    fn test_encode_word_pair_example_from_thesis() {
        // Example based on thesis page 52, but adjusted for corrected formula
        // w = "abcabb", x = "dacab", n = 3
        let encoding = encode_word_pair("abcabb", "dacab", 3)
            .expect("doc/test fixture: encode_word_pair with valid args");

        assert_eq!(encoding.len(), 5);

        // encoding[0] = β(d, s₃(w, 1))
        // start=1-3=-2, end=min(6,1+3)=4
        // Positions -2,-1,0,1,2,3,4 = "$$$abca" (7 chars)
        assert_eq!(encoding[0].to_string(), "0000000"); // d matches nothing

        // encoding[1] = β(a, s₃(w, 2))
        // start=2-3=-1, end=min(6,2+3)=5
        // Positions -1,0,1,2,3,4,5 = "$$abcab" (7 chars)
        assert_eq!(encoding[1].to_string(), "0010010");

        // encoding[2] = β(c, s₃(w, 3))
        // start=3-3=0, end=min(6,3+3)=6
        // Positions 0,1,2,3,4,5,6 = "$abcabb" (7 chars)
        assert_eq!(encoding[2].to_string(), "0001000");

        // encoding[3] = β(a, s₃(w, 4))
        // start=4-3=1, end=min(6,4+3)=6
        // Positions 1,2,3,4,5,6 = "abcabb" (6 chars)
        assert_eq!(encoding[3].to_string(), "100100");

        // encoding[4] = β(b, s₃(w, 5))
        // start=5-3=2, end=min(6,5+3)=6
        // Positions 2,3,4,5,6 = "bcabb" (5 chars)
        assert_eq!(encoding[4].to_string(), "10011");
    }

    #[test]
    fn test_encode_word_pair_simple() {
        // w = "ab", x = "ab", n = 1
        let encoding = encode_word_pair("ab", "ab", 1)
            .expect("doc/test fixture: encode_word_pair with valid args");

        assert_eq!(encoding.len(), 2);

        // encoding[0] = β(a, s₁(w, 1)) = β(a, "$ab")
        assert_eq!(encoding[0].to_string(), "010");

        // encoding[1] = β(b, s₁(w, 2)) = β(b, "ab")
        assert_eq!(encoding[1].to_string(), "01");
    }

    #[test]
    fn test_encode_word_pair_exact_match() {
        // w = "test", x = "test", n = 0
        let encoding = encode_word_pair("test", "test", 0)
            .expect("doc/test fixture: encode_word_pair with valid args");

        assert_eq!(encoding.len(), 4);

        // Each character should match only itself
        assert_eq!(encoding[0].to_string(), "1"); // t matches t
        assert_eq!(encoding[1].to_string(), "1"); // e matches e
        assert_eq!(encoding[2].to_string(), "1"); // s matches s
        assert_eq!(encoding[3].to_string(), "1"); // t matches t
    }

    #[test]
    fn test_encode_word_pair_too_long_input() {
        // w = "ab", x = "abcd", n = 1
        // |x| = 4 > |w| + n = 3, so encoding is undefined
        let encoding = encode_word_pair("ab", "abcd", 1);
        assert!(encoding.is_none());
    }

    #[test]
    fn test_encode_word_pair_empty_input() {
        let encoding = encode_word_pair("test", "", 2)
            .expect("doc/test fixture: encode_word_pair with valid args");
        assert_eq!(encoding.len(), 0);
    }

    #[test]
    fn test_encode_word_pair_single_character() {
        // w = "a", x = "a", n = 1
        let encoding = encode_word_pair("a", "a", 1)
            .expect("doc/test fixture: encode_word_pair with valid args");

        assert_eq!(encoding.len(), 1);
        // s₁(a, 1) = w_0w_1w_2 but w only has 1 char at position 1
        // From i-n=1-1=0 to min(1, 1+1+1)=min(1,3)=1
        // Position 0,1 = "$a"
        assert_eq!(encoding[0].to_string(), "01");
    }

    #[test]
    fn test_encode_word_pair_unicode_counts_characters() {
        let exact = encode_word_pair("é", "é", 0)
            .expect("test fixture: one Unicode scalar at distance 0 is encodable");
        assert_eq!(exact.len(), 1);
        assert_eq!(exact[0].to_string(), "1");

        let insertion = encode_word_pair("", "é", 1)
            .expect("test fixture: one Unicode scalar insertion is within distance 1");
        assert_eq!(insertion.len(), 1);
        assert_eq!(insertion[0].to_string(), "0");
    }

    #[test]
    fn test_encode_word_pair_no_matches() {
        // w = "aaa", x = "bbb", n = 1
        let encoding = encode_word_pair("aaa", "bbb", 1)
            .expect("doc/test fixture: encode_word_pair with valid args");

        assert_eq!(encoding.len(), 3);

        // All characteristic vectors should be all zeros
        for cv in encoding {
            assert!(cv.is_all_zeros());
        }
    }

    #[test]
    fn test_encode_word_pair_max_distance_zero() {
        // w = "abc", x = "xyz", n = 0
        let encoding = encode_word_pair("abc", "xyz", 0)
            .expect("doc/test fixture: encode_word_pair with valid args");

        assert_eq!(encoding.len(), 3);

        // With n=0, each position only sees itself
        assert_eq!(encoding[0].to_string(), "0"); // x ≠ a
        assert_eq!(encoding[1].to_string(), "0"); // y ≠ b
        assert_eq!(encoding[2].to_string(), "0"); // z ≠ c
    }

    // ==================== Edge Cases and Properties ====================

    #[test]
    fn test_characteristic_vector_equality() {
        let cv1 = CharacteristicVector::new('a', "banana");
        let cv2 = CharacteristicVector::new('a', "banana");
        let cv3 = CharacteristicVector::new('b', "banana");

        assert_eq!(cv1, cv2);
        assert_ne!(cv1, cv3);
    }

    #[test]
    fn test_characteristic_vector_clone() {
        let cv = CharacteristicVector::new('a', "banana");
        let cloned = cv.clone();

        assert_eq!(cv, cloned);
        assert_eq!(cv.to_string(), cloned.to_string());
    }
}
