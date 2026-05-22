//! IPA-based syllable counting and analysis.
//!
//! This module provides language-agnostic syllable analysis using IPA
//! phonetic properties. Syllables are counted by detecting vowel nuclei
//! in IPA transcriptions.
//!
//! # Design Principle
//!
//! IPA provides universal phonetic properties that work across all languages:
//! - **Vowels**: Well-defined set of vowel symbols (a, e, i, o, u, ə, ɪ, ʊ, etc.)
//! - **Diphthongs**: Adjacent vowels treated as single syllable nucleus
//! - **Length markers**: ː follows vowels but doesn't add syllables
//! - **Stress markers**: ˈ and ˌ mark syllable boundaries
//!
//! # Examples
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::ipa_syllable::ipa_syllable_count;
//!
//! // English examples
//! assert_eq!(ipa_syllable_count("kæt"), 1);      // "cat"
//! assert_eq!(ipa_syllable_count("ˈhæp.i"), 2);   // "happy"
//! assert_eq!(ipa_syllable_count("ˈbju.tɪ.fəl"), 3); // "beautiful"
//!
//! // Works with any IPA transcription
//! assert_eq!(ipa_syllable_count("kamal"), 2);    // Hindi "कमल"
//! ```

/// Length markers that don't add syllables.
const LENGTH_MARKERS: &[char] = &[
    'ː', // long (U+02D0)
    'ˑ', // half-long (U+02D1)
    '˘', // extra-short (U+02D8)
];

/// Stress markers that indicate syllable boundaries.
const STRESS_MARKERS: &[char] = &[
    'ˈ', // primary stress (U+02C8)
    'ˌ', // secondary stress (U+02CC)
];

/// Syllable boundary marker in IPA.
const SYLLABLE_BOUNDARY: char = '.';

/// Check if a character is an IPA vowel.
///
/// Returns true for all IPA vowel symbols including:
/// - Cardinal vowels (a, e, i, o, u)
/// - Central vowels (ə, ɜ, ɐ)
/// - Modified vowels (ɪ, ʊ, ɛ, ɔ, æ, ʌ, ɑ, ɒ)
/// - Front rounded vowels (y, ø, œ)
/// - Back unrounded vowels (ɯ, ɤ)
/// - Rhotic vowels (ɚ, ɝ)
#[inline]
pub fn is_ipa_vowel(c: char) -> bool {
    // Use match for common vowels (fast path)
    matches!(
        c,
        // Basic vowels
        'a' | 'e' | 'i' | 'o' | 'u' |
        'A' | 'E' | 'I' | 'O' | 'U' |
        // Common IPA vowels
        'ə' | 'ɪ' | 'ʊ' | 'ɛ' | 'ɔ' |
        'æ' | 'ʌ' | 'ɑ' | 'ɒ' | 'ɜ' | 'ɐ' |
        // Front rounded (German ü, ö, etc.)
        'y' | 'ʏ' | 'ø' | 'œ' | 'ɶ' |
        // Central vowels
        'ɨ' | 'ʉ' | 'ɘ' | 'ɵ' | 'ɞ' |
        // Back unrounded
        'ɯ' | 'ɤ' |
        // Rhotic vowels
        'ɚ' | 'ɝ'
    )
}

/// Check if a character is a length marker.
#[inline]
pub fn is_length_marker(c: char) -> bool {
    LENGTH_MARKERS.contains(&c)
}

/// Check if a character is a stress marker.
#[inline]
pub fn is_stress_marker(c: char) -> bool {
    STRESS_MARKERS.contains(&c)
}

/// Check if a character is a syllable boundary marker.
#[inline]
pub fn is_syllable_boundary(c: char) -> bool {
    c == SYLLABLE_BOUNDARY
}

/// Check if a character is an IPA consonant.
///
/// Returns true for any character that is not a vowel, length marker,
/// stress marker, or syllable boundary. This is a broad definition that
/// covers all IPA consonants including:
/// - Plosives: p, b, t, d, k, g, ʔ
/// - Fricatives: f, v, s, z, ʃ, ʒ, θ, ð, h
/// - Nasals: m, n, ŋ
/// - Approximants: l, r, ɹ, j, w
/// - And all other consonant symbols
#[inline]
pub fn is_ipa_consonant(c: char) -> bool {
    // Consonants are everything that's not a vowel or suprasegmental
    !is_ipa_vowel(c) && !is_length_marker(c) && !is_stress_marker(c) && !is_syllable_boundary(c)
}

/// Count syllables in an IPA transcription.
///
/// Uses vowel nuclei counting with the following rules:
/// 1. Each vowel or vowel group (diphthong) counts as one syllable
/// 2. Adjacent vowels are treated as a diphthong (same syllable)
/// 3. Length markers (ː) don't add syllables
/// 4. If the transcription contains explicit syllable boundaries (`.`),
///    those are counted instead
///
/// # Arguments
///
/// * `ipa` - An IPA transcription string
///
/// # Returns
///
/// The number of syllables (minimum 1 for non-empty strings)
///
/// # Examples
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::ipa_syllable::ipa_syllable_count;
///
/// // Simple words
/// assert_eq!(ipa_syllable_count("kæt"), 1);
/// assert_eq!(ipa_syllable_count("hæpi"), 2);
///
/// // With explicit syllable markers
/// assert_eq!(ipa_syllable_count("ˈhæp.i"), 2);
///
/// // Diphthongs count as one syllable
/// assert_eq!(ipa_syllable_count("haʊs"), 1);  // "house"
/// assert_eq!(ipa_syllable_count("kɔɪn"), 1);  // "coin"
/// ```
pub fn ipa_syllable_count(ipa: &str) -> usize {
    if ipa.is_empty() {
        return 0;
    }

    let chars: Vec<char> = ipa.chars().collect();

    // If explicit syllable boundaries are present, use those
    let boundary_count = chars.iter().filter(|&&c| c == SYLLABLE_BOUNDARY).count();
    if boundary_count > 0 {
        // Number of syllables = number of boundaries + 1
        return boundary_count + 1;
    }

    // Count vowel groups (nuclei)
    let mut count = 0;
    let mut in_vowel_group = false;

    for &c in &chars {
        if is_ipa_vowel(c) {
            if !in_vowel_group {
                count += 1;
                in_vowel_group = true;
            }
            // Adjacent vowels = diphthong = same syllable
        } else if !is_length_marker(c) && !is_stress_marker(c) {
            // Consonant or other non-vowel ends the vowel group
            in_vowel_group = false;
        }
        // Length markers don't end vowel groups (iː is still one syllable)
        // Stress markers don't end vowel groups either
    }

    // Ensure at least 1 syllable for non-empty strings
    count.max(1)
}

/// Get syllable boundary positions in an IPA transcription.
///
/// Returns a vector of character indices where syllables begin.
/// The first syllable always starts at index 0.
///
/// If the transcription contains explicit syllable boundary markers (`.`),
/// those are used directly. Otherwise, boundaries are computed using the
/// Maximum Onset Principle (consonants go with following vowel when possible).
///
/// # Arguments
///
/// * `ipa` - An IPA transcription string
///
/// # Returns
///
/// Vector of starting positions for each syllable
pub fn ipa_syllable_boundaries(ipa: &str) -> Vec<usize> {
    if ipa.is_empty() {
        return vec![];
    }

    let chars: Vec<char> = ipa.chars().collect();
    let mut boundaries = vec![0];

    // Check for explicit boundaries first
    for (i, &c) in chars.iter().enumerate() {
        if c == SYLLABLE_BOUNDARY && i + 1 < chars.len() {
            boundaries.push(i + 1);
        }
    }

    if boundaries.len() > 1 {
        return boundaries;
    }

    // Find vowel positions for implicit boundary detection
    let vowel_positions: Vec<usize> = chars
        .iter()
        .enumerate()
        .filter(|(_, &c)| is_ipa_vowel(c))
        .map(|(i, _)| i)
        .collect();

    // For each pair of vowels, find the syllable boundary between them
    for window in vowel_positions.windows(2) {
        let v1 = window[0];
        let v2 = window[1];

        // Skip to end of first vowel group (handle diphthongs)
        let mut v1_end = v1;
        while v1_end + 1 < v2
            && (is_ipa_vowel(chars[v1_end + 1]) || is_length_marker(chars[v1_end + 1]))
        {
            v1_end += 1;
        }

        // Skip past any length markers after second vowel
        let consonant_start = v1_end + 1;
        let consonant_end = v2;

        if consonant_start >= consonant_end {
            // No consonants between vowels (hiatus or adjacent in diphthong)
            continue;
        }

        // Maximum Onset Principle: give as many consonants to the following syllable
        // as form a valid onset. For simplicity, we'll put all consonants with the
        // following vowel (common cross-linguistic default)
        boundaries.push(consonant_start);
    }

    boundaries.sort();
    boundaries.dedup();
    boundaries
}

/// Check if a position is in the final syllable.
///
/// # Arguments
///
/// * `ipa` - The IPA transcription
/// * `pos` - Character position to check
///
/// # Returns
///
/// `true` if the position is in the last syllable
pub fn is_final_syllable(ipa: &str, pos: usize) -> bool {
    let boundaries = ipa_syllable_boundaries(ipa);
    if boundaries.is_empty() {
        return true;
    }

    let last_boundary = *boundaries.last().expect("non-empty checked above");
    pos >= last_boundary
}

/// Check if a position is in the initial syllable.
///
/// # Arguments
///
/// * `ipa` - The IPA transcription
/// * `pos` - Character position to check
///
/// # Returns
///
/// `true` if the position is in the first syllable
pub fn is_initial_syllable(ipa: &str, pos: usize) -> bool {
    let boundaries = ipa_syllable_boundaries(ipa);

    if boundaries.len() <= 1 {
        return true;
    }

    pos < boundaries[1]
}

/// Check if a syllable is open (ends with a vowel).
///
/// An open syllable has no coda consonant - it ends with the vowel nucleus.
///
/// # Arguments
///
/// * `ipa` - The IPA transcription
/// * `pos` - Position within the syllable to check
///
/// # Returns
///
/// `true` if the syllable containing `pos` is open
pub fn is_open_syllable(ipa: &str, pos: usize) -> bool {
    let chars: Vec<char> = ipa.chars().collect();
    let boundaries = ipa_syllable_boundaries(ipa);

    if boundaries.is_empty() || pos >= chars.len() {
        return false;
    }

    // Find which syllable this position is in
    let syllable_idx = boundaries
        .iter()
        .enumerate()
        .rev()
        .find(|(_, &b)| b <= pos)
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Find the end of this syllable
    let syllable_end = if syllable_idx + 1 < boundaries.len() {
        boundaries[syllable_idx + 1]
    } else {
        chars.len()
    };

    // Check if the syllable ends with a vowel (open) or consonant (closed)
    // Work backwards from syllable end, skipping length markers
    let mut check_pos = syllable_end.saturating_sub(1);
    while check_pos > 0 && is_length_marker(chars[check_pos]) {
        check_pos -= 1;
    }

    is_ipa_vowel(chars[check_pos])
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Vowel Detection Tests ====================

    #[test]
    fn test_is_ipa_vowel_basic() {
        assert!(is_ipa_vowel('a'));
        assert!(is_ipa_vowel('e'));
        assert!(is_ipa_vowel('i'));
        assert!(is_ipa_vowel('o'));
        assert!(is_ipa_vowel('u'));
    }

    #[test]
    fn test_is_ipa_vowel_ipa_symbols() {
        assert!(is_ipa_vowel('ə')); // schwa
        assert!(is_ipa_vowel('ɪ')); // near-close front
        assert!(is_ipa_vowel('ʊ')); // near-close back
        assert!(is_ipa_vowel('ɛ')); // open-mid front
        assert!(is_ipa_vowel('ɔ')); // open-mid back
        assert!(is_ipa_vowel('æ')); // near-open front
        assert!(is_ipa_vowel('ʌ')); // open-mid back unrounded
        assert!(is_ipa_vowel('ɑ')); // open back unrounded
    }

    #[test]
    fn test_is_ipa_vowel_front_rounded() {
        assert!(is_ipa_vowel('y')); // close front rounded
        assert!(is_ipa_vowel('ø')); // close-mid front rounded
        assert!(is_ipa_vowel('œ')); // open-mid front rounded
    }

    #[test]
    fn test_is_ipa_vowel_consonants_false() {
        assert!(!is_ipa_vowel('p'));
        assert!(!is_ipa_vowel('t'));
        assert!(!is_ipa_vowel('k'));
        assert!(!is_ipa_vowel('ʃ'));
        assert!(!is_ipa_vowel('ŋ'));
    }

    // ==================== Syllable Count Tests ====================

    #[test]
    fn test_syllable_count_monosyllables() {
        assert_eq!(ipa_syllable_count("kæt"), 1); // cat
        assert_eq!(ipa_syllable_count("dɔg"), 1); // dog
        assert_eq!(ipa_syllable_count("rʌn"), 1); // run
        assert_eq!(ipa_syllable_count("flaɪ"), 1); // fly (diphthong)
    }

    #[test]
    fn test_syllable_count_disyllables() {
        assert_eq!(ipa_syllable_count("hæpi"), 2); // happy
        assert_eq!(ipa_syllable_count("wɔːtər"), 2); // water
        assert_eq!(ipa_syllable_count("rʌnɪŋ"), 2); // running
    }

    #[test]
    fn test_syllable_count_trisyllables() {
        assert_eq!(ipa_syllable_count("bjuːtɪfəl"), 3); // beautiful
        assert_eq!(ipa_syllable_count("kəmpjuːtər"), 3); // computer
    }

    #[test]
    fn test_syllable_count_with_explicit_boundaries() {
        assert_eq!(ipa_syllable_count("ˈhæp.i"), 2);
        assert_eq!(ipa_syllable_count("ˈbju.tɪ.fəl"), 3);
        assert_eq!(ipa_syllable_count("kəm.ˈpju.tər"), 3);
    }

    #[test]
    fn test_syllable_count_with_length_markers() {
        assert_eq!(ipa_syllable_count("biː"), 1); // "bee" - long vowel
        assert_eq!(ipa_syllable_count("siː"), 1); // "see"
        assert_eq!(ipa_syllable_count("wɔːtər"), 2); // "water" with long vowel
    }

    #[test]
    fn test_syllable_count_diphthongs() {
        assert_eq!(ipa_syllable_count("haʊs"), 1); // house
        assert_eq!(ipa_syllable_count("kɔɪn"), 1); // coin
        assert_eq!(ipa_syllable_count("baɪ"), 1); // buy
        assert_eq!(ipa_syllable_count("goʊ"), 1); // go
    }

    #[test]
    fn test_syllable_count_hindi() {
        // Hindi words in IPA (after schwa deletion rules apply)
        assert_eq!(ipa_syllable_count("kamal"), 2); // कमल (lotus)
        assert_eq!(ipa_syllable_count("namaste"), 3); // नमस्ते
    }

    #[test]
    fn test_syllable_count_empty() {
        assert_eq!(ipa_syllable_count(""), 0);
    }

    // ==================== Syllable Position Tests ====================

    #[test]
    fn test_is_final_syllable() {
        // In "hæpi" (happy), position 3 (the 'i') is in final syllable
        assert!(is_final_syllable("hæpi", 3));
        // Position 0 is not in final syllable
        assert!(!is_final_syllable("hæpi", 0));
        // In monosyllables, everything is in final syllable
        assert!(is_final_syllable("kæt", 0));
        assert!(is_final_syllable("kæt", 2));
    }

    #[test]
    fn test_is_initial_syllable() {
        // In "hæpi", position 0 is in initial syllable
        assert!(is_initial_syllable("hæpi", 0));
        // Position 3 (the 'i') is NOT in initial syllable
        assert!(!is_initial_syllable("hæpi", 3));
        // In monosyllables, everything is in initial syllable
        assert!(is_initial_syllable("kæt", 0));
        assert!(is_initial_syllable("kæt", 2));
    }

    #[test]
    fn test_is_open_syllable() {
        // "bi" (bee) is open - ends with vowel
        assert!(is_open_syllable("bi", 0));
        // "kæt" (cat) is closed - ends with consonant
        assert!(!is_open_syllable("kæt", 1));
    }

    // ==================== Consonant Detection Tests ====================

    #[test]
    fn test_is_ipa_consonant() {
        assert!(is_ipa_consonant('p'));
        assert!(is_ipa_consonant('t'));
        assert!(is_ipa_consonant('k'));
        assert!(is_ipa_consonant('ʃ'));
        assert!(is_ipa_consonant('ŋ'));
        assert!(is_ipa_consonant('θ'));
        assert!(is_ipa_consonant('ð'));

        // Vowels are not consonants
        assert!(!is_ipa_consonant('a'));
        assert!(!is_ipa_consonant('ə'));

        // Suprasegmentals are not consonants
        assert!(!is_ipa_consonant('ː'));
        assert!(!is_ipa_consonant('ˈ'));
    }
}
