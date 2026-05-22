// ============================================================================
// Tests
// ============================================================================

use super::*;

#[test]
fn test_phone_pattern_char() {
    let p = PhonePattern::Char('a');
    assert!(p.is_char());
    assert!(!p.is_digraph());
    assert_eq!(p.as_char(), Some('a'));
    assert_eq!(p.as_digraph(), None);
    assert!(p.matches_char('a'));
    assert!(!p.matches_char('b'));
}

#[test]
fn test_phone_pattern_digraph() {
    let p = PhonePattern::Digraph('s', 'h');
    assert!(!p.is_char());
    assert!(p.is_digraph());
    assert_eq!(p.as_char(), None);
    assert_eq!(p.as_digraph(), Some(('s', 'h')));
    assert!(p.matches_digraph('s', 'h'));
    assert!(!p.matches_digraph('s', 'z'));
}

#[test]
fn test_get_named_class_vowel() {
    let vowels = get_named_class("vowel").expect("vowel class should exist");
    assert!(!vowels.patterns.is_empty());

    // Should contain ASCII vowels
    assert!(vowels.patterns.contains(&PhonePattern::Char('a')));
    assert!(vowels.patterns.contains(&PhonePattern::Char('e')));
    assert!(vowels.patterns.contains(&PhonePattern::Char('i')));
    assert!(vowels.patterns.contains(&PhonePattern::Char('o')));
    assert!(vowels.patterns.contains(&PhonePattern::Char('u')));

    // Should contain IPA vowels
    assert!(vowels.patterns.contains(&PhonePattern::Char('ə')));
    assert!(vowels.patterns.contains(&PhonePattern::Char('ɪ')));
}

#[test]
fn test_case_insensitive_lookup() {
    let v1 = get_named_class("vowel").expect("lowercase");
    let v2 = get_named_class("VOWEL").expect("uppercase");
    let v3 = get_named_class("Vowel").expect("mixed case");

    assert_eq!(v1.patterns.len(), v2.patterns.len());
    assert_eq!(v2.patterns.len(), v3.patterns.len());
}

#[test]
fn test_full_word_alias() {
    // "plosive" is an alias for "stop"
    let s1 = get_named_class("stop").expect("full name");
    let s2 = get_named_class("plosive").expect("alias");
    assert_eq!(s1.patterns.len(), s2.patterns.len());

    // "semivowel" is an alias for "glide"
    let g1 = get_named_class("glide").expect("full name");
    let g2 = get_named_class("semivowel").expect("alias");
    assert_eq!(g1.patterns.len(), g2.patterns.len());
}

#[test]
fn test_fricative_has_digraphs() {
    let fric = get_named_class("fricative").expect("fricative class");

    // Should have IPA chars
    assert!(fric.patterns.contains(&PhonePattern::Char('ʃ')));
    assert!(fric.patterns.contains(&PhonePattern::Char('θ')));

    // Should have digraphs
    assert!(fric.patterns.contains(&PhonePattern::Digraph('s', 'h')));
    assert!(fric.patterns.contains(&PhonePattern::Digraph('t', 'h')));
}

#[test]
fn test_nasal_has_ng_digraph() {
    let nasal = get_named_class("nasal").expect("nasal class");

    assert!(nasal.patterns.contains(&PhonePattern::Char('ŋ')));
    assert!(nasal.patterns.contains(&PhonePattern::Digraph('n', 'g')));
}

#[test]
fn test_is_builtin_class() {
    assert!(is_builtin_class("vowel"));
    assert!(is_builtin_class("VOWEL"));
    assert!(is_builtin_class("plosive")); // alias for stop
    assert!(is_builtin_class("fricative"));
    assert!(is_builtin_class("alpha"));

    // Single-letter names are NOT built-in (reserved for user symbols)
    assert!(!is_builtin_class("V"));
    assert!(!is_builtin_class("C"));
    assert!(!is_builtin_class("not_a_class"));
    assert!(!is_builtin_class("CUSTOM"));
}

#[test]
fn test_get_chars_only() {
    let chars = get_chars_only("fricative").expect("fricative exists");

    // Should have single chars
    assert!(chars.contains(&'f'));
    assert!(chars.contains(&'ʃ'));

    // Should NOT have digraph components as separate chars
    // (digraphs are filtered out)
}

#[test]
fn test_get_digraphs_only() {
    let digraphs = get_digraphs_only("fricative").expect("fricative exists");

    assert!(digraphs.contains(&('s', 'h')));
    assert!(digraphs.contains(&('t', 'h')));
}

#[test]
fn test_posix_classes() {
    let alpha = get_named_class("alpha").expect("alpha");
    let digit = get_named_class("digit").expect("digit");
    let alnum = get_named_class("alnum").expect("alnum");

    // Alpha should have letters only
    assert!(alpha.patterns.contains(&PhonePattern::Char('a')));
    assert!(alpha.patterns.contains(&PhonePattern::Char('Z')));
    assert!(!alpha.patterns.contains(&PhonePattern::Char('0')));

    // Digit should have digits only
    assert!(digit.patterns.contains(&PhonePattern::Char('0')));
    assert!(digit.patterns.contains(&PhonePattern::Char('9')));
    assert!(!digit.patterns.contains(&PhonePattern::Char('a')));

    // Alnum should have both
    assert!(alnum.patterns.contains(&PhonePattern::Char('a')));
    assert!(alnum.patterns.contains(&PhonePattern::Char('0')));
}

#[test]
fn test_ascii_vs_ipa_subsets() {
    let ascii_v = get_named_class("ascii_vowel").expect("ascii_vowel");
    let ipa_v = get_named_class("ipa_vowel").expect("ipa_vowel");

    // ASCII should only have basic vowels
    assert!(ascii_v.patterns.contains(&PhonePattern::Char('a')));
    assert!(!ascii_v.patterns.contains(&PhonePattern::Char('ə')));

    // IPA should only have IPA vowels
    assert!(ipa_v.patterns.contains(&PhonePattern::Char('ə')));
    assert!(!ipa_v.patterns.contains(&PhonePattern::Char('a')));
}

#[test]
fn test_all_builtin_class_names() {
    let names = all_builtin_class_names();

    assert!(names.contains(&"vowel"));
    assert!(names.contains(&"consonant"));
    assert!(names.contains(&"alpha"));
    assert!(names.contains(&"fricative"));
}

// =========================================================================
// Feature Bundle Helper Tests
// =========================================================================

#[test]
fn test_get_all_phonetic_chars() {
    let all = get_all_phonetic_chars();

    // Should contain vowels
    assert!(all.contains(&'a'));
    assert!(all.contains(&'e'));
    assert!(all.contains(&'ə')); // IPA schwa

    // Should contain consonants
    assert!(all.contains(&'b'));
    assert!(all.contains(&'p'));
    assert!(all.contains(&'ŋ')); // IPA eng

    // Should NOT contain digits or punctuation
    assert!(!all.contains(&'0'));
    assert!(!all.contains(&'.'));
}

#[test]
fn test_intersect_char_sets_voiced_stop() {
    let voiced = get_chars_only("voiced").expect("voiced class");
    let stop = get_chars_only("stop").expect("stop class");
    let result = intersect_char_sets(&[voiced, stop]);

    // Voiced stops: b, d, g
    assert!(result.contains(&'b'));
    assert!(result.contains(&'d'));
    assert!(result.contains(&'g'));

    // Voiceless stops should NOT be in result
    assert!(!result.contains(&'p'));
    assert!(!result.contains(&'t'));
    assert!(!result.contains(&'k'));

    // Non-stops should NOT be in result
    assert!(!result.contains(&'v')); // voiced fricative
    assert!(!result.contains(&'z')); // voiced fricative
}

#[test]
fn test_intersect_char_sets_empty() {
    let result = intersect_char_sets(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_intersect_char_sets_single() {
    let stop = get_chars_only("stop").expect("stop class");
    let result = intersect_char_sets(&[stop.clone()]);

    // Single set intersection should return the same set
    assert_eq!(result.len(), stop.len());
    for c in &stop {
        assert!(result.contains(c));
    }
}

#[test]
fn test_intersect_char_sets_three_features() {
    // high + front + vowel should give high front vowels (i, ɪ, y, ʏ, etc.)
    let high = get_chars_only("high_vowel").expect("high_vowel class");
    let front = get_chars_only("front_vowel").expect("front_vowel class");
    let vowel = get_chars_only("vowel").expect("vowel class");
    let result = intersect_char_sets(&[high, front, vowel]);

    // i should be in high front vowels
    assert!(result.contains(&'i'));
    assert!(result.contains(&'I'));

    // a should NOT be in result (low, not high)
    assert!(!result.contains(&'a'));

    // u should NOT be in result (back, not front)
    assert!(!result.contains(&'u'));
}

#[test]
fn test_negate_char_set_nasal() {
    let nasal = get_chars_only("nasal").expect("nasal class");
    let not_nasal = negate_char_set(&nasal);

    // Nasals should NOT be in result
    assert!(!not_nasal.contains(&'m'));
    assert!(!not_nasal.contains(&'n'));
    assert!(!not_nasal.contains(&'ŋ'));

    // Other consonants should be in result
    assert!(not_nasal.contains(&'p'));
    assert!(not_nasal.contains(&'b'));
    assert!(not_nasal.contains(&'t'));

    // Vowels should be in result
    assert!(not_nasal.contains(&'a'));
    assert!(not_nasal.contains(&'e'));
}

#[test]
fn test_negate_char_set_empty() {
    let not_empty = negate_char_set(&[]);
    let all = get_all_phonetic_chars();

    // Negating empty set should give all chars
    assert_eq!(not_empty.len(), all.len());
}

#[test]
fn test_intersect_with_negation() {
    // Test the pattern: [:!nasal stop:] = oral stops (p, t, k, b, d, g)
    let nasal = get_chars_only("nasal").expect("nasal class");
    let not_nasal = negate_char_set(&nasal);
    let stop = get_chars_only("stop").expect("stop class");
    let result = intersect_char_sets(&[not_nasal, stop]);

    // All non-nasal stops
    assert!(result.contains(&'p'));
    assert!(result.contains(&'t'));
    assert!(result.contains(&'k'));
    assert!(result.contains(&'b'));
    assert!(result.contains(&'d'));
    assert!(result.contains(&'g'));

    // Nasal consonants should NOT be in result (m, n are not stops anyway)
    assert!(!result.contains(&'m'));
    assert!(!result.contains(&'n'));

    // Other consonants should NOT be in result
    assert!(!result.contains(&'f'));
    assert!(!result.contains(&'s'));
}

// =========================================================================
// Extended IPA Class Tests
// =========================================================================

#[test]
fn test_retroflex_class() {
    let retroflex = get_named_class("retroflex").expect("retroflex class");

    // Should have retroflex consonants
    assert!(retroflex.patterns.contains(&PhonePattern::Char('ʈ'))); // U+0288
    assert!(retroflex.patterns.contains(&PhonePattern::Char('ɖ'))); // U+0256
    assert!(retroflex.patterns.contains(&PhonePattern::Char('ɻ'))); // U+027B
    assert!(retroflex.patterns.contains(&PhonePattern::Char('ʂ'))); // U+0282
}

#[test]
fn test_uvular_class() {
    let uvular = get_named_class("uvular").expect("uvular class");

    assert!(uvular.patterns.contains(&PhonePattern::Char('ʁ'))); // U+0281
    assert!(uvular.patterns.contains(&PhonePattern::Char('ɢ'))); // U+0262
    assert!(uvular.patterns.contains(&PhonePattern::Char('ʀ'))); // U+0280
}

#[test]
fn test_click_class() {
    let click = get_named_class("click").expect("click class");

    assert!(click.patterns.contains(&PhonePattern::Char('ʘ'))); // bilabial
    assert!(click.patterns.contains(&PhonePattern::Char('ǀ'))); // dental
    assert!(click.patterns.contains(&PhonePattern::Char('ǃ'))); // alveolar
}

#[test]
fn test_implosive_class() {
    let implosive = get_named_class("implosive").expect("implosive class");

    assert!(implosive.patterns.contains(&PhonePattern::Char('ɓ'))); // U+0253
    assert!(implosive.patterns.contains(&PhonePattern::Char('ɗ'))); // U+0257
    assert!(implosive.patterns.contains(&PhonePattern::Char('ɠ'))); // U+0260
}

#[test]
fn test_front_rounded_class() {
    let front_rounded = get_named_class("front_rounded").expect("front_rounded class");

    // German ü, ö sounds
    assert!(front_rounded.patterns.contains(&PhonePattern::Char('y'))); // close front rounded
    assert!(front_rounded.patterns.contains(&PhonePattern::Char('ø'))); // close-mid front rounded
    assert!(front_rounded.patterns.contains(&PhonePattern::Char('œ'))); // open-mid front rounded
}

#[test]
fn test_stress_class() {
    let stress = get_named_class("stress").expect("stress class");

    assert!(stress.patterns.contains(&PhonePattern::Char('ˈ'))); // primary
    assert!(stress.patterns.contains(&PhonePattern::Char('ˌ'))); // secondary
}

#[test]
fn test_length_class() {
    let length = get_named_class("length").expect("length class");

    assert!(length.patterns.contains(&PhonePattern::Char('ː'))); // long
    assert!(length.patterns.contains(&PhonePattern::Char('ˑ'))); // half-long
}

#[test]
fn test_tone_class() {
    let tone = get_named_class("tone").expect("tone class");

    // Chao tone letters
    assert!(tone.patterns.contains(&PhonePattern::Char('˥'))); // extra high
    assert!(tone.patterns.contains(&PhonePattern::Char('˩'))); // extra low
}

#[test]
fn test_diacritic_class() {
    let diacritic = get_named_class("diacritic").expect("diacritic class");

    // Should have common diacritics
    assert!(diacritic.patterns.contains(&PhonePattern::Char('\u{0325}'))); // voiceless
    assert!(diacritic.patterns.contains(&PhonePattern::Char('\u{0303}'))); // nasalized
    assert!(diacritic.patterns.contains(&PhonePattern::Char('\u{02B0}'))); // aspirated
}

#[test]
fn test_ipa_affricate_class() {
    let affricate = get_named_class("ipa_affricate").expect("ipa_affricate class");

    // Precomposed affricates
    assert!(affricate.patterns.contains(&PhonePattern::Char('ʧ'))); // tʃ
    assert!(affricate.patterns.contains(&PhonePattern::Char('ʤ'))); // dʒ
    assert!(affricate.patterns.contains(&PhonePattern::Char('ʦ'))); // ts
}

#[test]
fn test_comprehensive_ipa_class() {
    let ipa = get_named_class("ipa").expect("ipa class");

    // Should have all major IPA categories
    // Vowels
    assert!(ipa.patterns.contains(&PhonePattern::Char('ə'))); // schwa
    assert!(ipa.patterns.contains(&PhonePattern::Char('ɚ'))); // rhotic schwa

    // Consonants
    assert!(ipa.patterns.contains(&PhonePattern::Char('ŋ'))); // velar nasal
    assert!(ipa.patterns.contains(&PhonePattern::Char('ʃ'))); // sh sound
    assert!(ipa.patterns.contains(&PhonePattern::Char('ʁ'))); // uvular fricative

    // Clicks
    assert!(ipa.patterns.contains(&PhonePattern::Char('ǃ'))); // alveolar click

    // Implosives
    assert!(ipa.patterns.contains(&PhonePattern::Char('ɓ'))); // bilabial implosive

    // Suprasegmentals
    assert!(ipa.patterns.contains(&PhonePattern::Char('ˈ'))); // primary stress
    assert!(ipa.patterns.contains(&PhonePattern::Char('ː'))); // long
}

#[test]
fn test_tap_flap_alias() {
    let tap = get_named_class("tap").expect("tap class");
    let flap = get_named_class("flap").expect("flap alias");

    assert_eq!(tap.patterns.len(), flap.patterns.len());
    assert!(tap.patterns.contains(&PhonePattern::Char('ɾ'))); // alveolar tap
}

#[test]
fn test_place_of_articulation_classes() {
    // Test various place of articulation classes
    let bilabial = get_named_class("bilabial").expect("bilabial");
    let alveolar = get_named_class("alveolar").expect("alveolar");
    let velar = get_named_class("velar").expect("velar");
    let palatal = get_named_class("palatal").expect("palatal");

    assert!(bilabial.patterns.contains(&PhonePattern::Char('p')));
    assert!(bilabial.patterns.contains(&PhonePattern::Char('m')));

    assert!(alveolar.patterns.contains(&PhonePattern::Char('t')));
    assert!(alveolar.patterns.contains(&PhonePattern::Char('n')));

    assert!(velar.patterns.contains(&PhonePattern::Char('k')));
    assert!(velar.patterns.contains(&PhonePattern::Char('ŋ')));

    assert!(palatal.patterns.contains(&PhonePattern::Char('ç'))); // voiceless palatal fricative
    assert!(palatal.patterns.contains(&PhonePattern::Char('j'))); // palatal approximant
}

#[test]
fn test_manner_of_articulation_classes() {
    let trill = get_named_class("trill").expect("trill");
    let approximant = get_named_class("approximant").expect("approximant");
    let lateral = get_named_class("lateral").expect("lateral");

    assert!(trill.patterns.contains(&PhonePattern::Char('r')));
    assert!(trill.patterns.contains(&PhonePattern::Char('ʀ'))); // uvular trill

    assert!(approximant.patterns.contains(&PhonePattern::Char('ɹ'))); // alveolar approx
    assert!(approximant.patterns.contains(&PhonePattern::Char('j'))); // palatal approx

    assert!(lateral.patterns.contains(&PhonePattern::Char('l')));
    assert!(lateral.patterns.contains(&PhonePattern::Char('ɬ'))); // lateral fricative
}

#[test]
fn test_prenasalized_click_class() {
    let prenasalized =
        get_named_class("prenasalized_click").expect("prenasalized_click class should exist");

    // Test alias lookup
    let nasal_click = get_named_class("nasal_click").expect("nasal_click alias should work");
    assert_eq!(prenasalized.patterns.len(), nasal_click.patterns.len());

    // Test trigraphs (prenasalized aspirated clicks - Xhosa)
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Trigraph('ŋ', 'ǀ', 'ʰ'))); // ŋǀʰ
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Trigraph('ŋ', 'ǃ', 'ʰ'))); // ŋǃʰ
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Trigraph('ŋ', 'ǁ', 'ʰ'))); // ŋǁʰ

    // Test trigraphs (prenasalized voiced clicks - Zulu)
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Trigraph('ŋ', 'ɡ', 'ǀ'))); // ŋɡǀ
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Trigraph('ŋ', 'ɡ', 'ǃ'))); // ŋɡǃ
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Trigraph('ŋ', 'ɡ', 'ǁ'))); // ŋɡǁ

    // Test tetragraphs (prenasalized voiced aspirated clicks - rare)
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Tetragraph('ŋ', 'ɡ', 'ǀ', 'ʰ'))); // ŋɡǀʰ
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Tetragraph('ŋ', 'ɡ', 'ǃ', 'ʰ'))); // ŋɡǃʰ
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Tetragraph('ŋ', 'ɡ', 'ǁ', 'ʰ'))); // ŋɡǁʰ

    // Test tetragraphs (prenasalized labialized clicks without aspiration)
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Tetragraph('ŋ', 'ɡ', 'ǀ', 'ʷ'))); // ŋɡǀʷ
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Tetragraph('ŋ', 'ɡ', 'ǃ', 'ʷ'))); // ŋɡǃʷ
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Tetragraph('ŋ', 'ɡ', 'ǁ', 'ʷ'))); // ŋɡǁʷ

    // Test pentagraphs (prenasalized voiced aspirated labialized clicks - Khoisan)
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Pentagraph('ŋ', 'ɡ', 'ǀ', 'ʰ', 'ʷ'))); // ŋɡǀʰʷ
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Pentagraph('ŋ', 'ɡ', 'ǃ', 'ʰ', 'ʷ'))); // ŋɡǃʰʷ
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Pentagraph('ŋ', 'ɡ', 'ǁ', 'ʰ', 'ʷ'))); // ŋɡǁʰʷ

    // Test hexagraphs (glottalized prenasalized voiced aspirated labialized clicks - Khoisan)
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Hexagraph('ŋ', 'ɡ', 'ǀ', 'ʰ', 'ʷ', 'ʼ'))); // ŋɡǀʰʷʼ
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Hexagraph('ŋ', 'ɡ', 'ǃ', 'ʰ', 'ʷ', 'ʼ'))); // ŋɡǃʰʷʼ
    assert!(prenasalized
        .patterns
        .contains(&PhonePattern::Hexagraph('ŋ', 'ɡ', 'ǁ', 'ʰ', 'ʷ', 'ʼ'))); // ŋɡǁʰʷʼ

    // Verify count: 6 trigraphs + 6 tetragraphs + 3 pentagraphs + 3 hexagraphs = 18 patterns
    assert_eq!(prenasalized.patterns.len(), 18);
}

#[test]
fn test_aspirated_affricate_class() {
    let aspirated =
        get_named_class("aspirated_affricate").expect("aspirated_affricate class should exist");

    // Test trigraphs for aspirated affricates
    assert!(aspirated
        .patterns
        .contains(&PhonePattern::Trigraph('t', 's', 'ʰ'))); // t͡sʰ - alveolar
    assert!(aspirated
        .patterns
        .contains(&PhonePattern::Trigraph('t', 'ʃ', 'ʰ'))); // t͡ʃʰ - postalveolar
    assert!(aspirated
        .patterns
        .contains(&PhonePattern::Trigraph('t', 'ɕ', 'ʰ'))); // t͡ɕʰ - alveo-palatal
    assert!(aspirated
        .patterns
        .contains(&PhonePattern::Trigraph('t', 'ʂ', 'ʰ'))); // t͡ʂʰ - retroflex

    // Verify count: 4 trigraph patterns
    assert_eq!(aspirated.patterns.len(), 4);
}

// =========================================================================
// N-Graph Pattern Tests (Pentagraph, Hexagraph, Heptagraph)
// =========================================================================

#[test]
fn test_phone_pattern_pentagraph() {
    let p = PhonePattern::Pentagraph('a', 'b', 'c', 'd', 'e');
    assert!(p.is_pentagraph());
    assert!(!p.is_char());
    assert!(!p.is_digraph());
    assert!(!p.is_trigraph());
    assert!(!p.is_tetragraph());
    assert!(!p.is_hexagraph());
    assert!(!p.is_heptagraph());
    assert_eq!(p.len(), 5);
    assert_eq!(p.as_pentagraph(), Some(('a', 'b', 'c', 'd', 'e')));
    assert!(p.matches_pentagraph('a', 'b', 'c', 'd', 'e'));
    assert!(!p.matches_pentagraph('a', 'b', 'c', 'd', 'f'));
    // Display
    assert_eq!(format!("{}", p), "abcde");
}

#[test]
fn test_phone_pattern_hexagraph() {
    let p = PhonePattern::Hexagraph('a', 'b', 'c', 'd', 'e', 'f');
    assert!(p.is_hexagraph());
    assert!(!p.is_char());
    assert!(!p.is_digraph());
    assert!(!p.is_trigraph());
    assert!(!p.is_tetragraph());
    assert!(!p.is_pentagraph());
    assert!(!p.is_heptagraph());
    assert_eq!(p.len(), 6);
    assert_eq!(p.as_hexagraph(), Some(('a', 'b', 'c', 'd', 'e', 'f')));
    assert!(p.matches_hexagraph('a', 'b', 'c', 'd', 'e', 'f'));
    assert!(!p.matches_hexagraph('a', 'b', 'c', 'd', 'e', 'g'));
    // Display
    assert_eq!(format!("{}", p), "abcdef");
}

#[test]
fn test_phone_pattern_heptagraph() {
    let p = PhonePattern::Heptagraph('a', 'b', 'c', 'd', 'e', 'f', 'g');
    assert!(p.is_heptagraph());
    assert!(!p.is_char());
    assert!(!p.is_digraph());
    assert!(!p.is_trigraph());
    assert!(!p.is_tetragraph());
    assert!(!p.is_pentagraph());
    assert!(!p.is_hexagraph());
    assert_eq!(p.len(), 7);
    assert_eq!(p.as_heptagraph(), Some(('a', 'b', 'c', 'd', 'e', 'f', 'g')));
    assert!(p.matches_heptagraph('a', 'b', 'c', 'd', 'e', 'f', 'g'));
    assert!(!p.matches_heptagraph('a', 'b', 'c', 'd', 'e', 'f', 'h'));
    // Display
    assert_eq!(format!("{}", p), "abcdefg");
}

#[test]
fn test_phone_pattern_helper_constructors() {
    // Test the helper constructor functions
    let penta = PhonePattern::pentagraph('p', 'e', 'n', 't', 'a');
    assert!(penta.is_pentagraph());
    assert_eq!(penta.len(), 5);

    let hexa = PhonePattern::hexagraph('h', 'e', 'x', 'a', 'g', 'r');
    assert!(hexa.is_hexagraph());
    assert_eq!(hexa.len(), 6);

    let hepta = PhonePattern::heptagraph('h', 'e', 'p', 't', 'a', 'g', 'r');
    assert!(hepta.is_heptagraph());
    assert_eq!(hepta.len(), 7);
}

#[test]
fn test_phone_pattern_sequence_for_long_patterns() {
    // Sequence should be used for 8+ character patterns
    let seq = PhonePattern::Sequence(vec!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']);
    assert!(!seq.is_char());
    assert!(!seq.is_pentagraph());
    assert!(!seq.is_hexagraph());
    assert!(!seq.is_heptagraph());
    assert_eq!(seq.len(), 8);
    // Display
    assert_eq!(format!("{}", seq), "abcdefgh");
}

#[test]
fn test_phone_pattern_ipa_complex_patterns() {
    // Test with realistic IPA complex patterns
    // Prenasalized labialized click: ŋɡǀʰʷ (5 chars)
    let prenasalized_labialized_click = PhonePattern::Pentagraph('ŋ', 'ɡ', 'ǀ', 'ʰ', 'ʷ');
    assert!(prenasalized_labialized_click.is_pentagraph());
    assert_eq!(prenasalized_labialized_click.len(), 5);

    // Prenasalized labialized ejective affricate: ⁿt͡sʷʼ (simplified to 6 chars)
    let complex_affricate = PhonePattern::Hexagraph('ⁿ', 't', 's', 'ʷ', 'ʼ', ' ');
    assert!(complex_affricate.is_hexagraph());
    assert_eq!(complex_affricate.len(), 6);
}
