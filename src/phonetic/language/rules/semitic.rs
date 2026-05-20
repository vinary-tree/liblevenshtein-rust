//! Per-language rule aggregators for Semitic languages (and Persian/Urdu,
//! which share heavy Arabic-script and lexical influence and are grouped here).
//!
//! Covers: Hebrew, Arabic, Urdu, Persian/Farsi, Maltese.

use crate::phonetic::types::RewriteRuleChar;

/// Get Hebrew rule set.
///
/// Returns the complete phonetic normalization rules for Hebrew:
/// - 22 consonant letters to Latin transliteration
/// - Final forms (ך, ם, ן, ף, ץ)
/// - Dagesh modifications (בּ→b, כּ→k, פּ→p)
/// - Shin/Sin distinction (שׁ→sh, שׂ→s)
/// - Niqqud vowel points
pub(crate) fn hebrew_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::hebrew;
    hebrew::base().rules.clone()
}

/// Get Arabic rule set.
///
/// Returns the complete phonetic normalization rules for Arabic:
/// - Basic consonants (ب→b, ت→t, ك→k, etc.)
/// - Special consonants (ث→TH, ذ→DH, ش→SH, خ→KH, غ→GH)
/// - Emphatic consonants (ص→S, ض→D, ط→T, ظ→Z)
/// - Pharyngeal consonants (ح→H, ع→E)
/// - Hamza variants (ء→', أ→a, إ→i, ؤ→u, ئ→i)
/// - Special forms (آ→A, ة→a, ى→a)
/// - Diacritics (harakat) handling
/// - Arabic-Indic numerals (٠-٩ → 0-9)
pub(crate) fn arabic_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::arabic;
    arabic::base().rules.clone()
}

/// Get Urdu rule set.
///
/// Returns the complete phonetic normalization rules for Urdu:
/// - Arabic-inherited consonants (ب→b, ت→t, ج→j, ح→H, خ→KH, etc.)
/// - Persian/Urdu additions (پ→p, چ→C, ژ→ZH, گ→g)
/// - Retroflex consonants (ٹ→TT, ڈ→DD, ڑ→RR)
/// - Special letters (ں→N, ے→e, ہ→h, ھ→h)
/// - Aspirated consonants (بھ→bh, پھ→ph, کھ→kh, گھ→gh)
/// - Diacritics (zabar, zer, pesh, jazm, tashdid, tanwin)
/// - Extended Arabic-Indic numerals (۰-۹ → 0-9)
pub(crate) fn urdu_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::urdu;
    urdu::base().rules.clone()
}

/// Get Persian/Farsi rule set.
///
/// Returns the complete phonetic normalization rules for Persian (فارسی):
/// - Arabic script with Persian additions
/// - Persian-specific letters: پ(p), چ(ch), ژ(zh), گ(g)
/// - Arabic consonants with Persian pronunciation differences
/// - Short vowel diacritics
pub(crate) fn persian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::persian;
    persian::base().rules.clone()
}

/// Get Maltese rule set.
///
/// Returns the complete phonetic normalization rules for Maltese (Malti):
/// - Latin script for Semitic language (unique!)
/// - Special letters: Ċ(ch), Ġ(j), Għ(silent), Ħ(h), Ż(z)
/// - X = sh sound, Q = glottal stop (often silent)
/// - IE diphthong
pub(crate) fn maltese_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::maltese;
    maltese::base().rules.clone()
}
