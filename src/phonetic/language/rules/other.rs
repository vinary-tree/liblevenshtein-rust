//! Per-language rule aggregators for languages that do not cleanly fit one
//! of the broader family modules.
//!
//! Covers: Turkish, Hungarian, Finnish, Basque, Greek, Georgian, Armenian.

use crate::phonetic::types::RewriteRuleChar;

/// Get Turkish rule set.
///
/// Returns the complete phonetic normalization rules for Turkish:
/// - Special consonants (ş→S, ç→C, ğ→G)
/// - Dotted/undotted I distinction (ı→I, İ→i)
/// - Front vowels (ö→O, ü→U)
/// - Consonant transforms (c→dj, j→Z)
/// - Simplification rules for doubled markers
pub(crate) fn turkish_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::turkish;
    turkish::base().rules.clone()
}

/// Get Hungarian rule set.
///
/// Returns the complete phonetic normalization rules for Hungarian (Magyar):
/// - 9 digraphs treated as single letters:
///   - cs→CH, dz→DZ, dzs→DZS, gy→GY, ly→Y, ny→NY, sz→S, ty→TY, zs→ZS
/// - S alone → SH (unique Hungarian feature!)
/// - Long vowels with acute accent (á, é, í, ó, ú)
/// - Front rounded vowels (ö→OE, ü→UE)
/// - Double-acute vowels (ő→OE, ű→UE)
/// - Geminate digraphs (ccs→CH, ssz→S, etc.)
pub(crate) fn hungarian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::hungarian;
    hungarian::base().rules.clone()
}

/// Get Finnish rule set.
///
/// Returns the complete phonetic normalization rules for Finnish (Suomi):
/// - Front vowels: ä→AE, ö→OE, y→Y (front rounded)
/// - Vowel harmony: Front (ä, ö, y) vs back (a, o, u)
/// - Digraphs: ng→NG (velar nasal), nk→NK
/// - Nearly phonemic orthography
/// - Loanword consonant adaptations: b→p, d→t, g→k (in native words)
pub(crate) fn finnish_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::finnish;
    finnish::base().rules.clone()
}

/// Get Basque rule set.
///
/// Returns the complete phonetic normalization rules for Basque (Euskara):
/// - Digraphs: tx→CH, ts→TS, tz→TZ, tt→TT, dd→DD, rr→RR
/// - X → SH (like English "sh")
/// - Z → S (like English "s", NOT "z"!)
/// - Ñ → NY (palatal nasal)
/// - Language isolate with unique phonology
pub(crate) fn basque_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::basque;
    basque::base().rules.clone()
}

/// Get Greek rule set.
///
/// Returns the complete phonetic normalization rules for Greek (Ελληνικά):
/// - Greek alphabet (24 letters)
/// - Vowel digraphs: αι(e), ει(i), οι(i), υι(i), αυ(av/af), ευ(ev/ef), ου(u)
/// - Consonant combinations: μπ(b), ντ(d), γκ(g), γγ(ng)
/// - Accent marks stripped for matching
pub(crate) fn greek_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::greek;
    greek::base().rules.clone()
}

/// Get Georgian rule set.
///
/// Returns the complete phonetic normalization rules for Georgian (ქართული):
/// - Unique Mkhedruli script
/// - 33 letters, no uppercase/lowercase distinction
/// - Ejective consonants: k'(ყ), p'(პ), t'(ტ), ts'(წ), ch'(ჭ), q'(ყ)
/// - Nearly phonemic orthography
pub(crate) fn georgian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::georgian;
    georgian::base().rules.clone()
}

/// Get Armenian rule set.
///
/// Returns the complete phonetic normalization rules for Armenian (Hayeren):
/// - Unique Armenian script with uppercase/lowercase
/// - 39 letters (originally 36)
/// - Aspirated consonants: T, P, K
/// - Two dialects: Eastern vs Western Armenian
pub(crate) fn armenian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::armenian;
    armenian::base().rules.clone()
}
