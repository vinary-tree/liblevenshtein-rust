//! Per-language rule aggregators for Southeast Asian languages.
//!
//! Covers: Thai, Vietnamese, Indonesian, Tagalog (Filipino).

use crate::phonetic::types::RewriteRuleChar;

/// Get Thai rule set.
///
/// Returns the complete phonetic normalization rules for Thai (ไทย):
/// - Brahmic-derived abugida
/// - No spaces between words
/// - 5 tones (partially marked)
/// - 44 consonants, vowels can appear before/after/above/below consonants
pub(crate) fn thai_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::thai;
    thai::base().rules.clone()
}

/// Get Vietnamese rule set.
///
/// Returns the complete phonetic normalization rules for Vietnamese (Tiếng Việt):
/// - Latin script with extensive diacritics
/// - 6 tones marked with diacritics (stripped for phonetic matching)
/// - Special vowels: ă, â, ê, ô, ơ, ư
/// - Special consonant: đ (voiced alveolar)
/// - Digraphs: ch, gh, gi, kh, ng, ngh, nh, ph, qu, th, tr
pub(crate) fn vietnamese_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::vietnamese;
    vietnamese::base().rules.clone()
}

/// Get Indonesian rule set.
///
/// Returns the complete phonetic normalization rules for Indonesian (Bahasa Indonesia):
/// - Digraphs: ng→NG (velar nasal), ny→NY (palatal nasal), sy→SH
/// - C → CH (like English "ch", not "k")
/// - KH → KH (voiceless velar fricative, Arabic loans)
/// - V → F (typically pronounced as F in Indonesian)
/// - Nearly phonemic Latin orthography
pub(crate) fn indonesian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::indonesian;
    indonesian::base().rules.clone()
}

/// Tagalog phonetic rules.
///
/// Returns rules for normalizing Tagalog/Filipino text.
///
/// # Key Features
///
/// - **NG digraph**: ng→NG (velar nasal)
/// - **Spanish loanword adaptations**: ll→ly, ñ→ny, qu→k
/// - **Borrowed consonant normalization**: f→p, v→b, z→s
/// - **Simple vowel system**: a, e, i, o, u
pub(crate) fn tagalog_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::tagalog;
    tagalog::base().rules.clone()
}
