//! Per-language rule aggregators for Indic languages.
//!
//! Covers: Hindi, Marathi, Bengali, Gujarati, Telugu, Tamil,
//! Punjabi (Gurmukhi), Punjabi (Shahmukhi).

use crate::phonetic::types::RewriteRuleChar;

/// Get Hindi rule set.
///
/// Returns the complete phonetic normalization rules for Hindi (Devanagari):
/// - Independent vowels (अ→a, आ→A, इ→i, ई→I, उ→u, ऊ→U, ऋ→RI, etc.)
/// - Velar consonants (क→k, ख→kh, ग→g, घ→gh, ङ→N)
/// - Palatal consonants (च→c, छ→ch, ज→j, झ→jh, ञ→NY)
/// - Retroflex consonants (ट→TT, ठ→TTh, ड→DD, ढ→DDh, ण→NN)
/// - Dental consonants (त→t, थ→th, द→d, ध→dh, न→n)
/// - Labial consonants (प→p, फ→ph, ब→b, भ→bh, म→m)
/// - Semi-vowels (य→y, र→r, ल→l, व→v)
/// - Sibilants (श→SH, ष→SS, स→s, ह→h)
/// - Nukta consonants (क़→q, ख़→KH, ग़→GH, ज़→z, फ़→f, ड़→RR, ढ़→RRh)
/// - Vowel matras (ा→A, ि→i, ी→I, ु→u, ू→U, ृ→RI, े→e, ै→AI, ो→o, ौ→AU)
/// - Diacritics (virama ्, anusvara ं→M, chandrabindu ँ→M, visarga ः→H)
/// - Devanagari numerals (०-९ → 0-9)
pub(crate) fn hindi_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::hindi;
    hindi::base().rules.clone()
}

/// Get Marathi rule set.
///
/// Returns the complete phonetic normalization rules for Marathi (मराठी):
/// - Devanagari script (same as Hindi)
/// - Special ळ (retroflex lateral) unique to Marathi
/// - Schwa deletion patterns differ from Hindi
pub(crate) fn marathi_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::marathi;
    marathi::base().rules.clone()
}

/// Get Bengali rule set.
///
/// Returns the complete phonetic normalization rules for Bengali (বাংলা):
/// - Brahmic abugida script
/// - Inherent vowel is 'o' (not 'a' like Hindi!)
/// - Chandrabindu for nasalization
/// - Hasanta (virama) marks consonant clusters
pub(crate) fn bengali_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::bengali;
    bengali::base().rules.clone()
}

/// Get Gujarati rule set.
///
/// Returns the complete phonetic normalization rules for Gujarati (ગુજરાતી):
/// - Brahmic abugida derived from Devanagari
/// - No headline (shirorekha) unlike Devanagari
/// - Nukta for foreign sounds
pub(crate) fn gujarati_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::gujarati;
    gujarati::base().rules.clone()
}

/// Get Telugu rule set.
///
/// Returns the complete phonetic normalization rules for Telugu (తెలుగు):
/// - Brahmic abugida similar to Kannada
/// - 14 vowels, 36 consonants
/// - Sunna (anusvara) for nasalization
pub(crate) fn telugu_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::telugu;
    telugu::base().rules.clone()
}

/// Get Tamil rule set.
///
/// Returns the complete phonetic normalization rules for Tamil (தமிழ்):
/// - Unique Brahmic script (oldest Dravidian)
/// - Grantha letters for Sanskrit loanwords
/// - Unique letters: ழ (retroflex approximant), ற (alveolar trill), ன (alveolar nasal)
pub(crate) fn tamil_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::tamil;
    tamil::base().rules.clone()
}

/// Get Punjabi (Gurmukhi) rule set.
///
/// Returns the complete phonetic normalization rules for Punjabi in Gurmukhi script (ਪੰਜਾਬੀ):
/// - Brahmic abugida used in India
/// - Tonal language (3 tones, not marked in script)
/// - Nukta for foreign sounds
pub(crate) fn punjabi_gurmukhi_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::punjabi;
    punjabi::gurmukhi().rules.clone()
}

/// Get Punjabi (Shahmukhi) rule set.
///
/// Returns the complete phonetic normalization rules for Punjabi in Shahmukhi script:
/// - Arabic-derived script used in Pakistan
/// - Similar to Urdu script
/// - Right-to-left writing
pub(crate) fn punjabi_shahmukhi_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::punjabi;
    punjabi::shahmukhi().rules.clone()
}
