//! Per-language rule aggregators for Germanic languages.
//!
//! Covers: American English, British English, German, Dutch, Swedish,
//! Norwegian, Danish, Icelandic.

use crate::phonetic::types::RewriteRuleChar;

/// Combine American English rule sets into a single vector.
///
/// This includes:
/// - Base orthographic rules (spelling normalization)
/// - American dialect rules (yod-dropping, t-flapping)
/// - Homophone rules (words that sound alike)
/// - Text-speak rules (SMS/text abbreviations)
pub(crate) fn american_english_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::english;

    let mut rules = Vec::new();
    rules.extend(english::base().rules.iter().cloned());
    rules.extend(english::american().rules.iter().cloned());
    rules.extend(english::homophones().rules.iter().cloned());
    rules.extend(english::text_speak().rules.iter().cloned());
    rules
}

/// Combine American English rules with broad CMUdict homophone coverage.
///
/// This includes [`american_english_rules`] plus the large CMUdict-derived
/// lexical homophone layer. Use this profile when recall over American
/// dictionary homophones is more important than the extra embedded rule-set
/// size and first-use parse cost.
pub(crate) fn american_english_cmudict_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::english;

    let mut rules = american_english_rules();
    rules.extend(english::cmudict_homophones().rules.iter().cloned());
    rules
}

/// Combine British English rule sets into a single vector.
///
/// This includes:
/// - Base orthographic rules (spelling normalization)
/// - British dialect rules (r-dropping, broad 'a' in BATH words)
/// - Homophone rules (words that sound alike)
/// - Text-speak rules (SMS/text abbreviations)
pub(crate) fn british_english_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::english;

    let mut rules = Vec::new();
    rules.extend(english::base().rules.iter().cloned());
    rules.extend(english::british().rules.iter().cloned());
    rules.extend(english::homophones().rules.iter().cloned());
    rules.extend(english::text_speak().rules.iter().cloned());
    rules
}

/// Get Standard German rule set.
///
/// Returns the complete phonetic normalization rules for German (Hochdeutsch):
/// - Umlaut normalization (ä→ae, ö→oe, ü→ue)
/// - Eszett handling (ß→ss)
/// - CH variations (ich-Laut, ach-Laut)
/// - Final devoicing (b→p, d→t, g→k)
/// - W/V pronunciation
/// - SP/ST patterns
pub(crate) fn german_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::german;
    german::base().rules.clone()
}

/// Get Standard Dutch rule set.
///
/// Returns the complete phonetic normalization rules for Dutch (Nederlands):
/// - IJ digraph handling (ij→EI)
/// - G/CH guttural sounds (→X)
/// - OE→U, EU→OE, UI→OY vowel patterns
/// - W→V approximant
/// - SCH→sX (not sh like German)
/// - Final devoicing (b→p, d→t)
pub(crate) fn dutch_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::dutch;
    dutch::base().rules.clone()
}

/// Get Swedish rule set.
///
/// Returns the complete phonetic normalization rules for Swedish (Svenska):
/// - Three extra vowels: å→O, ä→AE, ö→OE
/// - SJ-sound: sj/skj/stj→SJ (unique Swedish voiceless fricative)
/// - TJ-sound: tj/kj→TJ (voiceless palatal fricative)
/// - Silent consonant clusters: dj/gj/hj/lj→J
/// - G/K before front vowels: g→J, k→TJ
/// - Standard consonant normalizations
pub(crate) fn swedish_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::swedish;
    swedish::base().rules.clone()
}

/// Get Norwegian rule set.
///
/// Returns the complete phonetic normalization rules for Norwegian (Norsk):
/// - Three extra vowels: æ→AE, ø→OE, å→O
/// - KJ-sound: kj→KJ (voiceless palatal fricative)
/// - SJ-sound: sj/skj→SJ
/// - Silent consonant clusters: hv→v, hj→J
/// - Velar nasal: ng→NG
/// - Standard consonant normalizations
///
/// Note: Covers both Bokmål and Nynorsk spelling variants.
pub(crate) fn norwegian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::norwegian;
    norwegian::base().rules.clone()
}

/// Get Danish rule set.
///
/// Returns the complete phonetic normalization rules for Danish (Dansk):
/// - Three extra vowels: æ→AE, ø→OE, å→O
/// - Old spelling: aa→O (historical spelling for å)
/// - SJ-sound: sj→SJ
/// - Silent consonant clusters: hv→v, hj→J
/// - Velar nasal: ng→NG
/// - Standard consonant normalizations
///
/// Note: Stød (glottal stop) is not represented in spelling and thus not handled.
pub(crate) fn danish_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::danish;
    danish::base().rules.clone()
}

/// Get Icelandic rule set.
///
/// Returns the complete phonetic normalization rules for Icelandic (Íslenska):
/// - Archaic Norse orthography
/// - Unique letters: Þ/þ(th), Ð/ð(dh), Æ/æ(ai)
/// - Accented vowels indicate different sounds
/// - Consonant clusters: ll→tl, rl→rtl, hv→kv
pub(crate) fn icelandic_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::icelandic;
    icelandic::base().rules.clone()
}
