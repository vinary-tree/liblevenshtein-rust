//! Per-language rule aggregators for Slavic languages.
//!
//! Covers: Russian, Polish, Czech, Slovak, Croatian, Serbian (Cyrillic),
//! Serbian (Latin), Bulgarian, Belarusian, Ukrainian.

use crate::phonetic::types::RewriteRuleChar;

/// Get Standard Russian rule set.
///
/// Returns the complete phonetic normalization rules for Russian:
/// - Cyrillic to Latin transliteration
/// - Complex consonants (ж→zh, ш→sh, щ→shch, ч→ch, ц→ts, х→kh)
/// - Iotated vowels (е→ye, ё→yo, ю→yu, я→ya)
/// - Soft/hard sign removal
/// - Final devoicing
pub(crate) fn russian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::russian;
    russian::base().rules.clone()
}

/// Get Polish rule set.
///
/// Returns the complete phonetic normalization rules for Polish:
/// - Special Polish letters (ą, ę, ć, ś, ź, ż, ł, ó, ń)
/// - Nasal vowels (ą→on, ę→en)
/// - Digraphs (sz→sh, cz→ch, rz→zh, dz, dź→dj, dż→dzh, ch→kh)
/// - Ł→w, ó→u equivalence
/// - Polish consonant and vowel mappings
pub(crate) fn polish_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::polish;
    polish::base().rules.clone()
}

/// Get Czech rule set.
///
/// Returns the complete phonetic normalization rules for Czech (Čeština):
/// - Háčky (caron): č→CH, š→SH, ž→ZH, ř→RZH
/// - Soft consonants: ď→DJ, ť→TJ, ň→NJ
/// - Long vowels (čárka): á, é, í, ó, ú, ý → short equivalents
/// - Kroužek: ů→u
/// - Y/I merger: y→i
pub(crate) fn czech_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::czech;
    czech::base().rules.clone()
}

/// Get Slovak rule set.
///
/// Returns the complete phonetic normalization rules for Slovak (Slovenčina):
/// - Similar to Czech but NO ř
/// - Soft consonants: ď→DJ, ť→TJ, ň→NJ, ľ→LJ
/// - Diphthong: ô→UO
/// - Digraphs: dž→DZH, dz→DZ
/// - Long vowels (čárka): á, é, í, ó, ú, ý → short equivalents
pub(crate) fn slovak_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::slovak;
    slovak::base().rules.clone()
}

/// Get Croatian rule set.
///
/// Returns the complete phonetic normalization rules for Croatian (Hrvatski):
/// - Digraphs as letters: lj→LJ, nj→NJ, dž→DZH
/// - Diacritics: č→CH, ć→TJ, š→SH, ž→ZH, đ→DJ
/// - Perfect phonemic spelling
pub(crate) fn croatian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::croatian;
    croatian::base().rules.clone()
}

/// Get Serbian Cyrillic rule set.
///
/// Returns the complete phonetic normalization rules for Serbian Cyrillic (Српски језик):
/// - Perfect phonemic orthography: one letter = one sound
/// - Unique letters: Љ(lj), Њ(nj), Џ(dž), Ћ(ć), Ђ(đ)
/// - Standard Cyrillic consonant/vowel mappings
pub(crate) fn serbian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::serbian;
    serbian::base().rules.clone()
}

/// Get Serbian Latin rule set.
///
/// Returns the complete phonetic normalization rules for Serbian Latin (Srpski jezik):
/// - Perfect phonemic orthography: one letter = one sound
/// - Digraphs: lj, nj, dž (single phonemes)
/// - Diacritics: č, ć, š, ž, đ
pub(crate) fn serbian_latin_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::serbian;
    serbian::latin().rules.clone()
}

/// Get Bulgarian rule set.
///
/// Returns the complete phonetic normalization rules for Bulgarian (Български език):
/// - Щ = sht (unlike Russian shch!)
/// - Ъ = schwa vowel (very common, NOT a hard sign like Russian)
/// - No Ы, Ё, Э (unlike Russian)
/// - Standard Cyrillic consonant/vowel mappings
pub(crate) fn bulgarian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::bulgarian;
    bulgarian::base().rules.clone()
}

/// Get Belarusian rule set.
///
/// Returns the complete phonetic normalization rules for Belarusian (Беларуская мова):
/// - Unique Ў (short u/w) - only Belarusian has this!
/// - Г = h sound (not g like Russian)
/// - І instead of И (like Ukrainian)
/// - No Щ (uses ШЧ instead)
/// - Ё commonly used
pub(crate) fn belarusian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::belarusian;
    belarusian::base().rules.clone()
}

/// Get Ukrainian rule set.
///
/// Returns the complete phonetic normalization rules for Ukrainian (Cyrillic):
/// - Unique Ukrainian letters (ї→yi, є→ye, і→i, ґ→g)
/// - Ukrainian г→h (voiced glottal, NOT plosive g)
/// - Ukrainian и→y (sounds like Russian ы)
/// - Complex consonants (щ→shch, ж→zh, ш→sh, х→kh, ц→ts, ч→ch)
/// - Iotated vowels (ю→yu, я→ya)
/// - Soft sign removal
/// - Final devoicing
pub(crate) fn ukrainian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::ukrainian;
    ukrainian::base().rules.clone()
}
