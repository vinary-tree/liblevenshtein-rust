//! Per-language rule aggregators for CJK languages.
//!
//! Covers: Korean (Hangul), Korean Romanization, Japanese Romaji,
//! Chinese Pinyin, Chinese Hanzi.

use crate::phonetic::types::RewriteRuleChar;

/// Get Korean rule set.
///
/// Returns the complete phonetic normalization rules for Korean Hangul:
/// - Hangul jamo to Latin transliteration
/// - Double consonants (ㄲ→kk, ㄸ→tt, ㅃ→pp, ㅆ→ss, ㅉ→jj)
/// - Aspirated consonants (ㅋ→k, ㅌ→t, ㅍ→p, ㅊ→ch)
/// - Compound finals (ㄳ→ks, ㄵ→nj, etc.)
/// - Vowels and diphthongs
pub(crate) fn korean_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::korean;
    korean::base().rules.clone()
}

/// Get Korean Romanization rule set.
///
/// Returns the complete phonetic normalization rules for Korean romanization:
/// - McCune-Reischauer breve vowels (ŏ→eo, ŭ→eu)
/// - Aspirate markers (k'→k, t'→t, p'→p, ch'→ch)
/// - Long vowel macrons removal (ā→a, ē→e, etc.)
/// - Variant vowel digraphs (oe→we, ui→i)
pub(crate) fn korean_romanization_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::korean;
    korean::romanization().rules.clone()
}

/// Get Japanese Romaji rule set.
///
/// Returns the complete phonetic normalization rules for Japanese romanization:
/// - Long vowels (macrons: ā→A, ē→E, ī→I, ō→O, ū→U)
/// - Romanization variants (ti→C, tu→TS, si→S, hu→F)
/// - Digraphs (shi→S, chi→C, tsu→TS)
/// - Gemination (kk→K, pp→P, tt→T)
/// - Syllabic N (n'→N)
pub(crate) fn japanese_romaji_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::japanese;
    japanese::romaji().rules.clone()
}

/// Get Chinese Pinyin rule set.
///
/// Returns the complete phonetic normalization rules for Chinese Pinyin:
/// - Tone marks (ā á ǎ à → a, etc.)
/// - Ü handling (ü, v → U)
/// - Retroflex consonants (zh→Z, ch→C, sh→S, r→R)
/// - Palatal consonants (x→X, q→Q)
/// - Affricate c (c→TS)
pub(crate) fn chinese_pinyin_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::chinese;
    chinese::pinyin().rules.clone()
}

/// Chinese Hanzi phonetic rules.
///
/// Returns rules for normalizing Chinese characters (汉字) to Pinyin.
///
/// # Key Features
///
/// - **HSK 1-4 characters**: ~400 most common characters
/// - **Character→Pinyin mapping**: 我→wo, 你→ni, 好→hao
/// - **Pronouns**: 我→wo, 你→ni, 他/她/它→ta
/// - **Numbers**: 一→yi through 十→shi
/// - **Common verbs**: 是→shi, 有→you, 在→zai, 去→qu, 来→lai
/// - **Common nouns**: 人→ren, 家→jia, 学→xue, 中→zhong, 国→guo
/// - **Multi-character compounds**: 我们→women, 什么→shenme
pub(crate) fn chinese_hanzi_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::chinese;
    chinese::characters().rules.clone()
}
