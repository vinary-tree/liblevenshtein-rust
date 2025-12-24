//! Chinese phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Chinese loaded from embedded `.llev` files.
//! These rules are parsed at first use and cached for subsequent calls.
//!
//! This module provides two rule sets:
//! - [`pinyin()`] - Pinyin romanization normalization (~45 rules)
//! - [`characters()`] - Hanzi character to Pinyin mapping (~400 rules)
//!
//! # Key Features
//!
//! ## Pinyin Normalization
//!
//! Chinese Pinyin phonetic normalization handles:
//! - **Tone marks**: ā á ǎ à → a (all four tones stripped)
//! - **Ü handling**: ü, v → U marker (common keyboard substitution)
//! - **Retroflex consonants**: zh→Z, ch→C, sh→S, r→R
//! - **Palatal consonants**: x→X, q→Q
//! - **Affricate c**: c→TS (alveolar affricate)
//!
//! ## Hanzi Character Mapping
//!
//! Chinese Hanzi character normalization handles:
//! - **HSK 1-4 characters**: ~400 most common characters
//! - **Character→Pinyin mapping**: 我→wo, 你→ni, 好→hao
//! - **Polyphonic characters**: Most common pronunciation used
//! - **Multi-character compounds**: 我们→women, 什么→shenme
//!
//! # Phonetic Markers
//!
//! Uses uppercase markers to avoid rule reprocessing:
//! - Z = retroflex zh (voiced affricate)
//! - C = retroflex ch (aspirated affricate)
//! - S = retroflex sh (fricative)
//! - R = retroflex r (approximant)
//! - X = palatal x (alveolo-palatal fricative)
//! - Q = palatal q (aspirated alveolo-palatal affricate)
//! - U = ü vowel (front rounded high vowel)
//! - TS = alveolar c (aspirated affricate)
//!
//! # Chinese Writing System
//!
//! Pinyin is the official romanization system for Standard Mandarin Chinese.
//! It uses Latin letters with optional diacritics (tone marks) to represent
//! Chinese pronunciation.
//!
//! Hanzi (汉字) is the logographic script used to write Chinese. Each character
//! represents a morpheme (meaningful unit) rather than a phoneme. This module
//! maps characters to their Pinyin pronunciation for phonetic matching.
//!
//! # Available Rule Sets
//!
//! - [`pinyin()`] - Chinese Pinyin phonetic rules (~45 rules)
//! - [`characters()`] - Chinese Hanzi character mappings (~400 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::chinese;
//!
//! let rules = chinese::pinyin();
//!
//! // Tone mark removal
//! let result = rules.apply("nǐ hǎo");
//! assert!(!result.contains('ǐ'), "tones should be stripped");
//!
//! // Retroflex consonants
//! let result = rules.apply("zhōngguó");
//! assert!(result.contains('Z'), "zh → Z");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Chinese Pinyin phonetic rules.
///
/// Complete phonetic normalization rules for Chinese Pinyin:
///
/// ## Tone Mark Removal (weight 0.05)
/// - First tone (macron): ā ē ī ō ū ǖ → a e i o u U
/// - Second tone (acute): á é í ó ú ǘ → a e i o u U
/// - Third tone (caron): ǎ ě ǐ ǒ ǔ ǚ → a e i o u U
/// - Fourth tone (grave): à è ì ò ù ǜ → a e i o u U
///
/// ## Ü Normalization (weight 0.1)
/// - ü → U (standard Pinyin)
/// - v → U (keyboard shortcut)
/// - nv → nU, lv → lU (common patterns)
///
/// ## Retroflex Consonants (weight 0.1)
/// - zh → Z, ch → C, sh → S, r → R
///
/// ## Palatal/Affricate Consonants (weight 0.1)
/// - x → X, q → Q, c → TS
pub fn pinyin() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/chinese/pinyin.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded chinese/pinyin.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Chinese pinyin rules - this is a bug in liblevenshtein")
    })
}

/// Chinese Hanzi character phonetic rules.
///
/// Maps Chinese characters (汉字) to their Pinyin pronunciations for
/// phonetic matching. This rule set covers the ~400 most common characters
/// organized by HSK (Hanyu Shuiping Kaoshi) proficiency levels.
///
/// ## Coverage
///
/// - **HSK 1** (~150 characters): Basic vocabulary
/// - **HSK 2** (~100 characters): Elementary vocabulary
/// - **HSK 3-4** (~100 characters): Intermediate vocabulary
/// - **Common high-frequency** (~50 characters): Additional common characters
///
/// ## Key Mappings
///
/// ### Pronouns
/// - 我→wo, 你→ni, 他/她/它→ta
/// - 我们→women, 你们→nimen, 他们→tamen
///
/// ### Numbers
/// - 一→yi, 二→er, 三→san, 四→si, 五→wu
/// - 六→liu, 七→qi, 八→ba, 九→jiu, 十→shi
///
/// ### Common Verbs
/// - 是→shi, 有→you, 在→zai, 去→qu, 来→lai
/// - 看→kan, 说→shuo, 听→ting, 做→zuo, 吃→chi
///
/// ### Common Nouns
/// - 人→ren, 家→jia, 学→xue, 中→zhong, 国→guo
///
/// ## Polyphony
///
/// For polyphonic characters (characters with multiple pronunciations),
/// the most common pronunciation is used. Examples:
/// - 行: xíng (walk) - more common than háng (row)
/// - 了: le (aspect marker) - more common than liǎo (finish)
/// - 还: hái (still) - more common than huán (return)
///
/// ## Use with Pinyin Rules
///
/// For best results with romanized input, use [`pinyin()`] rules.
/// For native Chinese character input, use this function.
pub fn characters() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/chinese/characters.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded chinese/characters.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Chinese character rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinyin_loads() {
        let rules = pinyin();
        assert!(!rules.is_empty(), "Chinese pinyin rules should not be empty");
        assert!(
            rules.len() > 35,
            "expected >35 pinyin rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_tone_mark_first() {
        let rules = pinyin();
        // First tone (macron)
        let result = rules.apply("ā");
        assert!(
            result.contains('a'),
            "ā should become a, got: {}",
            result
        );
        let result = rules.apply("ē");
        assert!(
            result.contains('e'),
            "ē should become e, got: {}",
            result
        );
    }

    #[test]
    fn test_tone_mark_second() {
        let rules = pinyin();
        // Second tone (acute accent)
        let result = rules.apply("á");
        assert!(
            result.contains('a'),
            "á should become a, got: {}",
            result
        );
        let result = rules.apply("é");
        assert!(
            result.contains('e'),
            "é should become e, got: {}",
            result
        );
    }

    #[test]
    fn test_tone_mark_third() {
        let rules = pinyin();
        // Third tone (caron)
        let result = rules.apply("ǎ");
        assert!(
            result.contains('a'),
            "ǎ should become a, got: {}",
            result
        );
        let result = rules.apply("ě");
        assert!(
            result.contains('e'),
            "ě should become e, got: {}",
            result
        );
    }

    #[test]
    fn test_tone_mark_fourth() {
        let rules = pinyin();
        // Fourth tone (grave accent)
        let result = rules.apply("à");
        assert!(
            result.contains('a'),
            "à should become a, got: {}",
            result
        );
        let result = rules.apply("è");
        assert!(
            result.contains('e'),
            "è should become e, got: {}",
            result
        );
    }

    #[test]
    fn test_u_umlaut() {
        let rules = pinyin();
        // ü → U
        let result = rules.apply("ü");
        assert!(
            result.contains('U'),
            "ü should become U, got: {}",
            result
        );
        // v → U (keyboard shortcut)
        let result = rules.apply("v");
        assert!(
            result.contains('U'),
            "v should become U, got: {}",
            result
        );
    }

    #[test]
    fn test_retroflex_consonants() {
        let rules = pinyin();
        // zh → Z
        let result = rules.apply("zh");
        assert!(
            result.contains('Z'),
            "zh should become Z, got: {}",
            result
        );
        // ch → C
        let result = rules.apply("ch");
        assert!(
            result.contains('C'),
            "ch should become C, got: {}",
            result
        );
        // sh → S
        let result = rules.apply("sh");
        assert!(
            result.contains('S'),
            "sh should become S, got: {}",
            result
        );
        // r → R
        let result = rules.apply("r");
        assert!(
            result.contains('R'),
            "r should become R, got: {}",
            result
        );
    }

    #[test]
    fn test_palatal_consonants() {
        let rules = pinyin();
        // x → X
        let result = rules.apply("x");
        assert!(
            result.contains('X'),
            "x should become X, got: {}",
            result
        );
        // q → Q
        let result = rules.apply("q");
        assert!(
            result.contains('Q'),
            "q should become Q, got: {}",
            result
        );
    }

    #[test]
    fn test_affricate_c() {
        let rules = pinyin();
        // c → TS
        let result = rules.apply("c");
        assert!(
            result.contains("TS"),
            "c should become TS, got: {}",
            result
        );
    }

    #[test]
    fn test_word_nihao() {
        let rules = pinyin();
        // nǐ hǎo (你好, hello)
        let result = rules.apply("nǐhǎo");
        // n stays, ǐ→i, h stays, ǎ→a, o stays
        assert!(
            result.contains('n') && result.contains('i') && result.contains('a') && result.contains('o'),
            "nǐhǎo should normalize to nihao-like, got: {}",
            result
        );
    }

    #[test]
    fn test_word_zhongguo() {
        let rules = pinyin();
        // zhōngguó (中国, China)
        let result = rules.apply("zhōngguó");
        // zh→Z, ō→o, ng stays, g stays, u stays, ó→o
        assert!(
            result.contains('Z') && result.contains('o'),
            "zhōngguó should have Z (from zh), got: {}",
            result
        );
    }

    #[test]
    fn test_word_nv() {
        let rules = pinyin();
        // nǚ (女, woman) - typed as nv on keyboards
        let result = rules.apply("nv");
        // nv→nU
        assert!(
            result.contains('n') && result.contains('U'),
            "nv should become nU, got: {}",
            result
        );
    }

    #[test]
    fn test_rules_sorted_by_weight() {
        let rules = pinyin();
        let weights: Vec<_> = rules.rules.iter().map(|r| r.weight).collect();
        let mut sorted_weights = weights.clone();
        sorted_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(weights, sorted_weights, "Rules should be sorted by weight");
    }

    // ============================================================
    // Character Rules Tests
    // ============================================================

    #[test]
    fn test_characters_loads() {
        let rules = characters();
        assert!(
            !rules.is_empty(),
            "Chinese character rules should not be empty"
        );
        assert!(
            rules.len() > 350,
            "expected >350 character rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_character_pronouns() {
        let rules = characters();
        // 我 → wo
        let result = rules.apply("我");
        assert!(
            result.contains("wo"),
            "我 should become wo, got: {}",
            result
        );
        // 你 → ni
        let result = rules.apply("你");
        assert!(
            result.contains("ni"),
            "你 should become ni, got: {}",
            result
        );
        // 他 → ta
        let result = rules.apply("他");
        assert!(
            result.contains("ta"),
            "他 should become ta, got: {}",
            result
        );
        // 她 → ta
        let result = rules.apply("她");
        assert!(
            result.contains("ta"),
            "她 should become ta, got: {}",
            result
        );
    }

    #[test]
    fn test_character_numbers() {
        let rules = characters();
        // 一 → yi
        let result = rules.apply("一");
        assert!(
            result.contains("yi"),
            "一 should become yi, got: {}",
            result
        );
        // 二 → er
        let result = rules.apply("二");
        assert!(
            result.contains("er"),
            "二 should become er, got: {}",
            result
        );
        // 三 → san
        let result = rules.apply("三");
        assert!(
            result.contains("san"),
            "三 should become san, got: {}",
            result
        );
        // 十 → shi
        let result = rules.apply("十");
        assert!(
            result.contains("shi"),
            "十 should become shi, got: {}",
            result
        );
    }

    #[test]
    fn test_character_common_verbs() {
        let rules = characters();
        // 是 → shi
        let result = rules.apply("是");
        assert!(
            result.contains("shi"),
            "是 should become shi, got: {}",
            result
        );
        // 有 → you
        let result = rules.apply("有");
        assert!(
            result.contains("you"),
            "有 should become you, got: {}",
            result
        );
        // 在 → zai
        let result = rules.apply("在");
        assert!(
            result.contains("zai"),
            "在 should become zai, got: {}",
            result
        );
        // 去 → qu
        let result = rules.apply("去");
        assert!(
            result.contains("qu"),
            "去 should become qu, got: {}",
            result
        );
    }

    #[test]
    fn test_character_common_nouns() {
        let rules = characters();
        // 人 → ren
        let result = rules.apply("人");
        assert!(
            result.contains("ren"),
            "人 should become ren, got: {}",
            result
        );
        // 家 → jia
        let result = rules.apply("家");
        assert!(
            result.contains("jia"),
            "家 should become jia, got: {}",
            result
        );
        // 学 → xue
        let result = rules.apply("学");
        assert!(
            result.contains("xue"),
            "学 should become xue, got: {}",
            result
        );
    }

    #[test]
    fn test_character_multi_char_compound() {
        let rules = characters();
        // 我们 → women
        let result = rules.apply("我们");
        assert!(
            result.contains("women"),
            "我们 should become women, got: {}",
            result
        );
        // 你们 → nimen
        let result = rules.apply("你们");
        assert!(
            result.contains("nimen"),
            "你们 should become nimen, got: {}",
            result
        );
    }

    #[test]
    fn test_character_sentence() {
        let rules = characters();
        // 你好 (nihao - hello)
        let result = rules.apply("你好");
        // Should become "nihao" through individual character mapping
        assert!(
            result.contains("ni") && result.contains("hao"),
            "你好 should contain ni and hao, got: {}",
            result
        );
    }

    #[test]
    fn test_character_china() {
        let rules = characters();
        // 中国 (zhongguo - China)
        let result = rules.apply("中国");
        assert!(
            result.contains("zhong") && result.contains("guo"),
            "中国 should contain zhong and guo, got: {}",
            result
        );
    }

    #[test]
    fn test_character_person() {
        let rules = characters();
        // 中国人 (zhongguoren - Chinese person)
        let result = rules.apply("中国人");
        assert!(
            result.contains("ren"),
            "中国人 should contain ren, got: {}",
            result
        );
    }

    #[test]
    fn test_character_rules_sorted_by_weight() {
        let rules = characters();
        let weights: Vec<_> = rules.rules.iter().map(|r| r.weight).collect();
        let mut sorted_weights = weights.clone();
        sorted_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            weights, sorted_weights,
            "Character rules should be sorted by weight"
        );
    }
}
