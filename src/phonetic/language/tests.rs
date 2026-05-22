//! Tests for the language module.
//!
//! Verifies the public dispatch API ([`super::dispatch`]), the tag constants
//! ([`super::tags`]), and each per-language rule aggregator
//! ([`super::rules`]) returns a non-empty rule set.

use super::dispatch::{default_language, is_supported, rules_for_language, supported_languages};
use super::rules::celtic::{irish_rules, welsh_rules};
use super::rules::cjk::{
    chinese_hanzi_rules, chinese_pinyin_rules, japanese_romaji_rules, korean_romanization_rules,
    korean_rules,
};
use super::rules::germanic::{
    american_english_rules, british_english_rules, danish_rules, dutch_rules, german_rules,
    icelandic_rules, norwegian_rules, swedish_rules,
};
use super::rules::indic::{
    bengali_rules, gujarati_rules, hindi_rules, marathi_rules, punjabi_gurmukhi_rules,
    punjabi_shahmukhi_rules, tamil_rules, telugu_rules,
};
use super::rules::other::{
    armenian_rules, basque_rules, finnish_rules, georgian_rules, greek_rules, hungarian_rules,
    turkish_rules,
};
use super::rules::romance::{
    brazilian_portuguese_rules, canadian_french_rules, castilian_spanish_rules, catalan_rules,
    european_portuguese_rules, italian_rules, latin_american_spanish_rules, romanian_rules,
    standard_french_rules,
};
use super::rules::semitic::{arabic_rules, hebrew_rules, maltese_rules, persian_rules, urdu_rules};
use super::rules::slavic::{
    belarusian_rules, bulgarian_rules, croatian_rules, czech_rules, polish_rules, russian_rules,
    serbian_latin_rules, serbian_rules, slovak_rules, ukrainian_rules,
};
use super::rules::southeast_asian::{
    indonesian_rules, tagalog_rules, thai_rules, vietnamese_rules,
};
use super::tags;

#[test]
fn test_rules_for_language_en_us() {
    let rules = rules_for_language("en-us");
    assert!(rules.is_some(), "en-us should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "en-us rules should not be empty");
}

#[test]
fn test_rules_for_language_en() {
    let rules = rules_for_language("en");
    assert!(rules.is_some(), "en should be supported");
}

#[test]
fn test_rules_for_language_english() {
    let rules = rules_for_language("english");
    assert!(rules.is_some(), "english should be supported");
}

#[test]
fn test_rules_for_language_en_gb() {
    let rules = rules_for_language("en-gb");
    assert!(rules.is_some(), "en-gb should be supported");
}

#[test]
fn test_rules_for_language_case_insensitive() {
    assert!(
        rules_for_language("EN-US").is_some(),
        "EN-US should be supported (case insensitive)"
    );
    assert!(
        rules_for_language("En-Us").is_some(),
        "En-Us should be supported (case insensitive)"
    );
    assert!(
        rules_for_language("ENGLISH").is_some(),
        "ENGLISH should be supported (case insensitive)"
    );
}

#[test]
fn test_unsupported_language() {
    assert!(
        rules_for_language("xx-xx").is_none(),
        "xx-xx should not be supported"
    );
    assert!(
        rules_for_language("ja").is_none(),
        "ja (Japanese native scripts) should not be supported yet"
    );
    // Note: "zh" is now supported for Chinese Hanzi characters
    assert!(
        rules_for_language("").is_none(),
        "empty string should not be supported"
    );
}

#[test]
fn test_is_supported() {
    // English
    assert!(is_supported("en-us"));
    assert!(is_supported("en"));
    assert!(is_supported("en-gb"));
    assert!(is_supported("english"));
    assert!(is_supported("american"));
    assert!(is_supported("british"));

    // Spanish
    assert!(is_supported("es"));
    assert!(is_supported("es-419"));
    assert!(is_supported("spanish"));
    assert!(is_supported("castilian"));

    // French
    assert!(is_supported("fr"));
    assert!(is_supported("fr-ca"));
    assert!(is_supported("french"));
    assert!(is_supported("quebec"));

    // Portuguese
    assert!(is_supported("pt"));
    assert!(is_supported("pt-pt"));
    assert!(is_supported("pt-br"));
    assert!(is_supported("portuguese"));
    assert!(is_supported("brazilian"));

    // Italian
    assert!(is_supported("it"));
    assert!(is_supported("italian"));

    // German
    assert!(is_supported("de"));
    assert!(is_supported("german"));
    assert!(is_supported("deutsch"));

    // Dutch
    assert!(is_supported("nl"));
    assert!(is_supported("dutch"));
    assert!(is_supported("nederlands"));

    // Russian
    assert!(is_supported("ru"));
    assert!(is_supported("russian"));
    assert!(is_supported("русский"));

    // Korean
    assert!(is_supported("ko"));
    assert!(is_supported("korean"));
    assert!(is_supported("한국어"));

    // Korean Romanization
    assert!(is_supported("ko-latn"));
    assert!(is_supported("korean-romanization"));

    // Hebrew
    assert!(is_supported("he"));
    assert!(is_supported("hebrew"));
    assert!(is_supported("עברית"));

    // Polish
    assert!(is_supported("pl"));
    assert!(is_supported("polish"));
    assert!(is_supported("polski"));

    // Turkish
    assert!(is_supported("tr"));
    assert!(is_supported("turkish"));
    assert!(is_supported("türkçe"));

    // Japanese Romaji
    assert!(is_supported("ja-latn"));
    assert!(is_supported("romaji"));
    assert!(is_supported("ローマ字"));

    // Chinese Pinyin
    assert!(is_supported("zh-latn"));
    assert!(is_supported("pinyin"));
    assert!(is_supported("拼音"));

    // Arabic
    assert!(is_supported("ar"));
    assert!(is_supported("arabic"));
    assert!(is_supported("العربية"));

    // Urdu
    assert!(is_supported("ur"));
    assert!(is_supported("urdu"));
    assert!(is_supported("اردو"));

    // Hindi
    assert!(is_supported("hi"));
    assert!(is_supported("hindi"));
    assert!(is_supported("हिन्दी"));
    assert!(is_supported("devanagari"));

    // Chinese Hanzi
    assert!(is_supported("zh"));
    assert!(is_supported("chinese"));
    assert!(is_supported("hanzi"));
    assert!(is_supported("汉字"));
    assert!(is_supported("中文"));

    // Tagalog
    assert!(is_supported("tl"));
    assert!(is_supported("tagalog"));
    assert!(is_supported("filipino"));

    // Ukrainian
    assert!(is_supported("uk"));
    assert!(is_supported("ukrainian"));
    assert!(is_supported("українська"));

    // Hungarian
    assert!(is_supported("hu"));
    assert!(is_supported("hungarian"));
    assert!(is_supported("magyar"));

    // Indonesian
    assert!(is_supported("id"));
    assert!(is_supported("indonesian"));
    assert!(is_supported("bahasa indonesia"));

    // Romanian
    assert!(is_supported("ro"));
    assert!(is_supported("romanian"));
    assert!(is_supported("română"));

    // Finnish
    assert!(is_supported("fi"));
    assert!(is_supported("finnish"));
    assert!(is_supported("suomi"));

    // Basque
    assert!(is_supported("eu"));
    assert!(is_supported("basque"));
    assert!(is_supported("euskara"));

    // Catalan
    assert!(is_supported("ca"));
    assert!(is_supported("catalan"));
    assert!(is_supported("català"));

    // Still unsupported (native scripts for CJK)
    assert!(!is_supported("ja")); // native Japanese Hiragana/Katakana/Kanji
    assert!(!is_supported("xx"));
}

#[test]
fn test_supported_languages() {
    let languages = supported_languages();
    // English
    assert!(languages.contains(&"en"), "should contain 'en'");
    assert!(languages.contains(&"en-us"), "should contain 'en-us'");
    assert!(languages.contains(&"en-gb"), "should contain 'en-gb'");
    // Spanish
    assert!(languages.contains(&"es"), "should contain 'es'");
    assert!(languages.contains(&"es-419"), "should contain 'es-419'");
    // French
    assert!(languages.contains(&"fr"), "should contain 'fr'");
    assert!(languages.contains(&"fr-ca"), "should contain 'fr-ca'");
    // Portuguese
    assert!(languages.contains(&"pt"), "should contain 'pt'");
    assert!(languages.contains(&"pt-br"), "should contain 'pt-br'");
    // Italian
    assert!(languages.contains(&"it"), "should contain 'it'");
    // German
    assert!(languages.contains(&"de"), "should contain 'de'");
    assert!(languages.contains(&"german"), "should contain 'german'");
    // Dutch
    assert!(languages.contains(&"nl"), "should contain 'nl'");
    assert!(languages.contains(&"dutch"), "should contain 'dutch'");
    // Russian
    assert!(languages.contains(&"ru"), "should contain 'ru'");
    assert!(languages.contains(&"russian"), "should contain 'russian'");
    // Korean
    assert!(languages.contains(&"ko"), "should contain 'ko'");
    assert!(languages.contains(&"korean"), "should contain 'korean'");
    // Korean Romanization
    assert!(languages.contains(&"ko-latn"), "should contain 'ko-latn'");
    assert!(
        languages.contains(&"korean-romanization"),
        "should contain 'korean-romanization'"
    );
    // Hebrew
    assert!(languages.contains(&"he"), "should contain 'he'");
    assert!(languages.contains(&"hebrew"), "should contain 'hebrew'");
    // Polish
    assert!(languages.contains(&"pl"), "should contain 'pl'");
    assert!(languages.contains(&"polish"), "should contain 'polish'");
    // Turkish
    assert!(languages.contains(&"tr"), "should contain 'tr'");
    assert!(languages.contains(&"turkish"), "should contain 'turkish'");
    // Japanese Romaji
    assert!(languages.contains(&"ja-latn"), "should contain 'ja-latn'");
    assert!(languages.contains(&"romaji"), "should contain 'romaji'");
    // Chinese Pinyin
    assert!(languages.contains(&"zh-latn"), "should contain 'zh-latn'");
    assert!(languages.contains(&"pinyin"), "should contain 'pinyin'");
    // Arabic
    assert!(languages.contains(&"ar"), "should contain 'ar'");
    assert!(languages.contains(&"arabic"), "should contain 'arabic'");
    // Urdu
    assert!(languages.contains(&"ur"), "should contain 'ur'");
    assert!(languages.contains(&"urdu"), "should contain 'urdu'");
    // Hindi
    assert!(languages.contains(&"hi"), "should contain 'hi'");
    assert!(languages.contains(&"hindi"), "should contain 'hindi'");
    assert!(languages.contains(&"हिन्दी"), "should contain 'हिन्दी'");
    // Chinese Hanzi
    assert!(languages.contains(&"zh"), "should contain 'zh'");
    assert!(languages.contains(&"chinese"), "should contain 'chinese'");
    assert!(languages.contains(&"hanzi"), "should contain 'hanzi'");
    assert!(languages.contains(&"汉字"), "should contain '汉字'");
    assert!(languages.contains(&"中文"), "should contain '中文'");
    // Tagalog
    assert!(languages.contains(&"tl"), "should contain 'tl'");
    assert!(languages.contains(&"tagalog"), "should contain 'tagalog'");
    assert!(languages.contains(&"filipino"), "should contain 'filipino'");
    // Ukrainian
    assert!(languages.contains(&"uk"), "should contain 'uk'");
    assert!(
        languages.contains(&"ukrainian"),
        "should contain 'ukrainian'"
    );
    assert!(
        languages.contains(&"українська"),
        "should contain 'українська'"
    );
    // Hungarian
    assert!(languages.contains(&"hu"), "should contain 'hu'");
    assert!(
        languages.contains(&"hungarian"),
        "should contain 'hungarian'"
    );
    assert!(languages.contains(&"magyar"), "should contain 'magyar'");
    // Indonesian
    assert!(languages.contains(&"id"), "should contain 'id'");
    assert!(
        languages.contains(&"indonesian"),
        "should contain 'indonesian'"
    );
    assert!(
        languages.contains(&"bahasa indonesia"),
        "should contain 'bahasa indonesia'"
    );
    // Romanian
    assert!(languages.contains(&"ro"), "should contain 'ro'");
    assert!(languages.contains(&"romanian"), "should contain 'romanian'");
    assert!(languages.contains(&"română"), "should contain 'română'");
    // Finnish
    assert!(languages.contains(&"fi"), "should contain 'fi'");
    assert!(languages.contains(&"finnish"), "should contain 'finnish'");
    assert!(languages.contains(&"suomi"), "should contain 'suomi'");
    // Basque
    assert!(languages.contains(&"eu"), "should contain 'eu'");
    assert!(languages.contains(&"basque"), "should contain 'basque'");
    assert!(languages.contains(&"euskara"), "should contain 'euskara'");
    // Catalan
    assert!(languages.contains(&"ca"), "should contain 'ca'");
    assert!(languages.contains(&"catalan"), "should contain 'catalan'");
    assert!(languages.contains(&"català"), "should contain 'català'");
}

#[test]
fn test_default_language() {
    assert_eq!(default_language(), "en-us");
    // Default language should be supported
    assert!(is_supported(default_language()));
}

// ============================================================
// English Tests
// ============================================================

#[test]
fn test_american_english_rules_not_empty() {
    let rules = american_english_rules();
    assert!(
        !rules.is_empty(),
        "American English rules should not be empty"
    );
    // Should have rules from zompist + american + homophones + text_speak
    assert!(
        rules.len() > 100,
        "American rules should have >100 rules (zompist + american + more), got {}",
        rules.len()
    );
}

#[test]
fn test_british_english_rules_not_empty() {
    let rules = british_english_rules();
    assert!(
        !rules.is_empty(),
        "British English rules should not be empty"
    );
    // Should have rules from zompist + british + homophones + text_speak
    assert!(
        rules.len() > 150,
        "British rules should have >150 rules (zompist + british + more), got {}",
        rules.len()
    );
}

#[test]
fn test_english_dialects_have_different_rule_counts() {
    let american = american_english_rules();
    let british = british_english_rules();
    // British has more rules due to extensive BATH and r-dropping rules
    assert_ne!(
        american.len(),
        british.len(),
        "American and British should have different rule counts"
    );
}

// ============================================================
// Spanish Tests
// ============================================================

#[test]
fn test_castilian_spanish_rules_not_empty() {
    let rules = castilian_spanish_rules();
    assert!(
        !rules.is_empty(),
        "Castilian Spanish rules should not be empty"
    );
    assert!(
        rules.len() > 30,
        "Castilian rules should have >30 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_latin_american_spanish_rules_not_empty() {
    let rules = latin_american_spanish_rules();
    assert!(
        !rules.is_empty(),
        "Latin American Spanish rules should not be empty"
    );
    assert!(
        rules.len() > 30,
        "Latin American rules should have >30 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_spanish_dialects_both_exist() {
    let castilian = castilian_spanish_rules();
    let latin_american = latin_american_spanish_rules();
    // Both dialects should have rules
    assert!(!castilian.is_empty(), "Castilian rules should not be empty");
    assert!(
        !latin_american.is_empty(),
        "Latin American rules should not be empty"
    );
    // Rule counts may be the same or different depending on rule structure
}

// ============================================================
// French Tests
// ============================================================

#[test]
fn test_standard_french_rules_not_empty() {
    let rules = standard_french_rules();
    assert!(
        !rules.is_empty(),
        "Standard French rules should not be empty"
    );
    assert!(
        rules.len() > 60,
        "Standard French rules should have >60 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_canadian_french_rules_not_empty() {
    let rules = canadian_french_rules();
    assert!(
        !rules.is_empty(),
        "Canadian French rules should not be empty"
    );
    assert!(
        rules.len() > 60,
        "Canadian French rules should have >60 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_french_dialects_differ() {
    let standard = standard_french_rules();
    let canadian = canadian_french_rules();
    // Canadian has affrication rules that standard doesn't
    assert_ne!(
        standard.len(),
        canadian.len(),
        "Standard and Canadian French should have different rule counts"
    );
}

// ============================================================
// Portuguese Tests
// ============================================================

#[test]
fn test_european_portuguese_rules_not_empty() {
    let rules = european_portuguese_rules();
    assert!(
        !rules.is_empty(),
        "European Portuguese rules should not be empty"
    );
    assert!(
        rules.len() > 50,
        "European Portuguese rules should have >50 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_brazilian_portuguese_rules_not_empty() {
    let rules = brazilian_portuguese_rules();
    assert!(
        !rules.is_empty(),
        "Brazilian Portuguese rules should not be empty"
    );
    assert!(
        rules.len() > 50,
        "Brazilian Portuguese rules should have >50 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_portuguese_dialects_differ() {
    let european = european_portuguese_rules();
    let brazilian = brazilian_portuguese_rules();
    // They should have different rule counts due to different features
    assert_ne!(
        european.len(),
        brazilian.len(),
        "European and Brazilian Portuguese should have different rule counts"
    );
}

// ============================================================
// Italian Tests
// ============================================================

#[test]
fn test_italian_rules_not_empty() {
    let rules = italian_rules();
    assert!(!rules.is_empty(), "Italian rules should not be empty");
    assert!(
        rules.len() > 40,
        "Italian rules should have >40 rules, got {}",
        rules.len()
    );
}

// ============================================================
// Tag Constants Tests
// ============================================================

#[test]
fn test_tags_constants() {
    // English
    assert_eq!(tags::ENGLISH, "en");
    assert_eq!(tags::ENGLISH_US, "en-us");
    assert_eq!(tags::ENGLISH_GB, "en-gb");

    // Spanish
    assert_eq!(tags::SPANISH, "es");
    assert_eq!(tags::SPANISH_419, "es-419");
    assert_eq!(tags::SPANISH_MX, "es-mx");
    assert_eq!(tags::SPANISH_AR, "es-ar");

    // French
    assert_eq!(tags::FRENCH, "fr");
    assert_eq!(tags::FRENCH_CA, "fr-ca");

    // Portuguese
    assert_eq!(tags::PORTUGUESE, "pt");
    assert_eq!(tags::PORTUGUESE_PT, "pt-pt");
    assert_eq!(tags::PORTUGUESE_BR, "pt-br");

    // Italian
    assert_eq!(tags::ITALIAN, "it");

    // German
    assert_eq!(tags::GERMAN, "de");

    // Dutch
    assert_eq!(tags::DUTCH, "nl");

    // Russian
    assert_eq!(tags::RUSSIAN, "ru");

    // Korean
    assert_eq!(tags::KOREAN, "ko");
    assert_eq!(tags::KOREAN_ROMANIZATION, "ko-latn");

    // Hebrew
    assert_eq!(tags::HEBREW, "he");

    // Turkish
    assert_eq!(tags::TURKISH, "tr");

    // Japanese Romaji
    assert_eq!(tags::JAPANESE_ROMAJI, "ja-latn");

    // Chinese Pinyin
    assert_eq!(tags::CHINESE_PINYIN, "zh-latn");

    // Arabic
    assert_eq!(tags::ARABIC, "ar");

    // Urdu
    assert_eq!(tags::URDU, "ur");

    // Hindi
    assert_eq!(tags::HINDI, "hi");

    // Ukrainian
    assert_eq!(tags::UKRAINIAN, "uk");

    // Hungarian
    assert_eq!(tags::HUNGARIAN, "hu");

    // Indonesian
    assert_eq!(tags::INDONESIAN, "id");

    // Romanian
    assert_eq!(tags::ROMANIAN, "ro");

    // Finnish
    assert_eq!(tags::FINNISH, "fi");

    // Basque
    assert_eq!(tags::BASQUE, "eu");

    // Catalan
    assert_eq!(tags::CATALAN, "ca");

    // All constants should be supported
    assert!(is_supported(tags::ENGLISH));
    assert!(is_supported(tags::ENGLISH_US));
    assert!(is_supported(tags::ENGLISH_GB));
    assert!(is_supported(tags::SPANISH));
    assert!(is_supported(tags::SPANISH_419));
    assert!(is_supported(tags::SPANISH_MX));
    assert!(is_supported(tags::SPANISH_AR));
    assert!(is_supported(tags::FRENCH));
    assert!(is_supported(tags::FRENCH_CA));
    assert!(is_supported(tags::PORTUGUESE));
    assert!(is_supported(tags::PORTUGUESE_PT));
    assert!(is_supported(tags::PORTUGUESE_BR));
    assert!(is_supported(tags::ITALIAN));
    assert!(is_supported(tags::GERMAN));
    assert!(is_supported(tags::DUTCH));
    assert!(is_supported(tags::RUSSIAN));
    assert!(is_supported(tags::KOREAN));
    assert!(is_supported(tags::KOREAN_ROMANIZATION));
    assert!(is_supported(tags::HEBREW));
    assert!(is_supported(tags::POLISH));
    assert!(is_supported(tags::TURKISH));
    assert!(is_supported(tags::JAPANESE_ROMAJI));
    assert!(is_supported(tags::CHINESE_PINYIN));
    assert!(is_supported(tags::ARABIC));
    assert!(is_supported(tags::URDU));
    assert!(is_supported(tags::HINDI));
    assert!(is_supported(tags::CHINESE_HANZI));
    assert!(is_supported(tags::TAGALOG));
    assert!(is_supported(tags::UKRAINIAN));
    assert!(is_supported(tags::HUNGARIAN));
    assert!(is_supported(tags::INDONESIAN));
    assert!(is_supported(tags::ROMANIAN));
    assert!(is_supported(tags::FINNISH));
    assert!(is_supported(tags::BASQUE));
    assert!(is_supported(tags::CATALAN));
}

// ============================================================
// German Tests
// ============================================================

#[test]
fn test_german_rules_not_empty() {
    let rules = german_rules();
    assert!(!rules.is_empty(), "German rules should not be empty");
    assert!(
        rules.len() > 40,
        "German rules should have >40 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_german_via_rules_for_language() {
    let rules = rules_for_language("de");
    assert!(rules.is_some(), "de should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "German rules should not be empty");
}

#[test]
fn test_german_alias_deutsch() {
    let rules = rules_for_language("deutsch");
    assert!(rules.is_some(), "deutsch should be supported");
}

// ============================================================
// Dutch Tests
// ============================================================

#[test]
fn test_dutch_rules_not_empty() {
    let rules = dutch_rules();
    assert!(!rules.is_empty(), "Dutch rules should not be empty");
    assert!(
        rules.len() > 40,
        "Dutch rules should have >40 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_dutch_via_rules_for_language() {
    let rules = rules_for_language("nl");
    assert!(rules.is_some(), "nl should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Dutch rules should not be empty");
}

#[test]
fn test_dutch_alias_nederlands() {
    let rules = rules_for_language("nederlands");
    assert!(rules.is_some(), "nederlands should be supported");
}

#[test]
fn test_dutch_alias_flemish() {
    let rules = rules_for_language("flemish");
    assert!(rules.is_some(), "flemish should be supported");
}

// ============================================================
// Russian Tests
// ============================================================

#[test]
fn test_russian_rules_not_empty() {
    let rules = russian_rules();
    assert!(!rules.is_empty(), "Russian rules should not be empty");
    assert!(
        rules.len() > 50,
        "Russian rules should have >50 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_russian_via_rules_for_language() {
    let rules = rules_for_language("ru");
    assert!(rules.is_some(), "ru should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Russian rules should not be empty");
}

#[test]
fn test_russian_alias_russian() {
    let rules = rules_for_language("russian");
    assert!(rules.is_some(), "russian should be supported");
}

#[test]
fn test_russian_alias_cyrillic() {
    let rules = rules_for_language("русский");
    assert!(rules.is_some(), "русский should be supported");
}

// ============================================================
// Korean Tests
// ============================================================

#[test]
fn test_korean_rules_not_empty() {
    let rules = korean_rules();
    assert!(!rules.is_empty(), "Korean rules should not be empty");
    assert!(
        rules.len() > 35,
        "Korean rules should have >35 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_korean_via_rules_for_language() {
    let rules = rules_for_language("ko");
    assert!(rules.is_some(), "ko should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Korean rules should not be empty");
}

#[test]
fn test_korean_alias_korean() {
    let rules = rules_for_language("korean");
    assert!(rules.is_some(), "korean should be supported");
}

#[test]
fn test_korean_alias_hangul() {
    let rules = rules_for_language("한국어");
    assert!(rules.is_some(), "한국어 should be supported");
}

// ============================================================
// Hebrew Tests
// ============================================================

#[test]
fn test_hebrew_rules_not_empty() {
    let rules = hebrew_rules();
    assert!(!rules.is_empty(), "Hebrew rules should not be empty");
    assert!(
        rules.len() > 35,
        "Hebrew rules should have >35 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_hebrew_via_rules_for_language() {
    let rules = rules_for_language("he");
    assert!(rules.is_some(), "he should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Hebrew rules should not be empty");
}

#[test]
fn test_hebrew_alias_hebrew() {
    let rules = rules_for_language("hebrew");
    assert!(rules.is_some(), "hebrew should be supported");
}

#[test]
fn test_hebrew_alias_ivrit() {
    let rules = rules_for_language("עברית");
    assert!(rules.is_some(), "עברית should be supported");
}

// ============================================================
// Polish Tests
// ============================================================

#[test]
fn test_polish_rules_not_empty() {
    let rules = polish_rules();
    assert!(!rules.is_empty(), "Polish rules should not be empty");
    assert!(
        rules.len() > 30,
        "Polish rules should have >30 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_polish_via_rules_for_language() {
    let rules = rules_for_language("pl");
    assert!(rules.is_some(), "pl should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Polish rules should not be empty");
}

#[test]
fn test_polish_alias_polish() {
    let rules = rules_for_language("polish");
    assert!(rules.is_some(), "polish should be supported");
}

#[test]
fn test_polish_alias_polski() {
    let rules = rules_for_language("polski");
    assert!(rules.is_some(), "polski should be supported");
}

// ============================================================
// Turkish Tests
// ============================================================

#[test]
fn test_turkish_rules_not_empty() {
    let rules = turkish_rules();
    assert!(!rules.is_empty(), "Turkish rules should not be empty");
    assert!(
        rules.len() > 20,
        "Turkish rules should have >20 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_turkish_via_rules_for_language() {
    let rules = rules_for_language("tr");
    assert!(rules.is_some(), "tr should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Turkish rules should not be empty");
}

#[test]
fn test_turkish_alias_turkish() {
    let rules = rules_for_language("turkish");
    assert!(rules.is_some(), "turkish should be supported");
}

#[test]
fn test_turkish_alias_turkce() {
    let rules = rules_for_language("türkçe");
    assert!(rules.is_some(), "türkçe should be supported");
}

// ============================================================
// Japanese Tests
// ============================================================

#[test]
fn test_japanese_romaji_rules_not_empty() {
    let rules = japanese_romaji_rules();
    assert!(
        !rules.is_empty(),
        "Japanese romaji rules should not be empty"
    );
    assert!(
        rules.len() > 35,
        "Japanese romaji rules should have >35 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_japanese_via_rules_for_language() {
    let rules = rules_for_language("ja-latn");
    assert!(rules.is_some(), "ja-latn should be supported");
    let rules = rules.expect("should have rules");
    assert!(
        !rules.is_empty(),
        "Japanese romaji rules should not be empty"
    );
}

#[test]
fn test_japanese_alias_romaji() {
    let rules = rules_for_language("romaji");
    assert!(rules.is_some(), "romaji should be supported");
}

#[test]
fn test_japanese_alias_native() {
    let rules = rules_for_language("ローマ字");
    assert!(rules.is_some(), "ローマ字 should be supported");
}

// ============================================================
// Chinese Tests
// ============================================================

#[test]
fn test_chinese_pinyin_rules_not_empty() {
    let rules = chinese_pinyin_rules();
    assert!(
        !rules.is_empty(),
        "Chinese pinyin rules should not be empty"
    );
    assert!(
        rules.len() > 35,
        "Chinese pinyin rules should have >35 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_chinese_via_rules_for_language() {
    let rules = rules_for_language("zh-latn");
    assert!(rules.is_some(), "zh-latn should be supported");
    let rules = rules.expect("should have rules");
    assert!(
        !rules.is_empty(),
        "Chinese pinyin rules should not be empty"
    );
}

#[test]
fn test_chinese_alias_pinyin() {
    let rules = rules_for_language("pinyin");
    assert!(rules.is_some(), "pinyin should be supported");
}

#[test]
fn test_chinese_alias_native() {
    let rules = rules_for_language("拼音");
    assert!(rules.is_some(), "拼音 should be supported");
}

// ============================================================
// Arabic Tests
// ============================================================

#[test]
fn test_arabic_rules_not_empty() {
    let rules = arabic_rules();
    assert!(!rules.is_empty(), "Arabic rules should not be empty");
    assert!(
        rules.len() > 50,
        "Arabic rules should have >50 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_arabic_via_rules_for_language() {
    let rules = rules_for_language("ar");
    assert!(rules.is_some(), "ar should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Arabic rules should not be empty");
}

#[test]
fn test_arabic_alias_arabic() {
    let rules = rules_for_language("arabic");
    assert!(rules.is_some(), "arabic should be supported");
}

#[test]
fn test_arabic_alias_native() {
    let rules = rules_for_language("العربية");
    assert!(rules.is_some(), "العربية should be supported");
}

// ============================================================
// Urdu Tests
// ============================================================

#[test]
fn test_urdu_rules_not_empty() {
    let rules = urdu_rules();
    assert!(!rules.is_empty(), "Urdu rules should not be empty");
    assert!(
        rules.len() > 60,
        "Urdu rules should have >60 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_urdu_via_rules_for_language() {
    let rules = rules_for_language("ur");
    assert!(rules.is_some(), "ur should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Urdu rules should not be empty");
}

#[test]
fn test_urdu_alias_urdu() {
    let rules = rules_for_language("urdu");
    assert!(rules.is_some(), "urdu should be supported");
}

#[test]
fn test_urdu_alias_native() {
    let rules = rules_for_language("اردو");
    assert!(rules.is_some(), "اردو should be supported");
}

// ============================================================
// Hindi Tests
// ============================================================

#[test]
fn test_hindi_rules_not_empty() {
    let rules = hindi_rules();
    assert!(!rules.is_empty(), "Hindi rules should not be empty");
    assert!(
        rules.len() > 65,
        "Hindi rules should have >65 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_hindi_via_rules_for_language() {
    let rules = rules_for_language("hi");
    assert!(rules.is_some(), "hi should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Hindi rules should not be empty");
}

#[test]
fn test_hindi_alias_hindi() {
    let rules = rules_for_language("hindi");
    assert!(rules.is_some(), "hindi should be supported");
}

#[test]
fn test_hindi_alias_native() {
    let rules = rules_for_language("हिन्दी");
    assert!(rules.is_some(), "हिन्दी should be supported");
}

#[test]
fn test_hindi_alias_devanagari() {
    let rules = rules_for_language("devanagari");
    assert!(rules.is_some(), "devanagari should be supported");
}

// ============================================================
// Chinese Hanzi Tests
// ============================================================

#[test]
fn test_chinese_hanzi_rules_not_empty() {
    let rules = chinese_hanzi_rules();
    assert!(!rules.is_empty(), "Chinese Hanzi rules should not be empty");
    assert!(
        rules.len() > 350,
        "Chinese Hanzi rules should have >350 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_chinese_hanzi_via_rules_for_language() {
    let rules = rules_for_language("zh");
    assert!(rules.is_some(), "zh should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Chinese Hanzi rules should not be empty");
}

#[test]
fn test_chinese_hanzi_alias_chinese() {
    let rules = rules_for_language("chinese");
    assert!(rules.is_some(), "chinese should be supported");
}

#[test]
fn test_chinese_hanzi_alias_hanzi() {
    let rules = rules_for_language("hanzi");
    assert!(rules.is_some(), "hanzi should be supported");
}

#[test]
fn test_chinese_hanzi_alias_native_hanzi() {
    let rules = rules_for_language("汉字");
    assert!(rules.is_some(), "汉字 should be supported");
}

#[test]
fn test_chinese_hanzi_alias_native_zhongwen() {
    let rules = rules_for_language("中文");
    assert!(rules.is_some(), "中文 should be supported");
}

// ============================================================
// Tagalog Tests
// ============================================================

#[test]
fn test_tagalog_rules_not_empty() {
    let rules = tagalog_rules();
    assert!(!rules.is_empty(), "Tagalog rules should not be empty");
    // ~25 rules: digraphs, Spanish adaptations, borrowed consonants, glottal, simplification
    assert!(
        rules.len() >= 20,
        "Tagalog rules should have >=20 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_tagalog_via_rules_for_language() {
    let rules = rules_for_language("tl");
    assert!(rules.is_some(), "tl should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Tagalog rules should not be empty");
}

#[test]
fn test_tagalog_alias_tagalog() {
    let rules = rules_for_language("tagalog");
    assert!(rules.is_some(), "tagalog should be supported");
}

#[test]
fn test_tagalog_alias_filipino() {
    let rules = rules_for_language("filipino");
    assert!(rules.is_some(), "filipino should be supported");
}

// ============================================================
// Korean Romanization Tests
// ============================================================

#[test]
fn test_korean_romanization_rules_not_empty() {
    let rules = korean_romanization_rules();
    assert!(
        !rules.is_empty(),
        "Korean romanization rules should not be empty"
    );
    assert!(
        rules.len() >= 25,
        "Korean romanization rules should have >=25 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_korean_romanization_via_rules_for_language() {
    let rules = rules_for_language("ko-latn");
    assert!(rules.is_some(), "ko-latn should be supported");
    let rules = rules.expect("should have rules");
    assert!(
        !rules.is_empty(),
        "Korean romanization rules should not be empty"
    );
}

#[test]
fn test_korean_romanization_alias() {
    let rules = rules_for_language("korean-romanization");
    assert!(rules.is_some(), "korean-romanization should be supported");
}

// ============================================================
// Ukrainian Tests
// ============================================================

#[test]
fn test_ukrainian_rules_not_empty() {
    let rules = ukrainian_rules();
    assert!(!rules.is_empty(), "Ukrainian rules should not be empty");
    assert!(
        rules.len() >= 40,
        "Ukrainian rules should have >=40 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_ukrainian_via_rules_for_language() {
    let rules = rules_for_language("uk");
    assert!(rules.is_some(), "uk should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Ukrainian rules should not be empty");
}

#[test]
fn test_ukrainian_alias_ukrainian() {
    let rules = rules_for_language("ukrainian");
    assert!(rules.is_some(), "ukrainian should be supported");
}

#[test]
fn test_ukrainian_alias_native() {
    let rules = rules_for_language("українська");
    assert!(rules.is_some(), "українська should be supported");
}

// ============================================================
// Hungarian Tests
// ============================================================

#[test]
fn test_hungarian_rules_not_empty() {
    let rules = hungarian_rules();
    assert!(!rules.is_empty(), "Hungarian rules should not be empty");
    assert!(
        rules.len() >= 50,
        "Hungarian rules should have >=50 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_hungarian_via_rules_for_language() {
    let rules = rules_for_language("hu");
    assert!(rules.is_some(), "hu should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Hungarian rules should not be empty");
}

#[test]
fn test_hungarian_alias_hungarian() {
    let rules = rules_for_language("hungarian");
    assert!(rules.is_some(), "hungarian should be supported");
}

#[test]
fn test_hungarian_alias_magyar() {
    let rules = rules_for_language("magyar");
    assert!(rules.is_some(), "magyar should be supported");
}

// ============================================================
// Indonesian Tests
// ============================================================

#[test]
fn test_indonesian_rules_not_empty() {
    let rules = indonesian_rules();
    assert!(!rules.is_empty(), "Indonesian rules should not be empty");
    assert!(
        rules.len() >= 5,
        "Indonesian rules should have >=5 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_indonesian_via_rules_for_language() {
    let rules = rules_for_language("id");
    assert!(rules.is_some(), "id should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Indonesian rules should not be empty");
}

#[test]
fn test_indonesian_alias_indonesian() {
    let rules = rules_for_language("indonesian");
    assert!(rules.is_some(), "indonesian should be supported");
}

#[test]
fn test_indonesian_alias_bahasa() {
    let rules = rules_for_language("bahasa indonesia");
    assert!(rules.is_some(), "bahasa indonesia should be supported");
}

// ============================================================
// Romanian Tests
// ============================================================

#[test]
fn test_romanian_rules_not_empty() {
    let rules = romanian_rules();
    assert!(!rules.is_empty(), "Romanian rules should not be empty");
    assert!(
        rules.len() >= 10,
        "Romanian rules should have >=10 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_romanian_via_rules_for_language() {
    let rules = rules_for_language("ro");
    assert!(rules.is_some(), "ro should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Romanian rules should not be empty");
}

#[test]
fn test_romanian_alias_romanian() {
    let rules = rules_for_language("romanian");
    assert!(rules.is_some(), "romanian should be supported");
}

#[test]
fn test_romanian_alias_native() {
    let rules = rules_for_language("română");
    assert!(rules.is_some(), "română should be supported");
}

// ============================================================
// Finnish Tests
// ============================================================

#[test]
fn test_finnish_rules_not_empty() {
    let rules = finnish_rules();
    assert!(!rules.is_empty(), "Finnish rules should not be empty");
    assert!(
        rules.len() >= 15,
        "Finnish rules should have >=15 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_finnish_via_rules_for_language() {
    let rules = rules_for_language("fi");
    assert!(rules.is_some(), "fi should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Finnish rules should not be empty");
}

#[test]
fn test_finnish_alias_finnish() {
    let rules = rules_for_language("finnish");
    assert!(rules.is_some(), "finnish should be supported");
}

#[test]
fn test_finnish_alias_suomi() {
    let rules = rules_for_language("suomi");
    assert!(rules.is_some(), "suomi should be supported");
}

// ============================================================
// Basque Tests
// ============================================================

#[test]
fn test_basque_rules_not_empty() {
    let rules = basque_rules();
    assert!(!rules.is_empty(), "Basque rules should not be empty");
    assert!(
        rules.len() >= 10,
        "Basque rules should have >=10 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_basque_via_rules_for_language() {
    let rules = rules_for_language("eu");
    assert!(rules.is_some(), "eu should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Basque rules should not be empty");
}

#[test]
fn test_basque_alias_basque() {
    let rules = rules_for_language("basque");
    assert!(rules.is_some(), "basque should be supported");
}

#[test]
fn test_basque_alias_euskara() {
    let rules = rules_for_language("euskara");
    assert!(rules.is_some(), "euskara should be supported");
}

// ============================================================
// Catalan Tests
// ============================================================

#[test]
fn test_catalan_rules_not_empty() {
    let rules = catalan_rules();
    assert!(!rules.is_empty(), "Catalan rules should not be empty");
    assert!(
        rules.len() >= 20,
        "Catalan rules should have >=20 rules, got {}",
        rules.len()
    );
}

#[test]
fn test_catalan_via_rules_for_language() {
    let rules = rules_for_language("ca");
    assert!(rules.is_some(), "ca should be supported");
    let rules = rules.expect("should have rules");
    assert!(!rules.is_empty(), "Catalan rules should not be empty");
}

#[test]
fn test_catalan_alias_catalan() {
    let rules = rules_for_language("catalan");
    assert!(rules.is_some(), "catalan should be supported");
}

#[test]
fn test_catalan_alias_catala() {
    let rules = rules_for_language("català");
    assert!(rules.is_some(), "català should be supported");
}

#[test]
fn test_every_aggregator_returns_non_empty_rule_set() {
    use crate::phonetic::types::RewriteRuleChar;
    let aggregators: &[(&str, fn() -> Vec<RewriteRuleChar>)] = &[
        ("american_english", american_english_rules),
        ("british_english", british_english_rules),
        ("german", german_rules),
        ("dutch", dutch_rules),
        ("swedish", swedish_rules),
        ("norwegian", norwegian_rules),
        ("danish", danish_rules),
        ("icelandic", icelandic_rules),
        ("castilian_spanish", castilian_spanish_rules),
        ("latin_american_spanish", latin_american_spanish_rules),
        ("standard_french", standard_french_rules),
        ("canadian_french", canadian_french_rules),
        ("european_portuguese", european_portuguese_rules),
        ("brazilian_portuguese", brazilian_portuguese_rules),
        ("italian", italian_rules),
        ("catalan", catalan_rules),
        ("romanian", romanian_rules),
        ("russian", russian_rules),
        ("polish", polish_rules),
        ("czech", czech_rules),
        ("slovak", slovak_rules),
        ("croatian", croatian_rules),
        ("serbian", serbian_rules),
        ("serbian_latin", serbian_latin_rules),
        ("bulgarian", bulgarian_rules),
        ("belarusian", belarusian_rules),
        ("ukrainian", ukrainian_rules),
        ("welsh", welsh_rules),
        ("irish", irish_rules),
        ("korean", korean_rules),
        ("korean_romanization", korean_romanization_rules),
        ("japanese_romaji", japanese_romaji_rules),
        ("chinese_pinyin", chinese_pinyin_rules),
        ("chinese_hanzi", chinese_hanzi_rules),
        ("hebrew", hebrew_rules),
        ("arabic", arabic_rules),
        ("urdu", urdu_rules),
        ("persian", persian_rules),
        ("maltese", maltese_rules),
        ("hindi", hindi_rules),
        ("marathi", marathi_rules),
        ("bengali", bengali_rules),
        ("gujarati", gujarati_rules),
        ("telugu", telugu_rules),
        ("tamil", tamil_rules),
        ("punjabi_gurmukhi", punjabi_gurmukhi_rules),
        ("punjabi_shahmukhi", punjabi_shahmukhi_rules),
        ("thai", thai_rules),
        ("vietnamese", vietnamese_rules),
        ("indonesian", indonesian_rules),
        ("tagalog", tagalog_rules),
        ("turkish", turkish_rules),
        ("hungarian", hungarian_rules),
        ("finnish", finnish_rules),
        ("basque", basque_rules),
        ("greek", greek_rules),
        ("georgian", georgian_rules),
        ("armenian", armenian_rules),
    ];

    for (name, agg) in aggregators {
        let rules = agg();
        assert!(!rules.is_empty(), "{name}_rules() returned empty Vec");
    }
}
