//! Central dispatch / lookup API for the language module.
//!
//! Maps BCP 47 language tags (and friendly aliases / native-script aliases) to
//! the per-language phonetic rule aggregator functions defined in
//! [`super::rules`]. This file is purely the routing layer: it owns no rule
//! data, only the `match` that selects which aggregator to call.

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
use crate::phonetic::types::RewriteRuleChar;

/// Get combined phonetic rules for a language tag.
///
/// Returns `Some(rules)` if the language is supported, `None` otherwise.
/// The rules are combined from multiple rule sets for comprehensive coverage.
///
/// # Supported Tags
///
/// ## English
/// - `"en"`, `"en-us"`, `"english"`, `"american"` - American English
/// - `"en-gb"`, `"british"` - British English (RP)
///
/// ## Spanish
/// - `"es"`, `"spanish"`, `"castilian"` - Castilian Spanish (distinción)
/// - `"es-419"`, `"es-mx"`, `"es-ar"`, `"latin-american"` - Latin American Spanish (seseo)
///
/// ## French
/// - `"fr"`, `"french"` - Standard/Metropolitan French
/// - `"fr-ca"`, `"quebec"`, `"quebecois"` - Canadian French (Québécois)
///
/// ## Portuguese
/// - `"pt"`, `"pt-pt"`, `"portuguese"`, `"european-portuguese"` - European Portuguese
/// - `"pt-br"`, `"brazilian"`, `"brazilian-portuguese"` - Brazilian Portuguese
///
/// ## Italian
/// - `"it"`, `"italian"` - Standard Italian
///
/// ## German
/// - `"de"`, `"german"`, `"deutsch"` - Standard German (Hochdeutsch)
///
/// ## Dutch
/// - `"nl"`, `"dutch"`, `"nederlands"` - Standard Dutch (Nederlands)
///
/// ## Russian
/// - `"ru"`, `"russian"`, `"русский"` - Standard Russian (Cyrillic script)
///
/// ## Korean
/// - `"ko"`, `"korean"`, `"한국어"` - Korean (Hangul script)
/// - `"ko-latn"`, `"korean-romanization"` - Korean Romanization (MR→RR normalization)
///
/// ## Hebrew
/// - `"he"`, `"hebrew"`, `"עברית"` - Hebrew (Hebrew script)
///
/// ## Polish
/// - `"pl"`, `"polish"`, `"polski"` - Polish (Latin script with diacritics)
///
/// ## Turkish
/// - `"tr"`, `"turkish"`, `"türkçe"` - Turkish (Latin script with diacritics)
///
/// ## Japanese Romaji
/// - `"ja-latn"`, `"romaji"`, `"ローマ字"` - Japanese romanization
///
/// ## Chinese Pinyin
/// - `"zh-latn"`, `"pinyin"`, `"拼音"` - Chinese romanization
///
/// ## Arabic
/// - `"ar"`, `"arabic"`, `"العربية"` - Arabic (Arabic script)
///
/// ## Urdu
/// - `"ur"`, `"urdu"`, `"اردو"` - Urdu (Nastaliq/Arabic-derived script)
///
/// ## Hindi
/// - `"hi"`, `"hindi"`, `"हिन्दी"`, `"devanagari"` - Hindi (Devanagari script)
///
/// ## Chinese (Hanzi)
/// - `"zh"`, `"chinese"`, `"hanzi"`, `"汉字"`, `"中文"` - Chinese characters (Hanzi)
///
/// ## Tagalog
/// - `"tl"`, `"tagalog"`, `"filipino"` - Tagalog/Filipino (Latin script)
///
/// ## Ukrainian
/// - `"uk"`, `"ukrainian"`, `"українська"` - Ukrainian (Cyrillic script)
///
/// ## Hungarian
/// - `"hu"`, `"hungarian"`, `"magyar"` - Hungarian (Latin script with digraphs)
///
/// Tags are case-insensitive: `"EN-US"`, `"en-US"`, `"en-us"` are all equivalent.
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::language::rules_for_language;
///
/// let rules = rules_for_language("en-us").expect("English should be supported");
/// assert!(!rules.is_empty());
///
/// // Case insensitive
/// assert!(rules_for_language("EN-US").is_some());
///
/// // Romance languages
/// assert!(rules_for_language("es").is_some());
/// assert!(rules_for_language("fr-ca").is_some());
/// assert!(rules_for_language("pt-br").is_some());
/// assert!(rules_for_language("it").is_some());
/// assert!(rules_for_language("de").is_some());
///
/// // Unsupported languages return None
/// assert!(rules_for_language("ja").is_none());
/// ```
pub fn rules_for_language(tag: &str) -> Option<Vec<RewriteRuleChar>> {
    let normalized = tag.to_lowercase();
    match normalized.as_str() {
        // English
        "en" | "en-us" | "english" | "american" => Some(american_english_rules()),
        "en-gb" | "british" => Some(british_english_rules()),

        // Spanish
        "es" | "spanish" | "castilian" => Some(castilian_spanish_rules()),
        "es-419" | "es-mx" | "es-ar" | "es-co" | "es-cl" | "es-pe" | "latin-american" => {
            Some(latin_american_spanish_rules())
        }

        // French
        "fr" | "french" => Some(standard_french_rules()),
        "fr-ca" | "quebec" | "quebecois" | "québécois" => Some(canadian_french_rules()),

        // Portuguese
        "pt" | "pt-pt" | "portuguese" | "european-portuguese" => Some(european_portuguese_rules()),
        "pt-br" | "brazilian" | "brazilian-portuguese" => Some(brazilian_portuguese_rules()),

        // Italian
        "it" | "italian" => Some(italian_rules()),

        // German
        "de" | "german" | "deutsch" => Some(german_rules()),

        // Dutch
        "nl" | "dutch" | "nederlands" | "flemish" => Some(dutch_rules()),

        // Russian
        "ru" | "russian" | "русский" => Some(russian_rules()),

        // Korean
        "ko" | "korean" | "한국어" | "hangul" => Some(korean_rules()),

        // Korean Romanization
        "ko-latn" | "korean-romanization" => Some(korean_romanization_rules()),

        // Hebrew
        "he" | "hebrew" | "עברית" | "ivrit" => Some(hebrew_rules()),

        // Polish
        "pl" | "polish" | "polski" => Some(polish_rules()),

        // Turkish
        "tr" | "turkish" | "türkçe" => Some(turkish_rules()),

        // Japanese Romaji
        "ja-latn" | "romaji" | "ローマ字" => Some(japanese_romaji_rules()),

        // Chinese Pinyin
        "zh-latn" | "pinyin" | "拼音" => Some(chinese_pinyin_rules()),

        // Arabic
        "ar" | "arabic" | "العربية" => Some(arabic_rules()),

        // Urdu
        "ur" | "urdu" | "اردو" => Some(urdu_rules()),

        // Hindi
        "hi" | "hindi" | "हिन्दी" | "devanagari" => Some(hindi_rules()),

        // Chinese (Hanzi characters)
        "zh" | "chinese" | "hanzi" | "汉字" | "中文" => Some(chinese_hanzi_rules()),

        // Tagalog
        "tl" | "tagalog" | "filipino" => Some(tagalog_rules()),

        // Ukrainian
        "uk" | "ukrainian" | "українська" => Some(ukrainian_rules()),

        // Hungarian
        "hu" | "hungarian" | "magyar" => Some(hungarian_rules()),

        // Indonesian
        "id" | "indonesian" | "bahasa indonesia" | "bahasa" => Some(indonesian_rules()),

        // Romanian
        "ro" | "romanian" | "română" => Some(romanian_rules()),

        // Finnish
        "fi" | "finnish" | "suomi" => Some(finnish_rules()),

        // Basque
        "eu" | "basque" | "euskara" => Some(basque_rules()),

        // Catalan
        "ca" | "catalan" | "català" => Some(catalan_rules()),

        // Swedish
        "sv" | "swedish" | "svenska" => Some(swedish_rules()),

        // Norwegian (covers both Bokmål and Nynorsk)
        "no" | "nb" | "nn" | "norwegian" | "norsk" | "bokmål" | "nynorsk" => {
            Some(norwegian_rules())
        }

        // Danish
        "da" | "danish" | "dansk" => Some(danish_rules()),

        // Czech
        "cs" | "czech" | "čeština" | "cestina" => Some(czech_rules()),

        // Slovak
        "sk" | "slovak" | "slovenčina" | "slovencina" => Some(slovak_rules()),

        // Croatian
        "hr" | "croatian" | "hrvatski" => Some(croatian_rules()),

        // Welsh
        "cy" | "welsh" | "cymraeg" => Some(welsh_rules()),

        // Irish
        "ga" | "irish" | "gaeilge" => Some(irish_rules()),

        // Bulgarian
        "bg" | "bulgarian" | "български" => Some(bulgarian_rules()),

        // Belarusian
        "be" | "belarusian" | "беларуская" => Some(belarusian_rules()),

        // Serbian (Cyrillic)
        "sr" | "serbian" | "српски" => Some(serbian_rules()),

        // Serbian (Latin)
        "sr-latn" | "serbian-latin" => Some(serbian_latin_rules()),

        // Vietnamese
        "vi" | "vietnamese" | "tiếng việt" => Some(vietnamese_rules()),

        // Greek
        "el" | "greek" | "ελληνικά" => Some(greek_rules()),

        // Icelandic
        "is" | "icelandic" | "íslenska" => Some(icelandic_rules()),

        // Persian/Farsi
        "fa" | "persian" | "farsi" | "فارسی" => Some(persian_rules()),

        // Maltese
        "mt" | "maltese" | "malti" => Some(maltese_rules()),

        // ================================================================
        // Batch 7: Indic Languages
        // ================================================================

        // Marathi (Devanagari script)
        "mr" | "marathi" | "मराठी" => Some(marathi_rules()),

        // Bengali
        "bn" | "bengali" | "বাংলা" => Some(bengali_rules()),

        // Gujarati
        "gu" | "gujarati" | "ગુજરાતી" => Some(gujarati_rules()),

        // Telugu
        "te" | "telugu" | "తెలుగు" => Some(telugu_rules()),

        // Tamil
        "ta" | "tamil" | "தமிழ்" => Some(tamil_rules()),

        // Punjabi (Gurmukhi - India)
        "pa" | "punjabi" | "ਪੰਜਾਬੀ" => Some(punjabi_gurmukhi_rules()),

        // Punjabi (Shahmukhi - Pakistan)
        "pa-arab" | "punjabi-shahmukhi" => Some(punjabi_shahmukhi_rules()),

        // ================================================================
        // Batch 8: Complex Scripts
        // ================================================================

        // Thai
        "th" | "thai" | "ไทย" => Some(thai_rules()),

        // Georgian
        "ka" | "georgian" | "ქართული" => Some(georgian_rules()),

        // Armenian
        "hy" | "armenian" | "hayeren" => Some(armenian_rules()),

        _ => None,
    }
}

/// Check if a language tag is supported.
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::language::is_supported;
///
/// assert!(is_supported("en-us"));
/// assert!(is_supported("en"));
/// assert!(is_supported("EN-US")); // Case insensitive
/// assert!(!is_supported("fr"));
/// ```
pub fn is_supported(tag: &str) -> bool {
    rules_for_language(tag).is_some()
}

/// List all supported language tags.
///
/// Returns a static slice of all language tags that [`rules_for_language`]
/// will accept.
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::language::supported_languages;
///
/// let languages = supported_languages();
/// assert!(languages.contains(&"en-us"));
/// assert!(languages.contains(&"es"));
/// assert!(languages.contains(&"fr-ca"));
/// assert!(languages.contains(&"pt-br"));
/// assert!(languages.contains(&"it"));
/// ```
pub fn supported_languages() -> &'static [&'static str] {
    &[
        // English
        "en",
        "en-us",
        "en-gb",
        "english",
        "american",
        "british",
        // Spanish
        "es",
        "es-419",
        "es-mx",
        "es-ar",
        "es-co",
        "es-cl",
        "es-pe",
        "spanish",
        "castilian",
        "latin-american",
        // French
        "fr",
        "fr-ca",
        "french",
        "quebec",
        "quebecois",
        "québécois",
        // Portuguese
        "pt",
        "pt-pt",
        "pt-br",
        "portuguese",
        "brazilian",
        "european-portuguese",
        "brazilian-portuguese",
        // Italian
        "it",
        "italian",
        // German
        "de",
        "german",
        "deutsch",
        // Dutch
        "nl",
        "dutch",
        "nederlands",
        "flemish",
        // Russian
        "ru",
        "russian",
        "русский",
        // Korean
        "ko",
        "korean",
        "한국어",
        "hangul",
        // Korean Romanization
        "ko-latn",
        "korean-romanization",
        // Hebrew
        "he",
        "hebrew",
        "עברית",
        "ivrit",
        // Polish
        "pl",
        "polish",
        "polski",
        // Turkish
        "tr",
        "turkish",
        "türkçe",
        // Japanese Romaji
        "ja-latn",
        "romaji",
        "ローマ字",
        // Chinese Pinyin
        "zh-latn",
        "pinyin",
        "拼音",
        // Arabic
        "ar",
        "arabic",
        "العربية",
        // Urdu
        "ur",
        "urdu",
        "اردو",
        // Hindi
        "hi",
        "hindi",
        "हिन्दी",
        "devanagari",
        // Chinese (Hanzi)
        "zh",
        "chinese",
        "hanzi",
        "汉字",
        "中文",
        // Tagalog
        "tl",
        "tagalog",
        "filipino",
        // Ukrainian
        "uk",
        "ukrainian",
        "українська",
        // Hungarian
        "hu",
        "hungarian",
        "magyar",
        // Indonesian
        "id",
        "indonesian",
        "bahasa indonesia",
        "bahasa",
        // Romanian
        "ro",
        "romanian",
        "română",
        // Finnish
        "fi",
        "finnish",
        "suomi",
        // Basque
        "eu",
        "basque",
        "euskara",
        // Catalan
        "ca",
        "catalan",
        "català",
        // Swedish
        "sv",
        "swedish",
        "svenska",
        // Norwegian
        "no",
        "nb",
        "nn",
        "norwegian",
        "norsk",
        "bokmål",
        "nynorsk",
        // Danish
        "da",
        "danish",
        "dansk",
        // Czech
        "cs",
        "czech",
        "čeština",
        "cestina",
        // Slovak
        "sk",
        "slovak",
        "slovenčina",
        "slovencina",
        // Croatian
        "hr",
        "croatian",
        "hrvatski",
        // Welsh
        "cy",
        "welsh",
        "cymraeg",
        // Irish
        "ga",
        "irish",
        "gaeilge",
        // Bulgarian
        "bg",
        "bulgarian",
        "български",
        // Belarusian
        "be",
        "belarusian",
        "беларуская",
        // Serbian (Cyrillic)
        "sr",
        "serbian",
        "српски",
        // Serbian (Latin)
        "sr-latn",
        "serbian-latin",
        // Vietnamese
        "vi",
        "vietnamese",
        "tiếng việt",
        // Greek
        "el",
        "greek",
        "ελληνικά",
        // Icelandic
        "is",
        "icelandic",
        "íslenska",
        // Persian/Farsi
        "fa",
        "persian",
        "farsi",
        "فارسی",
        // Maltese
        "mt",
        "maltese",
        "malti",
    ]
}

/// Get the default language tag.
///
/// This is the language used when no explicit language is specified.
/// Currently defaults to American English (`"en-us"`).
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::language::default_language;
///
/// assert_eq!(default_language(), "en-us");
/// ```
pub fn default_language() -> &'static str {
    tags::ENGLISH_US
}
