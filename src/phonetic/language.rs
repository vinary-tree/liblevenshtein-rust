//! Language and dialect support for phonetic rules.
//!
//! This module provides a language-aware API for selecting phonetic transformation
//! rules based on BCP 47 language tags (e.g., "en-us", "en-gb", "es", "fr-ca").
//!
//! # Supported Languages
//!
//! ## English
//! - `"en"` / `"en-us"` / `"english"` - American English (default)
//!   - Includes yod-dropping (tune → toon, new → noo)
//!   - Includes t-flapping (water → wader, butter → budder)
//! - `"en-gb"` / `"british"` - British English (RP)
//!   - Includes non-rhotic r-dropping (car → cah, farm → fahm)
//!   - Includes broad 'a' in BATH words (bath → bahth, class → clahs)
//!
//! ## Spanish
//! - `"es"` / `"spanish"` / `"castilian"` - Castilian Spanish (Spain)
//!   - Includes distinción: z, c(e,i) → θ (theta)
//! - `"es-419"` / `"es-mx"` / `"es-ar"` - Latin American Spanish
//!   - Includes seseo: z, c(e,i) → s
//!
//! ## French
//! - `"fr"` / `"french"` - Standard French (Metropolitan)
//!   - Standard pronunciation without affrication
//! - `"fr-ca"` / `"quebec"` - Canadian French (Québécois)
//!   - Includes affrication: ti → tsi, tu → tsu, di → dzi
//!
//! ## Portuguese
//! - `"pt"` / `"pt-pt"` / `"portuguese"` - European Portuguese (Portugal)
//!   - Includes vowel reduction, uvular R, s → sh before consonants
//! - `"pt-br"` / `"brazilian"` - Brazilian Portuguese
//!   - Includes t/d palatalization, l vocalization, h-like R
//!
//! ## Italian
//! - `"it"` / `"italian"` - Standard Italian
//!   - Includes c/g softening, geminate preservation, gn/gli patterns
//!
//! ## German
//! - `"de"` / `"german"` / `"deutsch"` - Standard German (Hochdeutsch)
//!   - Includes umlauts (ä, ö, ü), eszett (ß), CH variations, final devoicing
//!
//! ## Dutch
//! - `"nl"` / `"dutch"` / `"nederlands"` - Standard Dutch (Nederlands)
//!   - Includes IJ digraph, guttural G/CH, OE/EU/UI patterns
//!
//! ## Russian
//! - `"ru"` / `"russian"` / `"русский"` - Standard Russian (Cyrillic)
//!   - Includes Cyrillic to Latin transliteration, iotated vowels, palatalization
//!
//! ## Korean
//! - `"ko"` / `"korean"` / `"한국어"` - Korean (Hangul)
//!   - Includes Hangul jamo to Latin transliteration, double/aspirated consonants
//! - `"ko-latn"` / `"korean-romanization"` - Korean Romanization
//!   - Normalizes McCune-Reischauer to Revised Romanization variants
//!
//! ## Hebrew
//! - `"he"` / `"hebrew"` / `"עברית"` - Hebrew
//!   - Includes Hebrew letters to Latin, dagesh, shin/sin, final forms
//!
//! ## Polish
//! - `"pl"` / `"polish"` / `"polski"` - Polish
//!   - Includes nasal vowels (ą, ę), digraphs (sz, cz, rz), special letters
//!
//! ## Turkish
//! - `"tr"` / `"turkish"` / `"türkçe"` - Turkish (Türkçe)
//!   - Includes dotted/undotted I, special consonants (ş, ç, ğ), front vowels
//!
//! ## Japanese Romaji
//! - `"ja-latn"` / `"romaji"` / `"ローマ字"` - Japanese Romaji
//!   - Includes long vowels (macrons), romanization variants, gemination
//!
//! ## Chinese Pinyin
//! - `"zh-latn"` / `"pinyin"` / `"拼音"` - Chinese Pinyin
//!   - Includes tone mark removal, ü handling, retroflex consonants
//!
//! ## Arabic
//! - `"ar"` / `"arabic"` / `"العربية"` - Arabic (العربية)
//!   - Includes Arabic letters to Latin, emphatic consonants, diacritics
//!
//! ## Urdu
//! - `"ur"` / `"urdu"` / `"اردو"` - Urdu (اردو)
//!   - Includes Arabic-inherited consonants, Persian additions, retroflex, aspirated
//!
//! ## Hindi
//! - `"hi"` / `"hindi"` / `"हिन्दी"` - Hindi (हिन्दी)
//!   - Includes Devanagari consonants, vowel matras, retroflex, aspirated, nukta letters
//!
//! ## Ukrainian
//! - `"uk"` / `"ukrainian"` / `"українська"` - Ukrainian (Українська мова)
//!   - Unique letters: і, ї, є, ґ; Ukrainian г→h (not g!); и→y
//!
//! ## Hungarian
//! - `"hu"` / `"hungarian"` / `"magyar"` - Hungarian (Magyar)
//!   - 9 digraphs (cs, dz, dzs, gy, ly, ny, sz, ty, zs); S→SH; long/double-acute vowels
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::language::{rules_for_language, is_supported, default_language};
//!
//! // Check supported languages
//! assert!(is_supported("en-us"));
//! assert!(is_supported("es"));
//! assert!(is_supported("fr-ca"));
//! assert!(is_supported("pt-br"));
//! assert!(is_supported("it"));
//!
//! // Get rules for a language
//! let rules = rules_for_language("es-419").expect("Latin American Spanish is supported");
//! assert!(!rules.is_empty());
//!
//! // Get the default language
//! assert_eq!(default_language(), "en-us");
//! ```
//!
//! # Integration with PhoneticNormalizedDictionary
//!
//! ```rust,ignore
//! use liblevenshtein::dictionary::phonetic_normalized::PhoneticNormalizedDictionary;
//!
//! // Create dictionary with explicit language
//! let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_language(
//!     &["phone", "fone", "elephant"],
//!     "en-us"
//! );
//!
//! // Query as usual
//! let results = dict.query("fone", 2);
//! ```

use crate::phonetic::types::RewriteRuleChar;

/// BCP 47 language tag constants.
///
/// These constants provide well-known language tags for use with
/// [`rules_for_language`] and related functions.
pub mod tags {
    // ============================================================
    // English
    // ============================================================

    /// Generic English (maps to American English).
    pub const ENGLISH: &str = "en";

    /// American English (default).
    pub const ENGLISH_US: &str = "en-us";

    /// British English.
    pub const ENGLISH_GB: &str = "en-gb";

    // ============================================================
    // Spanish
    // ============================================================

    /// Generic Spanish (maps to Castilian).
    pub const SPANISH: &str = "es";

    /// Latin American Spanish (seseo).
    pub const SPANISH_419: &str = "es-419";

    /// Mexican Spanish (uses Latin American rules).
    pub const SPANISH_MX: &str = "es-mx";

    /// Argentine Spanish (uses Latin American rules).
    pub const SPANISH_AR: &str = "es-ar";

    // ============================================================
    // French
    // ============================================================

    /// Generic French (maps to Standard/Metropolitan French).
    pub const FRENCH: &str = "fr";

    /// Canadian French (Québécois).
    pub const FRENCH_CA: &str = "fr-ca";

    // ============================================================
    // Portuguese
    // ============================================================

    /// Generic Portuguese (maps to European Portuguese).
    pub const PORTUGUESE: &str = "pt";

    /// European Portuguese (Portugal).
    pub const PORTUGUESE_PT: &str = "pt-pt";

    /// Brazilian Portuguese.
    pub const PORTUGUESE_BR: &str = "pt-br";

    // ============================================================
    // Italian
    // ============================================================

    /// Italian (Standard).
    pub const ITALIAN: &str = "it";

    // ============================================================
    // German
    // ============================================================

    /// German (Standard / Hochdeutsch).
    pub const GERMAN: &str = "de";

    // ============================================================
    // Dutch
    // ============================================================

    /// Dutch (Standard / Nederlands).
    pub const DUTCH: &str = "nl";

    // ============================================================
    // Russian
    // ============================================================

    /// Russian (Standard / Русский).
    pub const RUSSIAN: &str = "ru";

    // ============================================================
    // Korean
    // ============================================================

    /// Korean (한국어).
    pub const KOREAN: &str = "ko";

    /// Korean Romanization (normalizes MR to RR variants).
    pub const KOREAN_ROMANIZATION: &str = "ko-latn";

    // ============================================================
    // Hebrew
    // ============================================================

    /// Hebrew (עברית).
    pub const HEBREW: &str = "he";

    // ============================================================
    // Polish
    // ============================================================

    /// Polish (polski).
    pub const POLISH: &str = "pl";

    // ============================================================
    // Turkish
    // ============================================================

    /// Turkish (Türkçe).
    pub const TURKISH: &str = "tr";

    // ============================================================
    // Japanese
    // ============================================================

    /// Japanese Romaji (ローマ字).
    pub const JAPANESE_ROMAJI: &str = "ja-latn";

    // ============================================================
    // Chinese
    // ============================================================

    /// Chinese Pinyin (汉语拼音).
    pub const CHINESE_PINYIN: &str = "zh-latn";

    // ============================================================
    // Arabic
    // ============================================================

    /// Arabic (العربية).
    pub const ARABIC: &str = "ar";

    // ============================================================
    // Urdu
    // ============================================================

    /// Urdu (اردو).
    pub const URDU: &str = "ur";

    // ============================================================
    // Hindi
    // ============================================================

    /// Hindi (हिन्दी).
    pub const HINDI: &str = "hi";

    // ============================================================
    // Chinese (Hanzi)
    // ============================================================

    /// Chinese (Hanzi characters, 汉字).
    pub const CHINESE_HANZI: &str = "zh";

    // ============================================================
    // Tagalog
    // ============================================================

    /// Tagalog (Filipino).
    pub const TAGALOG: &str = "tl";

    // ============================================================
    // Ukrainian
    // ============================================================

    /// Ukrainian (Українська).
    pub const UKRAINIAN: &str = "uk";

    // ============================================================
    // Hungarian
    // ============================================================

    /// Hungarian (Magyar).
    pub const HUNGARIAN: &str = "hu";

    // ============================================================
    // Indonesian
    // ============================================================

    /// Indonesian (Bahasa Indonesia).
    pub const INDONESIAN: &str = "id";

    // ============================================================
    // Romanian
    // ============================================================

    /// Romanian (Română).
    pub const ROMANIAN: &str = "ro";

    // ============================================================
    // Finnish
    // ============================================================

    /// Finnish (Suomi).
    pub const FINNISH: &str = "fi";

    // ============================================================
    // Basque
    // ============================================================

    /// Basque (Euskara).
    pub const BASQUE: &str = "eu";

    // ============================================================
    // Catalan
    // ============================================================

    /// Catalan (Català).
    pub const CATALAN: &str = "ca";

    // ============================================================
    // Nordic Languages
    // ============================================================

    /// Swedish (Svenska).
    pub const SWEDISH: &str = "sv";

    /// Norwegian (Norsk) - covers both Bokmål and Nynorsk.
    pub const NORWEGIAN: &str = "no";

    /// Danish (Dansk).
    pub const DANISH: &str = "da";

    // ============================================================
    // Slavic Languages (Latin script)
    // ============================================================

    /// Czech (Čeština).
    pub const CZECH: &str = "cs";

    /// Slovak (Slovenčina).
    pub const SLOVAK: &str = "sk";

    /// Croatian (Hrvatski).
    pub const CROATIAN: &str = "hr";

    // ============================================================
    // Celtic Languages
    // ============================================================

    /// Welsh (Cymraeg).
    pub const WELSH: &str = "cy";

    /// Irish (Gaeilge).
    pub const IRISH: &str = "ga";

    // ============================================================
    // Slavic Languages (Cyrillic) - Batch 5
    // ============================================================

    /// Bulgarian (Български).
    pub const BULGARIAN: &str = "bg";

    /// Belarusian (Беларуская).
    pub const BELARUSIAN: &str = "be";

    /// Serbian (Српски) - Cyrillic script.
    pub const SERBIAN: &str = "sr";

    /// Serbian Latin script variant.
    pub const SERBIAN_LATIN: &str = "sr-latn";

    // ============================================================
    // Batch 6: Medium Effort Languages
    // ============================================================

    /// Vietnamese (Tiếng Việt).
    pub const VIETNAMESE: &str = "vi";

    /// Greek (Ελληνικά).
    pub const GREEK: &str = "el";

    /// Icelandic (Íslenska).
    pub const ICELANDIC: &str = "is";

    /// Persian/Farsi (فارسی).
    pub const PERSIAN: &str = "fa";

    /// Maltese (Malti).
    pub const MALTESE: &str = "mt";
}

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
        "pt" | "pt-pt" | "portuguese" | "european-portuguese" => {
            Some(european_portuguese_rules())
        }
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

/// Combine American English rule sets into a single vector.
///
/// This includes:
/// - Base orthographic rules (spelling normalization)
/// - American dialect rules (yod-dropping, t-flapping)
/// - Homophone rules (words that sound alike)
/// - Text-speak rules (SMS/text abbreviations)
fn american_english_rules() -> Vec<RewriteRuleChar> {
    use super::rules::english;

    let mut rules = Vec::new();
    rules.extend(english::base().rules.iter().cloned());
    rules.extend(english::american().rules.iter().cloned());
    rules.extend(english::homophones().rules.iter().cloned());
    rules.extend(english::text_speak().rules.iter().cloned());
    rules
}

/// Combine British English rule sets into a single vector.
///
/// This includes:
/// - Base orthographic rules (spelling normalization)
/// - British dialect rules (r-dropping, broad 'a' in BATH words)
/// - Homophone rules (words that sound alike)
/// - Text-speak rules (SMS/text abbreviations)
fn british_english_rules() -> Vec<RewriteRuleChar> {
    use super::rules::english;

    let mut rules = Vec::new();
    rules.extend(english::base().rules.iter().cloned());
    rules.extend(english::british().rules.iter().cloned());
    rules.extend(english::homophones().rules.iter().cloned());
    rules.extend(english::text_speak().rules.iter().cloned());
    rules
}

// ============================================================
// Spanish
// ============================================================

/// Combine Castilian Spanish rule sets into a single vector.
///
/// This includes:
/// - Base Spanish orthographic rules
/// - Castilian dialect rules (distinción: z, c(e,i) → θ)
fn castilian_spanish_rules() -> Vec<RewriteRuleChar> {
    use super::rules::spanish;
    spanish::combined_castilian().rules.clone()
}

/// Combine Latin American Spanish rule sets into a single vector.
///
/// This includes:
/// - Base Spanish orthographic rules
/// - Latin American dialect rules (seseo: z, c(e,i) → s)
fn latin_american_spanish_rules() -> Vec<RewriteRuleChar> {
    use super::rules::spanish;
    spanish::combined_latin_american().rules.clone()
}

// ============================================================
// French
// ============================================================

/// Combine Standard French rule sets into a single vector.
///
/// This includes:
/// - Base French orthographic rules (silent letters, nasal vowels, digraphs)
/// - Standard French dialect rules
fn standard_french_rules() -> Vec<RewriteRuleChar> {
    use super::rules::french;
    french::combined_standard().rules.clone()
}

/// Combine Canadian French rule sets into a single vector.
///
/// This includes:
/// - Base French orthographic rules
/// - Canadian dialect rules (affrication: ti → tsi, tu → tsu, di → dzi)
fn canadian_french_rules() -> Vec<RewriteRuleChar> {
    use super::rules::french;
    french::combined_canadian().rules.clone()
}

// ============================================================
// Portuguese
// ============================================================

/// Combine European Portuguese rule sets into a single vector.
///
/// This includes:
/// - Base Portuguese orthographic rules
/// - European dialect rules (vowel reduction, uvular R, s → sh patterns)
fn european_portuguese_rules() -> Vec<RewriteRuleChar> {
    use super::rules::portuguese;
    portuguese::combined_european().rules.clone()
}

/// Combine Brazilian Portuguese rule sets into a single vector.
///
/// This includes:
/// - Base Portuguese orthographic rules
/// - Brazilian dialect rules (t/d palatalization, l vocalization, h-like R)
fn brazilian_portuguese_rules() -> Vec<RewriteRuleChar> {
    use super::rules::portuguese;
    portuguese::combined_brazilian().rules.clone()
}

// ============================================================
// Italian
// ============================================================

/// Get Standard Italian rule set.
///
/// Italian has relatively uniform pronunciation across regions,
/// so only a single comprehensive rule set is provided.
fn italian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::italian;
    italian::base().rules.clone()
}

// ============================================================
// German
// ============================================================

/// Get Standard German rule set.
///
/// Returns the complete phonetic normalization rules for German (Hochdeutsch):
/// - Umlaut normalization (ä→ae, ö→oe, ü→ue)
/// - Eszett handling (ß→ss)
/// - CH variations (ich-Laut, ach-Laut)
/// - Final devoicing (b→p, d→t, g→k)
/// - W/V pronunciation
/// - SP/ST patterns
fn german_rules() -> Vec<RewriteRuleChar> {
    use super::rules::german;
    german::base().rules.clone()
}

// ============================================================
// Dutch
// ============================================================

/// Get Standard Dutch rule set.
///
/// Returns the complete phonetic normalization rules for Dutch (Nederlands):
/// - IJ digraph handling (ij→EI)
/// - G/CH guttural sounds (→X)
/// - OE→U, EU→OE, UI→OY vowel patterns
/// - W→V approximant
/// - SCH→sX (not sh like German)
/// - Final devoicing (b→p, d→t)
fn dutch_rules() -> Vec<RewriteRuleChar> {
    use super::rules::dutch;
    dutch::base().rules.clone()
}

// ============================================================
// Russian
// ============================================================

/// Get Standard Russian rule set.
///
/// Returns the complete phonetic normalization rules for Russian:
/// - Cyrillic to Latin transliteration
/// - Complex consonants (ж→zh, ш→sh, щ→shch, ч→ch, ц→ts, х→kh)
/// - Iotated vowels (е→ye, ё→yo, ю→yu, я→ya)
/// - Soft/hard sign removal
/// - Final devoicing
fn russian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::russian;
    russian::base().rules.clone()
}

// ============================================================
// Korean
// ============================================================

/// Get Korean rule set.
///
/// Returns the complete phonetic normalization rules for Korean Hangul:
/// - Hangul jamo to Latin transliteration
/// - Double consonants (ㄲ→kk, ㄸ→tt, ㅃ→pp, ㅆ→ss, ㅉ→jj)
/// - Aspirated consonants (ㅋ→k, ㅌ→t, ㅍ→p, ㅊ→ch)
/// - Compound finals (ㄳ→ks, ㄵ→nj, etc.)
/// - Vowels and diphthongs
fn korean_rules() -> Vec<RewriteRuleChar> {
    use super::rules::korean;
    korean::base().rules.clone()
}

/// Get Korean Romanization rule set.
///
/// Returns the complete phonetic normalization rules for Korean romanization:
/// - McCune-Reischauer breve vowels (ŏ→eo, ŭ→eu)
/// - Aspirate markers (k'→k, t'→t, p'→p, ch'→ch)
/// - Long vowel macrons removal (ā→a, ē→e, etc.)
/// - Variant vowel digraphs (oe→we, ui→i)
fn korean_romanization_rules() -> Vec<RewriteRuleChar> {
    use super::rules::korean;
    korean::romanization().rules.clone()
}

// ============================================================
// Hebrew
// ============================================================

/// Get Hebrew rule set.
///
/// Returns the complete phonetic normalization rules for Hebrew:
/// - 22 consonant letters to Latin transliteration
/// - Final forms (ך, ם, ן, ף, ץ)
/// - Dagesh modifications (בּ→b, כּ→k, פּ→p)
/// - Shin/Sin distinction (שׁ→sh, שׂ→s)
/// - Niqqud vowel points
fn hebrew_rules() -> Vec<RewriteRuleChar> {
    use super::rules::hebrew;
    hebrew::base().rules.clone()
}

// ============================================================
// Polish
// ============================================================

/// Get Polish rule set.
///
/// Returns the complete phonetic normalization rules for Polish:
/// - Special Polish letters (ą, ę, ć, ś, ź, ż, ł, ó, ń)
/// - Nasal vowels (ą→on, ę→en)
/// - Digraphs (sz→sh, cz→ch, rz→zh, dz, dź→dj, dż→dzh, ch→kh)
/// - Ł→w, ó→u equivalence
/// - Polish consonant and vowel mappings
fn polish_rules() -> Vec<RewriteRuleChar> {
    use super::rules::polish;
    polish::base().rules.clone()
}

// ============================================================
// Turkish
// ============================================================

/// Get Turkish rule set.
///
/// Returns the complete phonetic normalization rules for Turkish:
/// - Special consonants (ş→S, ç→C, ğ→G)
/// - Dotted/undotted I distinction (ı→I, İ→i)
/// - Front vowels (ö→O, ü→U)
/// - Consonant transforms (c→dj, j→Z)
/// - Simplification rules for doubled markers
fn turkish_rules() -> Vec<RewriteRuleChar> {
    use super::rules::turkish;
    turkish::base().rules.clone()
}

// ============================================================
// Japanese
// ============================================================

/// Get Japanese Romaji rule set.
///
/// Returns the complete phonetic normalization rules for Japanese romanization:
/// - Long vowels (macrons: ā→A, ē→E, ī→I, ō→O, ū→U)
/// - Romanization variants (ti→C, tu→TS, si→S, hu→F)
/// - Digraphs (shi→S, chi→C, tsu→TS)
/// - Gemination (kk→K, pp→P, tt→T)
/// - Syllabic N (n'→N)
fn japanese_romaji_rules() -> Vec<RewriteRuleChar> {
    use super::rules::japanese;
    japanese::romaji().rules.clone()
}

// ============================================================
// Chinese
// ============================================================

/// Get Chinese Pinyin rule set.
///
/// Returns the complete phonetic normalization rules for Chinese Pinyin:
/// - Tone marks (ā á ǎ à → a, etc.)
/// - Ü handling (ü, v → U)
/// - Retroflex consonants (zh→Z, ch→C, sh→S, r→R)
/// - Palatal consonants (x→X, q→Q)
/// - Affricate c (c→TS)
fn chinese_pinyin_rules() -> Vec<RewriteRuleChar> {
    use super::rules::chinese;
    chinese::pinyin().rules.clone()
}

// ============================================================
// Arabic
// ============================================================

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
fn arabic_rules() -> Vec<RewriteRuleChar> {
    use super::rules::arabic;
    arabic::base().rules.clone()
}

// ============================================================
// Urdu
// ============================================================

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
fn urdu_rules() -> Vec<RewriteRuleChar> {
    use super::rules::urdu;
    urdu::base().rules.clone()
}

// ============================================================
// Hindi
// ============================================================

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
fn hindi_rules() -> Vec<RewriteRuleChar> {
    use super::rules::hindi;
    hindi::base().rules.clone()
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
fn chinese_hanzi_rules() -> Vec<RewriteRuleChar> {
    use super::rules::chinese;
    chinese::characters().rules.clone()
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
fn tagalog_rules() -> Vec<RewriteRuleChar> {
    use super::rules::tagalog;
    tagalog::base().rules.clone()
}

// ============================================================
// Ukrainian
// ============================================================

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
fn ukrainian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::ukrainian;
    ukrainian::base().rules.clone()
}

// ============================================================
// Hungarian
// ============================================================

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
fn hungarian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::hungarian;
    hungarian::base().rules.clone()
}

// ============================================================
// Indonesian
// ============================================================

/// Get Indonesian rule set.
///
/// Returns the complete phonetic normalization rules for Indonesian (Bahasa Indonesia):
/// - Digraphs: ng→NG (velar nasal), ny→NY (palatal nasal), sy→SH
/// - C → CH (like English "ch", not "k")
/// - KH → KH (voiceless velar fricative, Arabic loans)
/// - V → F (typically pronounced as F in Indonesian)
/// - Nearly phonemic Latin orthography
fn indonesian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::indonesian;
    indonesian::base().rules.clone()
}

// ============================================================
// Romanian
// ============================================================

/// Get Romanian rule set.
///
/// Returns the complete phonetic normalization rules for Romanian (Română):
/// - Special vowels: ă→a (schwa), â/î→i (close central unrounded)
/// - Special consonants: ș→SH, ț→TS
/// - Digraphs: ch→K (before e, i), gh→G (before e, i)
/// - J → ZH (like French "j")
/// - CE, CI → CHE, CHI (soft c)
fn romanian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::romanian;
    romanian::base().rules.clone()
}

// ============================================================
// Finnish
// ============================================================

/// Get Finnish rule set.
///
/// Returns the complete phonetic normalization rules for Finnish (Suomi):
/// - Front vowels: ä→AE, ö→OE, y→Y (front rounded)
/// - Vowel harmony: Front (ä, ö, y) vs back (a, o, u)
/// - Digraphs: ng→NG (velar nasal), nk→NK
/// - Nearly phonemic orthography
/// - Loanword consonant adaptations: b→p, d→t, g→k (in native words)
fn finnish_rules() -> Vec<RewriteRuleChar> {
    use super::rules::finnish;
    finnish::base().rules.clone()
}

// ============================================================
// Basque
// ============================================================

/// Get Basque rule set.
///
/// Returns the complete phonetic normalization rules for Basque (Euskara):
/// - Digraphs: tx→CH, ts→TS, tz→TZ, tt→TT, dd→DD, rr→RR
/// - X → SH (like English "sh")
/// - Z → S (like English "s", NOT "z"!)
/// - Ñ → NY (palatal nasal)
/// - Language isolate with unique phonology
fn basque_rules() -> Vec<RewriteRuleChar> {
    use super::rules::basque;
    basque::base().rules.clone()
}

// ============================================================
// Catalan
// ============================================================

/// Get Catalan rule set.
///
/// Returns the complete phonetic normalization rules for Catalan (Català):
/// - L·L (ela geminada) → LL (unique to Catalan)
/// - Digraphs: ny→NY, tx→CH, ll→Y, ss→S, tg/tj→J
/// - Ç → S (c cedilla)
/// - X → SH (like English "sh")
/// - H → (silent)
/// - V → B (b/v merger in most dialects)
/// - Accented vowels normalized
fn catalan_rules() -> Vec<RewriteRuleChar> {
    use super::rules::catalan;
    catalan::base().rules.clone()
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
fn swedish_rules() -> Vec<RewriteRuleChar> {
    use super::rules::swedish;
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
fn norwegian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::norwegian;
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
fn danish_rules() -> Vec<RewriteRuleChar> {
    use super::rules::danish;
    danish::base().rules.clone()
}

/// Get Czech rule set.
///
/// Returns the complete phonetic normalization rules for Czech (Čeština):
/// - Háčky (caron): č→CH, š→SH, ž→ZH, ř→RZH
/// - Soft consonants: ď→DJ, ť→TJ, ň→NJ
/// - Long vowels (čárka): á, é, í, ó, ú, ý → short equivalents
/// - Kroužek: ů→u
/// - Y/I merger: y→i
fn czech_rules() -> Vec<RewriteRuleChar> {
    use super::rules::czech;
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
fn slovak_rules() -> Vec<RewriteRuleChar> {
    use super::rules::slovak;
    slovak::base().rules.clone()
}

/// Get Croatian rule set.
///
/// Returns the complete phonetic normalization rules for Croatian (Hrvatski):
/// - Digraphs as letters: lj→LJ, nj→NJ, dž→DZH
/// - Diacritics: č→CH, ć→TJ, š→SH, ž→ZH, đ→DJ
/// - Perfect phonemic spelling
fn croatian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::croatian;
    croatian::base().rules.clone()
}

// ============================================================
// Celtic Languages
// ============================================================

/// Get Welsh rule set.
///
/// Returns the complete phonetic normalization rules for Welsh (Cymraeg):
/// - 8 digraphs as letters: ch→CH, dd→DH, ff→F, ng→NG, ll→LL, ph→F, rh→RH, th→TH
/// - Unique LL: Voiceless lateral fricative [ɬ]
/// - F = V sound (ff = F sound)
/// - W and Y as vowels
/// - Circumflex for long vowels: â, ê, î, ô, û, ŵ, ŷ
fn welsh_rules() -> Vec<RewriteRuleChar> {
    use super::rules::welsh;
    welsh::base().rules.clone()
}

/// Get Irish rule set.
///
/// Returns the complete phonetic normalization rules for Irish (Gaeilge):
/// - Séimhiú (lenition): bh→v, ch→CH, dh→GH, fh→(silent!), gh→GH, mh→v, ph→f, sh→h, th→h
/// - FH is completely silent
/// - Fadas (acute accent): á, é, í, ó, ú for long vowels
fn irish_rules() -> Vec<RewriteRuleChar> {
    use super::rules::irish;
    irish::base().rules.clone()
}

/// Get Bulgarian rule set.
///
/// Returns the complete phonetic normalization rules for Bulgarian (Български език):
/// - Щ = sht (unlike Russian shch!)
/// - Ъ = schwa vowel (very common, NOT a hard sign like Russian)
/// - No Ы, Ё, Э (unlike Russian)
/// - Standard Cyrillic consonant/vowel mappings
fn bulgarian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::bulgarian;
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
fn belarusian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::belarusian;
    belarusian::base().rules.clone()
}

/// Get Serbian Cyrillic rule set.
///
/// Returns the complete phonetic normalization rules for Serbian Cyrillic (Српски језик):
/// - Perfect phonemic orthography: one letter = one sound
/// - Unique letters: Љ(lj), Њ(nj), Џ(dž), Ћ(ć), Ђ(đ)
/// - Standard Cyrillic consonant/vowel mappings
fn serbian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::serbian;
    serbian::base().rules.clone()
}

/// Get Serbian Latin rule set.
///
/// Returns the complete phonetic normalization rules for Serbian Latin (Srpski jezik):
/// - Perfect phonemic orthography: one letter = one sound
/// - Digraphs: lj, nj, dž (single phonemes)
/// - Diacritics: č, ć, š, ž, đ
fn serbian_latin_rules() -> Vec<RewriteRuleChar> {
    use super::rules::serbian;
    serbian::latin().rules.clone()
}

// ============================================================
// Batch 6: Medium Effort Languages
// ============================================================

/// Get Vietnamese rule set.
///
/// Returns the complete phonetic normalization rules for Vietnamese (Tiếng Việt):
/// - Latin script with extensive diacritics
/// - 6 tones marked with diacritics (stripped for phonetic matching)
/// - Special vowels: ă, â, ê, ô, ơ, ư
/// - Special consonant: đ (voiced alveolar)
/// - Digraphs: ch, gh, gi, kh, ng, ngh, nh, ph, qu, th, tr
fn vietnamese_rules() -> Vec<RewriteRuleChar> {
    use super::rules::vietnamese;
    vietnamese::base().rules.clone()
}

/// Get Greek rule set.
///
/// Returns the complete phonetic normalization rules for Greek (Ελληνικά):
/// - Greek alphabet (24 letters)
/// - Vowel digraphs: αι(e), ει(i), οι(i), υι(i), αυ(av/af), ευ(ev/ef), ου(u)
/// - Consonant combinations: μπ(b), ντ(d), γκ(g), γγ(ng)
/// - Accent marks stripped for matching
fn greek_rules() -> Vec<RewriteRuleChar> {
    use super::rules::greek;
    greek::base().rules.clone()
}

/// Get Icelandic rule set.
///
/// Returns the complete phonetic normalization rules for Icelandic (Íslenska):
/// - Archaic Norse orthography
/// - Unique letters: Þ/þ(th), Ð/ð(dh), Æ/æ(ai)
/// - Accented vowels indicate different sounds
/// - Consonant clusters: ll→tl, rl→rtl, hv→kv
fn icelandic_rules() -> Vec<RewriteRuleChar> {
    use super::rules::icelandic;
    icelandic::base().rules.clone()
}

/// Get Persian/Farsi rule set.
///
/// Returns the complete phonetic normalization rules for Persian (فارسی):
/// - Arabic script with Persian additions
/// - Persian-specific letters: پ(p), چ(ch), ژ(zh), گ(g)
/// - Arabic consonants with Persian pronunciation differences
/// - Short vowel diacritics
fn persian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::persian;
    persian::base().rules.clone()
}

/// Get Maltese rule set.
///
/// Returns the complete phonetic normalization rules for Maltese (Malti):
/// - Latin script for Semitic language (unique!)
/// - Special letters: Ċ(ch), Ġ(j), Għ(silent), Ħ(h), Ż(z)
/// - X = sh sound, Q = glottal stop (often silent)
/// - IE diphthong
fn maltese_rules() -> Vec<RewriteRuleChar> {
    use super::rules::maltese;
    maltese::base().rules.clone()
}

// ============================================================================
// Batch 7: Indic Languages
// ============================================================================

/// Get Marathi rule set.
///
/// Returns the complete phonetic normalization rules for Marathi (मराठी):
/// - Devanagari script (same as Hindi)
/// - Special ळ (retroflex lateral) unique to Marathi
/// - Schwa deletion patterns differ from Hindi
fn marathi_rules() -> Vec<RewriteRuleChar> {
    use super::rules::marathi;
    marathi::base().rules.clone()
}

/// Get Bengali rule set.
///
/// Returns the complete phonetic normalization rules for Bengali (বাংলা):
/// - Brahmic abugida script
/// - Inherent vowel is 'o' (not 'a' like Hindi!)
/// - Chandrabindu for nasalization
/// - Hasanta (virama) marks consonant clusters
fn bengali_rules() -> Vec<RewriteRuleChar> {
    use super::rules::bengali;
    bengali::base().rules.clone()
}

/// Get Gujarati rule set.
///
/// Returns the complete phonetic normalization rules for Gujarati (ગુજરાતી):
/// - Brahmic abugida derived from Devanagari
/// - No headline (shirorekha) unlike Devanagari
/// - Nukta for foreign sounds
fn gujarati_rules() -> Vec<RewriteRuleChar> {
    use super::rules::gujarati;
    gujarati::base().rules.clone()
}

/// Get Telugu rule set.
///
/// Returns the complete phonetic normalization rules for Telugu (తెలుగు):
/// - Brahmic abugida similar to Kannada
/// - 14 vowels, 36 consonants
/// - Sunna (anusvara) for nasalization
fn telugu_rules() -> Vec<RewriteRuleChar> {
    use super::rules::telugu;
    telugu::base().rules.clone()
}

/// Get Tamil rule set.
///
/// Returns the complete phonetic normalization rules for Tamil (தமிழ்):
/// - Unique Brahmic script (oldest Dravidian)
/// - Grantha letters for Sanskrit loanwords
/// - Unique letters: ழ (retroflex approximant), ற (alveolar trill), ன (alveolar nasal)
fn tamil_rules() -> Vec<RewriteRuleChar> {
    use super::rules::tamil;
    tamil::base().rules.clone()
}

/// Get Punjabi (Gurmukhi) rule set.
///
/// Returns the complete phonetic normalization rules for Punjabi in Gurmukhi script (ਪੰਜਾਬੀ):
/// - Brahmic abugida used in India
/// - Tonal language (3 tones, not marked in script)
/// - Nukta for foreign sounds
fn punjabi_gurmukhi_rules() -> Vec<RewriteRuleChar> {
    use super::rules::punjabi;
    punjabi::gurmukhi().rules.clone()
}

/// Get Punjabi (Shahmukhi) rule set.
///
/// Returns the complete phonetic normalization rules for Punjabi in Shahmukhi script:
/// - Arabic-derived script used in Pakistan
/// - Similar to Urdu script
/// - Right-to-left writing
fn punjabi_shahmukhi_rules() -> Vec<RewriteRuleChar> {
    use super::rules::punjabi;
    punjabi::shahmukhi().rules.clone()
}

// ============================================================================
// Batch 8: Complex Scripts
// ============================================================================

/// Get Thai rule set.
///
/// Returns the complete phonetic normalization rules for Thai (ไทย):
/// - Brahmic-derived abugida
/// - No spaces between words
/// - 5 tones (partially marked)
/// - 44 consonants, vowels can appear before/after/above/below consonants
fn thai_rules() -> Vec<RewriteRuleChar> {
    use super::rules::thai;
    thai::base().rules.clone()
}

/// Get Georgian rule set.
///
/// Returns the complete phonetic normalization rules for Georgian (ქართული):
/// - Unique Mkhedruli script
/// - 33 letters, no uppercase/lowercase distinction
/// - Ejective consonants: k'(ყ), p'(პ), t'(ტ), ts'(წ), ch'(ჭ), q'(ყ)
/// - Nearly phonemic orthography
fn georgian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::georgian;
    georgian::base().rules.clone()
}

/// Get Armenian rule set.
///
/// Returns the complete phonetic normalization rules for Armenian (Hayeren):
/// - Unique Armenian script with uppercase/lowercase
/// - 39 letters (originally 36)
/// - Aspirated consonants: T, P, K
/// - Two dialects: Eastern vs Western Armenian
fn armenian_rules() -> Vec<RewriteRuleChar> {
    use super::rules::armenian;
    armenian::base().rules.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(languages.contains(&"ukrainian"), "should contain 'ukrainian'");
        assert!(
            languages.contains(&"українська"),
            "should contain 'українська'"
        );
        // Hungarian
        assert!(languages.contains(&"hu"), "should contain 'hu'");
        assert!(languages.contains(&"hungarian"), "should contain 'hungarian'");
        assert!(languages.contains(&"magyar"), "should contain 'magyar'");
        // Indonesian
        assert!(languages.contains(&"id"), "should contain 'id'");
        assert!(languages.contains(&"indonesian"), "should contain 'indonesian'");
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
        assert!(!rules.is_empty(), "American English rules should not be empty");
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
        assert!(!rules.is_empty(), "British English rules should not be empty");
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
        assert!(!rules.is_empty(), "Castilian Spanish rules should not be empty");
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
        assert!(
            !castilian.is_empty(),
            "Castilian rules should not be empty"
        );
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
        assert!(!rules.is_empty(), "Standard French rules should not be empty");
        assert!(
            rules.len() > 60,
            "Standard French rules should have >60 rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_canadian_french_rules_not_empty() {
        let rules = canadian_french_rules();
        assert!(!rules.is_empty(), "Canadian French rules should not be empty");
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
        assert!(!rules.is_empty(), "Japanese romaji rules should not be empty");
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
        assert!(!rules.is_empty(), "Japanese romaji rules should not be empty");
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
        assert!(!rules.is_empty(), "Chinese pinyin rules should not be empty");
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
        assert!(!rules.is_empty(), "Chinese pinyin rules should not be empty");
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
        assert!(
            !rules.is_empty(),
            "Chinese Hanzi rules should not be empty"
        );
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
            rules.len() >= 30,
            "Korean romanization rules should have >=30 rules, got {}",
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
            rules.len() >= 25,
            "Indonesian rules should have >=25 rules, got {}",
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
            rules.len() >= 35,
            "Romanian rules should have >=35 rules, got {}",
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
            rules.len() >= 30,
            "Finnish rules should have >=30 rules, got {}",
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
            rules.len() >= 30,
            "Basque rules should have >=30 rules, got {}",
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
            rules.len() >= 40,
            "Catalan rules should have >=40 rules, got {}",
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
}
