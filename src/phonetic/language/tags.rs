//! BCP 47 language tag constants.
//!
//! These constants provide well-known language tags for use with
//! [`super::dispatch::rules_for_language`] and related functions.

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
