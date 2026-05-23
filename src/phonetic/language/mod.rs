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

#[cfg(feature = "embedded-rules")]
pub mod dispatch;
#[cfg(feature = "embedded-rules")]
pub mod rules;
pub mod tags;

#[cfg(all(test, feature = "embedded-rules"))]
mod tests;

// Re-export the public dispatch API so existing call sites
// (`crate::phonetic::language::rules_for_language`, etc.) keep working.
#[cfg(feature = "embedded-rules")]
pub use dispatch::{default_language, is_supported, rules_for_language, supported_languages};
