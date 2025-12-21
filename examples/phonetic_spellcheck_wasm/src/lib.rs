//! WebAssembly entry points for phonetic spellcheck demo.
//!
//! This module provides a WASM-compatible API for the phonetic spellchecker,
//! using wasm-bindgen for JavaScript interoperability.

use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use wasm_bindgen::prelude::*;

mod core;
mod embedded;

use core::{PhoneticSpellchecker, QueryResult, SpellMatch, SpellcheckerConfig};

// Global spellchecker instance (lazy initialized)
static SPELLCHECKER: Mutex<Option<PhoneticSpellchecker>> = Mutex::new(None);

/// Initialize the spellchecker with embedded dictionary and rules.
///
/// Call this once before querying. Returns an error if initialization fails.
/// Note: Named `init_spellchecker` to avoid conflict with wasm-bindgen's `init()`.
#[wasm_bindgen(js_name = "initSpellchecker")]
pub fn init_spellchecker() -> Result<(), JsValue> {
    // Set up panic hook for better error messages in browser console
    console_error_panic_hook::set_once();

    let mut checker = SPELLCHECKER
        .lock()
        .map_err(|e| JsValue::from_str(&format!("Lock error: {}", e)))?;

    if checker.is_none() {
        let dictionary = embedded::dictionary();
        let rules = embedded::rules().clone();
        let config = SpellcheckerConfig::default();

        *checker = Some(PhoneticSpellchecker::new(dictionary, rules, config));
    }

    Ok(())
}

/// Check if the spellchecker is initialized.
#[wasm_bindgen]
pub fn is_initialized() -> bool {
    SPELLCHECKER
        .lock()
        .map(|c| c.is_some())
        .unwrap_or(false)
}

/// Query for spelling suggestions.
///
/// Returns a JSON object with:
/// - `original`: The original query word
/// - `normalized`: The phonetically normalized form
/// - `matches`: Array of {word, distance} objects
/// - `warning`: Optional warning message
/// - `from_cache`: Whether the result came from cache
#[wasm_bindgen]
pub fn query(word: &str) -> Result<JsValue, JsValue> {
    let mut checker = SPELLCHECKER
        .lock()
        .map_err(|e| JsValue::from_str(&format!("Lock error: {}", e)))?;

    let checker = checker
        .as_mut()
        .ok_or_else(|| JsValue::from_str("Spellchecker not initialized. Call init() first."))?;

    let result = checker.query(word);

    // Convert to JS-friendly format
    serde_wasm_bindgen::to_value(&WasmQueryResult::from(result))
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Clear the query cache.
#[wasm_bindgen]
pub fn clear_cache() -> Result<(), JsValue> {
    let mut checker = SPELLCHECKER
        .lock()
        .map_err(|e| JsValue::from_str(&format!("Lock error: {}", e)))?;

    if let Some(ref mut checker) = *checker {
        checker.clear_cache();
    }

    Ok(())
}

/// Get dictionary and rules statistics.
///
/// Returns a JSON object with:
/// - `dictionary_size`: Number of words in the dictionary
/// - `rules_count`: Number of phonetic rules
/// - `cache_size`: Number of cached queries
#[wasm_bindgen]
pub fn get_stats() -> Result<JsValue, JsValue> {
    let cache_size = SPELLCHECKER
        .lock()
        .map(|c| c.as_ref().map(|c| c.cache_size()).unwrap_or(0))
        .unwrap_or(0);

    let stats = WasmStats {
        dictionary_size: embedded::dictionary_size(),
        rules_count: embedded::rules_count(),
        cache_size,
    };

    serde_wasm_bindgen::to_value(&stats)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// === JS-friendly types ===

#[derive(Serialize, Deserialize)]
pub struct WasmQueryResult {
    pub original: String,
    pub normalized: String,
    pub matches: Vec<WasmSpellMatch>,
    pub warning: Option<String>,
    pub from_cache: bool,
}

impl From<QueryResult> for WasmQueryResult {
    fn from(r: QueryResult) -> Self {
        Self {
            original: r.original,
            normalized: r.normalized,
            matches: r.matches.into_iter().map(WasmSpellMatch::from).collect(),
            warning: r.warning,
            from_cache: r.from_cache,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct WasmSpellMatch {
    pub word: String,
    pub distance: usize,
}

impl From<SpellMatch> for WasmSpellMatch {
    fn from(m: SpellMatch) -> Self {
        Self {
            word: m.word,
            distance: m.distance,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct WasmStats {
    pub dictionary_size: usize,
    pub rules_count: usize,
    pub cache_size: usize,
}

// === WASM tests ===

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_init() {
        assert!(init_spellchecker().is_ok());
        assert!(is_initialized());
    }

    #[wasm_bindgen_test]
    fn test_query() {
        init_spellchecker().expect("init failed");
        let result = query("fone").expect("query failed");
        assert!(!result.is_null());
    }

    #[wasm_bindgen_test]
    fn test_get_stats() {
        init_spellchecker().expect("init failed");
        let stats = get_stats().expect("get_stats failed");
        assert!(!stats.is_null());
    }
}
