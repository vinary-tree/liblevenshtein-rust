//! Dictionary bindings for WebAssembly.

use wasm_bindgen::prelude::*;

use super::terms::{
    double_array_trie_from_sorted_terms, dynamic_dawg_from_sorted_terms, parse_sorted_terms,
};
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::Dictionary;

/// A WebAssembly-compatible Double Array Trie dictionary.
///
/// The Double Array Trie provides fast lookups for static dictionaries.
/// Best used when the dictionary won't change after construction.
///
/// # Example (JavaScript)
///
/// ```javascript
/// import { WasmDoubleArrayTrie } from 'liblevenshtein';
///
/// const dict = new WasmDoubleArrayTrie(["hello", "help", "world"]);
/// console.log(dict.contains("hello")); // true
/// console.log(dict.contains("helo"));  // false
/// console.log(dict.len());             // 3
/// ```
#[wasm_bindgen]
pub struct WasmDoubleArrayTrie {
    inner: DoubleArrayTrie<()>,
}

#[wasm_bindgen]
impl WasmDoubleArrayTrie {
    /// Create a new Double Array Trie from an array of terms.
    ///
    /// # Arguments
    ///
    /// * `terms` - Array of strings to add to the dictionary
    #[wasm_bindgen(constructor)]
    pub fn new(terms: Vec<JsValue>) -> Result<WasmDoubleArrayTrie, JsValue> {
        let terms = parse_sorted_terms(terms)?;

        Ok(WasmDoubleArrayTrie {
            inner: double_array_trie_from_sorted_terms(terms),
        })
    }

    /// Check if a term exists in the dictionary.
    pub fn contains(&self, term: &str) -> bool {
        self.inner.contains(term)
    }

    /// Get the number of terms in the dictionary.
    pub fn len(&self) -> usize {
        self.inner.len().unwrap_or(0)
    }

    /// Check if the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A WebAssembly-compatible Dynamic DAWG dictionary.
///
/// The Dynamic DAWG provides fast lookups with support for modifications
/// after construction. Use this when you need to add/remove terms dynamically.
///
/// # Example (JavaScript)
///
/// ```javascript
/// import { WasmDynamicDawg } from 'liblevenshtein';
///
/// const dict = new WasmDynamicDawg();
/// dict.insert("hello");
/// dict.insert("help");
/// console.log(dict.contains("hello")); // true
/// dict.remove("hello");
/// console.log(dict.contains("hello")); // false
/// ```
#[wasm_bindgen]
pub struct WasmDynamicDawg {
    inner: DynamicDawg<()>,
}

#[wasm_bindgen]
impl WasmDynamicDawg {
    /// Create a new empty Dynamic DAWG.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmDynamicDawg {
        WasmDynamicDawg {
            inner: DynamicDawg::new(),
        }
    }

    /// Create a Dynamic DAWG from an array of terms.
    ///
    /// # Arguments
    ///
    /// * `terms` - Array of strings to add to the dictionary
    #[wasm_bindgen(js_name = fromTerms)]
    pub fn from_terms(terms: Vec<JsValue>) -> Result<WasmDynamicDawg, JsValue> {
        let terms = parse_sorted_terms(terms)?;

        Ok(WasmDynamicDawg {
            inner: dynamic_dawg_from_sorted_terms(terms),
        })
    }

    /// Insert a term into the dictionary.
    ///
    /// Returns `true` if the term was newly inserted, `false` if it already existed.
    pub fn insert(&self, term: &str) -> bool {
        self.inner.insert(term)
    }

    /// Remove a term from the dictionary.
    ///
    /// Returns `true` if the term was removed, `false` if it didn't exist.
    pub fn remove(&self, term: &str) -> bool {
        self.inner.remove(term)
    }

    /// Check if a term exists in the dictionary.
    pub fn contains(&self, term: &str) -> bool {
        self.inner.contains(term)
    }

    /// Get the number of terms in the dictionary.
    pub fn len(&self) -> usize {
        self.inner.len().unwrap_or(0)
    }

    /// Check if the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for WasmDynamicDawg {
    fn default() -> Self {
        Self::new()
    }
}
