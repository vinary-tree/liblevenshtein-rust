//! Transducer bindings for WebAssembly.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::Dictionary;
use crate::transducer::{Algorithm, Transducer};

/// A fuzzy search result candidate.
#[derive(Serialize, Deserialize)]
pub struct WasmCandidate {
    /// The matched term from the dictionary
    pub term: String,
    /// The edit distance from the query
    pub distance: usize,
}

/// A WebAssembly-compatible Levenshtein transducer for fuzzy search (static dictionary).
///
/// Uses a DoubleArrayTrie backend for fast lookups. The dictionary cannot be
/// modified after creation.
///
/// # Example (JavaScript)
///
/// ```javascript
/// import { WasmTransducer } from 'liblevenshtein';
///
/// // Create from terms with algorithm type
/// const trans = new WasmTransducer(
///     ["hello", "help", "world", "helm"],
///     "standard"  // or "transposition", "merge_and_split"
/// );
///
/// // Query with max distance
/// const results = trans.query("helo", 2);
/// console.log(results);
/// // [{ term: "hello", distance: 1 }, { term: "help", distance: 2 }]
/// ```
#[wasm_bindgen]
pub struct WasmTransducer {
    inner: Transducer<DoubleArrayTrie<()>>,
}

#[wasm_bindgen]
impl WasmTransducer {
    /// Create a new transducer from an array of terms.
    ///
    /// # Arguments
    ///
    /// * `terms` - Array of strings to search through
    /// * `algorithm` - Algorithm type: "standard", "transposition", or "merge_and_split"
    ///
    /// # Algorithms
    ///
    /// - `standard`: Basic Levenshtein (insert, delete, substitute)
    /// - `transposition`: Adds transposition as a single edit (good for typos)
    /// - `merge_and_split`: Adds merge/split operations (good for spacing errors)
    #[wasm_bindgen(constructor)]
    pub fn new(terms: Vec<JsValue>, algorithm: &str) -> Result<WasmTransducer, JsValue> {
        let terms: Result<Vec<String>, _> = terms
            .into_iter()
            .map(|v| {
                v.as_string()
                    .ok_or_else(|| JsValue::from_str("all terms must be strings"))
            })
            .collect();
        let terms = terms?;
        let term_refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();

        let algorithm = parse_algorithm(algorithm)?;
        let dict = DoubleArrayTrie::from_terms(term_refs);
        let inner = Transducer::new(dict, algorithm);

        Ok(WasmTransducer { inner })
    }

    /// Query the transducer for fuzzy matches.
    ///
    /// # Arguments
    ///
    /// * `query` - The string to search for
    /// * `max_distance` - Maximum edit distance to consider
    ///
    /// # Returns
    ///
    /// Array of `{ term, distance }` objects sorted by distance.
    pub fn query(&self, query: &str, max_distance: usize) -> Result<JsValue, JsValue> {
        let candidates: Vec<WasmCandidate> = self
            .inner
            .query_with_distance(query, max_distance)
            .map(|c| WasmCandidate {
                term: c.term.clone(),
                distance: c.distance,
            })
            .collect();

        serde_wasm_bindgen::to_value(&candidates)
            .map_err(|e| JsValue::from_str(&format!("serialization error: {}", e)))
    }

    /// Query and return only the closest matches (lowest distance).
    ///
    /// # Arguments
    ///
    /// * `query` - The string to search for
    /// * `max_distance` - Maximum edit distance to consider
    ///
    /// # Returns
    ///
    /// Array of matches with the minimum distance found.
    #[wasm_bindgen(js_name = queryBest)]
    pub fn query_best(&self, query: &str, max_distance: usize) -> Result<JsValue, JsValue> {
        let mut candidates: Vec<WasmCandidate> = self
            .inner
            .query_with_distance(query, max_distance)
            .map(|c| WasmCandidate {
                term: c.term.clone(),
                distance: c.distance,
            })
            .collect();

        // Find minimum distance and filter
        if let Some(min_dist) = candidates.iter().map(|c| c.distance).min() {
            candidates.retain(|c| c.distance == min_dist);
        }

        serde_wasm_bindgen::to_value(&candidates)
            .map_err(|e| JsValue::from_str(&format!("serialization error: {}", e)))
    }

    /// Query and return a limited number of results.
    ///
    /// # Arguments
    ///
    /// * `query` - The string to search for
    /// * `max_distance` - Maximum edit distance to consider
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// Array of up to `limit` matches.
    #[wasm_bindgen(js_name = queryLimit)]
    pub fn query_limit(
        &self,
        query: &str,
        max_distance: usize,
        limit: usize,
    ) -> Result<JsValue, JsValue> {
        let candidates: Vec<WasmCandidate> = self
            .inner
            .query_with_distance(query, max_distance)
            .take(limit)
            .map(|c| WasmCandidate {
                term: c.term.clone(),
                distance: c.distance,
            })
            .collect();

        serde_wasm_bindgen::to_value(&candidates)
            .map_err(|e| JsValue::from_str(&format!("serialization error: {}", e)))
    }

    /// Check if a term exists in the dictionary (exact match).
    pub fn contains(&self, term: &str) -> bool {
        self.inner.dictionary().contains(term)
    }

    /// Get the number of terms in the dictionary.
    pub fn len(&self) -> usize {
        self.inner.dictionary().len().unwrap_or(0)
    }

    /// Check if the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A WebAssembly-compatible dynamic Levenshtein transducer for fuzzy search.
///
/// Uses a DynamicDawg backend for modifiable dictionaries. Supports insert/remove
/// operations after creation.
///
/// # Example (JavaScript)
///
/// ```javascript
/// import { WasmDynamicTransducer } from 'liblevenshtein';
///
/// const trans = WasmDynamicTransducer.new(["hello", "help"], "standard");
/// trans.insert("world");
/// trans.remove("help");
/// const results = trans.query("helo", 2);
/// ```
#[wasm_bindgen]
pub struct WasmDynamicTransducer {
    dict: DynamicDawg<()>,
    algorithm: Algorithm,
}

#[wasm_bindgen]
impl WasmDynamicTransducer {
    /// Create a new dynamic transducer from an array of terms.
    ///
    /// # Arguments
    ///
    /// * `terms` - Array of strings to search through
    /// * `algorithm` - Algorithm type: "standard", "transposition", or "merge_and_split"
    #[wasm_bindgen(constructor)]
    pub fn new(terms: Vec<JsValue>, algorithm: &str) -> Result<WasmDynamicTransducer, JsValue> {
        let terms: Result<Vec<String>, _> = terms
            .into_iter()
            .map(|v| {
                v.as_string()
                    .ok_or_else(|| JsValue::from_str("all terms must be strings"))
            })
            .collect();
        let terms = terms?;
        let term_refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();

        let algorithm = parse_algorithm(algorithm)?;
        let dict = DynamicDawg::from_terms(term_refs);

        Ok(WasmDynamicTransducer { dict, algorithm })
    }

    /// Create an empty dynamic transducer.
    #[wasm_bindgen(js_name = empty)]
    pub fn empty(algorithm: &str) -> Result<WasmDynamicTransducer, JsValue> {
        let algorithm = parse_algorithm(algorithm)?;
        Ok(WasmDynamicTransducer {
            dict: DynamicDawg::new(),
            algorithm,
        })
    }

    /// Query the transducer for fuzzy matches.
    ///
    /// # Arguments
    ///
    /// * `query` - The string to search for
    /// * `max_distance` - Maximum edit distance to consider
    ///
    /// # Returns
    ///
    /// Array of `{ term, distance }` objects.
    pub fn query(&self, query: &str, max_distance: usize) -> Result<JsValue, JsValue> {
        // Clone the dictionary (cheap - it's Arc-based)
        let transducer = Transducer::new(self.dict.clone(), self.algorithm);
        let candidates: Vec<WasmCandidate> = transducer
            .query_with_distance(query, max_distance)
            .map(|c| WasmCandidate {
                term: c.term.clone(),
                distance: c.distance,
            })
            .collect();

        serde_wasm_bindgen::to_value(&candidates)
            .map_err(|e| JsValue::from_str(&format!("serialization error: {}", e)))
    }

    /// Insert a term into the dictionary.
    ///
    /// Returns `true` if the term was newly inserted, `false` if it already existed.
    pub fn insert(&self, term: &str) -> bool {
        self.dict.insert(term)
    }

    /// Remove a term from the dictionary.
    ///
    /// Returns `true` if the term was removed, `false` if it didn't exist.
    pub fn remove(&self, term: &str) -> bool {
        self.dict.remove(term)
    }

    /// Check if a term exists in the dictionary (exact match).
    pub fn contains(&self, term: &str) -> bool {
        self.dict.contains(term)
    }

    /// Get the number of terms in the dictionary.
    pub fn len(&self) -> usize {
        self.dict.len().unwrap_or(0)
    }

    /// Check if the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn parse_algorithm(s: &str) -> Result<Algorithm, JsValue> {
    match s.to_lowercase().as_str() {
        "standard" | "levenshtein" => Ok(Algorithm::Standard),
        "transposition" | "damerau" | "damerau_levenshtein" | "damerau-levenshtein" => {
            Ok(Algorithm::Transposition)
        }
        "merge_and_split" | "merge-and-split" | "mergesplit" => Ok(Algorithm::MergeAndSplit),
        _ => Err(JsValue::from_str(
            "unknown algorithm; use 'standard', 'transposition', or 'merge_and_split'",
        )),
    }
}
