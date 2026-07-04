//! Transducer bindings for WebAssembly.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use super::terms::{
    double_array_trie_from_sorted_terms, dynamic_dawg_from_sorted_terms, parse_sorted_terms,
};
use crate::transducer::{Algorithm, OrderedCandidate, Transducer};
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::Dictionary;

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
        let terms = parse_sorted_terms(terms)?;
        let algorithm = parse_algorithm(algorithm)?;
        let dict = double_array_trie_from_sorted_terms(terms);
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
        let candidates = collect_ordered_candidates(self.inner.query_ordered(query, max_distance));
        serialize_candidates(&candidates)
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
        let candidates =
            collect_best_ordered_candidates(self.inner.query_ordered(query, max_distance));
        serialize_candidates(&candidates)
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
        let candidates =
            collect_ordered_candidates(self.inner.query_ordered(query, max_distance).take(limit));
        serialize_candidates(&candidates)
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
        let terms = parse_sorted_terms(terms)?;
        let algorithm = parse_algorithm(algorithm)?;
        let dict = dynamic_dawg_from_sorted_terms(terms);

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
        let transducer = Transducer::new(self.dict.clone(), self.algorithm);
        let candidates = collect_ordered_candidates(transducer.query_ordered(query, max_distance));
        serialize_candidates(&candidates)
    }

    /// Query and return only the closest matches (lowest distance).
    #[wasm_bindgen(js_name = queryBest)]
    pub fn query_best(&self, query: &str, max_distance: usize) -> Result<JsValue, JsValue> {
        let transducer = Transducer::new(self.dict.clone(), self.algorithm);
        let candidates =
            collect_best_ordered_candidates(transducer.query_ordered(query, max_distance));
        serialize_candidates(&candidates)
    }

    /// Query and return a limited number of closest results.
    #[wasm_bindgen(js_name = queryLimit)]
    pub fn query_limit(
        &self,
        query: &str,
        max_distance: usize,
        limit: usize,
    ) -> Result<JsValue, JsValue> {
        let transducer = Transducer::new(self.dict.clone(), self.algorithm);
        let candidates =
            collect_ordered_candidates(transducer.query_ordered(query, max_distance).take(limit));
        serialize_candidates(&candidates)
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
    if matches_algorithm_alias(s, &["standard", "levenshtein"]) {
        Ok(Algorithm::Standard)
    } else if matches_algorithm_alias(
        s,
        &[
            "transposition",
            "damerau",
            "damerau_levenshtein",
            "damerau-levenshtein",
        ],
    ) {
        Ok(Algorithm::Transposition)
    } else if matches_algorithm_alias(s, &["merge_and_split", "merge-and-split", "mergesplit"]) {
        Ok(Algorithm::MergeAndSplit)
    } else {
        Err(JsValue::from_str(
            "unknown algorithm; use 'standard', 'transposition', or 'merge_and_split'",
        ))
    }
}

fn matches_algorithm_alias(input: &str, aliases: &[&str]) -> bool {
    aliases
        .iter()
        .any(|alias| input.eq_ignore_ascii_case(alias))
}

fn serialize_candidates(candidates: &[WasmCandidate]) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(candidates)
        .map_err(|e| JsValue::from_str(&format!("serialization error: {}", e)))
}

fn collect_ordered_candidates<I>(candidates: I) -> Vec<WasmCandidate>
where
    I: IntoIterator<Item = OrderedCandidate>,
{
    candidates.into_iter().map(WasmCandidate::from).collect()
}

fn collect_best_ordered_candidates<I>(candidates: I) -> Vec<WasmCandidate>
where
    I: IntoIterator<Item = OrderedCandidate>,
{
    let mut candidates = candidates.into_iter();
    let Some(first) = candidates.next() else {
        return Vec::new();
    };

    let best_distance = first.distance;
    let mut best = vec![WasmCandidate::from(first)];

    for candidate in candidates {
        if candidate.distance != best_distance {
            break;
        }
        best.push(WasmCandidate::from(candidate));
    }

    best
}

impl From<OrderedCandidate> for WasmCandidate {
    fn from(candidate: OrderedCandidate) -> Self {
        Self {
            term: candidate.term,
            distance: candidate.distance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ordered_candidate(term: &str, distance: usize) -> OrderedCandidate {
        OrderedCandidate {
            term: term.to_owned(),
            distance,
        }
    }

    #[test]
    fn collect_best_ordered_candidates_keeps_only_minimum_distance_group() {
        let candidates = vec![
            ordered_candidate("alpha", 1),
            ordered_candidate("alpine", 1),
            ordered_candidate("aleph", 2),
        ];

        let best = collect_best_ordered_candidates(candidates);

        assert_eq!(best.len(), 2);
        assert_eq!(best[0].term, "alpha");
        assert_eq!(best[0].distance, 1);
        assert_eq!(best[1].term, "alpine");
        assert_eq!(best[1].distance, 1);
    }

    #[test]
    fn collect_best_ordered_candidates_returns_empty_for_no_matches() {
        let best = collect_best_ordered_candidates(Vec::new());
        assert!(best.is_empty());
    }

    #[test]
    fn collect_ordered_candidates_preserves_iterator_order() {
        let candidates = vec![
            ordered_candidate("alpha", 0),
            ordered_candidate("alpine", 1),
            ordered_candidate("aleph", 2),
        ];

        let collected = collect_ordered_candidates(candidates);

        assert_eq!(
            collected
                .into_iter()
                .map(|candidate| (candidate.term, candidate.distance))
                .collect::<Vec<_>>(),
            vec![
                ("alpha".to_owned(), 0),
                ("alpine".to_owned(), 1),
                ("aleph".to_owned(), 2),
            ]
        );
    }

    #[test]
    fn parse_algorithm_accepts_case_insensitive_aliases() {
        assert_eq!(parse_algorithm("STANDARD").unwrap(), Algorithm::Standard);
        assert_eq!(
            parse_algorithm("Damerau-Levenshtein").unwrap(),
            Algorithm::Transposition
        );
        assert_eq!(
            parse_algorithm("merge-and-split").unwrap(),
            Algorithm::MergeAndSplit
        );
    }
}
