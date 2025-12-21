//! Distance function bindings for WebAssembly.

use wasm_bindgen::prelude::*;

use crate::distance::{standard_distance, transposition_distance};

/// Calculate the Levenshtein distance between two strings.
///
/// The Levenshtein distance is the minimum number of single-character edits
/// (insertions, deletions, or substitutions) required to change one string
/// into another.
///
/// # Arguments
///
/// * `source` - The source string
/// * `target` - The target string
///
/// # Returns
///
/// The edit distance between the two strings.
///
/// # Example (JavaScript)
///
/// ```javascript
/// import { levenshtein } from 'liblevenshtein';
/// console.log(levenshtein("kitten", "sitting")); // 3
/// ```
#[wasm_bindgen]
pub fn levenshtein(source: &str, target: &str) -> usize {
    standard_distance(source, target)
}

/// Calculate the Levenshtein distance with early termination.
///
/// Returns the distance only if it's less than or equal to the threshold,
/// otherwise returns `null`. This is more efficient when you only care
/// about matches within a certain distance.
///
/// # Arguments
///
/// * `source` - The source string
/// * `target` - The target string
/// * `threshold` - Maximum distance to compute
///
/// # Returns
///
/// The distance if <= threshold, otherwise `null`.
#[wasm_bindgen]
pub fn levenshtein_threshold(source: &str, target: &str, threshold: usize) -> Option<usize> {
    let dist = standard_distance(source, target);
    if dist <= threshold {
        Some(dist)
    } else {
        None
    }
}

/// Calculate the Damerau-Levenshtein distance between two strings.
///
/// The Damerau-Levenshtein distance extends Levenshtein by also allowing
/// transpositions (swapping two adjacent characters) as a single edit.
///
/// # Arguments
///
/// * `source` - The source string
/// * `target` - The target string
///
/// # Returns
///
/// The edit distance between the two strings.
///
/// # Example (JavaScript)
///
/// ```javascript
/// import { damerau_levenshtein } from 'liblevenshtein';
/// console.log(damerau_levenshtein("ab", "ba")); // 1 (transposition)
/// ```
#[wasm_bindgen]
pub fn damerau_levenshtein(source: &str, target: &str) -> usize {
    transposition_distance(source, target)
}

/// Calculate the Damerau-Levenshtein distance with early termination.
///
/// # Arguments
///
/// * `source` - The source string
/// * `target` - The target string
/// * `threshold` - Maximum distance to compute
///
/// # Returns
///
/// The distance if <= threshold, otherwise `null`.
#[wasm_bindgen]
pub fn damerau_levenshtein_threshold(
    source: &str,
    target: &str,
    threshold: usize,
) -> Option<usize> {
    let dist = transposition_distance(source, target);
    if dist <= threshold {
        Some(dist)
    } else {
        None
    }
}

/// Calculate edit distances for multiple pairs in batch.
///
/// More efficient than calling `levenshtein` multiple times when you have
/// many pairs to compare.
///
/// # Arguments
///
/// * `pairs` - Array of [source, target] string pairs as a flat array
///   (e.g., ["a", "b", "c", "d"] for pairs (a,b) and (c,d))
///
/// # Returns
///
/// Array of distances corresponding to each pair.
#[wasm_bindgen]
pub fn levenshtein_batch(pairs: Vec<JsValue>) -> Result<Vec<usize>, JsValue> {
    if pairs.len() % 2 != 0 {
        return Err(JsValue::from_str(
            "pairs array must have even length (source, target pairs)",
        ));
    }

    let mut results = Vec::with_capacity(pairs.len() / 2);

    for chunk in pairs.chunks(2) {
        let source = chunk[0]
            .as_string()
            .ok_or_else(|| JsValue::from_str("source must be a string"))?;
        let target = chunk[1]
            .as_string()
            .ok_or_else(|| JsValue::from_str("target must be a string"))?;
        results.push(standard_distance(&source, &target));
    }

    Ok(results)
}
