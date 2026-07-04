use wasm_bindgen::JsValue;

use libdictenstein::double_array_trie::{DoubleArrayTrie, DoubleArrayTrieBuilder};
use libdictenstein::dynamic_dawg::DynamicDawg;

pub(super) fn parse_sorted_terms(terms: Vec<JsValue>) -> Result<Vec<String>, JsValue> {
    let mut parsed = Vec::with_capacity(terms.len());

    for value in terms {
        let term = value
            .as_string()
            .ok_or_else(|| JsValue::from_str("all terms must be strings"))?;
        parsed.push(term);
    }

    parsed.sort_unstable();
    parsed.dedup();
    Ok(parsed)
}

pub(super) fn double_array_trie_from_sorted_terms(
    terms: impl IntoIterator<Item = String>,
) -> DoubleArrayTrie<()> {
    let mut builder = DoubleArrayTrieBuilder::new();
    for term in terms {
        builder.insert(&term);
    }
    builder.build()
}

pub(super) fn dynamic_dawg_from_sorted_terms(terms: Vec<String>) -> DynamicDawg<()> {
    DynamicDawg::from_sorted_terms(terms)
}

#[cfg(test)]
mod tests {
    use super::{double_array_trie_from_sorted_terms, dynamic_dawg_from_sorted_terms};
    use libdictenstein::Dictionary;

    #[test]
    fn double_array_trie_helper_builds_expected_dictionary() {
        let terms = vec!["alpha".to_owned(), "beta".to_owned()];
        let dict = double_array_trie_from_sorted_terms(terms);

        assert_eq!(dict.len(), Some(2));
        assert!(dict.contains("alpha"));
        assert!(dict.contains("beta"));
        assert!(!dict.contains("gamma"));
    }

    #[test]
    fn dynamic_dawg_helper_builds_expected_dictionary() {
        let terms = vec!["alpha".to_owned(), "beta".to_owned()];
        let dict = dynamic_dawg_from_sorted_terms(terms);

        assert_eq!(dict.len(), Some(2));
        assert!(dict.contains("alpha"));
        assert!(dict.contains("beta"));
        assert!(!dict.contains("gamma"));
    }
}
