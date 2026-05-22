//! Phonetic rules bindings for WebAssembly.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::phonetic::llev::LLevError;
use crate::phonetic::{parse_str, RuleSetChar};

/// A phonetic rewrite variant.
#[derive(Serialize, Deserialize)]
pub struct WasmRewriteVariant {
    /// The rewritten text
    pub text: String,
}

/// A WebAssembly-compatible phonetic rule set.
///
/// Phonetic rules allow you to normalize text by applying sound-based
/// transformations. For example, "ph" -> "f" or "ough" -> ["o", "oo", "off"].
///
/// # Example (JavaScript)
///
/// ```javascript
/// import { WasmRuleSet } from 'liblevenshtein';
///
/// // Load rules from LLEV format
/// const rules = WasmRuleSet.parse(`
///     @name "My Rules"
///     ph -> f;
///     tion -> shun;
/// `);
///
/// // Apply transformations
/// const result = rules.rewrite("phone");
/// console.log(result); // "fone"
/// ```
#[wasm_bindgen]
pub struct WasmRuleSet {
    inner: RuleSetChar,
}

#[wasm_bindgen]
impl WasmRuleSet {
    /// Parse rules from LLEV format string.
    ///
    /// # Arguments
    ///
    /// * `source` - LLEV rules source code
    ///
    /// # Returns
    ///
    /// A new `WasmRuleSet` or an error if parsing fails.
    #[wasm_bindgen]
    pub fn parse(source: &str) -> Result<WasmRuleSet, JsValue> {
        // Parse the LLEV file
        let file = parse_str(source).map_err(|e: LLevError| JsValue::from_str(&e.to_string()))?;

        // Convert to runtime rule set
        let inner = RuleSetChar::from_llev(&file)
            .map_err(|e: LLevError| JsValue::from_str(&e.to_string()))?;

        Ok(WasmRuleSet { inner })
    }

    /// Get the number of rules in the set.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the rule set is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Apply rules to rewrite a term.
    ///
    /// # Arguments
    ///
    /// * `term` - The term to rewrite
    ///
    /// # Returns
    ///
    /// The rewritten term after applying all matching rules.
    pub fn rewrite(&self, term: &str) -> String {
        self.inner.apply(term)
    }

    /// Get the name of the rule set (if defined).
    pub fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    /// Get the version of the rule set (if defined).
    pub fn version(&self) -> Option<String> {
        self.inner.version.clone()
    }
}

/// Apply built-in English spelling normalization rules.
///
/// This uses the Zompist orthography rules to normalize English spellings:
/// - "ph" -> "f" (phone -> fone)
/// - "ough" -> various forms
/// - Silent letters removed
/// - etc.
///
/// # Example (JavaScript)
///
/// ```javascript
/// import { normalize_english } from 'liblevenshtein';
/// console.log(normalize_english("phone")); // "fone"
/// console.log(normalize_english("enough")); // "enuf"
/// ```
#[wasm_bindgen]
pub fn normalize_english(text: &str) -> String {
    use crate::phonetic::orthography_rules_char;
    let rules = orthography_rules_char();
    let ruleset = RuleSetChar {
        rules,
        name: None,
        version: None,
    };
    ruleset.apply(text)
}

/// Apply built-in phonetic transformation rules.
///
/// This uses comprehensive phonetic rules to transform text to a
/// phonetic representation.
///
/// # Example (JavaScript)
///
/// ```javascript
/// import { phonetic_transform } from 'liblevenshtein';
/// console.log(phonetic_transform("knight")); // transforms to phonetic form
/// ```
#[wasm_bindgen]
pub fn phonetic_transform(text: &str) -> String {
    use crate::phonetic::phonetic_rules_char;
    let rules = phonetic_rules_char();
    let ruleset = RuleSetChar {
        rules,
        name: None,
        version: None,
    };
    ruleset.apply(text)
}
