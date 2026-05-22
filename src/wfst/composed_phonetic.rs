//! Phonetic WFST Pipeline Builder.
//!
//! This module provides [`PhoneticPipelineBuilder`], a fluent API for constructing
//! multi-stage phonetic matching pipelines by composing WFSTs.
//!
//! # Pipeline Architecture
//!
//! A typical phonetic matching pipeline consists of:
//!
//! ```text
//! Input -> [PhoneticNFA/Rewrite] -> [Levenshtein] -> [Dictionary] -> [LM] -> Output
//!              ↓                        ↓               ↓            ↓
//!        phonetic_cost           edit_distance    dict_match    -log P
//! ```
//!
//! The total score is the sum of all component weights (tropical semiring).
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::wfst::PhoneticPipelineBuilder;
//! use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
//!
//! let dict = DynamicDawgChar::from_terms(vec!["phone", "fone", "help"]);
//!
//! // Build a phonetic pipeline
//! let pipeline = PhoneticPipelineBuilder::new()
//!     .phonetic_pattern("(ph|f)one")
//!     .max_edit_distance(2)
//!     .dictionary(&dict)
//!     .build()
//!     .expect("failed to build pipeline");
//! ```

use libdictenstein::{Dictionary, DictionaryNode};

use super::phonetic_rewrite_wfst::{RewriteRule, RewriteWfst};

/// Configuration for a phonetic matching pipeline.
///
/// This struct holds all configuration options for building a phonetic
/// WFST pipeline.
#[derive(Clone)]
pub struct PhoneticPipelineConfig {
    /// Phonetic pattern (regex syntax)
    pub pattern: Option<String>,
    /// Maximum edit distance
    pub max_distance: u8,
    /// Phonetic weight (cost for phonetic transformations)
    pub phonetic_weight: f64,
    /// Edit distance weight multiplier
    pub edit_weight: f64,
    /// Rewrite rules (alternative to pattern)
    pub rewrite_rules: Vec<RewriteRule>,
    /// Whether to allow identity (passthrough) transitions
    pub allow_identity: bool,
}

impl Default for PhoneticPipelineConfig {
    fn default() -> Self {
        Self {
            pattern: None,
            max_distance: 2,
            phonetic_weight: 0.0,
            edit_weight: 1.0,
            rewrite_rules: Vec::new(),
            allow_identity: true,
        }
    }
}

/// Builder for phonetic matching pipelines.
///
/// This provides a fluent API for configuring and constructing phonetic
/// WFST pipelines. The builder supports multiple configuration modes:
///
/// 1. **Pattern mode**: Use a phonetic regex pattern (e.g., "(ph|f)one")
/// 2. **Rule mode**: Use explicit rewrite rules (e.g., "ph -> f")
/// 3. **Combined mode**: Use both patterns and rules
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::wfst::PhoneticPipelineBuilder;
/// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
///
/// let dict = DynamicDawgChar::from_terms(vec!["phone", "fone"]);
///
/// // Pattern-based pipeline
/// let pattern_pipeline = PhoneticPipelineBuilder::new()
///     .phonetic_pattern("(ph|f)one")
///     .max_edit_distance(2)
///     .dictionary(&dict)
///     .build();
///
/// // Rule-based pipeline
/// let rule_pipeline = PhoneticPipelineBuilder::new()
///     .add_rewrite_rule("ph", "f", 0.1)
///     .add_rewrite_rule("c", "k", 0.2)
///     .max_edit_distance(1)
///     .dictionary(&dict)
///     .build();
/// ```
pub struct PhoneticPipelineBuilder<D = ()> {
    config: PhoneticPipelineConfig,
    dictionary: Option<D>,
}

impl PhoneticPipelineBuilder<()> {
    /// Create a new pipeline builder with default settings.
    pub fn new() -> PhoneticPipelineBuilder<()> {
        PhoneticPipelineBuilder {
            config: PhoneticPipelineConfig::default(),
            dictionary: None,
        }
    }
}

impl Default for PhoneticPipelineBuilder<()> {
    fn default() -> Self {
        PhoneticPipelineBuilder::new()
    }
}

// Configuration methods that don't require Dictionary bounds
impl<D> PhoneticPipelineBuilder<D> {
    /// Set the phonetic pattern (regex syntax).
    ///
    /// The pattern uses phonetic regex syntax:
    /// - `(a|b)`: Alternation (a or b)
    /// - `[abc]`: Character class
    /// - `a*`: Zero or more
    /// - `a+`: One or more
    /// - `a?`: Optional
    pub fn phonetic_pattern(mut self, pattern: &str) -> Self {
        self.config.pattern = Some(pattern.to_string());
        self
    }

    /// Set the maximum edit distance.
    pub fn max_edit_distance(mut self, distance: u8) -> Self {
        self.config.max_distance = distance;
        self
    }

    /// Set the phonetic weight (cost for phonetic transformations).
    pub fn phonetic_weight(mut self, weight: f64) -> Self {
        self.config.phonetic_weight = weight;
        self
    }

    /// Set the edit distance weight multiplier.
    pub fn edit_weight(mut self, weight: f64) -> Self {
        self.config.edit_weight = weight;
        self
    }

    /// Add a rewrite rule.
    pub fn add_rewrite_rule(mut self, input: &str, output: &str, cost: f64) -> Self {
        self.config
            .rewrite_rules
            .push(RewriteRule::with_cost(input, output, cost));
        self
    }

    /// Add multiple rewrite rules.
    pub fn add_rewrite_rules(mut self, rules: Vec<RewriteRule>) -> Self {
        self.config.rewrite_rules.extend(rules);
        self
    }

    /// Set whether to allow identity (passthrough) transitions.
    pub fn allow_identity(mut self, allow: bool) -> Self {
        self.config.allow_identity = allow;
        self
    }

    /// Set the dictionary to use.
    pub fn dictionary<D2>(self, dictionary: &D2) -> PhoneticPipelineBuilder<D2>
    where
        D2: Dictionary + Clone + Send + Sync + 'static,
        D2::Node: Send + Sync,
        <D2::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
    {
        PhoneticPipelineBuilder {
            config: self.config,
            dictionary: Some(dictionary.clone()),
        }
    }

    /// Build a rewrite WFST from the configured rules.
    ///
    /// This creates a standalone rewrite WFST without dictionary integration.
    pub fn build_rewrite_wfst(&self) -> RewriteWfst {
        let mut wfst = RewriteWfst::with_rules(self.config.rewrite_rules.clone());
        wfst.set_allow_identity(self.config.allow_identity);
        wfst
    }
}

// Phonetic NFA building (doesn't require dictionary)
#[cfg(feature = "phonetic-rules")]
impl<D> PhoneticPipelineBuilder<D> {
    /// Build a phonetic WFST from the configured pattern.
    ///
    /// This creates a phonetic NFA WFST from the pattern.
    pub fn build_phonetic_nfa(&self) -> Result<super::phonetic_nfa_wfst::PhoneticNfaWfst, String> {
        let pattern = self
            .config
            .pattern
            .as_ref()
            .ok_or_else(|| "No phonetic pattern specified".to_string())?;

        use crate::phonetic::nfa::compiler::compile;
        use crate::phonetic::regex::parse;

        let ast = parse(pattern).map_err(|e| format!("Parse error: {:?}", e))?;
        let nfa = compile(&ast).map_err(|e| format!("Compile error: {:?}", e))?;

        Ok(
            super::phonetic_nfa_wfst::PhoneticNfaWfst::with_phonetic_weight(
                nfa,
                self.config.phonetic_weight,
            ),
        )
    }
}

// Full pipeline building (requires dictionary)
#[cfg(feature = "phonetic-rules")]
impl<D> PhoneticPipelineBuilder<D>
where
    D: Dictionary + Clone + Send + Sync + 'static,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// Build a full phonetic pipeline with dictionary integration.
    ///
    /// This creates a PhoneticWfst that integrates the phonetic NFA,
    /// Levenshtein automaton, and dictionary.
    pub fn build(&self) -> Result<super::phonetic_wfst::PhoneticWfst<D>, String> {
        let dictionary = self
            .dictionary
            .as_ref()
            .ok_or_else(|| "No dictionary specified".to_string())?;

        let pattern = self
            .config
            .pattern
            .as_ref()
            .ok_or_else(|| "No phonetic pattern specified".to_string())?;

        use crate::phonetic::nfa::compiler::compile;
        use crate::phonetic::regex::parse;

        let ast = parse(pattern).map_err(|e| format!("Parse error: {:?}", e))?;
        let nfa = compile(&ast).map_err(|e| format!("Compile error: {:?}", e))?;

        Ok(super::phonetic_wfst::PhoneticWfst::with_phonetic_weight(
            dictionary,
            nfa,
            self.config.max_distance,
            self.config.phonetic_weight,
        ))
    }
}

/// Result of a phonetic pipeline search.
#[derive(Debug, Clone)]
pub struct PhoneticMatch {
    /// The matching term
    pub term: String,
    /// Total cost (phonetic + edit distance)
    pub total_cost: f64,
    /// Phonetic transformation cost component
    pub phonetic_cost: f64,
    /// Edit distance cost component
    pub edit_cost: f64,
}

impl PhoneticMatch {
    /// Create a new phonetic match result.
    pub fn new(term: String, phonetic_cost: f64, edit_cost: f64) -> Self {
        Self {
            term,
            total_cost: phonetic_cost + edit_cost,
            phonetic_cost,
            edit_cost,
        }
    }
}

impl Eq for PhoneticMatch {}

impl PartialEq for PhoneticMatch {
    fn eq(&self, other: &Self) -> bool {
        self.term == other.term && self.total_cost == other.total_cost
    }
}

impl PartialOrd for PhoneticMatch {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PhoneticMatch {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.total_cost.partial_cmp(&other.total_cost) {
            Some(std::cmp::Ordering::Equal) | None => self.term.cmp(&other.term),
            Some(ord) => ord,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PhoneticPipelineConfig::default();
        assert_eq!(config.max_distance, 2);
        assert_eq!(config.phonetic_weight, 0.0);
        assert!(config.allow_identity);
    }

    #[test]
    fn test_pipeline_builder_creation() {
        let builder = PhoneticPipelineBuilder::new();
        assert!(builder.config.pattern.is_none());
    }

    #[test]
    fn test_pipeline_builder_pattern() {
        let builder = PhoneticPipelineBuilder::new()
            .phonetic_pattern("(ph|f)one")
            .max_edit_distance(3)
            .phonetic_weight(0.5);

        assert_eq!(builder.config.pattern, Some("(ph|f)one".to_string()));
        assert_eq!(builder.config.max_distance, 3);
        assert_eq!(builder.config.phonetic_weight, 0.5);
    }

    #[test]
    fn test_pipeline_builder_rules() {
        let builder = PhoneticPipelineBuilder::new()
            .add_rewrite_rule("ph", "f", 0.1)
            .add_rewrite_rule("c", "k", 0.2);

        assert_eq!(builder.config.rewrite_rules.len(), 2);
    }

    #[test]
    fn test_pipeline_builder_rewrite_wfst() {
        let builder = PhoneticPipelineBuilder::new()
            .add_rewrite_rule("ph", "f", 0.1)
            .allow_identity(false);

        let wfst = builder.build_rewrite_wfst();
        assert_eq!(wfst.num_rules(), 1);
    }

    #[test]
    fn test_phonetic_match_ordering() {
        let m1 = PhoneticMatch::new("phone".to_string(), 0.0, 0.0);
        let m2 = PhoneticMatch::new("fone".to_string(), 0.1, 0.0);
        let m3 = PhoneticMatch::new("tone".to_string(), 0.0, 1.0);

        assert!(m1 < m2); // 0.0 < 0.1
        assert!(m1 < m3); // 0.0 < 1.0
        assert!(m2 < m3); // 0.1 < 1.0
    }

    #[test]
    #[cfg(feature = "phonetic-rules")]
    fn test_pipeline_builder_build_nfa() {
        let builder = PhoneticPipelineBuilder::new()
            .phonetic_pattern("(a|b)c")
            .phonetic_weight(0.1);

        let result = builder.build_phonetic_nfa();
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(feature = "phonetic-rules")]
    fn test_pipeline_builder_build_full() {
        use libdictenstein::dynamic_dawg_char::DynamicDawgChar;

        let dict = DynamicDawgChar::<()>::from_terms(vec!["phone", "fone", "help"]);

        let builder = PhoneticPipelineBuilder::new()
            .phonetic_pattern("(ph|f)one")
            .max_edit_distance(2)
            .dictionary(&dict);

        let result = builder.build();
        assert!(result.is_ok());

        let wfst = result.expect("test fixture: build must be Ok");
        assert_eq!(wfst.max_distance(), 2);
    }
}
