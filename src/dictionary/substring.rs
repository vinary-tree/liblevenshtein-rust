//! Substring search and bidirectional traversal traits for WallBreaker algorithm.
//!
//! This module provides the foundational abstractions needed for the WallBreaker
//! approximate string matching algorithm, which overcomes the "wall effect" in
//! traditional Levenshtein automata by:
//!
//! 1. **Splitting the query** into `b+1` pieces (pigeonhole principle)
//! 2. **Finding exact substring matches** for each piece in the dictionary
//! 3. **Extending bidirectionally** from matches using Levenshtein filters
//!
//! # Key Types
//!
//! - [`SubstringMatch`] - Result of an exact substring search
//! - [`SubstringDictionary`] - Trait for dictionaries supporting substring search
//! - [`BidirectionalDictionaryNode`] - Trait for nodes supporting reverse traversal
//!
//! # Algorithm Overview
//!
//! The WallBreaker algorithm addresses the "wall effect" where traditional
//! Levenshtein automata must explore ALL dictionary prefixes up to the error
//! bound `b` before any filtering occurs. By using the pigeonhole principle
//! (at least one of `b+1` pieces must match exactly), WallBreaker achieves
//! up to 5600x speedup for large error bounds.
//!
//! Reference: "WallBreaker - overcoming the wall effect in similarity search"
//! (Gerdjikov, Mihov, Mitankin, Schulz - EDBT/ICDT 2013)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::dictionary::substring::{SubstringMatch, SubstringDictionary};
//! use liblevenshtein::dictionary::scdawg::Scdawg;
//!
//! // Create an SCDAWG (Symmetric Compact DAWG) dictionary
//! let dict = Scdawg::<()>::from_terms(vec!["cathedral", "cathedrals", "category"]);
//!
//! // Find all terms containing "thedr"
//! let matches: Vec<SubstringMatch<_>> = dict.find_exact_substring("thedr");
//!
//! for m in &matches {
//!     println!("Found '{}' at position {} in '{}'",
//!         &m.term[m.position..m.position + m.length],
//!         m.position,
//!         m.term);
//! }
//! // Output:
//! // Found 'thedr' at position 2 in 'cathedral'
//! // Found 'thedr' at position 2 in 'cathedrals'
//! ```

use super::{CharUnit, Dictionary, DictionaryNode};

/// Result of finding an exact substring match in a dictionary.
///
/// When searching for a pattern `P` within dictionary terms, this struct
/// captures where the pattern was found and provides access to the dictionary
/// node at the end of the match for further traversal.
///
/// # Type Parameters
///
/// * `N` - The dictionary node type, which must implement [`DictionaryNode`].
///
/// # Fields
///
/// * `node` - The dictionary node at the end of the matched substring
/// * `term` - The complete dictionary term containing the match
/// * `position` - The 0-indexed character position where the match starts
/// * `length` - The length of the matched substring (in characters)
///
/// # Invariants
///
/// * `position + length <= term.chars().count()`
/// * `&term[char_offset(position)..char_offset(position + length)] == pattern`
///
/// # Example
///
/// ```rust,ignore
/// // Searching for "ello" in dictionary ["hello", "yellow", "mellow"]
/// // Returns matches:
/// SubstringMatch {
///     node: /* node at 'o' in "hello" */,
///     term: "hello".to_string(),
///     position: 1,  // 'e' is at index 1
///     length: 4,    // "ello" is 4 characters
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SubstringMatch<N>
where
    N: DictionaryNode,
{
    /// The dictionary node at the end of the matched substring.
    ///
    /// This node can be used for:
    /// - Forward extension (following `edges()` to continue matching)
    /// - Backward extension (following `parent()` if `BidirectionalDictionaryNode`)
    pub node: N,

    /// The complete dictionary term containing the matched substring.
    pub term: String,

    /// The 0-indexed character position where the match starts in `term`.
    ///
    /// For byte-level dictionaries, this is a byte offset.
    /// For character-level dictionaries, this is a character offset.
    pub position: usize,

    /// The length of the matched substring.
    ///
    /// For byte-level dictionaries, this is in bytes.
    /// For character-level dictionaries, this is in characters.
    pub length: usize,
}

impl<N: DictionaryNode> SubstringMatch<N> {
    /// Create a new substring match.
    ///
    /// # Arguments
    ///
    /// * `node` - The dictionary node at the end of the match
    /// * `term` - The complete dictionary term
    /// * `position` - Start position of the match in the term
    /// * `length` - Length of the matched substring
    pub fn new(node: N, term: String, position: usize, length: usize) -> Self {
        SubstringMatch {
            node,
            term,
            position,
            length,
        }
    }

    /// Get the matched substring.
    ///
    /// Returns the portion of `term` that was matched.
    ///
    /// # Note
    ///
    /// This method assumes character-level positioning. For byte-level
    /// dictionaries, use `matched_substring_bytes()` instead.
    pub fn matched_substring(&self) -> &str {
        // Handle both byte-level and char-level positioning
        let start_byte = self
            .term
            .char_indices()
            .nth(self.position)
            .map(|(i, _)| i)
            .unwrap_or(self.term.len());

        let end_byte = self
            .term
            .char_indices()
            .nth(self.position + self.length)
            .map(|(i, _)| i)
            .unwrap_or(self.term.len());

        &self.term[start_byte..end_byte]
    }

    /// Get the prefix of the term before the match.
    ///
    /// Returns the portion of `term` before the matched substring.
    pub fn prefix(&self) -> &str {
        let start_byte = self
            .term
            .char_indices()
            .nth(self.position)
            .map(|(i, _)| i)
            .unwrap_or(0);

        &self.term[..start_byte]
    }

    /// Get the suffix of the term after the match.
    ///
    /// Returns the portion of `term` after the matched substring.
    pub fn suffix(&self) -> &str {
        let end_byte = self
            .term
            .char_indices()
            .nth(self.position + self.length)
            .map(|(i, _)| i)
            .unwrap_or(self.term.len());

        &self.term[end_byte..]
    }

    /// Get the number of characters before the match (left context).
    #[inline]
    pub fn left_context_len(&self) -> usize {
        self.position
    }

    /// Get the number of characters after the match (right context).
    #[inline]
    pub fn right_context_len(&self) -> usize {
        self.term.chars().count().saturating_sub(self.position + self.length)
    }
}

/// Dictionary supporting exact substring search.
///
/// This trait extends [`Dictionary`] with the ability to find all terms
/// containing a given pattern as a substring. This is the foundation for
/// the WallBreaker algorithm's "piece matching" phase.
///
/// # Implementation Notes
///
/// Efficient substring search typically requires specialized data structures
/// like SCDAWG (Symmetric Compact DAWG) or suffix automata. The reference
/// implementation uses SCDAWG which provides:
///
/// - O(|pattern|) time to locate the pattern
/// - O(occurrences) time to enumerate all matches
/// - Linear space construction
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::dictionary::substring::SubstringDictionary;
///
/// fn find_candidates<D: SubstringDictionary>(dict: &D, piece: &str) {
///     let matches = dict.find_exact_substring(piece);
///     println!("Found {} terms containing '{}'", matches.len(), piece);
///
///     for m in matches {
///         println!("  {} (at position {})", m.term, m.position);
///     }
/// }
/// ```
pub trait SubstringDictionary: Dictionary {
    /// Find all dictionary terms containing the exact substring.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The substring to search for
    ///
    /// # Returns
    ///
    /// A vector of [`SubstringMatch`] structs, one for each occurrence of
    /// `pattern` in the dictionary. A term containing the pattern multiple
    /// times will appear multiple times with different `position` values.
    ///
    /// # Performance
    ///
    /// For SCDAWG-based dictionaries:
    /// - Time: O(|pattern| + occurrences)
    /// - The returned vector is typically small for non-trivial patterns
    fn find_exact_substring(&self, pattern: &str) -> Vec<SubstringMatch<Self::Node>>;

    /// Find all dictionary terms containing the exact substring, with limit.
    ///
    /// Like `find_exact_substring`, but stops after finding `limit` matches.
    /// Useful for early termination in search algorithms.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The substring to search for
    /// * `limit` - Maximum number of matches to return
    ///
    /// # Returns
    ///
    /// A vector of at most `limit` [`SubstringMatch`] structs.
    fn find_exact_substring_limited(
        &self,
        pattern: &str,
        limit: usize,
    ) -> Vec<SubstringMatch<Self::Node>> {
        let mut results = self.find_exact_substring(pattern);
        results.truncate(limit);
        results
    }

    /// Check if any term contains the exact substring.
    ///
    /// More efficient than `!find_exact_substring(pattern).is_empty()`
    /// for implementations that can short-circuit.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The substring to search for
    ///
    /// # Returns
    ///
    /// `true` if at least one term contains `pattern`, `false` otherwise.
    fn contains_substring(&self, pattern: &str) -> bool {
        !self.find_exact_substring_limited(pattern, 1).is_empty()
    }

    /// Count the number of terms containing the exact substring.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The substring to search for
    ///
    /// # Returns
    ///
    /// The number of (term, position) pairs where `pattern` occurs.
    fn count_substring_matches(&self, pattern: &str) -> usize {
        self.find_exact_substring(pattern).len()
    }
}

/// Dictionary node supporting bidirectional (forward and backward) traversal.
///
/// This trait extends [`DictionaryNode`] with the ability to traverse
/// backward toward the root, which is essential for the WallBreaker
/// algorithm's left-extension phase.
///
/// # Motivation
///
/// Traditional dictionary nodes only support forward traversal (root → leaves).
/// For WallBreaker's bidirectional extension, we need to:
///
/// 1. Find a substring match (the "anchor point")
/// 2. Extend LEFT by traversing backward toward the root
/// 3. Extend RIGHT by traversing forward toward leaves
///
/// # Implementation Notes
///
/// Backward traversal requires additional data structure support:
///
/// - **Parent links**: Direct pointer to spanning tree parent
/// - **Reverse edges**: Mapping from (child, label) → parents
/// - **Suffix/Prefix links**: For SCDAWG navigation
///
/// The SCDAWG implementation stores parent links during construction,
/// using ~8-16 extra bytes per node.
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::dictionary::substring::BidirectionalDictionaryNode;
///
/// fn reconstruct_path<N: BidirectionalDictionaryNode>(mut node: N) -> String {
///     let mut chars = Vec::new();
///
///     // Walk backward from node to root, collecting edge labels
///     while let Some(parent) = node.parent() {
///         if let Some(label) = node.parent_label() {
///             chars.push(label);
///         }
///         node = parent;
///     }
///
///     // Reverse to get the path from root to node
///     chars.reverse();
///     chars.into_iter().collect()
/// }
/// ```
pub trait BidirectionalDictionaryNode: DictionaryNode {
    /// Get the parent node in the spanning tree (toward root).
    ///
    /// Returns `None` if this is the root node.
    ///
    /// # Note
    ///
    /// For DAWG-like structures where a node may have multiple paths from root,
    /// this returns the canonical parent (typically from construction order).
    fn parent(&self) -> Option<Self>;

    /// Get the edge label leading from parent to this node.
    ///
    /// Returns `None` if this is the root node (no parent edge).
    fn parent_label(&self) -> Option<Self::Unit>;

    /// Iterate over reverse edges (edges pointing toward this node).
    ///
    /// For each edge, returns `(label, parent_node)` where following `label`
    /// from `parent_node` leads to `self`.
    ///
    /// # Note
    ///
    /// Unlike `parent()` which returns a single canonical parent, this
    /// returns ALL nodes that have edges pointing to this node. This is
    /// important for DAG structures where multiple paths may lead to the
    /// same node.
    fn reverse_edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_>;

    /// Perform a reverse transition: find all parent nodes reachable via the given label.
    ///
    /// Returns all nodes `P` such that `P.transition(label) == Some(self)`.
    ///
    /// # Arguments
    ///
    /// * `label` - The edge label to match
    ///
    /// # Returns
    ///
    /// A vector of parent nodes (empty if no such edges exist).
    fn reverse_transition(&self, label: Self::Unit) -> Vec<Self>;

    /// Get the depth of this node from the root.
    ///
    /// The root has depth 0, its children have depth 1, etc.
    ///
    /// For DAWG structures where a node may be reached by paths of different
    /// lengths, this returns the length of the canonical path (via `parent()`).
    fn depth(&self) -> usize;

    /// Check if this node is the root.
    fn is_root(&self) -> bool {
        self.parent().is_none()
    }

    /// Get the path from root to this node as a sequence of labels.
    ///
    /// This follows the canonical `parent()` path backward and reverses it.
    fn path_from_root(&self) -> Vec<Self::Unit> {
        let mut labels = Vec::new();
        let mut current = self.clone();

        while let Some(label) = current.parent_label() {
            labels.push(label);
            if let Some(parent) = current.parent() {
                current = parent;
            } else {
                break;
            }
        }

        labels.reverse();
        labels
    }

    /// Get the path from root to this node as a string.
    ///
    /// Convenience method that converts `path_from_root()` to a String.
    fn path_string(&self) -> String {
        let units: Vec<Self::Unit> = self.path_from_root();
        Self::Unit::to_string(&units)
    }
}

/// Extension result for WallBreaker algorithm.
///
/// Represents a potential match found by extending from a substring match
/// in both directions (left and right).
#[derive(Debug, Clone)]
pub struct ExtensionResult {
    /// The dictionary term that was matched.
    pub term: String,

    /// The total edit distance from the query to this term.
    pub distance: usize,

    /// The start position in the query that aligns with the term start.
    pub query_start: usize,

    /// The end position in the query that aligns with the term end.
    pub query_end: usize,
}

impl ExtensionResult {
    /// Create a new extension result.
    pub fn new(term: String, distance: usize, query_start: usize, query_end: usize) -> Self {
        ExtensionResult {
            term,
            distance,
            query_start,
            query_end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock node for testing SubstringMatch
    #[derive(Clone)]
    struct MockNode;

    impl DictionaryNode for MockNode {
        type Unit = char;

        fn is_final(&self) -> bool {
            false
        }

        fn transition(&self, _label: char) -> Option<Self> {
            None
        }

        fn edges(&self) -> Box<dyn Iterator<Item = (char, Self)> + '_> {
            Box::new(std::iter::empty())
        }
    }

    unsafe impl Send for MockNode {}
    unsafe impl Sync for MockNode {}

    #[test]
    fn test_substring_match_creation() {
        let node = MockNode;
        let m = SubstringMatch::new(node, "cathedral".to_string(), 2, 5);

        assert_eq!(m.term, "cathedral");
        assert_eq!(m.position, 2);
        assert_eq!(m.length, 5);
    }

    #[test]
    fn test_substring_match_matched_substring() {
        let node = MockNode;
        let m = SubstringMatch::new(node, "cathedral".to_string(), 2, 5);

        // "cathedral" with position=2, length=5 should give "thedr"
        assert_eq!(m.matched_substring(), "thedr");
    }

    #[test]
    fn test_substring_match_prefix() {
        let node = MockNode;
        let m = SubstringMatch::new(node, "cathedral".to_string(), 2, 5);

        // prefix is "ca"
        assert_eq!(m.prefix(), "ca");
    }

    #[test]
    fn test_substring_match_suffix() {
        let node = MockNode;
        let m = SubstringMatch::new(node, "cathedral".to_string(), 2, 5);

        // suffix is "al"
        assert_eq!(m.suffix(), "al");
    }

    #[test]
    fn test_substring_match_context_lengths() {
        let node = MockNode;
        let m = SubstringMatch::new(node, "cathedral".to_string(), 2, 5);

        assert_eq!(m.left_context_len(), 2);
        assert_eq!(m.right_context_len(), 2);
    }

    #[test]
    fn test_substring_match_unicode() {
        let node = MockNode;
        // "café" has 4 characters
        let m = SubstringMatch::new(node, "café".to_string(), 1, 2);

        // position=1, length=2 should give "af"
        assert_eq!(m.matched_substring(), "af");
        assert_eq!(m.prefix(), "c");
        assert_eq!(m.suffix(), "é");
    }

    #[test]
    fn test_substring_match_full_term() {
        let node = MockNode;
        let m = SubstringMatch::new(node, "hello".to_string(), 0, 5);

        assert_eq!(m.matched_substring(), "hello");
        assert_eq!(m.prefix(), "");
        assert_eq!(m.suffix(), "");
        assert_eq!(m.left_context_len(), 0);
        assert_eq!(m.right_context_len(), 0);
    }

    #[test]
    fn test_extension_result_creation() {
        let result = ExtensionResult::new("cathedral".to_string(), 2, 0, 9);

        assert_eq!(result.term, "cathedral");
        assert_eq!(result.distance, 2);
        assert_eq!(result.query_start, 0);
        assert_eq!(result.query_end, 9);
    }
}
