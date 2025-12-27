//! Bidirectional extension for WallBreaker algorithm.
//!
//! After finding an exact substring match, WallBreaker extends bidirectionally:
//! - **Left extension**: Traverse backward from match toward dictionary root
//! - **Right extension**: Traverse forward from match toward leaves
//!
//! Both directions use Levenshtein filters to prune the search space.

use crate::dictionary::substring::{BidirectionalDictionaryNode, SubstringMatch};
use crate::dictionary::DictionaryNode;

/// State during bidirectional extension.
///
/// Tracks the current position in both the query and dictionary,
/// along with accumulated edit distance.
#[derive(Debug, Clone)]
pub struct ExtensionState<N: DictionaryNode> {
    /// Current dictionary node.
    pub node: N,

    /// Position in the query (character index).
    pub query_pos: usize,

    /// Accumulated edit distance so far.
    pub distance: usize,

    /// Path labels collected during extension (for term reconstruction).
    pub labels: Vec<N::Unit>,
}

impl<N: DictionaryNode> ExtensionState<N> {
    /// Create a new extension state.
    pub fn new(node: N, query_pos: usize, distance: usize) -> Self {
        ExtensionState {
            node,
            query_pos,
            distance,
            labels: Vec::new(),
        }
    }

    /// Create with initial labels.
    pub fn with_labels(node: N, query_pos: usize, distance: usize, labels: Vec<N::Unit>) -> Self {
        ExtensionState {
            node,
            query_pos,
            distance,
            labels,
        }
    }
}

/// Bidirectional extension from a substring match.
///
/// Given an exact substring match, extends left (toward root) and right
/// (toward leaves) to find all complete terms within the distance bound.
pub struct BidirectionalExtension<'a, N>
where
    N: BidirectionalDictionaryNode,
    N::Unit: Into<u32>,
{
    /// The original substring match to extend from.
    match_info: &'a SubstringMatch<N>,

    /// Maximum allowed edit distance.
    max_distance: usize,

    /// The query string (as characters).
    query_chars: Vec<char>,

    /// Query piece that matched exactly.
    piece_start: usize,
    piece_end: usize,
}

impl<'a, N> BidirectionalExtension<'a, N>
where
    N: BidirectionalDictionaryNode,
    N::Unit: Into<u32>,
{
    /// Create a new bidirectional extension.
    ///
    /// # Arguments
    ///
    /// * `match_info` - The substring match to extend from
    /// * `query` - The full query string
    /// * `piece_start` - Start of the matched piece in the query
    /// * `piece_end` - End of the matched piece in the query
    /// * `max_distance` - Maximum allowed edit distance
    pub fn new(
        match_info: &'a SubstringMatch<N>,
        query: &str,
        piece_start: usize,
        piece_end: usize,
        max_distance: usize,
    ) -> Self {
        BidirectionalExtension {
            match_info,
            max_distance,
            query_chars: query.chars().collect(),
            piece_start,
            piece_end,
        }
    }

    /// Extend and find all valid (term, distance) pairs.
    ///
    /// Returns a vector of (term, total_distance) for all dictionary terms
    /// reachable within the distance bound.
    pub fn extend(&self) -> Vec<(String, usize)> {
        let mut results = Vec::new();

        // Get the left extension results (prefix possibilities)
        let left_states = self.extend_left();

        // For each left extension, extend right
        for left_state in left_states {
            let right_results = self.extend_right(&left_state);
            results.extend(right_results);
        }

        results
    }

    /// Extend leftward from the match position toward root.
    ///
    /// This handles the portion of the query before the matched piece.
    fn extend_left(&self) -> Vec<LeftExtensionState<N>> {
        let mut states = Vec::new();

        // Characters before the piece in the query
        let query_prefix: Vec<char> = self.query_chars[..self.piece_start].iter().copied().collect();

        // Start from the beginning of the matched substring in the dictionary
        // We need to find the node at the START of the match
        let start_node = self.find_match_start_node();

        if let Some(start) = start_node {
            // Initial state: at match start, need to match query prefix
            let initial = LeftExtensionState {
                node: start,
                query_remaining: query_prefix.len(),
                distance: 0,
                prefix_labels: Vec::new(),
            };

            self.extend_left_recursive(initial, &query_prefix, &mut states);
        }

        states
    }

    /// Find the node at the START of the matched substring.
    fn find_match_start_node(&self) -> Option<N> {
        // Navigate from root to the position just before the match
        let _term_chars: Vec<char> = self.match_info.term.chars().collect();

        // The match starts at position `match_info.position` in the term
        // We need to traverse to that position
        if self.match_info.position == 0 {
            // Match is at the beginning - start from root
            // We need to get root, but we only have the end node
            // Walk back through parent links
            let mut current = self.match_info.node.clone();
            let mut depth = 0;

            // Count how deep we need to go back
            while let Some(_) = current.parent() {
                depth += 1;
                if depth >= self.match_info.position + self.match_info.length {
                    break;
                }
                if let Some(parent) = current.parent() {
                    current = parent;
                } else {
                    break;
                }
            }

            // Now current should be near root
            // For position 0, we return the root's child at the first char
            Some(current)
        } else {
            // Navigate from root to position
            let mut current = self.match_info.node.clone();

            // Walk back to the start of the match
            for _ in 0..self.match_info.length {
                if let Some(parent) = current.parent() {
                    current = parent;
                } else {
                    break;
                }
            }

            Some(current)
        }
    }

    /// Recursive left extension using Levenshtein transitions.
    fn extend_left_recursive(
        &self,
        state: LeftExtensionState<N>,
        query_prefix: &[char],
        results: &mut Vec<LeftExtensionState<N>>,
    ) {
        // Base case: consumed all query prefix
        if state.query_remaining == 0 {
            // Must have reached root or a valid prefix point
            if state.node.is_root() || state.distance <= self.max_distance {
                results.push(state);
            }
            return;
        }

        // Pruning: if distance exceeds bound, stop
        if state.distance > self.max_distance {
            return;
        }

        // Get the query character we're trying to match (from the end of prefix)
        let query_idx = query_prefix.len() - state.query_remaining;
        let query_char = query_prefix[query_idx];

        // Try match (consume query char, move to parent)
        if let (Some(parent), Some(label)) = (state.node.parent(), state.node.parent_label()) {
            // Check if label matches query char
            let matches = label_matches_char(label, query_char);

            if matches {
                // Match: no distance increase
                let mut new_labels = state.prefix_labels.clone();
                new_labels.push(label);
                let new_state = LeftExtensionState {
                    node: parent,
                    query_remaining: state.query_remaining - 1,
                    distance: state.distance,
                    prefix_labels: new_labels,
                };
                self.extend_left_recursive(new_state, query_prefix, results);
            } else {
                // Substitution: distance + 1
                if state.distance + 1 <= self.max_distance {
                    let mut new_labels = state.prefix_labels.clone();
                    new_labels.push(label);
                    let new_state = LeftExtensionState {
                        node: parent,
                        query_remaining: state.query_remaining - 1,
                        distance: state.distance + 1,
                        prefix_labels: new_labels,
                    };
                    self.extend_left_recursive(new_state, query_prefix, results);
                }
            }
        }

        // Try insertion (consume query char without moving in dictionary)
        if state.distance + 1 <= self.max_distance {
            let new_state = LeftExtensionState {
                node: state.node.clone(),
                query_remaining: state.query_remaining - 1,
                distance: state.distance + 1,
                prefix_labels: state.prefix_labels.clone(),
            };
            self.extend_left_recursive(new_state, query_prefix, results);
        }

        // Try deletion (move in dictionary without consuming query)
        if let (Some(parent), Some(label)) = (state.node.parent(), state.node.parent_label()) {
            if state.distance + 1 <= self.max_distance {
                let mut new_labels = state.prefix_labels.clone();
                new_labels.push(label);
                let new_state = LeftExtensionState {
                    node: parent,
                    query_remaining: state.query_remaining,
                    distance: state.distance + 1,
                    prefix_labels: new_labels,
                };
                self.extend_left_recursive(new_state, query_prefix, results);
            }
        }
    }

    /// Extend rightward from a left extension state.
    fn extend_right(&self, left_state: &LeftExtensionState<N>) -> Vec<(String, usize)> {
        let mut results = Vec::new();

        // Characters after the piece in the query
        let query_suffix: Vec<char> = self.query_chars[self.piece_end..].iter().copied().collect();

        // Start from the end of the matched substring
        let initial = RightExtensionState {
            node: self.match_info.node.clone(),
            query_remaining: query_suffix.len(),
            distance: left_state.distance,
            suffix_labels: Vec::new(),
        };

        let right_states = self.extend_right_recursive(initial, &query_suffix);

        // Combine left prefix + matched piece + right suffix
        for right_state in right_states {
            if right_state.distance <= self.max_distance && right_state.node.is_final() {
                // Build the complete term
                // Left labels are in reverse order, need to reverse
                let mut term_chars: Vec<char> = Vec::new();

                // Add left prefix (reversed)
                for label in left_state.prefix_labels.iter().rev() {
                    if let Some(c) = label_to_char(*label) {
                        term_chars.push(c);
                    }
                }

                // Add the matched piece
                for c in self.match_info.term.chars() {
                    term_chars.push(c);
                }

                // Note: The matched piece is already part of the term,
                // we need to be careful not to double-count

                let term: String = term_chars.into_iter().collect();
                results.push((term, right_state.distance));
            }
        }

        results
    }

    /// Recursive right extension.
    fn extend_right_recursive(
        &self,
        state: RightExtensionState<N>,
        query_suffix: &[char],
    ) -> Vec<RightExtensionState<N>> {
        let mut results = Vec::new();

        // Base case: consumed all query suffix
        if state.query_remaining == 0 {
            // Check if we're at a final node
            if state.distance <= self.max_distance {
                results.push(state);
            }
            return results;
        }

        // Pruning
        if state.distance > self.max_distance {
            return results;
        }

        let query_idx = query_suffix.len() - state.query_remaining;
        let query_char = query_suffix[query_idx];

        // Try all forward edges
        for (label, child) in state.node.edges() {
            let matches = label_matches_char(label, query_char);

            if matches {
                // Match
                let mut new_labels = state.suffix_labels.clone();
                new_labels.push(label);
                let new_state = RightExtensionState {
                    node: child,
                    query_remaining: state.query_remaining - 1,
                    distance: state.distance,
                    suffix_labels: new_labels,
                };
                results.extend(self.extend_right_recursive(new_state, query_suffix));
            } else if state.distance + 1 <= self.max_distance {
                // Substitution
                let mut new_labels = state.suffix_labels.clone();
                new_labels.push(label);
                let new_state = RightExtensionState {
                    node: child,
                    query_remaining: state.query_remaining - 1,
                    distance: state.distance + 1,
                    suffix_labels: new_labels,
                };
                results.extend(self.extend_right_recursive(new_state, query_suffix));
            }
        }

        // Insertion (skip query char)
        if state.distance + 1 <= self.max_distance {
            let new_state = RightExtensionState {
                node: state.node.clone(),
                query_remaining: state.query_remaining - 1,
                distance: state.distance + 1,
                suffix_labels: state.suffix_labels.clone(),
            };
            results.extend(self.extend_right_recursive(new_state, query_suffix));
        }

        // Deletion (skip dictionary edge)
        for (label, child) in state.node.edges() {
            if state.distance + 1 <= self.max_distance {
                let mut new_labels = state.suffix_labels.clone();
                new_labels.push(label);
                let new_state = RightExtensionState {
                    node: child,
                    query_remaining: state.query_remaining,
                    distance: state.distance + 1,
                    suffix_labels: new_labels,
                };
                results.extend(self.extend_right_recursive(new_state, query_suffix));
            }
        }

        results
    }
}

/// State for left (backward) extension.
#[derive(Clone)]
struct LeftExtensionState<N: DictionaryNode> {
    node: N,
    query_remaining: usize,
    distance: usize,
    prefix_labels: Vec<N::Unit>,
}

/// State for right (forward) extension.
#[derive(Clone)]
struct RightExtensionState<N: DictionaryNode> {
    node: N,
    query_remaining: usize,
    distance: usize,
    suffix_labels: Vec<N::Unit>,
}

/// Check if a label matches a query character.
///
/// Works for both u8 (byte) and char (Unicode) labels.
fn label_matches_char<U>(label: U, query_char: char) -> bool
where
    U: Copy + PartialEq + Into<u32>,
{
    let label_u32: u32 = label.into();
    label_u32 == query_char as u32
}

/// Convert a label to a char (for term reconstruction).
fn label_to_char<U>(label: U) -> Option<char>
where
    U: Copy + Into<u32>,
{
    let label_u32: u32 = label.into();
    char::from_u32(label_u32)
}

#[cfg(test)]
mod tests {
    // Tests would require a mock dictionary node implementation
    // Real tests are in the parent module using Scdawg
}
