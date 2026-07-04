//! Completion candidate type for contextual completion.
//!
//! This module provides the `Completion` type representing a fuzzy match result
//! with context information, draft status, and edit distance.

use super::context_tree::ContextId;
use std::cmp::Ordering;

/// A completion candidate with metadata.
///
/// Represents a term that matches the user's query, along with metadata about
/// the match quality (distance), whether it's a draft or finalized term, and
/// which contexts it's visible in.
///
/// # Sorting
///
/// Completions are sorted by:
/// 1. Distance (ascending - closer matches first)
/// 2. Draft status (finalized terms before drafts)
/// 3. Term lexicographically (ascending)
///
/// # Examples
///
/// ```
/// use liblevenshtein::contextual::Completion;
///
/// // Create a finalized completion
/// let finalized = Completion::finalized("hello".to_string(), 1, vec![0, 1]);
/// assert!(!finalized.is_draft);
/// assert_eq!(finalized.distance, 1);
///
/// // Create a draft completion
/// let draft = Completion::draft("world".to_string(), 0, 2);
/// assert!(draft.is_draft);
/// assert_eq!(draft.contexts, vec![2]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    /// The completed term
    pub term: String,

    /// Edit distance from query
    pub distance: usize,

    /// Whether this is a draft (non-finalized) term
    pub is_draft: bool,

    /// Contexts where this term is visible
    pub contexts: Vec<ContextId>,
}

impl Completion {
    /// Create a completion for a finalized term.
    ///
    /// Finalized terms are stored in the dictionary and visible in the
    /// specified contexts.
    ///
    /// # Arguments
    ///
    /// * `term` - The completed term
    /// * `distance` - Edit distance from the query
    /// * `contexts` - Context IDs where this term is visible
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::Completion;
    ///
    /// let completion = Completion::finalized("test".to_string(), 0, vec![0]);
    /// assert!(!completion.is_draft);
    /// assert_eq!(completion.term, "test");
    /// assert_eq!(completion.distance, 0);
    /// assert_eq!(completion.contexts, vec![0]);
    /// ```
    pub fn finalized(term: String, distance: usize, contexts: Vec<ContextId>) -> Self {
        Self {
            term,
            distance,
            is_draft: false,
            contexts,
        }
    }

    /// Create a completion for a draft term.
    ///
    /// Draft terms are in-progress text in a specific context, not yet
    /// finalized into the dictionary.
    ///
    /// # Arguments
    ///
    /// * `term` - The draft term
    /// * `distance` - Edit distance from the query
    /// * `context` - Context ID where this draft exists
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::Completion;
    ///
    /// let completion = Completion::draft("draft".to_string(), 1, 5);
    /// assert!(completion.is_draft);
    /// assert_eq!(completion.term, "draft");
    /// assert_eq!(completion.distance, 1);
    /// assert_eq!(completion.contexts, vec![5]);
    /// ```
    pub fn draft(term: String, distance: usize, context: ContextId) -> Self {
        Self {
            term,
            distance,
            is_draft: true,
            contexts: vec![context],
        }
    }

    /// Check if this completion is visible in a specific context.
    ///
    /// # Arguments
    ///
    /// * `context` - Context ID to check
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::Completion;
    ///
    /// let completion = Completion::finalized("test".to_string(), 0, vec![0, 1, 2]);
    /// assert!(completion.is_visible_in(0));
    /// assert!(completion.is_visible_in(1));
    /// assert!(!completion.is_visible_in(3));
    /// ```
    pub fn is_visible_in(&self, context: ContextId) -> bool {
        self.contexts.contains(&context)
    }
}

impl Ord for Completion {
    /// Compare completions for sorting.
    ///
    /// Sort order:
    /// 1. Distance (ascending - closer matches first)
    /// 2. Draft status (finalized terms before drafts)
    /// 3. Term lexicographically (ascending)
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::Completion;
    ///
    /// let mut completions = vec![
    ///     Completion::draft("z".to_string(), 2, 0),
    ///     Completion::finalized("a".to_string(), 1, vec![0]),
    ///     Completion::draft("b".to_string(), 1, 0),
    ///     Completion::finalized("c".to_string(), 0, vec![0]),
    /// ];
    ///
    /// completions.sort();
    ///
    /// // Expected order: distance 0 (c), distance 1 finalized (a), distance 1 draft (b), distance 2 draft (z)
    /// assert_eq!(completions[0].term, "c");
    /// assert_eq!(completions[1].term, "a");
    /// assert_eq!(completions[2].term, "b");
    /// assert_eq!(completions[3].term, "z");
    /// ```
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.is_draft.cmp(&other.is_draft))
            .then_with(|| self.term.cmp(&other.term))
    }
}

impl PartialOrd for Completion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl std::fmt::Display for Completion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.is_draft { "draft" } else { "finalized" };
        write!(
            f,
            "{} (distance: {}, {}, contexts: {:?})",
            self.term, self.distance, status, self.contexts
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finalized_creation() {
        let completion = Completion::finalized("test".to_string(), 1, vec![0, 1]);
        assert_eq!(completion.term, "test");
        assert_eq!(completion.distance, 1);
        assert!(!completion.is_draft);
        assert_eq!(completion.contexts, vec![0, 1]);
    }

    #[test]
    fn test_draft_creation() {
        let completion = Completion::draft("draft".to_string(), 2, 5);
        assert_eq!(completion.term, "draft");
        assert_eq!(completion.distance, 2);
        assert!(completion.is_draft);
        assert_eq!(completion.contexts, vec![5]);
    }

    #[test]
    fn test_is_visible_in() {
        let completion = Completion::finalized("test".to_string(), 0, vec![0, 1, 2]);
        assert!(completion.is_visible_in(0));
        assert!(completion.is_visible_in(1));
        assert!(completion.is_visible_in(2));
        assert!(!completion.is_visible_in(3));
    }

    #[test]
    fn test_sorting_by_distance() {
        let mut completions = [
            Completion::finalized("c".to_string(), 2, vec![0]),
            Completion::finalized("b".to_string(), 1, vec![0]),
            Completion::finalized("a".to_string(), 0, vec![0]),
        ];

        completions.sort();

        assert_eq!(completions[0].term, "a"); // distance 0
        assert_eq!(completions[1].term, "b"); // distance 1
        assert_eq!(completions[2].term, "c"); // distance 2
    }

    #[test]
    fn test_sorting_by_draft_status() {
        let mut completions = [
            Completion::draft("draft".to_string(), 1, 0),
            Completion::finalized("finalized".to_string(), 1, vec![0]),
        ];

        completions.sort();

        // Finalized comes before draft at same distance
        assert_eq!(completions[0].term, "finalized");
        assert!(!completions[0].is_draft);
        assert_eq!(completions[1].term, "draft");
        assert!(completions[1].is_draft);
    }

    #[test]
    fn test_sorting_by_term() {
        let mut completions = [
            Completion::finalized("z".to_string(), 0, vec![0]),
            Completion::finalized("a".to_string(), 0, vec![0]),
            Completion::finalized("m".to_string(), 0, vec![0]),
        ];

        completions.sort();

        assert_eq!(completions[0].term, "a");
        assert_eq!(completions[1].term, "m");
        assert_eq!(completions[2].term, "z");
    }

    #[test]
    fn test_sorting_combined() {
        let mut completions = [
            Completion::draft("z".to_string(), 2, 0),
            Completion::finalized("a".to_string(), 1, vec![0]),
            Completion::draft("b".to_string(), 1, 0),
            Completion::finalized("c".to_string(), 0, vec![0]),
            Completion::draft("d".to_string(), 0, 0),
            Completion::finalized("e".to_string(), 2, vec![0]),
        ];

        completions.sort();

        // Order: distance 0 finalized (c), distance 0 draft (d),
        //        distance 1 finalized (a), distance 1 draft (b),
        //        distance 2 finalized (e), distance 2 draft (z)
        assert_eq!(completions[0].term, "c");
        assert!(!completions[0].is_draft);
        assert_eq!(completions[1].term, "d");
        assert!(completions[1].is_draft);
        assert_eq!(completions[2].term, "a");
        assert!(!completions[2].is_draft);
        assert_eq!(completions[3].term, "b");
        assert!(completions[3].is_draft);
        assert_eq!(completions[4].term, "e");
        assert!(!completions[4].is_draft);
        assert_eq!(completions[5].term, "z");
        assert!(completions[5].is_draft);
    }

    #[test]
    fn test_equality() {
        let c1 = Completion::finalized("test".to_string(), 1, vec![0]);
        let c2 = Completion::finalized("test".to_string(), 1, vec![0]);
        let c3 = Completion::draft("test".to_string(), 1, 0);

        assert_eq!(c1, c2);
        assert_ne!(c1, c3); // Different draft status
    }

    #[test]
    fn test_display() {
        let completion = Completion::finalized("hello".to_string(), 2, vec![0, 1]);
        let display = format!("{}", completion);
        assert!(display.contains("hello"));
        assert!(display.contains("distance: 2"));
        assert!(display.contains("finalized"));
        assert!(display.contains("[0, 1]"));

        let draft = Completion::draft("world".to_string(), 0, 5);
        let display = format!("{}", draft);
        assert!(display.contains("world"));
        assert!(display.contains("draft"));
        assert!(display.contains("[5]"));
    }
}
