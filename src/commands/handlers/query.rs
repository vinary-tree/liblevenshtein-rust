//! Shared query command handler

use crate::commands::core::{CommandResult, QueryParams};
use libdictenstein::Dictionary;
use crate::transducer::Transducer;

/// Execute a query against a dictionary
///
/// This is the core query logic shared by both CLI and REPL.
///
/// # Arguments
///
/// * `dict` - The dictionary to query
/// * `params` - Query parameters (term, distance, algorithm, etc.)
///
/// # Returns
///
/// Results as `(term, distance)` pairs sorted by distance then lexicographically
pub fn execute_query<D: Dictionary + Clone>(
    dict: &D,
    params: &QueryParams,
) -> Vec<(String, usize)> {
    let transducer = Transducer::new(dict.clone(), params.algorithm);

    let mut results: Vec<(String, usize)> = if params.prefix {
        // Prefix mode - find terms that start with query
        transducer
            .query_ordered(&params.term, params.max_distance)
            .prefix()
            .map(|c| (c.term, c.distance))
            .collect()
    } else {
        // Normal mode - find similar terms
        transducer
            .query_ordered(&params.term, params.max_distance)
            .map(|c| (c.term, c.distance))
            .collect()
    };

    // Apply limit if specified
    if let Some(limit) = params.limit {
        results.truncate(limit);
    }

    results
}

/// Format query results for display
///
/// # Arguments
///
/// * `results` - Query results as `(term, distance)` pairs
/// * `show_distances` - Whether to include distances in output
///
/// # Returns
///
/// Formatted string ready for display
pub fn format_results(results: &[(String, usize)], show_distances: bool) -> String {
    if results.is_empty() {
        return "No matches found".to_string();
    }

    if show_distances {
        results
            .iter()
            .map(|(term, distance)| format!("{} (distance: {})", term, distance))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        results
            .iter()
            .map(|(term, _)| term.clone())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Execute query and return formatted result
///
/// Convenience function that combines execute_query and format_results.
pub fn query_and_format<D: Dictionary + Clone>(dict: &D, params: &QueryParams) -> CommandResult {
    let results = execute_query(dict, params);
    let output = format_results(&results, params.show_distances);

    CommandResult::success(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use crate::transducer::Algorithm;

    #[test]
    fn test_execute_query_exact() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tested"]);
        let params = QueryParams {
            term: "test".to_string(),
            max_distance: 0,
            algorithm: Algorithm::Standard,
            prefix: false,
            show_distances: false,
            limit: None,
        };

        let results = execute_query(&dict, &params);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "test");
        assert_eq!(results[0].1, 0);
    }

    #[test]
    fn test_execute_query_fuzzy() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest"]);
        let params = QueryParams {
            term: "test".to_string(),
            max_distance: 1,
            algorithm: Algorithm::Standard,
            prefix: false,
            show_distances: false,
            limit: None,
        };

        let results = execute_query(&dict, &params);
        assert!(results.len() >= 3);
        assert!(results.iter().any(|(t, _)| t == "test"));
        assert!(results.iter().any(|(t, _)| t == "best"));
        assert!(results.iter().any(|(t, _)| t == "rest"));
    }

    #[test]
    fn test_execute_query_with_limit() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "nest"]);
        let params = QueryParams {
            term: "test".to_string(),
            max_distance: 1,
            algorithm: Algorithm::Standard,
            prefix: false,
            show_distances: false,
            limit: Some(2),
        };

        let results = execute_query(&dict, &params);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_format_results_no_distances() {
        let results = vec![
            ("apple".to_string(), 0),
            ("apply".to_string(), 1),
            ("ample".to_string(), 2),
        ];

        let output = format_results(&results, false);
        assert_eq!(output, "apple\napply\nample");
    }

    #[test]
    fn test_format_results_with_distances() {
        let results = vec![("apple".to_string(), 0), ("apply".to_string(), 1)];

        let output = format_results(&results, true);
        assert!(output.contains("apple (distance: 0)"));
        assert!(output.contains("apply (distance: 1)"));
    }

    #[test]
    fn test_format_results_empty() {
        let results = vec![];
        let output = format_results(&results, false);
        assert_eq!(output, "No matches found");
    }

    #[test]
    fn test_query_and_format() {
        let dict = DoubleArrayTrie::from_terms(vec!["test"]);
        let params = QueryParams {
            term: "test".to_string(),
            max_distance: 0,
            algorithm: Algorithm::Standard,
            prefix: false,
            show_distances: false,
            limit: None,
        };

        let result = query_and_format(&dict, &params);
        assert_eq!(result.output, "test");
        assert!(!result.modified);
    }
}
