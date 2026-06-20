//! Demonstration of OrderedQueryIterator for distance-first, lexicographic results
//!
//! This example shows how to use query_ordered() for:
//! 1. Top-k nearest neighbors (take first N results)
//! 2. Distance-bounded queries (take_while distance <= threshold)
//! 3. Ranked autocomplete/spell-check suggestions

use liblevenshtein::prelude::*;

fn main() {
    println!("=== OrderedQueryIterator Demonstration ===\n");

    // Create a sample dictionary of similar words
    let words = vec![
        "test", "tests", "tested", "testing", "tester", "best", "rest", "nest", "west", "fest",
        "zest", "taste", "text", "tent", "tame", "team",
    ];

    let dict = DoubleArrayTrie::from_terms(words.iter().copied());
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: {} words", words.len());
    println!("Query: \"tset\" (intentional misspelling of \"test\")");
    println!("Max distance: 3\n");

    // Example 1: Show all results in order
    println!("=== Example 1: All Results (Ordered) ===\n");
    println!("{:<3} {:<10} Term", "Rank", "Distance");
    println!("{:-<30}", "");

    for (rank, candidate) in transducer.query_ordered("tset", 3).enumerate() {
        println!(
            "{:<3}  {:<10} {}",
            rank + 1,
            candidate.distance,
            candidate.term
        );
    }

    println!("\n📊 Notice how results are grouped by distance (0, then 1, then 2, then 3)");
    println!("   and within each distance group, they're alphabetically sorted.\n");

    // Example 2: Top-K nearest neighbors
    println!("=== Example 2: Top-5 Nearest Neighbors ===\n");
    println!("Using .take(5) for efficient top-k query:\n");

    for candidate in transducer.query_ordered("tset", 3).take(5) {
        println!("  {} (distance: {})", candidate.term, candidate.distance);
    }

    println!("\n💡 Only 5 results computed due to lazy evaluation!\n");

    // Example 3: Distance-bounded query
    println!("=== Example 3: Distance-Bounded Query ===\n");
    println!("Using .take_while(|c| c.distance <= 1) to get only close matches:\n");

    for candidate in transducer
        .query_ordered("tset", 3)
        .take_while(|c| c.distance <= 1)
    {
        println!("  {} (distance: {})", candidate.term, candidate.distance);
    }

    println!("\n💡 Stops early when distance exceeds threshold!\n");

    // Example 4: Autocomplete scenario
    println!("=== Example 4: Autocomplete/Spell-Check Use Case ===\n");
    println!("Scenario: User types 'tst' - suggest top 3 corrections:\n");

    let suggestions: Vec<_> = transducer.query_ordered("tst", 2).take(3).collect();

    println!("Suggestions:");
    for (i, candidate) in suggestions.iter().enumerate() {
        println!(
            "  {}. {} (distance: {})",
            i + 1,
            candidate.term,
            candidate.distance
        );
    }

    println!("\n✨ Perfect for autocomplete - best matches first!\n");

    // Example 5: Comparison with unordered query
    println!("=== Example 5: Comparison with Unordered Query ===\n");

    println!("Ordered query (distance-first, then alphabetical):");
    let ordered: Vec<_> = transducer
        .query_ordered("test", 1)
        .map(|c| format!("{}(d={})", c.term, c.distance))
        .collect();
    println!("  {:?}", ordered);

    println!("\nUnordered query (depth-first traversal order):");
    let unordered: Vec<_> = transducer.query("test", 1).collect();
    println!("  {:?}", unordered);

    println!("\n📌 Same results, different order - choose based on your use case!\n");

    // Example 6: Distance distribution
    println!("=== Example 6: Distance Distribution ===\n");
    println!("Query: \"testing\" with max_distance=2\n");

    let mut by_distance: [Vec<String>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for candidate in transducer.query_ordered("testing", 2) {
        if candidate.distance < 3 {
            by_distance[candidate.distance].push(candidate.term);
        }
    }

    for (dist, terms) in by_distance.iter().enumerate() {
        if !terms.is_empty() {
            println!("  Distance {}: {} matches - {:?}", dist, terms.len(), terms);
        }
    }

    println!("\n=== Performance Notes ===\n");
    println!("✅ Lazy evaluation - only computes what you consume");
    println!("✅ Distance-stratified BFS - explores efficiently");
    println!("✅ Lexicographic order comes free from DAWG structure");
    println!("✅ Allocation reuse via StatePool");
    println!("\n🎯 Use Cases:");
    println!("   • Autocomplete/spell-check (top-k suggestions)");
    println!("   • Fuzzy search with quality thresholds");
    println!("   • Ranked similarity search");
    println!("   • Edit distance analysis");
}
