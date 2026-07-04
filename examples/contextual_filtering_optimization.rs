//! Demonstrates efficient contextual filtering via sub-trie construction
//!
//! Instead of filtering results after traversal, we create a sub-trie
//! containing only contextually relevant terms, dramatically reducing
//! search space.

use liblevenshtein::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdentifierType {
    Variable,
    Function,
    Class,
}

#[derive(Debug, Clone)]
struct Identifier {
    name: String,
    id_type: IdentifierType,
    is_public: bool,
}

impl Identifier {
    fn new(name: &str, id_type: IdentifierType, is_public: bool) -> Self {
        Self {
            name: name.to_string(),
            id_type,
            is_public,
        }
    }
}

fn main() {
    // Create a large codebase simulation
    let mut identifiers = vec![];

    // Add many identifiers of different types
    for i in 0..1000 {
        identifiers.push(Identifier::new(
            &format!("privateVar{}", i),
            IdentifierType::Variable,
            false,
        ));
        identifiers.push(Identifier::new(
            &format!("publicFunc{}", i),
            IdentifierType::Function,
            true,
        ));
        identifiers.push(Identifier::new(
            &format!("PrivateClass{}", i),
            IdentifierType::Class,
            false,
        ));
    }

    // Add some specific identifiers we'll search for
    identifiers.push(Identifier::new("getValue", IdentifierType::Function, true));
    identifiers.push(Identifier::new(
        "getVariable",
        IdentifierType::Function,
        true,
    ));
    identifiers.push(Identifier::new("getResult", IdentifierType::Function, true));
    identifiers.push(Identifier::new("setValue", IdentifierType::Function, true));
    identifiers.push(Identifier::new("calculate", IdentifierType::Function, true));

    println!("=== Contextual Filtering Optimization Demo ===\n");
    println!("Total identifiers: {}\n", identifiers.len());

    // Build full dictionary
    let all_names: Vec<&str> = identifiers.iter().map(|id| id.name.as_str()).collect();
    let full_dict = DoubleArrayTrie::from_terms(all_names.clone());

    println!("--- Strategy 1: Post-Filtering (Current Approach) ---\n");

    // Benchmark post-filtering
    let start = Instant::now();
    let mut count = 0;
    for candidate in Transducer::new(full_dict.clone(), Algorithm::Standard)
        .query_ordered("getVal", 1)
        .prefix()
        .filter(|c| {
            // Filter to only public functions
            identifiers.iter().any(|id| {
                id.name == c.term && id.id_type == IdentifierType::Function && id.is_public
            })
        })
        .take(5)
    {
        count += 1;
        println!("  {}: {}", candidate.term, candidate.distance);
    }
    let post_filter_time = start.elapsed();

    println!("\nPost-filtering stats:");
    println!("  Results: {}", count);
    println!("  Time: {:?}", post_filter_time);
    println!("  Dictionary size: {} terms", all_names.len());

    println!("\n--- Strategy 2: Pre-Filtering (Sub-Trie Construction) ---\n");

    // Pre-filter: build dictionary with only public functions
    let public_functions: Vec<&str> = identifiers
        .iter()
        .filter(|id| id.id_type == IdentifierType::Function && id.is_public)
        .map(|id| id.name.as_str())
        .collect();

    println!(
        "Pre-filtered dictionary size: {} terms ({}% reduction)",
        public_functions.len(),
        100 - (public_functions.len() * 100 / all_names.len())
    );

    let filtered_dict = DoubleArrayTrie::from_terms(public_functions.clone());

    // Benchmark pre-filtering
    let start = Instant::now();
    let mut count = 0;
    for candidate in Transducer::new(filtered_dict, Algorithm::Standard)
        .query_ordered("getVal", 1)
        .prefix()
        .take(5)
    {
        count += 1;
        println!("  {}: {}", candidate.term, candidate.distance);
    }
    let pre_filter_time = start.elapsed();

    println!("\nPre-filtering stats:");
    println!("  Results: {}", count);
    println!("  Time: {:?}", pre_filter_time);
    println!("  Dictionary size: {} terms", public_functions.len());

    if post_filter_time > pre_filter_time {
        let speedup = post_filter_time.as_nanos() as f64 / pre_filter_time.as_nanos() as f64;
        println!("\n🚀 Speedup: {:.2}x faster with pre-filtering!", speedup);
    }

    println!("\n--- Strategy 3: Context Switcher (Dynamic Sub-Tries) ---\n");
    println!("For interactive applications, maintain multiple pre-built sub-tries:\n");

    // Build context-specific dictionaries
    let contexts = vec![
        (
            "Public Functions",
            identifiers
                .iter()
                .filter(|id| id.id_type == IdentifierType::Function && id.is_public)
                .map(|id| id.name.as_str())
                .collect::<Vec<_>>(),
        ),
        (
            "Private Functions",
            identifiers
                .iter()
                .filter(|id| id.id_type == IdentifierType::Function && !id.is_public)
                .map(|id| id.name.as_str())
                .collect::<Vec<_>>(),
        ),
        (
            "Classes",
            identifiers
                .iter()
                .filter(|id| id.id_type == IdentifierType::Class)
                .map(|id| id.name.as_str())
                .collect::<Vec<_>>(),
        ),
        (
            "Variables",
            identifiers
                .iter()
                .filter(|id| id.id_type == IdentifierType::Variable)
                .map(|id| id.name.as_str())
                .collect::<Vec<_>>(),
        ),
    ];

    for (context_name, terms) in contexts {
        println!("  {} context: {} terms", context_name, terms.len());
    }

    println!("\n--- Strategy 4: Incremental Filtering ---\n");
    println!("For very large codebases, use incremental refinement:\n");

    // Start with a broad context, then refine
    let mut current_set: HashSet<&str> = all_names.iter().copied().collect();

    println!("1. Start: {} terms", current_set.len());

    // Refine: only functions
    current_set.retain(|name| {
        identifiers
            .iter()
            .any(|id| &id.name == name && id.id_type == IdentifierType::Function)
    });
    println!("2. Filter to functions: {} terms", current_set.len());

    // Refine: only public
    current_set.retain(|name| {
        identifiers
            .iter()
            .any(|id| &id.name == name && id.is_public)
    });
    println!("3. Filter to public: {} terms", current_set.len());

    // Build dictionary from refined set
    let refined_terms: Vec<&str> = current_set.iter().copied().collect();
    let refined_dict = DoubleArrayTrie::from_terms(refined_terms);

    println!("4. Query refined dictionary:");
    for candidate in Transducer::new(refined_dict, Algorithm::Standard)
        .query_ordered("get", 0)
        .prefix()
        .take(3)
    {
        println!("     {}: {}", candidate.term, candidate.distance);
    }

    println!("\n=== Performance Recommendations ===\n");
    println!("1. ✅ Pre-filter when context changes infrequently");
    println!("   - Build sub-trie per context");
    println!("   - Switch between pre-built tries");
    println!();
    println!("2. ✅ Cache sub-tries for common contexts");
    println!("   - \"current function scope\"");
    println!("   - \"current class members\"");
    println!("   - \"imported symbols\"");
    println!();
    println!("3. ✅ Use incremental filtering for complex criteria");
    println!("   - Chain multiple filters");
    println!("   - Build final dictionary once");
    println!();
    println!("4. ⚠️  Post-filter only when:");
    println!("   - Context changes every query");
    println!("   - Filter logic is very complex");
    println!("   - Dictionary is already small");
}
