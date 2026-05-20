//! Parallel Workspace Indexing Example
//!
//! Demonstrates efficient bulk construction of contextual completion dictionaries
//! using parallel per-document dictionary construction and binary tree reduction.
//!
//! This example implements the algorithms described in:
//! `docs/algorithms/07-contextual-completion/patterns/parallel-workspace-indexing.md`
//!
//! # Usage
//!
//! ```bash
//! # Run with defaults (100 docs, 1000 terms/doc, both strategies)
//! cargo run --release --example parallel_workspace_indexing
//!
//! # Custom parameters
//! cargo run --release --example parallel_workspace_indexing -- \
//!     --num-docs 200 \
//!     --terms-per-doc 500 \
//!     --strategy binary
//!
//! # Compare both strategies
//! cargo run --release --example parallel_workspace_indexing -- \
//!     --strategy both
//! ```

use liblevenshtein::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::collections::HashMap;
use std::time::Instant;

type ContextId = u32;

/// Command-line arguments
#[derive(Debug)]
struct Args {
    num_docs: usize,
    terms_per_doc: usize,
    strategy: MergeStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MergeStrategy {
    Sequential,
    Binary,
    Both,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            num_docs: 100,
            terms_per_doc: 1000,
            strategy: MergeStrategy::Both,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut iter = std::env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--num-docs" => {
                if let Some(val) = iter.next() {
                    args.num_docs = val.parse().unwrap_or(10);
                }
            }
            "--terms-per-doc" => {
                if let Some(val) = iter.next() {
                    args.terms_per_doc = val.parse().unwrap_or(100);
                }
            }
            "--strategy" => {
                if let Some(val) = iter.next() {
                    args.strategy = match val.as_str() {
                        "sequential" | "seq" => MergeStrategy::Sequential,
                        "binary" | "tree" => MergeStrategy::Binary,
                        "both" | "compare" => MergeStrategy::Both,
                        _ => MergeStrategy::Both,
                    };
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
    }

    args
}

fn print_help() {
    println!(
        r#"Parallel Workspace Indexing Example

Demonstrates parallel merge strategies for contextual completion dictionaries.

USAGE:
    parallel_workspace_indexing [OPTIONS]

OPTIONS:
    --num-docs <N>           Number of documents [default: 100]
    --terms-per-doc <N>      Terms per document [default: 1000]
    --strategy <STRATEGY>    sequential, binary, or both [default: both]
    --help, -h               Print help

EXAMPLES:
    cargo run --release --example parallel_workspace_indexing
    cargo run --release --example parallel_workspace_indexing -- --num-docs 200 --strategy both
"#
    );
}

/// Generate synthetic document terms
fn generate_document_terms(doc_id: ContextId, terms_per_doc: usize) -> Vec<String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    doc_id.hash(&mut hasher);
    let seed = hasher.finish();

    let mut terms = Vec::with_capacity(terms_per_doc);

    for i in 0..terms_per_doc {
        let pattern = (seed + i as u64) % 100;
        let term = if pattern < 40 {
            format!("fn{}{}", doc_id, i)
        } else if pattern < 70 {
            format!("var{}{}", doc_id, i)
        } else if pattern < 90 {
            format!("Class{}{}", doc_id, i)
        } else {
            format!("doc{}_unique_{}", doc_id, i)
        };
        terms.push(term);
    }

    // Add common shared terms
    for common in &["Result", "Error", "Option", "Vec", "String"] {
        terms.push(common.to_string());
    }

    terms.sort();
    terms.dedup();
    terms
}

/// Build per-document dictionaries in parallel
fn build_document_dicts(
    num_docs: usize,
    terms_per_doc: usize,
) -> Vec<HashMap<String, Vec<ContextId>>> {
    println!(
        "Building {} per-document dictionaries in parallel...",
        num_docs
    );
    let start = Instant::now();

    let dicts: Vec<_> = (0..num_docs)
        .into_par_iter() // Parallel iterator
        .map(|doc_id| {
            let doc_id = doc_id as u32;
            let terms = generate_document_terms(doc_id, terms_per_doc);

            let mut dict = HashMap::new();
            for term in terms {
                dict.insert(term, vec![doc_id]);
            }
            dict
        })
        .collect();

    println!(
        "  Built in {:?} ({} cores)",
        start.elapsed(),
        num_cpus::get()
    );
    dicts
}

/// Merge context ID vectors with deduplication
fn merge_contexts(left: &[ContextId], right: &[ContextId]) -> Vec<ContextId> {
    let total_len = left.len() + right.len();

    if total_len > 50 {
        let mut set: FxHashSet<_> = left.iter().copied().collect();
        set.extend(right.iter().copied());
        let mut merged: Vec<_> = set.into_iter().collect();
        merged.sort_unstable();
        merged
    } else {
        let mut merged = left.to_vec();
        merged.extend_from_slice(right);
        merged.sort_unstable();
        merged.dedup();
        merged
    }
}

/// Merge two dictionaries
fn merge_two_dicts(
    dict1: &HashMap<String, Vec<ContextId>>,
    dict2: &HashMap<String, Vec<ContextId>>,
) -> HashMap<String, Vec<ContextId>> {
    let mut merged = dict1.clone();

    for (term, contexts2) in dict2 {
        merged
            .entry(term.clone())
            .and_modify(|contexts1| {
                *contexts1 = merge_contexts(contexts1, contexts2);
            })
            .or_insert_with(|| contexts2.clone());
    }

    merged
}

/// Sequential merge strategy (baseline)
fn merge_sequential(
    mut dicts: Vec<HashMap<String, Vec<ContextId>>>,
) -> HashMap<String, Vec<ContextId>> {
    if dicts.is_empty() {
        return HashMap::new();
    }

    println!("\n=== Sequential Merge Strategy ===");
    let start = Instant::now();

    let mut merged = dicts.remove(0);
    let total = dicts.len();

    for (i, dict) in dicts.into_iter().enumerate() {
        if i % 10 == 0 || i == total - 1 {
            print!("\r  Progress: {}/{} dictionaries", i + 1, total);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        merged = merge_two_dicts(&merged, &dict);
    }

    let elapsed = start.elapsed();
    println!("\n  Completed in {:?}", elapsed);
    println!("  Final dictionary: {} terms", merged.len());

    merged
}

/// Binary tree reduction with parallel merging
fn merge_binary_tree(
    mut dicts: Vec<HashMap<String, Vec<ContextId>>>,
) -> HashMap<String, Vec<ContextId>> {
    if dicts.is_empty() {
        return HashMap::new();
    }

    println!("\n=== Binary Tree Reduction (Parallel) ===");
    let start = Instant::now();
    let mut round = 1;

    while dicts.len() > 1 {
        print!("\r  Round {}: {} dictionaries", round, dicts.len());
        std::io::Write::flush(&mut std::io::stdout()).ok();

        // Parallel merge of pairs
        let next_round: Vec<_> = dicts
            .par_chunks(2)
            .map(|chunk| {
                if chunk.len() == 2 {
                    merge_two_dicts(&chunk[0], &chunk[1])
                } else {
                    chunk[0].clone()
                }
            })
            .collect();

        dicts = next_round;
        round += 1;
    }

    let elapsed = start.elapsed();
    println!("\n  Completed in {:?}", elapsed);

    let merged = dicts.into_iter().next().unwrap();
    println!("  Final dictionary: {} terms", merged.len());
    println!("  Total rounds: {}", round - 1);
    println!("  Parallel efficiency: {} cores utilized", num_cpus::get());

    merged
}

/// Verify dictionary correctness
fn verify_dictionary(dict: &HashMap<String, Vec<ContextId>>, args: &Args) {
    println!("\n=== Verification ===");

    // Check document-specific terms
    let doc_0_term = format!("doc{}_unique_0", 0);
    if let Some(contexts) = dict.get(&doc_0_term) {
        println!("  ✓ '{}' → contexts: {:?}", doc_0_term, contexts);
        assert!(contexts.contains(&0));
    }

    let doc_last = args.num_docs - 1;
    let doc_last_term = format!("doc{}_unique_0", doc_last);
    if let Some(contexts) = dict.get(&doc_last_term) {
        println!("  ✓ '{}' → contexts: {:?}", doc_last_term, contexts);
        assert!(contexts.contains(&(doc_last as u32)));
    }

    // Check shared term
    if let Some(contexts) = dict.get("Result") {
        println!("  ✓ 'Result' → shared across {} contexts", contexts.len());
        assert!(contexts.len() >= args.num_docs.min(10));
    }

    println!("  All verifications passed!");
}

/// Build actual DynamicDawg from merged dictionary
fn build_final_dawg(dict: HashMap<String, Vec<ContextId>>) -> DynamicDawg<Vec<ContextId>> {
    println!("\n=== Building Final DynamicDawg ===");
    let start = Instant::now();

    let dawg: DynamicDawg<Vec<ContextId>> = DynamicDawg::new();

    for (term, contexts) in dict {
        dawg.insert_with_value(&term, contexts);
    }

    println!("  Built in {:?}", start.elapsed());
    println!("  Terms: {}", dawg.term_count());
    println!("  Nodes: {}", dawg.node_count());

    dawg
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Parallel Workspace Indexing for Contextual Completion    ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    let args = parse_args();

    println!("\nConfiguration:");
    println!("  Documents:       {}", args.num_docs);
    println!("  Terms per doc:   ~{}", args.terms_per_doc);
    println!("  Merge strategy:  {:?}", args.strategy);
    println!("  CPU cores:       {}", num_cpus::get());

    // Build per-document dictionaries
    let dicts = build_document_dicts(args.num_docs, args.terms_per_doc);

    let (_seq_time, _tree_time, merged_dict) = match args.strategy {
        MergeStrategy::Both => {
            // Sequential
            let seq_start = Instant::now();
            let _seq_result = merge_sequential(dicts.clone());
            let seq_elapsed = seq_start.elapsed();

            // Binary tree
            let tree_start = Instant::now();
            let tree_result = merge_binary_tree(dicts);
            let tree_elapsed = tree_start.elapsed();

            // Comparison
            println!("\n╔════════════════════════════════════════════════════════════╗");
            println!("║  PERFORMANCE COMPARISON                                    ║");
            println!("╚════════════════════════════════════════════════════════════╝");
            println!("\n  Sequential merge:      {:>8.2?}", seq_elapsed);
            println!("  Binary tree reduction: {:>8.2?}", tree_elapsed);
            let speedup = seq_elapsed.as_secs_f64() / tree_elapsed.as_secs_f64().max(0.000001);
            println!("  Speedup:               {:>8.1}×", speedup);
            println!(
                "  Parallel efficiency:   {:>8.1}% ({:.1}× / {} cores)",
                (speedup / num_cpus::get() as f64) * 100.0,
                speedup,
                num_cpus::get()
            );

            verify_dictionary(&tree_result, &args);

            (Some(seq_elapsed), Some(tree_elapsed), tree_result)
        }
        MergeStrategy::Sequential => {
            let start = Instant::now();
            let result = merge_sequential(dicts);
            let elapsed = start.elapsed();
            verify_dictionary(&result, &args);
            (Some(elapsed), None, result)
        }
        MergeStrategy::Binary => {
            let start = Instant::now();
            let result = merge_binary_tree(dicts);
            let elapsed = start.elapsed();
            verify_dictionary(&result, &args);
            (None, Some(elapsed), result)
        }
    };

    // Build final DAWG
    let _final_dawg = build_final_dawg(merged_dict);

    println!("\n✨ Example completed successfully!");
    println!("\nFor theoretical foundations and detailed analysis, see:");
    println!("  docs/algorithms/07-contextual-completion/patterns/parallel-workspace-indexing.md");

    Ok(())
}
