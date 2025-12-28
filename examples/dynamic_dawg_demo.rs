//! Demonstration of DynamicDawg with online modifications.
//!
//! This example shows:
//! - Creating a dynamic DAWG
//! - Online insertions and deletions
//! - Compaction to restore minimality
//! - Trade-offs vs static DAWG
//!
//! Run with: cargo run --example dynamic_dawg_demo

use liblevenshtein::prelude::*;

fn main() {
    println!("Dynamic DAWG Demonstration\n");
    println!("==========================\n");

    // Create an empty dynamic DAWG
    println!("1. Creating dynamic DAWG and adding terms...\n");
    let dawg: DynamicDawg<()> = DynamicDawg::new();

    dawg.insert("apple");
    dawg.insert("application");
    dawg.insert("apply");
    dawg.insert("apricot");

    println!("   Terms: {}", dawg.term_count());
    println!("   Nodes: {}", dawg.node_count());

    // Use with fuzzy search
    println!("\n2. Fuzzy search with dynamic DAWG...\n");
    let transducer = Transducer::new(dawg.clone(), Algorithm::Standard);

    let query = "aple";
    println!("   Query '{}' with distance 2:", query);
    let results: Vec<_> = transducer.query(query, 2).collect();
    for term in &results {
        println!("     - {}", term);
    }

    // Online insertion
    println!("\n3. Adding new term dynamically...\n");
    dawg.insert("applesauce");
    println!("   Added 'applesauce'");
    println!("   Terms: {} (was 4)", dawg.term_count());
    println!("   Nodes: {}", dawg.node_count());

    // Search again - new term is immediately available
    let query2 = "applesauc";
    println!("\n   Query '{}' with distance 1:", query2);
    let results2: Vec<_> = transducer.query(query2, 1).collect();
    for term in &results2 {
        println!("     - {}", term);
    }

    // Online deletion
    println!("\n4. Removing term dynamically...\n");
    dawg.remove("apricot");
    println!("   Removed 'apricot'");
    println!("   Terms: {} (was 5)", dawg.term_count());
    println!("   Nodes: {} (may have orphaned nodes)", dawg.node_count());
    println!("   Needs compaction: {}", dawg.needs_compaction());

    // Compaction
    println!("\n5. Compacting to restore minimality...\n");
    let removed_nodes = dawg.compact();
    println!("   Removed {} orphaned nodes", removed_nodes);
    println!("   Terms: {} (unchanged)", dawg.term_count());
    println!("   Nodes: {} (minimized)", dawg.node_count());
    println!("   Needs compaction: {}", dawg.needs_compaction());

    // Comparison with DoubleArrayTrie
    println!("\n6. Comparison: DynamicDawg vs DoubleArrayTrie\n");

    let terms = vec!["apple", "application", "apply", "applesauce"];

    // DoubleArrayTrie (built once, fast reads)
    let dat = DoubleArrayTrie::from_terms(terms.clone());
    println!("   DoubleArrayTrie:");
    println!("     Terms: {}", dat.len().unwrap_or(0));
    println!("     (Compact O(1) state transitions)");

    // Dynamic DAWG (after compaction)
    println!("\n   DynamicDawg (after compaction):");
    println!("     Terms: {}", dawg.term_count());
    println!("     Nodes: {}", dawg.node_count());

    // Performance characteristics
    println!("\n7. Performance Characteristics\n");
    println!("   DynamicDawg:");
    println!("     ✓ Online insertions: O(m) per term");
    println!("     ✓ Online deletions: O(m) per term");
    println!("     ✓ Compaction: O(n) total size");
    println!("     ✓ Thread-safe: RwLock for concurrent access");
    println!("     ✗ May become non-minimal between compactions");

    println!("\n   DoubleArrayTrie:");
    println!("     ✓ O(1) state transitions");
    println!("     ✓ Excellent cache locality");
    println!("     ✓ Compact memory representation");
    println!("     ✗ Expensive updates (requires rebuild)");

    // Use cases
    println!("\n8. Use Cases\n");
    println!("   Use DynamicDawg when:");
    println!("     • Dictionary changes frequently");
    println!("     • Real-time updates required");
    println!("     • Periodic compaction acceptable");
    println!("     • Examples: live spell checker, auto-complete");

    println!("\n   Use DoubleArrayTrie when:");
    println!("     • Dictionary is static or rarely changes");
    println!("     • Maximum query performance needed");
    println!("     • Memory efficiency important");
    println!("     • Examples: embedded systems, read-only dictionaries");

    println!("\n✓ Dynamic DAWG demonstration completed!");

    println!("\nKey Takeaways:");
    println!("• DynamicDawg supports insert(), remove(), and compact()");
    println!("• Maintains near-minimality with periodic compaction");
    println!("• Thread-safe with RwLock for concurrent access");
    println!("• Perfect for dictionaries that change over time");
    println!("• Compaction restores perfect minimality when needed");
}
