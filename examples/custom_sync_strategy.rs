//! Example showing how custom dictionary backends can specify
//! their synchronization strategy.

use liblevenshtein::dictionary::{Dictionary, DictionaryNode, SyncStrategy};
use liblevenshtein::transducer::{Algorithm, Transducer};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Example of a lock-free, persistent dictionary backend.
///
/// This is a simplified example showing how a backend that uses
/// persistent data structures (with structural sharing) can
/// declare itself as not needing external synchronization.
#[derive(Clone)]
struct PersistentDictionary {
    // In a real implementation, this would be an immutable/persistent trie
    // For demo purposes, we'll use a simple Vec with atomic operations
    terms: Arc<Vec<String>>,
    count: Arc<AtomicUsize>,
}

impl PersistentDictionary {
    fn new(terms: Vec<String>) -> Self {
        let count = terms.len();
        Self {
            terms: Arc::new(terms),
            count: Arc::new(AtomicUsize::new(count)),
        }
    }
}

#[derive(Clone)]
struct PersistentNode {
    terms: Arc<Vec<String>>,
    prefix: String,
    index: usize,
}

impl Dictionary for PersistentDictionary {
    type Node = PersistentNode;

    fn root(&self) -> Self::Node {
        PersistentNode {
            terms: Arc::clone(&self.terms),
            prefix: String::new(),
            index: 0,
        }
    }

    fn len(&self) -> Option<usize> {
        Some(self.count.load(Ordering::Relaxed))
    }

    fn sync_strategy(&self) -> SyncStrategy {
        // This backend uses Arc (immutable sharing) and atomics.
        // No external synchronization needed!
        SyncStrategy::Persistent
    }
}

impl DictionaryNode for PersistentNode {
    type Unit = u8;
    type SnapshotCursor = libdictenstein::SnapshotTraversalCursor;
    type SnapshotGraphValueHandle = libdictenstein::SnapshotTraversalCursor;

    fn is_final(&self) -> bool {
        self.terms.iter().any(|t| t == &self.prefix)
    }

    fn transition(&self, label: Self::Unit) -> Option<Self> {
        let c = label as char;
        let mut new_prefix = self.prefix.clone();
        new_prefix.push(c);

        // Check if any term starts with this prefix
        if self.terms.iter().any(|t| t.starts_with(&new_prefix)) {
            Some(PersistentNode {
                terms: Arc::clone(&self.terms),
                prefix: new_prefix,
                index: self.index + 1,
            })
        } else {
            None
        }
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        // Collect all possible next characters
        let mut chars = std::collections::HashSet::new();
        let prefix_len = self.prefix.len();

        for term in self.terms.iter() {
            if term.starts_with(&self.prefix) && term.len() > prefix_len {
                if let Some(c) = term.chars().nth(prefix_len) {
                    chars.insert(c as u8);
                }
            }
        }

        let terms = Arc::clone(&self.terms);
        let prefix = self.prefix.clone();
        let index = self.index;

        Box::new(chars.into_iter().map(move |byte| {
            let c = byte as char;
            let mut new_prefix = prefix.clone();
            new_prefix.push(c);

            (
                byte,
                PersistentNode {
                    terms: Arc::clone(&terms),
                    prefix: new_prefix,
                    index: index + 1,
                },
            )
        }))
    }
}

fn main() {
    println!("=== Custom Synchronization Strategy Example ===\n");

    // Create both types of dictionaries
    let pathmap_dict =
        liblevenshtein::prelude::DoubleArrayTrie::from_terms(vec!["test", "testing", "tested"]);

    let persistent_dict = PersistentDictionary::new(vec![
        "test".to_string(),
        "testing".to_string(),
        "tested".to_string(),
    ]);

    // Check their sync strategies
    println!("PathMap sync strategy: {:?}", pathmap_dict.sync_strategy());
    println!(
        "Persistent sync strategy: {:?}\n",
        persistent_dict.sync_strategy()
    );

    // Both work the same from the user's perspective
    let trans1 = Transducer::new(pathmap_dict, Algorithm::Standard);
    let trans2 = Transducer::new(persistent_dict, Algorithm::Standard);

    println!("Query 'test' with DoubleArrayTrie:");
    for term in trans1.query("test", 0) {
        println!("  - {}", term);
    }

    println!("\nQuery 'test' with PersistentDictionary:");
    for term in trans2.query("test", 0) {
        println!("  - {}", term);
    }

    println!("\n=== Performance Implications ===");
    println!();
    println!("DoubleArrayTrie (ExternalSync):");
    println!("  - Uses RwLock for synchronization");
    println!("  - Multiple concurrent reads: ✓");
    println!("  - Lock overhead: minimal");
    println!();
    println!("PersistentDictionary (Persistent):");
    println!("  - No locks needed for reads");
    println!("  - Zero-cost reads: ✓");
    println!("  - Writes use Arc::make_mut or atomic swap");
    println!();
    println!("Future optimization: When PathMap's thread-safety is confirmed,");
    println!("it could also declare Persistent or InternalSync strategy.");
}
