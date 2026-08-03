//! Example demonstrating dictionary serialization.
//!
//! This example shows how to:
//! - Create and populate a dictionary
//! - Serialize it to disk using the compact binary format
//! - Load it back from disk
//! - Verify the loaded dictionary works correctly
//!
//! Run with: cargo run --example serialization --features serialization

use liblevenshtein::prelude::*;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Dictionary Serialization Example\n");
    println!("=================================\n");

    // Create a dictionary with some terms
    println!("1. Creating dictionary with test data...");
    let terms = vec![
        "apple",
        "application",
        "apply",
        "apricot",
        "banana",
        "bandana",
        "band",
        "cherry",
        "chocolate",
        "chair",
        "test",
        "testing",
        "tested",
        "tester",
    ];
    let dict = DoubleArrayTrie::from_terms(terms.clone());
    println!("   Created dictionary with {} terms\n", dict.len().unwrap());

    // Serialize to bincode (compact binary format)
    println!("2. Serializing to bincode (binary format)...");
    let bincode_file = File::create("dict.bin")?;
    BincodeSerializer::serialize(&dict, bincode_file)?;
    let bincode_size = std::fs::metadata("dict.bin")?.len();
    println!("   Saved to dict.bin ({} bytes)\n", bincode_size);

    // Load from bincode
    println!("3. Loading from bincode...");
    let bincode_file = File::open("dict.bin")?;
    let loaded_bincode: DoubleArrayTrie = BincodeSerializer::deserialize(bincode_file)?;
    println!("   Loaded {} terms\n", loaded_bincode.len().unwrap());

    // Verify the loaded dictionary works correctly
    println!("4. Verifying loaded dictionary...");
    for term in &terms {
        assert!(
            loaded_bincode.contains(term),
            "Bincode dict missing: {}",
            term
        );
    }
    println!("   ✓ All terms present after the binary round trip\n");

    // Test fuzzy matching with loaded dictionary
    println!("5. Testing fuzzy search with loaded dictionary...");
    let transducer = Transducer::new(loaded_bincode, Algorithm::Standard);
    let results: Vec<_> = transducer.query("aple", 2).collect();
    println!("   Query 'aple' with distance 2:");
    for term in &results {
        println!("     - {}", term);
    }
    println!();

    // Cleanup
    println!("6. Cleaning up...");
    std::fs::remove_file("dict.bin")?;
    println!("   Removed scratch files\n");

    println!("✓ Serialization example completed successfully!");
    println!("\nKey takeaways:");
    println!("- Compact binary persistence used {} bytes", bincode_size);
    println!("- The binary round trip preserves dictionary functionality");
    println!("- Protobuf is the portable binary interchange format");

    Ok(())
}
