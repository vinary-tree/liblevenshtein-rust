//! Example demonstrating dictionary serialization.
//!
//! This example shows how to:
//! - Create and populate a dictionary
//! - Serialize it to disk (bincode and JSON)
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

    // Serialize to JSON (human-readable format)
    println!("3. Serializing to JSON (human-readable format)...");
    let json_file = File::create("dict.json")?;
    JsonSerializer::serialize(&dict, json_file)?;
    let json_size = std::fs::metadata("dict.json")?.len();
    println!("   Saved to dict.json ({} bytes)\n", json_size);

    // Show JSON contents (first 200 chars)
    let json_contents = std::fs::read_to_string("dict.json")?;
    let preview = if json_contents.len() > 200 {
        &json_contents[..200]
    } else {
        &json_contents
    };
    println!(
        "   JSON preview:\n   {}\n   ...\n",
        preview.replace('\n', "\n   ")
    );

    // Load from bincode
    println!("4. Loading from bincode...");
    let bincode_file = File::open("dict.bin")?;
    let loaded_bincode: DoubleArrayTrie = BincodeSerializer::deserialize(bincode_file)?;
    println!("   Loaded {} terms\n", loaded_bincode.len().unwrap());

    // Load from JSON
    println!("5. Loading from JSON...");
    let json_file = File::open("dict.json")?;
    let loaded_json: DoubleArrayTrie = JsonSerializer::deserialize(json_file)?;
    println!("   Loaded {} terms\n", loaded_json.len().unwrap());

    // Verify loaded dictionaries work correctly
    println!("6. Verifying loaded dictionaries...");
    for term in &terms {
        assert!(
            loaded_bincode.contains(term),
            "Bincode dict missing: {}",
            term
        );
        assert!(loaded_json.contains(term), "JSON dict missing: {}", term);
    }
    println!("   ✓ All terms present in both formats\n");

    // Test fuzzy matching with loaded dictionary
    println!("7. Testing fuzzy search with loaded dictionary...");
    let transducer = Transducer::new(loaded_bincode, Algorithm::Standard);
    let results: Vec<_> = transducer.query("aple", 2).collect();
    println!("   Query 'aple' with distance 2:");
    for term in &results {
        println!("     - {}", term);
    }
    println!();

    // Cleanup
    println!("8. Cleaning up...");
    std::fs::remove_file("dict.bin")?;
    std::fs::remove_file("dict.json")?;
    println!("   Removed scratch files\n");

    println!("✓ Serialization example completed successfully!");
    println!("\nKey takeaways:");
    println!(
        "- Bincode is more compact ({} bytes vs {} bytes JSON)",
        bincode_size, json_size
    );
    println!("- JSON is human-readable and easier to inspect");
    println!("- Both formats preserve dictionary functionality perfectly");
    println!("- Use bincode for production, JSON for development/inspection");

    Ok(())
}
