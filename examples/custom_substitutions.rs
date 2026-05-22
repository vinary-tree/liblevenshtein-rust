//! Custom Substitutions Example
//!
//! Demonstrates how to create and combine custom substitution sets for
//! domain-specific fuzzy matching scenarios.
//!
//! Run this example with:
//! ```bash
//! cargo run --example custom_substitutions
//! ```

use liblevenshtein::prelude::*;
use liblevenshtein::transducer::{Algorithm, Restricted, SubstitutionSet};

fn main() {
    println!("=== Custom Substitution Sets ===\n");

    // Example 1: Build custom substitution set from scratch
    println!("--- Example 1: Medical Terminology ---\n");

    let mut medical = SubstitutionSet::new();

    // Common medical spelling variations
    medical.allow('x', 'c');
    medical.allow('c', 'x'); // excision ↔ exsision

    medical.allow('f', 'p');
    medical.allow('p', 'f'); // pharmacy ↔ farmacy (f for ph)

    medical.allow('i', 'y');
    medical.allow('y', 'i'); // sinus ↔ synus

    println!("Medical substitutions:");
    println!("  - x ↔ c (excision/exsision)");
    println!("  - f ↔ p (pharmacy/farmacy, for 'ph')");
    println!("  - i ↔ y (sinus/synus)");
    println!();

    let medical_terms = vec!["excision", "pharmacy", "sinus", "diagnosis"];
    let dict = DoubleArrayTrie::from_terms(medical_terms.clone());
    let policy = Restricted::new(&medical);
    let transducer = Transducer::with_policy(dict, Algorithm::Standard, policy);

    let test_queries = vec![
        ("exsision", "excision"),
        ("farmacy", "pharmacy"),
        ("synus", "sinus"),
    ];

    for (query, expected) in test_queries {
        let results: Vec<String> = transducer.query(query, 1).collect();
        print!("Query: {:10} → ", query);

        if results.contains(&expected.to_string()) {
            println!("✓ Matched '{}'", expected);
        } else {
            println!("✗ Expected '{}', got: {:?}", expected, results);
        }
    }

    // Example 2: Combining multiple preset sets
    println!("\n--- Example 2: Combined Phonetic + Keyboard ---\n");

    // Start with phonetic base
    let mut combined = SubstitutionSet::phonetic_basic();
    println!("Started with: SubstitutionSet::phonetic_basic()");

    // Add selected keyboard adjacencies
    let keyboard_pairs = [
        ('q', 'w'),
        ('w', 'q'),
        ('e', 'r'),
        ('r', 'e'),
        ('a', 's'),
        ('s', 'a'),
        ('d', 'f'),
        ('f', 'd'),
    ];

    for (a, b) in &keyboard_pairs {
        combined.allow(*a, *b);
    }
    println!("Added keyboard adjacencies: q↔w, e↔r, a↔s, d↔f");
    println!("Total substitution pairs: {}\n", combined.len());

    let dict = DoubleArrayTrie::from_terms(vec!["hello", "world", "phone"]);
    let policy = Restricted::new(&combined);
    let transducer = Transducer::with_policy(dict, Algorithm::Standard, policy);

    let combined_queries = vec![
        ("hwllo", "hello"), // Keyboard: e→w
        ("fone", "phone"),  // Phonetic: f↔p
        ("workd", "world"), // Keyboard: l→k (not in set, won't match perfectly)
    ];

    for (query, expected) in combined_queries {
        let results: Vec<String> = transducer.query(query, 2).collect();
        print!("Query: {:10} → ", query);

        if results.contains(&expected.to_string()) {
            println!("✓ Found '{}'", expected);
        } else if !results.is_empty() {
            println!("? Got: {:?} (expected '{}')", results, expected);
        } else {
            println!("✗ No matches (expected '{}')", expected);
        }
    }

    // Example 3: Domain-specific: Chemical nomenclature
    println!("\n--- Example 3: Chemical Nomenclature ---\n");

    let mut chemical = SubstitutionSet::new();

    // Common chemical spelling variations
    chemical.allow('f', 'p');
    chemical.allow('p', 'f'); // phosphate/fosphate

    chemical.allow('i', 'y');
    chemical.allow('y', 'i'); // hydroxyl/hydroxil

    chemical.allow('s', 'z');
    chemical.allow('z', 's'); // sulphur/sulfur (British vs American)

    chemical.allow('f', 's');
    chemical.allow('s', 'f'); // sulphur/sulfur (ph→f)

    println!("Chemical substitutions:");
    println!("  - f ↔ p (phosphate variations)");
    println!("  - i ↔ y (hydroxyl/hydroxil)");
    println!("  - s ↔ z (sulphur/sulfur)");
    println!();

    let chemicals = vec!["phosphate", "hydroxyl", "sulphur", "benzene"];
    let dict = DoubleArrayTrie::from_terms(chemicals);
    let policy = Restricted::new(&chemical);
    let transducer = Transducer::with_policy(dict, Algorithm::Standard, policy);

    let chem_queries = vec![
        ("fosphate", "phosphate"),
        ("hydroxil", "hydroxyl"),
        ("sulfur", "sulphur"),
    ];

    for (query, expected) in chem_queries {
        let results: Vec<String> = transducer.query(query, 2).collect();
        print!("Query: {:10} → ", query);

        if results.contains(&expected.to_string()) {
            println!("✓ Matched '{}'", expected);
        } else if !results.is_empty() {
            println!("? Got: {:?} (expected '{}')", results, expected);
        } else {
            println!("✗ No matches",);
        }
    }

    // Example 4: Using from_pairs() for bulk construction
    println!("\n--- Example 4: Bulk Construction with from_pairs() ---\n");

    let vowel_pairs = vec![
        ('a', 'e'),
        ('e', 'a'),
        ('i', 'y'),
        ('y', 'i'),
        ('o', 'u'),
        ('u', 'o'),
    ];

    let vowel_set = SubstitutionSet::from_pairs(&vowel_pairs);
    println!(
        "Created vowel substitution set with {} pairs:",
        vowel_set.len()
    );
    println!("  - a ↔ e");
    println!("  - i ↔ y");
    println!("  - o ↔ u");
    println!();

    let dict = DoubleArrayTrie::from_terms(vec!["test", "happy", "color"]);
    let policy = Restricted::new(&vowel_set);
    let transducer = Transducer::with_policy(dict, Algorithm::Standard, policy);

    let vowel_queries = vec![
        ("tast", "test"),   // e→a
        ("happi", "happy"), // y→i
        ("cular", "color"), // o→u (in 'co'), o→a (in 'or')
    ];

    for (query, expected) in vowel_queries {
        let results: Vec<String> = transducer.query(query, 2).collect();
        print!("Query: {:10} → ", query);

        if results.contains(&expected.to_string()) {
            println!("✓ Matched '{}'", expected);
        } else if !results.is_empty() {
            println!("? Got: {:?} (expected '{}')", results, expected);
        } else {
            println!("✗ No matches");
        }
    }

    // Summary
    println!("\n=== Summary ===\n");
    println!("✓ Custom substitution sets for domain-specific matching");
    println!("✓ Combine presets with custom rules");
    println!("✓ Use from_pairs() for bulk construction");
    println!("✓ Tailor equivalences to your use case");
    println!("\n📌 Build substitution sets that match your domain's spelling patterns!");
}
