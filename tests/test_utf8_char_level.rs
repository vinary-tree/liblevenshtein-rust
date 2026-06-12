use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use liblevenshtein::prelude::*;

#[test]
fn test_unicode_empty_query_char_level() {
    println!("\n=== UTF-8 Char-Level: Empty Query → Unicode Character ===\n");

    // This was failing with byte-level: "" → "¡" required distance 2 (two bytes)
    // With char-level it should correctly be distance 1 (one character)
    let dict = DoubleArrayTrieChar::from_terms(vec!["¡"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"¡\"]");
    println!("Query: \"\" (empty)");
    println!("Max distance: 1");
    println!("Character '¡': {:?}", '¡');
    println!("Bytes: {:?} (length: {})", "¡".as_bytes(), "¡".len());
    println!(
        "Chars: {} (length: {})\n",
        "¡".chars().count(),
        "¡".chars().count()
    );

    let results: Vec<_> = transducer.query("", 1).collect();
    println!("Results: {:?}\n", results);

    assert!(
        results.contains(&"¡".to_string()),
        "Empty query should match \"¡\" at distance 1 (one character insertion)"
    );

    println!("✅ SUCCESS: Char-level correctly treats \"¡\" as distance 1 from empty string");
}

#[test]
fn test_unicode_various_distances() {
    println!("\n=== UTF-8 Char-Level: Various Distances ===\n");

    let dict = DoubleArrayTrieChar::from_terms(vec!["é", "ée", "éée"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"é\", \"ée\", \"éée\"]");
    println!("Each 'é' is 1 character (2 bytes in UTF-8)\n");

    for max_dist in 0..=3 {
        let results: Vec<_> = transducer.query("", max_dist).collect();
        println!("Empty query, max_distance={}: {:?}", max_dist, results);
    }

    // At distance 1: can insert 1 character → should find "é"
    let results_1: Vec<_> = transducer.query("", 1).collect();
    assert!(results_1.contains(&"é".to_string()));
    assert!(!results_1.contains(&"ée".to_string())); // requires 2 insertions

    // At distance 2: can insert 2 characters → should find "é" and "ée"
    let results_2: Vec<_> = transducer.query("", 2).collect();
    assert!(results_2.contains(&"é".to_string()));
    assert!(results_2.contains(&"ée".to_string()));
    assert!(!results_2.contains(&"éée".to_string())); // requires 3 insertions

    // At distance 3: should find all
    let results_3: Vec<_> = transducer.query("", 3).collect();
    assert!(results_3.contains(&"é".to_string()));
    assert!(results_3.contains(&"ée".to_string()));
    assert!(results_3.contains(&"éée".to_string()));

    println!("\n✅ SUCCESS: Character-level distances work correctly");
}

#[test]
fn test_emoji_distance() {
    println!("\n=== UTF-8 Char-Level: Emoji Distances ===\n");

    let dict = DoubleArrayTrieChar::from_terms(vec!["🎉", "hello🎉", "🎉world"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"🎉\", \"hello🎉\", \"🎉world\"]");
    println!("Emoji '🎉' is 1 character (4 bytes in UTF-8)\n");

    // Empty query at distance 1 should find solo emoji
    let results: Vec<_> = transducer.query("", 1).collect();
    println!("Empty query at distance 1: {:?}", results);
    assert!(results.contains(&"🎉".to_string()));

    // Query "hello" at distance 1 should find "hello🎉" (insert emoji at end)
    let results: Vec<_> = transducer.query("hello", 1).collect();
    println!("Query \"hello\" at distance 1: {:?}", results);
    assert!(results.contains(&"hello🎉".to_string()));

    println!("\n✅ SUCCESS: Emoji treated as single character");
}

#[test]
fn test_cjk_distance() {
    println!("\n=== UTF-8 Char-Level: CJK Character Distances ===\n");

    let dict = DoubleArrayTrieChar::from_terms(vec!["中", "中文", "中文字"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"中\", \"中文\", \"中文字\"]");
    println!("Each CJK character is 1 character (3 bytes in UTF-8)\n");

    // Empty query at distance 1: should find "中" (1 insertion)
    let results_1: Vec<_> = transducer.query("", 1).collect();
    println!("Distance 1: {:?}", results_1);
    assert!(results_1.contains(&"中".to_string()));

    // Empty query at distance 2: should find "中" and "中文" (2 insertions)
    let results_2: Vec<_> = transducer.query("", 2).collect();
    println!("Distance 2: {:?}", results_2);
    assert!(results_2.contains(&"中".to_string()));
    assert!(results_2.contains(&"中文".to_string()));

    // Query "中" at distance 1: should find "中文" (insert one char)
    let results: Vec<_> = transducer.query("中", 1).collect();
    println!("Query \"中\" at distance 1: {:?}", results);
    assert!(results.contains(&"中".to_string())); // exact match
    assert!(results.contains(&"中文".to_string())); // insert '文'

    println!("\n✅ SUCCESS: CJK characters treated correctly");
}

#[test]
fn test_mixed_unicode_query() {
    println!("\n=== UTF-8 Char-Level: Mixed Unicode Query ===\n");

    let dict =
        DoubleArrayTrieChar::from_terms(vec!["café", "cafe", "naïve", "naive", "résumé", "resume"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary contains accented and non-accented variants\n");

    // Query "cafe" should find "café" at distance 1 (substitute e→é)
    let results: Vec<_> = transducer.query("cafe", 1).collect();
    println!("Query \"cafe\" at distance 1: {:?}", results);
    assert!(results.contains(&"cafe".to_string())); // exact
    assert!(results.contains(&"café".to_string())); // one substitution

    // Query "café" should find "cafe" at distance 1
    let results: Vec<_> = transducer.query("café", 1).collect();
    println!("Query \"café\" at distance 1: {:?}", results);
    assert!(results.contains(&"café".to_string())); // exact
    assert!(results.contains(&"cafe".to_string())); // one substitution

    println!("\n✅ SUCCESS: Accented characters are single character edits");
}

#[test]
fn test_transposition_with_unicode() {
    println!("\n=== UTF-8 Char-Level: Transposition with Unicode ===\n");

    let dict = DoubleArrayTrieChar::from_terms(vec!["café", "éfac"]);
    let transducer = Transducer::new(dict, Algorithm::Transposition);

    println!("Dictionary: [\"café\", \"éfac\"]");
    println!("Using Transposition algorithm\n");

    // Swap 'é' and 'f' in "éfac" → "féac"
    let results: Vec<_> = transducer.query("féac", 1).collect();
    println!("Query \"féac\" at distance 1: {:?}", results);

    // Should find "éfac" via one transposition
    assert!(results.contains(&"éfac".to_string()));

    println!("\n✅ SUCCESS: Transposition works with Unicode characters");
}

#[test]
fn test_query_with_distance_unicode() {
    println!("\n=== UTF-8 Char-Level: Query with Distance (Unicode) ===\n");

    let dict = DoubleArrayTrieChar::from_terms(vec!["café", "naïve"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Query and get distances
    let results: Vec<_> = transducer.query_with_distance("cafe", 2).collect();

    println!("Query \"cafe\" at max_distance 2:");
    for candidate in &results {
        println!("  {}: distance {}", candidate.term, candidate.distance);
    }

    // Find "café" - should be distance 1 (substitute e→é)
    let cafe_result = results.iter().find(|c| c.term == "café");
    assert!(cafe_result.is_some());
    assert_eq!(cafe_result.unwrap().distance, 1);

    println!("\n✅ SUCCESS: Distances correctly computed for Unicode");
}

#[test]
fn test_combining_characters() {
    println!("\n=== UTF-8 Char-Level: Combining Characters ===\n");

    // NOTE: "é" can be represented as:
    // - NFC (composed): '\u{00E9}' - single code point
    // - NFD (decomposed): 'e' + '\u{0301}' (combining acute) - two code points

    // For this test, we use NFC (single character)
    let dict = DoubleArrayTrieChar::from_terms(vec!["café"]); // NFC form
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query("café", 0).collect(); // exact match
    println!("Query \"café\" (NFC) for exact match: {:?}", results);
    assert!(results.contains(&"café".to_string()));

    println!("\n✅ SUCCESS: NFC Unicode handled correctly");
    println!("Note: NFD (decomposed) would be treated as separate characters");
}

#[test]
fn test_grapheme_clusters_caveat() {
    println!("\n=== UTF-8 Char-Level: Grapheme Cluster Caveat ===\n");

    // CharUnit for char treats each Unicode scalar as one unit
    // But some "characters" are multiple scalars (grapheme clusters)
    // Example: 👨‍👩‍👧‍👦 (family emoji) is multiple code points joined with ZWJ

    let dict = DoubleArrayTrieChar::from_terms(vec!["hello"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // This works correctly for most Unicode
    let results: Vec<_> = transducer.query("héllo", 1).collect();
    assert!(results.contains(&"hello".to_string()));

    println!("✅ Current implementation: Unicode scalar values (char)");
    println!("Limitation: Grapheme clusters (like family emoji) are multiple chars");
    println!("This is acceptable for most use cases\n");
}
