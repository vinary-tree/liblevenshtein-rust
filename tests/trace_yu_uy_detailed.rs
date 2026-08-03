use liblevenshtein::distance::transposition_distance;
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::{Position, State};

#[test]
fn trace_yu_to_uy_detailed() {
    println!("\n=== Detailed Trace: 'yu' -> 'uy' ===\n");

    let query = "yu";
    let dict_words = vec!["uy".to_string()];
    let dict = DoubleArrayTrie::from_terms(dict_words.clone());

    let query_length = query.len(); // 2
    let max_distance = 1;
    let algorithm = Algorithm::Transposition;

    println!("Dictionary: {:?}", dict_words);
    println!("Query: \"{}\" (len={})", query, query_length);
    println!("Max distance: {}\n", max_distance);

    // Verify distance function
    let dist = transposition_distance(query, "uy");
    println!("transposition_distance(\"yu\", \"uy\") = {}", dist);
    assert_eq!(dist, 1, "Should be 1 transposition");
    println!();

    // Manual trace of expected automaton behavior
    println!("=== Expected Automaton Path ===\n");

    println!("Step 0: Initial state at root");
    println!("  State: [(0,0,false)]");
    println!();

    println!("Step 1: Process dictionary edge 'u' from root");
    println!("  Query: 'yu'");
    println!("  Current position: (0,0,false)");
    println!("  Dict char: 'u'");
    println!("  Query[0]: 'y'");
    println!("  CV for 'u' vs 'yu' at offset 0:");
    println!("    cv[0]: 'u' == 'y'? false");
    println!("    cv[1]: 'u' == 'u'? true");
    println!("  CV = [false, true]");
    println!();
    println!("  Match at index 1 -> transposition case!");
    println!("  Generated positions (from transition.rs lines 226-231):");
    println!("    (0,1,false) - insertion");
    println!("    (0,1,true)  - special transposition start <- KEY!");
    println!("    (1,1,false) - substitution");
    println!("    (2,1,false) - transposition complete (advanced by 2)");
    println!();

    // Check subsumption
    println!("  After subsumption:");
    let mut state1 = State::new();
    state1.insert(Position::new(0, 1), algorithm, query_length);
    state1.insert(Position::new_osa_transposing(0, 1), algorithm, query_length);
    state1.insert(Position::new(1, 1), algorithm, query_length);
    state1.insert(Position::new(2, 1), algorithm, query_length);

    let has_special = state1
        .positions()
        .iter()
        .any(|p| p.is_special() && p.term_index == 0 && p.num_errors == 1);

    if has_special {
        println!("    OK Special position (0,1,true) survived!");
    } else {
        println!("    FAIL Special position was removed!");
        panic!("Defect: special position removed");
    }

    println!("    Final state: {:?}", state1.positions());
    println!();

    println!("Step 2: Process dictionary edge 'y' from 'u' node");
    println!("  Need to check which positions generate valid transitions");
    println!();

    for pos in state1.positions() {
        println!("  Position {:?}:", pos);

        if pos.is_special() && pos.term_index == 0 && pos.num_errors == 1 {
            println!("    -> This is the SPECIAL transposition position!");
            println!("    CV for 'y' vs 'yu' at offset 0:");
            println!("      cv[0]: 'y' == 'y'? true <- MATCH!");
            println!();
            println!("    From transition.rs lines 286-290:");
            println!("      if t && cv[h]:");
            println!("        next.push(Position::new(i + 2, e));");
            println!("      -> Position::new(0 + 2, 1) = (2,1,false)");
            println!();
            println!("    This position (2,1,false) has:");
            println!("      term_index = 2 (consumed all query chars)");
            println!("      num_errors = 1");
            println!();
            println!("    At final node 'y', this should match with distance 1!");
        } else if pos.term_index == 2 {
            println!("    -> Already at term_index=2, consumed full query");
            println!("    At final node, distance should be: {}", pos.num_errors);
        }
    }
    println!();

    // Now test actual automaton
    println!("=== Actual Automaton Results ===\n");
    let transducer = Transducer::new(dict, algorithm);
    let results: Vec<_> = transducer
        .query_with_distance(query, max_distance)
        .collect();

    println!("Results: {:?}", results);
    println!();

    if results.is_empty() {
        println!("FAIL: No results found!");
        panic!("Should find 'uy'");
    }

    let candidate = results
        .iter()
        .find(|c| c.term == "uy")
        .expect("Should find 'uy'");
    println!("Found 'uy' at distance: {}", candidate.distance);
    println!();

    if candidate.distance == 1 {
        println!("SUCCESS: Correct distance!");
    } else {
        println!("FAIL: Wrong distance!");
        println!("  Expected: 1");
        println!("  Got: {}", candidate.distance);
        println!();
        println!("This means either:");
        println!("  1. The transposition path isn't being taken");
        println!("  2. The distance is being calculated incorrectly");
        println!("  3. A shorter non-transposition path is being found");
        panic!("Distance mismatch");
    }
}
