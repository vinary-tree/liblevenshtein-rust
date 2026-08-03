use liblevenshtein::prelude::*;
use liblevenshtein::transducer::{Position, State};

#[test]
fn manual_trace_ab_to_ba() {
    println!("\n=== Manual State Trace for 'ab' -> 'ba' ===\n");

    let query = "ab";
    let dict_words = vec!["ba".to_string()];
    let _dict = DoubleArrayTrie::from_terms(dict_words);

    let query_length = query.len(); // 2
    let max_distance = 1;
    let algorithm = Algorithm::Transposition;

    println!("Query: \"{}\" (len={})", query, query_length);
    println!("Dictionary: [\"ba\"]");
    println!("Max distance: {}", max_distance);
    println!("Algorithm: {:?}\n", algorithm);

    // Initial state at root
    println!("Step 0: Initial state at root");
    println!("  State: [(0,0,false)]");
    println!();

    // Step 1: Process first dictionary edge 'b'
    println!("Step 1: Process dictionary edge 'b' from root");
    println!("  Current position: (0,0,false)");
    println!("  Query char at 0: 'a' (97)");
    println!("  Dict char: 'b' (98)");
    println!("  CV for 'b' vs \"ab\" at offset 0:");
    println!("    cv[0]: 'b' == 'a'? false");
    println!("    cv[1]: 'b' == 'b'? true");
    println!("  CV = [false, true]");
    println!();
    println!("  Match found at index 1 → transposition case!");
    println!("  Generated positions:");
    println!("    (0,1,false) - insertion");
    println!("    (0,1,true)  - special transposition start ← KEY!");
    println!("    (1,1,false) - substitution");
    println!("    (2,1,false) - transposition complete (advanced by 2)");
    println!();

    // Now check if these survive subsumption
    println!("  After subsumption check:");
    let mut state1 = State::new();

    state1.insert(Position::new(0, 1), algorithm, query_length);
    println!("    Inserted (0,1,false)");
    println!("      State now: {:?}", state1.positions());

    state1.insert(Position::new_osa_transposing(0, 1), algorithm, query_length);
    println!("    Inserted (0,1,true)");
    println!("      State now: {:?}", state1.positions());

    let has_special = state1
        .positions()
        .iter()
        .any(|p| p.is_special() && p.term_index == 0 && p.num_errors == 1);
    if has_special {
        println!("      ✓ (0,1,true) survived!");
    } else {
        println!("      ❌ (0,1,true) was subsumed!");
        println!("      This is the defect.");
    }

    state1.insert(Position::new(1, 1), algorithm, query_length);
    println!("    Inserted (1,1,false)");
    println!("      State now: {:?}", state1.positions());

    state1.insert(Position::new(2, 1), algorithm, query_length);
    println!("    Inserted (2,1,false)");
    println!("      State now: {:?}", state1.positions());

    println!("  Final state after 'b': {:?}", state1.positions());
    println!();

    // Step 2: Process second dictionary edge 'a' from the 'b' node
    println!("Step 2: Process dictionary edge 'a' from 'b' node");
    println!("  Need to process all positions from state1");
    println!();

    for pos in state1.positions() {
        println!("  From position {:?}:", pos);
        println!(
            "    term_index={}, num_errors={}, is_special={}",
            pos.term_index,
            pos.num_errors,
            pos.is_special()
        );

        if pos.is_special() && pos.term_index == 0 && pos.num_errors == 1 {
            println!("    This is the special transposition position!");
            println!("    CV for 'a' vs \"ab\" at offset 0:");
            println!("      cv[0]: 'a' == 'a'? true ← Match!");
            println!("    According to line 287-289 in transition.rs:");
            println!("      if cv[h] → should create Position::new(i + 2, e)");
            println!("      → Position::new(0 + 2, 1) = (2,1,false)");
            println!("    This would complete the transposition and reach final state!");
        }
    }

    if !has_special {
        println!("\n  ❌ PROBLEM: Special position (0,1,true) doesn't exist!");
        println!("     It was subsumed during State::insert.");
        println!("     Need to check subsumption logic for transposition.");
    }

    println!("\n=== Subsumption Analysis ===");
    println!("Does (0,1,false) subsume (0,1,true)?");

    let p_normal = Position::new(0, 1);
    let p_special = Position::new_osa_transposing(0, 1);

    let subsumes = p_normal.subsumes(&p_special, algorithm, query_length);
    println!("  p_normal.subsumes(p_special) = {}", subsumes);

    if subsumes {
        println!("  ❌ REGRESSION FOUND: (0,1,false) incorrectly subsumes (0,1,true)!");
        println!("     These are DIFFERENT states in transposition logic.");
        println!("     They should coexist!");
        panic!("Defect confirmed: normal position incorrectly subsumes special position");
    } else {
        println!("  ✓ Subsumption check is correct.");
        if !has_special {
            panic!("Special position missing despite correct subsumption logic!");
        }
    }
}
