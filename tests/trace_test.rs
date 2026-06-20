//! Detailed trace test for diagnosing insertion issues

use liblevenshtein::prelude::*;
use liblevenshtein::transducer::transition::{initial_state, transition_state};
use liblevenshtein::transducer::Unrestricted;

#[test]
fn trace_aple_to_apple() {
    let dict = DoubleArrayTrie::from_terms(vec!["apple"]);
    let root = dict.root();

    let query = b"aple";
    let max_distance = 1;

    println!(
        "\n=== Tracing 'aple' against 'apple' with distance {} ===",
        max_distance
    );
    println!(
        "Query: {:?} (length {})",
        std::str::from_utf8(query).unwrap(),
        query.len()
    );

    let mut state = initial_state(query.len(), max_distance, Algorithm::Standard);
    println!("\nInitial state: {:?}", state);

    // Traverse 'a'
    let a_node = root.transition(b'a').expect("Should have 'a'");
    state = transition_state(
        &state,
        Unrestricted,
        b'a',
        query,
        max_distance,
        Algorithm::Standard,
        false,
    )
    .expect("Should have state after 'a'");
    println!("\nAfter 'a': {:?}", state);
    println!("Node is_final: {}", a_node.is_final());

    // Traverse first 'p'
    let p1_node = a_node.transition(b'p').expect("Should have first 'p'");
    state = transition_state(
        &state,
        Unrestricted,
        b'p',
        query,
        max_distance,
        Algorithm::Standard,
        false,
    )
    .expect("Should have state after first 'p'");
    println!("\nAfter 'ap': {:?}", state);
    println!("Node is_final: {}", p1_node.is_final());

    // Traverse second 'p' - this is where insertion should happen
    if let Some(p2_node) = p1_node.transition(b'p') {
        println!("\nFound second 'p' in dictionary");
        if let Some(new_state) = transition_state(
            &state,
            Unrestricted,
            b'p',
            query,
            max_distance,
            Algorithm::Standard,
            false,
        ) {
            state = new_state;
            println!("After 'app': {:?}", state);
            println!("Node is_final: {}", p2_node.is_final());

            // Traverse 'l'
            if let Some(l_node) = p2_node.transition(b'l') {
                println!("\nFound 'l' in dictionary");
                if let Some(new_state) = transition_state(
                    &state,
                    Unrestricted,
                    b'l',
                    query,
                    max_distance,
                    Algorithm::Standard,
                    false,
                ) {
                    state = new_state;
                    println!("After 'appl': {:?}", state);
                    println!("Node is_final: {}", l_node.is_final());

                    // Traverse 'e'
                    if let Some(e_node) = l_node.transition(b'e') {
                        println!("\nFound 'e' in dictionary");
                        if let Some(new_state) = transition_state(
                            &state,
                            Unrestricted,
                            b'e',
                            query,
                            max_distance,
                            Algorithm::Standard,
                            false,
                        ) {
                            state = new_state;
                            println!("After 'apple': {:?}", state);
                            println!("Node is_final: {}", e_node.is_final());

                            // Check if we should match
                            println!("\n=== Final Check ===");
                            println!("Query length: {}", query.len());
                            if let Some(distance) = state.infer_distance(query.len()) {
                                println!("Inferred distance: {}", distance);
                                println!("Max distance: {}", max_distance);
                                println!("Should match: {}", distance <= max_distance);
                            } else {
                                println!("Could not infer distance!");
                            }
                        } else {
                            println!("No state after 'e' - traversal stopped!");
                        }
                    } else {
                        println!("No 'e' edge from 'appl'");
                    }
                } else {
                    println!("No state after 'l' - traversal stopped!");
                }
            } else {
                println!("No 'l' edge from 'app'");
            }
        } else {
            println!("No state after second 'p' - traversal stopped!");
            println!("This is the problem! The state should continue with insertion.");
        }
    } else {
        println!("\nNo second 'p' edge");
    }
}
