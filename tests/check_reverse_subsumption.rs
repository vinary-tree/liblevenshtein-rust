use liblevenshtein::prelude::*;
use liblevenshtein::transducer::Position;

#[test]
fn check_transposition_subsumption_both_ways() {
    println!("\n=== Check Transposition Subsumption Both Ways ===\n");

    let algorithm = Algorithm::Transposition;
    let query_length = 2;

    let p_normal = Position::new(0, 1);
    let p_special = Position::new_special(0, 1);

    println!(
        "p_normal:  ({}, {}, false)",
        p_normal.term_index, p_normal.num_errors
    );
    println!(
        "p_special: ({}, {}, true)",
        p_special.term_index, p_special.num_errors
    );
    println!();

    // Check normal subsumes special
    let normal_subsumes_special = p_normal.subsumes(&p_special, algorithm, query_length);
    println!(
        "Does p_normal subsume p_special? {}",
        normal_subsumes_special
    );

    // Check special subsumes normal
    let special_subsumes_normal = p_special.subsumes(&p_normal, algorithm, query_length);
    println!(
        "Does p_special subsume p_normal? {}",
        special_subsumes_normal
    );
    println!();

    if special_subsumes_normal {
        println!("❌ REGRESSION FOUND!");
        println!("  p_special (0,1,true) subsumes p_normal (0,1,false)");
        println!("  This means when we insert p_special, it REMOVES p_normal!");
        println!("  Then when we insert positions afterward, p_special might get removed.");
        println!();
        println!("  The problem is in the subsumption logic for transposition.");
        println!("  Let's check the logic:");
        println!();
        println!("  From position.rs line 104-131 (Transposition algorithm):");
        println!("    if s (lhs is special):");
        println!("      if t (rhs is special): return i == j");
        println!("      else (rhs not special): return (f == query_length) && (i == j)");
        println!();
        println!("  For p_special.subsumes(p_normal):");
        println!("    s=true (lhs special), t=false (rhs not special)");
        println!(
            "    f={}, query_length={}, i={}, j={}",
            p_normal.num_errors, query_length, p_special.term_index, p_normal.term_index
        );
        println!("    Condition: (f == query_length) && (i == j)");
        println!(
            "    = ({} == {}) && ({} == {})",
            p_normal.num_errors, query_length, p_special.term_index, p_normal.term_index
        );
        println!("    = (1 == 2) && (0 == 0)");
        println!("    = false && true");
        println!("    = false");
        println!();
        println!("  Wait, that should be false! Let me check again...");
        panic!("Need to investigate further");
    } else if !normal_subsumes_special {
        println!("✓ Correct! Neither position subsumes the other.");
        println!("  They should coexist in the state.");
    } else {
        println!("⚠️  One-way subsumption:");
        println!("  p_normal subsumes p_special, but not vice versa.");
    }
}
