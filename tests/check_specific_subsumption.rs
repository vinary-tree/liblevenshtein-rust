use liblevenshtein::prelude::*;
use liblevenshtein::transducer::Position;

#[test]
fn check_1_1_false_subsumes_0_1_true() {
    println!("\n=== Check if (1,1,false) subsumes (0,1,true) ===\n");

    let algorithm = Algorithm::Transposition;
    let query_length = 2;

    let p_1_1 = Position::new(1, 1); // (1,1,false)
    let p_0_1_special = Position::new_osa_transposing(0, 1); // (0,1,true)

    println!(
        "p_1_1:          ({}, {}, false)",
        p_1_1.term_index, p_1_1.num_errors
    );
    println!(
        "p_0_1_special:  ({}, {}, true)",
        p_0_1_special.term_index, p_0_1_special.num_errors
    );
    println!();

    let subsumes = p_1_1.subsumes(&p_0_1_special, algorithm, query_length);
    println!("Does (1,1,false) subsume (0,1,true)? {}", subsumes);
    println!();

    if subsumes {
        println!("❌ REGRESSION FOUND!");
        println!("  (1,1,false) incorrectly subsumes (0,1,true)");
        println!();
        println!("  Let's trace through the subsumption logic:");
        println!("  From position.rs line 82-145 (subsumes function):");
        println!();
        println!("  Input:");
        println!(
            "    self (lhs): i={}, e={}, s={}",
            p_1_1.term_index,
            p_1_1.num_errors,
            p_1_1.is_special()
        );
        println!(
            "    other (rhs): j={}, f={}, t={}",
            p_0_1_special.term_index,
            p_0_1_special.num_errors,
            p_0_1_special.is_special()
        );
        println!("    algorithm: Transposition");
        println!("    query_length: {}", query_length);
        println!();
        println!("  Step 1: Check e > f?");
        println!(
            "    {} > {} = {}",
            p_1_1.num_errors,
            p_0_1_special.num_errors,
            p_1_1.num_errors > p_0_1_special.num_errors
        );
        println!("    If true, return false (cannot subsume with more errors)");
        println!();
        println!("  Step 2: Transposition algorithm logic");
        println!("    if s (lhs is special): {}", p_1_1.is_special());

        println!("      → NO, lhs is NOT special");
        println!();
        println!("    if t (rhs is special): {}", p_0_1_special.is_special());
        println!("      → YES, rhs IS special");
        println!();
        println!("    Lines 116-125 handle this case:");
        println!("      if t {{");
        println!("        // rhs is special: adjusted formula");
        println!("        let adjusted_diff = if j < i {{");
        println!("          i.saturating_sub(j).saturating_sub(1)");
        println!("        }} else {{");
        println!("          j.saturating_sub(i) + 1");
        println!("        }};");
        println!("        return adjusted_diff <= (f - e);");
        println!("      }}");
        println!();
        println!(
            "    Applying with i={}, j={}, e={}, f={}:",
            p_1_1.term_index, p_0_1_special.term_index, p_1_1.num_errors, p_0_1_special.num_errors
        );
        println!(
            "      j < i? {} < {} = {}",
            p_0_1_special.term_index,
            p_1_1.term_index,
            p_0_1_special.term_index < p_1_1.term_index
        );
        println!("      → YES, so adjusted_diff = i.saturating_sub(j).saturating_sub(1)");
        println!(
            "      adjusted_diff = {}.saturating_sub({}).saturating_sub(1)",
            p_1_1.term_index, p_0_1_special.term_index
        );
        println!(
            "      adjusted_diff = {}.saturating_sub(1)",
            p_1_1.term_index - p_0_1_special.term_index
        );
        println!(
            "      adjusted_diff = {}",
            (p_1_1.term_index - p_0_1_special.term_index).saturating_sub(1)
        );
        println!();
        println!("      Check: adjusted_diff <= (f - e)?");
        println!(
            "      {} <= ({} - {})",
            (p_1_1.term_index - p_0_1_special.term_index).saturating_sub(1),
            p_0_1_special.num_errors,
            p_1_1.num_errors
        );
        println!(
            "      {} <= {}",
            (p_1_1.term_index - p_0_1_special.term_index).saturating_sub(1),
            p_0_1_special.num_errors - p_1_1.num_errors
        );
        println!("      0 <= 0");
        println!("      → TRUE!");
        println!();
        println!("  This is the defect: the adjusted formula for transposition is incorrect.");
        println!("  A normal position at (1,1) should NOT subsume a special transposition");
        println!("  position at (0,1) - they represent different computational paths!");
        panic!("Defect confirmed: incorrect transposition subsumption formula");
    } else {
        println!("✓ Subsumption check is correct.");
    }
}
