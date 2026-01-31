//! Slice copying overhead analysis for phonetic rewrite rules.
//!
//! This program measures the overhead of slice copying operations in the
//! phonetic rewrite system by instrumenting Vec operations with atomic counters.
//!
//! **Usage**:
//! ```bash
//! cargo run --example phonetic_slice_analysis --features "phonetic-rules,perf-instrumentation"
//! ```

#[cfg(all(feature = "phonetic-rules", feature = "perf-instrumentation"))]
use liblevenshtein::phonetic::*;

#[cfg(not(all(feature = "phonetic-rules", feature = "perf-instrumentation")))]
fn main() {
    println!("This example requires both 'phonetic-rules' and 'perf-instrumentation' features.");
    println!("Run with:");
    println!("  cargo run --example phonetic_slice_analysis --features \"phonetic-rules,perf-instrumentation\"");
}

#[cfg(all(feature = "phonetic-rules", feature = "perf-instrumentation"))]
fn main() {
    use std::time::Instant;

    println!("=== Phonetic Rules Slice Copying Analysis ===\n");
    println!("Measuring bytes copied and allocations for different input sizes\n");

    let ortho_rules = orthography_rules();

    // Test different input sizes
    let test_cases = [
        (5, "phone"),   // 5 phones
        (10, "phonetics"), // ~9 phones
        (20, "phoneticsphonetics"), // ~18 phones
        (50, "phoneticsphoneticsphoneticsphoneticsphoneticsphone"), // ~50 phones
    ];

    println!("| Input Size | Actual | Bytes Copied | Allocations | Bytes/Phone | Allocs/Phone | Time (ns) |");
    println!("|------------|--------|--------------|-------------|-------------|--------------|-----------|");

    for (expected_size, word) in test_cases.iter() {
        let phones: Vec<PhoneByte> = word
            .bytes()
            .map(|b| {
                if matches!(b, b'a' | b'e' | b'i' | b'o' | b'u') {
                    PhoneByte::Vowel(b)
                } else {
                    PhoneByte::Consonant(b)
                }
            })
            .collect();

        let actual_size = phones.len();
        let fuel = actual_size * ortho_rules.len() * MAX_EXPANSION_FACTOR;

        // Reset counters
        application::reset_perf_stats();

        // Run and time the operation
        let start = Instant::now();
        let _result = apply_rules_seq(&ortho_rules, &phones, fuel);
        let duration = start.elapsed();

        // Get stats
        let (bytes_copied, allocations) = application::get_perf_stats();

        let bytes_per_phone = bytes_copied as f64 / actual_size as f64;
        let allocs_per_phone = allocations as f64 / actual_size as f64;

        println!(
            "| {} | {} | {} | {} | {:.1} | {:.2} | {} |",
            expected_size,
            actual_size,
            bytes_copied,
            allocations,
            bytes_per_phone,
            allocs_per_phone,
            duration.as_nanos(),
        );
    }

    println!("\n=== Analysis ===\n");
    println!("**Bytes Copied**:");
    println!("- Includes initial clone (to_vec) + all slice copies during rule applications");
    println!("- Each rule application: prefix + replacement + suffix");
    println!();
    println!("**Allocations**:");
    println!("- Initial clone: 1 allocation");
    println!("- Per rule application: 1 allocation");
    println!("- Total = 1 + (number of successful rule applications)");
    println!();
    println!("**Expected Scaling**:");
    println!("- If iterations ~ O(√n), then:");
    println!("  - Allocations ~ O(√n) [iterations × successful applications]");
    println!("  - Bytes copied ~ O(n × √n) = O(n^1.5) [allocations × average size]");
    println!();
    println!("**Overhead Calculation**:");
    println!("- Total time = computation + memory operations");
    println!("- Memory operations ≈ bytes_copied × (L1 cache latency)");
    println!("- L1 latency ≈ 1.7 ns @ 2.3 GHz (4 cycles)");
    println!("- Memory overhead % = (bytes_copied × 1.7 ns) / total_time × 100%");
}
