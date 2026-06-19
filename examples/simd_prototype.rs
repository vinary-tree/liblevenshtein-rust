//! Simple prototype to demonstrate SIMD concepts
//!
//! NOTE: This example is disabled because it requires the `pulp` crate which is not
//! currently a dependency. The liblevenshtein library has its own SIMD implementation
//! in src/transducer/simd.rs that uses x86_64 intrinsics directly.
//!
//! To see the actual SIMD implementation, check:
//! - src/transducer/simd.rs (always compiled on x86_64 targets)

fn main() {
    println!("=== SIMD Prototype (Disabled) ===\n");
    println!("This example is currently disabled because it requires the `pulp` crate.");
    println!(
        "The liblevenshtein library has its own SIMD implementation using x86_64 intrinsics.\n"
    );
    println!("SIMD in liblevenshtein:");
    println!("  - Automatic on x86_64 targets (no feature flag required)");
    println!("  - The SIMD code lives in: src/transducer/simd.rs");
    println!("  - Runtime CPU detection chooses AVX2/SSE4.1/scalar at call time");
    println!("If you want to enable this example, add `pulp` to Cargo.toml dependencies.");
}

// The code below is commented out because it requires the `pulp` crate
// which is not currently a dependency. Uncomment if you add pulp to Cargo.toml.

/*
#[cfg(target_arch = "x86_64")]
fn test_vectorized_add() {
    use pulp::{Arch, Simd, WithSimd};

    println!("=== Test 1: Vectorized Addition ===");

    struct VectorAdd {
        a: [u32; 8],
        b: [u32; 8],
        result: [u32; 8],
    }

    impl WithSimd for VectorAdd {
        type Output = [u32; 8];

        #[inline(always)]
        fn with_simd<S: Simd>(mut self, simd: S) -> Self::Output {
            // Simple scalar addition; pulp u32 operations are more complex.
            for i in 0..8 {
                self.result[i] = self.a[i] + self.b[i];
            }
            self.result
        }
    }

    let arch = Arch::new();
    let a = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let b = [8u32, 7, 6, 5, 4, 3, 2, 1];
    let result = arch.dispatch(VectorAdd {
        a,
        b,
        result: [0u32; 8],
    });

    println!("Input A:  {:?}", a);
    println!("Input B:  {:?}", b);
    println!("Sum:      {:?}", result);
    println!("Expected: [9, 9, 9, 9, 9, 9, 9, 9]\n");
}

#[cfg(target_arch = "x86_64")]
fn test_vectorized_min() {
    use pulp::{Arch, Simd, WithSimd};

    println!("=== Test 2: Vectorized Minimum (for DP) ===");

    struct VectorMin {
        deletion: [u32; 8],
        insertion: [u32; 8],
        substitution: [u32; 8],
    }

    impl WithSimd for VectorMin {
        type Output = [u32; 8];

        #[inline(always)]
        fn with_simd<S: Simd>(self, _simd: S) -> Self::Output {
            let mut result = [0u32; 8];
            // Compute min3 for each element
            for i in 0..8 {
                result[i] = self.deletion[i]
                    .min(self.insertion[i])
                    .min(self.substitution[i]);
            }
            result
        }
    }

    let arch = Arch::new();

    // Simulate three DP options: deletion, insertion, substitution
    let deletion = [5u32, 8, 3, 9, 2, 7, 4, 6];
    let insertion = [6u32, 7, 4, 8, 3, 6, 5, 7];
    let substitution = [4u32, 9, 2, 7, 4, 5, 6, 8];

    let min_result = arch.dispatch(VectorMin {
        deletion,
        insertion,
        substitution,
    });

    println!("Deletion:     {:?}", deletion);
    println!("Insertion:    {:?}", insertion);
    println!("Substitution: {:?}", substitution);
    println!("Min:          {:?}", min_result);
    println!("Expected:     [4, 7, 2, 7, 2, 5, 4, 6]");

    // Verify correctness
    let expected = [4u32, 7, 2, 7, 2, 5, 4, 6];
    let correct = min_result == expected;
    println!("Correct: {}\n", correct);
}
*/
