//! Stable code-generation witness for the Phase 5 variant seam.
//!
//! This is not a user-facing example. The zero-cost gate builds it with the
//! same optimized flags before and after the seam refactor, extracts the exact
//! bytes of [`liblevenshtein_phase5_transition_standard_probe`], and compares
//! them. A separate optimized-LLVM-IR audit checks that the constant Standard
//! call retains only the Standard leaf and no runtime selector. Keeping the
//! probe in Rust source makes both witnesses reproducible across machines with
//! the same toolchain and target CPU.

use liblevenshtein::transducer::transition::transition_position;
use liblevenshtein::transducer::{Algorithm, Position};
use std::hint::black_box;

/// Exercise the exact `Algorithm::Standard` position-transition hot path.
///
/// Scalar arguments keep the exported witness independent of Rust aggregate
/// ABI details. The fixed eight-element characteristic vector covers the
/// current inline fast path; `cv_len` is clamped before slicing.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn liblevenshtein_phase5_transition_standard_probe(
    term_index: usize,
    num_errors: usize,
    cv_bits: u8,
    cv_len: usize,
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
) -> usize {
    let mut cv = [false; 8];
    let cv_len = cv_len.min(cv.len());
    for (index, matched) in cv[..cv_len].iter_mut().enumerate() {
        *matched = cv_bits & (1 << index) != 0;
    }

    transition_position(
        &Position::new(term_index, num_errors),
        &cv[..cv_len],
        query_length,
        max_distance,
        Algorithm::Standard,
        prefix_mode,
    )
    .iter()
    .fold(0usize, |digest, position| {
        digest
            .wrapping_mul(1_099_511_628_211)
            .wrapping_add(position.term_index)
            .wrapping_add(position.num_errors.rotate_left(17))
    })
}

fn main() {
    black_box(liblevenshtein_phase5_transition_standard_probe(
        black_box(0),
        black_box(0),
        black_box(0b0000_0010),
        black_box(4),
        black_box(4),
        black_box(2),
        black_box(false),
    ));
}
