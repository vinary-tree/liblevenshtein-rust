//! Bounded UTF-8 source-filter scoring at the C boundary.

use super::{
    index::{boundary, utf8},
    LlevStatus,
};
use crate::filter::{jaro_similarity, jaro_winkler_similarity_scaled, HybridMatcher, NgramIndex};
use std::ffi::c_char;

/// Score two Unicode strings with native Jaro or scaled Jaro-Winkler.
///
/// A zero `prefix_scale` selects Jaro; values in `(0, 0.25]` select scaled
/// Jaro-Winkler. Both input byte lengths and the worst-case scalar comparison
/// count are checked before the native scorer allocates any scratch vectors.
///
/// # Safety
/// `left` and `right` must address readable UTF-8 bytes for their declared
/// lengths. Null is allowed only with zero length. `out_score` must address
/// writable storage disjoint from both inputs.
#[no_mangle]
pub unsafe extern "C" fn llev_jaro_similarity_utf8(
    left: *const c_char,
    left_len: usize,
    right: *const c_char,
    right_len: usize,
    prefix_scale: f64,
    max_input_bytes: usize,
    max_comparisons: usize,
    out_score: *mut f64,
) -> LlevStatus {
    boundary(|| {
        if out_score.is_null() {
            return Err((LlevStatus::NullPointer, "Jaro score output is null".into()));
        }
        out_score.write(0.0);
        if !prefix_scale.is_finite() || !(0.0..=0.25).contains(&prefix_scale) {
            return Err((
                LlevStatus::InvalidArgument,
                "Jaro prefix scale must be finite and within [0, 0.25]".into(),
            ));
        }
        if left_len > max_input_bytes || right_len > max_input_bytes {
            return Err((
                LlevStatus::LimitExceeded,
                "Jaro input exceeds max_input_bytes".into(),
            ));
        }
        let left = utf8(left, left_len)?;
        let right = utf8(right, right_len)?;
        let comparisons = left
            .chars()
            .count()
            .checked_mul(right.chars().count())
            .ok_or((
                LlevStatus::LimitExceeded,
                "Jaro comparison count overflows".into(),
            ))?;
        if comparisons > max_comparisons {
            return Err((
                LlevStatus::LimitExceeded,
                "Jaro comparison count exceeds max_comparisons".into(),
            ));
        }
        let score = if prefix_scale == 0.0 {
            jaro_similarity(left, right)
        } else {
            jaro_winkler_similarity_scaled(left, right, prefix_scale)
        };
        out_score.write(score);
        Ok(LlevStatus::Ok)
    })
}

/// Test one UTF-8 source candidate with the native n-gram or hybrid filter.
///
/// `mode` is 1 for n-gram only and 2 for n-gram plus Jaro-Winkler. The
/// candidate is indexed only for this call, so no mutable index handle or
/// borrowed source memory survives the boundary.
///
/// # Safety
/// `query` and `candidate` must address readable UTF-8 bytes for their
/// declared lengths. Null is allowed only with zero length. `out_accept`
/// must address writable storage disjoint from both inputs.
#[no_mangle]
pub unsafe extern "C" fn llev_source_filter_utf8(
    query: *const c_char,
    query_len: usize,
    candidate: *const c_char,
    candidate_len: usize,
    mode: u32,
    ngram_size: usize,
    max_distance: usize,
    jaro_threshold: f64,
    max_input_bytes: usize,
    max_comparisons: usize,
    out_accept: *mut u8,
) -> LlevStatus {
    boundary(|| {
        if out_accept.is_null() {
            return Err((
                LlevStatus::NullPointer,
                "source filter output is null".into(),
            ));
        }
        out_accept.write(0);
        if !matches!(mode, 1 | 2)
            || !jaro_threshold.is_finite()
            || !(0.0..=1.0).contains(&jaro_threshold)
            || (mode == 1 && jaro_threshold != 0.0)
        {
            return Err((
                LlevStatus::InvalidArgument,
                "invalid source filter mode or Jaro threshold".into(),
            ));
        }
        if query_len > max_input_bytes || candidate_len > max_input_bytes {
            return Err((
                LlevStatus::LimitExceeded,
                "source filter input exceeds max_input_bytes".into(),
            ));
        }
        let query = utf8(query, query_len)?;
        let candidate = utf8(candidate, candidate_len)?;
        if mode == 2 && jaro_threshold > 0.0 {
            let comparisons = query
                .chars()
                .count()
                .checked_mul(candidate.chars().count())
                .ok_or((
                    LlevStatus::LimitExceeded,
                    "hybrid Jaro comparison count overflows".into(),
                ))?;
            if comparisons > max_comparisons {
                return Err((
                    LlevStatus::LimitExceeded,
                    "hybrid Jaro comparison count exceeds max_comparisons".into(),
                ));
            }
        }
        let accepted = if mode == 1 {
            let mut index = NgramIndex::new(ngram_size);
            index.insert(candidate);
            index
                .find_candidates(query, max_distance)
                .contains(&candidate)
        } else {
            let matcher = HybridMatcher::with_config(
                std::iter::once(candidate.to_owned()),
                ngram_size,
                jaro_threshold,
            );
            matcher
                .filter_candidates(query, max_distance)
                .contains(&candidate)
        };
        out_accept.write(u8::from(accepted));
        Ok(LlevStatus::Ok)
    })
}
