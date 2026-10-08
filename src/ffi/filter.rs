//! Bounded UTF-8 source-filter scoring at the C boundary.

use super::{
    index::{boundary, utf8},
    LlevStatus,
};
use crate::filter::{jaro_similarity, jaro_winkler_similarity_scaled};
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
