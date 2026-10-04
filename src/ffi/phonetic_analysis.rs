//! Bounded, Unicode-scalar phonetic analysis shared by foreign bindings.

use super::index::boundary;
#[cfg(feature = "bindings-phonetic")]
use super::index::utf8;
use super::LlevStatus;
#[cfg(feature = "bindings-phonetic")]
use crate::phonetic::feature_distance::{
    articulatory_distance_weighted, articulatory_edit_distance_weighted, FeatureDistanceWeights,
};
use std::ffi::c_char;
#[cfg(feature = "bindings-phonetic")]
use std::slice;

/// Explicit dimension costs for articulatory substitution.
///
/// Fields have the same meaning and defaults as Rust's
/// `FeatureDistanceWeights`. Every field must be finite and nonnegative.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevPhoneticFeatureWeights {
    /// Voicing difference cost.
    pub voicing: f64,
    /// Cost per place-of-articulation step.
    pub place_step: f64,
    /// Fallback manner or missing-place cost.
    pub manner_default: f64,
    /// Multiplier on curated manner-pair costs.
    pub manner_table_scale: f64,
    /// Cost per vowel-height step.
    pub vowel_height_step: f64,
    /// Cost per vowel-backness step.
    pub vowel_backness_step: f64,
    /// Vowel-rounding difference cost.
    pub vowel_rounding: f64,
}

#[cfg(feature = "bindings-phonetic")]
pub(super) unsafe fn weights(
    input: *const LlevPhoneticFeatureWeights,
) -> Result<FeatureDistanceWeights, (LlevStatus, String)> {
    let Some(input) = input.as_ref() else {
        return Ok(FeatureDistanceWeights::standard());
    };
    let costs = [
        input.voicing,
        input.place_step,
        input.manner_default,
        input.manner_table_scale,
        input.vowel_height_step,
        input.vowel_backness_step,
        input.vowel_rounding,
    ];
    if costs.iter().any(|cost| !cost.is_finite() || *cost < 0.0) {
        return Err((
            LlevStatus::InvalidArgument,
            "phonetic feature weights must be finite and nonnegative".into(),
        ));
    }
    Ok(FeatureDistanceWeights {
        voicing: input.voicing,
        place_step: input.place_step,
        manner_default: input.manner_default,
        manner_table_scale: input.manner_table_scale,
        vowel_height_step: input.vowel_height_step,
        vowel_backness_step: input.vowel_backness_step,
        vowel_rounding: input.vowel_rounding,
    })
}

#[cfg(not(feature = "bindings-phonetic"))]
fn unavailable() -> (LlevStatus, String) {
    (
        LlevStatus::Unsupported,
        "phonetic bindings were not compiled; enable bindings-phonetic".into(),
    )
}

/// Compute the native feature-weighted distance between two Unicode scalars.
///
/// # Safety
///
/// `out_distance` and non-null `weights` must address valid storage.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_articulatory_distance(
    source: u32,
    target: u32,
    feature_weights: *const LlevPhoneticFeatureWeights,
    out_distance: *mut f64,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let output = out_distance
                .as_mut()
                .ok_or((LlevStatus::NullPointer, "out_distance is null".to_string()))?;
            let source = char::from_u32(source).ok_or((
                LlevStatus::InvalidArgument,
                "source is not a Unicode scalar".to_string(),
            ))?;
            let target = char::from_u32(target).ok_or((
                LlevStatus::InvalidArgument,
                "target is not a Unicode scalar".to_string(),
            ))?;
            *output = articulatory_distance_weighted(source, target, &weights(feature_weights)?);
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (source, target, feature_weights, out_distance);
            Err(unavailable())
        }
    })
}

/// Compute native articulatory edit distance under an explicit cell budget.
///
/// The product of source and target scalar counts must not exceed
/// `max_cells`; zero is invalid. A rejected call does not modify the result.
///
/// # Safety
///
/// Non-empty UTF-8 input buffers, non-null `weights`, and `out_distance` must
/// address valid storage.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_articulatory_edit_distance(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
    feature_weights: *const LlevPhoneticFeatureWeights,
    max_cells: usize,
    out_distance: *mut f64,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let output = out_distance
                .as_mut()
                .ok_or((LlevStatus::NullPointer, "out_distance is null".to_string()))?;
            if max_cells == 0 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "max_cells must be positive".into(),
                ));
            }
            let source = utf8(source, source_len)?;
            let target = utf8(target, target_len)?;
            let source_units = source.chars().count();
            let target_units = target.chars().count();
            if source_units > max_cells
                || target_units > max_cells
                || source_units
                    .checked_mul(target_units)
                    .is_none_or(|cells| cells > max_cells)
            {
                return Err((
                    LlevStatus::LimitExceeded,
                    "articulatory edit work exceeds max_cells".into(),
                ));
            }
            *output =
                articulatory_edit_distance_weighted(source, target, &weights(feature_weights)?);
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                source,
                source_len,
                target,
                target_len,
                feature_weights,
                max_cells,
                out_distance,
            );
            Err(unavailable())
        }
    })
}

#[cfg(feature = "bindings-phonetic")]
fn syllable_positions(text: &str, ipa: u8) -> Result<Vec<usize>, (LlevStatus, String)> {
    match ipa {
        0 => Ok(crate::phonetic::syllable_boundaries(text)),
        1 => Ok(crate::phonetic::ipa_syllable_boundaries(text)),
        _ => Err((
            LlevStatus::InvalidArgument,
            format!("phonetic IPA selector must be zero or one, got {ipa}"),
        )),
    }
}

/// Return the native English-orthographic or IPA syllable count.
///
/// `max_input_scalars` is a positive work ceiling. `ipa=0` selects English
/// orthography; `ipa=1` selects language-agnostic IPA heuristics.
///
/// # Safety
///
/// The UTF-8 buffer and output must address valid storage.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_syllable_count(
    input: *const c_char,
    input_len: usize,
    ipa: u8,
    max_input_scalars: usize,
    out_count: *mut usize,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let output = out_count
                .as_mut()
                .ok_or((LlevStatus::NullPointer, "out_count is null".to_string()))?;
            if max_input_scalars == 0 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "max_input_scalars must be positive".into(),
                ));
            }
            let input = utf8(input, input_len)?;
            if input.chars().count() > max_input_scalars {
                return Err((
                    LlevStatus::LimitExceeded,
                    "syllable input exceeds max_input_scalars".into(),
                ));
            }
            *output = match ipa {
                0 => crate::phonetic::syllable_count(input),
                1 => crate::phonetic::ipa_syllable_count(input),
                _ => {
                    return Err((
                        LlevStatus::InvalidArgument,
                        format!("phonetic IPA selector must be zero or one, got {ipa}"),
                    ));
                }
            };
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (input, input_len, ipa, max_input_scalars, out_count);
            Err(unavailable())
        }
    })
}

/// Copy native syllable-start offsets in Unicode scalars into caller storage.
///
/// The required count is written before a `LIMIT_EXCEEDED` return. A null
/// `out_positions` with capacity zero is a valid sizing call. No positions are
/// written on insufficient capacity.
///
/// # Safety
///
/// The UTF-8 buffer and non-null outputs must address valid storage.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_syllable_boundaries(
    input: *const c_char,
    input_len: usize,
    ipa: u8,
    max_input_scalars: usize,
    out_positions: *mut usize,
    capacity: usize,
    out_count: *mut usize,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let output_count = out_count
                .as_mut()
                .ok_or((LlevStatus::NullPointer, "out_count is null".to_string()))?;
            if max_input_scalars == 0 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "max_input_scalars must be positive".into(),
                ));
            }
            let input = utf8(input, input_len)?;
            if input.chars().count() > max_input_scalars {
                return Err((
                    LlevStatus::LimitExceeded,
                    "syllable input exceeds max_input_scalars".into(),
                ));
            }
            let positions = syllable_positions(input, ipa)?;
            *output_count = positions.len();
            if positions.len() > capacity {
                return Err((
                    LlevStatus::LimitExceeded,
                    "syllable boundary capacity is too small".into(),
                ));
            }
            if !positions.is_empty() {
                if out_positions.is_null() {
                    return Err((LlevStatus::NullPointer, "out_positions is null".into()));
                }
                slice::from_raw_parts_mut(out_positions, positions.len())
                    .copy_from_slice(&positions);
            }
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                input,
                input_len,
                ipa,
                max_input_scalars,
                out_positions,
                capacity,
                out_count,
            );
            Err(unavailable())
        }
    })
}
