//! Stable 42-bit phonetic feature projection for foreign-language bindings.

use super::index::boundary;
use super::{LlevPhoneticFeatureWeights, LlevStatus};
#[cfg(feature = "bindings-phonetic")]
use crate::phonetic::feature_distance::feature_set_distance_weighted;
#[cfg(feature = "bindings-phonetic")]
use crate::phonetic::{
    are_similar, chars_with_any_feature, chars_with_features, expand_feature_based, get_features,
    get_similar_chars, get_voicing_pair, is_free_substitution, PhoneticFeature,
};
#[cfg(feature = "bindings-phonetic")]
use rustc_hash::FxHashSet;
#[cfg(feature = "bindings-phonetic")]
use std::slice;

/// Stable feature bit positions published by the C ABI. These are deliberately
/// explicit instead of relying on Rust enum discriminants or layout.
#[cfg(feature = "bindings-phonetic")]
const FEATURES: [PhoneticFeature; 42] = {
    use PhoneticFeature::*;
    [
        Voiced,
        Voiceless,
        Stop,
        Fricative,
        Affricate,
        Nasal,
        Approximant,
        Lateral,
        Rhotic,
        Bilabial,
        Labiodental,
        Dental,
        Alveolar,
        PostAlveolar,
        Palatal,
        Velar,
        Glottal,
        Vowel,
        Consonant,
        High,
        Mid,
        Low,
        Front,
        Central,
        Back,
        Rounded,
        Unrounded,
        Sibilant,
        Aspirated,
        Tense,
        Pharyngealized,
        Labialized,
        Velarized,
        Retroflex,
        Uvular,
        Pharyngeal,
        Epiglottal,
        Tap,
        Trill,
        Ejective,
        Implosive,
        Click,
    ]
};

#[cfg(not(feature = "bindings-phonetic"))]
fn unavailable() -> (LlevStatus, String) {
    (
        LlevStatus::Unsupported,
        "phonetic bindings were not compiled".into(),
    )
}

#[cfg(feature = "bindings-phonetic")]
fn scalar(value: u32) -> Result<char, (LlevStatus, String)> {
    char::from_u32(value).ok_or((LlevStatus::InvalidArgument, "invalid Unicode scalar".into()))
}

#[cfg(feature = "bindings-phonetic")]
fn mask(features: &FxHashSet<PhoneticFeature>) -> u64 {
    FEATURES
        .iter()
        .enumerate()
        .fold(0u64, |bits, (index, feature)| {
            bits | (u64::from(features.contains(feature)) << index)
        })
}

#[cfg(feature = "bindings-phonetic")]
fn decode_mask(bits: u64) -> Result<Vec<PhoneticFeature>, (LlevStatus, String)> {
    if bits >> FEATURES.len() != 0 {
        return Err((
            LlevStatus::InvalidArgument,
            "unknown phonetic feature bits".into(),
        ));
    }
    Ok(FEATURES
        .iter()
        .enumerate()
        .filter_map(|(index, feature)| ((bits & (1u64 << index)) != 0).then_some(*feature))
        .collect())
}

#[cfg(feature = "bindings-phonetic")]
unsafe fn copy_chars(
    chars: &[char],
    out_chars: *mut u32,
    capacity: usize,
    out_count: *mut usize,
) -> Result<LlevStatus, (LlevStatus, String)> {
    if out_count.is_null() {
        return Err((LlevStatus::NullPointer, "out_count is null".into()));
    }
    if capacity > 0 && out_chars.is_null() {
        return Err((LlevStatus::NullPointer, "out_chars is null".into()));
    }
    out_count.write(chars.len());
    if chars.len() > capacity {
        return Err((
            LlevStatus::LimitExceeded,
            "character capacity insufficient".into(),
        ));
    }
    if !chars.is_empty() {
        for (target, character) in slice::from_raw_parts_mut(out_chars, chars.len())
            .iter_mut()
            .zip(chars)
        {
            *target = *character as u32;
        }
    }
    Ok(LlevStatus::Ok)
}

/// Encode native features of one Unicode scalar as stable low 42 bits.
/// Unknown scalars have zero features.
///
/// # Safety
///
/// Output must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_features(character: u32, out_mask: *mut u64) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_mask.is_null() {
                return Err((LlevStatus::NullPointer, "out_mask is null".into()));
            }
            out_mask.write(mask(&get_features(scalar(character)?)));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (character, out_mask);
            Err(unavailable())
        }
    })
}

/// Copy Unicode scalars matching all (`any=0`) or any (`any=1`) features.
/// Unknown mask bits and selector values are invalid. A zero mask follows
/// native semantics: no matches. Results are sorted by scalar value.
///
/// # Safety
///
/// Output pointers must be valid for capacity.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_chars_with_features(
    feature_mask: u64,
    any: u8,
    out_chars: *mut u32,
    capacity: usize,
    out_count: *mut usize,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if any > 1 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "any must be zero or one".into(),
                ));
            }
            let features = decode_mask(feature_mask)?;
            let mut result: Vec<_> = if any == 1 {
                chars_with_any_feature(&features)
            } else {
                chars_with_features(&features)
            }
            .into_iter()
            .collect();
            result.sort_unstable();
            copy_chars(&result, out_chars, capacity, out_count)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (feature_mask, any, out_chars, capacity, out_count);
            Err(unavailable())
        }
    })
}

/// Copy all native feature-similar Unicode scalars in scalar order.
///
/// # Safety
///
/// Output pointers must be valid for capacity.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_similar_chars(
    character: u32,
    out_chars: *mut u32,
    capacity: usize,
    out_count: *mut usize,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let mut result: Vec<_> = get_similar_chars(scalar(character)?).into_iter().collect();
            result.sort_unstable();
            copy_chars(&result, out_chars, capacity, out_count)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (character, out_chars, capacity, out_count);
            Err(unavailable())
        }
    })
}

/// Return an optional native voicing pair.
///
/// # Safety
///
/// Both outputs must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_voicing_pair(
    character: u32,
    out_character: *mut u32,
    out_found: *mut u8,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_character.is_null() || out_found.is_null() {
                return Err((LlevStatus::NullPointer, "voicing output is null".into()));
            }
            let result = get_voicing_pair(scalar(character)?);
            out_character.write(result.unwrap_or('\0') as u32);
            out_found.write(u8::from(result.is_some()));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (character, out_character, out_found);
            Err(unavailable())
        }
    })
}

/// Return native feature-similarity or free-substitution relation.
/// `free_substitution=0` selects similarity, one selects the cost-zero
/// substitution relation used by the phonetic matcher.
///
/// # Safety
///
/// Output must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_feature_relation(
    source: u32,
    target: u32,
    free_substitution: u8,
    out_matches: *mut u8,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_matches.is_null() {
                return Err((LlevStatus::NullPointer, "out_matches is null".into()));
            }
            if free_substitution > 1 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "relation selector must be zero or one".into(),
                ));
            }
            let source = scalar(source)?;
            let target = scalar(target)?;
            out_matches.write(u8::from(if free_substitution == 1 {
                is_free_substitution(source, target)
            } else {
                are_similar(source, target)
            }));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (source, target, free_substitution, out_matches);
            Err(unavailable())
        }
    })
}

/// Copy the native feature-expansion vector in native order, including any
/// intentional repeated entries. Capacity protocol matches similar_chars.
///
/// # Safety
///
/// Output pointers must be valid for capacity.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_expand_feature_based(
    character: u32,
    out_chars: *mut u32,
    capacity: usize,
    out_count: *mut usize,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let result = expand_feature_based(scalar(character)?);
            copy_chars(&result, out_chars, capacity, out_count)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (character, out_chars, capacity, out_count);
            Err(unavailable())
        }
    })
}

/// Native weighted distance between two phonetic feature sets. Null weights
/// select the native standard costs. Unknown mask bits are invalid.
///
/// # Safety
///
/// Optional weights and output must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_feature_set_distance(
    source_mask: u64,
    target_mask: u64,
    weights: *const LlevPhoneticFeatureWeights,
    out_distance: *mut f64,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_distance.is_null() {
                return Err((LlevStatus::NullPointer, "out_distance is null".into()));
            }
            let source: FxHashSet<_> = decode_mask(source_mask)?.into_iter().collect();
            let target: FxHashSet<_> = decode_mask(target_mask)?.into_iter().collect();
            out_distance.write(feature_set_distance_weighted(
                &source,
                &target,
                &super::phonetic_analysis::weights(weights)?,
            ));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (source_mask, target_mask, weights, out_distance);
            Err(unavailable())
        }
    })
}
