#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{llev_jaro_similarity_utf8, LlevStatus};
use liblevenshtein::filter::{
    jaro_similarity, jaro_winkler_similarity, jaro_winkler_similarity_scaled,
};
use std::ffi::c_char;

fn score(
    left: &str,
    right: &str,
    scale: f64,
    max_bytes: usize,
    max_work: usize,
) -> (LlevStatus, f64) {
    let mut value = f64::NAN;
    let status = unsafe {
        llev_jaro_similarity_utf8(
            left.as_ptr().cast::<c_char>(),
            left.len(),
            right.as_ptr().cast::<c_char>(),
            right.len(),
            scale,
            max_bytes,
            max_work,
            &mut value,
        )
    };
    (status, value)
}

#[test]
fn native_jaro_filter_scores_match_rust_surface() {
    for (left, right) in [
        ("", ""),
        ("", "abc"),
        ("MARTHA", "MARHTA"),
        ("café", "cafe"),
    ] {
        for (scale, expected) in [
            (0.0, jaro_similarity(left, right)),
            (0.1, jaro_winkler_similarity(left, right)),
            (0.2, jaro_winkler_similarity_scaled(left, right, 0.2)),
        ] {
            let (status, actual) = score(left, right, scale, 1024, 1024);
            assert_eq!(status, LlevStatus::Ok);
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn native_jaro_filter_rejects_limits_and_bad_utf8_before_scoring() {
    for (bytes, work, expected) in [
        (2, 100, LlevStatus::LimitExceeded),
        (100, 1, LlevStatus::LimitExceeded),
        (100, 100, LlevStatus::Ok),
    ] {
        let (status, value) = score("abcd", "abdc", 0.1, bytes, work);
        assert_eq!(status, expected);
        if status == LlevStatus::LimitExceeded {
            assert_eq!(value, 0.0);
        }
    }
    let (status, value) = score("abc", "abc", 0.5, 100, 100);
    assert_eq!(status, LlevStatus::InvalidArgument);
    assert_eq!(value, 0.0);

    let mut value = 9.0;
    let invalid = [0xffu8];
    let status = unsafe {
        llev_jaro_similarity_utf8(
            invalid.as_ptr().cast::<c_char>(),
            invalid.len(),
            std::ptr::null(),
            0,
            0.0,
            100,
            100,
            &mut value,
        )
    };
    assert_eq!(status, LlevStatus::InvalidUtf8);
    assert_eq!(value, 0.0);
}
