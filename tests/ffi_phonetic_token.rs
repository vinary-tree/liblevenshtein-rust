#![cfg(all(feature = "ffi", feature = "bindings-phonetic"))]

use liblevenshtein::ffi::{
    llev_phonetic_rules_free, llev_phonetic_rules_parse, llev_phonetic_token_free,
    llev_phonetic_token_matches_free, llev_phonetic_token_new, llev_phonetic_token_scan,
    LlevOwnedString, LlevPhoneticRuleSet, LlevPhoneticTokenGrep, LlevPhoneticTokenMatch,
    LlevStatus,
};
use liblevenshtein::phonetic::TokenGrep;
use proptest::prelude::*;
use std::ptr;

unsafe fn text(value: LlevOwnedString) -> String {
    if value.len == 0 {
        return String::new();
    }
    std::str::from_utf8(std::slice::from_raw_parts(value.data.cast(), value.len))
        .unwrap()
        .to_owned()
}

unsafe fn new(
    query: &str,
    rules: *const LlevPhoneticRuleSet,
    default_distance: u8,
) -> *mut LlevPhoneticTokenGrep {
    let mut output = ptr::null_mut();
    assert_eq!(
        llev_phonetic_token_new(
            query.as_ptr().cast(),
            query.len(),
            rules,
            default_distance,
            100,
            &mut output
        ),
        LlevStatus::Ok
    );
    output
}

#[test]
fn token_matches_and_details_agree_with_public_rust() {
    let query = "hello world";
    let document = "helo wrld and hello world";
    let native = TokenGrep::new(query, 1).unwrap().scan(document);
    let grep = unsafe { new(query, ptr::null(), 1) };
    let mut output = ptr::null_mut();
    let mut count = 0;
    assert_eq!(
        unsafe {
            llev_phonetic_token_scan(
                grep,
                document.as_ptr().cast(),
                document.len(),
                100,
                100,
                100,
                &mut output,
                &mut count,
            )
        },
        LlevStatus::Ok
    );
    let matches = unsafe { std::slice::from_raw_parts(output, count) };
    assert_eq!(matches.len(), native.len());
    for (ffi, expected) in matches.iter().zip(native.iter()) {
        assert_eq!((ffi.byte_start, ffi.byte_end), expected.byte_range);
        assert_eq!(ffi.total_distance, expected.total_distance);
        assert_eq!(unsafe { text(ffi.matched_text) }, expected.matched_text);
        let details = unsafe { std::slice::from_raw_parts(ffi.details, ffi.detail_count) };
        assert_eq!(details.len(), expected.token_matches.len());
        for (ffi, expected) in details.iter().zip(expected.token_matches.iter()) {
            assert_eq!(ffi.token_index, expected.token_index);
            assert_eq!((ffi.byte_start, ffi.byte_end), expected.byte_range);
            assert_eq!(unsafe { text(ffi.original_text) }, expected.original_text);
            assert_eq!(
                unsafe { text(ffi.normalized_text) },
                expected.normalized_text
            );
            assert_eq!(ffi.distance, expected.distance);
        }
    }
    unsafe {
        llev_phonetic_token_matches_free(output, count);
        llev_phonetic_token_free(grep);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn generated_token_spans_and_details_equal_native(document in "[a-z ]{0,18}") {
        let query = "cat dog";
        let native = TokenGrep::new(query, 1).unwrap().scan(&document);
        let grep = unsafe { new(query, ptr::null(), 1) };
        let mut output = ptr::null_mut();
        let mut count = 0;
        prop_assert_eq!(unsafe { llev_phonetic_token_scan(grep,
            document.as_ptr().cast(), document.len(), 100, 100, 100,
            &mut output, &mut count) }, LlevStatus::Ok);
        let matches = if count == 0 { &[][..] } else {
            unsafe { std::slice::from_raw_parts(output, count) }
        };
        prop_assert_eq!(matches.len(), native.len());
        for (ffi, expected) in matches.iter().zip(native.iter()) {
            prop_assert_eq!((ffi.byte_start, ffi.byte_end), expected.byte_range);
            prop_assert_eq!(ffi.total_distance, expected.total_distance);
            prop_assert_eq!(unsafe { text(ffi.matched_text) }, expected.matched_text.clone());
            let details = if ffi.detail_count == 0 { &[][..] } else {
                unsafe { std::slice::from_raw_parts(ffi.details, ffi.detail_count) }
            };
            prop_assert_eq!(details.len(), expected.token_matches.len());
            for (actual, expected) in details.iter().zip(expected.token_matches.iter()) {
                prop_assert_eq!(actual.token_index, expected.token_index);
                prop_assert_eq!((actual.byte_start, actual.byte_end), expected.byte_range);
                prop_assert_eq!(unsafe { text(actual.original_text) }, expected.original_text.clone());
                prop_assert_eq!(unsafe { text(actual.normalized_text) }, expected.normalized_text.clone());
                prop_assert_eq!(actual.distance, expected.distance);
            }
        }
        unsafe { llev_phonetic_token_matches_free(output, count); llev_phonetic_token_free(grep); }
    }

    #[test]
    fn generated_token_capacity_rejection_is_atomic(separator in "[a-z]{1,6}") {
        let document = format!("cat dog {separator} cat dog");
        let grep = unsafe { new("cat dog", ptr::null(), 0) };
        let mut output = 1usize as *mut LlevPhoneticTokenMatch;
        let mut count = 999;
        for (max_matches, max_details) in [(1, 100), (100, 1)] {
            prop_assert_eq!(unsafe { llev_phonetic_token_scan(grep,
                document.as_ptr().cast(), document.len(), 100, max_matches, max_details,
                &mut output, &mut count) }, LlevStatus::LimitExceeded);
            prop_assert_eq!(output as usize, 1);
            prop_assert_eq!(count, 999);
        }
        unsafe { llev_phonetic_token_free(grep) };
    }
}

#[test]
fn token_rules_copy_and_limits_are_transactional() {
    let source = "ph -> f;";
    let mut rules = ptr::null_mut();
    assert_eq!(
        unsafe { llev_phonetic_rules_parse(source.as_ptr().cast(), source.len(), &mut rules) },
        LlevStatus::Ok
    );
    let grep = unsafe { new("fone", rules, 0) };
    unsafe {
        llev_phonetic_rules_free(rules);
    }
    let mut output = 1usize as *mut LlevPhoneticTokenMatch;
    let mut count = 999;
    assert_eq!(
        unsafe {
            llev_phonetic_token_scan(
                grep,
                b"phone".as_ptr().cast(),
                5,
                4,
                100,
                100,
                &mut output,
                &mut count,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(output as usize, 1);
    assert_eq!(count, 999);
    assert_eq!(
        unsafe {
            llev_phonetic_token_scan(
                grep,
                b"phone".as_ptr().cast(),
                5,
                100,
                100,
                100,
                &mut output,
                &mut count,
            )
        },
        LlevStatus::Ok
    );
    let values = unsafe { std::slice::from_raw_parts(output, count) };
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].total_distance, 0);
    assert_eq!(unsafe { text(values[0].matched_text) }, "phone");
    unsafe {
        llev_phonetic_token_matches_free(output, count);
        llev_phonetic_token_free(grep);
    }
}
