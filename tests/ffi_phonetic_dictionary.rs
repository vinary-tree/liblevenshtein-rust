#![cfg(all(feature = "ffi", feature = "bindings-phonetic"))]

use liblevenshtein::dictionary::phonetic_normalized::{
    PhoneticNormalizedDictionary, PhoneticNormalizedTermIdDictionary,
};
use liblevenshtein::ffi::{
    llev_phonetic_candidates_free, llev_phonetic_dictionary_free, llev_phonetic_dictionary_new,
    llev_phonetic_dictionary_query, llev_phonetic_dictionary_update, LlevAlgorithm,
    LlevPhoneticCandidate, LlevPhoneticDictionary, LlevStatus, LlevUtf8View,
};
use proptest::prelude::*;
use std::ptr;

// INVARIANT-HOOK: LLEV-PHON-BOUND-1 — successful C results equal the complete native result.
// INVARIANT-HOOK: LLEV-PHON-BOUND-4 — pointer-result count stays untouched on limit.

unsafe fn new_dictionary(terms: &[&str], mode: u8) -> *mut LlevPhoneticDictionary {
    let views: Vec<LlevUtf8View> = terms
        .iter()
        .map(|term| LlevUtf8View {
            data: term.as_ptr().cast(),
            len: term.len(),
        })
        .collect();
    let mut dictionary = ptr::null_mut();
    assert_eq!(
        llev_phonetic_dictionary_new(
            views.as_ptr(),
            views.len(),
            ptr::null(),
            LlevAlgorithm::Standard as u32,
            mode,
            100,
            1_000,
            &mut dictionary,
        ),
        LlevStatus::Ok,
    );
    dictionary
}

unsafe fn query(
    dictionary: *mut LlevPhoneticDictionary,
    input: &str,
    bound: usize,
) -> Vec<(String, usize, String)> {
    let mut pointer: *mut LlevPhoneticCandidate = ptr::null_mut();
    let mut count = usize::MAX;
    assert_eq!(
        llev_phonetic_dictionary_query(
            dictionary,
            input.as_ptr().cast(),
            input.len(),
            bound,
            100,
            100,
            &mut pointer,
            &mut count
        ),
        LlevStatus::Ok,
    );
    let values = if count == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(pointer, count)
    };
    let result = values
        .iter()
        .map(|item| {
            let term = if item.term.len == 0 {
                String::new()
            } else {
                std::str::from_utf8(std::slice::from_raw_parts(
                    item.term.data.cast(),
                    item.term.len,
                ))
                .unwrap()
                .to_owned()
            };
            let normalized = if item.normalized_form.len == 0 {
                String::new()
            } else {
                std::str::from_utf8(std::slice::from_raw_parts(
                    item.normalized_form.data.cast(),
                    item.normalized_form.len,
                ))
                .unwrap()
                .to_owned()
            };
            (term, item.distance, normalized)
        })
        .collect();
    llev_phonetic_candidates_free(pointer, count);
    result
}

#[test]
fn both_backends_match_public_native_queries() {
    let terms = ["phone", "fone", "bone", "café", "écho", "phone"];
    let standard = PhoneticNormalizedDictionary::<()>::from_terms(terms);
    let compact = PhoneticNormalizedTermIdDictionary::from_terms(terms);
    for mode in [0, 1] {
        let dictionary = unsafe { new_dictionary(&terms, mode) };
        for (input, bound) in [("fone", 0), ("fone", 1), ("café", 1), ("🦀", 2)] {
            let ffi = unsafe { query(dictionary, input, bound) };
            let native = if mode == 0 {
                standard.query(input, bound)
            } else {
                compact.query(input, bound)
            };
            let native: Vec<_> = native
                .into_iter()
                .map(|item| (item.term, item.distance, item.normalized_form))
                .collect();
            assert_eq!(ffi, native, "mode={mode}, input={input}, bound={bound}");
        }
        unsafe { llev_phonetic_dictionary_free(dictionary) };
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn generated_mutable_and_compact_candidates_equal_native(
        terms in proptest::collection::vec("[a-z]{1,5}", 1..8),
        input in "[a-z]{0,5}",
        bound in 0usize..3,
    ) {
        let mut borrowed: Vec<&str> = terms.iter().map(String::as_str).collect();
        borrowed.push("");
        let standard = PhoneticNormalizedDictionary::<()>::from_terms(borrowed.clone());
        let compact = PhoneticNormalizedTermIdDictionary::from_terms(borrowed.clone());
        for mode in [0, 1] {
            let dictionary = unsafe { new_dictionary(&borrowed, mode) };
            let observed = unsafe { query(dictionary, &input, bound) };
            let expected: Vec<_> = (if mode == 0 {
                standard.query(&input, bound)
            } else {
                compact.query(&input, bound)
            }).into_iter().map(|item| (item.term, item.distance, item.normalized_form)).collect();
            if expected.len() > 1 {
                let mut pointer = 1usize as *mut LlevPhoneticCandidate;
                let mut count = 999;
                prop_assert_eq!(unsafe { llev_phonetic_dictionary_query(dictionary,
                    input.as_ptr().cast(), input.len(), bound, 100, expected.len() - 1,
                    &mut pointer, &mut count) }, LlevStatus::LimitExceeded);
                prop_assert_eq!(pointer as usize, 1);
                prop_assert_eq!(count, 999);
            }
            unsafe { llev_phonetic_dictionary_free(dictionary) };
            prop_assert_eq!(observed, expected);
        }
    }
}

#[test]
fn mutable_updates_and_immutable_rejection_are_exact() {
    let mutable = unsafe { new_dictionary(&["fone"], 0) };
    let compact = unsafe { new_dictionary(&["fone"], 1) };
    let mut changed = 3;
    assert_eq!(
        unsafe {
            llev_phonetic_dictionary_update(
                mutable,
                b"phone".as_ptr().cast(),
                5,
                0,
                20,
                &mut changed,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(changed, 1);
    assert!(unsafe { query(mutable, "fone", 0) }
        .iter()
        .any(|item| item.0 == "phone"));
    assert_eq!(
        unsafe {
            llev_phonetic_dictionary_update(
                mutable,
                b"phone".as_ptr().cast(),
                5,
                1,
                20,
                &mut changed,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(changed, 1);
    assert!(!unsafe { query(mutable, "fone", 0) }
        .iter()
        .any(|item| item.0 == "phone"));
    changed = 3;
    assert_eq!(
        unsafe {
            llev_phonetic_dictionary_update(
                compact,
                b"phone".as_ptr().cast(),
                5,
                0,
                20,
                &mut changed,
            )
        },
        LlevStatus::Unsupported
    );
    assert_eq!(changed, 3);
    unsafe {
        llev_phonetic_dictionary_free(mutable);
        llev_phonetic_dictionary_free(compact);
    }
}

#[test]
fn construction_and_query_limits_are_transactional() {
    let terms = [LlevUtf8View {
        data: b"phone".as_ptr().cast(),
        len: 5,
    }];
    let sentinel = 1usize as *mut LlevPhoneticDictionary;
    let mut output = sentinel;
    assert_eq!(
        unsafe {
            llev_phonetic_dictionary_new(
                terms.as_ptr(),
                1,
                ptr::null(),
                LlevAlgorithm::Standard as u32,
                0,
                1,
                4,
                &mut output,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(output, sentinel);
    let dictionary = unsafe { new_dictionary(&["phone", "fone"], 0) };
    let mut candidates = 1usize as *mut LlevPhoneticCandidate;
    let mut count = 999;
    assert_eq!(
        unsafe {
            llev_phonetic_dictionary_query(
                dictionary,
                b"fone".as_ptr().cast(),
                4,
                0,
                100,
                1,
                &mut candidates,
                &mut count,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(candidates as usize, 1);
    assert_eq!(count, 999);
    assert_eq!(
        unsafe {
            llev_phonetic_dictionary_query(
                dictionary,
                b"fone".as_ptr().cast(),
                4,
                0,
                3,
                100,
                &mut candidates,
                &mut count,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(candidates as usize, 1);
    assert_eq!(count, 999);
    unsafe {
        llev_phonetic_dictionary_free(dictionary);
    }
}
