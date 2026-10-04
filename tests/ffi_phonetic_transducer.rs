#![cfg(all(feature = "ffi", feature = "bindings-phonetic"))]

use liblevenshtein::ffi::{
    llev_owned_string_free, llev_phonetic_rules_free, llev_phonetic_rules_parse,
    llev_phonetic_transducer_feed, llev_phonetic_transducer_finish, llev_phonetic_transducer_free,
    llev_phonetic_transducer_new, llev_phonetic_transducer_normalize,
    llev_phonetic_transducer_reset, LlevOwnedString, LlevPhoneticTransducer, LlevStatus,
};
use liblevenshtein::phonetic::{
    llev::{parse_str, RuleSetChar},
    OnlinePhoneticTransducerChar,
};
use proptest::prelude::*;
use std::ptr;

// INVARIANT-HOOK: LLEV-PHON-STREAM-3 — rewrite output overflow resets state without publishing output.

unsafe fn take_text(value: &mut LlevOwnedString) -> String {
    let result = if value.len == 0 {
        String::new()
    } else {
        std::str::from_utf8(std::slice::from_raw_parts(value.data.cast(), value.len))
            .unwrap()
            .to_owned()
    };
    llev_owned_string_free(value);
    result
}

#[test]
fn incremental_and_whole_rewrites_match_public_rust() {
    let source = "ph -> f;";
    let mut rules = ptr::null_mut();
    assert_eq!(
        unsafe { llev_phonetic_rules_parse(source.as_ptr().cast(), source.len(), &mut rules) },
        LlevStatus::Ok
    );
    let mut ffi: *mut LlevPhoneticTransducer = ptr::null_mut();
    assert_eq!(
        unsafe { llev_phonetic_transducer_new(rules, &mut ffi) },
        LlevStatus::Ok
    );
    unsafe {
        llev_phonetic_rules_free(rules);
    }
    let mut emitted = String::new();
    let mut output = LlevOwnedString::default();
    for chunk in ["🦀 p", "hone"] {
        assert_eq!(
            unsafe {
                llev_phonetic_transducer_feed(
                    ffi,
                    chunk.as_ptr().cast(),
                    chunk.len(),
                    100,
                    100,
                    &mut output,
                )
            },
            LlevStatus::Ok
        );
        emitted.push_str(&unsafe { take_text(&mut output) });
    }
    assert_eq!(
        unsafe { llev_phonetic_transducer_finish(ffi, 100, &mut output) },
        LlevStatus::Ok
    );
    emitted.push_str(&unsafe { take_text(&mut output) });
    let expected = OnlinePhoneticTransducerChar::new(
        RuleSetChar::from_llev(&parse_str(source).unwrap())
            .unwrap()
            .rules,
    )
    .normalize("🦀 phone");
    assert_eq!(emitted, expected);
    assert_eq!(
        unsafe { llev_phonetic_transducer_finish(ffi, 100, &mut output) },
        LlevStatus::InvalidArgument
    );
    assert_eq!(
        unsafe {
            llev_phonetic_transducer_normalize(
                ffi,
                b"phone".as_ptr().cast(),
                5,
                100,
                100,
                &mut output,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(unsafe { take_text(&mut output) }, "fone");
    assert_eq!(
        unsafe { llev_phonetic_transducer_reset(ffi) },
        LlevStatus::Ok
    );
    assert_eq!(
        unsafe {
            llev_phonetic_transducer_feed(ffi, b"phone".as_ptr().cast(), 5, 100, 100, &mut output)
        },
        LlevStatus::Ok
    );
    let _ = unsafe { take_text(&mut output) };
    unsafe {
        llev_phonetic_transducer_free(ffi);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn generated_chunk_partitions_equal_whole_native_rewrite(
        input in "[a-z]{0,16}", split in 0usize..17,
    ) {
        let source = "ph -> f;";
        let native = OnlinePhoneticTransducerChar::new(
            RuleSetChar::from_llev(&parse_str(source).unwrap()).unwrap().rules)
            .normalize(&input);
        let mut rules = ptr::null_mut();
        prop_assert_eq!(unsafe { llev_phonetic_rules_parse(source.as_ptr().cast(),
            source.len(), &mut rules) }, LlevStatus::Ok);
        let mut transducer = ptr::null_mut();
        prop_assert_eq!(unsafe { llev_phonetic_transducer_new(rules, &mut transducer) }, LlevStatus::Ok);
        unsafe { llev_phonetic_rules_free(rules) };
        let boundary = split.min(input.len());
        let mut output = LlevOwnedString::default();
        let mut observed = String::new();
        for chunk in [&input[..boundary], &input[boundary..]] {
            prop_assert_eq!(unsafe { llev_phonetic_transducer_feed(transducer,
                chunk.as_ptr().cast(), chunk.len(), 100, 100, &mut output) }, LlevStatus::Ok);
            observed.push_str(&unsafe { take_text(&mut output) });
        }
        prop_assert_eq!(unsafe { llev_phonetic_transducer_finish(transducer, 100,
            &mut output) }, LlevStatus::Ok);
        observed.push_str(&unsafe { take_text(&mut output) });
        unsafe { llev_phonetic_transducer_free(transducer) };
        prop_assert_eq!(observed, native);
    }

    #[test]
    fn generated_output_overflow_is_atomic_and_resets(input in "[abc]{3,8}") {
        let mut transducer = ptr::null_mut();
        prop_assert_eq!(unsafe { llev_phonetic_transducer_new(ptr::null(),
            &mut transducer) }, LlevStatus::Ok);
        let mut output = LlevOwnedString { data: ptr::dangling_mut(), len: 999 };
        prop_assert_eq!(unsafe { llev_phonetic_transducer_feed(transducer,
            input.as_ptr().cast(), input.len(), 100, 1, &mut output) }, LlevStatus::LimitExceeded);
        prop_assert_eq!(output.data as usize, 1);
        prop_assert_eq!(output.len, 999);
        prop_assert_eq!(unsafe { llev_phonetic_transducer_feed(transducer,
            b"a".as_ptr().cast(), 1, 100, 100, &mut output) }, LlevStatus::Ok);
        let mut observed = unsafe { take_text(&mut output) };
        prop_assert_eq!(unsafe { llev_phonetic_transducer_finish(transducer,
            100, &mut output) }, LlevStatus::Ok);
        observed.push_str(&unsafe { take_text(&mut output) });
        prop_assert_eq!(observed, "a");
        unsafe { llev_phonetic_transducer_free(transducer) };
    }
}

#[test]
fn feed_limits_leave_outputs_unchanged_and_reset_on_output_overflow() {
    let mut ffi = ptr::null_mut();
    assert_eq!(
        unsafe { llev_phonetic_transducer_new(ptr::null(), &mut ffi) },
        LlevStatus::Ok
    );
    let mut output = LlevOwnedString {
        data: ptr::dangling_mut(),
        len: 999,
    };
    assert_eq!(
        unsafe {
            llev_phonetic_transducer_feed(ffi, b"ab".as_ptr().cast(), 2, 1, 100, &mut output)
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(output.len, 999);
    assert_eq!(
        unsafe {
            llev_phonetic_transducer_feed(ffi, b"abcdefgh".as_ptr().cast(), 8, 100, 1, &mut output)
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(output.len, 999);
    assert_eq!(
        unsafe {
            llev_phonetic_transducer_normalize(
                ffi,
                b"abcdefgh".as_ptr().cast(),
                8,
                100,
                1,
                &mut output,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(output.len, 999);
    unsafe {
        llev_phonetic_transducer_free(ffi);
    }
}
