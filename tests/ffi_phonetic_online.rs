#![cfg(all(feature = "ffi", feature = "bindings-phonetic"))]

use liblevenshtein::ffi::{
    llev_owned_string_free, llev_phonetic_online_free, llev_phonetic_online_matches_free,
    llev_phonetic_online_new, llev_phonetic_online_normalized_query, llev_phonetic_online_scan,
    llev_phonetic_online_stream_feed, llev_phonetic_online_stream_finish,
    llev_phonetic_online_stream_free, llev_phonetic_online_stream_new, llev_phonetic_rules_free,
    llev_phonetic_rules_parse, LlevOwnedString, LlevPhoneticOnlineGrep, LlevPhoneticOnlineMatch,
    LlevPhoneticOnlineStream, LlevPhoneticRuleSet, LlevStatus,
};
use liblevenshtein::phonetic::{PhoneticGrepOnline, ScanMatch};
use proptest::prelude::*;
use std::ptr;

// INVARIANT-HOOK: LLEV-PHON-STREAM-1 — scanner accepted feeds stay within the byte ceiling.
// INVARIANT-HOOK: LLEV-PHON-STREAM-2 — finish is single-use, including result-limit rejection.
// INVARIANT-HOOK: LLEV-PHON-STREAM-4 — a rejected scanner feed emits nothing.
// INVARIANT-HOOK: LLEV-PHON-BOUND-5 — successful scans never exceed the result capacity.

unsafe fn text(value: LlevOwnedString) -> String {
    if value.len == 0 {
        return String::new();
    }
    std::str::from_utf8(std::slice::from_raw_parts(value.data.cast(), value.len))
        .unwrap()
        .to_owned()
}

unsafe fn result(matches: *mut LlevPhoneticOnlineMatch, count: usize) -> Vec<ScanMatch> {
    let values = if count == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(matches, count)
    };
    let result = values
        .iter()
        .map(|item| ScanMatch {
            byte_range: (item.byte_start, item.byte_end),
            char_range: (item.char_start, item.char_end),
            original_text: text(item.original_text),
            normalized_text: text(item.normalized_text),
            distance: item.distance,
        })
        .collect();
    llev_phonetic_online_matches_free(matches, count);
    result
}

unsafe fn new(pattern: &str, rules: *const LlevPhoneticRuleSet) -> *mut LlevPhoneticOnlineGrep {
    let mut output = ptr::null_mut();
    assert_eq!(
        llev_phonetic_online_new(
            pattern.as_ptr().cast(),
            pattern.len(),
            rules,
            0,
            0,
            100,
            &mut output
        ),
        LlevStatus::Ok
    );
    output
}

#[test]
fn online_scan_and_stream_match_public_rust_exactly() {
    let native = PhoneticGrepOnline::without_rules("café", 0);
    let ffi = unsafe { new("café", ptr::null()) };
    for document in ["café", "🦀 café café", "unrelated", ""] {
        let mut matches = ptr::null_mut();
        let mut count = 0;
        assert_eq!(
            unsafe {
                llev_phonetic_online_scan(
                    ffi,
                    document.as_ptr().cast(),
                    document.len(),
                    100,
                    100,
                    &mut matches,
                    &mut count,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(unsafe { result(matches, count) }, native.scan(document));
    }
    let mut normalized = LlevOwnedString::default();
    assert_eq!(
        unsafe { llev_phonetic_online_normalized_query(ffi, &mut normalized) },
        LlevStatus::Ok
    );
    assert_eq!(unsafe { text(normalized) }, native.normalized_query());
    unsafe {
        llev_owned_string_free(&mut normalized);
    }

    let mut stream: *mut LlevPhoneticOnlineStream = ptr::null_mut();
    assert_eq!(
        unsafe { llev_phonetic_online_stream_new(ffi, 100, &mut stream) },
        LlevStatus::Ok
    );
    unsafe {
        llev_phonetic_online_free(ffi);
    }
    for chunk in ["🦀 ca", "fé ca", "fé"] {
        assert_eq!(
            unsafe { llev_phonetic_online_stream_feed(stream, chunk.as_ptr().cast(), chunk.len()) },
            LlevStatus::Ok
        );
    }
    let mut matches = ptr::null_mut();
    let mut count = 0;
    assert_eq!(
        unsafe { llev_phonetic_online_stream_finish(stream, 100, &mut matches, &mut count) },
        LlevStatus::Ok
    );
    assert_eq!(
        unsafe { result(matches, count) },
        native.scan("🦀 café café")
    );
    assert_eq!(
        unsafe { llev_phonetic_online_stream_finish(stream, 100, &mut matches, &mut count) },
        LlevStatus::InvalidArgument
    );
    unsafe {
        llev_phonetic_online_stream_free(stream);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn generated_scan_and_chunked_stream_equal_native(
        pattern in "[a-z]{1,4}",
        document in "[a-z ]{0,12}",
        split in 0usize..13,
    ) {
        let native = PhoneticGrepOnline::without_rules(&pattern, 0);
        let ffi = unsafe { new(&pattern, ptr::null()) };
        let mut matches = ptr::null_mut();
        let mut count = 0;
        prop_assert_eq!(unsafe { llev_phonetic_online_scan(ffi,
            document.as_ptr().cast(), document.len(), 100, 100,
            &mut matches, &mut count) }, LlevStatus::Ok);
        let observed = unsafe { result(matches, count) };
        prop_assert_eq!(observed, native.scan(&document));
        let mut stream = ptr::null_mut();
        prop_assert_eq!(unsafe { llev_phonetic_online_stream_new(ffi, 100, &mut stream) }, LlevStatus::Ok);
        let boundary = split.min(document.len());
        for chunk in [&document[..boundary], &document[boundary..]] {
            prop_assert_eq!(unsafe { llev_phonetic_online_stream_feed(stream,
                chunk.as_ptr().cast(), chunk.len()) }, LlevStatus::Ok);
        }
        matches = ptr::null_mut();
        count = 0;
        prop_assert_eq!(unsafe { llev_phonetic_online_stream_finish(stream, 100,
            &mut matches, &mut count) }, LlevStatus::Ok);
        let streamed = unsafe { result(matches, count) };
        prop_assert_eq!(streamed, native.scan(&document));
        unsafe { llev_phonetic_online_stream_free(stream); llev_phonetic_online_free(ffi); }
    }

    #[test]
    fn generated_rejected_feed_preserves_scanner_prefix(document in "[a-z]{2,12}") {
        let ffi = unsafe { new("a", ptr::null()) };
        let native = PhoneticGrepOnline::without_rules("a", 0);
        let mut stream = ptr::null_mut();
        prop_assert_eq!(unsafe { llev_phonetic_online_stream_new(ffi, document.len() - 1,
            &mut stream) }, LlevStatus::Ok);
        prop_assert_eq!(unsafe { llev_phonetic_online_stream_feed(stream,
            document.as_ptr().cast(), document.len()) }, LlevStatus::LimitExceeded);
        let prefix = &document[..document.len() - 1];
        prop_assert_eq!(unsafe { llev_phonetic_online_stream_feed(stream,
            prefix.as_ptr().cast(), prefix.len()) }, LlevStatus::Ok);
        let mut matches = ptr::null_mut();
        let mut count = 0;
        prop_assert_eq!(unsafe { llev_phonetic_online_stream_finish(stream, 100,
            &mut matches, &mut count) }, LlevStatus::Ok);
        prop_assert_eq!(unsafe { result(matches, count) }, native.scan(prefix));
        unsafe { llev_phonetic_online_stream_free(stream); llev_phonetic_online_free(ffi); }
    }
}

#[test]
fn online_rules_and_limits_preserve_output_transactionally() {
    let mut rules = ptr::null_mut();
    let source = "ph -> f;";
    assert_eq!(
        unsafe { llev_phonetic_rules_parse(source.as_ptr().cast(), source.len(), &mut rules) },
        LlevStatus::Ok
    );
    let grep = unsafe { new("phone", rules) };
    unsafe {
        llev_phonetic_rules_free(rules);
    }
    let mut normalized = LlevOwnedString::default();
    assert_eq!(
        unsafe { llev_phonetic_online_normalized_query(grep, &mut normalized) },
        LlevStatus::Ok
    );
    assert_eq!(unsafe { text(normalized) }, "fone");
    unsafe {
        llev_owned_string_free(&mut normalized);
    }
    let mut matches = 1usize as *mut LlevPhoneticOnlineMatch;
    let mut count = 999;
    assert_eq!(
        unsafe {
            llev_phonetic_online_scan(
                grep,
                b"fone phone".as_ptr().cast(),
                10,
                9,
                100,
                &mut matches,
                &mut count,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(matches as usize, 1);
    assert_eq!(count, 999);
    assert_eq!(
        unsafe {
            llev_phonetic_online_scan(
                grep,
                b"fone phone".as_ptr().cast(),
                10,
                100,
                1,
                &mut matches,
                &mut count,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(matches as usize, 1);
    assert_eq!(count, 999);
    let mut stream = ptr::null_mut();
    assert_eq!(
        unsafe { llev_phonetic_online_stream_new(grep, 4, &mut stream) },
        LlevStatus::Ok
    );
    assert_eq!(
        unsafe { llev_phonetic_online_stream_feed(stream, b"fone".as_ptr().cast(), 4) },
        LlevStatus::Ok
    );
    assert_eq!(
        unsafe { llev_phonetic_online_stream_feed(stream, b"x".as_ptr().cast(), 1) },
        LlevStatus::LimitExceeded
    );
    assert_eq!(
        unsafe { llev_phonetic_online_stream_finish(stream, 100, &mut matches, &mut count) },
        LlevStatus::Ok
    );
    assert_eq!(unsafe { result(matches, count) }.len(), 1);
    unsafe {
        llev_phonetic_online_stream_free(stream);
        llev_phonetic_online_free(grep);
    }
}
