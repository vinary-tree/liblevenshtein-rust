#![cfg(all(feature = "ffi", feature = "bindings-phonetic"))]

//! INVARIANT-HOOK: LLEV-PHON-AOT-1 — rejected AOT calls preserve outputs.
//! INVARIANT-HOOK: LLEV-PHON-AOT-2 — repeated owned-byte free is harmless.
//! INVARIANT-HOOK: LLEV-PHON-AOT-3 — decoded handles survive byte release.
//! INVARIANT-HOOK: LLEV-PHON-AOT-4 — disabled serialization is unsupported.
//! INVARIANT-HOOK: LLEV-PHON-AOT-5 — an owned byte buffer releases once.

use liblevenshtein::bindings::{PhoneticPattern, PhoneticRuleSet};
use liblevenshtein::ffi::*;
use proptest::prelude::*;
use std::ffi::c_char;
use std::ptr;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/phonetic_binding"
);

fn fixture(name: &str) -> String {
    format!("{FIXTURE}/{name}")
}

unsafe fn apply(rules: *const LlevPhoneticRuleSet, input: &str) -> String {
    let mut output = LlevOwnedString::default();
    assert_eq!(
        llev_phonetic_rules_apply(rules, input.as_ptr().cast(), input.len(), &mut output),
        LlevStatus::Ok
    );
    let result = if output.len == 0 {
        String::new()
    } else {
        String::from_utf8(std::slice::from_raw_parts(output.data.cast(), output.len).to_vec())
            .unwrap()
    };
    llev_owned_string_free(&mut output);
    result
}

unsafe fn matches(pattern: *const LlevPhoneticPattern, input: &str) -> bool {
    let mut output = 0;
    assert_eq!(
        llev_phonetic_pattern_matches(pattern, input.as_ptr().cast(), input.len(), &mut output),
        LlevStatus::Ok
    );
    output != 0
}

#[test]
fn file_loaders_resolve_includes_and_imports_like_native_rust() {
    let rule_path = fixture("rules.llev");
    let mut rules = ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_phonetic_rules_load_file(
                rule_path.as_ptr().cast(),
                rule_path.len(),
                ptr::null(),
                0,
                4096,
                &mut rules,
            )
        },
        LlevStatus::Ok
    );
    let native = PhoneticRuleSet::from_file(std::path::Path::new(&rule_path), &[]).unwrap();
    for word in ["phone", "café", "🦀 phone"] {
        assert_eq!(unsafe { apply(rules, word) }, native.apply(word));
    }
    unsafe { llev_phonetic_rules_free(rules) };

    let pattern_path = fixture("pattern.llre");
    let mut pattern = ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_phonetic_pattern_load_llre_file(
                pattern_path.as_ptr().cast(),
                pattern_path.len(),
                ptr::null(),
                0,
                4096,
                &mut pattern,
            )
        },
        LlevStatus::Ok
    );
    let native = PhoneticPattern::from_llre_file(std::path::Path::new(&pattern_path), &[]).unwrap();
    for word in ["phone", "café", "🦀"] {
        assert_eq!(unsafe { matches(pattern, word) }, native.matches(word));
    }
    unsafe { llev_phonetic_pattern_free(pattern) };
}

#[test]
fn file_loader_rejects_invalid_paths_transactionally() {
    let path = fixture("rules.llev");
    let mut rules = ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_phonetic_rules_load_file(
                path.as_ptr().cast(),
                path.len(),
                ptr::null(),
                0,
                2,
                &mut rules,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert!(rules.is_null());
    assert_eq!(
        unsafe {
            llev_phonetic_rules_load_file(
                path.as_ptr().cast(),
                path.len(),
                ptr::null(),
                65,
                4096,
                &mut rules,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert!(rules.is_null());
    let bad_utf8 = [0xffu8];
    assert_eq!(
        unsafe {
            llev_phonetic_rules_load_file(
                bad_utf8.as_ptr().cast(),
                1,
                ptr::null(),
                0,
                4096,
                &mut rules,
            )
        },
        LlevStatus::InvalidUtf8
    );
    assert!(rules.is_null());
    let missing = fixture("missing.llev");
    assert_eq!(
        unsafe {
            llev_phonetic_rules_load_file(
                missing.as_ptr().cast(),
                missing.len(),
                ptr::null(),
                0,
                4096,
                &mut rules,
            )
        },
        LlevStatus::InvalidArgument
    );
    assert!(rules.is_null());
    let view = LlevUtf8View {
        data: ptr::null::<c_char>(),
        len: 1,
    };
    assert_eq!(
        unsafe {
            llev_phonetic_rules_load_file(
                path.as_ptr().cast(),
                path.len(),
                &view,
                1,
                4096,
                &mut rules,
            )
        },
        LlevStatus::NullPointer
    );
    assert!(rules.is_null());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn generated_missing_local_paths_leave_rule_output_unchanged(name in "[a-z]{1,10}") {
        let path = format!("{FIXTURE}/missing-{name}.llev");
        let mut output = ptr::null_mut();
        prop_assert_eq!(unsafe { llev_phonetic_rules_load_file(path.as_ptr().cast(),
            path.len(), ptr::null(), 0, 4096, &mut output) }, LlevStatus::InvalidArgument);
        prop_assert!(output.is_null());
    }
}

#[cfg(feature = "serialization")]
#[test]
fn versioned_rule_and_pattern_bytes_roundtrip_exact_native_behavior() {
    assert_ne!(llev_build_features() & 4, 0);
    let rule_path = fixture("rules.llev");
    let mut rules = ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_phonetic_rules_load_file(
                rule_path.as_ptr().cast(),
                rule_path.len(),
                ptr::null(),
                0,
                4096,
                &mut rules,
            )
        },
        LlevStatus::Ok
    );
    let mut bytes = LlevOwnedBytes::default();
    assert_eq!(
        unsafe { llev_phonetic_rules_to_bytes(rules, 16 * 1024 * 1024, &mut bytes) },
        LlevStatus::Ok
    );
    let native_bytes = PhoneticRuleSet::from_file(std::path::Path::new(&rule_path), &[])
        .unwrap()
        .to_compiled_bytes()
        .unwrap();
    assert_eq!(
        unsafe { std::slice::from_raw_parts(bytes.data, bytes.len) },
        native_bytes
    );
    let mut restored = ptr::null_mut();
    assert_eq!(
        unsafe { llev_phonetic_rules_from_bytes(bytes.data, bytes.len, bytes.len, &mut restored) },
        LlevStatus::Ok
    );
    unsafe { llev_owned_bytes_free(&mut bytes) };
    assert!(bytes.data.is_null());
    unsafe { llev_owned_bytes_free(&mut bytes) };
    assert_eq!(bytes.len, 0);
    assert_eq!(unsafe { apply(restored, "phone") }, "fone");
    unsafe {
        llev_phonetic_rules_free(restored);
        llev_phonetic_rules_free(rules);
    }

    let pattern_path = fixture("pattern.llre");
    let mut pattern = ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_phonetic_pattern_load_llre_file(
                pattern_path.as_ptr().cast(),
                pattern_path.len(),
                ptr::null(),
                0,
                4096,
                &mut pattern,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(
        unsafe { llev_phonetic_pattern_to_bytes(pattern, 16 * 1024 * 1024, &mut bytes) },
        LlevStatus::Ok
    );
    let native_bytes = PhoneticPattern::from_llre_file(std::path::Path::new(&pattern_path), &[])
        .unwrap()
        .to_compiled_bytes()
        .unwrap();
    assert_eq!(
        unsafe { std::slice::from_raw_parts(bytes.data, bytes.len) },
        native_bytes
    );
    let mut restored = ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_phonetic_pattern_from_bytes(bytes.data, bytes.len, bytes.len, &mut restored)
        },
        LlevStatus::Ok
    );
    unsafe { llev_owned_bytes_free(&mut bytes) };
    assert!(unsafe { matches(restored, "phone") });
    assert!(!unsafe { matches(restored, "café") });
    unsafe {
        llev_phonetic_pattern_free(restored);
        llev_phonetic_pattern_free(pattern);
    }
}

#[cfg(feature = "serialization")]
#[test]
fn aot_limits_corruption_and_wrong_version_are_transactional() {
    let mut rules = ptr::null_mut();
    let source = "ph -> f;";
    assert_eq!(
        unsafe { llev_phonetic_rules_parse(source.as_ptr().cast(), source.len(), &mut rules) },
        LlevStatus::Ok
    );
    let mut bytes = LlevOwnedBytes::default();
    assert_eq!(
        unsafe { llev_phonetic_rules_to_bytes(rules, 1, &mut bytes) },
        LlevStatus::LimitExceeded
    );
    assert!(bytes.data.is_null());
    assert_eq!(
        unsafe { llev_phonetic_rules_to_bytes(rules, 16 * 1024 * 1024 + 1, &mut bytes) },
        LlevStatus::LimitExceeded
    );
    assert_eq!(
        unsafe { llev_phonetic_rules_to_bytes(rules, 16 * 1024 * 1024, &mut bytes) },
        LlevStatus::Ok
    );
    let mut output = ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_phonetic_rules_from_bytes(bytes.data, bytes.len, bytes.len - 1, &mut output)
        },
        LlevStatus::LimitExceeded
    );
    assert!(output.is_null());
    let mut corrupted = unsafe { std::slice::from_raw_parts(bytes.data, bytes.len) }.to_vec();
    corrupted[0] ^= 0xff;
    assert_eq!(
        unsafe {
            llev_phonetic_rules_from_bytes(
                corrupted.as_ptr(),
                corrupted.len(),
                corrupted.len(),
                &mut output,
            )
        },
        LlevStatus::InvalidArgument
    );
    assert!(output.is_null());
    let mut wrong_version = unsafe { std::slice::from_raw_parts(bytes.data, bytes.len) }.to_vec();
    wrong_version[4] ^= 0xff;
    assert_eq!(
        unsafe {
            llev_phonetic_rules_from_bytes(
                wrong_version.as_ptr(),
                wrong_version.len(),
                wrong_version.len(),
                &mut output,
            )
        },
        LlevStatus::InvalidArgument
    );
    assert!(output.is_null());
    unsafe {
        llev_owned_bytes_free(&mut bytes);
        llev_phonetic_rules_free(rules);
    }
}

#[cfg(feature = "serialization")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn generated_aot_roundtrip_survives_repeated_owned_byte_release(input in "[a-z]{0,8}") {
        let source = "ph -> f;";
        let native = PhoneticRuleSet::parse(source).unwrap();
        let mut rules = ptr::null_mut();
        prop_assert_eq!(unsafe { llev_phonetic_rules_parse(source.as_ptr().cast(),
            source.len(), &mut rules) }, LlevStatus::Ok);
        let mut bytes = LlevOwnedBytes::default();
        prop_assert_eq!(unsafe { llev_phonetic_rules_to_bytes(rules,
            16 * 1024 * 1024, &mut bytes) }, LlevStatus::Ok);
        let mut restored = ptr::null_mut();
        prop_assert_eq!(unsafe { llev_phonetic_rules_from_bytes(bytes.data,
            bytes.len, bytes.len, &mut restored) }, LlevStatus::Ok);
        unsafe { llev_owned_bytes_free(&mut bytes) };
        prop_assert!(bytes.data.is_null());
        prop_assert_eq!(bytes.len, 0);
        unsafe { llev_owned_bytes_free(&mut bytes) };
        prop_assert!(bytes.data.is_null());
        prop_assert_eq!(bytes.len, 0);
        prop_assert_eq!(unsafe { apply(restored, &input) }, native.apply(&input));
        unsafe { llev_phonetic_rules_free(restored); llev_phonetic_rules_free(rules); }
    }

    #[test]
    fn generated_header_corruption_never_returns_a_partial_rule_handle(
        position in 0usize..5, delta in 1u8..=255,
    ) {
        let source = "ph -> f;";
        let mut rules = ptr::null_mut();
        prop_assert_eq!(unsafe { llev_phonetic_rules_parse(source.as_ptr().cast(),
            source.len(), &mut rules) }, LlevStatus::Ok);
        let mut bytes = LlevOwnedBytes::default();
        prop_assert_eq!(unsafe { llev_phonetic_rules_to_bytes(rules,
            16 * 1024 * 1024, &mut bytes) }, LlevStatus::Ok);
        let mut corrupt = unsafe { std::slice::from_raw_parts(bytes.data, bytes.len) }.to_vec();
        corrupt[position] ^= delta;
        let mut restored = 1usize as *mut LlevPhoneticRuleSet;
        let status = unsafe { llev_phonetic_rules_from_bytes(corrupt.as_ptr(),
            corrupt.len(), corrupt.len(), &mut restored) };
        unsafe { llev_owned_bytes_free(&mut bytes); llev_phonetic_rules_free(rules); }
        prop_assert_eq!(status, LlevStatus::InvalidArgument);
        prop_assert_eq!(restored as usize, 1);
    }
}

#[cfg(not(feature = "serialization"))]
#[test]
fn aot_symbols_return_unsupported_without_serialization() {
    assert_eq!(llev_build_features() & 4, 0);
    let mut bytes = LlevOwnedBytes::default();
    assert_eq!(
        unsafe { llev_phonetic_rules_to_bytes(ptr::null(), 4096, &mut bytes) },
        LlevStatus::Unsupported
    );
    assert!(bytes.data.is_null());
    let mut rules = ptr::null_mut();
    assert_eq!(
        unsafe { llev_phonetic_rules_from_bytes(ptr::null(), 0, 4096, &mut rules) },
        LlevStatus::Unsupported
    );
    assert!(rules.is_null());
}

#[cfg(not(feature = "serialization"))]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn generated_disabled_aot_never_publishes_bytes_or_handles(
        input in proptest::collection::vec(any::<u8>(), 0..32),
        max_bytes in 1usize..1024,
    ) {
        prop_assert_eq!(llev_build_features() & 4, 0);
        let mut bytes = LlevOwnedBytes { data: 1usize as *mut u8, len: 999 };
        prop_assert_eq!(unsafe { llev_phonetic_rules_to_bytes(ptr::null(),
            max_bytes, &mut bytes) }, LlevStatus::Unsupported);
        prop_assert_eq!(bytes.data as usize, 1);
        prop_assert_eq!(bytes.len, 999);
        prop_assert_eq!(unsafe { llev_phonetic_pattern_to_bytes(ptr::null(),
            max_bytes, &mut bytes) }, LlevStatus::Unsupported);
        prop_assert_eq!(bytes.data as usize, 1);
        prop_assert_eq!(bytes.len, 999);
        let mut rules = 1usize as *mut LlevPhoneticRuleSet;
        prop_assert_eq!(unsafe { llev_phonetic_rules_from_bytes(input.as_ptr(),
            input.len(), max_bytes, &mut rules) }, LlevStatus::Unsupported);
        prop_assert_eq!(rules as usize, 1);
        let mut pattern = 1usize as *mut LlevPhoneticPattern;
        prop_assert_eq!(unsafe { llev_phonetic_pattern_from_bytes(input.as_ptr(),
            input.len(), max_bytes, &mut pattern) }, LlevStatus::Unsupported);
        prop_assert_eq!(pattern as usize, 1);
    }
}
