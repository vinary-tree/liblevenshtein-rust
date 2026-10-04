#![cfg(all(feature = "ffi", feature = "bindings-phonetic"))]

use liblevenshtein::ffi::{
    llev_phonetic_articulatory_distance, llev_phonetic_articulatory_edit_distance,
    llev_phonetic_grep_distance_config, llev_phonetic_grep_free, llev_phonetic_grep_matches,
    llev_phonetic_grep_new, llev_phonetic_grep_scan_line, llev_phonetic_grep_scan_text,
    llev_phonetic_rules_free, llev_phonetic_rules_parse, llev_phonetic_syllable_boundaries,
    llev_phonetic_syllable_count, LlevAlgorithm, LlevPhoneticFeatureWeights, LlevPhoneticGrep,
    LlevPhoneticGrepMatch, LlevPhoneticRuleSet, LlevStatus,
};
use liblevenshtein::phonetic::{
    articulatory_distance, articulatory_edit_distance, ipa_syllable_boundaries, ipa_syllable_count,
    syllable_boundaries, syllable_count,
};
use proptest::prelude::*;
use std::ptr;

// INVARIANT-HOOK: LLEV-PHON-BOUND-2 — a capacity failure does not publish partial matches.
// INVARIANT-HOOK: LLEV-PHON-BOUND-3 — word grep reports required capacity on limit.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn generated_articulatory_and_syllable_results_equal_native(
        source in "[a-z]{0,10}", target in "[a-z]{0,10}",
    ) {
        let mut distance = -1.0;
        prop_assert_eq!(unsafe { llev_phonetic_articulatory_edit_distance(
            source.as_ptr().cast(), source.len(), target.as_ptr().cast(),
            target.len(), ptr::null(), 100, &mut distance) }, LlevStatus::Ok);
        prop_assert_eq!(distance, articulatory_edit_distance(&source, &target));
        let mut count = usize::MAX;
        prop_assert_eq!(unsafe { llev_phonetic_syllable_count(source.as_ptr().cast(),
            source.len(), 0, 100, &mut count) }, LlevStatus::Ok);
        prop_assert_eq!(count, syllable_count(&source));
    }

    #[test]
    fn generated_word_grep_membership_equals_native(
        pattern in "[a-z]{1,4}", candidate in "[a-z]{0,8}",
    ) {
        let mut handle = ptr::null_mut();
        prop_assert_eq!(unsafe { llev_phonetic_grep_new(pattern.as_ptr().cast(),
            pattern.len(), ptr::null(), 1, LlevAlgorithm::Standard as u32, 0,
            &mut handle) }, LlevStatus::Ok);
        let native = liblevenshtein::phonetic::PhoneticGrep::from_pattern(&pattern, 1).unwrap();
        let mut distance = 0;
        let mut found = 0;
        prop_assert_eq!(unsafe { llev_phonetic_grep_matches(handle,
            candidate.as_ptr().cast(), candidate.len(), 100,
            &mut distance, &mut found) }, LlevStatus::Ok);
        unsafe { llev_phonetic_grep_free(handle) };
        let expected = native.matches(&candidate);
        prop_assert_eq!(found != 0, expected.is_some());
        prop_assert_eq!(distance, expected.unwrap_or(0));
    }

    #[test]
    fn generated_word_capacity_reports_exact_required_count_without_partial_payload(
        pattern in "[a-z]{1,4}",
    ) {
        let mut handle = ptr::null_mut();
        prop_assert_eq!(unsafe { llev_phonetic_grep_new(pattern.as_ptr().cast(),
            pattern.len(), ptr::null(), 0, LlevAlgorithm::Standard as u32, 0,
            &mut handle) }, LlevStatus::Ok);
        let document = format!("{pattern} {pattern}");
        let mut item = LlevPhoneticGrepMatch { line_number: 97,
            start_byte: 97, end_byte: 97, distance: 97, reserved: [97; 7] };
        let mut count = 999;
        prop_assert_eq!(unsafe { llev_phonetic_grep_scan_line(handle,
            document.as_ptr().cast(), document.len(), 100, &mut item, 1,
            &mut count) }, LlevStatus::LimitExceeded);
        unsafe { llev_phonetic_grep_free(handle) };
        prop_assert_eq!(count, 2);
        prop_assert_eq!(item.line_number, 97);
        prop_assert_eq!(item.start_byte, 97);
    }
}

#[test]
fn articulatory_scalar_boundary_matches_public_rust() {
    for source in ['p', 'b', 'ɲ', 'a', 'é', '🦀'] {
        for target in ['p', 't', 'ɲ', 'i', '🦀'] {
            let mut output = -1.0;
            let status = unsafe {
                llev_phonetic_articulatory_distance(
                    source as u32,
                    target as u32,
                    ptr::null(),
                    &mut output,
                )
            };
            assert_eq!(status, LlevStatus::Ok);
            assert_eq!(output, articulatory_distance(source, target));
        }
    }
    let mut unchanged = 42.0;
    assert_eq!(
        unsafe {
            llev_phonetic_articulatory_distance(0xd800, 'a' as u32, ptr::null(), &mut unchanged)
        },
        LlevStatus::InvalidArgument
    );
    assert_eq!(unchanged, 42.0);
}

#[test]
fn weighted_articulatory_boundary_preserves_native_costs() {
    let weights = LlevPhoneticFeatureWeights {
        voicing: 0.35,
        place_step: 0.25,
        manner_default: 0.75,
        manner_table_scale: 0.5,
        vowel_height_step: 0.2,
        vowel_backness_step: 0.3,
        vowel_rounding: 0.4,
    };
    let mut output = -1.0;
    assert_eq!(
        unsafe {
            llev_phonetic_articulatory_distance('p' as u32, 'b' as u32, &weights, &mut output)
        },
        LlevStatus::Ok
    );
    assert_eq!(
        output,
        liblevenshtein::phonetic::feature_distance::articulatory_distance_weighted(
            'p',
            'b',
            &liblevenshtein::phonetic::feature_distance::FeatureDistanceWeights {
                voicing: weights.voicing,
                place_step: weights.place_step,
                manner_default: weights.manner_default,
                manner_table_scale: weights.manner_table_scale,
                vowel_height_step: weights.vowel_height_step,
                vowel_backness_step: weights.vowel_backness_step,
                vowel_rounding: weights.vowel_rounding,
            },
        )
    );
    let bad = LlevPhoneticFeatureWeights {
        voicing: f64::NAN,
        ..weights
    };
    assert_eq!(
        unsafe { llev_phonetic_articulatory_distance('p' as u32, 'b' as u32, &bad, &mut output) },
        LlevStatus::InvalidArgument
    );
}

#[test]
fn articulatory_edit_boundary_matches_public_rust_and_limits_work() {
    for source in ["", "phone", "café", "🦀a"] {
        for target in ["", "fone", "cafe", "🦀b"] {
            let mut output = -1.0;
            assert_eq!(
                unsafe {
                    llev_phonetic_articulatory_edit_distance(
                        source.as_ptr().cast(),
                        source.len(),
                        target.as_ptr().cast(),
                        target.len(),
                        ptr::null(),
                        256,
                        &mut output,
                    )
                },
                LlevStatus::Ok
            );
            assert_eq!(output, articulatory_edit_distance(source, target));
        }
    }
    let mut unchanged = 42.0;
    assert_eq!(
        unsafe {
            llev_phonetic_articulatory_edit_distance(
                b"abc".as_ptr().cast(),
                3,
                b"def".as_ptr().cast(),
                3,
                ptr::null(),
                8,
                &mut unchanged,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(unchanged, 42.0);
    assert_eq!(
        unsafe {
            llev_phonetic_articulatory_edit_distance(
                [0xff].as_ptr().cast(),
                1,
                ptr::null(),
                0,
                ptr::null(),
                8,
                &mut unchanged,
            )
        },
        LlevStatus::InvalidUtf8
    );
}

#[test]
fn syllable_boundaries_and_counts_match_native_for_both_modes() {
    for word in ["", "cat", "happy", "beautiful", "café", "kæt", "ˈhæp.i"] {
        for (mode, count, positions) in [
            (0, syllable_count(word), syllable_boundaries(word)),
            (1, ipa_syllable_count(word), ipa_syllable_boundaries(word)),
        ] {
            let mut output_count = usize::MAX;
            assert_eq!(
                unsafe {
                    llev_phonetic_syllable_count(
                        word.as_ptr().cast(),
                        word.len(),
                        mode,
                        100,
                        &mut output_count,
                    )
                },
                LlevStatus::Ok
            );
            assert_eq!(output_count, count);
            let mut output = vec![usize::MAX; word.chars().count()];
            assert_eq!(
                unsafe {
                    llev_phonetic_syllable_boundaries(
                        word.as_ptr().cast(),
                        word.len(),
                        mode,
                        100,
                        output.as_mut_ptr(),
                        output.len(),
                        &mut output_count,
                    )
                },
                LlevStatus::Ok
            );
            assert_eq!(output_count, positions.len());
            assert_eq!(&output[..output_count], positions);
        }
    }
}

#[test]
fn syllable_capacity_and_invalid_selector_do_not_partially_write() {
    let mut positions = [usize::MAX; 1];
    let mut count = 0;
    assert_eq!(
        unsafe {
            llev_phonetic_syllable_boundaries(
                b"happy".as_ptr().cast(),
                5,
                0,
                100,
                positions.as_mut_ptr(),
                1,
                &mut count,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(count, syllable_boundaries("happy").len());
    assert_eq!(positions, [usize::MAX]);
    assert_eq!(
        unsafe { llev_phonetic_syllable_count(b"cat".as_ptr().cast(), 3, 2, 100, &mut count) },
        LlevStatus::InvalidArgument
    );
    assert_eq!(
        unsafe { llev_phonetic_syllable_count(b"cat".as_ptr().cast(), 3, 0, 2, &mut count) },
        LlevStatus::LimitExceeded
    );
}

#[test]
fn phonetic_grep_matches_public_rust_and_reports_byte_columns() {
    let mut handle: *mut LlevPhoneticGrep = ptr::null_mut();
    let pattern = b"phone";
    assert_eq!(
        unsafe {
            llev_phonetic_grep_new(
                pattern.as_ptr().cast(),
                pattern.len(),
                ptr::null(),
                1,
                LlevAlgorithm::Standard as u32,
                0,
                &mut handle,
            )
        },
        LlevStatus::Ok
    );
    assert!(!handle.is_null());
    let rust = liblevenshtein::phonetic::PhoneticGrep::from_pattern("phone", 1).unwrap();
    for candidate in ["phone", "phon", "tablet", "café"] {
        let mut distance = u8::MAX;
        let mut found = 7;
        assert_eq!(
            unsafe {
                llev_phonetic_grep_matches(
                    handle,
                    candidate.as_ptr().cast(),
                    candidate.len(),
                    100,
                    &mut distance,
                    &mut found,
                )
            },
            LlevStatus::Ok
        );
        let expected = rust.matches(candidate);
        assert_eq!(found, u8::from(expected.is_some()));
        assert_eq!(distance, expected.unwrap_or(0));
    }
    let text = "exact phone\nnear phon\nunrelated tablet";
    let mut output = vec![
        LlevPhoneticGrepMatch {
            line_number: 0,
            start_byte: 0,
            end_byte: 0,
            distance: 0,
            reserved: [0; 7],
        };
        8
    ];
    let mut count = 0;
    assert_eq!(
        unsafe {
            llev_phonetic_grep_scan_text(
                handle,
                text.as_ptr().cast(),
                text.len(),
                100,
                output.as_mut_ptr(),
                output.len(),
                &mut count,
            )
        },
        LlevStatus::Ok
    );
    let expected: Vec<_> = rust.grep_file(text).collect();
    assert_eq!(count, expected.iter().map(|line| line.matches.len()).sum());
    assert_eq!(
        (
            output[0].line_number,
            output[0].start_byte,
            output[0].end_byte
        ),
        (1, 6, 11)
    );
    assert_eq!(
        (
            output[1].line_number,
            output[1].start_byte,
            output[1].end_byte
        ),
        (2, 5, 9)
    );
    assert_eq!((output[0].distance, output[1].distance), (0, 1));
    unsafe { llev_phonetic_grep_free(handle) };
}

#[test]
fn phonetic_grep_owns_rules_and_enforces_transactional_capacity() {
    let mut rules: *mut LlevPhoneticRuleSet = ptr::null_mut();
    let source = b"ph -> f;";
    assert_eq!(
        unsafe { llev_phonetic_rules_parse(source.as_ptr().cast(), source.len(), &mut rules) },
        LlevStatus::Ok
    );
    let mut handle: *mut LlevPhoneticGrep = ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_phonetic_grep_new(
                b"fone".as_ptr().cast(),
                4,
                rules,
                0,
                LlevAlgorithm::Standard as u32,
                0,
                &mut handle,
            )
        },
        LlevStatus::Ok
    );
    unsafe { llev_phonetic_rules_free(rules) };
    let mut effective = u8::MAX;
    let mut local = u8::MAX;
    let mut has_local = u8::MAX;
    assert_eq!(
        unsafe {
            llev_phonetic_grep_distance_config(handle, &mut effective, &mut local, &mut has_local)
        },
        LlevStatus::Ok
    );
    assert_eq!((effective, local, has_local), (0, 0, 0));
    let mut distance = 7;
    let mut found = 7;
    assert_eq!(
        unsafe {
            llev_phonetic_grep_matches(
                handle,
                b"phone".as_ptr().cast(),
                5,
                100,
                &mut distance,
                &mut found,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!((found, distance), (1, 0));
    let mut item = LlevPhoneticGrepMatch {
        line_number: 97,
        start_byte: 97,
        end_byte: 97,
        distance: 97,
        reserved: [97; 7],
    };
    let mut count = 0;
    assert_eq!(
        unsafe {
            llev_phonetic_grep_scan_line(
                handle,
                b"phone phone".as_ptr().cast(),
                11,
                100,
                &mut item,
                1,
                &mut count,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(count, 2);
    assert_eq!(item.line_number, 97);
    unsafe { llev_phonetic_grep_free(handle) };
}
