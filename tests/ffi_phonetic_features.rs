#![cfg(all(feature = "ffi", feature = "bindings-phonetic"))]

use liblevenshtein::ffi::{
    llev_phonetic_chars_with_features, llev_phonetic_expand_feature_based,
    llev_phonetic_feature_relation, llev_phonetic_feature_set_distance, llev_phonetic_features,
    llev_phonetic_similar_chars, llev_phonetic_voicing_pair, LlevStatus,
};
use liblevenshtein::phonetic::{
    are_similar, chars_with_features, expand_feature_based, get_features, get_similar_chars,
    get_voicing_pair, is_free_substitution, PhoneticFeature,
};
use proptest::prelude::*;
use std::ptr;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn generated_ipa_relations_equal_public_native(
        source in proptest::sample::select(vec!['p', 'b', 't', 'd', 'a', 'i', 'ʃ', '🦀']),
        target in proptest::sample::select(vec!['p', 'b', 't', 'd', 'a', 'i', 'ʃ', '🦀']),
    ) {
        let mut feature_bits = u64::MAX;
        prop_assert_eq!(unsafe { llev_phonetic_features(source as u32,
            &mut feature_bits) }, LlevStatus::Ok);
        prop_assert_eq!(feature_bits.count_ones() as usize, get_features(source).len());
        for (selector, expected) in [(0, are_similar(source, target)),
            (1, is_free_substitution(source, target))] {
            let mut actual = 2;
            prop_assert_eq!(unsafe { llev_phonetic_feature_relation(source as u32,
                target as u32, selector, &mut actual) }, LlevStatus::Ok);
            prop_assert_eq!(actual != 0, expected);
        }
    }
}

unsafe fn copy_chars(symbol: u8, character_or_mask: u64) -> Vec<char> {
    let mut count = 0;
    let status = match symbol {
        0 => {
            llev_phonetic_chars_with_features(character_or_mask, 0, ptr::null_mut(), 0, &mut count)
        }
        1 => llev_phonetic_similar_chars(character_or_mask as u32, ptr::null_mut(), 0, &mut count),
        _ => llev_phonetic_expand_feature_based(
            character_or_mask as u32,
            ptr::null_mut(),
            0,
            &mut count,
        ),
    };
    assert!(matches!(status, LlevStatus::Ok | LlevStatus::LimitExceeded));
    let mut output = vec![0u32; count];
    let status = match symbol {
        0 => llev_phonetic_chars_with_features(
            character_or_mask,
            0,
            output.as_mut_ptr(),
            output.len(),
            &mut count,
        ),
        1 => llev_phonetic_similar_chars(
            character_or_mask as u32,
            output.as_mut_ptr(),
            output.len(),
            &mut count,
        ),
        _ => llev_phonetic_expand_feature_based(
            character_or_mask as u32,
            output.as_mut_ptr(),
            output.len(),
            &mut count,
        ),
    };
    assert_eq!(status, LlevStatus::Ok);
    output
        .into_iter()
        .map(|value| char::from_u32(value).unwrap())
        .collect()
}

#[test]
fn scalar_features_relations_and_expansions_match_native() {
    let voiced = 1u64 << 0;
    let stop = 1u64 << 2;
    let bilabial = 1u64 << 9;
    for (character, mask) in [
        ('b', voiced | stop | bilabial),
        ('p', stop | bilabial),
        ('🦀', 0),
    ] {
        let mut observed = u64::MAX;
        assert_eq!(
            unsafe { llev_phonetic_features(character as u32, &mut observed) },
            LlevStatus::Ok
        );
        assert_eq!(observed & mask, mask);
        if character == '🦀' {
            assert_eq!(observed, 0);
        }
        assert_eq!(observed == 0, get_features(character).is_empty());
    }
    let mut observed = 17;
    assert_eq!(
        unsafe { llev_phonetic_features(0xd800, &mut observed) },
        LlevStatus::InvalidArgument
    );
    assert_eq!(observed, 17);
    let mut expected: Vec<_> =
        chars_with_features(&[PhoneticFeature::Voiced, PhoneticFeature::Stop])
            .into_iter()
            .collect();
    expected.sort_unstable();
    assert_eq!(unsafe { copy_chars(0, voiced | stop) }, expected);
    let mut expected: Vec<_> = get_similar_chars('p').into_iter().collect();
    expected.sort_unstable();
    assert_eq!(unsafe { copy_chars(1, 'p' as u64) }, expected);
    assert_eq!(
        unsafe { copy_chars(2, 'p' as u64) },
        expand_feature_based('p')
    );
    let mut pair = 0;
    let mut found = 0;
    assert_eq!(
        unsafe { llev_phonetic_voicing_pair('p' as u32, &mut pair, &mut found) },
        LlevStatus::Ok
    );
    assert_eq!(found != 0, get_voicing_pair('p').is_some());
    if let Some(expected) = get_voicing_pair('p') {
        assert_eq!(pair, expected as u32);
    }
    for (source, target) in [('p', 'b'), ('a', 'u'), ('p', 'h')] {
        for (selector, expected) in [
            (0, are_similar(source, target)),
            (1, is_free_substitution(source, target)),
        ] {
            let mut actual = 99;
            assert_eq!(
                unsafe {
                    llev_phonetic_feature_relation(
                        source as u32,
                        target as u32,
                        selector,
                        &mut actual,
                    )
                },
                LlevStatus::Ok
            );
            assert_eq!(actual != 0, expected);
        }
    }
}

#[test]
fn feature_set_distance_and_mask_domain_are_exact() {
    let mut output = -1.0;
    let voiced = 1u64 << 0;
    let voiceless = 1u64 << 1;
    assert_eq!(
        unsafe { llev_phonetic_feature_set_distance(voiced, voiceless, ptr::null(), &mut output) },
        LlevStatus::Ok
    );
    let first = [PhoneticFeature::Voiced].into_iter().collect();
    let second = [PhoneticFeature::Voiceless].into_iter().collect();
    assert_eq!(
        output,
        liblevenshtein::phonetic::feature_distance::feature_set_distance(&first, &second)
    );
    assert_eq!(
        unsafe { llev_phonetic_feature_set_distance(1u64 << 63, voiced, ptr::null(), &mut output) },
        LlevStatus::InvalidArgument
    );
    assert_eq!(
        output,
        liblevenshtein::phonetic::feature_distance::feature_set_distance(&first, &second)
    );
}
