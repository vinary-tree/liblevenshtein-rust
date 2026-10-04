#![cfg(all(feature = "ffi", feature = "bindings-phonetic"))]

use liblevenshtein::ffi::{
    llev_owned_string_free, llev_phonetic_expand, llev_phonetic_expand_with_costs,
    llev_phonetic_rules_free, llev_phonetic_rules_parse, LlevOwnedString,
    LlevPhoneticExpansionLimits, LlevStatus,
};
use liblevenshtein::phonetic::{
    expand_phonetic_alternatives_char, expand_phonetic_alternatives_char_bounded,
    expand_with_costs,
    llev::{parse_str, RuleSetChar},
    ExpansionLimitError,
};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn generated_bounded_expansions_equal_native(input in "[a-z]{0,8}") {
        let source = "ph -> f;";
        let native_rules = RuleSetChar::from_llev(&parse_str(source).unwrap()).unwrap().rules;
        let mut rules = std::ptr::null_mut();
        prop_assert_eq!(unsafe { llev_phonetic_rules_parse(source.as_ptr().cast(),
            source.len(), &mut rules) }, LlevStatus::Ok);
        let mut output = LlevOwnedString::default();
        prop_assert_eq!(unsafe { llev_phonetic_expand(input.as_ptr().cast(), input.len(),
            rules, &limits(), &mut output) }, LlevStatus::Ok);
        let observed = unsafe { take_text(&mut output) };
        let expected = expand_phonetic_alternatives_char(&input, &native_rules);
        prop_assert_eq!(observed.as_str(), expected.as_str());
        if expected.len() > 1 {
            let too_small = LlevPhoneticExpansionLimits {
                max_output_bytes: expected.len() - 1,
                ..limits()
            };
            let mut sentinel = LlevOwnedString { data: 1usize as *mut _, len: 999 };
            prop_assert_eq!(unsafe { llev_phonetic_expand(input.as_ptr().cast(), input.len(),
                rules, &too_small, &mut sentinel) }, LlevStatus::LimitExceeded);
            prop_assert_eq!(sentinel.data as usize, 1);
            prop_assert_eq!(sentinel.len, 999);
        }
        let mut cost = -1.0;
        prop_assert_eq!(unsafe { llev_phonetic_expand_with_costs(input.as_ptr().cast(),
            input.len(), rules, &limits(), &mut output, &mut cost) }, LlevStatus::Ok);
        let observed = unsafe { take_text(&mut output) };
        unsafe { llev_phonetic_rules_free(rules) };
        let (expected, expected_cost) = expand_with_costs(&input, &native_rules);
        prop_assert_eq!(observed, expected);
        prop_assert_eq!(cost, expected_cost);
    }
}

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

fn limits() -> LlevPhoneticExpansionLimits {
    LlevPhoneticExpansionLimits {
        max_input_scalars: 100,
        max_rules: 100,
        max_rule_units: 1000,
        max_nodes: 1000,
        max_output_bytes: 100_000,
    }
}

#[test]
fn bounded_core_and_c_abi_match_public_expansion() {
    let source = "ph -> f;";
    let native_rules = RuleSetChar::from_llev(&parse_str(source).unwrap())
        .unwrap()
        .rules;
    let mut rules = std::ptr::null_mut();
    assert_eq!(
        unsafe { llev_phonetic_rules_parse(source.as_ptr().cast(), source.len(), &mut rules) },
        LlevStatus::Ok
    );
    for input in ["", "fone", "café", "🦀fone", "f"] {
        let expected = expand_phonetic_alternatives_char(input, &native_rules);
        assert_eq!(
            expand_phonetic_alternatives_char_bounded(input, &native_rules, 1000, 100_000).unwrap(),
            expected
        );
        let mut output = LlevOwnedString::default();
        assert_eq!(
            unsafe {
                llev_phonetic_expand(
                    input.as_ptr().cast(),
                    input.len(),
                    rules,
                    &limits(),
                    &mut output,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(unsafe { take_text(&mut output) }, expected);
        let mut cost = -1.0;
        assert_eq!(
            unsafe {
                llev_phonetic_expand_with_costs(
                    input.as_ptr().cast(),
                    input.len(),
                    rules,
                    &limits(),
                    &mut output,
                    &mut cost,
                )
            },
            LlevStatus::Ok
        );
        let (expected_pattern, expected_cost) = expand_with_costs(input, &native_rules);
        assert_eq!(unsafe { take_text(&mut output) }, expected_pattern);
        assert_eq!(cost, expected_cost);
    }
    unsafe {
        llev_phonetic_rules_free(rules);
    }
}

#[test]
fn expansion_limits_reject_without_partial_output() {
    let source = "ph -> f;";
    let native_rules = RuleSetChar::from_llev(&parse_str(source).unwrap())
        .unwrap()
        .rules;
    assert_eq!(
        expand_phonetic_alternatives_char_bounded("fone", &native_rules, 1, 100_000),
        Err(ExpansionLimitError::NodeLimit)
    );
    let mut rules = std::ptr::null_mut();
    assert_eq!(
        unsafe { llev_phonetic_rules_parse(source.as_ptr().cast(), source.len(), &mut rules) },
        LlevStatus::Ok
    );
    let mut output = LlevOwnedString {
        data: 1usize as *mut _,
        len: 999,
    };
    let small = LlevPhoneticExpansionLimits {
        max_nodes: 1,
        ..limits()
    };
    assert_eq!(
        unsafe { llev_phonetic_expand(b"fone".as_ptr().cast(), 4, rules, &small, &mut output) },
        LlevStatus::LimitExceeded
    );
    assert_eq!(output.len, 999);
    let small = LlevPhoneticExpansionLimits {
        max_output_bytes: 2,
        ..limits()
    };
    let mut cost = -1.0;
    assert_eq!(
        unsafe {
            llev_phonetic_expand_with_costs(
                b"fone".as_ptr().cast(),
                4,
                rules,
                &small,
                &mut output,
                &mut cost,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(output.len, 999);
    assert_eq!(cost, -1.0);
    unsafe {
        llev_phonetic_rules_free(rules);
    }
}
