//! Character-level Zompist English phonetic rewrite rules.
//!
//! This submodule contains the Unicode-character-level (`RewriteRuleChar` =
//! `RewriteRule<char>`) definitions of the Zompist English
//! spelling-to-pronunciation rules and the five public aggregator functions:
//!
//! - [`orthography_rules_char`]
//! - [`vowel_digraph_rules_char`]
//! - [`phonetic_rules_char`]
//! - [`test_rules_char`]
//! - [`zompist_rules_char`]
//!
//! These are re-exported from `crate::phonetic::rules` so external call sites
//! continue to resolve `crate::phonetic::rules::orthography_rules_char` etc.
//!
//! The structure mirrors `zompist_byte` exactly, byte-for-byte, with `Phone` /
//! `Context` / `RewriteRule<u8>` replaced by `PhoneChar` / `ContextChar` /
//! `RewriteRuleChar`.

use crate::phonetic::types::{ContextChar, PhoneChar, RewriteRuleChar};

// ============================================================================
// Helper constants (char-level)
// ============================================================================

/// Front vowels for velar softening (e, i)
const FRONT_VOWELS_CHAR: &[char] = &['e', 'i'];

/// All vowels (a, e, i, o, u)
const VOWELS_CHAR: &[char] = &['a', 'e', 'i', 'o', 'u'];

// ============================================================================
// Character-level rules
// ============================================================================

fn rule_ch_to_tsh_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 1,
        rule_name: "ch → ç (tsh sound)".to_string(),
        pattern: vec![PhoneChar::Consonant('c'), PhoneChar::Consonant('h')],
        replacement: vec![PhoneChar::Digraph('c', 'h')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

fn rule_sh_to_sh_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 2,
        rule_name: "sh → $ (sh sound)".to_string(),
        pattern: vec![PhoneChar::Consonant('s'), PhoneChar::Consonant('h')],
        replacement: vec![PhoneChar::Digraph('s', 'h')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

fn rule_ph_to_f_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 3,
        rule_name: "ph → f".to_string(),
        pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('h')],
        replacement: vec![PhoneChar::Consonant('f')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

fn rule_c_to_s_before_front_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 20,
        rule_name: "c → s / _[ie]".to_string(),
        pattern: vec![PhoneChar::Consonant('c')],
        replacement: vec![PhoneChar::Consonant('s')],
        context: ContextChar::BeforeVowel(FRONT_VOWELS_CHAR.to_vec()),
        weight: 0.0,
        syllable_condition: None,
    }
}

fn rule_c_to_k_elsewhere_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 21,
        rule_name: "c → k (elsewhere)".to_string(),
        pattern: vec![PhoneChar::Consonant('c')],
        replacement: vec![PhoneChar::Consonant('k')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

fn rule_g_to_j_before_front_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 22,
        rule_name: "g → j / _[ie]".to_string(),
        pattern: vec![PhoneChar::Consonant('g')],
        replacement: vec![PhoneChar::Consonant('j')],
        context: ContextChar::BeforeVowel(FRONT_VOWELS_CHAR.to_vec()),
        weight: 0.0,
        syllable_condition: None,
    }
}

fn rule_silent_e_final_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 33,
        rule_name: "e → ∅ / _#".to_string(),
        pattern: vec![PhoneChar::Vowel('e')],
        replacement: vec![PhoneChar::Silent],
        context: ContextChar::Final,
        weight: 0.0,
        syllable_condition: None,
    }
}

fn rule_gh_silent_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 34,
        rule_name: "gh → ∅".to_string(),
        pattern: vec![PhoneChar::Consonant('g'), PhoneChar::Consonant('h')],
        replacement: vec![PhoneChar::Silent],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

fn phonetic_th_to_t_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 100,
        rule_name: "th → t (phonetic)".to_string(),
        pattern: vec![PhoneChar::Consonant('t'), PhoneChar::Consonant('h')],
        replacement: vec![PhoneChar::Consonant('t')],
        context: ContextChar::Anywhere,
        weight: 0.15,
        syllable_condition: None,
    }
}

fn phonetic_qu_to_kw_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 101,
        rule_name: "qu → kw (phonetic)".to_string(),
        pattern: vec![PhoneChar::Consonant('q'), PhoneChar::Consonant('u')],
        replacement: vec![PhoneChar::Consonant('k'), PhoneChar::Consonant('w')],
        context: ContextChar::Anywhere,
        weight: 0.15,
        syllable_condition: None,
    }
}

fn phonetic_kw_to_qu_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 102,
        rule_name: "kw → qu (phonetic reverse)".to_string(),
        pattern: vec![PhoneChar::Consonant('k'), PhoneChar::Consonant('w')],
        replacement: vec![PhoneChar::Consonant('q'), PhoneChar::Consonant('u')],
        context: ContextChar::Anywhere,
        weight: 0.15,
        syllable_condition: None,
    }
}

fn rule_x_expand_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 200,
        rule_name: "x → yy (expansion test)".to_string(),
        pattern: vec![PhoneChar::Consonant('x')],
        replacement: vec![PhoneChar::Consonant('y'), PhoneChar::Consonant('y')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

fn rule_y_to_z_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 201,
        rule_name: "y → z (transformation test)".to_string(),
        pattern: vec![PhoneChar::Consonant('y')],
        replacement: vec![PhoneChar::Consonant('z')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// Initial cluster rules (char-level) - ID 4-11
// ============================================================================

/// Rule 4: wr → r at word start (write → rite)
fn rule_wr_initial_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 4,
        rule_name: "wr → r / #_".to_string(),
        pattern: vec![PhoneChar::Consonant('w'), PhoneChar::Consonant('r')],
        replacement: vec![PhoneChar::Consonant('r')],
        context: ContextChar::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 5: wh → w at word start (what → wat)
fn rule_wh_initial_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 5,
        rule_name: "wh → w / #_".to_string(),
        pattern: vec![PhoneChar::Consonant('w'), PhoneChar::Consonant('h')],
        replacement: vec![PhoneChar::Consonant('w')],
        context: ContextChar::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 6: gn → n at word start (gnome → nome)
fn rule_gn_initial_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 6,
        rule_name: "gn → n / #_".to_string(),
        pattern: vec![PhoneChar::Consonant('g'), PhoneChar::Consonant('n')],
        replacement: vec![PhoneChar::Consonant('n')],
        context: ContextChar::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 7: kn → n at word start (knife → nife)
fn rule_kn_initial_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 7,
        rule_name: "kn → n / #_".to_string(),
        pattern: vec![PhoneChar::Consonant('k'), PhoneChar::Consonant('n')],
        replacement: vec![PhoneChar::Consonant('n')],
        context: ContextChar::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 8: mn → n at word start (mnemonic → nemonic)
fn rule_mn_initial_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 8,
        rule_name: "mn → n / #_".to_string(),
        pattern: vec![PhoneChar::Consonant('m'), PhoneChar::Consonant('n')],
        replacement: vec![PhoneChar::Consonant('n')],
        context: ContextChar::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 9: pt → t at word start (pterodactyl → terodactyl)
fn rule_pt_initial_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 9,
        rule_name: "pt → t / #_".to_string(),
        pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('t')],
        replacement: vec![PhoneChar::Consonant('t')],
        context: ContextChar::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 10: ps → s at word start (psychology → sycology)
fn rule_ps_initial_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 10,
        rule_name: "ps → s / #_".to_string(),
        pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('s')],
        replacement: vec![PhoneChar::Consonant('s')],
        context: ContextChar::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 11: tm → m at word start (tmesis → mesis)
fn rule_tm_initial_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 11,
        rule_name: "tm → m / #_".to_string(),
        pattern: vec![PhoneChar::Consonant('t'), PhoneChar::Consonant('m')],
        replacement: vec![PhoneChar::Consonant('m')],
        context: ContextChar::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// GH rules (char-level) - ID 35-38
// ============================================================================

/// Rule 35: gh → g before vowels (ghost → gost)
fn rule_gh_before_vowel_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 35,
        rule_name: "gh → g / _[aeiou]".to_string(),
        pattern: vec![PhoneChar::Consonant('g'), PhoneChar::Consonant('h')],
        replacement: vec![PhoneChar::Consonant('g')],
        context: ContextChar::BeforeVowel(VOWELS_CHAR.to_vec()),
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 36: ough → o (dough → do)
fn rule_ough_pattern_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 36,
        rule_name: "ough → o".to_string(),
        pattern: vec![
            PhoneChar::Vowel('o'),
            PhoneChar::Vowel('u'),
            PhoneChar::Consonant('g'),
            PhoneChar::Consonant('h'),
        ],
        replacement: vec![PhoneChar::Vowel('o')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 37: aught → ot (caught → kot)
fn rule_aught_pattern_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 37,
        rule_name: "aught → ot".to_string(),
        pattern: vec![
            PhoneChar::Vowel('a'),
            PhoneChar::Vowel('u'),
            PhoneChar::Consonant('g'),
            PhoneChar::Consonant('h'),
            PhoneChar::Consonant('t'),
        ],
        replacement: vec![PhoneChar::Vowel('o'), PhoneChar::Consonant('t')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 38: ought → ot (bought → bot)
fn rule_ought_pattern_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 38,
        rule_name: "ought → ot".to_string(),
        pattern: vec![
            PhoneChar::Vowel('o'),
            PhoneChar::Vowel('u'),
            PhoneChar::Consonant('g'),
            PhoneChar::Consonant('h'),
            PhoneChar::Consonant('t'),
        ],
        replacement: vec![PhoneChar::Vowel('o'), PhoneChar::Consonant('t')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// X pronunciation rules (char-level) - ID 40-41
// ============================================================================

/// Rule 40: x → ks (box → boks)
fn rule_x_to_ks_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 40,
        rule_name: "x → ks".to_string(),
        pattern: vec![PhoneChar::Consonant('x')],
        replacement: vec![PhoneChar::Consonant('k'), PhoneChar::Consonant('s')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 41: x → gz after vowel and before vowel (exact → egzact)
fn rule_x_to_gz_voiced_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 41,
        rule_name: "x → gz / [aeiou]_[aeiou]".to_string(),
        pattern: vec![PhoneChar::Consonant('x')],
        replacement: vec![PhoneChar::Consonant('g'), PhoneChar::Consonant('z')],
        context: ContextChar::And(
            Box::new(ContextChar::AfterVowel(VOWELS_CHAR.to_vec())),
            Box::new(ContextChar::BeforeVowel(VOWELS_CHAR.to_vec())),
        ),
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// Vowel digraph rules (char-level) - ID 50-62
// ============================================================================

/// Rule 50: ea → e (meat → met)
fn rule_ea_digraph_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 50,
        rule_name: "ea → e".to_string(),
        pattern: vec![PhoneChar::Vowel('e'), PhoneChar::Vowel('a')],
        replacement: vec![PhoneChar::Vowel('e')],
        context: ContextChar::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 51: ee → e (feet → fet)
fn rule_ee_digraph_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 51,
        rule_name: "ee → e".to_string(),
        pattern: vec![PhoneChar::Vowel('e'), PhoneChar::Vowel('e')],
        replacement: vec![PhoneChar::Vowel('e')],
        context: ContextChar::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 52: ai → a (rain → ran)
fn rule_ai_digraph_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 52,
        rule_name: "ai → a".to_string(),
        pattern: vec![PhoneChar::Vowel('a'), PhoneChar::Vowel('i')],
        replacement: vec![PhoneChar::Vowel('a')],
        context: ContextChar::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 53: ay → a (day → da)
fn rule_ay_digraph_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 53,
        rule_name: "ay → a".to_string(),
        pattern: vec![PhoneChar::Vowel('a'), PhoneChar::Consonant('y')],
        replacement: vec![PhoneChar::Vowel('a')],
        context: ContextChar::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 54: oa → o (boat → bot)
fn rule_oa_digraph_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 54,
        rule_name: "oa → o".to_string(),
        pattern: vec![PhoneChar::Vowel('o'), PhoneChar::Vowel('a')],
        replacement: vec![PhoneChar::Vowel('o')],
        context: ContextChar::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 55: oe → o (toe → to)
fn rule_oe_digraph_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 55,
        rule_name: "oe → o".to_string(),
        pattern: vec![PhoneChar::Vowel('o'), PhoneChar::Vowel('e')],
        replacement: vec![PhoneChar::Vowel('o')],
        context: ContextChar::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 56: ou → ow (house → howse)
fn rule_ou_digraph_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 56,
        rule_name: "ou → ow".to_string(),
        pattern: vec![PhoneChar::Vowel('o'), PhoneChar::Vowel('u')],
        replacement: vec![PhoneChar::Vowel('o'), PhoneChar::Consonant('w')],
        context: ContextChar::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 58: oi → oy (coin → coyn)
fn rule_oi_digraph_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 58,
        rule_name: "oi → oy".to_string(),
        pattern: vec![PhoneChar::Vowel('o'), PhoneChar::Vowel('i')],
        replacement: vec![PhoneChar::Vowel('o'), PhoneChar::Consonant('y')],
        context: ContextChar::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 59: ey → e (they → the)
fn rule_ey_digraph_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 59,
        rule_name: "ey → e".to_string(),
        pattern: vec![PhoneChar::Vowel('e'), PhoneChar::Consonant('y')],
        replacement: vec![PhoneChar::Vowel('e')],
        context: ContextChar::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 60: ie → i (pie → pi)
fn rule_ie_digraph_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 60,
        rule_name: "ie → i".to_string(),
        pattern: vec![PhoneChar::Vowel('i'), PhoneChar::Vowel('e')],
        replacement: vec![PhoneChar::Vowel('i')],
        context: ContextChar::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 61: oo → u (food → fud)
fn rule_oo_digraph_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 61,
        rule_name: "oo → u".to_string(),
        pattern: vec![PhoneChar::Vowel('o'), PhoneChar::Vowel('o')],
        replacement: vec![PhoneChar::Vowel('u')],
        context: ContextChar::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 62: ue → u at word end (blue → blu)
fn rule_ue_final_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 62,
        rule_name: "ue → u / _#".to_string(),
        pattern: vec![PhoneChar::Vowel('u'), PhoneChar::Vowel('e')],
        replacement: vec![PhoneChar::Vowel('u')],
        context: ContextChar::Final,
        weight: 0.1,
        syllable_condition: None,
    }
}

// ============================================================================
// Double consonant simplification (char-level) - ID 80-92
// ============================================================================

/// Rule 80: bb → b (rubber → ruber)
fn rule_bb_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 80,
        rule_name: "bb → b".to_string(),
        pattern: vec![PhoneChar::Consonant('b'), PhoneChar::Consonant('b')],
        replacement: vec![PhoneChar::Consonant('b')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 81: cc → c (account → acount)
fn rule_cc_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 81,
        rule_name: "cc → c".to_string(),
        pattern: vec![PhoneChar::Consonant('c'), PhoneChar::Consonant('c')],
        replacement: vec![PhoneChar::Consonant('c')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 82: dd → d (add → ad)
fn rule_dd_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 82,
        rule_name: "dd → d".to_string(),
        pattern: vec![PhoneChar::Consonant('d'), PhoneChar::Consonant('d')],
        replacement: vec![PhoneChar::Consonant('d')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 83: ff → f (staff → staf)
fn rule_ff_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 83,
        rule_name: "ff → f".to_string(),
        pattern: vec![PhoneChar::Consonant('f'), PhoneChar::Consonant('f')],
        replacement: vec![PhoneChar::Consonant('f')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 84: gg → g (egg → eg)
fn rule_gg_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 84,
        rule_name: "gg → g".to_string(),
        pattern: vec![PhoneChar::Consonant('g'), PhoneChar::Consonant('g')],
        replacement: vec![PhoneChar::Consonant('g')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 85: ll → l (ball → bal)
fn rule_ll_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 85,
        rule_name: "ll → l".to_string(),
        pattern: vec![PhoneChar::Consonant('l'), PhoneChar::Consonant('l')],
        replacement: vec![PhoneChar::Consonant('l')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 86: mm → m (hammer → hamer)
fn rule_mm_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 86,
        rule_name: "mm → m".to_string(),
        pattern: vec![PhoneChar::Consonant('m'), PhoneChar::Consonant('m')],
        replacement: vec![PhoneChar::Consonant('m')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 87: nn → n (dinner → diner)
fn rule_nn_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 87,
        rule_name: "nn → n".to_string(),
        pattern: vec![PhoneChar::Consonant('n'), PhoneChar::Consonant('n')],
        replacement: vec![PhoneChar::Consonant('n')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 88: pp → p (happy → hapy)
fn rule_pp_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 88,
        rule_name: "pp → p".to_string(),
        pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('p')],
        replacement: vec![PhoneChar::Consonant('p')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 89: rr → r (carry → cary)
fn rule_rr_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 89,
        rule_name: "rr → r".to_string(),
        pattern: vec![PhoneChar::Consonant('r'), PhoneChar::Consonant('r')],
        replacement: vec![PhoneChar::Consonant('r')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 90: ss → s (class → clas)
fn rule_ss_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 90,
        rule_name: "ss → s".to_string(),
        pattern: vec![PhoneChar::Consonant('s'), PhoneChar::Consonant('s')],
        replacement: vec![PhoneChar::Consonant('s')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 91: tt → t (butter → buter)
fn rule_tt_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 91,
        rule_name: "tt → t".to_string(),
        pattern: vec![PhoneChar::Consonant('t'), PhoneChar::Consonant('t')],
        replacement: vec![PhoneChar::Consonant('t')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 92: zz → z (buzz → buz)
fn rule_zz_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 92,
        rule_name: "zz → z".to_string(),
        pattern: vec![PhoneChar::Consonant('z'), PhoneChar::Consonant('z')],
        replacement: vec![PhoneChar::Consonant('z')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// Affrication rules (char-level) - ID 110-113
// ============================================================================

/// Rule 110: tion → shun (nation → nashun)
fn rule_tion_ending_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 110,
        rule_name: "tion → shun".to_string(),
        pattern: vec![
            PhoneChar::Consonant('t'),
            PhoneChar::Vowel('i'),
            PhoneChar::Vowel('o'),
            PhoneChar::Consonant('n'),
        ],
        replacement: vec![
            PhoneChar::Consonant('s'),
            PhoneChar::Consonant('h'),
            PhoneChar::Vowel('u'),
            PhoneChar::Consonant('n'),
        ],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 111: sion → zhun (vision → vizhun)
fn rule_sion_ending_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 111,
        rule_name: "sion → zhun".to_string(),
        pattern: vec![
            PhoneChar::Consonant('s'),
            PhoneChar::Vowel('i'),
            PhoneChar::Vowel('o'),
            PhoneChar::Consonant('n'),
        ],
        replacement: vec![
            PhoneChar::Consonant('z'),
            PhoneChar::Consonant('h'),
            PhoneChar::Vowel('u'),
            PhoneChar::Consonant('n'),
        ],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 112: cious → shus (precious → preshus)
fn rule_cious_ending_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 112,
        rule_name: "cious → shus".to_string(),
        pattern: vec![
            PhoneChar::Consonant('c'),
            PhoneChar::Vowel('i'),
            PhoneChar::Vowel('o'),
            PhoneChar::Vowel('u'),
            PhoneChar::Consonant('s'),
        ],
        replacement: vec![
            PhoneChar::Consonant('s'),
            PhoneChar::Consonant('h'),
            PhoneChar::Vowel('u'),
            PhoneChar::Consonant('s'),
        ],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 113: tious → shus (cautious → kaushus)
fn rule_tious_ending_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 113,
        rule_name: "tious → shus".to_string(),
        pattern: vec![
            PhoneChar::Consonant('t'),
            PhoneChar::Vowel('i'),
            PhoneChar::Vowel('o'),
            PhoneChar::Vowel('u'),
            PhoneChar::Consonant('s'),
        ],
        replacement: vec![
            PhoneChar::Consonant('s'),
            PhoneChar::Consonant('h'),
            PhoneChar::Vowel('u'),
            PhoneChar::Consonant('s'),
        ],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// Additional orthographic rules (char-level) - ID 130-135
// ============================================================================

/// Rule 130: tch → ch (batch → bach)
fn rule_tch_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 130,
        rule_name: "tch → ch".to_string(),
        pattern: vec![
            PhoneChar::Consonant('t'),
            PhoneChar::Consonant('c'),
            PhoneChar::Consonant('h'),
        ],
        replacement: vec![PhoneChar::Consonant('c'), PhoneChar::Consonant('h')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 131: dge → j (judge → juj)
fn rule_dge_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 131,
        rule_name: "dge → j".to_string(),
        pattern: vec![
            PhoneChar::Consonant('d'),
            PhoneChar::Consonant('g'),
            PhoneChar::Vowel('e'),
        ],
        replacement: vec![PhoneChar::Consonant('j')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 132: ck → k (back → bak)
fn rule_ck_simplify_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 132,
        rule_name: "ck → k".to_string(),
        pattern: vec![PhoneChar::Consonant('c'), PhoneChar::Consonant('k')],
        replacement: vec![PhoneChar::Consonant('k')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 133: mb → m at word end (lamb → lam)
fn rule_mb_final_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 133,
        rule_name: "mb → m / _#".to_string(),
        pattern: vec![PhoneChar::Consonant('m'), PhoneChar::Consonant('b')],
        replacement: vec![PhoneChar::Consonant('m')],
        context: ContextChar::Final,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 134: bt → t (debt → det)
fn rule_bt_silent_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 134,
        rule_name: "bt → t".to_string(),
        pattern: vec![PhoneChar::Consonant('b'), PhoneChar::Consonant('t')],
        replacement: vec![PhoneChar::Consonant('t')],
        context: ContextChar::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 135: mn → m at word end (hymn → him)
fn rule_mn_final_char() -> RewriteRuleChar {
    RewriteRuleChar {
        rule_id: 135,
        rule_name: "mn → m / _#".to_string(),
        pattern: vec![PhoneChar::Consonant('m'), PhoneChar::Consonant('n')],
        replacement: vec![PhoneChar::Consonant('m')],
        context: ContextChar::Final,
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// Rule sets (char-level)
// ============================================================================

/// Character-level orthography rules
///
/// Contains all orthography rules for standard English transformations.
/// Rules are ordered by priority - more specific patterns come first.
pub fn orthography_rules_char() -> Vec<RewriteRuleChar> {
    vec![
        // Phase 1: Specific multi-character patterns (HIGH PRIORITY)
        rule_tion_ending_char(),   // ID 110
        rule_sion_ending_char(),   // ID 111
        rule_cious_ending_char(),  // ID 112
        rule_tious_ending_char(),  // ID 113
        rule_ough_pattern_char(),  // ID 36
        rule_aught_pattern_char(), // ID 37
        rule_ought_pattern_char(), // ID 38
        rule_tch_simplify_char(),  // ID 130
        rule_dge_simplify_char(),  // ID 131
        // Phase 2: GH rules (before generic gh deletion)
        rule_gh_before_vowel_char(), // ID 35
        // Phase 3: Digraph conversions
        rule_ch_to_tsh_char(), // ID 1
        rule_sh_to_sh_char(),  // ID 2
        rule_ph_to_f_char(),   // ID 3
        // Phase 4: Initial cluster simplifications
        rule_wr_initial_char(), // ID 4
        rule_wh_initial_char(), // ID 5
        rule_gn_initial_char(), // ID 6
        rule_kn_initial_char(), // ID 7
        rule_mn_initial_char(), // ID 8
        rule_pt_initial_char(), // ID 9
        rule_ps_initial_char(), // ID 10
        rule_tm_initial_char(), // ID 11
        // Phase 5: X pronunciation (compound context first)
        rule_x_to_gz_voiced_char(), // ID 41 - must come before ID 40
        rule_x_to_ks_char(),        // ID 40
        // Phase 6: Contextual single-character rules
        rule_c_to_s_before_front_char(), // ID 20
        rule_c_to_k_elsewhere_char(),    // ID 21
        rule_g_to_j_before_front_char(), // ID 22
        // Phase 7: Additional orthographic rules
        rule_ck_simplify_char(), // ID 132
        rule_mb_final_char(),    // ID 133
        rule_bt_silent_char(),   // ID 134
        rule_mn_final_char(),    // ID 135
        // Phase 8: Double consonant simplification
        rule_bb_simplify_char(), // ID 80
        rule_cc_simplify_char(), // ID 81
        rule_dd_simplify_char(), // ID 82
        rule_ff_simplify_char(), // ID 83
        rule_gg_simplify_char(), // ID 84
        rule_ll_simplify_char(), // ID 85
        rule_mm_simplify_char(), // ID 86
        rule_nn_simplify_char(), // ID 87
        rule_pp_simplify_char(), // ID 88
        rule_rr_simplify_char(), // ID 89
        rule_ss_simplify_char(), // ID 90
        rule_tt_simplify_char(), // ID 91
        rule_zz_simplify_char(), // ID 92
        // Phase 9: Default/fallback rules (LOW PRIORITY)
        rule_silent_e_final_char(), // ID 33
        rule_gh_silent_char(),      // ID 34
    ]
}

/// Character-level vowel digraph rules: vowel digraph simplifications (weight=0.1)
///
/// Contains rules for vowel digraph normalization.
pub fn vowel_digraph_rules_char() -> Vec<RewriteRuleChar> {
    vec![
        rule_ea_digraph_char(), // ID 50
        rule_ee_digraph_char(), // ID 51
        rule_ai_digraph_char(), // ID 52
        rule_ay_digraph_char(), // ID 53
        rule_oa_digraph_char(), // ID 54
        rule_oe_digraph_char(), // ID 55
        rule_ou_digraph_char(), // ID 56
        rule_oi_digraph_char(), // ID 58
        rule_ey_digraph_char(), // ID 59
        rule_ie_digraph_char(), // ID 60
        rule_oo_digraph_char(), // ID 61
        rule_ue_final_char(),   // ID 62
    ]
}

/// Character-level phonetic rules
pub fn phonetic_rules_char() -> Vec<RewriteRuleChar> {
    vec![
        phonetic_th_to_t_char(),
        phonetic_qu_to_kw_char(),
        phonetic_kw_to_qu_char(),
    ]
}

/// Character-level test rules
pub fn test_rules_char() -> Vec<RewriteRuleChar> {
    vec![rule_x_expand_char(), rule_y_to_z_char()]
}

/// Character-level complete rule set: all 62 rules
///
/// Combined set of orthography + vowel digraph + phonetic + test rules.
pub fn zompist_rules_char() -> Vec<RewriteRuleChar> {
    let mut rules = Vec::with_capacity(62);
    rules.extend(orthography_rules_char());
    rules.extend(vowel_digraph_rules_char());
    rules.extend(phonetic_rules_char());
    rules.extend(test_rules_char());
    rules
}
