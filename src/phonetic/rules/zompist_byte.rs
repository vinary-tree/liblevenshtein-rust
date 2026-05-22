//! Byte-level Zompist English phonetic rewrite rules.
//!
//! This submodule contains the byte-level (`RewriteRuleByte` = `RewriteRule<u8>`)
//! definitions of the Zompist English spelling-to-pronunciation rules and the
//! five public aggregator functions:
//!
//! - [`orthography_rules`]
//! - [`vowel_digraph_rules`]
//! - [`phonetic_rules`]
//! - [`test_rules`]
//! - [`zompist_rules`]
//!
//! These are re-exported from `crate::phonetic::rules` so external call sites
//! continue to resolve `crate::phonetic::rules::orthography_rules` etc.
//!
//! See `docs/verification/phonetic/zompist_rules.v` for the formal Coq/Rocq
//! verification of the rules defined here.

use crate::phonetic::types::{Context, Phone, RewriteRule, RewriteRuleByte};

// ============================================================================
// Helper constants (byte-level)
// ============================================================================

/// Front vowels for velar softening (e, i)
const FRONT_VOWELS: &[u8] = &[b'e', b'i'];

/// All vowels (a, e, i, o, u)
const VOWELS: &[u8] = &[b'a', b'e', b'i', b'o', b'u'];

// ============================================================================
// Orthography rules (byte-level) - weight = 0.0
// ============================================================================

/// Rule 1: ch → ç (digraph representation)
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:49-56`
///
/// Example: "church" → "çurç"
fn rule_ch_to_tsh() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 1,
        rule_name: "ch → ç (tsh sound)".to_string(),
        pattern: vec![Phone::Consonant(b'c'), Phone::Consonant(b'h')],
        replacement: vec![Phone::Digraph(b'c', b'h')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 2: sh → $ (digraph)
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:60-67`
fn rule_sh_to_sh() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 2,
        rule_name: "sh → $ (sh sound)".to_string(),
        pattern: vec![Phone::Consonant(b's'), Phone::Consonant(b'h')],
        replacement: vec![Phone::Digraph(b's', b'h')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 3: ph → f
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:71-78`
fn rule_ph_to_f() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 3,
        rule_name: "ph → f".to_string(),
        pattern: vec![Phone::Consonant(b'p'), Phone::Consonant(b'h')],
        replacement: vec![Phone::Consonant(b'f')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 20: c → s before front vowels (e, i)
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:84-91`
fn rule_c_to_s_before_front() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 20,
        rule_name: "c → s / _[ie]".to_string(),
        pattern: vec![Phone::Consonant(b'c')],
        replacement: vec![Phone::Consonant(b's')],
        context: Context::BeforeVowel(FRONT_VOWELS.to_vec()),
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 21: c → k elsewhere
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:95-102`
fn rule_c_to_k_elsewhere() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 21,
        rule_name: "c → k (elsewhere)".to_string(),
        pattern: vec![Phone::Consonant(b'c')],
        replacement: vec![Phone::Consonant(b'k')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 22: g → j before front vowels
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:107-114`
fn rule_g_to_j_before_front() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 22,
        rule_name: "g → j / _[ie]".to_string(),
        pattern: vec![Phone::Consonant(b'g')],
        replacement: vec![Phone::Consonant(b'j')],
        context: Context::BeforeVowel(FRONT_VOWELS.to_vec()),
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 33: Silent 'e' at end of word
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:121-128`
fn rule_silent_e_final() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 33,
        rule_name: "e → ∅ / _#".to_string(),
        pattern: vec![Phone::Vowel(b'e')],
        replacement: vec![Phone::Silent],
        context: Context::Final,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 34: gh → ∅ (silent)
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:132-139`
fn rule_gh_silent() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 34,
        rule_name: "gh → ∅".to_string(),
        pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
        replacement: vec![Phone::Silent],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// Initial cluster rules (byte-level) - ID 4-11
// ============================================================================

/// Rule 4: wr → r at word start (write → rite)
fn rule_wr_initial() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 4,
        rule_name: "wr → r / #_".to_string(),
        pattern: vec![Phone::Consonant(b'w'), Phone::Consonant(b'r')],
        replacement: vec![Phone::Consonant(b'r')],
        context: Context::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 5: wh → w at word start (what → wat)
fn rule_wh_initial() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 5,
        rule_name: "wh → w / #_".to_string(),
        pattern: vec![Phone::Consonant(b'w'), Phone::Consonant(b'h')],
        replacement: vec![Phone::Consonant(b'w')],
        context: Context::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 6: gn → n at word start (gnome → nome)
fn rule_gn_initial() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 6,
        rule_name: "gn → n / #_".to_string(),
        pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'n')],
        replacement: vec![Phone::Consonant(b'n')],
        context: Context::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 7: kn → n at word start (knife → nife)
fn rule_kn_initial() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 7,
        rule_name: "kn → n / #_".to_string(),
        pattern: vec![Phone::Consonant(b'k'), Phone::Consonant(b'n')],
        replacement: vec![Phone::Consonant(b'n')],
        context: Context::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 8: mn → n at word start (mnemonic → nemonic)
fn rule_mn_initial() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 8,
        rule_name: "mn → n / #_".to_string(),
        pattern: vec![Phone::Consonant(b'm'), Phone::Consonant(b'n')],
        replacement: vec![Phone::Consonant(b'n')],
        context: Context::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 9: pt → t at word start (pterodactyl → terodactyl)
fn rule_pt_initial() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 9,
        rule_name: "pt → t / #_".to_string(),
        pattern: vec![Phone::Consonant(b'p'), Phone::Consonant(b't')],
        replacement: vec![Phone::Consonant(b't')],
        context: Context::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 10: ps → s at word start (psychology → sycology)
fn rule_ps_initial() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 10,
        rule_name: "ps → s / #_".to_string(),
        pattern: vec![Phone::Consonant(b'p'), Phone::Consonant(b's')],
        replacement: vec![Phone::Consonant(b's')],
        context: Context::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 11: tm → m at word start (tmesis → mesis)
fn rule_tm_initial() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 11,
        rule_name: "tm → m / #_".to_string(),
        pattern: vec![Phone::Consonant(b't'), Phone::Consonant(b'm')],
        replacement: vec![Phone::Consonant(b'm')],
        context: Context::Initial,
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// GH rules (byte-level) - ID 35-38
// ============================================================================

/// Rule 35: gh → g before vowels (ghost → gost)
fn rule_gh_before_vowel() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 35,
        rule_name: "gh → g / _[aeiou]".to_string(),
        pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
        replacement: vec![Phone::Consonant(b'g')],
        context: Context::BeforeVowel(VOWELS.to_vec()),
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 36: ough → o (dough → do)
fn rule_ough_pattern() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 36,
        rule_name: "ough → o".to_string(),
        pattern: vec![
            Phone::Vowel(b'o'),
            Phone::Vowel(b'u'),
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
        ],
        replacement: vec![Phone::Vowel(b'o')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 37: aught → ot (caught → kot)
fn rule_aught_pattern() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 37,
        rule_name: "aught → ot".to_string(),
        pattern: vec![
            Phone::Vowel(b'a'),
            Phone::Vowel(b'u'),
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
            Phone::Consonant(b't'),
        ],
        replacement: vec![Phone::Vowel(b'o'), Phone::Consonant(b't')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 38: ought → ot (bought → bot)
fn rule_ought_pattern() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 38,
        rule_name: "ought → ot".to_string(),
        pattern: vec![
            Phone::Vowel(b'o'),
            Phone::Vowel(b'u'),
            Phone::Consonant(b'g'),
            Phone::Consonant(b'h'),
            Phone::Consonant(b't'),
        ],
        replacement: vec![Phone::Vowel(b'o'), Phone::Consonant(b't')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// X pronunciation rules (byte-level) - ID 40-41
// ============================================================================

/// Rule 40: x → ks (box → boks)
fn rule_x_to_ks() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 40,
        rule_name: "x → ks".to_string(),
        pattern: vec![Phone::Consonant(b'x')],
        replacement: vec![Phone::Consonant(b'k'), Phone::Consonant(b's')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 41: x → gz after vowel and before vowel (exact → egzact)
fn rule_x_to_gz_voiced() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 41,
        rule_name: "x → gz / [aeiou]_[aeiou]".to_string(),
        pattern: vec![Phone::Consonant(b'x')],
        replacement: vec![Phone::Consonant(b'g'), Phone::Consonant(b'z')],
        context: Context::And(
            Box::new(Context::AfterVowel(VOWELS.to_vec())),
            Box::new(Context::BeforeVowel(VOWELS.to_vec())),
        ),
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// Vowel digraph rules (byte-level) - ID 50-62
// ============================================================================

/// Rule 50: ea → e (meat → met)
fn rule_ea_digraph() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 50,
        rule_name: "ea → e".to_string(),
        pattern: vec![Phone::Vowel(b'e'), Phone::Vowel(b'a')],
        replacement: vec![Phone::Vowel(b'e')],
        context: Context::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 51: ee → e (feet → fet)
fn rule_ee_digraph() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 51,
        rule_name: "ee → e".to_string(),
        pattern: vec![Phone::Vowel(b'e'), Phone::Vowel(b'e')],
        replacement: vec![Phone::Vowel(b'e')],
        context: Context::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 52: ai → a (rain → ran)
fn rule_ai_digraph() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 52,
        rule_name: "ai → a".to_string(),
        pattern: vec![Phone::Vowel(b'a'), Phone::Vowel(b'i')],
        replacement: vec![Phone::Vowel(b'a')],
        context: Context::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 53: ay → a (day → da)
fn rule_ay_digraph() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 53,
        rule_name: "ay → a".to_string(),
        pattern: vec![Phone::Vowel(b'a'), Phone::Consonant(b'y')],
        replacement: vec![Phone::Vowel(b'a')],
        context: Context::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 54: oa → o (boat → bot)
fn rule_oa_digraph() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 54,
        rule_name: "oa → o".to_string(),
        pattern: vec![Phone::Vowel(b'o'), Phone::Vowel(b'a')],
        replacement: vec![Phone::Vowel(b'o')],
        context: Context::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 55: oe → o (toe → to)
fn rule_oe_digraph() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 55,
        rule_name: "oe → o".to_string(),
        pattern: vec![Phone::Vowel(b'o'), Phone::Vowel(b'e')],
        replacement: vec![Phone::Vowel(b'o')],
        context: Context::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 56: ou → ow (house → howse)
fn rule_ou_digraph() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 56,
        rule_name: "ou → ow".to_string(),
        pattern: vec![Phone::Vowel(b'o'), Phone::Vowel(b'u')],
        replacement: vec![Phone::Vowel(b'o'), Phone::Consonant(b'w')],
        context: Context::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 58: oi → oy (coin → coyn)
fn rule_oi_digraph() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 58,
        rule_name: "oi → oy".to_string(),
        pattern: vec![Phone::Vowel(b'o'), Phone::Vowel(b'i')],
        replacement: vec![Phone::Vowel(b'o'), Phone::Consonant(b'y')],
        context: Context::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 59: ey → e (they → the)
fn rule_ey_digraph() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 59,
        rule_name: "ey → e".to_string(),
        pattern: vec![Phone::Vowel(b'e'), Phone::Consonant(b'y')],
        replacement: vec![Phone::Vowel(b'e')],
        context: Context::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 60: ie → i (pie → pi)
fn rule_ie_digraph() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 60,
        rule_name: "ie → i".to_string(),
        pattern: vec![Phone::Vowel(b'i'), Phone::Vowel(b'e')],
        replacement: vec![Phone::Vowel(b'i')],
        context: Context::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 61: oo → u (food → fud)
fn rule_oo_digraph() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 61,
        rule_name: "oo → u".to_string(),
        pattern: vec![Phone::Vowel(b'o'), Phone::Vowel(b'o')],
        replacement: vec![Phone::Vowel(b'u')],
        context: Context::Anywhere,
        weight: 0.1,
        syllable_condition: None,
    }
}

/// Rule 62: ue → u at word end (blue → blu)
fn rule_ue_final() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 62,
        rule_name: "ue → u / _#".to_string(),
        pattern: vec![Phone::Vowel(b'u'), Phone::Vowel(b'e')],
        replacement: vec![Phone::Vowel(b'u')],
        context: Context::Final,
        weight: 0.1,
        syllable_condition: None,
    }
}

// ============================================================================
// Double consonant simplification (byte-level) - ID 80-92
// ============================================================================

/// Rule 80: bb → b (rubber → ruber)
fn rule_bb_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 80,
        rule_name: "bb → b".to_string(),
        pattern: vec![Phone::Consonant(b'b'), Phone::Consonant(b'b')],
        replacement: vec![Phone::Consonant(b'b')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 81: cc → c (account → acount)
fn rule_cc_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 81,
        rule_name: "cc → c".to_string(),
        pattern: vec![Phone::Consonant(b'c'), Phone::Consonant(b'c')],
        replacement: vec![Phone::Consonant(b'c')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 82: dd → d (add → ad)
fn rule_dd_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 82,
        rule_name: "dd → d".to_string(),
        pattern: vec![Phone::Consonant(b'd'), Phone::Consonant(b'd')],
        replacement: vec![Phone::Consonant(b'd')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 83: ff → f (staff → staf)
fn rule_ff_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 83,
        rule_name: "ff → f".to_string(),
        pattern: vec![Phone::Consonant(b'f'), Phone::Consonant(b'f')],
        replacement: vec![Phone::Consonant(b'f')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 84: gg → g (egg → eg)
fn rule_gg_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 84,
        rule_name: "gg → g".to_string(),
        pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'g')],
        replacement: vec![Phone::Consonant(b'g')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 85: ll → l (ball → bal)
fn rule_ll_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 85,
        rule_name: "ll → l".to_string(),
        pattern: vec![Phone::Consonant(b'l'), Phone::Consonant(b'l')],
        replacement: vec![Phone::Consonant(b'l')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 86: mm → m (hammer → hamer)
fn rule_mm_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 86,
        rule_name: "mm → m".to_string(),
        pattern: vec![Phone::Consonant(b'm'), Phone::Consonant(b'm')],
        replacement: vec![Phone::Consonant(b'm')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 87: nn → n (dinner → diner)
fn rule_nn_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 87,
        rule_name: "nn → n".to_string(),
        pattern: vec![Phone::Consonant(b'n'), Phone::Consonant(b'n')],
        replacement: vec![Phone::Consonant(b'n')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 88: pp → p (happy → hapy)
fn rule_pp_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 88,
        rule_name: "pp → p".to_string(),
        pattern: vec![Phone::Consonant(b'p'), Phone::Consonant(b'p')],
        replacement: vec![Phone::Consonant(b'p')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 89: rr → r (carry → cary)
fn rule_rr_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 89,
        rule_name: "rr → r".to_string(),
        pattern: vec![Phone::Consonant(b'r'), Phone::Consonant(b'r')],
        replacement: vec![Phone::Consonant(b'r')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 90: ss → s (class → clas)
fn rule_ss_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 90,
        rule_name: "ss → s".to_string(),
        pattern: vec![Phone::Consonant(b's'), Phone::Consonant(b's')],
        replacement: vec![Phone::Consonant(b's')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 91: tt → t (butter → buter)
fn rule_tt_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 91,
        rule_name: "tt → t".to_string(),
        pattern: vec![Phone::Consonant(b't'), Phone::Consonant(b't')],
        replacement: vec![Phone::Consonant(b't')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 92: zz → z (buzz → buz)
fn rule_zz_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 92,
        rule_name: "zz → z".to_string(),
        pattern: vec![Phone::Consonant(b'z'), Phone::Consonant(b'z')],
        replacement: vec![Phone::Consonant(b'z')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// Affrication rules (byte-level) - ID 110-113
// ============================================================================

/// Rule 110: tion → shun (nation → nashun)
fn rule_tion_ending() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 110,
        rule_name: "tion → shun".to_string(),
        pattern: vec![
            Phone::Consonant(b't'),
            Phone::Vowel(b'i'),
            Phone::Vowel(b'o'),
            Phone::Consonant(b'n'),
        ],
        replacement: vec![
            Phone::Consonant(b's'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'u'),
            Phone::Consonant(b'n'),
        ],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 111: sion → zhun (vision → vizhun)
fn rule_sion_ending() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 111,
        rule_name: "sion → zhun".to_string(),
        pattern: vec![
            Phone::Consonant(b's'),
            Phone::Vowel(b'i'),
            Phone::Vowel(b'o'),
            Phone::Consonant(b'n'),
        ],
        replacement: vec![
            Phone::Consonant(b'z'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'u'),
            Phone::Consonant(b'n'),
        ],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 112: cious → shus (precious → preshus)
fn rule_cious_ending() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 112,
        rule_name: "cious → shus".to_string(),
        pattern: vec![
            Phone::Consonant(b'c'),
            Phone::Vowel(b'i'),
            Phone::Vowel(b'o'),
            Phone::Vowel(b'u'),
            Phone::Consonant(b's'),
        ],
        replacement: vec![
            Phone::Consonant(b's'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'u'),
            Phone::Consonant(b's'),
        ],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 113: tious → shus (cautious → kaushus)
fn rule_tious_ending() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 113,
        rule_name: "tious → shus".to_string(),
        pattern: vec![
            Phone::Consonant(b't'),
            Phone::Vowel(b'i'),
            Phone::Vowel(b'o'),
            Phone::Vowel(b'u'),
            Phone::Consonant(b's'),
        ],
        replacement: vec![
            Phone::Consonant(b's'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'u'),
            Phone::Consonant(b's'),
        ],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// Additional orthographic rules (byte-level) - ID 130-135
// ============================================================================

/// Rule 130: tch → ch (batch → bach)
fn rule_tch_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 130,
        rule_name: "tch → ch".to_string(),
        pattern: vec![
            Phone::Consonant(b't'),
            Phone::Consonant(b'c'),
            Phone::Consonant(b'h'),
        ],
        replacement: vec![Phone::Consonant(b'c'), Phone::Consonant(b'h')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 131: dge → j (judge → juj)
fn rule_dge_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 131,
        rule_name: "dge → j".to_string(),
        pattern: vec![
            Phone::Consonant(b'd'),
            Phone::Consonant(b'g'),
            Phone::Vowel(b'e'),
        ],
        replacement: vec![Phone::Consonant(b'j')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 132: ck → k (back → bak)
fn rule_ck_simplify() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 132,
        rule_name: "ck → k".to_string(),
        pattern: vec![Phone::Consonant(b'c'), Phone::Consonant(b'k')],
        replacement: vec![Phone::Consonant(b'k')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 133: mb → m at word end (lamb → lam)
fn rule_mb_final() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 133,
        rule_name: "mb → m / _#".to_string(),
        pattern: vec![Phone::Consonant(b'm'), Phone::Consonant(b'b')],
        replacement: vec![Phone::Consonant(b'm')],
        context: Context::Final,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 134: bt → t (debt → det)
fn rule_bt_silent() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 134,
        rule_name: "bt → t".to_string(),
        pattern: vec![Phone::Consonant(b'b'), Phone::Consonant(b't')],
        replacement: vec![Phone::Consonant(b't')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Rule 135: mn → m at word end (hymn → him)
fn rule_mn_final() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 135,
        rule_name: "mn → m / _#".to_string(),
        pattern: vec![Phone::Consonant(b'm'), Phone::Consonant(b'n')],
        replacement: vec![Phone::Consonant(b'm')],
        context: Context::Final,
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// Phonetic rules (byte-level) - weight = 0.15
// ============================================================================

/// Phonetic: th → t
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:147-154`
fn phonetic_th_to_t() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 100,
        rule_name: "th → t (phonetic)".to_string(),
        pattern: vec![Phone::Consonant(b't'), Phone::Consonant(b'h')],
        replacement: vec![Phone::Consonant(b't')],
        context: Context::Anywhere,
        weight: 0.15,
        syllable_condition: None,
    }
}

/// Phonetic: qu → kw
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:158-165`
fn phonetic_qu_to_kw() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 101,
        rule_name: "qu → kw (phonetic)".to_string(),
        pattern: vec![Phone::Consonant(b'q'), Phone::Consonant(b'u')],
        replacement: vec![Phone::Consonant(b'k'), Phone::Consonant(b'w')],
        context: Context::Anywhere,
        weight: 0.15,
        syllable_condition: None,
    }
}

/// Phonetic: kw → qu (reverse)
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:167-174`
fn phonetic_kw_to_qu() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 102,
        rule_name: "kw → qu (phonetic reverse)".to_string(),
        pattern: vec![Phone::Consonant(b'k'), Phone::Consonant(b'w')],
        replacement: vec![Phone::Consonant(b'q'), Phone::Consonant(b'u')],
        context: Context::Anywhere,
        weight: 0.15,
        syllable_condition: None,
    }
}

// ============================================================================
// Test rules (byte-level) - for non-commutativity demonstration
// ============================================================================

/// Test Rule 200: x → yy (expansion)
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:187-194`
fn rule_x_expand() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 200,
        rule_name: "x → yy (expansion test)".to_string(),
        pattern: vec![Phone::Consonant(b'x')],
        replacement: vec![Phone::Consonant(b'y'), Phone::Consonant(b'y')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

/// Test Rule 201: y → z (transformation)
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:197-204`
fn rule_y_to_z() -> RewriteRuleByte {
    RewriteRule {
        rule_id: 201,
        rule_name: "y → z (transformation test)".to_string(),
        pattern: vec![Phone::Consonant(b'y')],
        replacement: vec![Phone::Consonant(b'z')],
        context: Context::Anywhere,
        weight: 0.0,
        syllable_condition: None,
    }
}

// ============================================================================
// Rule sets (byte-level)
// ============================================================================

/// Orthography rules: exact transformations (weight=0.0)
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:209-218`
///
/// Contains all orthography rules for standard English transformations.
/// Rules are ordered by priority - more specific patterns come first.
pub fn orthography_rules() -> Vec<RewriteRuleByte> {
    vec![
        // Phase 1: Specific multi-character patterns (HIGH PRIORITY)
        rule_tion_ending(),   // ID 110
        rule_sion_ending(),   // ID 111
        rule_cious_ending(),  // ID 112
        rule_tious_ending(),  // ID 113
        rule_ough_pattern(),  // ID 36
        rule_aught_pattern(), // ID 37
        rule_ought_pattern(), // ID 38
        rule_tch_simplify(),  // ID 130
        rule_dge_simplify(),  // ID 131
        // Phase 2: GH rules (before generic gh deletion)
        rule_gh_before_vowel(), // ID 35
        // Phase 3: Digraph conversions
        rule_ch_to_tsh(), // ID 1
        rule_sh_to_sh(),  // ID 2
        rule_ph_to_f(),   // ID 3
        // Phase 4: Initial cluster simplifications
        rule_wr_initial(), // ID 4
        rule_wh_initial(), // ID 5
        rule_gn_initial(), // ID 6
        rule_kn_initial(), // ID 7
        rule_mn_initial(), // ID 8
        rule_pt_initial(), // ID 9
        rule_ps_initial(), // ID 10
        rule_tm_initial(), // ID 11
        // Phase 5: X pronunciation (compound context first)
        rule_x_to_gz_voiced(), // ID 41 - must come before ID 40
        rule_x_to_ks(),        // ID 40
        // Phase 6: Contextual single-character rules
        rule_c_to_s_before_front(), // ID 20
        rule_c_to_k_elsewhere(),    // ID 21
        rule_g_to_j_before_front(), // ID 22
        // Phase 7: Additional orthographic rules
        rule_ck_simplify(), // ID 132
        rule_mb_final(),    // ID 133
        rule_bt_silent(),   // ID 134
        rule_mn_final(),    // ID 135
        // Phase 8: Double consonant simplification
        rule_bb_simplify(), // ID 80
        rule_cc_simplify(), // ID 81
        rule_dd_simplify(), // ID 82
        rule_ff_simplify(), // ID 83
        rule_gg_simplify(), // ID 84
        rule_ll_simplify(), // ID 85
        rule_mm_simplify(), // ID 86
        rule_nn_simplify(), // ID 87
        rule_pp_simplify(), // ID 88
        rule_rr_simplify(), // ID 89
        rule_ss_simplify(), // ID 90
        rule_tt_simplify(), // ID 91
        rule_zz_simplify(), // ID 92
        // Phase 9: Default/fallback rules (LOW PRIORITY)
        rule_silent_e_final(), // ID 33
        rule_gh_silent(),      // ID 34
    ]
}

/// Vowel digraph rules: vowel digraph simplifications (weight=0.1)
///
/// Contains rules for vowel digraph normalization.
pub fn vowel_digraph_rules() -> Vec<RewriteRuleByte> {
    vec![
        rule_ea_digraph(), // ID 50
        rule_ee_digraph(), // ID 51
        rule_ai_digraph(), // ID 52
        rule_ay_digraph(), // ID 53
        rule_oa_digraph(), // ID 54
        rule_oe_digraph(), // ID 55
        rule_ou_digraph(), // ID 56
        rule_oi_digraph(), // ID 58
        rule_ey_digraph(), // ID 59
        rule_ie_digraph(), // ID 60
        rule_oo_digraph(), // ID 61
        rule_ue_final(),   // ID 62
    ]
}

/// Phonetic rules: approximate transformations (weight=0.15)
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:221-225`
///
/// Contains 3 rules for phonetic approximations.
pub fn phonetic_rules() -> Vec<RewriteRuleByte> {
    vec![phonetic_th_to_t(), phonetic_qu_to_kw(), phonetic_kw_to_qu()]
}

/// Test rules: for demonstrating non-commutativity
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:228-231`
///
/// Contains 2 rules used in Theorem 3 (non-confluence proof).
pub fn test_rules() -> Vec<RewriteRuleByte> {
    vec![rule_x_expand(), rule_y_to_z()]
}

/// Complete Zompist rule set: all 62 rules
///
/// **Formal Specification**: `docs/verification/phonetic/zompist_rules.v:234-235`
///
/// Combined set of orthography + vowel digraph + phonetic + test rules.
pub fn zompist_rules() -> Vec<RewriteRuleByte> {
    let mut rules = Vec::with_capacity(62);
    rules.extend(orthography_rules());
    rules.extend(vowel_digraph_rules());
    rules.extend(phonetic_rules());
    rules.extend(test_rules());
    rules
}
