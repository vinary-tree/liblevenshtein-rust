//! Tests for the phonetic NFA implementations (character-level and byte-level).
//!
//! These tests exercise both [`super::NFAChar`] and [`super::NFA`] using the
//! types re-exported via [`super`]. The tests are kept in a sibling test file
//! because they mix both character- and byte-level constructions in a single
//! test surface.

#[cfg(test)]
use super::{
    types::{checked_state_id_add, state_id_from_len},
    NFAChar, TransitionLabelChar, NFA,
};

// --- NFAChar basic tests ---

#[test]
fn test_nfa_char_new() {
    let nfa = NFAChar::new();
    assert_eq!(nfa.num_states(), 1);
    assert_eq!(nfa.num_transitions(), 0);
    assert_eq!(nfa.start(), 0);
    assert!(!nfa.is_final(0));
}

#[test]
fn test_nfa_char_add_state() {
    let mut nfa = NFAChar::new();
    let q1 = nfa.add_state(false);
    let q2 = nfa.add_state(true);

    assert_eq!(q1, 1);
    assert_eq!(q2, 2);
    assert_eq!(nfa.num_states(), 3);
    assert!(!nfa.is_final(q1));
    assert!(nfa.is_final(q2));
}

#[test]
fn test_nfa_state_id_conversion_accepts_largest_u32_id() {
    assert_eq!(state_id_from_len(u32::MAX as usize), Some(u32::MAX));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn test_nfa_state_id_conversion_rejects_out_of_range_ids() {
    let out_of_range = (u32::MAX as usize) + 1;
    assert_eq!(state_id_from_len(out_of_range), None);
}

#[test]
fn test_nfa_state_id_addition_rejects_overflow() {
    assert_eq!(checked_state_id_add(u32::MAX - 1, 1), Some(u32::MAX));
    assert_eq!(checked_state_id_add(u32::MAX, 1), None);
}

#[test]
fn test_nfa_try_add_state_matches_add_state_for_char_and_byte() {
    let mut char_nfa = NFAChar::new();
    let mut byte_nfa = NFA::new();

    assert_eq!(char_nfa.try_add_state(true), Some(1));
    assert_eq!(byte_nfa.try_add_state(true), Some(1));
    assert!(char_nfa.is_final(1));
    assert!(byte_nfa.is_final(1));
}

#[test]
fn test_nfa_char_invalid_state_id_has_no_transitions() {
    let mut nfa = NFAChar::new();
    assert_eq!(nfa.transitions_from(u32::MAX).iter().count(), 0);

    let q1 = nfa.add_state(true);
    nfa.add_transition_char(nfa.start(), 'a', q1);
    assert_eq!(nfa.transitions_from(u32::MAX).iter().count(), 0);

    nfa.finalize();
    assert_eq!(nfa.transitions_from(u32::MAX).iter().count(), 0);
    nfa.set_final(u32::MAX, true);
    assert_eq!(nfa.finals().len(), 1);
}

#[test]
fn test_nfa_byte_invalid_state_id_has_no_transitions() {
    let mut nfa = NFA::new();
    assert_eq!(nfa.transitions_from(u32::MAX).iter().count(), 0);

    let q1 = nfa.add_state(true);
    nfa.add_transition_byte(nfa.start(), b'a', q1);
    assert_eq!(nfa.transitions_from(u32::MAX).iter().count(), 0);

    nfa.finalize();
    assert_eq!(nfa.transitions_from(u32::MAX).iter().count(), 0);
    nfa.set_final(u32::MAX, true);
    assert_eq!(nfa.finals().len(), 1);
}

#[test]
fn test_nfa_try_combinators_preserve_languages() {
    let mut char_a = NFAChar::new();
    let char_a_final = char_a.add_state(true);
    char_a.add_transition_char(char_a.start(), 'a', char_a_final);

    let mut char_b = NFAChar::new();
    let char_b_final = char_b.add_state(true);
    char_b.add_transition_char(char_b.start(), 'b', char_b_final);

    let char_union = char_a
        .clone()
        .try_union(char_b.clone())
        .expect("small NFAs fit in StateId");
    assert!(char_union.accepts("a"));
    assert!(char_union.accepts("b"));
    assert!(!char_union.accepts("c"));

    let char_concat = char_a
        .clone()
        .try_concatenate(char_b)
        .expect("small NFAs fit in StateId");
    assert!(char_concat.accepts("ab"));
    assert!(!char_concat.accepts("a"));

    let mut byte_a = NFA::new();
    let byte_a_final = byte_a.add_state(true);
    byte_a.add_transition_byte(byte_a.start(), b'a', byte_a_final);

    let byte_star = byte_a.try_kleene_star().expect("small NFAs fit in StateId");
    assert!(byte_star.accepts(b""));
    assert!(byte_star.accepts(b"aaa"));
    assert!(!byte_star.accepts(b"b"));
}

#[test]
fn test_nfa_char_simple_accept() {
    let mut nfa = NFAChar::new();
    let q0 = nfa.start();
    let q1 = nfa.add_state(true);

    nfa.add_transition_char(q0, 'a', q1);

    assert!(nfa.accepts("a"));
    assert!(!nfa.accepts("b"));
    assert!(!nfa.accepts(""));
    assert!(!nfa.accepts("aa"));
}

#[test]
fn test_nfa_char_epsilon_closure() {
    let mut nfa = NFAChar::new();
    let q0 = nfa.start();
    let q1 = nfa.add_state(false);
    let q2 = nfa.add_state(true);

    nfa.add_epsilon(q0, q1);
    nfa.add_epsilon(q1, q2);

    let closure = nfa.epsilon_closure_single(q0);
    assert!(closure.contains(&q0));
    assert!(closure.contains(&q1));
    assert!(closure.contains(&q2));
    assert_eq!(closure.len(), 3);
}

#[test]
fn test_nfa_char_accepts_with_epsilon() {
    let mut nfa = NFAChar::new();
    let q0 = nfa.start();
    let q1 = nfa.add_state(false);
    let q2 = nfa.add_state(true);

    nfa.add_epsilon(q0, q1);
    nfa.add_transition_char(q1, 'x', q2);

    assert!(nfa.accepts("x"));
    assert!(!nfa.accepts("y"));
}

// --- NFA combination tests ---

#[test]
fn test_nfa_char_union() {
    // NFA for "a"
    let mut nfa_a = NFAChar::new();
    let q0 = nfa_a.start();
    let q1 = nfa_a.add_state(true);
    nfa_a.add_transition_char(q0, 'a', q1);

    // NFA for "b"
    let mut nfa_b = NFAChar::new();
    let q0 = nfa_b.start();
    let q1 = nfa_b.add_state(true);
    nfa_b.add_transition_char(q0, 'b', q1);

    // Union: a | b
    let union = nfa_a.union(nfa_b);

    assert!(union.accepts("a"));
    assert!(union.accepts("b"));
    assert!(!union.accepts("c"));
    assert!(!union.accepts("ab"));
    assert!(!union.accepts(""));
}

#[test]
fn test_nfa_char_concatenate() {
    // NFA for "a"
    let mut nfa_a = NFAChar::new();
    let q0 = nfa_a.start();
    let q1 = nfa_a.add_state(true);
    nfa_a.add_transition_char(q0, 'a', q1);

    // NFA for "b"
    let mut nfa_b = NFAChar::new();
    let q0 = nfa_b.start();
    let q1 = nfa_b.add_state(true);
    nfa_b.add_transition_char(q0, 'b', q1);

    // Concatenation: ab
    let concat = nfa_a.concatenate(nfa_b);

    assert!(concat.accepts("ab"));
    assert!(!concat.accepts("a"));
    assert!(!concat.accepts("b"));
    assert!(!concat.accepts("ba"));
    assert!(!concat.accepts(""));
}

#[test]
fn test_nfa_char_kleene_star() {
    // NFA for "a"
    let mut nfa_a = NFAChar::new();
    let q0 = nfa_a.start();
    let q1 = nfa_a.add_state(true);
    nfa_a.add_transition_char(q0, 'a', q1);

    // Kleene star: a*
    let star = nfa_a.kleene_star();

    assert!(star.accepts(""));
    assert!(star.accepts("a"));
    assert!(star.accepts("aa"));
    assert!(star.accepts("aaa"));
    assert!(!star.accepts("b"));
    assert!(!star.accepts("ab"));
}

#[test]
fn test_nfa_char_kleene_plus() {
    // NFA for "a"
    let mut nfa_a = NFAChar::new();
    let q0 = nfa_a.start();
    let q1 = nfa_a.add_state(true);
    nfa_a.add_transition_char(q0, 'a', q1);

    // Kleene plus: a+
    let plus = nfa_a.kleene_plus();

    assert!(!plus.accepts(""));
    assert!(plus.accepts("a"));
    assert!(plus.accepts("aa"));
    assert!(plus.accepts("aaa"));
    assert!(!plus.accepts("b"));
}

#[test]
fn test_nfa_char_optional() {
    // NFA for "a"
    let mut nfa_a = NFAChar::new();
    let q0 = nfa_a.start();
    let q1 = nfa_a.add_state(true);
    nfa_a.add_transition_char(q0, 'a', q1);

    // Optional: a?
    let opt = nfa_a.optional();

    assert!(opt.accepts(""));
    assert!(opt.accepts("a"));
    assert!(!opt.accepts("aa"));
    assert!(!opt.accepts("b"));
}

// --- Byte-level NFA tests ---

#[test]
fn test_nfa_byte_simple() {
    let mut nfa = NFA::new();
    let q0 = nfa.start();
    let q1 = nfa.add_state(true);

    nfa.add_transition_byte(q0, b'a', q1);

    assert!(nfa.accepts(b"a"));
    assert!(nfa.accepts_str("a"));
    assert!(!nfa.accepts(b"b"));
    assert!(!nfa.accepts(b""));
}

#[test]
fn test_nfa_byte_union() {
    let mut nfa_a = NFA::new();
    let q0 = nfa_a.start();
    let q1 = nfa_a.add_state(true);
    nfa_a.add_transition_byte(q0, b'a', q1);

    let mut nfa_b = NFA::new();
    let q0 = nfa_b.start();
    let q1 = nfa_b.add_state(true);
    nfa_b.add_transition_byte(q0, b'b', q1);

    let union = nfa_a.union(nfa_b);

    assert!(union.accepts_str("a"));
    assert!(union.accepts_str("b"));
    assert!(!union.accepts_str("c"));
}

#[test]
fn test_nfa_byte_concatenate() {
    let mut nfa_a = NFA::new();
    let q0 = nfa_a.start();
    let q1 = nfa_a.add_state(true);
    nfa_a.add_transition_byte(q0, b'a', q1);

    let mut nfa_b = NFA::new();
    let q0 = nfa_b.start();
    let q1 = nfa_b.add_state(true);
    nfa_b.add_transition_byte(q0, b'b', q1);

    let concat = nfa_a.concatenate(nfa_b);

    assert!(concat.accepts_str("ab"));
    assert!(!concat.accepts_str("a"));
    assert!(!concat.accepts_str("b"));
}

// --- Complex pattern tests ---

#[test]
fn test_nfa_char_complex_pattern() {
    // Pattern: (a|b)*c
    // Build NFA for 'a'
    let mut nfa_a = NFAChar::new();
    let q0 = nfa_a.start();
    let q1 = nfa_a.add_state(true);
    nfa_a.add_transition_char(q0, 'a', q1);

    // Build NFA for 'b'
    let mut nfa_b = NFAChar::new();
    let q0 = nfa_b.start();
    let q1 = nfa_b.add_state(true);
    nfa_b.add_transition_char(q0, 'b', q1);

    // Build NFA for 'c'
    let mut nfa_c = NFAChar::new();
    let q0 = nfa_c.start();
    let q1 = nfa_c.add_state(true);
    nfa_c.add_transition_char(q0, 'c', q1);

    // (a|b)*
    let union = nfa_a.union(nfa_b);
    let star = union.kleene_star();

    // (a|b)*c
    let pattern = star.concatenate(nfa_c);

    assert!(pattern.accepts("c"));
    assert!(pattern.accepts("ac"));
    assert!(pattern.accepts("bc"));
    assert!(pattern.accepts("aac"));
    assert!(pattern.accepts("abc"));
    assert!(pattern.accepts("bac"));
    assert!(pattern.accepts("aabc"));
    assert!(!pattern.accepts(""));
    assert!(!pattern.accepts("a"));
    assert!(!pattern.accepts("b"));
    assert!(!pattern.accepts("ca"));
}

// --- Anchor tests ---

#[test]
fn test_nfa_char_start_of_line_anchor() {
    // Build NFA for ^hello
    let mut nfa = NFAChar::new();
    let q0 = nfa.start();
    let q1 = nfa.add_state(false);
    let q2 = nfa.add_state(false);
    let q3 = nfa.add_state(false);
    let q4 = nfa.add_state(false);
    let q5 = nfa.add_state(false);
    let q6 = nfa.add_state(true);

    // ^
    nfa.add_transition(q0, TransitionLabelChar::StartOfLine, q1);
    // hello
    nfa.add_transition_char(q1, 'h', q2);
    nfa.add_transition_char(q2, 'e', q3);
    nfa.add_transition_char(q3, 'l', q4);
    nfa.add_transition_char(q4, 'l', q5);
    nfa.add_transition_char(q5, 'o', q6);

    // Should match "hello" at start of input
    assert!(nfa.accepts("hello"));
    // Should not match with prefix
    assert!(!nfa.accepts("xhello"));
}

#[test]
fn test_nfa_char_end_of_line_anchor() {
    // Build NFA for hello$
    let mut nfa = NFAChar::new();
    let q0 = nfa.start();
    let q1 = nfa.add_state(false);
    let q2 = nfa.add_state(false);
    let q3 = nfa.add_state(false);
    let q4 = nfa.add_state(false);
    let q5 = nfa.add_state(false);
    let q6 = nfa.add_state(true);

    // hello
    nfa.add_transition_char(q0, 'h', q1);
    nfa.add_transition_char(q1, 'e', q2);
    nfa.add_transition_char(q2, 'l', q3);
    nfa.add_transition_char(q3, 'l', q4);
    nfa.add_transition_char(q4, 'o', q5);
    // $
    nfa.add_transition(q5, TransitionLabelChar::EndOfLine, q6);

    // Should match "hello" at end of input
    assert!(nfa.accepts("hello"));
}

#[test]
fn test_nfa_char_anchored_pattern() {
    // Build NFA for ^hello$
    let mut nfa = NFAChar::new();
    let q0 = nfa.start();
    let q1 = nfa.add_state(false);
    let q2 = nfa.add_state(false);
    let q3 = nfa.add_state(false);
    let q4 = nfa.add_state(false);
    let q5 = nfa.add_state(false);
    let q6 = nfa.add_state(false);
    let q7 = nfa.add_state(true);

    // ^
    nfa.add_transition(q0, TransitionLabelChar::StartOfLine, q1);
    // hello
    nfa.add_transition_char(q1, 'h', q2);
    nfa.add_transition_char(q2, 'e', q3);
    nfa.add_transition_char(q3, 'l', q4);
    nfa.add_transition_char(q4, 'l', q5);
    nfa.add_transition_char(q5, 'o', q6);
    // $
    nfa.add_transition(q6, TransitionLabelChar::EndOfLine, q7);

    // Should match exact "hello"
    assert!(nfa.accepts("hello"));
    // Should not match with extra chars
    assert!(!nfa.accepts("xhello"));
}

#[test]
fn test_nfa_char_multiline_start_of_line() {
    // Build NFA for ^line
    let mut nfa = NFAChar::new();
    let q0 = nfa.start();
    let q1 = nfa.add_state(false);
    let q2 = nfa.add_state(false);
    let q3 = nfa.add_state(false);
    let q4 = nfa.add_state(false);
    let q5 = nfa.add_state(true);

    // ^
    nfa.add_transition(q0, TransitionLabelChar::StartOfLine, q1);
    // line
    nfa.add_transition_char(q1, 'l', q2);
    nfa.add_transition_char(q2, 'i', q3);
    nfa.add_transition_char(q3, 'n', q4);
    nfa.add_transition_char(q4, 'e', q5);

    // Without multiline, should only match at start
    assert!(nfa.accepts_with_flags("line", false, false));

    // In multiline mode, ^ should match after newline
    // But the NFA expects the entire input to match, not a substring
    // So "first\nline" won't match ^line in our whole-string matching
}

#[test]
fn test_nfa_char_start_of_input_anchor() {
    // Build NFA for \Ahello
    let mut nfa = NFAChar::new();
    let q0 = nfa.start();
    let q1 = nfa.add_state(false);
    let q2 = nfa.add_state(false);
    let q3 = nfa.add_state(false);
    let q4 = nfa.add_state(false);
    let q5 = nfa.add_state(false);
    let q6 = nfa.add_state(true);

    // \A
    nfa.add_transition(q0, TransitionLabelChar::StartOfInput, q1);
    // hello
    nfa.add_transition_char(q1, 'h', q2);
    nfa.add_transition_char(q2, 'e', q3);
    nfa.add_transition_char(q3, 'l', q4);
    nfa.add_transition_char(q4, 'l', q5);
    nfa.add_transition_char(q5, 'o', q6);

    // \A only matches at absolute start, regardless of multiline mode
    assert!(nfa.accepts_with_flags("hello", true, false));
}

#[test]
fn test_nfa_char_end_of_input_anchor() {
    // Build NFA for hello\Z
    let mut nfa = NFAChar::new();
    let q0 = nfa.start();
    let q1 = nfa.add_state(false);
    let q2 = nfa.add_state(false);
    let q3 = nfa.add_state(false);
    let q4 = nfa.add_state(false);
    let q5 = nfa.add_state(false);
    let q6 = nfa.add_state(true);

    // hello
    nfa.add_transition_char(q0, 'h', q1);
    nfa.add_transition_char(q1, 'e', q2);
    nfa.add_transition_char(q2, 'l', q3);
    nfa.add_transition_char(q3, 'l', q4);
    nfa.add_transition_char(q4, 'o', q5);
    // \Z
    nfa.add_transition(q5, TransitionLabelChar::EndOfInput, q6);

    // \Z matches at end, optionally with trailing newline
    assert!(nfa.accepts("hello"));
    assert!(nfa.accepts("hello\n"));
}

#[test]
fn test_nfa_char_strict_end_of_input_anchor() {
    // Build NFA for hello\z
    let mut nfa = NFAChar::new();
    let q0 = nfa.start();
    let q1 = nfa.add_state(false);
    let q2 = nfa.add_state(false);
    let q3 = nfa.add_state(false);
    let q4 = nfa.add_state(false);
    let q5 = nfa.add_state(false);
    let q6 = nfa.add_state(true);

    // hello
    nfa.add_transition_char(q0, 'h', q1);
    nfa.add_transition_char(q1, 'e', q2);
    nfa.add_transition_char(q2, 'l', q3);
    nfa.add_transition_char(q3, 'l', q4);
    nfa.add_transition_char(q4, 'o', q5);
    // \z
    nfa.add_transition(q5, TransitionLabelChar::EndOfInputStrict, q6);

    // \z matches only at absolute end, no trailing newline
    assert!(nfa.accepts("hello"));
    assert!(!nfa.accepts("hello\n")); // \z doesn't allow trailing newline
}
