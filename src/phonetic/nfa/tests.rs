//! Tests for the phonetic NFA implementations (character-level and byte-level).
//!
//! These tests exercise both [`super::NFAChar`] and [`super::NFA`] using the
//! types re-exported via [`super`]. The tests are kept in a sibling test file
//! because they mix both character- and byte-level constructions in a single
//! test surface.

#[cfg(test)]
use super::{NFAChar, NFA, TransitionLabelChar};

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
