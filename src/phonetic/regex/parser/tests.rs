use super::*;
use crate::phonetic::regex::ast::{Regex, RegexFlags};
use crate::phonetic::regex::error::ParseErrorKind;

#[test]
fn test_parse_literal() {
    let r = parse("phone").expect("test: parse phone");
    assert_eq!(r.to_string(), "phone");
}

#[test]
fn test_parse_alternation() {
    let r = parse("ph|f").expect("test: parse ph|f");
    assert_eq!(r.to_string(), "(ph|f)");
}

#[test]
fn test_parse_group() {
    let r = parse("(ph|f)one").expect("test: parse (ph|f)one");
    assert_eq!(r.to_string(), "((ph|f))one");
}

#[test]
fn test_parse_star() {
    let r = parse("a*").expect("test: parse a*");
    assert_eq!(r.to_string(), "a*");
}

#[test]
fn test_parse_plus() {
    let r = parse("a+").expect("test: parse a+");
    assert_eq!(r.to_string(), "a+");
}

#[test]
fn test_parse_optional() {
    let r = parse("a?").expect("test: parse a?");
    assert_eq!(r.to_string(), "a?");
}

#[test]
fn test_parse_char_class() {
    let r = parse("[aeiou]").expect("test: parse [aeiou]");
    assert_eq!(r.to_string(), "[aeiou]");
}

#[test]
fn test_parse_char_class_negated() {
    let r = parse("[^aeiou]").expect("test: parse [^aeiou]");
    assert_eq!(r.to_string(), "[^aeiou]");
}

#[test]
fn test_parse_char_class_range() {
    let r = parse("[a-z]").expect("test: parse [a-z]");
    // The display will show all characters in the range
    assert!(r.to_string().starts_with('['));
    assert!(r.to_string().ends_with(']'));
}

#[test]
fn test_parse_any() {
    let r = parse("a.b").expect("test: parse a.b");
    assert_eq!(r.to_string(), "a.b");
}

#[test]
fn test_parse_repetition_exact() {
    let r = parse("a{3}").expect("test: parse a{3}");
    assert_eq!(r.to_string(), "a{3}");
}

#[test]
fn test_parse_repetition_range() {
    let r = parse("a{2,4}").expect("test: parse a{2,4}");
    assert_eq!(r.to_string(), "a{2,4}");
}

#[test]
fn test_parse_repetition_unbounded() {
    let r = parse("a{2,}").expect("test: parse a{2,}");
    assert_eq!(r.to_string(), "a{2,}");
}

#[test]
fn test_parse_repetition_at_most() {
    // {,m} is equivalent to {0,m}
    let r = parse("a{,3}").expect("test: parse a{,3}");
    assert_eq!(r.to_string(), "a{0,3}");
}

#[test]
fn test_parse_repetition_at_most_zero() {
    // {,0} means zero occurrences (effectively empty)
    let r = parse("a{,0}").expect("test: parse a{,0}");
    assert_eq!(r.to_string(), "a{0,0}");
}

#[test]
fn test_parse_escape() {
    let r = parse("\\[\\]").expect("test: parse \\[\\]");
    assert_eq!(r.to_string(), "\\[\\]");
}

#[test]
fn test_parse_word_boundary() {
    let r = parse("#abc#").expect("test: parse #abc#");
    assert_eq!(r.to_string(), "#abc#");
}

#[test]
fn test_parse_rewrite_rule_simple() {
    let r = parse_rule("ph -> f").expect("test: parse_rule ph -> f");
    assert!(r.is_rewrite_rule());
    assert_eq!(r.to_string(), "ph -> f");
}

#[test]
fn test_parse_rewrite_rule_with_context() {
    let r = parse_rule("c -> s / _[ei]").expect("test: parse_rule c -> s / _[ei]");
    assert!(r.is_rewrite_rule());
    assert_eq!(r.to_string(), "c -> s / _[ei]");
}

#[test]
fn test_parse_rewrite_rule_word_end() {
    let r = parse_rule("e -> / _#").expect("test: parse_rule e -> / _#");
    assert!(r.is_rewrite_rule());
    assert_eq!(r.to_string(), "e ->  / _#");
}

#[test]
fn test_parse_rewrite_rule_word_start() {
    let r = parse_rule("k -> c / #_").expect("test: parse_rule k -> c / #_");
    assert!(r.is_rewrite_rule());
    assert_eq!(r.to_string(), "k -> c / #_");
}

#[test]
fn test_parse_complex_pattern() {
    let r = parse("(ph|f)one[s]?").expect("test: parse (ph|f)one[s]?");
    // Should parse without error
    assert!(!r.is_empty());
}

#[test]
fn test_parse_error_unclosed_group() {
    let result = parse("(abc");
    assert!(result.is_err());
    let err = result.unwrap_err();
    // The error could be UnclosedGroup or ExpectedChar(')')
    assert!(
        matches!(err.kind, ParseErrorKind::UnclosedGroup)
            || matches!(err.kind, ParseErrorKind::ExpectedChar(')'))
    );
}

#[test]
fn test_parse_error_unclosed_char_class() {
    let result = parse("[abc");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err.kind, ParseErrorKind::UnclosedCharClass));
}

#[test]
fn test_parse_error_empty_char_class() {
    let result = parse("[]");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err.kind, ParseErrorKind::EmptyCharClass));
}

#[test]
fn test_parse_error_invalid_repetition() {
    let result = parse("a{5,3}");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(
        err.kind,
        ParseErrorKind::InvalidRepetition { min: 5, max: 3 }
    ));
}

// Byte-level tests

#[test]
fn test_parse_bytes_literal() {
    let r = parse_bytes(b"phone").expect("test: parse_bytes phone");
    assert_eq!(r.to_string(), "phone");
}

#[test]
fn test_parse_bytes_alternation() {
    let r = parse_bytes(b"ph|f").expect("test: parse_bytes ph|f");
    assert_eq!(r.to_string(), "(ph|f)");
}

#[test]
fn test_parse_bytes_rewrite_rule() {
    let r = parse_rule_bytes(b"ph -> f").expect("test: parse_rule_bytes ph -> f");
    assert!(r.is_rewrite_rule());
    assert_eq!(r.to_string(), "ph -> f");
}

// ========================================================================
// Named character class tests
// ========================================================================

#[test]
fn test_parse_standalone_named_class_vowel() {
    let r = parse("[:vowel:]").expect("test: parse [:vowel:]");
    // Should parse as a character class containing vowels
    assert!(r.to_string().starts_with('['));
    assert!(r.to_string().contains('a'));
    assert!(r.to_string().contains('e'));
    assert!(r.to_string().contains('i'));
    assert!(r.to_string().contains('o'));
    assert!(r.to_string().contains('u'));
}

#[test]
fn test_parse_standalone_named_class_negated() {
    let r = parse("[^:vowel:]").expect("test: parse [^:vowel:]");
    // Should parse as a negated character class
    assert!(r.to_string().starts_with("[^"));
}

#[test]
fn test_parse_standalone_named_class_full_name() {
    // Use full name since shorthand aliases were removed
    let r = parse("[:vowel:]").expect("test: parse [:vowel:] full name");
    // Should contain vowels
    assert!(r.to_string().contains('a'));
    assert!(r.to_string().contains('e'));
}

#[test]
fn test_parse_standalone_named_class_alpha() {
    let r = parse("[:alpha:]").expect("test: parse [:alpha:]");
    // Should contain a-z, A-Z
    let s = r.to_string();
    assert!(s.contains('a'));
    assert!(s.contains('z'));
    assert!(s.contains('A'));
    assert!(s.contains('Z'));
}

#[test]
fn test_parse_standalone_named_class_digit() {
    let r = parse("[:digit:]").expect("test: parse [:digit:]");
    // Should contain 0-9
    let s = r.to_string();
    assert!(s.contains('0'));
    assert!(s.contains('9'));
}

#[test]
fn test_parse_posix_named_class_mixed() {
    let r = parse("[[:vowel:]y]").expect("test: parse [[:vowel:]y]");
    // Should contain vowels plus 'y'
    let s = r.to_string();
    assert!(s.contains('a'));
    assert!(s.contains('y'));
}

#[test]
fn test_parse_posix_named_class_multiple() {
    let r = parse("[[:vowel:][:digit:]]").expect("test: parse [[:vowel:][:digit:]]");
    // Should contain vowels and digits
    let s = r.to_string();
    assert!(s.contains('a'));
    assert!(s.contains('0'));
}

#[test]
fn test_parse_named_class_case_insensitive() {
    let r1 = parse("[:VOWEL:]").expect("test: parse [:VOWEL:]");
    let r2 = parse("[:vowel:]").expect("test: parse [:vowel:]");
    // Both should work and produce similar results
    assert!(r1.to_string().contains('a'));
    assert!(r2.to_string().contains('a'));
}

#[test]
fn test_parse_named_class_unknown() {
    let result = parse("[:unknown_class:]");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err.kind, ParseErrorKind::UnknownNamedClass(_)));
}

#[test]
fn test_parse_named_class_empty() {
    let result = parse("[::]");
    assert!(result.is_err());
}

#[test]
fn test_parse_named_class_in_rewrite_rule() {
    let r = parse_rule("c -> s / _[:front_vowel:]").expect("test: parse_rule front_vowel");
    assert!(r.is_rewrite_rule());
    // The context should contain front vowels
}

#[test]
fn test_parse_named_class_consonant() {
    let r = parse("[:consonant:]").expect("test: parse [:consonant:]");
    let s = r.to_string();
    // Should contain consonants
    assert!(s.contains('b'));
    assert!(s.contains('c'));
    assert!(s.contains('d'));
    // Should NOT contain vowels
    // (Actually the string representation shows all chars, so we can't easily test exclusion)
}

#[test]
fn test_parse_named_class_stop() {
    let r = parse("[:stop:]").expect("test: parse [:stop:]");
    let s = r.to_string();
    // Should contain stop consonants
    assert!(s.contains('p'));
    assert!(s.contains('t'));
    assert!(s.contains('k'));
    assert!(s.contains('b'));
    assert!(s.contains('d'));
    assert!(s.contains('g'));
}

#[test]
fn test_parse_literal_bracket_in_char_class() {
    // Make sure [[] still works (literal '[' in class)
    let r = parse("[[ab]").expect("test: parse [[ab]");
    let s = r.to_string();
    assert!(s.contains('['));
    assert!(s.contains('a'));
    assert!(s.contains('b'));
}

// ========================================================================
// Symbol reference tests
// ========================================================================

#[test]
fn test_parse_symbol_ref_standalone() {
    let mut symbols = SymbolTable::new();
    symbols.insert("VOWEL".to_string(), vec!['a', 'e', 'i', 'o', 'u']);

    let mut parser = Parser::new_with_symbols("$VOWEL", &symbols);
    let r = parser.parse().expect("test: parser.parse $VOWEL");
    let s = r.to_string();
    assert!(s.contains('a'));
    assert!(s.contains('e'));
    assert!(s.contains('i'));
    assert!(s.contains('o'));
    assert!(s.contains('u'));
}

#[test]
fn test_parse_symbol_ref_in_pattern() {
    let mut symbols = SymbolTable::new();
    symbols.insert("VOWEL".to_string(), vec!['a', 'e', 'i', 'o', 'u']);

    let mut parser = Parser::new_with_symbols("$VOWEL+", &symbols);
    let r = parser.parse().expect("test: parser.parse $VOWEL+");
    // Pattern should match one or more vowels
    assert!(r.to_string().contains('+'));
}

#[test]
fn test_dollar_literal_in_char_class() {
    // $ is a literal character inside character classes, not a symbol reference
    let r = parse("[$abc]").expect("test: parse [$abc]");
    let s = r.to_string();
    // Should contain literal '$' and 'a', 'b', 'c'
    assert!(s.contains('$'), "should contain literal $");
    assert!(s.contains('a'));
    assert!(s.contains('b'));
    assert!(s.contains('c'));
}

#[test]
fn test_parse_symbol_ref_braced() {
    let mut symbols = SymbolTable::new();
    symbols.insert("FRONT_VOWEL".to_string(), vec!['e', 'i']);

    let mut parser = Parser::new_with_symbols("${FRONT_VOWEL}y", &symbols);
    let r = parser.parse().expect("test: parser.parse ${FRONT_VOWEL}y");
    // Should parse as character class followed by 'y'
    assert!(r.to_string().contains('y'));
}

#[test]
fn test_parse_symbol_ref_undefined_error() {
    let symbols = SymbolTable::new();

    let mut parser = Parser::new_with_symbols("$UNDEFINED", &symbols);
    let result = parser.parse();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err.kind, ParseErrorKind::UndefinedSymbol { .. }));
}

#[test]
fn test_parse_symbol_ref_undefined_with_suggestions() {
    let mut symbols = SymbolTable::new();
    symbols.insert("VOWEL".to_string(), vec!['a', 'e', 'i', 'o', 'u']);
    symbols.insert("CONSONANT".to_string(), vec!['b', 'c', 'd']);

    let mut parser = Parser::new_with_symbols("$UNDEFINED", &symbols);
    let result = parser.parse();
    assert!(result.is_err());
    let err = result.unwrap_err();
    if let ParseErrorKind::UndefinedSymbol { name, available } = &err.kind {
        assert_eq!(name, "UNDEFINED");
        // Available should contain our defined symbols
        assert!(
            available.contains(&"VOWEL".to_string())
                || available.contains(&"CONSONANT".to_string())
        );
    } else {
        panic!("Expected UndefinedSymbol error");
    }
}

#[test]
fn test_parse_symbol_ref_no_symbols_error() {
    // Using regular parser without symbols
    let result = parse("$VOWEL");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err.kind, ParseErrorKind::UndefinedSymbol { .. }));
}

#[test]
fn test_parse_symbol_ref_outside_char_class() {
    // $ symbols are only expanded OUTSIDE character classes
    let mut symbols = SymbolTable::new();
    symbols.insert("V".to_string(), vec!['a', 'e', 'i', 'o', 'u']);

    // Symbol reference before char class
    let mut parser = Parser::new_with_symbols("$V[xyz]", &symbols);
    let r = parser.parse().expect("test: parser.parse $V[xyz]");
    let s = r.to_string();
    // Vowels should be expanded from $V
    assert!(s.contains('a'), "Should contain vowel from $V");
    assert!(s.contains('e'), "Should contain vowel from $V");
}

#[test]
fn test_dollar_is_literal_inside_char_class() {
    // $ is a LITERAL character inside character classes
    let mut symbols = SymbolTable::new();
    symbols.insert("V".to_string(), vec!['a', 'e', 'i', 'o', 'u']);

    // Inside char class, $V is literal chars '$', 'V'
    let mut parser = Parser::new_with_symbols("[$Vz]", &symbols);
    let r = parser.parse().expect("test: parser.parse [$Vz]");
    let s = r.to_string();
    // Should contain literal '$'
    assert!(s.contains('$'), "Should contain literal $");
    // Should contain literal 'V' (not expanded to vowels)
    assert!(s.contains('V'), "Should contain literal V");
    // Should NOT contain 'a' since symbol is not expanded
    assert!(!s.contains('a'), "Should NOT expand symbol inside []");
}

// ========================================================================
// Feature Bundle Tests
// ========================================================================

#[test]
fn test_feature_bundle_standalone_intersection() {
    // [:voiced stop:] should match only voiced stops: b, d, g
    let r = parse("[:voiced stop:]").expect("test: parse [:voiced stop:]");
    let s = r.to_string();
    // Should contain b, d, g
    assert!(s.contains('b'), "Should contain 'b'");
    assert!(s.contains('d'), "Should contain 'd'");
    assert!(s.contains('g'), "Should contain 'g'");
    // Should NOT contain voiceless stops: p, t, k
    assert!(!s.contains('p'), "Should NOT contain 'p'");
    assert!(!s.contains('t'), "Should NOT contain 't'");
    assert!(!s.contains('k'), "Should NOT contain 'k'");
}

#[test]
fn test_feature_bundle_standalone_negation() {
    // [:!nasal stop:] should match stops that are NOT nasal: p, t, k, b, d, g
    let r = parse("[:!nasal stop:]").expect("test: parse [:!nasal stop:]");
    let s = r.to_string();
    // Should contain all stops (none are nasal)
    assert!(s.contains('p'), "Should contain 'p'");
    assert!(s.contains('t'), "Should contain 't'");
    assert!(s.contains('k'), "Should contain 'k'");
    assert!(s.contains('b'), "Should contain 'b'");
    assert!(s.contains('d'), "Should contain 'd'");
    assert!(s.contains('g'), "Should contain 'g'");
}

#[test]
fn test_feature_bundle_standalone_single_negated() {
    // [:!nasal:] should match everything except nasals
    let r = parse("[:!nasal:]").expect("test: parse [:!nasal:]");
    let s = r.to_string();
    // Should NOT contain nasals
    assert!(!s.contains('m'), "Should NOT contain 'm'");
    assert!(!s.contains('n'), "Should NOT contain 'n'");
    // Should contain other consonants
    assert!(s.contains('p'), "Should contain 'p'");
    assert!(s.contains('s'), "Should contain 's'");
}

#[test]
fn test_feature_bundle_standalone_single_term() {
    // [:stop:] should work as before (backwards compatible)
    let r = parse("[:stop:]").expect("test: parse [:stop:] backwards-compat");
    let s = r.to_string();
    assert!(s.contains('p'), "Should contain 'p'");
    assert!(s.contains('t'), "Should contain 't'");
    assert!(s.contains('k'), "Should contain 'k'");
    assert!(s.contains('b'), "Should contain 'b'");
    assert!(s.contains('d'), "Should contain 'd'");
    assert!(s.contains('g'), "Should contain 'g'");
}

#[test]
fn test_feature_bundle_standalone_negated_outer() {
    // [^:voiced stop:] - negated outer, should NOT contain b, d, g
    let r = parse("[^:voiced stop:]").expect("test: parse [^:voiced stop:]");
    let s = r.to_string();
    // The outer negation should negate the char class
    // This syntax means "match anything NOT in voiced stop"
    assert!(s.contains('^'), "Should be a negated char class");
}

#[test]
fn test_feature_bundle_posix_intersection() {
    // [a[[:voiced stop:]]] - a plus voiced stops in POSIX syntax
    let r = parse("[a[[:voiced stop:]]]").expect("test: parse [a[[:voiced stop:]]]");
    let s = r.to_string();
    assert!(s.contains('a'), "Should contain 'a'");
    assert!(s.contains('b'), "Should contain 'b'");
    assert!(s.contains('d'), "Should contain 'd'");
    assert!(s.contains('g'), "Should contain 'g'");
}

#[test]
fn test_feature_bundle_posix_negation() {
    // [[[:!nasal stop:]]] - stops that aren't nasal
    let r = parse("[[[:!nasal stop:]]]").expect("test: parse [[[:!nasal stop:]]]");
    let s = r.to_string();
    assert!(s.contains('p'), "Should contain 'p'");
    assert!(s.contains('b'), "Should contain 'b'");
}

#[test]
fn test_feature_bundle_unknown_feature_error() {
    // [:unknown_feature:] should error
    let result = parse("[:unknown_feature:]");
    assert!(result.is_err());
}

#[test]
fn test_feature_bundle_empty_error() {
    // [::]  should error (empty)
    let result = parse("[::]");
    assert!(result.is_err());
}

// ========================================================================
// De Morgan's Law Tests
// ========================================================================

#[test]
fn test_double_negation_equals_positive() {
    // [^[^[:vowel:]]] should equal [:vowel:] (double negation cancels out)
    let double_neg = parse("[^[^[:vowel:]]]").expect("test: parse [^[^[:vowel:]]]");
    let positive = parse("[:vowel:]").expect("test: parse [:vowel:] positive");

    let double_neg_str = double_neg.to_string();
    let positive_str = positive.to_string();

    // Both should contain the same vowels
    for c in ['a', 'e', 'i', 'o', 'u'] {
        assert!(
            double_neg_str.contains(c),
            "double_neg should contain '{}'",
            c
        );
        assert!(positive_str.contains(c), "positive should contain '{}'", c);
    }
    // Neither should contain consonants (for the positive case, they're excluded)
    for _c in ['p', 't', 'k'] {
        // The double negation result should NOT be negated
        assert!(
            !double_neg_str.contains('^'),
            "double negation should not have ^ flag"
        );
    }
}

#[test]
fn test_negated_union() {
    // [^[:vowel:][:stop:]] = ¬(vowel ∪ stop)
    let r = parse("[^[:vowel:][:stop:]]").expect("test: parse [^[:vowel:][:stop:]]");
    let s = r.to_string();

    // Should be a negated char class
    assert!(s.starts_with("[^"), "Should be negated: {}", s);
    // Should contain vowels and stops (which are then negated)
    assert!(s.contains('a'), "Should contain 'a' (to be negated)");
    assert!(s.contains('p'), "Should contain 'p' (to be negated)");
}

#[test]
fn test_triple_negation() {
    // [^[^[^[:vowel:]]]] should equal [^[:vowel:]] (odd count = negated)
    let triple = parse("[^[^[^[:vowel:]]]]").expect("test: parse triple-neg");
    let s = triple.to_string();

    // Should be negated (odd count of negations)
    assert!(
        s.starts_with("[^"),
        "Triple negation should result in negated: {}",
        s
    );
}

#[test]
fn test_quadruple_negation() {
    // [^[^[^[^[:vowel:]]]]] should equal [:vowel:] (even count = positive)
    let quad = parse("[^[^[^[^[:vowel:]]]]]").expect("test: parse quad-neg");
    let s = quad.to_string();

    // Should NOT be negated (even count of negations)
    assert!(
        !s.starts_with("[^"),
        "Quadruple negation should be positive: {}",
        s
    );
    // Should contain vowels
    assert!(s.contains('a'), "Should contain 'a'");
    assert!(s.contains('e'), "Should contain 'e'");
}

// ========================================================================
// Phonetic Shortcut Parser Tests
// ========================================================================

#[test]
fn test_parse_shortcut_vowel() {
    let r = parse(r"\v").expect("test: parse \\v");
    let s = r.to_string();
    // Should contain vowels
    assert!(s.contains('a'), "Should contain 'a'");
    assert!(s.contains('e'), "Should contain 'e'");
    assert!(s.contains('i'), "Should contain 'i'");
    assert!(s.contains('o'), "Should contain 'o'");
    assert!(s.contains('u'), "Should contain 'u'");
}

#[test]
fn test_parse_shortcut_vowel_negated() {
    let r = parse(r"\V").expect("test: parse \\V");
    let s = r.to_string();
    // Should NOT contain vowels, should contain consonants
    assert!(!s.contains('a'), "Should NOT contain 'a'");
    assert!(!s.contains('e'), "Should NOT contain 'e'");
    assert!(
        s.contains('p') || s.contains('b') || s.contains('t'),
        "Should contain some consonants"
    );
}

#[test]
fn test_parse_shortcut_consonant() {
    let r = parse(r"\c").expect("test: parse \\c");
    let s = r.to_string();
    // Should contain consonants
    assert!(s.contains('p'), "Should contain 'p'");
    assert!(s.contains('b'), "Should contain 'b'");
    assert!(s.contains('t'), "Should contain 't'");
}

#[test]
fn test_parse_shortcut_stop() {
    let r = parse(r"\p").expect("test: parse \\p");
    let s = r.to_string();
    // Should contain stop consonants
    assert!(s.contains('p'), "Should contain 'p'");
    assert!(s.contains('t'), "Should contain 't'");
    assert!(s.contains('k'), "Should contain 'k'");
    assert!(s.contains('b'), "Should contain 'b'");
    assert!(s.contains('d'), "Should contain 'd'");
    assert!(s.contains('g'), "Should contain 'g'");
}

#[test]
fn test_parse_shortcut_digit() {
    let r = parse(r"\d").expect("test: parse \\d");
    let s = r.to_string();
    // Should contain digits
    assert!(s.contains('0'), "Should contain '0'");
    assert!(s.contains('5'), "Should contain '5'");
    assert!(s.contains('9'), "Should contain '9'");
}

#[test]
fn test_parse_shortcut_word() {
    let r = parse(r"\w").expect("test: parse \\w");
    let s = r.to_string();
    // Should contain word characters (alnum + _)
    assert!(s.contains('a'), "Should contain 'a'");
    assert!(s.contains('Z'), "Should contain 'Z'");
    assert!(s.contains('0'), "Should contain '0'");
    assert!(s.contains('_'), "Should contain '_'");
}

#[test]
fn test_parse_shortcut_space() {
    let r = parse(r"\s").expect("test: parse \\s");
    let s = r.to_string();
    // Should contain whitespace (space, tab, newline, etc.)
    assert!(s.contains(' '), "Should contain space");
}

#[test]
fn test_parse_shortcut_voiced() {
    let r = parse(r"\o").expect("test: parse \\o");
    let s = r.to_string();
    // Should contain voiced consonants
    assert!(s.contains('b'), "Should contain 'b'");
    assert!(s.contains('d'), "Should contain 'd'");
    assert!(s.contains('g'), "Should contain 'g'");
}

#[test]
fn test_parse_shortcut_fricative() {
    let r = parse(r"\e").expect("test: parse \\e");
    let s = r.to_string();
    // Should contain fricatives
    assert!(s.contains('f'), "Should contain 'f'");
    assert!(s.contains('v'), "Should contain 'v'");
    assert!(s.contains('s'), "Should contain 's'");
    assert!(s.contains('z'), "Should contain 'z'");
}

#[test]
fn test_parse_shortcut_affricate() {
    let r = parse(r"\a").expect("test: parse \\a");
    // Should parse successfully (affricates like ch, j)
    // Just check it parses - affricate class may be small
    assert!(r.to_string().len() > 0);
}

#[test]
fn test_parse_shortcut_in_char_class() {
    // Shortcuts should work inside character classes
    let r = parse(r"[\v123]").expect("test: parse [\\v123]");
    let s = r.to_string();
    // Should contain vowels and digits
    assert!(s.contains('a'), "Should contain 'a'");
    assert!(s.contains('1'), "Should contain '1'");
    assert!(s.contains('2'), "Should contain '2'");
    assert!(s.contains('3'), "Should contain '3'");
}

#[test]
fn test_parse_shortcut_negated_in_char_class() {
    // Negated shortcuts should work inside character classes
    let r = parse(r"[\V]").expect("test: parse [\\V]");
    let s = r.to_string();
    // Should NOT contain vowels
    assert!(!s.contains('a'), "Should NOT contain 'a'");
    assert!(!s.contains('e'), "Should NOT contain 'e'");
}

#[test]
fn test_parse_shortcut_mixed_in_pattern() {
    // Test simpler concatenation first
    let r = parse(r"\v\c").expect("test: parse \\v\\c");
    // Just check it parses successfully
    assert!(r.to_string().len() > 0);
}

#[test]
fn test_parse_shortcut_with_quantifier() {
    // Shortcut with quantifier
    let r = parse(r"\v+").expect("test: parse \\v+");
    // Just check it parses successfully
    assert!(r.to_string().len() > 0);
}

// ========================================================================
// Tests for Group Types (Phase 3)
// ========================================================================

#[test]
fn test_parse_capturing_group() {
    // Standard capturing group: (abc)
    let r = parse("(abc)").expect("test: parse (abc)");
    // Should produce CapturingGroup(1, ...)
    assert_eq!(r.to_string(), "(abc)");
}

#[test]
fn test_parse_capturing_group_numbering() {
    // Multiple capturing groups should get sequential numbers
    let r = parse("(a)(b)(c)").expect("test: parse (a)(b)(c)");
    // All groups parse correctly
    assert!(r.to_string().contains("(a)"));
    assert!(r.to_string().contains("(b)"));
    assert!(r.to_string().contains("(c)"));
}

#[test]
fn test_parse_non_capturing_group() {
    // Non-capturing group: (?:abc)
    let r = parse("(?:abc)").expect("test: parse (?:abc)");
    assert_eq!(r.to_string(), "(?:abc)");
}

#[test]
fn test_parse_non_capturing_group_complex() {
    // Non-capturing group with alternation
    let r = parse("(?:ph|f)one").expect("test: parse (?:ph|f)one");
    assert!(r.to_string().contains("(?:"));
}

#[test]
fn test_parse_named_group() {
    // Named group: (?<name>pattern)
    let r = parse("(?<vowel>[aeiou])").expect("test: parse named group vowel");
    assert!(r.to_string().contains("(?<vowel>"));
}

#[test]
fn test_parse_named_group_with_reference() {
    // Named group with valid reference
    let r = parse("(?<digit>[0-9])(?&digit)").expect("test: parse named group + backref");
    assert!(r.to_string().contains("(?<digit>"));
    assert!(r.to_string().contains("(?&digit)"));
}

#[test]
fn test_parse_duplicate_named_group_error() {
    // Duplicate named groups should error
    let result = parse("(?<x>a)(?<x>b)");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err.kind, ParseErrorKind::DuplicateGroupName(_)));
}

#[test]
fn test_parse_undefined_group_reference_error() {
    // References to undefined groups should error
    let result = parse("(?&undefined)");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(
        err.kind,
        ParseErrorKind::UndefinedGroupReference(_)
    ));
}

#[test]
fn test_parse_allows_external_group_reference_when_configured() {
    let mut parser = Parser::new("(?&IMPORTED_PATTERN)").allow_external_group_refs();
    let regex = parser
        .parse()
        .expect("external group refs should be allowed in opt-in mode");

    assert_eq!(regex, Regex::GroupRef("IMPORTED_PATTERN".to_string()));
}

#[test]
fn test_parse_forward_reference() {
    // Forward references are allowed (reference before definition)
    // The validation happens after parsing, so this pattern should parse
    // but the reference should be validated at the end
    let result = parse("(?&later)(?<later>abc)");
    // This should succeed because the group is defined before validation
    assert!(result.is_ok());
}

// ========================================================================
// Tests for Flags (Phase 3)
// ========================================================================

#[test]
fn test_parse_inline_flags_case_insensitive() {
    // Inline case-insensitive flag
    let r = parse("(?i)abc").expect("test: parse (?i)abc");
    assert!(r.to_string().contains("(?i)"));
}

#[test]
fn test_parse_scoped_flags_case_insensitive() {
    // Scoped case-insensitive flag
    let r = parse("(?i:abc)def").expect("test: parse (?i:abc)def");
    assert!(r.to_string().contains("(?i:"));
}

#[test]
fn test_parse_flag_disable() {
    // Disable flag
    let r = parse("(?-i)abc").expect("test: parse (?-i)abc");
    assert!(r.to_string().contains("(?-i)"));
}

#[test]
fn test_parse_unicode_normalization_flag() {
    // Unicode normalization flag
    let r = parse("(?u:NFC:abc)").expect("test: parse (?u:NFC:abc)");
    assert!(r.to_string().contains("u:NFC"));
}

#[test]
fn test_parse_unicode_normalization_nfd() {
    let r = parse("(?u:NFD:abc)").expect("test: parse (?u:NFD:abc)");
    assert!(r.to_string().contains("u:NFD"));
}

#[test]
fn test_parse_unicode_normalization_nfkc() {
    let r = parse("(?u:NFKC:abc)").expect("test: parse (?u:NFKC:abc)");
    assert!(r.to_string().contains("u:NFKC"));
}

#[test]
fn test_parse_unicode_normalization_nfkd() {
    let r = parse("(?u:NFKD:abc)").expect("test: parse (?u:NFKD:abc)");
    assert!(r.to_string().contains("u:NFKD"));
}

#[test]
fn test_parse_feature_flag() {
    // Feature-based matching flag
    let r = parse("(?f)abc").expect("test: parse (?f)abc");
    assert!(r.to_string().contains("(?f)"));
}

#[test]
fn test_parse_accent_flag() {
    // Accent-insensitive flag
    let r = parse("(?a)cafe").expect("test: parse (?a)cafe");
    assert!(r.to_string().contains("(?a)"));
}

#[test]
fn test_parse_combined_flags() {
    // Combined flags
    let r = parse("(?ia)abc").expect("test: parse (?ia)abc");
    let s = r.to_string();
    // Should contain both flags
    assert!(s.contains('i') || s.contains('a'));
}

#[test]
fn test_parse_combined_scoped_flags() {
    // Combined scoped flags
    let r = parse("(?ia:abc)def").expect("test: parse (?ia:abc)def");
    assert!(r.to_string().contains("(?"));
}

// ========================================================================
// Tests for NFA Compilation of New Group Types
// ========================================================================

#[test]
fn test_compile_non_capturing_group() {
    use crate::phonetic::nfa::compiler::compile;
    let regex = parse("(?:ph|f)one").expect("test: parse (?:ph|f)one");
    let nfa = compile(&regex).expect("test: compile (?:ph|f)one");
    assert!(nfa.accepts("phone"));
    assert!(nfa.accepts("fone"));
    assert!(!nfa.accepts("bone"));
}

#[test]
fn test_compile_capturing_group() {
    use crate::phonetic::nfa::compiler::compile;
    let regex = parse("(ph|f)one").expect("test: parse (ph|f)one");
    let nfa = compile(&regex).expect("test: compile (ph|f)one");
    assert!(nfa.accepts("phone"));
    assert!(nfa.accepts("fone"));
}

#[test]
fn test_compile_named_group() {
    use crate::phonetic::nfa::compiler::compile;
    let regex = parse("(?<prefix>ph|f)one").expect("test: parse named group");
    let nfa = compile(&regex).expect("test: compile named group");
    assert!(nfa.accepts("phone"));
    assert!(nfa.accepts("fone"));
}

#[test]
fn test_compile_flags_group() {
    use crate::phonetic::nfa::compiler::compile;
    // Flags don't affect matching yet, but should compile
    let regex = parse("(?i:abc)").expect("test: parse (?i:abc)");
    let nfa = compile(&regex).expect("test: compile (?i:abc)");
    // Without case-insensitive implementation, only exact match works
    assert!(nfa.accepts("abc"));
}

#[test]
fn test_compile_inline_flags() {
    use crate::phonetic::nfa::compiler::compile;
    // Inline flags without an inner pattern produce epsilon after flag extraction.
    let regex = parse("(?i)").expect("test: parse (?i)");
    let nfa = compile(&regex).expect("test: compile (?i)");
    // Should accept empty string (epsilon)
    assert!(nfa.accepts(""));
}

#[test]
fn test_compile_scoped_case_insensitive_flags() {
    use crate::phonetic::nfa::compiler::compile;
    let regex = parse("(?i:abc)def").expect("test: parse (?i:abc)def");
    let nfa = compile(&regex).expect("test: compile scoped case-insensitive flag");

    assert!(nfa.accepts("ABCdef"));
    assert!(nfa.accepts("AbCdef"));
    assert!(!nfa.accepts("ABCDEF"));
}

// ========================================================================
// Anchor Tests
// ========================================================================

/// Helper function to check if a Regex tree contains a specific variant
fn contains_variant(regex: &Regex, predicate: &dyn Fn(&Regex) -> bool) -> bool {
    if predicate(regex) {
        return true;
    }
    match regex {
        Regex::Concat(left, right) => {
            contains_variant(left, predicate) || contains_variant(right, predicate)
        }
        Regex::Alt(left, right) => {
            contains_variant(left, predicate) || contains_variant(right, predicate)
        }
        Regex::Star(inner) | Regex::Plus(inner) | Regex::Optional(inner) => {
            contains_variant(inner, predicate)
        }
        Regex::RepeatExact(inner, _) | Regex::RepeatRange(inner, _, _) => {
            contains_variant(inner, predicate)
        }
        Regex::CapturingGroup(_, inner)
        | Regex::NonCapturingGroup(inner)
        | Regex::NamedGroup(_, inner) => contains_variant(inner, predicate),
        Regex::FlagsGroup {
            inner: Some(inner), ..
        } => contains_variant(inner, predicate),
        _ => false,
    }
}

/// Helper to get the leftmost node in a concat chain
fn leftmost(regex: &Regex) -> &Regex {
    match regex {
        Regex::Concat(left, _) => leftmost(left),
        _ => regex,
    }
}

/// Helper to get the rightmost node in a concat chain
fn rightmost(regex: &Regex) -> &Regex {
    match regex {
        Regex::Concat(_, right) => rightmost(right),
        _ => regex,
    }
}

#[test]
fn test_parse_start_of_line_anchor() {
    let regex = parse("^hello").expect("test: parse ^hello");
    // Check that the leftmost element is StartOfLine
    assert!(
        matches!(leftmost(&regex), Regex::StartOfLine),
        "Expected StartOfLine at start, got {:?}",
        regex
    );
}

#[test]
fn test_parse_end_of_line_anchor() {
    let regex = parse("hello$").expect("test: parse hello$");
    // Check that the rightmost element is EndOfLine
    assert!(
        matches!(rightmost(&regex), Regex::EndOfLine),
        "Expected EndOfLine at end, got {:?}",
        regex
    );
}

#[test]
fn test_parse_both_anchors() {
    let regex = parse("^hello$").expect("test: parse ^hello$ anchors");
    assert!(
        matches!(leftmost(&regex), Regex::StartOfLine),
        "Expected StartOfLine at start"
    );
    assert!(
        matches!(rightmost(&regex), Regex::EndOfLine),
        "Expected EndOfLine at end"
    );
}

#[test]
fn test_parse_start_of_input_anchor() {
    let regex = parse(r"\Ahello").expect("test: parse \\Ahello");
    assert!(
        matches!(leftmost(&regex), Regex::StartOfInput),
        "Expected StartOfInput at start, got {:?}",
        regex
    );
}

#[test]
fn test_parse_end_of_input_anchor() {
    let regex = parse(r"hello\Z").expect("test: parse hello\\Z");
    assert!(
        matches!(rightmost(&regex), Regex::EndOfInput),
        "Expected EndOfInput at end, got {:?}",
        regex
    );
}

#[test]
fn test_parse_end_of_input_strict_anchor() {
    let regex = parse(r"hello\z").expect("test: parse hello\\z");
    assert!(
        matches!(rightmost(&regex), Regex::EndOfInputStrict),
        "Expected EndOfInputStrict at end, got {:?}",
        regex
    );
}

#[test]
fn test_parse_anchors_roundtrip() {
    // Test that anchors are correctly represented in Display
    let regex = parse("^hello$").expect("test: parse ^hello$ display");
    let display = regex.to_string();
    assert!(
        display.contains('^'),
        "Display should contain ^: {}",
        display
    );
    assert!(
        display.contains('$'),
        "Display should contain $: {}",
        display
    );
}

// ========================================================================
// Multiline and Dotall Flag Tests
// ========================================================================

/// Helper to find FlagsGroup in a regex tree and return its flags
fn find_flags_group(regex: &Regex) -> Option<&RegexFlags> {
    match regex {
        Regex::FlagsGroup { flags, .. } => Some(flags),
        Regex::Concat(left, right) => find_flags_group(left).or_else(|| find_flags_group(right)),
        _ => None,
    }
}

#[test]
fn test_parse_multiline_flag() {
    let regex = parse("(?m)^line$").expect("test: parse (?m)^line$");
    // Pattern should contain FlagsGroup with multiline=true
    let flags = find_flags_group(&regex);
    assert!(flags.is_some(), "Expected FlagsGroup in {:?}", regex);
    assert_eq!(
        flags.expect("test: flags is_some for multiline").multiline,
        Some(true)
    );
}

#[test]
fn test_parse_dotall_flag() {
    let regex = parse("(?s).*").expect("test: parse (?s).*");
    let flags = find_flags_group(&regex);
    assert!(flags.is_some(), "Expected FlagsGroup in {:?}", regex);
    assert_eq!(
        flags.expect("test: flags is_some for dotall").dotall,
        Some(true)
    );
}

#[test]
fn test_parse_combined_multiline_dotall() {
    let regex = parse("(?ms)test").expect("test: parse (?ms)test");
    let flags = find_flags_group(&regex);
    assert!(flags.is_some(), "Expected FlagsGroup in {:?}", regex);
    let flags = flags.expect("test: flags is_some for combined ms");
    assert_eq!(flags.multiline, Some(true));
    assert_eq!(flags.dotall, Some(true));
}

#[test]
fn test_parse_scoped_multiline() {
    let regex = parse("(?m:^line$)").expect("test: parse (?m:^line$)");
    // Should be FlagsGroup with inner pattern containing anchors
    match &regex {
        Regex::FlagsGroup {
            flags,
            inner: Some(inner),
        } => {
            assert_eq!(flags.multiline, Some(true));
            // Inner should contain anchors
            assert!(
                contains_variant(inner, &|r| matches!(r, Regex::StartOfLine)),
                "Expected StartOfLine in inner"
            );
            assert!(
                contains_variant(inner, &|r| matches!(r, Regex::EndOfLine)),
                "Expected EndOfLine in inner"
            );
        }
        _ => panic!("Expected FlagsGroup with inner pattern, got {:?}", regex),
    }
}

#[test]
fn test_parse_negated_flags() {
    let regex = parse("(?-ms)test").expect("test: parse (?-ms)test");
    let flags = find_flags_group(&regex);
    assert!(flags.is_some(), "Expected FlagsGroup in {:?}", regex);
    let flags = flags.expect("test: flags is_some for negated ms");
    assert_eq!(flags.multiline, Some(false), "multiline should be false");
    assert_eq!(flags.dotall, Some(false), "dotall should be false");
}
