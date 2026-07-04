//! Unit tests for the phonetic-regex lexer.

use crate::phonetic::regex::error::ParseErrorKind;

use super::*;

#[test]
fn test_lexer_simple() {
    let mut lexer = Lexer::new("abc");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('c')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Eof
    );
}

#[test]
fn test_lexer_operators() {
    let mut lexer = Lexer::new("a*b+c?");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Star
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Plus
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('c')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Question
    );
}

#[test]
fn test_lexer_alternation() {
    let mut lexer = Lexer::new("a|b");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Pipe
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
}

#[test]
fn test_lexer_groups() {
    let mut lexer = Lexer::new("(ab)");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::GroupStart
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::GroupEnd
    );
}

#[test]
fn test_lexer_char_class() {
    let mut lexer = Lexer::new("[abc]");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassStart
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('c')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassEnd
    );
}

#[test]
fn test_lexer_char_class_negated() {
    let mut lexer = Lexer::new("[^abc]");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassStart
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Caret
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('c')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassEnd
    );
}

#[test]
fn test_lexer_char_class_range() {
    let mut lexer = Lexer::new("[a-z]");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassStart
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Dash
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('z')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassEnd
    );
}

#[test]
fn test_lexer_quantifier() {
    let mut lexer = Lexer::new("a{2,5}");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::QuantifierStart
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Number(2)
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Comma
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Number(5)
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::QuantifierEnd
    );
}

#[test]
fn char_lexer_rejects_oversized_number_tokens() {
    let oversized = format!("{}0", usize::MAX);
    let mut lexer = Lexer::new(&oversized);

    let err = lexer
        .next_token()
        .expect_err("oversized char-level numeric token must be rejected");

    assert!(matches!(
        err.kind,
        ParseErrorKind::InvalidQuantifier(message)
            if message.contains("usize::MAX")
    ));
}

#[test]
fn char_lexer_keeps_non_ascii_decimal_digits_as_characters() {
    let mut lexer = Lexer::new("٣3");

    assert_eq!(
        lexer
            .next_token()
            .expect("test: non-ASCII digit-like char must lex"),
        Token::Char('٣')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: ASCII digit after non-ASCII char must lex"),
        Token::Number(3)
    );
}

#[test]
fn test_lexer_arrow() {
    let mut lexer = Lexer::new("ph -> f");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('p')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('h')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Arrow
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('f')
    );
}

#[test]
fn test_lexer_context() {
    let mut lexer = Lexer::new("c -> s / _[ei]");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('c')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Arrow
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('s')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Slash
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Underscore
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassStart
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('e')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('i')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassEnd
    );
}

#[test]
fn test_lexer_word_boundary() {
    let mut lexer = Lexer::new("e -> / _#");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('e')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Arrow
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Slash
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Underscore
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Hash
    );
}

#[test]
fn test_lexer_escape() {
    let mut lexer = Lexer::new("\\[\\]\\*");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('[')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char(']')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('*')
    );
}

#[test]
fn test_lexer_hex_escape() {
    let mut lexer = Lexer::new("\\x41\\x42");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('A')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('B')
    );
}

#[test]
fn test_lexer_unicode_escape() {
    let mut lexer = Lexer::new("\\u00E9");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('é')
    );
}

#[test]
fn test_lexer_dot() {
    let mut lexer = Lexer::new("a.b");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Dot
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
}

#[test]
fn test_lexer_peek() {
    let mut lexer = Lexer::new("ab");
    assert_eq!(
        *lexer.peek().expect("test: lexer.peek must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        *lexer.peek().expect("test: lexer.peek still 'a'"),
        Token::Char('a')
    ); // Still 'a'
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
}

#[test]
fn test_lexer_whitespace() {
    let mut lexer = Lexer::new("a b c");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('c')
    );
}

// Symbol reference tests

#[test]
fn test_lexer_symbol_ref_simple() {
    let mut lexer = Lexer::new("$VOWEL");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::SymbolRef("VOWEL".to_string())
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Eof
    );
}

#[test]
fn test_lexer_symbol_ref_braced() {
    let mut lexer = Lexer::new("${FRONT_VOWEL}");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::SymbolRef("FRONT_VOWEL".to_string())
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Eof
    );
}

#[test]
fn test_lexer_symbol_ref_in_pattern() {
    let mut lexer = Lexer::new("a$VOWEL+");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::SymbolRef("VOWEL".to_string())
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Plus
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Eof
    );
}

#[test]
fn test_dollar_literal_in_char_class() {
    // $ is now a literal character inside character classes
    let mut lexer = Lexer::new("[$abc]");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassStart
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('$')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('c')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassEnd
    );
}

#[test]
fn test_lexer_symbol_ref_braced_adjacent() {
    // Test that ${NAME}x parses correctly
    let mut lexer = Lexer::new("${FRONT}y");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::SymbolRef("FRONT".to_string())
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('y')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Eof
    );
}

#[test]
fn test_lexer_dollar_followed_by_space_is_anchor() {
    // $ followed by space is EndOfLine anchor (not a symbol ref error)
    let mut lexer = Lexer::new("$ ");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::EndOfLine
    );
}

#[test]
fn test_lexer_symbol_ref_empty_braced_error() {
    let mut lexer = Lexer::new("${}");
    let err = lexer.next_token().unwrap_err();
    assert!(
        format!("{:?}", err).contains("empty"),
        "Error should mention empty"
    );
}

// Byte-level tests

#[test]
fn test_lexer_byte_simple() {
    let mut lexer = LexerByte::new(b"abc");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        TokenByte::Byte(b'a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        TokenByte::Byte(b'b')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        TokenByte::Byte(b'c')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        TokenByte::Eof
    );
}

#[test]
fn test_lexer_byte_operators() {
    let mut lexer = LexerByte::new(b"a*b+");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        TokenByte::Byte(b'a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        TokenByte::Star
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        TokenByte::Byte(b'b')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        TokenByte::Plus
    );
}

#[test]
fn byte_lexer_rejects_oversized_number_tokens() {
    let oversized = format!("{}0", usize::MAX);
    let mut lexer = LexerByte::new(oversized.as_bytes());

    let err = lexer
        .next_token()
        .expect_err("oversized byte-level numeric token must be rejected");

    assert!(matches!(
        err.kind,
        ParseErrorKind::InvalidQuantifier(message)
            if message.contains("usize::MAX")
    ));
}

#[test]
fn test_lexer_byte_arrow() {
    let mut lexer = LexerByte::new(b"ph -> f");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        TokenByte::Byte(b'p')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        TokenByte::Byte(b'h')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        TokenByte::Arrow
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        TokenByte::Byte(b'f')
    );
}

// ========================================================================
// Phonetic Shortcut Tests
// ========================================================================

#[test]
fn test_lexer_phonetic_shortcut_vowel() {
    let mut lexer = Lexer::new(r"\v\V");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "vowel",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "vowel",
            negated: true
        }
    );
}

#[test]
fn test_lexer_phonetic_shortcut_consonant() {
    let mut lexer = Lexer::new(r"\c\C");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "consonant",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "consonant",
            negated: true
        }
    );
}

#[test]
fn test_lexer_phonetic_shortcut_stop() {
    let mut lexer = Lexer::new(r"\p\P");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "stop",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "stop",
            negated: true
        }
    );
}

#[test]
fn test_lexer_phonetic_shortcut_voiced() {
    let mut lexer = Lexer::new(r"\o\O");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "voiced",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "voiced",
            negated: true
        }
    );
}

#[test]
fn test_lexer_phonetic_shortcut_fricative() {
    let mut lexer = Lexer::new(r"\e\E");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "fricative",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "fricative",
            negated: true
        }
    );
}

#[test]
fn test_lexer_phonetic_shortcut_affricate() {
    // Outside char class: \a = affricate shortcut, \A = StartOfInput anchor
    let mut lexer = Lexer::new(r"\a");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "affricate",
            negated: false
        }
    );

    // \A outside char class is StartOfInput anchor (not negated affricate)
    let mut lexer2 = Lexer::new(r"\A");
    assert_eq!(
        lexer2
            .next_token()
            .expect("test: lexer2.next_token must be Ok"),
        Token::StartOfInput
    );

    // Inside char class: both \a and \A are phonetic shortcuts
    let mut lexer3 = Lexer::new(r"[\a\A]");
    assert_eq!(
        lexer3
            .next_token()
            .expect("test: lexer3.next_token must be Ok"),
        Token::CharClassStart
    );
    assert_eq!(
        lexer3
            .next_token()
            .expect("test: lexer3.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "affricate",
            negated: false
        }
    );
    assert_eq!(
        lexer3
            .next_token()
            .expect("test: lexer3.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "affricate",
            negated: true
        }
    );
}

#[test]
fn test_lexer_standard_shortcut_digit() {
    let mut lexer = Lexer::new(r"\d\D");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "digit",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "digit",
            negated: true
        }
    );
}

#[test]
fn test_lexer_standard_shortcut_word() {
    let mut lexer = Lexer::new(r"\w\W");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "word",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "word",
            negated: true
        }
    );
}

#[test]
fn test_lexer_standard_shortcut_space() {
    let mut lexer = Lexer::new(r"\s\S");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "space",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "space",
            negated: true
        }
    );
}

#[test]
fn test_lexer_shortcut_in_char_class() {
    // Shortcuts should work inside character classes too
    let mut lexer = Lexer::new(r"[\v\d]");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassStart
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "vowel",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "digit",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassEnd
    );
}

#[test]
fn test_lexer_shortcut_all_vowel_types() {
    let mut lexer = Lexer::new(r"\f\k\h\l\m");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "front_vowel",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "back_vowel",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "high_vowel",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "low_vowel",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "mid_vowel",
            negated: false
        }
    );
}

#[test]
fn test_lexer_shortcut_consonant_types() {
    // \g = glide, \q = liquid (outside char class)
    // Note: \z is EndOfInputStrict anchor outside char class
    let mut lexer = Lexer::new(r"\g\q");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "glide",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "liquid",
            negated: false
        }
    );

    // \z outside char class is EndOfInputStrict anchor
    let mut lexer2 = Lexer::new(r"\z");
    assert_eq!(
        lexer2
            .next_token()
            .expect("test: lexer2.next_token must be Ok"),
        Token::EndOfInputStrict
    );

    // Inside char class: \z is nasal phonetic shortcut
    let mut lexer3 = Lexer::new(r"[\z]");
    assert_eq!(
        lexer3
            .next_token()
            .expect("test: lexer3.next_token must be Ok"),
        Token::CharClassStart
    );
    assert_eq!(
        lexer3
            .next_token()
            .expect("test: lexer3.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "nasal",
            negated: false
        }
    );
}

// ========================================================================
// Special Group Tests
// ========================================================================

#[test]
fn test_lexer_non_capturing_group() {
    let mut lexer = Lexer::new("(?:abc)");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::NonCapturingGroupStart
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('c')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::GroupEnd
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Eof
    );
}

#[test]
fn test_lexer_named_group() {
    let mut lexer = Lexer::new("(?<vowel>[aeiou])");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::NamedGroupStart("vowel".to_string())
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassStart
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('e')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('i')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('o')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('u')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassEnd
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::GroupEnd
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Eof
    );
}

#[test]
fn test_lexer_named_group_with_underscore() {
    let mut lexer = Lexer::new("(?<front_vowel>[ei])");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::NamedGroupStart("front_vowel".to_string())
    );
}

#[test]
fn test_lexer_group_reference() {
    let mut lexer = Lexer::new("(?&vowel)");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::GroupReference("vowel".to_string())
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Eof
    );
}

#[test]
fn test_lexer_inline_flags_case_insensitive() {
    let mut lexer = Lexer::new("(?i)abc");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.case_insensitive, Some(true));
            assert_eq!(flags.feature_based, None);
            assert_eq!(flags.accent_insensitive, None);
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
}

#[test]
fn test_lexer_inline_flags_negated() {
    let mut lexer = Lexer::new("(?-i)abc");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.case_insensitive, Some(false));
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
}

#[test]
fn test_lexer_inline_flags_combined() {
    let mut lexer = Lexer::new("(?ia)abc");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.case_insensitive, Some(true));
            assert_eq!(flags.accent_insensitive, Some(true));
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
}

#[test]
fn test_lexer_scoped_flags() {
    let mut lexer = Lexer::new("(?i:abc)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::ScopedFlagsStart(flags) => {
            assert_eq!(flags.case_insensitive, Some(true));
        }
        _ => panic!("Expected ScopedFlagsStart, got {:?}", token),
    }
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('c')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::GroupEnd
    );
}

#[test]
fn test_lexer_unicode_normalization_flag() {
    let mut lexer = Lexer::new("(?u:NFC)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.unicode_normalization, Some("NFC".to_string()));
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
}

#[test]
fn test_lexer_unicode_normalization_scoped() {
    let mut lexer = Lexer::new("(?u:NFD:cafe)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::ScopedFlagsStart(flags) => {
            assert_eq!(flags.unicode_normalization, Some("NFD".to_string()));
        }
        _ => panic!("Expected ScopedFlagsStart, got {:?}", token),
    }
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('c')
    );
}

#[test]
fn test_lexer_feature_flag() {
    let mut lexer = Lexer::new("(?f)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.feature_based, Some(true));
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
}

#[test]
fn test_lexer_named_group_empty_error() {
    let mut lexer = Lexer::new("(?<>abc)");
    let err = lexer.next_token().unwrap_err();
    assert!(
        format!("{:?}", err).contains("empty"),
        "Error should mention empty group name"
    );
}

#[test]
fn test_lexer_group_reference_empty_error() {
    let mut lexer = Lexer::new("(?&)");
    let err = lexer.next_token().unwrap_err();
    assert!(
        format!("{:?}", err).contains("empty"),
        "Error should mention empty group reference"
    );
}

#[test]
fn test_lexer_invalid_group_syntax() {
    let mut lexer = Lexer::new("(?x)");
    let err = lexer.next_token().unwrap_err();
    assert!(
        format!("{:?}", err).contains("InvalidGroupSyntax")
            || format!("{:?}", err).contains("InvalidFlag"),
        "Error should be InvalidGroupSyntax or InvalidFlag"
    );
}

#[test]
fn test_lexer_capturing_group_unchanged() {
    // Regular (abc) should still produce GroupStart
    let mut lexer = Lexer::new("(abc)");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::GroupStart
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('c')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::GroupEnd
    );
}

// ========================================================================
// Anchor Tests
// ========================================================================

#[test]
fn test_lexer_anchor_start_of_line() {
    let mut lexer = Lexer::new("^hello");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::StartOfLine
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('h')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('e')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('l')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('l')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('o')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Eof
    );
}

#[test]
fn test_lexer_anchor_end_of_line() {
    let mut lexer = Lexer::new("hello$");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('h')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('e')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('l')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('l')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('o')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::EndOfLine
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Eof
    );
}

#[test]
fn test_lexer_anchor_both() {
    let mut lexer = Lexer::new("^hello$");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::StartOfLine
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('h')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('e')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('l')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('l')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('o')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::EndOfLine
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Eof
    );
}

#[test]
fn test_lexer_anchor_start_of_input() {
    // \A outside char class = StartOfInput anchor
    let mut lexer = Lexer::new(r"\Ahello");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::StartOfInput
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('h')
    );
}

#[test]
fn test_lexer_anchor_end_of_input() {
    // \Z outside char class = EndOfInput anchor
    let mut lexer = Lexer::new(r"hello\Z");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('h')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('e')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('l')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('l')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('o')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::EndOfInput
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Eof
    );
}

#[test]
fn test_lexer_anchor_end_of_input_strict() {
    // \z outside char class = EndOfInputStrict anchor
    let mut lexer = Lexer::new(r"hello\z");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('h')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('e')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('l')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('l')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('o')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::EndOfInputStrict
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Eof
    );
}

#[test]
fn test_lexer_caret_in_char_class() {
    // ^ inside char class is negation (Caret), not StartOfLine
    let mut lexer = Lexer::new("[^abc]");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassStart
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Caret
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
}

#[test]
fn test_lexer_anchor_escapes_in_char_class() {
    // \A, \Z, \z inside char class should remain phonetic shortcuts
    let mut lexer = Lexer::new(r"[\A\Z\z]");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassStart
    );
    // \A inside = affricate (negated)
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "affricate",
            negated: true
        }
    );
    // \Z inside = nasal (negated)
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "nasal",
            negated: true
        }
    );
    // \z inside = nasal (non-negated)
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::PhoneticShortcut {
            class_name: "nasal",
            negated: false
        }
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::CharClassEnd
    );
}

#[test]
fn test_lexer_dollar_symbol_vs_anchor() {
    // $NAME = SymbolRef, $ alone or before non-identifier = EndOfLine
    let mut lexer = Lexer::new("$VOWEL");
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::SymbolRef("VOWEL".to_string())
    );

    let mut lexer2 = Lexer::new("$");
    assert_eq!(
        lexer2
            .next_token()
            .expect("test: lexer2.next_token must be Ok"),
        Token::EndOfLine
    );

    let mut lexer3 = Lexer::new("$ ");
    assert_eq!(
        lexer3
            .next_token()
            .expect("test: lexer3.next_token must be Ok"),
        Token::EndOfLine
    );
}

// ========================================================================
// Multiline and Dotall Flag Tests
// ========================================================================

#[test]
fn test_lexer_multiline_flag() {
    let mut lexer = Lexer::new("(?m)^line$");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.multiline, Some(true));
            assert_eq!(flags.dotall, None);
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::StartOfLine
    );
}

#[test]
fn test_lexer_dotall_flag() {
    let mut lexer = Lexer::new("(?s).*");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.dotall, Some(true));
            assert_eq!(flags.multiline, None);
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Dot
    );
}

#[test]
fn test_lexer_multiline_dotall_combined() {
    let mut lexer = Lexer::new("(?ms)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.multiline, Some(true));
            assert_eq!(flags.dotall, Some(true));
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
}

#[test]
fn test_lexer_multiline_negated() {
    let mut lexer = Lexer::new("(?-m)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.multiline, Some(false));
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
}

#[test]
fn test_lexer_dotall_negated() {
    let mut lexer = Lexer::new("(?-s)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.dotall, Some(false));
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
}

#[test]
fn test_lexer_scoped_multiline() {
    let mut lexer = Lexer::new("(?m:^line$)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::ScopedFlagsStart(flags) => {
            assert_eq!(flags.multiline, Some(true));
        }
        _ => panic!("Expected ScopedFlagsStart, got {:?}", token),
    }
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::StartOfLine
    );
}

#[test]
fn test_lexer_all_flags_combined() {
    let mut lexer = Lexer::new("(?ims)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.case_insensitive, Some(true));
            assert_eq!(flags.multiline, Some(true));
            assert_eq!(flags.dotall, Some(true));
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
}

#[test]
fn test_lexer_levenshtein_distance_inline() {
    // (?;2) - inline distance only
    let mut lexer = Lexer::new("(?;2)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.levenshtein_distance, Some(2));
            assert_eq!(flags.case_insensitive, None);
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
}

#[test]
fn test_lexer_levenshtein_distance_with_flags() {
    // (?i;0) - case-insensitive with distance 0
    let mut lexer = Lexer::new("(?i;0)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.case_insensitive, Some(true));
            assert_eq!(flags.levenshtein_distance, Some(0));
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
}

#[test]
fn test_lexer_levenshtein_distance_scoped() {
    // (?;1:...) - scoped distance only
    let mut lexer = Lexer::new("(?;1:abc)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::ScopedFlagsStart(flags) => {
            assert_eq!(flags.levenshtein_distance, Some(1));
            assert_eq!(flags.case_insensitive, None);
        }
        _ => panic!("Expected ScopedFlagsStart, got {:?}", token),
    }
    // Continue reading "abc)"
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('a')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('b')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::Char('c')
    );
    assert_eq!(
        lexer
            .next_token()
            .expect("test: lexer.next_token must be Ok"),
        Token::GroupEnd
    );
}

#[test]
fn test_lexer_levenshtein_distance_with_flags_scoped() {
    // (?i;0:...) - case-insensitive with distance 0, scoped
    let mut lexer = Lexer::new("(?i;0:rpo)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::ScopedFlagsStart(flags) => {
            assert_eq!(flags.case_insensitive, Some(true));
            assert_eq!(flags.levenshtein_distance, Some(0));
        }
        _ => panic!("Expected ScopedFlagsStart, got {:?}", token),
    }
}

#[test]
fn test_lexer_levenshtein_distance_multi_digit() {
    // (?;255) - max distance value
    let mut lexer = Lexer::new("(?;255)");
    let token = lexer
        .next_token()
        .expect("test: lexer.next_token must be Ok");
    match token {
        Token::InlineFlags(flags) => {
            assert_eq!(flags.levenshtein_distance, Some(255));
        }
        _ => panic!("Expected InlineFlags, got {:?}", token),
    }
}

#[test]
fn test_lexer_levenshtein_distance_missing_number() {
    // (?;) - error: missing distance number
    let mut lexer = Lexer::new("(?;)");
    let result = lexer.next_token();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("expected distance number"));
}

#[test]
fn test_lexer_levenshtein_distance_overflow() {
    // (?;256) - error: distance out of range (u8 max is 255)
    let mut lexer = Lexer::new("(?;256)");
    let result = lexer.next_token();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("out of range"));
}
