// ============================================================================
// PhonePattern - Single char or digraph
// ============================================================================

/// A phonetic class element - supports single characters through arbitrary-length sequences.
///
/// Phonetic classes can contain:
/// - Single Unicode characters (including IPA): `Char('ʃ')`
/// - Digraphs (2 characters): `Digraph('s', 'h')` for "sh"
/// - Trigraphs (3 characters): `Trigraph('t', 's', 'ʼ')` for ejective affricate t͡sʼ
/// - Tetragraphs (4 characters): `Tetragraph('ŋ', 'ɡ', 'ǀ', 'ʰ')` for prenasalized aspirated click
/// - Pentagraphs (5 characters): `Pentagraph(...)` for prenasalized labialized clicks
/// - Hexagraphs (6 characters): `Hexagraph(...)` for prenasalized labialized ejective affricates
/// - Heptagraphs (7 characters): `Heptagraph(...)` for theoretical maximum complex phonemes
/// - Sequences (8+ characters): `Sequence(vec![...])` for rare longer patterns
///
/// Fixed-size variants (Char through Heptagraph) avoid heap allocation.
/// Sequence uses `Vec<char>` for rare 8+ character patterns.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PhonePattern {
    /// Single character (e.g., 'a', 'ʃ', 'θ')
    Char(char),
    /// Two-character digraph (e.g., ('s', 'h') for "sh")
    Digraph(char, char),
    /// Three-character trigraph (e.g., ('t', 's', 'ʼ') for ejective affricate t͡sʼ)
    Trigraph(char, char, char),
    /// Four-character tetragraph (e.g., ('ŋ', 'ɡ', 'ǀ', 'ʰ') for prenasalized aspirated click)
    Tetragraph(char, char, char, char),
    /// Five-character pentagraph (e.g., prenasalized labialized click)
    Pentagraph(char, char, char, char, char),
    /// Six-character hexagraph (e.g., prenasalized labialized ejective affricate)
    Hexagraph(char, char, char, char, char, char),
    /// Seven-character heptagraph (theoretical maximum for complex phonemes)
    Heptagraph(char, char, char, char, char, char, char),
    /// Arbitrary-length sequence (8+ characters) - heap allocated
    Sequence(Vec<char>),
}

impl PhonePattern {
    /// Create a single character pattern.
    pub const fn char(c: char) -> Self {
        PhonePattern::Char(c)
    }

    /// Create a digraph pattern.
    pub const fn digraph(c1: char, c2: char) -> Self {
        PhonePattern::Digraph(c1, c2)
    }

    /// Create a trigraph pattern (for ejective affricates, etc.).
    pub const fn trigraph(c1: char, c2: char, c3: char) -> Self {
        PhonePattern::Trigraph(c1, c2, c3)
    }

    /// Check if this pattern matches a single character.
    pub fn matches_char(&self, c: char) -> bool {
        matches!(self, PhonePattern::Char(pc) if *pc == c)
    }

    /// Check if this pattern matches a two-character sequence.
    pub fn matches_digraph(&self, c1: char, c2: char) -> bool {
        matches!(self, PhonePattern::Digraph(d1, d2) if *d1 == c1 && *d2 == c2)
    }

    /// Check if this pattern matches a three-character sequence.
    pub fn matches_trigraph(&self, c1: char, c2: char, c3: char) -> bool {
        matches!(self, PhonePattern::Trigraph(t1, t2, t3) if *t1 == c1 && *t2 == c2 && *t3 == c3)
    }

    /// Returns true if this is a single character.
    pub fn is_char(&self) -> bool {
        matches!(self, PhonePattern::Char(_))
    }

    /// Returns true if this is a digraph.
    pub fn is_digraph(&self) -> bool {
        matches!(self, PhonePattern::Digraph(_, _))
    }

    /// Returns true if this is a trigraph.
    pub fn is_trigraph(&self) -> bool {
        matches!(self, PhonePattern::Trigraph(_, _, _))
    }

    /// Get the character if this is a single char pattern.
    pub fn as_char(&self) -> Option<char> {
        match self {
            PhonePattern::Char(c) => Some(*c),
            _ => None,
        }
    }

    /// Get the digraph characters if this is a digraph pattern.
    pub fn as_digraph(&self) -> Option<(char, char)> {
        match self {
            PhonePattern::Digraph(c1, c2) => Some((*c1, *c2)),
            _ => None,
        }
    }

    /// Get the trigraph characters if this is a trigraph pattern.
    pub fn as_trigraph(&self) -> Option<(char, char, char)> {
        match self {
            PhonePattern::Trigraph(c1, c2, c3) => Some((*c1, *c2, *c3)),
            _ => None,
        }
    }

    /// Create a tetragraph pattern (for prenasalized aspirated clicks, etc.).
    pub const fn tetragraph(c1: char, c2: char, c3: char, c4: char) -> Self {
        PhonePattern::Tetragraph(c1, c2, c3, c4)
    }

    /// Create a pentagraph pattern (for 5-character patterns).
    pub const fn pentagraph(c1: char, c2: char, c3: char, c4: char, c5: char) -> Self {
        PhonePattern::Pentagraph(c1, c2, c3, c4, c5)
    }

    /// Create a hexagraph pattern (for 6-character patterns).
    pub const fn hexagraph(c1: char, c2: char, c3: char, c4: char, c5: char, c6: char) -> Self {
        PhonePattern::Hexagraph(c1, c2, c3, c4, c5, c6)
    }

    /// Create a heptagraph pattern (for 7-character patterns).
    pub const fn heptagraph(
        c1: char,
        c2: char,
        c3: char,
        c4: char,
        c5: char,
        c6: char,
        c7: char,
    ) -> Self {
        PhonePattern::Heptagraph(c1, c2, c3, c4, c5, c6, c7)
    }

    /// Create a sequence pattern (for 8+ character patterns).
    pub fn sequence(chars: Vec<char>) -> Self {
        debug_assert!(chars.len() >= 8, "Sequence should have 8+ characters; use fixed-size variants for shorter patterns");
        PhonePattern::Sequence(chars)
    }

    /// Check if this pattern matches a four-character sequence.
    pub fn matches_tetragraph(&self, c1: char, c2: char, c3: char, c4: char) -> bool {
        matches!(self, PhonePattern::Tetragraph(t1, t2, t3, t4) if *t1 == c1 && *t2 == c2 && *t3 == c3 && *t4 == c4)
    }

    /// Check if this pattern matches a five-character sequence.
    pub fn matches_pentagraph(&self, c1: char, c2: char, c3: char, c4: char, c5: char) -> bool {
        matches!(self, PhonePattern::Pentagraph(p1, p2, p3, p4, p5)
            if *p1 == c1 && *p2 == c2 && *p3 == c3 && *p4 == c4 && *p5 == c5)
    }

    /// Check if this pattern matches a six-character sequence.
    pub fn matches_hexagraph(
        &self,
        c1: char,
        c2: char,
        c3: char,
        c4: char,
        c5: char,
        c6: char,
    ) -> bool {
        matches!(self, PhonePattern::Hexagraph(h1, h2, h3, h4, h5, h6)
            if *h1 == c1 && *h2 == c2 && *h3 == c3 && *h4 == c4 && *h5 == c5 && *h6 == c6)
    }

    /// Check if this pattern matches a seven-character sequence.
    pub fn matches_heptagraph(
        &self,
        c1: char,
        c2: char,
        c3: char,
        c4: char,
        c5: char,
        c6: char,
        c7: char,
    ) -> bool {
        matches!(self, PhonePattern::Heptagraph(h1, h2, h3, h4, h5, h6, h7)
            if *h1 == c1 && *h2 == c2 && *h3 == c3 && *h4 == c4 && *h5 == c5 && *h6 == c6 && *h7 == c7)
    }

    /// Check if this pattern matches an arbitrary-length sequence.
    pub fn matches_sequence(&self, chars: &[char]) -> bool {
        match self {
            PhonePattern::Sequence(s) => s.as_slice() == chars,
            _ => false,
        }
    }

    /// Returns true if this is a tetragraph.
    pub fn is_tetragraph(&self) -> bool {
        matches!(self, PhonePattern::Tetragraph(_, _, _, _))
    }

    /// Returns true if this is a pentagraph.
    pub fn is_pentagraph(&self) -> bool {
        matches!(self, PhonePattern::Pentagraph(_, _, _, _, _))
    }

    /// Returns true if this is a hexagraph.
    pub fn is_hexagraph(&self) -> bool {
        matches!(self, PhonePattern::Hexagraph(_, _, _, _, _, _))
    }

    /// Returns true if this is a heptagraph.
    pub fn is_heptagraph(&self) -> bool {
        matches!(self, PhonePattern::Heptagraph(_, _, _, _, _, _, _))
    }

    /// Returns true if this is a sequence.
    pub fn is_sequence(&self) -> bool {
        matches!(self, PhonePattern::Sequence(_))
    }

    /// Get the tetragraph characters if this is a tetragraph pattern.
    pub fn as_tetragraph(&self) -> Option<(char, char, char, char)> {
        match self {
            PhonePattern::Tetragraph(c1, c2, c3, c4) => Some((*c1, *c2, *c3, *c4)),
            _ => None,
        }
    }

    /// Get the pentagraph characters if this is a pentagraph pattern.
    pub fn as_pentagraph(&self) -> Option<(char, char, char, char, char)> {
        match self {
            PhonePattern::Pentagraph(c1, c2, c3, c4, c5) => Some((*c1, *c2, *c3, *c4, *c5)),
            _ => None,
        }
    }

    /// Get the hexagraph characters if this is a hexagraph pattern.
    pub fn as_hexagraph(&self) -> Option<(char, char, char, char, char, char)> {
        match self {
            PhonePattern::Hexagraph(c1, c2, c3, c4, c5, c6) => Some((*c1, *c2, *c3, *c4, *c5, *c6)),
            _ => None,
        }
    }

    /// Get the heptagraph characters if this is a heptagraph pattern.
    pub fn as_heptagraph(&self) -> Option<(char, char, char, char, char, char, char)> {
        match self {
            PhonePattern::Heptagraph(c1, c2, c3, c4, c5, c6, c7) => {
                Some((*c1, *c2, *c3, *c4, *c5, *c6, *c7))
            }
            _ => None,
        }
    }

    /// Get the sequence if this is a sequence pattern.
    pub fn as_sequence(&self) -> Option<&[char]> {
        match self {
            PhonePattern::Sequence(s) => Some(s.as_slice()),
            _ => None,
        }
    }

    /// Get the length of this pattern in characters.
    pub fn len(&self) -> usize {
        match self {
            PhonePattern::Char(_) => 1,
            PhonePattern::Digraph(_, _) => 2,
            PhonePattern::Trigraph(_, _, _) => 3,
            PhonePattern::Tetragraph(_, _, _, _) => 4,
            PhonePattern::Pentagraph(_, _, _, _, _) => 5,
            PhonePattern::Hexagraph(_, _, _, _, _, _) => 6,
            PhonePattern::Heptagraph(_, _, _, _, _, _, _) => 7,
            PhonePattern::Sequence(s) => s.len(),
        }
    }

    /// Returns true if this pattern is empty (only possible for empty Sequence).
    pub fn is_empty(&self) -> bool {
        matches!(self, PhonePattern::Sequence(s) if s.is_empty())
    }
}

impl std::fmt::Display for PhonePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhonePattern::Char(c) => write!(f, "{}", c),
            PhonePattern::Digraph(c1, c2) => write!(f, "{}{}", c1, c2),
            PhonePattern::Trigraph(c1, c2, c3) => write!(f, "{}{}{}", c1, c2, c3),
            PhonePattern::Tetragraph(c1, c2, c3, c4) => write!(f, "{}{}{}{}", c1, c2, c3, c4),
            PhonePattern::Pentagraph(c1, c2, c3, c4, c5) => {
                write!(f, "{}{}{}{}{}", c1, c2, c3, c4, c5)
            }
            PhonePattern::Hexagraph(c1, c2, c3, c4, c5, c6) => {
                write!(f, "{}{}{}{}{}{}", c1, c2, c3, c4, c5, c6)
            }
            PhonePattern::Heptagraph(c1, c2, c3, c4, c5, c6, c7) => {
                write!(f, "{}{}{}{}{}{}{}", c1, c2, c3, c4, c5, c6, c7)
            }
            PhonePattern::Sequence(s) => {
                for c in s {
                    write!(f, "{}", c)?;
                }
                Ok(())
            }
        }
    }
}
