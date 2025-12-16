//! Position tracking for error reporting.

use std::fmt;

/// Position in the input string where an error occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    /// Line number (1-indexed)
    pub line: usize,
    /// Column number (1-indexed)
    pub column: usize,
    /// Byte offset in the input
    pub offset: usize,
}

impl Position {
    /// Create a new position.
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self {
            line,
            column,
            offset,
        }
    }

    /// Create a position at the start of input.
    pub fn start() -> Self {
        Self {
            line: 1,
            column: 1,
            offset: 0,
        }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_display() {
        let pos = Position::new(5, 10, 42);
        assert_eq!(pos.to_string(), "line 5, column 10");
    }

    #[test]
    fn test_position_start() {
        let pos = Position::start();
        assert_eq!(pos.line, 1);
        assert_eq!(pos.column, 1);
        assert_eq!(pos.offset, 0);
    }

    #[test]
    fn test_position_equality() {
        let pos1 = Position::new(1, 1, 0);
        let pos2 = Position::start();
        assert_eq!(pos1, pos2);
    }

    #[test]
    fn test_position_clone() {
        let pos = Position::new(5, 10, 42);
        let cloned = pos;
        assert_eq!(pos, cloned);
    }
}
