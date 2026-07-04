//! Character-level draft buffer with rollback support.
//!
//! This module implements a buffer for tracking tentative text input with
//! efficient character-level insertion and deletion (backspace) operations.

/// Buffer for managing draft text with character-level operations.
///
/// DraftBuffer stores text as contiguous UTF-8 while tracking length in
/// characters, making forward typing, backspace, and string materialization
/// efficient for editor-style drafts.
///
/// # Memory Efficiency
///
/// - Small allocations: ~32 bytes base + UTF-8 byte storage
/// - ASCII drafts use 1 byte per character instead of 4 bytes per `char`
/// - No allocations for backspace (just decrements length)
///
/// # Use Cases
///
/// - Code editor: Track partial identifier as user types
/// - Autocomplete: Build query string incrementally
/// - Undo/redo: Checkpoint and restore buffer state
///
/// # Examples
///
/// ```
/// use liblevenshtein::contextual::DraftBuffer;
///
/// let mut buffer = DraftBuffer::new();
///
/// // User types "he"
/// buffer.insert('h');
/// buffer.insert('e');
/// assert_eq!(buffer.as_str(), "he");
///
/// // User types "l"
/// buffer.insert('l');
/// assert_eq!(buffer.as_str(), "hel");
///
/// // User hits backspace
/// assert_eq!(buffer.delete(), Some('l'));
/// assert_eq!(buffer.as_str(), "he");
/// ```
#[derive(Debug, Clone)]
pub struct DraftBuffer {
    /// UTF-8 draft text.
    text: String,
    /// Cached character length for O(1) checkpoint creation.
    char_len: usize,
}

impl DraftBuffer {
    /// Create a new empty draft buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DraftBuffer;
    ///
    /// let buffer = DraftBuffer::new();
    /// assert_eq!(buffer.len(), 0);
    /// assert!(buffer.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            text: String::new(),
            char_len: 0,
        }
    }

    /// Create a draft buffer with the given initial capacity.
    ///
    /// This avoids reallocation if you know the approximate size.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity in characters
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DraftBuffer;
    ///
    /// // Preallocate for typical identifier length
    /// let buffer = DraftBuffer::with_capacity(32);
    /// assert!(buffer.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            text: String::with_capacity(capacity.saturating_mul(4)),
            char_len: 0,
        }
    }

    /// Create a draft buffer from an existing string.
    ///
    /// # Arguments
    ///
    /// * `s` - Initial string content
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DraftBuffer;
    ///
    /// let buffer = DraftBuffer::from_string("hello");
    /// assert_eq!(buffer.as_str(), "hello");
    /// assert_eq!(buffer.len(), 5);
    /// ```
    pub fn from_string(s: &str) -> Self {
        Self {
            text: s.to_owned(),
            char_len: s.chars().count(),
        }
    }

    fn from_owned_string(text: String) -> Self {
        let char_len = text.chars().count();
        Self { text, char_len }
    }

    /// Insert a character at the end of the buffer.
    ///
    /// # Arguments
    ///
    /// * `ch` - Character to insert
    ///
    /// # Performance
    ///
    /// O(1) amortized. May trigger reallocation if capacity exceeded.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DraftBuffer;
    ///
    /// let mut buffer = DraftBuffer::new();
    /// buffer.insert('a');
    /// buffer.insert('b');
    /// assert_eq!(buffer.as_str(), "ab");
    /// ```
    pub fn insert(&mut self, ch: char) {
        self.text.push(ch);
        self.char_len += 1;
    }

    /// Insert a string at the end of the buffer.
    ///
    /// # Arguments
    ///
    /// * `s` - String slice to append
    ///
    /// # Performance
    ///
    /// O(n) in the inserted string length. This performs one append into the
    /// underlying UTF-8 buffer and updates the cached character length.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DraftBuffer;
    ///
    /// let mut buffer = DraftBuffer::new();
    /// buffer.insert_str("hello");
    /// buffer.insert_str(" 世界");
    /// assert_eq!(buffer.as_str(), "hello 世界");
    /// assert_eq!(buffer.len(), 8);
    /// ```
    pub fn insert_str(&mut self, s: &str) {
        self.text.push_str(s);
        self.char_len += s.chars().count();
    }

    /// Delete the last character from the buffer (backspace).
    ///
    /// # Returns
    ///
    /// `Some(ch)` if a character was deleted, `None` if buffer was empty.
    ///
    /// # Performance
    ///
    /// O(1). No allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DraftBuffer;
    ///
    /// let mut buffer = DraftBuffer::from_string("test");
    /// assert_eq!(buffer.delete(), Some('t'));
    /// assert_eq!(buffer.delete(), Some('s'));
    /// assert_eq!(buffer.as_str(), "te");
    /// assert_eq!(buffer.len(), 2);
    /// ```
    pub fn delete(&mut self) -> Option<char> {
        let ch = self.text.pop()?;
        self.char_len -= 1;
        Some(ch)
    }

    /// Get the buffer length in characters.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DraftBuffer;
    ///
    /// let buffer = DraftBuffer::from_string("hello");
    /// assert_eq!(buffer.len(), 5);
    /// ```
    pub fn len(&self) -> usize {
        self.char_len
    }

    /// Check if the buffer is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DraftBuffer;
    ///
    /// let buffer = DraftBuffer::new();
    /// assert!(buffer.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.char_len == 0
    }

    /// Get the buffer content as a borrowed string slice.
    ///
    /// # Performance
    ///
    /// O(1). This borrows the underlying UTF-8 buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DraftBuffer;
    ///
    /// let buffer = DraftBuffer::from_string("test");
    /// assert_eq!(buffer.as_slice(), "test");
    /// ```
    pub fn as_slice(&self) -> &str {
        self.text.as_str()
    }

    /// Get the buffer content as an owned string.
    ///
    /// # Performance
    ///
    /// O(n) allocation to clone the underlying UTF-8 buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DraftBuffer;
    ///
    /// let buffer = DraftBuffer::from_string("test");
    /// assert_eq!(buffer.as_str(), "test");
    /// ```
    pub fn as_str(&self) -> String {
        self.text.clone()
    }

    /// Get the buffer content as a byte vector (UTF-8).
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DraftBuffer;
    ///
    /// let buffer = DraftBuffer::from_string("test");
    /// assert_eq!(buffer.as_bytes(), b"test");
    /// ```
    pub fn as_bytes(&self) -> Vec<u8> {
        self.text.as_bytes().to_vec()
    }

    /// Clear all content from the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DraftBuffer;
    ///
    /// let mut buffer = DraftBuffer::from_string("test");
    /// buffer.clear();
    /// assert!(buffer.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.text.clear();
        self.char_len = 0;
    }

    /// Truncate the buffer to the specified length.
    ///
    /// If `len` is greater than the current length, this has no effect.
    ///
    /// # Arguments
    ///
    /// * `len` - Target length in characters
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::DraftBuffer;
    ///
    /// let mut buffer = DraftBuffer::from_string("hello");
    /// buffer.truncate(3);
    /// assert_eq!(buffer.as_str(), "hel");
    /// ```
    pub fn truncate(&mut self, len: usize) {
        if len >= self.char_len {
            return;
        }
        if len == 0 {
            self.clear();
            return;
        }

        let byte_len = self
            .text
            .char_indices()
            .nth(len)
            .map(|(index, _)| index)
            .unwrap_or(self.text.len());
        self.text.truncate(byte_len);
        self.char_len = len;
    }
}

impl Default for DraftBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl From<String> for DraftBuffer {
    fn from(s: String) -> Self {
        Self::from_owned_string(s)
    }
}

impl From<&str> for DraftBuffer {
    fn from(s: &str) -> Self {
        Self::from_string(s)
    }
}

impl std::fmt::Display for DraftBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_slice())
    }
}

impl AsRef<str> for DraftBuffer {
    fn as_ref(&self) -> &str {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let buffer = DraftBuffer::new();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert_eq!(buffer.as_str(), "");
    }

    #[test]
    fn test_insert() {
        let mut buffer = DraftBuffer::new();
        buffer.insert('a');
        buffer.insert('b');
        buffer.insert('c');
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.as_str(), "abc");
    }

    #[test]
    fn test_insert_str() {
        let mut buffer = DraftBuffer::new();
        buffer.insert_str("hello");
        buffer.insert_str(" 世界");
        assert_eq!(buffer.len(), 8);
        assert_eq!(buffer.as_str(), "hello 世界");
    }

    #[test]
    fn test_delete() {
        let mut buffer = DraftBuffer::from_string("test");
        assert_eq!(buffer.delete(), Some('t'));
        assert_eq!(buffer.as_str(), "tes");
        assert_eq!(buffer.delete(), Some('s'));
        assert_eq!(buffer.as_str(), "te");
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_delete_empty() {
        let mut buffer = DraftBuffer::new();
        assert_eq!(buffer.delete(), None);
    }

    #[test]
    fn test_from_str() {
        let buffer = DraftBuffer::from_string("hello");
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.as_str(), "hello");
    }

    #[test]
    fn test_clear() {
        let mut buffer = DraftBuffer::from_string("test");
        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.as_str(), "");
    }

    #[test]
    fn test_truncate() {
        let mut buffer = DraftBuffer::from_string("hello");
        buffer.truncate(3);
        assert_eq!(buffer.as_str(), "hel");
        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn test_truncate_longer() {
        let mut buffer = DraftBuffer::from_string("hi");
        buffer.truncate(10);
        assert_eq!(buffer.as_str(), "hi");
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_unicode() {
        let mut buffer = DraftBuffer::new();
        buffer.insert('😀');
        buffer.insert('世');
        buffer.insert('界');
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.as_str(), "😀世界");
        assert_eq!(buffer.delete(), Some('界'));
        assert_eq!(buffer.as_str(), "😀世");
    }

    #[test]
    fn test_truncate_unicode_boundary() {
        let mut buffer = DraftBuffer::from_string("é😀ab");
        buffer.truncate(2);
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.as_str(), "é😀");
        assert_eq!(buffer.delete(), Some('😀'));
        assert_eq!(buffer.as_str(), "é");
    }

    #[test]
    fn test_as_bytes() {
        let buffer = DraftBuffer::from_string("test");
        assert_eq!(buffer.as_bytes(), b"test");
    }

    #[test]
    fn test_as_bytes_unicode() {
        let buffer = DraftBuffer::from_string("é😀");
        assert_eq!(buffer.as_bytes(), "é😀".as_bytes());
    }

    #[test]
    fn test_as_slice() {
        let buffer = DraftBuffer::from_string("test");
        assert_eq!(buffer.as_slice(), "test");
        assert_eq!(buffer.as_ref(), "test");
    }

    #[test]
    fn test_display() {
        let buffer = DraftBuffer::from_string("test");
        assert_eq!(format!("{}", buffer), "test");
    }

    #[test]
    fn test_from_string() {
        let buffer = DraftBuffer::from(String::from("hello"));
        assert_eq!(buffer.as_str(), "hello");
    }

    #[test]
    fn test_with_capacity() {
        let buffer = DraftBuffer::with_capacity(100);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_incremental_typing() {
        let mut buffer = DraftBuffer::new();

        // Simulate typing "hello"
        for ch in "hello".chars() {
            buffer.insert(ch);
        }
        assert_eq!(buffer.as_str(), "hello");

        // Simulate backspace twice
        buffer.delete();
        buffer.delete();
        assert_eq!(buffer.as_str(), "hel");

        // Continue typing "p"
        buffer.insert('p');
        assert_eq!(buffer.as_str(), "help");
    }
}
