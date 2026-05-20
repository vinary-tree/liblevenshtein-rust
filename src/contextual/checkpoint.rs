//! Checkpoint system for undo/redo operations.
//!
//! This module provides checkpoint management for editor undo/redo support,
//! allowing the draft buffer state to be saved and restored.

use super::DraftBuffer;

/// A checkpoint capturing the state of a draft buffer at a point in time.
///
/// Checkpoints are lightweight snapshots that enable undo/redo functionality
/// by storing only the buffer length (not the full content). The buffer itself
/// can be truncated to restore to a checkpoint.
///
/// # Memory Efficiency
///
/// - Checkpoint: 8 bytes (just a usize length)
/// - No content duplication (uses buffer's internal storage)
/// - Undo stack: ~8 bytes per checkpoint
///
/// # Use Cases
///
/// - **Editor undo**: Checkpoint before significant edits
/// - **Transaction boundaries**: Mark points that can be rolled back
/// - **Speculative execution**: Try operations, rollback if failed
///
/// # Examples
///
/// ```
/// use liblevenshtein::contextual::{DraftBuffer, Checkpoint};
///
/// let mut buffer = DraftBuffer::new();
/// buffer.insert('h');
/// buffer.insert('e');
///
/// // Save checkpoint
/// let checkpoint = Checkpoint::from_buffer(&buffer);
/// assert_eq!(checkpoint.position(), 2);
///
/// // Continue typing
/// buffer.insert('l');
/// buffer.insert('l');
/// buffer.insert('o');
/// assert_eq!(buffer.as_str(), "hello");
///
/// // Restore to checkpoint (undo)
/// checkpoint.restore(&mut buffer);
/// assert_eq!(buffer.as_str(), "he");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Checkpoint {
    /// Position in the buffer (length in characters)
    position: usize,
}

impl Checkpoint {
    /// Create a checkpoint from the current buffer state.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The draft buffer to checkpoint
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::{DraftBuffer, Checkpoint};
    ///
    /// let buffer = DraftBuffer::from_string("test");
    /// let checkpoint = Checkpoint::from_buffer(&buffer);
    /// assert_eq!(checkpoint.position(), 4);
    /// ```
    pub fn from_buffer(buffer: &DraftBuffer) -> Self {
        Self {
            position: buffer.len(),
        }
    }

    /// Create a checkpoint at a specific position.
    ///
    /// # Arguments
    ///
    /// * `position` - Character position (length)
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::Checkpoint;
    ///
    /// let checkpoint = Checkpoint::at(5);
    /// assert_eq!(checkpoint.position(), 5);
    /// ```
    pub fn at(position: usize) -> Self {
        Self { position }
    }

    /// Get the checkpoint position (buffer length).
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::Checkpoint;
    ///
    /// let checkpoint = Checkpoint::at(10);
    /// assert_eq!(checkpoint.position(), 10);
    /// ```
    pub fn position(&self) -> usize {
        self.position
    }

    /// Restore the buffer to this checkpoint.
    ///
    /// Truncates the buffer to the checkpoint position, effectively
    /// discarding any characters added after the checkpoint was created.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The draft buffer to restore
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::{DraftBuffer, Checkpoint};
    ///
    /// let mut buffer = DraftBuffer::from_string("hello");
    /// let checkpoint = Checkpoint::at(3);
    ///
    /// checkpoint.restore(&mut buffer);
    /// assert_eq!(buffer.as_str(), "hel");
    /// ```
    pub fn restore(&self, buffer: &mut DraftBuffer) {
        buffer.truncate(self.position);
    }
}

/// Stack-based checkpoint manager for undo/redo operations.
///
/// Manages a stack of checkpoints with efficient push/pop operations.
/// Uses a Vec for O(1) amortized push/pop.
///
/// # Memory Efficiency
///
/// - Base: ~24 bytes (Vec overhead)
/// - Per checkpoint: 8 bytes
/// - Typical undo stack (50 entries): ~424 bytes
///
/// # Examples
///
/// ```
/// use liblevenshtein::contextual::{DraftBuffer, CheckpointStack};
///
/// let mut buffer = DraftBuffer::new();
/// let mut stack = CheckpointStack::new();
///
/// // Type and checkpoint
/// buffer.insert('h');
/// buffer.insert('e');
/// stack.push_from_buffer(&buffer);
///
/// buffer.insert('l');
/// buffer.insert('l');
/// buffer.insert('o');
/// assert_eq!(buffer.as_str(), "hello");
///
/// // Undo
/// if let Some(checkpoint) = stack.pop() {
///     checkpoint.restore(&mut buffer);
///     assert_eq!(buffer.as_str(), "he");
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CheckpointStack {
    /// Stack of checkpoints (most recent at end)
    checkpoints: Vec<Checkpoint>,
}

impl CheckpointStack {
    /// Create a new empty checkpoint stack.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::CheckpointStack;
    ///
    /// let stack = CheckpointStack::new();
    /// assert!(stack.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            checkpoints: Vec::new(),
        }
    }

    /// Create a checkpoint stack with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity for checkpoints
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::CheckpointStack;
    ///
    /// let stack = CheckpointStack::with_capacity(50);
    /// assert!(stack.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            checkpoints: Vec::with_capacity(capacity),
        }
    }

    /// Push a checkpoint onto the stack.
    ///
    /// # Arguments
    ///
    /// * `checkpoint` - The checkpoint to push
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::{Checkpoint, CheckpointStack};
    ///
    /// let mut stack = CheckpointStack::new();
    /// let checkpoint = Checkpoint::at(5);
    /// stack.push(checkpoint);
    /// assert_eq!(stack.len(), 1);
    /// ```
    pub fn push(&mut self, checkpoint: Checkpoint) {
        self.checkpoints.push(checkpoint);
    }

    /// Create a checkpoint from the buffer and push it onto the stack.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The draft buffer to checkpoint
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::{DraftBuffer, CheckpointStack};
    ///
    /// let buffer = DraftBuffer::from_string("test");
    /// let mut stack = CheckpointStack::new();
    /// stack.push_from_buffer(&buffer);
    /// assert_eq!(stack.len(), 1);
    /// ```
    pub fn push_from_buffer(&mut self, buffer: &DraftBuffer) {
        self.push(Checkpoint::from_buffer(buffer));
    }

    /// Pop the most recent checkpoint from the stack.
    ///
    /// # Returns
    ///
    /// `Some(checkpoint)` if the stack is not empty, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::{Checkpoint, CheckpointStack};
    ///
    /// let mut stack = CheckpointStack::new();
    /// stack.push(Checkpoint::at(5));
    ///
    /// let checkpoint = stack.pop();
    /// assert_eq!(checkpoint.expect("doc example: pop on non-empty stack").position(), 5);
    /// assert!(stack.is_empty());
    /// ```
    pub fn pop(&mut self) -> Option<Checkpoint> {
        self.checkpoints.pop()
    }

    /// Peek at the most recent checkpoint without removing it.
    ///
    /// # Returns
    ///
    /// `Some(&checkpoint)` if the stack is not empty, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::{Checkpoint, CheckpointStack};
    ///
    /// let mut stack = CheckpointStack::new();
    /// stack.push(Checkpoint::at(5));
    ///
    /// assert_eq!(stack.peek().expect("test/doc fixture: peek on non-empty stack").position(), 5);
    /// assert_eq!(stack.len(), 1); // Still on stack
    /// ```
    pub fn peek(&self) -> Option<&Checkpoint> {
        self.checkpoints.last()
    }

    /// Get the number of checkpoints in the stack.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::{Checkpoint, CheckpointStack};
    ///
    /// let mut stack = CheckpointStack::new();
    /// assert_eq!(stack.len(), 0);
    /// stack.push(Checkpoint::at(5));
    /// assert_eq!(stack.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Check if the stack is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::CheckpointStack;
    ///
    /// let stack = CheckpointStack::new();
    /// assert!(stack.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }

    /// Clear all checkpoints from the stack.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::{Checkpoint, CheckpointStack};
    ///
    /// let mut stack = CheckpointStack::new();
    /// stack.push(Checkpoint::at(5));
    /// stack.clear();
    /// assert!(stack.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.checkpoints.clear();
    }
}

impl Default for CheckpointStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_from_buffer() {
        let buffer = DraftBuffer::from_string("test");
        let checkpoint = Checkpoint::from_buffer(&buffer);
        assert_eq!(checkpoint.position(), 4);
    }

    #[test]
    fn test_checkpoint_at() {
        let checkpoint = Checkpoint::at(10);
        assert_eq!(checkpoint.position(), 10);
    }

    #[test]
    fn test_checkpoint_restore() {
        let mut buffer = DraftBuffer::from_string("hello");
        let checkpoint = Checkpoint::at(3);
        checkpoint.restore(&mut buffer);
        assert_eq!(buffer.as_str(), "hel");
    }

    #[test]
    fn test_checkpoint_stack_new() {
        let stack = CheckpointStack::new();
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
    }

    #[test]
    fn test_checkpoint_stack_push_pop() {
        let mut stack = CheckpointStack::new();
        let checkpoint = Checkpoint::at(5);
        stack.push(checkpoint);
        assert_eq!(stack.len(), 1);

        let popped = stack.pop();
        assert_eq!(popped.expect("test fixture: pop on non-empty stack").position(), 5);
        assert!(stack.is_empty());
    }

    #[test]
    fn test_checkpoint_stack_peek() {
        let mut stack = CheckpointStack::new();
        stack.push(Checkpoint::at(5));

        assert_eq!(stack.peek().expect("test/doc fixture: peek on non-empty stack").position(), 5);
        assert_eq!(stack.len(), 1);
    }

    #[test]
    fn test_checkpoint_stack_from_buffer() {
        let buffer = DraftBuffer::from_string("test");
        let mut stack = CheckpointStack::new();
        stack.push_from_buffer(&buffer);
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.peek().expect("test/doc fixture: peek on non-empty stack").position(), 4);
    }

    #[test]
    fn test_checkpoint_stack_clear() {
        let mut stack = CheckpointStack::new();
        stack.push(Checkpoint::at(5));
        stack.push(Checkpoint::at(10));
        stack.clear();
        assert!(stack.is_empty());
    }

    #[test]
    fn test_undo_workflow() {
        let mut buffer = DraftBuffer::new();
        let mut undo_stack = CheckpointStack::new();

        // Start with empty checkpoint
        undo_stack.push_from_buffer(&buffer);

        // Type "hello"
        for ch in "hello".chars() {
            buffer.insert(ch);
            undo_stack.push_from_buffer(&buffer);
        }
        assert_eq!(buffer.as_str(), "hello");
        assert_eq!(undo_stack.len(), 6); // empty + 5 chars

        // Undo twice (remove "o" then "l")
        for _ in 0..2 {
            undo_stack.pop(); // Pop current state
            if let Some(checkpoint) = undo_stack.peek() {
                checkpoint.restore(&mut buffer);
            }
        }
        assert_eq!(buffer.as_str(), "hel");
        assert_eq!(undo_stack.len(), 4); // empty + h + e + l

        // Undo one more time
        undo_stack.pop();
        if let Some(checkpoint) = undo_stack.peek() {
            checkpoint.restore(&mut buffer);
        }
        assert_eq!(buffer.as_str(), "he");
        assert_eq!(undo_stack.len(), 3);
    }

    #[test]
    fn test_multiple_checkpoints() {
        let mut buffer = DraftBuffer::new();
        let mut stack = CheckpointStack::new();

        // Start with empty
        stack.push_from_buffer(&buffer);

        // Type "a", checkpoint
        buffer.insert('a');
        stack.push_from_buffer(&buffer);

        // Type "b", checkpoint
        buffer.insert('b');
        stack.push_from_buffer(&buffer);

        // Type "c", checkpoint
        buffer.insert('c');
        stack.push_from_buffer(&buffer);

        assert_eq!(buffer.as_str(), "abc");
        assert_eq!(stack.len(), 4); // empty, a, ab, abc

        // Undo to "ab" (pop abc, restore to ab)
        stack.pop();
        stack.peek().expect("test fixture: peek on non-empty stack").restore(&mut buffer);
        assert_eq!(buffer.as_str(), "ab");

        // Undo to "a" (pop ab, restore to a)
        stack.pop();
        stack.peek().expect("test fixture: peek on non-empty stack").restore(&mut buffer);
        assert_eq!(buffer.as_str(), "a");

        // Undo to "" (pop a, restore to empty)
        stack.pop();
        stack.peek().expect("test fixture: peek on non-empty stack").restore(&mut buffer);
        assert_eq!(buffer.as_str(), "");
    }
}
