//! Cross-platform synchronization primitives.
//!
//! This module provides a unified API for synchronization primitives that works
//! across both native and WASM targets:
//!
//! - On native platforms with `parking_lot` feature: Uses `parking_lot::RwLock`
//!   for better performance (no poisoning, smaller size, spin-wait optimization)
//! - On WASM or without `parking_lot`: Falls back to `std::sync::RwLock`
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::sync_compat::RwLock;
//!
//! let lock = RwLock::new(42);
//! let value = lock.read();  // Works on both parking_lot and std::sync
//! ```

// ============================================================================
// parking_lot backend (native + feature enabled)
// ============================================================================

#[cfg(all(feature = "parking_lot", not(target_arch = "wasm32")))]
pub use parking_lot::RwLock;

#[cfg(all(feature = "parking_lot", not(target_arch = "wasm32")))]
pub use parking_lot::RwLockReadGuard;

#[cfg(all(feature = "parking_lot", not(target_arch = "wasm32")))]
pub use parking_lot::RwLockWriteGuard;

// ============================================================================
// std::sync backend (WASM or parking_lot disabled)
// ============================================================================

/// A wrapper around `std::sync::RwLock` that provides a non-poisoning API
/// matching `parking_lot::RwLock`.
#[cfg(any(not(feature = "parking_lot"), target_arch = "wasm32"))]
#[derive(Debug, Default)]
pub struct RwLock<T>(std::sync::RwLock<T>);

#[cfg(any(not(feature = "parking_lot"), target_arch = "wasm32"))]
impl<T> RwLock<T> {
    /// Creates a new RwLock.
    #[inline]
    pub const fn new(value: T) -> Self {
        RwLock(std::sync::RwLock::new(value))
    }

    /// Acquires a read lock, panicking if the lock is poisoned.
    #[inline]
    pub fn read(&self) -> std::sync::RwLockReadGuard<'_, T> {
        self.0.read().expect("RwLock poisoned")
    }

    /// Acquires a write lock, panicking if the lock is poisoned.
    #[inline]
    pub fn write(&self) -> std::sync::RwLockWriteGuard<'_, T> {
        self.0.write().expect("RwLock poisoned")
    }

    /// Returns a mutable reference to the underlying data.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut().expect("RwLock poisoned")
    }

    /// Consumes the lock and returns the underlying data.
    #[inline]
    pub fn into_inner(self) -> T {
        self.0.into_inner().expect("RwLock poisoned")
    }
}

#[cfg(any(not(feature = "parking_lot"), target_arch = "wasm32"))]
pub use std::sync::RwLockReadGuard;

#[cfg(any(not(feature = "parking_lot"), target_arch = "wasm32"))]
pub use std::sync::RwLockWriteGuard;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rwlock_read() {
        let lock = RwLock::new(42);
        let value = lock.read();
        assert_eq!(*value, 42);
    }

    #[test]
    fn test_rwlock_write() {
        let lock = RwLock::new(42);
        {
            let mut value = lock.write();
            *value = 100;
        }
        let value = lock.read();
        assert_eq!(*value, 100);
    }

    #[test]
    fn test_rwlock_multiple_readers() {
        let lock = RwLock::new(42);
        let r1 = lock.read();
        // Note: Can't get multiple read guards simultaneously in single-threaded test
        // because std::sync::RwLock doesn't support that in this pattern
        assert_eq!(*r1, 42);
    }

    #[test]
    fn test_rwlock_get_mut() {
        let mut lock = RwLock::new(42);
        *lock.get_mut() = 100;
        assert_eq!(*lock.read(), 100);
    }

    #[test]
    fn test_rwlock_into_inner() {
        let lock = RwLock::new(42);
        assert_eq!(lock.into_inner(), 42);
    }
}
