use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[inline]
pub(crate) fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[inline]
pub(crate) fn read_lock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

#[inline]
pub(crate) fn write_lock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn lock_mutex_recovers_after_poison() {
        let mutex = Arc::new(Mutex::new(0));
        let thread_mutex = Arc::clone(&mutex);
        let result = std::thread::spawn(move || {
            *lock_mutex(&thread_mutex) = 42;
            panic!("intentional mutex poisoning for contextual locking test");
        })
        .join();

        assert!(result.is_err());
        assert_eq!(*lock_mutex(&mutex), 42);
    }

    #[test]
    fn rwlock_recovers_after_poison() {
        let lock = Arc::new(RwLock::new(0));
        let thread_lock = Arc::clone(&lock);
        let result = std::thread::spawn(move || {
            *write_lock(&thread_lock) = 42;
            panic!("intentional rwlock poisoning for contextual locking test");
        })
        .join();

        assert!(result.is_err());
        assert_eq!(*read_lock(&lock), 42);

        *write_lock(&lock) = 100;
        assert_eq!(*read_lock(&lock), 100);
    }
}
