//! String handling utilities for FFI.

use std::ffi::{c_char, CString};

/// Free a string allocated by liblevenshtein.
///
/// # Safety
///
/// - `ptr` must have been allocated by a liblevenshtein FFI function
/// - `ptr` must not be used after this call
/// - Passing NULL is safe (no-op)
#[no_mangle]
pub unsafe extern "C" fn llev_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Free a string array allocated by liblevenshtein.
///
/// # Safety
///
/// - The array and all strings within must have been allocated by liblevenshtein
/// - The array and strings must not be used after this call
#[no_mangle]
pub unsafe extern "C" fn llev_string_array_free(arr: *mut *mut c_char, len: usize) {
    if arr.is_null() {
        return;
    }

    // Free each string
    let slice = std::slice::from_raw_parts_mut(arr, len);
    for ptr in slice.iter() {
        if !ptr.is_null() {
            drop(CString::from_raw(*ptr));
        }
    }

    // Free the array itself
    drop(Vec::from_raw_parts(arr, len, len));
}

/// Duplicate a string.
///
/// Returns a heap-allocated copy that must be freed with `llev_string_free`.
///
/// # Safety
///
/// - `src` must be a valid null-terminated C string
/// - Returns NULL if allocation fails or input is NULL
#[no_mangle]
pub unsafe extern "C" fn llev_string_dup(src: *const c_char) -> *mut c_char {
    if src.is_null() {
        return std::ptr::null_mut();
    }

    match std::ffi::CStr::from_ptr(src).to_str() {
        Ok(s) => match CString::new(s) {
            Ok(cstr) => cstr.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}
