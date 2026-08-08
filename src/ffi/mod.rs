//! C-compatible FFI for liblevenshtein.
//!
//! This module provides raw C-compatible functions (`extern "C"`) for use with:
//! - WASI runtimes (Wasmtime, WasmEdge)
//! - Native FFI from other languages (Python, Ruby, Go, etc.)
//!
//! # Memory Management
//!
//! Owned strings returned by the legacy string helpers must be freed using
//! [`llev_string_free`](crate::ffi::llev_string_free). Query matches and their
//! UTF-8 term bytes are borrowed from a cursor and are invalidated by its next
//! advance; they must never be freed by the caller.
//!
//! # Safety
//!
//! All FFI functions are unsafe as they operate on raw pointers. Callers must:
//! - Ensure required pointers are valid and non-null
//! - Ensure length-bearing buffers are valid for their provided byte length
//! - Free returned memory using the appropriate free function
//! - Treat cursor match views as borrowed storage
//! - Not use freed pointers
//!
//! # Example (C)
//!
//! ```c
//! #include <liblevenshtein.h>
//! #include <stdio.h>
//!
//! int main() {
//!     // Calculate distance
//!     size_t dist = llev_distance("hello", 5, "helo", 4);
//!     printf("Distance: %zu\n", dist);
//!
//!     LlevIndex* index = NULL;
//!     uint8_t inserted = 0;
//!     llev_index_new(&index);
//!     llev_index_insert(index, "hello", 5, 1, 7, &inserted);
//!
//!     // Query lazily; every returned match is borrowed from this cursor.
//!     LlevQueryCursor* cursor = NULL;
//!     llev_index_query(index, "helo", 4, 2, LLEV_ALGORITHM_STANDARD,
//!                      LLEV_QUERY_ORDER_TRAVERSAL, &cursor);
//!     const LlevMatch* match = NULL;
//!     while (llev_query_cursor_next(cursor, &match) == LLEV_STATUS_OK) {
//!         printf("%.*s: %zu\n", (int)match->term_len, match->term,
//!                match->distance);
//!     }
//!
//!     llev_query_cursor_free(cursor);
//!     llev_index_free(index);
//!
//!     return 0;
//! }
//! ```

mod distance;
mod generated;
mod index;
mod phonetic;
mod string;

pub use distance::*;
pub use generated::*;
pub use index::*;
pub use phonetic::*;
pub use string::*;

use std::{ffi::c_char, slice, str};

impl From<LlevAlgorithm> for crate::transducer::Algorithm {
    fn from(algo: LlevAlgorithm) -> Self {
        match algo {
            LlevAlgorithm::Standard => crate::transducer::Algorithm::Standard,
            LlevAlgorithm::Transposition => crate::transducer::Algorithm::Transposition,
            LlevAlgorithm::MergeAndSplit => crate::transducer::Algorithm::MergeAndSplit,
            LlevAlgorithm::DamerauLevenshtein => crate::transducer::Algorithm::DamerauLevenshtein,
        }
    }
}

/// Convert a length-bearing UTF-8 buffer to a Rust string slice.
///
/// # Safety
///
/// The input pointer must be valid for `len` bytes and must point to UTF-8.
unsafe fn cbuf_to_str<'a>(ptr: *const c_char, len: usize) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }

    let bytes = slice::from_raw_parts(ptr.cast::<u8>(), len);
    str::from_utf8(bytes).ok()
}
