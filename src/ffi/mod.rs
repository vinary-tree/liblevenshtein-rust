//! C-compatible FFI for liblevenshtein.
//!
//! This module provides raw C-compatible functions (`extern "C"`) for use with:
//! - WASI runtimes (Wasmtime, WasmEdge)
//! - Native FFI from other languages (Python, Ruby, Go, etc.)
//!
//! # Memory Management
//!
//! All strings returned by FFI functions must be freed using [`llev_string_free`].
//! All arrays returned must be freed using their specific free function.
//!
//! # Safety
//!
//! All FFI functions are unsafe as they operate on raw pointers. Callers must:
//! - Ensure pointers are valid and non-null
//! - Free returned memory using the appropriate free function
//! - Not use freed pointers
//!
//! # Example (C)
//!
//! ```c
//! #include <liblevenshtein.h>
//!
//! int main() {
//!     // Calculate distance
//!     size_t dist = llev_distance("hello", 5, "helo", 4);
//!     printf("Distance: %zu\n", dist);
//!
//!     // Create dictionary
//!     const char* terms[] = {"hello", "help", "world"};
//!     LlevDictionary* dict = llev_dict_new(terms, 3);
//!
//!     // Create transducer
//!     LlevTransducer* trans = llev_transducer_new(dict, LLEV_ALGORITHM_STANDARD);
//!
//!     // Query
//!     LlevCandidateArray results = llev_transducer_query(trans, "helo", 4, 2);
//!     for (size_t i = 0; i < results.len; i++) {
//!         printf("%s: %zu\n", results.data[i].term, results.data[i].distance);
//!     }
//!
//!     // Cleanup
//!     llev_candidates_free(results);
//!     llev_transducer_free(trans);
//!     llev_dict_free(dict);
//!
//!     return 0;
//! }
//! ```

mod distance;
mod string;

pub use distance::*;
pub use string::*;

use std::ffi::{c_char, CStr};

/// Algorithm type for transducers.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlevAlgorithm {
    /// Standard Levenshtein (insert, delete, substitute)
    Standard = 0,
    /// Damerau-Levenshtein (adds transposition)
    Transposition = 1,
    /// Merge and split operations
    MergeAndSplit = 2,
}

impl From<LlevAlgorithm> for crate::transducer::Algorithm {
    fn from(algo: LlevAlgorithm) -> Self {
        match algo {
            LlevAlgorithm::Standard => crate::transducer::Algorithm::Standard,
            LlevAlgorithm::Transposition => crate::transducer::Algorithm::Transposition,
            LlevAlgorithm::MergeAndSplit => crate::transducer::Algorithm::MergeAndSplit,
        }
    }
}

/// Convert a C string to a Rust string slice.
///
/// # Safety
///
/// The input pointer must be valid and null-terminated.
unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}
