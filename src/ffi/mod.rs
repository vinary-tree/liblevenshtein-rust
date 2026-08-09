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
//! Dictionaries are owned by `libdictenstein`; this library consumes any
//! `vt.dictionary.v1` resource (here one produced by
//! `ldict_dictionary_resource`) and streams matches through leased batches.
//!
//! ```c
//! #include <libdictenstein.h>
//! #include <liblevenshtein.h>
//! #include <stdio.h>
//!
//! int main(void) {
//!     // Distance functions need no dictionary at all.
//!     size_t distance = llev_distance("hello", 5, "helo", 4);
//!     printf("distance: %zu\n", distance);
//!
//!     // Build a dictionary with its owning library...
//!     LdictDictionary* dictionary = NULL;
//!     ldict_dynamic_dawg_new(VT_UNIT_DOMAIN_UNICODE_SCALAR, &dictionary);
//!     LdictOptionalU64 value = {7, 1, {0}};
//!     uint8_t inserted = 0;
//!     ldict_dictionary_insert_text(dictionary, (const uint8_t*)"hello", 5,
//!                                  value, &inserted);
//!
//!     // ...export it as a shared resource and hand it to a transducer.
//!     VtResource resource;
//!     ldict_dictionary_resource(dictionary, &resource);
//!     LlevTransducer* transducer = NULL;
//!     llev_transducer_new(&resource, LLEV_ALGORITHM_STANDARD, &transducer);
//!
//!     // Query lazily; each leased batch is borrowed until released.
//!     LlevQueryCursor* cursor = NULL;
//!     llev_transducer_query_utf8(transducer, "helo", 4, 2,
//!                                LLEV_QUERY_ORDER_TRAVERSAL, &cursor);
//!     LlevMatchBatchView batch;
//!     while (llev_query_cursor_next_batch(cursor, LLEV_DEFAULT_MATCH_BATCH,
//!                                         &batch) == LLEV_STATUS_OK &&
//!            batch.len > 0) {
//!         for (size_t i = 0; i < batch.len; ++i) {
//!             const LlevMatch* match = &batch.matches[i];
//!             printf("%.*s: %zu\n", (int)match->byte_len,
//!                    (const char*)match->term_data, match->distance);
//!         }
//!         llev_query_cursor_release_batch(cursor, batch.generation);
//!     }
//!
//!     llev_query_cursor_free(cursor);
//!     llev_transducer_free(transducer);
//!     ldict_dictionary_free(dictionary);
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
