//! WebAssembly bindings for liblevenshtein.
//!
//! This module provides JavaScript-friendly APIs via wasm-bindgen for use in
//! browsers and Node.js environments.
//!
//! # Features
//!
//! - **Distance Functions**: `levenshtein()`, `damerau_levenshtein()`
//! - **Phonetic Rules**: `WasmRuleSet` (with `wasm-phonetic` feature)
//!
//! # Usage from JavaScript
//!
//! ```javascript
//! import init, { levenshtein } from '@vinary-tree/vinary-tree/liblevenshtein';
//!
//! async function main() {
//!     await init();
//!
//!     // Distance functions
//!     console.log(levenshtein("hello", "helo")); // 1
//!
//!     // Cross-project dictionary/automaton bindings are linked by the
//!     // @vinary-tree/vinary-tree umbrella runtime so every facade shares one
//!     // WebAssembly instance and one resource table.
//! }
//! ```

mod distance;

#[cfg(feature = "phonetic-rules")]
mod phonetic;

pub use distance::*;

#[cfg(feature = "phonetic-rules")]
pub use phonetic::*;

use wasm_bindgen::prelude::*;

/// Initialize the WASM module.
///
/// This function sets up panic hooks for better error messages in the browser console.
/// It is automatically called when the module loads, but can be called explicitly
/// if needed.
#[wasm_bindgen(start)]
pub fn wasm_init() {
    // Set up panic hook for better error messages
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get the library version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
