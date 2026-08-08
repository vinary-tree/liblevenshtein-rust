#![deny(missing_docs)]
//! Shared WebAssembly implementation behind `@vinary-tree/vinary-tree`.

#[cfg(all(feature = "browser", target_arch = "wasm32", target_os = "unknown"))]
mod browser;

#[cfg(all(feature = "browser", target_arch = "wasm32", target_os = "unknown"))]
pub use browser::*;

#[cfg(all(feature = "wasi", target_arch = "wasm32", target_os = "wasi"))]
mod wasi;
