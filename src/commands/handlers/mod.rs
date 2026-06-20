//! Command handler implementations
//!
//! This module provides shared handler functions that can be used by both the
//! CLI and the REPL for common operations like querying and I/O.

pub mod query;

#[cfg(feature = "cli")]
pub mod io;

// Modification operations (insert/delete/clear) are backend-specific
// and handled directly by dictionary containers (REPL's DictContainer, etc.)
// They don't fit into a shared handler pattern since the Dictionary trait
// is immutable.

// Re-exports for convenience
pub use query::execute_query;

#[cfg(feature = "cli")]
pub use io::{execute_deserialize, execute_serialize, get_dict_info};
