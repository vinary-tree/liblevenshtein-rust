//! Interactive REPL for liblevenshtein.
//!
//! A comprehensive Read-Eval-Print Loop for exploring and manipulating Levenshtein
//! dictionaries interactively. All dictionary I/O and querying is delegated to the
//! shared primitives in [`crate::cli::commands`], so the REPL and the one-shot CLI
//! behave identically.
//!
//! ## Submodules
//!
//! - [`command`](crate::repl::command) — parses and executes a line of REPL input
//!   ([`Command`](crate::repl::Command), [`CommandResult`](crate::repl::CommandResult)).
//! - [`state`](crate::repl::state) — the persistent session state (loaded dictionary, backend, format).
//! - [`state_machine`](crate::repl::state_machine) — drives the prompt's input lifecycle.
//! - [`helper`](crate::repl::helper) — `rustyline` integration (completion, history, hints).
//! - [`highlighter`](crate::repl::highlighter) — syntax highlighting of the input line.

pub mod command;
pub mod helper;
pub mod highlighter;
pub mod state;
pub mod state_machine;

pub use command::{Command, CommandResult};
pub use helper::LevenshteinHelper;
pub use state::ReplState;
pub use state_machine::{ReplEvent, ReplPhase, ReplStateMachine, Transition};

/// REPL configuration
#[derive(Debug, Clone)]
pub struct ReplConfig {
    /// Prompt string
    pub prompt: String,
    /// Continuation prompt for multi-line input
    pub continuation_prompt: String,
    /// History file path
    pub history_file: Option<std::path::PathBuf>,
    /// Maximum history entries
    pub max_history: usize,
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            prompt: "liblevenshtein> ".to_string(),
            continuation_prompt: "...> ".to_string(),
            history_file: Some(
                dirs::home_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join(".liblevenshtein_history"),
            ),
            max_history: 1000,
        }
    }
}
