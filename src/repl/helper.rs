//! Rustyline helper integration
//!
//! Provides completion, hinting, highlighting, and validation for the REPL.

use super::highlighter::CommandHighlighter;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::hint::{Hinter, HistoryHinter};
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Context, Helper};
use std::borrow::Cow;

/// REPL helper
pub struct LevenshteinHelper {
    highlighter: CommandHighlighter,
    hinter: HistoryHinter,
    commands: Vec<String>,
    backends: Vec<String>,
    algorithms: Vec<String>,
}

impl LevenshteinHelper {
    /// Create a new helper instance
    pub fn new() -> Self {
        Self {
            highlighter: CommandHighlighter::new(),
            hinter: HistoryHinter::new(),
            commands: vec![
                "query",
                "insert",
                "delete",
                "contains",
                "load",
                "save",
                "backend",
                "algorithm",
                "distance",
                "prefix",
                "show-distances",
                "limit",
                "clear",
                "compact",
                "dump",
                "stats",
                "settings",
                "help",
                "exit",
                "quit",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            backends: vec!["pathmap", "dawg", "dynamic-dawg"]
                .into_iter()
                .map(String::from)
                .collect(),
            algorithms: vec!["standard", "transposition", "merge-and-split"]
                .into_iter()
                .map(String::from)
                .collect(),
        }
    }
}

impl Default for LevenshteinHelper {
    fn default() -> Self {
        Self::new()
    }
}

impl Helper for LevenshteinHelper {}

impl Completer for LevenshteinHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> Result<(usize, Vec<Pair>), ReadlineError> {
        let line = &line[..pos];
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.is_empty() {
            // Complete command names
            let candidates = self
                .commands
                .iter()
                .map(|cmd| Pair {
                    display: cmd.clone(),
                    replacement: format!("{} ", cmd),
                })
                .collect();
            return Ok((0, candidates));
        }

        let cmd = parts[0].to_lowercase();
        let start = line.rfind(char::is_whitespace).map(|i| i + 1).unwrap_or(0);

        // Command completion if still typing first word
        if parts.len() == 1 && !line.ends_with(char::is_whitespace) {
            let prefix = parts[0].to_lowercase();
            let candidates = self
                .commands
                .iter()
                .filter(|cmd| cmd.starts_with(&prefix))
                .map(|cmd| Pair {
                    display: cmd.clone(),
                    replacement: format!("{} ", cmd),
                })
                .collect();
            return Ok((start, candidates));
        }

        // Context-specific completions
        match cmd.as_str() {
            "backend" | "use" if parts.len() <= 2 => {
                let prefix = parts.last().map(|s| s.to_lowercase()).unwrap_or_default();
                let candidates = self
                    .backends
                    .iter()
                    .filter(|b| b.starts_with(&prefix))
                    .map(|b| Pair {
                        display: b.clone(),
                        replacement: format!("{} ", b),
                    })
                    .collect();
                Ok((start, candidates))
            }
            "algorithm" | "algo" if parts.len() <= 2 => {
                let prefix = parts.last().map(|s| s.to_lowercase()).unwrap_or_default();
                let candidates = self
                    .algorithms
                    .iter()
                    .filter(|a| a.starts_with(&prefix))
                    .map(|a| Pair {
                        display: a.clone(),
                        replacement: format!("{} ", a),
                    })
                    .collect();
                Ok((start, candidates))
            }
            "prefix" | "show-distances" if parts.len() <= 2 => {
                let options = ["on", "off"];
                let prefix = parts.last().map(|s| s.to_lowercase()).unwrap_or_default();
                let candidates = options
                    .iter()
                    .filter(|o| o.starts_with(&prefix))
                    .map(|o| Pair {
                        display: o.to_string(),
                        replacement: format!("{} ", o),
                    })
                    .collect();
                Ok((start, candidates))
            }
            "help" | "?" if parts.len() <= 2 => {
                let prefix = parts.last().map(|s| s.to_lowercase()).unwrap_or_default();
                let candidates = self
                    .commands
                    .iter()
                    .filter(|cmd| cmd.starts_with(&prefix))
                    .map(|cmd| Pair {
                        display: cmd.clone(),
                        replacement: format!("{} ", cmd),
                    })
                    .collect();
                Ok((start, candidates))
            }
            _ => Ok((0, vec![])),
        }
    }
}

impl Hinter for LevenshteinHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, ctx: &Context<'_>) -> Option<Self::Hint> {
        self.hinter.hint(line, pos, ctx)
    }
}

impl Highlighter for LevenshteinHelper {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> Cow<'l, str> {
        self.highlighter.highlight(line, pos)
    }

    fn highlight_char(&self, line: &str, pos: usize, kind: CmdKind) -> bool {
        self.highlighter.highlight_char(line, pos, kind)
    }
}

impl Validator for LevenshteinHelper {
    fn validate(&self, _ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        // Always accept input (validation happens during execution)
        Ok(ValidationResult::Valid(None))
    }
}
