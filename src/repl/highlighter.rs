//! Syntax highlighting for REPL input.
//!
//! Implements the `rustyline` [`Highlighter`] trait to
//! colourise commands, arguments, and string literals as the user types, giving the
//! interactive prompt immediate visual feedback. Purely presentational — it does not
//! parse or execute input (that is [`crate::repl::command`]).

use colored::Colorize;
use rustyline::highlight::{CmdKind, Highlighter};

/// Command highlighter
pub struct CommandHighlighter {
    commands: Vec<String>,
}

impl CommandHighlighter {
    /// Create a new command highlighter
    pub fn new() -> Self {
        Self {
            commands: vec![
                "query",
                "q",
                "insert",
                "add",
                "delete",
                "remove",
                "rm",
                "contains",
                "has",
                "load",
                "save",
                "backend",
                "use",
                "algorithm",
                "algo",
                "distance",
                "dist",
                "prefix",
                "show-distances",
                "show-dist",
                "limit",
                "clear",
                "compact",
                "minimize",
                "dump",
                "list",
                "stats",
                "info",
                "settings",
                "config",
                "set",
                "help",
                "?",
                "exit",
                "quit",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        }
    }

    fn highlight_command(&self, line: &str) -> String {
        if line.trim().is_empty() {
            return line.to_string();
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return line.to_string();
        }

        let cmd = parts[0].to_lowercase();

        // Check if it's a known command
        if self.commands.iter().any(|c| c == &cmd) {
            // Highlight the command in bold blue
            let highlighted_cmd = parts[0].blue().bold().to_string();

            // Reconstruct the line with highlighted command
            if parts.len() == 1 {
                highlighted_cmd
            } else {
                let rest = line[parts[0].len()..].to_string();
                format!("{}{}", highlighted_cmd, self.highlight_args(&rest, &cmd))
            }
        } else {
            line.to_string()
        }
    }

    fn highlight_args(&self, args: &str, cmd: &str) -> String {
        let mut result = String::new();
        let mut in_option = false;

        for part in args.split_whitespace() {
            result.push(' ');

            if part.starts_with("--") || part.starts_with('-') {
                // Highlight options/flags in yellow
                result.push_str(&part.yellow().to_string());
                in_option = true;
            } else if in_option {
                // Highlight option values in cyan
                result.push_str(&part.cyan().to_string());
                in_option = false;
            } else if part.parse::<usize>().is_ok() {
                // Highlight numbers in magenta
                result.push_str(&part.magenta().to_string());
            } else if matches!(
                part.to_lowercase().as_str(),
                "on" | "off" | "true" | "false"
            ) {
                // Highlight boolean keywords in green
                result.push_str(&part.green().to_string());
            } else if Self::is_backend_name(part) || Self::is_algorithm_name(part) {
                // Highlight backend/algorithm names in cyan
                result.push_str(&part.cyan().to_string());
            } else if Self::is_format_name(part) {
                // Highlight format names in cyan
                result.push_str(&part.cyan().to_string());
            } else if part.ends_with(".txt")
                || part.ends_with(".bin")
                || part.ends_with(".json")
                || part.ends_with(".pb")
                || part.ends_with(".paths")
            {
                // Highlight file paths in bright white
                result.push_str(&part.bright_white().to_string());
            } else if matches!(
                cmd,
                "query" | "q" | "insert" | "add" | "delete" | "remove" | "rm" | "contains" | "has"
            ) {
                // Highlight query terms and dictionary terms in white
                result.push_str(part);
            } else {
                // Regular arguments
                result.push_str(part);
            }
        }

        result
    }

    fn is_backend_name(s: &str) -> bool {
        matches!(
            s.to_lowercase().as_str(),
            "pathmap"
                | "path-map"
                | "double-array-trie"
                | "doublearraytrie"
                | "dat"
                | "dawg"
                | "optimized-dawg"
                | "optimizeddawg"
                | "dynamic-dawg"
                | "dynamicdawg"
                | "suffix-automaton"
                | "suffixautomaton"
        )
    }

    fn is_algorithm_name(s: &str) -> bool {
        matches!(
            s.to_lowercase().as_str(),
            "standard" | "std" | "transposition" | "trans" | "merge-and-split" | "mas"
        )
    }

    fn is_format_name(s: &str) -> bool {
        matches!(
            s.to_lowercase().as_str(),
            "text"
                | "txt"
                | "bincode"
                | "bin"
                | "json"
                | "protobuf"
                | "pb"
                | "bincode-gzip"
                | "json-gzip"
                | "protobuf-gzip"
                | "paths-native"
                | "paths"
        )
    }
}

impl Default for CommandHighlighter {
    fn default() -> Self {
        Self::new()
    }
}

impl Highlighter for CommandHighlighter {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> std::borrow::Cow<'l, str> {
        std::borrow::Cow::Owned(self.highlight_command(line))
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _kind: CmdKind) -> bool {
        true
    }
}
