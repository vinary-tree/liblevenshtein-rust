//! Command parsing and execution
//!
//! Defines all REPL commands and their execution logic.

use super::state::{DictionaryBackend, ReplState};
use crate::cli::args::SerializationFormat;
use crate::transducer::Algorithm;
use anyhow::{Context, Result};
use colored::Colorize;
use std::io::{self, Write};
use std::path::PathBuf;

/// REPL command
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Query the dictionary: query <term> [distance] [--prefix] [--limit N]
    Query {
        /// Query term
        term: String,
        /// Maximum edit distance
        distance: Option<usize>,
        /// Enable prefix matching
        prefix: bool,
        /// Result limit
        limit: Option<usize>,
    },
    /// Insert term(s): insert <term> [term2] [term3] ...
    Insert {
        /// Terms to insert
        terms: Vec<String>,
    },
    /// Delete term(s): delete <term> [term2] [term3] ...
    Delete {
        /// Terms to delete
        terms: Vec<String>,
    },
    /// Check if term exists: contains <term>
    Contains {
        /// Term to check
        term: String,
    },
    /// Load dictionary from file: load <path> [backend]
    Load {
        /// Path to dictionary file
        path: PathBuf,
        /// Backend type (auto-detected if not specified)
        backend: Option<DictionaryBackend>,
    },
    /// Save dictionary to file: save <path>
    Save {
        /// Path to save to
        path: PathBuf,
    },
    /// Insert terms from file: insert-from <path>
    InsertFrom {
        /// Path to file containing terms
        path: PathBuf,
    },
    /// Remove terms from file: remove-from <path>
    RemoveFrom {
        /// Path to file containing terms to remove
        path: PathBuf,
    },
    /// Replace all terms with those from file: replace-with <path>
    ReplaceWith {
        /// Path to file containing replacement terms
        path: PathBuf,
    },
    /// Change backend: backend <type>
    Backend {
        /// Backend type
        backend: DictionaryBackend,
    },
    /// Change algorithm: algorithm <type>
    Algorithm {
        /// Algorithm type
        algorithm: Algorithm,
    },
    /// Set max distance: distance <n>
    Distance {
        /// Maximum edit distance
        distance: usize,
    },
    /// Toggle prefix mode: prefix [on|off]
    PrefixMode {
        /// Enable or disable prefix mode
        enable: Option<bool>,
    },
    /// Toggle distance display: show-distances [on|off]
    ShowDistances {
        /// Enable or disable distance display
        enable: Option<bool>,
    },
    /// Set result limit: limit <n>
    Limit {
        /// Result limit
        limit: Option<usize>,
    },
    /// Change serialization format: format <type>
    Format {
        /// Serialization format
        format: SerializationFormat,
    },
    /// Toggle auto-sync: auto-sync [on|off]
    AutoSync {
        /// Enable or disable auto-sync
        enable: Option<bool>,
    },
    /// Clear dictionary: clear
    Clear,
    /// Compact/minimize dictionary: compact
    Compact,
    /// Dump all terms: dump [limit]
    Dump {
        /// Limit number of terms to dump
        limit: Option<usize>,
    },
    /// Show statistics: stats | info
    Stats,
    /// Show settings: settings
    Settings,
    /// Manage config file: config [path]
    Config {
        /// Path to config file
        path: Option<PathBuf>,
    },
    /// Show help: help [command]
    Help {
        /// Help topic
        topic: Option<String>,
    },
    /// Enable cache: cache enable <strategy> [max-size]
    CacheEnable {
        /// Cache strategy
        strategy: String,
        /// Maximum cache size
        max_size: Option<usize>,
    },
    /// Disable cache: cache disable
    CacheDisable,
    /// Show cache statistics: cache stats
    CacheStats,
    /// Clear cache: cache clear
    CacheClear,
    /// Exit REPL: exit | quit
    Exit,
}

/// Command result
#[derive(Debug, Clone)]
pub enum CommandResult {
    /// Continue REPL
    Continue(String),
    /// Exit REPL
    Exit,
    /// No output
    Silent,
}

impl Command {
    /// Parse command from input string
    pub fn parse(input: &str) -> Result<Self> {
        let input = input.trim();

        if input.is_empty() {
            return Err(anyhow::anyhow!("Empty command"));
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        let cmd = parts[0].to_lowercase();

        match cmd.as_str() {
            "query" | "q" => Self::parse_query(&parts[1..]),
            "insert" | "add" => Self::parse_insert(&parts[1..]),
            "delete" | "remove" | "rm" => Self::parse_delete(&parts[1..]),
            "contains" | "has" => Self::parse_contains(&parts[1..]),
            "load" => Self::parse_load(&parts[1..]),
            "save" => Self::parse_save(&parts[1..]),
            "insert-from" | "merge" => Self::parse_insert_from(&parts[1..]),
            "remove-from" | "subtract" => Self::parse_remove_from(&parts[1..]),
            "replace-with" | "replace" => Self::parse_replace_with(&parts[1..]),
            "backend" | "use" => Self::parse_backend(&parts[1..]),
            "algorithm" | "algo" => Self::parse_algorithm(&parts[1..]),
            "distance" | "dist" => Self::parse_distance(&parts[1..]),
            "prefix" => Self::parse_prefix_mode(&parts[1..]),
            "show-distances" | "show-dist" => Self::parse_show_distances(&parts[1..]),
            "limit" => Self::parse_limit(&parts[1..]),
            "format" | "serialization" => Self::parse_format(&parts[1..]),
            "auto-sync" | "autosync" | "sync" => Self::parse_auto_sync(&parts[1..]),
            "clear" => Ok(Self::Clear),
            "compact" | "minimize" => Ok(Self::Compact),
            "dump" | "list" => Self::parse_dump(&parts[1..]),
            "stats" | "info" => Ok(Self::Stats),
            "settings" | "set" | "options" | "opts" => Ok(Self::Settings),
            "config" | "configuration" => Self::parse_config(&parts[1..]),
            "help" | "?" => Self::parse_help(&parts[1..]),
            "cache" => Self::parse_cache(&parts[1..]),
            "exit" | "quit" => Ok(Self::Exit),
            _ => Err(anyhow::anyhow!(
                "Unknown command: '{}'. Type 'help' for available commands.",
                cmd
            )),
        }
    }

    fn parse_query(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            return Err(anyhow::anyhow!(
                "Usage: query <term> [distance] [--prefix] [--limit N]"
            ));
        }

        let mut term = args[0].to_string();
        let mut distance = None;
        let mut prefix = false;
        let mut limit = None;
        let mut i = 1;

        while i < args.len() {
            match args[i] {
                "--prefix" | "-p" => prefix = true,
                "--limit" | "-l" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(anyhow::anyhow!("--limit requires a value"));
                    }
                    limit = Some(args[i].parse().context("Invalid limit value")?);
                }
                arg => {
                    if arg.starts_with("--") {
                        return Err(anyhow::anyhow!("Unknown option: {}", arg));
                    }
                    // Try to parse as distance
                    if distance.is_none() {
                        distance = Some(arg.parse().context("Invalid distance value")?);
                    } else {
                        // Treat as part of term
                        term.push(' ');
                        term.push_str(arg);
                    }
                }
            }
            i += 1;
        }

        Ok(Self::Query {
            term,
            distance,
            prefix,
            limit,
        })
    }

    fn parse_insert(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            return Err(anyhow::anyhow!("Usage: insert <term> [term2] [term3] ..."));
        }
        Ok(Self::Insert {
            terms: args.iter().map(|s| s.to_string()).collect(),
        })
    }

    fn parse_delete(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            return Err(anyhow::anyhow!("Usage: delete <term> [term2] [term3] ..."));
        }
        Ok(Self::Delete {
            terms: args.iter().map(|s| s.to_string()).collect(),
        })
    }

    fn parse_contains(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            return Err(anyhow::anyhow!("Usage: contains <term>"));
        }
        Ok(Self::Contains {
            term: args.join(" "),
        })
    }

    fn parse_load(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            return Err(anyhow::anyhow!("Usage: load <path> [backend]"));
        }

        let path = PathBuf::from(args[0]);
        let backend = if args.len() > 1 {
            Some(args[1].parse()?)
        } else {
            None
        };

        Ok(Self::Load { path, backend })
    }

    fn parse_save(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            return Err(anyhow::anyhow!("Usage: save <path>"));
        }
        Ok(Self::Save {
            path: PathBuf::from(args[0]),
        })
    }

    fn parse_insert_from(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            return Err(anyhow::anyhow!("Usage: insert-from <path>"));
        }
        Ok(Self::InsertFrom {
            path: PathBuf::from(args[0]),
        })
    }

    fn parse_remove_from(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            return Err(anyhow::anyhow!("Usage: remove-from <path>"));
        }
        Ok(Self::RemoveFrom {
            path: PathBuf::from(args[0]),
        })
    }

    fn parse_replace_with(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            return Err(anyhow::anyhow!("Usage: replace-with <path>"));
        }
        Ok(Self::ReplaceWith {
            path: PathBuf::from(args[0]),
        })
    }

    fn parse_backend(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            // Show current backend if no argument provided
            return Ok(Self::Settings);
        }
        Ok(Self::Backend {
            backend: args[0].parse()?,
        })
    }

    fn parse_algorithm(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            // Show current algorithm if no argument provided
            return Ok(Self::Settings);
        }

        let algorithm = match args[0].to_lowercase().as_str() {
            "standard" | "std" => Algorithm::Standard,
            "transposition" | "trans" => Algorithm::Transposition,
            "merge-and-split" | "merge" | "ms" => Algorithm::MergeAndSplit,
            _ => {
                return Err(anyhow::anyhow!(
                    "Unknown algorithm: {}. Valid: standard, transposition, merge-and-split",
                    args[0]
                ))
            }
        };

        Ok(Self::Algorithm { algorithm })
    }

    fn parse_distance(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            // Show current max distance if no argument provided
            return Ok(Self::Settings);
        }
        Ok(Self::Distance {
            distance: args[0].parse().context("Invalid distance value")?,
        })
    }

    fn parse_prefix_mode(args: &[&str]) -> Result<Self> {
        let enable = if args.is_empty() {
            None
        } else {
            Some(match args[0].to_lowercase().as_str() {
                "on" | "true" | "yes" | "1" => true,
                "off" | "false" | "no" | "0" => false,
                _ => return Err(anyhow::anyhow!("Usage: prefix [on|off]")),
            })
        };
        Ok(Self::PrefixMode { enable })
    }

    fn parse_show_distances(args: &[&str]) -> Result<Self> {
        let enable = if args.is_empty() {
            None
        } else {
            Some(match args[0].to_lowercase().as_str() {
                "on" | "true" | "yes" | "1" => true,
                "off" | "false" | "no" | "0" => false,
                _ => return Err(anyhow::anyhow!("Usage: show-distances [on|off]")),
            })
        };
        Ok(Self::ShowDistances { enable })
    }

    fn parse_limit(args: &[&str]) -> Result<Self> {
        let limit = if args.is_empty() || args[0].to_lowercase() == "none" || args[0] == "0" {
            None
        } else {
            Some(args[0].parse().context("Invalid limit value")?)
        };
        Ok(Self::Limit { limit })
    }

    fn parse_format(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            // Show current format if no argument provided
            return Ok(Self::Settings);
        }

        let format = match args[0].to_lowercase().as_str() {
            "text" | "txt" => SerializationFormat::Text,
            "bincode" | "bin" => SerializationFormat::Bincode,
            "json" => SerializationFormat::Json,
            #[cfg(feature = "protobuf")]
            "protobuf" | "proto" | "pb" => SerializationFormat::Protobuf,
            "paths" => SerializationFormat::PathsNative,
            _ => {
                #[cfg(feature = "protobuf")]
                let valid_formats = "text, bincode, json, protobuf, paths";
                #[cfg(not(feature = "protobuf"))]
                let valid_formats = "text, bincode, json, paths";

                return Err(anyhow::anyhow!(
                    "Invalid format: '{}'. Valid formats: {}",
                    args[0],
                    valid_formats
                ));
            }
        };
        Ok(Self::Format { format })
    }

    fn parse_auto_sync(args: &[&str]) -> Result<Self> {
        let enable = if args.is_empty() {
            None
        } else {
            Some(match args[0].to_lowercase().as_str() {
                "on" | "true" | "yes" | "1" => true,
                "off" | "false" | "no" | "0" => false,
                _ => return Err(anyhow::anyhow!("Usage: auto-sync [on|off]")),
            })
        };
        Ok(Self::AutoSync { enable })
    }

    fn parse_dump(args: &[&str]) -> Result<Self> {
        let limit = if args.is_empty() {
            None
        } else {
            Some(args[0].parse().context("Invalid limit value")?)
        };
        Ok(Self::Dump { limit })
    }

    fn parse_config(args: &[&str]) -> Result<Self> {
        let path = if args.is_empty() {
            None
        } else {
            Some(PathBuf::from(args[0]))
        };
        Ok(Self::Config { path })
    }

    fn parse_help(args: &[&str]) -> Result<Self> {
        Ok(Self::Help {
            topic: args.first().map(|s| s.to_string()),
        })
    }

    fn parse_cache(args: &[&str]) -> Result<Self> {
        if args.is_empty() {
            return Err(anyhow::anyhow!(
                "Usage: cache <subcommand>\n\
                 Subcommands: enable <strategy> [max-size], disable, stats, clear"
            ));
        }

        match args[0].to_lowercase().as_str() {
            "enable" => {
                if args.len() < 2 {
                    return Err(anyhow::anyhow!(
                        "Usage: cache enable <strategy> [max-size]\n\
                         Strategies: lru, lfu, ttl, age, cost-aware, memory-pressure, manual"
                    ));
                }

                let strategy = args[1].to_string();
                let max_size = if args.len() > 2 {
                    Some(args[2].parse().context("Invalid max-size value")?)
                } else {
                    None
                };

                Ok(Self::CacheEnable { strategy, max_size })
            }
            "disable" | "off" => Ok(Self::CacheDisable),
            "stats" | "info" => Ok(Self::CacheStats),
            "clear" => Ok(Self::CacheClear),
            _ => Err(anyhow::anyhow!(
                "Unknown cache subcommand: '{}'. Valid: enable, disable, stats, clear",
                args[0]
            )),
        }
    }

    /// Execute command
    pub fn execute(&self, state: &mut ReplState) -> Result<CommandResult> {
        match self {
            Self::Query {
                term,
                distance,
                prefix,
                limit,
            } => {
                // Save current settings
                let saved_distance = state.max_distance;
                let saved_prefix = state.prefix_mode;
                let saved_limit = state.result_limit;

                // Apply command-specific settings
                if let Some(d) = distance {
                    state.max_distance = *d;
                }
                if *prefix {
                    state.prefix_mode = true;
                }
                if let Some(l) = limit {
                    state.result_limit = Some(*l);
                }

                let results = state.query(term);

                // Restore settings
                state.max_distance = saved_distance;
                state.prefix_mode = saved_prefix;
                state.result_limit = saved_limit;

                let output = Self::format_query_results(&results, state.show_distances);
                Ok(CommandResult::Continue(output))
            }

            Self::Insert { terms } => {
                let mut inserted = 0;
                let mut skipped = 0;

                for term in terms {
                    match state.dictionary.insert(term) {
                        Ok(true) => inserted += 1,
                        Ok(false) => skipped += 1,
                        Err(e) => return Err(e),
                    }
                }
                if inserted > 0 {
                    state.invalidate_cache();
                }

                let msg = if skipped > 0 {
                    format!(
                        "Inserted {} term(s), {} already existed",
                        inserted.to_string().green().bold(),
                        skipped.to_string().yellow()
                    )
                } else {
                    format!("Inserted {} term(s)", inserted.to_string().green().bold())
                };

                Ok(CommandResult::Continue(msg))
            }

            Self::Delete { terms } => {
                let mut deleted = 0;
                let mut not_found = 0;

                for term in terms {
                    match state.dictionary.remove(term) {
                        Ok(true) => deleted += 1,
                        Ok(false) => not_found += 1,
                        Err(e) => return Err(e),
                    }
                }
                if deleted > 0 {
                    state.invalidate_cache();
                }

                let msg = if not_found > 0 {
                    format!(
                        "Deleted {} term(s), {} not found",
                        deleted.to_string().green().bold(),
                        not_found.to_string().yellow()
                    )
                } else {
                    format!("Deleted {} term(s)", deleted.to_string().green().bold())
                };

                Ok(CommandResult::Continue(msg))
            }

            Self::Contains { term } => {
                let exists = state.dictionary.contains(term);
                let msg = if exists {
                    format!("{}: {}", "Found".green().bold(), term)
                } else {
                    format!("{}: {}", "Not found".red(), term)
                };
                Ok(CommandResult::Continue(msg))
            }

            Self::Load { path, backend } => {
                let backend = backend.unwrap_or(state.backend);
                let count = state.load_from_file(path, backend)?;
                state.auto_sync_path = Some(path.clone());
                let msg = format!(
                    "Loaded {} term(s) from {} using {} backend",
                    count.to_string().green().bold(),
                    path.display().to_string().cyan(),
                    backend.to_string().yellow()
                );
                Ok(CommandResult::Continue(msg))
            }

            Self::Save { path } => {
                use crate::cli::paths::validate_dict_path;

                // Validate dictionary file extension matches serialization format
                validate_dict_path(path, state.serialization_format)
                    .with_context(|| format!(
                        "Cannot save to {}. Use a file with the correct extension or change the serialization format with 'format <type>'",
                        path.display()
                    ))?;

                let count = state.save_to_file(path)?;
                state.auto_sync_path = Some(path.clone());
                let msg = format!(
                    "Saved {} term(s) to {}",
                    count.to_string().green().bold(),
                    path.display().to_string().cyan()
                );
                Ok(CommandResult::Continue(msg))
            }

            Self::InsertFrom { path } => {
                // Load terms from file
                let terms = Self::load_terms_from_file(path)?;
                let total = terms.len();

                // Insert each term
                let mut inserted = 0;
                let mut skipped = 0;
                for term in terms {
                    match state.dictionary.insert(&term) {
                        Ok(true) => inserted += 1,
                        Ok(false) => skipped += 1,
                        Err(e) => return Err(e),
                    }
                }
                if inserted > 0 {
                    state.invalidate_cache();
                }

                let msg = if skipped > 0 {
                    format!(
                        "Inserted {} term(s) from {}, {} already existed (total: {})",
                        inserted.to_string().green().bold(),
                        path.display().to_string().cyan(),
                        skipped.to_string().yellow(),
                        total.to_string().yellow()
                    )
                } else {
                    format!(
                        "Inserted {} term(s) from {}",
                        inserted.to_string().green().bold(),
                        path.display().to_string().cyan()
                    )
                };
                Ok(CommandResult::Continue(msg))
            }

            Self::RemoveFrom { path } => {
                // Load terms from file
                let terms = Self::load_terms_from_file(path)?;
                let total = terms.len();

                // Remove each term
                let mut removed = 0;
                let mut not_found = 0;
                for term in terms {
                    match state.dictionary.remove(&term) {
                        Ok(true) => removed += 1,
                        Ok(false) => not_found += 1,
                        Err(e) => return Err(e),
                    }
                }
                if removed > 0 {
                    state.invalidate_cache();
                }

                let msg = if not_found > 0 {
                    format!(
                        "Removed {} term(s) from dictionary, {} not found (total in file: {})",
                        removed.to_string().green().bold(),
                        not_found.to_string().yellow(),
                        total.to_string().yellow()
                    )
                } else {
                    format!(
                        "Removed {} term(s) from dictionary",
                        removed.to_string().green().bold()
                    )
                };
                Ok(CommandResult::Continue(msg))
            }

            Self::ReplaceWith { path } => {
                let old_count = state.dictionary.len();

                // Confirm replacement
                if old_count > 0 {
                    let message = format!(
                        "Replace {} existing term(s) with terms from {}?",
                        old_count.to_string().red().bold(),
                        path.display().to_string().cyan()
                    );
                    if !Self::confirm(&message)? {
                        return Ok(CommandResult::Continue(
                            "Replace cancelled".yellow().to_string(),
                        ));
                    }
                }

                // Clear dictionary
                state.dictionary.clear()?;

                // Load terms from file
                let terms = Self::load_terms_from_file(path)?;
                let new_count = terms.len();

                // Insert all terms
                for term in terms {
                    state.dictionary.insert(&term)?;
                }
                state.invalidate_cache();

                let msg = format!(
                    "Replaced {} term(s) with {} term(s) from {}",
                    old_count.to_string().yellow(),
                    new_count.to_string().green().bold(),
                    path.display().to_string().cyan()
                );
                Ok(CommandResult::Continue(msg))
            }

            Self::Backend { backend } => {
                if state.backend == *backend {
                    return Ok(CommandResult::Continue(format!(
                        "Already using {} backend",
                        backend.to_string().yellow()
                    )));
                }

                let old_backend = state.backend;
                state.change_backend(*backend)?;

                // Save config after backend change
                #[cfg(feature = "cli")]
                state.save_config().ok();

                let msg = format!(
                    "Migrated from {} to {} ({} terms)",
                    old_backend.to_string().yellow(),
                    backend.to_string().green().bold(),
                    state.dictionary.len()
                );
                Ok(CommandResult::Continue(msg))
            }

            Self::Algorithm { algorithm } => {
                state.algorithm = *algorithm;
                state.invalidate_cache();

                // Save config after algorithm change
                #[cfg(feature = "cli")]
                state.save_config().ok();

                let msg = format!(
                    "Algorithm set to {}",
                    format!("{:?}", algorithm).green().bold()
                );
                Ok(CommandResult::Continue(msg))
            }

            Self::Distance { distance } => {
                state.max_distance = *distance;
                state.invalidate_cache();

                // Save config after distance change
                #[cfg(feature = "cli")]
                state.save_config().ok();

                let msg = format!(
                    "Max distance set to {}",
                    distance.to_string().green().bold()
                );
                Ok(CommandResult::Continue(msg))
            }

            Self::PrefixMode { enable } => {
                let new_state = enable.unwrap_or(!state.prefix_mode);
                state.prefix_mode = new_state;
                state.invalidate_cache();

                // Save config after prefix mode change
                #[cfg(feature = "cli")]
                state.save_config().ok();

                let msg = format!(
                    "Prefix mode {}",
                    if new_state {
                        "enabled".green().bold()
                    } else {
                        "disabled".red()
                    }
                );
                Ok(CommandResult::Continue(msg))
            }

            Self::ShowDistances { enable } => {
                let new_state = enable.unwrap_or(!state.show_distances);
                state.show_distances = new_state;

                // Save config after show distances change
                #[cfg(feature = "cli")]
                state.save_config().ok();

                let msg = format!(
                    "Distance display {}",
                    if new_state {
                        "enabled".green().bold()
                    } else {
                        "disabled".red()
                    }
                );
                Ok(CommandResult::Continue(msg))
            }

            Self::Limit { limit } => {
                state.result_limit = *limit;
                state.invalidate_cache();

                // Save config after limit change
                #[cfg(feature = "cli")]
                state.save_config().ok();

                let msg = if let Some(n) = limit {
                    format!("Result limit set to {}", n.to_string().green().bold())
                } else {
                    "Result limit removed".to_string()
                };
                Ok(CommandResult::Continue(msg))
            }

            Self::Format { format } => {
                use crate::cli::paths::{change_extension, file_extension, validate_dict_path};

                // Check if we have a dictionary path that would need its extension changed
                if let Some(ref dict_path) = state.auto_sync_path {
                    // Check if current path matches the new format
                    if validate_dict_path(dict_path, *format).is_err() {
                        let new_path = change_extension(dict_path, *format);
                        let expected_ext = file_extension(*format);

                        // Prompt for confirmation
                        print!(
                            "Changing format will require renaming dictionary file extension to .{}.\n  Old: {}\n  New: {}\nProceed? (y/N): ",
                            expected_ext,
                            dict_path.display().to_string().yellow(),
                            new_path.display().to_string().cyan()
                        );
                        io::stdout().flush()?;

                        let mut response = String::new();
                        io::stdin().read_line(&mut response)?;

                        if !response.trim().eq_ignore_ascii_case("y") {
                            return Err(anyhow::anyhow!(
                                "Format change cancelled. Dictionary path extension must match serialization format."
                            ));
                        }

                        // Update the path
                        state.auto_sync_path = Some(new_path.clone());
                        println!(
                            "  Dictionary path updated to {}",
                            new_path.display().to_string().cyan()
                        );
                    }
                }

                state.serialization_format = *format;

                // Save config after format change
                #[cfg(feature = "cli")]
                state.save_config().ok();

                let msg = format!(
                    "Serialization format set to {}",
                    format.to_string().green().bold()
                );
                Ok(CommandResult::Continue(msg))
            }

            Self::AutoSync { enable } => {
                let new_state = if let Some(enabled) = enable {
                    state.auto_sync = *enabled;
                    *enabled
                } else {
                    state.auto_sync = !state.auto_sync;
                    state.auto_sync
                };

                // Save config after auto-sync change
                #[cfg(feature = "cli")]
                state.save_config().ok();

                if new_state && state.auto_sync_path.is_none() {
                    return Ok(CommandResult::Continue(
                        format!("{}: Auto-sync enabled but no dictionary path set. Use 'save <path>' first or load a dictionary.",
                            "Warning".yellow().bold())
                    ));
                }

                let msg = format!(
                    "Auto-sync {}",
                    if new_state {
                        "enabled".green().bold()
                    } else {
                        "disabled".red()
                    }
                );
                Ok(CommandResult::Continue(msg))
            }

            Self::Clear => {
                let count = state.dictionary.len();
                if count == 0 {
                    return Ok(CommandResult::Continue(
                        "Dictionary is already empty".yellow().to_string(),
                    ));
                }

                let message = format!(
                    "Clear {} term(s) from dictionary?",
                    count.to_string().red().bold()
                );
                if !Self::confirm(&message)? {
                    return Ok(CommandResult::Continue(
                        "Clear cancelled".yellow().to_string(),
                    ));
                }

                state.dictionary.clear()?;
                state.invalidate_cache();
                let msg = format!("Cleared {} term(s)", count.to_string().green().bold());
                Ok(CommandResult::Continue(msg))
            }

            Self::Compact => {
                state.dictionary.compact()?;
                state.invalidate_cache();
                let msg = "Dictionary compacted/minimized".green().to_string();
                Ok(CommandResult::Continue(msg))
            }

            Self::Dump { limit } => {
                let terms = state.dictionary.terms();
                let total = terms.len();
                let terms: Vec<_> = if let Some(n) = limit {
                    terms.into_iter().take(*n).collect()
                } else {
                    terms
                };

                let mut output = String::new();
                for (i, term) in terms.iter().enumerate() {
                    output.push_str(&format!("{:4}. {}\n", i + 1, term));
                }

                if let Some(n) = limit {
                    if total > *n {
                        output.push_str(&format!(
                            "\n... {} more terms (showing {}/{})\n",
                            (total - n).to_string().yellow(),
                            n,
                            total
                        ));
                    }
                } else {
                    output.push_str(&format!("\nTotal: {} terms\n", total.to_string().green()));
                }

                Ok(CommandResult::Continue(output))
            }

            Self::Stats => {
                let stats = state.stats();
                Ok(CommandResult::Continue(format!("{}", stats)))
            }

            Self::Settings => {
                let mut output = String::new();
                output.push_str(&format!("{}\n\n", "Current Settings:".bold().underline()));

                // Dictionary Path - show actual default
                let dict_path = if let Some(ref path) = state.auto_sync_path {
                    path.display().to_string()
                } else {
                    crate::cli::paths::default_dict_path(state.backend, state.serialization_format)
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|_| "(auto)".to_string())
                };
                output.push_str(&format!("  Dictionary Path:    {}\n", dict_path.cyan()));

                output.push_str(&format!(
                    "  Backend:            {}\n",
                    state.backend.to_string().yellow()
                ));
                output.push_str(&format!(
                    "  Serialization:      {}\n",
                    state.serialization_format.to_string().yellow()
                ));
                output.push_str(&format!(
                    "  Algorithm:          {}\n",
                    state.algorithm.to_string().yellow()
                ));
                output.push_str(&format!(
                    "  Max Distance:       {}\n",
                    state.max_distance.to_string().yellow()
                ));
                output.push_str(&format!(
                    "  Prefix Mode:        {}\n",
                    if state.prefix_mode {
                        "enabled".green()
                    } else {
                        "disabled".red()
                    }
                ));
                output.push_str(&format!(
                    "  Show Distances:     {}\n",
                    if state.show_distances {
                        "enabled".green()
                    } else {
                        "disabled".red()
                    }
                ));
                output.push_str(&format!(
                    "  Result Limit:       {}\n",
                    state
                        .result_limit
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "none".to_string())
                        .yellow()
                ));
                output.push_str(&format!(
                    "  Auto-sync:          {}\n",
                    if state.auto_sync {
                        "enabled".green()
                    } else {
                        "disabled".red()
                    }
                ));
                output.push_str(&format!(
                    "  Terms:              {}\n",
                    state.dictionary.len().to_string().yellow()
                ));

                // Show config file path
                if let Ok(config_path) = crate::cli::paths::config_file_path() {
                    output.push_str(&format!(
                        "\n  Config file: {}\n",
                        config_path.display().to_string().cyan()
                    ));
                }

                Ok(CommandResult::Continue(output))
            }

            Self::Config { path } => {
                use crate::cli::commands::load_dictionary;
                use crate::cli::detect::detect_format;
                use crate::cli::paths::{AppConfig, PersistentConfig};

                if let Some(new_path) = path {
                    // User wants to switch to a new config file
                    let mut app_config = AppConfig::load()?;
                    let current_config = PersistentConfig::load()?;

                    // Check if destination exists
                    if !new_path.exists() {
                        // Copy current config to new location
                        current_config.copy_to(new_path)?;
                        println!(
                            "  Copied current configuration to {}",
                            new_path.display().to_string().cyan()
                        );
                    }

                    // Switch app config to point to new user config
                    app_config.switch_user_config(new_path.clone())?;
                    println!("  Updated app config to use new config file");

                    // Load new config and apply all settings to state
                    let new_config = PersistentConfig::load()?;

                    // Apply all settings from config
                    state.backend = new_config.backend.unwrap_or(state.backend);
                    state.serialization_format =
                        new_config.format.unwrap_or(state.serialization_format);
                    state.algorithm = new_config.algorithm.unwrap_or(state.algorithm);
                    state.max_distance = new_config.max_distance.unwrap_or(state.max_distance);
                    state.prefix_mode = new_config.prefix_mode.unwrap_or(state.prefix_mode);
                    state.show_distances =
                        new_config.show_distances.unwrap_or(state.show_distances);
                    state.result_limit = new_config.result_limit.unwrap_or(state.result_limit);
                    state.auto_sync = new_config.auto_sync.unwrap_or(state.auto_sync);
                    state.invalidate_cache();

                    // Load dictionary from config's path if set
                    if let Some(ref dict_path) = new_config.dict_path {
                        if dict_path.exists() {
                            match detect_format(dict_path, new_config.backend, new_config.format) {
                                Ok(detection) => {
                                    match load_dictionary(dict_path, detection.format) {
                                        Ok(container) => {
                                            let count = container.len();
                                            state.dictionary = container;
                                            state.backend = detection.format.backend;
                                            state.auto_sync_path = Some(dict_path.clone());
                                            state.invalidate_cache();
                                            println!(
                                                "  Loaded {} term(s) from {}",
                                                count.to_string().green(),
                                                dict_path.display().to_string().cyan()
                                            );
                                        }
                                        Err(e) => {
                                            println!(
                                                "  {}: Could not load dictionary: {}",
                                                "Warning".yellow(),
                                                e
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!(
                                        "  {}: Could not detect dictionary format: {}",
                                        "Warning".yellow(),
                                        e
                                    );
                                }
                            }
                        }
                    }

                    // Clear the session override since we've updated the app config
                    state.config_file_path = None;

                    let msg = format!(
                        "Now using config file: {}",
                        new_path.display().to_string().cyan()
                    );
                    Ok(CommandResult::Continue(msg))
                } else {
                    // Show current config file paths
                    let app_config = AppConfig::load()?;
                    let msg = format!(
                        "App Config:  {}\nUser Config: {}",
                        crate::cli::paths::app_config_path()?
                            .display()
                            .to_string()
                            .cyan(),
                        app_config.user_config_path.display().to_string().cyan()
                    );
                    Ok(CommandResult::Continue(msg))
                }
            }

            Self::Help { topic } => {
                let help_text = if let Some(topic) = topic {
                    Self::command_help(topic)
                } else {
                    Self::general_help()
                };
                Ok(CommandResult::Continue(help_text))
            }

            Self::CacheEnable { strategy, max_size } => {
                state.enable_cache(strategy, *max_size)?;
                Ok(CommandResult::Continue(format!(
                    "Cache enabled: strategy={}, capacity={}",
                    strategy.green().bold(),
                    max_size.unwrap_or(1000).to_string().green().bold()
                )))
            }

            Self::CacheDisable => {
                let was_enabled = state.cache_enabled();
                state.disable_cache();
                let msg = if was_enabled {
                    "Cache disabled".red().to_string()
                } else {
                    "Cache already disabled".yellow().to_string()
                };
                Ok(CommandResult::Continue(msg))
            }

            Self::CacheStats => Ok(CommandResult::Continue(state.cache_stats())),

            Self::CacheClear => {
                state.clear_cache()?;
                Ok(CommandResult::Continue("Cache cleared".green().to_string()))
            }

            Self::Exit => Ok(CommandResult::Exit),
        }
    }

    fn format_query_results(results: &[(String, usize)], show_distances: bool) -> String {
        if results.is_empty() {
            return "No matches found".yellow().to_string();
        }

        let mut output = String::new();
        for (i, (term, distance)) in results.iter().enumerate() {
            if show_distances {
                output.push_str(&format!(
                    "{:4}. {} ({})\n",
                    i + 1,
                    term.cyan(),
                    format!("d={}", distance).yellow()
                ));
            } else {
                output.push_str(&format!("{:4}. {}\n", i + 1, term.cyan()));
            }
        }
        output.push_str(&format!(
            "\n{} match(es) found\n",
            results.len().to_string().green()
        ));
        output
    }

    fn general_help() -> String {
        format!(
            r#"{}

{}
  The prompt shows your current state:
    {} Ready to accept input
    {} Entering multi-line command (end line with \)
    {} Success indicator
    {} Warning/recoverable error

  Current settings shown in prompt: backend/algorithm/max_distance
  Example: path-map/Standard/d2

{}
  Tab       - Auto-complete commands and arguments
  Ctrl+C    - Cancel current line (use 'exit' or Ctrl+D to quit)
  Ctrl+D    - Exit REPL
  Ctrl+R    - Search command history
  Up/Down   - Navigate command history

{}
  1. Load a dictionary:      load /usr/share/dict/words
  2. Query for similar term: query test
  3. Adjust settings:        distance 3
  4. Query again:            query exampl
  5. Get help on command:    help query

{}
  query, q <term> [distance] [--prefix] [--limit N]
                              Search for similar terms
  insert, add <term> ...    Insert term(s) into dictionary
  delete, remove, rm <term> ...
                              Delete term(s) from dictionary
  contains, has <term>      Check if term exists

{}
  load <path> [backend]     Load dictionary from file
  save <path>               Save dictionary to file
  dump, list [limit]        Show all terms (optionally limited)

{}
  backend, use <type>       Change dictionary backend
                            Types: pathmap, double-array-trie, dawg,
                                   optimized-dawg, dynamic-dawg, suffix-automaton
  algorithm, algo <type>    Change Levenshtein algorithm
                            Types: standard, transposition, merge-and-split
  distance, dist <n>        Set maximum edit distance
  prefix [on|off]           Toggle prefix matching mode
  show-distances [on|off]   Toggle distance display in results
  limit <n>                 Set result limit (0 or 'none' to remove)
  format <type>             Set serialization format
                            Types: text, bincode, json, protobuf
  auto-sync [on|off]        Toggle auto-save on modifications

{}
  clear                     Remove all terms from dictionary
  compact, minimize         Compact/minimize dictionary
  stats, info               Show dictionary statistics
  settings, config, set     Show current settings

{}
  cache enable <strategy> [max-size]
                            Enable fuzzy cache with specified strategy
                            Strategies: lru, lfu, ttl, age, cost-aware,
                                       memory-pressure, manual
  cache disable             Disable fuzzy cache
  cache stats               Show cache metrics and performance
  cache clear               Clear all cached entries

{}
  help, ? [command]         Show this help or help for specific command
  exit, quit                Exit REPL

{}
  liblevenshtein> query test 2 --prefix --limit 10
  liblevenshtein> insert testing tester tested
  liblevenshtein> delete obsolete
  liblevenshtein> load /usr/share/dict/words pathmap
  liblevenshtein> backend dynamic-dawg
  liblevenshtein> algorithm transposition
  liblevenshtein> prefix on
  liblevenshtein> query tes

{}
  For detailed help on any command: help <command>
  For tips on getting started:      help quickstart
  Report issues: https://github.com/anthropics/liblevenshtein-rust
"#,
            "liblevenshtein REPL - Interactive Levenshtein Dictionary Explorer"
                .bold()
                .underline(),
            "Understanding the Prompt:".bold(),
            "🔵",
            "🟡",
            "✓".green(),
            "⚠".yellow(),
            "Keyboard Shortcuts:".bold(),
            "Quick Start:".bold(),
            "Dictionary Operations:".bold(),
            "File Operations:".bold(),
            "Configuration:".bold(),
            "Maintenance:".bold(),
            "Cache Operations:".bold(),
            "Utility:".bold(),
            "Examples:".bold(),
            "More Help:".bold(),
        )
    }

    fn command_help(topic: &str) -> String {
        match topic.to_lowercase().as_str() {
            "query" | "q" => format!(
                r#"{}

{}
  query <term> [distance] [--prefix] [--limit N]
  q <term> [distance] [-p] [-l N]

{}
  Search for terms similar to <term> within edit distance.
  Results are ordered by distance (closest first), then alphabetically.

{}
  term              Query term to search for
  distance          Override max distance for this query (optional)
  --prefix, -p      Use prefix matching for this query
  --limit N, -l N   Limit results to N matches

{}
  query test                    # Search for "test" with current settings
  query test 1                  # Search with distance=1
  query test --prefix           # Prefix matching
  query test 2 --limit 10       # Top 10 results within distance 2
  query "hello world" --prefix  # Multi-word queries
"#,
                "query - Search Dictionary".bold().underline(),
                "Usage:".bold(),
                "Description:".bold(),
                "Arguments:".bold(),
                "Examples:".bold(),
            ),
            "insert" | "add" => format!(
                r#"{}

{}
  insert <term> [term2] [term3] ...
  add <term> [term2] [term3] ...

{}
  Insert one or more terms into the dictionary.
  Terms already in the dictionary will be skipped.
  Only works with mutable backends (PathMap, DynamicDAWG).

{}
  insert hello                  # Insert single term
  insert foo bar baz            # Insert multiple terms
  add testing                   # Alias for insert
"#,
                "insert - Add Terms".bold().underline(),
                "Usage:".bold(),
                "Description:".bold(),
                "Examples:".bold(),
            ),
            "delete" | "remove" | "rm" => format!(
                r#"{}

{}
  delete <term> [term2] [term3] ...
  remove <term> [term2] [term3] ...
  rm <term> [term2] [term3] ...

{}
  Remove one or more terms from the dictionary.
  Terms not found will be reported.
  Only works with mutable backends (PathMap, DynamicDAWG).

{}
  delete obsolete               # Delete single term
  remove foo bar baz            # Delete multiple terms
  rm testing                    # Alias for delete
"#,
                "delete - Remove Terms".bold().underline(),
                "Usage:".bold(),
                "Description:".bold(),
                "Examples:".bold(),
            ),
            "backend" | "use" => format!(
                r#"{}

{}
  backend <type>
  use <type>

{}
  Change the dictionary backend. This migrates all current terms
  to the new backend type.

{}
  pathmap       Fast trie, supports all operations
  dawg          Compressed DAWG, read-only
  dynamic-dawg  Compressed DAWG, supports modifications

{}
  backend pathmap               # Switch to PathMap
  backend dawg                  # Switch to DAWG (read-only)
  use dynamic-dawg              # Alias for backend
"#,
                "backend - Change Dictionary Backend".bold().underline(),
                "Usage:".bold(),
                "Description:".bold(),
                "Backends:".bold(),
                "Examples:".bold(),
            ),
            "algorithm" | "algo" => format!(
                r#"{}

{}
  algorithm <type>
  algo <type>

{}
  Change the Levenshtein algorithm used for queries.

{}
  standard          Insert, delete, substitute only
  transposition     Includes character transposition
  merge-and-split   Includes merge and split operations

{}
  algorithm standard            # Basic Levenshtein
  algo transposition            # Allow swaps
  algorithm merge-and-split     # Most permissive
"#,
                "algorithm - Change Levenshtein Algorithm"
                    .bold()
                    .underline(),
                "Usage:".bold(),
                "Description:".bold(),
                "Algorithms:".bold(),
                "Examples:".bold(),
            ),
            "cache" => format!(
                r#"{}

{}
  cache enable <strategy> [max-size]
  cache disable
  cache stats
  cache clear

{}
  Manage fuzzy query result caching to accelerate repeated queries.

  When enabled, the cache stores results from fuzzy queries to avoid
  redundant computations. Different eviction strategies optimize for
  different usage patterns.

{}
  cache enable <strategy> [max-size]
      Enable caching with specified strategy and optional size limit.

      Available strategies:
        lru               Least Recently Used - evicts oldest accessed entries
        lfu               Least Frequently Used - evicts rarely accessed entries
        ttl               Time-To-Live - evicts expired entries (5min default)
        age               FIFO - evicts oldest entries first
        cost-aware        Balances age, size, and access frequency
        memory-pressure   Prioritizes evicting large entries
        manual            Only evicts when full (FIFO), no automatic eviction

      max-size: Maximum cache entries (default: 1000)

  cache disable
      Disable caching and free cache memory.

  cache stats
      Show cache metrics including hits, misses, evictions, and hit rate.

  cache clear
      Remove all cached entries (keeps cache enabled).

{}
  # Enable LRU cache with default size
  cache enable lru

  # Enable LFU cache with custom size
  cache enable lfu 5000

  # Manual eviction for testing
  cache enable manual 100

  # View performance metrics
  cache stats

  # Clear cached results
  cache clear

  # Disable caching
  cache disable

{}
  - Enable cache before loading large dictionaries for best performance
  - LRU strategy works well for general-purpose use cases
  - Manual strategy is useful for testing or explicit cache control
  - Check cache stats regularly to optimize strategy choice
  - Clear cache when switching dictionaries or algorithms
"#,
                "cache - Fuzzy Query Result Caching".bold().underline(),
                "Usage:".bold(),
                "Description:".bold(),
                "Commands:".bold(),
                "Examples:".bold(),
                "Tips:".bold(),
            ),
            _ => format!(
                "No help available for '{}'. Try 'help' for general help.",
                topic
            ),
        }
    }

    /// Load terms from file (supports plain text and serialized formats)
    fn load_terms_from_file(path: &PathBuf) -> Result<Vec<String>> {
        // Try to detect if it's a plain text file first
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        // Check if it looks like plain text (one term per line)
        // Plain text files should have mostly printable ASCII/UTF-8 and newlines
        if contents
            .chars()
            .all(|c| c.is_alphanumeric() || c.is_whitespace() || c.is_ascii_punctuation())
        {
            // Parse as plain text
            let terms: Vec<String> = contents
                .lines()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .map(|s| s.to_string())
                .collect();

            if !terms.is_empty() {
                return Ok(terms);
            }
        }

        // If not plain text or empty, try to load as serialized dictionary and extract terms
        // We'll try using the CLI's detect_format and load_dictionary functions
        use crate::cli::commands::load_dictionary;
        use crate::cli::detect::detect_format;

        let detection =
            detect_format(path, None, None).context("Failed to detect dictionary format")?;

        let container = load_dictionary(path, detection.format)
            .context("Failed to load dictionary from file")?;

        // Extract terms from the loaded dictionary
        Ok(container.terms())
    }

    /// Prompt user for confirmation
    fn confirm(message: &str) -> Result<bool> {
        print!("{} {} ", message.yellow(), "(y/N):".yellow().bold());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let response = input.trim().to_lowercase();
        Ok(matches!(response.as_str(), "y" | "yes"))
    }
}
