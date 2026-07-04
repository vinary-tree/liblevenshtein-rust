//! liblevenshtein - Fast fuzzy string matching with Levenshtein automata
//!
//! Provides CLI utilities and an interactive REPL for dictionary operations.

use clap::Parser;
use colored::Colorize;
use std::process;

use liblevenshtein::cli::commands;
use liblevenshtein::cli::paths::PersistentConfig;
use liblevenshtein::cli::Cli;
use liblevenshtein::repl::{Command, LevenshteinHelper, ReplConfig, ReplState};
use rustyline::error::ReadlineError;
use rustyline::{Config, Editor};

struct ReplLaunchOptions {
    dict_path: Option<std::path::PathBuf>,
    backend: Option<liblevenshtein::repl::state::DictionaryBackend>,
    format: Option<liblevenshtein::cli::args::SerializationFormat>,
    algorithm: liblevenshtein::transducer::Algorithm,
    max_distance: usize,
    prefix: bool,
    show_distances: bool,
    limit: Option<usize>,
    auto_sync: bool,
}

fn main() {
    let cli = Cli::parse();

    let result = if cli.repl {
        // Launch REPL with provided configuration
        run_repl(ReplLaunchOptions {
            dict_path: cli.dict.clone(),
            backend: cli.backend,
            format: cli.format,
            algorithm: cli.algorithm,
            max_distance: cli.max_distance,
            prefix: cli.prefix,
            show_distances: cli.show_distances,
            limit: cli.limit,
            auto_sync: cli.auto_sync,
        })
    } else {
        // Execute CLI command
        commands::execute(&cli)
    };

    if let Err(e) = result {
        eprintln!("{}: {}", "Error".red().bold(), e);
        process::exit(1);
    }
}

fn run_repl(options: ReplLaunchOptions) -> anyhow::Result<()> {
    use liblevenshtein::cli::detect::detect_format;
    use liblevenshtein::cli::paths::default_dict_path;

    let ReplLaunchOptions {
        dict_path,
        backend: backend_opt,
        format: format_opt,
        algorithm,
        max_distance,
        prefix,
        show_distances,
        limit,
        auto_sync,
    } = options;

    // Load persistent config
    let config = PersistentConfig::load().unwrap_or_default();

    // Merge with CLI options
    let cli_overrides = PersistentConfig {
        dict_path: dict_path.clone(),
        backend: backend_opt,
        format: format_opt,
        algorithm: Some(algorithm),
        max_distance: Some(max_distance),
        prefix_mode: Some(prefix),
        show_distances: Some(show_distances),
        result_limit: Some(limit),
        auto_sync: Some(auto_sync),
    };
    let merged_config = config.merge_with_cli(&cli_overrides);

    // NOTE: We don't save config at startup to avoid overwriting user preferences
    // with CLI defaults. Config will be saved when user changes settings during
    // the REPL session.

    // Print banner
    print_banner();

    // Initialize REPL state with configured options
    let mut state = ReplState::new();
    state.algorithm = merged_config.algorithm.unwrap_or(algorithm);
    state.max_distance = merged_config.max_distance.unwrap_or(max_distance);
    state.prefix_mode = merged_config.prefix_mode.unwrap_or(prefix);
    state.show_distances = merged_config.show_distances.unwrap_or(show_distances);
    state.result_limit = merged_config.result_limit.unwrap_or(limit);
    if let Some(fmt) = merged_config.format.or(format_opt) {
        state.serialization_format = fmt;
    }

    // Determine dictionary path
    let dict_path = if let Some(path) = dict_path {
        Some(path)
    } else if let Some(path) = merged_config.dict_path {
        Some(path)
    } else {
        // Use default path if it exists
        let backend = merged_config.backend.unwrap_or(state.backend);
        let format = merged_config
            .format
            .unwrap_or(liblevenshtein::cli::args::SerializationFormat::Text);
        default_dict_path(backend, format).ok()
    };

    // Try to auto-load dictionary if path exists
    if let Some(ref path) = dict_path {
        if path.exists() {
            match detect_format(
                path,
                backend_opt.or(merged_config.backend),
                format_opt.or(merged_config.format),
            ) {
                Ok(detection) => {
                    println!(
                        "  Loading dictionary from {} ({}, {})...",
                        path.display().to_string().cyan(),
                        detection.format.backend.to_string().green(),
                        detection.format.format.to_string().green()
                    );

                    match liblevenshtein::cli::commands::load_dictionary(path, detection.format) {
                        Ok(container) => {
                            let count = container.len();
                            state.dictionary = container;
                            state.backend = detection.format.backend;
                            state.auto_sync_path = Some(path.clone());
                            println!("  Loaded {} term(s)", count.to_string().green().bold());
                            println!();
                        }
                        Err(e) => {
                            eprintln!("  {}: Could not load dictionary: {}", "Warning".yellow(), e);
                            println!();
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "  {}: Could not detect dictionary format: {}",
                        "Warning".yellow(),
                        e
                    );
                    println!();
                }
            }
        }
    }

    let repl_config = ReplConfig::default();

    // Initialize Rustyline
    let rustyline_config = Config::builder()
        .auto_add_history(true)
        .history_ignore_dups(true)?
        .history_ignore_space(true)
        .build();

    let helper = LevenshteinHelper::new();
    let mut editor: Editor<LevenshteinHelper, rustyline::history::DefaultHistory> =
        Editor::with_config(rustyline_config)?;
    editor.set_helper(Some(helper));

    // Load history if it exists
    if let Some(history_path) = &repl_config.history_file {
        if history_path.exists() {
            let _ = editor.load_history(history_path);
        }
    }

    // Initialize state machine
    use liblevenshtein::repl::{ReplEvent, ReplStateMachine};
    let mut state_machine = ReplStateMachine::new();

    // Main REPL loop with state machine
    let mut line_num = 0;
    loop {
        // Check if state machine is in terminal state
        if state_machine.is_terminal() {
            break;
        }

        line_num += 1;

        // Generate prompt with state indicator and context
        let state_indicator = state_machine.phase().status_indicator();
        let backend_str = format!("{}", state.backend).bright_yellow();
        let algo_str = format!("{:?}", state.algorithm).bright_green();
        let dist_str = format!("d{}", state.max_distance).bright_magenta();

        let context = format!("{}/{}/{}", backend_str, algo_str, dist_str);

        let prompt = match state_machine.phase() {
            liblevenshtein::repl::ReplPhase::Continuation { .. } => {
                format!(
                    "{} {}[{}] {}...> ",
                    state_indicator,
                    "liblevenshtein".bright_cyan().bold(),
                    line_num,
                    context
                )
            }
            liblevenshtein::repl::ReplPhase::Error { .. } => {
                format!(
                    "{} {}[{}] {}> ",
                    state_indicator,
                    "liblevenshtein".bright_cyan().bold(),
                    line_num,
                    context
                )
            }
            _ => {
                format!(
                    "{} {}[{}] {}> ",
                    state_indicator,
                    "liblevenshtein".bright_cyan().bold(),
                    line_num,
                    context
                )
            }
        };

        // Read input
        let readline = editor.readline(&prompt);

        // Convert readline result to event
        let event = match readline {
            Ok(line) => {
                let line = line.trim().to_string();
                ReplEvent::LineSubmitted { line }
            }
            Err(ReadlineError::Interrupted) => ReplEvent::Interrupted,
            Err(ReadlineError::Eof) => ReplEvent::Eof,
            Err(err) => {
                eprintln!("{}: {:?}", "Readline error".red().bold(), err);
                break;
            }
        };

        // Process event through state machine
        match state_machine.process_event(event.clone()) {
            Ok(transition) => {
                // Display output if any
                if let Some(output) = transition.output {
                    println!("{}", output);
                }

                let mut follow_up = transition.follow_up;
                while let Some(event) = follow_up {
                    match state_machine.process_event(event) {
                        Ok(follow_up_transition) => {
                            if let Some(output) = follow_up_transition.output {
                                println!("{}", output);
                            }
                            follow_up = follow_up_transition.follow_up;
                        }
                        Err(e) => {
                            eprintln!("{}: State machine error: {}", "Error".red().bold(), e);
                            state_machine.reset();
                            follow_up = None;
                        }
                    }
                }

                // Handle command execution if in Executing phase
                if let liblevenshtein::repl::ReplPhase::Executing { ref command } =
                    state_machine.phase()
                {
                    // Check if command modifies dictionary
                    let modifies_dict = matches!(
                        command,
                        Command::Insert { .. }
                            | Command::Delete { .. }
                            | Command::Clear
                            | Command::Load { .. }
                    );

                    // Execute command
                    match command.execute(&mut state) {
                        Ok(result) => {
                            // Process execution result
                            let exec_event = ReplEvent::CommandExecuted { result };
                            if let Ok(exec_transition) = state_machine.process_event(exec_event) {
                                if let Some(exec_output) = exec_transition.output {
                                    println!("{}", exec_output);
                                }
                            }

                            // Auto-sync if enabled and command modified dictionary
                            if modifies_dict && state.auto_sync {
                                if let Some(ref path) = state.auto_sync_path {
                                    if let Err(e) = state.save_to_file(path) {
                                        eprintln!(
                                            "{}: Failed to auto-save dictionary: {}",
                                            "Warning".yellow(),
                                            e
                                        );
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            // Handle execution error
                            let error_event = ReplEvent::ExecutionError {
                                message: e.to_string(),
                                recoverable: true,
                            };
                            if let Ok(error_transition) = state_machine.process_event(error_event) {
                                if let Some(error_output) = error_transition.output {
                                    println!("{}", error_output);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("{}: State machine error: {}", "Error".red().bold(), e);
                state_machine.reset();
            }
        }
    }

    // Save history
    if let Some(history_path) = &repl_config.history_file {
        if let Err(e) = editor.save_history(history_path) {
            eprintln!("{}: Failed to save history: {}", "Warning".yellow(), e);
        }
    }

    // Save config on exit
    if let Err(e) = state.save_config() {
        eprintln!("{}: Failed to save config: {}", "Warning".yellow(), e);
    }

    Ok(())
}

fn print_banner() {
    println!();
    println!(
        "{}",
        "═══════════════════════════════════════════════════════".bright_cyan()
    );
    println!(
        "{}",
        "   liblevenshtein - Fast Fuzzy String Matching"
            .bright_cyan()
            .bold()
    );
    println!(
        "{}",
        "═══════════════════════════════════════════════════════".bright_cyan()
    );
    println!();
    println!("  Version: {}", env!("CARGO_PKG_VERSION").green());
    println!("  Type {} for available commands", "'help'".yellow().bold());
    println!(
        "  Type {} or press {} to exit",
        "'exit'".yellow().bold(),
        "Ctrl+D".yellow().bold()
    );
    println!();
    println!("{}", "  Quick Start:".bold());
    println!(
        "    • Load a dictionary: {}",
        "load /usr/share/dict/words".cyan()
    );
    println!("    • Query for matches: {}", "query test 2".cyan());
    println!("    • Insert terms:      {}", "insert hello world".cyan());
    println!("    • Show settings:     {}", "settings".cyan());
    println!();
}
