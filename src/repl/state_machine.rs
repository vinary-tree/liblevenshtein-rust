//! REPL state machine
//!
//! Provides a structured state machine for managing REPL execution flow,
//! inspired by The Elm Architecture (TEA) and functional state management patterns.

use super::command::{Command, CommandResult};
use anyhow::Result;
use colored::Colorize;

/// REPL execution phase
///
/// Represents the current state of the REPL's execution cycle.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ReplPhase {
    /// Ready to accept new input
    #[default]
    Ready,

    /// Accepting multi-line input (continuation)
    Continuation {
        /// Buffer containing accumulated input
        buffer: String,
    },

    /// Executing a command
    Executing {
        /// The command being executed
        command: Command,
    },

    /// Displaying query results
    DisplayingResults {
        /// Results to display
        output: String,
    },

    /// Error state
    Error {
        /// Error message
        message: String,
        /// Whether the error is recoverable
        recoverable: bool,
    },

    /// Exiting the REPL
    Exiting,
}

impl ReplPhase {
    /// Check if the phase is terminal (requires exit)
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Exiting
                | Self::Error {
                    recoverable: false,
                    ..
                }
        )
    }

    /// Check if the phase is recoverable from error
    pub fn is_recoverable(&self) -> bool {
        !matches!(
            self,
            Self::Error {
                recoverable: false,
                ..
            }
        )
    }

    /// Get a status indicator string for display
    pub fn status_indicator(&self) -> String {
        match self {
            Self::Ready => "🔵".to_string(),
            Self::Continuation { .. } => "🟡".to_string(),
            Self::Executing { .. } => "🟢".to_string(),
            Self::DisplayingResults { .. } => "✓".green().to_string(),
            Self::Error {
                recoverable: true, ..
            } => "⚠".yellow().to_string(),
            Self::Error {
                recoverable: false, ..
            } => "✗".red().to_string(),
            Self::Exiting => "👋".to_string(),
        }
    }
}

/// REPL event
///
/// Represents events that trigger state transitions in the REPL.
#[derive(Debug, Clone)]
pub enum ReplEvent {
    /// User submitted a line of input
    LineSubmitted {
        /// The input line
        line: String,
    },

    /// Command was successfully parsed
    CommandParsed {
        /// The parsed command
        command: Command,
    },

    /// Command execution completed
    CommandExecuted {
        /// The execution result
        result: CommandResult,
    },

    /// User interrupted (Ctrl+C)
    Interrupted,

    /// End of file (Ctrl+D)
    Eof,

    /// Parse error occurred
    ParseError {
        /// Error message
        message: String,
    },

    /// Execution error occurred
    ExecutionError {
        /// Error message
        message: String,
        /// Whether the error is recoverable
        recoverable: bool,
    },

    /// Continuation line needed
    ContinuationNeeded {
        /// Current buffer
        buffer: String,
    },

    /// Results ready to display
    ResultsReady {
        /// Output to display
        output: String,
    },
}

/// State transition result
#[derive(Debug)]
pub struct Transition {
    /// New phase after transition
    pub new_phase: ReplPhase,
    /// Optional output message
    pub output: Option<String>,
    /// Optional follow-up event to process
    pub follow_up: Option<ReplEvent>,
}

impl Transition {
    /// Create a simple transition with no output or follow-up
    pub fn to(phase: ReplPhase) -> Self {
        Self {
            new_phase: phase,
            output: None,
            follow_up: None,
        }
    }

    /// Create a transition with output
    pub fn to_with_output(phase: ReplPhase, output: String) -> Self {
        Self {
            new_phase: phase,
            output: Some(output),
            follow_up: None,
        }
    }

    /// Create a transition with a follow-up event
    pub fn to_with_follow_up(phase: ReplPhase, follow_up: ReplEvent) -> Self {
        Self {
            new_phase: phase,
            output: None,
            follow_up: Some(follow_up),
        }
    }

    /// Create a transition with both output and follow-up
    pub fn to_with_both(phase: ReplPhase, output: String, follow_up: ReplEvent) -> Self {
        Self {
            new_phase: phase,
            output: Some(output),
            follow_up: Some(follow_up),
        }
    }
}

/// State machine for REPL execution
pub struct ReplStateMachine {
    /// Current phase
    phase: ReplPhase,
}

impl ReplStateMachine {
    /// Create a new state machine in Ready phase
    pub fn new() -> Self {
        Self {
            phase: ReplPhase::Ready,
        }
    }

    /// Get the current phase
    pub fn phase(&self) -> &ReplPhase {
        &self.phase
    }

    /// Check if the state machine is in a terminal state
    pub fn is_terminal(&self) -> bool {
        self.phase.is_terminal()
    }

    /// Process an event and transition to a new state
    pub fn process_event(&mut self, event: ReplEvent) -> Result<Transition> {
        let transition = match (&self.phase, &event) {
            // Ready state transitions
            (ReplPhase::Ready, ReplEvent::LineSubmitted { line }) => {
                if line.is_empty() {
                    // Empty line, stay in Ready
                    Transition::to(ReplPhase::Ready)
                } else if line.ends_with('\\') {
                    // Continuation needed
                    let buffer = line.trim_end_matches('\\').to_string();
                    Transition::to(ReplPhase::Continuation { buffer })
                } else {
                    // Try to parse command
                    match Command::parse(line) {
                        Ok(command) => Transition::to_with_follow_up(
                            ReplPhase::Executing {
                                command: command.clone(),
                            },
                            ReplEvent::CommandParsed { command },
                        ),
                        Err(e) => Transition::to_with_output(
                            ReplPhase::Ready,
                            format!("{}: {}", "Parse error".red().bold(), e),
                        ),
                    }
                }
            }

            (ReplPhase::Ready, ReplEvent::Interrupted) => Transition::to_with_output(
                ReplPhase::Ready,
                "^C (Use 'exit' or Ctrl+D to quit)".yellow().to_string(),
            ),

            (ReplPhase::Ready, ReplEvent::Eof) => {
                Transition::to_with_output(ReplPhase::Exiting, "Goodbye!".green().to_string())
            }

            // Continuation state transitions
            (ReplPhase::Continuation { buffer }, ReplEvent::LineSubmitted { line }) => {
                let mut new_buffer = buffer.clone();
                new_buffer.push(' ');
                new_buffer.push_str(line);

                if line.ends_with('\\') {
                    // Still continuing
                    let trimmed = new_buffer.trim_end_matches('\\').to_string();
                    Transition::to(ReplPhase::Continuation { buffer: trimmed })
                } else {
                    // Try to parse complete command
                    match Command::parse(&new_buffer) {
                        Ok(command) => Transition::to_with_follow_up(
                            ReplPhase::Executing {
                                command: command.clone(),
                            },
                            ReplEvent::CommandParsed { command },
                        ),
                        Err(e) => Transition::to_with_output(
                            ReplPhase::Ready,
                            format!("{}: {}", "Parse error".red().bold(), e),
                        ),
                    }
                }
            }

            (ReplPhase::Continuation { .. }, ReplEvent::Interrupted) => Transition::to_with_output(
                ReplPhase::Ready,
                "Continuation cancelled".yellow().to_string(),
            ),

            // Executing state transitions
            (ReplPhase::Executing { .. }, ReplEvent::CommandParsed { command }) => {
                Transition::to(ReplPhase::Executing {
                    command: command.clone(),
                })
            }

            (ReplPhase::Executing { .. }, ReplEvent::CommandExecuted { result }) => match result {
                CommandResult::Continue(output) => {
                    if output.is_empty() {
                        Transition::to(ReplPhase::Ready)
                    } else {
                        Transition::to_with_output(ReplPhase::Ready, output.clone())
                    }
                }
                CommandResult::Exit => {
                    Transition::to_with_output(ReplPhase::Exiting, "Goodbye!".green().to_string())
                }
                CommandResult::Silent => Transition::to(ReplPhase::Ready),
            },

            (
                ReplPhase::Executing { .. },
                ReplEvent::ExecutionError {
                    message,
                    recoverable,
                },
            ) => {
                if *recoverable {
                    Transition::to_with_output(
                        ReplPhase::Ready,
                        format!("{}: {}", "Error".red().bold(), message),
                    )
                } else {
                    Transition::to(ReplPhase::Error {
                        message: message.clone(),
                        recoverable: false,
                    })
                }
            }

            // Error state transitions
            (
                ReplPhase::Error {
                    recoverable: true, ..
                },
                ReplEvent::LineSubmitted { .. },
            ) => {
                // Recoverable error, go back to Ready
                Transition::to(ReplPhase::Ready)
            }

            // Exiting state (terminal)
            (ReplPhase::Exiting, _) => {
                // Already exiting, ignore events
                Transition::to(ReplPhase::Exiting)
            }

            // Catch-all for unexpected transitions
            (current, event) => {
                eprintln!(
                    "{}: Unexpected event {:?} in phase {:?}",
                    "Warning".yellow(),
                    event,
                    current
                );
                Transition::to(ReplPhase::Ready)
            }
        };

        // Update internal state
        self.phase = transition.new_phase.clone();

        Ok(transition)
    }

    /// Reset to Ready phase
    pub fn reset(&mut self) {
        self.phase = ReplPhase::Ready;
    }
}

impl Default for ReplStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ready_to_executing() {
        let mut sm = ReplStateMachine::new();
        assert!(matches!(sm.phase(), ReplPhase::Ready));

        let result = sm.process_event(ReplEvent::LineSubmitted {
            line: "help".to_string(),
        });
        assert!(result.is_ok());
        assert!(matches!(sm.phase(), ReplPhase::Executing { .. }));
    }

    #[test]
    fn test_ready_to_executing_emits_command_parsed_follow_up() {
        let mut sm = ReplStateMachine::new();

        let transition = sm
            .process_event(ReplEvent::LineSubmitted {
                line: "help".to_string(),
            })
            .expect("help should parse");

        assert!(matches!(
            transition.follow_up,
            Some(ReplEvent::CommandParsed { .. })
        ));
    }

    #[test]
    fn test_command_parsed_follow_up_preserves_executing_phase() {
        let mut sm = ReplStateMachine::new();

        let transition = sm
            .process_event(ReplEvent::LineSubmitted {
                line: "help".to_string(),
            })
            .expect("help should parse");

        let follow_up = transition.follow_up.expect("expected CommandParsed event");
        sm.process_event(follow_up)
            .expect("CommandParsed follow-up should be accepted");

        assert!(matches!(sm.phase(), ReplPhase::Executing { .. }));
    }

    #[test]
    fn test_continuation() {
        let mut sm = ReplStateMachine::new();

        // Submit line ending with backslash
        let result = sm.process_event(ReplEvent::LineSubmitted {
            line: "query test\\".to_string(),
        });
        assert!(result.is_ok());
        assert!(matches!(sm.phase(), ReplPhase::Continuation { .. }));
    }

    #[test]
    fn test_interrupt_recovery() {
        let mut sm = ReplStateMachine::new();

        // Interrupt in Ready state
        let result = sm.process_event(ReplEvent::Interrupted);
        assert!(result.is_ok());
        assert!(matches!(sm.phase(), ReplPhase::Ready));
    }

    #[test]
    fn test_eof_exits() {
        let mut sm = ReplStateMachine::new();

        let result = sm.process_event(ReplEvent::Eof);
        assert!(result.is_ok());
        assert!(matches!(sm.phase(), ReplPhase::Exiting));
        assert!(sm.is_terminal());
    }
}
