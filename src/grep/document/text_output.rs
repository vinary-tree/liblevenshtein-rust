//! Shared output builders for document extractors.

use std::fmt::{self, Write as _};

pub(super) fn begin_line(output: &mut String) {
    if !output.is_empty() {
        output.push('\n');
    }
}

pub(super) fn push_line(output: &mut String, line: &str) {
    begin_line(output);
    output.push_str(line);
}

fn begin_section(output: &mut String) {
    if !output.is_empty() {
        output.push_str("\n\n");
    }
}

pub(super) fn push_section(output: &mut String, section: &str) {
    begin_section(output);
    output.push_str(section);
}

pub(super) fn write_section(output: &mut String, args: fmt::Arguments<'_>) {
    begin_section(output);
    output
        .write_fmt(args)
        .expect("writing to String cannot fail");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_line_matches_single_newline_join() {
        let mut output = String::new();
        push_line(&mut output, "first");
        push_line(&mut output, "second");
        push_line(&mut output, "");
        assert_eq!(output, "first\nsecond\n");
    }

    #[test]
    fn push_section_matches_double_newline_join() {
        let mut output = String::new();
        push_section(&mut output, "metadata");
        push_section(&mut output, "");
        push_section(&mut output, "chapter");
        assert_eq!(output, "metadata\n\n\n\nchapter");
    }
}
