#!/usr/bin/env raku

# doc-math-transpose.raku — one-time repair for the MathJax delimiter-transposition
# antipattern introduced by the 2026-07-13 math campaign.
#
# The campaign's converter (scripts/doc-math-convert.raku) emitted inline math with its
# delimiters transposed: it wrote a backtick code span whose CONTENT is dollar-delimited —
#
#     `$\mathcal{O}(\lvert W\rvert)$`        ← ANTIPATTERN (backticks OUTSIDE, dollars inside)
#
# which GitHub renders as literal monospace text `$…$`, NOT as math. The correct GitHub
# MathJax inline form is a dollar-delimited span whose CONTENT is a backtick code span —
#
#     $`\mathcal{O}(\lvert W\rvert)`$        ← CORRECT (dollars OUTSIDE, backticks inside)
#
# This script transposes the former into the latter. It is fence-aware and TOKENIZER-based
# (it pairs backtick runs exactly like scripts/doc-math-prescan.raku), which guarantees two
# properties a naive `\`\$…\$\`` regex cannot:
#
#   * it never matches the "glue" between two already-correct adjacent spans — e.g. the
#     incidental `` `$ and $` `` substring inside `` $`a`$ and $`b`$ `` is NOT a code span,
#     so it is left untouched;
#   * it is IDEMPOTENT — after transposition a span's content is the bare LaTeX (no
#     surrounding `$`), so a second run matches nothing.
#
# Only SINGLE-backtick spans whose content both starts and ends with `$` (length ≥ 3) are
# rewritten; fenced code blocks, multi-backtick spans, and currency/variable dollars that do
# not form a `$…$` span are never touched.
#
# Usage:
#   raku scripts/doc-math-transpose.raku [--dry] FILE ...
#   raku scripts/doc-math-transpose.raku --dry $(fd -e md)
#
#   --dry : print changed lines (old → new) and per-file counts; do NOT write.
#
# GUARD: docs/DOCUMENTATION_OVERHAUL_LEDGER.md is skipped — its `$…$`-in-backticks spans are
# *illustrative* meta-examples describing this very antipattern, not live math.

my constant $LEDGER-GUARD = 'DOCUMENTATION_OVERHAUL_LEDGER';

# Transpose every `$…$`-content single-backtick span on one (non-fence) line.
sub transpose-line(Str $line --> Str) {
    $line.subst(
        / ('`'+) (.*?) $0 /,
        -> $m {
            my $ticks   = $m[0].Str;
            my $content = $m[1].Str;
            if $ticks.chars == 1
               && $content.chars >= 3
               && $content.starts-with('$')
               && $content.ends-with('$') {
                my $inner = $content.substr(1, $content.chars - 2);   # strip outer $ … $
                '$`' ~ $inner ~ '`$'                                  # dollars → outside
            } else {
                $m.Str                                               # code / multi-tick: keep
            }
        },
        :g,
    );
}

sub process(Str $path, Bool :$dry --> Int) {
    my $raw          = $path.IO.slurp;
    my $had-final-nl = $raw.ends-with("\n");
    my @out;
    my $in-fence = False;
    my $marker   = '';
    my $changed  = 0;
    for $raw.lines -> $line {
        my $lead = $line.trim-leading;
        if $lead.starts-with('```') || $lead.starts-with('~~~') {
            my $m = $lead.substr(0, 3);
            if !$in-fence { $in-fence = True; $marker = $m }
            elsif $m eq $marker { $in-fence = False; $marker = '' }
            @out.push($line);
            next;
        }
        if $in-fence { @out.push($line); next }
        my $new = transpose-line($line);
        if $new ne $line { $changed++; say "  - $line\n  + $new" if $dry }
        @out.push($new);
    }
    # Write ONLY when the transposition actually changed something, and preserve the file's
    # original end-of-file newline convention exactly — never re-normalise (or even re-touch)
    # a file the transposition did not modify.
    if $changed > 0 && !$dry {
        spurt $path, @out.join("\n") ~ ($had-final-nl ?? "\n" !! "");
    }
    return $changed;
}

sub MAIN(*@files, Bool :$dry = False) {
    unless @files {
        note "usage: doc-math-transpose.raku [--dry] FILE ...";
        exit 2;
    }
    my $total = 0;
    my $touched = 0;
    for @files -> $f {
        unless $f.IO.e { note "skip (missing): $f"; next }
        if $f.contains($LEDGER-GUARD) { note "skip (meta-doc guard): $f"; next }
        my $c = process($f, :$dry);
        next unless $c;
        $touched++;
        $total += $c;
        say "{$dry ?? '[dry] ' !! ''}$f: $c line(s) changed";
    }
    say "──────────────────────────────────────────────────────────────";
    say "TOTAL: $total line(s) across $touched file(s) " ~ ($dry ?? 'would change' !! 'changed');
}
