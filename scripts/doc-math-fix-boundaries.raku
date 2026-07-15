#!/usr/bin/env raku

# doc-math-fix-boundaries.raku — one-time repair for MATH-SPAN BOUNDARY corruption left by the
# 2026-07-13 campaign's converter. That converter did not just transpose delimiters (fixed
# separately by doc-math-transpose.raku); its span-bounding logic also SWALLOWED adjacent
# markdown structure INTO the math span, e.g.
#
#     **Alphabet $`(\Sigma )** -`$ The set …      ← bold-close `**` + em-dash trapped in the math
#     $`- \chi = t`$: With transposition          ← a list bullet trapped in the math
#     $`| **\pi , \rho ** |`$ Position variables  ← table pipes + bold trapped in the math
#
# In every case the LaTeX belongs inside `$`…`$` and the markdown tokens (bullet `- `/`N. `,
# bold `**`, table pipes `| … |`, and the trailing definition dash) belong OUTSIDE. This script
# hoists them back out:
#
#     **Alphabet $`(\Sigma )`$** - The set …
#     - $`\chi = t`$: With transposition
#     | **$`\pi , \rho`$** | Position variables …
#
# It is DELIBERATELY conservative — it only rewrites a span when it can fully account for the
# non-math tokens, and it BAILS (leaving the span untouched, to be hand-fixed) whenever `**`
# would remain embedded in the middle of the LaTeX. Two context gates keep it from mangling
# legitimate math:
#   * table pipes are hoisted ONLY on a recognised corrupted table row (the line, trimmed,
#     begins with `$`|` — as the glossary symbol tables do), so a cardinality/abs `| … |` inside
#     ordinary prose math is never disturbed;
#   * a leading bullet `- ` is hoisted ONLY when the span begins the line (modulo indentation),
#     so a genuine leading minus such as `$`- \times X`$` mid-sentence is left alone.
#
# Usage:  raku scripts/doc-math-fix-boundaries.raku [--dry] FILE ...
#   --dry : print every (old → new) line and per-file counts; write nothing.
#
# GUARD: docs/DOCUMENTATION_OVERHAUL_LEDGER.md is skipped (its spans are meta-examples).

my constant $LEDGER-GUARD = 'DOCUMENTATION_OVERHAUL_LEDGER';

# Rebuild a single span's content, hoisting leading/trailing markdown out of the `$`…`$`.
# Returns the full replacement text (prefix ~ $`core`$ ~ suffix) or Nil to leave the span as-is.
sub rebound(Str $content, Bool :$table-row, Bool :$line-start --> Str) {
    my $c   = $content;
    my $pre = '';
    my $suf = '';

    # ── leading tokens (in source order: table-pipe, bullet/number, bold) ──
    if $table-row && $c ~~ s/^ '|' \s+ // { $pre ~= '| '; }
    if $line-start && $c ~~ s/^ ( '-' \s+ | \d+ '.' \s+ ) // { my $b = ~$0; $b ~~ s/\s+$//; $pre ~= "$b "; }
    if $c ~~ s/^ '**' // { $pre ~= '**'; }

    # ── trailing tokens (in source order from the end: table-pipe, then bold + optional dash) ──
    if $table-row && $c ~~ s/ \s+ '|' $// { $suf = " |$suf"; }
    if $c ~~ s/ '**' \s* (<[ - \x[2013] \x[2014] ]>)? \s* $// {
        my $dash = $0 ?? " {~$0}" !! '';
        $suf = "**$dash$suf";
    }

    # Confine changes to genuine boundary corruption: if no markdown token was hoisted out, leave
    # the span exactly as-is (do NOT normalise the harmless internal double-spaces the old
    # converter left, which would balloon the diff far beyond the corruption being repaired).
    return Str if $pre eq '' && $suf eq '';
    # If bold markers are still embedded in the LaTeX, this span is irregular — hand-fix it.
    return Str if $c.contains('**');
    # On a table row, a pipe still inside the core means the span straddled a cell boundary
    # (e.g. `\beta(x,w) | N/A`) — too tangled to auto-repair; leave it for hand reconstruction.
    return Str if $table-row && $c.contains('|');

    $c ~~ s/^ \s+ //;              # trim only the whitespace exposed at the hoist boundary
    $c ~~ s/ \s+ $//;
    return Str if $c eq '';

    return "{$pre}\$`{$c}`\$$suf";
}

sub fix-line(Str $line --> Str) {
    my $table-row = $line.trim-leading.starts-with('$`|');
    my $out = '';
    my $pos = 0;
    for ($line ~~ m:g/ '$`' (<-[`]>+) '`$' /) -> $m {
        $out ~= $line.substr($pos, $m.from - $pos);        # text before this span
        my $before = $line.substr(0, $m.from);
        my $line-start = so $before ~~ / ^ \s* $ /;         # only whitespace precedes the span
        my $rep = rebound($m[0].Str, :$table-row, :$line-start);
        $out ~= $rep.defined ?? $rep !! $m.Str;
        $pos = $m.to;
    }
    $out ~= $line.substr($pos);
    return $out;
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
        my $new = fix-line($line);
        if $new ne $line { $changed++; say "  - $line\n  + $new" if $dry }
        @out.push($new);
    }
    if $changed > 0 && !$dry {
        spurt $path, @out.join("\n") ~ ($had-final-nl ?? "\n" !! "");
    }
    return $changed;
}

sub MAIN(*@files, Bool :$dry = False) {
    unless @files { note "usage: doc-math-fix-boundaries.raku [--dry] FILE ..."; exit 2; }
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
