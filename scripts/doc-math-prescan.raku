#!/usr/bin/env raku

# doc-math-prescan.raku — fence-aware MathJax-conformance scanner for the liblevenshtein docs.
#
# The documentation house style requires MathJax LaTeX for all mathematical prose:
#   * inline math  = a backtick code span whose content is dollar-delimited, e.g.
#                    `$\mathcal{O}(\lvert W\rvert)$`
#   * display math = a fenced block whose info-string is `math`
# and forbids (a) Unicode-literal formulae (`𝒪(∣W∣)`, `⟨i,e⟩`, `≤` …) whether backticked or
# bare, (b) bare undelimited big-O (`O(n)`) in prose, and (c) bare `$…$` / `$$…$$`.
#
# This scanner reports what still needs converting. It NEVER edits anything.
#
# Modes:
#   (default)   brief  — per-file, human-readable list of residual old-style math + a summary.
#   --lint             — CI gate: same detection, machine-terse; exit 1 if ANY file still
#                        carries old-style math (Unicode-literal or bare-O()) or leaked
#                        bare-dollar LaTeX; exit 0 when clean.
#   --key              — print the Unicode→LaTeX conversion key and exit.
#
# Usage:
#   raku scripts/doc-math-prescan.raku [--lint] FILE ...
#   raku scripts/doc-math-prescan.raku --lint $(cat living-docs.txt)
#   raku scripts/doc-math-prescan.raku --key
#
# Guards (never flagged / never converted): the U+00B5 MICRO SIGN (µs, microseconds) as
# distinct from U+03BC GREEK MU; contents of fenced code blocks; IPA letters; MeTTa `$var`,
# currency `$5`, and regex `$` in prose; ASCII `|` table delimiters and code pipes; context
# `×`/`·`/arrows outside a formula.

# ── The Unicode → LaTeX conversion key (the clearly-mathematical glyphs) ─────────────────
my constant %KEY = (
    # script / blackboard / brackets
    "\x[1D4AA]" => '\mathcal{O}', "\x[1D4AB]" => '\mathcal{P}',
    "\x[2115]" => '\mathbb{N}', "\x[211D]" => '\mathbb{R}', "\x[2124]" => '\mathbb{Z}', "\x[211A]" => '\mathbb{Q}',
    "\x[27E8]" => '\langle', "\x[27E9]" => '\rangle', "\x[27E6]" => '\llbracket', "\x[27E7]" => '\rrbracket',
    "\x[2308]" => '\lceil', "\x[2309]" => '\rceil', "\x[230A]" => '\lfloor', "\x[230B]" => '\rfloor',
    "\x[2223]" => '\mid',                       # cardinality bar (also \lvert…\rvert in |…| context)
    # relations
    "\x[2264]" => '\le', "\x[2265]" => '\ge', "\x[2260]" => '\ne', "\x[2248]" => '\approx',
    "\x[2261]" => '\equiv', "\x[2291]" => '\sqsubseteq', "\x[2292]" => '\sqsupseteq',
    "\x[226A]" => '\ll', "\x[226B]" => '\gg', "\x[221D]" => '\propto', "\x[2280]" => '\nprec',
    # set / logic
    "\x[2208]" => '\in', "\x[2209]" => '\notin', "\x[222A]" => '\cup', "\x[2229]" => '\cap',
    "\x[2286]" => '\subseteq', "\x[2282]" => '\subset', "\x[2287]" => '\supseteq', "\x[2283]" => '\supset',
    "\x[2205]" => '\emptyset', "\x[2294]" => '\sqcup', "\x[2293]" => '\sqcap',
    "\x[2200]" => '\forall', "\x[2203]" => '\exists', "\x[2204]" => '\nexists',
    "\x[2227]" => '\land', "\x[2228]" => '\lor', "\x[00AC]" => '\lnot',
    "\x[22A2]" => '\vdash', "\x[22A8]" => '\models', "\x[22A4]" => '\top', "\x[22A5]" => '\bot',
    "\x[220E]" => '\blacksquare', "\x[22C3]" => '\bigcup', "\x[22C2]" => '\bigcap', "\x[22C0]" => '\bigwedge',
    # operators / accents
    "\x[2218]" => '\circ', "\x[2211]" => '\sum', "\x[220F]" => '\prod',
    "\x[2295]" => '\oplus', "\x[2297]" => '\otimes', "\x[2299]" => '\odot',
    "\x[221E]" => '\infty', "\x[221A]" => '\sqrt{}', "\x[2207]" => '\nabla', "\x[2202]" => '\partial',
    "\x[00B1]" => '\pm', "\x[2212]" => '-', "\x[2234]" => '\therefore', "\x[2225]" => '\parallel',
    "\x[00F7]" => '\div',
    # greek lowercase (math)
    "\x[03B1]" => '\alpha', "\x[03B2]" => '\beta', "\x[03B3]" => '\gamma', "\x[03B4]" => '\delta',
    "\x[03B5]" => '\varepsilon', "\x[03B6]" => '\zeta', "\x[03B7]" => '\eta', "\x[03B8]" => '\theta',
    "\x[03B9]" => '\iota', "\x[03BA]" => '\kappa', "\x[03BB]" => '\lambda', "\x[03BC]" => '\mu',
    "\x[03BD]" => '\nu', "\x[03BE]" => '\xi', "\x[03C0]" => '\pi', "\x[03C1]" => '\rho',
    "\x[03C3]" => '\sigma', "\x[03C2]" => '\varsigma', "\x[03C4]" => '\tau', "\x[03C5]" => '\upsilon',
    "\x[03C6]" => '\varphi', "\x[03C7]" => '\chi', "\x[03C8]" => '\psi', "\x[03C9]" => '\omega',
    # greek uppercase (math)
    "\x[03A3]" => '\Sigma', "\x[0393]" => '\Gamma', "\x[0394]" => '\Delta', "\x[03A9]" => '\Omega',
    "\x[0398]" => '\Theta', "\x[039B]" => '\Lambda', "\x[03A0]" => '\Pi', "\x[03A6]" => '\Phi',
    "\x[03A5]" => '\Upsilon', "\x[03A8]" => '\Psi', "\x[039E]" => '\Xi',
);

# The set of glyphs whose presence — inside a backtick span, or bare in prose — signals a
# formula that must be converted. Explicitly EXCLUDES U+00B5 (µ, microseconds) and the
# context-gated glyphs (× · … → ↔) which are only math inside an existing formula.
my constant \MATH = %KEY.keys.Set;

sub print-key() {
    say "Unicode -> LaTeX conversion key (wrap the result in an inline backtick-dollar span or a fenced math block):";
    for %KEY.sort(*.key) -> $p {
        my $u = 'U+' ~ $p.key.ord.base(16).fmt('%04s');
        say "  $u  {$p.key}  ->  {$p.value}";
    }
    say "  U+2223 |  (inside |…| cardinality/abs-value)  ->  \\lvert … \\rvert";
    say "  sub/superscripts ² ³ ⁿ ₀…₉ ₙ ᵢ …            ->  ^\{…\} / _\{…\}";
    say "  GUARDS (never convert): µ U+00B5 (microseconds); IPA; var/currency/regex dollars; table |; × · -> outside a formula";
}

# Split a text line into (code-span contents, prose-with-spans-blanked). Inline code spans are
# runs of one-or-more backticks … matching run; we blank them in the prose so bare-O()/`$…$`
# scanning does not see code, and we return their contents so we can test them for math glyphs.
sub split-code-spans(Str $line) {
    my @spans;
    my $prose = $line.subst(
        / ('`'+) ( .*? ) $0 /,
        -> $m { @spans.push($m[1].Str); "\x[2420]" x $m.Str.chars },   # blank (SYMBOL FOR SPACE)
        :g,
    );
    return { spans => @spans, prose => $prose };
}

sub has-math(Str $s --> Bool) { so $s.comb.any ∈ MATH }

# ── Scan one file ───────────────────────────────────────────────────────────────────────
sub scan-file(Str $path) {
    my @findings;                      # (line-no, kind, excerpt)
    my $in-fence = False;
    my $fence-marker = '';
    my @lines = $path.IO.lines;
    for @lines.kv -> $i, $line {
        my $lno = $i + 1;
        # fence tracking (``` or ~~~, length ≥ 3); info-strings ride the opening fence only
        my $lead = $line.trim-leading;
        if $lead.starts-with('```') || $lead.starts-with('~~~') {
            my $marker = $lead.substr(0, 3);
            if !$in-fence { $in-fence = True; $fence-marker = $marker }
            elsif $marker eq $fence-marker { $in-fence = False; $fence-marker = '' }
            next;
        }
        next if $in-fence;             # code-fence body: never math

        my %cs    = split-code-spans($line);
        my @spans = %cs<spans>.list;
        my $prose = %cs<prose>;

        # (a) old-style math inside an inline code span. Greek-letter terminology (π-calculus,
        # λ-theory, ε-transition, α-approximation) is a proper-noun NAME, not a formula, so a span
        # that is exactly "<Greek>-<word>" is exempt (Unicode is correct for such names).
        for @spans -> $s {
            next if $s ~~ / ^ <[ \x[0391]..\x[03A9] \x[03B1]..\x[03C9] ]> '-' <[A..Za..z]>+ $ /;
            @findings.push([$lno, 'backticked-unicode-math', "`$s`"]) if has-math($s);
        }
        # (b) bare Unicode math glyph in prose (display math written as plain text). Blank the same
        # Greek-hyphen-word terminology before the check so a bare `λ-calculus` is not flagged.
        my $proseb = $prose;
        $proseb ~~ s:g/ <[ \x[0391]..\x[03A9] \x[03B1]..\x[03C9] ]> '-' <?before <[A..Za..z]>> /-/;
        $proseb ~~ s:g/ '](' <-[)]>* ')' / /;   # blank Markdown link URLs — anchors are not math
        my @bare = $proseb.comb.grep(* ∈ MATH).unique;
        @findings.push([$lno, 'bare-unicode-math', @bare.join(' ')]) if @bare;
        # (c) bare undelimited big-O in prose (complexity, not an identifier)
        if $prose ~~ / << 'O(' <[ 0..9 A..Z a..z ( \x[2308] \x[2223] | ]> / {
            @findings.push([$lno, 'bare-O', ~$/]);
        }
        # (d) leaked bare-dollar LaTeX in prose (dollars around a backslash-command, not
        #     currency / MeTTa var): `$ … \cmd … $`
        if $prose ~~ / '$' <-[$]>* '\\' <[a..zA..Z]>+ <-[$]>* '$' / {
            @findings.push([$lno, 'leaked-dollar-latex', ~$/]);
        }
        # (e) target-form hygiene: an ASCII word char abutting a `$ opening delimiter
        if $line ~~ / <[0..9A..Za..z]> '`$' / {
            @findings.push([$lno, 'letter-abuts-open', ~$/]);
        }
    }
    return @findings;
}

# ── Main ────────────────────────────────────────────────────────────────────────────────
sub MAIN(*@files, Bool :$lint = False, Bool :$key = False) {
    if $key { print-key(); exit 0 }
    unless @files {
        note "usage: doc-math-prescan.raku [--lint] [--key] FILE ...";
        exit 2;
    }
    my $total = 0;
    my %by-kind;
    for @files -> $f {
        unless $f.IO.e { note "skip (not found): $f"; next }
        my @found = scan-file($f);
        next unless @found;
        $total += @found.elems;
        %by-kind{.[1]}++ for @found;
        if $lint {
            say "$f:{.[0]}: {.[1]}: {.[2]}" for @found;
        } else {
            say "── $f  ({@found.elems} finding(s))";
            say "   L{.[0]}  {.[1].fmt('%-24s')} {.[2]}" for @found;
        }
    }
    unless $lint {
        say "";
        say "Summary: $total finding(s) across {@files.elems} file(s)";
        say "  {.value.fmt('%4d')}  {.key}" for %by-kind.sort(-*.value);
        say "Run with --key for the Unicode→LaTeX conversion table.";
    }
    exit($total > 0 && $lint ?? 1 !! 0);
}
