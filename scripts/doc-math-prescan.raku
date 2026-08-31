#!/usr/bin/env raku

# doc-math-prescan.raku — fence-aware MathJax conformance scanner.
#
# The documentation house style is deliberately byte-exact:
#
#   * inline math has dollars outside one backtick span:  $`x_i`$
#   * display math uses a fenced block whose info string is `math`
#
# Bare dollar math, dollar-delimited content inside a code span, a stray
# backtick on either side of a valid inline span, and Unicode mathematical
# formulae are rejected. Literal dollars remain valid inside ordinary code
# spans, and fenced source examples are not interpreted as mathematical prose.
#
# With --repository-root, the scanner discovers every Markdown and Rust source
# file through `rg --files`. Living Markdown and every Rustdoc comment are
# scanned. Append-only scientific/release evidence is classified but never
# rewritten; a Markdown path outside the documented repository roots fails as
# `unclassified-markdown` so a new documentation island cannot silently escape
# the gate. Explicit FILE arguments remain available for focused checks.
#
# Usage:
#   raku scripts/doc-math-prescan.raku --lint --repository-root=.
#   raku scripts/doc-math-prescan.raku --lint FILE ...
#   raku scripts/doc-math-prescan.raku --key

# ── Unicode mathematical glyphs and their LaTeX spellings ──────────────────
my constant %KEY = (
    "\x[1D4AA]" => '\mathcal{O}', "\x[1D4AB]" => '\mathcal{P}',
    "\x[2115]" => '\mathbb{N}', "\x[211D]" => '\mathbb{R}',
    "\x[2124]" => '\mathbb{Z}', "\x[211A]" => '\mathbb{Q}',
    "\x[27E8]" => '\langle', "\x[27E9]" => '\rangle',
    "\x[27E6]" => '\llbracket', "\x[27E7]" => '\rrbracket',
    "\x[2308]" => '\lceil', "\x[2309]" => '\rceil',
    "\x[230A]" => '\lfloor', "\x[230B]" => '\rfloor',
    "\x[2223]" => '\mid',
    "\x[2264]" => '\le', "\x[2265]" => '\ge', "\x[2260]" => '\ne',
    "\x[2248]" => '\approx', "\x[2261]" => '\equiv',
    "\x[2291]" => '\sqsubseteq', "\x[2292]" => '\sqsupseteq',
    "\x[226A]" => '\ll', "\x[226B]" => '\gg', "\x[221D]" => '\propto',
    "\x[2280]" => '\nprec', "\x[2208]" => '\in', "\x[2209]" => '\notin',
    "\x[222A]" => '\cup', "\x[2229]" => '\cap',
    "\x[2286]" => '\subseteq', "\x[2282]" => '\subset',
    "\x[2287]" => '\supseteq', "\x[2283]" => '\supset',
    "\x[2205]" => '\emptyset', "\x[2294]" => '\sqcup',
    "\x[2293]" => '\sqcap', "\x[2200]" => '\forall',
    "\x[2203]" => '\exists', "\x[2204]" => '\nexists',
    "\x[2227]" => '\land', "\x[2228]" => '\lor', "\x[00AC]" => '\lnot',
    "\x[22A2]" => '\vdash', "\x[22A8]" => '\models',
    "\x[22A4]" => '\top', "\x[22A5]" => '\bot',
    "\x[220E]" => '\blacksquare', "\x[22C3]" => '\bigcup',
    "\x[22C2]" => '\bigcap', "\x[22C0]" => '\bigwedge',
    "\x[2218]" => '\circ', "\x[2211]" => '\sum', "\x[220F]" => '\prod',
    "\x[2295]" => '\oplus', "\x[2297]" => '\otimes',
    "\x[2299]" => '\odot', "\x[221E]" => '\infty',
    "\x[221A]" => '\sqrt{}', "\x[2207]" => '\nabla',
    "\x[2202]" => '\partial', "\x[00B1]" => '\pm', "\x[2212]" => '-',
    "\x[2234]" => '\therefore', "\x[2225]" => '\parallel',
    "\x[00F7]" => '\div',
    "\x[03B1]" => '\alpha', "\x[03B2]" => '\beta',
    "\x[03B3]" => '\gamma', "\x[03B4]" => '\delta',
    "\x[03B5]" => '\varepsilon', "\x[03B6]" => '\zeta',
    "\x[03B7]" => '\eta', "\x[03B8]" => '\theta',
    "\x[03B9]" => '\iota', "\x[03BA]" => '\kappa',
    "\x[03BB]" => '\lambda', "\x[03BC]" => '\mu',
    "\x[03BD]" => '\nu', "\x[03BE]" => '\xi', "\x[03C0]" => '\pi',
    "\x[03C1]" => '\rho', "\x[03C3]" => '\sigma',
    "\x[03C2]" => '\varsigma', "\x[03C4]" => '\tau',
    "\x[03C5]" => '\upsilon', "\x[03C6]" => '\varphi',
    "\x[03C7]" => '\chi', "\x[03C8]" => '\psi', "\x[03C9]" => '\omega',
    "\x[03A3]" => '\Sigma', "\x[0393]" => '\Gamma',
    "\x[0394]" => '\Delta', "\x[03A9]" => '\Omega',
    "\x[0398]" => '\Theta', "\x[039B]" => '\Lambda',
    "\x[03A0]" => '\Pi', "\x[03A6]" => '\Phi',
    "\x[03A5]" => '\Upsilon', "\x[03A8]" => '\Psi', "\x[039E]" => '\Xi',
);

my constant \MATH = %KEY.keys.Set;
my constant $BLANK = "\x[2420]";

# These paths are historical evidence. The scanner classifies and excludes
# them; corrections belong in a dated append-only erratum.
my constant @APPEND-ONLY-PREFIXES = (
    'benchmarks/causal/evidence/',
    'docs/archive/',
    'docs/scientific-ledger/',
);

my constant @STRICT-RUSTDOC-PATHS = (
    'src/time_series/elastic/interval.rs',
    'src/time_series/elastic/walker.rs',
    'src/time_series/kernels/dtw.rs',
    'src/time_series/kernels/soft_dtw.rs',
    'src/time_series/timestamped_twed_index.rs',
    'src/transducer/language/dyck.rs',
    'src/transducer/operation_set.rs',
    'src/transducer/presets.rs',
    'src/transducer/variants/damerau.rs',
);

sub print-key() {
    say 'Unicode -> LaTeX conversion key (wrap inline results with dollars outside backticks):';
    for %KEY.sort(*.key) -> $pair {
        my $u = 'U+' ~ $pair.key.ord.base(16).fmt('%04s');
        say "  $u  {$pair.key}  ->  {$pair.value}";
    }
    say 'Display formulae use a fenced block whose info string is math.';
    say 'Guards: U+00B5 MICRO SIGN, IPA, literal/currency/regex dollars, and source fences.';
}

sub has-math(Str $text --> Bool) {
    so $text.comb.first({ $_ ∈ MATH }).defined
}

sub is-greek-name(Str $text --> Bool) {
    so $text ~~ /^ <[ \x[0391]..\x[03A9] \x[03B1]..\x[03C9] ]> '-' <[A..Za..z]>+ $/
}

# Return inline-code contents and prose with each complete code span blanked.
sub split-code-spans(Str $line) {
    my @spans;
    my $prose = $line.subst(
        / ('`'+) ( .*? ) $0 /,
        -> $match {
            @spans.push($match[1].Str);
            $BLANK x $match.Str.chars
        },
        :g,
    );
    return { spans => @spans, prose => $prose };
}

# Blank complete double-quoted literals before dollar-delimiter inspection.
# Documentation frequently renders automaton alphabets and padded words as
# strings such as `"$$abc"`; their dollars are data.  Mathematical prose around
# a quoted string remains visible, including any enclosing old-style span.
sub blank-double-quoted-literals(Str $line --> Str) {
    my $blanked = $line;
    my $search = 0;
    loop {
        my $open = $line.index('"', $search);
        last unless $open.defined;
        my $close = $line.index('"', $open + 1);
        last unless $close.defined;
        my $length = $close + 1 - $open;
        $blanked.substr-rw($open, $length) = $BLANK x $length;
        $search = $close + 1;
    }
    $blanked
}

sub looks-like-bare-math(Str $content --> Bool) {
    my $text = $content.trim;
    return False unless $text.chars;
    return True if has-math($text) || $text.contains('\\');
    return True if $text ~~ /^ <[A..Za..z0..9]>+ $/;
    return so ('=', '<', '>', '_', '^', '{', '}', '(', ')', '[', ']',
        '+', '*', '/', '|').first({ $text.contains($_) }).defined;
}

# An old display delimiter is either a standalone exact `$$` token (the usual
# multiline form) or a pair of exact `$$` runs on one prose line.  Requiring an
# exact run and a pair prevents byte strings such as `"$$abc"` and
# `"$$$abc"` from being mistaken for display mathematics.
sub contains-old-display-delimiter(Str $prose --> Bool) {
    return True if $prose.trim eq '$$';

    my @exact-runs;
    my $search = 0;
    loop {
        my $position = $prose.index('$$', $search);
        last unless $position.defined;
        my $before = $position > 0 ?? $prose.substr($position - 1, 1) !! '';
        my $after-position = $position + 2;
        my $after = $after-position < $prose.chars
            ?? $prose.substr($after-position, 1)
            !! '';
        @exact-runs.push($position) if $before ne '$' && $after ne '$';
        $search = $position + 2;
    }
    @exact-runs.elems >= 2
}

# Blank every valid inline span and report malformed boundary bytes or Unicode
# formulae inside it. This runs before ordinary code-span tokenization because
# the backticks are part of the valid GitHub math delimiter.
sub inspect-and-blank-valid-math(Str $line, Int $line-number, @findings --> Str) {
    my $blanked = $line;
    my $search = 0;
    loop {
        my $open = $line.index('$`', $search);
        last unless $open.defined;
        my $close = $line.index('`$', $open + 2);
        last unless $close.defined;
        my $content = $line.substr($open + 2, $close - $open - 2);
        # A valid inline math span cannot contain another backtick. If it does,
        # these bytes are the closing/opening delimiters of two ordinary code
        # spans (for example regex documentation containing both `^x$` and
        # `$`), not one mathematical span.
        if $content.contains('`') {
            $search = $open + 2;
            next;
        }
        my $leading-extra = $open > 0 && $line.substr($open - 1, 1) eq '`';
        my $after = $close + 2;
        my $trailing-extra = $after < $line.chars && $line.substr($after, 1) eq '`';
        if $leading-extra || $trailing-extra {
            @findings.push([
                $line-number,
                'one-sided-malformed-inline-math',
                $line.trim,
            ]);
        }
        if $open > 0 && $line.substr($open - 1, 1) ~~ /<[0..9A..Za..z]>/ {
            @findings.push([$line-number, 'letter-abuts-open', $line.trim]);
        }
        if has-math($content) {
            @findings.push([$line-number, 'unicode-in-mathjax', $content]);
        }
        my $span-length = $close + 2 - $open;
        $blanked.substr-rw($open, $span-length) = $BLANK x $span-length;
        $search = $close + 2;
    }
    $blanked
}

sub inspect-bare-dollars(Str $prose, Int $line-number, @findings) {
    if contains-old-display-delimiter($prose) {
        @findings.push([$line-number, 'old-display-dollar-math', $prose.trim]);
        return;
    }

    my $search = 0;
    loop {
        my $open = $prose.index('$', $search);
        last unless $open.defined;
        my $close = $prose.index('$', $open + 1);
        last unless $close.defined;
        my $content = $prose.substr($open + 1, $close - $open - 1);
        my $after-open = $open + 1 < $prose.chars
            ?? $prose.substr($open + 1, 1)
            !! '';
        my $after-close = $close + 1 < $prose.chars
            ?? $prose.substr($close + 1, 1)
            !! '';
        # `$5 and $10` is two currency amounts, not a mathematical span.
        my $currency-pair = $after-open ~~ /<[0..9]>/ && $after-close ~~ /<[0..9]>/;
        # In a bold Markdown label such as `**$ ↔ s**`, the dollar is a
        # literal symbol and a later example can contain another literal
        # dollar. Do not pair those two symbols as mathematical delimiters.
        my $bold-symbol = $open >= 2
            && $prose.substr($open - 2, 2) eq '**'
            && $content.contains('**');
        if !$currency-pair && !$bold-symbol && looks-like-bare-math($content) {
            @findings.push([$line-number, 'bare-dollar-math', '$' ~ $content ~ '$']);
        }
        $search = $close + 1;
    }
}

# Convert one Rust source into its rendered Rustdoc lines while preserving
# original line numbers. Ordinary source and non-doc comments are excluded.
sub documentation-lines(Str $path) {
    my @raw = $path.IO.lines;
    return @raw.kv.map(-> $index, $line { [$index + 1, $line] })
        unless $path.ends-with('.rs');

    my @docs;
    my $in-block = False;
    for @raw.kv -> $index, $raw-line {
        my $trimmed = $raw-line.trim-leading;
        if $in-block {
            my $end = $trimmed.index('*/');
            my $content = $end.defined ?? $trimmed.substr(0, $end) !! $trimmed;
            $content = $content.substr(1) if $content.starts-with('*');
            @docs.push([$index + 1, $content.trim-leading]);
            $in-block = False if $end.defined;
            next;
        }
        if $trimmed.starts-with('///') && !$trimmed.starts-with('////') {
            @docs.push([$index + 1, $trimmed.substr(3).trim-leading]);
        } elsif $trimmed.starts-with('//!') {
            @docs.push([$index + 1, $trimmed.substr(3).trim-leading]);
        } elsif ($trimmed.starts-with('/**') && !$trimmed.starts-with('/***'))
                || $trimmed.starts-with('/*!') {
            my $content = $trimmed.substr(3);
            my $end = $content.index('*/');
            @docs.push([
                $index + 1,
                ($end.defined ?? $content.substr(0, $end) !! $content).trim-leading,
            ]);
            $in-block = !$end.defined;
        }
    }
    @docs
}

# ── Markdown table structure ────────────────────────────────────────────────
sub table-cell-count(Str $row --> Int) {
    my $text = $row.trim;
    return -1 unless $text.contains('|');
    $text = $text.subst(/ ('`'+) .*? $0 /, -> $m { '#' x $m.Str.chars }, :g);
    $text ~~ s:g/ \\ '|' /##/;
    return -1 unless $text.contains('|');
    $text ~~ s/ ^ \s* '|' //;
    $text ~~ s/ '|' \s* $ //;
    $text.split('|').elems
}

sub is-separator(Str $row --> Bool) {
    my $text = $row.trim;
    so $text ~~ /^ <[ | \- : \s ]>+ $/ && $text.contains('-') && $text.contains('|')
}

sub scan-file(Str $path, Bool :$strict-prose = True) {
    my @findings;
    my $in-fence = False;
    my $math-fence = False;
    my $fence-marker = '';
    my $previous-row = Str;
    my $table-columns = 0;

    for documentation-lines($path) -> $entry {
        my ($line-number, $line) = $entry.List;
        my $lead = $line.trim-leading;
        if $lead.starts-with('```') || $lead.starts-with('~~~') {
            my $marker = $lead.substr(0, 3);
            if !$in-fence {
                $in-fence = True;
                $fence-marker = $marker;
                $math-fence = $lead.substr(3).trim eq 'math';
            } elsif $marker eq $fence-marker {
                $in-fence = False;
                $math-fence = False;
                $fence-marker = '';
            }
            next;
        }
        if $in-fence {
            if $strict-prose && $math-fence && has-math($line) {
                @findings.push([$line-number, 'unicode-in-display-math', $line.trim]);
            }
            next;
        }

        my $math-blanked = inspect-and-blank-valid-math(
            $line,
            $line-number,
            @findings,
        );
        my %code = split-code-spans($math-blanked);
        my @spans = %code<spans>.list;
        my $prose = %code<prose>;

        for @spans -> $span {
            next if is-greek-name($span);
            @findings.push([$line-number, 'backticked-unicode-math', "`$span`"])
                if $strict-prose && has-math($span);
            @findings.push([$line-number, 'code-wrapped-dollar-math', "`$span`"])
                if $span.chars >= 3 && $span.starts-with('$') && $span.ends-with('$');
        }

        if $strict-prose {
            my $bare-prose = $prose;
            $bare-prose ~~ s:g/ <[ \x[0391]..\x[03A9] \x[03B1]..\x[03C9] ]> '-'
                <?before <[A..Za..z]>> /-/;
            $bare-prose ~~ s:g/ '](' <-[)]>* ')' / /;
            my @bare = $bare-prose.comb.grep(* ∈ MATH).unique;
            @findings.push([$line-number, 'bare-unicode-math', @bare.join(' ')]) if @bare;

            if $prose ~~ / << 'O(' <[ 0..9 A..Z a..z ( \x[2308] \x[2223] | ]> / {
                @findings.push([$line-number, 'bare-O', ~$/]);
            }
        }
        # Dollar/backtick structure is unambiguous enough to enforce on every
        # Rustdoc surface, including prose whose Unicode alphabet is not
        # formula-like (for example phonetic documentation).
        inspect-bare-dollars(
            blank-double-quoted-literals($prose),
            $line-number,
            @findings,
        );

        if is-separator($line) {
            my $columns = table-cell-count($line);
            with $previous-row {
                my $header-columns = table-cell-count($_);
                @findings.push([$line-number - 1, 'table-column-mismatch', $_.trim])
                    if $header-columns > $columns;
            }
            $table-columns = $columns;
        } elsif $table-columns > 0 {
            my $columns = table-cell-count($line);
            if $columns <= 0 {
                $table-columns = 0;
            } elsif $columns > $table-columns {
                @findings.push([$line-number, 'table-column-mismatch', $line.trim]);
            }
        }
        $previous-row = $line;
    }
    @findings
}

sub is-append-only(Str $path --> Bool) {
    return True if @APPEND-ONLY-PREFIXES.first({ $path.starts-with($_) }).defined;
    return True if $path.starts-with('docs/releases/')
        && $path ne 'docs/releases/README.md';
    return True if $path eq 'docs/bindings/FINDINGS_LEDGER.md'
        || $path eq 'docs/verification/FINDINGS_LEDGER.md'
        || $path eq 'benchmarks/cross-language/legacy/javascript/vendor/LICENSE.vendored.md';
    False
}

sub load-classification(Str $root, Str $relative) {
    my $path = $root.IO.add($relative);
    die "missing documentation classification: $relative" unless $path.e;
    $path.lines
        .grep({ .trim.chars && !.trim.starts-with('#') })
        .map({ .trim => True })
        .Hash
}

sub repository-targets(Str $root) {
    my $proc = run(
        'rg', '--files', '-g', '*.md', '-g', '*.rs',
        :cwd($root), :out, :err,
    );
    my @paths = $proc.out.lines.sort;
    my $errors = $proc.err.slurp;
    $proc.out.close;
    $proc.err.close;
    die "rg repository discovery failed: $errors" if $proc.exitcode != 0;

    my @targets;
    my @unclassified;
    my %living = load-classification($root, 'docs/.mathlint-include.txt');
    my %legacy = load-classification($root, 'docs/.mathlint-legacy.txt');
    for @paths -> $path {
        # These are scanner fixtures rather than project documentation.
        next if $path.starts-with('tests/fixtures/doc-math/');
        if $path.ends-with('.rs') {
            my $strict = so @STRICT-RUSTDOC-PATHS.first({ $path eq $_ }).defined;
            @targets.push([$path, $root.IO.add($path).Str, $strict]);
        } elsif is-append-only($path) {
            next;
        } elsif %living{$path}:exists {
            @targets.push([$path, $root.IO.add($path).Str, True]);
        } elsif %legacy{$path}:exists {
            next;
        } else {
            @unclassified.push($path);
        }
    }
    { targets => @targets, unclassified => @unclassified }
}

sub MAIN(
    *@files,
    Bool :$lint = False,
    Bool :$key = False,
    Str :$repository-root,
) {
    if $key {
        print-key();
        exit 0;
    }
    if @files && $repository-root.defined {
        note 'choose explicit FILE arguments or --repository-root, not both';
        exit 2;
    }

    my @targets;
    my @unclassified;
    if $repository-root.defined {
        my %repository = repository-targets($repository-root);
        @targets = %repository<targets>.list;
        @unclassified = %repository<unclassified>.list;
    } else {
        unless @files {
            note 'usage: doc-math-prescan.raku [--lint] [--repository-root=DIR] FILE ...';
            exit 2;
        }
        @targets = @files.map({ [$_, $_, True] });
    }

    my $total = 0;
    my %by-kind;
    for @unclassified -> $path {
        $total++;
        %by-kind<unclassified-markdown>++;
        say "$path:0: unclassified-markdown: classify this path as living or append-only"
            if $lint;
    }

    for @targets -> $target {
        my ($display, $path, $strict-prose) = $target.List;
        unless $path.IO.e {
            note "skip (not found): $display";
            next;
        }
        my @found = scan-file($path, :$strict-prose);
        next unless @found;
        $total += @found.elems;
        %by-kind{.[1]}++ for @found;
        if $lint {
            say "$display:{.[0]}: {.[1]}: {.[2]}" for @found;
        } else {
            say "-- $display  ({@found.elems} finding(s))";
            say "   L{.[0]}  {.[1].fmt('%-34s')} {.[2]}" for @found;
        }
    }

    unless $lint {
        say '';
        say "Summary: $total finding(s) across {@targets.elems} scanned file(s)";
        say "  {.value.fmt('%4d')}  {.key}" for %by-kind.sort(-*.value);
        say 'Run with --key for the Unicode-to-LaTeX conversion table.';
    }
    exit($total > 0 && $lint ?? 1 !! 0);
}
