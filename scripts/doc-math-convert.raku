#!/usr/bin/env raku

# doc-math-convert.raku — guarded mechanical converter: backticked-Unicode-literal math spans and
# bare `O(...)` complexity → MathJax `$…$`. Fence-aware; operates ONLY on the given file list
# (pass LIVING docs only — never records). Leaves bare-Unicode-math *prose lines* (inference
# rules / display math) for hand-conversion to ```math blocks; the scanner still flags those.
#
# Usage:  raku scripts/doc-math-convert.raku [--dry] FILE ...
#   --dry : print unified-ish preview (changed lines only) to stdout, do NOT write.
#
# It converts, inside a backtick code span that contains at least one clearly-math glyph, every
# math glyph to LaTeX (pairing ∣…∣ / |…| → \lvert…\rvert, grouping sub/superscripts, − → -), and
# rewrites the span as `$…$`. Separately, bare `O(args)` in prose → `$\mathcal{O}(args)$`.
# GUARDS (never touched): fenced code blocks; `µ` U+00B5; the glyph set excludes it; MeTTa `$var`
# and currency `$` are never introduced/removed; table `|` outside a math span is left.

my %SUP = '⁰'=>'0','¹'=>'1','²'=>'2','³'=>'3','⁴'=>'4','⁵'=>'5','⁶'=>'6','⁷'=>'7','⁸'=>'8','⁹'=>'9',
          'ⁿ'=>'n','ⁱ'=>'i','⁺'=>'+','⁻'=>'-','⁼'=>'=','⁽'=>'(','⁾'=>')';
my %SUB = '₀'=>'0','₁'=>'1','₂'=>'2','₃'=>'3','₄'=>'4','₅'=>'5','₆'=>'6','₇'=>'7','₈'=>'8','₉'=>'9',
          'ₙ'=>'n','ᵢ'=>'i','ⱼ'=>'j','ₖ'=>'k','ₗ'=>'l','ₘ'=>'m','ₚ'=>'p','ₛ'=>'s','ₜ'=>'t',
          'ₐ'=>'a','ₑ'=>'e','ₒ'=>'o','ₓ'=>'x','₊'=>'+','₋'=>'-','₌'=>'=','₍'=>'(','₎'=>')';

# glyph → LaTeX. Commands get a trailing space (safe next to letters/digits); brace-terminated and
# closing-delimiter commands do not. ∣ and the sub/superscripts are handled separately (not here).
my @GLYPH =                       # ordered list of (glyph, latex) pairs
  '𝒪'=>'\mathcal{O}', '𝒫'=>'\mathcal{P}', 'ℕ'=>'\mathbb{N} ', 'ℝ'=>'\mathbb{R} ', 'ℤ'=>'\mathbb{Z} ', 'ℚ'=>'\mathbb{Q} ',
  '⟨'=>'\langle ', '⟩'=>'\rangle', '⟦'=>'\llbracket ', '⟧'=>'\rrbracket', '⌈'=>'\lceil ', '⌉'=>'\rceil', '⌊'=>'\lfloor ', '⌋'=>'\rfloor',
  '≤'=>'\le ', '≥'=>'\ge ', '≠'=>'\ne ', '≈'=>'\approx ', '≡'=>'\equiv ', '⊑'=>'\sqsubseteq ', '⊒'=>'\sqsupseteq ', '≪'=>'\ll ', '≫'=>'\gg ', '∝'=>'\propto ', '⊀'=>'\nprec ', '≺'=>'\prec ', '≻'=>'\succ ', '⪯'=>'\preceq ', '⪰'=>'\succeq ',
  '∈'=>'\in ', '∉'=>'\notin ', '∪'=>'\cup ', '∩'=>'\cap ', '⊆'=>'\subseteq ', '⊂'=>'\subset ', '⊇'=>'\supseteq ', '⊃'=>'\supset ', '∅'=>'\emptyset ', '⊔'=>'\sqcup ', '⊓'=>'\sqcap ',
  '∀'=>'\forall ', '∃'=>'\exists ', '∄'=>'\nexists ', '∧'=>'\land ', '∨'=>'\lor ', '¬'=>'\lnot ', '⊢'=>'\vdash ', '⊨'=>'\models ', '⊤'=>'\top ', '⊥'=>'\bot ', '∎'=>'\blacksquare ', '⋃'=>'\bigcup ', '⋂'=>'\bigcap ', '⋀'=>'\bigwedge ',
  '∘'=>'\circ ', '∑'=>'\sum ', '∏'=>'\prod ', '⊕'=>'\oplus ', '⊗'=>'\otimes ', '⊙'=>'\odot ', '∞'=>'\infty ', '√'=>'\sqrt ', '∇'=>'\nabla ', '∂'=>'\partial ', '±'=>'\pm ', '∓'=>'\mp ', '×'=>'\times ', '·'=>'\cdot ', '÷'=>'\div ', '∴'=>'\therefore ', '∵'=>'\because ', '∥'=>'\parallel ', '…'=>'\dots ',
  '↦'=>'\mapsto ', '⇒'=>'\Rightarrow ', '⇐'=>'\Leftarrow ', '⇔'=>'\Leftrightarrow ', '⟹'=>'\implies ', '⟺'=>'\iff ', '⟶'=>'\longrightarrow ', '→'=>'\to ', '←'=>'\leftarrow ', '↔'=>'\leftrightarrow ', '⇝'=>'\rightsquigarrow ', '↪'=>'\hookrightarrow ',
  'α'=>'\alpha ', 'β'=>'\beta ', 'γ'=>'\gamma ', 'δ'=>'\delta ', 'ε'=>'\varepsilon ', 'ζ'=>'\zeta ', 'η'=>'\eta ', 'θ'=>'\theta ', 'ι'=>'\iota ', 'κ'=>'\kappa ', 'λ'=>'\lambda ', 'μ'=>'\mu ', 'ν'=>'\nu ', 'ξ'=>'\xi ', 'π'=>'\pi ', 'ρ'=>'\rho ', 'σ'=>'\sigma ', 'ς'=>'\varsigma ', 'τ'=>'\tau ', 'υ'=>'\upsilon ', 'φ'=>'\varphi ', 'χ'=>'\chi ', 'ψ'=>'\psi ', 'ω'=>'\omega ',
  'Σ'=>'\Sigma ', 'Γ'=>'\Gamma ', 'Δ'=>'\Delta ', 'Ω'=>'\Omega ', 'Θ'=>'\Theta ', 'Λ'=>'\Lambda ', 'Π'=>'\Pi ', 'Φ'=>'\Phi ', 'Υ'=>'\Upsilon ', 'Ψ'=>'\Psi ', 'Ξ'=>'\Xi ',
  '−'=>'-', '⁻'=>'-';

# Context-gated glyphs: real math *inside* a formula, but also common in prose/code/benchmarks
# (arrows, ×, ·, …). They are CONVERTED when inside a triggered span, but do NOT by themselves
# mark a span as math — mirroring the scanner, so we never mangle `lexer → AST` or `3 × 4`.
my \GATED = set ('→','←','↔','↦','⇒','⇐','⇔','⟹','⟺','⟶','⇝','↪','×','·','…','∥');
# TRIGGER = clear-math glyphs whose presence in a backtick span marks it as a formula.
# ∣ (U+2223, handled separately by pair-bars) is added explicitly so bar-only spans trigger.
my \TRIGGER = set('∣', |@GLYPH.map(*.key).grep({ $_ ∉ GATED }));
# ALLMATH = every math glyph (trigger + gated); used as a run-token test in wrap-bare-math.
my \ALLMATH = set('∣', |@GLYPH.map(*.key));

sub group-scripts(Str $in) {
    my $s = $in;
    $s ~~ s:g/ (<[⁰¹²³⁴⁵⁶⁷⁸⁹ⁿⁱ⁺⁻⁼⁽⁾]>+) /{ '^{' ~ $0.Str.comb.map({ %SUP{$_} // $_ }).join ~ '}' }/;
    $s ~~ s:g/ (<[₀₁₂₃₄₅₆₇₈₉ₙᵢⱼₖₗₘₚₛₜₐₑₒₓ₊₋₌₍₎]>+) /{ '_{' ~ $0.Str.comb.map({ %SUB{$_} // $_ }).join ~ '}' }/;
    return $s;
}

sub pair-bars(Str $in, Str $bar) {
    my $s = $in; my $open = True;
    $s ~~ s:g/ $bar /{ my $r = $open ?? '\lvert ' !! '\rvert'; $open = !$open; $r }/;
    return $s;
}

# Known math functions/operators spelled with letters — set upright as operators.
my %FUNC = 'log'=>'\log ', 'ln'=>'\ln ', 'lg'=>'\lg ', 'min'=>'\min ', 'max'=>'\max ',
           'exp'=>'\exp ', 'sup'=>'\sup ', 'inf'=>'\inf ', 'lim'=>'\lim ', 'gcd'=>'\gcd ',
           'mod'=>'\bmod ', 'sin'=>'\sin ', 'cos'=>'\cos ', 'tan'=>'\tan ', 'det'=>'\det ',
           'arg'=>'\arg ', 'dim'=>'\dim ', 'ker'=>'\ker ', 'deg'=>'\deg ';

# Convert the inner content of a math expression (already known to be math).
sub convert-content(Str $in) {
    my $s = $in;
    # Multi-letter ASCII words → \func (known operators) or upright \text{…}; lone letters stay
    # italic variables. Run before glyph conversion so it never touches emitted \commands.
    $s ~~ s:g/ (<[A..Za..z]> ** 2..*) /{ my $w = $0.Str; %FUNC{$w.lc} // ('\text{' ~ $w ~ '}') }/;
    $s = group-scripts($s);
    $s = pair-bars($s, '∣');                 # U+2223 cardinality (unambiguous)
    # ASCII '|' is deliberately NOT paired: it is ambiguous (cardinality vs Rholang-parallel vs
    # bitwise-or vs escaped table pipe `\|`) and already renders as a bar in math mode.
    for @GLYPH -> $p { $s ~~ s:g/ $($p.key) /$($p.value)/ }
    $s ~~ s:g/ \s+ $ //;                      # trim trailing space introduced by a command
    return $s;
}

sub has-math(Str $s --> Bool) { so $s.comb.any (elem) TRIGGER }

my \SUBSUP = set(|%SUP.keys, |%SUB.keys);
my $LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
my $DIGITS  = '0123456789';
my $MATHOPS = "+-/=()[].,|'^_<>";      # ASCII operators/brackets valid inside a math run
                                       # ('*' excluded: it is markdown bold/italic far more often
                                       #  than multiplication here, which uses × / · / \times.)

# Wrap BARE (non-backticked) math runs in prose as `$…$`. A "run" is a maximal contiguous span of
# math tokens (glyphs, digits, operators, sub/superscripts, and LONE single letters) joined by single
# spaces, that contains at least one TRIGGER glyph. The "lone letter" rule (a letter is a token only
# when neither neighbour is a letter) means a run can never swallow a prose WORD (2+ letters), so
# prose is safe. Multi-glyph expressions like ⟨i, e⟩ or δ = |x − y| are captured whole; a bare
# inference rule Γ ⊢ t : A becomes an inline `$…$` span (conformant, if not a display block).
sub wrap-bare-math(Str $prose) {
    my @c = $prose.comb;
    my $n = @c.elems;
    return $prose if $n == 0;
    my sub letter($i) { 0 <= $i < $n && $LETTERS.contains(@c[$i]) }
    my sub tok($i) {
        return False unless 0 <= $i < $n;
        my $ch = @c[$i];
        return True if $ch (elem) ALLMATH;                 # any math glyph
        return True if $ch (elem) SUBSUP;                  # sub/superscript
        return True if $DIGITS.contains($ch);
        return True if $MATHOPS.contains($ch);
        return True if $LETTERS.contains($ch) && !letter($i-1) && !letter($i+1);   # lone letter
        return False;
    }
    my @out; my $i = 0;
    while $i < $n {
        unless tok($i) { @out.push(@c[$i]); $i++; next }
        my $start = $i; my $j = $i;
        loop {
            if tok($j+1) { $j++ }
            elsif $j+2 < $n && @c[$j+1] eq ' ' && tok($j+2) { $j += 2 }
            else { last }
        }
        # Trim trailing/leading "loose" punctuation (space, forward-opening brackets, dangling
        # binary ops) — but keep balanced closers — so a neighbouring prose word (e.g. the
        # "(Standard)" after "χ = ε (") does not make the run look embedded.
        my ($ts, $tj) = ($start, $j);
        my $tail = ' ([{+=,/<.:;-';
        my $lead = ' )]}+=,/>.:;';
        while $tj >= $ts && $tail.contains(@c[$tj])   { $tj-- }
        while $ts <= $tj && $lead.contains(@c[$ts])   { $ts++ }
        my $has-trigger = $ts <= $tj && so ($ts..$tj).grep({ @c[$_] (elem) TRIGGER });
        # Embedded guard: if the (trimmed) run directly abuts (no space) ASCII-math structure —
        # {, }, ^, _, or a letter — it is a FRAGMENT of a larger expression (e.g. `A^{ND,χ}_n`).
        my $embedded = ($ts > 0     && ('{}^_' ~ $LETTERS).contains(@c[$ts-1]))
                    || ($tj < $n-1  && ('{}^_' ~ $LETTERS).contains(@c[$tj+1]));
        if $has-trigger && !$embedded {
            @out.push(@c[$start ..^ $ts].join) if $ts > $start;              # trimmed lead
            @out.push('`$' ~ convert-content(@c[$ts .. $tj].join) ~ '$`');   # wrapped core
            @out.push(@c[($tj+1) .. $j].join) if $tj < $j;                   # trimmed tail
        } else {
            @out.push(@c[$start..$j].join);
        }
        $i = $j + 1;
    }
    return @out.join;
}

my $BARE-ONLY = False;   # --bare-only: skip backticked-span conversion (for signature-bearing files)

sub convert-line(Str $line) {
    my @spans;
    # 1. Protect backtick spans behind placeholders, converting the math ones → `$…$`.
    my $prose = $line.subst(
        / ('`'+) ( .*? ) $0 /,
        -> $m {
            my $v = (!$BARE-ONLY && $m[0].Str.chars == 1 && has-math($m[1].Str))
                ?? '`$' ~ convert-content($m[1].Str) ~ '$`'
                !! $m.Str;                                   # code / multi-backtick: leave as-is
            @spans.push($v);
            "\x[FDD0]{@spans.end}\x[FDD1]"                    # noncharacter placeholder
        },
        :g,
    );
    # 2. Bare O(...) complexity in prose → protected `$\mathcal{O}(…)$`.
    $prose ~~ s:g/ << 'O(' (<-[)]>+) ')' /{
        @spans.push('`$\mathcal{O}(' ~ convert-content($0.Str) ~ ')$`');
        "\x[FDD0]{@spans.end}\x[FDD1]"
    }/;
    # 2.5 Math-identifier expressions with ASCII super/subscripts — X^{…}_n(…), d^χ_L(…), d²_L,
    #     L^χ_{Lev}(n,w) — wrapped whole (only when they carry a real math signal: a ^, a {, or a
    #     non-ASCII glyph — so plain snake_case identifiers are not swept up).
    $prose ~~ s:g/
        ( [ <[A..Za..z]> | <[ \x[0391]..\x[03A9] \x[03B1]..\x[03C9] ]> ]+ [ <[_^]> [ '{' <-[}]>* '}' | <[A..Za..z0..9]>+ | <-[\s(){}\[\]_^]> ] ]+ [ '(' <-[()]>* ')' ]? )
    /{
        my $mm = $0.Str;
        # Require a literal '^' (superscript): that is what math notation like A^{ND,χ}_n / L^χ
        # carries and what UPPER_SNAKE_CASE pseudocode (ELEMENTARY_TRANSITION) lacks — so we never
        # sweep up a snake_case identifier that merely has a Unicode param in its (…) args.
        if $mm.contains('^') {
            @spans.push('`$' ~ convert-content($mm) ~ '$`');
            "\x[FDD0]{@spans.end}\x[FDD1]"
        } else { $mm }
    }/;
    # 2.7 Bare set-literals {…} (non-nested) that contain a Unicode math glyph → `$\{…\}$`.
    $prose ~~ s:g/ '{' (<-[{}]>*) '}' /{
        my $c = $0.Str;
        if so $c.comb.grep(*.ord > 127) {
            @spans.push('`$\{' ~ convert-content($c) ~ '\}$`');
            "\x[FDD0]{@spans.end}\x[FDD1]"
        } else { '{' ~ $c ~ '}' }
    }/;
    # 3. Bare Unicode-math runs in the remaining prose → `$…$`.
    $prose = wrap-bare-math($prose);
    # 4. Restore protected spans.
    $prose ~~ s:g/ \x[FDD0] (\d+) \x[FDD1] /{ @spans[$0.Int] }/;
    return $prose;
}

sub process(Str $path, Bool :$dry) {
    my @out; my $in-fence = False; my $marker = ''; my $changed = 0;
    for $path.IO.lines -> $line {
        my $lead = $line.trim-leading;
        if $lead.starts-with('```') || $lead.starts-with('~~~') {
            my $m = $lead.substr(0,3);
            if !$in-fence { $in-fence = True; $marker = $m }
            elsif $m eq $marker { $in-fence = False; $marker = '' }
            @out.push($line); next;
        }
        if $in-fence { @out.push($line); next }
        my $new = convert-line($line);
        if $new ne $line { $changed++; say "  - $line\n  + $new" if $dry }
        @out.push($new);
    }
    unless $dry { spurt $path, @out.join("\n") ~ "\n" }
    return $changed;
}

sub MAIN(*@files, Bool :$dry = False, Bool :$bare-only = False) {
    $BARE-ONLY = $bare-only;
    my $total = 0;
    for @files -> $f {
        unless $f.IO.e { note "skip (missing): $f"; next }
        my $c = process($f, :$dry);
        $total += $c;
        say "{$dry ?? '[dry] ' !! ''}$f: $c line(s) changed";
    }
    say "TOTAL: $total line(s) " ~ ($dry ?? 'would change' !! 'changed');
}
