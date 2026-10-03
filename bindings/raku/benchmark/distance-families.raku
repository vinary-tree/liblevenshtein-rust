use Liblevenshtein;

# Execute all seven families in all three public unit domains, exact and
# thresholded. Ratios compare a dispatcher to the standard-distance dispatcher
# in the same domain/mode; this removes much host/NativeCall and machine-speed
# variation. Use a quiet single-worker run with the RC baseline before
# tightening these deliberately generous regression ceilings.
my constant %MAX-RATIO =
    standard => 1.5,
    osa => 2.5,
    true => 4.5,
    merge => 2.5,
    hamming => 2.5,
    indel => 3,
    affine => 3.5;

my $iterations = (%*ENV<RAKU_DISTANCE_BENCH_ITERATIONS> // 2_000).Int;
my $samples = (%*ENV<RAKU_DISTANCE_BENCH_SAMPLES> // 5).Int;
die 'iterations and samples must be positive' unless $iterations > 0 && $samples > 0;
my $check = @*ARGS.grep(* eq '--check').elems > 0;
die 'only --check is accepted' if @*ARGS.grep(* ne '--check').elems;

my $costs = AffineGapCosts.new(gap-open => 2, gap-extend => 1,
    substitution => 3);
my %inputs =
    text => ['kittenkitten', 'sittenkitten'],
    bytes => [Buf.new('kittenkitten'.encode.list),
              Buf.new('sittenkitten'.encode.list)],
    tokens => ['kittenkitten'.ords.Array, 'sittenkitten'.ords.Array];

sub measure(&operation --> Num:D) {
    my $sink = 0;
    $sink += operation() // 0 for ^100;
    my @times = gather for ^$samples {
        my $started = now;
        $sink += operation() // 0 for ^$iterations;
        take ((now - $started) * 1_000_000_000 / $iterations).Num;
    }
    die 'unreachable benchmark sink' if $sink < 0;
    @times.sort[$samples div 2]
}

sub score(Str:D $family, $left, $right, Bool:D $bounded) {
    given $family {
        when 'standard' { $bounded ?? distance($left, $right, :threshold(3)) !! distance($left, $right) }
        when 'osa' { $bounded ?? damerau-distance($left, $right, :threshold(3)) !! damerau-distance($left, $right) }
        when 'true' { $bounded ?? true-damerau-distance($left, $right, :threshold(3)) !! true-damerau-distance($left, $right) }
        when 'merge' { $bounded ?? merge-and-split-distance($left, $right, :threshold(3)) !! merge-and-split-distance($left, $right) }
        when 'hamming' { $bounded ?? hamming-distance($left, $right, :threshold(3)) !! hamming-distance($left, $right) }
        when 'indel' { $bounded ?? indel-distance($left, $right, :threshold(3)) !! indel-distance($left, $right) }
        when 'affine' { $bounded ?? affine-gap-distance($left, $right, $costs, :threshold(3)) !! affine-gap-distance($left, $right, $costs) }
        default { die "unknown family $family" }
    }
}

say "domain\tmode\tfamily\tmedian_ns_per_call\tstandard_ratio\tmax_ratio";
my @failures;
for <text bytes tokens> -> $domain {
    my ($left, $right) = %inputs{$domain}.list;
    for False, True -> $bounded {
        my $mode = $bounded ?? 'bounded' !! 'exact';
        my $baseline = measure({ score('standard', $left, $right, $bounded) });
        for <standard osa true merge hamming indel affine> -> $family {
            my $elapsed = $family eq 'standard'
                ?? $baseline
                !! measure({ score($family, $left, $right, $bounded) });
            my $ratio = $elapsed / $baseline;
            say "$domain\t$mode\t$family\t{$elapsed.fmt('%.1f')}\t{$ratio.fmt('%.2f')}\t{%MAX-RATIO{$family}}";
            @failures.push("$domain/$mode/$family: {$ratio.fmt('%.2f')} > {%MAX-RATIO{$family}}")
                if $ratio > %MAX-RATIO{$family};
        }
    }
}
die @failures.join("\n") if $check && @failures;
