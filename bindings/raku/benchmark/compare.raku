use Liblevenshtein;

sub sample(&operation, Int:D :$warmup = 10_000, Int:D :$iterations = 100_000,
    Int:D :$samples = 9 --> Map:D) {
    operation() for ^$warmup;
    my @values = gather for ^$samples {
        my $started = now;
        operation() for ^$iterations;
        take (now - $started) * 1_000_000_000 / $iterations;
    }
    my @sorted = @values.sort;
    Map.new(
        minimum => @sorted[0],
        median => @sorted[@sorted.elems div 2],
        maximum => @sorted[*-1],
    )
}

say 'standard distance ns/op: ',
    sample({ distance('levenshtein', 'liblevenshtein') });
say 'thresholded distance ns/op: ',
    sample({ distance('levenshtein', 'liblevenshtein', :threshold(4)) });
