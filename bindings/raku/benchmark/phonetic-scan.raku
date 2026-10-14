use Liblevenshtein;

my $input = 'cat dog ' x 128;
my $matcher = PhoneticOnlineGrep.new(pattern => 'cat');
LEAVE $matcher.close;
my @samples;
for ^25 -> $iteration {
    my $start = now;
    my @hits = $matcher.scan($input, max-input-bytes => $input.encode.elems,
        max-matches => 128);
    die 'native match count changed' unless @hits.elems == 128;
    @samples.push((now - $start).Numeric * 1_000_000)
        if $iteration >= 5;
}
@samples = @samples.sort;
say sprintf('raku_scan median_us=%.3f matches=128 bytes=%d runs=20',
    (@samples[9] + @samples[10]) / 2, $input.encode.elems);
