use Liblevenshtein;

module Producer {
    use Libdictenstein;
    our sub dictionary() { dynamic-dawg() }
}

my $dictionary = Producer::dictionary();
for ^128 -> $index { $dictionary{"term-$index"} = $index }
my $provider = $dictionary.snapshot;
my $transducer = Transducer.new(:dictionary($provider));
LEAVE $transducer.close;

my @hot = (^512).map({ "term-{$_ % 8}" }).Array;
my @scan = (^512).map({ "term-{$_ % 128}" }).Array;
my @mixed = (^512).map({ $_ % 4 == 0 ?? "term-{$_ % 128}" !! "term-{$_ % 8}" }).Array;
my %workloads = hot => @hot, scan => @scan, mixed => @mixed;
my %policies =
    default => QueryCachePolicy.new(max-entries => 16, max-weight => 2**20),
    tuned => QueryCachePolicy.new(
        max-entries => 16, max-weight => 2**20,
        sketch-width => 128, age-after => 64, seed => 42,
    ),
    disabled => QueryCachePolicy.disabled;

say 'workload,policy,queries,elapsed_ms,hits,misses,admissions,rejections,evictions,resident_entries,resident_weight,checksum';
for %workloads.keys.sort -> $workload {
    for %policies.keys.sort -> $name {
        my $cache = QueryCache.new(:$transducer, policy => %policies{$name});
        my $checksum = 0;
        my $start = now;
        for %workloads{$workload}.list -> $query {
            $checksum += $cache.query($query, 1).list.elems;
        }
        my $elapsed-ms = (now - $start).Num * 1000;
        my $stats = $cache.stats;
        die 'request accounting mismatch'
            unless $stats.requests == $stats.hits + $stats.misses;
        die 'entry bound exceeded' if $stats.resident-entries > 32;
        die 'weight bound exceeded' if $stats.resident-weight > 2 * 2**20;
        say join ',', $workload, $name, $stats.requests,
            sprintf('%.3f', $elapsed-ms),
            $stats.hits, $stats.misses, $stats.admissions,
            $stats.rejections, $stats.evictions,
            $stats.resident-entries, $stats.resident-weight, $checksum;
        $cache.close;
    }
}
