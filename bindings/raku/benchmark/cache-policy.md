# Raku query-cache policy workloads

Run `cache-policy.raku` with the liblevenshtein, libdictenstein, and Vinary
Tree Raku modules on `RAKULIB` and the two native libraries selected through
`LIBLEVENSHTEIN_LIBRARY` and `LIBDICTENSTEIN_LIBRARY`. It builds a fixed
128-term Unicode dictionary and runs 512 complete queries per policy on three
traces: eight repeated hot terms, a 128-term scan, and a 75% hot mixture.

The CSV includes elapsed time and native requests, hits, misses, admissions,
rejections, evictions, residency, and an exact-result checksum. The script
checks request accounting and the two-shard hard bounds. `default` and
`tuned` both use native TinyLFU/SIEVE; `tuned` fixes the seed and uses a
128-column sketch with an age interval of 64. `disabled` measures exact
uncached traversal. Timings are workload observations, not API guarantees.

The dictionary eviction wrappers have different semantics and are not used
as query-cache policies. Their metadata maps are not bounded by the
`QueryCache` limits, and TTL changes provider visibility rather than simply
evicting a memoized result.

One smoke run on 2026-10-14 used a debug `raku-bindings` native library,
Rakudo, a 1 GiB memory cap, and a one-CPU quota. Every policy returned the
same checksum for each workload. These elapsed times include Raku iteration,
NativeCall, and exact result materialization; they are not native-kernel
microbenchmarks and are too short for stable speed ratios.

| Workload | Policy | Elapsed ms | Hits | Misses | Admissions | Rejections | Evictions | Final entries |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hot | default | 374.048 | 504 | 8 | 8 | 0 | 0 | 8 |
| hot | tuned | 306.620 | 504 | 8 | 8 | 0 | 0 | 8 |
| hot | disabled | 335.267 | 0 | 512 | 0 | 512 | 0 | 0 |
| mixed | default | 312.200 | 378 | 134 | 93 | 41 | 77 | 16 |
| mixed | tuned | 311.991 | 378 | 134 | 114 | 20 | 98 | 16 |
| mixed | disabled | 345.832 | 0 | 512 | 0 | 512 | 0 | 0 |
| scan | default | 277.512 | 0 | 512 | 193 | 319 | 177 | 16 |
| scan | tuned | 258.382 | 0 | 512 | 324 | 188 | 308 | 16 |
| scan | disabled | 262.001 | 0 | 512 | 0 | 512 | 0 | 0 |
