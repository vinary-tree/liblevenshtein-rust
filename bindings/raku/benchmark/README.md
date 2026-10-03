# Raku standalone-distance dispatch budget

`distance-families.raku` measures every public standalone family over text,
byte, and u64-token domains, in exact and thresholded modes: 42 dispatcher
paths. It uses the same fixed equal-length, one-substitution input pattern in
each domain. Thus Hamming has a defined score, and the thresholded runs retain
an in-bound result. The measured value includes Raku argument validation,
domain marshalling, NativeCall, and the native kernel; it is **not** a
native-only kernel benchmark. The facade owns a `CArray` copy of each byte
argument for the duration of a call; its one-slot backing allocation for an
empty argument still passes logical length zero. This makes the boundary
stable under Rakudo's garbage collection and repeated empty/nonempty calls.

From the repository root, after building the matching native library and
making its dependencies available to the loader:

```sh
LIBLEVENSHTEIN_LIBRARY=/absolute/path/to/libliblevenshtein.so \
  raku -Ibindings/raku/lib bindings/raku/benchmark/distance-families.raku --check
```

`RAKU_DISTANCE_BENCH_ITERATIONS` (default 2,000) and
`RAKU_DISTANCE_BENCH_SAMPLES` (default 5) control run size. The script prints
median nanoseconds per call, a ratio to the standard-distance path on the
**same domain and mode**, and that family's maximum permitted ratio. The
ratio budget applies independently to each of the 42 paths. These tolerant
ceilings are regression alarms, not claims of absolute throughput or a
requirement that a weighted affine alignment equal Levenshtein's cost.
Measure a quiet, pinned worker and repeat before judging a budget failure;
record CPU, build profile, inputs, iterations, samples, and all results. The
existing `compare.raku` remains a smaller standard-distance smoke benchmark.

The first complete budget run on 2026-10-03 used an AMD Ryzen Threadripper PRO
5975WX (32 physical cores), a `cargo build --locked -j1 --lib --features ffi`
debug native library, 2,000 iterations in each of five samples per path, an 8 GiB memory
cap, and a 200% CPU quota. The worker was not core-pinned, so these are
provisional comparative medians, not cross-machine throughput promises. All
42 checks passed. The highest observed ratio in each family across the six
domain/mode combinations was:

| Family | Highest observed ratio | Current per-path budget |
|---|---:|---:|
| Standard | 1.00 (baseline) | 1.5 |
| Optimal-string-alignment | 1.41 | 2.5 |
| Unrestricted Damerau | 2.79 | 4.5 |
| Merge/split | 1.38 | 2.5 |
| Hamming | 1.36 | 2.5 |
| Indel | 1.50 | 3.0 |
| Affine gap | 1.91 | 3.5 |

These bounds leave roughly 60% or more headroom over the observed worst
ratio while still detecting large dispatcher regressions. Before changing a
budget, compare multiple controlled runs and retain the failing evidence.

Specialized-path scope follows the Rust implementation: standard Unicode
ASCII inputs at most 64 bytes can take Myers, longer applicable Unicode
inputs can take runtime SIMD, and raw byte standard distance can take Myers
when its shorter operand fits one word. These are internal exact-equivalent
dispatches. Raku has no user-selectable Myers/SIMD function because the
native public contract exposes distance semantics, not kernel controls.
Hamming is linear, indel bounded calls use a diagonal band, and affine
bounded calls still compute the exact Gotoh result before comparison. The
current budget covers one short input shape; profiling and size-sweep
experiments are needed before interpreting it as a long-input guarantee.
