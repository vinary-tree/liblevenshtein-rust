# Liblevenshtein.jl

Fast Unicode edit distances and snapshot-consistent fuzzy search from Julia,
backed by liblevenshtein-rust.

## Install and verify

This RC package is tested from its source tree. It requires Julia 1.10 or newer,
the `VinaryTreeInterop` Julia package, and the matching native libraries. Point
the loader at the exact liblevenshtein build:

```sh
export LIBLEVENSHTEIN_LIBRARY="$PWD/target/release/libliblevenshtein.so"
julia --project=bindings/julia/Liblevenshtein \
  -e 'using Pkg; Pkg.test()'
```

macOS uses `libliblevenshtein.dylib`; Windows uses
`liblevenshtein.dll`. A future registered artifact will select the platform
binary without this development override.

## Quick start

```julia
using Liblevenshtein

distance("kitten", "sitting")                         # 3
distance("kitten", "sitting"; threshold=2)            # nothing
damerau_distance("ab", "ba")                         # 1
```

For dictionary search, create or receive a `VinaryTreeInterop.Dictionary`, then
retain it in a transducer. Every query captures one immutable dictionary
revision:

```julia
transducer = Transducer(dictionary, ALGORITHM_STANDARD)
try
    matches = collect(query(transducer, "speling", 2;
        order=ORDER_DISTANCE_THEN_TERM))
finally
    close(transducer)
end
```

`dictionary` may come from libdictenstein or a customer-defined provider that
implements `vt.dictionary.v1`; liblevenshtein does not depend on a concrete
dictionary backend.

## Choose an automaton

| Julia value | Semantics | Metric? |
|---|---|---:|
| `ALGORITHM_STANDARD` | Insert, delete, and substitute | yes |
| `ALGORITHM_TRANSPOSITION` | Optimal string alignment with adjacent swaps | no |
| `ALGORITHM_MERGE_AND_SPLIT` | Standard edits plus symmetric merge/split | yes |
| `ALGORITHM_DAMERAU_LEVENSHTEIN` | Unrestricted, history-composable adjacent swaps | yes |

Optimal string alignment and unrestricted Damerau-Levenshtein differ on edit
histories: the former assigns distance 3 from `CA` to `ABC`, while the latter
assigns distance 2. Choose the algorithm when constructing `Transducer`; query
domains, batching, and snapshot behavior are otherwise identical.

## Common and intended usage

- Use `distance`, `damerau_distance`, and `true_damerau_distance` for pairwise
  Unicode-scalar metrics. A thresholded call returns `nothing` when the exact
  value exceeds the inclusive threshold.
- Reuse a `Transducer` for repeated queries with the same dictionary and
  algorithm. Use `snapshot(transducer)` when several queries must observe the
  same revision even while the live dictionary changes.
- Iterate a `QueryCursor` for independently owned `Match` values. Use
  `reduce_batches!` for high-volume processing where callback-scoped native
  batches amortize the FFI boundary.
- Compile reusable `PhoneticPattern` and `PhoneticRuleSet` values only when the
  native `BUILD_FEATURE_PHONETIC` bit is present.

## API reference

| API | Contract |
|---|---|
| `distance(a, b; threshold=nothing)` | Standard Unicode Levenshtein distance. |
| `damerau_distance(a, b; threshold=nothing)` | Optimal-string-alignment distance. |
| `true_damerau_distance(a, b; threshold=nothing)` | Unrestricted Damerau-Levenshtein distance. |
| `Transducer(resource, algorithm)` | Retains a `VinaryTreeInterop.Resource` or `.Dictionary`. |
| `snapshot(transducer)` | Owned immutable transducer revision. |
| `unit_domain(transducer)` | `BYTE`, `UNICODE_SCALAR`, or `U64`. |
| `query(transducer, input, k; order=...)` | Lazy query over `String`, `Vector{UInt8}`, integer tokens, or a pattern. |
| `next_batch!(cursor, maximum)` | Copy one bounded leased batch into owned matches. |
| `reduce_batches!(f, initial, cursor; batch_size=256)` | Invoke `f(accumulator, BorrowedBatch)` on lexical zero-copy batches and consume the cursor. |
| `PhoneticPattern(source; llre=false)` | Compile regex or import-free LLRE source. |
| `input in pattern`, `size(pattern)` | Membership and structural size. |
| `PhoneticRuleSet(source_or_kind)` | Parse rules or select a built-in rule set. |
| `rules(input)`, `length(rules)` | Rewrite text and count enabled rules. |
| `close`, `isopen` | Deterministic lifecycle for every native owner. |

`Match.term` is a `String`, `Vector{UInt8}`, or `Vector{UInt64}` according to
`Match.unit_domain`; `Match.id` is `nothing` or `UInt64`. Enum values and ABI
constants are generated from `bindings/api.json`.

## Ownership, snapshots, and batching

Constructors retain provider resources; callers keep ownership of their own
handles. A cursor owns its query-start snapshot and remains valid after its
source transducer or dictionary closes. Ordinary iteration copies terms before
releasing each generation. A `BorrowedMatch` and every view derived from it are
valid only during the current `reduce_batches!` callback; access afterward
throws. The reducer callback is contained at the C boundary, and a Julia
exception is rethrown only after native traversal has returned and the cursor
has closed.

Close long-lived values deterministically with `close` in `finally`. Finalizers
are leak containment, not a scheduling guarantee. A cursor is exclusive and
single-consumer. Independent cursors and immutable transducers may be queried
from separate tasks; do not race `close` against another operation on the same
wrapper.

## Errors, compatibility, and security

Fallible native statuses become `NativeError` with the exact numeric status,
operation, and a copied thread-local diagnostic. ABI generation 1 is checked at
module initialization; newer additive API revisions are accepted. Unit domains
remain distinct and invalid UTF-8, domain mismatches, stale leases, closed
providers, and malformed provider output fail explicitly.

Treat custom dictionary providers as untrusted code. Native negotiation
validates their versioned vtables and converts provider failures into statuses.
Set application-specific traversal and result limits, avoid retaining lexical
batch pointers, and report vulnerabilities through the repository security
policy.

## Performance and release

Transducer construction is constant-time resource negotiation. Query work is
lazy over a captured revision. Iteration allocates owned host terms by design;
`reduce_batches!` keeps descriptors and term storage native for the callback and
uses the ABI default of 256 matches to amortize calls. Benchmark native,
pairwise-FFI, iterator, and reducer paths separately on an idle host before
making performance claims.

The Julia package name is `Liblevenshtein`, without an organization prefix.
Release publication is intentionally disabled on this RC5 feature branch; a
signed source tag and registry review remain required before General-registry
registration and Documenter deployment.
