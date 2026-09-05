# Domain-preserving standalone distances

## Purpose

The standalone distance API computes exact or thresholded edit distance
without constructing a dictionary. API revision 4 exposes the same four
unit-cost families over three distinct input domains, so a host binding never
needs to reimplement an algorithm or reinterpret binary data as text.

| Domain | Native representation | Equality | Typical use |
|---|---|---|---|
| Unicode scalar | Valid UTF-8 plus byte length | Decoded scalar value | Human-language text |
| Byte | Arbitrary `uint8_t` sequence | Byte value | Binary identifiers and already encoded alphabets |
| Token | Aligned `uint64_t` sequence | Unsigned token value | Words, labels, phonemes, or customer-defined symbols |

The domains are intentionally not interchangeable. For example, byte `0xff`
is valid binary input but cannot appear alone in a valid UTF-8 string.

## Distance families

Let $`D(i,j)`$ be the minimum cost of transforming the first $`i`$ source
units into the first $`j`$ target units. Every family includes unit-cost
insertion, deletion, and substitution. They differ only in the additional
history transitions they admit.

| Julia function | Additional edit | Mathematical property |
|---|---|---|
| `distance` | None | Metric |
| `optimal_string_alignment_distance` | One adjacent swap, with a substring edited at most once | Symmetric, but not a metric |
| `true_damerau_distance` | History-composable adjacent swaps | Metric under the published unit costs |
| `merge_and_split_distance` | Two source units to one target unit, or one source unit to two target units | Metric |

The merge/split recurrence adds these two candidates to the ordinary
Levenshtein recurrence:

```math
D(i,j) = \min\!\left(D(i-2,j-1)+1,\;D(i-1,j-2)+1,\;\ldots\right).
```

The unrestricted Damerau implementation uses the Lowrance--Wagner
last-occurrence recurrence. Optimal string alignment is deliberately a
different operation: `CA` to `ABC` costs 3 under optimal string alignment and
2 under unrestricted Damerau--Levenshtein. The underlying transposition model
was introduced by [Lowrance and Wagner](https://doi.org/10.1145/321879.321880).

## Shared implementation

The generic kernels re-exported from Rust's `distance` module are the semantic
source for scalar, byte, and token computation. Standard byte distance may
select Myers's bit-vector implementation when its pattern fits a machine word;
the fallback and every other domain retain the same result. See
[Myers's bit-vector algorithm](https://doi.org/10.1145/316542.316550).

Exact standard distance stores two rows. Optimal string alignment and
merge/split store three rows. Each row uses the shorter operand as its column
dimension, and semantics-preserving common affixes are removed before the
merge/split matrix is allocated. Thresholded versions evaluate only the
diagonal band that can still reach the inclusive bound. Unrestricted Damerau
retains its full history matrix because a valid transposition may refer to a
non-adjacent earlier row.

For bound $`k`$, source length $`m`$, and target length $`n`$, the banded
families follow this fail-closed procedure. `family_cell` denotes the selected
standard, optimal-string-alignment, or merge/split recurrence; its inputs are
the retained predecessor rows and the current source and target units.

```text
BOUNDED-DISTANCE(family, source, target, k):
    remove a semantics-preserving common affix when selected by the family
    let m and n be the retained source and target lengths
    if |m - n| > k: return ABOVE-THRESHOLD
    initialize each retained row to the sentinel k + 1
    for i from 1 through m:
        low  := max(1, i - k)
        high := min(n, i + k)
        reset the current row to k + 1
        for j from low through high:
            current[j] := family_cell(i, j)
        rotate the retained rows
    if final cell <= k: return final cell
    return ABOVE-THRESHOLD
```

The unrestricted Damerau family shares the length lower-bound rejection but
computes its exact Lowrance--Wagner recurrence before comparing with $`k`$;
discarding older rows there would make non-adjacent transpositions unsound.

The merge/split string compatibility API still accepts a `MemoCache`, but its
miss path is iterative and stack-safe. The cache stores the completed pair;
the dynamic program does not recursively populate unbounded substring keys.

## Native ABI

Unicode functions retain their original names. Byte and token operations add
the suffixes `_bytes` and `_u64`; `_threshold` is always the final suffix.
Thus `llev_true_damerau_distance_u64_threshold` is the bounded unrestricted
Damerau operation over tokens.

Every input is a pointer plus a logical unit count. `(NULL, 0)` denotes an
empty input. A nonzero token length requires `uint64_t` alignment. Exact calls
return `SIZE_MAX` for invalid input. Thresholded calls return `SIZE_MAX` for
invalid input and `SIZE_MAX-1` when the exact distance exceeds the bound.

## Julia dispatch

All four Julia functions use the same domain mapping:

```julia
distance("café", "cafe")                              # Unicode scalars: 1
distance(UInt8[0xff, 0x00], UInt8[0xff, 0x01])        # bytes: 1
distance(UInt64[10, 20], UInt64[20, 10])              # tokens: 2
merge_and_split_distance("m", "rn"; threshold=1)     # 1
```

Both operands must have the same domain. `Vector{UInt8}` and `Vector{UInt64}`
cross the ABI without copying. Other `AbstractVector` implementations are
materialized into contiguous temporary storage, kept alive with
`GC.@preserve`, and released after the call. Negative thresholds raise
`ArgumentError`; values outside `Csize_t` raise `OverflowError`; a result above
the bound is `nothing`.

## Verification strategy

The Rust and Julia suites generate short sequences containing zero, high byte
values, and the maximum u64 token. Together they compare every native ABI
operation with its generic kernel and check every tested threshold against the
exact result. The Julia suite maps the same generated abstract alphabet into
text, byte, and token representations, then requires cross-domain equality for
all four families and both exact and thresholded calls. Separate negative
controls cover invalid UTF-8, misaligned token pointers, negative and
overflowing thresholds, and mixed or unsupported Julia element types.
