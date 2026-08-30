# Liblevenshtein.jl

`Liblevenshtein` provides Unicode edit-distance functions and lazy,
snapshot-consistent fuzzy search over any negotiated Vinary Tree dictionary
provider. Its Julia API preserves byte, Unicode-scalar, and unsigned-token
domains through multiple dispatch.

## Common usage

```julia
using Liblevenshtein

distance("kitten", "sitting")
distance("kitten", "sitting"; threshold=2)
damerau_distance("ab", "ba")
```

## Resource-backed search

A `Transducer` accepts a `VinaryTreeInterop.Resource` or
`VinaryTreeInterop.Dictionary`. Reuse it across queries and close it
deterministically:

```julia
transducer = Transducer(dictionary, ALGORITHM_STANDARD)
try
    for match in query(transducer, "speling", 2)
        println(match.term, " ", match.distance)
    end
finally
    close(transducer)
end
```

Use `reduce_batches!` when matches do not need to escape the callback. Each
`BorrowedMatch` expires when its callback returns; call `materialize` inside
the callback when an independently owned value is required.

## Automata

`Transducer` accepts `ALGORITHM_STANDARD`, `ALGORITHM_TRANSPOSITION`,
`ALGORITHM_MERGE_AND_SPLIT`, or `ALGORITHM_DAMERAU_LEVENSHTEIN`. Standard,
merge-and-split, and unrestricted Damerau-Levenshtein are metrics. The
optimal-string-alignment transposition variant is intentionally non-metric and
does not compose repeated edits through the same substring.

## API

```@autodocs
Modules = [Liblevenshtein]
Private = false
```
