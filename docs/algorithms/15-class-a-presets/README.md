# Literate Class-A preset algorithms

This chapter presents the executable reference algorithms for Hamming,
insertion/deletion, and bounded-skip distance. The corresponding
[`OperationSet` design](../../design/class-a-presets.md) defines their public
semantics and validation boundary.

## 1. Hamming: lock-step comparison

Hamming admits only edges that consume one source and one target scalar. A
single pass can therefore count mismatches, but it must detect unequal lengths
instead of truncating `zip` to the shorter input.

```pseudocode
procedure HAMMING(source, target)
    mismatches ← 0
    loop
        left ← next scalar from source
        right ← next scalar from target
        if left and right are both absent then return mismatches
        if exactly one is absent then return undefined
        if left differs from right then mismatches ← mismatches + 1
    end loop
end procedure
```

The loop invariant is: after $`i`$ iterations, both prefixes have length
$`i`$ and `mismatches` equals their mismatch count. This invariant yields
identity and symmetry directly. Coordinate-wise Boolean triangle inequality
gives the metric triangle law on every fixed-length space.

## 2. Indel: two-row dynamic programming

Let $`D[i,j]`$ be the minimum insertion/deletion cost between prefixes of
length $`i`$ and $`j`$. There is no substitution edge:

```math
D[i,j]=\min\left(
D[i-1,j]+1,
D[i,j-1]+1,
\begin{cases}D[i-1,j-1],&x_i=y_j\\\top,&x_i\ne y_j.\end{cases}
\right).
```

Only the preceding row is needed:

```pseudocode
procedure INDEL(source, target)
    columns ← the shorter input
    rows ← the longer input
    previous[j] ← j for every column j
    for each row scalar x_i do
        current[0] ← i
        for each column scalar y_j do
            current[j] ← minimum(previous[j] + 1, current[j-1] + 1)
            if x_i equals y_j then
                current[j] ← minimum(current[j], previous[j-1])
            end if
        end for
        swap(previous, current)
    end for
    return previous[length(columns)]
end procedure
```

Reversing an edit script swaps insertions and deletions without changing cost,
which proves symmetry. Concatenating scripts adds their costs, which proves the
triangle inequality. Counting length change in a script gives
$`||x|-|y||\le d_I(x,y)`$; counting insert/delete parity gives the parity
invariant used by property tests.

## 3. Thresholded indel band

An affordable path of cost at most $`k`$ cannot visit a cell with
$`|i-j|>k`$. The bounded algorithm retains at most $`2k+1`$ diagonals and
uses $`k+1`$ as an unreachable cap.

```pseudocode
procedure BOUNDED_INDEL(source, target, k)
    if either side is empty then
        return its opposite length exactly when that length is at most k
    end if
    if absolute length difference exceeds k then return undefined

    initialize row zero only through column k
    for row i from 1 through length(source) do
        start ← maximum(1, i-k)
        finish ← minimum(length(target), i+k)
        reset current row to unreachable
        evaluate recurrence only from start through finish
        if every retained cell exceeds k then return undefined
        swap rows
    end for
    return terminal value exactly when it is at most k
end procedure
```

The explicit empty-side branch is essential. Without it, the band has no
interior cell to process and can incorrectly reject an affordable deletion-only
or insertion-only path. The minimized generated regression is `"a"` versus
`""` at budget 1.

## 4. Bounded skip: a subsequence scan

The match/delete operation set says that every target scalar must consume the
same source scalar, while extra source scalars may be deleted:

```pseudocode
procedure BOUNDED_SKIP_REFERENCE(source, target)
    expected ← first target scalar
    for each source scalar do
        if source scalar equals expected then expected ← next target scalar
    end for
    if a target scalar remains then return undefined
    return length(source) - length(target)
end procedure
```

The procedure is directional. Its loop invariant states that the consumed
target prefix is the longest prefix embedded in the consumed source prefix.

## 5. Three-way conformance harness

For a generated pair, each preset is compared with both an explicitly built
operation list and a structurally independent reference:

```pseudocode
for each generated (source, target) do
    reference ← direct algorithm(source, target)
    preset ← generalized_grid(source, target, built_in_preset)
    explicit ← generalized_grid(source, target, manually_listed_operations)
    assert preset equals explicit
    assert explicit equals reference
end for
```

This triangulation catches mistakes in the convenience constructor, the
generalized grid, and the direct reference without treating any one
implementation as infallible. The generated suite also checks metric laws,
budget equivalence, inter-metric ordering, Unicode-scalar behavior, and
validation guards. An exhaustive Birkbeck pass repeats the three reference
comparisons over 42,395 real corrected pairs.

## 6. Operation-set validation

```pseudocode
procedure VALIDATE(operations)
    total ← 0
    for each indexed operation t do
        reject unless weight(t) is finite and non-negative
        reject if source_consumption(t) + target_consumption(t) equals zero
        reject if weight(t) equals zero and the two consumptions differ
        total ← checked_add(total,
                            checked_add(source_consumption(t),
                                        target_consumption(t)))
        reject if total exceeds 4096
    end for
    accept
end procedure
```

The prefix invariant is $`0\le\text{total}\le4096`$ after every accepted
operation. Checked addition prevents wraparound from converting a huge rule
into a cheap-looking one. Validation precedes generalized traversal and is
also repeated at fallible evaluation boundaries.
