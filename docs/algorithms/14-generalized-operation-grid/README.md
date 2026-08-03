# Exact generalized-operation grid

This chapter develops the algorithm behind `GeneralizedAutomaton` in literate
pseudocode. Read the [repair design](../../design/generalized-automaton-repair.md)
first for the public contract and compatibility boundary.

## 1. Inputs and output

The algorithm receives a source string, a target string, an integer budget,
and a finite operation set. Each operation declares source consumption, target
consumption, a non-negative decimal weight, and an optional allowed-pair set.
It returns the least exact scaled cost when that cost is within budget.

The central invariant is:

> `pending[(i, j)]` is the least cost discovered for consuming exactly the
> first `i` source scalars and first `j` target scalars.

## 2. Preparation

UTF-8 byte offsets are computed once so an operation can count scalars while
restriction lookup borrows exact string slices.

```pseudocode
procedure PREPARE(text)
    offsets ← every UTF-8 byte index at which a scalar starts
    append byte_length(text) to offsets
    return offsets
end procedure
```

One cost scale is derived before any graph work:

```pseudocode
procedure PREPARE_COSTS(operations, integer_budget)
    scale ← least common denominator of reduced decimal operation weights
    budget ← checked_multiply(integer_budget, scale.denominator)
    weighted ← empty list

    for each operation in operations do
        step ← scale.to_scaled(operation.weight)
        append (operation, step) to weighted
    end for

    return (scale, budget, weighted)
end procedure
```

Any invalid decimal or overflow returns an error. There is no fallback to a
rounded floating-point cost.

## 3. Sparse topological traversal

Each operation moves right, down, or diagonally in the alignment grid. A
zero-consumption operation is ignored because it cannot advance a finite
alignment. Every other edge moves to a lexicographically later coordinate, so
a sorted map supplies topological order without a separate visited set.

```pseudocode
procedure SCALED_DISTANCE(source, target, operations, integer_budget)
    source_offsets ← PREPARE(source)
    target_offsets ← PREPARE(target)
    (scale, budget, weighted) ← PREPARE_COSTS(operations, integer_budget)

    if MAX_GENERALIZED_ALIGNMENT_STATES equals 0 then
        return resource-limit error with observed = 1
    end if
    pending ← sorted map containing ((0, 0), 0)
    discovered ← 1

    while pending is not empty do
        ((i, j), accumulated) ← remove lexicographically first entry

        if i equals scalar_length(source)
           and j equals scalar_length(target) then
            return accumulated
        end if

        for each (operation, step) in weighted do
            if operation consumes neither side then
                continue
            end if

            next_i ← checked_add(i, operation.consume_source)
            next_j ← checked_add(j, operation.consume_target)
            if either destination exceeds its string length then
                continue
            end if

            source_slice ← source scalars in [i, next_i)
            target_slice ← target scalars in [j, next_j)
            if operation does not apply to both slices then
                continue
            end if

            next_cost ← checked_add(accumulated, step)
            if next_cost exceeds budget then
                continue
            end if

            if pending contains (next_i, next_j) then
                pending[(next_i, next_j)] ← minimum of
                    pending[(next_i, next_j)], next_cost
            else
                next_discovered ← checked_add(discovered, 1)
                if next_discovered exceeds MAX_GENERALIZED_ALIGNMENT_STATES then
                    return resource-limit error before insertion
                end if
                discovered ← next_discovered
                pending[(next_i, next_j)] ← next_cost
            end if
        end for
    end while

    return no in-budget alignment
end procedure
```

## 4. Why the first removal is final

Consider a cell $`(i,j)`$. Every incoming edge starts at $`(p,q)`$ with either
$`p<i`$, or $`p=i`$ and $`q<j`$. Thus every predecessor is removed before
$`(i,j)`$. All its candidate costs have already been merged by minimum when the
cell is removed. This is dynamic programming in sparse graph form, not
Dijkstra's algorithm; edge non-negativity supports budget pruning, while
topological order supplies finality.

## 5. Worked fractional example

Let match cost zero, substitution cost `0.15`, and integer budget `1`. The
derived denominator is 20, each substitution costs 3, and the budget is 20.

| Mismatches | Scaled cost | Real cost | Accepted? |
|---:|---:|---:|---|
| 0 | 0 | 0.00 | yes |
| 6 | 18 | 0.90 | yes |
| 7 | 21 | 1.05 | no |

The comparison is integer-exact. No epsilon or floating-point accumulation is
involved.

## 6. Derived preset invariants

The same traversal becomes several familiar algorithms by changing only the
operation set:

- Hamming: every edge consumes one scalar from each side, so accepted strings
  have equal length.
- Indel: substitution is absent; the least cost equals
  $`\lvert x\rvert+\lvert y\rvert-2\operatorname{LCS}(x,y)`$.
- Bounded skip: match and source deletion only; the target must be a
  subsequence of the source.
- Standard Levenshtein: match, substitution, insertion, and deletion reproduce
  the ordinary dynamic-programming distance.

These statements are executable differential properties, not examples alone.

## 7. Failure handling

Fallible APIs distinguish invalid cost domains, checked arithmetic failure,
and resource exhaustion. The Boolean compatibility method maps all such
failures to rejection. This fail-closed policy is appropriate for matching;
diagnostic and configuration code should call `try_accepts` or
`scaled_distance` so it can report the cause.

The discovery count measures unique materialized cells, not processed cells.
It is checked before a vacant map entry is inserted, so an adversarial frontier
cannot allocate beyond the ceiling while waiting to be popped.
