# Exact finite-lookback generalized-operation automaton

This chapter develops the production algorithm behind
`GeneralizedOnlineAutomaton` in literate pseudocode. Read the
[repair design](../../design/generalized-automaton-repair.md) first for its
public contract, mathematical recurrence, and compatibility boundary.

## 1. Inputs, output, and online contract

The machine receives one fixed source string, an integer budget, and a finite
operation set. Each operation declares source consumption, target consumption,
a non-negative decimal weight, and an optional allowed-pair set. After
construction, `advance` consumes one Unicode target scalar and returns the
least exact scaled cost for the newly committed target prefix when that cost
is within budget.

Let $`D_j[i]`$ be the least in-budget cost after consuming the first $`i`$
source scalars and first $`j`$ target scalars. The central invariant is:

> A row-ring slot tagged generation $`j`$ contains exactly $`D_j`$; an
> untagged or differently tagged slot is never read as that generation.

## 2. Preparation

UTF-8 byte offsets for the fixed source are computed once so an operation can
count Unicode scalars while restriction lookup borrows exact string slices.
One exact cost scale is also derived before target work begins.

```text
PROCEDURE PREPARE(automaton, source, limits)
  validate every operation; reject a zero/zero consuming operation
  sourceOffsets <- UTF-8 scalar boundaries of source plus byte length
  scale <- least common denominator of reduced decimal operation weights
  budget <- checkedMultiply(integerBudget, scale.denominator)
  compiled <- each operation paired with scale.toScaled(operation.weight)

  r <- maximum target consumption among compiled operations
  width <- scalarLength(source) + 1, using checked addition
  retainedCells <- checkedMultiply(r + 2, width)
  stepWork <- checkedMultiply(width, numberOf(compiled))
  reject before allocation if either configured ceiling is exceeded

  rows <- r + 1 rows of TOP, each with width cells
  tags <- r + 1 absent generation tags
  scratch <- one width-cell row of TOP
  targetWindow <- capacity for at most r scalars

  COMPUTE-GENERATION(0, origin = true)
  COMMIT(0)
END PROCEDURE
```

`try_reserve_exact` precedes each allocation. Invalid decimal costs,
arithmetic overflow, allocation failure, or a resource ceiling returns a
tagged `GeneralizedAutomatonError`; none is converted into an empty successful
alignment.

## 3. Why finite lookback is complete

Let operation $`t`$ consume $`t^x`$ source scalars and $`t^y`$ target
scalars. Its predecessor for cell $`(i,j)`$ is
$`(i-t^x,j-t^y)`$. If $`r=\max_t t^y`$, every target-consuming predecessor
of generation $`j`$ lies in generations $`j-r`$ through $`j-1`$.

An operation with $`t^y=0`$ reads the current scratch row at a smaller source
coordinate. Because source coordinates are processed in increasing order,
that predecessor is already final. Thus $`r+1`$ committed rows plus one
scratch row form a complete topological schedule of the alignment graph.

## 4. Literate online transition

**Purpose.** Commit the exact next target-prefix row without retaining any
older target history than an operation can inspect.

**Invariant.** Before relaxing source coordinate $`i`$, every current-row
predecessor at a smaller coordinate and every tagged earlier-row predecessor
already equals its exact least in-budget cost.

```text
PROCEDURE ADVANCE(targetScalar)
  generation <- checkedAdd(committedGeneration, 1)

  prospectiveWindow <- suffix(targetWindow + targetScalar, maximumLength = r)
  rebuild UTF-8 text and scalar offsets for prospectiveWindow
  fill scratch with TOP

  for sourceEnd from 0 through sourceLength do
    best <- scratch[sourceEnd]

    for each (operation, stepCost) in compiled do
      sx <- operation.sourceConsumption
      ty <- operation.targetConsumption
      skip if sx > sourceEnd or ty > generation

      predecessorSource <- sourceEnd - sx
      if ty = 0 then
        predecessor <- scratch[predecessorSource]
      else
        predecessorGeneration <- generation - ty
        slot <- predecessorGeneration modulo numberOf(rows)
        skip unless tags[slot] = predecessorGeneration
        predecessor <- rows[slot][predecessorSource]
      end if
      skip if predecessor is TOP

      sourceSlice <- fixed source scalars [sourceEnd - sx, sourceEnd)
      targetSlice <- prospective target suffix of length ty
      skip unless operation applies to these exact slices

      candidate <- checkedAdd(predecessor, stepCost)
      if candidate <= budget then best <- minimum(best, candidate)
    end for

    scratch[sourceEnd] <- best
  end for

  COMMIT(generation):
    swap scratch with rows[generation modulo numberOf(rows)]
    tag that slot with generation
    replace targetWindow with prospectiveWindow
    publish final-cell distance and active-position count
END PROCEDURE
```

Prospective window construction and row computation occur before commit. If a
checked operation fails, the old row tags, committed generation, target window,
and public observation remain unchanged.

## 5. Batch evaluation is an online fold

`GeneralizedAutomaton::scaled_distance(source, target)` constructs the online
machine, feeds `target.chars()` one at a time, and reads the final observation.
It does not materialize a target-sized DP matrix or sparse map.

The former sorted-map evaluator remains under `cfg(test)` as an independent
oracle. It uses a different storage shape and is exhaustively compared to the
row ring for all source/target strings of length at most three over
`{ "a", "b", "é" }` across Standard, transposition, and merge/split operation
sets.

## 6. Worked fractional example

Let match cost zero, substitution cost `0.15`, and integer budget `1`. The
derived denominator is 20, each substitution costs 3, and the budget is 20.

| Mismatches | Scaled cost | Real cost | Accepted? |
|---:|---:|---:|---|
| 0 | 0 | 0.00 | yes |
| 6 | 18 | 0.90 | yes |
| 7 | 21 | 1.05 | no |

The comparison is integer-exact. No epsilon or floating-point accumulation is
involved.

## 7. Complexity and stable retention

For fixed source length $`m`$, consumed target length $`n`$, maximum target
consumption $`r`$, and operation count $`|\mathcal{O}|`$, the batch fold
takes:

```math
\mathcal{O}\left(n(m+1)|\mathcal{O}|\right)
```

operation relaxations. Retained DP memory is:

```math
\mathcal{O}\left((r+2)(m+1)\right),
```

plus the fixed source, compiled operations, and $`r`$ target scalars. It is
independent of $`n`$. A 100,000-scalar unit test pins the exact retained-cell
count while the consumed generation grows.

## 8. Derived preset invariants

The same automaton becomes familiar algorithms by changing the operation set:

- Hamming: every edge consumes one scalar from each side, so accepted strings
  have equal length.
- Indel: substitution is absent; the least cost equals
  $`\lvert x\rvert+\lvert y\rvert-2\operatorname{LCS}(x,y)`$.
- Bounded skip: match and source deletion only; the target must be a
  subsequence of the source.
- Standard Levenshtein: match, substitution, insertion, and deletion reproduce
  ordinary dynamic-programming distance.

These statements are formal and executable differential properties, not
examples alone. An arbitrary custom operation set is an alignment cost model;
it is not automatically a metric.

## 9. References

- P. Mitankin, S. Mihov, and K. U. Schulz, “Deciding word neighborhood with
  universal neighborhood automata,” *Theoretical Computer Science* 412(22),
  2011. [doi:10.1016/j.tcs.2011.01.013](https://doi.org/10.1016/j.tcs.2011.01.013)
- R. A. Wagner and M. J. Fischer, “The string-to-string correction problem,”
  *Journal of the ACM* 21(1), 1974.
  [doi:10.1145/321796.321811](https://doi.org/10.1145/321796.321811)
