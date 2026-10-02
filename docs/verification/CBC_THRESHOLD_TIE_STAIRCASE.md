# Threshold-indexed tie floors for ordered metric search

**Status:** representation design for `cbc-04-summary`. It is not an
implemented automaton optimization. The exact-natural statements below
assume a complete captured region and sound per-original or per-subregion
lower bounds. Binary64 endpoint production and Rust traversal correspondence
remain separate obligations.

## What the summary must answer

Let $`U`$ be the finite set of original occurrences owned by one captured
region. Each original $`x`$ has authoritative exact cost $`d(x)`$, injective
tie key $`t(x)`$, and a checked lower bound $`b(x)\le d(x)`$. For an inclusive
threshold $`c`$, define

```math
S_U(c)=\{x\in U:b(x)\le c\},\qquad
T_U(c)=\min\{t(x):x\in S_U(c)\}
```

with $`T_U(c)`$ undefined when $`S_U(c)`$ is empty. The summary has three
semantic results:

| Result | Required evidence | Meaning |
|---|---|---|
| `Unknown` | Coverage, bound soundness, or current query scope is unavailable | No rank improvement is certified |
| `Empty` | Complete coverage proves $`S_U(c)=\varnothing`$ | Every original costs strictly more than $`c`$ |
| `Known(t)` | Complete coverage and a proved lower tie floor $`t\le t(x)`$ for every $`x\in S_U(c)`$ | Any original whose cost equals $`c`$ has tie at least $`t`$ |

If a separate certificate says every member costs at least $`L`$, then
`Empty` at $`c=L`$ gives a strict cost cut. `Known(t)` at $`c=L`$ gives a
conditioned pair $`(L,t)`$. The summary itself does not supply the cost floor
$`L`$. A full verified heap may use either certificate for lexicographic
pruning only through the strict BF-2 comparison; an underfull heap cannot
discard a still-eligible original to fill a vacant result slot.

**Coverage proof.** If $`d(x)=L`$ and $`b(x)\le d(x)`$, then $`b(x)\le L`$,
so $`x\in S_U(L)`$. If the set is empty, no equality-cost member exists.
If it is nonempty, its minimum tie is no greater than the tie of every
equality-cost member. This proof requires the complete set of original
occurrences, including multiple slots in one compressed bucket. A summary
over one representative per bucket can be unsound.

## Finite staircase representation

For a complete list of $`n`$ original entries $`(b(x),t(x),id(x))`$, sort
by nondecreasing bound and group equal bounds. After each group with bound
$`c_i`$, store the prefix minimum tie $`m_i`$. The retained staircase is
$`[(c_1,m_1),\ldots,(c_h,m_h)]`$ for distinct bound values, where
$`h\le n`$. A lookup takes the last entry whose bound is at most the
requested threshold. Before the first entry, it returns `Empty`; at or
after an entry, it returns `Known(m_i)`. It returns `Unknown` whenever the
complete-enumeration or scope certificate is absent, even if a partial
inspection found no entry.

```text
build(complete originals, scoped sound bounds):
    entries := one (bound, tie, identity) per original occurrence
    sort entries by bound, retaining every colliding original
    running_min := None
    for each equal-bound group in ascending bound order:
        running_min := min(running_min, every tie in group)
        append (group.bound, running_min) to staircase
    return (scope, coverage proof, staircase)

lookup(summary, requested scope, cutoff):
    if scope is incompatible or coverage is not proved: return Unknown
    i := last staircase index with bound <= cutoff
    if no such index exists: return Empty
    return Known(staircase[i].minimum_tie)
```

The invariant after group $`i`$ is
$`m_i=\min\{t(x):b(x)\le c_i\}`$. It follows by induction: the first group
establishes its own minimum, and each next group takes the minimum of the
previous prefix and every new tie. The lookup picks exactly the prefix whose
bounds are at most $`c`$. Therefore `Known(m_i)` is the exact minimum tie
of the **bound-eligible** set. It may be lower than the true equality-slice
minimum because the bound-eligible set includes false positives; this loses
pruning power but preserves correctness.

If entries are unsorted, comparison sorting takes $`O(n\log n)`$ time and
$`O(n)`$ temporary storage in a conventional sorting implementation. The
prefix pass takes $`O(n)`$ time and retains at most $`n`$ pairs plus scope
and coverage metadata. Binary-search lookup takes $`O(\log h)`$ comparisons.
If a source producer already emits verified entries ordered by bound, the
sorting charge can be omitted only with a proof of that order. Each query or
parameter change that alters $`b`$ may require a rebuild; a stored staircase
cannot silently cross snapshot, query, arithmetic, or bound-function scope.

For an illustrative set $`(b,t)=(2,9),(2,4),(5,7)`$, the staircase is
$`(2,4),(5,4)`$. The two originals at bound two must both be included. At
cutoff one the result is `Empty`; at two it is `Known(4)`. Static minimum tie
over all three is also four and gives no advantage in this example; a region
with an early tie only at a large bound can make the staircase stronger at
small cutoffs.

## Compositional and lazy variants

Building the per-original staircase by evaluating a query-dependent bound
for every original may cost as much as the search it was meant to accelerate.
A compositional alternative uses a disjoint, complete partition of terminals,
private work, and child regions. An entry may represent a whole subregion
only when its bound applies to **every** original in that subregion and its
stored tie is a lower bound for **every** represented tie. The union summary
takes the minimum tie of all bound-eligible pieces. A piece with an unknown
bound or incomplete membership keeps the union `Unknown` where it could
contain a better equality candidate. Empty children contribute nothing, but
an early-tie live terminal remains a separate piece.

A query-independent subtree minimum tie is cheap to retain and can be paired
with a query-specific certified subregion cost bound. This is safe but often
weaker than per-original entries. Lazy construction can defer summary work
until a full heap and equality-bound prune opportunity exist; its cost must
include the construction, cache lookup, invalidation, and retained metadata.
A changed bound function can reuse an old staircase only after a checked
eligible-set inclusion in the direction needed for the floor; decreasing a
threshold with fixed bounds shrinks the eligible set, while increasing it
can introduce an earlier tie and invalidate an old floor.

## Space and time decision

The staircase is attractive only if its build and retained-space costs are
recovered by avoided queue work and exact verifications on the declared
workload. Charge sorting scratch, grouped-entry storage, scope keys, cache
maintenance, and repeated construction after revision changes. Compare it
with a static query-independent subtree minimum and with no tie summary.
The [acquisition cost model](CBC_EVIDENCE_ACQUISITION_COST.md) defines the
work coordinates; the [performance adoption gate](CBC_PERFORMANCE_ADOPTION_GATE.md)
requires paired end-to-end time and peak-space nonregression before any
production use.

A summary costing twenty primitive units to avoid five unit-cost exact
verifications is a negative control: it improves the number of verifications
but worsens total work. The representation proof establishes a sound floor,
not a runtime speedup or a minimum-memory implementation.
