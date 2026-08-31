# Lowrance–Wagner 1975: engineering summary

## Bibliographic record

R. Lowrance and R. A. Wagner, “An extension of the string-to-string correction
problem,” *Journal of the ACM* 22(2), 177–183 (1975).
[DOI 10.1145/321879.321880](https://doi.org/10.1145/321879.321880).

## Question

The classical recurrence aligns symbols once. The paper asks for the minimum
cost when transpositions compose with later edits. Its decisive contribution
is a last-occurrence recurrence that remembers the earlier matching endpoints
of a transposition macro.

## Recurrence term used here

If $`k'`$ and $`l`$ are the previous matching endpoints, the additional
candidate is:

```math
D[k'-1,l-1]+(i-k'-1)+(j-l-1)+1.
```

The three added quantities are source-interior deletions, target-interior
insertions, and one endpoint transposition. Under budget $`k`$, their sum
implies the joint lookback bound $`(i-k')+(j-l)\le k+1`$.

## Translation into the crate

The reference DP retains the paper's last-occurrence table. The dictionary
automaton refines one recurrence edge into a streaming chain:

1. enter and prepay the query interior plus transposition;
2. extend once per target-interior symbol;
3. resolve when the opposite endpoint matches.

The equality $`\delta+b=(\delta-1)+b+1`$ proves the chain has exactly the
paper recurrence's local cost. See the [literate algorithm chapter](../../algorithms/11-true-damerau/README.md)
and [formal tree](../../verification/damerau/).

## What the paper does not supply

The paper is a pairwise dynamic program, not a trie-state canonicalization or
resource policy. This implementation additionally requires kind-aware
subsumption, a finite payload representation, an explicit maximum representable
budget, and empirical state-size evidence. Those are library engineering
obligations rather than claims attributed to the paper.
