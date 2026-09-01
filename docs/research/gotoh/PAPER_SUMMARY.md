# Gotoh 1982: engineering summary

## Bibliographic record

O. Gotoh, “An improved algorithm for matching biological sequences,” *Journal
of Molecular Biology* 162(3), 705–708 (1982).
[DOI 10.1016/0022-2836(82)90398-9](https://doi.org/10.1016/0022-2836(82)90398-9).

## Contribution

A naive affine-gap alignment recurrence may inspect every possible preceding
gap length at every cell. Gotoh separates alignments by their final operation,
allowing the gap continuation minimum to be updated in constant time per cell.
The resulting pairwise dynamic program is $`\mathcal{O}(mn)`$.

## Translation into this crate

The three recurrence matrices become three `PositionKind` layers. Query gaps
are epsilon transitions; dictionary gaps consume trie edges; diagonals reset to
`Normal`. The direct three-matrix implementation remains beside the optimized
automaton as an independent oracle.

The paper supplies the recurrence, not a trie canonicalization theorem. B-1
through B-5, the fused skip-and-consume realization, exact scaling, resource
bounds, backend genericity, and the closure counterexample that motivated the
fusion are library-specific engineering obligations.

## Cost-convention note

This crate defines a length-$`r`$ run as $`g_o+r g_e`$. Readers comparing
reported parameters must check whether another source charges the first symbol
inside the opening term. The recurrence is equivalent after parameter
translation; the public API intentionally does not guess a convention.
