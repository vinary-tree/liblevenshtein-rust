# Substitution policy implementation status

**Current through:** 2026-09-02

**Status:** Complete in the native transition and universal-automaton engines

**Scientific evidence:** [Universal policy-aware characteristic-vector encoding](../scientific-ledger/universal-policy-aware-encoding-2026-09-02.md)

## Semantics

A substitution policy declares directional zero-cost equivalences. The first
argument is a dictionary/source unit and the second is a query/target unit. For
dictionary unit $`d`$, query unit $`q`$, and policy $`P`$, a transition consumes
the pair without substitution cost when:

```math
d = q \lor P(d, q).
```

Policies are not implicitly symmetric. Add both ordered pairs when both
directions are intended. Insertions, deletions, transpositions, merge/split
operations, and other configured edits retain their ordinary costs.

## Public policy types

| Policy | Unit domain | Ownership | Distinct zero-cost pairs |
|---|---|---|---|
| `Unrestricted` | every `CharUnit` | zero-sized value | none; Standard Levenshtein behavior |
| `Restricted<'a>` | `u8` | borrowed `SubstitutionSet` | configured byte pairs |
| `OwnedRestricted` | `u8` | `Arc<SubstitutionSet>` | configured byte pairs |
| `RestrictedChar<'a>` | `char` | borrowed `SubstitutionSetChar` | configured Unicode-scalar pairs |
| `OwnedRestrictedChar` | `char` | `Arc<SubstitutionSetChar>` | configured Unicode-scalar pairs |
| customer implementation | any supported `U: CharUnit` | implementation-defined | `SubstitutionPolicyFor<U>` result |

`SubstitutionPolicyFor<U>` is the authoritative unit-generic interface.
`SubstitutionPolicyChar` remains a compatibility convenience and now accepts
owned, cloneable policies as well as copyable borrowed policies.

## Engine coverage

The policy participates in all dictionary transition families that accept a
policy parameter, including Standard, adjacent transposition, merge/split, and
true Damerau surfaces where applicable. It is also applied by:

- ordinary, ordered, prefix, priority, ranked, value-filtered, and weighted
  query iterators;
- packed and cached characteristic-vector implementations;
- batch and stable-online `UniversalAutomaton` execution;
- byte, Unicode-scalar, `u64`, and external `CharUnit` universal inputs when
  the policy implements the corresponding `SubstitutionPolicyFor<U>` trait.

The universal API provides `accepts`/`online` for Unicode strings,
`accepts_bytes`/`online_bytes` for raw bytes, and
`accepts_units`/`online_units` for arbitrary unit slices. Policy application is
centralized in characteristic-vector construction so every universal position
variant observes identical equivalence semantics.

## Zero-cost default specialization

`Unrestricted` is a zero-sized type and sets
`SubstitutionPolicy::MAY_MATCH_DISTINCT_UNITS` to `false`. Monomorphized
transition engines select an exact-comparison-only loop at compile time. Custom
policies conservatively default the capability to `true`, preserving semantics
for external implementations without requiring another method.

The 2026-09-02 pinned-core experiment measured the explicit unrestricted
constructor slightly faster than the default constructor; both instantiate the
same `UniversalAutomaton<Standard, Unrestricted>` code. A test policy whose
lookup panics also proves that an exact-only capability never invokes the
lookup. See the linked scientific ledger for protocol, confidence intervals,
hardware, and interpretation.

## Verification contract

Policy changes must retain all of the following evidence:

1. Directional positive and negative examples for byte and Unicode policies.
2. Exact matches under policies that reject every distinct pair.
3. Raw non-UTF-8 byte coverage.
4. Borrowed and owned lifetime coverage.
5. Stable-online and batch agreement.
6. Standard, Transposition, and Merge-and-Split universal coverage.
7. Random comparison with a directional dynamic-programming oracle.
8. Differential comparison with singleton dictionary transducers.
9. Padding isolation under a policy that accepts every concrete pair.
10. Pinned-core timing and exact-only specialization evidence.

Foreign-language exposure is governed separately by the generated family
completeness matrix. A native policy implementation does not, by itself, prove
that a language binding exposes the corresponding constructor, lifecycle, and
unit-domain surface.
