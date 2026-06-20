# Verification — Core Theories

This directory holds the Rocq (Coq) proof artifacts for the core edit-distance
correctness development: top-level theory sources (`Distance.v`,
`MainTheorems.v`, `TraceLowerBound.v`) and the modular subdirectories
`Automaton/`, `Cardinality/`, `Composition/`, `Core/`, `DPMatrix/`,
`LowerBound/`, `OptimalTrace/`, `Trace/`, and `Triangle/`, together with the
build files (`_CoqProject`, `Makefile*`) and a `DECOMPOSITION_SUMMARY.md`
describing the modular split. These proof artifacts are machine-checked and are
never reworded by hand.

For the authoritative status of every artifact — which files are *trusted* and
support public correctness claims versus debug/legacy/partial — defer to the
verification manifest, which is the single source of truth:

→ [../../FORMAL_VERIFICATION_MANIFEST.tsv](../../FORMAL_VERIFICATION_MANIFEST.tsv)

See also [../../README.md](../../README.md) and `../../README_FORMAL_GATES.md`
for the verification workflow and gate-checking policy.

[← Documentation Index](../../../README.md)
