# Verification — Core Theories: Automaton

This directory holds the Rocq (Coq) proof artifacts for the Levenshtein
automaton sub-development: `.v` sources (`Position.v`, `State.v`,
`Transition.v`, `CharVector.v`, `Acceptance.v`, `AntiChain.v`, `Subsumption.v`,
`Soundness.v`, `Completeness.v`, `MainTheorem.v`) and their compiled
`.glob`/`.vo`/`.vok`/`.vos` outputs, plus a `PROOF_STATUS.md` prose summary.
These proof artifacts are machine-checked and are never reworded by hand.

For the authoritative status of every artifact — which files are *trusted* and
support public correctness claims versus debug/legacy/partial — defer to the
verification manifest, which is the single source of truth:

→ [../../../FORMAL_VERIFICATION_MANIFEST.tsv](../../../FORMAL_VERIFICATION_MANIFEST.tsv)

See also [../../../README.md](../../../README.md) and
`../../../README_FORMAL_GATES.md` for the verification workflow and gate-checking
policy.

[← Documentation Index](../../../../README.md)
