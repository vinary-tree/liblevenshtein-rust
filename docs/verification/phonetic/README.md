# Verification — Phonetic

This directory holds Rocq (Coq) proof artifacts (`.v` sources such as
`position_skipping_proof.v`, `rewrite_rules.v`, `zompist_rules.v`, and the
`test_*.v` debug files, together with their compiled `.glob`/`.vo`/`.vok`/`.vos`
outputs) and build files (`_CoqProject`, `Makefile*`) for the phonetic
position-skipping verification, alongside numerous prose status/strategy notes
and a `theories/` subdirectory. The proof artifacts are machine-checked and are
never reworded by hand.

For the authoritative status of every artifact — which files are *trusted* and
support public correctness claims versus debug/legacy/partial, and which lemmas
remain admitted — defer to the verification manifest, which is the single source
of truth:

→ [../FORMAL_VERIFICATION_MANIFEST.tsv](../FORMAL_VERIFICATION_MANIFEST.tsv)

See also [../README.md](../README.md) and `../README_FORMAL_GATES.md` for the
verification workflow and gate-checking policy.

[← Documentation Index](../../README.md)
