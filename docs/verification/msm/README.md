# Verification — MSM

This directory holds Rocq (Coq) proof artifacts (`.v` sources and their compiled
`.glob`/`.vo`/`.vok`/`.vos` outputs) together with the build files
(`_CoqProject`, `Makefile`, `Makefile.coq*`) for the MSM (Move-Split-Merge)
interval-relaxed trie-search verification, plus the `INTERVAL_MSM.md` design note
and a `theories/` subdirectory. These proof artifacts are machine-checked and are
never reworded by hand.

For the authoritative status of every artifact — which files are *trusted* and
support public correctness claims versus debug/legacy/partial — defer to the
verification manifest, which is the single source of truth:

→ [../FORMAL_VERIFICATION_MANIFEST.tsv](../FORMAL_VERIFICATION_MANIFEST.tsv)

See also [../README.md](../README.md) and `../README_FORMAL_GATES.md` for the
verification workflow and gate-checking policy.

[← Documentation Index](../../README.md)
