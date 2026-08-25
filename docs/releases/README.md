# Release evidence ledgers

This directory preserves the factual execution record for each coordinated
Vinary Tree release train. The living policy and reusable command syntax live
in [Releasing the Vinary Tree language bindings](../releasing-language-bindings.md);
these ledgers record what actually happened: source commits, validation runs,
registry coordinates, digests, public-byte smoke tests, incidents, and recovery
decisions.

Release ledgers are historical evidence. Append corrections and later results;
do not silently rewrite an earlier failure into a success. Registry acceptance
is recorded only after the exact public coordinate resolves and a clean
consumer exercises it.

## Ledgers

- [`4.0.0-rc.4`](4.0.0-rc.4.md) — current corrective release train. It makes
  native JavaScript/JVM prerequisites explicit, brings Cargo lockfiles under
  release synchronization, and rehearses both repairs from a build-clean family.
- [`4.0.0-rc.3`](4.0.0-rc.3.md) — rejected corrective candidate. Its verified
  interop and libdictenstein public subset is retained, but two clean-build
  package jobs failed and the remaining publication stopped.
- [`4.0.0-rc.2`](4.0.0-rc.2.md) — rejected corrective candidate. Its valid
  public interop/runtime subset and installed-byte proofs are retained, but an
  invalid libdictenstein Cabal descriptor prevents promotion of the train.
- [`4.0.0-rc.1`](4.0.0-rc.1.md) — synchronized version-4 release candidate,
  standalone interop and JavaScript runtime decomposition, scoped npm package
  bootstrap migration, and multi-registry language-binding publication. This
  ledger is historical: one published facade was rejected during installed-byte
  verification, so unfinished coordinates moved to RC.2 rather than rewriting
  RC.1.
