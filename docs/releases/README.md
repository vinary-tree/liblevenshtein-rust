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

- [`4.0.0-rc.1`](4.0.0-rc.1.md) — synchronized version-4 release candidate,
  standalone interop and JavaScript runtime decomposition, scoped npm package
  bootstrap migration, and multi-registry language-binding publication.

