# vinary-tree-interop

`vinary-tree-interop` defines the small, stable C ABI used to exchange live
resources between independently packaged vinary-tree libraries and language
bindings. It contains no algorithms and does not depend on any vinary-tree
project.

The ABI deliberately consists of a two-word resource (`context`, `vtable`),
reference counting, versioned interface discovery, and batched provider
interfaces. Native object handoff is O(1); it neither serializes the object nor
depends on Rust's dynamic-library ABI.

The first interface is an immutable dictionary snapshot provider. Mutable
dictionary resources capture an O(1), structurally shared snapshot at the start
of every consuming operation. Node edges cross the ABI in batches instead of
one callback per edge.

