# Vinary Tree Go bindings

The Go package is an idiomatic streaming wrapper around liblevenshtein's stable
C ABI. It supports Go 1.25 and newer. `Iterator.Next` owns only the current Go
result while the iterator leases one bounded native batch; it never collects an
entire query into a slice. `Close` is deterministic and finalizers are a safety
net.

Dictionary producers implement the small `interop.DictionaryResource`
interface from
`github.com/vinary-tree/liblevenshtein-rust/vinary-tree-interop/bindings/go`.
Native Vinary
Tree producers lend two pointer-sized words for the constructor and are retained
in O(1) by the transducer.

During source-tree development, set `CGO_LDFLAGS=-L../../target/debug` and put
that directory on the platform loader path. Published archives place the shared
library in the platform package or may use the separately installed CMake
package. C/C++ consumers may select static or shared linkage.
