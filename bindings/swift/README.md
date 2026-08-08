# liblevenshtein Swift bindings

The SwiftPM product `Liblevenshtein` consumes any `DictionaryResource` from
`VinaryTreeInterop`; concrete dictionaries remain in libdictenstein's Swift
package. `QueryCursor` is a `Sequence` and `IteratorProtocol`, advances through
leased native batches, and copies only the current batch into Swift ownership.
The native CMake package may be linked dynamically or statically by the parent
application; the Swift system-library target uses the installed shared library
by default.
