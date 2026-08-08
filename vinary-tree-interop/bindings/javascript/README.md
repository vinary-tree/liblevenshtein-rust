# @vinary-tree/interop

Shared TypeScript and runtime guards for retained resources passed between
modular Vinary Tree packages. The package does not create a second resource
table: resources are created by the selected `@vinary-tree/vinary-tree`
native, WASM, or WASI runtime and carry that runtime's identity.

Project facades use `assertSameRuntime` before accepting a resource and use
`assertDictionaryResource` to require `vt.dictionary.v1`. Concrete dictionary
constructors and CRUD remain in `@vinary-tree/libdictenstein`.
