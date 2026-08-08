// Package interop defines allocation-free retained-resource handoff between
// independently packaged Vinary Tree Go bindings.
package interop

// DictionaryResource lends the two native words of a VtResource to a
// synchronous callback while keeping its owning handle alive. A native
// consumer may retain the resource through its vtable before the callback
// returns.
type DictionaryResource interface {
	WithResource(func(context, vtable uintptr) error) error
}
