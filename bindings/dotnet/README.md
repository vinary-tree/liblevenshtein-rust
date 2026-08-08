# Vinary Tree .NET bindings

`VinaryTree.Liblevenshtein` is an idiomatic, streaming .NET binding for the
stable native ABI. `VinaryTree.Interop` contains the shared two-word retained
resource contract used by independently packaged dictionary producers.

The package targets .NET 8 (the oldest supported LTS) and uses the latest C#
language standard. Query enumerators retain the query-start dictionary revision
and lease only one native result batch at a time. Dispose transducers and
enumerators deterministically; `SafeHandle` supplies the leak-safe fallback.

Build and test with `dotnet run --project tests/VinaryTree.Liblevenshtein.Tests`.
NuGet packaging is produced with `dotnet pack -c Release`.
