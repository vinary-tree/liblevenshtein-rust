# Use liblevenshtein from F#

The `Liblevenshtein` NuGet package publishes the same .NET assembly for C#
and F#. No separate F# wrapper is needed. A dictionary producer such as
`Libdictenstein` supplies a `VinaryTree.Interop.IDictionaryResource` to the
reusable `Transducer`.

This example is compiled against both .NET 8 and .NET 10 by the binding CI:

```fsharp
module VinaryTree.Liblevenshtein.FSharpUsage.Search

open VinaryTree.Interop
open VinaryTree.Liblevenshtein

let suggestions (dictionary: IDictionaryResource) (text: string) : Match list =
    use transducer = new Transducer(dictionary, Algorithm.Standard)
    use query = transducer.Query(text, unativeint 2, QueryOrder.DistanceThenTerm)
    query |> Seq.toList
```

The two `use` bindings close native resources deterministically, including
when enumeration raises an exception. A query captures the dictionary
revision visible at its start; subsequent producer mutations do not change
that query's results. `Seq.toList` materializes host-owned `Match` values;
iterate directly for a large result set and still close the query after an
early exit. Reuse a transducer across independent queries, and use a separate
`QueryCache` per worker when complete-result caching is worthwhile.

The [versioned .NET API reference](../../docs/api/dotnet/index.md) describes
the exact shared public types and their ownership rules. The
[.NET binding guide](README.md) covers native loading, phonetic operations,
status handling, and cross-project dictionary collections. The
[compiled source](tests/VinaryTree.Liblevenshtein.FSharpUsage/Search.fs)
is the executable syntax authority for this example.
