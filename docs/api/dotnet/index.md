# liblevenshtein .NET API — 4.0.0-rc.6

The `Liblevenshtein` NuGet package exposes edit-distance automata, streaming
fuzzy searches over a retained dictionary, reusable phonetic patterns, and a
bounded result cache through the `VinaryTree.Liblevenshtein` namespace. The
`VinaryTree.Interop` package supplies `IDictionaryResource`; a producer such
as `Libdictenstein` creates the dictionary. The two packages exchange a
versioned native resource, not a copy of the dictionary.

## Search a dictionary

```csharp
using System.Collections.Generic;
using System.Linq;
using VinaryTree.Interop;
using VinaryTree.Liblevenshtein;

static IReadOnlyList<Match> Search(IDictionaryResource dictionary, string text)
{
    using var transducer = new Transducer(dictionary, Algorithm.Standard);
    using var query = transducer.Query(text, 2, QueryOrder.DistanceThenTerm);
    return query.ToArray();
}
```

`Query` retains the dictionary revision visible at query creation. A later
mutation of the producer does not change that query's results. `Query` is a
one-shot `IEnumerable<Match>`; dispose it even when enumeration stops early.
Each `Match` owns its term after the current native batch has been released.

For repeated queries, keep the `Transducer` and create a fresh `Query` each
time. `QueryCache` is an optional bounded memo for complete repeated results;
create one cache per worker, set both entry and weight limits, and dispose it
when the worker ends. Its counters are available through `QueryCache.Stats`.

```csharp
using var transducer = new Transducer(dictionary, Algorithm.Transposition);
using var cache = new QueryCache(
    transducer, maximumEntries: 512, maximumWeight: 8 * 1024 * 1024);
using var cachedQuery = cache.Query("cat", 2, QueryOrder.DistanceThenTerm);
foreach (var match in cachedQuery)
    Console.WriteLine($"{match.Term}: {match.Distance}");
Console.WriteLine($"cache hits: {cache.Stats.Hits}");
```

`Algorithm.Standard` selects ordinary insert/delete/substitute edits;
`Transposition` adds adjacent swaps under optimal-string-alignment rules;
`MergeAndSplit` adds symmetric two-to-one and one-to-two edits; and
`DamerauLevenshtein` permits unrestricted adjacent transpositions. The
`Transducer.Query` overloads preserve three separate dictionary domains:
Unicode `string`, raw `ReadOnlySpan<byte>`, and `ReadOnlySpan<ulong>` tokens.
Only the Unicode overload currently accepts an explicit result order; the
byte and token overloads use traversal order. A phonetic-pattern query is a
fourth overload and composes the dictionary with the compiled pattern.

## Distance and phonetic operations

`Distance.Levenshtein` counts insertion, deletion, and substitution. The
`Damerau` overload uses optimal string alignment: one adjacent transposition
cannot be reused on the same substring. `TrueDamerau` uses unrestricted
Damerau–Levenshtein edits. Thresholded overloads return the exact distance
within the bound, or `nuint.MaxValue - 1` when it is exceeded. This sentinel
is not an exact distance.

`PhoneticPattern` and `PhoneticRuleSet` are independently disposable native
resources. Compile or parse them once when applying them repeatedly. A
`LiblevenshteinException` carries both a typed `Status` and the exact numeric
`StatusCode`, so callers can inspect unknown statuses from a compatible newer
native ABI without parsing diagnostic text.

F# callers can use the same types with lexical `use` bindings; the
[compile-checked F# guide](https://github.com/vinary-tree/liblevenshtein-rust/blob/v4.0.0-rc.6/bindings/dotnet/fsharp.md) gives the
complete example. For installation, native-library loading, full examples,
and cross-project dictionary ownership, use the
[.NET binding guide](https://github.com/vinary-tree/liblevenshtein-rust/blob/v4.0.0-rc.6/bindings/dotnet/README.md).
