module VinaryTree.Liblevenshtein.FSharpUsage.Search

open VinaryTree.Interop
open VinaryTree.Liblevenshtein

/// Compile-checked F# usage of the same public package shipped to C# callers.
let suggestions (dictionary: IDictionaryResource) (text: string) : Match list =
    use transducer = new Transducer(dictionary, Algorithm.Standard)
    use query = transducer.Query(text, unativeint 2, QueryOrder.DistanceThenTerm)
    query |> Seq.toList
