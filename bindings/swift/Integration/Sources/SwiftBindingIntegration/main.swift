import Foundation
import Libdictenstein
import Liblevenshtein

func terms(_ cursor: QueryCursor) -> [(String, UInt64?)] {
    Array(cursor).map { match in
        guard case let .text(term) = match.term else { fatalError("expected text") }
        return (term, match.id)
    }
}

do {
    for trace in 0..<64 {
        let dictionary = try DynamicDAWG()
        for index in 0..<16 {
            try dictionary.put("t\(trace)-\(index)", value: UInt64(index))
        }
        let transducer = try Transducer(dictionary: dictionary)
        let expected = terms(try transducer.query("", maximumDistance: 64, order: .distanceThenTerm))
        let cursor = try transducer.query("", maximumDistance: 64, order: .distanceThenTerm)
        var actual = [cursor.next()!]
        try dictionary.remove("t\(trace)-1")
        try dictionary.put("t\(trace)-2", value: 999)
        actual.append(cursor.next()!)
        try dictionary.clear()
        _ = try dictionary.compact()
        try dictionary.put("after-\(trace)", value: 1000)
        actual.append(contentsOf: cursor)
        let normalized = actual.map { match -> (String, UInt64?) in
            guard case let .text(term) = match.term else { fatalError("expected text") }
            return (term, match.id)
        }
        precondition(normalized.elementsEqual(expected, by: { $0.0 == $1.0 && $0.1 == $1.1 }))
        dictionary.close()
        let freshTerms = terms(try transducer.query("", maximumDistance: 64)).map(\.0)
        precondition(freshTerms == ["after-\(trace)"])
        transducer.close()
    }

    let dat = try DoubleArrayTrie(entries: [("café", 7), ("caff", nil)])
    let datLookup = try dat.get("caff")
    precondition(datLookup == Lookup(found: true, value: nil))
    dat.close()
    let suffixes = try SCDAWG()
    try suffixes.put("cat", value: 1)
    try suffixes.put("cot", value: 2)
    let containsSuffix = try suffixes.containsSubstring("ot")
    let suffixFrequency = try suffixes.substringFrequency("t")
    precondition(containsSuffix)
    precondition(suffixFrequency == 2)
    suffixes.close()

    let path = FileManager.default.temporaryDirectory
        .appendingPathComponent("vinary-tree-swift-\(UUID().uuidString).artrie").path
    var persistent: PersistentARTrie? = try .create(at: path)
    try persistent!.put("cat", value: 1)
    try persistent!.checkpoint()
    persistent!.close()
    persistent = try .open(at: path)
    let persistentLookup = try persistent!.get("cat")
    precondition(persistentLookup == Lookup(found: true, value: 1))
    persistent!.close()

    let pattern = try PhoneticPattern.regex("c[ao]t")
    let patternMatches = try pattern.matches("cat")
    precondition(patternMatches)
    // LLEV-B11: PhoneticPattern.size now surfaces the automaton dimensions.
    let patternSize = try pattern.size()
    precondition(patternSize.states > 0 && patternSize.transitions > 0)
    pattern.close()

    precondition(EditDistance.levenshtein("kitten", "sitting") == 3)
    // LLEV-B11: all three threshold overloads are bound; nil means the exact
    // distance exceeds the bound (native usize::MAX - 1 sentinel).
    precondition(EditDistance.levenshtein("kitten", "sitting", threshold: 3) == 3)
    precondition(EditDistance.levenshtein("kitten", "sitting", threshold: 2) == nil)
    precondition(EditDistance.damerauOSA("ab", "ba", threshold: 1) == 1)
    // "ca" -> "abc": OSA is 3 but unrestricted Damerau-Levenshtein is 2.
    precondition(EditDistance.damerauOSA("ca", "abc", threshold: 2) == nil)
    precondition(EditDistance.damerauLevenshtein("ca", "abc", threshold: 2) == 2)
    precondition(EditDistance.damerauLevenshtein("ca", "abc", threshold: 1) == nil)

    // LLEV-B11: the PhoneticRuleSet facade (parse/builtin/count/apply/close).
    let rules = try PhoneticRuleSet.builtin(.englishOrthography)
    let builtinCount = try rules.count()
    let builtinApplied = try rules.apply("phone")
    precondition(builtinCount > 0)
    precondition(!builtinApplied.isEmpty)
    rules.close()
    let parsed = try PhoneticRuleSet.parse("ph -> f\ngh ->\n")
    let parsedCount = try parsed.count()
    let parsedApplied = try parsed.apply("phgh")
    precondition(parsedCount == 2)
    precondition(parsedApplied == "f")
    parsed.close()

    print("Swift binding integration passed")
} catch {
    fatalError("Swift binding integration failed: \(error)")
}
