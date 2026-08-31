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
    let stableStatuses: [Status] = [
        .ok, .end, .invalidArgument, .invalidUtf8, .nullPointer, .panic,
        .unsupported, .ioError, .closed, .limitExceeded, .providerError,
        .batchInUse, .domainMismatch,
    ]
    precondition(stableStatuses.map(\.rawValue) == Array(0...12))
    let futureStatus = Status(rawValue: UInt32.max)
    precondition(futureStatus.rawValue == UInt32.max)
    precondition(futureStatus.description == "UNKNOWN(4294967295)")

    let algorithms: [Algorithm] = [
        .standard, .transposition, .mergeAndSplit, .damerauLevenshtein,
    ]
    let orders: [QueryOrder] = [.traversal, .distanceThenTerm]
    let ruleSetKinds: [PhoneticRuleSetKind] = [.englishOrthography, .englishPhonetic]
    precondition(algorithms.map(\.rawValue) == Array(0...3))
    precondition(orders.map(\.rawValue) == [0, 1])
    precondition(ruleSetKinds.map(\.rawValue) == [0, 1])

    do {
        _ = try PhoneticPattern.regex("(")
        preconditionFailure("invalid regex unexpectedly compiled")
    } catch let failure as LiblevenshteinError {
        precondition(failure.status == .invalidArgument)
        precondition(!failure.description.isEmpty)
    }

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

    let algorithmDictionary = try DynamicDAWG()
    try algorithmDictionary.put("cat", value: 1)
    try algorithmDictionary.put("cot", value: 2)
    try algorithmDictionary.put("cut", value: 3)
    for algorithm in algorithms {
        let transducer = try Transducer.init(
            dictionary: algorithmDictionary,
            algorithm: algorithm
        )
        let cursor = try transducer.query(
            "cat",
            maximumDistance: 0,
            order: QueryOrder.traversal
        )
        precondition(Array(cursor).map(\.term) == [.text("cat")])
        transducer.close()
    }
    let batchTransducer = try Transducer.init(dictionary: algorithmDictionary)
    let batchCursor = try batchTransducer.query("cat", maximumDistance: 1)
    let firstBatch = try batchCursor.nextBatch(maximum: 1)
    let remainingCount = batchCursor.reduceBatches(0, batchSize: 1) { count, batch in
        count += batch.count
    }
    precondition(firstBatch.count == 1 && remainingCount == 2)
    batchCursor.close()
    let cache = try QueryCache(
        transducer: batchTransducer,
        maximumEntries: 4,
        maximumWeight: 4096
    )
    _ = Array(try cache.query("cat", maximumDistance: 1, order: .distanceThenTerm))
    _ = Array(try cache.query("cat", maximumDistance: 1, order: .distanceThenTerm))
    let warmStats = try cache.stats()
    precondition(warmStats.requests == 2 && warmStats.hits == 1)
    try algorithmDictionary.remove("cot")
    batchTransducer.close()
    algorithmDictionary.close()
    let refreshed = terms(
        try cache.query("cat", maximumDistance: 1, order: .distanceThenTerm)
    ).map(\.0)
    precondition(refreshed == ["cat", "cut"])
    _ = try cache.resetStats()
    let resetCacheStats = try cache.stats()
    precondition(resetCacheStats.requests == 0)
    _ = try cache.clear()
    let clearedCacheStats = try cache.stats()
    precondition(clearedCacheStats.residentEntries == 0)
    cache.close()

    let byteDictionary = try DynamicDAWG(unitDomain: .byte)
    try byteDictionary.put(bytes: [0x63, 0x61, 0x74], value: 11)
    let byteTransducer = try Transducer(dictionary: byteDictionary)
    do {
        _ = try byteTransducer.query(
            [UInt8(0x63), 0x61, 0x74],
            maximumDistance: 0,
            order: .distanceThenTerm
        )
        preconditionFailure("ordered byte query unexpectedly succeeded")
    } catch let failure as LiblevenshteinError {
        precondition(failure.status == .unsupported)
    }
    let byteQuery: [UInt8] = [0x63, 0x61, 0x74]
    let byteMatches = Array(try byteTransducer.query(byteQuery, maximumDistance: 0))
    precondition(byteMatches.count == 1)
    precondition(byteMatches[0].term == MatchTerm.bytes(byteQuery))
    precondition(byteMatches[0].distance == 0 && byteMatches[0].id == 11)
    byteTransducer.close()
    byteDictionary.close()

    let tokenDictionary = try DynamicDAWG(unitDomain: .u64)
    try tokenDictionary.put([1, UInt64.max], value: 12)
    let tokenTransducer = try Transducer(dictionary: tokenDictionary)
    do {
        _ = try tokenTransducer.query(
            [1, UInt64.max],
            maximumDistance: 0,
            order: .distanceThenTerm
        )
        preconditionFailure("ordered u64 query unexpectedly succeeded")
    } catch let failure as LiblevenshteinError {
        precondition(failure.status == .unsupported)
    }
    let tokenMatches = Array(
        try tokenTransducer.query([1, UInt64.max], maximumDistance: 0)
    )
    precondition(tokenMatches.count == 1)
    precondition(tokenMatches[0].term == .u64([1, UInt64.max]))
    precondition(tokenMatches[0].distance == 0 && tokenMatches[0].id == 12)
    tokenTransducer.close()
    tokenDictionary.close()

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
    let llre = try PhoneticPattern.llre("@name \"Greeting\"\n^hello$")
    let llreAccepted = try llre.matches("hello")
    let llreRejected = try llre.matches("world")
    precondition(llreAccepted && !llreRejected)
    llre.close()

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
    let rules = try PhoneticRuleSet.builtin(PhoneticRuleSetKind.englishOrthography)
    let builtinCount = try rules.count()
    let builtinApplied = try rules.apply("phone")
    precondition(builtinCount > 0)
    precondition(!builtinApplied.isEmpty)
    rules.close()
    let phoneticRules = try PhoneticRuleSet.builtin(PhoneticRuleSetKind.englishPhonetic)
    let phoneticRuleCount = try phoneticRules.count()
    precondition(phoneticRuleCount > 0)
    phoneticRules.close()
    let parsed = try PhoneticRuleSet.parse("ph -> f\ngh ->\n")
    let parsedCount = try parsed.count()
    let parsedApplied = try parsed.apply("phgh")
    precondition(parsedCount == 2)
    precondition(parsedApplied == "f")
    parsed.close()

    try PropertyTests.run()
    try LeakTests.run()

    print("Swift binding integration passed")
} catch {
    fatalError("Swift binding integration failed: \(error)")
}
