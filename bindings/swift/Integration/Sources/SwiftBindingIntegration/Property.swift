import Foundation
import Libdictenstein
import Liblevenshtein

// C8 native property-based tests for the Swift facade: a seeded pseudo-PBT loop
// checked against an in-language Levenshtein oracle.
//
//   (a) distance symmetry d(a,b)==d(b,a), identity d(a,a)==0, and threshold
//       consistency dThreshold(a,b,k) == (d<=k ? d : nil) across the
//       Levenshtein/OSA/true-Damerau variants, plus standard oracle agreement;
//   (b) query result set == {t in dict : lev(query,t) <= k} over a random
//       libdictenstein DynamicDAWG dictionary, with exact distances and value
//       round-trips;
//   (c) u64 value round-trips, with 0 and UInt64.max pinned.
//
// A SplitMix64 RNG seeded from a constant makes each run reproducible. Distances
// count Unicode scalar values, and the alphabet's characters are all single
// scalars, so the oracle over `unicodeScalars` matches the native semantics.

/// Deterministic, seedable RNG (SplitMix64) for reproducible property runs.
struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

enum PropertyTests {
    static let alphabet: [Character] = ["a", "b", "c", "é"]

    static func generate(_ minLength: Int, _ maxLength: Int, _ rng: inout SeededRNG) -> String {
        let length = Int.random(in: minLength...maxLength, using: &rng)
        var result = ""
        for _ in 0..<length { result.append(alphabet.randomElement(using: &rng)!) }
        return result
    }

    static func oracle(_ a: String, _ b: String) -> Int {
        if a == b { return 0 }
        let left = Array(a.unicodeScalars)
        let right = Array(b.unicodeScalars)
        if left.isEmpty { return right.count }
        if right.isEmpty { return left.count }
        var previous = Array(0...right.count)
        for i in 1...left.count {
            var current = [i] + Array(repeating: 0, count: right.count)
            for j in 1...right.count {
                let cost = left[i - 1] == right[j - 1] ? 0 : 1
                current[j] = Swift.min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            }
            previous = current
        }
        return previous[right.count]
    }

    static func runQuery(
        _ entries: [(String, UInt64?)], _ query: String, _ maximumDistance: Int
    ) throws -> [String: (Int, UInt64?)] {
        let dictionary = try DynamicDAWG()
        for (term, value) in entries { _ = try dictionary.put(term, value: value) }
        let transducer = try Transducer(dictionary: dictionary)
        var result: [String: (Int, UInt64?)] = [:]
        let cursor = try transducer.query(query, maximumDistance: maximumDistance)
        for match in cursor {
            guard case let .text(term) = match.term else { fatalError("expected text term") }
            result[term] = (match.distance, match.id)
        }
        transducer.close()
        dictionary.close()
        return result
    }

    static func run() throws {
        var rng = SeededRNG(seed: 20_260_809)

        let variants: [(dist: (String, String) -> Int, thr: (String, String, Int) -> Int?)] = [
            ({ EditDistance.levenshtein($0, $1) }, { EditDistance.levenshtein($0, $1, threshold: $2) }),
            ({ EditDistance.damerauOSA($0, $1) }, { EditDistance.damerauOSA($0, $1, threshold: $2) }),
            ({ EditDistance.damerauLevenshtein($0, $1) }, { EditDistance.damerauLevenshtein($0, $1, threshold: $2) }),
        ]

        // (a) distance self-consistency and oracle agreement.
        for _ in 0..<400 {
            let a = generate(0, 6, &rng)
            let b = generate(0, 6, &rng)
            let k = Int.random(in: 0...3, using: &rng)
            precondition(EditDistance.levenshtein(a, b) == oracle(a, b), "standard oracle agreement")
            for variant in variants {
                let full = variant.dist(a, b)
                precondition(variant.dist(b, a) == full, "symmetry")
                precondition(variant.dist(a, a) == 0, "identity")
                let bounded = variant.thr(a, b, k)
                precondition(bounded == (full <= k ? full : nil), "threshold consistency")
            }
        }

        // (b) query result set equals the oracle, with exact distances and (c)
        // value round-trips (including nil values).
        for _ in 0..<150 {
            var entries: [String: UInt64?] = [:]
            let count = Int.random(in: 0...8, using: &rng)
            for _ in 0..<count {
                let term = generate(0, 6, &rng)
                let value: UInt64? = Int.random(in: 0...3, using: &rng) == 0
                    ? nil : UInt64.random(in: 0...UInt64.max, using: &rng)
                entries.updateValue(value, forKey: term)
            }
            let query = generate(0, 6, &rng)
            let k = Int.random(in: 0...3, using: &rng)
            let got = try runQuery(entries.map { ($0.key, $0.value) }, query, k)

            var expected: [String: UInt64?] = [:]
            for (term, value) in entries where oracle(query, term) <= k {
                expected.updateValue(value, forKey: term)
            }
            precondition(Set(got.keys) == Set(expected.keys), "result set equals oracle")
            for (term, observed) in got {
                precondition(observed.0 == oracle(query, term), "exact distance")
                precondition(observed.1 == expected[term]!, "value round-trip")
            }
        }

        // (c) u64 boundaries pinned.
        for value in [UInt64(0), 1, UInt64(1) << 63, UInt64.max] {
            let got = try runQuery([("term", value)], "term", 0)
            let match = got["term"]!
            precondition(match.0 == 0 && match.1 == value, "u64 round-trip \(value)")
        }

        print("Swift property tests passed")
    }
}
