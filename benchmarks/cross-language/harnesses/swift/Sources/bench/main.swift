// Swift harness for the cross-language benchmark program.
//
// Implements harnesses/common/PROTOCOL.md over the Swift facades
// (Liblevenshtein + Libdictenstein, sharing the VinaryTreeInterop resource
// ABI). Fairness notes (PROTOCOL.md §10): release build, default runtime;
// the DynamicDAWG facade exposes no batch insert, so construction uses the
// facade's put() loop — recorded in every cell's notes (PROTOCOL.md §4).
//
// Clock (PROTOCOL.md §9): ContinuousClock (monotonic), durations reduced to
// whole nanoseconds via Duration.components.

import Foundation
import Libdictenstein
import Liblevenshtein

let wallCapSeconds = 300.0
let batchSize = 256 // facade cursor drain capacity (LLEV_DEFAULT_MATCH_BATCH)
let libraryVersion = "0.10.0"
let sampleDefinition =
    "one full pass over the query set; every cursor fully drained and " +
    "(term, distance) materialized"

// ---------------------------------------------------------------------------
// Diagnostics (stderr only; stdout stays empty per PROTOCOL.md §1)
// ---------------------------------------------------------------------------

func die(_ message: String) -> Never {
    FileHandle.standardError.write(Data("bench-cross-swift: \(message)\n".utf8))
    exit(2)
}

// ---------------------------------------------------------------------------
// Checksum primitives (PROTOCOL.md §8) — UInt64 with &* / &+ wrapping ops —
// and the startup self-test (§2).
// ---------------------------------------------------------------------------

let fnvOffset: UInt64 = 0xcbf2_9ce4_8422_2325
let fnvPrime: UInt64 = 0x0000_0100_0000_01b3

@inline(__always)
func fnvUpdate(_ hash: UInt64, _ byte: UInt8) -> UInt64 {
    (hash ^ UInt64(byte)) &* fnvPrime
}

func fnv1a64<Bytes: Sequence>(_ bytes: Bytes) -> UInt64 where Bytes.Element == UInt8 {
    var hash = fnvOffset
    for byte in bytes { hash = fnvUpdate(hash, byte) }
    return hash
}

func entryHash(_ term: String, _ distance: UInt64) -> UInt64 {
    var hash = fnvOffset
    for byte in term.utf8 { hash = fnvUpdate(hash, byte) }
    hash = fnvUpdate(hash, 0x00) // separator
    var value = distance
    for _ in 0..<8 { // LE64(distance)
        hash = fnvUpdate(hash, UInt8(truncatingIfNeeded: value))
        value >>= 8
    }
    return hash
}

func selfTest() {
    func expect(_ actual: UInt64, _ wanted: UInt64, _ label: String) {
        if actual != wanted {
            die("checksum self-test failed for \(label): got \(String(actual, radix: 16)), " +
                "want \(String(wanted, radix: 16))")
        }
    }
    expect(fnv1a64([]), 0xcbf2_9ce4_8422_2325, "fnv1a64(\"\")")
    expect(fnv1a64([0x61]), 0xaf63_dc4c_8601_ec8c, "fnv1a64(\"a\")")
    expect(entryHash("cat", 1), 0x9697_fa3e_5046_4bc4, "entry(cat,1)")
    expect(entryHash("cat", 0), 0xb592_c147_5b35_95e5, "entry(cat,0)")
    expect(entryHash("cot", 1), 0xb8ac_c5d3_816b_cdea, "entry(cot,1)")
    expect(entryHash("cat", 0) &+ entryHash("cot", 1), 0x6e3f_871a_dca1_63cf, "checksum{2}")
    expect(0, 0x0000_0000_0000_0000, "checksum{}")
}

func hex16(_ value: UInt64) -> String {
    let digits = String(value, radix: 16) // lowercase
    return String(repeating: "0", count: 16 - digits.count) + digits
}

// ---------------------------------------------------------------------------
// Monotonic clock (PROTOCOL.md §9)
// ---------------------------------------------------------------------------

let clock = ContinuousClock()

@inline(__always)
func nsSince(_ start: ContinuousClock.Instant) -> Int64 {
    let elapsed = start.duration(to: clock.now)
    return elapsed.components.seconds &* 1_000_000_000
        &+ elapsed.components.attoseconds / 1_000_000_000
}

// ---------------------------------------------------------------------------
// CLI (PROTOCOL.md §1)
// ---------------------------------------------------------------------------

struct Args {
    var mode = ""
    var algorithm: String?
    var maxDistance = -1
    var dictionary: String?
    var queries: String?
    var backend = ""
    var out: String?
    var samples = 30
    var warmupSeconds = 3.0
    var gateLimit = 200
    var reps = 10
    var cells: String?
}

func parseArgs(_ argv: [String]) -> Args {
    var args = Args()
    var index = 1
    while index < argv.count {
        guard index + 1 < argv.count else { die("flag requires a value: \(argv[index])") }
        let flag = argv[index]
        let value = argv[index + 1]
        switch flag {
        case "--mode": args.mode = value
        case "--algorithm": args.algorithm = value
        case "--max-distance": args.maxDistance = Int(value) ?? -1
        case "--dictionary": args.dictionary = value
        case "--queries": args.queries = value
        case "--backend": args.backend = value
        case "--out": args.out = value
        case "--samples": args.samples = Int(value) ?? args.samples
        case "--warmup-seconds": args.warmupSeconds = Double(value) ?? args.warmupSeconds
        case "--gate-limit": args.gateLimit = Int(value) ?? args.gateLimit
        case "--reps": args.reps = Int(value) ?? args.reps
        case "--cells": args.cells = value
        default: die("unknown flag: \(flag)")
        }
        index += 2
    }
    if args.mode.isEmpty || args.dictionary == nil || args.backend.isEmpty {
        die("--mode, --dictionary, --backend are required")
    }
    return args
}

func parseAlgorithm(_ name: String) -> Algorithm {
    switch name {
    case "standard": return .standard
    case "transposition": return .transposition
    case "merge_and_split": return .mergeAndSplit
    case "damerau_levenshtein": return .damerauLevenshtein
    default: die("unknown algorithm: \(name)")
    }
}

// ---------------------------------------------------------------------------
// Input loading (PROTOCOL.md §3)
// ---------------------------------------------------------------------------

func readLines(_ path: String) -> [String] {
    guard let data = FileManager.default.contents(atPath: path) else {
        die("cannot open \(path)")
    }
    let raw = String(decoding: data, as: UTF8.self)
    let lines = raw.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
    if lines.isEmpty { die("\(path) contains no lines") }
    return lines
}

func assertStrictlySorted(_ lines: [String], _ path: String) {
    for i in 0..<(lines.count - 1) {
        // UTF-8 lexicographic comparison == strict bytewise order (§3.2).
        if !lines[i].utf8.lexicographicallyPrecedes(lines[i + 1].utf8) {
            die("\(path) is not strictly byte-sorted at line \(i + 1): " +
                "\"\(lines[i])\" >= \"\(lines[i + 1])\"")
        }
    }
}

// ---------------------------------------------------------------------------
// Dictionary + transducer side (PROTOCOL.md §4–5)
// ---------------------------------------------------------------------------

struct Triple: Equatable {
    var matches: UInt64 = 0
    var bytes: UInt64 = 0
    var distanceSum: UInt64 = 0
}

final class Side {
    var dictionary: Libdictenstein.Dictionary?
    var transducer: Transducer?
    var preparedEntries: [(String, UInt64?)]?

    func buildDictionary(_ terms: [String], backend: String) {
        do {
            switch backend {
            case "dynamic_dawg":
                // The Swift facade exposes no batch insert (unlike PutAll /
                // update_many / put_all elsewhere); the put() loop below IS
                // the facade's bulk path and is recorded in the cell notes.
                let dawg = try DynamicDAWG(unitDomain: .unicodeScalar)
                for term in terms {
                    _ = try dawg.put(term)
                }
                dictionary = dawg
            case "double_array_trie":
                if preparedEntries == nil {
                    var entries: [(String, UInt64?)] = []
                    entries.reserveCapacity(terms.count) // preallocated once
                    for term in terms { entries.append((term, nil)) }
                    preparedEntries = entries
                }
                dictionary = try DoubleArrayTrie(entries: preparedEntries!, unitDomain: .unicodeScalar)
            default:
                die("unknown backend: \(backend)")
            }
        } catch {
            die("dictionary construction failed: \(error)")
        }
    }

    func freeDictionary() {
        transducer?.close()
        transducer = nil
        dictionary?.close()
        dictionary = nil
    }

    func createTransducer(_ algorithm: String) {
        transducer?.close()
        do {
            transducer = try Transducer(dictionary: dictionary!, algorithm: parseAlgorithm(algorithm))
        } catch {
            die("transducer construction failed: \(error)")
        }
    }

    func fullPass(_ queries: [String], limit: Int, maxDistance: Int,
                  checksum: inout UInt64, withChecksum: Bool) -> Triple {
        var triple = Triple()
        guard let machine = transducer else { die("createTransducer must run before fullPass") }
        do {
            for i in 0..<limit {
                let cursor = try machine.query(queries[i], maximumDistance: maxDistance)
                while let match = cursor.next() {
                    guard case let .text(term) = match.term else { die("expected text match") }
                    triple.matches &+= 1
                    triple.bytes &+= UInt64(term.utf8.count)
                    triple.distanceSum &+= UInt64(match.distance)
                    if withChecksum {
                        checksum = checksum &+ entryHash(term, UInt64(match.distance))
                    }
                }
                cursor.close()
            }
        } catch {
            die("query failed: \(error)")
        }
        return triple
    }
}

// ---------------------------------------------------------------------------
// Result JSON (PROTOCOL.md §11: harness-filled fields only), hand-emitted so
// numbers stay exact and the field order mirrors the C harness.
// ---------------------------------------------------------------------------

func jsonEscape(_ text: String) -> String {
    var escaped = ""
    escaped.reserveCapacity(text.count)
    for scalar in text.unicodeScalars {
        switch scalar {
        case "\"": escaped += "\\\""
        case "\\": escaped += "\\\\"
        case "\n": escaped += "\\n"
        case "\r": escaped += "\\r"
        case "\t": escaped += "\\t"
        default:
            if scalar.value < 0x20 {
                escaped += String(format: "\\u%04x", scalar.value)
            } else {
                escaped.unicodeScalars.append(scalar)
            }
        }
    }
    return escaped
}

func jsonNumber(_ value: Double) -> String {
    if value == value.rounded() && abs(value) < 1e15 {
        return String(Int64(value))
    }
    return String(value)
}

func swiftRuntimeVersion() -> String {
    // Swift exposes no runtime API for the toolchain version; a
    // compile-time compiler() ladder pins the minor release.
    #if compiler(>=6.4)
    return "Swift compiler >=6.4"
    #elseif compiler(>=6.3)
    return "Swift compiler 6.3"
    #elseif compiler(>=6.2)
    return "Swift compiler 6.2"
    #elseif compiler(>=6.1)
    return "Swift compiler 6.1"
    #elseif compiler(>=6.0)
    return "Swift compiler 6.0"
    #else
    return "Swift compiler <6.0"
    #endif
}

struct CellOutput {
    var mode: String
    var structure: String
    var algorithm: String
    var maxDistance: Int
    var dictionaryFile: String
    var termCount: Int
    var constructNs: Int64?
    var queriesFile: String
    var queryCount: Int
    var warmupPasses = 0
    var samplesRequested: Int
    var warmupSeconds: Double
    var samplesNs: [Int64] = []
    var triple = Triple()
    var checksum: UInt64 = 0
    var constructTimes: [Int64]?
    var status: String
    var notes: [String]
}

func writeResult(_ outPath: String, _ cell: CellOutput) {
    let timestampFormatter = ISO8601DateFormatter()
    timestampFormatter.formatOptions = [.withInternetDateTime]
    let queryset = (cell.queriesFile as NSString).lastPathComponent
        .replacingOccurrences(of: ".txt", with: "")

    var json = "{\n"
    json += "  \"schema_version\": \"1.0.0\",\n"
    json += "  \"suite\": \"cross-language-v1\",\n"
    json += "  \"timestamp_utc\": \"\(timestampFormatter.string(from: Date()))\",\n"
    json += "  \"target\": {\n"
    json += "    \"language\": \"swift\",\n"
    json += "    \"implementation\": \"vinary-tree\",\n"
    json += "    \"backend\": \"systemlib\",\n"
    json += "    \"runtime_version\": \"\(jsonEscape(swiftRuntimeVersion()))\",\n"
    json += "    \"library_version\": \"\(libraryVersion)\",\n"
    json += "    \"artifact\": { \"kind\": \"local-build\", \"id\": \"liblevenshtein-swift@\(libraryVersion)\" }\n"
    json += "  },\n"
    json += "  \"dictionary\": {\n"
    json += "    \"file\": \"\(jsonEscape(cell.dictionaryFile))\",\n"
    json += "    \"term_count\": \(cell.termCount),\n"
    json += "    \"structure\": \"\(cell.structure)\",\n"
    json += "    \"unit_domain\": \"unicode_scalar\""
    if let constructNs = cell.constructNs {
        json += ",\n    \"construct_ns\": \(constructNs)\n"
    } else {
        json += "\n"
    }
    json += "  },\n"
    json += "  \"workload\": {\n"
    json += "    \"queryset\": \"\(jsonEscape(queryset))\",\n"
    json += "    \"file\": \"\(jsonEscape(cell.queriesFile))\",\n"
    json += "    \"query_count\": \(cell.queryCount)\n"
    json += "  },\n"
    json += "  \"algorithm\": \"\(cell.algorithm)\",\n"
    json += "  \"max_distance\": \(cell.maxDistance),\n"
    json += "  \"mode\": \"\(cell.mode == "memory-child" ? "memory" : cell.mode)\",\n"
    json += "  \"protocol\": {\n"
    json += "    \"timer\": \"monotonic\",\n"
    json += "    \"harness\": \"self-timed\",\n"
    json += "    \"warmup_seconds_min\": \(jsonNumber(cell.warmupSeconds)),\n"
    json += "    \"warmup_passes\": \(cell.warmupPasses),\n"
    json += "    \"samples_requested\": \(cell.samplesRequested),\n"
    json += "    \"sample_definition\": \"\(jsonEscape(sampleDefinition))\",\n"
    json += "    \"batch_size\": \(batchSize),\n"
    json += "    \"wall_cap_seconds\": \(jsonNumber(wallCapSeconds))\n"
    json += "  },\n"
    if let constructTimes = cell.constructTimes {
        json += "  \"construct\": {\n"
        json += "    \"reps\": \(constructTimes.count),\n"
        json += "    \"times_ns\": [\(constructTimes.map(String.init).joined(separator: ", "))],\n"
        json += "    \"term_count\": \(cell.termCount)\n"
        json += "  },\n"
    } else {
        json += "  \"measurements\": {\n"
        json += "    \"samples_ns\": [\(cell.samplesNs.map(String.init).joined(separator: ", "))],\n"
        json += "    \"sample_count\": \(cell.samplesNs.count),\n"
        json += "    \"matches_per_pass\": \(cell.triple.matches),\n"
        json += "    \"term_bytes_per_pass\": \(cell.triple.bytes),\n"
        json += "    \"distance_sum_per_pass\": \(cell.triple.distanceSum),\n"
        json += "    \"checksum_hex\": \"\(hex16(cell.checksum))\"\n"
        json += "  },\n"
    }
    json += "  \"status\": \"\(cell.status)\",\n"
    json += "  \"notes\": [\(cell.notes.map { "\"\(jsonEscape($0))\"" }.joined(separator: ", "))]\n"
    json += "}\n"

    let parent = (outPath as NSString).deletingLastPathComponent
    if !parent.isEmpty {
        try? FileManager.default.createDirectory(
            atPath: parent, withIntermediateDirectories: true)
    }
    do {
        try json.write(toFile: outPath, atomically: true, encoding: .utf8)
    } catch {
        die("cannot write \(outPath): \(error)")
    }
}

func baseNotes(_ backend: String) -> [String] {
    var notes = [
        "Swift systemLibrary facade over the shared C ABI",
        "clock: ContinuousClock, Duration reduced to whole nanoseconds",
        "runtime_version pinned by a compile-time compiler() ladder (no runtime toolchain API)",
    ]
    if backend == "dynamic_dawg" {
        notes.append(
            "dynamic_dawg construction uses the facade's put() loop (no batch API in the Swift facade)")
    }
    return notes
}

// ---------------------------------------------------------------------------
// Modes (PROTOCOL.md §6)
// ---------------------------------------------------------------------------

func runQueryCell(side: Side, args: Args, queries: [String], algorithm: String,
                  maxDistance: Int, queriesPath: String, outPath: String,
                  termCount: Int, constructNs: Int64) {
    var checksum: UInt64 = 0
    let gate = side.fullPass(queries, limit: queries.count, maxDistance: maxDistance,
                             checksum: &checksum, withChecksum: true)

    let warmStart = clock.now
    let warmupNs = Int64(args.warmupSeconds * 1e9)
    var warmupPasses = 0
    var lastPassNs: Int64 = 0
    var ignored: UInt64 = 0
    while nsSince(warmStart) < warmupNs || warmupPasses < 2 {
        let t0 = clock.now
        let triple = side.fullPass(queries, limit: queries.count, maxDistance: maxDistance,
                                   checksum: &ignored, withChecksum: false)
        lastPassNs = nsSince(t0)
        if triple != gate { die("nondeterministic result during warmup") }
        warmupPasses += 1
    }

    var sampleCount = args.samples
    var status = "ok"
    var notes = baseNotes(args.backend)
    let lastPassSeconds = Double(lastPassNs) / 1e9
    if Double(sampleCount) * lastPassSeconds > wallCapSeconds {
        let reduced = max(10, Int(wallCapSeconds / lastPassSeconds))
        notes.append(
            "samples reduced from \(sampleCount) to \(reduced) by the " +
            "\(jsonNumber(wallCapSeconds))s wall cap " +
            "(estimated pass \(String(format: "%.3f", lastPassSeconds))s)")
        sampleCount = reduced
        status = "degraded"
    }

    var samplesNs = [Int64](repeating: 0, count: sampleCount) // preallocated (§3.4)
    for i in 0..<sampleCount {
        let t0 = clock.now
        let triple = side.fullPass(queries, limit: queries.count, maxDistance: maxDistance,
                                   checksum: &ignored, withChecksum: false)
        samplesNs[i] = nsSince(t0)
        if triple != gate { die("nondeterministic result during measurement") }
    }

    writeResult(outPath, CellOutput(
        mode: "query",
        structure: args.backend,
        algorithm: algorithm,
        maxDistance: maxDistance,
        dictionaryFile: args.dictionary!,
        termCount: termCount,
        constructNs: constructNs,
        queriesFile: queriesPath,
        queryCount: queries.count,
        warmupPasses: warmupPasses,
        samplesRequested: args.samples,
        warmupSeconds: args.warmupSeconds,
        samplesNs: samplesNs,
        triple: gate,
        checksum: checksum,
        status: status,
        notes: notes))
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

selfTest()
let args = parseArgs(CommandLine.arguments)

let terms = readLines(args.dictionary!)
assertStrictlySorted(terms, args.dictionary!)
let side = Side()

if args.mode == "construct" {
    guard let outPath = args.out else { die("--out is required for construct mode") }
    side.buildDictionary(terms, backend: args.backend) // warmup build
    side.freeDictionary()
    var times = [Int64](repeating: 0, count: args.reps) // preallocated
    for r in 0..<args.reps {
        let t0 = clock.now
        side.buildDictionary(terms, backend: args.backend)
        times[r] = nsSince(t0)
        side.freeDictionary()
    }
    var notes = baseNotes(args.backend)
    notes.append("construct mode: timed region is the build from the pre-sorted in-memory list only")
    writeResult(outPath, CellOutput(
        mode: "construct",
        structure: args.backend,
        algorithm: "standard",
        maxDistance: 1,
        dictionaryFile: args.dictionary!,
        termCount: terms.count,
        constructNs: nil,
        queriesFile: args.queries ?? "workload/queries/hits.txt",
        queryCount: 1,
        warmupPasses: 1,
        samplesRequested: args.reps,
        warmupSeconds: args.warmupSeconds,
        constructTimes: times,
        status: "ok",
        notes: notes))
    exit(0)
}

let buildStart = clock.now
side.buildDictionary(terms, backend: args.backend)
let constructNs = nsSince(buildStart)

@MainActor
func runOne(algorithm: String, maxDistance: Int, queriesPath: String, outPath: String) {
    side.createTransducer(algorithm)
    let queries = readLines(queriesPath)
    switch args.mode {
    case "verify":
        let limit = min(args.gateLimit, queries.count)
        var checksum: UInt64 = 0
        let triple = side.fullPass(queries, limit: limit, maxDistance: maxDistance,
                                   checksum: &checksum, withChecksum: true)
        writeResult(outPath, CellOutput(
            mode: "verify",
            structure: args.backend,
            algorithm: algorithm,
            maxDistance: maxDistance,
            dictionaryFile: args.dictionary!,
            termCount: terms.count,
            constructNs: constructNs,
            queriesFile: queriesPath,
            queryCount: limit,
            samplesRequested: 0,
            warmupSeconds: args.warmupSeconds,
            triple: triple,
            checksum: checksum,
            status: "ok",
            notes: baseNotes(args.backend)))
    case "memory-child":
        var checksum: UInt64 = 0
        let triple = side.fullPass(queries, limit: queries.count, maxDistance: maxDistance,
                                   checksum: &checksum, withChecksum: true)
        writeResult(outPath, CellOutput(
            mode: "memory-child",
            structure: args.backend,
            algorithm: algorithm,
            maxDistance: maxDistance,
            dictionaryFile: args.dictionary!,
            termCount: terms.count,
            constructNs: constructNs,
            queriesFile: queriesPath,
            queryCount: queries.count,
            samplesRequested: 0,
            warmupSeconds: args.warmupSeconds,
            triple: triple,
            checksum: checksum,
            status: "ok",
            notes: baseNotes(args.backend)))
    case "query":
        runQueryCell(side: side, args: args, queries: queries, algorithm: algorithm,
                     maxDistance: maxDistance, queriesPath: queriesPath, outPath: outPath,
                     termCount: terms.count, constructNs: constructNs)
    default:
        die("unknown mode: \(args.mode)")
    }
}

if let cellsPath = args.cells {
    guard let data = FileManager.default.contents(atPath: cellsPath) else {
        die("cannot open cells file \(cellsPath)")
    }
    let rows = String(decoding: data, as: UTF8.self)
        .split(separator: "\n", omittingEmptySubsequences: true)
        .map { $0.trimmingCharacters(in: .whitespaces) }
        .filter { !$0.isEmpty && !$0.hasPrefix("#") }
    for row in rows {
        let fields = row.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
        if fields.count != 4 { die("cells row needs 4 tab-separated fields: \(row)") }
        guard let distance = Int(fields[1]) else { die("bad max_distance in cells row: \(row)") }
        runOne(algorithm: fields[0], maxDistance: distance, queriesPath: fields[2], outPath: fields[3])
    }
} else {
    guard let algorithm = args.algorithm, args.maxDistance >= 0,
          let queriesPath = args.queries, let outPath = args.out else {
        die("--algorithm, --max-distance, --queries, --out are required")
    }
    runOne(algorithm: algorithm, maxDistance: args.maxDistance,
           queriesPath: queriesPath, outPath: outPath)
}

side.freeDictionary()
exit(0)
