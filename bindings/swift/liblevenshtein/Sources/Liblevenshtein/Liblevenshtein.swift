import CLiblevenshtein
import VinaryTreeInterop

public struct LiblevenshteinError: Error, CustomStringConvertible, Sendable {
    public let description: String
    init(_ fallback: String) {
        description = llev_last_error_message().map(String.init(cString:)) ?? fallback
    }
}

private func checked(_ status: LlevStatus) throws {
    guard status == LLEV_STATUS_OK else {
        throw LiblevenshteinError("liblevenshtein status \(status.rawValue)")
    }
}

public enum Algorithm: UInt32, Sendable {
    case standard = 0
    case transposition = 1
    case mergeAndSplit = 2
    case damerauLevenshtein = 3
}

public enum QueryOrder: UInt32, Sendable {
    case traversal = 0
    case distanceThenTerm = 1
}

public enum MatchTerm: Sendable, Equatable {
    case text(String)
    case bytes([UInt8])
    case u64([UInt64])
}

public struct Match: Sendable, Equatable {
    public let term: MatchTerm
    public let distance: Int
    public let id: UInt64?
}

/// One lazy query-start snapshot. Cursor lifetime is independent of both the
/// source dictionary facade and transducer facade.
public final class QueryCursor: Sequence, IteratorProtocol, @unchecked Sendable {
    private var raw: OpaquePointer?
    private var batch: [Match] = []
    private var index = 0
    private let batchSize: Int

    init(raw: OpaquePointer, batchSize: Int = Int(LLEV_DEFAULT_MATCH_BATCH)) {
        self.raw = raw
        self.batchSize = batchSize
    }

    deinit { close() }

    public func makeIterator() -> QueryCursor { self }

    public func next() -> Match? {
        if index == batch.count {
            do { batch = try nextBatch(maximum: batchSize) }
            catch { preconditionFailure("query cursor failed: \(error)") }
            index = 0
        }
        guard index < batch.count else { return nil }
        defer { index += 1 }
        return batch[index]
    }

    public func nextBatch(maximum: Int) throws -> [Match] {
        guard maximum > 0 else { throw LiblevenshteinError("batch size must be positive") }
        guard let raw else { return [] }
        var view = LlevMatchBatchView()
        let status = llev_query_cursor_next_batch(raw, maximum, &view)
        if status == LLEV_STATUS_END { return [] }
        try checked(status)
        defer {
            let release = llev_query_cursor_release_batch(raw, view.generation)
            precondition(release == LLEV_STATUS_OK, "native batch lease release failed")
        }
        guard let descriptors = view.matches else { return [] }
        return (0..<view.len).map { offset in
            let value = descriptors.advanced(by: offset).pointee
            let term: MatchTerm
            switch value.unit_domain {
            case VT_UNIT_DOMAIN_UNICODE_SCALAR:
                let bytes = UnsafeRawBufferPointer(start: value.term_data, count: value.byte_len)
                term = .text(String(decoding: bytes, as: UTF8.self))
            case VT_UNIT_DOMAIN_BYTE:
                let bytes = UnsafeRawBufferPointer(start: value.term_data, count: value.byte_len)
                term = .bytes(Array(bytes))
            case VT_UNIT_DOMAIN_U64:
                let tokens = value.term_data!
                    .assumingMemoryBound(to: UInt64.self)
                term = .u64(Array(UnsafeBufferPointer(start: tokens, count: value.term_len)))
            default:
                preconditionFailure("unknown unit domain")
            }
            return Match(
                term: term,
                distance: value.distance,
                id: value.has_id == 0 ? nil : value.id
            )
        }
    }

    public func reduceBatches<Result>(
        _ initial: Result,
        batchSize: Int = Int(LLEV_DEFAULT_MATCH_BATCH),
        _ reducer: (inout Result, [Match]) throws -> Void
    ) rethrows -> Result {
        var result = initial
        while true {
            let values: [Match]
            do { values = try nextBatch(maximum: batchSize) }
            catch { preconditionFailure("query cursor failed: \(error)") }
            if values.isEmpty { return result }
            try reducer(&result, values)
        }
    }

    public func close() {
        if let raw {
            let status = llev_query_cursor_free(raw)
            precondition(status == LLEV_STATUS_OK, "cursor closed with an outstanding batch")
            self.raw = nil
        }
    }
}

public final class Transducer: @unchecked Sendable {
    private var raw: OpaquePointer?

    public init(dictionary: some DictionaryResource, algorithm: Algorithm = .standard) throws {
        var output: OpaquePointer?
        try dictionary.withVtResource { resource in
            try checked(llev_transducer_new(resource, algorithm.rawValue, &output))
        }
        raw = output!
    }

    deinit { close() }

    private func handle() throws -> OpaquePointer {
        guard let raw else { throw LiblevenshteinError("transducer is closed") }
        return raw
    }

    public func query(
        _ text: String,
        maximumDistance: Int,
        order: QueryOrder = .traversal
    ) throws -> QueryCursor {
        let bytes = Array(text.utf8)
        var output: OpaquePointer?
        try bytes.withUnsafeBufferPointer { buffer in
            try checked(llev_transducer_query_utf8(
                try handle(), buffer.baseAddress, buffer.count,
                maximumDistance, order.rawValue, &output
            ))
        }
        return QueryCursor(raw: output!)
    }

    public func query(_ bytes: [UInt8], maximumDistance: Int) throws -> QueryCursor {
        var output: OpaquePointer?
        try bytes.withUnsafeBufferPointer { buffer in
            try checked(llev_transducer_query_bytes(
                try handle(), buffer.baseAddress, buffer.count,
                maximumDistance, QueryOrder.traversal.rawValue, &output
            ))
        }
        return QueryCursor(raw: output!)
    }

    public func query(_ tokens: [UInt64], maximumDistance: Int) throws -> QueryCursor {
        var output: OpaquePointer?
        try tokens.withUnsafeBufferPointer { buffer in
            try checked(llev_transducer_query_u64(
                try handle(), buffer.baseAddress, buffer.count,
                maximumDistance, QueryOrder.traversal.rawValue, &output
            ))
        }
        return QueryCursor(raw: output!)
    }

    public func query(_ pattern: PhoneticPattern, maximumDistance: UInt8) throws -> QueryCursor {
        var output: OpaquePointer?
        try checked(llev_transducer_query_pattern(
            try handle(), try pattern.handle(), maximumDistance, &output
        ))
        return QueryCursor(raw: output!)
    }

    public func close() {
        if let raw {
            llev_transducer_free(raw)
            self.raw = nil
        }
    }
}

public final class PhoneticPattern: @unchecked Sendable {
    private var raw: OpaquePointer?
    private init(_ raw: OpaquePointer) { self.raw = raw }
    deinit { close() }

    public static func regex(_ source: String) throws -> PhoneticPattern {
        try compile(source, llev_phonetic_pattern_compile_regex)
    }
    public static func llre(_ source: String) throws -> PhoneticPattern {
        try compile(source, llev_phonetic_pattern_compile_llre)
    }
    private static func compile(
        _ source: String,
        _ compiler: (UnsafePointer<CChar>?, Int, UnsafeMutablePointer<OpaquePointer?>?) -> LlevStatus
    ) throws -> PhoneticPattern {
        var output: OpaquePointer?
        try source.withCString { pointer in
            try checked(compiler(pointer, source.utf8.count, &output))
        }
        return PhoneticPattern(output!)
    }
    fileprivate func handle() throws -> OpaquePointer {
        guard let raw else { throw LiblevenshteinError("phonetic pattern is closed") }
        return raw
    }
    public func matches(_ input: String) throws -> Bool {
        var result: UInt8 = 0
        try input.withCString { pointer in
            try checked(llev_phonetic_pattern_matches(
                try handle(), pointer, input.utf8.count, &result
            ))
        }
        return result != 0
    }
    public func close() {
        if let raw {
            llev_phonetic_pattern_free(raw)
            self.raw = nil
        }
    }
}

public enum EditDistance {
    public static func levenshtein(_ source: String, _ target: String) -> Int {
        source.withCString { sourcePointer in
            target.withCString { targetPointer in
                llev_distance(sourcePointer, source.utf8.count, targetPointer, target.utf8.count)
            }
        }
    }
    public static func damerauOSA(_ source: String, _ target: String) -> Int {
        source.withCString { sourcePointer in
            target.withCString { targetPointer in
                llev_damerau_distance(sourcePointer, source.utf8.count, targetPointer, target.utf8.count)
            }
        }
    }
    public static func damerauLevenshtein(_ source: String, _ target: String) -> Int {
        source.withCString { sourcePointer in
            target.withCString { targetPointer in
                llev_true_damerau_distance(sourcePointer, source.utf8.count, targetPointer, target.utf8.count)
            }
        }
    }
}
