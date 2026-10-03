import CLiblevenshtein
import VinaryTreeInterop

/// A native operation failure with a typed, forward-compatible status and copied diagnostic.
///
/// Inspect ``status`` for control flow. The message is human-readable context and may
/// change across native library versions.
public struct LiblevenshteinError: Error, CustomStringConvertible, Sendable {
    /// The exact native status, including raw values unknown to this Swift release.
    public let status: Status
    /// A diagnostic copied before another native call can replace it.
    public let description: String

    init(nativeStatus: LlevStatus, fallback: String) {
        status = Status(rawValue: nativeStatus.rawValue)
        description = llev_last_error_message().map(String.init(cString:)) ?? fallback
    }

    init(status: Status, message: String) {
        self.status = status
        description = message
    }
}

private func checked(_ status: LlevStatus) throws {
    guard status == LLEV_STATUS_OK else {
        throw LiblevenshteinError(
            nativeStatus: status,
            fallback: "liblevenshtein status \(status.rawValue)"
        )
    }
}

/// A matched dictionary key in the same unit domain as the query.
///
/// Keeping the cases distinct avoids lossy transcoding of arbitrary bytes or
/// unsigned 64-bit tokens into Unicode text.
public enum MatchTerm: Sendable, Equatable {
    /// A Unicode-text term.
    case text(String)
    /// An exact byte sequence, including embedded zero bytes.
    case bytes([UInt8])
    /// An exact unsigned 64-bit token sequence.
    case u64([UInt64])
}

/// An owned fuzzy match copied from a bounded native result batch.
///
/// The term, distance, and optional dictionary identifier remain valid after
/// the cursor advances or closes.
public struct Match: Sendable, Equatable {
    /// The matched term in its original unit domain.
    public let term: MatchTerm
    /// The selected edit distance between query and term.
    public let distance: Int
    /// The dictionary's optional unsigned 64-bit identifier.
    public let id: UInt64?
}

/// A one-shot, lazy traversal of one query-start dictionary revision.
///
/// Cursor lifetime is independent of both the source dictionary facade and
/// transducer facade. Use ``nextBatch(maximum:)`` when failures must be handled
/// as thrown errors; `Sequence` iteration cannot throw and traps on a late
/// native failure. Close a cursor explicitly after early termination.
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

    /// Return this one-shot cursor as its iterator; this does not rewind it.
    public func makeIterator() -> QueryCursor { self }

    /// Return the next owned match, or `nil` at the end of the stream.
    ///
    /// A late native failure traps because `IteratorProtocol.next()` cannot
    /// throw. Use ``nextBatch(maximum:)`` for recoverable error handling.
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

    /// Copy and return at most `maximum` matches from the next native batch.
    ///
    /// The native lease is released before this method returns. An empty array
    /// means the cursor is exhausted; `maximum` must be strictly positive.
    public func nextBatch(maximum: Int) throws -> [Match] {
        guard maximum > 0 else {
            throw LiblevenshteinError(
                status: .invalidArgument,
                message: "batch size must be positive"
            )
        }
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

    /// Fold bounded batches without materializing the complete result set.
    ///
    /// The reducer receives Swift-owned matches. A reducer error propagates;
    /// a late native query error traps because this convenience API is
    /// `rethrows`. Use ``nextBatch(maximum:)`` to handle both kinds of error.
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

    /// Release this cursor's native resource; repeated calls are harmless.
    ///
    /// Matches returned earlier remain owned Swift values.
    public func close() {
        if let raw {
            let status = llev_query_cursor_free(raw)
            precondition(status == LLEV_STATUS_OK, "cursor closed with an outstanding batch")
            self.raw = nil
        }
    }
}

/// A reusable edit-distance automaton retaining a dictionary resource.
///
/// Each query captures the dictionary revision visible when that query begins.
/// Construction retains the provider rather than copying its entries.
public final class Transducer: @unchecked Sendable {
    private var raw: OpaquePointer?

    /// Retain `dictionary` and select the edit algorithm for subsequent queries.
    ///
    /// - Parameters:
    ///   - dictionary: A producer-provided resource such as a libdictenstein dictionary.
    ///   - algorithm: Edit semantics shared by all queries on this transducer.
    public init(dictionary: some DictionaryResource, algorithm: Algorithm = .standard) throws {
        var output: OpaquePointer?
        try dictionary.withVtResource { resource in
            try checked(llev_transducer_new(resource, algorithm.rawValue, &output))
        }
        raw = output!
    }

    deinit { close() }

    fileprivate func handle() throws -> OpaquePointer {
        guard let raw else {
            throw LiblevenshteinError(status: .closed, message: "transducer is closed")
        }
        return raw
    }

    /// Start a Unicode-text query against the current dictionary revision.
    ///
    /// `maximumDistance` must be nonnegative. The caller owns and should close
    /// the returned cursor, especially after early termination.
    public func query(
        _ text: String,
        maximumDistance: Int,
        order: QueryOrder = .traversal
    ) throws -> QueryCursor {
        guard maximumDistance >= 0 else {
            throw LiblevenshteinError(status: .invalidArgument, message: "maximumDistance must be nonnegative")
        }
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

    /// Start an exact-byte query without Unicode decoding.
    ///
    /// The dictionary must use the byte unit domain. Only traversal ordering
    /// is supported for this domain by the current native ABI.
    public func query(
        _ bytes: [UInt8],
        maximumDistance: Int,
        order: QueryOrder = .traversal
    ) throws -> QueryCursor {
        guard maximumDistance >= 0 else {
            throw LiblevenshteinError(status: .invalidArgument, message: "maximumDistance must be nonnegative")
        }
        var output: OpaquePointer?
        try bytes.withUnsafeBufferPointer { buffer in
            try checked(llev_transducer_query_bytes(
                try handle(), buffer.baseAddress, buffer.count,
                maximumDistance, order.rawValue, &output
            ))
        }
        return QueryCursor(raw: output!)
    }

    /// Start an unsigned 64-bit token query without narrowing token values.
    ///
    /// The dictionary must use the token unit domain. Only traversal ordering
    /// is supported for this domain by the current native ABI.
    public func query(
        _ tokens: [UInt64],
        maximumDistance: Int,
        order: QueryOrder = .traversal
    ) throws -> QueryCursor {
        guard maximumDistance >= 0 else {
            throw LiblevenshteinError(status: .invalidArgument, message: "maximumDistance must be nonnegative")
        }
        var output: OpaquePointer?
        try tokens.withUnsafeBufferPointer { buffer in
            try checked(llev_transducer_query_u64(
                try handle(), buffer.baseAddress, buffer.count,
                maximumDistance, order.rawValue, &output
            ))
        }
        return QueryCursor(raw: output!)
    }

    /// Intersect this dictionary with a compiled phonetic pattern and a bound.
    ///
    /// The pattern and cursor retain their own native resources. Close both
    /// when they are no longer needed.
    public func query(_ pattern: PhoneticPattern, maximumDistance: UInt8) throws -> QueryCursor {
        var output: OpaquePointer?
        try checked(llev_transducer_query_pattern(
            try handle(), try pattern.handle(), maximumDistance, &output
        ))
        return QueryCursor(raw: output!)
    }

    /// Release this transducer; already-created cursors remain valid.
    public func close() {
        if let raw {
            llev_transducer_free(raw)
            self.raw = nil
        }
    }
}

/// Immutable TinyLFU/SIEVE counters and current bounded native residency.
/// A snapshot of bounded query-cache activity and current residency.
public struct QueryCacheStats: Sendable, Equatable {
    /// Number of cache lookups.
    public let requests: UInt64
    /// Number of resident complete-result hits.
    public let hits: UInt64
    /// Number of lookups requiring native traversal.
    public let misses: UInt64
    /// Number of complete results admitted to the cache.
    public let admissions: UInt64
    /// Number of complete results rejected by the admission policy.
    public let rejections: UInt64
    /// Number of resident results evicted to enforce a bound.
    public let evictions: UInt64
    /// Current number of resident result entries.
    public let residentEntries: Int
    /// Current aggregate resident weight in native accounting units.
    public let residentWeight: Int
}

/// An exclusive, synchronization-free bounded memo for complete repeated queries.
///
/// Limits apply independently to traversal and distance-then-term result order.
/// Create one instance per worker for parallel workloads; do not share a cache
/// for concurrent mutation. The underlying transducer retains its dictionary.
public final class QueryCache {
    private var raw: OpaquePointer?

    /// Create a bounded cache over `transducer`.
    ///
    /// Both limits must be nonnegative. A zero limit disables residency while
    /// preserving query correctness. The cache keeps its own transducer retain.
    public init(
        transducer: Transducer,
        maximumEntries: Int = 1024,
        maximumWeight: Int = 64 * 1024 * 1024
    ) throws {
        guard maximumEntries >= 0, maximumWeight >= 0 else {
            throw LiblevenshteinError(status: .invalidArgument, message: "cache limits must be nonnegative")
        }
        var output: OpaquePointer?
        try checked(llev_query_cache_new(
            try transducer.handle(), maximumEntries, maximumWeight, &output
        ))
        raw = output!
    }

    deinit { close() }

    private func handle() throws -> OpaquePointer {
        guard let raw else {
            throw LiblevenshteinError(status: .closed, message: "query cache is closed")
        }
        return raw
    }

    /// Copy aggregate policy counters and current residency.
    public func stats() throws -> QueryCacheStats {
        var value = LlevQueryCacheStats()
        try checked(llev_query_cache_stats(try handle(), &value))
        return QueryCacheStats(
            requests: value.requests, hits: value.hits, misses: value.misses,
            admissions: value.admissions, rejections: value.rejections,
            evictions: value.evictions, residentEntries: value.resident_entries,
            residentWeight: value.resident_weight
        )
    }

    /// Drop resident results while preserving policy counters.
    @discardableResult public func clear() throws -> QueryCache {
        try checked(llev_query_cache_clear(try handle()))
        return self
    }

    /// Reset counters while preserving residency and frequency state.
    @discardableResult public func resetStats() throws -> QueryCache {
        try checked(llev_query_cache_reset_stats(try handle()))
        return self
    }

    /// Query Unicode text, using a resident complete result when available.
    ///
    /// The returned cursor is owned by the caller and retains its own
    /// query-start dictionary revision.
    public func query(
        _ text: String,
        maximumDistance: Int,
        order: QueryOrder = .traversal
    ) throws -> QueryCursor {
        guard maximumDistance >= 0 else {
            throw LiblevenshteinError(status: .invalidArgument, message: "maximumDistance must be nonnegative")
        }
        let bytes = Array(text.utf8)
        var output: OpaquePointer?
        try bytes.withUnsafeBufferPointer { buffer in
            try checked(llev_query_cache_query_utf8(
                try handle(), buffer.baseAddress, buffer.count,
                maximumDistance, order.rawValue, &output
            ))
        }
        return QueryCursor(raw: output!)
    }

    /// Query exact bytes through the bounded cache without Unicode conversion.
    public func query(
        _ bytes: [UInt8],
        maximumDistance: Int,
        order: QueryOrder = .traversal
    ) throws -> QueryCursor {
        guard maximumDistance >= 0 else {
            throw LiblevenshteinError(status: .invalidArgument, message: "maximumDistance must be nonnegative")
        }
        var output: OpaquePointer?
        try bytes.withUnsafeBufferPointer { buffer in
            try checked(llev_query_cache_query_bytes(
                try handle(), buffer.baseAddress, buffer.count,
                maximumDistance, order.rawValue, &output
            ))
        }
        return QueryCursor(raw: output!)
    }

    /// Query unsigned 64-bit tokens through the bounded cache.
    public func query(
        _ tokens: [UInt64],
        maximumDistance: Int,
        order: QueryOrder = .traversal
    ) throws -> QueryCursor {
        guard maximumDistance >= 0 else {
            throw LiblevenshteinError(status: .invalidArgument, message: "maximumDistance must be nonnegative")
        }
        var output: OpaquePointer?
        try tokens.withUnsafeBufferPointer { buffer in
            try checked(llev_query_cache_query_u64(
                try handle(), buffer.baseAddress, buffer.count,
                maximumDistance, order.rawValue, &output
            ))
        }
        return QueryCursor(raw: output!)
    }

    /// Release the cache; cursors already returned by it remain valid.
    public func close() {
        if let raw {
            llev_query_cache_free(raw)
            self.raw = nil
        }
    }
}

/// A compiled, reusable phonetic language for matching or dictionary queries.
///
/// Pattern compilation is separate from traversal. A pattern is a native
/// resource and should be closed explicitly after its final use.
public final class PhoneticPattern: @unchecked Sendable {
    private var raw: OpaquePointer?
    private init(_ raw: OpaquePointer) { self.raw = raw }
    deinit { close() }

    /// Compile the phonetic regular-expression syntax.
    public static func regex(_ source: String) throws -> PhoneticPattern {
        try compile(source, llev_phonetic_pattern_compile_regex)
    }
    /// Compile an import-free LLRE pattern document.
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
        guard let raw else {
            throw LiblevenshteinError(status: .closed, message: "phonetic pattern is closed")
        }
        return raw
    }
    /// Test whether the complete input string is accepted by this pattern.
    public func matches(_ input: String) throws -> Bool {
        var result: UInt8 = 0
        try input.withCString { pointer in
            try checked(llev_phonetic_pattern_matches(
                try handle(), pointer, input.utf8.count, &result
            ))
        }
        return result != 0
    }
    /// Compiled `(states, transitions)` counts of the pattern automaton.
    public func size() throws -> (states: Int, transitions: Int) {
        var states = 0
        var transitions = 0
        try checked(llev_phonetic_pattern_size(try handle(), &states, &transitions))
        return (states, transitions)
    }
    /// Release this compiled pattern; repeated calls are harmless.
    public func close() {
        if let raw {
            llev_phonetic_pattern_free(raw)
            self.raw = nil
        }
    }
}

/// A reusable Unicode phonetic rewrite-rule set.
///
/// Parsing and validation are paid once; each `apply` result is independently
/// owned Swift text.
public final class PhoneticRuleSet: @unchecked Sendable {
    private var raw: OpaquePointer?
    private init(_ raw: OpaquePointer) { self.raw = raw }
    deinit { close() }

    /// Parse an import-free `.llev` rewrite-rule document.
    public static func parse(_ source: String) throws -> PhoneticRuleSet {
        var output: OpaquePointer?
        try source.withCString { pointer in
            try checked(llev_phonetic_rules_parse(pointer, source.utf8.count, &output))
        }
        return PhoneticRuleSet(output!)
    }
    /// Construct a built-in rewrite-rule set.
    public static func builtin(_ kind: PhoneticRuleSetKind) throws -> PhoneticRuleSet {
        var output: OpaquePointer?
        try checked(llev_phonetic_rules_builtin(kind.rawValue, &output))
        return PhoneticRuleSet(output!)
    }
    private func handle() throws -> OpaquePointer {
        guard let raw else {
            throw LiblevenshteinError(status: .closed, message: "phonetic rule set is closed")
        }
        return raw
    }
    /// Number of enabled rewrite rules.
    public func count() throws -> Int {
        var length = 0
        try checked(llev_phonetic_rules_len(try handle(), &length))
        return length
    }
    /// Apply the rewrite rules and return owned UTF-8.
    public func apply(_ input: String) throws -> String {
        var output = LlevOwnedString()
        try input.withCString { pointer in
            try checked(llev_phonetic_rules_apply(try handle(), pointer, input.utf8.count, &output))
        }
        defer { llev_owned_string_free(&output) }
        guard let data = output.data, output.len > 0 else { return "" }
        let bytes = UnsafeRawBufferPointer(start: data, count: output.len)
        return String(decoding: bytes, as: UTF8.self)
    }
    /// Release this rewrite-rule set; repeated calls are harmless.
    public func close() {
        if let raw {
            llev_phonetic_rules_free(raw)
            self.raw = nil
        }
    }
}

/// Standalone Unicode edit-distance functions independent of a dictionary.
///
/// Use a ``Transducer`` when searching a dictionary rather than repeatedly
/// comparing the query to every term yourself.
public enum EditDistance {
    /// Standard Levenshtein insertion, deletion, and substitution distance.
    public static func levenshtein(_ source: String, _ target: String) -> Int {
        source.withCString { sourcePointer in
            target.withCString { targetPointer in
                llev_distance(sourcePointer, source.utf8.count, targetPointer, target.utf8.count)
            }
        }
    }
    /// Optimal-string-alignment distance with adjacent transpositions.
    ///
    /// Unlike unrestricted Damerau–Levenshtein, a substring is not edited
    /// more than once in an alignment.
    public static func damerauOSA(_ source: String, _ target: String) -> Int {
        source.withCString { sourcePointer in
            target.withCString { targetPointer in
                llev_damerau_distance(sourcePointer, source.utf8.count, targetPointer, target.utf8.count)
            }
        }
    }
    /// Unrestricted Damerau–Levenshtein distance with adjacent transpositions.
    public static func damerauLevenshtein(_ source: String, _ target: String) -> Int {
        source.withCString { sourcePointer in
            target.withCString { targetPointer in
                llev_true_damerau_distance(sourcePointer, source.utf8.count, targetPointer, target.utf8.count)
            }
        }
    }
    /// Levenshtein distance if it is at most `threshold`, otherwise `nil`.
    public static func levenshtein(_ source: String, _ target: String, threshold: Int) -> Int? {
        bounded(source, target, threshold, llev_distance_threshold)
    }
    /// Adjacent-transposition (OSA) distance if it is at most `threshold`, otherwise `nil`.
    public static func damerauOSA(_ source: String, _ target: String, threshold: Int) -> Int? {
        bounded(source, target, threshold, llev_damerau_distance_threshold)
    }
    /// Unrestricted Damerau-Levenshtein distance if it is at most `threshold`, otherwise `nil`.
    public static func damerauLevenshtein(_ source: String, _ target: String, threshold: Int) -> Int? {
        bounded(source, target, threshold, llev_true_damerau_distance_threshold)
    }
    private static func bounded(
        _ source: String,
        _ target: String,
        _ threshold: Int,
        _ function: (UnsafePointer<CChar>?, Int, UnsafePointer<CChar>?, Int, Int) -> Int
    ) -> Int? {
        let raw = source.withCString { sourcePointer in
            target.withCString { targetPointer in
                function(sourcePointer, source.utf8.count, targetPointer, target.utf8.count, threshold)
            }
        }
        // The native contract returns usize::MAX (invalid input) or usize::MAX - 1
        // (exceeded the bound); both arrive as negative Ints under the size_t ->
        // Int import, so any negative result means "not within the threshold".
        return raw < 0 ? nil : raw
    }
}
