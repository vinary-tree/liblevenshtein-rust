/** @file
 * @brief Move-only C++23 resource wrappers for liblevenshtein's stable C ABI.
 */
#ifndef LIBLEVENSHTEIN_HPP
#define LIBLEVENSHTEIN_HPP

#include "liblevenshtein.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

/** Idiomatic, exception-safe C++ access to the liblevenshtein resource ABI. */
namespace vinary_tree::liblevenshtein {

/** A typed native failure with a copied thread-local diagnostic. */
class error final : public std::runtime_error {
public:
    /** Copy the current native diagnostic and preserve its stable status.
     * @param status the failing native status returned by the C ABI
     */
    explicit error(LlevStatus status)
        : std::runtime_error(message()), status_(status) {}

    /** Return the stable native failure category.
     * @return the exact status supplied at construction
     */
    [[nodiscard]] LlevStatus status() const noexcept { return status_; }

private:
    [[nodiscard]] static std::string message() {
        const auto* value = llev_last_error_message();
        return value != nullptr ? value : "native operation failed";
    }

    LlevStatus status_;
};

/** Raise error unless a native operation completed successfully.
 * @param status the status returned by the C ABI
 * @throws error when @p status is not LLEV_STATUS_OK
 */
inline void check(LlevStatus status) {
    if (status != LLEV_STATUS_OK) {
        throw error(status);
    }
}

/** Edit-distance automaton used by a transducer. */
enum class algorithm : std::uint32_t {
    standard = LLEV_ALGORITHM_STANDARD, ///< Insert, delete, and substitute.
    transposition = LLEV_ALGORITHM_TRANSPOSITION, ///< Add adjacent transposition.
    merge_and_split = LLEV_ALGORITHM_MERGE_AND_SPLIT, ///< Add merge and split edits.
    damerau_levenshtein = LLEV_ALGORITHM_DAMERAU_LEVENSHTEIN, ///< Unrestricted Damerau-Levenshtein.
};

/** Observable ordering of a lazy result stream. */
enum class query_order : std::uint32_t {
    traversal = LLEV_QUERY_ORDER_TRAVERSAL, ///< Provider traversal order with bounded buffering.
    distance_then_term = LLEV_QUERY_ORDER_DISTANCE_THEN_TERM, ///< Increasing distance, then term.
};

/** @cond INTERNAL */
namespace detail {

struct cursor_state final {
    explicit cursor_state(LlevQueryCursor* cursor) noexcept : value(cursor) {}
    cursor_state(const cursor_state&) = delete;
    cursor_state& operator=(const cursor_state&) = delete;
    ~cursor_state() {
        if (value != nullptr) {
            (void)llev_query_cursor_free(value);
        }
    }

    LlevQueryCursor* value;
};

} // namespace detail
/** @endcond */

/** One move-only lease over a cursor-owned contiguous match batch.
 *
 * Every descriptor and term view returned by this object remains valid only
 * until the batch is destroyed, move-assigned, or moved into another batch.
 * Destruction releases the exact native generation. A default-constructed or
 * moved-from batch is empty and owns no lease.
 */
class batch final {
public:
    /** Construct an empty, non-owning batch. */
    batch() = default;
    batch(const batch&) = delete;
    batch& operator=(const batch&) = delete;
    /** Move a lease without releasing its generation.
     * @param other batch whose lease transfers to this object
     */
    batch(batch&& other) noexcept
        : cursor_(std::move(other.cursor_)),
          view_(std::exchange(other.view_, LlevMatchBatchView{})) {}
    /** Release the current lease, then take another batch's lease.
     * @param other batch whose lease transfers to this object
     * @return this batch
     */
    batch& operator=(batch&& other) noexcept {
        if (this != &other) {
            release();
            cursor_ = std::move(other.cursor_);
            view_ = std::exchange(other.view_, LlevMatchBatchView{});
        }
        return *this;
    }
    /** Release the native batch generation when one is owned. */
    ~batch() { release(); }

    /** Borrow all match descriptors for this batch's lexical lifetime.
     * @return a contiguous descriptor view, empty for an end/moved-from batch
     */
    [[nodiscard]] std::span<const LlevMatch> matches() const noexcept {
        return {view_.matches, view_.len};
    }

    /** Borrow the bytes of a byte-domain or Unicode descriptor.
     * @param item descriptor belonging to this live batch
     * @return a zero-copy byte view valid only for the batch lifetime
     * @throws std::invalid_argument when @p item belongs to the u64 domain
     */
    [[nodiscard]] static std::span<const std::byte> bytes(const LlevMatch& item) {
        if (item.unit_domain == VT_UNIT_DOMAIN_U64) {
            throw std::invalid_argument("u64 match is not a byte term");
        }
        return {static_cast<const std::byte*>(item.term_data), item.byte_len};
    }

    /** Borrow the UTF-8 encoding of a Unicode-scalar descriptor.
     * @param item descriptor belonging to this live batch
     * @return a string view valid only for the batch lifetime
     * @throws std::invalid_argument when @p item is not Unicode text
     */
    [[nodiscard]] static std::string_view utf8(const LlevMatch& item) {
        if (item.unit_domain != VT_UNIT_DOMAIN_UNICODE_SCALAR) {
            throw std::invalid_argument("match is not Unicode text");
        }
        return {static_cast<const char*>(item.term_data), item.byte_len};
    }

    /** Borrow the aligned tokens of a u64-domain descriptor.
     * @param item descriptor belonging to this live batch
     * @return a zero-copy token view valid only for the batch lifetime
     * @throws std::invalid_argument when @p item is not a u64 term
     */
    [[nodiscard]] static std::span<const std::uint64_t> tokens(const LlevMatch& item) {
        if (item.unit_domain != VT_UNIT_DOMAIN_U64) {
            throw std::invalid_argument("match is not a u64 term");
        }
        return {static_cast<const std::uint64_t*>(item.term_data), item.term_len};
    }

private:
    friend class query_cursor;

    batch(std::shared_ptr<detail::cursor_state> cursor,
          LlevMatchBatchView view) noexcept
        : cursor_(std::move(cursor)), view_(view) {}

    void release() noexcept {
        if (cursor_) {
            (void)llev_query_cursor_release_batch(cursor_->value, view_.generation);
            cursor_.reset();
        }
        view_ = {};
    }

    std::shared_ptr<detail::cursor_state> cursor_;
    LlevMatchBatchView view_{};
};

/** Exclusive, move-only traversal over one query-start dictionary snapshot.
 *
 * Different cursors are independent and may run concurrently. One cursor must
 * be used by only one thread at a time. It retains its immutable provider
 * revision and therefore may outlive both the transducer and source dictionary
 * handle from which it was created.
 */
class query_cursor final {
public:
    query_cursor(const query_cursor&) = delete;
    query_cursor& operator=(const query_cursor&) = delete;
    /** Transfer cursor ownership; the source remains destructible and assignable. */
    query_cursor(query_cursor&&) noexcept = default;
    /** Release the current cursor and transfer ownership from another cursor.
     * @param other source cursor whose ownership transfers to this object
     * @return this cursor
     */
    query_cursor& operator=(query_cursor&& other) noexcept = default;
    /** Close the cursor and release its snapshot when no batch lease remains. */
    ~query_cursor() = default;

    /** Borrow the next bounded result batch.
     * @param maximum positive maximum number of matches to borrow
     * @return a live batch, or an empty batch after the stream reaches its end
     * @throws error on native traversal/provider failure
     * @throws std::logic_error when called on a moved-from cursor
     */
    [[nodiscard]] batch next_batch(std::size_t maximum = LLEV_DEFAULT_MATCH_BATCH) {
        if (!state_) {
            throw std::logic_error("query cursor is moved from");
        }
        LlevMatchBatchView view{};
        const auto status = llev_query_cursor_next_batch(state_->value, maximum, &view);
        if (status == LLEV_STATUS_END) {
            return {};
        }
        check(status);
        return batch(state_, view);
    }

private:
    friend class transducer;
    friend class query_cache;

    explicit query_cursor(LlevQueryCursor* value) try
        : state_(std::make_shared<detail::cursor_state>(value)) {}
    catch (...) {
        (void)llev_query_cursor_free(value);
        throw;
    }

    std::shared_ptr<detail::cursor_state> state_;
};

/** Shareable automaton configuration retaining a live dictionary resource.
 *
 * Construction is constant-time with respect to dictionary size. Each query
 * captures an immutable provider revision and returns an independently owned
 * cursor. The wrapper is move-only and releases its retained resource at scope
 * exit.
 */
class transducer final {
public:
    /** Retain a dictionary resource and select an edit-distance algorithm.
     * @param dictionary borrowed resource copied and retained by the native ABI
     * @param selected edit-distance automaton for subsequent queries
     * @throws error when negotiation, validation, or retention fails
     */
    transducer(const VtResource& dictionary,
               algorithm selected = algorithm::standard) {
        check(llev_transducer_new(&dictionary,
                                  static_cast<std::uint32_t>(selected),
                                  &value_));
    }
    transducer(const transducer&) = delete;
    transducer& operator=(const transducer&) = delete;
    /** Transfer ownership of the retained native transducer.
     * @param other source whose native handle transfers to this object
     */
    transducer(transducer&& other) noexcept
        : value_(std::exchange(other.value_, nullptr)) {}
    /** Release the current resource and transfer another transducer's resource.
     * @param other source whose native handle transfers to this object
     * @return this transducer
     */
    transducer& operator=(transducer&& other) noexcept {
        if (this != &other) {
            llev_transducer_free(value_);
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }
    /** Release the retained dictionary resource. Existing cursors stay valid. */
    ~transducer() { llev_transducer_free(value_); }

    /** Start a lazy Unicode query against the revision visible now.
     * @param text valid UTF-8 query text
     * @param maximum_distance inclusive edit-distance bound
     * @param order traversal order or distance-then-term ordering
     * @return an exclusive cursor retaining the captured revision
     * @throws error on domain mismatch, malformed UTF-8, or provider failure
     */
    [[nodiscard]] query_cursor query(
        std::string_view text,
        std::size_t maximum_distance,
        query_order order = query_order::traversal) const {
        LlevQueryCursor* cursor = nullptr;
        check(llev_transducer_query_utf8(value_, text.data(), text.size(),
                                         maximum_distance,
                                         static_cast<std::uint32_t>(order),
                                         &cursor));
        return query_cursor(cursor);
    }

    /** Start a lazy raw-byte query against the revision visible now.
     * @param bytes arbitrary byte sequence
     * @param maximum_distance inclusive edit-distance bound
     * @return an exclusive traversal-order cursor retaining the captured revision
     * @throws error on domain mismatch or provider failure
     */
    [[nodiscard]] query_cursor query(
        std::span<const std::byte> bytes,
        std::size_t maximum_distance,
        query_order order = query_order::traversal) const {
        LlevQueryCursor* cursor = nullptr;
        check(llev_transducer_query_bytes(
            value_, reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size(),
            maximum_distance, static_cast<std::uint32_t>(order), &cursor));
        return query_cursor(cursor);
    }

    /** Start a lazy u64-token query against the revision visible now.
     * @param tokens logical token sequence
     * @param maximum_distance inclusive edit-distance bound
     * @return an exclusive traversal-order cursor retaining the captured revision
     * @throws error on domain mismatch or provider failure
     */
    [[nodiscard]] query_cursor query(
        std::span<const std::uint64_t> tokens,
        std::size_t maximum_distance,
        query_order order = query_order::traversal) const {
        LlevQueryCursor* cursor = nullptr;
        check(llev_transducer_query_u64(value_, tokens.data(), tokens.size(),
                                        maximum_distance,
                                        static_cast<std::uint32_t>(order),
                                        &cursor));
        return query_cursor(cursor);
    }

private:
    friend class query_cache;
    LlevTransducer* value_ = nullptr;
};

/** Exclusive, synchronization-free bounded cache for complete query results.
 *
 * Limits apply independently to traversal and distance-then-term result-order
 * shards. Construct one cache per worker for parallel workloads. Returned
 * cursors own their immutable result and may outlive this object.
 */
class query_cache final {
public:
    /** Retain a transducer and configure hard per-order bounds.
     * @param source live transducer whose provider is retained
     * @param maximum_entries maximum resident results in each order shard
     * @param maximum_weight maximum logical result weight in each order shard
     */
    query_cache(const transducer& source,
                std::size_t maximum_entries = 1024,
                std::size_t maximum_weight = 64 * 1024 * 1024) {
        check(llev_query_cache_new(source.value_, maximum_entries,
                                   maximum_weight, &value_));
    }
    query_cache(const query_cache&) = delete;
    query_cache& operator=(const query_cache&) = delete;
    /** Transfer exclusive cache ownership. */
    query_cache(query_cache&& other) noexcept
        : value_(std::exchange(other.value_, nullptr)) {}
    /** Release current residency and transfer exclusive ownership. */
    query_cache& operator=(query_cache&& other) noexcept {
        if (this != &other) {
            llev_query_cache_free(value_);
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }
    /** Release all resident results and the retained transducer. */
    ~query_cache() { llev_query_cache_free(value_); }

    /** Copy aggregate policy counters and current residency. */
    [[nodiscard]] LlevQueryCacheStats stats() const {
        LlevQueryCacheStats output{};
        check(llev_query_cache_stats(value_, &output));
        return output;
    }

    /** Drop resident results while preserving policy counters. */
    void clear() { check(llev_query_cache_clear(value_)); }
    /** Reset counters while preserving residency and frequency state. */
    void reset_stats() { check(llev_query_cache_reset_stats(value_)); }

    /** Query Unicode text through the bounded complete-result cache. */
    [[nodiscard]] query_cursor query(
        std::string_view text,
        std::size_t maximum_distance,
        query_order order = query_order::traversal) {
        LlevQueryCursor* cursor = nullptr;
        check(llev_query_cache_query_utf8(
            value_, text.data(), text.size(), maximum_distance,
            static_cast<std::uint32_t>(order), &cursor));
        return query_cursor(cursor);
    }

    /** Query an exact byte sequence through the bounded result cache. */
    [[nodiscard]] query_cursor query(
        std::span<const std::byte> bytes,
        std::size_t maximum_distance,
        query_order order = query_order::traversal) {
        LlevQueryCursor* cursor = nullptr;
        check(llev_query_cache_query_bytes(
            value_, reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size(),
            maximum_distance, static_cast<std::uint32_t>(order), &cursor));
        return query_cursor(cursor);
    }

    /** Query exact u64 tokens through the bounded result cache. */
    [[nodiscard]] query_cursor query(
        std::span<const std::uint64_t> tokens,
        std::size_t maximum_distance,
        query_order order = query_order::traversal) {
        LlevQueryCursor* cursor = nullptr;
        check(llev_query_cache_query_u64(
            value_, tokens.data(), tokens.size(), maximum_distance,
            static_cast<std::uint32_t>(order), &cursor));
        return query_cursor(cursor);
    }

private:
    LlevQueryCache* value_ = nullptr;
};

} // namespace vinary_tree::liblevenshtein

#endif /* LIBLEVENSHTEIN_HPP */
