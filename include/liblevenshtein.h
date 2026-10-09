/** @file
 * @brief Stable C API for distance functions and resource-backed fuzzy queries.
 *
 * Every fallible resource operation returns LlevStatus and copies its diagnostic
 * into a thread-local slot. Query cursors capture an immutable dictionary
 * revision and lend bounded batches whose generation must be released exactly.
 * The complete status, ownership, concurrency, and complexity contract is
 * available in the versioned package guide and is summarized per declaration
 * below.
 * Opaque owning handles are valid only until a successful free consumes them.
 * Reuse or double-free afterward is undefined behavior, not a status result;
 * independently retained resources and snapshots keep their own lifetimes.
 */
#ifndef LIBLEVENSHTEIN_H
#define LIBLEVENSHTEIN_H

#include "liblevenshtein_abi.h"

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(LIBLEVENSHTEIN_BUILDING_DLL)
#    define LLEV_API __declspec(dllexport)
#  elif defined(LIBLEVENSHTEIN_USING_DLL)
#    define LLEV_API __declspec(dllimport)
#  else
#    define LLEV_API
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define LLEV_API __attribute__((visibility("default")))
#else
#  define LLEV_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Return the binary ABI generation implemented by the loaded library.
 * @return LLEV_ABI_VERSION; this total constant-time operation cannot fail
 */
LLEV_API uint32_t llev_abi_version(void);
/** Return the additive API revision within the current ABI generation.
 * @return LLEV_API_REVISION; this total constant-time operation cannot fail
 */
LLEV_API uint32_t llev_api_revision(void);
/** Report the capabilities compiled into this library.
 * @return a bitset of LLEV_BUILD_FEATURE_CORE and optional feature bits
 */
LLEV_API uint64_t llev_build_features(void);
/** Borrow the current thread's native diagnostic.
 * @return a never-NULL, library-owned UTF-8 string valid until the next llev_*
 * call on this thread; callers must copy data they need to retain and never free it
 */
LLEV_API const char* llev_last_error_message(void);

/** Compute Unicode-scalar Levenshtein distance.
 * @param source UTF-8 bytes, or NULL only when source_len is zero
 * @param source_len source byte length
 * @param target UTF-8 bytes, or NULL only when target_len is zero
 * @param target_len target byte length
 * @return exact distance, or SIZE_MAX for a NULL/invalid-UTF-8 input
 */
LLEV_API size_t llev_distance(const char* source, size_t source_len,
                              const char* target, size_t target_len);
/** Compute thresholded Unicode-scalar Levenshtein distance.
 * @param source UTF-8 bytes, or NULL only when source_len is zero
 * @param source_len source byte length
 * @param target UTF-8 bytes, or NULL only when target_len is zero
 * @param target_len target byte length
 * @param threshold inclusive distance bound
 * @return exact distance, SIZE_MAX for invalid input, or SIZE_MAX-1 above the bound
 */
LLEV_API size_t llev_distance_threshold(const char* source, size_t source_len,
                                        const char* target, size_t target_len,
                                        size_t threshold);
/** Compute Unicode-scalar optimal-string-alignment distance.
 * @param source UTF-8 bytes, or NULL only when source_len is zero
 * @param source_len source byte length
 * @param target UTF-8 bytes, or NULL only when target_len is zero
 * @param target_len target byte length
 * @return exact restricted-Damerau distance, or SIZE_MAX for invalid input
 */
LLEV_API size_t llev_damerau_distance(const char* source, size_t source_len,
                                      const char* target, size_t target_len);
/** Compute thresholded Unicode-scalar optimal-string-alignment distance.
 * @param source UTF-8 bytes, or NULL only when source_len is zero
 * @param source_len source byte length
 * @param target UTF-8 bytes, or NULL only when target_len is zero
 * @param target_len target byte length
 * @param threshold inclusive distance bound
 * @return exact distance, SIZE_MAX for invalid input, or SIZE_MAX-1 above the bound
 */
LLEV_API size_t llev_damerau_distance_threshold(const char* source,
                                                size_t source_len,
                                                const char* target,
                                                size_t target_len,
                                                size_t threshold);
/** Compute unrestricted Unicode-scalar Damerau-Levenshtein distance.
 * @param source UTF-8 bytes, or NULL only when source_len is zero
 * @param source_len source byte length
 * @param target UTF-8 bytes, or NULL only when target_len is zero
 * @param target_len target byte length
 * @return exact true-Damerau distance, or SIZE_MAX for invalid input
 */
LLEV_API size_t llev_true_damerau_distance(const char* source,
                                           size_t source_len,
                                           const char* target,
                                           size_t target_len);
/** Compute thresholded unrestricted Unicode-scalar Damerau-Levenshtein distance.
 * @param source UTF-8 bytes, or NULL only when source_len is zero
 * @param source_len source byte length
 * @param target UTF-8 bytes, or NULL only when target_len is zero
 * @param target_len target byte length
 * @param threshold inclusive distance bound
 * @return exact distance, SIZE_MAX for invalid input, or SIZE_MAX-1 above the bound
 */
LLEV_API size_t llev_true_damerau_distance_threshold(const char* source,
                                                     size_t source_len,
                                                     const char* target,
                                                     size_t target_len,
                                                     size_t threshold);

/** Compute Unicode-scalar merge-and-split distance.
 *
 * Merge and split each cost one: a merge consumes two source scalars and one
 * target scalar, while a split consumes one source scalar and two target
 * scalars. Insert, delete, and substitute remain available at unit cost.
 * @return exact distance, or SIZE_MAX for a NULL/invalid-UTF-8 input
 */
LLEV_API size_t llev_merge_and_split_distance(const char* source,
                                              size_t source_len,
                                              const char* target,
                                              size_t target_len);
/** Compute thresholded Unicode-scalar merge-and-split distance.
 * @return exact distance, SIZE_MAX for invalid input, or SIZE_MAX-1 above the bound
 */
LLEV_API size_t llev_merge_and_split_distance_threshold(const char* source,
                                                        size_t source_len,
                                                        const char* target,
                                                        size_t target_len,
                                                        size_t threshold);

/** @name Hamming, insertion/deletion, and affine-gap distances
 *
 * UTF-8 entry points compare Unicode scalar values, without normalizing
 * canonically equivalent spellings. Byte and uint64_t entry points compare
 * their native units without transcoding. NULL is valid only with zero length.
 *
 * All variants return SIZE_MAX for invalid input. A thresholded defined
 * distance above its inclusive bound returns SIZE_MAX-1. Hamming distance is
 * undefined for unequal unit counts and returns SIZE_MAX-2 in either variant.
 * Affine gap returns SIZE_MAX-2 when arithmetic overflows or the finite
 * result enters the reserved sentinel range. These cases never become a
 * fabricated finite distance.
 *
 * Affine costs are nonnegative integers sharing the same caller-selected
 * scale. A gap of k units costs gap_open + k*gap_extend; substitutions cost
 * substitution. The threshold uses that same scale. Bounded affine calls
 * currently compute the exact Gotoh result before comparison, whereas bounded
 * indel calls use an affordable diagonal band.
 * @{ */
LLEV_API size_t llev_hamming_distance(const char* source, size_t source_len,
                                      const char* target, size_t target_len);
LLEV_API size_t llev_hamming_distance_threshold(const char* source,
                                                size_t source_len,
                                                const char* target,
                                                size_t target_len,
                                                size_t threshold);
LLEV_API size_t llev_indel_distance(const char* source, size_t source_len,
                                    const char* target, size_t target_len);
LLEV_API size_t llev_indel_distance_threshold(const char* source,
                                              size_t source_len,
                                              const char* target,
                                              size_t target_len,
                                              size_t threshold);
LLEV_API size_t llev_affine_gap_distance(const char* source, size_t source_len,
                                         const char* target, size_t target_len,
                                         size_t gap_open, size_t gap_extend,
                                         size_t substitution);
LLEV_API size_t llev_affine_gap_distance_threshold(
    const char* source, size_t source_len, const char* target,
    size_t target_len, size_t gap_open, size_t gap_extend,
    size_t substitution, size_t threshold);

LLEV_API size_t llev_hamming_distance_bytes(const uint8_t* source,
                                            size_t source_len,
                                            const uint8_t* target,
                                            size_t target_len);
LLEV_API size_t llev_hamming_distance_bytes_threshold(
    const uint8_t* source, size_t source_len, const uint8_t* target,
    size_t target_len, size_t threshold);
LLEV_API size_t llev_hamming_distance_u64(const uint64_t* source,
                                          size_t source_len,
                                          const uint64_t* target,
                                          size_t target_len);
LLEV_API size_t llev_hamming_distance_u64_threshold(
    const uint64_t* source, size_t source_len, const uint64_t* target,
    size_t target_len, size_t threshold);

LLEV_API size_t llev_indel_distance_bytes(const uint8_t* source,
                                          size_t source_len,
                                          const uint8_t* target,
                                          size_t target_len);
LLEV_API size_t llev_indel_distance_bytes_threshold(
    const uint8_t* source, size_t source_len, const uint8_t* target,
    size_t target_len, size_t threshold);
LLEV_API size_t llev_indel_distance_u64(const uint64_t* source,
                                        size_t source_len,
                                        const uint64_t* target,
                                        size_t target_len);
LLEV_API size_t llev_indel_distance_u64_threshold(
    const uint64_t* source, size_t source_len, const uint64_t* target,
    size_t target_len, size_t threshold);

LLEV_API size_t llev_affine_gap_distance_bytes(
    const uint8_t* source, size_t source_len, const uint8_t* target,
    size_t target_len, size_t gap_open, size_t gap_extend, size_t substitution);
LLEV_API size_t llev_affine_gap_distance_bytes_threshold(
    const uint8_t* source, size_t source_len, const uint8_t* target,
    size_t target_len, size_t gap_open, size_t gap_extend,
    size_t substitution, size_t threshold);
LLEV_API size_t llev_affine_gap_distance_u64(
    const uint64_t* source, size_t source_len, const uint64_t* target,
    size_t target_len, size_t gap_open, size_t gap_extend, size_t substitution);
LLEV_API size_t llev_affine_gap_distance_u64_threshold(
    const uint64_t* source, size_t source_len, const uint64_t* target,
    size_t target_len, size_t gap_open, size_t gap_extend,
    size_t substitution, size_t threshold);
/** @} */

/** @name Domain-explicit standalone distance functions
 *
 * Byte functions accept arbitrary binary data and never interpret UTF-8.
 * Token functions compare aligned uint64_t application tokens by value.
 * For every function, NULL is permitted only when the corresponding length is
 * zero. Exact functions return SIZE_MAX for invalid input. Thresholded
 * functions additionally return SIZE_MAX-1 when the exact distance exceeds
 * the inclusive threshold.
 * @{ */
LLEV_API size_t llev_distance_bytes(const uint8_t* source, size_t source_len,
                                    const uint8_t* target, size_t target_len);
LLEV_API size_t llev_distance_bytes_threshold(const uint8_t* source,
                                              size_t source_len,
                                              const uint8_t* target,
                                              size_t target_len,
                                              size_t threshold);
LLEV_API size_t llev_distance_u64(const uint64_t* source, size_t source_len,
                                  const uint64_t* target, size_t target_len);
LLEV_API size_t llev_distance_u64_threshold(const uint64_t* source,
                                            size_t source_len,
                                            const uint64_t* target,
                                            size_t target_len,
                                            size_t threshold);

LLEV_API size_t llev_damerau_distance_bytes(const uint8_t* source,
                                            size_t source_len,
                                            const uint8_t* target,
                                            size_t target_len);
LLEV_API size_t llev_damerau_distance_bytes_threshold(const uint8_t* source,
                                                      size_t source_len,
                                                      const uint8_t* target,
                                                      size_t target_len,
                                                      size_t threshold);
LLEV_API size_t llev_damerau_distance_u64(const uint64_t* source,
                                          size_t source_len,
                                          const uint64_t* target,
                                          size_t target_len);
LLEV_API size_t llev_damerau_distance_u64_threshold(const uint64_t* source,
                                                    size_t source_len,
                                                    const uint64_t* target,
                                                    size_t target_len,
                                                    size_t threshold);

LLEV_API size_t llev_true_damerau_distance_bytes(const uint8_t* source,
                                                 size_t source_len,
                                                 const uint8_t* target,
                                                 size_t target_len);
LLEV_API size_t llev_true_damerau_distance_bytes_threshold(
    const uint8_t* source, size_t source_len, const uint8_t* target,
    size_t target_len, size_t threshold);
LLEV_API size_t llev_true_damerau_distance_u64(const uint64_t* source,
                                               size_t source_len,
                                               const uint64_t* target,
                                               size_t target_len);
LLEV_API size_t llev_true_damerau_distance_u64_threshold(
    const uint64_t* source, size_t source_len, const uint64_t* target,
    size_t target_len, size_t threshold);

LLEV_API size_t llev_merge_and_split_distance_bytes(const uint8_t* source,
                                                    size_t source_len,
                                                    const uint8_t* target,
                                                    size_t target_len);
LLEV_API size_t llev_merge_and_split_distance_bytes_threshold(
    const uint8_t* source, size_t source_len, const uint8_t* target,
    size_t target_len, size_t threshold);
LLEV_API size_t llev_merge_and_split_distance_u64(const uint64_t* source,
                                                  size_t source_len,
                                                  const uint64_t* target,
                                                  size_t target_len);
LLEV_API size_t llev_merge_and_split_distance_u64_threshold(
    const uint64_t* source, size_t source_len, const uint64_t* target,
    size_t target_len, size_t threshold);
/** @} */

/** Release a NUL-terminated string allocated by llev_string_dup.
 * @param value owned string to consume; NULL is a no-op
 */
LLEV_API void llev_string_free(char* value);
/** Release an owned array and each non-NULL string it contains.
 * @param values owned array to consume; NULL is a no-op
 * @param len number of string slots in values
 */
LLEV_API void llev_string_array_free(char** values, size_t len);
/** Duplicate one valid NUL-terminated UTF-8 string in the library allocator.
 * @param value borrowed string; NULL, embedded NUL, and invalid UTF-8 are rejected
 * @return a caller-owned string for llev_string_free, or NULL on failure
 */
LLEV_API char* llev_string_dup(const char* value);

/**
 * Retain a live dictionary resource and construct an automaton configuration.
 * Dictionary construction and CRUD are intentionally supplied by
 * libdictenstein, not by this library.
 * @param dictionary borrowed resource copied and retained on success
 * @param algorithm one published LlevAlgorithm numeric value
 * @param out_transducer receives one caller-owned handle on success
 * @return OK, NULL_POINTER, INVALID_ARGUMENT, UNSUPPORTED, a mapped provider
 * status, PROVIDER_ERROR for malformed negotiation, or PANIC
 */
LLEV_API LlevStatus llev_transducer_new(const VtResource* dictionary,
                                        uint32_t algorithm,
                                        LlevTransducer** out_transducer);
/**
 * Capture one immutable revision for a read-only query batch. The returned
 * transducer does not observe later dictionary mutations and shares validated
 * provider-node data across its query cursors.
 * @param transducer live source configuration
 * @param out_transducer receives a caller-owned immutable configuration
 * @return OK, a mapped provider status, PROVIDER_ERROR, or PANIC
 */
LLEV_API LlevStatus llev_transducer_snapshot(
    const LlevTransducer* transducer,
    LlevTransducer** out_transducer);
/** Release a transducer's dictionary retain without invalidating its cursors.
 * @param transducer owned handle to consume; NULL is a no-op
 */
LLEV_API void llev_transducer_free(LlevTransducer* transducer);
/** Read the query-unit domain accepted by this transducer.
 * @param transducer live configuration
 * @param out_domain receives BYTE, UNICODE_SCALAR, or U64 on success
 * @return OK, NULL_POINTER, or PANIC
 */
LLEV_API LlevStatus llev_transducer_unit_domain(
    const LlevTransducer* transducer,
    VtUnitDomain* out_domain);

/** Create an opt-in bounded complete-result cache retaining the transducer.
 *
 * The cache uses TinyLFU approximate-frequency admission and SIEVE eviction.
 * Approximation changes residency only: every miss computes the exact result.
 * Limits are hard bounds applied independently to traversal-order and
 * distance-then-term shards. The handle has no internal lock and is exclusive;
 * shard one cache per worker for parallel workloads.
 *
 * @param transducer borrowed configuration retained by the cache
 * @param max_entries_per_order hard resident-entry bound for each order shard;
 * zero disables admission while preserving exact computation
 * @param max_weight_per_order hard logical-byte bound for each order shard;
 * zero disables admission while preserving exact computation
 * @param out_cache receives one caller-owned exclusive cache handle
 * @return OK, NULL_POINTER, or PANIC
 */
LLEV_API LlevStatus llev_query_cache_new(
    const LlevTransducer* transducer,
    size_t max_entries_per_order,
    size_t max_weight_per_order,
    LlevQueryCache** out_cache);
/** Drop every resident result while retaining the source transducer and counters.
 * @param cache live, exclusively borrowed cache
 * @return OK, NULL_POINTER, or PANIC
 */
LLEV_API LlevStatus llev_query_cache_clear(LlevQueryCache* cache);
/** Reset policy counters without changing residency or frequency estimates.
 * @param cache live, exclusively borrowed cache
 * @return OK, NULL_POINTER, or PANIC
 */
LLEV_API LlevStatus llev_query_cache_reset_stats(LlevQueryCache* cache);
/** Copy aggregate counters and current residency without changing policy state.
 * @param cache live cache that is not concurrently mutated
 * @param out_stats receives counters across both result-order shards
 * @return OK, NULL_POINTER, or PANIC
 */
LLEV_API LlevStatus llev_query_cache_stats(
    const LlevQueryCache* cache,
    LlevQueryCacheStats* out_stats);
/** Release the cache and its resident results; existing cursors remain valid.
 * @param cache owned handle to consume; NULL is a no-op
 */
LLEV_API void llev_query_cache_free(LlevQueryCache* cache);

/** Capture the provider revision now and start a lazy Unicode query.
 * @param transducer live configuration over a Unicode-scalar dictionary
 * @param query UTF-8 bytes, or NULL only when query_len is zero
 * @param query_len query byte length
 * @param max_distance inclusive edit-distance bound
 * @param order one published LlevQueryOrder numeric value
 * @param out_cursor receives an exclusive caller-owned cursor on success
 * @return OK, NULL_POINTER, INVALID_UTF8, INVALID_ARGUMENT, DOMAIN_MISMATCH,
 * UNSUPPORTED, a mapped provider status, PROVIDER_ERROR, or PANIC
 */
LLEV_API LlevStatus llev_transducer_query_utf8(
    const LlevTransducer* transducer,
    const char* query,
    size_t query_len,
    size_t max_distance,
    uint32_t order,
    LlevQueryCursor** out_cursor);

/** Capture a Unicode revision and filter values before constructing terms.
 * The callback and context remain borrowed until the cursor is freed. The
 * callback executes synchronously on the thread advancing the cursor and
 * must not unwind across the C boundary. Results retain traversal order and
 * their optional provider IDs.
 */
LLEV_API LlevStatus llev_transducer_query_filtered_utf8(
    const LlevTransducer* transducer,
    const char* query,
    size_t query_len,
    size_t max_distance,
    LlevValueFilterCallback callback,
    void* context,
    LlevQueryCursor** out_cursor);

/** Capture the provider revision now and start a lazy raw-byte query.
 * @param transducer live configuration over a byte dictionary
 * @param query arbitrary bytes, or NULL only when query_len is zero
 * @param query_len query byte length
 * @param max_distance inclusive edit-distance bound
 * @param order must be LLEV_QUERY_ORDER_TRAVERSAL
 * @param out_cursor receives an exclusive caller-owned cursor on success
 * @return OK, NULL_POINTER, INVALID_ARGUMENT, DOMAIN_MISMATCH, UNSUPPORTED, a
 * mapped provider status, PROVIDER_ERROR, or PANIC
 */
LLEV_API LlevStatus llev_transducer_query_bytes(
    const LlevTransducer* transducer,
    const uint8_t* query,
    size_t query_len,
    size_t max_distance,
    uint32_t order,
    LlevQueryCursor** out_cursor);

/** Capture the provider revision now and start a lazy u64-token query.
 * @param transducer live configuration over a u64-token dictionary
 * @param query aligned tokens, or NULL only when query_len is zero
 * @param query_len number of u64 tokens
 * @param max_distance inclusive edit-distance bound
 * @param order must be LLEV_QUERY_ORDER_TRAVERSAL
 * @param out_cursor receives an exclusive caller-owned cursor on success
 * @return OK, NULL_POINTER, INVALID_ARGUMENT, DOMAIN_MISMATCH, UNSUPPORTED, a
 * mapped provider status, PROVIDER_ERROR, or PANIC
 */
LLEV_API LlevStatus llev_transducer_query_u64(
    const LlevTransducer* transducer,
    const uint64_t* query,
    size_t query_len,
    size_t max_distance,
    uint32_t order,
    LlevQueryCursor** out_cursor);

/** Validate decimal affine costs and report the exact selected denominator.
 * A zero scale_denominator derives the least exact decimal scale; a nonzero
 * denominator must represent all three weights exactly.
 */
LLEV_API LlevStatus llev_affine_costs_validate(
    const LlevAffineCosts* costs, uint32_t* out_denominator);

/** Fetch native weighted-operation preset 0 standard, 1 typo, or 2 OCR. */
LLEV_API LlevStatus llev_operation_costs_preset(
    uint32_t preset, LlevOperationCostsF64* out_costs);
/** Validate finite nonnegative costs with zero match cost. */
LLEV_API LlevStatus llev_operation_costs_validate(
    const LlevOperationCostsF64* costs);

/** Capture one provider revision and query with exact affine-gap costs.
 * `unit_domain` selects Unicode UTF-8, raw bytes, or aligned u64 tokens.
 * `max_cost` must be exactly representable at the selected scale. Results
 * expose both an f64 presentation cost and the exact scaled numerator.
 */
LLEV_API LlevStatus llev_transducer_query_affine(
    const LlevTransducer* transducer, uint32_t unit_domain,
    const void* query, size_t query_len, double max_cost,
    const LlevAffineCosts* costs, LlevCostCursor** out_cursor);

/** Capture one provider revision and query with weighted edit operations.
 * The transducer's standard, transposition, or merge/split algorithm selects
 * the native edit repertoire. Non-finite costs and negative budgets fail.
 */
LLEV_API LlevStatus llev_transducer_query_weighted(
    const LlevTransducer* transducer, uint32_t unit_domain,
    const void* query, size_t query_len, double max_cost,
    const LlevOperationCostsF64* costs, LlevCostCursor** out_cursor);

/** Borrow one bounded, generation-checked cost-result batch. */
LLEV_API LlevStatus llev_cost_cursor_next_batch(
    LlevCostCursor* cursor, size_t maximum, LlevCostBatchView* out_batch);
/** Release exactly the live cost-result batch generation. */
LLEV_API LlevStatus llev_cost_cursor_release_batch(
    LlevCostCursor* cursor, uint64_t generation);
/** Consume bounded cost-result batches on the caller thread. */
LLEV_API LlevStatus llev_cost_cursor_reduce(
    LlevCostCursor* cursor, size_t batch_size, LlevCostBatchReducer reducer,
    void* context, size_t* out_count);
/** Free a cost cursor; refuses a live batch lease. */
LLEV_API LlevStatus llev_cost_cursor_free(LlevCostCursor* cursor);

/** Capture a Unicode revision and run a lazy context-dependent edit query.
 * The callback and context remain borrowed until the cursor is freed. The
 * caller must keep them live and must not unwind across the C boundary.
 * `minimum_nonzero_cost` is a strictly positive finite lower bound for all
 * nonzero costs returned by the callback; false bounds invalidate pruning.
 * Callback operation codes are declared by LlevContextualCostCallback.
 * @return OK, INVALID_ARGUMENT for invalid bounds, or a provider error
 */
LLEV_API LlevStatus llev_transducer_query_contextual_utf8(
    const LlevTransducer* transducer,
    const char* query,
    size_t query_len,
    double max_cost,
    double minimum_nonzero_cost,
    LlevContextualCostCallback callback,
    void* context,
    LlevSpecializedCursor** out_cursor);

/** Capture a Unicode revision and run a lazy balanced prefix-pruned DFS.
 * The callback and context remain borrowed until the cursor is freed. The
 * caller must keep them live and must not unwind across the C boundary.
 * Enter and leave calls are balanced, including rejected subtrees and an early
 * cursor close. Result order is dictionary DFS order.
 */
LLEV_API LlevStatus llev_transducer_query_pruned_utf8(
    const LlevTransducer* transducer,
    const char* query,
    size_t query_len,
    size_t max_distance,
    LlevPrefixCallback callback,
    void* context,
    LlevSpecializedCursor** out_cursor);

/** Borrow a bounded generation-checked specialized match batch.
 * @return OK, END, BATCH_IN_USE, INVALID_ARGUMENT, or a provider error
 */
LLEV_API LlevStatus llev_specialized_cursor_next_batch(
    LlevSpecializedCursor* cursor,
    size_t maximum,
    LlevSpecializedBatchView* out_batch);
/** Release the exact live specialized batch generation. */
LLEV_API LlevStatus llev_specialized_cursor_release_batch(
    LlevSpecializedCursor* cursor,
    uint64_t generation);
/** Reduce bounded borrowed specialized batches on the caller's thread. */
LLEV_API LlevStatus llev_specialized_cursor_reduce(
    LlevSpecializedCursor* cursor,
    size_t batch_size,
    LlevSpecializedBatchReducer reducer,
    void* context,
    size_t* out_count);
/** Close a specialized cursor; refuses a live batch lease. */
LLEV_API LlevStatus llev_specialized_cursor_free(
    LlevSpecializedCursor* cursor);

/** Query Unicode scalars through a bounded complete-result cache.
 *
 * A miss captures one immutable dictionary revision and materializes the exact
 * result before admission. A hit returns an independent cursor over shared
 * immutable matches. The provider must publish snapshot identity so mutations
 * invalidate stale residency exactly.
 *
 * @param cache live, exclusively borrowed Unicode cache
 * @param query UTF-8 bytes, or NULL only when query_len is zero
 * @param query_len query byte length
 * @param max_distance inclusive edit-distance bound
 * @param order one published LlevQueryOrder numeric value
 * @param out_cursor receives an exclusive caller-owned cursor on success
 * @return OK, NULL_POINTER, INVALID_UTF8, INVALID_ARGUMENT, DOMAIN_MISMATCH,
 * UNSUPPORTED when snapshot identity is absent, a provider status, or PANIC
 */
LLEV_API LlevStatus llev_query_cache_query_utf8(
    LlevQueryCache* cache,
    const char* query,
    size_t query_len,
    size_t max_distance,
    uint32_t order,
    LlevQueryCursor** out_cursor);
/** Query raw bytes through a bounded complete-result cache.
 * @param cache live, exclusively borrowed byte-domain cache
 * @param query arbitrary bytes, or NULL only when query_len is zero
 * @param query_len query byte length
 * @param max_distance inclusive edit-distance bound
 * @param order must be LLEV_QUERY_ORDER_TRAVERSAL
 * @param out_cursor receives an exclusive caller-owned cursor on success
 * @return OK or the corresponding pointer/domain/order/provider/cache failure
 */
LLEV_API LlevStatus llev_query_cache_query_bytes(
    LlevQueryCache* cache,
    const uint8_t* query,
    size_t query_len,
    size_t max_distance,
    uint32_t order,
    LlevQueryCursor** out_cursor);
/** Query u64 tokens through a bounded complete-result cache.
 * @param cache live, exclusively borrowed u64-domain cache
 * @param query aligned tokens, or NULL only when query_len is zero
 * @param query_len number of u64 tokens
 * @param max_distance inclusive edit-distance bound
 * @param order must be LLEV_QUERY_ORDER_TRAVERSAL
 * @param out_cursor receives an exclusive caller-owned cursor on success
 * @return OK or the corresponding pointer/domain/order/provider/cache failure
 */
LLEV_API LlevStatus llev_query_cache_query_u64(
    LlevQueryCache* cache,
    const uint64_t* query,
    size_t query_len,
    size_t max_distance,
    uint32_t order,
    LlevQueryCursor** out_cursor);

/**
 * Borrow at most max_matches descriptors backed by cursor-owned contiguous
 * arenas. The returned generation must be released before advancing or closing
 * the cursor.
 * @param cursor live exclusive cursor with no outstanding lease
 * @param max_matches positive maximum number of descriptors to borrow
 * @param out_batch always zeroed first, then populated only on OK
 * @return OK, END, BATCH_IN_USE, INVALID_ARGUMENT, NULL_POINTER, a mapped
 * traversal/provider status, or PANIC
 */
LLEV_API LlevStatus llev_query_cursor_next_batch(
    LlevQueryCursor* cursor,
    size_t max_matches,
    LlevMatchBatchView* out_batch);
/** Settle the exact live batch generation and invalidate all its borrowed views.
 * @param cursor cursor that owns the lease
 * @param generation nonzero generation returned in the live batch
 * @return OK, INVALID_ARGUMENT for a stale/missing generation, NULL_POINTER, or PANIC
 */
LLEV_API LlevStatus llev_query_cursor_release_batch(
    LlevQueryCursor* cursor,
    uint64_t generation);

/** Consume the remaining cursor with one callback per reusable batch.
 * @param cursor live exclusive cursor with no outstanding lease
 * @param batch_size positive maximum callback batch size
 * @param reducer callback invoked on the calling thread with lexical borrows
 * @param context opaque value forwarded unchanged to reducer
 * @param out_count receives descriptors delivered when the operation succeeds
 * @return OK for completion or reducer END, BATCH_IN_USE, NULL_POINTER,
 * INVALID_ARGUMENT, a traversal failure, the reducer's abort status, or PANIC
 */
LLEV_API LlevStatus llev_query_cursor_reduce(
    LlevQueryCursor* cursor,
    size_t batch_size,
    LlevBatchReducer reducer,
    void* context,
    size_t* out_count);

/** Close a cursor, refusing to free storage while a batch lease is live.
 * @param cursor owned cursor to consume; NULL is a successful no-op
 * @return OK when consumed, BATCH_IN_USE while a lease exists, or PANIC
 */
LLEV_API LlevStatus llev_query_cursor_free(LlevQueryCursor* cursor);

/** @name Standalone generalized and universal automata
 *
 * These handles execute the native online automata without a dictionary.
 * Constructors copy all operation and substitution-policy data; borrowed
 * descriptors need remain valid only for the constructor call. A NULL limits
 * pointer selects the documented native defaults. Complete evaluation and
 * online advancement use the same native transition kernels.
 * @{ */

/** Construct a generalized Unicode automaton from runtime edit operations.
 * @param max_distance inclusive integral cost budget before fixed-point scaling
 * @param operations non-empty borrowed operation descriptors
 * @param operation_count number of descriptors, subject to native hard limits
 * @param out_automaton receives a caller-owned immutable handle
 * @return OK, NULL_POINTER, INVALID_UTF8, INVALID_ARGUMENT, LIMIT_EXCEEDED, or PANIC
 */
LLEV_API LlevStatus llev_generalized_automaton_new(
    uint8_t max_distance,
    const LlevGeneralizedOperation* operations,
    size_t operation_count,
    LlevGeneralizedAutomaton** out_automaton);
/** Release a generalized automaton; NULL is a no-op. */
LLEV_API void llev_generalized_automaton_free(LlevGeneralizedAutomaton* automaton);
/** Evaluate one complete UTF-8 source/target pair with exact scaled costs.
 * @param configured_limits optional hard ceilings; NULL selects defaults
 * @param out_observation receives the committed final observation
 * @return OK or the corresponding pointer/UTF-8/configuration/limit failure
 */
LLEV_API LlevStatus llev_generalized_automaton_evaluate_utf8(
    const LlevGeneralizedAutomaton* automaton,
    const char* source,
    size_t source_len,
    const char* target,
    size_t target_len,
    const LlevAutomatonLimits* configured_limits,
    LlevGeneralizedObservation* out_observation);
/** Bind a generalized automaton to one UTF-8 source for prefix processing. */
LLEV_API LlevStatus llev_generalized_online_new_utf8(
    const LlevGeneralizedAutomaton* automaton,
    const char* source,
    size_t source_len,
    const LlevAutomatonLimits* configured_limits,
    LlevGeneralizedOnlineAutomaton** out_online);
/** Commit one Unicode scalar to a generalized online state transactionally.
 *
 * `current_row_nonempty == 0` is not permanent death: a multi-target-unit
 * operation may connect an older retained row to a later generation.
 */
LLEV_API LlevStatus llev_generalized_online_advance(
    LlevGeneralizedOnlineAutomaton* online,
    uint32_t scalar,
    LlevGeneralizedObservation* out_observation);
/** Observe a generalized online state without advancing it. */
LLEV_API LlevStatus llev_generalized_online_observation(
    const LlevGeneralizedOnlineAutomaton* online,
    LlevGeneralizedObservation* out_observation);
/** Release a generalized online state; NULL is a no-op. */
LLEV_API void llev_generalized_online_free(LlevGeneralizedOnlineAutomaton* online);

/** Construct a universal automaton with an optional directional zero-cost policy.
 *
 * Set policy_unit_domain to zero and equivalence_count to zero for the
 * specialized unrestricted policy. Otherwise it must be one VtUnitDomain and
 * every pair is interpreted as dictionary/source to query/target. The policy's
 * domain subsequently fixes the domain accepted by evaluate and online_new.
 * @return OK, NULL_POINTER, INVALID_ARGUMENT, LIMIT_EXCEEDED, or PANIC
 */
LLEV_API LlevStatus llev_universal_automaton_new(
    uint8_t max_distance,
    uint32_t variant,
    uint32_t policy_unit_domain,
    const LlevUniversalEquivalence* equivalences,
    size_t equivalence_count,
    LlevUniversalAutomaton** out_automaton);
/** Release a universal automaton; NULL is a no-op. */
LLEV_API void llev_universal_automaton_free(LlevUniversalAutomaton* automaton);
/** Evaluate one complete pair in the selected byte, Unicode, or u64 domain.
 *
 * Unicode buffers use UTF-8 byte lengths; byte buffers use byte lengths; u64
 * buffers must be aligned and lengths count tokens.
 */
LLEV_API LlevStatus llev_universal_automaton_evaluate(
    const LlevUniversalAutomaton* automaton,
    uint32_t unit_domain,
    const void* source_data,
    size_t source_len,
    const void* target_data,
    size_t target_len,
    const LlevAutomatonLimits* configured_limits,
    LlevUniversalObservation* out_observation);
/** Bind a universal automaton to one source for domain-native prefix processing. */
LLEV_API LlevStatus llev_universal_online_new(
    const LlevUniversalAutomaton* automaton,
    uint32_t unit_domain,
    const void* source_data,
    size_t source_len,
    const LlevAutomatonLimits* configured_limits,
    LlevUniversalOnlineAutomaton** out_online);
/** Commit one scalar, byte, or u64 token to a universal online state.
 *
 * Unicode scalars and bytes are range-checked before mutation. Once `alive`
 * becomes zero, the universal frontier is permanently dead.
 */
LLEV_API LlevStatus llev_universal_online_advance(
    LlevUniversalOnlineAutomaton* online,
    uint64_t unit,
    LlevUniversalObservation* out_observation);
/** Observe a universal online state without advancing it. */
LLEV_API LlevStatus llev_universal_online_observation(
    const LlevUniversalOnlineAutomaton* online,
    LlevUniversalObservation* out_observation);
/** Release a universal online state; NULL is a no-op. */
LLEV_API void llev_universal_online_free(LlevUniversalOnlineAutomaton* online);

/** @} */

/** Borrowed UTF-8 view; empty strings may use {NULL, 0}. */
typedef struct LlevUtf8View {
    const char* data;
    size_t len;
} LlevUtf8View;

/** Owned arbitrary AOT bytes; not text and not NUL-terminated. */
typedef struct LlevOwnedBytes {
    uint8_t* data;
    size_t len;
} LlevOwnedBytes;

/** Compile an import-free Unicode phonetic regular expression.
 * @param source UTF-8 expression bytes
 * @param source_len expression byte length
 * @param out_pattern receives a caller-owned immutable pattern
 * @return OK, NULL_POINTER, INVALID_UTF8, INVALID_ARGUMENT, UNSUPPORTED, or PANIC
 */
LLEV_API LlevStatus llev_phonetic_pattern_compile_regex(
    const char* source, size_t source_len, LlevPhoneticPattern** out_pattern);
/** Compile an import-free LLRE document into a Unicode phonetic automaton.
 * @param source UTF-8 LLRE document bytes
 * @param source_len document byte length
 * @param out_pattern receives a caller-owned immutable pattern
 * @return OK, NULL_POINTER, INVALID_UTF8, INVALID_ARGUMENT, UNSUPPORTED, or PANIC
 */
LLEV_API LlevStatus llev_phonetic_pattern_compile_llre(
    const char* source, size_t source_len, LlevPhoneticPattern** out_pattern);
/** Resolve .llre imports from a trusted local file, then compile under the
 * same NFA state ceiling. Search paths are borrowed UTF-8 directory views;
 * zero paths select native loader defaults. Path limits do not bound the
 * contents of transitively imported files. */
LLEV_API LlevStatus llev_phonetic_pattern_load_llre_file(
    const char* path, size_t path_len,
    const LlevUtf8View* search_paths, size_t search_path_count,
    size_t max_total_path_bytes, LlevPhoneticPattern** out_pattern);
/** Release a compiled pattern; cursors retain independent pattern products.
 * @param pattern owned pattern to consume; NULL is a no-op
 */
LLEV_API void llev_phonetic_pattern_free(LlevPhoneticPattern* pattern);
/** Read the compiled pattern automaton's structural size.
 * @param pattern live immutable pattern
 * @param out_states receives the number of NFA states
 * @param out_transitions receives the number of NFA transitions
 * @return OK, NULL_POINTER, UNSUPPORTED, or PANIC
 */
LLEV_API LlevStatus llev_phonetic_pattern_size(
    const LlevPhoneticPattern* pattern,
    size_t* out_states,
    size_t* out_transitions);
/** Decide complete-string membership in a compiled phonetic language.
 * @param pattern live immutable pattern
 * @param input UTF-8 input bytes
 * @param input_len input byte length
 * @param out_matches receives zero or one
 * @return OK, NULL_POINTER, INVALID_UTF8, UNSUPPORTED, or PANIC
 */
LLEV_API LlevStatus llev_phonetic_pattern_matches(
    const LlevPhoneticPattern* pattern,
    const char* input,
    size_t input_len,
    uint8_t* out_matches);
/** Query Unicode dictionary terms near any word in a phonetic language.
 * @param transducer live Unicode-scalar configuration
 * @param pattern live immutable phonetic pattern
 * @param max_distance inclusive edit-distance bound
 * @param out_cursor receives an exclusive caller-owned cursor on success
 * @return OK, NULL_POINTER, DOMAIN_MISMATCH, UNSUPPORTED, a mapped provider
 * status, PROVIDER_ERROR, or PANIC
 */
LLEV_API LlevStatus llev_transducer_query_pattern(
    const LlevTransducer* transducer,
    const LlevPhoneticPattern* pattern,
    uint8_t max_distance,
    LlevQueryCursor** out_cursor);

/** Parse an import-free UTF-8 .llev rewrite-rule document.
 * @param source document bytes
 * @param source_len document byte length
 * @param out_rules receives a caller-owned immutable rule set
 * @return OK, NULL_POINTER, INVALID_UTF8, INVALID_ARGUMENT, UNSUPPORTED, or PANIC
 */
LLEV_API LlevStatus llev_phonetic_rules_parse(
    const char* source, size_t source_len, LlevPhoneticRuleSet** out_rules);
/** Resolve .llev includes from a trusted local file. Search paths are
 * borrowed UTF-8 directory views. Zero paths use native loader defaults;
 * at most 64 paths and positive max_total_path_bytes are accepted. The path
 * ceiling does not bound contents of files loaded transitively. */
LLEV_API LlevStatus llev_phonetic_rules_load_file(
    const char* path, size_t path_len,
    const LlevUtf8View* search_paths, size_t search_path_count,
    size_t max_total_path_bytes, LlevPhoneticRuleSet** out_rules);
/** Construct one built-in phonetic rewrite-rule set.
 * @param kind one published LlevPhoneticRuleSetKind numeric value
 * @param out_rules receives a caller-owned immutable rule set
 * @return OK, NULL_POINTER, INVALID_ARGUMENT, UNSUPPORTED, or PANIC
 */
LLEV_API LlevStatus llev_phonetic_rules_builtin(
    uint32_t kind, LlevPhoneticRuleSet** out_rules);
/** Release a compiled rewrite-rule set.
 * @param rules owned rule set to consume; NULL is a no-op
 */
LLEV_API void llev_phonetic_rules_free(LlevPhoneticRuleSet* rules);
/** Read the number of enabled rules.
 * @param rules live immutable rule set
 * @param out_len receives the enabled-rule count
 * @return OK, NULL_POINTER, UNSUPPORTED, or PANIC
 */
LLEV_API LlevStatus llev_phonetic_rules_len(
    const LlevPhoneticRuleSet* rules, size_t* out_len);
/** Rewrite UTF-8 text to a bounded fixed point.
 * @param rules live immutable rule set
 * @param input UTF-8 input bytes
 * @param input_len input byte length
 * @param out_text receives an owned, length-bearing UTF-8 result
 * @return OK, NULL_POINTER, INVALID_UTF8, UNSUPPORTED, or PANIC
 */
LLEV_API LlevStatus llev_phonetic_rules_apply(
    const LlevPhoneticRuleSet* rules,
    const char* input,
    size_t input_len,
    LlevOwnedString* out_text);
/** Release and zero a length-bearing string returned by the phonetic API.
 * @param value owned string structure; NULL and {NULL, 0} are no-ops
 */
LLEV_API void llev_owned_string_free(LlevOwnedString* value);

/** Native versioned phonetic AOT serialization. All operations require
 * LLEV_BUILD_FEATURE_PHONETIC_AOT; otherwise they return UNSUPPORTED without
 * touching outputs. Byte ceilings must be positive and at most 16 MiB.
 * Deserializers reject invalid magic/version and oversized NFAs. The formats
 * are native `.llev` / `.llre` formats, not a stable cross-version wire ABI. */
LLEV_API LlevStatus llev_phonetic_rules_to_bytes(
    const LlevPhoneticRuleSet* rules, size_t max_output_bytes,
    LlevOwnedBytes* out_bytes);
LLEV_API LlevStatus llev_phonetic_rules_from_bytes(
    const uint8_t* data, size_t data_len, size_t max_input_bytes,
    LlevPhoneticRuleSet** out_rules);
LLEV_API LlevStatus llev_phonetic_pattern_to_bytes(
    const LlevPhoneticPattern* pattern, size_t max_output_bytes,
    LlevOwnedBytes* out_bytes);
LLEV_API LlevStatus llev_phonetic_pattern_from_bytes(
    const uint8_t* data, size_t data_len, size_t max_input_bytes,
    LlevPhoneticPattern** out_pattern);
/** Release and clear an exact owned AOT byte buffer. NULL is a no-op. */
LLEV_API void llev_owned_bytes_free(LlevOwnedBytes* value);

/** Dimension costs for native articulatory distance. Every field must be
 * finite and nonnegative. NULL selects Rust's FeatureDistanceWeights::standard.
 * The layout is seven consecutive IEEE-754 binary64 values. */
typedef struct LlevPhoneticFeatureWeights {
    double voicing;
    double place_step;
    double manner_default;
    double manner_table_scale;
    double vowel_height_step;
    double vowel_backness_step;
    double vowel_rounding;
} LlevPhoneticFeatureWeights;

/** Native IPA feature distance between two Unicode scalar values.
 * Surrogate and out-of-range inputs are INVALID_ARGUMENT. The result is
 * written only on success; a NULL weights pointer selects native defaults. */
LLEV_API LlevStatus llev_phonetic_articulatory_distance(
    uint32_t source, uint32_t target,
    const LlevPhoneticFeatureWeights* weights, double* out_distance);

/** Native weighted edit distance with insertion/deletion cost one and
 * articulatory substitution cost. max_cells must be positive and bounds the
 * source/target scalar product as well as either individual length. Exceeding
 * it returns LIMIT_EXCEEDED before allocation; output is then unchanged. */
LLEV_API LlevStatus llev_phonetic_articulatory_edit_distance(
    const char* source, size_t source_len,
    const char* target, size_t target_len,
    const LlevPhoneticFeatureWeights* weights,
    size_t max_cells, double* out_distance);

/** Count syllables under the English-orthographic (ipa=0) or IPA (ipa=1)
 * heuristic. max_input_scalars must be positive. Empty input
 * has count zero. The result is unchanged on error. */
LLEV_API LlevStatus llev_phonetic_syllable_count(
    const char* input, size_t input_len, uint8_t ipa,
    size_t max_input_scalars, size_t* out_count);

/** Copy scalar-indexed syllable start positions. out_count receives the
 * required number of positions even when capacity is insufficient. The
 * (NULL, 0) sizing call is valid; no positions are written on failure.
 * English and IPA may report a number of boundary starts distinct from the
 * corresponding heuristic syllable count; both match their native Rust APIs. */
LLEV_API LlevStatus llev_phonetic_syllable_boundaries(
    const char* input, size_t input_len, uint8_t ipa,
    size_t max_input_scalars, size_t* out_positions,
    size_t capacity, size_t* out_count);

/** Immutable, reusable word-boundary phonetic grep configuration. */
typedef struct LlevPhoneticGrep LlevPhoneticGrep;

/** One copied grep result. Byte offsets are zero-based, end-exclusive within
 * the one-based logical line number. All reserved bytes are zero. */
typedef struct LlevPhoneticGrepMatch {
    size_t line_number;
    size_t start_byte;
    size_t end_byte;
    uint8_t distance;
    uint8_t reserved[7];
} LlevPhoneticGrepMatch;

/** Compile a bounded phonetic grep pattern. NULL rules mean no rewrite rules;
 * a non-NULL rule set is cloned, so the matcher outlives its source rules.
 * `algorithm` is one published LlevAlgorithm value and `case_insensitive` is
 * zero or one. Pattern NFA states obey the shared language-product ceiling. */
LLEV_API LlevStatus llev_phonetic_grep_new(
    const char* pattern, size_t pattern_len,
    const LlevPhoneticRuleSet* rules,
    uint8_t max_distance, uint32_t algorithm, uint8_t case_insensitive,
    LlevPhoneticGrep** out_grep);
/** Consume an owned grep handle. NULL is a no-op. */
LLEV_API void llev_phonetic_grep_free(LlevPhoneticGrep* grep);
/** Report effective distance and optional inline `(?;N:...)` override. */
LLEV_API LlevStatus llev_phonetic_grep_distance_config(
    const LlevPhoneticGrep* grep,
    uint8_t* out_effective, uint8_t* out_local, uint8_t* out_has_local);
/** Return optional distance for one candidate. The caller sets a positive
 * max_candidate_bytes work ceiling; both outputs change only on success. */
LLEV_API LlevStatus llev_phonetic_grep_matches(
    const LlevPhoneticGrep* grep,
    const char* candidate, size_t candidate_len, size_t max_candidate_bytes,
    uint8_t* out_distance, uint8_t* out_matches);
/** Copy all word matches in one line. The caller sets a positive input ceiling
 * and owns a descriptor array. (NULL,0) is valid for sizing. Insufficient
 * capacity returns LIMIT_EXCEEDED, sets out_count, and writes no descriptors. */
LLEV_API LlevStatus llev_phonetic_grep_scan_line(
    const LlevPhoneticGrep* grep,
    const char* line, size_t line_len, size_t max_input_bytes,
    LlevPhoneticGrepMatch* out_matches, size_t capacity, size_t* out_count);
/** Scan all lines using Rust's str::lines and preserve one-based line numbers;
 * capacity and ownership follow llev_phonetic_grep_scan_line. */
LLEV_API LlevStatus llev_phonetic_grep_scan_text(
    const LlevPhoneticGrep* grep,
    const char* document, size_t document_len, size_t max_input_bytes,
    LlevPhoneticGrepMatch* out_matches, size_t capacity, size_t* out_count);

/** One owned normalized-space candidate. Free only via candidates_free. */
typedef struct LlevPhoneticCandidate {
    LlevOwnedString term;
    size_t distance;
    LlevOwnedString normalized_form;
} LlevPhoneticCandidate;
/** Native phonetic-normalized dictionary; mode 0 mutable, mode 1 compact
 * immutable term-ID payloads. */
typedef struct LlevPhoneticDictionary LlevPhoneticDictionary;
/** Build a native normalized dictionary from length-bearing UTF-8 terms.
 * Optional rules are copied; NULL selects English Zompist rules. All terms
 * are copied. Positive max_terms and max_total_bytes are strict ceilings.
 * Algorithm must be a published LlevAlgorithm and mode must be 0 or 1. */
LLEV_API LlevStatus llev_phonetic_dictionary_new(
    const LlevUtf8View* terms, size_t term_count,
    const LlevPhoneticRuleSet* rules, uint32_t algorithm, uint8_t mode,
    size_t max_terms, size_t max_total_bytes,
    LlevPhoneticDictionary** out_dictionary);
/** Consume a dictionary; NULL is a no-op. */
LLEV_API void llev_phonetic_dictionary_free(LlevPhoneticDictionary* dictionary);
/** Query in native relevance order; max_query_scalars and max_results must be
 * positive. An excessive result count returns LIMIT_EXCEEDED with both outputs
 * unchanged. The successful returned array and its strings are caller-owned. */
LLEV_API LlevStatus llev_phonetic_dictionary_query(
    const LlevPhoneticDictionary* dictionary,
    const char* query, size_t query_len, size_t max_distance,
    size_t max_query_scalars, size_t max_results,
    LlevPhoneticCandidate** out_candidates, size_t* out_count);
/** Consume an exact array/count pair returned by dictionary_query. */
LLEV_API void llev_phonetic_candidates_free(
    LlevPhoneticCandidate* candidates, size_t count);
/** Modify mode-0 dictionary. remove=0 inserts; remove=1 removes. Compact
 * mode reports UNSUPPORTED. max_term_scalars must be positive. */
LLEV_API LlevStatus llev_phonetic_dictionary_update(
    LlevPhoneticDictionary* dictionary, const char* term, size_t term_len,
    uint8_t remove, size_t max_term_scalars, uint8_t* out_changed);

/** Character-level phonetic grep, distinct from word-boundary matching.
 * All span offsets are zero-based, end-exclusive, and refer to the original
 * UTF-8 document. Both strings are independently owned by the result array. */
typedef struct LlevPhoneticOnlineMatch {
    size_t byte_start;
    size_t byte_end;
    size_t char_start;
    size_t char_end;
    LlevOwnedString original_text;
    LlevOwnedString normalized_text;
    uint8_t distance;
    uint8_t reserved[7];
} LlevPhoneticOnlineMatch;
typedef struct LlevPhoneticOnlineGrep LlevPhoneticOnlineGrep;
typedef struct LlevPhoneticOnlineStream LlevPhoneticOnlineStream;
/** Compile a reusable matcher. NULL rules mean no rewrite rules; otherwise
 * rules are cloned. max_pattern_scalars must be positive, and
 * case_insensitive must be 0 or 1. */
LLEV_API LlevStatus llev_phonetic_online_new(
    const char* pattern, size_t pattern_len,
    const LlevPhoneticRuleSet* rules, uint8_t max_distance,
    uint8_t case_insensitive, size_t max_pattern_scalars,
    LlevPhoneticOnlineGrep** out_grep);
LLEV_API void llev_phonetic_online_free(LlevPhoneticOnlineGrep* grep);
/** Return a copied, owned normalized query; free with owned_string_free. */
LLEV_API LlevStatus llev_phonetic_online_normalized_query(
    const LlevPhoneticOnlineGrep* grep, LlevOwnedString* out_text);
/** Scan a complete document. Positive byte/result ceilings are required;
 * results and nested strings are owned and freed as one array. */
LLEV_API LlevStatus llev_phonetic_online_scan(
    const LlevPhoneticOnlineGrep* grep,
    const char* document, size_t document_len,
    size_t max_input_bytes, size_t max_matches,
    LlevPhoneticOnlineMatch** out_matches, size_t* out_count);
LLEV_API void llev_phonetic_online_matches_free(
    LlevPhoneticOnlineMatch* matches, size_t count);
/** Create a scanner whose input is buffered up to max_total_bytes. The
 * matcher is cloned; both handles may be freed independently. */
LLEV_API LlevStatus llev_phonetic_online_stream_new(
    const LlevPhoneticOnlineGrep* grep, size_t max_total_bytes,
    LlevPhoneticOnlineStream** out_stream);
LLEV_API void llev_phonetic_online_stream_free(LlevPhoneticOnlineStream* stream);
LLEV_API LlevStatus llev_phonetic_online_stream_feed(
    LlevPhoneticOnlineStream* stream, const char* chunk, size_t chunk_len);
/** Finish once. A LIMIT_EXCEEDED result still consumes stream state but
 * leaves outputs untouched. A second finish/feed returns INVALID_ARGUMENT. */
LLEV_API LlevStatus llev_phonetic_online_stream_finish(
    LlevPhoneticOnlineStream* stream, size_t max_matches,
    LlevPhoneticOnlineMatch** out_matches, size_t* out_count);

/** Native token-query grep with full per-token phonetic/edit details. */
typedef struct LlevPhoneticTokenGrep LlevPhoneticTokenGrep;
typedef struct LlevPhoneticTokenDetail {
    size_t token_index;
    size_t byte_start;
    size_t byte_end;
    LlevOwnedString original_text;
    LlevOwnedString normalized_text;
    uint8_t distance;
    uint8_t reserved[7];
} LlevPhoneticTokenDetail;
typedef struct LlevPhoneticTokenMatch {
    size_t byte_start;
    size_t byte_end;
    uint8_t total_distance;
    uint8_t reserved[7];
    LlevOwnedString matched_text;
    LlevPhoneticTokenDetail* details;
    size_t detail_count;
} LlevPhoneticTokenMatch;
/** Compile native TokenGrep query syntax. Optional rules are cloned. */
LLEV_API LlevStatus llev_phonetic_token_new(
    const char* query, size_t query_len,
    const LlevPhoneticRuleSet* rules, uint8_t default_distance,
    size_t max_query_bytes, LlevPhoneticTokenGrep** out_grep);
LLEV_API void llev_phonetic_token_free(LlevPhoneticTokenGrep* grep);
/** Scan one UTF-8 document. Every ceiling must be positive; byte offsets
 * refer to that document. The result and every nested string/detail array
 * are owned together and freed by token_matches_free. */
LLEV_API LlevStatus llev_phonetic_token_scan(
    const LlevPhoneticTokenGrep* grep,
    const char* document, size_t document_len,
    size_t max_input_bytes, size_t max_matches, size_t max_details,
    LlevPhoneticTokenMatch** out_matches, size_t* out_count);
LLEV_API void llev_phonetic_token_matches_free(
    LlevPhoneticTokenMatch* matches, size_t count);

/** Incremental phonetic rewrite transducer, distinct from fixed-point rules_apply.
 * Context-sensitive rules may delay emission until later input or finish. */
typedef struct LlevPhoneticTransducer LlevPhoneticTransducer;
/** Clone optional rules; NULL selects identity rewriting. */
LLEV_API LlevStatus llev_phonetic_transducer_new(
    const LlevPhoneticRuleSet* rules, LlevPhoneticTransducer** out_transducer);
LLEV_API void llev_phonetic_transducer_free(LlevPhoneticTransducer* transducer);
/** Feed UTF-8 text. An output ceiling failure resets the transducer, leaves
 * out_text unchanged, and requires restarting the logical stream. */
LLEV_API LlevStatus llev_phonetic_transducer_feed(
    LlevPhoneticTransducer* transducer, const char* chunk, size_t chunk_len,
    size_t max_input_scalars, size_t max_output_bytes, LlevOwnedString* out_text);
/** Flush buffered context. A successful finish requires reset before reuse. */
LLEV_API LlevStatus llev_phonetic_transducer_finish(
    LlevPhoneticTransducer* transducer, size_t max_output_bytes,
    LlevOwnedString* out_text);
LLEV_API LlevStatus llev_phonetic_transducer_reset(LlevPhoneticTransducer* transducer);
/** Normalize an independent UTF-8 string without changing incremental state. */
LLEV_API LlevStatus llev_phonetic_transducer_normalize(
    const LlevPhoneticTransducer* transducer,
    const char* input, size_t input_len,
    size_t max_input_scalars, size_t max_output_bytes,
    LlevOwnedString* out_text);

/** Exhaustive and cost-aware reverse-phonetic expansion work ceilings. All
 * fields must be positive. max_rule_units sums pattern/replacement phone units.
 * The exhaustive function accounts intermediate strings conservatively before
 * deduplication, so a sufficient final-output ceiling may still reject. */
typedef struct LlevPhoneticExpansionLimits {
    size_t max_input_scalars;
    size_t max_rules;
    size_t max_rule_units;
    size_t max_nodes;
    size_t max_output_bytes;
} LlevPhoneticExpansionLimits;
/** Expand every reverse-phonetic segmentation to a regex pattern. NULL rules
 * mean identity. The result is owned and freed with owned_string_free. */
LLEV_API LlevStatus llev_phonetic_expand(
    const char* input, size_t input_len, const LlevPhoneticRuleSet* rules,
    const LlevPhoneticExpansionLimits* limits, LlevOwnedString* out_pattern);
/** Greedy reverse expansion with native maximum-rule-cost accounting.
 * Both outputs are transactional. This is not the exhaustive segmentation. */
LLEV_API LlevStatus llev_phonetic_expand_with_costs(
    const char* input, size_t input_len, const LlevPhoneticRuleSet* rules,
    const LlevPhoneticExpansionLimits* limits,
    LlevOwnedString* out_pattern, double* out_cost);

/** Stable bit indices for the native IPA feature classification. These are
 * independent of Rust enum discriminants and fixed for this API revision. */
typedef enum LlevPhoneticFeatureIndex {
    LLEV_FEATURE_VOICED = 0,
    LLEV_FEATURE_VOICELESS = 1,
    LLEV_FEATURE_STOP = 2,
    LLEV_FEATURE_FRICATIVE = 3,
    LLEV_FEATURE_AFFRICATE = 4,
    LLEV_FEATURE_NASAL = 5,
    LLEV_FEATURE_APPROXIMANT = 6,
    LLEV_FEATURE_LATERAL = 7,
    LLEV_FEATURE_RHOTIC = 8,
    LLEV_FEATURE_BILABIAL = 9,
    LLEV_FEATURE_LABIODENTAL = 10,
    LLEV_FEATURE_DENTAL = 11,
    LLEV_FEATURE_ALVEOLAR = 12,
    LLEV_FEATURE_POST_ALVEOLAR = 13,
    LLEV_FEATURE_PALATAL = 14,
    LLEV_FEATURE_VELAR = 15,
    LLEV_FEATURE_GLOTTAL = 16,
    LLEV_FEATURE_VOWEL = 17,
    LLEV_FEATURE_CONSONANT = 18,
    LLEV_FEATURE_HIGH = 19,
    LLEV_FEATURE_MID = 20,
    LLEV_FEATURE_LOW = 21,
    LLEV_FEATURE_FRONT = 22,
    LLEV_FEATURE_CENTRAL = 23,
    LLEV_FEATURE_BACK = 24,
    LLEV_FEATURE_ROUNDED = 25,
    LLEV_FEATURE_UNROUNDED = 26,
    LLEV_FEATURE_SIBILANT = 27,
    LLEV_FEATURE_ASPIRATED = 28,
    LLEV_FEATURE_TENSE = 29,
    LLEV_FEATURE_PHARYNGEALIZED = 30,
    LLEV_FEATURE_LABIALIZED = 31,
    LLEV_FEATURE_VELARIZED = 32,
    LLEV_FEATURE_RETROFLEX = 33,
    LLEV_FEATURE_UVULAR = 34,
    LLEV_FEATURE_PHARYNGEAL = 35,
    LLEV_FEATURE_EPIGLOTTAL = 36,
    LLEV_FEATURE_TAP = 37,
    LLEV_FEATURE_TRILL = 38,
    LLEV_FEATURE_EJECTIVE = 39,
    LLEV_FEATURE_IMPLOSIVE = 40,
    LLEV_FEATURE_CLICK = 41
} LlevPhoneticFeatureIndex;
#define LLEV_PHONETIC_FEATURE_BIT(index) (UINT64_C(1) << (index))
/** Return the low-42-bit native feature mask for one Unicode scalar. Unknown
 * characters have zero features; invalid scalar values are INVALID_ARGUMENT. */
LLEV_API LlevStatus llev_phonetic_features(uint32_t character, uint64_t* out_mask);
/** Query the native table for characters with all (any=0) or any (any=1)
 * selected features. Unknown bits are INVALID_ARGUMENT. Empty mask returns
 * an empty result. Sizing calls use (NULL,0); insufficient capacity writes
 * required out_count but no characters. Results are scalar-sorted. */
LLEV_API LlevStatus llev_phonetic_chars_with_features(
    uint64_t feature_mask, uint8_t any,
    uint32_t* out_chars, size_t capacity, size_t* out_count);
LLEV_API LlevStatus llev_phonetic_similar_chars(
    uint32_t character, uint32_t* out_chars,
    size_t capacity, size_t* out_count);
/** Return an optional voicing counterpart. out_found controls whether
 * out_character is meaningful; both outputs are set only on success. */
LLEV_API LlevStatus llev_phonetic_voicing_pair(
    uint32_t character, uint32_t* out_character, uint8_t* out_found);
/** Test native shared-feature similarity (selector 0) or a native cost-zero
 * substitution (selector 1). These are distinct predicates. */
LLEV_API LlevStatus llev_phonetic_feature_relation(
    uint32_t source, uint32_t target, uint8_t free_substitution,
    uint8_t* out_matches);
/** Copy native feature-based expansion in its own vector order. */
LLEV_API LlevStatus llev_phonetic_expand_feature_based(
    uint32_t character, uint32_t* out_chars,
    size_t capacity, size_t* out_count);
/** Native weighted distance between two feature sets; null weights select
 * defaults. Invalid feature bits or costs are INVALID_ARGUMENT. */
LLEV_API LlevStatus llev_phonetic_feature_set_distance(
    uint64_t source_mask, uint64_t target_mask,
    const LlevPhoneticFeatureWeights* weights, double* out_distance);

/* Revision 6: finite-owned Unicode WallBreaker. This is not a dictionary
 * resource adapter: vt.dictionary.v1 has no exact-substring capability. */
typedef struct LlevWallBreaker LlevWallBreaker;
typedef struct LlevWallBreakerCursor LlevWallBreakerCursor;

typedef struct LlevWallBreakerTerm {
    const char* data;
    size_t byte_len;
} LlevWallBreakerTerm;

/** Logical limits for a matcher. Every field is positive. Implementation
 * maxima, respectively: 4096 terms, 1 MiB total term bytes, 256 scalars per
 * term/query, 16 MiB candidate-clone bound, 4096 results, 1 MiB result bytes.
 * These are not process-RSS limits. Distance has a separate hard maximum 8.
 * Construction preflights term, UTF-8, scalar and distance limits before
 * building the owned SCDAWG. Selective queries enforce the conservative
 * candidate-clone bound before materializing substring occurrences; complete-
 * term streaming queries do not allocate those candidates. */
typedef struct LlevWallBreakerLimits {
    size_t max_terms;
    size_t max_total_term_bytes;
    size_t max_term_scalars;
    size_t max_query_scalars;
    size_t max_candidate_clone_bytes;
    size_t max_results;
    size_t max_result_bytes;
} LlevWallBreakerLimits;

typedef struct LlevWallBreakerResultView {
    const uint8_t* term_data;
    size_t byte_len;
    size_t distance;
} LlevWallBreakerResultView;

typedef struct LlevWallBreakerBatchView {
    const LlevWallBreakerResultView* results;
    size_t len;
    uint64_t generation;
} LlevWallBreakerBatchView;

typedef struct LlevPatternPiece {
    size_t byte_offset;
    size_t byte_len;
    size_t start_scalar;
    size_t end_scalar;
    size_t piece_index;
} LlevPatternPiece;

/** Copy caller-owned UTF-8 terms into a fixed immutable Unicode SCDAWG.
 * Zero terms are allowed; (NULL,0) denotes an empty term. Valid algorithm
 * values are LLEV_ALGORITHM_*. On error *out is NULL, except if out is NULL.
 * Invalid UTF-8 reports INVALID_UTF8; null required pointers NULL_POINTER;
 * malformed non-null inputs INVALID_ARGUMENT;
 * exceeded term/distance limits LIMIT_EXCEEDED. */
LLEV_API LlevStatus llev_wallbreaker_new_utf8(
    const LlevWallBreakerTerm* terms, size_t term_count,
    LlevWallBreakerLimits limits, uint32_t algorithm, size_t max_distance,
    LlevWallBreaker** out);
/** Free a matcher; any completed result cursor remains independent. */
LLEV_API void llev_wallbreaker_free(LlevWallBreaker* matcher);
/** Eagerly verify all results in one immutable matcher revision. Short or
 * unrestricted-Damerau queries borrow complete terms one at a time; other
 * queries materialize bounded substring candidates. A limit failure publishes
 * no cursor or partial result; *out is NULL. Results preserve first-seen
 * candidate order and are deduplicated by complete term. */
LLEV_API LlevStatus llev_wallbreaker_query_utf8(
    const LlevWallBreaker* matcher, const char* query, size_t query_len,
    LlevWallBreakerCursor** out);
/** Publish one borrowed batch, or END with an empty output. At most one lease
 * is active. If the next term alone exceeds max_bytes, LIMIT_EXCEEDED does
 * not advance. All descriptors and term bytes live until release or free;
 * cancelling during a lease preserves that lease until release/free. */
LLEV_API LlevStatus llev_wallbreaker_cursor_next_batch(
    LlevWallBreakerCursor* cursor, size_t max_entries, size_t max_bytes,
    LlevWallBreakerBatchView* out);
/** Release exactly the active generation; stale/double releases fail. */
LLEV_API LlevStatus llev_wallbreaker_cursor_release_batch(
    LlevWallBreakerCursor* cursor, uint64_t generation);
/** Cancel further advances (CLOSED); an active lease remains releasable. */
LLEV_API LlevStatus llev_wallbreaker_cursor_cancel(LlevWallBreakerCursor* cursor);
/** Free an exclusive cursor and invalidate any active batch view. */
LLEV_API void llev_wallbreaker_cursor_free(LlevWallBreakerCursor* cursor);
/** Project native PatternSplitter pieces with Unicode scalar and UTF-8 byte
 * coordinates. A capacity query uses pieces=NULL/capacity=0 and returns
 * LIMIT_EXCEEDED with *out_required; insufficient capacity writes no pieces.
 * The input is borrowed only for the call. */
LLEV_API LlevStatus llev_wallbreaker_split_utf8(
    const char* query, size_t query_len, uint32_t algorithm,
    size_t max_distance, LlevPatternPiece* pieces, size_t capacity,
    size_t* out_required);

/** Compare two UTF-8 strings with native Jaro (prefix_scale=0) or scaled
 * Jaro-Winkler (0<prefix_scale<=0.25). Input bytes are borrowed only during
 * the call. The caller chooses a per-input byte ceiling and a worst-case
 * scalar comparison ceiling; exceeding either returns LIMIT_EXCEEDED before
 * scoring. out_score is initialized before validation and must be disjoint
 * from both inputs. A zero-length input may use NULL. */
LLEV_API LlevStatus llev_jaro_similarity_utf8(
    const char* left, size_t left_len,
    const char* right, size_t right_len,
    double prefix_scale, size_t max_input_bytes,
    size_t max_comparisons, double* out_score);

/** Check a UTF-8 source against the native n-gram (mode=1) or hybrid
 * n-gram/Jaro-Winkler (mode=2) filter. The candidate is indexed for this
 * call; no native handle or source memory is retained. ngram_size=0 uses
 * unigrams as in the native index. jaro_threshold must be zero for mode=1.
 * The inputs are borrowed and capped per input by max_input_bytes. Hybrid
 * Jaro work is additionally capped by max_comparisons. out_accept is set to
 * zero before validation and must be disjoint from both inputs. */
LLEV_API LlevStatus llev_source_filter_utf8(
    const char* query, size_t query_len,
    const char* candidate, size_t candidate_len,
    uint32_t mode, size_t ngram_size, size_t max_distance,
    double jaro_threshold, size_t max_input_bytes,
    size_t max_comparisons, uint8_t* out_accept);

/** Persistent native source filter. Mode 1 uses an n-gram postings index;
 * mode 2 additionally applies Jaro-Winkler. ngram_size must be positive,
 * reserved must be zero, and threshold must be finite in [0,1] (zero for
 * mode 1). Insertion copies UTF-8 terms and deduplicates them. */
typedef struct LlevSourceFilterIndexConfig {
    uint32_t mode;
    uint32_t reserved;
    size_t ngram_size;
    double jaro_threshold;
    size_t max_terms;
    size_t max_term_bytes;
    size_t max_source_bytes;
    size_t max_query_bytes;
} LlevSourceFilterIndexConfig;

/** Fail-closed query ceilings. max_candidates bounds source cardinality;
 * max_results bounds the complete result; max_comparisons bounds a
 * conservative query-character x source-byte hybrid comparison count. */
typedef struct LlevSourceFilterIndexLimits {
    size_t max_candidates;
    size_t max_results;
    size_t max_comparisons;
} LlevSourceFilterIndexLimits;

typedef struct LlevSourceFilterIndex LlevSourceFilterIndex;

/** Freeze is idempotent. Queries require a frozen index and copy zero-based
 * insertion IDs into caller storage in source order. Duplicate terms retain
 * their original ID. The query publishes no partial output on failure;
 * out_reason=1 means source cardinality, 2 means result count, 3 means output
 * capacity, and 4 means hybrid comparisons. On success out_reason is zero.
 * The copied IDs remain valid after index free. */
LLEV_API LlevStatus llev_source_filter_index_new(
    const LlevSourceFilterIndexConfig* config,
    LlevSourceFilterIndex** out_index);
LLEV_API LlevStatus llev_source_filter_index_insert(
    LlevSourceFilterIndex* index, const char* term, size_t term_len,
    size_t* out_id);
LLEV_API LlevStatus llev_source_filter_index_freeze(
    LlevSourceFilterIndex* index);
LLEV_API LlevStatus llev_source_filter_index_query(
    const LlevSourceFilterIndex* index, const char* query, size_t query_len,
    size_t max_distance, const LlevSourceFilterIndexLimits* limits,
    size_t* out_ids, size_t capacity, size_t* out_len, uint32_t* out_reason);
LLEV_API void llev_source_filter_index_free(LlevSourceFilterIndex* index);

/** Scalar temporal kernels. Every value is a stable wire constant. */
#define LLEV_TEMPORAL_MSM 1u
#define LLEV_TEMPORAL_ERP 2u
#define LLEV_TEMPORAL_TWED 3u
#define LLEV_TEMPORAL_DTW 4u
#define LLEV_TEMPORAL_FRECHET 5u
#define LLEV_TEMPORAL_SOFT_DTW 6u

/** Complete temporal result kinds. SOFT_DTW is a loss and can be negative. */
#define LLEV_TEMPORAL_FINITE 0u
#define LLEV_TEMPORAL_ABOVE_CUTOFF 1u
#define LLEV_TEMPORAL_NO_ALIGNMENT 2u
#define LLEV_TEMPORAL_INCOMPLETE 3u

/** Scalar algorithm and inclusive cutoff. Positive infinity requests the
 * unthresholded result. parameter0 is MSM cost, ERP gap, TWED stiffness, or
 * Soft-DTW gamma. parameter1 is TWED gap penalty. band is DTW half-width.
 * Unused fields and reserved must be zero. */
typedef struct LlevTemporalConfig {
    uint32_t algorithm;
    uint32_t reserved;
    double parameter0;
    double parameter1;
    size_t band;
    double cutoff;
} LlevTemporalConfig;

/** Hard caller-chosen ceilings for one temporal comparison. The default
 * values are 1,000,000 samples per input, 100,000,000 DP cells, 200,000,000
 * work units, and 512 MiB temporary storage. */
typedef struct LlevTemporalLimits {
    size_t max_series_len;
    size_t max_dp_cells;
    size_t max_work_units;
    size_t max_scratch_bytes;
} LlevTemporalLimits;

/** Finite score or exact non-score disposition. reason is 1 DP cells, 2 work
 * units, 3 scratch bytes, or 4 arithmetic/numeric overflow when incomplete.
 * Counters represent reserved whole-operation capacity, not elapsed time. */
typedef struct LlevTemporalDistanceResult {
    double value;
    uint32_t kind;
    uint32_t reason;
    size_t dp_cells;
    size_t work_units;
    size_t scratch_bytes;
} LlevTemporalDistanceResult;

/** Evaluate one native temporal scalar kernel after validating both finite
 * input series and reserving the complete DP/work/scratch budget. Both series
 * are borrowed only during this call. A zero-length input may use NULL; all
 * nonempty inputs must be aligned and readable for their declared lengths.
 * config, limits, and out_result must be valid, with writable output storage
 * disjoint from all inputs. Output is initialized before validation. Invalid
 * requests return INVALID_ARGUMENT or NULL_POINTER; budget and numeric
 * incompletion return LIMIT_EXCEEDED. No partial score is published. */
LLEV_API LlevStatus llev_temporal_distance(
    const double* left, size_t left_len,
    const double* right, size_t right_len,
    const LlevTemporalConfig* config,
    const LlevTemporalLimits* limits,
    LlevTemporalDistanceResult* out_result);

/** Exact channel/unit identity with a positive fold-local scale and weight.
 * All UTF-8 fields are copied at metric construction. */
typedef struct LlevVectorChannelView {
    const uint8_t* channel;
    size_t channel_len;
    const uint8_t* unit;
    size_t unit_len;
    double scale;
    double weight;
} LlevVectorChannelView;

/** Immutable fixed-channel point metric and exact scale provenance. */
typedef struct LlevVectorMetricView {
    const LlevVectorChannelView* channels;
    size_t channel_count;
    const uint8_t* training_fold;
    size_t training_fold_len;
    const uint8_t* estimator_revision;
    size_t estimator_revision_len;
} LlevVectorMetricView;

/** Point samples are consecutive columns, each with dimension coordinates.
 * Timestamp fields are used only for physical-time vector TWED; its unit is
 * 1 seconds, 2 milliseconds, 3 microseconds, or 4 nanoseconds. */
typedef struct LlevVectorSeriesView {
    const double* coordinates;
    size_t sample_count;
    size_t dimension;
    const double* timestamps;
    uint32_t timestamp_unit;
    uint32_t reserved;
    double origin;
} LlevVectorSeriesView;

/** Algorithm 2 ERP, 4 banded DTW, 5 discrete Frechet, or 7 timestamped TWED.
 * ERP and TWED require one finite gap_or_sentinel point. Unused fields must
 * be zero or NULL; cutoff is inclusive or positive infinity. Vector MSM is
 * unsupported because no canonical vector betweenness is defined. */
typedef struct LlevVectorTemporalConfig {
    uint32_t algorithm;
    uint32_t reserved;
    const double* gap_or_sentinel;
    double parameter0;
    double parameter1;
    size_t band;
    double cutoff;
} LlevVectorTemporalConfig;

/** Hard limits on native input copies, vector width, DP, and work. */
typedef struct LlevVectorTemporalLimits {
    LlevTemporalLimits scalar;
    size_t max_dimension;
    size_t max_band_width;
} LlevVectorTemporalLimits;

typedef struct LlevVectorMetric LlevVectorMetric;

/** Copy a fixed typed metric once for reuse by concurrent comparisons.
 * The caller must keep the handle live throughout each comparison. */
LLEV_API LlevStatus llev_vector_metric_new(
    const LlevVectorMetricView* view, size_t max_dimension,
    LlevVectorMetric** out_metric);
LLEV_API void llev_vector_metric_free(LlevVectorMetric* metric);

/** Score typed native vector ERP, DTW, Frechet, or physical-time TWED.
 * Both series are copied under max_scratch_bytes before native kernels run.
 * The result uses LlevTemporalDistanceResult kinds and reason codes. */
LLEV_API LlevStatus llev_vector_temporal_distance(
    const LlevVectorMetric* metric,
    const LlevVectorSeriesView* left,
    const LlevVectorSeriesView* right,
    const LlevVectorTemporalConfig* config,
    const LlevVectorTemporalLimits* limits,
    LlevTemporalDistanceResult* out_result);

/** Exact vector discrete Fréchet using audited untyped point distance:
 * ground=1 L1, 2 L2, 3 L-infinity. Both equal-dimensional untimestamped
 * paths are copied within the common vector temporal limits and normalized
 * modulo consecutive equal points. Result kinds match scalar temporal scores. */
LLEV_API LlevStatus llev_vector_frechet_ground_distance(
    uint32_t ground, const LlevVectorSeriesView* left,
    const LlevVectorSeriesView* right, double cutoff,
    const LlevVectorTemporalLimits* limits,
    LlevTemporalDistanceResult* out_result);

typedef struct LlevVectorFrechetOnline LlevVectorFrechetOnline;

/** Borrowed nonempty scalar samples with strictly increasing physical
 * timestamps. Unit is 1 seconds, 2 milliseconds, 3 microseconds, or
 * 4 nanoseconds. Origin is finite and no later than the first timestamp.
 * Reserved must be zero. The arrays are copied into bounded native series
 * for this call and never retained afterward. */
typedef struct LlevTimestampedSeriesView {
    const double* values;
    const double* timestamps;
    size_t len;
    uint32_t unit;
    uint32_t reserved;
    double origin;
} LlevTimestampedSeriesView;

/** Exact metric TWED over physical timestamps. Both series must use the same
 * canonical unit and origin; stiffness must be finite and positive, and gap
 * penalty finite and nonnegative. Cutoff is inclusive and nonnegative, or
 * positive infinity for the full score. The output uses the scalar temporal
 * result kinds and resource reason codes; invalid input publishes no score.
 * All pointers must address valid, mutually disjoint storage. */
LLEV_API LlevStatus llev_timestamped_twed_distance(
    const LlevTimestampedSeriesView* left,
    const LlevTimestampedSeriesView* right,
    double stiffness, double gap_penalty, double cutoff,
    const LlevTemporalLimits* limits,
    LlevTemporalDistanceResult* out_result);

/** Bounded witness extraction reuses scalar DP limits and adds a hard peak
 * witness-storage ceiling. */
typedef struct LlevTemporalAlignmentLimits {
    LlevTemporalLimits temporal;
    size_t max_witness_bytes;
} LlevTemporalAlignmentLimits;

/** One paged alignment operation. For MSM, operation is 1 move, 2 merge,
 * or 3 split and the remaining fields are zero. For ERP, TWED, DTW, and
 * Frechet, operation is 1 align, 2 advance query, or 3 advance candidate;
 * flags bits 0 and 1 indicate present zero-based endpoints. local_cost_bits
 * retains the exact IEEE-754 encoding of the native local cost. */
typedef struct LlevTemporalAlignmentStep {
    uint32_t operation;
    uint32_t flags;
    uint64_t query_endpoint;
    uint64_t candidate_endpoint;
    uint64_t local_cost_bits;
} LlevTemporalAlignmentStep;

/** Kind uses LLEV_TEMPORAL_* tags. A finite outcome transfers one immutable
 * owning witness handle; other outcomes leave it NULL. Incompletion reason
 * uses the existing resource reason codes, plus 5 for witness bytes. */
typedef struct LlevTemporalAlignmentOutcome {
    uint32_t kind;
    uint32_t reason;
    double distance;
    size_t step_count;
    size_t dp_cells;
    size_t work_units;
    size_t scratch_bytes;
    size_t witness_bytes;
} LlevTemporalAlignmentOutcome;

typedef struct LlevTemporalAlignment LlevTemporalAlignment;

/** Extract a deterministic MSM, ERP, unit-grid TWED, banded DTW, or
 * discrete Frechet witness. Soft-DTW has no alignment witness. Inputs are
 * borrowed for this call; the returned witness owns its bounded steps. */
LLEV_API LlevStatus llev_temporal_alignment_new(
    const double* query, size_t query_len,
    const double* candidate, size_t candidate_len,
    const LlevTemporalConfig* config,
    const LlevTemporalAlignmentLimits* limits,
    LlevTemporalAlignment** out_alignment,
    LlevTemporalAlignmentOutcome* out_outcome);

/** Extract a metric physical-time TWED witness over copied, validated
 * timestamped operands in the same unit and origin. */
LLEV_API LlevStatus llev_timestamped_twed_alignment_new(
    const LlevTimestampedSeriesView* query,
    const LlevTimestampedSeriesView* candidate,
    double stiffness, double gap_penalty, double cutoff,
    const LlevTemporalAlignmentLimits* limits,
    LlevTemporalAlignment** out_alignment,
    LlevTemporalAlignmentOutcome* out_outcome);

/** Copy at most capacity steps from a stable zero-based offset. A zero
 * capacity may use NULL for out_steps. The handle remains caller-owned. */
LLEV_API LlevStatus llev_temporal_alignment_page(
    const LlevTemporalAlignment* alignment,
    size_t start, LlevTemporalAlignmentStep* out_steps,
    size_t capacity, size_t* out_written);

/** Recompute and validate a scalar witness against supplied operands and
 * the native configuration captured at extraction. */
LLEV_API LlevStatus llev_temporal_alignment_replay(
    const LlevTemporalAlignment* alignment,
    const double* query, size_t query_len,
    const double* candidate, size_t candidate_len,
    double* out_distance);

/** Recompute and validate a physical-time TWED witness. */
LLEV_API LlevStatus llev_timestamped_twed_alignment_replay(
    const LlevTemporalAlignment* alignment,
    const LlevTimestampedSeriesView* query,
    const LlevTimestampedSeriesView* candidate,
    double* out_distance);

/** Release an alignment witness once. */
LLEV_API void llev_temporal_alignment_free(LlevTemporalAlignment* alignment);


/** Complete Soft-DTW loss and gradients with respect to both nonempty finite
 * operands. gamma must be finite and positive. Output buffers are caller
 * owned and have capacities at least left_len and right_len respectively.
 * No gradient element is written on invalid input or an incomplete result.
 * The result uses kind=0 on completion or kind=3 plus a resource/overflow
 * reason on LIMIT_EXCEEDED. All buffers must be valid and mutually disjoint. */
LLEV_API LlevStatus llev_soft_dtw_gradient(
    const double* left, size_t left_len,
    const double* right, size_t right_len,
    double gamma, const LlevTemporalLimits* limits,
    double* left_gradient, size_t left_capacity,
    double* right_gradient, size_t right_capacity,
    LlevTemporalDistanceResult* out_result);

/** Native temporal lower-bound selectors. */
#define LLEV_TEMPORAL_BOUND_ERP_GAP_MASS 1u
#define LLEV_TEMPORAL_BOUND_FRECHET_ENDPOINTS 2u
#define LLEV_TEMPORAL_BOUND_FRECHET_HAUSDORFF 3u
#define LLEV_TEMPORAL_BOUND_FRECHET_CANDIDATE 4u
#define LLEV_TEMPORAL_BOUND_KEOGH 5u
#define LLEV_TEMPORAL_BOUND_MSM_LENGTH 6u
#define LLEV_TEMPORAL_HEURISTIC_MSM_EUCLIDEAN 7u
#define LLEV_TEMPORAL_HEURISTIC_MSM_L1 8u
#define LLEV_TEMPORAL_HEURISTIC_MSM_COMBINED 9u

/** Compute a native temporal bound or explicitly selected heuristic under
 * work and scratch limits. parameter0 is the finite ERP gap or nonnegative
 * MSM split/merge cost; band applies only to Keogh. Only MSM length is safe
 * for exact MSM pruning: the three MSM heuristic modes can exceed MSM.
 * Reuses the scalar temporal finite/no-alignment/incomplete result tags. */
LLEV_API LlevStatus llev_temporal_lower_bound(
    const double* left, size_t left_len,
    const double* right, size_t right_len,
    uint32_t algorithm, double parameter0, size_t band,
    const LlevTemporalLimits* limits,
    LlevTemporalDistanceResult* out_result);

/** Native TWED length-only bound under explicit limits. */
LLEV_API LlevStatus llev_twed_length_lower_bound(
    size_t left_len, size_t right_len, double gap_penalty,
    const LlevTemporalLimits* limits,
    LlevTemporalDistanceResult* out_result);

/** Reusable native Keogh envelope; the constructor copies its finite query.
 * A plan requires a nonempty query and preserves its construction band.
 * out_has=0 denotes an unreachable target position. */
typedef struct LlevKeoghPlan LlevKeoghPlan;
LLEV_API LlevStatus llev_keogh_plan_new(
    const double* query, size_t query_len, size_t band,
    const LlevTemporalLimits* limits, LlevKeoghPlan** out_plan);
LLEV_API LlevStatus llev_keogh_plan_bounds_at(
    const LlevKeoghPlan* plan, size_t target_index,
    uint8_t* out_has, double* out_low, double* out_high);
LLEV_API LlevStatus llev_keogh_plan_score(
    const LlevKeoghPlan* plan, const double* candidate, size_t candidate_len,
    uint8_t squared, const LlevTemporalLimits* limits,
    LlevTemporalDistanceResult* out_result);
LLEV_API void llev_keogh_plan_free(LlevKeoghPlan* plan);

/** Bounded construction of a native quantized temporal index. The temporal
 * cutoff must be positive infinity; Soft-DTW is unsupported because it has
 * no elastic index. All three source limits are explicit. */
typedef struct LlevTemporalIndexConfig {
    LlevTemporalConfig temporal;
    double quant_min;
    double quant_max;
    uint32_t quant_bins;
    uint32_t reserved;
    size_t max_entries;
    size_t max_total_samples;
    size_t max_series_len;
} LlevTemporalIndexConfig;

/** Cumulative hard ceilings of one indexed range search. */
typedef struct LlevTemporalSearchLimits {
    size_t max_series_len;
    size_t max_dp_cells;
    size_t max_work_units;
    size_t max_scratch_bytes;
    size_t max_trie_nodes;
    size_t max_trie_edges;
    size_t max_candidates;
    size_t max_results;
    size_t max_queue_entries;
    size_t max_continuation_bytes;
} LlevTemporalSearchLimits;

/** Persistent PAA-ranked approximate MSM index. Feature selection is
 * advisory unless every episode is exactly reranked. All inserted samples
 * must be finite. The source and feature ceilings bound retained storage. */
typedef struct LlevApproxMsmIndexConfig {
    size_t segments;
    size_t candidate_limit;
    double split_merge_cost;
    size_t max_entries;
    size_t max_total_samples;
    size_t max_series_len;
    size_t max_total_features;
} LlevApproxMsmIndexConfig;

typedef struct LlevApproxMsmNeighbor {
    uint64_t id;
    size_t insertion_index;
    double distance;
} LlevApproxMsmNeighbor;

/** Kind 1 is exhaustive and proves recall; kind 2 is advisory and makes no
 * recall or absence claim; kind 3 is incomplete and may contain an exact
 * partial neighbor list. The reason uses temporal incompletion codes.
 * Coverage counts only exact MSM decisions, not PAA feature inspections. */
typedef struct LlevApproxMsmOutcome {
    uint32_t kind;
    uint32_t reason;
    size_t neighbor_count;
    size_t indexed_entries;
    size_t candidate_entries;
    size_t exact_reranked;
    size_t dp_cells;
    size_t work_units;
    size_t scratch_bytes;
    size_t candidates;
    size_t results;
} LlevApproxMsmOutcome;

typedef struct LlevApproxMsmIndex LlevApproxMsmIndex;

/** Mutate before freeze; frozen indexes support independent concurrent
 * queries. Query output storage is caller owned and must hold min(k, len)
 * neighbors. Each emitted distance is exact MSM. Incomplete outcomes remain
 * tagged, even when they contain an exact partial subset. */
LLEV_API LlevStatus llev_approx_msm_index_new(
    const LlevApproxMsmIndexConfig* config,
    LlevApproxMsmIndex** out_index);
LLEV_API LlevStatus llev_approx_msm_index_insert(
    LlevApproxMsmIndex* index, uint64_t id,
    const double* samples, size_t len, size_t* out_position);
LLEV_API LlevStatus llev_approx_msm_index_freeze(LlevApproxMsmIndex* index);
LLEV_API LlevStatus llev_approx_msm_index_query_knn(
    const LlevApproxMsmIndex* index, const double* query,
    size_t query_len, size_t k, const LlevTemporalSearchLimits* limits,
    LlevApproxMsmNeighbor* out_neighbors, size_t capacity,
    LlevApproxMsmOutcome* out_outcome);
LLEV_API void llev_approx_msm_index_free(LlevApproxMsmIndex* index);

/** Typed physical-time quantization, validated metric configuration, and
 * explicit bounded ingestion. Value and timestamp domains must be finite,
 * increasing, and the time minimum must follow the shared origin. Bin counts
 * are in [1, 2^31]. Quantization only prunes: exact scores use retained full
 * precision episodes. */
typedef struct LlevTimestampedTwedIndexConfig {
    uint32_t unit;
    uint32_t reserved;
    double origin;
    double value_min;
    double value_max;
    double time_min;
    double time_max;
    uint32_t value_bins;
    uint32_t time_bins;
    double stiffness;
    double gap_penalty;
    size_t max_entries;
    size_t max_total_samples;
    size_t max_series_len;
} LlevTimestampedTwedIndexConfig;

/** Common cumulative search limits plus bounded product-state arenas. Query
 * copy bytes are charged against scratch and continuation ceilings before
 * native search starts. */
typedef struct LlevTimestampedTwedSearchLimits {
    LlevTemporalSearchLimits common;
    size_t max_product_states;
    size_t max_product_positions;
    size_t max_transition_cache_entries;
} LlevTimestampedTwedSearchLimits;

/** Full-precision exact match: caller metadata, stable insertion position,
 * and physical-time TWED distance. Duplicate metadata IDs remain distinct
 * episodes. */
typedef struct LlevTimestampedTwedMatch {
    uint64_t id;
    uint64_t episode_id;
    double distance;
} LlevTimestampedTwedMatch;

typedef struct LlevTimestampedTwedIndex LlevTimestampedTwedIndex;
typedef struct LlevTimestampedTwedCursor LlevTimestampedTwedCursor;

/** Build, insert, freeze, and free one native timestamped index. Insertion
 * copies finite values and timestamps; `out_episode_id` identifies the new
 * episode even when caller metadata duplicates another episode. Mutation
 * requires exclusive access. Freezing is idempotent; frozen indexes reject
 * insertion. Cursors retain the frozen revision after index free. */
LLEV_API LlevStatus llev_timestamped_twed_index_new(
    const LlevTimestampedTwedIndexConfig* config,
    LlevTimestampedTwedIndex** out_index);
LLEV_API LlevStatus llev_timestamped_twed_index_insert(
    LlevTimestampedTwedIndex* index, uint64_t id,
    const LlevTimestampedSeriesView* series, uint64_t* out_episode_id);
LLEV_API LlevStatus llev_timestamped_twed_index_freeze(
    LlevTimestampedTwedIndex* index);
LLEV_API void llev_timestamped_twed_index_free(LlevTimestampedTwedIndex* index);

/** Start a bounded exact range query. The query is copied; its unit and
 * bitwise origin must match the index. A cursor retains the frozen index
 * revision and can outlive the index handle. Cutoff is inclusive. */
LLEV_API LlevStatus llev_timestamped_twed_index_query_range(
    const LlevTimestampedTwedIndex* index,
    const LlevTimestampedSeriesView* query, double cutoff,
    const LlevTimestampedTwedSearchLimits* limits,
    LlevTimestampedTwedCursor** out_cursor);

/** Exact k-nearest neighbors from a frozen revision, sorted by distance then
 * stable episode ID. This strict full scan returns no partial results on
 * LIMIT_EXCEEDED; out_reason uses the temporal index reason codes. The query
 * copy counts against scratch bytes. Caller storage must hold min(k, len)
 * matches, or the call fails before scanning. */
LLEV_API LlevStatus llev_timestamped_twed_index_query_knn(
    const LlevTimestampedTwedIndex* index,
    const LlevTimestampedSeriesView* query, size_t k,
    const LlevTemporalSearchLimits* limits,
    LlevTimestampedTwedMatch* out_matches, size_t capacity,
    size_t* out_len, uint32_t* out_reason);

/** Advance at most one bounded page. Zero matches with out_done=0 means
 * paused. LIMIT_EXCEEDED leaves previous matches an exact incomplete subset;
 * out_reason uses the temporal index reason codes, including 14 for a page
 * too small to advance. Output storage is caller owned. */
LLEV_API LlevStatus llev_timestamped_twed_cursor_next_batch(
    LlevTimestampedTwedCursor* cursor, LlevTimestampedTwedMatch* out_matches,
    size_t capacity, size_t page_work_units, size_t page_results,
    size_t* out_len, uint8_t* out_done, uint32_t* out_reason);
LLEV_API void llev_timestamped_twed_cursor_free(LlevTimestampedTwedCursor* cursor);

/** One copied exact match; DTW uses root-distance units. */
typedef struct LlevTemporalIndexMatch {
    uint64_t id;
    double distance;
} LlevTemporalIndexMatch;

typedef struct LlevTemporalIndex LlevTemporalIndex;
typedef struct LlevTemporalIndexCursor LlevTemporalIndexCursor;

/** Build, mutate, freeze, and release an index. Mutation requires exclusive
 * access. A frozen index may start independent concurrent cursors. A cursor
 * retains its immutable snapshot even after the index handle is freed. */
LLEV_API LlevStatus llev_temporal_index_new(
    const LlevTemporalIndexConfig* config, LlevTemporalIndex** out_index);
LLEV_API LlevStatus llev_temporal_index_insert(
    LlevTemporalIndex* index, uint64_t id,
    const double* samples, size_t len);
LLEV_API LlevStatus llev_temporal_index_freeze(LlevTemporalIndex* index);
LLEV_API void llev_temporal_index_free(LlevTemporalIndex* index);

/** Start a bounded exact range query. query is copied into cursor state;
 * a null pointer is allowed only with zero query_len. cutoff is inclusive
 * and nonnegative. Every limit field is a cumulative ceiling. */
LLEV_API LlevStatus llev_temporal_index_query_range(
    const LlevTemporalIndex* index, const double* query, size_t query_len,
    double cutoff, const LlevTemporalSearchLimits* limits,
    LlevTemporalIndexCursor** out_cursor);

/** Advance at most one native page. Zero output with out_done=0 is a valid
 * paused page and must be retried. LIMIT_EXCEEDED means the exact subset
 * produced earlier is incomplete; it never proves absence. Caller owns the
 * output array and must provide positive capacity and page budgets.
 * out_reason is zero on success; LIMIT_EXCEEDED uses 1 DP cells, 2 work,
 * 3 scratch, 4 trie nodes, 5 trie edges, 6 candidates, 7 results, 8 queue,
 * 9 continuation bytes, 10 overflow/other, 11 invalid stored data,
 * 12 unsupported, 13 allocation, 14 page too small, or 15 cancellation. */
LLEV_API LlevStatus llev_temporal_index_cursor_next_batch(
    LlevTemporalIndexCursor* cursor, LlevTemporalIndexMatch* out_matches,
    size_t capacity, size_t page_work_units, size_t page_results,
    size_t* out_len, uint8_t* out_done, uint32_t* out_reason);
LLEV_API void llev_temporal_index_cursor_free(LlevTemporalIndexCursor* cursor);

/** Exact, fail-closed top-k query from a frozen scalar index. Native search
 * scans every stored candidate under cumulative limits before returning a
 * cursor over the complete result, ordered by score and stable index order.
 * DTW matches use public root-distance units. On LIMIT_EXCEEDED no cursor is
 * returned and out_reason uses the temporal reason codes above. A successful
 * cursor remains valid after index free. */
typedef struct LlevTemporalKnnCursor LlevTemporalKnnCursor;
LLEV_API LlevStatus llev_temporal_index_query_knn(
    const LlevTemporalIndex* index, const double* query, size_t query_len,
    size_t k, const LlevTemporalSearchLimits* limits,
    LlevTemporalKnnCursor** out_cursor, uint32_t* out_reason);
/** Copy one page of complete top-k results in native order. Capacity must
 * be positive. out_done=1 means all results were copied. */
LLEV_API LlevStatus llev_temporal_knn_cursor_next_batch(
    LlevTemporalKnnCursor* cursor, LlevTemporalIndexMatch* out_matches,
    size_t capacity, size_t* out_len, uint8_t* out_done);
LLEV_API void llev_temporal_knn_cursor_free(LlevTemporalKnnCursor* cursor);

/** All ceilings apply to one complete exact range certificate. The search
 * ceilings include query, traversal, and result resources; witness bytes,
 * record count, path bytes, and certificate work are cumulative. */
typedef struct LlevTemporalCertificateLimits {
    LlevTemporalSearchLimits search;
    size_t max_witness_bytes;
    size_t max_records;
    size_t max_path_bytes;
    size_t max_work_units;
} LlevTemporalCertificateLimits;

/** Exact native certificate shape and charged resources. DTW cutoff is in
 * squared-distance units, even though public matches use root distance. */
typedef struct LlevTemporalCertificateInfo {
    size_t query_len;
    size_t evidence_len;
    size_t result_len;
    double cutoff_native;
    size_t work_units;
    size_t path_bytes;
    size_t witness_bytes;
    uint8_t snapshot_present;
    uint8_t reserved[7];
    uint8_t snapshot_identity[32];
} LlevTemporalCertificateInfo;

/** Ordered K1--K4 evidence. Kinds 1,2,3,4,5 denote prefix, subtree,
 * terminal, candidate prune, and exact candidate respectively. */
typedef struct LlevTemporalCertificateEvidenceHeader {
    uint32_t kind;
    uint32_t reserved;
    size_t path_len;
    uint64_t stable_id;
    double lower_bound;
    double exact;
    uint8_t has_exact;
    uint8_t survived;
    uint8_t reserved_tail[6];
} LlevTemporalCertificateEvidenceHeader;

/** Caller-owned projection for exact replay. Each path is path_len bytes. */
typedef struct LlevTemporalCertificateEvidenceView {
    LlevTemporalCertificateEvidenceHeader header;
    const uint8_t* path;
} LlevTemporalCertificateEvidenceView;

typedef struct LlevTemporalCertificateView {
    LlevTemporalCertificateInfo info;
    const uint64_t* query_bits;
    const LlevTemporalCertificateEvidenceView* evidence;
    const LlevTemporalIndexMatch* results;
} LlevTemporalCertificateView;

typedef struct LlevTemporalRangeCertificate LlevTemporalRangeCertificate;

/** Produce complete exact range evidence from a frozen scalar index. The
 * certificate owns its snapshot and remains valid after index free. No
 * certificate is returned on any limit or validation failure. */
LLEV_API LlevStatus llev_temporal_index_query_certified(
    const LlevTemporalIndex* index, const double* query, size_t query_len,
    double cutoff, const LlevTemporalCertificateLimits* limits,
    LlevTemporalRangeCertificate** out_certificate);
LLEV_API LlevStatus llev_temporal_certificate_info(
    const LlevTemporalRangeCertificate* certificate,
    LlevTemporalCertificateInfo* out_info);
/** Copy at most capacity elements starting at start; out_written is exact. */
LLEV_API LlevStatus llev_temporal_certificate_query_bits(
    const LlevTemporalRangeCertificate* certificate, size_t start,
    uint64_t* out_words, size_t capacity, size_t* out_written);
LLEV_API LlevStatus llev_temporal_certificate_matches(
    const LlevTemporalRangeCertificate* certificate, size_t start,
    LlevTemporalIndexMatch* out_matches, size_t capacity, size_t* out_written);
/** Read one decision; null out_path and zero capacity query path_len only.
 * Short nonzero buffers fail with LIMIT_EXCEEDED. */
LLEV_API LlevStatus llev_temporal_certificate_evidence_at(
    const LlevTemporalRangeCertificate* certificate, size_t index,
    LlevTemporalCertificateEvidenceHeader* out_header, uint8_t* out_path,
    size_t path_capacity, size_t* out_path_written);
/** Verify caller-supplied complete evidence and results against the retained
 * snapshot. An altered well-formed view returns OK with out_valid=0. */
LLEV_API LlevStatus llev_temporal_certificate_verify(
    const LlevTemporalRangeCertificate* certificate,
    const LlevTemporalCertificateView* view, uint8_t* out_valid);
LLEV_API void llev_temporal_certificate_free(
    LlevTemporalRangeCertificate* certificate);

/** Fixed-query online temporal automaton. It retains bounded query and
 * frontier state independent of target stream length. MSM, ERP, unit-grid
 * TWED, banded DTW, and scalar Fréchet are supported. Soft-DTW is not.
 * Non-ERP kernels require a finite inclusive cutoff. DTW cutoff and scores
 * are in squared-distance units, matching the native online kernel. */
typedef struct LlevTemporalOnlineAutomaton LlevTemporalOnlineAutomaton;

/** Hard query, frontier, per-sample work, and scratch ceilings. Defaults are
 * 1,000,000; 1,000,001; 100,000,000; and 256 MiB, respectively. */
typedef struct LlevTemporalOnlineLimits {
    size_t max_query_len;
    size_t max_frontier_positions;
    size_t max_step_work_units;
    size_t max_scratch_bytes;
} LlevTemporalOnlineLimits;

/** Exact observation for the committed target prefix. Optional scores are
 * valid only when their corresponding has_* byte is one. */
typedef struct LlevTemporalOnlineObservation {
    size_t consumed_target_len;
    size_t active_positions;
    double distance_within_cutoff;
    double minimum_active_cost;
    uint8_t has_distance;
    uint8_t has_minimum;
    uint8_t reserved[6];
} LlevTemporalOnlineObservation;

/** kind=0 commits the sample and carries an exact observation. kind=1 is
 * incomplete: the sample was not consumed and observation is zeroed. reason
 * uses the indexed temporal codes 1-13 and 15; zero means complete. */
typedef struct LlevTemporalOnlineStep {
    LlevTemporalOnlineObservation observation;
    uint32_t kind;
    uint32_t reason;
    size_t dp_cells;
    size_t work_units;
    size_t scratch_bytes;
    size_t queue_entries;
} LlevTemporalOnlineStep;

/** Construction copies the finite query and preflights all configured
 * ceilings. Each advance needs exclusive handle access. An incomplete step
 * leaves the current observation unchanged; callers may read it separately.
 * A null query pointer is allowed only for zero query_len. */
LLEV_API LlevStatus llev_temporal_online_new(
    const double* query, size_t query_len,
    const LlevTemporalConfig* config,
    const LlevTemporalOnlineLimits* limits,
    LlevTemporalOnlineAutomaton** out_machine);
LLEV_API LlevStatus llev_temporal_online_observation(
    const LlevTemporalOnlineAutomaton* machine,
    LlevTemporalOnlineObservation* out_observation);
LLEV_API LlevStatus llev_temporal_online_advance(
    LlevTemporalOnlineAutomaton* machine, double sample,
    LlevTemporalOnlineStep* out_step);
LLEV_API LlevStatus llev_temporal_online_scratch_bytes(
    const LlevTemporalOnlineAutomaton* machine, size_t* out_bytes);
LLEV_API void llev_temporal_online_free(LlevTemporalOnlineAutomaton* machine);

/** Copy a typed, untimestamped query into a bounded online vector Fréchet
 * machine. The machine owns a metric copy and can outlive its metric handle.
 * Subsequent operations on one machine require exclusive access. */
LLEV_API LlevStatus llev_vector_frechet_online_new(
    const LlevVectorMetric* metric,
    const LlevVectorSeriesView* query,
    double cutoff,
    const LlevTemporalOnlineLimits* limits,
    LlevVectorFrechetOnline** out_machine);

/** Copy an untimestamped query into an online Fréchet machine using ground
 * 1 L1, 2 L2, or 3 L-infinity. Use the common observation, advance, scratch,
 * and free functions below; each machine requires exclusive access. */
LLEV_API LlevStatus llev_vector_frechet_ground_online_new(
    uint32_t ground, const LlevVectorSeriesView* query, double cutoff,
    const LlevTemporalOnlineLimits* limits,
    LlevVectorFrechetOnline** out_machine);

/** Observe a committed vector-target prefix without advancing. */
LLEV_API LlevStatus llev_vector_frechet_online_observation(
    const LlevVectorFrechetOnline* machine,
    LlevTemporalOnlineObservation* out_observation);

/** Advance by one whole vector point. A resource-incomplete step does not
 * consume the point; LlevTemporalOnlineStep carries its exact stop reason. */
LLEV_API LlevStatus llev_vector_frechet_online_advance(
    LlevVectorFrechetOnline* machine,
    const double* point, size_t point_len,
    LlevTemporalOnlineStep* out_step);

/** Query fixed retained logical bytes and release the machine. */
LLEV_API LlevStatus llev_vector_frechet_online_scratch_bytes(
    const LlevVectorFrechetOnline* machine, size_t* out_bytes);
LLEV_API void llev_vector_frechet_online_free(
    LlevVectorFrechetOnline* machine);

/** Closed coordinate interval in one fixed-channel vector box. */
typedef struct LlevVectorInterval {
    double low;
    double high;
} LlevVectorInterval;

/** Typed vector box with a physical-time interval. Unit codes match scalar
 * timestamped TWED: 1 seconds, 2 milliseconds, 3 microseconds, 4 nanoseconds.
 * Reserved must be zero. */
typedef struct LlevTimestampedVectorBoxView {
    const LlevVectorInterval* coordinates;
    size_t dimension;
    double time_low;
    double time_high;
    uint32_t unit;
    uint32_t reserved;
} LlevTimestampedVectorBoxView;

/** Native fixed-channel K1 bounds. Limits cap constructed interval storage. */
LLEV_API LlevStatus llev_vector_point_box_lower_bound(
    const LlevVectorMetric* metric,
    const double* coordinates, size_t dimension,
    const LlevVectorInterval* intervals, size_t interval_count,
    const LlevVectorTemporalLimits* limits, double* out_bound);
LLEV_API LlevStatus llev_vector_box_box_lower_bound(
    const LlevVectorMetric* metric,
    const LlevVectorInterval* left, size_t left_len,
    const LlevVectorInterval* right, size_t right_len,
    const LlevVectorTemporalLimits* limits, double* out_bound);

/** Native K4 bound for ERP, banded DTW, Fréchet, or timestamped TWED.
 * Config uses the corresponding vector score algorithm and positive infinity
 * cutoff. Input copies are bounded by limits. */
LLEV_API LlevStatus llev_vector_temporal_candidate_lower_bound(
    const LlevVectorMetric* metric,
    const LlevVectorSeriesView* left,
    const LlevVectorSeriesView* right,
    const LlevVectorTemporalConfig* config,
    const LlevVectorTemporalLimits* limits,
    double* out_bound);

/** Native timestamped vector TWED K1 interval bound. Mode 1 deletes two
 * consecutive candidate boxes and requires null query points and zero query
 * dimension; mode 2 matches two exact query points to the boxes. Config uses
 * algorithm 7, positive infinity cutoff, sentinel, stiffness, and gap cost. */
LLEV_API LlevStatus llev_vector_twed_interval_lower_bound(
    const LlevVectorMetric* metric, uint32_t mode,
    const double* query_current, const double* query_previous,
    size_t query_dimension,
    double query_current_time, double query_previous_time,
    const LlevTimestampedVectorBoxView* candidate_current,
    const LlevTimestampedVectorBoxView* candidate_previous,
    const LlevVectorTemporalConfig* config,
    const LlevVectorTemporalLimits* limits,
    double* out_bound);

#ifdef __cplusplus
}
#endif

#endif /* LIBLEVENSHTEIN_H */
