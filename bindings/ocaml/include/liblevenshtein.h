/** @file
 * @brief Stable C API for distance functions and resource-backed fuzzy queries.
 *
 * Every fallible resource operation returns LlevStatus and copies its diagnostic
 * into a thread-local slot. Query cursors capture an immutable dictionary
 * revision and lend bounded batches whose generation must be released exactly.
 * The complete status, ownership, concurrency, and complexity contract is
 * available in the versioned package guide and is summarized per declaration
 * below.
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

#ifdef __cplusplus
}
#endif

#endif /* LIBLEVENSHTEIN_H */
