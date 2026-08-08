/* Stable C API for liblevenshtein's project-owned binding surface. */
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

LLEV_API uint32_t llev_abi_version(void);
LLEV_API uint32_t llev_api_revision(void);
LLEV_API uint64_t llev_build_features(void);
LLEV_API const char* llev_last_error_message(void);

LLEV_API size_t llev_distance(const char* source, size_t source_len,
                              const char* target, size_t target_len);
LLEV_API size_t llev_distance_threshold(const char* source, size_t source_len,
                                        const char* target, size_t target_len,
                                        size_t threshold);
LLEV_API size_t llev_damerau_distance(const char* source, size_t source_len,
                                      const char* target, size_t target_len);
LLEV_API size_t llev_damerau_distance_threshold(const char* source,
                                                size_t source_len,
                                                const char* target,
                                                size_t target_len,
                                                size_t threshold);
LLEV_API size_t llev_true_damerau_distance(const char* source,
                                           size_t source_len,
                                           const char* target,
                                           size_t target_len);
LLEV_API size_t llev_true_damerau_distance_threshold(const char* source,
                                                     size_t source_len,
                                                     const char* target,
                                                     size_t target_len,
                                                     size_t threshold);

/* Legacy allocation helpers retained only for their independent string ABI. */
LLEV_API void llev_string_free(char* value);
LLEV_API void llev_string_array_free(char** values, size_t len);
LLEV_API char* llev_string_dup(const char* value);

/**
 * Retain a live dictionary resource and construct an automaton configuration.
 * Dictionary construction and CRUD are intentionally supplied by
 * libdictenstein, not by this library.
 */
LLEV_API LlevStatus llev_transducer_new(const VtResource* dictionary,
                                        uint32_t algorithm,
                                        LlevTransducer** out_transducer);
LLEV_API void llev_transducer_free(LlevTransducer* transducer);
LLEV_API LlevStatus llev_transducer_unit_domain(
    const LlevTransducer* transducer,
    VtUnitDomain* out_domain);

/** Capture the provider revision now and start a lazy Unicode query. */
LLEV_API LlevStatus llev_transducer_query_utf8(
    const LlevTransducer* transducer,
    const char* query,
    size_t query_len,
    size_t max_distance,
    uint32_t order,
    LlevQueryCursor** out_cursor);

/** Capture the provider revision now and start a lazy raw-byte query. */
LLEV_API LlevStatus llev_transducer_query_bytes(
    const LlevTransducer* transducer,
    const uint8_t* query,
    size_t query_len,
    size_t max_distance,
    uint32_t order,
    LlevQueryCursor** out_cursor);

/** Capture the provider revision now and start a lazy u64-token query. */
LLEV_API LlevStatus llev_transducer_query_u64(
    const LlevTransducer* transducer,
    const uint64_t* query,
    size_t query_len,
    size_t max_distance,
    uint32_t order,
    LlevQueryCursor** out_cursor);

/**
 * Borrow at most max_matches descriptors backed by cursor-owned contiguous
 * arenas. The returned generation must be released before advancing or closing
 * the cursor.
 */
LLEV_API LlevStatus llev_query_cursor_next_batch(
    LlevQueryCursor* cursor,
    size_t max_matches,
    LlevMatchBatchView* out_batch);
LLEV_API LlevStatus llev_query_cursor_release_batch(
    LlevQueryCursor* cursor,
    uint64_t generation);

/** Consume the remaining cursor with one callback per reusable batch. */
LLEV_API LlevStatus llev_query_cursor_reduce(
    LlevQueryCursor* cursor,
    size_t batch_size,
    LlevBatchReducer reducer,
    void* context,
    size_t* out_count);

/** Close a cursor; returns BATCH_IN_USE without closing when a lease is live. */
LLEV_API LlevStatus llev_query_cursor_free(LlevQueryCursor* cursor);

/* Optional project-owned phonetic automata and rewrite rules. */
LLEV_API LlevStatus llev_phonetic_pattern_compile_regex(
    const char* source, size_t source_len, LlevPhoneticPattern** out_pattern);
LLEV_API LlevStatus llev_phonetic_pattern_compile_llre(
    const char* source, size_t source_len, LlevPhoneticPattern** out_pattern);
LLEV_API void llev_phonetic_pattern_free(LlevPhoneticPattern* pattern);
LLEV_API LlevStatus llev_phonetic_pattern_size(
    const LlevPhoneticPattern* pattern,
    size_t* out_states,
    size_t* out_transitions);
LLEV_API LlevStatus llev_phonetic_pattern_matches(
    const LlevPhoneticPattern* pattern,
    const char* input,
    size_t input_len,
    uint8_t* out_matches);
LLEV_API LlevStatus llev_transducer_query_pattern(
    const LlevTransducer* transducer,
    const LlevPhoneticPattern* pattern,
    uint8_t max_distance,
    LlevQueryCursor** out_cursor);

LLEV_API LlevStatus llev_phonetic_rules_parse(
    const char* source, size_t source_len, LlevPhoneticRuleSet** out_rules);
LLEV_API LlevStatus llev_phonetic_rules_builtin(
    uint32_t kind, LlevPhoneticRuleSet** out_rules);
LLEV_API void llev_phonetic_rules_free(LlevPhoneticRuleSet* rules);
LLEV_API LlevStatus llev_phonetic_rules_len(
    const LlevPhoneticRuleSet* rules, size_t* out_len);
LLEV_API LlevStatus llev_phonetic_rules_apply(
    const LlevPhoneticRuleSet* rules,
    const char* input,
    size_t input_len,
    LlevOwnedString* out_text);
LLEV_API void llev_owned_string_free(LlevOwnedString* value);

#ifdef __cplusplus
}
#endif

#endif /* LIBLEVENSHTEIN_H */
