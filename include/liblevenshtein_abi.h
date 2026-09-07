/** @file
 * @brief Generated from bindings/api.json; do not edit numeric values manually.
 */
#ifndef LIBLEVENSHTEIN_ABI_H
#define LIBLEVENSHTEIN_ABI_H

#include <stddef.h>
#include <stdint.h>
#ifndef VT_INTEROP_HEADER
/** Overrideable include spelling for the shared resource-ABI header. */
#define VT_INTEROP_HEADER "vinary_tree_interop.h"
#endif
#include VT_INTEROP_HEADER

/** Binary ABI generation implemented by this header and library. */
#define LLEV_ABI_VERSION 1u
/** Additive API revision within LLEV_ABI_VERSION. */
#define LLEV_API_REVISION 5u
/** Default maximum descriptors borrowed by one cursor batch. */
#define LLEV_DEFAULT_MATCH_BATCH 256u

/** Build-feature bit: Core distance, transducer, cursor, and batch surface. */
#define LLEV_BUILD_FEATURE_CORE UINT64_C(1)
/** Build-feature bit: Compiled phonetic patterns and rewrite-rule sets. */
#define LLEV_BUILD_FEATURE_PHONETIC UINT64_C(2)

/** Result of a fallible native operation. */
typedef enum LlevStatus {
    LLEV_STATUS_OK = 0, /**< The operation completed successfully. */
    LLEV_STATUS_END = 1, /**< A finite cursor reached the end of its stream. */
    LLEV_STATUS_INVALID_ARGUMENT = 2, /**< An argument violated the operation's contract. */
    LLEV_STATUS_INVALID_UTF8 = 3, /**< Input advertised as text was not valid UTF-8. */
    LLEV_STATUS_NULL_POINTER = 4, /**< A required native pointer was null. */
    LLEV_STATUS_PANIC = 5, /**< A contained Rust panic crossed the failure boundary. */
    LLEV_STATUS_UNSUPPORTED = 6, /**< The requested capability is unavailable in this build. */
    LLEV_STATUS_IO_ERROR = 7, /**< An input/output operation failed. */
    LLEV_STATUS_CLOSED = 8, /**< The target resource was already closed. */
    LLEV_STATUS_LIMIT_EXCEEDED = 9, /**< A configured resource or traversal limit was exceeded. */
    LLEV_STATUS_PROVIDER_ERROR = 10, /**< A foreign dictionary provider reported a failure. */
    LLEV_STATUS_BATCH_IN_USE = 11, /**< A cursor was advanced while its previous batch remained borrowed. */
    LLEV_STATUS_DOMAIN_MISMATCH = 12 /**< The query and dictionary use different unit domains. */
} LlevStatus;

/** Edit-distance algorithm. */
typedef enum LlevAlgorithm {
    LLEV_ALGORITHM_STANDARD = 0, /**< Standard insert/delete/substitute distance. */
    LLEV_ALGORITHM_TRANSPOSITION = 1, /**< Optimal string alignment with adjacent transposition. */
    LLEV_ALGORITHM_MERGE_AND_SPLIT = 2, /**< Merge-and-split edit distance. */
    LLEV_ALGORITHM_DAMERAU_LEVENSHTEIN = 3 /**< Unrestricted Damerau-Levenshtein distance. */
} LlevAlgorithm;

/** Lazy result ordering. */
typedef enum LlevQueryOrder {
    LLEV_QUERY_ORDER_TRAVERSAL = 0, /**< Provider traversal order with bounded buffering. */
    LLEV_QUERY_ORDER_DISTANCE_THEN_TERM = 1 /**< Distance then term, buffering at most one distance layer. */
} LlevQueryOrder;

/** Built-in phonetic rewrite-rule set. */
typedef enum LlevPhoneticRuleSetKind {
    LLEV_PHONETIC_RULE_SET_ENGLISH_ORTHOGRAPHY = 0, /**< English orthography normalization. */
    LLEV_PHONETIC_RULE_SET_ENGLISH_PHONETIC = 1 /**< English phonetic transformation. */
} LlevPhoneticRuleSetKind;

/** Runtime generalized-operation applicability predicate. */
typedef enum LlevOperationApplicability {
    LLEV_OPERATION_APPLICABILITY_ANY = 0, /**< Apply without inspecting consumed units. */
    LLEV_OPERATION_APPLICABILITY_EQUAL = 1, /**< Apply only when the consumed source and target slices are equal. */
    LLEV_OPERATION_APPLICABILITY_ADJACENT_TRANSPOSE = 2, /**< Apply only to an adjacent two-unit transposition. */
    LLEV_OPERATION_APPLICABILITY_LISTED = 3 /**< Apply only to a configured directional source/target pair. */
} LlevOperationApplicability;

/** Universal edit-automaton variant. */
typedef enum LlevUniversalVariant {
    LLEV_UNIVERSAL_STANDARD = 0, /**< Standard insert/delete/substitute universal automaton. */
    LLEV_UNIVERSAL_TRANSPOSITION = 1, /**< Universal automaton with adjacent transposition. */
    LLEV_UNIVERSAL_MERGE_AND_SPLIT = 2 /**< Universal automaton with merge-and-split edits. */
} LlevUniversalVariant;

/** Opaque, shareable automaton configuration retaining a dictionary resource. */
typedef struct LlevTransducer LlevTransducer;
/** Opaque, exclusive lazy traversal over one immutable query-start snapshot. */
typedef struct LlevQueryCursor LlevQueryCursor;
/** Opaque, exclusive bounded complete-query cache. */
typedef struct LlevQueryCache LlevQueryCache;
/** Opaque, immutable compiled phonetic-language automaton. */
typedef struct LlevPhoneticPattern LlevPhoneticPattern;
/** Opaque, immutable compiled phonetic rewrite-rule set. */
typedef struct LlevPhoneticRuleSet LlevPhoneticRuleSet;
/** Opaque, immutable runtime-configured generalized automaton. */
typedef struct LlevGeneralizedAutomaton LlevGeneralizedAutomaton;
/** Opaque, exclusive generalized online prefix state. */
typedef struct LlevGeneralizedOnlineAutomaton LlevGeneralizedOnlineAutomaton;
/** Opaque, immutable universal automaton and substitution policy. */
typedef struct LlevUniversalAutomaton LlevUniversalAutomaton;
/** Opaque, exclusive universal online prefix state. */
typedef struct LlevUniversalOnlineAutomaton LlevUniversalOnlineAutomaton;

/** Hard ceilings for standalone online and complete evaluation. */
typedef struct LlevAutomatonLimits {
    size_t max_source_units; /**< Maximum source units. */
    size_t max_target_units; /**< Maximum committed target units. */
    size_t max_retained_cells; /**< Generalized row-ring and scratch ceiling. */
    size_t max_step_work_units; /**< Generalized relaxations per target unit. */
} LlevAutomatonLimits;

/** One borrowed directional source/target restriction for a listed operation. */
typedef struct LlevGeneralizedRestriction {
    const char* source_data; /**< Borrowed UTF-8 source bytes. */
    size_t source_len; /**< Source byte length. */
    const char* target_data; /**< Borrowed UTF-8 target bytes. */
    size_t target_len; /**< Target byte length. */
} LlevGeneralizedRestriction;

/** One borrowed runtime generalized edit operation. */
typedef struct LlevGeneralizedOperation {
    size_t consume_source; /**< Unicode source scalars consumed. */
    size_t consume_target; /**< Unicode target scalars consumed. */
    double weight; /**< Non-negative finite decimal cost. */
    const char* name_data; /**< Borrowed non-empty UTF-8 diagnostic name. */
    size_t name_len; /**< Name byte length. */
    uint32_t applicability; /**< One LlevOperationApplicability value. */
    uint32_t reserved; /**< Must be zero. */
    const LlevGeneralizedRestriction* restrictions; /**< LISTED pairs. */
    size_t restriction_count; /**< Number of restriction descriptors. */
} LlevGeneralizedOperation;

/** One directional zero-cost universal source/target substitution pair. */
typedef struct LlevUniversalEquivalence {
    uint64_t source; /**< Dictionary/source unit. */
    uint64_t target; /**< Query/target unit. */
} LlevUniversalEquivalence;

/** Exact observation of a generalized target prefix. */
typedef struct LlevGeneralizedObservation {
    size_t consumed_target_len; /**< Committed Unicode target scalars. */
    size_t active_positions; /**< In-budget cells in the current generation. */
    size_t scaled_distance; /**< Fixed-point numerator when present. */
    uint32_t scale_denominator; /**< Shared exact decimal denominator. */
    uint8_t current_row_nonempty; /**< Not a permanent-death signal. */
    uint8_t accepting; /**< Complete source is currently in budget. */
    uint8_t has_distance; /**< scaled_distance is present. */
    uint8_t reserved; /**< Must be ignored; fixed to zero. */
} LlevGeneralizedObservation;

/** Exact observation of a universal target prefix. */
typedef struct LlevUniversalObservation {
    size_t consumed_target_len; /**< Committed target units. */
    size_t source_len; /**< Bound source units. */
    uint8_t alive; /**< Universal frontier is non-empty. */
    uint8_t accepting; /**< Current prefix is accepted. */
    uint8_t reserved[6]; /**< Must be ignored; fixed to zero. */
} LlevUniversalObservation;

/** One borrowed match descriptor inside a live cursor batch lease. */
typedef struct LlevMatch {
    const void* term_data; /**< Borrowed term bytes or aligned u64 tokens. */
    size_t term_len; /**< Logical unit count: scalars, bytes, or u64 tokens. */
    size_t byte_len; /**< Physical bytes addressed by term_data. */
    size_t distance; /**< Exact edit distance from the query. */
    uint64_t id; /**< Dictionary value when has_id is one. */
    VtUnitDomain unit_domain; /**< Encoding and alignment of term_data. */
    uint8_t has_id; /**< One when id is present, otherwise zero. */
    uint8_t reserved[3]; /**< Must be ignored; reserved for additive evolution. */
} LlevMatch;

/** Borrowed contiguous match lease returned by cursor advancement. */
typedef struct LlevMatchBatchView {
    const LlevMatch* matches; /**< Descriptor array valid until exact release. */
    size_t len; /**< Number of initialized descriptors. */
    uint64_t generation; /**< Nonzero identity required by release_batch. */
} LlevMatchBatchView;

/** Aggregate query-cache policy and residency counters. */
typedef struct LlevQueryCacheStats {
    uint64_t requests; /**< Total cached-query requests. */
    uint64_t hits; /**< Requests served by resident immutable results. */
    uint64_t misses; /**< Requests whose exact product walk ran. */
    uint64_t admissions; /**< Computed results admitted to residency. */
    uint64_t rejections; /**< Computed results rejected by limits or admission. */
    uint64_t evictions; /**< Resident results displaced by SIEVE. */
    size_t resident_entries; /**< Entries across both result-order shards. */
    size_t resident_weight; /**< Logical weight across both shards. */
} LlevQueryCacheStats;

/** Heap-owned, length-bearing UTF-8 returned by the phonetic-rule API. */
typedef struct LlevOwnedString {
    char* data; /**< Owned bytes, not necessarily NUL-terminated. */
    size_t len; /**< Number of initialized UTF-8 bytes. */
} LlevOwnedString;

/** Consume one borrowed reducer batch on the calling thread.
 * @param context unchanged caller context supplied to cursor_reduce
 * @param matches descriptors valid only for this callback invocation
 * @param len number of descriptors in matches
 * @return OK to continue, END to stop successfully, or another published status to abort
 */
typedef LlevStatus (*LlevBatchReducer)(void* context,
                                       const LlevMatch* matches,
                                       size_t len);

#endif /* LIBLEVENSHTEIN_ABI_H */
