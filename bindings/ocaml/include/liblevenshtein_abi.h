/* Generated from bindings/api.json; do not edit numeric values manually. */
#ifndef LIBLEVENSHTEIN_ABI_H
#define LIBLEVENSHTEIN_ABI_H

#include <stddef.h>
#include <stdint.h>
#ifndef VT_INTEROP_HEADER
#define VT_INTEROP_HEADER "vinary_tree_interop.h"
#endif
#include VT_INTEROP_HEADER

#define LLEV_ABI_VERSION 1u
#define LLEV_API_REVISION 1u
#define LLEV_DEFAULT_MATCH_BATCH 256u

#define LLEV_BUILD_FEATURE_CORE UINT64_C(1)
#define LLEV_BUILD_FEATURE_PHONETIC UINT64_C(2)

typedef enum LlevStatus {
    LLEV_STATUS_OK = 0,
    LLEV_STATUS_END = 1,
    LLEV_STATUS_INVALID_ARGUMENT = 2,
    LLEV_STATUS_INVALID_UTF8 = 3,
    LLEV_STATUS_NULL_POINTER = 4,
    LLEV_STATUS_PANIC = 5,
    LLEV_STATUS_UNSUPPORTED = 6,
    LLEV_STATUS_IO_ERROR = 7,
    LLEV_STATUS_CLOSED = 8,
    LLEV_STATUS_LIMIT_EXCEEDED = 9,
    LLEV_STATUS_PROVIDER_ERROR = 10,
    LLEV_STATUS_BATCH_IN_USE = 11,
    LLEV_STATUS_DOMAIN_MISMATCH = 12
} LlevStatus;

typedef enum LlevAlgorithm {
    LLEV_ALGORITHM_STANDARD = 0,
    LLEV_ALGORITHM_TRANSPOSITION = 1,
    LLEV_ALGORITHM_MERGE_AND_SPLIT = 2,
    LLEV_ALGORITHM_DAMERAU_LEVENSHTEIN = 3
} LlevAlgorithm;

typedef enum LlevQueryOrder {
    LLEV_QUERY_ORDER_TRAVERSAL = 0,
    LLEV_QUERY_ORDER_DISTANCE_THEN_TERM = 1
} LlevQueryOrder;

typedef enum LlevPhoneticRuleSetKind {
    LLEV_PHONETIC_RULE_SET_ENGLISH_ORTHOGRAPHY = 0,
    LLEV_PHONETIC_RULE_SET_ENGLISH_PHONETIC = 1
} LlevPhoneticRuleSetKind;

typedef struct LlevTransducer LlevTransducer;
typedef struct LlevQueryCursor LlevQueryCursor;
typedef struct LlevPhoneticPattern LlevPhoneticPattern;
typedef struct LlevPhoneticRuleSet LlevPhoneticRuleSet;

typedef struct LlevMatch {
    const void* term_data;
    size_t term_len;
    size_t byte_len;
    size_t distance;
    uint64_t id;
    VtUnitDomain unit_domain;
    uint8_t has_id;
    uint8_t reserved[3];
} LlevMatch;

typedef struct LlevMatchBatchView {
    const LlevMatch* matches;
    size_t len;
    uint64_t generation;
} LlevMatchBatchView;

typedef struct LlevOwnedString {
    char* data;
    size_t len;
} LlevOwnedString;

typedef LlevStatus (*LlevBatchReducer)(void* context,
                                       const LlevMatch* matches,
                                       size_t len);

#endif /* LIBLEVENSHTEIN_ABI_H */
