/* Stable resource ABI for modular vinary-tree language bindings. */
#ifndef VINARY_TREE_INTEROP_H
#define VINARY_TREE_INTEROP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VT_ABI_VERSION 1u
#define VT_DICTIONARY_INTERFACE_VERSION 1u
#define VT_WFST_INTERFACE_VERSION 1u
#define VT_RECOMMENDED_EDGE_BATCH 256u
#define VT_RECOMMENDED_ARC_BATCH 256u

#define VT_DICTIONARY_FLAG_PARALLEL_REENTRANT UINT64_C(1)
#define VT_DICTIONARY_FLAG_SUFFIX_BASED UINT64_C(2)
#define VT_DICTIONARY_FLAG_IMMUTABLE UINT64_C(4)

typedef enum VtStatus {
    VT_STATUS_OK = 0,
    VT_STATUS_END = 1,
    VT_STATUS_INVALID_ARGUMENT = 2,
    VT_STATUS_NULL_POINTER = 3,
    VT_STATUS_UNSUPPORTED = 4,
    VT_STATUS_IO_ERROR = 5,
    VT_STATUS_CLOSED = 6,
    VT_STATUS_LIMIT_EXCEEDED = 7,
    VT_STATUS_PROVIDER_ERROR = 8
} VtStatus;

typedef struct VtInterfaceId { uint8_t bytes[16]; } VtInterfaceId;

struct VtResourceVTable;
typedef struct VtResource {
    void* context;
    const struct VtResourceVTable* vtable;
} VtResource;

typedef struct VtResourceVTable {
    size_t struct_size;
    uint32_t abi_version;
    uint32_t reserved;
    void (*retain)(void* context);
    void (*release)(void* context);
    VtStatus (*query_interface)(void* context,
                                const VtInterfaceId* interface_id,
                                uint32_t minimum_version,
                                const void** out_vtable);
} VtResourceVTable;

typedef enum VtUnitDomain {
    VT_UNIT_DOMAIN_BYTE = 1,
    VT_UNIT_DOMAIN_UNICODE_SCALAR = 2,
    VT_UNIT_DOMAIN_U64 = 3
} VtUnitDomain;

typedef enum VtValueDomain {
    VT_VALUE_DOMAIN_UNIT = 0,
    VT_VALUE_DOMAIN_OPTIONAL_U64 = 1,
    VT_VALUE_DOMAIN_BYTES = 2
} VtValueDomain;

typedef struct VtOptionalU64 {
    uint64_t value;
    uint8_t has_value;
    uint8_t reserved[7];
} VtOptionalU64;

typedef struct VtDictionaryEdge {
    uint64_t label;
    uint64_t node;
} VtDictionaryEdge;

typedef struct VtDictionaryVTable {
    size_t struct_size;
    uint32_t interface_version;
    VtUnitDomain unit_domain;
    VtValueDomain value_domain;
    uint64_t flags;
    VtStatus (*snapshot)(void* context, VtResource* out_snapshot);
    VtStatus (*root)(void* context, uint64_t* out_node);
    VtStatus (*len)(void* context, size_t* out_len, uint8_t* out_known);
    VtStatus (*node_is_final)(void* context, uint64_t node,
                              uint8_t* out_is_final);
    VtStatus (*node_value_u64)(void* context, uint64_t node,
                               VtOptionalU64* out_value);
    VtStatus (*node_transition)(void* context, uint64_t node, uint64_t label,
                                uint64_t* out_child, uint8_t* out_found);
    VtStatus (*node_edges)(void* context, uint64_t node, size_t start,
                           VtDictionaryEdge* out_edges, size_t capacity,
                           size_t* out_written, size_t* out_total);
} VtDictionaryVTable;

/* Scalar semirings with a portable f64 representation. */
typedef enum VtWeightDomain {
    VT_WEIGHT_DOMAIN_TROPICAL_F64 = 1,
    VT_WEIGHT_DOMAIN_LOG_F64 = 2,
    VT_WEIGHT_DOMAIN_PROBABILITY_F64 = 3,
    VT_WEIGHT_DOMAIN_ARCTIC_F64 = 4,
    VT_WEIGHT_DOMAIN_SIGNED_TROPICAL_F64 = 5,
    VT_WEIGHT_DOMAIN_COUNT_F64 = 6,
    VT_WEIGHT_DOMAIN_BOOLEAN_F64 = 7
} VtWeightDomain;

#define VT_WFST_FLAG_PARALLEL_REENTRANT UINT64_C(1)
#define VT_WFST_FLAG_IMMUTABLE UINT64_C(2)
#define VT_WFST_FLAG_LAZY UINT64_C(4)
#define VT_WFST_FLAG_ACYCLIC UINT64_C(8)

/* One arc in a caller-owned page. Epsilon labels have has_* == 0. */
typedef struct VtWfstArc {
    uint64_t input_label;
    uint64_t output_label;
    uint64_t target_state;
    double weight;
    uint8_t has_input;
    uint8_t has_output;
    uint8_t reserved[6];
} VtWfstArc;

/*
 * Versioned immutable scalar-WFST interface. State identifiers are scoped to
 * the retained snapshot. Implementations may expand states lazily, but a
 * provider advertising PARALLEL_REENTRANT must do so safely without imposing
 * a process-wide or resource-wide sequential call gate.
 */
typedef struct VtWfstVTable {
    size_t struct_size;
    uint32_t interface_version;
    VtUnitDomain unit_domain;
    VtWeightDomain weight_domain;
    uint32_t reserved;
    uint64_t flags;
    VtStatus (*snapshot)(void* context, VtResource* out_snapshot);
    VtStatus (*start)(void* context, uint64_t* out_state);
    VtStatus (*num_states)(void* context, size_t* out_count,
                           uint8_t* out_known);
    VtStatus (*state_info)(void* context, uint64_t state,
                           uint8_t* out_valid, uint8_t* out_is_final,
                           double* out_final_weight);
    VtStatus (*state_arcs)(void* context, uint64_t state, size_t start,
                           VtWfstArc* out_arcs, size_t capacity,
                           size_t* out_written, size_t* out_total);
} VtWfstVTable;

static const VtInterfaceId VT_DICTIONARY_INTERFACE_ID = {
    { 'v','t','.','d','i','c','t','i','o','n','a','r','y','.','v','1' }
};

static const VtInterfaceId VT_WFST_INTERFACE_ID = {
    { 'v','t','.','s','c','a','l','a','r','-','w','f','s','t','.','1' }
};

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus)
static_assert(sizeof(VtResource) == 2 * sizeof(void*),
              "VtResource must remain a two-word handle");
#endif

#endif /* VINARY_TREE_INTEROP_H */
