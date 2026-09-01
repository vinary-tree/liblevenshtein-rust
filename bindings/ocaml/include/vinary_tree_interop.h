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
#define VT_DICTIONARY_VISIT_INTERFACE_VERSION 1u
#define VT_DICTIONARY_GRAPH_INTERFACE_VERSION 1u
#define VT_DICTIONARY_ENTRIES_INTERFACE_VERSION 1u
#define VT_SNAPSHOT_IDENTITY_INTERFACE_VERSION 1u
#define VT_WFST_INTERFACE_VERSION 1u
#define VT_LATTICE_INTERFACE_VERSION 1u
#define VT_SEMIRING_INTERFACE_VERSION 1u
#define VT_SEMIRING_DIVISION_INTERFACE_VERSION 1u
#define VT_SEMIRING_STAR_INTERFACE_VERSION 1u
#define VT_SEMIRING_NUMERIC_INTERFACE_VERSION 1u
#define VT_SEMIRING_PROPERTIES_INTERFACE_VERSION 1u
#define VT_RECOMMENDED_EDGE_BATCH 256u
#define VT_RECOMMENDED_ARC_BATCH 256u
#define VT_RECOMMENDED_LATTICE_BATCH 256u
#define VT_RECOMMENDED_SEMIRING_BATCH 256u

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
    VT_STATUS_PROVIDER_ERROR = 8,
    VT_STATUS_BATCH_IN_USE = 9
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

/* Optional fused finality + edge-page inspection. */
typedef struct VtDictionaryVisitVTable {
    size_t struct_size;
    uint32_t interface_version;
    uint32_t reserved;
    VtStatus (*node_visit)(void* context, uint64_t node, size_t start,
                           uint8_t* out_is_final,
                           VtDictionaryEdge* out_edges, size_t capacity,
                           size_t* out_written, size_t* out_total);
} VtDictionaryVisitVTable;

/* One node in an immutable compact dictionary graph. The edge range indexes
 * VtDictionaryGraphView.edges. value_cursor is an opaque snapshot-local token
 * consumed only by VtDictionaryGraphVTable.node_value_u64. */
typedef struct VtDictionaryGraphNode {
    uint64_t edge_start;
    uint64_t edge_len;
    uint64_t value_cursor;
    uint8_t is_final;
    uint8_t reserved[7];
} VtDictionaryGraphNode;

/* One compact-graph edge. target is a zero-based node-array index. */
typedef struct VtDictionaryGraphEdge {
    uint64_t label;
    uint64_t target;
} VtDictionaryGraphEdge;

/* Borrowed immutable slices owned by the retained snapshot resource. Null is
 * permitted only for an empty slice. Providers must write reserved as zero. */
typedef struct VtDictionaryGraphView {
    const VtDictionaryGraphNode* nodes;
    size_t node_count;
    const VtDictionaryGraphEdge* edges;
    size_t edge_count;
    uint64_t root;
    uint64_t reserved;
} VtDictionaryGraphView;

/* Optional compact immutable snapshot-graph interface. It is legal only for
 * dictionary resources advertising VT_DICTIONARY_FLAG_IMMUTABLE. Returned
 * slices remain valid and unchanged while the resource is retained. */
typedef struct VtDictionaryGraphVTable {
    size_t struct_size;
    uint32_t interface_version;
    uint32_t reserved;
    VtStatus (*graph)(void* context, VtDictionaryGraphView* out_graph);
    VtStatus (*node_value_u64)(void* context, uint64_t value_cursor,
                               VtOptionalU64* out_value);
} VtDictionaryGraphVTable;

/* Optional process-local immutable producer/revision identity. */
typedef struct VtSnapshotIdentity {
    uint64_t producer;
    uint64_t revision;
} VtSnapshotIdentity;

typedef struct VtSnapshotIdentityVTable {
    size_t struct_size;
    uint32_t interface_version;
    uint32_t reserved;
    VtStatus (*identity)(void* context, VtSnapshotIdentity* out_identity);
} VtSnapshotIdentityVTable;

/* Ordering guaranteed by a dictionary-entries cursor. */
typedef enum VtDictionaryEntryOrder {
    VT_DICTIONARY_ENTRY_ORDER_LEXICOGRAPHIC = 1
} VtDictionaryEntryOrder;

#define VT_DICTIONARY_ENTRIES_INFO_FLAG_EXACT_LEN UINT64_C(1)
#define VT_DICTIONARY_ENTRIES_INFO_FLAG_SNAPSHOT_IDENTITY UINT64_C(2)

/* One entry descriptor into the parallel unit and optional-u64 arenas. All
 * offsets and lengths count arena elements, not bytes. */
typedef struct VtDictionaryEntry {
    size_t unit_offset;
    size_t unit_len;
    size_t value_offset;
    size_t value_len;
    uint64_t reserved;
} VtDictionaryEntry;

/* Hard upper bounds for one returned batch. max_entries must be nonzero. */
typedef struct VtDictionaryEntryBatchLimits {
    size_t max_entries;
    size_t max_units;
    size_t max_values;
    uint64_t reserved;
} VtDictionaryEntryBatchLimits;

/* Cursor-owned batch lease. Pointer element types are selected by unit_domain:
 * uint8_t for BYTE, uint32_t for UNICODE_SCALAR, and uint64_t for U64. */
typedef struct VtDictionaryEntryBatchView {
    const VtDictionaryEntry* entries;
    size_t entry_count;
    const void* units;
    size_t unit_count;
    const uint64_t* values;
    size_t value_count;
    uint64_t generation;
    uint64_t reserved;
} VtDictionaryEntryBatchView;

/* Immutable metadata captured with a cursor. Domain/order fields are raw
 * discriminants so consumers can reject unknown values before enum conversion. */
typedef struct VtDictionaryEntriesInfo {
    uint32_t unit_domain;
    uint32_t value_domain;
    uint32_t order;
    uint32_t reserved0;
    uint64_t flags;
    size_t exact_len;
    VtSnapshotIdentity identity;
    uint64_t reserved[2];
} VtDictionaryEntriesInfo;

struct VtDictionaryEntriesVTable;
typedef struct VtDictionaryEntriesCursor {
    void* context;
    const struct VtDictionaryEntriesVTable* vtable;
} VtDictionaryEntriesCursor;

typedef VtStatus (*VtDictionaryEntryReducer)(
    void* reducer_context,
    const VtDictionaryEntryBatchView* batch);

/* Optional finite lexicographic dictionary-entry stream. A cursor is a
 * move-only two-word owned handle; copying its words does not duplicate it. */
typedef struct VtDictionaryEntriesVTable {
    size_t struct_size;
    uint32_t interface_version;
    uint32_t reserved;
    VtStatus (*open)(void* resource_context,
                     VtDictionaryEntriesCursor* out_cursor,
                     VtDictionaryEntriesInfo* out_info);
    VtStatus (*next_batch)(VtDictionaryEntriesCursor* cursor,
                           const VtDictionaryEntryBatchLimits* limits,
                           VtDictionaryEntryBatchView* out_batch);
    VtStatus (*release_batch)(VtDictionaryEntriesCursor* cursor,
                              uint64_t generation);
    VtStatus (*reduce)(VtDictionaryEntriesCursor* cursor,
                       const VtDictionaryEntryBatchLimits* limits,
                       VtDictionaryEntryReducer reducer,
                       void* reducer_context,
                       size_t* out_count);
    VtStatus (*cancel)(VtDictionaryEntriesCursor* cursor);
    VtStatus (*close)(VtDictionaryEntriesCursor* cursor);
} VtDictionaryEntriesVTable;

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

#define VT_LATTICE_FLAG_THREAD_BOUND UINT64_C(1)
#define VT_LATTICE_FLAG_PARALLEL_REENTRANT UINT64_C(2)
#define VT_LATTICE_FLAG_STABLE_BYTES UINT64_C(4)
#define VT_LATTICE_FLAG_BATCH UINT64_C(8)

/*
 * One immutable lattice value. Binary operations accept only resources whose
 * VtLatticeVTable has the same domain_id. Every successful operation writes
 * one new owned VtResource. join_many/meet_many are associative left folds;
 * an empty operand array returns an independent retain of the receiver.
 */
typedef struct VtLatticeVTable {
    size_t struct_size;
    uint32_t interface_version;
    uint32_t reserved;
    uint64_t flags;
    VtInterfaceId domain_id;
    VtStatus (*join)(void* context, const VtResource* other,
                     VtResource* out_value);
    VtStatus (*meet)(void* context, const VtResource* other,
                     VtResource* out_value);
    VtStatus (*equal)(void* context, const VtResource* other,
                      uint8_t* out_equal);
    VtStatus (*stable_bytes)(void* context, uint8_t* out_bytes,
                             size_t capacity, size_t* out_written,
                             size_t* out_required);
    VtStatus (*diagnostic)(void* context, uint8_t* out_bytes,
                           size_t capacity, size_t* out_written,
                           size_t* out_required);
    VtStatus (*join_many)(void* context, const VtResource* others,
                          size_t count, VtResource* out_value);
    VtStatus (*meet_many)(void* context, const VtResource* others,
                          size_t count, VtResource* out_value);
} VtLatticeVTable;

#define VT_SEMIRING_FLAG_THREAD_BOUND UINT64_C(1)
#define VT_SEMIRING_FLAG_PARALLEL_REENTRANT UINT64_C(2)
#define VT_SEMIRING_FLAG_STABLE_BYTES UINT64_C(4)
#define VT_SEMIRING_FLAG_BATCH UINT64_C(8)

#define VT_SEMIRING_PROPERTY_HASHABLE UINT64_C(1)
#define VT_SEMIRING_PROPERTY_IDEMPOTENT_PLUS UINT64_C(2)
#define VT_SEMIRING_PROPERTY_K_CLOSED UINT64_C(4)
#define VT_SEMIRING_PROPERTY_ZERO_SUM_FREE UINT64_C(8)
#define VT_SEMIRING_PROPERTY_COMMUTATIVE_TIMES UINT64_C(16)
#define VT_SEMIRING_PROPERTY_TOTALLY_ORDERED UINT64_C(32)
#define VT_SEMIRING_PROPERTY_NONNEGATIVE UINT64_C(64)

#define VT_SEMIRING_ORDER_BETTER (-1)
#define VT_SEMIRING_ORDER_EQUAL 0
#define VT_SEMIRING_ORDER_WORSE 1
#define VT_SEMIRING_ORDER_INCOMPARABLE 2

/*
 * Compact provider-scoped weight token. A token is valid only while the
 * retained semiring operation-context resource that issued it remains alive.
 * Providers may encode an inline value or a generational arena identifier in
 * these two words. Consumers must clone and release owned tokens through the
 * semiring vtable; copying the words does not clone ownership.
 */
typedef struct VtSemiringValue {
    uint64_t word0;
    uint64_t word1;
} VtSemiringValue;

/*
 * Base dynamic-semiring capability. Every successful constructor or algebra
 * callback writes one owned value token. release_values consumes each token
 * exactly once. Batch folds use the additive or multiplicative identity for
 * an empty input. No callback may retain caller-owned input-array storage.
 */
typedef struct VtSemiringVTable {
    size_t struct_size;
    uint32_t interface_version;
    uint32_t reserved;
    uint64_t flags;
    VtInterfaceId domain_id;
    VtStatus (*zero)(void* context, VtSemiringValue* out_value);
    VtStatus (*one)(void* context, VtSemiringValue* out_value);
    VtStatus (*clone_value)(void* context, const VtSemiringValue* value,
                            VtSemiringValue* out_value);
    VtStatus (*release_values)(void* context, VtSemiringValue* values,
                               size_t count);
    VtStatus (*plus)(void* context, const VtSemiringValue* left,
                     const VtSemiringValue* right,
                     VtSemiringValue* out_value);
    VtStatus (*times)(void* context, const VtSemiringValue* left,
                      const VtSemiringValue* right,
                      VtSemiringValue* out_value);
    VtStatus (*equal)(void* context, const VtSemiringValue* left,
                      const VtSemiringValue* right, uint8_t* out_equal);
    VtStatus (*approx_equal)(void* context, const VtSemiringValue* left,
                             const VtSemiringValue* right, double epsilon,
                             uint8_t* out_equal);
    VtStatus (*natural_order)(void* context, const VtSemiringValue* left,
                              const VtSemiringValue* right,
                              int32_t* out_order);
    VtStatus (*stable_bytes)(void* context, const VtSemiringValue* value,
                             uint8_t* out_bytes, size_t capacity,
                             size_t* out_written, size_t* out_required);
    VtStatus (*diagnostic)(void* context, const VtSemiringValue* value,
                           uint8_t* out_bytes, size_t capacity,
                           size_t* out_written, size_t* out_required);
    VtStatus (*plus_many)(void* context, const VtSemiringValue* values,
                          size_t count, VtSemiringValue* out_value);
    VtStatus (*times_many)(void* context, const VtSemiringValue* values,
                           size_t count, VtSemiringValue* out_value);
} VtSemiringVTable;

/* Optional division capabilities. VT_STATUS_END denotes an undefined result. */
typedef struct VtSemiringDivisionVTable {
    size_t struct_size;
    uint32_t interface_version;
    uint32_t reserved;
    VtStatus (*divide)(void* context, const VtSemiringValue* dividend,
                       const VtSemiringValue* divisor,
                       VtSemiringValue* out_value);
    VtStatus (*left_divide)(void* context, const VtSemiringValue* value,
                            const VtSemiringValue* divisor,
                            VtSemiringValue* out_value);
} VtSemiringDivisionVTable;

/* Optional Kleene-closure capability. VT_STATUS_END denotes divergence. */
typedef struct VtSemiringStarVTable {
    size_t struct_size;
    uint32_t interface_version;
    uint32_t reserved;
    VtStatus (*star)(void* context, const VtSemiringValue* value,
                     VtSemiringValue* out_value);
} VtSemiringStarVTable;

/* Optional numerical projections used by specialized algorithms. */
typedef struct VtSemiringNumericVTable {
    size_t struct_size;
    uint32_t interface_version;
    uint32_t reserved;
    VtStatus (*numerical_value)(void* context,
                                const VtSemiringValue* value,
                                double* out_value);
    VtStatus (*quantize)(void* context, const VtSemiringValue* value,
                         double epsilon, int64_t* out_value);
    VtStatus (*to_probability)(void* context,
                               const VtSemiringValue* value,
                               double* out_value);
} VtSemiringNumericVTable;

/* Optional declared algebraic laws and bounded-closure metadata. */
typedef struct VtSemiringPropertiesVTable {
    size_t struct_size;
    uint32_t interface_version;
    uint32_t reserved;
    uint64_t properties;
    VtStatus (*closure_bound)(void* context, size_t* out_bound,
                              uint8_t* out_known);
} VtSemiringPropertiesVTable;

static const VtInterfaceId VT_DICTIONARY_INTERFACE_ID = {
    { 'v','t','.','d','i','c','t','i','o','n','a','r','y','.','v','1' }
};

static const VtInterfaceId VT_DICTIONARY_VISIT_INTERFACE_ID = {
    { 'v','t','.','d','i','c','t','.','v','i','s','i','t','.','v','1' }
};

static const VtInterfaceId VT_DICTIONARY_GRAPH_INTERFACE_ID = {
    { 'v','t','.','d','i','c','t','.','g','r','a','p','h','.','v','1' }
};

static const VtInterfaceId VT_DICTIONARY_ENTRIES_INTERFACE_ID = {
    { 'v','t','.','d','i','c','t','.','e','n','t','r','y','.','v','1' }
};

static const VtInterfaceId VT_SNAPSHOT_IDENTITY_INTERFACE_ID = {
    { 'v','t','.','s','n','a','p','s','h','o','t','.','i','d','.','1' }
};

static const VtInterfaceId VT_WFST_INTERFACE_ID = {
    { 'v','t','.','s','c','a','l','a','r','-','w','f','s','t','.','1' }
};

static const VtInterfaceId VT_LATTICE_INTERFACE_ID = {
    { 'v','t','.','l','a','t','t','i','c','e','.','v','a','l','.','1' }
};

static const VtInterfaceId VT_SEMIRING_INTERFACE_ID = {
    { 'v','t','.','s','e','m','i','r','i','n','g','.','v','a','l','1' }
};

static const VtInterfaceId VT_SEMIRING_DIVISION_INTERFACE_ID = {
    { 'v','t','.','s','e','m','i','r','i','n','g','.','d','i','v','1' }
};

static const VtInterfaceId VT_SEMIRING_STAR_INTERFACE_ID = {
    { 'v','t','.','s','e','m','i','r','i','n','g','.','s','t','r','1' }
};

static const VtInterfaceId VT_SEMIRING_NUMERIC_INTERFACE_ID = {
    { 'v','t','.','s','e','m','i','r','i','n','g','.','n','u','m','1' }
};

static const VtInterfaceId VT_SEMIRING_PROPERTIES_INTERFACE_ID = {
    { 'v','t','.','s','e','m','i','r','i','n','g','.','p','r','p','1' }
};

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus)
static_assert(sizeof(VtResource) == 2 * sizeof(void*),
              "VtResource must remain a two-word handle");
static_assert(sizeof(VtDictionaryEntriesCursor) == 2 * sizeof(void*),
              "VtDictionaryEntriesCursor must remain a two-word handle");
#endif

#endif /* VINARY_TREE_INTEROP_H */
