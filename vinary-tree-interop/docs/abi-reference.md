# The vinary-tree resource ABI, annotated

This document is the literate, normative walk of
[`include/vinary_tree_interop.h`](../include/vinary_tree_interop.h) — the
166-line C header that *is* the family ABI. Every declaration is quoted and
then explained: what it means, who may rely on it, and which law it carries.
The Rust definitions in [`src/lib.rs`](../src/lib.rs) are the same ABI spelled
in a second language; the two are kept bit-identical by the executable
contract in [`tests/`](../tests/) (see [§ 9](#9-the-executable-contract)).

Companion documents: [README](../README.md) (portal and catalog),
[ABI evolution policy](abi-evolution.md) (how this surface may change),
[security model](security-model.md) (what may go wrong and whose duty it is
to stop it).

---

## 1. Terms

Every symbol and acronym used below, defined before use:

| Term | Definition |
|---|---|
| ABI | Application Binary Interface: the byte-level contract (struct layouts, calling convention, ownership rules) between *independently compiled* binaries. Unlike an API, an ABI cannot be re-checked by a compiler at the boundary — drift is silent corruption. |
| vtable | Virtual-function table: a struct of function pointers the provider fills in. The consumer calls through the pointers without knowing the implementation. |
| resource | A live object exchanged across the ABI: two words (`context`, `vtable`) plus a reference count the provider maintains. |
| provider / producer | The binary that implements a resource's vtables and owns its memory. |
| consumer | The binary that receives a resource and calls through its vtables. |
| retain / release | The two reference-counting operations: `retain` adds one owned reference, `release` removes one. The count is the provider's ledger of live owners (Collins [[1]](#11-references)). |
| reference counting | Manual shared-ownership discipline: an object stays alive while its count is positive and is destroyed by the release that drops it to zero. |
| `query_interface` | Versioned capability discovery: the consumer names an interface and a minimum version; the provider either hands back a vtable for it or says `Unsupported`. Direct lineage: COM's `IUnknown::QueryInterface` (Box [[2]](#11-references)). |
| interface identifier | A 16-byte string compared byte-for-byte (no hashing, no NUL terminator) naming one interface contract, e.g. `vt.dictionary.v1`. |
| interface version | A `uint32_t` counter *within* one interface identifier, negotiated as a minimum by the consumer. |
| snapshot | An immutable revision of a possibly-mutable resource, captured in $`\mathcal{O}(1)`$ by structural sharing (Driscoll et al. [[3]](#11-references)). |
| revision | The logical value of a dictionary or WFST at one instant; a snapshot pins one revision forever. |
| structural sharing | Persistence technique: a new revision shares all unchanged substructure with its predecessors, so capturing a revision copies nothing. |
| unit domain | The value space of edge/arc labels: raw bytes, Unicode scalar values, or opaque `u64` tokens. |
| value domain | What a final dictionary node carries: nothing (set semantics), an optional `u64`, or (reserved) opaque bytes. |
| weight domain | Which scalar semiring the `double` in a WFST arc denotes. |
| semiring | An algebraic structure $`\langle K, \oplus, \otimes, \bar{0}, \bar{1} \rangle`$: a carrier set with two associative operations, where $`\oplus`$ (path alternation) is commutative with identity $`\bar{0}`$, $`\otimes`$ (path extension) has identity $`\bar{1}`$, $`\otimes`$ distributes over $`\oplus`$, and $`\bar{0}`$ annihilates $`\otimes`$. |
| epsilon label | The empty label $`\varepsilon`$ on a transducer arc — consumed no input or emitted no output. Encoded here by a *flag*, never by a magic label value. |
| page / paging | Transferring a variable-length edge or arc list through a fixed-capacity caller-owned buffer, in one or more calls. |
| POD | Plain Old Data: a struct of scalar fields with no pointers, invariant layout across targets. |
| `no_std` | A Rust crate attribute: no standard library, no allocator — nothing but type definitions can hide here. |
| null-pointer optimization (NPO) | Rust's guarantee that `Option<extern "C" fn>` occupies exactly one pointer with `None` represented as the null pointer — what lets a NULL C function pointer and a Rust `None` be the same bytes. |
| struct-size handshake | Additive-evolution mechanism: every vtable's first field is its own size in bytes; a consumer uses it to know which trailing fields exist. |
| machine word | The natural pointer width of the target: 8 bytes on 64-bit, 4 bytes on 32-bit. |

The published surface at a glance (fields, laws, and relations; every box is a
type quoted below):

![Class model of the published ABI: the two-word VtResource, its base VtResourceVTable, the dictionary and scalar-WFST interface vtables discovered through query_interface, the caller-owned page payload types, and the shared status/domain enums, annotated with the two-word and struct-size laws.](../../docs/diagrams/bindings/vt-structs-class.svg)

---

## 2. Prologue: what kind of header this is

```c
/* Stable resource ABI for modular vinary-tree language bindings. */
#ifndef VINARY_TREE_INTEROP_H
#define VINARY_TREE_INTEROP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
```

The header includes exactly two freestanding headers — `stddef.h` for
`size_t`, `stdint.h` for exact-width integers — and nothing else. There are
**no functions** in this ABI. It declares only layouts and constants; every
function that exists is a *pointer inside a struct*, supplied at runtime by a
provider. That is the crate's whole design: one ABI, zero algorithms. A
binary that only reads this header can neither link against nor depend on any
vinary-tree project.

```c
#define VT_ABI_VERSION 1u
#define VT_DICTIONARY_INTERFACE_VERSION 1u
#define VT_WFST_INTERFACE_VERSION 1u
#define VT_RECOMMENDED_EDGE_BATCH 256u
#define VT_RECOMMENDED_ARC_BATCH 256u
```

Three of these are version counters with precisely delimited jurisdictions
(the full model is the [evolution policy](abi-evolution.md)):

- `VT_ABI_VERSION` gates the **base resource protocol** — the layout and
  meaning of `VtResource` and `VtResourceVTable`. The reference consumer
  checks it for *exact equality* (`validate_base` in liblevenshtein's
  `src/bindings.rs` rejects `abi_version != 1`), so bumping it is a
  coordinated, family-wide breaking event.
- `VT_DICTIONARY_INTERFACE_VERSION` and `VT_WFST_INTERFACE_VERSION` each gate
  **one interface's contract** and are negotiated per-resource through
  `query_interface` as a minimum, not an exact match.

The two batch constants are *recommendations*, not limits: a consumer that
expands a node through a 256-entry buffer completes almost every expansion in
one crossing and pages only pathological nodes, giving

```math
\left\lceil \frac{\deg(v)}{256} \right\rceil
```

boundary crossings for a node of out-degree $`\deg(v)`$. Any positive
capacity is legal.

```c
#define VT_DICTIONARY_FLAG_PARALLEL_REENTRANT UINT64_C(1)
#define VT_DICTIONARY_FLAG_SUFFIX_BASED UINT64_C(2)
#define VT_DICTIONARY_FLAG_IMMUTABLE UINT64_C(4)
```

Capability **claims** a dictionary provider may make about itself (a bitset
stored in `VtDictionaryVTable.flags`); their meanings are given with that
struct in [§ 6.3](#63-dictionary_flags-claims-not-permissions).

---

## 3. `VtStatus` — the one error currency

```c
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
```

Every interop callback returns a `VtStatus`; there is no other error channel
(no `errno`, no exceptions, no unwinding — see the
[security model](security-model.md) for why unwinding across this boundary is
forbidden). The nine values, their semantics, and who may return each:

| Value | Name | Meaning | Who returns it |
|---|---|---|---|
| 0 | `Ok` | The operation completed; every advertised output was written. | Any callback. The **only** success value: `is_ok()` holds exactly for `Ok`, pinned by `tests/discriminant_pins.rs`. |
| 1 | `End` | An iterator or paginated stream is exhausted. | **Never legal from the interface callbacks in this header.** Paging termination is expressed through `out_written` / `out_total` (§ 7), so `End` is reserved for the project-level cursor surfaces built *above* this ABI (e.g. liblevenshtein's `llev_query_cursor_next`). The reference consumer treats any non-`Ok` interface return — including `End` — as a provider error (`status()` in liblevenshtein `src/bindings.rs`). This is family contract pin F5. |
| 2 | `InvalidArgument` | An argument was outside the callback's domain (e.g. an unknown node identifier). | Providers. |
| 3 | `NullPointer` | A required pointer argument was NULL. Failed calls make **no partial writes**: outputs are untouched. | Providers; also the value consumers expect from `query_interface` given a NULL identifier or output slot (pinned by `tests/vtable_evolution.rs`). |
| 4 | `Unsupported` | The requested interface, version, or operation is not offered. A *negotiation* outcome, not a fault. | `query_interface` (wrong identifier, or `minimum_version` above support); any op a provider legitimately cannot honor. |
| 5 | `IoError` | A storage-backed provider failed on I/O. | Providers with persistent backends. |
| 6 | `Closed` | The resource was already torn down. A best-effort courtesy: a consumer holding a retain must never *need* it. | Providers. |
| 7 | `LimitExceeded` | A configured resource limit (memory, node budget, result cap) was hit. | Providers. |
| 8 | `ProviderError` | The provider failed with no more specific portable status — including a panic or foreign exception the provider contained at its own boundary. | Providers; also what consumers map *unknown* future statuses to. |

Two hardening rules follow from the type being `#[repr(u32)]` on the Rust
side. First, a consumer must receive the value from a foreign callback as a
raw `uint32_t` and **validate it is in the range zero through eight before
converting** — an out-of-range discriminant materialized as a Rust `VtStatus`
is instant undefined behavior (family finding LLEV-B6, ledgered in
liblevenshtein's `docs/bindings/FINDINGS_LEDGER.md`). Second, new statuses
may only ever be *appended* (see the [evolution policy](abi-evolution.md)),
so "in range for version 1" is a stable predicate.

---

## 4. `VtInterfaceId` — identity as sixteen exact bytes

```c
typedef struct VtInterfaceId { uint8_t bytes[16]; } VtInterfaceId;
```

An interface identifier is sixteen bytes compared **byte-for-byte** — no
hashing, no case folding, no NUL terminator, no registry. The two published
identifiers (quoted in [§ 8.3](#83-the-published-identifiers)) are ASCII
mnemonics whose *sixteenth byte is the version fork counter*:
`vt.dictionary.v1` and `vt.scalar-wfst.1`. Exactness is the point: two
independently built binaries agree on an interface exactly when they contain
the same sixteen bytes, and a *breaking* interface revision changes the
string itself so the two contracts can never be confused
(`tests/layout_contract.rs` pins both identifiers and their inequality,
invariant hook VT-ABI-4).

---

## 5. The resource: two words and a base vtable

### 5.1 `VtResource`

```c
struct VtResourceVTable;
typedef struct VtResource {
    void* context;
    const struct VtResourceVTable* vtable;
} VtResource;
```

A live resource is **exactly two machine words**: an opaque
provider-defined `context` pointer and a pointer to an immutable
provider-owned vtable. This is the *two-word law*:

```math
\mathrm{sizeof}(\mathtt{VtResource}) \;=\; 2 \cdot \mathrm{sizeof}(\mathtt{void*}),
\qquad
\mathrm{alignof}(\mathtt{VtResource}) \;=\; \mathrm{alignof}(\mathtt{void*}).
```

It is enforced three times: at compile time in Rust (the `const` assertion
block at the bottom of [`src/lib.rs`](../src/lib.rs)), at compile time in C++
(the `static_assert` in this header's epilogue, § 8.3), and at runtime on the
JVM (`BindingArchitectureTest.sharedResourceIsExactlyTwoWords` in
liblevenshtein's `bindings/jvm` test suite, which has no compile-time layout
checking to lean on). Passing a resource is passing sixteen bytes on a 64-bit
target; native object handoff is $`\mathcal{O}(1)`$ and never serializes the
object.

**The copy-is-not-retain law.** Copying the two words is free and confers
nothing:

> A non-null resource owns one retain. Copying the two words does not retain
> automatically: the receiver must call `retain` before storing another owned
> copy and must eventually call `release` once for every owned retain.
> — `src/lib.rs`, `VtResource` documentation

A resource is *null* when **either** word is null (`VtResource::is_null`),
and `VtResource::NULL` (both words null) is the conventional output
initializer. `tests/layout_contract.rs` pins the either-word semantics.

### 5.2 `VtResourceVTable`

```c
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
```

The base vtable is the whole resource *protocol*; everything else is
discovered through it. Field by field:

- **`struct_size`** — the provider writes `sizeof` of the vtable *it built*.
  A future ABI revision may append trailing members; a consumer discovers
  them by comparing `struct_size` against the offsets it knows
  (the struct-size handshake, proven end-to-end by
  `tests/vtable_evolution.rs`).
- **`abi_version`** — must equal `VT_ABI_VERSION`. The base-protocol gate.
- **`reserved`** — **must be zero.** Reserved fields are the ABI's blank
  tape: they may acquire meaning in a future version precisely because
  today's providers all write zero and careful consumers verify it.
- **`retain`** — add one owned reference for `context`. No status: retain on
  a live resource cannot fail.
- **`release`** — remove one owned reference. The release that drops the
  ledger to zero destroys the resource; the two words are dangling
  afterwards.
- **`query_interface`** — versioned discovery. The consumer passes an
  identifier and the *minimum* interface version it can consume. On success
  the provider writes a pointer to a **provider-owned, immutable** interface
  vtable and returns `Ok`; the pointer remains valid *exactly as long as the
  resource stays retained* — discovery is not retention, so the consumer must
  already hold the retain that keeps it alive. On any failure the output is
  untouched: wrong identifier or unsatisfiable version yields `Unsupported`,
  NULL arguments yield `NullPointer` (invariant hooks VT-QI-1 through
  VT-QI-3, pinned by `tests/vtable_evolution.rs`).

All three function pointers are nullable at the layout level (that is what
lets a provider omit optional *interface* ops), but a resource whose base
vtable omits any of the three is not a usable resource: the reference
consumer rejects it (`IncompatibleResourceAbi`).

The design is COM's `IUnknown` reduced to its portable core — `AddRef`,
`Release`, `QueryInterface` over an opaque `this` (Box
[[2]](#11-references)) — with reference counting in the form Collins
introduced for list structures (Collins [[1]](#11-references)). What is
deliberately *not* imported from COM: apartments, marshaling, registries,
HRESULTs, and any global allocator contract.

The full negotiation, both failure branches, and the balancing release on
every path:

![Activity diagram of interface negotiation: copy the two words, validate the base vtable, retain, call query_interface with its NullPointer and Unsupported branches, on success use the provider-owned vtable that stays valid while retained, and release exactly once on every path.](../../docs/diagrams/bindings/interface-negotiation-activity.svg)

### 5.3 The refcount laws

Let $`\mathrm{retains}_{\le t}(r)`$ and $`\mathrm{releases}_{\le t}(r)`$
count the retain and release calls issued for resource $`r`$ up to time
$`t`$, with the provider's own construction counting as the first retain
(a resource is *born owning one retain*, which it hands to its first owner).
The ledger law:

```math
\mathrm{live}_t(r) \;=\; \mathrm{retains}_{\le t}(r) \;-\; \mathrm{releases}_{\le t}(r) \;\ge\; 0
\qquad \text{for every } t .
```

The validity-window law — every callback through any of $`r`$'s vtables at
time $`t`$, and every dereference of an interface vtable pointer obtained
from $`r`$, requires

```math
\mathrm{live}_t(r) \;>\; 0 ,
```

and the balance law — each owned retain is paired with **exactly one**
release over the resource's life:

```math
\lim_{t \to \infty} \mathrm{live}_t(r) \;=\; 0 ,
```

with no release ever issued for a retain the caller does not own
(over-release is destruction of someone else's reference and, past zero, a
double free). Node and state identifiers inherit the validity window: they
are meaningful only against a snapshot whose $`\mathrm{live}`$ count the
holder keeps positive. The retain/release protocol is formally modeled in
`docs/verification/tla/AbiResourceLifecycle.tla` (invariant IDs VT-LIFE-1
through VT-LIFE-6) in the liblevenshtein repository, and
`tests/vtable_evolution.rs` pins the ledger-balancing surface behavior.

---

## 6. The dictionary interface

### 6.1 `VtUnitDomain` and `VtValueDomain`

```c
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
```

Labels cross the ABI as `uint64_t` regardless of domain; the **unit domain**
says what the integer *means* and thereby which values are representable:

- `BYTE` — raw byte labels; every label is below 256.
- `UNICODE_SCALAR` — Unicode scalar values, i.e. code points excluding the
  surrogate range: $`[0, \mathrm{D7FF}_{16}] \cup [\mathrm{E000}_{16}, \mathrm{10FFFF}_{16}]`$.
- `U64` — opaque unsigned 64-bit tokens (tokenizer output, phoneme
  identifiers, grammar symbols); any value is admissible.

A label outside its declared domain is a provider fault, and consumers must
reject it rather than truncate (see the
[security model](security-model.md)). The **value domain** states what a
final node carries: `UNIT` is plain set membership, `OPTIONAL_U64` attaches
an optional integer payload readable through `node_value_u64`, and `BYTES` is
*reserved* — declared now so the discriminant is pinned, usable only after a
future interface version defines its operations (the reference consumer
rejects `BYTES` providers today).

### 6.2 `VtOptionalU64` and `VtDictionaryEdge`

```c
typedef struct VtOptionalU64 {
    uint64_t value;
    uint8_t has_value;
    uint8_t reserved[7];
} VtOptionalU64;

typedef struct VtDictionaryEdge {
    uint64_t label;
    uint64_t node;
} VtDictionaryEdge;
```

Both are POD payloads built purely from `uint64_t`/`uint8_t`, so their
layouts (16 bytes each) are identical on **every** supported target — unlike
the vtables, whose word-sized members vary between 32-bit and 64-bit tiers.
`has_value` is exactly zero or one; the seven `reserved` bytes **must be
written zero** by providers. `tests/layout_contract.rs` pins offsets and
sizes; `tests/discriminant_pins.rs` pins that the all-zero value is the
default.

### 6.3 `dictionary_flags` — claims, not permissions

The three flag macros quoted in § 2 populate `VtDictionaryVTable.flags`:

- `PARALLEL_REENTRANT` — *"calls may run concurrently and re-enter the
  provider."* This is a **provider claim about itself**, not a consumer
  obligation. In its absence the reference consumer serializes every callback
  to a captured provider behind a mutex (the `CallGate` in liblevenshtein
  `src/bindings.rs`); when the claim is present the gate is elided. A false
  claim harms only the claimant: the resulting data races are inside the
  provider's own state (the consumer's memory is never handed out for the
  provider to synchronize), though the garbage such races produce still meets
  the consumer's output validation like any other hostile input.
- `SUFFIX_BASED` — the dictionary implements suffix/substring distance
  semantics (a suffix-automaton backend), so a consumer's traversal semantics
  must match.
- `IMMUTABLE` — the resource is *already* an immutable revision, so
  `snapshot` may simply retain the resource itself and hand back the same two
  words.

Consumers must **ignore flag bits they do not know**; future bits are
additive claims (see the [evolution policy](abi-evolution.md)).

### 6.4 `VtDictionaryVTable`, op by op

```c
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
```

The header (`struct_size`, `interface_version`, domains, `flags`) is the
interface's self-description; `interface_version` must be at least
`VT_DICTIONARY_INTERFACE_VERSION` for a v1 consumer, which checks it with a
*minimum*, not equality — a v2-capable provider still serves v1 consumers.
The `context` passed to every op is the resource's context word.

- **`snapshot(context, out_snapshot)`** — capture the revision visible at
  operation start, as a *new* resource (born owning one retain, handed to the
  caller). The defining constraint is the **capture-cost contract**:

  > Implementations must make `snapshot` $`\mathcal{O}(1)`$ with structural
  > sharing (or an equivalent immutable revision); copying the whole
  > dictionary or holding a long-lived read lock violates the interface
  > contract. — `src/lib.rs`, `VtDictionaryVTable` documentation

  This is what makes a mutable dictionary safely consumable: the consumer
  captures once at the start of a query and traverses a frozen revision while
  writers proceed unblocked. It is the ABI-shaped form of the persistence
  construction of Driscoll, Sarnak, Sleator, and Tarjan
  [[3]](#11-references). Formal statement in § 6.5, law (S).
- **`root(context, out_node)`** — the root node identifier of this
  *immutable* resource. Node identifiers are provider-defined opaque
  `uint64_t` values scoped to the captured snapshot: **valid only while the
  snapshot resource is retained**, never comparable across snapshots.
- **`len(context, out_len, out_known)`** — the stored-term count *when
  cheaply available*. A provider that would have to walk its structure to
  count writes `out_known = 0` instead of stalling; `out_len` is meaningful
  only when `out_known` is one.
- **`node_is_final(context, node, out_is_final)`** — whether `node`
  terminates a stored term (writes zero or one).
- **`node_value_u64(context, node, out_value)`** — the optional payload of a
  final node. Required exactly when `value_domain` is `OPTIONAL_U64`
  (the reference consumer refuses an `OPTIONAL_U64` interface without it);
  legitimately NULL under `UNIT`.
- **`node_transition(context, node, label, out_child, out_found)`** — follow
  one edge without enumerating siblings. Absence is a *result*
  (`out_found = 0`, status `Ok`), not an error — exact-lookup walks stay on
  the happy path.
- **`node_edges(context, node, start, out_edges, capacity, out_written,
  out_total)`** — copy one page of outgoing edges into caller-owned
  contiguous storage. `start` is an *edge offset* within the node's edge
  list; `out_total` receives the complete edge count (stable for a given node
  across a snapshot's lifetime); `out_written` receives how many edges were
  copied, never more than `capacity`. A consumer sizes its buffer at
  `VT_RECOMMENDED_EDGE_BATCH`, expands nearly every node in one call, and
  pages the rare high-degree node by advancing `start`. The precise algebra
  is § 6.5, law (P).

Only `snapshot`, `root`, `node_is_final`, and `node_edges` are unconditionally
required by the reference consumer; `len` and `node_transition` are optional
accelerations, and `node_value_u64` is conditionally required as stated
above.

### 6.5 The dictionary laws

**(P) The paging law.** Fix a node $`v`$ of a retained snapshot and let its
edge list be the sequence

```math
E(v) = \bigl[\, e_0, e_1, \ldots, e_{T-1} \,\bigr],
\qquad T = \mathrm{total}(v),
```

whose order is provider-chosen but **fixed for the snapshot's lifetime**.
Every `node_edges` call with offset $`s`$ and capacity $`c`$ that returns
`Ok` with outputs $`(w, T')`$ satisfies

```math
T' = T,
\qquad
w \;=\; \min\bigl(c,\; T - \min(s, T)\bigr),
\qquad
\mathtt{out\_edges}[i] = e_{s+i} \quad \text{for } 0 \le i < w .
```

Consequences a consumer may rely on (and must verify — a foreign provider
can violate any of them): $`w \le c`$ always; $`T`$ is identical across
every page of the same node; an offset at or past $`T`$ yields the empty
page $`w = 0`$; and the concatenation of successive pages taken at offsets
$`s_0 = 0`$, $`s_{k+1} = s_k + w_k`$ until an empty page reconstructs the
edge list exactly:

```math
\mathrm{page}(s_0) \mathbin{+\!\!+} \mathrm{page}(s_1) \mathbin{+\!\!+} \cdots \;=\; E(v),
```

where $`\mathbin{+\!\!+}`$ is sequence concatenation. Since each non-final
page has $`w = c > 0`$, paging terminates in
$`\lceil T / c \rceil`$ calls. The consumer-side acceptance predicate is
being harmonized across the family to one proven form (family finding F3;
formal home `docs/verification/abi/theories/PagingLaws.v` and
`ConsumerAcceptance.v`, invariant IDs VT-PAGE-1 through VT-PAGE-6, in the
liblevenshtein repository).

**(S) The snapshot laws.** Let $`D_t`$ denote the source dictionary's
revision at time $`t`$, and let `snapshot` at time $`t_0`$ yield resource
$`\sigma`$. Then:

```math
\text{(capture cost)} \qquad
\mathrm{cost}\bigl(\mathtt{snapshot}\bigr) = \mathcal{O}(1)
\quad \text{— independent of } \lvert D_{t_0} \rvert,
```

```math
\text{(immutability)} \qquad
\forall\, t \ge t_0,\ \forall\, \text{reads } q :\quad
q(\sigma \text{ at } t) \;=\; q(D_{t_0}),
```

i.e. every read through $`\sigma`$ — `root`, `node_is_final`,
`node_transition`, `node_edges`, `node_value_u64` — observes the captured
revision forever, regardless of how the source has mutated since. Node
identifiers read from $`\sigma`$ are valid while $`\mathrm{live}(\sigma) > 0`$
and carry no meaning against any other snapshot. The consumer-side law is
formalized in `CursorSnapshotSemantics.v` (VT-SNAP-1 through VT-SNAP-3) and
exercised by liblevenshtein's `tests/query_start_snapshot_semantics.rs`.

---

## 7. The scalar-WFST interface

### 7.1 `VtWeightDomain` — seven semirings in one `double`

```c
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
```

Arc weights cross the ABI as IEEE-754 `double`; the weight domain declares
which semiring $`\langle K, \oplus, \otimes, \bar{0}, \bar{1} \rangle`$
(§ 1) that scalar denotes, and thereby which bit patterns are *valid* — the
per-domain representation predicate consumers must enforce at ingestion.

**`TROPICAL_F64` — min-plus costs.** Shortest-path weight algebra: alternation
takes the cheaper branch, extension adds costs; $`+\infty`$ means
"unreachable" and is the additive identity, so the header's own summary reads
"positive infinity is zero and 0 is one":

```math
\mathcal{T} \;=\; \bigl\langle\; \mathbb{R} \cup \{+\infty\},\;\; \min,\;\; +,\;\; +\infty,\;\; 0 \;\bigr\rangle .
```

Valid representations: finite values or $`+\infty`$ — never NaN, never
$`-\infty`$.

**`LOG_F64` — negative-log probabilities.** The probability semiring mapped
through $`x \mapsto -\ln x`$ so products become sums and tiny probabilities
stay representable; alternation is log-sum-exp:

```math
\mathcal{L} \;=\; \bigl\langle\; \mathbb{R} \cup \{+\infty\},\;\; \oplus_{\log},\;\; +,\;\; +\infty,\;\; 0 \;\bigr\rangle,
\qquad
x \oplus_{\log} y \;=\; -\ln\!\bigl(e^{-x} + e^{-y}\bigr).
```

**`PROBABILITY_F64` — direct probabilities.** Sum-of-paths semantics in the
raw:

```math
\mathcal{P} \;=\; \bigl\langle\; \mathbb{R}_{\ge 0},\;\; +,\;\; \times,\;\; 0,\;\; 1 \;\bigr\rangle .
```

**`ARCTIC_F64` — max-plus scores.** The mirror of tropical for
reward-maximization (Viterbi-style best score); $`-\infty`$ is the
unreachable element:

```math
\mathcal{A} \;=\; \bigl\langle\; \mathbb{R} \cup \{-\infty\},\;\; \max,\;\; +,\;\; -\infty,\;\; 0 \;\bigr\rangle .
```

Valid representations: finite values or $`-\infty`$ — never NaN, never
$`+\infty`$.

**`SIGNED_TROPICAL_F64` — costs and rewards.** The same signature as
$`\mathcal{T}`$, but the carrier deliberately admits negative reals as
rewards alongside positive costs:

```math
\mathcal{S} \;=\; \bigl\langle\; \mathbb{R} \cup \{+\infty\},\;\; \min,\;\; +,\;\; +\infty,\;\; 0 \;\bigr\rangle .
```

The Kleene star $`w^{*} = \bar{1} \oplus w \oplus (w \otimes w) \oplus \cdots`$
converges only for $`w \ge 0`$; algorithms requiring closure must guard
against negative-weight cycles.

**`COUNT_F64` — path counting.** The counting semiring carried in the `f64`
slot; values are non-negative integers, exact up to $`2^{53}`$:

```math
\mathcal{C} \;=\; \bigl\langle\; \mathbb{N},\;\; +,\;\; \times,\;\; 0,\;\; 1 \;\bigr\rangle .
```

**`BOOLEAN_F64` — reachability.** Classical acceptance embedded as exactly
the two values 0 and 1:

```math
\mathcal{B} \;=\; \bigl\langle\; \{0, 1\},\;\; \lor,\;\; \land,\;\; 0,\;\; 1 \;\bigr\rangle .
```

**The IEEE-754 caveat — stated once, bindingly.** The `f64`
*representation* does **not** satisfy the exact semiring laws wherever an
operation rounds. Floating-point addition is not associative, so for
$`\mathcal{T}`$, $`\mathcal{S}`$, and $`\mathcal{A}`$ the extension
$`\otimes = +`$ is only approximately associative; for $`\mathcal{P}`$ and
$`\mathcal{L}`$ *both* operations round ($`\oplus_{\log}`$ compounds two
transcendental evaluations); $`\mathcal{C}`$ is exact only below
$`2^{53}`$. Exactness survives where no rounding occurs: $`\min`$ /
$`\max`$ over non-NaN values (the $`\oplus`$ of the three tropical-family
domains) and all of $`\mathcal{B}`$ on its two values. No document or
implementation in this family may claim the exact laws for the `f64`
representation; correctness claims are made for the *mathematical* semiring
and accompanied by representation predicates. The formal treatment —
per-domain `repr_ok` predicates, exactness lemmas where true, and rounding
envelopes where not — lives in lling-llang's `proofs/coq/abi/WeightBridge.v`
(family obligation #16, invariant IDs LLING-BRIDGE-1 through
LLING-BRIDGE-4), motivated by the confirmed ingestion finding LLING-B2
described in the [security model](security-model.md).

### 7.2 `wfst_flags`

```c
#define VT_WFST_FLAG_PARALLEL_REENTRANT UINT64_C(1)
#define VT_WFST_FLAG_IMMUTABLE UINT64_C(2)
#define VT_WFST_FLAG_LAZY UINT64_C(4)
#define VT_WFST_FLAG_ACYCLIC UINT64_C(8)
```

- `PARALLEL_REENTRANT` — the same self-claim as the dictionary flag
  (§ 6.3), with one sharpening the header spells out: a lazily expanding
  provider advertising this flag must make its expansions safe for
  independent concurrent callers "without imposing a process-wide or
  resource-wide sequential call gate".
- `IMMUTABLE` — `snapshot` may retain the resource itself directly.
- `LAZY` — states or arcs may be materialized on demand during `state_info`
  or `state_arcs`; consumers must not assume the state space is enumerable
  up front.
- `ACYCLIC` — the provider **guarantees** the reachable graph is acyclic,
  licensing single-pass algorithms (topological shortest distance) and
  bounded-frontier traversal. A false `ACYCLIC` claim makes a consumer's
  bounded traversal unsound, which is why claims are still subject to
  consumer resource limits (see the [security model](security-model.md)).

### 7.3 `VtWfstArc` — epsilon is a flag, not a value

```c
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
```

A 40-byte POD arc, layout-identical on every target
(`tests/layout_contract.rs` pins all seven offsets). The epsilon encoding is
the design decision worth pausing on: transducer arcs may consume no input
($`\varepsilon`$-input) or emit no output ($`\varepsilon`$-output), and this
ABI encodes that as `has_input = 0` / `has_output = 0` with the
corresponding label field *meaningless* — *not* as a reserved magic label
value (OpenFst's convention of label 0). With `U64` unit domains every one
of the $`2^{64}`$ label values remains usable, and "no label" cannot collide
with a real token. An arc with both flags zero is a pure
$`\varepsilon`$-transition (weight and target still meaningful). The six
`reserved` bytes must be zero — and this is the reserved field consumers
already validate today (lling-llang rejects arcs with nonzero reserved
bytes at capture).

### 7.4 `VtWfstVTable`, op by op

```c
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
```

The header mirrors the dictionary's (`struct_size` handshake,
minimum-version semantics, flags), adds `weight_domain`, and carries one
explicit `reserved` word that must be zero. The operations:

- **`snapshot(context, out_snapshot)`** — capture the revision visible at
  operation start; same $`\mathcal{O}(1)`$ contract and laws as § 6.5 (S).
  State identifiers are scoped to the retained snapshot exactly as node
  identifiers are.
- **`start(context, out_state)`** — the initial state.
- **`num_states(context, out_count, out_known)`** — the state count when
  cheaply known; a `LAZY` provider typically writes `out_known = 0` because
  its state space is materialized on demand.
- **`state_info(context, state, out_valid, out_is_final, out_final_weight)`**
  — validate a state identifier and read its finality. `out_valid` and
  `out_is_final` are each zero or one; `out_final_weight` is meaningful only
  when `out_is_final` is one, and must then satisfy the weight domain's
  representation predicate (§ 7.1). Validation-as-an-operation exists
  because lazy providers may *mint* states during expansion; a consumer
  holding a suspicious identifier asks rather than crashes.
- **`state_arcs(context, state, start, out_arcs, capacity, out_written,
  out_total)`** — copy one page of outgoing arcs into caller-owned storage.
  The paging algebra is identical to § 6.5 (P) with arcs in place of edges
  and `VT_RECOMMENDED_ARC_BATCH` as the suggested capacity. A `LAZY`
  provider may expand the state on first touch; under `PARALLEL_REENTRANT`
  that expansion must tolerate concurrent independent callers.

---

## 8. Epilogue of the header

### 8.1 The published identifiers

```c
static const VtInterfaceId VT_DICTIONARY_INTERFACE_ID = {
    { 'v','t','.','d','i','c','t','i','o','n','a','r','y','.','v','1' }
};

static const VtInterfaceId VT_WFST_INTERFACE_ID = {
    { 'v','t','.','s','c','a','l','a','r','-','w','f','s','t','.','1' }
};
```

The two constants every conforming binary shares, spelled as character
arrays so the byte-exactness is visible: `vt.dictionary.v1` (16 bytes) and
`vt.scalar-wfst.1` (16 bytes). They are `static const` so the header stays
usable from any C translation unit without a home object file.

### 8.2 The C++ guard

```c
#if defined(__cplusplus)
static_assert(sizeof(VtResource) == 2 * sizeof(void*),
              "VtResource must remain a two-word handle");
#endif
```

The two-word law (§ 5.1) re-asserted for C++ consumers at their own compile
time — the third of the law's three enforcement points.

### 8.3 What is *not* in the header

No allocation contract (each provider frees what it allocated, inside
`release`), no thread identity rules (only the `PARALLEL_REENTRANT` claim),
no string encoding (labels are integers; text encoding is a project-level
concern), no I/O, and no functions. Everything a project adds — cursors,
builders, error strings — lives in that project's own `*_abi` surface above
this header.

---

## 9. The executable contract

Three test suites in [`tests/`](../tests/) are the ABI's executable half;
CI runs them on every push (`cargo test --locked -p vinary-tree-interop`),
and any layout or semantic drift fails there before it can ship:

| Suite | What it pins | Invariant hooks |
|---|---|---|
| [`layout_contract.rs`](../tests/layout_contract.rs) | Sizes, alignments, and every field offset of all seven `#[repr(C)]` types — exact tables for the 64-bit tier and the 32-bit ARM EABI tier, plus target-independent packing laws; the two-word law; either-word null semantics; byte-exact interface identifiers; the `Option<extern "C" fn>` null niche. | VT-ABI-1, VT-ABI-2, VT-ABI-3, VT-ABI-4, VT-ABI-6 |
| [`discriminant_pins.rs`](../tests/discriminant_pins.rs) | Every enum discriminant and flag bit, twice: exact numeric pins, and wildcard-free `match` tables so *adding* a variant fails compilation until the pins (and the [evolution policy](abi-evolution.md)) are consulted; zeroed defaults for all reserved fields. | VT-ABI-5 |
| [`vtable_evolution.rs`](../tests/vtable_evolution.rs) | The `query_interface` negotiation surface against a hand-rolled provider: wrong identifier and future `minimum_version` yield `Unsupported` with the output untouched, NULL arguments yield `NullPointer`, success hands out a provider-owned vtable without retaining, the retain/release ledger balances, and a strictly larger future vtable remains consumable through its v1 prefix via `struct_size`. | VT-QI-1, VT-QI-2, VT-QI-3 |

---

## 10. A complete minimal provider in C

The smallest honest provider: a static three-term byte dictionary
(`{"car", "cat", "dog"}`) implementing the full base protocol (atomic
refcount, negotiation) and the full dictionary interface (correct paging
included). It compiles clean under
`cc -std=c17 -Wall -Wextra -Werror -fsyntax-only -I include`, and every
contract explained above is visible in miniature: the born-owning-one-retain
constructor, discovery-is-not-retention, the untouched-output failure
branches, the empty page past the end, and the legitimately NULL
`node_value_u64` slot under the `UNIT` value domain.

```c
/*
 * static_dictionary_provider.c — a complete, minimal interop provider.
 *
 * Publishes the three-term byte dictionary {"car", "cat", "dog"} as a
 * VtResource implementing VtResourceVTable (atomic refcount + interface
 * negotiation) and VtDictionaryVTable (snapshot, root, len, node_is_final,
 * node_transition, node_edges with correct paging).
 */
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "vinary_tree_interop.h"

/* ── the immutable trie: {"car", "cat", "dog"} ──────────────────────────── */
/*
 *        (0) ──c──▶ (1) ──a──▶ (3) ──r──▶ (5)*   * = final node
 *         │                     └───t──▶ (6)*
 *         └──d──▶ (2) ──o──▶ (4) ──g──▶ (7)*
 */
static const VtDictionaryEdge EDGES_NODE_0[] = { { 'c', 1 }, { 'd', 2 } };
static const VtDictionaryEdge EDGES_NODE_1[] = { { 'a', 3 } };
static const VtDictionaryEdge EDGES_NODE_2[] = { { 'o', 4 } };
static const VtDictionaryEdge EDGES_NODE_3[] = { { 'r', 5 }, { 't', 6 } };
static const VtDictionaryEdge EDGES_NODE_4[] = { { 'g', 7 } };

typedef struct StaticNode {
    const VtDictionaryEdge* edges;
    size_t edge_count;
    uint8_t is_final;
} StaticNode;

static const StaticNode NODES[] = {
    { EDGES_NODE_0, 2, 0 }, /* 0: root        */
    { EDGES_NODE_1, 1, 0 }, /* 1: "c"         */
    { EDGES_NODE_2, 1, 0 }, /* 2: "d"         */
    { EDGES_NODE_3, 2, 0 }, /* 3: "ca"        */
    { EDGES_NODE_4, 1, 0 }, /* 4: "do"        */
    { NULL,         0, 1 }, /* 5: "car" final */
    { NULL,         0, 1 }, /* 6: "cat" final */
    { NULL,         0, 1 }, /* 7: "dog" final */
};

#define NODE_COUNT ((uint64_t)(sizeof(NODES) / sizeof(NODES[0])))
#define TERM_COUNT ((size_t)3)

/* ── provider context: one atomic ledger of owned retains ───────────────── */

typedef struct StaticDictionary {
    atomic_size_t refs; /* number of live owned retains */
} StaticDictionary;

/* Forward declarations so the vtables can be defined before the bodies. */
static void dictionary_retain(void* context);
static void dictionary_release(void* context);
static VtStatus dictionary_query_interface(void* context,
                                           const VtInterfaceId* interface_id,
                                           uint32_t minimum_version,
                                           const void** out_vtable);
static VtStatus dictionary_snapshot(void* context, VtResource* out_snapshot);
static VtStatus dictionary_root(void* context, uint64_t* out_node);
static VtStatus dictionary_len(void* context, size_t* out_len,
                               uint8_t* out_known);
static VtStatus dictionary_node_is_final(void* context, uint64_t node,
                                         uint8_t* out_is_final);
static VtStatus dictionary_node_transition(void* context, uint64_t node,
                                           uint64_t label, uint64_t* out_child,
                                           uint8_t* out_found);
static VtStatus dictionary_node_edges(void* context, uint64_t node,
                                      size_t start, VtDictionaryEdge* out_edges,
                                      size_t capacity, size_t* out_written,
                                      size_t* out_total);

/* ── the two provider-owned immutable vtables ───────────────────────────── */

static const VtResourceVTable RESOURCE_VTABLE = {
    .struct_size = sizeof(VtResourceVTable),
    .abi_version = VT_ABI_VERSION,
    .reserved = 0, /* reserved fields are always written as zero */
    .retain = dictionary_retain,
    .release = dictionary_release,
    .query_interface = dictionary_query_interface,
};

static const VtDictionaryVTable DICTIONARY_VTABLE = {
    .struct_size = sizeof(VtDictionaryVTable),
    .interface_version = VT_DICTIONARY_INTERFACE_VERSION,
    .unit_domain = VT_UNIT_DOMAIN_BYTE,
    .value_domain = VT_VALUE_DOMAIN_UNIT,
    /* Both flags are TRUE claims here: the tables are immutable statics and
     * the refcount is atomic, so concurrent reentrant calls are safe. */
    .flags = VT_DICTIONARY_FLAG_IMMUTABLE | VT_DICTIONARY_FLAG_PARALLEL_REENTRANT,
    .snapshot = dictionary_snapshot,
    .root = dictionary_root,
    .len = dictionary_len,
    .node_is_final = dictionary_node_is_final,
    .node_value_u64 = NULL, /* value_domain is UNIT: op legitimately absent */
    .node_transition = dictionary_node_transition,
    .node_edges = dictionary_node_edges,
};

/* ── base resource protocol ─────────────────────────────────────────────── */

static void dictionary_retain(void* context) {
    StaticDictionary* dictionary = (StaticDictionary*)context;
    atomic_fetch_add_explicit(&dictionary->refs, 1, memory_order_relaxed);
}

static void dictionary_release(void* context) {
    StaticDictionary* dictionary = (StaticDictionary*)context;
    /* Release-acquire pairing: the final releaser observes every write made
     * while the resource was shared, then frees exactly once. */
    if (atomic_fetch_sub_explicit(&dictionary->refs, 1, memory_order_release) == 1) {
        atomic_thread_fence(memory_order_acquire);
        free(dictionary);
    }
}

static VtStatus dictionary_query_interface(void* context,
                                           const VtInterfaceId* interface_id,
                                           uint32_t minimum_version,
                                           const void** out_vtable) {
    (void)context; /* negotiation needs no per-instance state here */
    if (interface_id == NULL || out_vtable == NULL) {
        return VT_STATUS_NULL_POINTER;
    }
    if (memcmp(interface_id->bytes, VT_DICTIONARY_INTERFACE_ID.bytes, 16) != 0) {
        return VT_STATUS_UNSUPPORTED; /* wrong interface: output untouched */
    }
    if (minimum_version > VT_DICTIONARY_INTERFACE_VERSION) {
        return VT_STATUS_UNSUPPORTED; /* consumer is from the future */
    }
    /* Discovery is not retention: hand out the provider-owned vtable without
     * touching the refcount. The pointer stays valid while the resource is
     * retained. */
    *out_vtable = &DICTIONARY_VTABLE;
    return VT_STATUS_OK;
}

/* ── dictionary interface ───────────────────────────────────────────────── */

static VtStatus dictionary_snapshot(void* context, VtResource* out_snapshot) {
    if (out_snapshot == NULL) {
        return VT_STATUS_NULL_POINTER;
    }
    /* The dictionary is immutable (flag IMMUTABLE is set), so the O(1)
     * snapshot is the resource itself: add one owned retain for the snapshot
     * handle and return the same two words. A mutable provider would instead
     * hand out its current structurally shared revision. */
    dictionary_retain(context);
    out_snapshot->context = context;
    out_snapshot->vtable = &RESOURCE_VTABLE;
    return VT_STATUS_OK;
}

static VtStatus dictionary_root(void* context, uint64_t* out_node) {
    (void)context;
    if (out_node == NULL) {
        return VT_STATUS_NULL_POINTER;
    }
    *out_node = 0;
    return VT_STATUS_OK;
}

static VtStatus dictionary_len(void* context, size_t* out_len,
                               uint8_t* out_known) {
    (void)context;
    if (out_len == NULL || out_known == NULL) {
        return VT_STATUS_NULL_POINTER;
    }
    *out_len = TERM_COUNT;
    *out_known = 1; /* cheaply available for a static table */
    return VT_STATUS_OK;
}

static VtStatus dictionary_node_is_final(void* context, uint64_t node,
                                         uint8_t* out_is_final) {
    (void)context;
    if (out_is_final == NULL) {
        return VT_STATUS_NULL_POINTER;
    }
    if (node >= NODE_COUNT) {
        return VT_STATUS_INVALID_ARGUMENT;
    }
    *out_is_final = NODES[node].is_final;
    return VT_STATUS_OK;
}

static VtStatus dictionary_node_transition(void* context, uint64_t node,
                                           uint64_t label, uint64_t* out_child,
                                           uint8_t* out_found) {
    (void)context;
    if (out_child == NULL || out_found == NULL) {
        return VT_STATUS_NULL_POINTER;
    }
    if (node >= NODE_COUNT) {
        return VT_STATUS_INVALID_ARGUMENT;
    }
    const StaticNode* entry = &NODES[node];
    for (size_t i = 0; i < entry->edge_count; ++i) {
        if (entry->edges[i].label == label) {
            *out_child = entry->edges[i].node;
            *out_found = 1;
            return VT_STATUS_OK;
        }
    }
    *out_found = 0; /* absence is a result, not an error */
    return VT_STATUS_OK;
}

static VtStatus dictionary_node_edges(void* context, uint64_t node,
                                      size_t start, VtDictionaryEdge* out_edges,
                                      size_t capacity, size_t* out_written,
                                      size_t* out_total) {
    (void)context;
    if (out_written == NULL || out_total == NULL) {
        return VT_STATUS_NULL_POINTER;
    }
    if (node >= NODE_COUNT) {
        return VT_STATUS_INVALID_ARGUMENT;
    }
    const StaticNode* entry = &NODES[node];
    const size_t total = entry->edge_count;
    *out_total = total; /* stable across every page of this node */
    if (start >= total || capacity == 0) {
        *out_written = 0; /* an empty page terminates paging cleanly */
        return VT_STATUS_OK;
    }
    if (out_edges == NULL) {
        return VT_STATUS_NULL_POINTER;
    }
    const size_t remaining = total - start;
    const size_t written = remaining < capacity ? remaining : capacity;
    memcpy(out_edges, entry->edges + start, written * sizeof(VtDictionaryEdge));
    *out_written = written; /* out_written <= capacity, always */
    return VT_STATUS_OK;
}

/* ── constructor: the resource is born owning one retain ────────────────── */

VtResource static_dictionary_new(void) {
    StaticDictionary* dictionary = malloc(sizeof(StaticDictionary));
    if (dictionary == NULL) {
        return (VtResource){ NULL, NULL }; /* null resource signals failure */
    }
    atomic_init(&dictionary->refs, 1); /* the caller owns this first retain */
    return (VtResource){ dictionary, &RESOURCE_VTABLE };
}
```

A consumer exercises it exactly as the
[negotiation diagram](../../docs/diagrams/bindings/interface-negotiation-activity.svg)
prescribes: `static_dictionary_new()` hands over the first owned retain;
`query_interface(&VT_DICTIONARY_INTERFACE_ID, 1, &vt)` yields
`DICTIONARY_VTABLE`; `snapshot` mints a second owned handle (here the same
two words, refcount two); each handle is released exactly once, and the
second release frees the context.

---

## 11. References

1. George E. Collins. 1960. *A method for overlapping and erasure of
   lists.* Communications of the ACM 3(12), 655-657.
   DOI: [10.1145/367487.367501](https://doi.org/10.1145/367487.367501).
   — The original reference-counting discipline `retain` / `release`
   implements.
2. Don Box. 1998. *Essential COM.* Addison-Wesley. ISBN 0-201-63446-5.
   — The `IUnknown` triple (`AddRef`, `Release`, `QueryInterface`) whose
   portable core § 5.2 adopts, and the definitive treatment of why
   interface identity plus versioned discovery keeps independently shipped
   binaries compatible.
3. James R. Driscoll, Neil Sarnak, Daniel D. Sleator, and Robert E.
   Tarjan. 1989. *Making data structures persistent.* Journal of Computer
   and System Sciences 38(1), 86-124.
   DOI: [10.1016/0022-0000(89)90034-2](<https://doi.org/10.1016/0022-0000(89)90034-2>).
   — The persistence construction behind the $`\mathcal{O}(1)`$
   structurally shared snapshot contract of § 6.5.

<!--
DOI verification (2026-08-08): curl -sI --max-redirs 0 https://doi.org/<doi>
  10.1145/367487.367501        -> 302 (handle API responseCode 1)
  10.1016/0022-0000(89)90034-2 -> 302 (handle API responseCode 1; full-chain curl -sIL 200)
The ACM landing host (dl.acm.org) answers 403 to non-browser HEAD requests
after the 302, so the doi.org hop plus the handle API are the resolution
evidence; negative control 10.1145/9999999.9999999 -> 404 / responseCode 100.
-->

---

*Family canon:* this document is the normative ABI reference for every
vinary-tree repository. Producers and consumers cite it rather than restate
it: [liblevenshtein](https://github.com/vinary-tree/liblevenshtein-rust) ·
[libdictenstein](https://github.com/vinary-tree/libdictenstein) ·
[lling-llang](https://github.com/vinary-tree/lling-llang) ·
[duallity](https://github.com/vinary-tree/duallity).
