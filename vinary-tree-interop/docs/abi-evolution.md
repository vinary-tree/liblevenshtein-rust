# ABI evolution policy

How the surface defined in [`vinary_tree_interop.h`](../include/vinary_tree_interop.h)
is allowed to change. The policy has one organizing idea, inherited from the
COM lineage of the [ABI reference](abi-reference.md): **published bytes are
forever**. Compatible growth *adds* — trailing vtable members, new enum
values, new flag bits, new interfaces — and is discovered at runtime;
anything else *forks the identity* it changes, and old and new identities
coexist through `query_interface` until the old one is unconsumed.

![State diagram of ABI evolution: the published v1 surface grows through additive states (trailing vtable operations, appended enum values, claimed flag bits, new optional interfaces) that keep the same identity, or forks across the breaking boundary into a new interface ID or a coordinated VT_ABI_VERSION flag day, coexisting with v1 through negotiation.](../../docs/diagrams/bindings/abi-evolution-timeline.svg)

---

## 1. The four version counters

Four *distinct* counters version four distinct things. Conflating them is
the classic ABI failure mode, so each row states exactly what its counter
gates — and, as importantly, what it does **not**.

| Counter | Today | Gates | Checked where | Does not gate |
|---|---|---|---|---|
| `VT_ABI_VERSION` | 1 | The **base resource protocol**: the layout and meaning of `VtResource` and `VtResourceVTable` — two words, retain/release semantics, the `query_interface` signature. | Written by every provider into `VtResourceVTable.abi_version`; the reference consumer checks **exact equality** (`validate_base` in liblevenshtein `src/bindings.rs` rejects any value other than 1). A bump is therefore a coordinated, family-wide flag day: every producer and consumer rebuilds together. | Interface contracts, project C surfaces, package versions. |
| Interface version (`VT_DICTIONARY_INTERFACE_VERSION` = 1, `VT_WFST_INTERFACE_VERSION` = 1) | 1 / 1 | **One interface's contract** under one 16-byte identifier: its vtable's guaranteed prefix and operation semantics. | Negotiated per resource: the consumer passes its `minimum_version` to `query_interface`; the provider answers `Unsupported` if it cannot honor it. Consumers validate the discovered vtable with a **minimum** check (`interface_version >=`), never equality. The identifier string itself (`…v1`, `….1`) carries the *fork* counter for breaking revisions. | The base protocol; sibling interfaces; anything outside the named interface. |
| Project API revision (llev 1 · ldict 4 · lling 1 · duallity 1) | see left | Each project's **own C surface** above the interop layer: an additive counter bumped when the project adds `llev_*` / `ldict_*` / `lling_*` / `dual_*` functions. Declared as `apiRevision` in each repo's `bindings/api.json` and surfaced at runtime where the project exposes it (e.g. `LDICT_API_REVISION` = 4 in libdictenstein `src/ffi.rs`, returned by `ldict_api_revision()`). | Facade preflight checks in the language bindings: a facade built against revision $`n`$ refuses a library reporting less than $`n`$. | The interop structs — a project may add fifty functions without touching this crate. |
| Package semver (interop 0.1.0 · llev 0.10.0 · ldict 0.2.1 · lling 0.2.0 · duallity 0.3.0) | see left | **Distribution only**: crates.io / npm / PyPI / Maven coordinates and the version pins between packages. | Package managers and each repo's `bindings/related-projects.json` pins. | Any byte of the ABI. A patch release must not change layouts; conversely an additive interface version may ship in a minor package release. |

The separation is what makes mixed deployments work: a Python wheel built
against llev apiRevision 1 + ABI 1 loads a newer `liblevenshtein` shared
library (apiRevision, say, 3) and consumes dictionaries from a newer
`libdictenstein` — every counter that matters is checked at its own gate,
at runtime, with a portable failure (`Unsupported` / preflight error) rather
than corruption.

---

## 2. Additive evolution — same identity, one binary serves everyone

### 2.1 Trailing vtable members, gated by `struct_size`

Every vtable's first member is its own size in bytes. A future revision may
append members **strictly at the tail**; a consumer decides whether a member
exists by the gate

```math
\text{member exists} \iff \mathtt{struct\_size} \;\ge\; \mathrm{offset}(\text{member}) + \mathrm{width}(\text{member}),
```

which is monotone in `struct_size` because appended members only grow it.
Old consumers never look past the v1 prefix; new consumers degrade
gracefully when the gate fails. This is not aspiration — it is *proven
executable* by `tests/vtable_evolution.rs`
(`successful_negotiation_yields_a_prefix_consumable_future_vtable`): a
hand-rolled provider publishes a vtable strictly larger than v1 with one
trailing operation, a v1-only consumer reads the prefix untroubled, and an
extension-aware consumer discovers the trailing operation through
`struct_size` and calls it.

Appending a member is normally paired with bumping the interface-version
constant (worked example A below) so that consumers who *require* the new
operation can demand it via `minimum_version` instead of probing sizes.

### 2.2 New optional interfaces

A new capability ships as a **fresh 16-byte identifier** plus its own vtable
type and version constant — never as a semantic change to an existing
interface. Legacy consumers ask for identifiers they know and receive
`Unsupported` for ones the provider lacks; negotiation degrades gracefully
in both directions. (This is also the mechanism for breaking forks, § 3:
`vt.dictionary.v2` is "a new optional interface" from the protocol's point
of view.)

### 2.3 New enum values — per-enum rules

All interop enums are `uint32_t` with pinned, contiguous discriminants
(`tests/discriminant_pins.rs` asserts both properties and its wildcard-free
`match` tables make *any* addition a compile error until the pins — and this
policy — are consulted). Values are only ever **appended at the next
contiguous discriminant**; removal or renumbering is impossible (§ 3.2).
What a v1 consumer does with an unknown value differs per enum, and that
difference is the policy:

| Enum | Next free value | Unknown-value behavior required of consumers |
|---|---|---|
| `VtStatus` | 9 | **Degrade**: receive as raw `uint32_t`, validate against the range known at build time, and map anything outside it to the project's provider-error class. Unknown statuses are always failures, so degrading is safe. |
| `VtUnitDomain` | 4 | **Reject** the interface (incompatible): a consumer that cannot interpret the label encoding cannot traverse at all. |
| `VtValueDomain` | 3 (`Bytes` = 2 is already declared-but-reserved) | **Reject** interfaces whose value domain the consumer does not implement — exactly what the reference consumer does with `Bytes` today. |
| `VtWeightDomain` | 8 | **Reject**: weights in an unknown semiring cannot be combined, compared, or validated. |

Reserved *fields* follow the same append-only spirit: they must be written
zero today (§ 5.2 of the [ABI reference](abi-reference.md)) precisely so a
future version can assign them meaning — but only a meaning for which zero
denotes the v1 behavior; otherwise the change is breaking.

### 2.4 New capability flag bits

Flags are self-claims (`dictionary_flags`, `wfst_flags`). A new claim takes
the **next free bit** (dictionary: bit 3; WFST: bit 4). Consumers must
ignore bits they do not know — a claim a consumer cannot exploit is simply
unexploited — and the reference consumer does exactly that (`validate_dictionary`
inspects only the bits it understands). Bits are never reused (§ 3.2).

---

## 3. Breaking changes — fork the identity, never edit it

### 3.1 What is breaking

Any of the following invalidates an existing contract and **must not** be
made under the existing identity:

- **Layout**: reordering, inserting mid-struct, resizing, or removing any
  published field of any published type (would silently shear every offset
  behind it — the failure `tests/layout_contract.rs` exists to catch).
- **Ownership**: changing retain/release semantics, who frees what, or the
  copy-is-not-retain law — this is *base-protocol* territory and forces a
  `VT_ABI_VERSION` flag day.
- **Status semantics**: changing what an existing `VtStatus` value means,
  or making a previously illegal value legal (e.g. `End` from interface
  callbacks — pinned illegal, family contract F5).
- **Callback signatures**: any parameter or return change to a published
  function-pointer type.
- **Guarantee weakening**: relaxing the $`\mathcal{O}(1)`$ snapshot-capture
  contract, snapshot immutability, node/state-identifier validity windows,
  or the paging algebra — consumers are *built on* these laws
  (§ 6.5 of the [ABI reference](abi-reference.md)).

### 3.2 How a fork works

A breaking interface revision mints a **new 16-byte identifier** — the
version fork counter lives in the identifier string itself, so
`vt.dictionary.v1` becomes `vt.dictionary.v2` (and `vt.scalar-wfst.1`,
`vt.scalar-wfst.2`; the sixteen-byte budget is why the WFST name spends only
one byte on it). Coexistence is then ordinary negotiation:

- a migrating **provider** answers *both* identifiers from one
  `query_interface`, serving v1 consumers the v1 vtable and v2 consumers the
  v2 vtable, from the same resource;
- an updated **consumer** asks for v2 first and falls back to v1 on
  `Unsupported`;
- v1 is retired only when telemetry/pins say nothing consumes it — and its
  identifier bytes, discriminants, and flag bits remain **burned forever**:
  they are never reassigned, so a stale binary can never misinterpret a new
  contract as an old one.

A base-protocol break (`VT_ABI_VERSION` 2) has no such coexistence story —
the reference consumers check equality, by design — which is why the base
vtable was kept to the minimal, maximally stable triple in the first place.

---

## 4. Worked examples

### A. Add a vtable operation (`node_degree_hint` on the dictionary)

*Change class:* trailing vtable member (§ 2.1). *Identity kept.*

1. `src/lib.rs` — append
   `pub node_degree_hint: Option<unsafe extern "C" fn(context: *mut c_void, node: u64) -> u64>`
   as the **last** field of `VtDictionaryVTable`, with its contract in the
   doc comment; bump `VT_DICTIONARY_INTERFACE_VERSION` to 2.
2. `include/vinary_tree_interop.h` — append the matching member last in
   `VtDictionaryVTable`; bump `VT_DICTIONARY_INTERFACE_VERSION` to `2u`.
3. `tests/layout_contract.rs` — extend the field table in
   `every_struct_is_packed_in_declaration_order` and the exact pins: size
   96 / offset 88 on the 64-bit tier, size 60 / offset 56 on the 32-bit ARM
   tier; `published_constants_are_pinned` gains the new version value.
4. `tests/vtable_evolution.rs` — the `ExtendedDictionaryVTable` fixture
   becomes the published layout; add a negotiation case for
   `minimum_version = 2` against a v1-only provider (expects `Unsupported`).
5. The nine language mirrors under [`bindings/`](../bindings/) — append the
   member to each struct mirror (Fortran, Go, Haskell, JavaScript, JVM, Lua,
   OCaml, Python, Swift) and regenerate the conformance table via
   `scripts/generate-bindings.py`.
6. Docs and formal homes — new op-by-op entry in the
   [ABI reference](abi-reference.md) § 6.4; new row in the layout manifest
   (`docs/verification/ABI_LAYOUT_MANIFEST.tsv` and its `abi_layout.smt2`
   self-consistency model in the liblevenshtein repository); update § 5
   below.
7. Providers now advertise `interface_version = 2` with
   `struct_size = sizeof` of the extended table — and keep serving
   `minimum_version = 1` consumers through the unchanged prefix.

### B. Add a weight domain (`VT_WEIGHT_DOMAIN_MIN_MAX_F64 = 8`)

*Change class:* appended enum value (§ 2.3). *No version bump anywhere —
pure addition.*

1. `src/lib.rs` — append `MinMaxF64 = 8` to `VtWeightDomain` with a doc
   comment naming the semiring.
2. `include/vinary_tree_interop.h` — append
   `VT_WEIGHT_DOMAIN_MIN_MAX_F64 = 8` to the enum.
3. `tests/discriminant_pins.rs` — the wildcard-free `match` in
   `weight_domain_discriminants_are_pinned` **fails to compile** the moment
   step 1 lands; extend the exact pin (8), the universe array, the `seen`
   table, and the contiguity range. This forcing function is the policy's
   enforcement mechanism working as designed.
4. The [ABI reference](abi-reference.md) § 7.1 — define the semiring as a
   display-math signature $`\langle K, \oplus, \otimes, \bar{0}, \bar{1} \rangle`$
   with its representation predicate (which bit patterns are valid).
5. lling-llang `proofs/coq/abi/WeightBridge.v` — add the domain's
   `repr_ok` predicate and exactness lemmas (for min/max both operations
   are rounding-free, so this domain's laws hold exactly — worth stating).
6. The nine language mirrors — add the constant.
7. Providers may advertise the new domain immediately; every existing
   consumer already rejects unknown weight domains (§ 2.3), so nothing
   breaks while consumers learn it one by one.

### C. Retire a capability flag (hypothetically, `SUFFIX_BASED`)

*Change class:* deprecation. *Nothing is deleted — bits are burned, never
freed.*

1. `src/lib.rs` — keep `pub const SUFFIX_BASED: u64 = 1 << 1;` forever; mark
   it `#[deprecated]` with a note pointing at the replacement, and say so in
   the doc comment.
2. `include/vinary_tree_interop.h` — keep
   `VT_DICTIONARY_FLAG_SUFFIX_BASED` defined; annotate it deprecated in the
   comment.
3. `tests/discriminant_pins.rs` — the bit pins **stay**: the value of bit 1
   is an ABI fact independent of whether anyone sets it.
4. Providers stop setting the bit; consumers keep (or no-op) their handling
   — an unset claim requires nothing of anyone, which is why flag
   retirement is the gentlest change class in the table.
5. Record the burned bit in § 5; the next dictionary flag claims bit 3, not
   the retired bit 1 — a stale provider still setting bit 1 must never be
   misread as making some new claim.

---

## 5. Decision table and current compatibility matrix

### 5.1 Change class → required action

| Change class | Additive? | Required action |
|---|---|---|
| Append trailing vtable member | yes | Bump the interface-version constant; `struct_size` gates presence; extend layout pins, mirrors, manifest (example A). |
| Append enum value | yes | Next contiguous discriminant only; extend `discriminant_pins.rs` (compile-forced); consumers keep reject/degrade behavior per § 2.3 (example B). |
| Claim new flag bit | yes | Next free bit; consumers ignore unknown bits; document the claim. |
| Publish new interface | yes | Fresh 16-byte identifier + vtable type + version constant; legacy consumers get `Unsupported`. |
| Give meaning to a reserved field | conditional | Only if zero continues to denote the v1 behavior; otherwise it is a layout change → fork. |
| Deprecate a flag / operation | yes | Keep the constant and pins forever; stop setting/calling; ledger the burned bit (example C). |
| Reorder / insert / resize / remove a field | **no** | New interface identifier (§ 3.2); the old layout ships until unconsumed. |
| Change retain/release or copy semantics | **no** | New `VT_ABI_VERSION` — family-wide flag day; exhaust every alternative first. |
| Change an existing status's meaning | **no** | Fork every interface whose contract references it. |
| Change a callback signature | **no** | New interface identifier. |
| Weaken snapshot / paging / validity-window guarantees | **no** | New interface identifier — consumers are built on the laws. |
| Remove or renumber an enum value / flag bit | **never** | Impossible under this policy; values are burned forever. |

### 5.2 Compatibility matrix (verified 2026-08-08)

Every constant below was read from the named source in the sibling checkout,
not quoted from memory.

| Surface | Version | Producers | Consumers | Pinned by |
|---|---|---|---|---|
| Base resource ABI | `VT_ABI_VERSION` = 1 | all four projects | all four projects | `src/lib.rs` · header · `tests/layout_contract.rs` (`published_constants_are_pinned`) |
| `vt.dictionary.v1` | interface version 1 | libdictenstein (4 backends) | liblevenshtein · duallity | `VT_DICTIONARY_INTERFACE_VERSION` in `src/lib.rs` / header; negotiation pinned by `tests/vtable_evolution.rs` |
| `vt.scalar-wfst.1` | interface version 1 | lling-llang · duallity | lling-llang (composition) · duallity | `VT_WFST_INTERFACE_VERSION` in `src/lib.rs` / header |
| liblevenshtein C surface | apiRevision 1 · package 0.10.0 | — | 15 language facades | `bindings/api.json` |
| libdictenstein C surface | apiRevision 4 · package 0.2.1 | — | 13 language facades | `bindings/api.json`; `LDICT_API_REVISION` in `src/ffi.rs` via `ldict_api_revision()` |
| lling-llang C surface | apiRevision 1 · package 0.2.0 | — | JS + C/C++ facades | `bindings/api.json` |
| duallity C surface | apiRevision 1 · package 0.3.0 | — | JS + C/C++ facades | `bindings/api.json` |
| interop crate | package 0.1.0 · Rust 1.95 or newer · `no_std` | — | all of the above + 9 interop mirrors | `Cargo.toml` |
| umbrella JS runtime | hosts all four projects + interop | — | Node / WASI / browser | `bindings/javascript-runtime/rust/Cargo.toml` |

Known pin inconsistencies (release execution is out of scope for this
program) are ledgered, not fixed here: see finding LLEV-B9 in
liblevenshtein's `docs/bindings/FINDINGS_LEDGER.md`.

---

*See also:* [ABI reference](abi-reference.md) — the surface this policy
governs · [security model](security-model.md) — why unknown values are
rejected, not trusted · [README](../README.md) — the catalog of who
produces and consumes each interface.
