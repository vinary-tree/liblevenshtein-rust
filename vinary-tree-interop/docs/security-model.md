# Security model — the family trust model for the resource ABI

This is the canonical trust-model document for every vinary-tree repository
that produces or consumes resources over
[`vinary_tree_interop.h`](../include/vinary_tree_interop.h). Per-repo
security documents instantiate this model for their own surfaces; they cite
it rather than restate it. The threat analysis is grounded in *confirmed*
findings from the family's own ledgers, cited inline — this model was
written after auditing real code, not before.

Companions: [ABI reference](abi-reference.md) (the contract being defended),
[evolution policy](abi-evolution.md) (why unknown values are rejected, not
guessed at).

---

## 1. Assets

What the model defends, in priority order:

1. **Host-process memory safety.** No input arriving through the ABI —
   status codes, node identifiers, page counts, weights, labels — may cause
   undefined behavior in a consumer that performs its documented validation
   duties (§ 5).
2. **Availability.** No panic, foreign exception, or hostile input may
   abort, deadlock, or wedge the host process; faults surface as status
   codes on the same channel as every other error (§ 3, § 6).
3. **Ledger integrity.** The retain/release count of every resource balances
   on every path, including every error path — a leak is a slow outage and a
   double release is memory corruption (§ 5, § 6).
4. **Result integrity.** Data accepted from a provider is *validated against
   the ABI's laws* before it influences computation: garbage may be
   rejected, but must never be silently trusted (§ 5).
5. **Capability confinement.** Sandboxed deployments (WASI) receive exactly
   the filesystem capabilities the embedder granted — nothing ambient (§ 7).

## 2. Trust zones

![Component diagram of the trust zones: host application and language facade in the host-trusted zone, the four project C ABIs as hardened gates, the interop struct exchange plane, and the foreign provider zone behind the red trust boundary, with the containment duty annotated at each crossing.](../../docs/diagrams/bindings/trust-zones.svg)

One host process, one address space, five zones:

| Zone | Contents | Trust stance |
|---|---|---|
| Host application | The user's Python / JVM / JS / Go / C# / … code | Trusted; owns the process. |
| Language facade | The per-language binding layer (SafeHandle, Arena + Cleaner, ctypes + gc, RAII, …) | Trusted; owns handle lifetime and total error mapping. |
| Project C ABIs | `llev_*` · `ldict_*` · `lling_*` · `duallity_*` entry points | Trusted code, **hostile inputs**: every entry point assumes its arguments may be wrong and every panic must die inside (§ 3). |
| Interop exchange plane | This crate's structs and constants — two-word handles, POD payloads | Neutral: layouts and constants only; no code to trust or distrust. |
| Foreign provider | Whatever implementation stands behind a received vtable: another family repo, another *version* of a family repo, a third party, or an adversary | **Untrusted.** Every function pointer is adversarial-or-buggy until its outputs pass validation. |

The threat population behind "foreign" is deliberately broad because the
common case is not malice: it is a **buggy sibling** or a **version-skewed
sibling** (an old `libdictenstein.so` under a new consumer). The same
validation defends against all three; the difference malice makes is only
which guarantees are *out of scope* (§ 8).

## 3. The panic and exception containment law

**No unwinding may cross an `extern "C"` boundary, in either direction.**
Since Rust 1.81, a panic that reaches the edge of an `extern "C"` function
does not corrupt the caller — it **aborts the whole process**. Abort is
memory-safe and availability-fatal, so containment cannot be delegated to
callers or wrappers: it must live *inside* each implementation, at every
entry point, in every repository.

The family's implementations of the law, by file and line (verified at the
commit introducing this document):

- **liblevenshtein (consumer):** every `llev_*` entry point runs through
  `boundary()` at `src/ffi/index.rs:124-148` — `catch_unwind` wraps the
  operation; a caught panic is downcast to its message, stored in the
  thread-local `LAST_ERROR` slot (`src/ffi/index.rs:89-96`, read back by
  `llev_last_error_message`), and surfaced as the `Panic` status. Success
  paths clear the slot; failure paths store the message and return the
  mapped status.
- **libdictenstein (producer):** the equivalent `catch_unwind` boundary at
  `src/ffi.rs:352`.
- **lling-llang (producer + consumer):** `src/ffi.rs:76`.
- **duallity (consumer + producer):** `src/ffi.rs:73`.

Symmetrically, a provider written in an unwinding language (C++, exceptions;
Rust, panics) owes the same containment at its *own* vtable edge: a foreign
exception escaping a `retain`, `release`, `query_interface`, or interface
operation is undefined behavior or an abort in whichever runtime it lands.
Interface callbacks have a status channel — a contained provider fault is
`ProviderError` (§ 3 of the [ABI reference](abi-reference.md)); `retain` and
`release` return nothing, so they must be infallible by construction.

**Known gap (ledgered, scheduled):** the umbrella JS runtime crate's WASI
and browser entry points still contain panic-class sites
(unwrap/expect/unreachable) instead of status returns — finding LLEV-B4 in
liblevenshtein's `docs/bindings/FINDINGS_LEDGER.md`, fix scheduled with
no-panic regression gates. In a WASM instance a trap kills only the
instance, not the embedding process, which is why this gap is severity-lower
than a native equivalent would be — but the family discipline is
panic-free boundaries everywhere, and this document does not consider the
gap acceptable steady state.

## 4. Threading: serialization by default, parallelism by claim

The ABI's threading default is maximally conservative: **a consumer
serializes every callback to a captured provider** unless that provider
claims `PARALLEL_REENTRANT` for itself. The reference implementation is
liblevenshtein's `CallGate` (`src/bindings.rs:203-206`): at capture time the
consumer inspects the claim and installs either a pass-through
(`CallGate::Parallel`) or a mutex (`CallGate::Serial`) through which *every*
subsequent callback — including the final `release` in `Drop`
(`src/bindings.rs:142-155`) — is funneled. The gate's domain is one captured
provider object, not the process and not the resource (family contract pin
VT-GATE-1); independent providers proceed concurrently.

Consequences of a **false** `PARALLEL_REENTRANT` claim are, by construction,
scoped to the claimant:

- the races execute inside the *provider's* state — the consumer shares none
  of its own mutable memory with the provider, so the consumer's memory
  safety does not depend on the claim being true;
- the garbage a racy provider returns re-enters the consumer through the
  same validation (§ 5) as any other hostile output — it may produce wrong
  *results*, never wrong *memory*;
- the provider's own memory, however, is genuinely corrupted by its own
  races. A provider should claim the flag only with the receipts (immutable
  state, atomics, or internal locks — the
  [ABI reference's example provider](abi-reference.md#10-a-complete-minimal-provider-in-c)
  shows the honest minimal case).

The residual risk a consumer cannot remove: a provider that *blocks* inside
a callback stalls the calling thread (and, under a serial gate, every thread
queued on that provider). In-process there is no preemption to offer;
deployments that must bound this run the provider out-of-process or in a
sandbox (§ 8).

## 5. Input-validation duties

Everything a provider writes is input. The consumer duties, each grounded
in the law it enforces and — where one exists — the confirmed finding that
motivates it:

| Hostile input class | Example | Consumer duty | Law / evidence |
|---|---|---|---|
| Out-of-range status discriminant | Callback returns 42 as its `VtStatus` | Receive as raw `uint32_t`; validate membership in the known range **before** any enum conversion; map unknown to the provider-error class. In Rust, materializing an invalid `#[repr(u32)]` enum is undefined behavior *before any check can run*. | Finding **LLEV-B6** (high, confirmed): consumer callback sites received `VtStatus` by value — llev `docs/bindings/FINDINGS_LEDGER.md`. |
| Invalid weight representation | Tropical arc carrying $`-\infty`$; any NaN | Enforce the per-domain representation predicate at *ingestion* (capture, expansion, import) — not merely NaN-rejection. The motivating real case: lling-llang validated only `is_nan()`, so $`-\infty`$ entered tropical composition, where IEEE-754 yields $`(+\infty) + (-\infty) = \mathrm{NaN}`$ — manufacturing NaN *inside* the consumer and, on the import path, a reachable panic. | Finding **LLING-B2** = pre-registered family finding **F1** (high, confirmed, 11 sites): `lling-llang/docs/scientific-ledger/bindings-findings-ledger.md`; fix under invariant LLING-BRIDGE-4, formal home `proofs/coq/abi/WeightBridge.v`. |
| Nonzero reserved bytes | `VtOptionalU64.reserved` carrying nonzero bytes | Reject the value: reserved bytes are the evolution mechanism (they may *mean* something to a newer peer), so accepting garbage there today forecloses meaning tomorrow and masks provider bugs now. | Family law "reserved must be zero" (`src/lib.rs`); asymmetry finding **LLEV-B7** = F2: lling validates `VtWfstArc.reserved`, llev did not validate `VtOptionalU64.reserved` — harmonization under VT-ABI-5. |
| Paging inconsistency | `out_written` above `capacity`; `out_total` shrinking between pages; a page that never advances | Enforce the paging law (§ 6.5 (P) of the [ABI reference](abi-reference.md)): reject `written > capacity`, unstable totals, and non-progress; treat violation as `InvalidProviderOutput`, never as data. | Finding **LLEV-B8** = F3 (confirmed asymmetry): three consumers, three subtly different acceptance predicates — harmonized to the single proven predicate in `ConsumerAcceptance.v` (VT-PAGE-1..6). |
| Label outside the unit domain | Byte-domain edge labeled 700; a surrogate code point under `UNICODE_SCALAR` | Range-check labels against the declared domain and reject the expansion; truncating instead would alias distinct labels. | Unit-domain definitions, [ABI reference § 6.1](abi-reference.md#61-vtunitdomain-and-vtvaluedomain). |
| Bogus out-parameters | `out_valid` = 7; `has_value` = 3; a "boolean" of 200 | Accept only the documented value sets (zero or one); reject otherwise. lling's capture path already rejects `valid > 1` and `is_final > 1`. | Zero-or-one contracts throughout the header; `tests/discriminant_pins.rs` pins the zeroed defaults consumers compare against. |
| Untouched-output violation | `query_interface` fails but scribbles on `out_vtable` | Initialize outputs to poison/null before the call and trust them only on `Ok`. The negotiation tests poison the output slot and assert failed negotiations leave it untouched. | VT-QI-3; `tests/vtable_evolution.rs`. |
| Null vtable on success | `query_interface` returns `Ok` with a null interface pointer | Explicit null check after every successful negotiation (llev: `InvalidProviderOutput("query_interface returned a null vtable")`). | Reference consumer, `src/bindings.rs`. |

The through-line: **validation is cheap where it must run** — every check
above is $`\mathcal{O}(1)`$ per value crossing the boundary — and the
alternative to validating is not "trusting a peer", it is *importing the
peer's bugs as undefined behavior*.

## 6. Resource-exhaustion vectors

Memory-safe inputs can still be unaffordable inputs. The vectors and their
bounds:

| Vector | Attack shape | Bound |
|---|---|---|
| Inflated `out_total` | A page reports $`2^{60}`$ total edges to bait a full-list preallocation | Never allocate `out_total` up front. Page through a fixed-capacity buffer (`VT_RECOMMENDED_EDGE_BATCH` = 256) and let *traversal budgets* — automaton frontier size, result caps, node budgets — bound total work. `out_total` is a paging-consistency check (§ 5), not an allocation size. |
| Pathological degree / fan-out | A node with millions of real edges, repeated at every level | Same paging discipline plus the consumer's own budget; a consumer with a limit surfaces `LimitExceeded` instead of thrashing. |
| Unbounded lazy expansion | A `LAZY` WFST that mints fresh states forever; an `ACYCLIC` claim that is false | Bound the frontier and visited-set growth; treat exceeding the budget as a failure, not a reason to trust the claim. A false `ACYCLIC` claim must degrade a traversal into its budget, never into non-termination. |
| Refcount leaks | A consumer that forgets `release`; a provider whose `release` forgets to free | Leases and facade finalizers (SafeHandle / Cleaner / gc callbacks) make release structural on the consumer side; leak-discipline tests across the family's binding suites make it observable. A provider-side leak is the provider's own memory, but still a host-process outage vector — deployment-level memory caps are the final backstop. |
| Poisoned serial gate | A provider panic mid-callback leaves the consumer's gate mutex poisoned | The gate recovers the lock rather than propagating poison (llev `src/bindings.rs`: `unwrap_or_else(PoisonError::into_inner)` on both call and drop paths), so one contained fault does not wedge every later caller. |
| Message-channel flooding | Fault messages sized by the provider | The fault channel stores one `BindingError` per provider (`src/bindings.rs:121`, `fault: Arc<Mutex<Option<BindingError>>>`) and the thread-local `LAST_ERROR` holds one message per thread — constant space per provider and per thread by construction. |

## 7. WASI capability policy

Sandboxed deployments follow capability discipline — the sandbox's isolation
is only as good as the capabilities handed in:

- **Preopens only, no ambient filesystem.** The umbrella runtime
  instantiates WASI with an explicit guest-to-host preopen map
  (`bindings/javascript-runtime/wasi-runtime.mjs:28-33`: an isolated
  instance whose default grant is a single `/workspace` preopen, and the
  embedder chooses what — if anything — it maps to). No path outside the
  preopen set exists for the guest, and the runtime never widens the grant
  on its own.
- **Persistent backends only against real, granted storage.** The
  filesystem-backed persistent dictionary (`vt_persistent_artrie_create` /
  `vt_persistent_artrie_open`, `bindings/javascript-runtime/rust/src/wasi.rs`)
  operates **at preopened WASI paths only**, and only in builds whose `wasi`
  feature deliberately enables `libdictenstein/persistent-artrie`. The
  browser build ships no persistent backend at all — a durability guarantee
  that cannot be honored (no real mmap/fsync substrate) is not offered,
  rather than silently downgraded.
- **Trap containment at the instance boundary.** A WASM trap kills the
  instance, not the embedding host — the module-level analogue of § 3 — and
  the embedder decides whether to re-instantiate. The formal basis for
  relying on the sandbox's memory isolation is the WebAssembly semantics
  itself (Haas et al. [[2]](#9-references)).

## 8. Non-goals — stated so nobody relies on them

1. **In-process memory safety against a hostile provider is out of scope.**
   A foreign shared library in the same address space can scribble over any
   byte of the process without ever going through this ABI; no validation
   at the boundary can prevent that. This is not a defect of the model but
   arithmetic of the deployment shape. The formal framing: RustBelt
   [[1]](#9-references) proves that *safe* Rust is memory- and thread-safe
   provided every unsafe component upholds its verified contract — a
   theorem whose premises quantify over the program's own components.
   Arbitrary native code in the address space is outside those premises,
   so at this boundary safe-Rust guarantees mean: *the consumer adds no
   unsafety of its own* (its unsafe glue upholds documented contracts,
   ledgered in the family's unsafe-contract inventories), not that it can
   police its neighbor. What the validation duties (§ 5) buy against a
   *malicious* provider is precise: the consumer is never the one who
   converts hostile data into undefined behavior.
2. **Isolation, when required, is a deployment choice, not an ABI
   feature.** Run the untrusted provider inside a WebAssembly sandbox
   (whose isolation is part of the language's formal semantics — Haas et
   al. [[2]](#9-references)) or a separate process with an IPC bridge. The
   ABI's POD-and-status design is deliberately friendly to both.
3. **No confidentiality inside the process.** Any component can read any
   mapped memory; secrets are managed at process granularity.
4. **No supply-chain or build-integrity guarantees.** Signing, provenance,
   and reproducibility of the shared libraries are the packaging pipeline's
   concern.
5. **No liveness guarantee against a stalled callback** (§ 4): a provider
   that never returns holds its calling thread. Watchdogs exist at process
   granularity only.

## 9. References

1. Ralf Jung, Jacques-Henri Jourdan, Robbert Krebbers, and Derek Dreyer.
   2018. *RustBelt: securing the foundations of the Rust programming
   language.* Proceedings of the ACM on Programming Languages 2, POPL,
   Article 66. DOI: [10.1145/3158154](https://doi.org/10.1145/3158154).
   — What "safe Rust" formally guarantees, and thereby the precise scope of
   § 8.1.
2. Andreas Haas, Andreas Rossberg, Derek L. Schuff, Ben L. Titzer, Michael
   Holman, Dan Gohman, Luke Wagner, Alon Zakai, and JF Bastien. 2017.
   *Bringing the web up to speed with WebAssembly.* PLDI 2017, 185-200.
   DOI: [10.1145/3062341.3062363](https://doi.org/10.1145/3062341.3062363).
   — The formal sandbox model §§ 7-8 lean on for isolation-by-deployment.
3. Family ledgers cited inline: liblevenshtein
   `docs/bindings/FINDINGS_LEDGER.md` (LLEV-B4, LLEV-B6, LLEV-B7, LLEV-B8)
   and lling-llang `docs/scientific-ledger/bindings-findings-ledger.md`
   (LLING-B2 / F1).

<!--
DOI verification (2026-08-08): curl -sI --max-redirs 0 https://doi.org/<doi>
  10.1145/3158154         -> 302 (handle API responseCode 1)
  10.1145/3062341.3062363 -> 302 (handle API responseCode 1)
The ACM landing host answers 403 to non-browser HEAD requests after the
302; the doi.org hop plus handle-API responseCode 1 are the resolution
evidence (negative control 10.1145/9999999.9999999 -> 404 / 100).
-->

---

*Family canon:* per-repo instantiations cite this model —
[liblevenshtein](https://github.com/vinary-tree/liblevenshtein-rust) (consumer
specifics) · [libdictenstein](https://github.com/vinary-tree/libdictenstein)
(producer specifics) · [lling-llang](https://github.com/vinary-tree/lling-llang) ·
[duallity](https://github.com/vinary-tree/duallity).
