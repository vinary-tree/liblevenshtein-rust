# Security & Threat Model

This document describes liblevenshtein's trust boundaries, the untrusted-input
surfaces it exposes, and the posture at each. It is aimed at integrators
embedding the library in a service or another language runtime.

liblevenshtein is a **library**, not a network service: it has no listening
sockets or ambient authority. Its filesystem input/output is limited to APIs
that callers explicitly invoke, such as dictionary serialization. Its security
profile is therefore the profile of the **inputs it parses** and the **language
boundaries it crosses**.

## Trust boundaries at a glance

| Surface | Untrusted input | Trust boundary | Posture / mitigation |
|---|---|---|---|
| Core query API | query term $`W`$, distance $`k`$ | none (pure, in-process) | total functions; $`k`$ bounds the search; no allocation proportional to attacker-chosen constants beyond $`\mathcal{O}(\lvert W\rvert)`$ state |
| `.llre` / regex DSL | a regular expression | `src/phonetic/regex`, `src/phonetic/llre` | compiled to an NFA (Thompson/Glushkov) → **linear-time matching, no catastrophic backtracking (ReDoS-resistant by construction)** |
| Serialization (`serialization`) | a serialized dictionary file | `src/serialization` | deserialize only trusted/own-produced artifacts; treat third-party blobs as untrusted |
| FFI (`ffi`) | raw C pointers | `src/ffi` (`unsafe extern "C"`) | documented caller contract; the boundary is `unsafe` by nature |
| WASM (`wasm`) | values from JavaScript | `src/wasm` (`wasm_bindgen`) | sandboxed by the Wasm runtime; validate term sizes at the host |

## 1 · Core query API — no trust boundary

The transducer query path is pure and in-process. Query construction is
$`\mathcal{O}(\lvert W\rvert)`$; each step is $`\mathcal{O}(k)`$; the search is bounded by
the dictionary and the error bound $`k`$. A hostile $`(W, k)`$ cannot induce unbounded work
beyond what $`k`$ and the dictionary size permit, and there is no path-dependent allocation an
attacker can exploit. No special handling is required.

## 2 · The `.llre` / regex DSL — ReDoS-resistant by construction

User-supplied regular expressions are a classic denial-of-service vector
(*ReDoS*): backtracking engines can take exponential time on adversarial
patterns. liblevenshtein **does not backtrack**. A `.llre` pattern is compiled to
a nondeterministic finite automaton via Thompson/Glushkov construction
(`src/phonetic/nfa/thompson.rs`) and simulated in time linear in the input
length. The construction also exposes size/complexity hooks in
`src/phonetic/llre/ast.rs`, so a pattern's compiled size is bounded and knowable.

**Guidance.** Accepting third-party `.llre`/regex patterns is safe with respect
to matching time. If you accept very large *pattern sources*, bound the input
length before compilation as you would any parser input.

## 3 · Binary persistence — bound every decode

The `serialization` feature loads bincode dictionaries; `protobuf` adds Protocol
Buffers, and `compression` permits gzip around either binary stream. JSON, TOML,
and newline text are not persistence formats. Deserialization of **untrusted**
data is a general risk class (resource exhaustion on malformed input, logic
errors on crafted structures). The formats are data-only, but a crafted blob can
still drive large allocations or decompression work.

**Guidance.** Prefer authenticated artifacts. Bound compressed input,
decompressed output, decoder allocations, and elapsed work. The generalized
`OperationSet` bincode envelope validates magic, version, flags, exact payload
length/consumption, semantic invariants, and caller-configurable resource
limits. Its protobuf decoder first scans the wire without allocating decoded
collections, enforces operation/pair/name/text limits, requires a supported
container version, and then validates the semantic model. Protobuf unknown
fields are skipped for forward compatibility; they are not equivalent to
unstructured trailing junk. Operation-set gzip accepts exactly one checksummed
member, rejects concatenated members/trailing bytes, and caps inflated output
before invoking either inner decoder.

## 4 · FFI — the documented `unsafe` contract

Every FFI function is `unsafe extern "C"` (`src/ffi/`), as it dereferences raw
caller pointers. The contract, lifted from the module documentation, is:

- **Pointers must be valid and non-null**, and C strings must be **NUL-terminated**
  (`cstr_to_str` returns `None` on a null pointer but cannot validate an
  out-of-bounds or non-terminated pointer).
- **Returned memory is owned by the caller** and must be freed with the matching
  function — strings with `llev_string_free`, candidate arrays with
  `llev_candidates_free`, dictionaries/transducers with their `*_free` functions.
- **Freed pointers must not be reused** (no use-after-free, no double-free).

Violating the contract is undefined behaviour. Memory safety across this boundary
is the **caller's** responsibility; the Rust side upholds its half (it never hands
back a dangling pointer and validates nullness where it can).

## 5 · WASM — sandboxed, but validate sizes at the host

The `wasm` bindings (`src/wasm/`) run inside the host's WebAssembly sandbox, which
provides memory isolation. The residual concern is resource use: a caller can ask
for a large dictionary or a high-$`k`$ query. Validate term counts/sizes and $`k`$
at the JavaScript host before crossing into Wasm.

## Scope

- **In scope:** memory-safety of the safe Rust API; denial-of-service resistance of
  the matching engines; the documented FFI contract; and the parsing posture of
  serialization and the domain-specific languages (DSLs).
- **Out of scope:** misuse of the `unsafe` FFI contract by the caller; security of
  host-application authorization (the library has no notion of users or
  permissions); and the CLI application's archive, compression, document, and
  optical character recognition (OCR) parsers. See the CLI repository's
  [security guide](https://github.com/vinary-tree/liblevenshtein-rust-cli/blob/master/docs/security.md)
  for those application-owned boundaries.

## Reporting

Report suspected vulnerabilities via the project's GitHub repository
(`https://github.com/vinary-tree/liblevenshtein-rust`) security advisory /
issue channel. Please include a reproducer and the affected feature flags.

---

[← Documentation Index](README.md)
