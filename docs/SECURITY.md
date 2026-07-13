# Security & Threat Model

This document describes liblevenshtein's trust boundaries, the untrusted-input
surfaces it exposes, and the posture at each. It is aimed at integrators
embedding the library in a service or another language runtime.

liblevenshtein is a **library**, not a network service: it has no listening
sockets, no ambient authority, and performs no I/O of its own except where a
feature explicitly reads files (grep, serialization). Its security profile is
therefore the profile of the **inputs it parses** and the **language boundaries
it crosses**.

## Trust boundaries at a glance

| Surface | Untrusted input | Trust boundary | Posture / mitigation |
|---|---|---|---|
| Core query API | query term `$W$`, distance `$k$` | none (pure, in-process) | total functions; `$k$` bounds the search; no allocation proportional to attacker-chosen constants beyond `$\mathcal{O}(\lvert W\rvert)$` state |
| `.llre` / regex DSL | a regular expression | `src/phonetic/regex`, `src/phonetic/llre` | compiled to an NFA (Thompson/Glushkov) → **linear-time matching, no catastrophic backtracking (ReDoS-resistant by construction)** |
| Grep (`grep-*` features) | files, archives, compressed & document formats | `src/grep/source.rs` | streaming extraction with archive-entry filtering; integrators must bound decompression (see below) |
| Serialization (`serialization`) | a serialized dictionary file | `src/serialization` | deserialize only trusted/own-produced artifacts; treat third-party blobs as untrusted |
| FFI (`ffi`) | raw C pointers | `src/ffi` (`unsafe extern "C"`) | documented caller contract; the boundary is `unsafe` by nature |
| WASM (`wasm`) | values from JavaScript | `src/wasm` (`wasm_bindgen`) | sandboxed by the Wasm runtime; validate term sizes at the host |

## 1 · Core query API — no trust boundary

The transducer query path is pure and in-process. Query construction is
`$\mathcal{O}(\lvert W\rvert)$`; each step is `$\mathcal{O}(k)$`; the search is bounded by
the dictionary and the error bound `$k$`. A hostile `$(W, k)$` cannot induce unbounded work
beyond what `$k$` and the dictionary size permit, and there is no path-dependent allocation an
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

## 3 · Grep — untrusted archives, compressed streams, and documents

The grep subsystem (`src/grep/source.rs`, behind the `grep-*` features) is the
widest untrusted-input surface: it transparently peels compressed streams
(gzip, zstd, xz, bzip2), walks archives (tar, zip) with an entry-glob filter, and
extracts text from document formats (PDF, DOCX, XLSX, EPUB, ODT). The relevant
classes of risk and the posture:

- **Decompression bombs.** A small archive can expand to a huge stream. The
  pipeline is *streaming* (it does not require materialising whole files), which
  bounds peak memory, but integrators scanning untrusted archives should still
  impose an overall output-size / time budget at the call site.
- **Path traversal (zip-slip).** Archive entries are matched against an
  entry-glob filter rather than being written to arbitrary paths; grep reads
  entry *contents* for matching and does not extract entries to the filesystem,
  which removes the classic zip-slip write primitive. If you extend grep to
  *write* extracted entries, sanitise entry paths.
- **Malformed documents.** Document extractors (PDF/DOCX/…) parse complex,
  attacker-controllable formats via third-party crates; treat extraction of
  untrusted documents as you would any untrusted parser — sandbox or resource-limit
  the process when scanning hostile corpora.

**Guidance.** When scanning untrusted input, run with an OS-level memory/time
limit (e.g. `systemd-run --scope -p MemoryMax=… -p RuntimeMaxSec=…`) and enable
only the document extractors you need.

## 4 · Serialization — deserialize only trusted artifacts

The `serialization` feature loads dictionaries from bincode, JSON, and protobuf
(optionally gzip-wrapped). Deserialization of **untrusted** data is a general
risk class (resource exhaustion on malformed input, logic errors on crafted
structures). The formats here are data-only (no code execution), but a crafted
blob can still drive large allocations.

**Guidance.** Deserialize only artifacts you produced or trust. For third-party
blobs, validate provenance (e.g. a signature) and deserialize under a resource
limit.

## 5 · FFI — the documented `unsafe` contract

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

## 6 · WASM — sandboxed, but validate sizes at the host

The `wasm` bindings (`src/wasm/`) run inside the host's WebAssembly sandbox, which
provides memory isolation. The residual concern is resource use: a caller can ask
for a large dictionary or a high-`$k$` query. Validate term counts/sizes and `$k$`
at the JavaScript host before crossing into Wasm.

## Scope

- **In scope:** memory-safety of the safe Rust API; denial-of-service resistance of
  the matching engines; the documented FFI contract; the parsing posture of the
  grep, serialization, and DSL surfaces.
- **Out of scope:** misuse of the `unsafe` FFI contract by the caller; security of
  the third-party document-parsing crates beyond how this crate invokes them;
  host-application authorization (the library has no notion of users or
  permissions).

## Reporting

Report suspected vulnerabilities via the project's GitHub repository
(`https://github.com/universal-automata/liblevenshtein-rust`) security advisory /
issue channel. Please include a reproducer and the affected feature flags.

---

[← Documentation Index](README.md)
