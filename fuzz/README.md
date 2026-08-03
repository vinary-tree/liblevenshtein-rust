# Resource-exhaustion fuzz harness

This standalone cargo-fuzz workspace exercises the six attacker-controlled
surfaces listed in the
[resource-exhaustion guide](../docs/security/resource-exhaustion.md). Keeping
the harness outside the main workspace prevents `libfuzzer-sys` and nightly
compiler requirements from entering the library's normal dependency graph.

## Target-to-contract map

| Target | Generated input | Checked invariant |
|---|---|---|
| `regex_nfa_resource` | arbitrary, lossily decoded pattern bytes | parsing, expansion preflight, NFA construction, and query setup do not bypass the state ceiling or crash |
| `bracket_state_growth` | every byte-sized bracket-kind and depth pair | successful construction never exceeds 4,096 states; excessive geometric growth is rejected before allocation |
| `true_damerau_budget` | bounded Unicode dictionaries, queries, and the complete `u8` budget domain | character-level automaton results and reported distances equal the Lowrance–Wagner reference |
| `banded_dtw` | arbitrary IEEE-754 samples and an explicit band | exact distance is symmetric and never NaN |
| `cost_scale_overflow` | arbitrary denominators, IEEE-754 weights, and machine-sized costs | scaling is deterministic and all inexact, non-finite, negative, or overflowing values return an error instead of wrapping |
| `msm_nonfinite` | arbitrary IEEE-754 bit patterns plus explicit NaN and both infinities | malformed inputs fail closed and neither the returned lower bound nor retained cells are NaN |

The true-Damerau target uses `DoubleArrayTrieChar`: both its oracle and the
backend must count Unicode scalar values. A byte-labelled trie intentionally
counts UTF-8 bytes and is therefore not a valid differential oracle for
non-ASCII text.

## Running the harness

Install a nightly toolchain and cargo-fuzz, then run:

```console
cargo +nightly fuzz check
cargo +nightly fuzz run regex_nfa_resource
cargo +nightly fuzz run bracket_state_growth
cargo +nightly fuzz run true_damerau_budget
cargo +nightly fuzz run banded_dtw
cargo +nightly fuzz run cost_scale_overflow
cargo +nightly fuzz run msm_nonfinite
```

Use a time or run limit in continuous integration; open-ended fuzzing is for a
dedicated worker. Under a debugger, tracer, or another ptrace-based runner,
LeakSanitizer cannot inspect the process. In that environment, set
`ASAN_OPTIONS=detect_leaks=0` and retain AddressSanitizer. This is an execution
environment accommodation, not a claim that leak checking ran.

Deterministic unit and compile-fail tests separately force every guard branch,
so a short fuzz smoke run need not rediscover a specific rejection seed. The
security guide records those tests and the corresponding formal invariant.
