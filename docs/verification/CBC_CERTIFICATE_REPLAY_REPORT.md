# CBC certificate rejection model report

The finite [MetricCertificateReplay.tla](tla/MetricCertificateReplay.tla)
model checks a small certificate acceptance interface. A certificate has a
claim kind, full request scope, source and target names, a rewrite chain, and
a witness-presence marker. The model returns a **rejection reason** from each
check, so a malformed certificate cannot pass merely because a later,
unrelated check happens to fail.

This is a finite interface model. Its expression names and arithmetic labels
are strings; they have no numerical semantics in TLC. Universal rewrite
soundness for exact-natural, score-only expressions is proved in
[CertifiedContracts.v](temporal_automata/theories/CertifiedContracts.v) and
[CertificateChecking.v](temporal_automata/theories/CertificateChecking.v).
[CertificateScope.v](temporal_automata/theories/CertificateScope.v) proves
stable-request comparison and the separate cutoff rules for that expression
fragment. None of these files yet proves correspondence to a Rust executable.

## The acceptance path

`CheckExact` first checks the claim kind, then rejects unsupported arithmetic,
missing witness presence, unsupported observation profiles, and scope
mismatch. `Replay` checks each step's source, rule identity, premise presence,
rule pattern, and target before advancing. `CheckExact` finally compares the
replayed expression name with the claimed target. `CheckLower` is a separate
path with its own rule and direction checks. `ScopeAccepted` compares the
entire **eight-field model record** (domain, query, gap, stiffness,
arithmetic, revision, cutoff, observation); `BypassScopeCheck` exists only to
test the negative control. Other fields in the wider CBC contract need their
own model and source binding.

The valid exact chain has two steps:

```text
x_plus_zero_plus_zero --add_zero_outer--> x_plus_zero
x_plus_zero           --add_zero_right--> x
```

That chain makes truncation a real missing-step case: retaining only the first
step ends at `x_plus_zero`, while the certificate still claims `x`.

| Case | `CheckExact` result | Rejection point |
|---|---|---|
| Valid two-step exact chain | `accepted` | Both steps and final target agree |
| Valid witness-marked chain | `accepted` | Abstract witness-presence interface |
| Lawful lower certificate | `wrong_kind` | Exact and lower claims use separate paths |
| Stale revision | `scope_mismatch` | Captured revision differs |
| Changed domain, query, gap, stiffness, or cutoff | `scope_mismatch` | Full request record differs |
| Matching rounded arithmetic in request and certificate, or a changed request arithmetic | `unsupported_arithmetic` | Only the natural arithmetic tag is supported, even when scope fields match |
| Missing witness marker under an otherwise matching witness scope | `missing_witness` | A witness-profile request needs the marker |
| Wrong second-step source | `step_source_mismatch` | Chain adjacency fails |
| Fabricated first-step target | `rule_target_mismatch` | Named rule does not produce the claimed target |
| Missing second-step premise | `missing_premise` | Required premise is absent |
| Unsupported second-step rule | `unsupported_rule` | No rule case authorizes it |
| Truncated chain or changed final claim | `final_target_mismatch` | Replayed endpoint differs from claim |
| Lower/filter result disguised as exact | `final_target_mismatch` | An empty exact chain cannot prove the lower target |

The lawful lower certificate is accepted by `CheckLower`. Reversing its
source and target gives `lower_direction_mismatch`; replacing its rule gives
`unsupported_lower_rule`. Rounded arithmetic and witness observation tags
give `unsupported_arithmetic` and `unsupported_observation`, respectively,
even before scope comparison. The table contains **19 distinct exact-check
scenarios**. These are deliberately small and enumerated, not an exhaustive
enumeration of all certificates.

## Reproduction and observed results

Run from the repository root with TLC 2.19:

```sh
tlc -config docs/verification/tla/MetricCertificateReplay.cfg docs/verification/tla/MetricCertificateReplay.tla
tlc -config docs/verification/tla/MetricCertificateReplayMutant.cfg docs/verification/tla/MetricCertificateReplay.tla
```

On 2026-09-30, the clean configuration generated 38 states, found 19 distinct
states, had depth 1, and reported no invariant violation. Its checked
invariants were `ReasonClassification`, `ExactAndLowerAreDistinct`,
`LowerRuleDirectionIsChecked`, `LowerProfileIsChecked`, `ScopeMutationDetected`,
`PositiveControlsAccepted`, and `FilterCannotClaimExact`.

The scope-bypass mutant exited with status **151** and reported
`The invariant of ScopeMutationDetected is equal to FALSE`. With scope
comparison disabled, the stale-revision certificate passes its otherwise
valid exact chain. The mutant is required to fail; its failure is the model's
negative control.

The exact command output was saved outside `target/` in
`/tmp/cbc-rejection-clean.log` and `/tmp/cbc-rejection-mutant.log` for the
review run. TLC's `Next` stutters, so the 19 initial scenarios are the whole
reachable corpus; the depth-one result is expected. TLC checks each finite
scenario's classification and the named control properties. It does not
establish an unbounded theorem, the validity of a witness, or a result for a
Rust implementation.

The positive witness-marker case describes a **future** witness-aware
checker interface. The current Rocq expression checker accepts only
`ScoreOnly`; it correctly rejects `CanonicalWitness` even if a marker is
present. Before a witness-capable executable can use the model's positive
case, it needs a checker and proof for witness content, canonical tie rules,
and production correspondence. Similarly, this model's string arithmetic
tag cannot certify binary64 semantics. The proof and model claims must remain
separate until those obligations are discharged.
