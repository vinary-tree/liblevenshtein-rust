# CBC certificate rejection model report

The finite model [MetricCertificateReplay.tla](tla/MetricCertificateReplay.tla)
abstracts a certificate to a kind, scope, source and target names, a short
rewrite chain, and a witness-presence flag. It checks exact certificate
acceptance under a fixed requested scope. It does not model the Rust verifier,
the full ORC intermediate representation, or whether a supplied witness is
correct. The universal natural-expression rewrite theorem is separately
checked in [CertifiedContracts.v](temporal_automata/theories/CertifiedContracts.v).

## Reproduction

From the repository root:

```sh
tlc -config docs/verification/tla/MetricCertificateReplay.cfg docs/verification/tla/MetricCertificateReplay.tla
tlc -config docs/verification/tla/MetricCertificateReplayMutant.cfg docs/verification/tla/MetricCertificateReplay.tla
```

On 2026-09-29, TLC 2.19 explored 12 distinct states with the clean
configuration, found no invariant violation, and reported depth 1. The
scope-bypass mutant exited with status 151 and reported
`The invariant of ScopeMutationDetected is equal to FALSE`. This is the
expected negative control: a certificate carrying an old revision is accepted
if request/certificate scope equality is disabled.

| Scenario | Expected exact acceptance | Reason |
|---|---:|---|
| Valid exact rewrite | Yes | Rule, source, target, and scope agree |
| Valid witness-marked exact rewrite | Yes | Requested witness profile and marker agree |
| Lower-bound certificate | No | Certificate kind is not exact; the separate lower checker accepts its lawful direction |
| Stale revision or changed arithmetic | No | Scope equality fails |
| Missing witness marker | No | Requested witness profile is stronger |
| Wrong step source or target | No | Replay chain does not connect |
| Missing premise or unknown rule | No | Rule application is unavailable |
| Truncated chain or changed final claim | No | Replay result differs from target |

The separate lower checker rejects reversed and unsupported lower rules in
every reachable state. These cases are finite controls for the distinction
between exact and lower evidence, not a proof that a Rust implementation
cannot confuse the types.

The model's `Next` stutters; the 12 initial states are the entire finite
scenario corpus. TLC checks every scenario, not an unbounded execution or a
general theorem. Its arithmetic string is only a scope discriminator; the
Rocq checker separately restricts its semantics to `ExactNaturals` and
`ScoreOnly`. In particular, the model's positive witness-marker scenario
illustrates a future acceptance interface and is not accepted by the current
Rocq score checker.
