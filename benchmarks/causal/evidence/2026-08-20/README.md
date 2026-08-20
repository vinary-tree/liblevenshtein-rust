# Closing AMD uProf evidence

These are the final post-optimization, CLI-only AMD uProf 5.3.521.0 captures
for the Java-parity campaign. Both binaries were built with
`RUSTFLAGS='-C target-cpu=native'`, pinned to CPU 3 by `profile-headless.sh`,
and captured with the `hotspots` preset and call-stack depth 64.

| capture | exact binary SHA-256 | report SHA-256 |
|---|---|---|
| query | `ab520efc3dacc8891d60b0c1c16960670abd1ce2705f586dad88d29431f11a0a` | `fc8bc953a3d94b733b4bb4dc6b45628ba1d2ffe09d923335039c6becbe070810` |
| unordered construction | `7886db69c52dc8fb8ca04db290890a11b7b88573935562b05d22dda25f666808` | `c370d90ab6049bc500c5ad52b54b4db7ee38d75d4f061d96346407a888efa16e` |

The query capture contains 500 full 1,000-query `standard/d1/hits` passes.
The construction capture contains five warmups and 300 builds of the shuffled
79,343-term byte dictionary through `from_terms`. Each directory's
`environment.txt` records the exact command, machine, kernel, tool versions,
and affinity; `collect.log` retains program output; `report.csv` is the
detailed text report; and `uprof/` contains the raw session.

`uprof-profile-host-load.jsonl` is the topology ledger (SHA-256
`eb930eab8d7f2580ce23de2ff5a97c9923111c2e8045e523c49f1e764bcc8105`).
The first sample, taken immediately after compilation, was rejected at 11.49%
LLC-mean utilization and did not start a profile. The admitted retry and both
pre/post capture gates satisfy 10% selected-CPU, SMT-sibling, and LLC-mean
limits plus the 20% LLC-peer limit.

No GUI command was used. The wrapper invokes only `AMDuProfCLI collect` and
`AMDuProfCLI report`; the campaign's Heaptrack mode likewise mandates
`--record-only` and `heaptrack_print`.
