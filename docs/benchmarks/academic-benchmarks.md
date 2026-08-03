# Academic Benchmark Reproduction

This page describes the repeatable entry points for the academic benchmark
workloads used in the elastic time-series and phonetic automata experiments.

The canonical runner is:

```bash
scripts/run-academic-benchmarks.sh all
```

It stores corpora and outputs under `target/academic-benchmarks/`, never under
`/tmp`. It also runs heavy Cargo commands through `systemd-run --user --scope`
with `MemoryMax` and `MemorySwapMax` controls by default.

## Workloads

| Workload | Academic data | Primary command | Output |
| --- | --- | --- | --- |
| Shared elastic exact 1-NN | UCR/UEA 2018 univariate time-series archive | `scripts/run-academic-benchmarks.sh elastic-ucr-all` | `target/academic-benchmarks/results/elastic_ucr_{msm,erp,twed,frechet,dtw}_*.csv` |
| One elastic measure | The same selected UCR slice | `scripts/run-academic-benchmarks.sh elastic-ucr --measure dtw` | one raw CSV plus `elastic_ucr_dtw_summary.tsv` |
| Legacy MSM exact 1-NN | UCR/UEA 2018 univariate time-series archive | `scripts/run-academic-benchmarks.sh msm-ucr` | `target/academic-benchmarks/results/msm_ucr_archive_*.csv` |
| Phonetic homophones | CMU Pronouncing Dictionary homophone groups | `scripts/run-academic-benchmarks.sh phonetic-cmudict` | `target/academic-benchmarks/results/phonetic_cmudict_*.jsonl` |

The elastic benchmark uses the UCR Time Series Archive described by Dau et al.
([DOI 10.48550/arXiv.1810.07758](https://doi.org/10.48550/arXiv.1810.07758)). The phonetic benchmark
uses the public CMUdict word-to-pronunciation lexicon
([cmusphinx/cmudict](https://github.com/cmusphinx/cmudict)).

## Runner Semantics

The runner's operational commands are:

```bash
scripts/run-academic-benchmarks.sh all
scripts/run-academic-benchmarks.sh elastic-ucr --measure msm
scripts/run-academic-benchmarks.sh elastic-ucr --measure erp
scripts/run-academic-benchmarks.sh elastic-ucr --measure twed
scripts/run-academic-benchmarks.sh elastic-ucr --measure frechet
scripts/run-academic-benchmarks.sh elastic-ucr --measure dtw
scripts/run-academic-benchmarks.sh elastic-ucr-all
scripts/run-academic-benchmarks.sh msm-ucr
scripts/run-academic-benchmarks.sh phonetic-cmudict
scripts/run-academic-benchmarks.sh clean-raw
```

`all` prepares both corpora, runs all five elastic arms, and then runs the
phonetic workload. `elastic-ucr` requires one allowlisted measure name;
`elastic-ucr-all` runs those names sequentially so their process time and peak
memory do not overlap. Both commands download or use an existing UCR archive,
extract it to
`target/academic-benchmarks/msm/Univariate_ts`, then runs the archive summary
and exact one-nearest-neighbor benchmark. `msm-ucr` preserves the older
MSM-only artifact format. `phonetic-cmudict` downloads or
uses an existing CMUdict file, then evaluates three profiles:

```text
zompist-default
american.llev + homophones.llev + names.llev, extensions before primary rules
en-us-cmudict
```

`clean-raw` removes downloaded and extracted corpora while preserving result
artifacts. Use it after recording artifacts in pgmcp or another experiment
ledger.

The runner writes metadata, checksum files, raw benchmark output, and compact
summary files under:

```text
target/academic-benchmarks/results/
```

Each generalized summary row reports flat candidate-bound pruning and exact
cutoff work, plus trie nodes, edges, prefix prunes, columns built, column
prunes, queued-subtree prunes, candidate prunes, exact evaluations, and cutoff
abandonments. It also reports the summed per-dataset elapsed time, process
high-water resident memory, configuration, and a native-distance checksum.
Two normalized case rows retain the paired majority/measure correctness
outcomes.

After all arms complete, run the independent artifact gate:

```bash
scripts/verify-elastic-ucr-gate.sh
```

It checks the 32-field schema, the same 51 dataset names and 13,754 case keys,
identical majority outcomes, every flat/trie accounting partition, and exact
historical MSM accuracy and pruning counts. The analytic rationale and observed
results are in the
[shared scientific ledger](../scientific-ledger/elastic-ucr-harness-2026-08-01.md).

## Resource Controls

Heavy benchmark commands are memory-capped by default:

```bash
MEMORY_MAX=4G MEMORY_SWAP_MAX=0 scripts/run-academic-benchmarks.sh all
```

The script also refuses wall-time-producing benchmark runs when the one-minute
load average is higher than `LOAD_LIMIT_FACTOR * core_count`. The default factor
is `0.75`.

```bash
LOAD_LIMIT_FACTOR=0.50 scripts/run-academic-benchmarks.sh msm-ucr
```

To intentionally bypass that guard:

```bash
ALLOW_HIGH_LOAD=1 scripts/run-academic-benchmarks.sh phonetic-cmudict
```

`systemd-run` is required unless explicitly bypassed:

```bash
ALLOW_UNCAPPED=1 scripts/run-academic-benchmarks.sh phonetic-cmudict
```

That uncapped path is for systems without user-scoped systemd. It should not be
used for large runs on constrained machines.

## Corpus Inputs

The default CMUdict URL is:

```text
https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
```

The script checks the observed CMUdict SHA-256 by default:

```text
81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22
```

Override it only when intentionally updating the corpus snapshot:

```bash
CMUDICT_SHA256_EXPECTED= scripts/run-academic-benchmarks.sh phonetic-cmudict
```

The UCR archive has moved between public archive paths over time, so the script
tries several known archive URLs and also supports explicit inputs:

```bash
UCR_ARCHIVE_URL=https://example.invalid/Univariate2018_ts.zip \
  scripts/run-academic-benchmarks.sh msm-ucr

UCR_ARCHIVE_PATH=/path/to/Univariate2018_ts.zip \
  scripts/run-academic-benchmarks.sh msm-ucr

UCR_ARCHIVE_ROOT=/path/to/Univariate_ts \
  scripts/run-academic-benchmarks.sh msm-ucr
```

The observed UCR archive checksum is recorded to:

```text
target/academic-benchmarks/results/ucr-archive-checksum.txt
```

## Tuning

The default MSM selector matches the completed academic run:

```bash
MSM_MAX_CELLS=1000000000 MSM_MAX_DATASETS=1000 \
  scripts/run-academic-benchmarks.sh msm-ucr
```

The generalized selector defaults to those same values:

```bash
ELASTIC_UCR_MAX_CELLS=1000000000 ELASTIC_UCR_MAX_DATASETS=1000 \
  scripts/run-academic-benchmarks.sh elastic-ucr-all
```

Its fixed configurations are MSM `$`c=1`$`, ERP `$`g=0`$`, TWED
`$`\nu=1`$` and `$`\lambda=1`$`, discrete Fréchet, and banded DTW with
`$`w=\max(1,\lceil0.1L\rceil)`$`. Quantization uses 256 uniform bins derived
from training data only. These settings are preregistered and are not tuned
from test accuracy or timing.

The default phonetic settings match the completed CMUdict homophone run:

```bash
PHONETIC_CASES=2048 PHONETIC_MAX_DISTANCE=2 PHONETIC_RECALL_K=5 \
  scripts/run-academic-benchmarks.sh phonetic-cmudict
```

Use `DRY_RUN=1` to validate the planned commands without downloading corpora or
running benchmarks:

```bash
DRY_RUN=1 scripts/run-academic-benchmarks.sh all
```

## Cargo Aliases

Cargo aliases are available for already-prepared corpora:

```bash
cargo academic-msm-ucr-summary
cargo academic-msm-ucr
cargo academic-phonetic-cmudict
cargo academic-phonetic-cmudict-diagnostic
```

These aliases do not download corpora, clean raw data, cap RSS, or perform the
load guard. Use the script for full reproduction and use the aliases when a
corpus has already been prepared in `target/academic-benchmarks/`.

## Artifact Flow

```text
public corpus
  -> target/academic-benchmarks/{msm,phonetic}/
  -> systemd-run capped Cargo example
  -> target/academic-benchmarks/results/raw artifacts
  -> pgmcp experiment artifact records
  -> scripts/run-academic-benchmarks.sh clean-raw
```

The raw artifacts are intentionally line-oriented CSV or JSONL so that pgmcp can
ingest paired binary counts, measurement summaries, or diagnostic categories
without rerunning the workload.
