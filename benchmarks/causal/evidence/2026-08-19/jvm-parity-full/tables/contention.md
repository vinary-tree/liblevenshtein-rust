# Cells measured under external load

7 of 90 cells were measured with a 1-minute load average above 8.0 on a 32-core host; 83 were measured quiet and 0 lack a snapshot (a cell only receives one when its batch completes, so in-flight batches appear here). Contention inflates dispersion rather than shifting medians systematically, which is why medians with MAD and bootstrap intervals are reported instead of means; cells listed below can be re-measured on a quiet machine. No cell overlapped a harness process invoked outside the runner's control.

# Cells with load average above the threshold

| cell | 1-min load | status |
| --- | --- | --- |
| jvm-vinary__dynamic_dawg__query__transposition__d2__oov | 16.180 | ok |
| jvm-legacy__own__query__transposition__d3__tr-d1 | 13.870 | ok |
| jvm-vinary__dynamic_dawg__query__transposition__d2__tr-d3 | 12.620 | ok |
| jvm-legacy__own__query__merge_and_split__d3__oov | 11.120 | ok |
| jvm-vinary__dynamic_dawg__query__standard__d3__std-d3 | 9.070 | ok |
| jvm-legacy__own__query__standard__d2__std-d2 | 8.680 | ok |
| jvm-legacy__own__query__merge_and_split__d3__std-d3 | 8.630 | ok |
