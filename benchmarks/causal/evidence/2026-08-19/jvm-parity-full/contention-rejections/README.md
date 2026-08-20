# Rejected JVM timing attempts

This directory is outside the accepted-cell and accepted-admission ledgers.
Nothing stored here contributes to `cells/`, `host-load-admission.jsonl`, or
the paired matrix statistics.

The retained `standard/d1/hits` log predates the invocation-tree guard fix. The
process scanner saw the Gradle launcher and its forked JMH JVM, both of whose
command lines name `VinaryBench`, and incorrectly classified the two
descendants of one timed command as two concurrent cells. The monitor rejected
the raw result. That raw JMH JSON is retained under `jmh/invalid-retries/`.

The corrected guard anchors ownership at the direct child of
`run-with-contention-monitor.sh`, treats all descendants of that child as one
managed-runtime invocation, and still rejects any runner-owned harness outside
the monitored tree. The coordinate was then measured again from scratch and
admitted only after its exact pre/post topology records committed.
