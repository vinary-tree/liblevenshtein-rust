# Conservative quiet-host rerun

These 29 otherwise valid cells passed the mandatory selected-CPU, sibling,
LLC, compiler, and foreign-harness gates, but their pre-window one-minute
system load average exceeded 8.0 on this 32-core host. They were removed from
the active evidence set on 2026-08-20 and retained here for audit before a
selective quiet-host rerun.

The resumable JVM runner skips the other 61 committed cells and remeasures only
these coordinates with the unchanged two-fork, five-warmup, ten-measurement
protocol. Files in this directory are historical candidates, not inputs to the
final aggregate.
