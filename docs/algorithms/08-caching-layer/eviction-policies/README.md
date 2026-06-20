# Caching Layer — Eviction Policies

**Configurable cache eviction strategies for the caching layer (Layer 8).**

The caching layer stores query results behind a lock-free concurrent map and
evicts entries according to a pluggable policy. Each document here describes one
policy: its eviction criterion, the metadata it tracks, and the access patterns
it serves best. For the layer overview and integration, see the
[caching layer](../) index.

## Policies

| Document | Purpose |
|----------|---------|
| [lru.md](lru.md) | LRU (Least Recently Used) — evicts the least recently accessed entry; exploits temporal locality. |
| [lfu.md](lfu.md) | LFU (Least Frequently Used) — evicts the entry with the fewest accesses; frequency-based. |
| [ttl.md](ttl.md) | TTL (Time-to-Live) — evicts entries after a fixed expiry duration. |
| [age.md](age.md) | Age (FIFO) — evicts the oldest-inserted entry regardless of access. |
| [cost-aware.md](cost-aware.md) | CostAware — weights eviction by entry size / recomputation cost. |
| [memory-pressure.md](memory-pressure.md) | MemoryPressure — evicts in response to system memory pressure signals. |

**Status: Living reference.**

[← Documentation Index](../../../README.md)
