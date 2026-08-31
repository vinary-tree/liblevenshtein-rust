package io.vinarytree.liblevenshtein;

/** Immutable TinyLFU/SIEVE counters and current bounded native residency. */
public record QueryCacheStats(
        long requests,
        long hits,
        long misses,
        long admissions,
        long rejections,
        long evictions,
        long residentEntries,
        long residentWeight) {}
