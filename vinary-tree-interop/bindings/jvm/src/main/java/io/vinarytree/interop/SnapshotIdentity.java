package io.vinarytree.interop;

/** Opaque process-local identity of one immutable producer revision. */
public record SnapshotIdentity(long producer, long revision) {}
