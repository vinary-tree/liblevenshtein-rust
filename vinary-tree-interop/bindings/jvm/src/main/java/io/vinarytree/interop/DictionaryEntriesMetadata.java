package io.vinarytree.interop;

import java.util.Optional;
import java.util.OptionalLong;

/** Immutable metadata captured together with a dictionary entries cursor. */
public record DictionaryEntriesMetadata(
        DictionaryUnitDomain unitDomain,
        DictionaryValueDomain valueDomain,
        OptionalLong exactLength,
        Optional<SnapshotIdentity> snapshotIdentity) {}
