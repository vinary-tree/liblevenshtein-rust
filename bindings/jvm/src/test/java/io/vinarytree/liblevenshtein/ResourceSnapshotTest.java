package io.vinarytree.liblevenshtein;

import org.junit.jupiter.api.Test;

/** End-to-end FFM snapshot semantics over a Java-defined provider. */
final class ResourceSnapshotTest {
    @Test
    void partiallyConsumedCursorKeepsItsQueryStartRevisionAndOutlivesOwners() {
        ResourceSnapshotSmoke.verify();
    }
}
