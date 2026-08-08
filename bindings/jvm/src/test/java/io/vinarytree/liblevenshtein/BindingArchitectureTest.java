package io.vinarytree.liblevenshtein;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import io.vinarytree.interop.InteropLayouts;
import org.junit.jupiter.api.Test;

/** Binding ownership and ABI layout tests. */
final class BindingArchitectureTest {
    @Test
    void sharedResourceIsExactlyTwoWords() {
        assertEquals(2L * Long.BYTES, InteropLayouts.RESOURCE.byteSize());
    }

    @Test
    void liblevenshteinDoesNotPublishDictionaryConstructors() {
        assertFalse(java.util.Arrays.stream(Transducer.class.getConstructors())
                .anyMatch(constructor -> constructor.getParameterCount() == 0));
    }
}
