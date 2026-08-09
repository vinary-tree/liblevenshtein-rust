package io.vinarytree.liblevenshtein;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

    // C2/C3: a compiled pattern reports a positive size, closes idempotently, and
    // rejects use after close with IllegalStateException rather than crashing.
    @Test
    void phoneticPatternSizeIdempotentCloseAndClosedGuard() {
        PhoneticPattern pattern = PhoneticPattern.compileRegex("c[ao]t");
        AutomatonSize size = pattern.size();
        assertTrue(size.states() > 0 && size.transitions() > 0);
        assertTrue(pattern.matches("cat"));
        assertFalse(pattern.matches("cut"));
        pattern.close();
        pattern.close(); // idempotent
        assertThrows(IllegalStateException.class, () -> pattern.matches("cat"));
    }
}
