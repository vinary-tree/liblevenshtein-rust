package io.vinarytree.liblevenshtein;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;

/** Reusable Unicode phonetic language automaton. */
public final class PhoneticPattern implements AutoCloseable {
    private MemorySegment handle;

    private PhoneticPattern(MemorySegment handle) { this.handle = handle; }

    /** Compile a phonetic regular expression. */
    public static PhoneticPattern compileRegex(String source) {
        return compile(source, false);
    }

    /** Compile an import-free LLRE document. */
    public static PhoneticPattern compileLlre(String source) {
        return compile(source, true);
    }

    private static PhoneticPattern compile(String source, boolean llre) {
        byte[] bytes = source.getBytes(StandardCharsets.UTF_8);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(Math.max(1, bytes.length), 1);
            if (bytes.length != 0) input.asSlice(0, bytes.length).copyFrom(MemorySegment.ofArray(bytes));
            MemorySegment out = arena.allocate(ADDRESS);
            Native.check(llre
                    ? Native.patternLlre(input, bytes.length, out)
                    : Native.patternRegex(input, bytes.length, out));
            return new PhoneticPattern(out.get(ADDRESS, 0));
        }
    }

    /** Test complete-string acceptance. */
    public boolean matches(String input) {
        byte[] bytes = input.getBytes(StandardCharsets.UTF_8);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment text = arena.allocate(Math.max(1, bytes.length), 1);
            if (bytes.length != 0) text.asSlice(0, bytes.length).copyFrom(MemorySegment.ofArray(bytes));
            MemorySegment out = arena.allocate(JAVA_BYTE);
            Native.check(Native.patternMatches(handle(), text, bytes.length, out));
            return out.get(JAVA_BYTE, 0) != 0;
        }
    }

    /** Return state and transition counts. */
    public AutomatonSize size() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment states = arena.allocate(JAVA_LONG);
            MemorySegment transitions = arena.allocate(JAVA_LONG);
            Native.check(Native.patternSize(handle(), states, transitions));
            return new AutomatonSize(states.get(JAVA_LONG, 0), transitions.get(JAVA_LONG, 0));
        }
    }

    MemorySegment handle() {
        if (handle == null || handle.equals(MemorySegment.NULL)) throw new IllegalStateException("pattern is closed");
        return handle;
    }

    /** Release this pattern. Existing query cursors remain valid. */
    @Override public void close() {
        if (handle != null && !handle.equals(MemorySegment.NULL)) {
            Native.patternFree(handle);
            handle = MemorySegment.NULL;
        }
    }
}
