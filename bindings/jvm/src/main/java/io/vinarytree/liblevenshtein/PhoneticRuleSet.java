package io.vinarytree.liblevenshtein;

import static java.lang.foreign.MemoryLayout.PathElement.groupElement;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;

/** Reusable Unicode phonetic rewrite-rule set. */
public final class PhoneticRuleSet implements AutoCloseable {
    private static final long DATA = Native.OWNED_STRING.byteOffset(groupElement("data"));
    private static final long LEN = Native.OWNED_STRING.byteOffset(groupElement("len"));
    private MemorySegment handle;

    private PhoneticRuleSet(MemorySegment handle) { this.handle = handle; }

    /** Parse an import-free LLEV document. */
    public static PhoneticRuleSet parse(String source) {
        byte[] bytes = source.getBytes(StandardCharsets.UTF_8);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(Math.max(1, bytes.length), 1);
            if (bytes.length != 0) input.asSlice(0, bytes.length).copyFrom(MemorySegment.ofArray(bytes));
            MemorySegment out = arena.allocate(ADDRESS);
            Native.check(Native.rulesParse(input, bytes.length, out));
            return new PhoneticRuleSet(out.get(ADDRESS, 0));
        }
    }

    /** Construct a built-in rule set. */
    public static PhoneticRuleSet builtin(PhoneticRuleSetKind kind) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ADDRESS);
            Native.check(Native.rulesBuiltin(kind.nativeValue(), out));
            return new PhoneticRuleSet(out.get(ADDRESS, 0));
        }
    }

    /** Number of enabled rules. */
    public long size() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(JAVA_LONG);
            Native.check(Native.rulesLen(handle(), out));
            return out.get(JAVA_LONG, 0);
        }
    }

    /** Apply rules to a fixed point. */
    public String apply(String input) {
        byte[] bytes = input.getBytes(StandardCharsets.UTF_8);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment text = arena.allocate(Math.max(1, bytes.length), 1);
            if (bytes.length != 0) text.asSlice(0, bytes.length).copyFrom(MemorySegment.ofArray(bytes));
            MemorySegment out = arena.allocate(Native.OWNED_STRING);
            Native.check(Native.rulesApply(handle(), text, bytes.length, out));
            try {
                long length = out.get(JAVA_LONG, LEN);
                MemorySegment data = out.get(ADDRESS, DATA).reinterpret(length);
                return new String(data.toArray(JAVA_BYTE), StandardCharsets.UTF_8);
            } finally {
                Native.ownedStringFree(out);
            }
        }
    }

    private MemorySegment handle() {
        if (handle == null || handle.equals(MemorySegment.NULL)) throw new IllegalStateException("rule set is closed");
        return handle;
    }

    /** Release this rule set. */
    @Override public void close() {
        if (handle != null && !handle.equals(MemorySegment.NULL)) {
            Native.rulesFree(handle);
            handle = MemorySegment.NULL;
        }
    }
}
