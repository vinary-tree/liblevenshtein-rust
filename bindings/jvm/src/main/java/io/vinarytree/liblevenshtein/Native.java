package io.vinarytree.liblevenshtein;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

/** Java FFM downcalls for the stable project ABI. No JNI is used. */
final class Native {
    static final int OK = GeneratedAbi.STATUS_OK;
    static final int END = GeneratedAbi.STATUS_END;
    static final int DOMAIN_BYTE = 1;
    static final int DOMAIN_UNICODE = 2;
    static final int DOMAIN_U64 = 3;

    static final MemoryLayout MATCH = MemoryLayout.structLayout(
            ADDRESS.withName("term_data"),
            JAVA_LONG.withName("term_len"),
            JAVA_LONG.withName("byte_len"),
            JAVA_LONG.withName("distance"),
            JAVA_LONG.withName("id"),
            JAVA_INT.withName("unit_domain"),
            JAVA_BYTE.withName("has_id"),
            MemoryLayout.sequenceLayout(3, JAVA_BYTE).withName("reserved"));
    static final MemoryLayout BATCH = MemoryLayout.structLayout(
            ADDRESS.withName("matches"),
            JAVA_LONG.withName("len"),
            JAVA_LONG.withName("generation"));
    static final MemoryLayout OWNED_STRING = MemoryLayout.structLayout(
            ADDRESS.withName("data"), JAVA_LONG.withName("len"));

    private static final Linker LINKER = Linker.nativeLinker();
    private static final MethodHandle LAST_ERROR;
    private static final MethodHandle TRANSDUCER_NEW;
    private static final MethodHandle TRANSDUCER_SNAPSHOT;
    private static final MethodHandle TRANSDUCER_FREE;
    private static final MethodHandle QUERY_UTF8;
    private static final MethodHandle QUERY_BYTES;
    private static final MethodHandle QUERY_U64;
    private static final MethodHandle QUERY_PATTERN;
    private static final MethodHandle NEXT_BATCH;
    private static final MethodHandle RELEASE_BATCH;
    private static final MethodHandle CURSOR_FREE;
    private static final MethodHandle PATTERN_REGEX;
    private static final MethodHandle PATTERN_LLRE;
    private static final MethodHandle PATTERN_FREE;
    private static final MethodHandle PATTERN_SIZE;
    private static final MethodHandle PATTERN_MATCHES;
    private static final MethodHandle RULES_PARSE;
    private static final MethodHandle RULES_BUILTIN;
    private static final MethodHandle RULES_FREE;
    private static final MethodHandle RULES_LEN;
    private static final MethodHandle RULES_APPLY;
    private static final MethodHandle OWNED_STRING_FREE;

    static {
        NativeLibraryLoader.load();
        SymbolLookup symbols = SymbolLookup.loaderLookup();
        LAST_ERROR = downcall(symbols, "llev_last_error_message", FunctionDescriptor.of(ADDRESS));
        TRANSDUCER_NEW = downcall(symbols, "llev_transducer_new",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));
        TRANSDUCER_SNAPSHOT = downcall(symbols, "llev_transducer_snapshot",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        TRANSDUCER_FREE = downcall(symbols, "llev_transducer_free", FunctionDescriptor.ofVoid(ADDRESS));
        QUERY_UTF8 = queryHandle(symbols, "llev_transducer_query_utf8");
        QUERY_BYTES = queryHandle(symbols, "llev_transducer_query_bytes");
        QUERY_U64 = queryHandle(symbols, "llev_transducer_query_u64");
        QUERY_PATTERN = downcall(symbols, "llev_transducer_query_pattern",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_BYTE, ADDRESS));
        NEXT_BATCH = downcall(symbols, "llev_query_cursor_next_batch",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
        RELEASE_BATCH = downcall(symbols, "llev_query_cursor_release_batch",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
        CURSOR_FREE = downcall(symbols, "llev_query_cursor_free", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        PATTERN_REGEX = textConstructor(symbols, "llev_phonetic_pattern_compile_regex");
        PATTERN_LLRE = textConstructor(symbols, "llev_phonetic_pattern_compile_llre");
        PATTERN_FREE = downcall(symbols, "llev_phonetic_pattern_free", FunctionDescriptor.ofVoid(ADDRESS));
        PATTERN_SIZE = downcall(symbols, "llev_phonetic_pattern_size",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
        PATTERN_MATCHES = downcall(symbols, "llev_phonetic_pattern_matches",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));
        RULES_PARSE = textConstructor(symbols, "llev_phonetic_rules_parse");
        RULES_BUILTIN = downcall(symbols, "llev_phonetic_rules_builtin",
                FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS));
        RULES_FREE = downcall(symbols, "llev_phonetic_rules_free", FunctionDescriptor.ofVoid(ADDRESS));
        RULES_LEN = downcall(symbols, "llev_phonetic_rules_len",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        RULES_APPLY = downcall(symbols, "llev_phonetic_rules_apply",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));
        OWNED_STRING_FREE = downcall(symbols, "llev_owned_string_free", FunctionDescriptor.ofVoid(ADDRESS));
    }

    private Native() {}

    private static MethodHandle queryHandle(SymbolLookup symbols, String name) {
        return downcall(symbols, name,
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS));
    }
    private static MethodHandle textConstructor(SymbolLookup symbols, String name) {
        return downcall(symbols, name, FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
    }
    private static MethodHandle downcall(SymbolLookup symbols, String name, FunctionDescriptor descriptor) {
        return LINKER.downcallHandle(symbols.find(name).orElseThrow(), descriptor);
    }

    static String lastError() {
        try {
            MemorySegment address = (MemorySegment) LAST_ERROR.invoke();
            return address.equals(MemorySegment.NULL)
                    ? "native operation failed"
                    : address.reinterpret(4096).getString(0);
        } catch (Throwable throwable) {
            throw rethrow(throwable);
        }
    }
    static void check(int status) {
        if (status != OK) throw new NativeException(status, lastError());
    }

    static int transducerNew(MemorySegment resource, int algorithm, MemorySegment out) { return call(TRANSDUCER_NEW, resource, algorithm, out); }
    static int transducerSnapshot(MemorySegment transducer, MemorySegment out) { return call(TRANSDUCER_SNAPSHOT, transducer, out); }
    static void transducerFree(MemorySegment value) { run(TRANSDUCER_FREE, value); }
    static int queryUtf8(MemorySegment value, MemorySegment query, long len, long distance, int order, MemorySegment out) { return call(QUERY_UTF8, value, query, len, distance, order, out); }
    static int queryBytes(MemorySegment value, MemorySegment query, long len, long distance, int order, MemorySegment out) { return call(QUERY_BYTES, value, query, len, distance, order, out); }
    static int queryU64(MemorySegment value, MemorySegment query, long len, long distance, int order, MemorySegment out) { return call(QUERY_U64, value, query, len, distance, order, out); }
    static int queryPattern(MemorySegment value, MemorySegment pattern, byte distance, MemorySegment out) { return call(QUERY_PATTERN, value, pattern, distance, out); }
    static int nextBatch(MemorySegment cursor, long size, MemorySegment out) { return call(NEXT_BATCH, cursor, size, out); }
    static int releaseBatch(MemorySegment cursor, long generation) { return call(RELEASE_BATCH, cursor, generation); }
    static int cursorFree(MemorySegment cursor) { return call(CURSOR_FREE, cursor); }
    static int patternRegex(MemorySegment source, long len, MemorySegment out) { return call(PATTERN_REGEX, source, len, out); }
    static int patternLlre(MemorySegment source, long len, MemorySegment out) { return call(PATTERN_LLRE, source, len, out); }
    static void patternFree(MemorySegment value) { run(PATTERN_FREE, value); }
    static int patternSize(MemorySegment value, MemorySegment states, MemorySegment transitions) { return call(PATTERN_SIZE, value, states, transitions); }
    static int patternMatches(MemorySegment value, MemorySegment input, long len, MemorySegment out) { return call(PATTERN_MATCHES, value, input, len, out); }
    static int rulesParse(MemorySegment source, long len, MemorySegment out) { return call(RULES_PARSE, source, len, out); }
    static int rulesBuiltin(int kind, MemorySegment out) { return call(RULES_BUILTIN, kind, out); }
    static void rulesFree(MemorySegment value) { run(RULES_FREE, value); }
    static int rulesLen(MemorySegment value, MemorySegment out) { return call(RULES_LEN, value, out); }
    static int rulesApply(MemorySegment value, MemorySegment input, long len, MemorySegment out) { return call(RULES_APPLY, value, input, len, out); }
    static void ownedStringFree(MemorySegment value) { run(OWNED_STRING_FREE, value); }

    private static int call(MethodHandle handle, Object... arguments) {
        try { return (int) handle.invokeWithArguments(arguments); }
        catch (Throwable throwable) { throw rethrow(throwable); }
    }
    private static void run(MethodHandle handle, Object... arguments) {
        try { handle.invokeWithArguments(arguments); }
        catch (Throwable throwable) { throw rethrow(throwable); }
    }
    private static RuntimeException rethrow(Throwable throwable) {
        return throwable instanceof RuntimeException runtime ? runtime : new RuntimeException(throwable);
    }
}
