package io.vinarytree.interop;

import java.lang.foreign.MemorySegment;

/** A live two-word {@code VtResource} implementing dictionary interface v1. */
public interface DictionaryResource extends AutoCloseable {
    /** Borrow the native resource struct for one synchronous FFM call. */
    MemorySegment resourceSegment();

    /** Release this facade's owned resource retain. */
    @Override
    void close();
}
