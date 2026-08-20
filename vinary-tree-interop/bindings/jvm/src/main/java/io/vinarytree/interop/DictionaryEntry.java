package io.vinarytree.interop;

import java.util.Objects;
import java.util.Optional;

/** One present dictionary key and its optional unsigned-64 mapped value. */
public record DictionaryEntry(DictionaryKey key, Optional<UnsignedLong> value) {
    /** Validate immutable entry components. */
    public DictionaryEntry {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
    }
}
