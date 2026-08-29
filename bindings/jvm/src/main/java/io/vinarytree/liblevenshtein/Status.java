package io.vinarytree.liblevenshtein;

/** Result of a fallible native operation.
 * <p>Generated from bindings/api.json; do not edit numeric values manually.
 */
public enum Status {
    /** The operation completed successfully. */
    OK(0),
    /** A finite cursor reached the end of its stream. */
    END(1),
    /** An argument violated the operation's contract. */
    INVALID_ARGUMENT(2),
    /** Input advertised as text was not valid UTF-8. */
    INVALID_UTF8(3),
    /** A required native pointer was null. */
    NULL_POINTER(4),
    /** A contained Rust panic crossed the failure boundary. */
    PANIC(5),
    /** The requested capability is unavailable in this build. */
    UNSUPPORTED(6),
    /** An input/output operation failed. */
    IO_ERROR(7),
    /** The target resource was already closed. */
    CLOSED(8),
    /** A configured resource or traversal limit was exceeded. */
    LIMIT_EXCEEDED(9),
    /** A foreign dictionary provider reported a failure. */
    PROVIDER_ERROR(10),
    /** A cursor was advanced while its previous batch remained borrowed. */
    BATCH_IN_USE(11),
    /** The query and dictionary use different unit domains. */
    DOMAIN_MISMATCH(12),
    /** A status introduced by a newer compatible ABI revision. */
    UNKNOWN(-1);

    private final int nativeValue;

    Status(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    /**
     * Return this known status's stable numeric ABI value.
     *
     * <p>{@link #UNKNOWN} returns {@code -1}; use
     * {@link NativeException#statusCode()} to recover an unknown status's
     * original forward-compatible value.
     *
     * @return stable ABI value, or {@code -1} for {@link #UNKNOWN}
     */
    public int code() {
        return nativeValue;
    }

    static Status fromNativeValue(int value) {
        return switch (value) {
            case 0 -> OK;
            case 1 -> END;
            case 2 -> INVALID_ARGUMENT;
            case 3 -> INVALID_UTF8;
            case 4 -> NULL_POINTER;
            case 5 -> PANIC;
            case 6 -> UNSUPPORTED;
            case 7 -> IO_ERROR;
            case 8 -> CLOSED;
            case 9 -> LIMIT_EXCEEDED;
            case 10 -> PROVIDER_ERROR;
            case 11 -> BATCH_IN_USE;
            case 12 -> DOMAIN_MISMATCH;
            default -> UNKNOWN;
        };
    }
}
