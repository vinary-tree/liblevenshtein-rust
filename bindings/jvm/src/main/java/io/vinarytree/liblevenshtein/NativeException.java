package io.vinarytree.liblevenshtein;

/** Failure reported by the stable native ABI. */
public final class NativeException extends RuntimeException {
    private final Status status;
    private final int statusCode;

    NativeException(int status, String message) {
        super(message);
        this.status = Status.fromNativeValue(status);
        this.statusCode = status;
    }

    /**
     * Return the typed status from the versioned C ABI.
     *
     * <p>A status introduced by a newer compatible ABI revision is represented
     * as {@link Status#UNKNOWN}; {@link #statusCode()} retains its raw value.
     *
     * @return typed native status
     */
    public Status status() {
        return status;
    }

    /**
     * Return the exact numeric status supplied by the native ABI.
     *
     * @return raw native status code, including values unknown to this JAR
     */
    public int statusCode() {
        return statusCode;
    }
}
