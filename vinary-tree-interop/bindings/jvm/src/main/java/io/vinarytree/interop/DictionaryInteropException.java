package io.vinarytree.interop;

/** A malformed provider response or non-success shared-ABI status. */
public final class DictionaryInteropException extends RuntimeException {
    private static final long serialVersionUID = 1L;
    private final int status;

    /** Create a failure carrying the raw {@code VtStatus} value. */
    public DictionaryInteropException(int status, String message) {
        super(message);
        this.status = status;
    }

    /** Raw unsigned status discriminant. */
    public int status() {
        return status;
    }
}
