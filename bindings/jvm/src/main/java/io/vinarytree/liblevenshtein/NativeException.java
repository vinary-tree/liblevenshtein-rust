package io.vinarytree.liblevenshtein;

/** Failure reported by the stable native ABI. */
public final class NativeException extends RuntimeException {
    /** Numeric status from the versioned C ABI. */
    private final int status;

    NativeException(int status, String message) {
        super(message);
        this.status = status;
    }

    /**
     * Return the numeric status from the versioned C ABI.
     *
     * @return native status code
     */
    public int status() {
        return status;
    }
}
