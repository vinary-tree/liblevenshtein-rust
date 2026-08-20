package io.vinarytree.interop;

/** Hard upper bounds for one native dictionary entries batch. */
public record DictionaryBatchLimits(long maxEntries, long maxUnits, long maxValues) {
    /** Balanced defaults that avoid one call per entry while bounding each arena. */
    public static final DictionaryBatchLimits DEFAULT =
            new DictionaryBatchLimits(256, 65_536, 256);

    /** Validate positive descriptor capacity and nonnegative arena bounds. */
    public DictionaryBatchLimits {
        if (maxEntries <= 0 || maxUnits < 0 || maxValues < 0) {
            throw new IllegalArgumentException("invalid dictionary batch limits");
        }
    }
}
