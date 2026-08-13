package bench;

/**
 * The accumulators of one full pass: the O(1) triple (always) plus the FNV
 * checksum (gate passes only; 0 in timed passes).
 */
public record PassResult(long matches, long termBytes, long distanceSum, long checksum) {
    public boolean tripleEquals(PassResult other) {
        return matches == other.matches
            && termBytes == other.termBytes
            && distanceSum == other.distanceSum;
    }
}
