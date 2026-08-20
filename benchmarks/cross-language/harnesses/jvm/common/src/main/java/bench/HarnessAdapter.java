package bench;

import java.util.List;

/**
 * The five operations a side (vinary or legacy) must implement for the
 * shared {@link ProtocolRunner}. One adapter instance handles exactly one
 * process's lifecycle; the runner guarantees call order:
 * buildDictionary → createTransducer → pass* → (freeDictionary in construct
 * loops).
 */
public interface HarnessAdapter {
    /** Build the dictionary from the pre-sorted term list (timed by caller). */
    void buildDictionary(List<String> terms, String backend);

    /** Release the current dictionary (construct-mode reps; untimed). */
    void freeDictionary();

    /**
     * Validate the current dictionary outside every timed construction
     * region. Implementations must check exact size and every input term.
     */
    ConstructionProof validateDictionary(List<String> terms);

    /** Create/replace the transducer for the given algorithm name. */
    void createTransducer(String algorithm);

    /**
     * One full pass over the queries: drain every cursor, materialize
     * (term, distance). withChecksum selects the gate accumulators.
     */
    PassResult pass(List<String> queries, int maxDistance, boolean withChecksum);

    /** Static facts for the result JSON. */
    TargetInfo targetInfo(String backend);

    record ConstructionProof(
        long termCount,
        long membershipChecks,
        long checksum) {}

    record TargetInfo(
        String implementation,   // "vinary-tree" | "legacy"
        String backendLabel,     // e.g. "jvm-ffm" | "jvm-legacy-jar"
        String libraryVersion,   // "0.10.0" | "3.0.0"
        String artifactKind,     // "local-build" | "maven"
        String artifactId,
        String dictionaryStructure, // dynamic_dawg | double_array_trie | legacy_sorted_dawg
        String unitDomain,          // unicode_scalar | jvm_string
        Integer batchSize) {}       // 256 | null
}
