package bench;

import io.vinarytree.libdictenstein.Dictionary;
import io.vinarytree.libdictenstein.DoubleArrayTrie;
import io.vinarytree.libdictenstein.DynamicDawg;
import io.vinarytree.liblevenshtein.Algorithm;
import io.vinarytree.liblevenshtein.Match;
import io.vinarytree.liblevenshtein.QueryCursor;
import io.vinarytree.liblevenshtein.Term;
import io.vinarytree.liblevenshtein.Transducer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;

/**
 * vinary-tree side: libdictenstein dictionaries (unicode-scalar domain)
 * handed to the FFM Transducer as DictionaryResources — the same objects and
 * calls a migrating user writes.
 */
public final class VinaryAdapter implements HarnessAdapter {
    private Dictionary dictionary;
    private Transducer transducer;
    private List<Dictionary.TextEntry> preparedEntries;
    private Map<String, OptionalLong> preparedMap;

    @Override
    public void buildDictionary(List<String> terms, String backend) {
        switch (backend) {
            case "dynamic_dawg" -> {
                if (preparedEntries == null) {
                    // Entry preparation (UTF-8 encoding) happens once and is
                    // NOT part of the timed build: the timed operation is the
                    // one-crossing batch insert, matching every other facade.
                    preparedEntries = new ArrayList<>(terms.size());
                    for (String term : terms) {
                        preparedEntries.add(new Dictionary.TextEntry(
                            term.getBytes(StandardCharsets.US_ASCII), OptionalLong.empty()));
                    }
                }
                DynamicDawg dawg = new DynamicDawg();
                long inserted = dawg.putAllBytes(preparedEntries);
                if (inserted != preparedEntries.size()) {
                    throw new IllegalStateException(
                        "batch insert count mismatch: " + inserted + " != " + preparedEntries.size());
                }
                dictionary = dawg;
            }
            case "double_array_trie" -> {
                if (preparedMap == null) {
                    preparedMap = new LinkedHashMap<>(terms.size() * 2);
                    for (String term : terms) {
                        preparedMap.put(term, OptionalLong.empty());
                    }
                }
                dictionary = new DoubleArrayTrie(preparedMap);
            }
            default -> throw new IllegalArgumentException("unknown backend: " + backend);
        }
    }

    @Override
    public void freeDictionary() {
        if (transducer != null) {
            transducer.close();
            transducer = null;
        }
        if (dictionary != null) {
            dictionary.close();
            dictionary = null;
        }
    }

    @Override
    public void createTransducer(String algorithm) {
        if (transducer != null) {
            transducer.close();
        }
        try (Transducer live = new Transducer(dictionary, parseAlgorithm(algorithm))) {
            transducer = live.snapshot();
        }
    }

    public static Algorithm parseAlgorithm(String algorithm) {
        return switch (algorithm) {
            case "standard" -> Algorithm.STANDARD;
            case "transposition" -> Algorithm.TRANSPOSITION;
            case "merge_and_split" -> Algorithm.MERGE_AND_SPLIT;
            case "damerau_levenshtein" -> Algorithm.DAMERAU_LEVENSHTEIN;
            default -> throw new IllegalArgumentException("unknown algorithm: " + algorithm);
        };
    }

    @Override
    public PassResult pass(List<String> queries, int maxDistance, boolean withChecksum) {
        PassAccumulator result = new PassAccumulator(withChecksum);
        for (String query : queries) {
            transducer.forEachMatch(query, maxDistance, result::acceptMaterialized);
        }
        return result.finish();
    }

    /** Causal treatment that reduces borrowed descriptors without decoding terms. */
    public PassResult passBorrowed(
            List<String> queries, int maxDistance, boolean withChecksum) {
        PassAccumulator result = new PassAccumulator(withChecksum);
        for (String query : queries) {
            try (QueryCursor cursor = transducer.query(query, maxDistance)) {
                cursor.forEachBatch(batch -> {
                    for (int index = 0; index < batch.size(); index++) {
                        long distance = batch.distance(index);
                        result.matches++;
                        result.bytes += batch.byteLength(index);
                        result.distanceSum += distance;
                        if (result.withChecksum) {
                            result.checksum += Fnv.entry(batch.bytes(index), distance);
                        }
                    }
                });
            }
        }
        return new PassResult(
            result.matches, result.bytes, result.distanceSum, result.checksum);
    }

    private static final class PassAccumulator {
        private final boolean withChecksum;
        private long matches;
        private long bytes;
        private long distanceSum;
        private long checksum;

        private PassAccumulator(boolean withChecksum) {
            this.withChecksum = withChecksum;
        }

        private void acceptMaterialized(Match match) {
            String term = ((Term.Utf8) match.term()).value();
            matches++;
            bytes += term.length();
            distanceSum += match.distance();
            if (withChecksum) {
                checksum += Fnv.entryAscii(term, match.distance());
            }
        }

        private PassResult finish() {
            return new PassResult(matches, bytes, distanceSum, checksum);
        }
    }

    @Override
    public TargetInfo targetInfo(String backend) {
        return new TargetInfo(
            "vinary-tree",
            "jvm-ffm",
            "0.10.0",
            "local-build",
            "io.vinarytree:liblevenshtein:0.10.0",
            backend,
            "unicode_scalar",
            256);
    }
}
