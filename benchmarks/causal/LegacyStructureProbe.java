import com.github.liblevenshtein.collection.dictionary.DawgNode;
import com.github.liblevenshtein.collection.dictionary.SortedDawg;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/** Identity-based graph census for the published Java 3.0.0 SortedDawg. */
public final class LegacyStructureProbe {
    private LegacyStructureProbe() {}

    public static void main(String[] argv) throws Exception {
        if (argv.length != 1) {
            throw new IllegalArgumentException("usage: LegacyStructureProbe <sorted-dictionary>");
        }
        List<String> terms = Files.readAllLines(Path.of(argv[0]));
        terms.removeIf(String::isEmpty);
        for (int index = 1; index < terms.size(); index++) {
            if (terms.get(index - 1).compareTo(terms.get(index)) >= 0) {
                throw new IllegalArgumentException("dictionary is not strictly sorted at line " + index);
            }
        }

        SortedDawg dictionary = new SortedDawg(terms);
        Set<DawgNode> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        ArrayDeque<DawgNode> pending = new ArrayDeque<>();
        DawgNode root = dictionary.root();
        seen.add(root);
        pending.add(root);
        long edges = 0;
        long finals = 0;
        while (!pending.isEmpty()) {
            DawgNode node = pending.removeFirst();
            if (node.isFinal()) {
                finals++;
            }
            for (DawgNode child : node.edges().values()) {
                edges++;
                if (seen.add(child)) {
                    pending.addLast(child);
                }
            }
        }

        System.out.printf(
            "{\n"
                + "  \"schema\": \"liblevenshtein.legacy-java-structure.v1\",\n"
                + "  \"term_count\": %d,\n"
                + "  \"identity_nodes\": %d,\n"
                + "  \"physical_edges\": %d,\n"
                + "  \"final_nodes\": %d\n"
                + "}\n",
            terms.size(), seen.size(), edges, finals);
    }
}
