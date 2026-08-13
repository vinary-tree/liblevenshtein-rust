package bench;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Input loading with the sortedness invariant (PROTOCOL.md §3). */
public final class Workload {
    private Workload() {}

    public static List<String> readLines(Path path) {
        final byte[] raw;
        try {
            raw = Files.readAllBytes(path);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + path, e);
        }
        int estimate = 1;
        for (byte b : raw) {
            if (b == '\n') {
                estimate++;
            }
        }
        List<String> lines = new ArrayList<>(estimate);
        int start = 0;
        for (int i = 0; i <= raw.length; i++) {
            if (i == raw.length || raw[i] == '\n') {
                if (i > start) {
                    lines.add(new String(raw, start, i - start, StandardCharsets.UTF_8));
                }
                start = i + 1;
            }
        }
        if (lines.isEmpty()) {
            throw new IllegalStateException(path + " contains no lines");
        }
        return lines;
    }

    /** Strict byte-ascending order (compares UTF-8 bytes, not UTF-16). */
    public static void assertStrictlySorted(List<String> lines, Path path) {
        byte[] previous = lines.get(0).getBytes(StandardCharsets.UTF_8);
        for (int i = 1; i < lines.size(); i++) {
            byte[] current = lines.get(i).getBytes(StandardCharsets.UTF_8);
            if (compareBytes(previous, current) >= 0) {
                throw new IllegalStateException(
                    path + " is not strictly byte-sorted at line " + (i + 1)
                        + ": \"" + lines.get(i - 1) + "\" >= \"" + lines.get(i) + "\"");
            }
            previous = current;
        }
    }

    private static int compareBytes(byte[] a, byte[] b) {
        int limit = Math.min(a.length, b.length);
        for (int i = 0; i < limit; i++) {
            int cmp = Integer.compare(a[i] & 0xff, b[i] & 0xff);
            if (cmp != 0) {
                return cmp;
            }
        }
        return Integer.compare(a.length, b.length);
    }
}
