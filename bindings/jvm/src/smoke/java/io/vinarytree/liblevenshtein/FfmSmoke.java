package io.vinarytree.liblevenshtein;

/** Minimal FFM smoke test for project-owned phonetic resources. */
public final class FfmSmoke {
    private FfmSmoke() {}

    /** Run the smoke test. */
    public static void main(String[] arguments) {
        try (PhoneticPattern pattern = PhoneticPattern.compileRegex("c(at|ot)")) {
            if (!pattern.matches("cat") || !pattern.matches("cot")) {
                throw new AssertionError("phonetic pattern mismatch");
            }
        }
    }
}
