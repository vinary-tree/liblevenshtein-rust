package bench;

/** CLI entry point for the legacy side's verify/construct/memory/query cells. */
public final class LegacyVerifyMain {
    private LegacyVerifyMain() {}

    public static void main(String[] args) {
        System.exit(new ProtocolRunner(new LegacyAdapter()).run(args));
    }
}
