package bench;

/** CLI entry point for the vinary side's verify/construct/memory/query cells. */
public final class VerifyMain {
    private VerifyMain() {}

    public static void main(String[] args) {
        System.exit(new ProtocolRunner(new VinaryAdapter()).run(args));
    }
}
