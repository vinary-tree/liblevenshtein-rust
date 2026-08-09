using VinaryTree.Libdictenstein;
using VinaryTree.Liblevenshtein;

/// <summary>
/// C9 leak-discipline suite for the .NET facade.
///
/// A >=10,000-cycle create/use/free loop must reach a managed-memory steady
/// state. Native handles are freed deterministically at <see cref="IDisposable.Dispose"/>
/// (SafeHandle-backed), so a native leak is impossible while the loop disposes;
/// these tests assert the managed heap (GC.GetTotalMemory after a forced full
/// collection) does not drift upward across the cycles.
/// </summary>
internal static class LeakTests
{
    private const int Cycles = 10_000;
    private const int Warmup = 2_000;
    // Managed-heap measurement is precise after a forced collection; a per-cycle
    // handle leak would accrue far more than this ceiling over 10k cycles.
    private const long MaxGrowthBytes = 8L * 1024 * 1024;

    public static void Run()
    {
        AssertSteady("transducer iterator", () =>
        {
            using var dawg = new DynamicDawg();
            dawg.Put("cat", 1UL);
            dawg.Put("cot", 2UL);
            dawg.Put("cut", 3UL);
            dawg.Put("scat", null);
            using var transducer = new Transducer(dawg);
            using var query = transducer.Query("cat", (nuint)2);
            foreach (Match _ in query)
            {
                // drain
            }
        });

        AssertSteady("distance", () =>
        {
            _ = Distance.Levenshtein("kitten", "sitting");
            _ = Distance.Levenshtein("kitten", "sitting", (nuint)2);
            _ = Distance.TrueDamerau("ca", "abc");
        });

        AssertSteady("phonetic pattern", () =>
        {
            using var pattern = PhoneticPattern.CompileRegex("c[ao]t");
            _ = pattern.Matches("cat");
        });

        AssertSteady("phonetic rules", () =>
        {
            using var rules = PhoneticRuleSet.Builtin(PhoneticRuleSetKind.EnglishOrthography);
            _ = rules.Apply("phone");
        });

        Console.WriteLine(".NET leak tests passed");
    }

    private static void AssertSteady(string label, Action cycle)
    {
        for (int i = 0; i < Warmup; i++) cycle();
        long baseline = Settled();
        for (int i = 0; i < Cycles; i++) cycle();
        long growth = Settled() - baseline;
        if (growth >= MaxGrowthBytes)
        {
            throw new InvalidOperationException(
                $"{label}: managed heap grew {growth} bytes over {Cycles} cycles");
        }
    }

    private static long Settled()
    {
        for (int i = 0; i < 4; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
        return GC.GetTotalMemory(forceFullCollection: true);
    }
}
