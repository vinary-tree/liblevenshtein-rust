using System.Text;
using VinaryTree.Libdictenstein;
using VinaryTree.Liblevenshtein;

if (Distance.Levenshtein("kitten", "sitting") != 3) throw new Exception("distance mismatch");
if (Distance.Damerau("ab", "ba") != 1) throw new Exception("Damerau mismatch");
if (Distance.TrueDamerau("ca", "abc") != 2) throw new Exception("true Damerau mismatch");

// LLEV-B13: all three threshold overloads are bound and share the native
// exceeded-bound sentinel (nuint.MaxValue - 1), not threshold + 1.
nuint exceeded = nuint.MaxValue - 1;
if (Distance.Levenshtein("kitten", "sitting", 3) != 3) throw new Exception("Levenshtein threshold within-bound mismatch");
if (Distance.Levenshtein("kitten", "sitting", 2) != exceeded) throw new Exception("Levenshtein threshold sentinel mismatch");
if (Distance.Damerau("ab", "ba", 1) != 1) throw new Exception("Damerau threshold within-bound mismatch");
if (Distance.Damerau("ab", "ba", 0) != exceeded) throw new Exception("Damerau threshold sentinel mismatch");
// "ca" -> "abc": OSA is 3 but unrestricted Damerau-Levenshtein is 2, so a
// threshold of 2 separates the true-Damerau variant from OSA.
if (Distance.Damerau("ca", "abc", 2) != exceeded) throw new Exception("OSA threshold must exceed 2 for ca->abc");
if (Distance.TrueDamerau("ca", "abc", 2) != 2) throw new Exception("true Damerau threshold within-bound mismatch");
if (Distance.TrueDamerau("ca", "abc", 1) != exceeded) throw new Exception("true Damerau threshold sentinel mismatch");
using PhoneticPattern pattern = PhoneticPattern.CompileRegex("cat");
if (!pattern.Matches("cat") || pattern.Matches("cot")) throw new Exception("pattern mismatch");

// C6: distances count Unicode scalar values, not UTF-16 code units or bytes, so
// a multi-byte character is a single edit. The facade passes UTF-8 to the ABI,
// which guards against a naive string.Length (UTF-16) regression.
if (Distance.Levenshtein("café", "cafe") != 1) throw new Exception("café/cafe scalar distance"); // é: 2 UTF-8 bytes, 1 scalar
if (Distance.Levenshtein("🦀", "x") != 1) throw new Exception("astral scalar distance"); // 🦀: 1 scalar, 2 UTF-16 units
if (Distance.Levenshtein("é", "e") != 1) throw new Exception("combining scalar distance");
if (Distance.Levenshtein("café", "cafe", 1) != 1) throw new Exception("Unicode within-bound threshold");
if (Distance.Levenshtein("café", "cafe", 0) != exceeded) throw new Exception("Unicode exceeded-bound sentinel");

// C2/C3: Size is a live accessor; Dispose is idempotent; a disposed pattern
// rejects further use with ObjectDisposedException rather than crashing.
PhoneticPattern lifecycle = PhoneticPattern.CompileRegex("c[ao]t");
(nuint states, nuint transitions) = lifecycle.Size();
if (states == 0 || transitions == 0) throw new Exception("pattern size must be positive");
lifecycle.Dispose();
lifecycle.Dispose(); // idempotent
try { lifecycle.Matches("cat"); throw new Exception("closed pattern must reject Matches"); }
catch (ObjectDisposedException) { }

// C6/C2/C3: rule-set parse/count/apply, idempotent Dispose, closed-handle guard.
PhoneticRuleSet rules = PhoneticRuleSet.Parse("ph -> f\ngh ->\n");
if (rules.Count != 2) throw new Exception($"rule count={rules.Count}");
if (rules.Apply("phgh") != "f") throw new Exception("rule application");
rules.Dispose();
rules.Dispose(); // idempotent
try { _ = rules.Count; throw new Exception("closed rule set must reject Count"); }
catch (ObjectDisposedException) { }

foreach (PhoneticRuleSetKind kind in Enum.GetValues<PhoneticRuleSetKind>())
{
    using PhoneticRuleSet builtin = PhoneticRuleSet.Builtin(kind);
    if (builtin.Count == 0) throw new Exception($"built-in rule count must be positive: {kind}");
    if (builtin.Apply("phone").Length == 0) throw new Exception($"built-in rule application: {kind}");
}

VerifySelectorsOrdersDomainsAndProductQuery();
VerifyLlreAndTypedFailure();
VerifyGeneratedEnumsAndLifecycle();

Console.WriteLine(".NET binding conformance passed");

DictionaryCollectionTests.Run();

PropertyTests.Run();

LeakTests.Run();

static void VerifySelectorsOrdersDomainsAndProductQuery()
{
    using var dictionary = new DynamicDawg();
    dictionary.Put("ab", 1);
    dictionary.Put("c", 2);
    dictionary.Put("abc", 3);
    dictionary.Put("bat", 4);
    dictionary.Put("cat", 5);
    dictionary.Put("cats", 6);

    List<Match> standard = QueryText(dictionary, Algorithm.Standard, "ba", 1, QueryOrder.Traversal);
    Require(!ContainsText(standard, "ab", 1), "standard unexpectedly accepted a transposition");

    List<Match> transposition = QueryText(dictionary, Algorithm.Transposition, "ba", 1, QueryOrder.Traversal);
    Require(ContainsText(transposition, "ab", 1), "transposition did not accept an adjacent swap");

    List<Match> mergeAndSplit = QueryText(dictionary, Algorithm.MergeAndSplit, "ab", 1, QueryOrder.Traversal);
    Require(ContainsText(mergeAndSplit, "c", 1), "merge-and-split did not accept its distinguishing edit");

    List<Match> damerau = QueryText(dictionary, Algorithm.DamerauLevenshtein, "ca", 2, QueryOrder.Traversal);
    Require(ContainsText(damerau, "abc", 2), "unrestricted Damerau-Levenshtein distinguishing edit");

    List<Match> traversal = QueryText(dictionary, Algorithm.Standard, "cat", 1, QueryOrder.Traversal);
    List<Match> ranked = QueryText(dictionary, Algorithm.Standard, "cat", 1, QueryOrder.DistanceThenTerm);
    Require(Describe(traversal).SequenceEqual(["bat:1", "cat:0", "cats:1"]),
            $"traversal order changed: {string.Join(",", Describe(traversal))}");
    Require(Describe(ranked).SequenceEqual(["cat:0", "bat:1", "cats:1"]),
            $"distance-then-term order changed: {string.Join(",", Describe(ranked))}");

    using (PhoneticPattern productPattern = PhoneticPattern.CompileRegex("c[ao]t"))
    using (var transducer = new Transducer(dictionary))
    using (Query query = transducer.Query(productPattern, 0))
    {
        List<Match> matches = query.ToList();
        Require(matches.Count == 1 && (string)matches[0].Term == "cat",
                "dictionary-pattern product query changed");
    }

    using (var byteDictionary = new DynamicDawg(UnitDomain.Byte))
    {
        byteDictionary.Put("bat", 7);
        using var transducer = new Transducer(byteDictionary);
        using Query query = transducer.Query(Encoding.UTF8.GetBytes("cat"), 1);
        Match match = query.Single(result => result.Term is byte[] bytes
                                             && bytes.SequenceEqual(Encoding.UTF8.GetBytes("bat")));
        Require(match.Distance == 1 && match.Id == 7, "raw-byte query lost its term, distance, or value");
    }

    using (var tokenDictionary = new DynamicDawg(UnitDomain.U64))
    {
        tokenDictionary.Put([10UL, 20UL], ulong.MaxValue);
        using var transducer = new Transducer(tokenDictionary);
        using Query query = transducer.Query([10UL, 21UL], 1);
        Match match = query.Single(result => result.Term is ulong[] tokens
                                             && tokens.SequenceEqual([10UL, 20UL]));
        Require(match.Distance == 1 && match.Id == ulong.MaxValue,
                "u64 query lost its term, distance, or value");
    }
}

static void VerifyLlreAndTypedFailure()
{
    using (PhoneticPattern llre = PhoneticPattern.CompileLlre("@name \"Greeting\"\n^hello$"))
    {
        Require(llre.Matches("hello"), "LLRE rejected hello");
        Require(!llre.Matches("world"), "LLRE accepted world");
    }

    try
    {
        using PhoneticPattern _ = PhoneticPattern.CompileRegex("(");
        throw new Exception("invalid regex unexpectedly compiled");
    }
    catch (LiblevenshteinException failure)
    {
        Require(failure.Status == Status.InvalidArgument,
                $"invalid regex returned typed status {failure.Status}");
        Require(failure.StatusCode == (uint)Status.InvalidArgument,
                "typed and raw native statuses disagree");
        Require(!string.IsNullOrWhiteSpace(failure.Message),
                "native failure omitted its copied diagnostic");
    }
}

static void VerifyGeneratedEnumsAndLifecycle()
{
    Require((uint)Status.DomainMismatch == 12, "generated status constants drifted");
    Require((uint)Algorithm.DamerauLevenshtein == 3, "generated algorithm constants drifted");
    Require((uint)QueryOrder.DistanceThenTerm == 1, "generated query-order constants drifted");
    Require((uint)PhoneticRuleSetKind.EnglishPhonetic == 1, "generated rule-set constants drifted");
    Status futureStatus = (Status)uint.MaxValue;
    Require((uint)futureStatus == uint.MaxValue, "unknown compatible status did not remain lossless");

    using var dictionary = new DynamicDawg();
    dictionary.Put("cat");
    var transducer = new Transducer(dictionary, Algorithm.Standard);
    Query cursor = transducer.Query("cat", 0, QueryOrder.Traversal);
    cursor.Dispose();
    cursor.Dispose();
    try
    {
        using IEnumerator<Match> enumerator = cursor.GetEnumerator();
        _ = enumerator.MoveNext();
        throw new Exception("closed query accepted iteration");
    }
    catch (ObjectDisposedException) { }

    transducer.Dispose();
    transducer.Dispose();
    try
    {
        using Query _ = transducer.Query("cat", 0, QueryOrder.Traversal);
        throw new Exception("closed transducer accepted a query");
    }
    catch (ObjectDisposedException) { }
}

static List<Match> QueryText(
    DynamicDawg dictionary,
    Algorithm algorithm,
    string input,
    nuint maximumDistance,
    QueryOrder order)
{
    using var transducer = new Transducer(dictionary, algorithm);
    using Query query = transducer.Query(input, maximumDistance, order);
    return query.ToList();
}

static bool ContainsText(IEnumerable<Match> matches, string term, nuint distance) =>
    matches.Any(match => match.Term is string text && text == term && match.Distance == distance);

static IEnumerable<string> Describe(IEnumerable<Match> matches) =>
    matches.Select(match => $"{(string)match.Term}:{match.Distance}");

static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}
