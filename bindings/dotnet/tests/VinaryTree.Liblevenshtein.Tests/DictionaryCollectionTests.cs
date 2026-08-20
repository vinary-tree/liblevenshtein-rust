using System.Text;
using VinaryTree.Interop;
using VinaryTree.Libdictenstein;
using DictionaryBatchLimits = VinaryTree.Interop.DictionaryBatchLimits;
using DictionaryEntryEnumerator = VinaryTree.Interop.DictionaryEntryEnumerator;
using DictionaryEntryStream = VinaryTree.Interop.DictionaryEntryStream;
using DictionaryKey = VinaryTree.Interop.DictionaryKey;
using DictionarySnapshot = VinaryTree.Interop.DictionarySnapshot;

internal static class DictionaryCollectionTests
{
    private static readonly DictionaryBatchLimits SmallBatch = new(2, 16, 2);

    internal static void Run()
    {
        MaterializedViewsCaptureOneUnicodeRevision();
        ByteAndU64KeysRemainLosslessAndOrdered();
        StreamingSurfacesCancelEarly();
        Console.WriteLine(".NET dictionary collection tests passed");
    }

    private static void MaterializedViewsCaptureOneUnicodeRevision()
    {
        using var dictionary = new DynamicDawg(UnitDomain.UnicodeScalar);
        dictionary.Put(string.Empty);
        dictionary.Put("café", ulong.MaxValue);
        dictionary.Put("cat");

        DictionarySnapshot snapshot = DictionaryCollectionExtensions.SnapshotEntries(dictionary, SmallBatch);
        Check(snapshot.Metadata.UnitDomain == DictionaryUnitDomain.UnicodeScalar, "Unicode metadata");
        Check(snapshot.Metadata.ValueDomain == DictionaryValueDomain.OptionalU64, "value metadata");
        Check(snapshot.Metadata.ExactLength == 3, "exact length");
        Check(snapshot.Metadata.SnapshotIdentity is not null, "snapshot identity");
        Check(snapshot.Keys.Select(key => key.ToUnicodeString()).SequenceEqual(["", "café", "cat"]), "lexicographic keys");
        Check(snapshot.Entries.TryGetValue(DictionaryKey.FromString("café"), out ulong? mapped)
              && mapped == ulong.MaxValue, "mapped ulong max");
        Check(snapshot.Entries.TryGetValue(DictionaryKey.FromString("cat"), out ulong? unvalued)
              && unvalued is null, "present key without value");
        Check(!snapshot.Entries.ContainsKey(DictionaryKey.FromString("dog")), "absent key");
        DictionaryKey[] originalKeys = [
            DictionaryKey.FromString(string.Empty),
            DictionaryKey.FromString("café"),
            DictionaryKey.FromString("cat"),
        ];
        Check(snapshot.Keys.SetEquals(originalKeys), "standard set equality");
        Check(snapshot.Keys.IsProperSubsetOf(originalKeys.Append(DictionaryKey.FromString("dog"))),
              "standard proper-subset relation");
        Check(snapshot.Keys.IsSupersetOf(originalKeys.Take(2)), "standard superset relation");
        Check(snapshot.Keys.Overlaps([DictionaryKey.FromString("cat"), DictionaryKey.FromString("dog")]),
              "standard overlap relation");
        Check(snapshot.Entries.Keys.SequenceEqual(snapshot.Keys), "map and set key order");
        Check(snapshot.Entries.Select(entry => entry.Key).SequenceEqual(snapshot.Keys),
              "map enumeration preserves native order");

        dictionary.Put("dog", 7);
        Check(snapshot.Count == 3 && !snapshot.Keys.Contains(DictionaryKey.FromString("dog")), "snapshot immutability");
        DictionarySnapshot fresh = DictionaryCollectionExtensions.SnapshotEntries(dictionary, SmallBatch);
        Check(fresh.Count == 4, "fresh revision length");
        Check(fresh.Metadata.SnapshotIdentity != snapshot.Metadata.SnapshotIdentity, "identity changes by revision");
    }

    private static void ByteAndU64KeysRemainLosslessAndOrdered()
    {
        DictionaryKey arbitrary = DictionaryKey.FromBytes([0, 0xff, 0x80]);
        Check(arbitrary.UnitCount == 3, "byte unit count");
        Check(arbitrary.ToByteArray().SequenceEqual(new byte[] { 0, 0xff, 0x80 }), "arbitrary bytes");
        Check(arbitrary.Equals(DictionaryKey.FromBytes([0, 0xff, 0x80])), "byte value equality");

        using (var bytes = new DynamicDawg(UnitDomain.Byte))
        {
            bytes.Put(string.Empty);
            bytes.Put("\0", 0);
            bytes.Put("ÿ", ulong.MaxValue);
            DictionarySnapshot snapshot = DictionaryCollectionExtensions.SnapshotEntries(bytes, SmallBatch);
            byte[][] expected = [[], [0], Encoding.UTF8.GetBytes("ÿ")];
            Check(snapshot.Keys.Select(key => key.ToByteArray()).Zip(expected)
                .All(pair => pair.First.SequenceEqual(pair.Second)), "byte seam order and payload");
        }

        using (var tokens = new DynamicDawg(UnitDomain.U64))
        {
            tokens.Put([0UL]);
            tokens.Put([1UL << 63], 1UL << 63);
            tokens.Put([ulong.MaxValue], ulong.MaxValue);
            DictionarySnapshot snapshot = DictionaryCollectionExtensions.SnapshotEntries(tokens, SmallBatch);
            ulong[][] expected = [[0], [1UL << 63], [ulong.MaxValue]];
            Check(snapshot.Keys.Select(key => key.ToU64Array()).Zip(expected)
                .All(pair => pair.First.SequenceEqual(pair.Second)), "unsigned token order");
            Check(snapshot.Keys.All(key => key.UnitCount == 1), "u64 unit count");
            Check(snapshot.Entries[DictionaryKey.FromU64([ulong.MaxValue])] == ulong.MaxValue, "unsigned value bits");
        }
    }

    private static void StreamingSurfacesCancelEarly()
    {
        using var dictionary = new DynamicDawg();
        foreach (string key in new[] { "a", "b", "c", "d" }) dictionary.Put(key);

        using (DictionaryEntryEnumerator cursor =
               DictionaryCollectionExtensions.OpenEntryEnumerator(dictionary, SmallBatch))
        {
            Check(cursor.MoveNext() && cursor.Current.Key.ToUnicodeString() == "a", "stream first entry");
        }

        using (DictionaryEntryStream stream = DictionaryCollectionExtensions.StreamEntries(dictionary, SmallBatch))
        {
            Check(stream.Take(2).Select(entry => entry.Key.ToUnicodeString()).SequenceEqual(["a", "b"]), "stream early take");
        }

        Check(DictionaryCollectionExtensions.SnapshotEntries(dictionary, SmallBatch).Count == 4,
              "cursor cleanup leaves provider usable");
    }

    private static void Check(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException($"dictionary collections: {message}");
    }
}
