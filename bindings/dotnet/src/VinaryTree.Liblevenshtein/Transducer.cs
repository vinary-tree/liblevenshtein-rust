using System.Collections;
using System.Runtime.InteropServices;
using System.Text;
using VinaryTree.Interop;

namespace VinaryTree.Liblevenshtein;

/// <summary>One owned result materialized from a leased native batch.</summary>
public sealed record Match(object Term, nuint Distance, ulong? Id);

/// <summary>A retained automaton configuration over a modular dictionary resource.</summary>
public sealed class Transducer : IDisposable
{
    private readonly TransducerHandle handle;
    internal TransducerHandle Handle => handle;

    /// <summary>Retain a dictionary in O(1).</summary>
    public unsafe Transducer(IDictionaryResource dictionary, Algorithm algorithm = Algorithm.Standard)
    {
        ArgumentNullException.ThrowIfNull(dictionary);
        nint created = 0;
        dictionary.WithResource(resource =>
        {
            NativeStatus.Check(NativeMethods.TransducerNew(resource, (uint)algorithm, out created));
            return 0;
        });
        handle = created != 0 ? new TransducerHandle(created)
                              : throw new InvalidOperationException("native transducer was not returned");
    }

    /// <summary>Start a lazy Unicode query and capture the producer revision now.</summary>
    public unsafe Query Query(string text, nuint maximumDistance, QueryOrder order = QueryOrder.Traversal)
    {
        byte[] input = Encoding.UTF8.GetBytes(text);
        fixed (byte* data = input)
        {
            NativeStatus.Check(NativeMethods.QueryUtf8(handle, data, (nuint)input.Length, maximumDistance, (uint)order, out nint cursor));
            return new Query(new CursorHandle(cursor));
        }
    }

    /// <summary>Start a lazy raw-byte query.</summary>
    public unsafe Query Query(ReadOnlySpan<byte> bytes, nuint maximumDistance)
    {
        fixed (byte* data = bytes)
        {
            NativeStatus.Check(NativeMethods.QueryBytes(handle, data, (nuint)bytes.Length, maximumDistance, 0, out nint cursor));
            return new Query(new CursorHandle(cursor));
        }
    }

    /// <summary>Start a lazy u64-token query.</summary>
    public unsafe Query Query(ReadOnlySpan<ulong> tokens, nuint maximumDistance)
    {
        fixed (ulong* data = tokens)
        {
            NativeStatus.Check(NativeMethods.QueryU64(handle, data, (nuint)tokens.Length, maximumDistance, 0, out nint cursor));
            return new Query(new CursorHandle(cursor));
        }
    }

    /// <summary>Start a dictionary × phonetic-language product query.</summary>
    public Query Query(PhoneticPattern pattern, byte maximumDistance)
    {
        NativeStatus.Check(NativeMethods.QueryPattern(handle, pattern.Handle, maximumDistance, out nint cursor));
        return new Query(new CursorHandle(cursor));
    }

    /// <inheritdoc />
    public void Dispose() => handle.Dispose();
}

/// <summary>Immutable TinyLFU/SIEVE counters and current bounded residency.</summary>
public sealed record QueryCacheStats(
    ulong Requests,
    ulong Hits,
    ulong Misses,
    ulong Admissions,
    ulong Rejections,
    ulong Evictions,
    nuint ResidentEntries,
    nuint ResidentWeight);

/// <summary>
/// Exclusive synchronization-free cache for complete repeated query results.
/// Limits apply independently to traversal and distance-then-term order.
/// Create one cache per worker for parallel workloads.
/// </summary>
public sealed class QueryCache : IDisposable
{
    private readonly QueryCacheHandle handle;

    /// <summary>Retain a transducer and configure hard per-order bounds.</summary>
    public QueryCache(
        Transducer transducer,
        nuint maximumEntries = 1024,
        nuint maximumWeight = 64 * 1024 * 1024)
    {
        ArgumentNullException.ThrowIfNull(transducer);
        NativeStatus.Check(NativeMethods.QueryCacheNew(
            transducer.Handle, maximumEntries, maximumWeight, out nint created));
        handle = created != 0 ? new QueryCacheHandle(created)
                              : throw new InvalidOperationException("native query cache was not returned");
    }

    /// <summary>Copy aggregate policy counters and current residency.</summary>
    public QueryCacheStats Stats
    {
        get
        {
            NativeStatus.Check(NativeMethods.QueryCacheStats(handle, out var stats));
            return new QueryCacheStats(
                stats.Requests, stats.Hits, stats.Misses, stats.Admissions,
                stats.Rejections, stats.Evictions, stats.ResidentEntries,
                stats.ResidentWeight);
        }
    }

    /// <summary>Drop resident results while preserving policy counters.</summary>
    public void Clear() => NativeStatus.Check(NativeMethods.QueryCacheClear(handle));

    /// <summary>Reset counters while preserving residency and frequency state.</summary>
    public void ResetStats() => NativeStatus.Check(NativeMethods.QueryCacheResetStats(handle));

    /// <summary>Query Unicode text through the bounded complete-result cache.</summary>
    public unsafe Query Query(
        string text,
        nuint maximumDistance,
        QueryOrder order = QueryOrder.Traversal)
    {
        byte[] input = Encoding.UTF8.GetBytes(text);
        fixed (byte* data = input)
        {
            NativeStatus.Check(NativeMethods.QueryCacheUtf8(
                handle, data, (nuint)input.Length, maximumDistance, (uint)order,
                out nint cursor));
            return new Query(new CursorHandle(cursor));
        }
    }

    /// <summary>Query exact bytes through the bounded complete-result cache.</summary>
    public unsafe Query Query(
        ReadOnlySpan<byte> bytes,
        nuint maximumDistance,
        QueryOrder order = QueryOrder.Traversal)
    {
        fixed (byte* data = bytes)
        {
            NativeStatus.Check(NativeMethods.QueryCacheBytes(
                handle, data, (nuint)bytes.Length, maximumDistance, (uint)order,
                out nint cursor));
            return new Query(new CursorHandle(cursor));
        }
    }

    /// <summary>Query exact u64 tokens through the bounded complete-result cache.</summary>
    public unsafe Query Query(
        ReadOnlySpan<ulong> tokens,
        nuint maximumDistance,
        QueryOrder order = QueryOrder.Traversal)
    {
        fixed (ulong* data = tokens)
        {
            NativeStatus.Check(NativeMethods.QueryCacheU64(
                handle, data, (nuint)tokens.Length, maximumDistance, (uint)order,
                out nint cursor));
            return new Query(new CursorHandle(cursor));
        }
    }

    /// <inheritdoc />
    public void Dispose() => handle.Dispose();
}

/// <summary>A one-shot enumerable retaining its query-start dictionary snapshot.</summary>
public sealed class Query : IEnumerable<Match>, IDisposable
{
    private readonly CursorHandle cursor;
    private int claimed;
    internal Query(CursorHandle cursor) => this.cursor = cursor;

    /// <inheritdoc />
    public IEnumerator<Match> GetEnumerator()
    {
        if (Interlocked.Exchange(ref claimed, 1) != 0) throw new InvalidOperationException("query is one-shot");
        return new Enumerator(cursor);
    }
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    /// <inheritdoc />
    public void Dispose() => cursor.Dispose();

    private sealed unsafe class Enumerator : IEnumerator<Match>
    {
        private readonly CursorHandle cursor;
        private NativeBatch batch;
        private nuint index;
        private bool leased;
        private bool ended;

        internal Enumerator(CursorHandle cursor) => this.cursor = cursor;
        public Match Current { get; private set; } = null!;
        object IEnumerator.Current => Current;

        public bool MoveNext()
        {
            if (ended) return false;
            if (!leased || index == batch.Length)
            {
                Release();
                Status status = NativeMethods.NextBatch(cursor, 256, out batch);
                if (status == Status.End) { ended = true; cursor.Dispose(); return false; }
                NativeStatus.Check(status);
                leased = true;
                index = 0;
            }
            ref NativeMatch item = ref batch.Matches[index++];
            object term = item.UnitDomain switch
            {
                1 => new ReadOnlySpan<byte>(item.TermData, checked((int)item.ByteLength)).ToArray(),
                2 => Encoding.UTF8.GetString(new ReadOnlySpan<byte>(item.TermData, checked((int)item.ByteLength))),
                3 => new ReadOnlySpan<ulong>(item.TermData, checked((int)item.TermLength)).ToArray(),
                _ => throw new InvalidDataException($"unknown term domain {item.UnitDomain}"),
            };
            Current = new Match(term, item.Distance, item.HasId != 0 ? item.Id : null);
            return true;
        }

        private void Release()
        {
            if (!leased) return;
            NativeStatus.Check(NativeMethods.ReleaseBatch(cursor, batch.Generation));
            leased = false;
        }
        public void Reset() => throw new NotSupportedException();
        public void Dispose() { Release(); cursor.Dispose(); ended = true; }
    }
}
