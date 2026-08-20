using System.Collections;
using System.Buffers;
using System.Runtime.InteropServices;
using System.Text;

namespace VinaryTree.Interop;

/// <summary>Shared resource-ABI status values.</summary>
public enum DictionaryInteropStatus : uint
{
    /// <summary>Operation completed.</summary>
    Ok = 0,
    /// <summary>Finite stream exhausted.</summary>
    End = 1,
    /// <summary>Invalid argument.</summary>
    InvalidArgument = 2,
    /// <summary>Required pointer was null.</summary>
    NullPointer = 3,
    /// <summary>Optional interface or operation is unavailable.</summary>
    Unsupported = 4,
    /// <summary>Provider I/O failure.</summary>
    IoError = 5,
    /// <summary>Resource is closed.</summary>
    Closed = 6,
    /// <summary>Configured bound was exceeded.</summary>
    LimitExceeded = 7,
    /// <summary>Malformed or failed provider.</summary>
    ProviderError = 8,
    /// <summary>A borrowed batch lease is already live.</summary>
    BatchInUse = 9,
}

/// <summary>A shared-ABI dictionary collection failure.</summary>
public sealed class DictionaryInteropException : Exception
{
    internal DictionaryInteropException(DictionaryInteropStatus status, string message)
        : base(message) => Status = status;

    /// <summary>Stable shared-ABI status.</summary>
    public DictionaryInteropStatus Status { get; }
}

/// <summary>Native element representation of a dictionary key.</summary>
public enum DictionaryUnitDomain : uint
{
    /// <summary>Arbitrary bytes.</summary>
    Byte = 1,
    /// <summary>Unicode scalar values.</summary>
    UnicodeScalar = 2,
    /// <summary>Unsigned 64-bit tokens.</summary>
    U64 = 3,
}

/// <summary>Value representation attached to present dictionary keys.</summary>
public enum DictionaryValueDomain : uint
{
    /// <summary>Set semantics.</summary>
    Unit = 0,
    /// <summary>Present keys may optionally carry one unsigned-64 value.</summary>
    OptionalU64 = 1,
}

/// <summary>Immutable, value-equal dictionary key preserving its native domain.</summary>
public sealed class DictionaryKey : IEquatable<DictionaryKey>, IComparable<DictionaryKey>
{
    private readonly byte[]? bytes;
    private readonly int[]? scalars;
    private readonly ulong[]? tokens;

    private DictionaryKey(DictionaryUnitDomain domain, byte[]? bytes, int[]? scalars, ulong[]? tokens)
    {
        Domain = domain;
        this.bytes = bytes;
        this.scalars = scalars;
        this.tokens = tokens;
    }

    /// <summary>Native unit domain.</summary>
    public DictionaryUnitDomain Domain { get; }

    /// <summary>Number of native units without copying key storage.</summary>
    public int UnitCount => Domain switch
    {
        DictionaryUnitDomain.Byte => bytes!.Length,
        DictionaryUnitDomain.UnicodeScalar => scalars!.Length,
        DictionaryUnitDomain.U64 => tokens!.Length,
        _ => throw new InvalidOperationException("unknown key domain"),
    };

    /// <summary>Copy an arbitrary byte key.</summary>
    public static DictionaryKey FromBytes(ReadOnlySpan<byte> value) =>
        new(DictionaryUnitDomain.Byte, value.ToArray(), null, null);

    /// <summary>Copy and validate a Unicode-scalar key.</summary>
    public static DictionaryKey FromUnicodeScalars(ReadOnlySpan<int> value)
    {
        int[] copy = value.ToArray();
        foreach (int scalar in copy)
        {
            if (!Rune.IsValid(scalar)) throw new ArgumentException("key contains a non-scalar code point", nameof(value));
        }
        return new(DictionaryUnitDomain.UnicodeScalar, null, copy, null);
    }

    /// <summary>Convert a .NET string to its exact Unicode-scalar sequence.</summary>
    public static DictionaryKey FromString(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        var result = new List<int>(value.Length);
        ReadOnlySpan<char> remaining = value;
        while (!remaining.IsEmpty)
        {
            OperationStatus status = Rune.DecodeFromUtf16(remaining, out Rune rune, out int consumed);
            if (status != OperationStatus.Done) throw new ArgumentException("key contains invalid UTF-16", nameof(value));
            result.Add(rune.Value);
            remaining = remaining[consumed..];
        }
        return FromUnicodeScalars(CollectionsMarshal.AsSpan(result));
    }

    /// <summary>Copy unsigned-64 tokens losslessly.</summary>
    public static DictionaryKey FromU64(ReadOnlySpan<ulong> value) =>
        new(DictionaryUnitDomain.U64, null, null, value.ToArray());

    /// <summary>Return a defensive copy of a byte key.</summary>
    public byte[] ToByteArray()
    {
        Require(DictionaryUnitDomain.Byte);
        return (byte[])bytes!.Clone();
    }

    /// <summary>Return a defensive copy of Unicode scalar values.</summary>
    public int[] ToUnicodeScalars()
    {
        Require(DictionaryUnitDomain.UnicodeScalar);
        return (int[])scalars!.Clone();
    }

    /// <summary>Convert a Unicode-scalar key to a .NET string.</summary>
    public string ToUnicodeString()
    {
        Require(DictionaryUnitDomain.UnicodeScalar);
        var builder = new StringBuilder(scalars!.Length);
        foreach (int scalar in scalars) builder.Append(new Rune(scalar).ToString());
        return builder.ToString();
    }

    /// <summary>Return a defensive copy of unsigned-64 tokens.</summary>
    public ulong[] ToU64Array()
    {
        Require(DictionaryUnitDomain.U64);
        return (ulong[])tokens!.Clone();
    }

    private void Require(DictionaryUnitDomain expected)
    {
        if (Domain != expected) throw new InvalidOperationException($"key domain is {Domain}, not {expected}");
    }

    /// <inheritdoc />
    public bool Equals(DictionaryKey? other)
    {
        if (ReferenceEquals(this, other)) return true;
        if (other is null || Domain != other.Domain) return false;
        return Domain switch
        {
            DictionaryUnitDomain.Byte => bytes!.AsSpan().SequenceEqual(other.bytes),
            DictionaryUnitDomain.UnicodeScalar => scalars!.AsSpan().SequenceEqual(other.scalars),
            DictionaryUnitDomain.U64 => tokens!.AsSpan().SequenceEqual(other.tokens),
            _ => false,
        };
    }

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is DictionaryKey other && Equals(other);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(Domain);
        switch (Domain)
        {
            case DictionaryUnitDomain.Byte:
                foreach (byte value in bytes!) hash.Add(value);
                break;
            case DictionaryUnitDomain.UnicodeScalar:
                foreach (int value in scalars!) hash.Add(value);
                break;
            case DictionaryUnitDomain.U64:
                foreach (ulong value in tokens!) hash.Add(value);
                break;
        }
        return hash.ToHashCode();
    }

    /// <inheritdoc />
    public int CompareTo(DictionaryKey? other)
    {
        if (other is null) return 1;
        int domain = Domain.CompareTo(other.Domain);
        if (domain != 0) return domain;
        return Domain switch
        {
            DictionaryUnitDomain.Byte => Compare(bytes!, other.bytes!),
            DictionaryUnitDomain.UnicodeScalar => Compare(scalars!, other.scalars!),
            DictionaryUnitDomain.U64 => Compare(tokens!, other.tokens!),
            _ => throw new InvalidOperationException("unknown key domain"),
        };
    }

    private static int Compare<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right) where T : IComparable<T>
    {
        int common = Math.Min(left.Length, right.Length);
        for (int index = 0; index < common; index++)
        {
            int compared = left[index].CompareTo(right[index]);
            if (compared != 0) return compared;
        }
        return left.Length.CompareTo(right.Length);
    }

    /// <inheritdoc />
    public override string ToString() => Domain switch
    {
        DictionaryUnitDomain.Byte => $"[{string.Join(", ", bytes!)}]",
        DictionaryUnitDomain.UnicodeScalar => ToUnicodeString(),
        DictionaryUnitDomain.U64 => $"[{string.Join(", ", tokens!)}]",
        _ => string.Empty,
    };
}

/// <summary>One present key and its optional unsigned-64 mapped value.</summary>
public sealed record DictionaryEntry(DictionaryKey Key, ulong? Value);

/// <summary>Opaque process-local identity of an immutable producer revision.</summary>
public readonly record struct DictionarySnapshotIdentity(ulong Producer, ulong Revision);

/// <summary>Metadata captured at the same linearization point as an entries cursor.</summary>
public sealed record DictionaryEntriesMetadata(
    DictionaryUnitDomain UnitDomain,
    DictionaryValueDomain ValueDomain,
    nuint? ExactLength,
    DictionarySnapshotIdentity? SnapshotIdentity);

/// <summary>Hard upper bounds for one native dictionary entries batch.</summary>
public readonly record struct DictionaryBatchLimits(nuint MaxEntries, nuint MaxUnits, nuint MaxValues)
{
    /// <summary>Balanced defaults: 256 descriptors, 65,536 units, and 256 values.</summary>
    public static DictionaryBatchLimits Default { get; } = new(256, 65_536, 256);

    internal void Validate()
    {
        if (MaxEntries == 0) throw new ArgumentOutOfRangeException(nameof(MaxEntries));
    }
}

/// <summary>Natural collection extensions over the optional entries-v1 capability.</summary>
public static class DictionaryCollectionExtensions
{
    /// <summary>Open a disposable streaming enumerator over the current revision.</summary>
    public static DictionaryEntryEnumerator OpenEntryEnumerator(
        this IDictionaryResource resource, DictionaryBatchLimits? limits = null) =>
        DictionaryEntryEnumerator.Open(resource, limits ?? DictionaryBatchLimits.Default);

    /// <summary>Open a one-shot enumerable stream; dispose it when stopping early.</summary>
    public static DictionaryEntryStream StreamEntries(
        this IDictionaryResource resource, DictionaryBatchLimits? limits = null) =>
        new(resource.OpenEntryEnumerator(limits));

    /// <summary>Materialize immutable collection, set, and map views of one revision.</summary>
    public static DictionarySnapshot SnapshotEntries(
        this IDictionaryResource resource, DictionaryBatchLimits? limits = null) =>
        DictionarySnapshot.Materialize(resource, limits ?? DictionaryBatchLimits.Default);
}

/// <summary>A one-shot enumerable owning a native entries cursor.</summary>
public sealed class DictionaryEntryStream : IEnumerable<DictionaryEntry>, IDisposable
{
    private readonly DictionaryEntryEnumerator enumerator;
    private int claimed;

    internal DictionaryEntryStream(DictionaryEntryEnumerator enumerator) => this.enumerator = enumerator;

    /// <summary>Metadata captured with this stream.</summary>
    public DictionaryEntriesMetadata Metadata => enumerator.Metadata;

    /// <inheritdoc />
    public IEnumerator<DictionaryEntry> GetEnumerator()
    {
        if (Interlocked.Exchange(ref claimed, 1) != 0) throw new InvalidOperationException("entry stream is one-shot");
        return enumerator;
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>Cancel unread entries and close the cursor.</summary>
    public void Dispose() => enumerator.Dispose();
}

/// <summary>Disposable streaming enumerator over bounded native entry batches.</summary>
public sealed unsafe class DictionaryEntryEnumerator : IEnumerator<DictionaryEntry>
{
    private const ulong ExactLengthFlag = 1;
    private const ulong SnapshotIdentityFlag = 2;
    private const uint LexicographicOrder = 1;
    private static ReadOnlySpan<byte> EntriesId => "vt.dict.entry.v1"u8;

    private readonly DictionaryBatchLimits limits;
    private readonly Queue<DictionaryEntry> buffered = new();
    private NativeEntriesCursor cursor;
    private DictionaryKey? previous;
    private nuint delivered;
    private bool ended;
    private bool disposed;

    private DictionaryEntryEnumerator(
        NativeEntriesCursor cursor,
        DictionaryEntriesMetadata metadata,
        DictionaryBatchLimits limits)
    {
        this.cursor = cursor;
        Metadata = metadata;
        this.limits = limits;
    }

    /// <summary>Metadata captured with this cursor.</summary>
    public DictionaryEntriesMetadata Metadata { get; }

    /// <inheritdoc />
    public DictionaryEntry Current { get; private set; } = null!;

    object IEnumerator.Current => Current;

    internal static DictionaryEntryEnumerator Open(IDictionaryResource resource, DictionaryBatchLimits limits)
    {
        ArgumentNullException.ThrowIfNull(resource);
        limits.Validate();
        var result = new OpenResult();
        try
        {
            resource.WithResource(native =>
            {
                NativeEntriesVTable* entries = QueryEntries(native);
                fixed (NativeEntriesCursor* opened = &result.Cursor)
                fixed (NativeEntriesInfo* info = &result.Info)
                {
                    DictionaryInteropStatus status = entries->Open(native->Context, opened, info);
                    Check(status, "open dictionary entries cursor");
                }
                result.OwnsCursor = true;
                return 0;
            });
            fixed (NativeEntriesCursor* opened = &result.Cursor)
            fixed (NativeEntriesInfo* info = &result.Info)
            {
                ValidateCursor(opened);
                DictionaryEntriesMetadata metadata = ValidateInfo(info);
                return new(result.Cursor, metadata, limits);
            }
        }
        catch
        {
            if (result.OwnsCursor)
            {
                fixed (NativeEntriesCursor* opened = &result.Cursor)
                {
                    BestEffortClose(opened, cancel: false);
                }
            }
            throw;
        }
    }

    /// <inheritdoc />
    public bool MoveNext()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (buffered.Count == 0 && !ended)
        {
            try { Refill(); }
            catch (Exception failure)
            {
                if (!disposed) FailAndClose(failure);
                throw;
            }
        }
        if (buffered.Count == 0) return false;
        Current = buffered.Dequeue();
        return true;
    }

    private void Refill()
    {
        NativeEntriesBatchLimits nativeLimits = new()
        {
            MaxEntries = limits.MaxEntries,
            MaxUnits = limits.MaxUnits,
            MaxValues = limits.MaxValues,
        };
        NativeEntriesBatch batch = default;
        NativeEntriesVTable* table = CursorVTable();
        DictionaryInteropStatus status;
        fixed (NativeEntriesCursor* cursorPointer = &cursor)
        {
            status = table->NextBatch(cursorPointer, &nativeLimits, &batch);
        }
        if (status == DictionaryInteropStatus.End)
        {
            ValidateEmpty(&batch);
            ended = true;
            VerifyExactLength();
            CloseCore(cancel: false, throwOnError: true);
            return;
        }
        if (status != DictionaryInteropStatus.Ok)
        {
            FailAndClose(new DictionaryInteropException(status, $"entries next_batch returned {status}"));
        }

        Exception? failure = null;
        try { CopyBatch(&batch); }
        catch (Exception exception) { failure = exception; }
        DictionaryInteropStatus release;
        fixed (NativeEntriesCursor* cursorPointer = &cursor)
        {
            release = table->ReleaseBatch(cursorPointer, batch.Generation);
        }
        if (release != DictionaryInteropStatus.Ok)
        {
            var releaseFailure = new DictionaryInteropException(release, "release dictionary entries batch failed");
            failure = failure is null ? releaseFailure : new AggregateException(failure, releaseFailure);
        }
        if (failure is not null) FailAndClose(failure);
    }

    private void CopyBatch(NativeEntriesBatch* batch)
    {
        if (batch->EntryCount == 0 || batch->EntryCount > limits.MaxEntries
            || batch->UnitCount > limits.MaxUnits || batch->ValueCount > limits.MaxValues
            || batch->Generation == 0 || batch->Reserved != 0)
        {
            Malformed("batch count, generation, or reserved fields are invalid");
        }
        RequirePointer(batch->Entries, batch->EntryCount, "entry descriptors");
        RequirePointer(batch->Units, batch->UnitCount, "unit arena");
        RequirePointer(batch->Values, batch->ValueCount, "value arena");
        nuint unitAlignment = Metadata.UnitDomain switch
        {
            DictionaryUnitDomain.Byte => 1,
            DictionaryUnitDomain.UnicodeScalar => 4,
            DictionaryUnitDomain.U64 => 8,
            _ => throw new InvalidOperationException(),
        };
        if ((nuint)batch->Entries % 8 != 0
            || batch->UnitCount != 0 && (nuint)batch->Units % unitAlignment != 0
            || batch->ValueCount != 0 && (nuint)batch->Values % 8 != 0)
        {
            Malformed("entry arena is misaligned");
        }

        nuint nextUnit = 0;
        nuint nextValue = 0;
        for (nuint index = 0; index < batch->EntryCount; index++)
        {
            NativeDictionaryEntry descriptor = batch->Entries[index];
            if (descriptor.Reserved != 0 || descriptor.UnitOffset != nextUnit
                || descriptor.ValueOffset != nextValue
                || descriptor.UnitOffset > batch->UnitCount
                || descriptor.ValueOffset > batch->ValueCount
                || descriptor.UnitLength > batch->UnitCount - descriptor.UnitOffset
                || descriptor.ValueLength > batch->ValueCount - descriptor.ValueOffset)
            {
                Malformed("entry descriptor arenas are not canonical and packed");
            }
            if (Metadata.ValueDomain == DictionaryValueDomain.Unit && descriptor.ValueLength != 0
                || Metadata.ValueDomain == DictionaryValueDomain.OptionalU64 && descriptor.ValueLength > 1)
            {
                Malformed("entry value length does not match its value domain");
            }

            DictionaryKey key = DecodeKey(batch->Units, descriptor.UnitOffset, descriptor.UnitLength);
            if (previous is not null && previous.CompareTo(key) >= 0)
            {
                Malformed("entries are not globally strict lexicographic order");
            }
            previous = key;
            ulong? value = descriptor.ValueLength == 0 ? null : batch->Values[descriptor.ValueOffset];
            buffered.Enqueue(new(key, value));
            nextUnit = checked(descriptor.UnitOffset + descriptor.UnitLength);
            nextValue = checked(descriptor.ValueOffset + descriptor.ValueLength);
        }
        if (nextUnit != batch->UnitCount || nextValue != batch->ValueCount)
        {
            Malformed("entry descriptors do not consume their complete arenas");
        }
        delivered = checked(delivered + batch->EntryCount);
        if (Metadata.ExactLength is nuint exact && delivered > exact)
        {
            Malformed("cursor yielded more than its advertised exact length");
        }
    }

    private DictionaryKey DecodeKey(void* units, nuint offset, nuint length)
    {
        int count = checked((int)length);
        return Metadata.UnitDomain switch
        {
            DictionaryUnitDomain.Byte => DictionaryKey.FromBytes(new ReadOnlySpan<byte>((byte*)units + offset, count)),
            DictionaryUnitDomain.UnicodeScalar => DecodeScalars((uint*)units + offset, count),
            DictionaryUnitDomain.U64 => DictionaryKey.FromU64(new ReadOnlySpan<ulong>((ulong*)units + offset, count)),
            _ => throw new InvalidOperationException("unknown unit domain"),
        };
    }

    private static DictionaryKey DecodeScalars(uint* units, int count)
    {
        int[] scalars = new int[count];
        for (int index = 0; index < count; index++)
        {
            uint scalar = units[index];
            if (scalar > int.MaxValue || !Rune.IsValid((int)scalar)) Malformed("entry contains a non-scalar code point");
            scalars[index] = (int)scalar;
        }
        return DictionaryKey.FromUnicodeScalars(scalars);
    }

    private void VerifyExactLength()
    {
        if (Metadata.ExactLength is nuint exact && delivered != exact)
        {
            Malformed("cursor exhausted before its advertised exact length");
        }
    }

    /// <inheritdoc />
    public void Reset() => throw new NotSupportedException();

    /// <summary>Cancel unread entries and close the native cursor.</summary>
    public void Dispose()
    {
        CloseCore(cancel: !ended, throwOnError: true);
        GC.SuppressFinalize(this);
    }

    private void CloseCore(bool cancel, bool throwOnError)
    {
        if (disposed) return;
        NativeEntriesVTable* table = CursorVTable();
        DictionaryInteropStatus cancelStatus = DictionaryInteropStatus.Ok;
        DictionaryInteropStatus closeStatus;
        fixed (NativeEntriesCursor* cursorPointer = &cursor)
        {
            if (cancel) cancelStatus = table->Cancel(cursorPointer);
            closeStatus = table->Close(cursorPointer);
        }
        if (closeStatus == DictionaryInteropStatus.Ok)
        {
            disposed = true;
            cursor = default;
        }
        if (!throwOnError) return;
        if (cancelStatus != DictionaryInteropStatus.Ok) Check(cancelStatus, "cancel dictionary entries cursor");
        Check(closeStatus, "close dictionary entries cursor");
    }

    private void FailAndClose(Exception failure)
    {
        try { CloseCore(cancel: true, throwOnError: true); }
        catch (Exception cleanup) { throw new AggregateException(failure, cleanup); }
        throw failure;
    }

    private NativeEntriesVTable* CursorVTable()
    {
        fixed (NativeEntriesCursor* cursorPointer = &cursor)
        {
            return ValidateCursor(cursorPointer);
        }
    }

    /// <summary>Best-effort leak containment for an abandoned cursor.</summary>
    ~DictionaryEntryEnumerator()
    {
        if (disposed) return;
        fixed (NativeEntriesCursor* cursorPointer = &cursor)
        {
            BestEffortClose(cursorPointer, cancel: true);
        }
    }

    private sealed class OpenResult
    {
        internal NativeEntriesCursor Cursor;
        internal NativeEntriesInfo Info;
        internal bool OwnsCursor;
    }

    private static NativeEntriesVTable* QueryEntries(NativeResource* resource)
    {
        if (resource == null || resource->IsNull) Malformed("resource descriptor is null");
        NativeResourceVTable* baseTable = (NativeResourceVTable*)resource->VTable;
        if (baseTable->StructSize < (nuint)sizeof(NativeResourceVTable)
            || baseTable->AbiVersion != 1 || baseTable->Reserved != 0
            || baseTable->Retain == null || baseTable->Release == null || baseTable->QueryInterface == null)
        {
            Malformed("resource base vtable is incompatible");
        }
        nint result = 0;
        fixed (byte* identifier = EntriesId)
        {
            DictionaryInteropStatus status = baseTable->QueryInterface(resource->Context, identifier, 1, &result);
            if (status == DictionaryInteropStatus.Unsupported)
            {
                throw new NotSupportedException("resource does not implement vt.dict.entry.v1");
            }
            Check(status, "query dictionary entries interface");
        }
        return ValidateVTable((NativeEntriesVTable*)result);
    }

    private static NativeEntriesVTable* ValidateCursor(NativeEntriesCursor* value)
    {
        if (value == null || value->Context == 0 || value->VTable == 0) Malformed("entries cursor is null or half-null");
        return ValidateVTable((NativeEntriesVTable*)value->VTable);
    }

    private static NativeEntriesVTable* ValidateVTable(NativeEntriesVTable* value)
    {
        if (value == null || value->StructSize < (nuint)sizeof(NativeEntriesVTable)
            || value->InterfaceVersion < 1 || value->Reserved != 0
            || value->Open == null || value->NextBatch == null || value->ReleaseBatch == null
            || value->Reduce == 0 || value->Cancel == null || value->Close == null)
        {
            Malformed("dictionary entries vtable is incompatible");
        }
        return value;
    }

    private static DictionaryEntriesMetadata ValidateInfo(NativeEntriesInfo* info)
    {
        if (info->Order != LexicographicOrder || info->Reserved0 != 0
            || info->Reserved1 != 0 || info->Reserved2 != 0)
        {
            Malformed("entries metadata order/reserved fields are invalid");
        }
        DictionaryUnitDomain unit = info->UnitDomain switch
        {
            1 => DictionaryUnitDomain.Byte,
            2 => DictionaryUnitDomain.UnicodeScalar,
            3 => DictionaryUnitDomain.U64,
            _ => throw new DictionaryInteropException(DictionaryInteropStatus.ProviderError, "unknown unit domain"),
        };
        DictionaryValueDomain value = info->ValueDomain switch
        {
            0 => DictionaryValueDomain.Unit,
            1 => DictionaryValueDomain.OptionalU64,
            _ => throw new DictionaryInteropException(DictionaryInteropStatus.ProviderError, "unsupported value domain"),
        };
        nuint? exact = (info->Flags & ExactLengthFlag) != 0 ? info->ExactLength : null;
        if (exact is null && info->ExactLength != 0) Malformed("absent exact length must be zero");
        DictionarySnapshotIdentity? identity = (info->Flags & SnapshotIdentityFlag) != 0
            ? new(info->IdentityProducer, info->IdentityRevision)
            : null;
        if (identity is null && (info->IdentityProducer != 0 || info->IdentityRevision != 0))
        {
            Malformed("absent snapshot identity must be zero");
        }
        return new(unit, value, exact, identity);
    }

    private static void ValidateEmpty(NativeEntriesBatch* batch)
    {
        if (batch->Entries != null || batch->EntryCount != 0 || batch->Units != null
            || batch->UnitCount != 0 || batch->Values != null || batch->ValueCount != 0
            || batch->Generation != 0 || batch->Reserved != 0)
        {
            Malformed("End did not return the canonical empty batch");
        }
    }

    private static void RequirePointer(void* pointer, nuint count, string name)
    {
        if ((pointer == null) != (count == 0)) Malformed($"{name} pointer/count mismatch");
    }

    private static void Check(DictionaryInteropStatus status, string operation)
    {
        if (status != DictionaryInteropStatus.Ok) throw new DictionaryInteropException(status, $"{operation} returned {status}");
    }

    private static void Malformed(string message) =>
        throw new DictionaryInteropException(DictionaryInteropStatus.ProviderError, message);

    private static void BestEffortClose(NativeEntriesCursor* value, bool cancel)
    {
        try
        {
            if (value == null || value->Context == 0 || value->VTable == 0) return;
            NativeEntriesVTable* table = (NativeEntriesVTable*)value->VTable;
            if (cancel && table->Cancel != null) table->Cancel(value);
            if (table->Close != null) table->Close(value);
        }
        catch { }
    }
}

/// <summary>Materialized immutable collection plus Set/Map views of one revision.</summary>
public sealed class DictionarySnapshot : IReadOnlyCollection<DictionaryEntry>
{
    private readonly DictionaryEntry[] ordered;

    private DictionarySnapshot(DictionaryEntriesMetadata metadata, DictionaryEntry[] ordered)
    {
        Metadata = metadata;
        this.ordered = ordered;
        Keys = new SnapshotSet(ordered);
        Entries = new SnapshotMap(ordered);
    }

    /// <summary>Metadata captured with this snapshot.</summary>
    public DictionaryEntriesMetadata Metadata { get; }

    /// <summary>Immutable value-equal set in native lexicographic iteration order.</summary>
    public IReadOnlySet<DictionaryKey> Keys { get; }

    /// <summary>Immutable map preserving unvalued versus valued present keys.</summary>
    public IReadOnlyDictionary<DictionaryKey, ulong?> Entries { get; }

    /// <inheritdoc />
    public int Count => ordered.Length;

    internal static DictionarySnapshot Materialize(IDictionaryResource resource, DictionaryBatchLimits limits)
    {
        using DictionaryEntryEnumerator cursor = resource.OpenEntryEnumerator(limits);
        if (cursor.Metadata.ExactLength is nuint exact && exact > int.MaxValue)
        {
            throw new DictionaryInteropException(DictionaryInteropStatus.LimitExceeded, "snapshot is too large for a materialized .NET collection");
        }
        if (cursor.Metadata.ExactLength is nuint size)
        {
            var exactEntries = new DictionaryEntry[checked((int)size)];
            int index = 0;
            while (cursor.MoveNext()) exactEntries[index++] = cursor.Current;
            return new(cursor.Metadata, exactEntries);
        }

        var entries = new List<DictionaryEntry>();
        while (cursor.MoveNext()) entries.Add(cursor.Current);
        return new(cursor.Metadata, entries.ToArray());
    }

    /// <inheritdoc />
    public IEnumerator<DictionaryEntry> GetEnumerator() => ((IEnumerable<DictionaryEntry>)ordered).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => ordered.GetEnumerator();

    private sealed class SnapshotSet : IReadOnlySet<DictionaryKey>
    {
        private readonly DictionaryEntry[] ordered;

        internal SnapshotSet(DictionaryEntry[] entries) => ordered = entries;

        public int Count => ordered.Length;
        public bool Contains(DictionaryKey item) => Find(ordered, item) >= 0;

        public bool IsProperSubsetOf(IEnumerable<DictionaryKey> other)
        {
            var candidates = new HashSet<DictionaryKey>(other);
            return Count < candidates.Count && this.All(candidates.Contains);
        }

        public bool IsProperSupersetOf(IEnumerable<DictionaryKey> other)
        {
            var candidates = new HashSet<DictionaryKey>(other);
            return Count > candidates.Count && candidates.All(Contains);
        }

        public bool IsSubsetOf(IEnumerable<DictionaryKey> other)
        {
            var candidates = new HashSet<DictionaryKey>(other);
            return this.All(candidates.Contains);
        }

        public bool IsSupersetOf(IEnumerable<DictionaryKey> other) => other.All(Contains);
        public bool Overlaps(IEnumerable<DictionaryKey> other) => other.Any(Contains);

        public bool SetEquals(IEnumerable<DictionaryKey> other)
        {
            var candidates = new HashSet<DictionaryKey>(other);
            return Count == candidates.Count && candidates.All(Contains);
        }

        public IEnumerator<DictionaryKey> GetEnumerator()
        {
            foreach (DictionaryEntry entry in ordered) yield return entry.Key;
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    private sealed class SnapshotMap : IReadOnlyDictionary<DictionaryKey, ulong?>
    {
        private readonly DictionaryEntry[] ordered;

        internal SnapshotMap(DictionaryEntry[] entries) => ordered = entries;

        public int Count => ordered.Length;
        public IEnumerable<DictionaryKey> Keys => EnumerateKeys();
        public IEnumerable<ulong?> Values => EnumerateValues();

        public ulong? this[DictionaryKey key]
        {
            get
            {
                int index = Find(ordered, key);
                return index >= 0 ? ordered[index].Value : throw new KeyNotFoundException();
            }
        }

        public bool ContainsKey(DictionaryKey key) => Find(ordered, key) >= 0;

        public bool TryGetValue(DictionaryKey key, out ulong? value)
        {
            int index = Find(ordered, key);
            if (index < 0)
            {
                value = null;
                return false;
            }
            value = ordered[index].Value;
            return true;
        }

        public IEnumerator<KeyValuePair<DictionaryKey, ulong?>> GetEnumerator()
        {
            foreach (DictionaryEntry entry in ordered)
            {
                yield return new KeyValuePair<DictionaryKey, ulong?>(entry.Key, entry.Value);
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        private IEnumerable<DictionaryKey> EnumerateKeys()
        {
            foreach (DictionaryEntry entry in ordered) yield return entry.Key;
        }

        private IEnumerable<ulong?> EnumerateValues()
        {
            foreach (DictionaryEntry entry in ordered) yield return entry.Value;
        }
    }

    private static int Find(DictionaryEntry[] ordered, DictionaryKey key)
    {
        ArgumentNullException.ThrowIfNull(key);
        int low = 0;
        int high = ordered.Length - 1;
        while (low <= high)
        {
            int middle = (low + high) >>> 1;
            int comparison = ordered[middle].Key.CompareTo(key);
            if (comparison < 0) low = middle + 1;
            else if (comparison > 0) high = middle - 1;
            else return middle;
        }
        return -1;
    }

    private static void Malformed(string message) =>
        throw new DictionaryInteropException(DictionaryInteropStatus.ProviderError, message);
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeResourceVTable
{
    internal nuint StructSize;
    internal uint AbiVersion;
    internal uint Reserved;
    internal delegate* unmanaged<nint, void> Retain;
    internal delegate* unmanaged<nint, void> Release;
    internal delegate* unmanaged<nint, byte*, uint, nint*, DictionaryInteropStatus> QueryInterface;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDictionaryEntry
{
    internal nuint UnitOffset;
    internal nuint UnitLength;
    internal nuint ValueOffset;
    internal nuint ValueLength;
    internal ulong Reserved;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeEntriesBatchLimits
{
    internal nuint MaxEntries;
    internal nuint MaxUnits;
    internal nuint MaxValues;
    internal ulong Reserved;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeEntriesBatch
{
    internal NativeDictionaryEntry* Entries;
    internal nuint EntryCount;
    internal void* Units;
    internal nuint UnitCount;
    internal ulong* Values;
    internal nuint ValueCount;
    internal ulong Generation;
    internal ulong Reserved;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeEntriesInfo
{
    internal uint UnitDomain;
    internal uint ValueDomain;
    internal uint Order;
    internal uint Reserved0;
    internal ulong Flags;
    internal nuint ExactLength;
    internal ulong IdentityProducer;
    internal ulong IdentityRevision;
    internal ulong Reserved1;
    internal ulong Reserved2;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeEntriesCursor
{
    internal nint Context;
    internal nint VTable;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeEntriesVTable
{
    internal nuint StructSize;
    internal uint InterfaceVersion;
    internal uint Reserved;
    internal delegate* unmanaged<nint, NativeEntriesCursor*, NativeEntriesInfo*, DictionaryInteropStatus> Open;
    internal delegate* unmanaged<NativeEntriesCursor*, NativeEntriesBatchLimits*, NativeEntriesBatch*, DictionaryInteropStatus> NextBatch;
    internal delegate* unmanaged<NativeEntriesCursor*, ulong, DictionaryInteropStatus> ReleaseBatch;
    internal nint Reduce;
    internal delegate* unmanaged<NativeEntriesCursor*, DictionaryInteropStatus> Cancel;
    internal delegate* unmanaged<NativeEntriesCursor*, DictionaryInteropStatus> Close;
}
