using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace VinaryTree.Liblevenshtein;

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeMatch
{
    internal void* TermData;
    internal nuint TermLength;
    internal nuint ByteLength;
    internal nuint Distance;
    internal ulong Id;
    internal uint UnitDomain;
    internal byte HasId;
    private fixed byte Reserved[3];
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeBatch
{
    internal NativeMatch* Matches;
    internal nuint Length;
    internal ulong Generation;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeOwnedString
{
    internal byte* Data;
    internal nuint Length;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeQueryCacheStats
{
    internal ulong Requests;
    internal ulong Hits;
    internal ulong Misses;
    internal ulong Admissions;
    internal ulong Rejections;
    internal ulong Evictions;
    internal nuint ResidentEntries;
    internal nuint ResidentWeight;
}

internal sealed class TransducerHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    internal TransducerHandle(nint value) : base(true) => SetHandle(value);
    protected override bool ReleaseHandle() { NativeMethods.TransducerFree(handle); return true; }
}

internal sealed class CursorHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    internal CursorHandle(nint value) : base(true) => SetHandle(value);
    protected override bool ReleaseHandle() => NativeMethods.CursorFree(handle) == Status.Ok;
}

internal sealed class QueryCacheHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    internal QueryCacheHandle(nint value) : base(true) => SetHandle(value);
    protected override bool ReleaseHandle() { NativeMethods.QueryCacheFree(handle); return true; }
}

internal sealed class PatternHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    internal PatternHandle(nint value) : base(true) => SetHandle(value);
    protected override bool ReleaseHandle() { NativeMethods.PatternFree(handle); return true; }
}

internal sealed class RuleSetHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    internal RuleSetHandle(nint value) : base(true) => SetHandle(value);
    protected override bool ReleaseHandle() { NativeMethods.RulesFree(handle); return true; }
}

internal static unsafe partial class NativeMethods
{
    private const string Library = "liblevenshtein";

    [LibraryImport(Library, EntryPoint = "llev_last_error_message")]
    internal static partial nint LastErrorMessage();
    [LibraryImport(Library, EntryPoint = "llev_distance")]
    internal static partial nuint Distance(byte* source, nuint sourceLength, byte* target, nuint targetLength);
    [LibraryImport(Library, EntryPoint = "llev_distance_threshold")]
    internal static partial nuint DistanceThreshold(byte* source, nuint sourceLength, byte* target, nuint targetLength, nuint threshold);
    [LibraryImport(Library, EntryPoint = "llev_damerau_distance")]
    internal static partial nuint DamerauDistance(byte* source, nuint sourceLength, byte* target, nuint targetLength);
    [LibraryImport(Library, EntryPoint = "llev_damerau_distance_threshold")]
    internal static partial nuint DamerauDistanceThreshold(byte* source, nuint sourceLength, byte* target, nuint targetLength, nuint threshold);
    [LibraryImport(Library, EntryPoint = "llev_true_damerau_distance")]
    internal static partial nuint TrueDamerauDistance(byte* source, nuint sourceLength, byte* target, nuint targetLength);
    [LibraryImport(Library, EntryPoint = "llev_true_damerau_distance_threshold")]
    internal static partial nuint TrueDamerauDistanceThreshold(byte* source, nuint sourceLength, byte* target, nuint targetLength, nuint threshold);
    [LibraryImport(Library, EntryPoint = "llev_transducer_new")]
    internal static partial Status TransducerNew(VinaryTree.Interop.NativeResource* resource, uint algorithm, out nint handle);
    [LibraryImport(Library, EntryPoint = "llev_transducer_free")]
    internal static partial void TransducerFree(nint handle);
    [LibraryImport(Library, EntryPoint = "llev_transducer_query_utf8")]
    internal static partial Status QueryUtf8(TransducerHandle transducer, byte* query, nuint length, nuint maximumDistance, uint order, out nint cursor);
    [LibraryImport(Library, EntryPoint = "llev_transducer_query_bytes")]
    internal static partial Status QueryBytes(TransducerHandle transducer, byte* query, nuint length, nuint maximumDistance, uint order, out nint cursor);
    [LibraryImport(Library, EntryPoint = "llev_transducer_query_u64")]
    internal static partial Status QueryU64(TransducerHandle transducer, ulong* query, nuint length, nuint maximumDistance, uint order, out nint cursor);
    [LibraryImport(Library, EntryPoint = "llev_query_cache_new")]
    internal static partial Status QueryCacheNew(TransducerHandle transducer, nuint maximumEntries, nuint maximumWeight, out nint cache);
    [LibraryImport(Library, EntryPoint = "llev_query_cache_clear")]
    internal static partial Status QueryCacheClear(QueryCacheHandle cache);
    [LibraryImport(Library, EntryPoint = "llev_query_cache_reset_stats")]
    internal static partial Status QueryCacheResetStats(QueryCacheHandle cache);
    [LibraryImport(Library, EntryPoint = "llev_query_cache_stats")]
    internal static partial Status QueryCacheStats(QueryCacheHandle cache, out NativeQueryCacheStats stats);
    [LibraryImport(Library, EntryPoint = "llev_query_cache_free")]
    internal static partial void QueryCacheFree(nint cache);
    [LibraryImport(Library, EntryPoint = "llev_query_cache_query_utf8")]
    internal static partial Status QueryCacheUtf8(QueryCacheHandle cache, byte* query, nuint length, nuint maximumDistance, uint order, out nint cursor);
    [LibraryImport(Library, EntryPoint = "llev_query_cache_query_bytes")]
    internal static partial Status QueryCacheBytes(QueryCacheHandle cache, byte* query, nuint length, nuint maximumDistance, uint order, out nint cursor);
    [LibraryImport(Library, EntryPoint = "llev_query_cache_query_u64")]
    internal static partial Status QueryCacheU64(QueryCacheHandle cache, ulong* query, nuint length, nuint maximumDistance, uint order, out nint cursor);
    [LibraryImport(Library, EntryPoint = "llev_query_cursor_next_batch")]
    internal static partial Status NextBatch(CursorHandle cursor, nuint maximum, out NativeBatch batch);
    [LibraryImport(Library, EntryPoint = "llev_query_cursor_release_batch")]
    internal static partial Status ReleaseBatch(CursorHandle cursor, ulong generation);
    [LibraryImport(Library, EntryPoint = "llev_query_cursor_free")]
    internal static partial Status CursorFree(nint cursor);
    [LibraryImport(Library, EntryPoint = "llev_phonetic_pattern_compile_regex")]
    internal static partial Status PatternRegex(byte* source, nuint length, out nint pattern);
    [LibraryImport(Library, EntryPoint = "llev_phonetic_pattern_compile_llre")]
    internal static partial Status PatternLlre(byte* source, nuint length, out nint pattern);
    [LibraryImport(Library, EntryPoint = "llev_phonetic_pattern_free")]
    internal static partial void PatternFree(nint pattern);
    [LibraryImport(Library, EntryPoint = "llev_phonetic_pattern_matches")]
    internal static partial Status PatternMatches(PatternHandle pattern, byte* text, nuint length, out byte matches);
    [LibraryImport(Library, EntryPoint = "llev_phonetic_pattern_size")]
    internal static partial Status PatternSize(PatternHandle pattern, out nuint states, out nuint transitions);
    [LibraryImport(Library, EntryPoint = "llev_transducer_query_pattern")]
    internal static partial Status QueryPattern(TransducerHandle transducer, PatternHandle pattern, byte maximumDistance, out nint cursor);
    [LibraryImport(Library, EntryPoint = "llev_phonetic_rules_parse")]
    internal static partial Status RulesParse(byte* source, nuint length, out nint rules);
    [LibraryImport(Library, EntryPoint = "llev_phonetic_rules_builtin")]
    internal static partial Status RulesBuiltin(uint kind, out nint rules);
    [LibraryImport(Library, EntryPoint = "llev_phonetic_rules_free")]
    internal static partial void RulesFree(nint rules);
    [LibraryImport(Library, EntryPoint = "llev_phonetic_rules_len")]
    internal static partial Status RulesLength(RuleSetHandle rules, out nuint length);
    [LibraryImport(Library, EntryPoint = "llev_phonetic_rules_apply")]
    internal static partial Status RulesApply(RuleSetHandle rules, byte* text, nuint length, out NativeOwnedString output);
    [LibraryImport(Library, EntryPoint = "llev_owned_string_free")]
    internal static partial void OwnedStringFree(ref NativeOwnedString value);
}

internal static class NativeStatus
{
    internal static void Check(Status status)
    {
        if (status == Status.Ok) return;
        string message = Marshal.PtrToStringUTF8(NativeMethods.LastErrorMessage()) ?? status.ToString();
        throw new LiblevenshteinException(status, message);
    }
}

/// <summary>A native binding failure with typed and lossless status information.</summary>
public sealed class LiblevenshteinException : Exception
{
    internal LiblevenshteinException(Status status, string message) : base(message)
    {
        Status = status;
        StatusCode = (uint)status;
    }

    /// <summary>The symbolic status when known, or its forward-compatible enum value.</summary>
    public Status Status { get; }

    /// <summary>The exact stable ABI status value, including values introduced later.</summary>
    public uint StatusCode { get; }
}
