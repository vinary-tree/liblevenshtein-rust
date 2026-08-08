using System.Text;

namespace VinaryTree.Liblevenshtein;

/// <summary>A compiled phonetic regular language.</summary>
public sealed class PhoneticPattern : IDisposable
{
    internal PatternHandle Handle { get; }
    private PhoneticPattern(PatternHandle handle) => Handle = handle;
    /// <summary>Compile the phonetic regular-expression syntax.</summary>
    public static PhoneticPattern CompileRegex(string source) => Compile(source, false);
    /// <summary>Compile an import-free LLRE document.</summary>
    public static PhoneticPattern CompileLlre(string source) => Compile(source, true);
    private static unsafe PhoneticPattern Compile(string source, bool llre)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(source);
        fixed (byte* data = bytes)
        {
            Status status = llre ? NativeMethods.PatternLlre(data, (nuint)bytes.Length, out nint result)
                                 : NativeMethods.PatternRegex(data, (nuint)bytes.Length, out result);
            NativeStatus.Check(status);
            return new PhoneticPattern(new PatternHandle(result));
        }
    }
    /// <summary>Test complete-string acceptance.</summary>
    public unsafe bool Matches(string text)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        fixed (byte* data = bytes)
        {
            NativeStatus.Check(NativeMethods.PatternMatches(Handle, data, (nuint)bytes.Length, out byte result));
            return result != 0;
        }
    }
    /// <summary>Return state and transition counts.</summary>
    public (nuint States, nuint Transitions) Size()
    {
        NativeStatus.Check(NativeMethods.PatternSize(Handle, out nuint states, out nuint transitions));
        return (states, transitions);
    }
    /// <inheritdoc />
    public void Dispose() => Handle.Dispose();
}

/// <summary>A built-in rewrite-rule set.</summary>
public enum PhoneticRuleSetKind : uint
{
    /// <summary>English spelling-oriented rewrites.</summary>
    EnglishOrthography = 0,
    /// <summary>English pronunciation-oriented rewrites.</summary>
    EnglishPhonetic = 1,
}

/// <summary>A reusable phonetic rewrite-rule set.</summary>
public sealed class PhoneticRuleSet : IDisposable
{
    private readonly RuleSetHandle handle;
    private PhoneticRuleSet(RuleSetHandle handle) => this.handle = handle;
    /// <summary>Parse an import-free LLEV document.</summary>
    public static unsafe PhoneticRuleSet Parse(string source)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(source);
        fixed (byte* data = bytes)
        {
            NativeStatus.Check(NativeMethods.RulesParse(data, (nuint)bytes.Length, out nint result));
            return new PhoneticRuleSet(new RuleSetHandle(result));
        }
    }
    /// <summary>Load a built-in set.</summary>
    public static PhoneticRuleSet Builtin(PhoneticRuleSetKind kind)
    {
        NativeStatus.Check(NativeMethods.RulesBuiltin((uint)kind, out nint result));
        return new PhoneticRuleSet(new RuleSetHandle(result));
    }
    /// <summary>Number of enabled rules.</summary>
    public nuint Count { get { NativeStatus.Check(NativeMethods.RulesLength(handle, out nuint result)); return result; } }
    /// <summary>Apply the rules to a fixed point.</summary>
    public unsafe string Apply(string input)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(input);
        fixed (byte* data = bytes)
        {
            NativeStatus.Check(NativeMethods.RulesApply(handle, data, (nuint)bytes.Length, out NativeOwnedString result));
            try { return Encoding.UTF8.GetString(new ReadOnlySpan<byte>(result.Data, checked((int)result.Length))); }
            finally { NativeMethods.OwnedStringFree(ref result); }
        }
    }
    /// <inheritdoc />
    public void Dispose() => handle.Dispose();
}
