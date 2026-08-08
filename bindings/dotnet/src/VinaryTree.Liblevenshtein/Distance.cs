using System.Text;

namespace VinaryTree.Liblevenshtein;

/// <summary>Allocation-bounded Unicode edit-distance functions.</summary>
public static class Distance
{
    /// <summary>Standard Levenshtein distance over UTF-8 input.</summary>
    public static unsafe nuint Levenshtein(string source, string target) => Invoke(source, target, NativeMethods.Distance);
    /// <summary>Adjacent-transposition optimal-string-alignment distance.</summary>
    public static unsafe nuint Damerau(string source, string target) => Invoke(source, target, NativeMethods.DamerauDistance);
    /// <summary>True unrestricted Damerau-Levenshtein distance.</summary>
    public static unsafe nuint TrueDamerau(string source, string target) => Invoke(source, target, NativeMethods.TrueDamerauDistance);
    /// <summary>Thresholded Levenshtein distance; returns threshold + 1 when exceeded.</summary>
    public static unsafe nuint Levenshtein(string source, string target, nuint threshold)
    {
        byte[] left = Encoding.UTF8.GetBytes(source);
        byte[] right = Encoding.UTF8.GetBytes(target);
        fixed (byte* l = left)
        fixed (byte* r = right)
            return NativeMethods.DistanceThreshold(l, (nuint)left.Length, r, (nuint)right.Length, threshold);
    }

    private unsafe delegate nuint DistanceCall(byte* left, nuint leftLength, byte* right, nuint rightLength);
    private static unsafe nuint Invoke(string source, string target, DistanceCall call)
    {
        byte[] left = Encoding.UTF8.GetBytes(source);
        byte[] right = Encoding.UTF8.GetBytes(target);
        fixed (byte* l = left)
        fixed (byte* r = right)
            return call(l, (nuint)left.Length, r, (nuint)right.Length);
    }
}
