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
Console.WriteLine(".NET binding conformance passed");
