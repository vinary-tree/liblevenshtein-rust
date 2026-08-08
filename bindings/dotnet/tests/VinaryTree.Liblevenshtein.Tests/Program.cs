using VinaryTree.Liblevenshtein;

if (Distance.Levenshtein("kitten", "sitting") != 3) throw new Exception("distance mismatch");
if (Distance.Damerau("ab", "ba") != 1) throw new Exception("Damerau mismatch");
if (Distance.TrueDamerau("ca", "abc") != 2) throw new Exception("true Damerau mismatch");
using PhoneticPattern pattern = PhoneticPattern.CompileRegex("cat");
if (!pattern.Matches("cat") || pattern.Matches("cot")) throw new Exception("pattern mismatch");
Console.WriteLine(".NET binding conformance passed");
