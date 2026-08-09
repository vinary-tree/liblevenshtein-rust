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

using PhoneticRuleSet builtin = PhoneticRuleSet.Builtin(PhoneticRuleSetKind.EnglishOrthography);
if (builtin.Count == 0) throw new Exception("built-in rule count must be positive");
if (builtin.Apply("phone").Length == 0) throw new Exception("built-in rule application");

Console.WriteLine(".NET binding conformance passed");

PropertyTests.Run();
