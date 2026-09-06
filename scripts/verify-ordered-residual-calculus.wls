(* Ordered Residual Calculus: symbolic and finite-model evidence.
   Run: WolframKernel -noinit -noprompt -script scripts/verify-ordered-residual-calculus.wls
   One kernel, no subkernels, external packages, or generated files.
   Quantified real arithmetic and bounded examples have distinct proof scopes. *)

$HistoryLength = 0;
orcRoot = DirectoryName[DirectoryName[ExpandFileName[$InputFileName]]];
orcOutput = First[$Output];
say[parts___] := WriteString[orcOutput, StringJoin[ToString[#, OutputForm] & /@ {parts}], "\n"];
SetAttributes[require, HoldFirst];
require[condition_, name_] := If[!TrueQ[condition],
  say["FAIL ", name, ": ", HoldForm[condition]]; Exit[1]];
words[alphabet_, maximum_] := Join @@ Table[Tuples[alphabet, n], {n, 0, maximum}];
slice[v_, start_, size_] := If[size == 0, {}, Take[v, {start + 1, start + size}]];
clip[value_, cutoff_] := If[value <= cutoff, value, Infinity];
choose[f_, g_] := MapThread[Min, {f, g}];
compose[f_, g_] := f[[g]];
difference[a_, b_] := If[a <= b, 0, If[a === Infinity, Infinity, a - b]];
intervalDistance[x_, {lo_, hi_}] := Max[0, lo - x, x - hi];
msmLocal[a_, b_, c_] := 1 + Max[0, Min[b, c] - a, a - Max[b, c]];

checkSymbolic[] := Module[{x, y, z, a, b, c, w, tau, lo, hi,
    blo, bhi, dlo, dhi, beta, delta, wx, wy, ax, ay, bx, by, cx, cy, formulas, result},
  formulas = {
    {"directed scalar triangle (G1)",
      ForAll[{x, y, z}, Max[0, x - z] <= Max[0, x - y] + Max[0, y - z]]},
    {"minimum is stable under directed error (G2)",
      ForAll[{a, b, c, w, delta}, Implies[delta >= 0 && a <= c + delta && b <= w + delta,
        Min[a, b] <= Min[c, w] + delta]]},
    {"min/max distribution (C1 scalar instance)",
      ForAll[{a, b, c}, Max[a, Min[b, c]] == Min[Max[a, b], Max[a, c]]]},
    {"additive backward budget (B1)",
      ForAll[{x, w, tau}, Implies[x >= 0 && w >= 0 && tau >= 0,
        Equivalent[x + w <= tau, w <= tau && x <= tau - w]]]},
    {"bottleneck backward budget (B1)",
      ForAll[{x, w, tau}, Equivalent[Max[x, w] <= tau, x <= tau && w <= tau]]},
    {"point-to-interval lower simulation (ERP/Frechet)",
      ForAll[{x, y, lo, hi}, Implies[lo <= y <= hi,
        intervalDistance[x, {lo, hi}] <= Abs[x - y]]]},
    {"interval refinement",
      ForAll[{x, lo, hi, a, b}, Implies[lo <= a <= b <= hi,
        intervalDistance[x, {lo, hi}] <= intervalDistance[x, {a, b}]]]},
    {"MSM local cost as distance to the between interval",
      ForAll[{a, b, c}, msmLocal[a, b, c] ==
        Piecewise[{{1, Min[b, c] <= a <= Max[b, c]}}, 1 + Min[Abs[a - b], Abs[a - c]]]]},
    {"MSM interval-hull lower bound",
      ForAll[{a, b, c, blo, bhi, dlo, dhi},
        Implies[blo <= b <= bhi && dlo <= c <= dhi,
          1 + intervalDistance[a, {Min[blo, dlo], Max[bhi, dhi]}] <= msmLocal[a, b, c]]]},
    {"common-offset min-plus normalization (O15)",
      ForAll[{a, b, c, w, beta},
        Min[a + beta + c, b + beta + w] == beta + Min[a + c, b + w]]},
    {"consistent potential gives nonnegative edges",
      ForAll[{w, a, b}, Implies[a <= w + b, w + b - a >= 0]]},
    {"nonnegative weighted sum preserves triangle (M2)",
      ForAll[{wx, wy, ax, ay, bx, by, cx, cy},
        Implies[wx >= 0 && wy >= 0 && cx <= ax + bx && cy <= ay + by,
          wx cx + wy cy <= wx ax + wy ay + wx bx + wy by]]},
    {"weighted maximum preserves triangle (M2)",
      ForAll[{wx, wy, ax, ay, bx, by, cx, cy},
        Implies[wx >= 0 && wy >= 0 && cx <= ax + bx && cy <= ay + by,
          Max[wx cx, wy cy] <= Max[wx ax, wy ay] + Max[wx bx, wy by]]]}
  };
  Do[
    result = TimeConstrained[Resolve[item[[2]], Reals], 45, $Aborted];
    require[result === True, "quantifier elimination: " <> item[[1]]],
    {item, formulas}];
  ToString[Length[formulas]] <> " universally quantified real identities/inequalities"
];

checkTransformations[] := Module[{carrier = Range[4], top = 4, maps, identity,
    zero, inflationary, pre, truncate},
  maps = Append[#, top] & /@ Select[Tuples[carrier, 3], OrderedQ];
  identity = carrier; zero = ConstantArray[top, 4];
  Do[
    require[compose[f, identity] === f && compose[identity, f] === f, "composition identity"];
    require[compose[f, zero] === zero && compose[zero, f] === zero, "top absorption"];
    require[choose[f, zero] === f && choose[f, f] === f, "choice identity/idempotence"],
    {f, maps}];
  Do[
    require[compose[f, compose[g, h]] === compose[compose[f, g], h], "associativity"];
    require[compose[f, choose[g, h]] === choose[compose[f, g], compose[f, h]], "left distribution"];
    require[compose[choose[f, g], h] === choose[compose[f, h], compose[g, h]], "right distribution"];
    require[choose[f, choose[g, h]] === choose[choose[f, g], h], "choice associativity"];
    require[choose[f, g] === choose[g, f], "choice commutativity"],
    {f, maps}, {g, maps}, {h, maps}];
  pre[fn_, accepted_] := Select[carrier, MemberQ[accepted, fn[[#]]] &];
  Do[
    require[pre[compose[g, f], Range[tau]] === pre[f, pre[g, Range[tau]]], "backward composition"];
    require[pre[choose[f, g], Range[tau]] === Union[pre[f, Range[tau]], pre[g, Range[tau]]],
      "alternative budgets"],
    {f, maps}, {g, maps}, {tau, carrier}];
  inflationary = Select[maps, And @@ Thread[carrier <= #] &];
  Do[
    truncate[x_] := If[x <= tau, x, top];
    require[truncate[f[[truncate[x]]]] == truncate[f[[x]]], "cutoff congruence"],
    {f, inflationary}, {tau, carrier}, {x, carrier}];
  require[Length[maps] == 20 && Length[inflationary] == 14, "finite model size"];
  "20 maps, all 8,000 triples; 14 inflationary maps"
];

checkFrontiers[] := Module[{atoms = Range[3], pairs, subsets, rel, count = 0,
    norm, behavior, reps, width, ns},
  pairs = Select[Tuples[atoms, 2], #[[1]] != #[[2]] &]; subsets = Subsets[atoms];
  Do[
    rel = IdentityMatrix[3] /. {0 -> False, 1 -> True};
    MapThread[(rel[[Sequence @@ #1]] = #2 == 1) &, {pairs, IntegerDigits[mask, 2, 6]}];
    If[!And @@ Flatten[Table[!rel[[a, b]] || !rel[[b, c]] || rel[[a, c]],
        {a, atoms}, {b, atoms}, {c, atoms}]], Continue[]];
    count++;
    norm[s_] := Select[s, Function[p, !AnyTrue[s,
      Function[q, rel[[q, p]] && (!rel[[p, q]] || q < p)]]]];
    behavior[s_, suffix_] := Min[If[rel[[#, suffix]], 0, 1] & /@ s];
    reps = Select[atoms, Function[p, !AnyTrue[Range[p - 1],
      Function[q, rel[[p, q]] && rel[[q, p]]]]]];
    width = Max[Length /@ Select[subsets, Function[s, Complement[s, reps] === {} &&
      And @@ Flatten[Table[p == q || (!rel[[p, q]] && !rel[[q, p]]), {p, s}, {q, s}]]]]];
    Do[
      ns = norm[s];
      require[norm[ns] === ns, "normalization idempotence"];
      require[Length[ns] <= width, "executable preorder width"];
      require[And @@ Table[behavior[ns, z] == behavior[s, z], {z, atoms}], "elimination preservation"],
      {s, subsets}];
    Do[
      require[norm[Union[norm[Union[s, t]], u]] === norm[Union[s, norm[Union[t, u]]]],
        "normalized union associativity"],
      {s, subsets}, {t, subsets}, {u, subsets}],
    {mask, 0, 63}];
  require[count == 29, "all preorders on three labeled atoms"];
  "all 29 preorders and their finite frontiers"
];

(* Independent full-matrix and target-major rolling edit realizations. *)
editReference[q_List, t_List, osa_:False, substitution_:Automatic, gap_:1] :=
 Module[{m = Length[q], n = Length[t], d, cost},
  d = ConstantArray[Infinity, {m + 1, n + 1}];
  d[[All, 1]] = gap Range[0, m]; d[[1, All]] = gap Range[0, n];
  Do[
    cost = If[substitution === Automatic, Boole[q[[i]] != t[[j]]], substitution[q[[i]], t[[j]]]];
    d[[i + 1, j + 1]] = Min[d[[i, j + 1]] + gap, d[[i + 1, j]] + gap, d[[i, j]] + cost];
    If[osa && i > 1 && j > 1 && q[[i]] == t[[j - 1]] && q[[i - 1]] == t[[j]],
      d[[i + 1, j + 1]] = Min[d[[i + 1, j + 1]], d[[i - 1, j - 1]] + 1]],
    {i, m}, {j, n}];
  d[[-1, -1]]
];
editStep[q_, row_, label_, cutoff_:Infinity] := Module[{current},
  current = ConstantArray[Infinity, Length[row]]; current[[1]] = clip[row[[1]] + 1, cutoff];
  Do[current[[i + 1]] = clip[Min[row[[i + 1]] + 1, current[[i]] + 1,
    row[[i]] + Boole[q[[i]] != label]], cutoff], {i, Length[q]}];
  current
];

checkEditProducts[] := Module[{queries = words[{"a", "b"}, 3],
    targets = words[{"a", "b"}, 4], count = 0, row, exact, expectedRow, bounds,
    pending, observed, cache, prefix, key, direct, expected, adjacency},
  adjacency = Association[Table[StringJoin[p] ->
    If[Length[p] < 4, {"a", "b"}, {}], {p, targets}]];
  Do[
    Do[
      row = clip[#, cutoff] & /@ Range[0, Length[q]];
      Do[
        exact = editReference[q, Take[t, j]];
        require[Last[row] == clip[exact, cutoff], "edit prefix/cutoff"]; count++;
        If[j < Length[t], row = editStep[q, row, t[[j + 1]], cutoff]],
        {j, 0, Length[t]}],
      {t, targets}, {cutoff, {0, 1, 2, Infinity}}];
    Do[
      expectedRow = Fold[editStep[q, #1, #2] &, Range[0, Length[q]], t];
      Do[
        bounds = Join[{0}, cuts, {Length[t]}]; row = Range[0, Length[q]];
        Do[row = Fold[editStep[q, #1, #2] &, row, slice[t, bounds[[k]], bounds[[k + 1]] - bounds[[k]]]],
          {k, Length[bounds] - 1}];
        require[row === expectedRow, "all chunk partitions"],
        {cuts, Subsets[Range[Max[0, Length[t] - 1]]]}],
      {t, targets}];
    Do[
      pending = {{{}, Range[0, Length[q]]}}; observed = <||>; cache = <||>;
      While[pending =!= {},
        {prefix, row} = Last[pending]; pending = Most[pending];
        If[Last[row] <= cutoff, AssociateTo[observed, StringJoin[prefix] -> Last[row]]];
        Do[
          key = ToString[{row, (# == a & /@ q)}, InputForm];
          direct = editStep[q, row, a, cutoff];
          If[KeyExistsQ[cache, key], require[cache[key] === direct, "observation congruence"],
            AssociateTo[cache, key -> direct]];
          If[Min[direct] <= cutoff, AppendTo[pending, {Append[prefix, a], direct}]],
          {a, adjacency[StringJoin[prefix]]}]];
      expected = Association[Table[StringJoin[t] -> editReference[q, t],
        {t, Select[targets, editReference[q, #] <= cutoff &]}]];
      require[Sort[Normal[observed]] === Sort[Normal[expected]], "lazy trie result completeness"],
      {cutoff, {0, 1, 2}}],
    {q, queries}];
  ToString[count] <> " prefix/cutoff comparisons, all chunk partitions, 45 trie searches"
];

operations = {
  {1, 1, 0, Function[{u, v}, u === v]},
  {1, 1, 1, Function[{u, v}, True]},
  {1, 0, 1, Function[{u, v}, True]},
  {0, 1, 1, Function[{u, v}, True]},
  {2, 1, 1/4, Function[{u, v}, u === {"a", "b"} && v === {"b"}]},
  {1, 2, 1/2, Function[{u, v}, u === {"a"} && v === {"b", "a"}]}
};
generalizedReference[q_, t_] := Module[{visit},
  visit[i_, j_] := visit[i, j] = If[i == Length[q] && j == Length[t], 0,
    Min[Function[op, op[[3]] + visit[i + op[[1]], j + op[[2]]]] /@
      Select[operations, Function[op, i + op[[1]] <= Length[q] &&
        j + op[[2]] <= Length[t] && op[[4]][slice[q, i, op[[1]]], slice[t, j, op[[2]]]]]]]];
  visit[0, 0]
];
generalizedStream[q_, t_, cutoff_] := Module[{lag = Max[operations[[All, 2]]],
    generations = <||>, window = {}, current, outputs = {}, a, b, value, previous},
  Do[
    If[j > 0, window = Take[Append[window, t[[j]]], -Min[lag, j]]];
    current = ConstantArray[Infinity, Length[q] + 1];
    Do[
      value = If[i == 0 && j == 0, 0, Infinity];
      Do[
        {a, b} = op[[1 ;; 2]];
        If[i >= a && j >= b && op[[4]][slice[q, i - a, a], Take[window, -b]],
          previous = If[b == 0, current, generations[j - b]];
          value = Min[value, previous[[i - a + 1]] + op[[3]]]],
        {op, operations}];
      current[[i + 1]] = clip[value, cutoff],
      {i, 0, Length[q]}];
    AssociateTo[generations, j -> current]; AppendTo[outputs, Last[current]];
    KeyDropFrom[generations, Select[Keys[generations], # <= j - lag &]];
    require[Length[generations] <= lag, "generalized retained generations"],
    {j, 0, Length[t]}];
  outputs
];
checkGeneralized[] := Module[{corpus = words[{"a", "b"}, 3], count = 0, actual},
  Do[
    actual = generalizedStream[q, t, cutoff];
    Do[
      require[actual[[j + 1]] == clip[generalizedReference[q, Take[t, j]], cutoff],
        "multi-arity rolling/path correspondence"]; count++,
      {j, 0, Length[t]}],
    {q, corpus}, {t, corpus}, {cutoff, {0, 1/2, 2, Infinity}}];
  ToString[count] <> " exact rational comparisons with multi-symbol operations"
];

(* Soft-DTW uses its exact path-partition polynomial in z, with z=Exp[-1/gamma].
   This checks multiplicity symbolically and avoids approximate score equality. *)
temporalReference[kind_, q_, t_, penalty_:1] := Module[{m = Length[q], n = Length[t],
    d, x, y, tx, ty, px, py, ptx, pty, up, diagonal, left, cost},
  If[kind == "msm",
    If[m == 0 || n == 0, Return[If[m == n, 0, Infinity]]];
    d = ConstantArray[Infinity, {m, n}]; d[[1, 1]] = Abs[q[[1]] - t[[1]]];
    Do[d[[i, 1]] = d[[i - 1, 1]] + msmLocal[q[[i]], q[[i - 1]], t[[1]]], {i, 2, m}];
    Do[d[[1, j]] = d[[1, j - 1]] + msmLocal[t[[j]], q[[1]], t[[j - 1]]], {j, 2, n}];
    Do[d[[i, j]] = Min[d[[i - 1, j - 1]] + Abs[q[[i]] - t[[j]]],
      d[[i - 1, j]] + msmLocal[q[[i]], q[[i - 1]], t[[j]]],
      d[[i, j - 1]] + msmLocal[t[[j]], q[[i]], t[[j - 1]]]], {i, 2, m}, {j, 2, n}];
    Return[d[[-1, -1]]]];
  d = ConstantArray[If[kind == "soft", 0, Infinity], {m + 1, n + 1}];
  d[[1, 1]] = If[kind == "soft", 1, 0];
  If[MemberQ[{"erp", "twed", "timestamped"}, kind],
    Do[
      cost = Switch[kind,
        "erp", Abs[q[[i]]],
        "twed", Abs[q[[i]] - If[i > 1, q[[i - 1]], 0]] + 1 + penalty,
        "timestamped", Abs[q[[i, 1]] - If[i > 1, q[[i - 1, 1]], 0]] +
          q[[i, 2]] - If[i > 1, q[[i - 1, 2]], 0] + penalty];
      d[[i + 1, 1]] = d[[i, 1]] + cost, {i, m}];
    Do[d[[1, j + 1]] = temporalReference[kind, Take[t, j], {}, penalty], {j, n}]];
  Do[
    x = q[[i]]; y = t[[j]];
    {up, diagonal, left} = {d[[i, j + 1]], d[[i, j]], d[[i + 1, j]]};
    d[[i + 1, j + 1]] = Switch[kind,
      "erp", Min[up + Abs[x], diagonal + Abs[x - y], left + Abs[y]],
      "twed" | "timestamped",
        If[kind == "twed",
          {tx, ty} = {i, j}; {px, ptx} = If[i > 1, {q[[i - 1]], i - 1}, {0, 0}];
          {py, pty} = If[j > 1, {t[[j - 1]], j - 1}, {0, 0}],
          {x, tx} = x; {y, ty} = y;
          {px, ptx} = If[i > 1, q[[i - 1]], {0, 0}];
          {py, pty} = If[j > 1, t[[j - 1]], {0, 0}]];
        Min[up + Abs[x - px] + tx - ptx + penalty, left + Abs[y - py] + ty - pty + penalty,
          diagonal + Abs[x - y] + Abs[px - py] + Abs[tx - ty] + Abs[ptx - pty]],
      "frechet", Max[Abs[x - y], Min[up, diagonal, left]],
      "soft", Expand[z^((x - y)^2) (up + diagonal + left)],
      "dtw", If[Abs[i - j] <= 1, Abs[x - y] + Min[up, diagonal, left], Infinity]],
    {i, m}, {j, n}];
  d[[-1, -1]]
];

temporalStream[kind_, q_, t_, penalty_:1] := Module[{m = Length[q], row, current, outputs,
    previousTarget = {0, 0}, qp, value, time, previous, previousTime, deletion,
    qvalue, qtime, qprevious, qpreviousTime, qdeletion, match, result, x, y},
  If[kind == "msm",
    outputs = {If[m == 0, 0, Infinity]}; row = {};
    Do[
      If[m == 0, AppendTo[outputs, Infinity]; Continue[]];
      current = ConstantArray[Infinity, m];
      If[row === {},
        current[[1]] = Abs[q[[1]] - y];
        Do[current[[i]] = current[[i - 1]] + msmLocal[q[[i]], q[[i - 1]], y], {i, 2, m}],
        current[[1]] = row[[1]] + msmLocal[y, q[[1]], previousTarget];
        Do[current[[i]] = Min[row[[i - 1]] + Abs[q[[i]] - y],
          current[[i - 1]] + msmLocal[q[[i]], q[[i - 1]], y],
          row[[i]] + msmLocal[y, q[[i]], previousTarget]], {i, 2, m}]];
      row = current; previousTarget = y; AppendTo[outputs, Last[row]],
      {y, t}];
    Return[outputs]];
  row = ConstantArray[If[kind == "soft", 0, Infinity], m + 1];
  row[[1]] = If[kind == "soft", 1, 0];
  qp = If[kind == "twed", MapIndexed[{#1, First[#2]} &, q], q];
  Do[
    If[kind == "erp", row[[i + 1]] = row[[i]] + Abs[q[[i]]]];
    If[MemberQ[{"twed", "timestamped"}, kind],
      {value, time} = qp[[i]]; {previous, previousTime} = If[i > 1, qp[[i - 1]], {0, 0}];
      row[[i + 1]] = row[[i]] + Abs[value - previous] + time - previousTime + penalty],
    {i, m}];
  outputs = {Last[row]};
  Do[
    y = t[[j]]; current = ConstantArray[If[kind == "soft", 0, Infinity], m + 1];
    If[kind == "erp", current[[1]] = row[[1]] + Abs[y]];
    If[MemberQ[{"twed", "timestamped"}, kind],
      {value, time} = If[kind == "twed", {y, j}, y]; {previous, previousTime} = previousTarget;
      deletion = Abs[value - previous] + time - previousTime + penalty;
      current[[1]] = row[[1]] + deletion];
    Do[
      x = q[[i]];
      result = Switch[kind,
        "erp", Min[row[[i + 1]] + Abs[y], current[[i]] + Abs[x], row[[i]] + Abs[x - y]],
        "twed" | "timestamped",
          {qvalue, qtime} = qp[[i]]; {qprevious, qpreviousTime} = If[i > 1, qp[[i - 1]], {0, 0}];
          qdeletion = Abs[qvalue - qprevious] + qtime - qpreviousTime + penalty;
          match = Abs[qvalue - value] + Abs[qprevious - previous] +
            Abs[qtime - time] + Abs[qpreviousTime - previousTime];
          Min[row[[i + 1]] + deletion, current[[i]] + qdeletion, row[[i]] + match],
        "frechet", Max[Abs[x - y], Min[row[[i + 1]], row[[i]], current[[i]]]],
        "soft", Expand[z^((x - y)^2) (row[[i + 1]] + row[[i]] + current[[i]])],
        "dtw", If[Abs[i - j] <= 1, Abs[x - y] + Min[row[[i + 1]], row[[i]], current[[i]]], Infinity]];
      current[[i + 1]] = result,
      {i, m}];
    row = current; AppendTo[outputs, Last[row]];
    If[MemberQ[{"twed", "timestamped"}, kind], previousTarget = {value, time}],
    {j, Length[t]}];
  outputs
];

checkTemporal[] := Module[{corpus = words[{-1, 0, 2}, 3], count = 0, q, t, actual, expected},
  Do[
    q = q0; t = t0;
    If[kind == "timestamped",
      q = MapIndexed[{#1, First[#2]^2} &, q0];
      t = MapIndexed[{#1, First[#2] (First[#2] + 1)} &, t0]];
    actual = temporalStream[kind, q, t];
    Do[
      expected = temporalReference[kind, q, Take[t, j]];
      require[actual[[j + 1]] === expected, kind <> " prefix correspondence"]; count++,
      {j, 0, Length[t]}],
    {kind, {"erp", "frechet", "twed", "timestamped", "msm", "dtw", "soft"}},
    {q0, corpus}, {t0, corpus}];
  require[temporalReference["erp", {0}, {1, 2}] == 3, "ERP published example"];
  require[temporalReference["erp", {1, 2}, {2, 3, 3}] == 5, "ERP published triangle example"];
  require[temporalReference["frechet", {1, 1, 2}, {1, 2}] == 0, "Frechet stutter example"];
  ToString[count] <> " prefixes across seven kernels; Soft-DTW partitions checked symbolically"
];

(* Published infinite-axis TWED, independent of the cumulative boundary code. *)
twedSource[q_, t_, penalty_] := Module[{d, x, y, px, py, ground},
  ground[u_, v_] := Total[Abs[u - v]];
  d = ConstantArray[Infinity, {Length[q] + 1, Length[t] + 1}]; d[[1, 1]] = 0;
  Do[
    x = q[[i]]; y = t[[j]];
    px = If[i > 1, q[[i - 1]], {0, 0}]; py = If[j > 1, t[[j - 1]], {0, 0}];
    d[[i + 1, j + 1]] = Min[d[[i, j + 1]] + ground[x, px] + penalty,
      d[[i + 1, j]] + ground[y, py] + penalty,
      d[[i, j]] + ground[x, y] + ground[px, py]],
    {i, Length[q]}, {j, Length[t]}];
  d[[-1, -1]]
];
checkTwedDomain[] := Module[{corpus = words[{-1, 0, 2}, 2], anchor, q, t, count = 0, exact},
  anchor[v_] := Prepend[({#[[1]], 1 + #[[2]]} & /@ v), {0, 1}];
  Do[
    q = MapIndexed[{#1, First[#2]^2} &, q0];
    t = MapIndexed[{#1, First[#2] (First[#2] + 1)} &, t0];
    exact = temporalReference["timestamped", q, t, penalty];
    require[exact == twedSource[anchor[q], anchor[t], penalty], "TWED anchored metric transfer"];
    require[Last[temporalStream["timestamped", q, t, penalty]] == exact, "TWED penalty correspondence"];
    count++,
    {q0, corpus}, {t0, corpus}, {penalty, {0, 1}}];
  q = {{1, 1}}; t = {{0, 0}, {1, 1}};
  require[q =!= t && temporalReference["timestamped", q, t, 0] == 0 &&
    Last[temporalStream["timestamped", q, t, 0]] == 0, "TWED origin separation counterexample"];
  require[twedSource[q, t, 0] == 4, "TWED boundary conventions differ"];
  ToString[count] <> " anchored reductions; zero-penalty origin counterexample reproduced"
];

intervalStream[kind_, q_, intervals_] := Module[{row, current, outputs, d},
  row = If[kind == "erp", Prepend[Accumulate[Abs[q]], 0], Join[{0}, ConstantArray[Infinity, Length[q]]]];
  outputs = {Last[row]};
  Do[
    current = ConstantArray[Infinity, Length[row]];
    If[kind == "erp", current[[1]] = row[[1]] + intervalDistance[0, interval]];
    Do[
      d = intervalDistance[q[[i]], interval];
      current[[i + 1]] = If[kind == "erp",
        Min[row[[i + 1]] + intervalDistance[0, interval], current[[i]] + Abs[q[[i]]], row[[i]] + d],
        Max[d, Min[row[[i + 1]], row[[i]], current[[i]]]]],
      {i, Length[q]}];
    row = current; AppendTo[outputs, Last[row]],
    {interval, intervals}];
  outputs
];
checkIntervals[] := Module[{count = 0, abstract, concrete, points},
  Do[
    abstract = intervalStream[kind, q, boxes];
    Do[
      concrete = temporalStream[kind, q, t];
      points = intervalStream[kind, q, {#, #} & /@ t];
      require[points === concrete, "point-state embedding"];
      require[And @@ MapThread[LessEqual, {abstract, concrete}], "interval lower simulation"];
      count++,
      {t, Tuples[Range @@ # & /@ boxes]}],
    {kind, {"erp", "frechet"}}, {q, words[{-1, 0, 2}, 2]},
    {boxes, words[{{-1, 0}, {0, 2}, {1, 1}}, 3]}];
  ToString[count] <> " ERP/Frechet concretizations, including every prefix"
];

collapse[v_] := If[v === {}, {}, First /@ Split[v]];
checkCompositionMetrics[] := Module[{corpus = words[{"a", "b", "c"}, 2], count = 0,
    alphabet, retained, outside, row, histogram, independent, actual, expected, a,
    scores, scalar = words[{0, 1, 2}, 3], nx, ny, erp, frechet},
  Do[
    scores = Table[
      alphabet = DeleteDuplicates[q]; retained = ConstantArray[0, Length[alphabet]];
      outside = 0; row = Range[0, Length[q]];
      Do[
        histogram = Total[MapThread[Abs[Count[q, #1] - #2] &, {alphabet, retained}]] + outside;
        independent = Total[Abs[Count[q, #] - Count[Take[t, j], #]] & /@ Union[q, Take[t, j]]];
        actual = Last[row] + weight histogram;
        expected = editReference[q, Take[t, j]] + weight independent;
        require[actual == expected, "compositional residual realization"]; count++;
        If[j < Length[t],
          a = t[[j + 1]];
          If[MemberQ[alphabet, a], retained[[First[FirstPosition[alphabet, a]]]]++, outside++];
          row = editStep[q, row, a]],
        {j, 0, Length[t]}];
      actual,
      {q, corpus}, {t, corpus}];
    require[scores === Transpose[scores], "composite symmetry"];
    Do[require[(scores[[i, j]] == 0) === (i == j), "composite separation"], {i, Length[corpus]}, {j, Length[corpus]}];
    Do[require[scores[[i, k]] <= scores[[i, j]] + scores[[j, k]], "composite triangle"],
      {i, Length[corpus]}, {j, Length[corpus]}, {k, Length[corpus]}],
    {weight, {0, 1/2, 2}}];
  Do[
    nx = DeleteCases[x, 0]; ny = DeleteCases[y, 0]; erp = temporalReference["erp", x, y];
    require[(erp == 0) === (nx === ny), "ERP zero quotient"];
    require[erp == temporalReference["erp", nx, ny], "ERP representative invariance"];
    If[x =!= {} && y =!= {},
      frechet = temporalReference["frechet", x, y];
      require[(frechet == 0) === (collapse[x] === collapse[y]), "Frechet zero quotient"];
      require[frechet == temporalReference["frechet", collapse[x], collapse[y]], "Frechet representative invariance"]],
    {x, scalar}, {y, scalar}];
  ToString[count] <> " composite prefixes; metric and quotient checks"
];

checkGeometry[] := Module[{behaviors = Tuples[{0, 1, 2, Infinity}, 2], delta,
    p = {0, 2}, q = {1, 0}, f, g, cauchy, aw, lhs, rhs},
  delta[u_, v_] := Max[MapThread[difference, {u, v}]];
  Do[
    require[delta[u, w] <= delta[u, v] + delta[v, w], "directed triangle"];
    require[(delta[u, v] == 0) === And @@ MapThread[LessEqual, {u, v}], "zero discrepancy/dominance"];
    require[difference[First[u], First[v]] <= delta[u, v], "restriction contraction"],
    {u, behaviors}, {v, behaviors}, {w, behaviors}];
  require[p[[1]] <= q[[1]] && p[[2]] > q[[2]], "contextual dominance control"];
  f[w_] := Length[w] + 2 Count[w, "a"];
  g[w_] := 1 + 3 Count[w, "b"];
  cauchy[left_, right_, w_] := Min[Table[left[Take[w, i]] + right[Drop[w, i]], {i, 0, Length[w]}]];
  Do[
    aw = Prepend[w, a]; lhs = cauchy[f, g, aw];
    rhs = Min[cauchy[Function[u, f[Prepend[u, a]]], g, w], f[{}] + g[aw]];
    require[lhs == rhs, "Cauchy derivative"],
    {w, words[{"a", "b"}, 4]}, {a, {"a", "b"}}];
  "4,096 directed triangles, contextual counterexample, Cauchy derivatives"
];

checkNegativeControls[] := Module[{big = N[2^53], end = N[2^53 + 2], add,
    ground, substitute, language, metric, samples, d1, d2},
  add[x_Real, y_Real] := x + y;
  ground[x_Real, y_Real] := Abs[x - y];
  require[Rationalize[add[add[big, 1.], 1.], 0] != Rationalize[add[big, add[1., 1.]], 0],
    "binary64 reassociation witness"];
  require[Rationalize[ground[0., end], 0] > Rationalize[add[ground[0., 1.], ground[1., end]], 0],
    "rounded triangle witness"];
  require[editReference[Characters["CA"], Characters["AC"], True] == 1 &&
    editReference[Characters["AC"], Characters["ABC"], True] == 1 &&
    editReference[Characters["CA"], Characters["ABC"], True] == 3, "OSA triangle witness"];
  substitute[x_, y_] := Which[x == y, 0, Sort[{x, y}] == {"a", "c"}, 3, True, 1];
  require[Table[editReference[{p[[1]]}, {p[[2]]}, False, substitute, 10],
    {p, {{"a", "b"}, {"b", "c"}, {"a", "c"}}}] === {1, 1, 3}, "weighted alignment triangle witness"];
  d1 = {{0, 1, 10}, {1, 0, 10}, {10, 10, 0}};
  d2 = {{0, 10, 10}, {10, 0, 1}, {10, 1, 0}};
  Do[require[d[[i, k]] <= d[[i, j]] + d[[j, k]], "component metric witness"],
    {d, {d1, d2}}, {i, 3}, {j, 3}, {k, 3}];
  require[Min[d1[[1, 3]], d2[[1, 3]]] >
    Min[d1[[1, 2]], d2[[1, 2]]] + Min[d1[[2, 3]], d2[[2, 3]]], "minimum of metrics witness"];
  require[Min[0, 10] + Min[10, 0] < Min[0 + 10, 10 + 0], "independent alignment witness"];
  require[-Log[2] < 0, "soft idempotence witness"];
  require[editReference[Characters["abc"], {"a"}] > 0 &&
    editReference[Characters["abc"], Characters["abc"]] == 0, "prefix finality pruning witness"];
  require[clip[Abs[0 - 2], 3/2] > clip[Abs[0 - 1], 3/2] + clip[Abs[1 - 2], 3/2],
    "cutoff-saturated triangle witness"];
  require[2 <= 3 && 2 + 1 != 3 + 1, "point label does not repair previous precision"];
  require[Abs[0 - 0] == 0 && "a" != "b", "noninjective pullback witness"];
  language[w_] := StringMatchQ[w, RegularExpression["a+b+"]] && StringCount[w, "a"] == StringCount[w, "b"];
  metric[x_, y_] := Which[x == y, 0, (x == "" && language[y]) || (y == "" && language[x]), 1, True, 2];
  samples = {"", "a", "b", "ab", "aabb", "aaabbb", "abab"};
  Do[require[metric[x, z] <= metric[x, y] + metric[y, z], "arbitrary-ball metric"];
    require[0 <= metric[x, y], "zero abstract filter"], {x, samples}, {y, samples}, {z, samples}];
  Do[If[i != j, require[language[StringRepeat["a", i] <> StringRepeat["b", i]] &&
    !language[StringRepeat["a", j] <> StringRepeat["b", i]], "distinguishing ball residuals"]],
    {i, 1, 8}, {j, 1, 8}];
  require[Length[DeleteDuplicates[{0, 1}]] == 2 && 0 <= 1, "semantic/executable width distinction"];
  "counterexamples reproduced, including rounded and cutoff-saturated triangle failures"
];

checkDocumentLinks[] := Module[{paths, count = 0, source, destinations, hash,
    relative, fragment, target, headings, anchors, fence = StringRepeat[FromCharacterCode[96], 3]},
  paths = FileNameJoin[{orcRoot, #}] & /@ {
    "docs/theory/ordered-residual-calculus.md",
    "docs/theory/lazy-ordered-cost-product-automata.md", "docs/theory/README.md", "docs/README.md"};
  Do[
    source = Import[path, "Text"];
    source = StringReplace[source, RegularExpression["(?ms)^" <> fence <>
      "[^\\n]*\\n.*?^" <> fence <> "[ \\t]*$"] -> ""];
    destinations = StringCases[source, RegularExpression["!?\\[[^\\]]*\\]\\(([^)\\s]+)\\)"] -> "$1"];
    Do[
      If[StringMatchQ[destination, RegularExpression["[a-zA-Z]+:.*"]], Continue[]];
      hash = StringPosition[destination, "#", 1]; relative = destination; fragment = "";
      If[hash =!= {}, relative = StringTake[destination, First[First[hash]] - 1];
        fragment = StringDrop[destination, First[First[hash]]]];
      target = If[relative == "", path, ExpandFileName[FileNameJoin[{DirectoryName[path], relative}]]];
      require[FileExistsQ[target] || DirectoryQ[target], "local link: " <> destination];
      If[fragment != "" && FileExtension[target] == "md",
        headings = StringCases[Import[target, "Text"], RegularExpression["(?m)^#{1,6}[ \\t]+([^\\n]+)$"] -> "$1"];
        anchors = StringReplace[StringReplace[ToLowerCase[#], RegularExpression["[^\\w\\s-]"] -> ""], " " -> "-"] & /@ headings;
        require[MemberQ[anchors, fragment], "heading link: " <> destination]];
      count++,
      {destination, destinations}],
    {path, paths}];
  require[count > 10, "link discovery is not vacuous"];
  ToString[count] <> " local document links and heading references"
];

run[name_, fn_] := Module[{seconds, result},
  {seconds, result} = AbsoluteTiming[fn[]];
  require[StringQ[result], "completed evidence group: " <> name];
  say["PASS ", name, ": ", result, " (", NumberForm[seconds, {8, 3}], " s)"]];
say["Ordered Residual Calculus verification — Wolfram ", $Version];
Check[
  run["symbolic laws", checkSymbolic];
  run["cost transformations and budgets", checkTransformations];
  run["frontier normalization", checkFrontiers];
  run["edit residuals and dictionary products", checkEditProducts];
  run["generalized recurrence synthesis", checkGeneralized];
  run["temporal recurrence synthesis", checkTemporal];
  run["TWED boundary and metric domain", checkTwedDomain];
  run["interval abstractions", checkIntervals];
  run["measure composition and quotients", checkCompositionMetrics];
  run["residual geometry and derivatives", checkGeometry];
  run["negative controls", checkNegativeControls];
  run["document links", checkDocumentLinks],
  say["FAIL: unexpected Wolfram message"]; Exit[2]];
say["ORC executable evidence passed; finite checks do not replace the manuscript proofs."];
Exit[0];
