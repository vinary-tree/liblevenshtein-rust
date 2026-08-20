#!/usr/bin/env wolframscript
(* Reproduce the paired post-optimization Rust/legacy-Java summary figure.
   Raw rows remain authoritative; this script performs presentation only. *)

ClearAll["Global`*"];

scriptDirectory = DirectoryName[$InputFileName];
repository = Nest[DirectoryName, scriptDirectory, 3];
evidence = FileNameJoin[{repository, "benchmarks", "causal", "evidence", "2026-08-19"}];
output = FileNameJoin[{repository, "docs", "diagrams", "benchmarks",
    "postoptimization-paired-latency.svg"}];

experiments = {
  <|"label" -> "unordered\nconstruction",
    "directory" -> "direct-construction-from-terms"|>,
  <|"label" -> "ordered\nconstruction",
    "directory" -> "direct-construction-from-sorted-terms"|>,
  <|"label" -> "standard d=1\nquery pass",
    "directory" -> "direct-standard-d1-hits"|>
};

readRows[directory_] := Module[{table, header},
  table = Import[FileNameJoin[{evidence, directory, "samples.csv"}], "CSV"];
  header = First[table];
  AssociationThread[header, #] & /@ Rest[table]
];

readAnalysis[directory_] :=
  Import[FileNameJoin[{evidence, directory, "analysis.json"}], "RawJSON"];

armValue[group_, arm_] :=
  SelectFirst[group, #["arm"] == arm &]["elapsed_ns"];

speedupValues[data_] := Module[{groups},
  groups = GroupBy[data, #["replicate"] &];
  Values[Map[
    armValue[#, "java"]/armValue[#, "rust"] &,
    groups
  ]]
];

rows = readRows[#["directory"]] & /@ experiments;
analyses = readAnalysis[#["directory"]] & /@ experiments;
labels = #["label"] & /@ experiments;
rustMedians = (#["rust"]["median_ns"] & /@ analyses)/10.^6;
javaMedians = (#["java"]["median_ns"] & /@ analyses)/10.^6;
speedups = speedupValues /@ rows;

rustColor = RGBColor[0.14, 0.48, 0.73];
javaColor = RGBColor[0.88, 0.37, 0.22];
accentColor = RGBColor[0.23, 0.64, 0.45];

latencyPlot = ListLogPlot[
  {
    Transpose[{Range[Length[labels]] - 0.12, rustMedians}],
    Transpose[{Range[Length[labels]] + 0.12, javaMedians}]
  },
  PlotStyle -> {Directive[rustColor, PointSize[0.018]],
    Directive[javaColor, PointSize[0.018]]},
  PlotMarkers -> {{"●", 18}, {"◆", 18}},
  PlotLegends -> Placed[{"Rust", "legacy Java"}, Above],
  Frame -> True,
  Axes -> False,
  FrameLabel -> {None, "median wall time (ms, log scale)"},
  FrameTicks -> {{Automatic, None}, {Thread[{Range[Length[labels]], labels}], None}},
  PlotRange -> All,
  GridLines -> {None, Automatic},
  GridLinesStyle -> Directive[GrayLevel[0.88], Dashed],
  ImageSize -> 1050,
  PlotLabel -> Style["Admitted 51-pair post-optimization latency", 17, Bold]
];

speedupPlot = BoxWhiskerChart[
  speedups,
  ChartStyle -> Directive[accentColor, Opacity[0.8]],
  ChartLabels -> Placed[labels, Below],
  Frame -> True,
  Axes -> False,
  FrameLabel -> {None, "paired speedup: legacy Java time / Rust time"},
  GridLines -> {None, {1}},
  GridLinesStyle -> Directive[GrayLevel[0.45], Dashed],
  ImageSize -> 1050,
  PlotLabel -> Style["Distribution across alternating-order pairs", 17, Bold]
];

figure = GraphicsColumn[
  {latencyPlot, speedupPlot},
  Spacings -> 25,
  ImageSize -> 1100,
  Background -> White
];

Export[output, figure, "SVG"];
Print[output];
