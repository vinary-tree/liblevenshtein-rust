// msm-wavefront.asy — the MSM dynamic-programming cost matrix between two
// series X (length m = 6) and Y (length n = 6), with the diagonal WAVEFRONT
// frontier of the DP highlighted.
//
// The exact DP fills Cost(i, j) cell-by-cell; an early-abandoning / automaton
// implementation instead advances a single ANTI-DIAGONAL frontier (cells where
// i + j is constant), because every cell depends only on its W/N/NW neighbours.
//
// Color legend (time-series / MSM concept):
//   frontier cells   fill rgb("B2DFDB"), border rgb("00695C")  = the wavefront
//   plain cells      fill rgb("E0F2F1"), border rgb("00695C")  = the DP matrix
//   lower-bound band fill rgb("F0F4C3")                         = main-diagonal
//                                                                 (length / L1) band
// Neutral scaffolding (axes, title) is dark slate, per the house style.

import fontsize;
settings.outformat = "svg";
size(300);
defaultpen(fontsize(9pt) + rgb("102027"));

int m = 6;        // |X|  (columns, horizontal)
int n = 6;        // |Y|  (rows, vertical)
int frontier = 5; // the anti-diagonal i + j == frontier (a full diagonal sweep)

pen cellFill     = rgb("E0F2F1");
pen cellBorder   = rgb("00695C");
pen frontierFill = rgb("B2DFDB");
pen bandFill     = rgb("F0F4C3");
pen axisPen      = rgb("455A64") + linewidth(1.0);

// 1. Thin lower-bound band along the main diagonal (drawn first, underneath).
for (int d = 0; d < m; ++d) {
  fill(box((d, d), (d + 1, d + 1)), bandFill);
}

// 2. The full grid of DP cells, with the anti-diagonal frontier highlighted.
for (int i = 0; i < m; ++i) {        // i = X index (column)
  for (int j = 0; j < n; ++j) {      // j = Y index (row)
    path cell = box((i, j), (i + 1, j + 1));
    if (i + j == frontier) {
      fill(cell, frontierFill);
    } else if (i == j) {
      // keep the lime band visible: outline only, no overpaint
    } else {
      fill(cell, cellFill);
    }
    draw(cell, cellBorder + linewidth(0.6));
  }
}

// 3. Emphasise the wavefront with a heavier outline on each frontier cell.
for (int i = 0; i < m; ++i) {
  int j = frontier - i;
  if (j >= 0 && j < n) {
    draw(box((i, j), (i + 1, j + 1)), rgb("00695C") + linewidth(1.6));
  }
}

// 4. Axes and arrows.
draw((0, 0)--(m + 0.4, 0), axisPen, Arrow(4));
draw((0, 0)--(0, n + 0.4), axisPen, Arrow(4));
label("X (query series) $\to$", (m / 2, -0.7), rgb("455A64"));
label(rotate(90) * "Y (candidate) $\to$", (-0.7, n / 2), rgb("455A64"));

// 5. Title.
label("MSM wavefront — frontier of the cost matrix", (m / 2, n + 1.1),
      fontsize(11pt) + rgb("102027"));
