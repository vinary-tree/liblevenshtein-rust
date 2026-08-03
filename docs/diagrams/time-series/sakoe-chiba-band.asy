// sakoe-chiba-band.asy — why DTW's required band is semantic and operational.
// The left grid shows the only live cells under |i-j| <= w. The upper-right
// chart contrasts an unbanded zero-stutter lower bound with a band-constrained
// prefix whose bound must expose length/value divergence. The lower-right
// cascade makes the O(1) prefix gate occur before the O(w) interval column.
//
// Color legend:
//   live band / admissible geometry  teal
//   valid warping path              blue
//   unreachable cells              rose
//   first prefix gate               amber
//   band-column gate                violet
//   exact verification              green
import fontsize;
import graph;
settings.outformat = "svg";
size(760, 0);
defaultpen(fontsize(9pt) + rgb("102027"));

pen tealFill = rgb("B2DFDB");
pen teal = rgb("00695C") + linewidth(1.25pt);
pen blueFill = rgb("BBDEFB");
pen blue = rgb("1565C0") + linewidth(2.0pt);
pen roseFill = rgb("FFCDD2");
pen rose = rgb("B71C1C") + linewidth(0.8pt);
pen amberFill = rgb("FFE0B2");
pen amber = rgb("E65100") + linewidth(1.25pt);
pen violetFill = rgb("E1BEE7");
pen violet = rgb("6A1B9A") + linewidth(1.25pt);
pen greenFill = rgb("C8E6C9");
pen green = rgb("2E7D32") + linewidth(1.25pt);
pen slate = rgb("455A64") + linewidth(1.0pt);
pen lightGrid = rgb("B0BEC5") + linewidth(0.45pt);

// Panel A: a symmetric w=1 grid.
real gx = 0.6;
real gy = 1.0;
real cell = 0.55;
int count = 8;
for (int i = 0; i < count; ++i) {
  for (int j = 0; j < count; ++j) {
    pair lo = (gx + j * cell, gy + (count - 1 - i) * cell);
    pair hi = lo + (cell, cell);
    if (abs(i - j) <= 1) {
      filldraw(box(lo, hi), tealFill, teal);
    } else {
      filldraw(box(lo, hi), roseFill + opacity(0.48), rose);
    }
  }
}

pair[] path = {
  (gx + 0.5 * cell, gy + 7.5 * cell),
  (gx + 1.5 * cell, gy + 6.5 * cell),
  (gx + 1.5 * cell, gy + 5.5 * cell),
  (gx + 2.5 * cell, gy + 4.5 * cell),
  (gx + 3.5 * cell, gy + 3.5 * cell),
  (gx + 4.5 * cell, gy + 2.5 * cell),
  (gx + 5.5 * cell, gy + 2.5 * cell),
  (gx + 6.5 * cell, gy + 1.5 * cell),
  (gx + 7.5 * cell, gy + 0.5 * cell)
};
draw(graph(path), blue, Arrow(5));
for (pair point : path) filldraw(circle(point, 0.075), blueFill, blue);

label("query index $i$", (gx - 0.30, gy + count * cell / 2), rotate(90) * slate);
label("target prefix index $j$", (gx + count * cell / 2, gy - 0.46), slate);
label("live: $|i-j|\le w$, here $w=1$", (gx + count * cell / 2, gy + count * cell + 0.46), teal);
label("$+\infty$", (gx + 3.7 * cell, gy + 7.45 * cell), rose);
label("valid pinned path", (gx + 2.05 * cell, gy + 5.1 * cell), blue);

// Panel B: lower-bound growth intuition.
real cx = 6.1;
real cy = 4.15;
real cw = 5.5;
real ch = 2.15;
draw((cx, cy)--(cx + cw, cy), slate, Arrow(4));
draw((cx, cy)--(cx, cy + ch), slate, Arrow(4));
label("prefix depth $d$", (cx + cw / 2, cy - 0.38), slate);
label("subtree lower bound", (cx - 0.48, cy + ch / 2), rotate(90) * slate);

pair[] flat = {(cx, cy + 0.22), (cx + cw, cy + 0.22)};
draw(graph(flat), rose + dashed);
label("unbanded stutters can stall at zero", (cx + 3.0, cy + 0.48), rose);

pair[] rising = {
  (cx, cy + 0.22), (cx + 0.8, cy + 0.22), (cx + 1.6, cy + 0.48),
  (cx + 2.5, cy + 0.80), (cx + 3.4, cy + 1.12),
  (cx + 4.35, cy + 1.55), (cx + 5.2, cy + 1.93)
};
draw(graph(rising), teal + linewidth(2pt), Arrow(5));
label("band exposes divergence", (cx + 3.15, cy + 1.55), teal);
draw((cx, cy + 1.05)--(cx + cw, cy + 1.05), amber + dashed);
label("cutoff $\tau$", (cx + cw - 0.62, cy + 1.24), amber);

// Panel C: prefix-first exact-search cascade.
real bx = 6.0;
real by = 1.0;
real bw = 1.62;
real bh = 0.78;
void stage(pair lo, string title, string cost, pen fillPen, pen edgePen) {
  filldraw(box(lo, lo + (bw, bh)), fillPen, edgePen);
  label(title, lo + (bw / 2, 0.52), edgePen);
  label(cost, lo + (bw / 2, 0.23), fontsize(8pt) + slate);
}
stage((bx, by), "prefix LB", "$O(1)$", amberFill, amber);
stage((bx + 2.12, by), "band column", "$O(2w+1)$", violetFill, violet);
stage((bx + 4.24, by), "exact score", "survivors only", greenFill, green);
draw((bx + bw, by + bh / 2)--(bx + 2.12, by + bh / 2), slate, Arrow(5));
draw((bx + 2.12 + bw, by + bh / 2)--(bx + 4.24, by + bh / 2), slate, Arrow(5));
label("pass", (bx + 1.87, by + 0.58), slate);
label("pass", (bx + 3.99, by + 0.58), slate);
draw((bx + bw / 2, by)--(bx + bw / 2, by - 0.42), amber, Arrow(5));
draw((bx + 2.12 + bw / 2, by)--(bx + 2.12 + bw / 2, by - 0.42), violet, Arrow(5));
label("prune", (bx + bw / 2, by - 0.62), amber);
label("prune", (bx + 2.12 + bw / 2, by - 0.62), violet);
label("first gate precedes allocation and DP", (bx + 3.05, by + 1.15), amber);

label("Required band: bounded wavefront, earlier evidence, exact survivors",
      (6.15, 7.15), fontsize(13pt) + rgb("00695C"));
