// dyck-state-explosion.asy — why exact multi-kind nesting is pushdown work.
// Teaches the linear one-kind counter versus exponential stack memories for
// two or more kinds; the fooling-set annotation explains this is not merely a
// determinization artifact.
import fontsize;
import graph;
settings.outformat = "svg";
size(720, 0);
defaultpen(fontsize(9pt) + rgb("102027"));

pen teal = rgb("00695C") + linewidth(2pt);
pen blue = rgb("1565C0") + linewidth(2pt);
pen violet = rgb("6A1B9A") + linewidth(2pt);
pen amberFill = rgb("FFE0B2");
pen amber = rgb("E65100") + linewidth(1.2pt);
pen slate = rgb("455A64") + linewidth(1pt);
pen grid = rgb("CFD8DC") + linewidth(0.5pt);

real x0 = 1.2;
real y0 = 1.0;
real w = 10.5;
real h = 5.2;
real xmax = 20;
real ymax = 10.2;
real xm(real depth) { return x0 + w * depth / xmax; }
real ym(real exponent) { return y0 + h * exponent / ymax; }

for (int depth = 0; depth <= 20; depth += 5) {
  draw((xm(depth), y0)--(xm(depth), y0 + h), grid);
}
for (int exponent = 0; exponent <= 10; exponent += 2) {
  draw((x0, ym(exponent))--(x0 + w, ym(exponent)), grid);
}
label("0", (xm(0), y0 - 0.30), slate);
label("5", (xm(5), y0 - 0.30), slate);
label("10", (xm(10), y0 - 0.30), slate);
label("15", (xm(15), y0 - 0.30), slate);
label("20", (xm(20), y0 - 0.30), slate);
label("$10^0$", (x0 - 0.45, ym(0)), slate);
label("$10^2$", (x0 - 0.45, ym(2)), slate);
label("$10^4$", (x0 - 0.45, ym(4)), slate);
label("$10^6$", (x0 - 0.45, ym(6)), slate);
label("$10^8$", (x0 - 0.45, ym(8)), slate);
label("$10^{10}$", (x0 - 0.45, ym(10)), slate);
draw((x0, y0)--(x0 + w + 0.25, y0), slate, Arrow(5));
draw((x0, y0)--(x0, y0 + h + 0.25), slate, Arrow(5));
label("maximum nesting depth D", (x0 + w / 2, y0 - 0.72), slate);
label("remembered stack words (log scale)", (x0 - 1.05, y0 + h / 2), rotate(90) * slate);

real stackStates(int kinds, int depth) {
  if (kinds == 1) return depth + 1;
  return (kinds^(depth + 1) - 1) / (kinds - 1);
}
pair[] one;
pair[] two;
pair[] three;
for (int depth = 0; depth <= 20; ++depth) {
  one.push((xm(depth), ym(log10(stackStates(1, depth)))));
  two.push((xm(depth), ym(log10(stackStates(2, depth)))));
  three.push((xm(depth), ym(log10(stackStates(3, depth)))));
}
draw(graph(one), teal);
draw(graph(two), blue);
draw(graph(three), violet);
label("one kind: counter", one[18] + (0.0, 0.32), teal);
label("two kinds", two[15] + (0.30, 0.10), blue);
label("three kinds", three[11] + (0.55, 0.22), violet);

filldraw(box((6.8, 5.45), (12.0, 6.75)), amberFill, amber);
label("Fooling set at depth $D$ has $k^D$ members", (9.4, 6.35), amber);
label("Every member needs a different reversed closer suffix", (9.4, 5.93), amber);
label("Therefore any NFA needs at least $k^D$ states", (9.4, 5.62), amber);

label("Exact multi-kind Dyck recognition crosses from finite-state to pushdown memory",
      (6.45, 7.45), fontsize(12pt) + rgb("00695C"));
