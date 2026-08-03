// prune-duality.asy — min-plus and max-plus branch-and-bound are order duals.
// Teaches the shared traversal skeleton and the opposite rejection inequalities.
import fontsize;
settings.outformat = "svg";
size(720, 0);
defaultpen(fontsize(9pt) + rgb("102027"));

pen blueFill = rgb("BBDEFB");
pen blue = rgb("1565C0") + linewidth(1.5pt);
pen skyFill = rgb("B3E5FC");
pen sky = rgb("0277BD") + linewidth(1.5pt);
pen amberFill = rgb("FFE0B2");
pen amber = rgb("E65100") + linewidth(1.3pt);
pen greenFill = rgb("C8E6C9");
pen green = rgb("2E7D32") + linewidth(1.3pt);
pen roseFill = rgb("FFCDD2");
pen rose = rgb("C62828") + linewidth(1.3pt);
pen slate = rgb("455A64") + linewidth(1pt);

void node(pair center, pen fillPen, pen edgePen) {
  filldraw(box(center - (1.55, 0.38), center + (1.55, 0.38)), fillPen, edgePen);
}

label("Min-plus bounded distance", (3.3, 6.4), fontsize(12pt) + blue);
node((3.3, 5.45), blueFill, blue);
node((3.3, 4.15), amberFill, amber);
node((1.55, 2.65), greenFill, green);
node((5.05, 2.65), roseFill, rose);
label("dictionary prefix p", (3.3, 5.45), blue);
label("cost(p) + lower bound", (3.3, 4.15), amber);
label("keep / descend", (1.55, 2.65), green);
label("prune subtree", (5.05, 2.65), rose);
draw((3.3, 5.05)--(3.3, 4.55), slate, Arrow(5));
draw((3.3, 3.77)--(1.55, 3.05), green, Arrow(5));
draw((3.3, 3.77)--(5.05, 3.05), rose, Arrow(5));
label("$\leq$ cutoff $\tau$", (1.95, 3.55), green);
label("$>$ cutoff $\tau$", (4.72, 3.55), rose);

label("Max-plus scored matching", (10.3, 6.4), fontsize(12pt) + sky);
node((10.3, 5.45), skyFill, sky);
node((10.3, 4.15), amberFill, amber);
node((8.55, 2.65), greenFill, green);
node((12.05, 2.65), roseFill, rose);
label("dictionary prefix p", (10.3, 5.45), sky);
label("score(p) + upper bound", (10.3, 4.15), amber);
label("keep / descend", (8.55, 2.65), green);
label("prune subtree", (12.05, 2.65), rose);
draw((10.3, 5.05)--(10.3, 4.55), slate, Arrow(5));
draw((10.3, 3.77)--(8.55, 3.05), green, Arrow(5));
draw((10.3, 3.77)--(12.05, 3.05), rose, Arrow(5));
label("$\geq$ kth score $\sigma_k$", (8.95, 3.55), green);
label("$<$ kth score $\sigma_k$", (11.75, 3.55), rose);

draw((6.6, 5.75)--(7.0, 5.75), slate, Arrows(5));
draw((6.6, 4.15)--(7.0, 4.15), slate, Arrows(5));
label("same DFS skeleton", (6.8, 6.05), slate);
label("dual order", (6.8, 4.45), slate);

filldraw(box((3.1, 0.55), (10.5, 1.55)), rgb("ECEFF1"), slate);
label("liblevenshtein supplies balanced structural DFS", (6.8, 1.28), slate);
label("the measure supplies the lawful lower or upper bound", (6.8, 0.86), slate);

label("Pruning duality: identical control flow, opposite admissibility direction",
      (6.8, 7.2), fontsize(13pt) + rgb("455A64"));
