// interval-relaxation.asy — the geometric fact behind ERP and Fréchet pruning.
// A concrete target y is unknown inside the quantization bin [lo, hi]. The
// closest point to scalar v is clamp(v, lo, hi), so the colored segment is both
// the exact box minimum and a lower bound for every concrete |v-y|. The right
// panel maps this fact to ERP's match, delete, and insert leaves. Fréchet uses
// the same match leaf under bottleneck max; point bins collapse either
// relaxation back to the scalar recurrence.
//
// Color legend (time-series family):
//   interval / admissible set  teal     rgb("B2DFDB") / rgb("00695C")
//   exact minimum             amber    rgb("FFE0B2") / rgb("E65100")
//   concrete realization     blue     rgb("BBDEFB") / rgb("1565C0")
//   exact fixed leaf          green    rgb("C8E6C9") / rgb("2E7D32")
//   neutral scaffolding       slate    rgb("455A64")
import fontsize;
settings.outformat = "svg";
size(620, 0);
defaultpen(fontsize(9pt) + rgb("102027"));

pen tealFill = rgb("B2DFDB");
pen teal = rgb("00695C") + linewidth(1.4pt);
pen amberFill = rgb("FFE0B2");
pen amber = rgb("E65100") + linewidth(1.4pt);
pen blueFill = rgb("BBDEFB");
pen blue = rgb("1565C0") + linewidth(1.2pt);
pen greenFill = rgb("C8E6C9");
pen green = rgb("2E7D32") + linewidth(1.2pt);
pen slate = rgb("455A64") + linewidth(1.0pt);

// Left panel: scalar-to-interval geometry.
real axisY = 2.8;
real v = 0.8;
real lo = 2.4;
real hi = 5.2;
real y = 4.3;
draw((0.2, axisY)--(6.0, axisY), slate, Arrows(4));
filldraw(box((lo, axisY - 0.24), (hi, axisY + 0.24)), tealFill, teal);
label("quantization bin $[\\ell,h]$", ((lo + hi) / 2, axisY + 0.55), teal);

filldraw(circle((v, axisY), 0.10), amberFill, amber);
label("query scalar $v$", (v - 0.34, axisY - 0.48), amber);
filldraw(circle((lo, axisY), 0.10), amberFill, amber);
label("$clamp(v,\\ell,h)=\\ell$", (lo + 0.20, axisY - 0.72), amber);
filldraw(circle((y, axisY), 0.09), blueFill, blue);
label("concrete $y$", (y, axisY - 0.48), blue);

draw((v, axisY + 0.16)--(lo, axisY + 0.16), amber, Arrows(4));
label("exact box minimum", ((v + lo) / 2, axisY + 0.43), amber);
draw((v, axisY - 0.16)--(y, axisY - 0.16), blue, Arrows(4));
label("every concrete distance is no smaller", ((v + y) / 2, axisY - 1.02), blue);

label("$d(v,[\\ell,h]) \\le |v-y|$ for every $y\\in[\\ell,h]$",
      (3.1, 1.15), fontsize(10pt) + rgb("102027"));

// Right panel: the three ERP recurrence leaves.
real px = 7.1;
real py = 4.7;
real w = 5.9;
real h = 0.72;

void leaf(real top, string name, string scalarCost, string relaxedCost,
          pen fillPen, pen edgePen) {
  filldraw(box((px, top - h), (px + w, top)), fillPen, edgePen);
  label(name, (px + 0.70, top - h / 2), edgePen);
  label(scalarCost, (px + 2.05, top - h / 2), slate);
  label("→", (px + 3.08, top - h / 2), slate);
  label(relaxedCost, (px + 4.55, top - h / 2), fontsize(8pt) + edgePen);
}

label("ERP leaf relaxation for target bin $B_j$", (px + w / 2, py + 0.48),
      fontsize(11pt) + rgb("102027"));
leaf(py, "match", "$|x_i-y_j|$", "$d(x_i,B_j)$", tealFill, teal);
leaf(py - 1.05, "delete", "$|x_i-g|$", "exact", greenFill, green);
leaf(py - 2.10, "insert", "$|y_j-g|$", "$d(g,B_j)$", amberFill, amber);

filldraw(box((px, 0.85), (px + w, 1.55)), rgb("ECEFF1"), slate);
label("point bin $B_j=[y_j,y_j]$  →  scalar DP exactly",
      (px + w / 2, 1.20), slate);

filldraw(box((px, -0.18), (px + w, 0.57)), rgb("E8EAF6"), rgb("5E35B1") + linewidth(1.2pt));
label("Fréchet: use only $d(x_i,B_j)$, then compose with $\min/\max$",
      (px + w / 2, 0.20), fontsize(8pt) + rgb("5E35B1"));

label("Exact interval leaves compose into admissible ERP and Fréchet columns",
      (6.2, 5.75), fontsize(12pt) + rgb("00695C"));
