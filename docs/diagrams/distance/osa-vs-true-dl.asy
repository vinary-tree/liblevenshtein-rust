// osa-vs-true-dl.asy — same operation vocabulary, different composition semantics.
// Teaches why OSA reports 3 for CA -> ABC while unrestricted Damerau reports 2,
// and pins the resulting OSA triangle-inequality counterexample.
import fontsize;
settings.outformat = "svg";
size(720, 0);
defaultpen(fontsize(9pt) + rgb("102027"));

pen blueFill = rgb("BBDEFB");
pen blue = rgb("1565C0") + linewidth(1.5pt);
pen greenFill = rgb("C8E6C9");
pen green = rgb("2E7D32") + linewidth(1.5pt);
pen roseFill = rgb("FFCDD2");
pen rose = rgb("C62828") + linewidth(1.5pt);
pen amberFill = rgb("FFE0B2");
pen amber = rgb("E65100") + linewidth(1.2pt);
pen slate = rgb("455A64") + linewidth(1pt);

void grid(pair origin, pen fillPen, pen edgePen) {
  real cell = 0.72;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      pair lo = origin + (col * cell, (2 - row) * cell);
      filldraw(box(lo, lo + (cell, cell)), fillPen + opacity(0.55), edgePen);
    }
  }
}

pair left = (0.9, 3.0);
pair right = (7.0, 3.0);
grid(left, roseFill, rose);
grid(right, greenFill, green);
label("$\epsilon$", left + (-0.32, 1.80), slate);
label("C", left + (-0.32, 1.08), slate);
label("A", left + (-0.32, 0.36), slate);
label("$\epsilon$", left + (0.36, 2.44), slate);
label("A", left + (1.08, 2.44), slate);
label("B", left + (1.80, 2.44), slate);
label("C", left + (2.52, 2.44), slate);
label("$\epsilon$", right + (-0.32, 1.80), slate);
label("C", right + (-0.32, 1.08), slate);
label("A", right + (-0.32, 0.36), slate);
label("$\epsilon$", right + (0.36, 2.44), slate);
label("A", right + (1.08, 2.44), slate);
label("B", right + (1.80, 2.44), slate);
label("C", right + (2.52, 2.44), slate);
label("Optimal string alignment", left + (1.08, 2.94), fontsize(12pt) + rose);
label("$d_{OSA}(CA,ABC)=3$", left + (1.08, -0.38), rose);
label("True Damerau-Levenshtein", right + (1.08, 2.94), fontsize(12pt) + green);
label("$d_{DL}(CA,ABC)=2$", right + (1.08, -0.38), green);

draw(left + (0.36, 1.80)--left + (1.08, 1.08)--left + (1.80, 0.36)--left + (2.52, 0.36),
     rose, Arrow(5));
label("no re-edit inside a transposed pair", left + (1.45, 2.55), rose);

draw(right + (0.36, 1.80)--right + (2.52, 0.36), green, Arrow(6));
label("macro: transpose CA, then insert B", right + (1.55, 2.55), green);

filldraw(box((2.1, 0.45), (11.6, 1.65)), amberFill, amber);
label("OSA triangle counterexample", (3.8, 1.30), amber);
label("d(CA, AC) = 1", (5.8, 1.30), blue);
label("+", (7.05, 1.30), slate);
label("d(AC, ABC) = 1", (8.35, 1.30), blue);
label("<", (9.85, 1.30), rose);
label("d(CA, ABC) = 3", (10.75, 1.30), rose);
label("Same named operations do not imply the same distance function", (6.85, 0.82), slate);

label("Alignment restriction versus unrestricted script composition",
      (6.55, 6.85), fontsize(13pt) + rgb("1565C0"));
