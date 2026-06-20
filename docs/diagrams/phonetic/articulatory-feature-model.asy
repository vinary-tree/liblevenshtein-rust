// articulatory-feature-model.asy — the articulatory feature space that grounds
// phonetic edit costs. Consonants are placed by PLACE of articulation on the
// x-axis (bilabial = 1, alveolar = 2, velar = 3) and by MANNER on the y-axis
// (plosive = 1, nasal = 2). Voiced/voiceless cognate pairs share a cell and are
// nudged apart so their labels do not collide: a VOICELESS sound is an open
// (unfilled) dot, a VOICED sound is a filled dot. A dashed arc between the
// cognate pair p/b annotates the small "feature distance" that a phonetic metric
// assigns to a single voicing change.
// Colour mapping (per the canonical legend — Phonetic engine):
//   marker fill   rgb("F8BBD0")   (voiced dots, filled)
//   marker stroke rgb("AD1457")   (all dots; voiceless dots are stroke-only)
//   axes/frame    neutral slate    rgb("455A64")
// Output: SVG via Asymptote's native driver (no LaTeX; built-in fonts).
import fontsize;
settings.outformat = "svg";
size(280);
defaultpen(fontsize(10pt) + rgb("102027"));

pen phonStroke = rgb("AD1457") + linewidth(1.0pt);
pen phonFill   = rgb("F8BBD0");
pen axisPen    = rgb("455A64") + linewidth(0.9pt);
pen gridPen    = rgb("CFD8DC") + linewidth(0.4pt) + linetype(new real[] {3,3});

// ── light reference grid for the two manner rows / three place columns ──
for (int x = 1; x <= 3; ++x) draw((x, 0.5) -- (x, 2.5), gridPen);
for (int y = 1; y <= 2; ++y) draw((0.5, y) -- (3.5, y), gridPen);

// ── axes (place →, manner →) ──
draw((0.5, 0.5) -- (3.7, 0.5), axisPen, Arrow(4));
draw((0.5, 0.5) -- (0.5, 2.7), axisPen, Arrow(4));
label("place →", (2.1, 0.18), axisPen);
label(rotate(90) * "manner →", (0.16, 1.6), axisPen);

// place (x) tick labels
label("bilabial",  (1, 0.34), fontsize(8pt) + rgb("455A64"));
label("alveolar",  (2, 0.34), fontsize(8pt) + rgb("455A64"));
label("velar",     (3, 0.34), fontsize(8pt) + rgb("455A64"));

// manner (y) tick labels
label("plosive", (0.30, 1), rotate(90) * E, fontsize(8pt) + rgb("455A64"));
label("nasal",   (0.30, 2), rotate(90) * E, fontsize(8pt) + rgb("455A64"));

real dx = 0.16;   // horizontal nudge separating each voiceless/voiced cognate pair
real r  = 0.085;  // marker radius

// voiceless cognate: open (stroke-only) dot, label above-left
void voiceless(pair c, string name) {
  draw(circle(c, r), phonStroke);
  label(name, c + (-0.04, 0.20), phonStroke);
}
// voiced cognate: filled dot, label below-right
void voiced(pair c, string name) {
  filldraw(circle(c, r), phonFill, phonStroke);
  label(name, c + (0.06, -0.21), phonStroke);
}

// ── plosives (manner = 1): p/b bilabial, t/d alveolar, k/g velar ──
pair pP = (1 - dx, 1); pair pB = (1 + dx, 1);
pair pT = (2 - dx, 1); pair pD = (2 + dx, 1);
pair pK = (3 - dx, 1); pair pG = (3 + dx, 1);
voiceless(pP, "p"); voiced(pB, "b");
voiceless(pT, "t"); voiced(pD, "d");
voiceless(pK, "k"); voiced(pG, "g");

// ── nasals (manner = 2): m bilabial, n alveolar (both voiced) ──
pair pM = (1, 2); pair pN = (2, 2);
voiced(pM, "m"); voiced(pN, "n");

// ── feature-distance annotation: dashed arc between the cognates p and b ──
pen distPen = rgb("AD1457") + linewidth(0.9pt) + linetype(new real[] {4,3});
path arc_pb = pP{up} .. (1, 1.34) .. {down}pB;
draw(arc_pb, distPen, Arrow(4));
label("feature distance", (1, 1.52), fontsize(8pt) + rgb("AD1457"));

// ── title ──
label("Articulatory feature space — consonants by place × manner",
      (2.1, 2.92), fontsize(10pt) + rgb("AD1457"));
label("(open = voiceless, filled = voiced)",
      (2.1, 2.66), fontsize(8pt) + rgb("455A64"));
