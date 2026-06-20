# Cologne Phonetic (Kölner Phonetik) Algorithm Extraction

> **Terminology.** A **phoneme** is a contrastive unit of sound. The IPA symbols used below denote: `ʃ` the "sh" sound (a voiceless post-alveolar fricative) and `ts` the affricate in German *z*. *Place of articulation* is where the vocal tract is constricted; *manner of articulation* is how airflow is shaped; *voicing* is whether the vocal folds vibrate. See [`../GLOSSARY.md`](../GLOSSARY.md) for fuller definitions.

## Algorithm Overview

The Cologne Phonetic (Kölner Phonetik) algorithm was developed by Hans Joachim Postel in 1969 for indexing German names. It is more suitable for German than Soundex because it accounts for German pronunciation patterns. Each extracted rule carries a `weight` in `[0, 1]` expressing the residual edit cost of treating the two spellings as equivalent.

## Encoding Table

| Letter | Code | Context |
|--------|------|---------|
| A, E, I, O, U | 0 | - |
| B | 1 | - |
| P | 1 | not before H |
| D, T | 2 | not before C, S, Z |
| F, V, W | 3 | - |
| P | 3 | before H |
| G, K, Q | 4 | - |
| C | 4 | initial before A, H, K, L, O, Q, R, U, X |
| C | 4 | before A, H, K, O, Q, U, X (not after S, Z) |
| X | 48 | not after C, K, Q |
| L | 5 | - |
| M, N | 6 | - |
| R | 7 | - |
| S, Z | 8 | - |
| C | 8 | after S, Z |
| C | 8 | initial (not before A, H, K, L, O, Q, R, U, X) |
| D, T | 8 | before C, S, Z |
| X | 8 | after C, K, Q |

## Extracted Rules

### C Context Rules
```llev
[id: 2400, name: "c initial before ahkloqrux cologne", weight: 0.10, group: cologne_c]
c -> k / #_[ahkloqrux];  // Cologne, Clara, Christ

[id: 2401, name: "c after sz cologne", weight: 0.10, group: cologne_c]
c -> s / [sz]_;  // SC, ZC combinations

[id: 2402, name: "c before ei cologne", weight: 0.10, group: cologne_c]
c -> ts / _[ei];  // German soft c (Celsius, Cäsar)

[id: 2403, name: "c internal before ahkoqux cologne", weight: 0.10, group: cologne_c]
c -> k / [^sz]_[ahkoqux];  // internal hard c
```

### D/T before sibilants
```llev
[id: 2410, name: "dt before c cologne", weight: 0.10, group: cologne_dt]
dt -> ts / _c;  // Stadt→Statc

[id: 2411, name: "ds cologne", weight: 0.10, group: cologne_dt]
ds -> ts;  // Landsmann

[id: 2412, name: "dz cologne", weight: 0.10, group: cologne_dt]
dz -> ts;  // variants

[id: 2413, name: "ts cologne", weight: 0.10, group: cologne_dt]
ts -> ts;  // normalize
```

### X decomposition
```llev
[id: 2420, name: "x after ckq cologne", weight: 0.10, group: cologne_x]
x -> s / [ckq]_;  // CKX, KX, QX → same code

[id: 2421, name: "x standalone cologne", weight: 0.10, group: cologne_x]
x -> ks / [^ckq]_;  // AX, EX → 048
```

### PH handling
```llev
[id: 2425, name: "ph to f cologne", weight: 0.10, group: cologne_ph]
ph -> f;  // Philipp, Stephan
```

### ST/SP preservation
```llev
[id: 2430, name: "st initial cologne", weight: 0.05, group: cologne_clusters]
st -> ʃt / #_;  // German initial st → ʃt

[id: 2431, name: "sp initial cologne", weight: 0.05, group: cologne_clusters]
sp -> ʃp / #_;  // German initial sp → ʃp
```

### V/W equivalence
```llev
[id: 2435, name: "v to f cologne", weight: 0.15, group: cologne_vw]
v -> f;  // Vogel (German V)

[id: 2436, name: "w to v cologne", weight: 0.15, group: cologne_vw]
w -> v;  // Wagner (German W)
```

## German-Specific Patterns

### Umlauts
```llev
[id: 2440, name: "ä to ae cologne", weight: 0.10, group: cologne_umlaut]
ä -> e;  // Bär, Mädchen

[id: 2441, name: "ö to oe cologne", weight: 0.10, group: cologne_umlaut]
ö -> o;  // König, schön

[id: 2442, name: "ü to ue cologne", weight: 0.10, group: cologne_umlaut]
ü -> u;  // München, für
```

### ß handling
```llev
[id: 2445, name: "ß to ss cologne", weight: 0.05, group: cologne_eszett]
ß -> ss;  // Straße, groß
```

## Comparison with Soundex

| Feature | Soundex | Cologne Phonetic |
|---------|---------|------------------|
| Target Language | English | German |
| Code Length | Fixed (4) | Variable |
| Vowel Handling | Removed | Code 0 |
| Context Sensitivity | None | Yes (C, D, T, X) |
| Initial Letter | Retained | Encoded |

## References

1. Postel, Hans Joachim (1969). "Die Kölner Phonetik. Ein Verfahren zur Identifizierung von Personennamen auf der Grundlage der Gestaltanalyse". *IBM-Nachrichten* 19, pp. 925–931. (No DOI; canonical source of the Kölner Phonetik algorithm.)

---

[← Documentation Index](../README.md)
