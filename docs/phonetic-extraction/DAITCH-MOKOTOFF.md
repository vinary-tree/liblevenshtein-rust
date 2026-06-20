# Daitch-Mokotoff Soundex Algorithm Extraction

> **Terminology.** A **phoneme** is a contrastive unit of sound. The IPA symbols used below denote, among others: `ʃ` ("sh"), `tʃ` ("ch"), `dʒ` ("j"), `ʒ` (the "s" in *measure*), and `x` (the voiceless velar fricative, as in Scottish *loch* or Hebrew *chet*). *Place of articulation* is where the vocal tract is constricted; *manner of articulation* is how airflow is shaped; *voicing* is whether the vocal folds vibrate. See [`../GLOSSARY.md`](../GLOSSARY.md) for fuller definitions.

## Algorithm Overview

Daitch-Mokotoff Soundex (D-M Soundex) was developed by Gary Mokotoff and Randy Daitch in 1985 specifically for Jewish and Eastern European surnames. It addresses limitations of American Soundex for these names. Each extracted rule carries a `weight` in `[0, 1]` expressing the residual edit cost of treating the two spellings as equivalent.

## Key Differences from American Soundex

| Feature | American Soundex | Daitch-Mokotoff |
|---------|------------------|-----------------|
| Code Length | 4 characters | 6 digits |
| First Letter | Retained literally | Encoded |
| Multiple Codes | No | Yes (branching) |
| Slavic Sounds | Poor | Excellent |
| Hebrew Sounds | Poor | Good |

## Encoding Rules

D-M Soundex uses a more extensive mapping with context-sensitivity:

### Vowel Codes
| Pattern | Start | Before Vowel | Other |
|---------|-------|--------------|-------|
| A, E, I, O, U, Y | 0 | - | - |
| AI, AJ, AY | 0 | 1 | - |
| AU | 0 | 7 | - |
| EI, EJ, EY | 0 | 1 | - |
| EU | 1 | 1 | - |
| OI, OJ, OY | 0 | 1 | - |

### Consonant Codes
| Pattern | Start | Before Vowel | Other |
|---------|-------|--------------|-------|
| B | 7 | 7 | 7 |
| CH | 5,4 | 5,4 | 5,4 |
| CK | 5,45 | 5,45 | 5,45 |
| CZ, CS, CSZ | 4 | 4 | 4 |
| D | 3 | 3 | 3 |
| DRZ, DRS | 4 | 4 | 4 |
| DS, DSH, DSZ | 4 | 4 | 4 |
| DT | 3 | 3 | 3 |
| DZ, DZH, DZS | 4 | 4 | 4 |
| G | 5 | 5 | 5 |
| H | 5 | 5 | - |
| K | 5 | 5 | 5 |
| KH | 5 | 5 | 5 |
| KS | 54 | 54 | 54 |
| L | 8 | 8 | 8 |
| M | 6 | 6 | 6 |
| MN | 66 | 66 | 66 |
| N | 6 | 6 | 6 |
| NM | 66 | 66 | 66 |
| P, PH, PF | 7 | 7 | 7 |
| R | 9 | 9 | 9 |
| RZ, RS | 94,4 | 94,4 | 94,4 |
| S | 4 | 4 | 4 |
| SCH | 4 | 4 | 4 |
| SH | 4 | 4 | 4 |
| SCHTCH, SCHTSCH | 2 | 4 | 4 |
| SHTCH, SHCH, SHTSH | 2 | 4 | 4 |
| ST, STRZ, STRS, STSH | 2 | 4 | 4 |
| SZCZ, SZCS | 2 | 4 | 4 |
| SZ | 4 | 4 | 4 |
| T | 3 | 3 | 3 |
| TCH, TTCH, TTSCH | 4 | 4 | 4 |
| TH | 3 | 3 | 3 |
| TRZ, TRS | 4 | 4 | 4 |
| TSCH, TSH | 4 | 4 | 4 |
| TS, TSZ, TZ, TTZ | 4 | 4 | 4 |
| V | 7 | 7 | 7 |
| W | 7 | 7 | 7 |
| X | 5,54 | 54 | 54 |
| Z | 4 | 4 | 4 |
| ZDZ, ZDZH, ZHDZH | 2 | 4 | 4 |
| ZD | 2 | 4 | 4 |
| ZH, ZS | 4 | 4 | 4 |

## Extracted Rules

### Polish Consonant Clusters
```llev
[id: 3000, name: "szcz dm", weight: 0.10, group: dm_clusters, ipa: "/ʃtʃ/"]
szcz -> ʃtʃ;

[id: 3001, name: "strz dm", weight: 0.10, group: dm_clusters, ipa: "/ʃtʃ/"]
strz -> ʃtʃ;

[id: 3002, name: "strs dm", weight: 0.10, group: dm_clusters]
strs -> ʃtʃ;

[id: 3003, name: "stsh dm", weight: 0.10, group: dm_clusters]
stsh -> ʃtʃ;

[id: 3010, name: "cz dm", weight: 0.10, group: dm_cz, ipa: "/tʃ/"]
cz -> tʃ;

[id: 3011, name: "cs dm", weight: 0.10, group: dm_cz]
cs -> tʃ;

[id: 3012, name: "csz dm", weight: 0.10, group: dm_cz]
csz -> tʃ;

[id: 3015, name: "sz dm", weight: 0.10, group: dm_sz, ipa: "/ʃ/"]
sz -> ʃ;
```

### DRZ/DRS/DZ Patterns
```llev
[id: 3020, name: "drz dm", weight: 0.10, group: dm_drz, ipa: "/dʒ/"]
drz -> dʒ;

[id: 3021, name: "drs dm", weight: 0.10, group: dm_drz]
drs -> dʒ;

[id: 3022, name: "dz dm", weight: 0.10, group: dm_dz, ipa: "/dz/"]
dz -> dz;

[id: 3023, name: "dzh dm", weight: 0.10, group: dm_dz]
dzh -> dʒ;

[id: 3024, name: "dzs dm", weight: 0.10, group: dm_dz]
dzs -> dʒ;
```

### ZDZ Patterns (Word-initial)
```llev
[id: 3030, name: "zdz initial dm", weight: 0.10, group: dm_zdz]
zdz -> ʒdʒ / #_;

[id: 3031, name: "zdzh initial dm", weight: 0.10, group: dm_zdz]
zdzh -> ʒdʒ / #_;

[id: 3032, name: "zhdzh initial dm", weight: 0.10, group: dm_zdz]
zhdzh -> ʒdʒ / #_;
```

### TCH/TSH Patterns
```llev
[id: 3040, name: "tch dm", weight: 0.10, group: dm_tch, ipa: "/tʃ/"]
tch -> tʃ;

[id: 3041, name: "ttch dm", weight: 0.10, group: dm_tch]
ttch -> tʃ;

[id: 3042, name: "ttsch dm", weight: 0.10, group: dm_tch]
ttsch -> tʃ;

[id: 3043, name: "tsch dm", weight: 0.10, group: dm_tch]
tsch -> tʃ;

[id: 3044, name: "tsh dm", weight: 0.10, group: dm_tch]
tsh -> tʃ;
```

### TRZ/TRS Patterns
```llev
[id: 3050, name: "trz dm", weight: 0.10, group: dm_trz, ipa: "/tʃ/"]
trz -> tʃ;

[id: 3051, name: "trs dm", weight: 0.10, group: dm_trz]
trs -> tʃ;
```

### CH Variants
```llev
[id: 3055, name: "ch to kh dm", weight: 0.15, group: dm_ch, ipa: "/x/"]
ch -> x;  // Hebrew/Yiddish sound

[id: 3056, name: "ch to tsh dm", weight: 0.15, group: dm_ch, ipa: "/tʃ/"]
ch -> tʃ;  // Slavic variant
```

### RZ/RS Patterns
```llev
[id: 3060, name: "rz dm", weight: 0.10, group: dm_rz, ipa: "/ʒ/"]
rz -> ʒ;

[id: 3061, name: "rs dm", weight: 0.15, group: dm_rz]
rs -> ʒ;  // In Polish context
```

### Diphthongs
```llev
[id: 3070, name: "ai dm", weight: 0.10, group: dm_diphthong]
ai -> aj;

[id: 3071, name: "aj dm", weight: 0.10, group: dm_diphthong]
aj -> aj;

[id: 3072, name: "ay dm", weight: 0.10, group: dm_diphthong]
ay -> aj;

[id: 3073, name: "ei dm", weight: 0.10, group: dm_diphthong]
ei -> aj;

[id: 3074, name: "ej dm", weight: 0.10, group: dm_diphthong]
ej -> aj;

[id: 3075, name: "ey dm", weight: 0.10, group: dm_diphthong]
ey -> aj;

[id: 3076, name: "oi dm", weight: 0.10, group: dm_diphthong]
oi -> oj;

[id: 3077, name: "oj dm", weight: 0.10, group: dm_diphthong]
oj -> oj;

[id: 3078, name: "oy dm", weight: 0.10, group: dm_diphthong]
oy -> oj;

[id: 3079, name: "au dm", weight: 0.10, group: dm_diphthong]
au -> au;

[id: 3080, name: "eu dm", weight: 0.10, group: dm_diphthong]
eu -> oj;  // Yiddish pronunciation
```

## Common Jewish Name Elements

```llev
// Gold- element
[id: 3090, name: "gold prefix dm", weight: 0.05, group: dm_element]
gold -> gold / #_;

// -berg element
[id: 3091, name: "berg suffix dm", weight: 0.05, group: dm_element]
berg -> berg / _#;

// -stein element
[id: 3092, name: "stein suffix dm", weight: 0.05, group: dm_element]
stein -> ʃtajn / _#;

// Rosen- element
[id: 3093, name: "rosen prefix dm", weight: 0.05, group: dm_element]
rosen -> rozen / #_;

// -witz/-vitz element
[id: 3094, name: "witz suffix dm", weight: 0.05, group: dm_element]
witz -> vits / _#;

// -baum element
[id: 3095, name: "baum suffix dm", weight: 0.05, group: dm_element]
baum -> boim / _#;

// -feld element
[id: 3096, name: "feld suffix dm", weight: 0.05, group: dm_element]
feld -> feld / _#;
```

## Branching Codes

D-M Soundex produces multiple codes when a pattern can be interpreted multiple ways. In LLev, this maps to non-deterministic rules with varying weights:

```llev
// CH can be /x/ (Hebrew) or /tʃ/ (Slavic)
[id: 3055, weight: 0.15]
ch -> x;

[id: 3056, weight: 0.20]
ch -> tʃ;
```

Both outputs are generated; the weight difference prioritizes the more common interpretation while still matching the alternate.

## Use Cases

D-M Soundex excels at matching:
- Jewish surnames (Ashkenazic, Sephardic)
- Polish surnames
- Russian surnames
- Ukrainian surnames
- Hungarian surnames
- Romanian surnames

## References

1. Mokotoff, Gary & Daitch, Randy (1985). *Daitch-Mokotoff Soundex System*. Avotaynu, Bergenfield, NJ. (No DOI; canonical source of the D-M Soundex algorithm.)
2. Mokotoff, Gary (1997). "Soundexing and Genealogy". *Avotaynu* 13(3). (No DOI.)
3. Avotaynu, "Daitch-Mokotoff Soundex Coding". https://www.avotaynu.com/soundex.htm

---

[← Documentation Index](../README.md)
