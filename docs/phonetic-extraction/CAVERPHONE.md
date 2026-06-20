# Caverphone Algorithm Extraction

> **Terminology.** A **phoneme** is a contrastive unit of sound. The IPA symbols used below denote: `ʃ` the "sh" sound (a voiceless post-alveolar fricative), `ŋ` the velar nasal ("ng" in *sing*), `ʍ` the voiceless "wh", and `ə` the mid-central vowel (schwa). *Place of articulation* is where the vocal tract is constricted; *manner of articulation* is how airflow is shaped; *voicing* is whether the vocal folds vibrate. See [`../GLOSSARY.md`](../GLOSSARY.md) for fuller definitions.

## Algorithm Overview

Caverphone was developed by David Hood at the University of Otago in New Zealand in 2002. It is specifically designed for matching New Zealand English names, particularly surnames derived from Māori and British sources. Each extracted rule carries a `weight` in `[0, 1]` expressing the residual edit cost of treating the two spellings as equivalent.

## Versions

- **Caverphone 1.0** (2002): Original 6-character code
- **Caverphone 2.0** (2004): Extended 10-character code with improved accuracy

## Key Features

1. Designed for NZ English pronunciation patterns
2. Handles Maori name conventions
3. Extensive vowel simplification
4. Specific consonant cluster handling

## Algorithm Steps (Caverphone 2.0)

1. Convert to lowercase
2. Remove anything not A-Z
3. Remove final 'e'
4. Apply initial transformations for common patterns
5. Apply main substitution rules
6. Remove vowels (except initial)
7. Pad/truncate to 10 characters

## Extracted Rules

### Initial Consonant Clusters
```llev
[id: 2700, name: "cough initial caverphone", weight: 0.10, group: caverphone_initial]
cough -> cof2 / #_;

[id: 2701, name: "rough initial caverphone", weight: 0.10, group: caverphone_initial]
rough -> rof2 / #_;

[id: 2702, name: "tough initial caverphone", weight: 0.10, group: caverphone_initial]
tough -> tof2 / #_;

[id: 2703, name: "enough initial caverphone", weight: 0.10, group: caverphone_initial]
enough -> enof2 / #_;

[id: 2704, name: "gn initial caverphone", weight: 0.10, group: caverphone_initial]
gn -> 2n / #_;

[id: 2705, name: "mb final caverphone", weight: 0.10, group: caverphone_final]
mb -> m2 / _#;
```

### GH Patterns (NZ specific)
```llev
[id: 2710, name: "gh to f caverphone", weight: 0.15, group: caverphone_gh]
gh -> f / [aou]_;

[id: 2711, name: "gh silent caverphone", weight: 0.15, group: caverphone_gh]
gh ->  / [ei]_;  // night, weight
```

### TCH/CK Patterns
```llev
[id: 2715, name: "tch to 2 caverphone", weight: 0.10, group: caverphone_tch]
tch -> 2;

[id: 2716, name: "ck to k caverphone", weight: 0.10, group: caverphone_ck]
ck -> k;
```

### C Patterns
```llev
[id: 2720, name: "c to s caverphone", weight: 0.10, group: caverphone_c]
c -> s / _[eiy];  // soft c

[id: 2721, name: "c to k caverphone", weight: 0.10, group: caverphone_c]
c -> k / _[aou];  // hard c
```

### DG Pattern
```llev
[id: 2725, name: "dg to 2 caverphone", weight: 0.10, group: caverphone_dg]
dg -> 2 / _[eiy];  // edge, badge
```

### Q/QU Patterns
```llev
[id: 2730, name: "qu to kw caverphone", weight: 0.10, group: caverphone_qu]
qu -> kw;

[id: 2731, name: "q to k caverphone", weight: 0.10, group: caverphone_qu]
q -> k;
```

### TI/SI Patterns (before vowels)
```llev
[id: 2735, name: "tio to sio caverphone", weight: 0.15, group: caverphone_tion]
tio -> ʃo;  // nation

[id: 2736, name: "tia to sia caverphone", weight: 0.15, group: caverphone_tion]
tia -> ʃa;  // spatial

[id: 2737, name: "sio to so caverphone", weight: 0.15, group: caverphone_sion]
sio -> ʃo;  // vision

[id: 2738, name: "sia to sa caverphone", weight: 0.15, group: caverphone_sion]
sia -> ʃa;  // Asia
```

### Double Consonant Simplification
```llev
[id: 2740, name: "double t caverphone", weight: 0.10, group: caverphone_double]
tt -> t;

[id: 2741, name: "double d caverphone", weight: 0.10, group: caverphone_double]
dd -> d;

[id: 2742, name: "double l caverphone", weight: 0.10, group: caverphone_double]
ll -> l;

[id: 2743, name: "double n caverphone", weight: 0.10, group: caverphone_double]
nn -> n;

[id: 2744, name: "double m caverphone", weight: 0.10, group: caverphone_double]
mm -> m;

[id: 2745, name: "double s caverphone", weight: 0.10, group: caverphone_double]
ss -> s;

[id: 2746, name: "double r caverphone", weight: 0.10, group: caverphone_double]
rr -> r;

[id: 2747, name: "double f caverphone", weight: 0.10, group: caverphone_double]
ff -> f;

[id: 2748, name: "double p caverphone", weight: 0.10, group: caverphone_double]
pp -> p;

[id: 2749, name: "double b caverphone", weight: 0.10, group: caverphone_double]
bb -> b;
```

### W Handling
```llev
[id: 2755, name: "wh to w caverphone", weight: 0.15, group: caverphone_wh]
wh -> w / #_;  // NZ pronunciation

[id: 2756, name: "w to vowel caverphone", weight: 0.20, group: caverphone_wh]
w ->  / [aeiou]_[aeiou];  // intervocalic
```

### Final E Removal
```llev
[id: 2760, name: "final e caverphone", weight: 0.10, group: caverphone_final]
e ->  / _#;  // Silent final e
```

### NZ-Specific Vowel Patterns
```llev
[id: 2765, name: "ou to u caverphone", weight: 0.15, group: caverphone_vowel]
ou -> u;

[id: 2766, name: "oo to u caverphone", weight: 0.15, group: caverphone_vowel]
oo -> u;

[id: 2767, name: "ea to e caverphone", weight: 0.15, group: caverphone_vowel]
ea -> e;

[id: 2768, name: "ai to e caverphone", weight: 0.15, group: caverphone_vowel]
ai -> e;

[id: 2769, name: "ay to e caverphone", weight: 0.15, group: caverphone_vowel]
ay -> e;
```

### Maori Name Patterns
```llev
[id: 2775, name: "wh to f maori caverphone", weight: 0.20, group: caverphone_maori]
wh -> f;  // Traditional Maori

[id: 2776, name: "ng to ng maori caverphone", weight: 0.05, group: caverphone_maori]
ng -> ŋ;  // Maori velar nasal
```

## NZ English vs British English

Key pronunciation differences captured:

| Pattern | British | NZ | Rule |
|---------|---------|-----|------|
| final 'er' | /ə/ | /ə/ | Same |
| 'wh' | /w/ | /ʍ/ or /f/ | 2755, 2775 |
| short 'i' | /ɪ/ | /ə/ | flattening |
| 'air' | /eə/ | /e/ | merger |

## Use Cases

Caverphone is particularly effective for:
- New Zealand electoral rolls
- New Zealand genealogical records
- Matching names with Maori/British dual heritage
- NZ healthcare patient matching

## References

1. Hood, David (2002). "Caverphone". *University of Otago Technical Report*, Caversham Project Occasional Paper, Dunedin, NZ. (No DOI; canonical source of the Caverphone algorithm.)
2. Hood, David (2004). "Caverphone Revisited" (Caverphone 2.0). *University of Otago Technical Report*, Dunedin, NZ. (No DOI.)

---

[← Documentation Index](../README.md)
