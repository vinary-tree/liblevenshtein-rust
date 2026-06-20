# NYSIIS Algorithm Extraction

> **Terminology.** A **phoneme** is a contrastive unit of sound; a **phonetic algorithm** maps spelling to an approximation of pronunciation so that sound-alike strings collide. *Place of articulation* (where the vocal tract is constricted), *manner of articulation* (how airflow is shaped), and *voicing* (whether the vocal folds vibrate) are the dimensions that distinguish consonants. See [`../GLOSSARY.md`](../GLOSSARY.md) for fuller definitions.

## Algorithm Overview

NYSIIS (New York State Identification and Intelligence System) was published by Robert L. Taft for the New York State Identification and Intelligence System in 1970. It was designed for matching American names, particularly for law enforcement and social services applications. Each extracted rule carries a `weight` in `[0, 1]` expressing the residual edit cost of treating the two spellings as equivalent.

## Key Features

1. Designed specifically for American name patterns
2. Retains more vowel information than Soundex
3. Special handling of common American name prefixes
4. Focus on phonetic accuracy over code brevity

## Algorithm Steps

1. Translate first characters of name:
   - MAC → MCC
   - KN → N
   - K → C
   - PH, PF → FF
   - SCH → SSS
2. Translate last characters:
   - EE, IE → Y
   - DT, RT, RD, NT, ND → D
3. Translate remaining characters with context rules
4. Remove trailing S and A
5. Remove trailing AY, replace with Y
6. Collapse duplicate characters
7. Optionally truncate to fixed length

## Extracted Rules

### Initial Prefix Transformations
```llev
[id: 2800, name: "mac to mcc nysiis", weight: 0.10, group: nysiis_prefix]
mac -> mcc / #_;  // MacDonald → McCDonald

[id: 2801, name: "mc to mcc nysiis", weight: 0.10, group: nysiis_prefix]
mc -> mcc / #_;  // McDonald → McCDonald

[id: 2802, name: "kn to n nysiis", weight: 0.10, group: nysiis_prefix]
kn -> n / #_;  // Knight → Night

[id: 2803, name: "k to c nysiis", weight: 0.10, group: nysiis_prefix]
k -> c / #_;  // Katz → Catz

[id: 2804, name: "pf to ff nysiis", weight: 0.10, group: nysiis_prefix]
pf -> ff / #_;  // Pfeiffer → FFeiffer

[id: 2805, name: "ph to ff nysiis", weight: 0.10, group: nysiis_prefix]
ph -> ff / #_;  // Phelps → FFelps

[id: 2806, name: "sch to sss nysiis", weight: 0.10, group: nysiis_prefix]
sch -> sss / #_;  // Schmidt → SSShmidt
```

### Final Suffix Transformations
```llev
[id: 2810, name: "ee final to y nysiis", weight: 0.10, group: nysiis_suffix]
ee -> y / _#;  // McGee → McGy

[id: 2811, name: "ie final to y nysiis", weight: 0.10, group: nysiis_suffix]
ie -> y / _#;  // Christie → Christy

[id: 2812, name: "dt final to d nysiis", weight: 0.10, group: nysiis_suffix]
dt -> d / _#;  // Schmidt → Schmid

[id: 2813, name: "rt final to d nysiis", weight: 0.10, group: nysiis_suffix]
rt -> d / _#;  // Hart → Hard

[id: 2814, name: "rd final to d nysiis", weight: 0.10, group: nysiis_suffix]
rd -> d / _#;  // Howard → Howad

[id: 2815, name: "nt final to d nysiis", weight: 0.10, group: nysiis_suffix]
nt -> d / _#;  // Grant → Grand

[id: 2816, name: "nd final to d nysiis", weight: 0.10, group: nysiis_suffix]
nd -> d / _#;  // Roland → Rolad
```

### EV to AF Transformation
```llev
[id: 2820, name: "ev to af nysiis", weight: 0.15, group: nysiis_vowel]
ev -> af;  // Steven → Stafan
```

### K/Q Transformations
```llev
[id: 2825, name: "k to c nysiis internal", weight: 0.10, group: nysiis_k]
k -> c;  // Uniform K → C

[id: 2826, name: "q to g nysiis", weight: 0.15, group: nysiis_q]
q -> g;  // Quentin → Guentin
```

### SCH Internal
```llev
[id: 2830, name: "sch to s nysiis", weight: 0.15, group: nysiis_sch]
sch -> s;  // Internal sch simplification
```

### PH Handling
```llev
[id: 2835, name: "ph to f nysiis", weight: 0.10, group: nysiis_ph]
ph -> f;  // Phonetic standard
```

### H Handling
```llev
[id: 2840, name: "h after vowel nysiis", weight: 0.15, group: nysiis_h]
h ->  / [aeiou]_;  // Drop H after vowel

[id: 2841, name: "h before consonant nysiis", weight: 0.15, group: nysiis_h]
h ->  / _[^aeiou];  // Drop H before consonant
```

### W Handling
```llev
[id: 2845, name: "w after vowel nysiis", weight: 0.15, group: nysiis_w]
w ->  / [aeiou]_;  // Vowel absorbs W
```

### AW to A
```llev
[id: 2850, name: "aw to a nysiis", weight: 0.10, group: nysiis_aw]
aw -> a;  // Hawkins → Hakins
```

### M/N Before Consonant
```llev
[id: 2855, name: "m to n before consonant nysiis", weight: 0.15, group: nysiis_mn]
m -> n / _[^aeiou];  // Thompson → Thonpson
```

### Vowel Transformations
```llev
[id: 2860, name: "a to vowel nysiis", weight: 0.20, group: nysiis_vowel]
a -> a;  // Keep A

[id: 2861, name: "e to vowel nysiis", weight: 0.20, group: nysiis_vowel]
e -> a;  // E → A

[id: 2862, name: "i to vowel nysiis", weight: 0.20, group: nysiis_vowel]
i -> a;  // I → A

[id: 2863, name: "o to vowel nysiis", weight: 0.20, group: nysiis_vowel]
o -> a;  // O → A

[id: 2864, name: "u to vowel nysiis", weight: 0.20, group: nysiis_vowel]
u -> a;  // U → A
```

### Double Letter Simplification
```llev
[id: 2870, name: "double letter nysiis", weight: 0.10, group: nysiis_double]
([a-z])\1 -> \1;  // Any doubled letter → single
```

### SH/SCH to S
```llev
[id: 2875, name: "sh to s nysiis", weight: 0.10, group: nysiis_sh]
sh -> s;  // Simplified

[id: 2876, name: "sch to s internal nysiis", weight: 0.10, group: nysiis_sh]
sch -> s;  // Internal position
```

### Z to S
```llev
[id: 2880, name: "z to s nysiis", weight: 0.10, group: nysiis_z]
z -> s;  // Uniform sibilant
```

### GHT to GT
```llev
[id: 2885, name: "ght to gt nysiis", weight: 0.10, group: nysiis_ght]
ght -> gt;  // Knight → Knigt
```

### DG to G
```llev
[id: 2890, name: "dg to g nysiis", weight: 0.10, group: nysiis_dg]
dg -> g;  // Badge → Bage
```

## Common American Name Patterns

```llev
// -son endings
[id: 2895, name: "son ending nysiis", weight: 0.05, group: nysiis_patronymic]
son -> san / _#;

// -sen endings (Scandinavian)
[id: 2896, name: "sen ending nysiis", weight: 0.10, group: nysiis_patronymic]
sen -> san / _#;

// -stein endings (Germanic)
[id: 2897, name: "stein ending nysiis", weight: 0.10, group: nysiis_patronymic]
stein -> stan / _#;

// -ski/-sky endings (Slavic)
[id: 2898, name: "ski ending nysiis", weight: 0.10, group: nysiis_patronymic]
ski -> scy / _#;

[id: 2899, name: "sky ending nysiis", weight: 0.10, group: nysiis_patronymic]
sky -> scy / _#;
```

## Comparison with Soundex

| Feature | Soundex | NYSIIS |
|---------|---------|--------|
| Output Length | 4 (fixed) | Variable (typically 6) |
| First Character | Retained | Encoded |
| Vowel Handling | Removed | Normalized to 'A' |
| Prefix Rules | None | MAC, KN, PH, etc. |
| Suffix Rules | None | EE→Y, DT→D, etc. |
| Target Population | General English | American names |

## Use Cases

NYSIIS is particularly effective for:
- Law enforcement databases
- Social services record matching
- Immigration record matching
- Hospital patient matching
- American genealogical research

## References

1. Taft, Robert L. (1970). "Name Search Techniques". *New York State Identification and Intelligence System*, Special Report No. 1, Albany, NY. (No DOI; canonical source of the NYSIIS algorithm.)

---

[← Documentation Index](../README.md)
