# Beider-Morse Phonetic Matching Algorithm Extraction

> **Terminology.** A **phoneme** is a contrastive unit of sound. The IPA symbols used below denote, among others: `ʃ` ("sh"), `tʃ` ("ch"), `dʒ` ("j"), `ʒ` (the "s" in *measure*), `x` (voiceless velar fricative), `ɲ` (palatal nasal), and the palatalized consonants `dʲ`, `tʲ`. *Place of articulation* is where the vocal tract is constricted; *manner of articulation* is how airflow is shaped; *voicing* is whether the vocal folds vibrate; **final devoicing** turns a voiced consonant voiceless at a word boundary. See [`../GLOSSARY.md`](../GLOSSARY.md) for fuller definitions.

## Algorithm Overview

Beider-Morse Phonetic Matching (BMPM) was developed by Alexander Beider and Stephen P. Morse in 2008 for matching names of various ethnic origins. Each extracted rule carries a `weight` in `[0, 1]` expressing the residual edit cost of treating the two forms as equivalent. Unlike single-language algorithms, BMPM:

1. Detects the likely language(s) of a name
2. Applies language-specific phonetic rules
3. Generates multiple phonetic encodings for cross-language matching

## Language Branches

BMPM supports multiple language branches:

### Ashkenazic (Jewish names from Central/Eastern Europe)
- German
- Hebrew
- Hungarian
- Polish
- Romanian
- Russian
- Spanish (Ladino)
- Ukrainian

### Sephardic (Jewish names from Iberian Peninsula)
- French
- Hebrew
- Italian
- Portuguese
- Spanish

### Generic (General European names)
- English
- French
- German
- Greek
- Hungarian
- Italian
- Polish
- Portuguese
- Romanian
- Russian
- Spanish

## Extracted Rules by Language

### Germanic Branch
```llev
// Initial W → V (German pronunciation)
[id: 2900, name: "w to v initial bm germanic", weight: 0.15, group: bm_germanic]
w -> v / #_;

// EI diphthong
[id: 2901, name: "ei to ai bm germanic", weight: 0.15, group: bm_germanic]
ei -> ai;

// IE combination
[id: 2902, name: "ie to i bm germanic", weight: 0.15, group: bm_germanic]
ie -> i;

// TZ cluster
[id: 2903, name: "tz to ts bm germanic", weight: 0.10, group: bm_germanic]
tz -> ts;

// SCH → ʃ
[id: 2904, name: "sch to sh bm germanic", weight: 0.10, group: bm_germanic]
sch -> ʃ;
```

### Slavic Branch
```llev
// Polish clusters
[id: 3100, name: "szcz bm slavic", weight: 0.10, group: bm_slavic]
szcz -> ʃtʃ;

[id: 3101, name: "cz bm slavic", weight: 0.10, group: bm_slavic]
cz -> tʃ;

[id: 3102, name: "sz bm slavic", weight: 0.10, group: bm_slavic]
sz -> ʃ;

[id: 3103, name: "rz bm slavic", weight: 0.10, group: bm_slavic]
rz -> ʒ;

// Russian patterns
[id: 3110, name: "ogo to ovo bm russian", weight: 0.15, group: bm_russian]
ого -> ovo;  // Ivanov pronunciation

// Final devoicing (universal Slavic)
[id: 3120, name: "final b to p bm slavic", weight: 0.10, group: bm_slavic_devoicing]
b -> p / _#;

[id: 3121, name: "final d to t bm slavic", weight: 0.10, group: bm_slavic_devoicing]
d -> t / _#;

[id: 3122, name: "final g to k bm slavic", weight: 0.10, group: bm_slavic_devoicing]
g -> k / _#;
```

### Hungarian Branch
```llev
// Digraphs
[id: 3500, name: "cs to ch bm hungarian", weight: 0.10, group: bm_hungarian]
cs -> tʃ;

[id: 3501, name: "gy to dy bm hungarian", weight: 0.10, group: bm_hungarian]
gy -> dʲ;

[id: 3502, name: "ly to y bm hungarian", weight: 0.10, group: bm_hungarian]
ly -> j;

[id: 3503, name: "ny to ny bm hungarian", weight: 0.10, group: bm_hungarian]
ny -> ɲ;

[id: 3504, name: "sz to s bm hungarian", weight: 0.10, group: bm_hungarian]
sz -> s;

[id: 3505, name: "ty to ty bm hungarian", weight: 0.10, group: bm_hungarian]
ty -> tʲ;

[id: 3506, name: "zs to zh bm hungarian", weight: 0.10, group: bm_hungarian]
zs -> ʒ;
```

### Greek Branch
```llev
// Transliteration variants
[id: 3700, name: "ph to f bm greek", weight: 0.10, group: bm_greek]
ph -> f;

[id: 3701, name: "th to t bm greek", weight: 0.15, group: bm_greek]
th -> t;  // Modern Greek pronunciation

[id: 3702, name: "ch to kh bm greek", weight: 0.10, group: bm_greek]
ch -> x;

// Vowel digraphs
[id: 3710, name: "ou to u bm greek", weight: 0.10, group: bm_greek_vowel]
ou -> u;

[id: 3711, name: "ai to e bm greek", weight: 0.15, group: bm_greek_vowel]
ai -> e;  // Modern Greek
```

### Hebrew Branch
```llev
// Transliteration variants
[id: 3800, name: "ch to kh bm hebrew", weight: 0.10, group: bm_hebrew]
ch -> x;  // ח/כ

[id: 3801, name: "kh to kh bm hebrew", weight: 0.10, group: bm_hebrew]
kh -> x;

[id: 3810, name: "tz to ts bm hebrew", weight: 0.10, group: bm_hebrew]
tz -> ts;  // צ

[id: 3811, name: "sh to sh bm hebrew", weight: 0.10, group: bm_hebrew]
sh -> ʃ;  // ש
```

## Cross-Language Patterns

BMPM's key insight is that names migrate across languages with predictable transformations:

### German ↔ Polish
```llev
// -stein → -sztajn
[id: 3150, name: "stein to sztajn cross", weight: 0.15, group: bm_cross]
stein -> ʃtajn / _#;

// -berg → -berk
[id: 3151, name: "berg to berk cross", weight: 0.15, group: bm_cross]
berg -> berk / _#;
```

### Russian ↔ English
```llev
// -ov/-ev endings
[id: 3160, name: "ov ending variants", weight: 0.10, group: bm_cross]
ov -> of / _#;

[id: 3161, name: "ev ending variants", weight: 0.10, group: bm_cross]
ev -> ef / _#;
```

## Name Element Patterns

BMPM recognizes common name elements across languages:

### Theophoric Elements
```llev
// -el (God) - Daniel, Michael, Gabriel
[id: 3200, name: "el theophoric bm", weight: 0.05, group: bm_theophoric]
el -> el / _#;

// -yahu/-yah (YHWH) - Isaiah, Jeremiah
[id: 3201, name: "yahu theophoric bm", weight: 0.10, group: bm_theophoric]
yahu -> ya / _#;
```

### Patronymic Suffixes
```llev
// -ovich/-evich (Russian)
[id: 3210, name: "ovich patronymic bm", weight: 0.05, group: bm_patronymic]
ovich -> ovitʃ / _#;

// -ski/-sky (Polish/Russian)
[id: 3211, name: "ski suffix bm", weight: 0.05, group: bm_patronymic]
ski -> ski / _#;

// -son/-sen (Germanic)
[id: 3212, name: "son patronymic bm", weight: 0.10, group: bm_patronymic]
son -> son / _#;
```

## Implementation Strategy

BMPM's multi-encoding approach maps to LLev through:

1. **Language detection**: Rules grouped by language origin
2. **Multiple outputs**: Non-deterministic rules with varying weights
3. **Cross-language matching**: Lower weights for same-language, higher for cross-language

The weight system allows:
- Same-language matches: Low distance (0.1-0.2)
- Related-language matches: Medium distance (0.2-0.4)
- Cross-family matches: Higher distance (0.4-0.6)

## References

1. Beider, Alexander & Morse, Stephen P. (2008). "Beider-Morse Phonetic Matching: An Alternative to Soundex with Fewer False Hits". *Avotaynu: The International Review of Jewish Genealogy* 24(2). (No DOI; canonical source of the BMPM algorithm.)
2. Beider, Alexander (2001). *A Dictionary of Ashkenazic Given Names: Their Origins, Structure, Pronunciation, and Migrations*. Avotaynu, Bergenfield, NJ. ISBN 978-1-886223-12-1.
3. Morse, Stephen P. "Phonetic Matching (Beider-Morse)". https://stevemorse.org/phonetics/bmpm.htm

---

[← Documentation Index](../README.md)
