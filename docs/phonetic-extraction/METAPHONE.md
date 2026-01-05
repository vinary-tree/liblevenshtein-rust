# Metaphone and DoubleMetaphone Algorithm Extraction

## Metaphone Overview

Metaphone was developed by Lawrence Philips in 1990 as an improvement over Soundex. It uses context-sensitive rules based on English pronunciation patterns rather than simple character mapping.

## Key Improvements over Soundex

1. Context-sensitive consonant handling
2. Vowel preservation at word start
3. Recognition of silent letters
4. Special handling of letter combinations

## Metaphone Rules Extracted

### SCH Handling
```llev
[id: 2200, name: "sch to sk metaphone", weight: 0.10, group: metaphone_clusters]
sch -> sk;  // English "school", not German "sch→ʃ"
```

### DG before front vowels
```llev
[id: 2201, name: "dge to j metaphone", weight: 0.10, group: metaphone_clusters]
dge -> j;   // edge, badge, judge

[id: 2202, name: "dgi to j metaphone", weight: 0.10, group: metaphone_clusters]
dgi -> ji;  // digit
```

### GH patterns
```llev
[id: 2205, name: "gh silent metaphone", weight: 0.15, group: metaphone_gh]
gh ->  / [aeiou]_[aeiou];  // night, daughter (silent)

[id: 2206, name: "gh to f metaphone", weight: 0.10, group: metaphone_gh]
gh -> f / [aou]_#;  // rough, cough, laugh
```

### Initial X
```llev
[id: 2210, name: "initial x to s metaphone", weight: 0.10, group: metaphone_initial]
x -> s / #_;  // xylophone, Xerox
```

### KN, GN, PN, WR initial clusters
```llev
[id: 2215, name: "kn initial metaphone", weight: 0.10, group: metaphone_initial]
kn -> n / #_;  // knee, knife, know

[id: 2216, name: "gn initial metaphone", weight: 0.10, group: metaphone_initial]
gn -> n / #_;  // gnome, gnat

[id: 2217, name: "pn initial metaphone", weight: 0.10, group: metaphone_initial]
pn -> n / #_;  // pneumonia, pneumatic

[id: 2218, name: "wr initial metaphone", weight: 0.10, group: metaphone_initial]
wr -> r / #_;  // write, wrong, wrist
```

### WH initial
```llev
[id: 2220, name: "wh initial metaphone", weight: 0.15, group: metaphone_initial]
wh -> w / #_;  // what, where, when (most dialects)
```

---

## DoubleMetaphone Overview

DoubleMetaphone (2000) extends Metaphone to handle surnames of various ethnic origins. It returns two codes: primary (American pronunciation) and alternate (ethnic pronunciation).

## Etymology-Specific Rules Extracted

### Germanic Origins
```llev
[id: 2300, name: "w to v germanic dm", weight: 0.15, group: dm_germanic]
w -> v / #_;  // Wagner, Werner (German W = English V)

[id: 2301, name: "sch to sh germanic dm", weight: 0.10, group: dm_germanic]
sch -> ʃ;  // Schmidt, Schiller

[id: 2302, name: "ei to ai germanic dm", weight: 0.15, group: dm_germanic]
ei -> ai;  // Einstein, Weinstein
```

### Slavic Origins
```llev
[id: 2310, name: "cz to ch slavic dm", weight: 0.10, group: dm_slavic]
cz -> tʃ;  // Czech, Kowalczyk

[id: 2311, name: "sz to sh slavic dm", weight: 0.10, group: dm_slavic]
sz -> ʃ;  // Szymanski

[id: 2312, name: "initial j slavic dm", weight: 0.15, group: dm_slavic]
j -> y / #_;  // Jablonski (Polish J = Y)
```

### Greek Origins
```llev
[id: 2320, name: "ps initial greek dm", weight: 0.10, group: dm_greek]
ps -> s / #_;  // psychology, Psarogiorgos

[id: 2321, name: "pt initial greek dm", weight: 0.10, group: dm_greek]
pt -> t / #_;  // Ptolemy

[id: 2322, name: "ch to k greek dm", weight: 0.15, group: dm_greek]
ch -> k;  // chaos, Christ (Greek chi)

[id: 2323, name: "ph to f greek dm", weight: 0.10, group: dm_greek]
ph -> f;  // philosophy, Philadelphia
```

### Italian Origins
```llev
[id: 2330, name: "cci to chi italian dm", weight: 0.10, group: dm_italian]
cci -> tʃi;  // Gucci, Bucci

[id: 2331, name: "cce to che italian dm", weight: 0.10, group: dm_italian]
cce -> tʃe;  // Puccetti

[id: 2332, name: "gn to ny italian dm", weight: 0.10, group: dm_italian]
gn -> ɲ;  // Bologna, lasagna

[id: 2333, name: "gli to ly italian dm", weight: 0.10, group: dm_italian]
gli -> ʎi;  // famiglia, Caravaglio
```

### Spanish Origins
```llev
[id: 2340, name: "ll to y spanish dm", weight: 0.10, group: dm_spanish]
ll -> j;  // Castillo, Villa (yeismo)

[id: 2341, name: "ñ to ny spanish dm", weight: 0.10, group: dm_spanish]
ñ -> ɲ;  // España, niño
```

## Dual-Code Strategy

DoubleMetaphone's dual-code approach is captured in LLev by:
1. **Primary rules**: Lower weights for standard American pronunciation
2. **Alternate rules**: Higher weights for ethnic pronunciation variants

Both apply during matching, allowing cross-ethnic name matching with appropriate distance penalties.

## References

1. Philips, L. (1990). "Hanging on the Metaphone". Computer Language Magazine
2. Philips, L. (2000). "The Double Metaphone Search Algorithm". C/C++ Users Journal
