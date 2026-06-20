# Soundex Algorithm Extraction

## Algorithm Overview

Soundex is a *phonetic algorithm* for indexing names by sound, patented by Robert C. Russell and Margaret King Odell in the early 1900s. It encodes homophones to the same representation so they can be matched despite minor spelling differences. Here, a **phoneme** is a contrastive unit of sound in a language, and a **phonetic algorithm** maps orthography (spelling) to an approximation of pronunciation so that sound-alike strings collide.

> **Terminology.** *Place of articulation* — where in the vocal tract the airflow is constricted (e.g. the lips, the alveolar ridge); *manner of articulation* — how the airflow is shaped (e.g. complete closure for a stop, turbulence for a fricative); *voicing* — whether the vocal folds vibrate (`b` is voiced, `p` is voiceless). The Soundex equivalence classes below group consonants that share an approximate place/manner. See [`../GLOSSARY.md`](../GLOSSARY.md) for fuller definitions.

![Phonetic NFA product pipeline: phonetic rules normalize a term, a Thompson-constructed NFA is intersected with a Levenshtein automaton, and the product walks the dictionary trie.](../diagrams/phonetic/nfa-product-pipeline.svg)

## Encoding Rules

The American Soundex algorithm:

1. Retain the first letter of the name
2. Replace consonants with digits:
   - `B, F, P, V` → `1` (labials)
   - `C, G, J, K, Q, S, X, Z` → `2` (gutturals/sibilants)
   - `D, T` → `3` (dentals)
   - `L` → `4` (lateral)
   - `M, N` → `5` (nasals)
   - `R` → `6` (rhotic)
3. Remove vowels (`A, E, I, O, U`) and `H, W, Y`
4. Remove duplicate adjacent digits
5. Truncate or pad to `4` characters

## Extracted Rules

### Labial Equivalences (B↔F↔P↔V)
```llev
[id: 2100, name: "b_f soundex labial", weight: 0.25, group: soundex_labials]
b -> f;

[id: 2101, name: "f_p soundex labial", weight: 0.25, group: soundex_labials]
f -> p;

[id: 2102, name: "p_v soundex labial", weight: 0.25, group: soundex_labials]
p -> v;

[id: 2103, name: "v_b soundex labial", weight: 0.25, group: soundex_labials]
v -> b;
```

### Guttural/Sibilant Equivalences
```llev
[id: 2110, name: "c_k soundex guttural", weight: 0.25, group: soundex_gutturals]
c -> k;

[id: 2111, name: "g_j soundex guttural", weight: 0.25, group: soundex_gutturals]
g -> j;

[id: 2112, name: "k_q soundex guttural", weight: 0.25, group: soundex_gutturals]
k -> q;

[id: 2113, name: "s_z soundex sibilant", weight: 0.25, group: soundex_gutturals]
s -> z;

[id: 2114, name: "x_s soundex sibilant", weight: 0.25, group: soundex_gutturals]
x -> ks;
```

### Dental Equivalences (D↔T)
```llev
[id: 2120, name: "d_t soundex dental", weight: 0.20, group: soundex_dentals]
d -> t;

[id: 2121, name: "t_d soundex dental", weight: 0.20, group: soundex_dentals]
t -> d;
```

### Nasal Equivalences (M↔N)
```llev
[id: 2125, name: "m_n soundex nasal", weight: 0.30, group: soundex_nasals]
m -> n;

[id: 2126, name: "n_m soundex nasal", weight: 0.30, group: soundex_nasals]
n -> m;
```

## Rationale for Weight Selection

- **Labials (0.25)**: B/F/P/V share lip articulation but differ in voicing and manner
- **Gutturals (0.25)**: Back-of-throat consonants merged somewhat aggressively
- **Dentals (0.20)**: D/T differ only in voicing, acoustically similar
- **Nasals (0.30)**: M/N more perceptually distinct, higher penalty

## Limitations

Soundex was designed for English names and performs poorly on:
- Non-English phonetic patterns
- Names beginning with vowels (first letter retained literally)
- Short names (truncation/padding artifacts)

The extracted rules capture the equivalence classes without the mechanical encoding, allowing phonetic matching to benefit from Soundex insights while avoiding its limitations.

## References

1. Russell, R.C. (1918). US Patent 1,261,167
2. Knuth, D.E. (1973). The Art of Computer Programming, Vol. 3
