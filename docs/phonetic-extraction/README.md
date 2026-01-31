# Phonetic Algorithm Extraction for LLev Rules

This document describes the enrichment of liblevenshtein's LLev phonetic rules through extraction of knowledge from classic phonetic algorithms.

## Overview

The LLev (Levenshtein Language) rule system provides phonetic normalization through rewrite transformation rules. This project enriched the existing 125+ language rule sets by systematically extracting patterns from established phonetic matching algorithms.

## Algorithms Integrated

| Algorithm | Languages | Rule IDs | Rules Added |
|-----------|-----------|----------|-------------|
| Soundex | English | 2100-2199 | ~20 |
| Metaphone | English | 2200-2299 | ~35 |
| DoubleMetaphone | Multi-origin English | 2300-2399 | ~40 |
| Caverphone | NZ English | 2700-2799 | ~25 |
| NYSIIS | American names | 2800-2899 | ~25 |
| ColognePhonetic | German | 2400-2499 | ~25 |
| SpanishMetaphone | Spanish | 2500-2599 | ~20 |
| PHONEX | French | 2600-2699 | ~30 |
| Beider-Morse | Multi-language | 2900-2999, 3100-3899 | ~300 |
| Daitch-Mokotoff | Jewish/E. European | 3000-3099 | ~50 |

**Total: ~570 new rules**

## Implementation by Language

### Phase 1: English
- **Modified**: `/data/rules/english/base.llev`
  - Added Soundex consonant equivalence classes (labials, gutturals, dentals, nasals)
  - Added Metaphone context-sensitive rules (sch, dg, gh, x patterns)
  - Added DoubleMetaphone etymology rules (Germanic, Slavic, Greek, Italian origins)
- **Created**: `/data/rules/english/new_zealand.llev` - Caverphone rules for NZ English
- **Created**: `/data/rules/english/names.llev` - NYSIIS rules for American names

### Phase 2: German
- **Modified**: `/data/rules/german/base.llev`
  - Added ColognePhonetic C-context rules
  - Added D/T sibilant patterns
  - Added Beider-Morse Germanic patterns
- **Created**: `/data/rules/german/names.llev` - Germanic surname patterns

### Phase 3: Spanish
- **Modified**: `/data/rules/spanish/base.llev`
  - Added SpanishMetaphone RR/R distinction
  - Added dialectal patterns (Caribbean aspiration, final D weakening)
- **Modified**: `/data/rules/spanish/latin_american.llev` - Regional variants

### Phase 4: French
- **Modified**: `/data/rules/french/base.llev`
  - Added PHONEX ILL/IL yod patterns
  - Added OIN nasal patterns
  - Added liaison and elision patterns

### Phase 5: Slavic Languages
- **Modified**: `/data/rules/polish/base.llev`
  - Added Beider-Morse Slavic patterns (szcz, strz, cz, sz clusters)
  - Added regressive voicing assimilation
  - Added nasal vowel decomposition (ą, ę)
- **Created**: `/data/rules/polish/names.llev` - Polish surname patterns
- **Modified**: `/data/rules/russian/base.llev`
  - Added G-weakening patterns (ого → ово)
  - Added palatalization rules
  - Added cluster simplification
- **Created**: `/data/rules/russian/names.llev` - Russian patronymic and surname patterns
- **Created**: `/data/rules/jewish/names.llev` - Daitch-Mokotoff Soundex rules for Jewish/Eastern European names

### Phase 6: Other Languages
- **Modified**: `/data/rules/hungarian/base.llev`
  - Added Beider-Morse Hungarian patterns
  - Added cross-border patterns (German, Polish, Slavic influences)
  - Added historical spelling variants
- **Modified**: `/data/rules/italian/base.llev`
  - Added DoubleMetaphone Italian patterns
  - Added geminate consonant patterns (cci, ggi, gli, gn)
  - Added regional dialect patterns (Sicilian, Neapolitan, Venetian)
- **Modified**: `/data/rules/greek/base.llev`
  - Added Beider-Morse Greek patterns
  - Added Latin transliteration variants
  - Added name suffix patterns (-opoulos, -idis, -akis)
- **Modified**: `/data/rules/hebrew/base.llev`
  - Added Beider-Morse Hebrew transliteration patterns
  - Added theophoric element patterns (-el, -yahu)
  - Added common Hebrew name variants

## Rule ID Allocation

To prevent conflicts, rules use non-overlapping ID ranges:

| Range | Algorithm/Category |
|-------|-------------------|
| 2100-2199 | Soundex |
| 2200-2299 | Metaphone |
| 2300-2399 | DoubleMetaphone |
| 2400-2499 | ColognePhonetic |
| 2500-2599 | SpanishMetaphone |
| 2600-2699 | PHONEX (French) |
| 2700-2799 | Caverphone |
| 2800-2899 | NYSIIS |
| 2900-2999 | Beider-Morse (Generic) |
| 3000-3099 | Daitch-Mokotoff |
| 3100-3199 | BM Polish |
| 3200-3299 | Polish Names |
| 3300-3399 | Russian Names |
| 3400-3499 | BM Russian |
| 3500-3599 | BM Hungarian |
| 3600-3699 | DM Italian |
| 3700-3799 | BM Greek |
| 3800-3899 | BM Hebrew |

## Complementary: Articulatory Distance

The rule-based phonetic patterns described above handle *known* phonetic equivalences through explicit transformations. For *residual* substitutions not covered by rules, liblevenshtein also provides **articulatory distance**—a feature-based approach that computes phonetic similarity between any two IPA characters based on their articulatory properties.

| Approach | Strengths | Use Case |
|----------|-----------|----------|
| **LLev Rules** | Context-sensitive, language-specific, explicit patterns | Known phonetic alternations (tion→ʃən, ph→f) |
| **Articulatory Distance** | Universal, handles novel pairs, gradient costs | Residual substitutions after rule application |

The two approaches complement each other: rules handle predictable patterns while articulatory distance provides principled costs for everything else.

For full details on articulatory distance, see: [Articulatory Distance Guide](../guides/articulatory-distance.md)

## Weight System

Rules use a consistent weight system:
- **0.0-0.05**: Orthographic equivalence (different spellings of same sound)
- **0.05-0.10**: Near-equivalent sounds (minor phonetic differences)
- **0.10-0.20**: Phonetic approximations (dialectal variants, historical changes)
- **0.20-0.30**: Broader phonetic classes (algorithm-derived equivalences)
- **0.30+**: Distant matches (cross-language patterns)

## LLev Rule Format

```llev
[id: 2100, name: "rule_name", weight: 0.15, group: category, ipa: "/IPA/"]
pattern -> replacement / left_context _ right_context;
```

Components:
- **id**: Unique numeric identifier
- **name**: Human-readable rule name
- **weight**: Distance penalty (0.0-1.0)
- **group**: Category for rule organization
- **ipa**: Optional IPA pronunciation
- **pattern**: Input pattern to match
- **replacement**: Output transformation
- **context**: Optional left/right context constraints

Context syntax:
- `#_`: Word start
- `_#`: Word end
- `$VOWEL`: Any vowel character
- `$CONSONANT`: Any consonant character
- `[abc]`: Character class

## Algorithm Details

See individual documentation files for detailed algorithm analysis:
- [SOUNDEX.md](SOUNDEX.md) - American Soundex
- [METAPHONE.md](METAPHONE.md) - Metaphone and DoubleMetaphone
- [COLOGNE.md](COLOGNE.md) - Cologne Phonetic (Kölner Phonetik)
- [CAVERPHONE.md](CAVERPHONE.md) - Caverphone for NZ English
- [NYSIIS.md](NYSIIS.md) - New York State Identification and Intelligence System
- [BEIDER-MORSE.md](BEIDER-MORSE.md) - Beider-Morse Phonetic Matching
- [DAITCH-MOKOTOFF.md](DAITCH-MOKOTOFF.md) - Daitch-Mokotoff Soundex

## Testing Methodology

1. **Unit tests**: Each rule tested in isolation
2. **Equivalence tests**: Words with same algorithm code verified to have low LLev distance
3. **Regression tests**: Existing functionality preserved
4. **Cross-language tests**: Name variants across languages correctly matched

## References

1. Knuth, D.E. (1973). The Art of Computer Programming, Vol. 3: Sorting and Searching
2. Philips, L. (1990). "Hanging on the Metaphone". Computer Language Magazine
3. Philips, L. (2000). "The Double Metaphone Search Algorithm". C/C++ Users Journal
4. Postel, H.J. (1969). "Die Kölner Phonetik". IBM-Nachrichten
5. Hood, D. (2002). "Caverphone: Phonetic matching algorithm". Technical report
6. Beider, A. & Morse, S.P. (2008). "Beider-Morse Phonetic Matching"
7. Mokotoff, G. (1997). "Soundexing and Genealogy". Avotaynu
