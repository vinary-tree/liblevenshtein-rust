# Universal Levenshtein Automaton Documentation

This directory contains documentation for the Universal Levenshtein Automaton implementation, including its optimal-string-alignment transposition variant (restricted Damerau distance, not unrestricted Damerau–Levenshtein).

## Overview

The Universal Levenshtein Automaton is a parameter-free automaton that efficiently computes approximate string matching using offset-based positions rather than word-specific absolute positions.

## Key Documents

### Implementation Summary
- **[SESSION_COMPLETION_2025-11-13.md](SESSION_COMPLETION_2025-11-13.md)** - Complete session summary with final status
- **[transposition_phase2_summary.md](transposition_phase2_summary.md)** - Technical summary of Phase 2 transposition implementation
- **[transposition_implementation_plan.md](transposition_implementation_plan.md)** - Original implementation plan

### Phase Completion Documents
- **[transposition_phase1_complete.md](transposition_phase1_complete.md)** - Phase 1: Infrastructure and trait-based dispatch
- **[transposition_phase2_complete.md](transposition_phase2_complete.md)** - Phase 2: Transposition logic implementation
- **[merge_split_phase3_complete.md](merge_split_phase3_complete.md)** - Phase 3: Merge/Split logic implementation

### Technical Analysis
- **[lazy_to_universal_mapping.md](lazy_to_universal_mapping.md)** - Cross-validation between Universal and lazy automaton
- **[hypothesis_h5_offset_completion_defect.md](hypothesis_h5_offset_completion_defect.md)** - Critical defect fix: transposition completion offset
- **[hypothesis_h6_test_assertions.md](hypothesis_h6_test_assertions.md)** - Test assertion corrections
- **[merge_split_analysis.md](merge_split_analysis.md)** - Phase 3 research: merge/split operations analysis and offset calculations

### Debugging Documents
- **[transposition_fix_needed.md](transposition_fix_needed.md)** - Initial debugging analysis
- **[transposition_phase2_debugging_needed.md](transposition_phase2_debugging_needed.md)** - Phase 2 debugging session
- **[transposition_manual_trace.md](transposition_manual_trace.md)** - Manual simulation traces
- **[hypothesis_h4_bit_vector_semantics.md](hypothesis_h4_bit_vector_semantics.md)** - Bit vector indexing analysis

## Implementation Status

### Phase 1: Infrastructure ✅ COMPLETE
- Trait-based dispatch with `PositionVariant` trait
- Variant state tracking with associated `State` type
- Support for Standard, Transposition, and MergeAndSplit variants
- All backward compatibility tests passing

### Phase 2: Transposition ✅ COMPLETE
- Adjacent character swap support (⟨2,2,1⟩ operation)
- Transposition entry and completion logic
- Cross-validated against lazy automaton
- 12/12 transposition tests passing
- 168/168 universal automaton tests passing
- 617/617 total tests passing

### Phase 3: Merge/Split ✅ COMPLETE
- Merge operation ⟨2,1,1⟩ support (two input chars → one word char)
- Split operation ⟨1,2,1⟩ support (one input char → two word chars)
- Two-step split state machine (enter/complete)
- Cross-validated against lazy automaton
- 13/13 merge/split tests passing
- 181/181 universal automaton tests passing
- 630/630 total tests passing

## Key Technical Insights

### Universal Automaton Offsets
Position `I+offset#e` at input position `k` represents word position `i = offset + k`. This relative offset model differs from lazy automaton's absolute positions.

### Transposition State Machine
The ⟨2,2,1⟩ operation for adjacent character swaps:
- **Enter**: `i#e` → `i#(e+1)_t` (Universal: `offset - 1`)
- **Complete**: `i#(e+1)_t` → `(i+2)#e` (Universal: `offset + 1`)

### Standard Operations Inclusion
Transposition is ADDITIVE - it includes ALL standard operations (insertion, deletion, substitution) plus adjacent character swaps.

Similarly, merge and split operations are ADDITIVE - they include all standard operations plus the merge/split operations.

## Source Code Locations

- **Core Implementation**: `src/transducer/universal/position.rs`
- **Automaton**: `src/transducer/universal/automaton.rs`
- **Tests**: `src/transducer/universal/automaton.rs` (lines 467-720)

## Cross-Validation

The Universal automaton implementation has been thoroughly cross-validated against:
- Lazy automaton (`src/transducer/transition.rs`)
- Mitankin's thesis on Universal Levenshtein Automata

Both implementations agree completely on transposition semantics and behavior.

## Changes Summary

- **520 insertions, 55 deletions** across 3 source files
- Trait-based dispatch system for variant-specific logic
- Comprehensive test suite with 12 transposition-specific tests
- Detailed inline documentation explaining offset calculations

## Next Steps

1. **Phase 3**: Implement merge/split successor logic
2. **Phase 5**: Integrate Universal transposition with GeneralizedAutomaton Phase 2d
3. Consider performance optimization if needed

## References

- Mitankin, Petar. "Universal Levenshtein Automata - Building and Properties"
- See `/home/dylon/Papers/Approximate String Matching/Universal Levenshtein Automata - Building and Properties/`
