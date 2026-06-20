# Lazy to Universal Automaton Mapping for Transposition

**Date**: 2025-11-13
**Purpose**: Derive correct Universal automaton formulas from the working lazy implementation

## Lazy Automaton Position Semantics

From `src/transducer/position.rs`, the lazy Position is:
```rust
struct Position {
    term_index: usize,    // i - position in dictionary term
    num_errors: usize,    // e - errors consumed
    is_special: bool,     // t - transposition flag
}
```

## Lazy Transposition Logic (transition.rs:252-376)

### Case 1: Not in Transposition State (is_special == false)

**When match at position 1** (lines 327-331):
```rust
Some(1) => {
    next.push(Position::new(i, e + 1));           // insertion
    next.push(Position::new_special(i, e + 1));   // transposition start
    next.push(Position::new(i + 1, e + 1));       // substitution
    next.push(Position::new(i + 2, e + 1));       // direct jump
}
```

**Interpretation**:
- Current position: i#e (not transposing)
- cv[1] = true means: input character matches word[i+1]
- This suggests characters might be swapped!
- Generate 4 successors:
  1. i#(e+1): Standard insertion
  2. i#(e+1)_t: Enter transposition state (will verify on next input)
  3. (i+1)#(e+1): Standard substitution
  4. (i+2)#(e+1): Direct transposition (assumes swap without verification)

### Case 2: In Transposition State (is_special == true)

**When match at position 0** (lines 344-349):
```rust
if cv[h] {  // cv[0] - match at current position
    next.push(Position::new(i + 2, e));
}
```

**Interpretation**:
- Current position: i#e_t (in transposition state)
- cv[0] = true means: current input matches word[i]
- This CONFIRMS the transposition (previous input matched word[i+1], current matches word[i])
- Result: Jump to (i+2)#e (both characters consumed, no additional error)

## Universal Automaton Mapping

### Key Difference: Offset-Based Positions

Universal uses I+offset#errors where offset = i - errors (approximately).

For position I+offset#errors processing input at position k:
- Word position i = k + offset (from line 734 in position.rs)
- Bit vector index for word position j: index = n + (j - (k-n)) = n + j - k + n = 2n + j - k

Wait, let me recalculate this more carefully...

From position.rs:734-736:
```rust
// For I+offset#errors at input k, the concrete word position is i = offset + k
// In bit vector s_n(w,k) (which starts at k-n), position i corresponds to index: i - (k-n) = offset + n
let match_index = (max_distance as i32 + offset) as usize;
```

So:
- Word position i = offset + k (where k is current input position)
- Bit vector starts at word position k-n
- Word position i maps to bit vector index: i - (k-n) = (offset+k) - (k-n) = offset + n

**For checking word position j** relative to current input k:
- Bit vector index = j - (k-n) = j - k + n = n + (j - k)

### Transposition State Machine for Universal

**Usual State at position I+offset#errors, input position k**:

Word position: i = k + offset

Check if cv[1] = true (word position i+1):
- Word position to check: i+1 = k + offset + 1
- Bit vector index: n + ((k+offset+1) - k) = n + offset + 1

✓ This matches my implementation! (next_match_index = n + offset + 1)

**Transposing State at position I+offset#errors, input position k**:

This position was created from (i-1)#(e-1) entering transposition:
- Previous word position: i-1 = (k-1) + offset_prev
- New position: i#e = ((i-1)+1)#((e-1)+1) = i#e
- New offset: i - e = (i-1+1) - (e-1+1) = i-1 - (e-1) = offset_prev

So offset DOESN'T change! But we're now at the NEXT input character (k instead of k-1).

Word position: i = k + offset

We want to check if current input matches word position i-1:
- Word position to check: i-1 = k + offset - 1
- Bit vector index: n + ((k+offset-1) - k) = n + offset - 1

❌ My implementation checks n + offset (wrong!)

## The Defect

In Transposing state, I should check `match_index - 1` not `match_index`:

```rust
// WRONG (current):
let match_index = (max_distance as i32 + offset) as usize;
if match_index < bit_vector.len() && bit_vector.is_match(match_index)

// CORRECT:
let prev_match_index = (max_distance as i32 + offset - 1) as usize;
if prev_match_index < bit_vector.len() && bit_vector.is_match(prev_match_index)
```

Wait, but that means checking position n+offset-1, which is checking word position i-1. Let me verify this is correct...

Actually, looking at the lazy automaton again at line 345:
```rust
if cv[h] {  // h = 0, so cv[0]
```

And cv is "offset-adjusted" (line 263), so cv[0] corresponds to the current position i, not i-1.

Hmm, let me reconsider the semantics...

## Re-analysis: What Does Transposing State Represent?

Looking at line 287 again:
```rust
next.push(Position::new_special(i, e + 1));   // transposition start
```

This creates position i#(e+1)_t. This is:
- Same word position i (not i+1!)
- One more error
- In transposing state

Then on the NEXT input character, at line 345:
```rust
if cv[h] {  // Check if next input matches current word position i
    next.push(Position::new(i + 2, e));
}
```

So the state machine is:
1. At position i#e, see that input matches word[i+1] → enter i#(e+1)_t
2. At position i#(e+1)_t with NEXT input, check if it matches word[i]
3. If yes → jump to (i+2)#e (consumed both swapped characters)

## Corrected Universal Mapping

**Transposing State I+offset#errors at input position k**:

The position represents word position i where we haven't advanced yet!
- Word position: i = k + offset (but we're checking against NEXT input!)
- We want: does current input match word position i?
- Bit vector index: n + offset

✓ My implementation checking `match_index = n + offset` is CORRECT!

But wait, then why is it failing?

## The Real Defect: Position Creation

Looking at my code at line 229-233:
```rust
if let Ok(trans) = UniversalPosition::new_i_with_state(
    offset + 1,  // ← defect candidate
    errors + 1,
    max_distance,
    TranspositionState::Transposing,
)
```

I'm creating position with offset+1, but the lazy automaton creates position with SAME i!

Let me check: if we're at I+offset#errors (word position i = k+offset), and we enter transposition:
- Lazy creates: i#(e+1)_t (same i, one more error)
- Universal should create: I+offset#(errors+1)_t (same offset!)

**The defect is that I'm incrementing offset when entering transposition!**

It should be:
```rust
UniversalPosition::new_i_with_state(
    offset,      // Same offset, not offset+1!
    errors + 1,
    max_distance,
    TranspositionState::Transposing,
)
```

And then in the completion, we should create:
```rust
UniversalPosition::new_i_with_state(
    offset + 2,  // Jump ahead by 2
    errors,      // Back to original error count
    max_distance,
    TranspositionState::Usual,
)
```

Let me verify this makes sense:
- Enter transposition: i#e → i#(e+1)_t (offset stays same)
- Complete transposition: i#(e+1)_t → (i+2)#e (offset increases by 2, errors decreases by 1)
  - Wait, that's not right either. Let me recalculate offset:
  - Position (i+2)#e has offset = (i+2) - e

Actually, the offset calculation gets complex. Let me just follow the lazy pattern exactly!

## Correct Fix

Match the lazy automaton behavior exactly:

**Enter transposition** (from i#e):
- Create i#(e+1)_t
- Universal offset = i - (e+1) = i - e - 1 = offset - 1

**Complete transposition** (from i#(e+1)_t):
- Create (i+2)#e
- Universal offset = (i+2) - e = i - e + 2 = (offset-1) + 2 = offset + 1

So:
- Enter: `new_i_with_state(offset - 1, errors + 1, ...)`
- Complete: `new_i_with_state(offset + 1, errors, ...)`

Let me double-check with concrete example...

Actually, I think the confusion is because Universal uses I^ε conversion. Let me just empirically test different combinations!
