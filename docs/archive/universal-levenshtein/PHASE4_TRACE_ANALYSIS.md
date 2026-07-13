# Phase 4: Trace Analysis - "test" → "text" Substitution Bug

## Test Case
- Word: "test" (positions 0='t', 1='e', 2='s', 3='t')
- Input: "text"
- Max distance: n = 2
- Expected: ACCEPT (1 substitution at position 2: 's'→'x')
- Actual: REJECT

## Execution Trace

### Step 1: Process 't' (input[0])
- Input char: 't', Word char: word[0]='t'
- Subword: s_n(w,1) = "$$test" (padded)
- Bit vector β('t', "$$test") = [false, false, true, false, false, true]
  - Matches at positions 2 and 5 (word positions 0 and 3)
- Current state: [I+0#0]
- **MATCH** at match_index = 0 + 2 = 2 ✓
- Next state: [I+0#0] ✓

### Step 2: Process 'e' (input[1])
- Input char: 'e', Word char: word[1]='e'
- Subword: s_n(w,2) = "$test"
- Bit vector β('e', "$test") = [false, false, true, false, false]
  - Matches at position 2 (word position 1)
- Current state: [I+0#0]
- **MATCH** at match_index = 0 + 2 = 2 ✓
- Next state: [I+0#0] ✓

### Step 3: Process 'x' (input[2])
- Input char: 'x', Word char: word[2]='s'
- Subword: s_n(w,3) = "test"
- Bit vector β('x', "test") = [false, false, false, false]
  - **NO MATCHES** (x not in "test")
- Current state: [I+0#0]
- **NO MATCH** at match_index = 0 + 2 = 2
- Generates error transitions:
  - DELETE: I-1#1 (skip word char, errors++)
  - SUBSTITUTE: I+0#1 (both advance, errors++)
- Next state: [I-1#1, I+0#1] ✓

### Step 4: Process 't' (input[3]) - THE BUG!
- Input char: 't', Word char: word[3]='t'
- Subword: s_n(w,4) = word[max(0,4-2)..min(4,4+2+1)]
           = word[2..4] = "st"
  - **WAIT!** The trace shows subword = "est"! This is wrong!

Let me recalculate the subword:
- From thesis page 51: s_n(w, i) = w_{i-n}...w_v where v = min(|w|, i + n + 1)
- For i=4, n=2, |w|=4:
  - Start: i - n = 4 - 2 = 2 (1-indexed) → word position 1 (0-indexed)
  - End: min(4, 4+2+1) = min(4,7) = 4 → up to word position 3 (0-indexed)
  - So s_n(w,4) = word[1..4] = "est" ✓ (trace is correct!)

- Bit vector β('t', "est") = [false, false, true]
  - Position 0: 't' == 'e' = false
  - Position 1: 't' == 's' = false
  - Position 2: 't' == 't' = **true** ✓

Now let's check each position in the current state:

#### Position I-1#1 at k=4:
- Concrete word position: i = k + offset = 4 + (-1) = 3
- Word char: word[3] = 't'
- Subword "est" starts at word position 1
- So bit_vector[j] corresponds to word position 1+j
- To check word[3], we need j where 1+j = 3 → j = 2
- bit_vector[2] = **true** → **THIS IS A MATCH!**

But what does the code check?
```rust
let match_index = (max_distance as i32 + offset) as usize;
                = (2 + (-1)) as usize
                = 1
```

So it checks bit_vector[1] = false → **NO MATCH DETECTED!** ❌

**THE BUG**: The match_index calculation is wrong!

## Root Cause

The formula `match_index = n + offset` is incorrect for checking matches.

The correct calculation should be:
- Concrete word position: i = k + offset
- Subword starts at: k - n (in 1-indexed terms) = (k-n-1) in 0-indexed
- To check word position i, we need bit_vector[i - (k-n-1)]
- Simplifying: match_index = i - (k-n-1) = (k+offset) - (k-n-1) = offset + n + 1

Wait, let me recalculate more carefully using 1-indexed positions as in the thesis:

In thesis (1-indexed):
- s_n(w, k) starts at position k-n
- bit_vector[0] corresponds to word position k-n
- bit_vector[j] corresponds to word position k-n+j

For universal position I+offset#e at input position k:
- Concrete word position (1-indexed): i = k + offset
- bit_vector[j] corresponds to word position k-n+j
- We want to find j such that k-n+j = i = k+offset
- So j = i - (k-n) = k+offset - k + n = offset + n

**Current code**: match_index = n + offset ✓

Wait, this is correct! Let me check the numbers again:
- k = 4 (1-indexed input position)
- offset = -1
- match_index = 2 + (-1) = 1

Subword "est" in 1-indexed terms:
- Starts at position k-n = 4-2 = 2 (word position 'e')
- bit_vector[0] → word position 2 ('e')
- bit_vector[1] → word position 3 ('s')
- bit_vector[2] → word position 4 ('t')

Concrete word position for I-1#1 at k=4:
- i = k + offset = 4 + (-1) = 3 (word position 's')

**AH!** The concrete position is 3, not 4! Let me recalculate:
- i = 3 (word position 's' in 1-indexed)
- bit_vector[j] where k-n+j = i
- 4-2+j = 3
- j = 1
- bit_vector[1] → word position 3 ('s')
- Input 't' vs word 's' → **NO MATCH** ✓

So for position I-1#1 at k=4, we're actually checking word[2]='s', not word[3]='t'!

## The Real Problem

The issue is that position I-1#1 represents "we're 1 position behind in the word relative to the input."

At input position 4:
- I+0#e would be at word position 4
- I-1#e would be at word position 3
- I-2#e would be at word position 2

In 0-indexed terms (what the word string uses):
- I+0#e at k=4 → word[3]
- I-1#e at k=4 → word[2]
- I-2#e at k=4 → word[1]

After step 3, we have `[I-1#1, I+0#1]` which means:
- I-1#1: We consumed 2 input chars, consumed 3 word chars, made 1 error
- I+0#1: We consumed 2 input chars, consumed 2 word chars, made 1 error (the substitution!)

Wait, let me think about the I^ε conversion again. From thesis page 23:
- I^ε({i#e}) = {I + (i-1)#e}

So for a concrete position (i,e) after processing k chars of input:
- If we're at input position k and word position i with e errors
- The universal position is: I + (i-k)#e

After processing 3 chars of input ("tex"):
- If we did the substitution: word position 3, errors 1
  - Universal: I + (3-3)#1 = I+0#1 ✓
- If we deleted word[2]: word position 4, errors 1
  - Universal: I + (4-3)#1 = I+1#1 ❌ (but we have I-1#1)

Hmm, I think I have the offset direction backwards! Let me check the I-type successor code.
