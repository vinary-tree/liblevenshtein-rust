# Literate downstream-query algorithms

This chapter gives executable pseudocode for the APIs specified in the
[downstream design](../../design/downstream-query-surfaces.md). The algorithms
state their invariants at the point where an implementation must preserve
them.

## 1. Prefix-pruned subsequence DFS

```pseudocode
procedure SUBSEQUENCE-DFS(root, query, visitor)
    stack ← [frame(root, matched=0, entered_by=none)]
    prefix ← empty sequence

    while stack is not empty do
        frame ← top(stack)
        if frame finality has not been checked then
            mark finality checked
            if frame is final
               and frame.matched = length(query)
               and visitor.permits_accept(prefix) then
                yield (copy(prefix), visitor.accept(prefix))
            end if
        end if

        if frame has another edge (unit, child) then
            depth ← length(prefix) + 1
            allowed ← visitor.enter(unit, depth)
            if not allowed then
                visitor.leave(unit, depth)
                continue
            end if
            matched ← frame.matched
            if matched < length(query) and unit = query[matched] then
                matched ← matched + 1
            end if
            append unit to prefix
            push frame(child, matched, entered_by=unit)
        else
            finished ← pop(stack)
            if finished.entered_by exists then
                unit ← pop(prefix)
                visitor.leave(unit, length(prefix) + 1)
            end if
        end if
    end while
end procedure
```

The stack invariant is:

```math
|\text{stack}|=|\text{prefix}|+1,
```

and the top frame's `matched` value is the length of the greedily embedded
query prefix. Every successful `enter` remains live until its child frame is
popped; every rejected `enter` is paired immediately.

## 2. Prefix-pruned fuzzy DFS

```pseudocode
procedure FUZZY-DFS(root, query, budget, automaton, visitor)
    stack ← [frame(root, automaton.initial(query, budget), entered_by=none)]
    prefix ← empty sequence

    while stack is not empty do
        frame ← top(stack)
        if frame is an unchecked terminal then
            mark terminal checked
            distance ← automaton.final_distance(frame.state, length(query))
            if distance ≤ budget and visitor.permits_accept(prefix) then
                yield (copy(prefix), distance, visitor.accept(prefix))
            end if
        end if

        if frame has another edge (unit, child) then
            depth ← length(prefix) + 1
            if not visitor.enter(unit, depth) then
                count external subtree prune
                visitor.leave(unit, depth)
                continue
            end if
            next ← automaton.transition(frame.state, unit, query, budget)
            if next does not exist then
                count automaton subtree prune
                visitor.leave(unit, depth)
                continue
            end if
            append unit to prefix
            push frame(child, next, entered_by=unit)
        else
            finished ← pop(stack)
            recycle finished.state
            if finished.entered_by exists then
                unit ← pop(prefix)
                visitor.leave(unit, length(prefix) + 1)
            end if
        end if
    end while
end procedure
```

The automaton state and visitor state describe the same root-to-node path.
That invariant holds naturally on the DFS stack. A BFS implementation would
instead need a visitor snapshot in every frontier entry or would need to
replay each retained prefix before transitioning it.

## 3. Building an exact prefix set

```pseudocode
procedure ALLOWED-PREFIXES(terms)
    prefixes ← {empty sequence}
    terminals ← empty set
    for each term in terms do
        insert term into terminals
        for length from 1 through length(term) do
            insert term[0..length] into prefixes
        end for
    end for
    return visitor(prefixes, terminals)
end procedure
```

Membership in `prefixes` controls descent; membership in `terminals` controls
yielding. Combining the predicates would incorrectly admit a non-terminal
prefix or prune an admitted descendant.

## 4. Distance-layered suggestion ranking

```pseudocode
procedure NEXT-SUGGESTION
    if sorted current-layer buffer is non-empty then
        return pop best suggestion
    end if

    while current_distance ≤ maximum_distance do
        while current distance queue is non-empty do
            intersection ← pop queue
            if intersection is a valued final node then
                exact ← infer terminal distance
                if exact = current_distance and term has not been seen then
                    score ← normalize scorer(term, exact, value)
                    append suggestion to current-layer buffer
                else if exact > current_distance then
                    requeue intersection in layer exact
                    continue
                end if
            end if
            transition each child and queue it by minimum reachable distance
        end while

        if current-layer buffer is non-empty then
            sort by confidence descending, then term ascending
            return pop best suggestion
        end if
        current_distance ← current_distance + 1
    end while
    return end-of-stream
end procedure
```

The outer loop makes distance primary without one global result heap. The
current-layer buffer is the only set of materialized results; traversal states
for later layers may be queued, but their terms, values, and scores are not
materialized.

## 5. Match-mode filtering

```pseudocode
procedure QUERY-MODE(query, mode)
    (minimum, maximum) ← validate inclusive bounds of mode
    for candidate in QUERY-ORDERED(query, maximum) do
        if candidate.distance ≥ minimum then
            yield candidate
        end if
    end for
end procedure
```

Only `maximum` is an automaton budget. `minimum` is applied at terminals,
because an intermediate prefix distance is not a lower bound on all completed
descendants.

## 6. Kind-erased bracket lower bound

```pseudocode
procedure BALANCE-LOWER-BOUND(tokens, kinds)
    opens ← 0
    closes ← 0
    for token in tokens do
        if token < kinds then
            opens ← opens + 1
        else if token < 2 × kinds then
            if opens > 0 then opens ← opens - 1
            else closes ← closes + 1
        else
            return unknown-token error
        end if
    end for
    return ceiling(opens / 2) + ceiling(closes / 2)
end procedure
```

The scan invariant says that `opens` counts unmatched projected openings in
the consumed prefix and `closes` counts projected closings that had no opening
available. A substitution can repair two unmatched brackets of the same
direction; insertion or deletion repairs one. Kind erasure can only turn a
substitution into a match, so the result is admissible for every number of
kinds.

## 7. Contextual DP column

```pseudocode
procedure CONTEXTUAL-CHILD-COLUMN(parent, prefix, unit, query, costs)
    current[0] ← parent[0] + valid-or-infinity(
        costs.insertion(context(query, 0, prefix, unit), unit))

    for i from 1 through length(query) do
        ctx ← context(query, i-1, prefix, unit)
        insert ← parent[i] + valid-or-infinity(costs.insertion(ctx, unit))
        delete ← current[i-1] + valid-or-infinity(costs.deletion(ctx, query[i-1]))
        replace ← parent[i-1]
                   + valid-or-infinity(costs.substitution(ctx, query[i-1], unit))
        current[i] ← minimum(insert, delete, replace)
    end for
    return current
end procedure
```

`valid-or-infinity` rejects `None`, NaN, infinities, and negative costs while
incrementing diagnostics. A child is queued only if at least one current cell
fits the threshold. The parent column is immutable during the update, so each
cell uses exactly the three Wagner–Fischer predecessors.

## 8. Conformance harness

Each formal invariant has an executable analogue:

```pseudocode
for 2,000 generated examples per property do
    compare subsequence DFS with a flat scan
    compare allowed-prefix traversal with exact set intersection
    count enter and leave events
    compare fuzzy DFS with BFS across every unit-cost algorithm
    unwind callbacks after early iterator termination
    compare MatchMode ranges with completed-candidate filtering
    compare ranked multiset with query_values
    check rank-window order
    compare projected bracket bound with brute-force Dyck distance
    compare contextual standard adapter with QueryIteratorF64
    check realignment symmetry
    serialize and deserialize every valid float-cost record exactly
end for
```

Backend examples repeat ranking on `DoubleArrayTrie` and `DynamicDawg`; raw
`u64` examples cover the units-native APIs. The ignored Birkbeck gate then
checks 42,395 real correction pairs.
