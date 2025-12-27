# SCDAWG Operations

This document covers the operations supported by the SCDAWG, including substring search, bidirectional pattern extension, and Inverted File (IS) features from Blumer et al. (1987).

## Substring Search

The fundamental operation: given a pattern P, determine if P is a substring of the indexed text.

### Algorithm

```
fn contains_substring(pattern: &[Char]) -> bool {
    let mut current = source;
    let mut consumed = 0;

    while consumed < pattern.len() {
        // Find edge starting with pattern[consumed]
        let c = pattern[consumed];

        match nodes[current].edges.get(&c) {
            Some(edge) => {
                // Match characters along the edge
                let edge_label = &text[edge.start..=edge.end];
                let remaining = &pattern[consumed..];

                let match_len = common_prefix_length(edge_label, remaining);

                if match_len < edge_label.len() && match_len < remaining.len() {
                    // Mismatch within edge
                    return false;
                }

                consumed += match_len;

                if match_len == edge_label.len() {
                    // Fully consumed edge, move to target
                    current = edge.target;
                }
                // else: pattern ends within edge (still a match)
            }
            None => return false,
        }
    }

    true
}
```

### Complexity

**Time**: O(|pattern|) - each character is examined once.

**Space**: O(1) additional space beyond the SCDAWG itself.

### Finding the Representative Node

Often we want not just existence, but a handle to continue operations:

```
fn find_substring(pattern: &[Char]) -> Option<SubstringHandle> {
    let mut current = source;
    let mut consumed = 0;
    let mut within_edge: Option<(Edge, usize)> = None;

    while consumed < pattern.len() {
        let c = pattern[consumed];

        match nodes[current].edges.get(&c) {
            Some(edge) => {
                let edge_label = &text[edge.start..=edge.end];
                let remaining = &pattern[consumed..];
                let match_len = common_prefix_length(edge_label, remaining);

                if match_len < edge_label.len() && match_len < remaining.len() {
                    return None;
                }

                consumed += match_len;

                if match_len == edge_label.len() {
                    current = edge.target;
                    within_edge = None;
                } else {
                    // Ended within edge
                    within_edge = Some((edge.clone(), match_len));
                }
            }
            None => return None,
        }
    }

    Some(SubstringHandle {
        node: current,
        within_edge,
        pattern_len: pattern.len(),
    })
}
```

## Bidirectional Extension

The SCDAWG supports both right extension (appending) and left extension (prepending).

### Right Extension

Navigate from a pattern V to V·σ (append character σ):

```
fn right_extend(handle: &SubstringHandle, c: Char) -> Option<SubstringHandle> {
    let node = handle.node;

    // If we're within an edge, check the next character
    if let Some((edge, pos)) = &handle.within_edge {
        let next_char = text[edge.start + pos + 1];
        if next_char == c {
            // Continue along this edge
            let new_pos = pos + 1;
            if new_pos >= edge.end - edge.start {
                // Reached edge target
                return Some(SubstringHandle {
                    node: edge.target,
                    within_edge: None,
                    pattern_len: handle.pattern_len + 1,
                });
            } else {
                return Some(SubstringHandle {
                    node,
                    within_edge: Some((edge.clone(), new_pos)),
                    pattern_len: handle.pattern_len + 1,
                });
            }
        }
        return None;
    }

    // At explicit node - look for outgoing edge
    nodes[node].edges.get(&c).map(|edge| {
        if edge.end == edge.start {
            // Single-character edge
            SubstringHandle {
                node: edge.target,
                within_edge: None,
                pattern_len: handle.pattern_len + 1,
            }
        } else {
            // Multi-character edge
            SubstringHandle {
                node,
                within_edge: Some((edge.clone(), 0)),
                pattern_len: handle.pattern_len + 1,
            }
        }
    })
}
```

### Left Extension

Navigate from pattern V to σ·V (prepend character σ):

```
fn left_extend(handle: &SubstringHandle, c: Char) -> Option<SubstringHandle> {
    let node = handle.node;

    // Left extension uses sext_edges (left extension edges)
    // These edges lead to patterns with c prepended

    nodes[node].sext_edges.get(&c).map(|sext_edge| {
        SubstringHandle {
            node: sext_edge.target,
            within_edge: None,  // Sext edges lead to explicit nodes
            pattern_len: handle.pattern_len + 1,
        }
    })
}
```

### Enumerating Extensions

List all possible right/left extensions:

```
fn right_extensions(handle: &SubstringHandle) -> impl Iterator<Item = (Char, SubstringHandle)> {
    if handle.within_edge.is_some() {
        // Only one possible extension (next char in edge)
        // Return that single option
        ...
    } else {
        // At explicit node - all outgoing edges
        nodes[handle.node].edges.iter().map(|(c, edge)| {
            (*c, SubstringHandle { ... })
        })
    }
}

fn left_extensions(handle: &SubstringHandle) -> impl Iterator<Item = (Char, SubstringHandle)> {
    // All sext edges from current node
    nodes[handle.node].sext_edges.iter().map(|(c, edge)| {
        (*c, SubstringHandle { ... })
    })
}
```

## Inverted File (IS) Features

Blumer et al. (1987) describe the SCDAWG as supporting **Inverted File** (IS) features, which provide document-level information about substrings.

### Frequency: freq(x)

Return the number of times pattern x occurs in the text:

```
fn freq(handle: &SubstringHandle) -> usize {
    // Count is stored at nodes or computed from subtree
    let node = handle.node;

    // Option 1: Pre-computed during construction
    nodes[node].occurrence_count

    // Option 2: Count accepting states in subtree
    count_accepting_states(node)
}
```

**Pre-computing frequencies**:
```
fn compute_frequencies() {
    // Post-order traversal
    for node in topological_order().rev() {
        if nodes[node].is_final {
            nodes[node].freq = 1;
        } else {
            nodes[node].freq = 0;
        }

        // Add children's frequencies
        for edge in nodes[node].edges.values() {
            nodes[node].freq += nodes[edge.target].freq;
        }
    }
}
```

### Locations: locations(x)

Return all positions where pattern x occurs:

```
fn locations(handle: &SubstringHandle) -> Vec<Position> {
    let node = handle.node;
    let pattern_len = handle.pattern_len;
    let mut positions = Vec::new();

    collect_positions(node, pattern_len, &mut positions);
    positions
}

fn collect_positions(node: NodeId, pattern_len: usize, out: &mut Vec<Position>) {
    // If this node is final, compute its position
    if nodes[node].is_final {
        let end_pos = nodes[node].end_position;  // Position where this suffix ends
        let start_pos = end_pos - pattern_len;
        out.push(start_pos);
    }

    // Recurse to children
    for edge in nodes[node].edges.values() {
        collect_positions(edge.target, pattern_len, out);
    }
}
```

### Find: find(x)

Search for pattern x and return an IS handle if found:

```
fn find(pattern: &[Char]) -> Option<ISHandle> {
    find_substring(pattern).map(|handle| {
        ISHandle {
            node: handle.node,
            pattern_len: handle.pattern_len,
        }
    })
}
```

The IS handle can then be used with freq() and locations().

### Multi-String Dictionary Locations

For dictionaries with multiple terms, we track which term(s) contain each substring:

```
struct OccurrenceInfo {
    term_index: usize,
    position: usize,
}

fn locations_with_terms(handle: &SubstringHandle) -> Vec<OccurrenceInfo> {
    let mut occurrences = Vec::new();

    fn collect(node: NodeId, pattern_len: usize, out: &mut Vec<OccurrenceInfo>) {
        for &term_idx in &nodes[node].term_indices {
            let pos = compute_position(node, term_idx, pattern_len);
            out.push(OccurrenceInfo { term_index: term_idx, position: pos });
        }

        for edge in nodes[node].edges.values() {
            collect(edge.target, pattern_len, out);
        }
    }

    collect(handle.node, handle.pattern_len, &mut occurrences);
    occurrences
}
```

## WallBreaker Integration

The SCDAWG satisfies all requirements from Gerdjikov et al. (2013):

### Requirement (1a): Substring Check

```rust
impl WallBreakerSupport for Scdawg {
    fn contains_substring(&self, v: &[Char]) -> bool {
        self.find_substring(v).is_some()
    }
}
```

**Complexity**: O(|v|)

### Requirement (1b): Right Extension

```rust
impl WallBreakerSupport for Scdawg {
    fn right_extend(&self, handle: &Handle, sigma: Char) -> Option<Handle> {
        self.try_right_extend(handle, sigma)
    }

    fn right_extensions(&self, handle: &Handle) -> Vec<(Char, Handle)> {
        self.enumerate_right_extensions(handle).collect()
    }
}
```

**Complexity**: O(1) for single extension, O(|Σ|) for enumeration

### Requirement (1c): Left Extension

```rust
impl WallBreakerSupport for Scdawg {
    fn left_extend(&self, handle: &Handle, sigma: Char) -> Option<Handle> {
        self.try_left_extend(handle, sigma)  // Uses sext_edges
    }

    fn left_extensions(&self, handle: &Handle) -> Vec<(Char, Handle)> {
        self.enumerate_left_extensions(handle).collect()
    }
}
```

**Complexity**: O(1) for single extension, O(|Σ|) for enumeration

## Traversal Patterns

### Depth-First Enumeration of All Substrings

```
fn enumerate_all_substrings<F: FnMut(&[Char])>(callback: &mut F) {
    let mut path = Vec::new();

    fn dfs(node: NodeId, path: &mut Vec<Char>, callback: &mut F) {
        callback(path);  // Report current substring

        for (c, edge) in &nodes[node].edges {
            // Extend path with edge label
            for i in edge.start..=edge.end {
                path.push(text[i]);
            }

            dfs(edge.target, path, callback);

            // Backtrack
            for _ in edge.start..=edge.end {
                path.pop();
            }
        }
    }

    dfs(source, &mut path, callback);
}
```

### Finding Maximal Repeats

A **maximal repeat** is a string that:
1. Occurs more than once
2. Cannot be extended left or right without reducing frequency

```
fn find_maximal_repeats() -> Vec<(String, usize)> {
    let mut results = Vec::new();

    for node in 0..nodes.len() {
        let freq = nodes[node].freq;
        if freq >= 2 {
            // Check if maximal (left-diverse and right-diverse)
            let left_diverse = nodes[node].sext_edges.len() >= 2
                || (nodes[node].sext_edges.len() == 1 && is_suffix(node));
            let right_diverse = nodes[node].edges.len() >= 2
                || (nodes[node].edges.len() == 1 && is_prefix(node));

            if left_diverse && right_diverse {
                let repr = get_representative(node);
                results.push((repr, freq));
            }
        }
    }

    results
}
```

## Performance Considerations

### Caching Frequencies

For frequent freq() queries, pre-compute and cache:

```
fn precompute_all_frequencies(&mut self) {
    // Topological sort nodes
    let order = topological_sort(&self.nodes);

    // Process in reverse order (leaves first)
    for &node_idx in order.iter().rev() {
        let mut freq = 0;

        if self.nodes[node_idx].is_final {
            freq += 1;
        }

        for edge in self.nodes[node_idx].edges.values() {
            freq += self.nodes[edge.target].cached_freq;
        }

        self.nodes[node_idx].cached_freq = freq;
    }
}
```

### Lazy Location Enumeration

For large result sets, use iterators:

```
fn locations_iter(&self, handle: &SubstringHandle) -> impl Iterator<Item = Position> {
    LocationIterator::new(self, handle.node, handle.pattern_len)
}

struct LocationIterator<'a> {
    scdawg: &'a Scdawg,
    stack: Vec<NodeId>,
    pattern_len: usize,
}

impl Iterator for LocationIterator<'_> {
    type Item = Position;

    fn next(&mut self) -> Option<Position> {
        while let Some(node) = self.stack.pop() {
            // Push children
            for edge in self.scdawg.nodes[node].edges.values() {
                self.stack.push(edge.target);
            }

            // Return position if final
            if self.scdawg.nodes[node].is_final {
                return Some(compute_position(node, self.pattern_len));
            }
        }
        None
    }
}
```

## Summary of Operations

| Operation | Description | Complexity |
|-----------|-------------|------------|
| `contains_substring(P)` | Check if P exists | O(\|P\|) |
| `find_substring(P)` | Get handle to P if exists | O(\|P\|) |
| `right_extend(h, c)` | Navigate to h·c | O(1) |
| `left_extend(h, c)` | Navigate to c·h | O(1) |
| `right_extensions(h)` | List all right extensions | O(\|Σ\|) |
| `left_extensions(h)` | List all left extensions | O(\|Σ\|) |
| `freq(h)` | Count occurrences | O(1) cached, O(n) uncached |
| `locations(h)` | List all positions | O(occ) where occ = output size |

**Key insight**: The SCDAWG provides O(|pattern|) substring search with O(1) bidirectional extension, making it ideal for algorithms like WallBreaker that need to grow patterns in both directions.

**Next**: [07-references](07-references.md) - Annotated bibliography of source papers
