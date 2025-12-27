# On-line SCDAWG Construction

This document describes the on-line O(n) algorithm for constructing the SCDAWG, based on Inenaga et al. (2001, 2005). The algorithm processes characters left-to-right, enabling dynamic updates as strings are added.

## Overview

The construction combines:
1. **Inenaga et al. (2005)**: On-line CDAWG construction
2. **Inenaga et al. (2001)**: Sext link maintenance for left extensions

Key insight from Inenaga (2001): **Sext links of CDAWG(w) equal edges of CDAWG(w^rev)**. This means left extension edges can be maintained incrementally during construction, without building a separate reversed automaton.

## Data Structures

### Reference Pairs

A **reference pair** represents a location in the CDAWG, which may be **explicit** (at a node) or **implicit** (within an edge):

```
struct ReferencePair {
    node: NodeId,      // Explicit ancestor node
    start: Position,   // Start position in text for edge label
    end: Position,     // End position (exclusive); start > end means explicit
}
```

**Explicit**: When start > end, the reference is at the node itself.

**Implicit**: When start ≤ end, the reference is within the edge starting at `node` with first character `text[start]`.

### Node Structure

Each CDAWG node stores:

```
struct CdawgNode {
    // Standard CDAWG fields
    edges: Map<Char, Edge>,       // Outgoing edges (right extensions)
    suffix_link: Option<NodeId>,  // Points to longest proper suffix node
    length: usize,                // Length of longest string reaching this node

    // SCDAWG additions
    sext_edges: Map<Char, Edge>,  // Sext links = left extension edges
}

struct Edge {
    start: Position,   // Start position in text
    end: Position,     // End position (or 'open' marker)
    target: NodeId,    // Target node
}
```

### Global State

```
struct CdawgBuilder {
    nodes: Vec<CdawgNode>,
    text: Vec<Char>,
    source: NodeId,           // Root node (empty string)
    sink: NodeId,             // Current longest string node
    active_point: ReferencePair,  // Current update position
    e: Position,              // Global end marker for open edges
}
```

## The On-line Update Algorithm

### High-Level Flow

For each new character `c`:

```
fn update(c: Char) {
    e += 1;  // Extend all open edges automatically
    text.push(c);

    // Create new sink for the extended string
    let new_sink = create_node();

    // Update active point until we find c or reach source
    while !check_end_point(c) {
        if active_point.is_implicit() {
            let (r, split_occurred) = handle_implicit_case(c, new_sink);
        } else {
            handle_explicit_case(c, new_sink);
        }
        move_to_next_suffix();
    }

    // Finalize the update
    finalize_update(new_sink);
    sink = new_sink;
}
```

### Check End Point

The `check_end_point` function determines if the current character already exists:

```
fn check_end_point(c: Char) -> bool {
    if active_point.is_explicit() {
        // At an explicit node - check if edge with c exists
        nodes[active_point.node].edges.contains_key(c)
    } else {
        // Implicit - check if next char in edge matches c
        let edge = get_edge(active_point);
        text[edge.start + (active_point.end - active_point.start + 1)] == c
    }
}
```

### Handling Implicit Case

When the active point is within an edge:

```
fn handle_implicit_case(c: Char, new_sink: NodeId) -> (NodeId, bool) {
    // Check for redirect (node merging)
    if let Some(target) = try_redirect(c) {
        // The edge already leads where we want
        redirect_edge_to(target);
        return (target, false);
    }

    // Split the edge
    let split_node = split_edge();

    // Create edge from split node to new sink
    add_edge(split_node, e-1, OPEN_END, new_sink);

    // Update suffix links
    update_suffix_links(split_node);

    // Update sext links (for SCDAWG)
    update_sext_links(split_node);

    (split_node, true)
}
```

### Handling Explicit Case

When the active point is at a node:

```
fn handle_explicit_case(c: Char, new_sink: NodeId) {
    let node = active_point.node;

    // Create edge from current node to new sink
    add_edge(node, e-1, OPEN_END, new_sink);

    // Check if node separation is needed
    if needs_separation(node) {
        separate_node(node);
    }
}
```

### Canonize

The `canonize` function advances through edges to find the explicit ancestor:

```
fn canonize(node: NodeId, start: Position, end: Position) -> ReferencePair {
    if start > end {
        // Already canonical
        return ReferencePair { node, start, end };
    }

    loop {
        let edge = nodes[node].edges[text[start]];
        let edge_len = edge.end - edge.start;

        if edge_len <= end - start {
            // Move past this edge
            start += edge_len + 1;
            node = edge.target;
            if start > end {
                break;
            }
        } else {
            // Stop within this edge
            break;
        }
    }

    ReferencePair { node, start, end }
}
```

### Move to Next Suffix

```
fn move_to_next_suffix() {
    let (node, start, end) = (
        active_point.node,
        active_point.start,
        active_point.end
    );

    // Follow suffix link
    let suffix_node = nodes[node].suffix_link.unwrap_or(source);

    // Canonize the new position
    active_point = canonize(suffix_node, start, end);
}
```

## Edge Splitting

When the active point is implicit and we need to add a new edge, we split:

```
fn split_edge() -> NodeId {
    let ReferencePair { node, start, end } = active_point;

    // Get the edge to split
    let first_char = text[start];
    let edge = nodes[node].edges[first_char].clone();

    // Create new intermediate node
    let split_node = create_node();
    let split_pos = edge.start + (end - start);

    // Modify original edge: node -> split_node
    nodes[node].edges[first_char] = Edge {
        start: edge.start,
        end: split_pos,
        target: split_node,
    };

    // Create continuation edge: split_node -> original_target
    let cont_char = text[split_pos + 1];
    nodes[split_node].edges[cont_char] = Edge {
        start: split_pos + 1,
        end: edge.end,
        target: edge.target,
    };

    // Set length
    nodes[split_node].length = nodes[node].length + (split_pos - edge.start + 1);

    split_node
}
```

## Node Separation

Node separation handles the case where a node needs to represent two different equivalence classes:

**Condition**: A node needs separation when:
- It has an incoming edge from a recent split
- Its longest string differs from the string used to reach it

```
fn separate_node(node: NodeId) {
    // Clone the node
    let clone = clone_node(node);

    // Adjust lengths
    nodes[clone].length = /* computed from path length */;

    // Redirect some incoming edges to clone
    for incoming in find_incoming_edges(node) {
        if should_redirect_to_clone(incoming) {
            redirect_to(incoming, clone);
        }
    }

    // Update suffix links
    nodes[clone].suffix_link = nodes[node].suffix_link;
    nodes[node].suffix_link = Some(clone);
}
```

## Sext Link Maintenance

The key innovation from Inenaga et al. (2001) is maintaining sext links (left extension edges) during construction.

### Sext Link Property

**Theorem**: sext_link(x) in CDAWG(w) equals the edge (y, x) in CDAWG(w^rev).

### Updating Sext Links

When we create or split a node, we update sext links:

```
fn update_sext_links(node: NodeId) {
    // For each suffix link pointing TO node,
    // create a sext link FROM node

    if let Some(suffix_target) = nodes[node].suffix_link {
        // Compute the label for the sext link
        let gamma = compute_left_extension_label(node, suffix_target);

        // Add sext link from suffix_target to node
        nodes[suffix_target].sext_edges.insert(gamma[0], Edge {
            start: /* gamma start */,
            end: /* gamma end */,
            target: node,
        });
    }
}
```

### Deriving Left Extensions from Suffix Links

After construction (or incrementally):

```
fn add_left_extensions() {
    for node_idx in 0..nodes.len() {
        if let Some(suffix_target) = nodes[node_idx].suffix_link {
            // The characters dropped by suffix link form the left extension label
            let dropped_len = nodes[node_idx].length - nodes[suffix_target].length;

            if dropped_len > 0 {
                // Get the first character of the dropped prefix
                let first_char = get_representative(node_idx)[0];

                // Add left extension edge
                nodes[suffix_target].sext_edges.entry(first_char).or_insert(Edge {
                    start: /* appropriate position */,
                    end: /* appropriate position */,
                    target: node_idx,
                });
            }
        }
    }
}
```

## Multi-String Support

For dictionaries with multiple strings, we use unique end markers:

```
const END_MARKER_BASE: Char = 0x100;  // Outside normal ASCII

fn add_term(term: &str, term_idx: usize) {
    let end_marker = END_MARKER_BASE + term_idx;

    // Process each character
    for c in term.chars() {
        update(c);
    }

    // Add unique end marker
    update(end_marker);

    // Mark sink as final for this term
    nodes[sink].is_final = true;
    nodes[sink].term_indices.push(term_idx);

    // Reset for next term (but keep automaton state)
    // The end marker ensures separation between terms
}
```

### Why Unique End Markers?

Consider adding "cat" and "catalog":
- Without end markers: "cat" becomes a prefix of "catalog", sharing nodes
- With end markers: "cat$₁" and "catalog$₂" are distinct, each gets its own sink

The end markers ensure:
1. Each term has a unique accepting state
2. Suffix links don't cross term boundaries incorrectly
3. Occurrence counting remains accurate

## Open Edges Trick

A key efficiency optimization: **open edges** automatically extend as new characters are added.

```
struct Edge {
    start: Position,
    end: OpenEnd,  // Either a position or 'OPEN'
    target: NodeId,
}

enum OpenEnd {
    Fixed(Position),
    Open,  // Means "end = e" (current position)
}
```

When `update()` increments `e`, all open edges extend automatically without explicit modification.

**Freezing**: When an edge no longer grows (e.g., at a split point), we "freeze" it:
```
edge.end = OpenEnd::Fixed(current_position);
```

## Complexity Analysis

**Theorem** (Inenaga et al., 2005):
The on-line CDAWG construction runs in O(n) time and space.

**Proof sketch**:
1. Each character is processed once
2. Amortized O(1) operations per character via suffix link traversal
3. At most O(n) nodes and edges created total
4. Canonize operations traverse at most O(n) total edge length

**SCDAWG Extension**:
Adding sext links adds O(1) work per suffix link, maintaining O(n) complexity.

## Worked Example: "abcabcab"

Let's trace construction step by step:

### Initial State

```
Nodes: [source]
sink = source
e = 0
```

### After 'a' (e=1)

```
Create sink₁ for "a"
source --a[0,open]--> sink₁
suffix_link(sink₁) = source

Nodes: [source, sink₁]
```

### After 'b' (e=2)

```
Create sink₂ for "ab"
source --b[1,open]--> sink₂  (secondary edge for "b")
sink₁ --b[1,open]--> sink₂   (primary edge for "ab")
suffix_link(sink₂) = source  (since "b" leads to source-equivalent)

Nodes: [source, sink₁, sink₂]
```

### After 'c' (e=3)

```
Create sink₃ for "abc"
source --c[2,open]--> sink₃  (for "c")
sink₂ --c[2,open]--> sink₃   (for "abc")
suffix_link(sink₃) = source

Nodes: [source, sink₁, sink₂, sink₃]
```

### After 'a' (e=4)

```
Edge already exists: source --a--> sink₁
Extend implicitly (open edges handle it)

Create sink₄ for "abca"
sink₃ --a[3,open]--> sink₄
suffix_link(sink₄) = sink₁

Nodes: [source, sink₁, sink₂, sink₃, sink₄]
```

### After 'b' (e=5)

```
Check from sink₄:
- Needs edge for 'b'
- Follow suffix_link to sink₁, check there
- sink₁ has edge 'b' -> sink₂
- But we're at suffix of "abca" = "a", need to extend "a" -> "ab"
- "ab" already exists, so we're at implicit point in path

Create sink₅ for "abcab"
... (details involve checking active points)
```

Construction continues similarly, with nodes being created and potentially split as patterns emerge.

## Summary

| Concept | Description |
|---------|-------------|
| Reference pair | (node, start, end) - explicit or implicit position |
| Check end point | Does character already exist at current position? |
| Split edge | Create intermediate node when adding diverging edge |
| Separate node | Clone node when equivalence class needs splitting |
| Open edges | Edges with end = 'open' extend automatically |
| Sext links | Left extension edges, maintained via suffix links |
| Unique end markers | $₁, $₂, ... for multi-string dictionaries |

**Key insight**: On-line construction enables dynamic dictionary updates while maintaining O(n) complexity. Sext links provide left extensions without building CDAWG(w^rev) explicitly.

**Next**: [06-operations](06-operations.md) - Using the SCDAWG for search and IS features
