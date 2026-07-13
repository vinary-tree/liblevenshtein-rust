# Code Completion Use Case

## Overview

Code completion is the flagship use case for contextual completion, providing IDE-like autocomplete functionality with scope-aware suggestions. This document details integration patterns, LSP server design, and real-world implementation strategies.

## Key Features

1. **Scope-Aware Suggestions** - Variables visible only in their lexical scope
2. **Incremental Updates** - Real-time completion as users type
3. **Fuzzy Matching** - Typo-tolerant suggestions with Levenshtein distance
4. **Hierarchical Contexts** - Functions, blocks, classes, modules
5. **Performance** - Sub-millisecond completion queries
6. **Unicode Support** - Correct handling of emoji, CJK identifiers

## Architecture

### IDE Integration Layers

```
┌─────────────────────────────────────────────────────────┐
│                    IDE / Editor                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Text Buffer │  │  AST Parser  │  │ UI Dropdown  │  │
│  └───────┬──────┘  └──────┬───────┘  └──────▲───────┘  │
│          │                 │                  │          │
└──────────┼─────────────────┼──────────────────┼──────────┘
           │                 │                  │
           │ Text Events     │ Scope Info       │ Results
           ▼                 ▼                  │
┌─────────────────────────────────────────────────────────┐
│              LSP Server / Completion Service            │
│  ┌──────────────────────────────────────────────────┐   │
│  │       DynamicContextualCompletionEngine          │   │
│  │  ┌────────────┐  ┌──────────┐  ┌─────────────┐  │   │
│  │  │ ContextTree│  │ DraftBuf │  │ Checkpoints │  │   │
│  │  └────────────┘  └──────────┘  └─────────────┘  │   │
│  │  ┌─────────────────────────────────────────────┐ │   │
│  │  │    PathMapDictionary / DynamicDawg          │ │   │
│  │  └─────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Performance Target |
|-----------|----------------|-------------------|
| **Text Buffer** | Track cursor position, character edits | < 1µs per keystroke |
| **AST Parser** | Extract scope boundaries (functions, blocks) | < 10ms incremental parse |
| **Completion Service** | Map scopes to contexts, query engine | < 500µs per query |
| **ContextTree** | Maintain scope hierarchy | < 100ns visibility check |
| **DraftBuffer** | Track variable names per scope | < 10ns insert_char |
| **Dictionary** | Store finalized/library symbols | < 10µs fuzzy query |
| **UI Dropdown** | Display ranked results | < 16ms render (60fps) |

## LSP Server Design

### Language Server Protocol Integration

The [Language Server Protocol](https://microsoft.github.io/language-server-protocol/) defines a standard for IDE-agnostic code intelligence. Here's how contextual completion integrates:

#### 1. Server Initialization

```rust
use liblevenshtein::contextual::{DynamicContextualCompletionEngine, ContextId};
use libdictenstein::pathmap::PathMapDictionary;

struct LspCompletionServer {
    engine: DynamicContextualCompletionEngine<PathMapDictionary<()>>,

    // Map from file URI + scope position to ContextId
    scope_map: HashMap<(String, Range), ContextId>,

    // Root context per file
    file_contexts: HashMap<String, ContextId>,
}

impl LspCompletionServer {
    fn new() -> Self {
        let dict = PathMapDictionary::new();
        let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

        // Pre-populate with standard library / language keywords
        let root = engine.root_context();
        engine.finalize(root, "String").unwrap();
        engine.finalize(root, "Vec").unwrap();
        engine.finalize(root, "HashMap").unwrap();
        // ... add stdlib symbols

        Self {
            engine,
            scope_map: HashMap::new(),
            file_contexts: HashMap::new(),
        }
    }
}
```

#### 2. Document Open/Close Events

```rust
impl LspCompletionServer {
    /// Handle textDocument/didOpen notification
    fn did_open(&mut self, uri: &str, text: &str) -> Result<()> {
        // Create root context for file
        let root = self.engine.root_context();
        let file_ctx = self.engine.create_context(root)?;
        self.file_contexts.insert(uri.to_string(), file_ctx);

        // Parse file to extract scopes
        self.update_scopes(uri, text)?;

        Ok(())
    }

    /// Handle textDocument/didClose notification
    fn did_close(&mut self, uri: &str) {
        if let Some(ctx) = self.file_contexts.remove(uri) {
            self.engine.remove_context(ctx);
        }

        // Remove all scopes for this file
        self.scope_map.retain(|(file, _), _| file != uri);
    }
}
```

#### 3. Incremental Text Changes

```rust
impl LspCompletionServer {
    /// Handle textDocument/didChange notification
    fn did_change(&mut self, uri: &str, changes: Vec<TextDocumentContentChangeEvent>)
        -> Result<()>
    {
        for change in changes {
            if let Some(range) = change.range {
                // Find affected context
                let ctx = self.find_context_at_position(uri, range.start)?;

                // Apply character insertions/deletions to draft buffer
                for ch in change.text.chars() {
                    if ch == '\n' {
                        // Newline might indicate variable declaration complete
                        let draft = self.engine.draft_content(ctx)?;
                        if self.is_complete_declaration(&draft) {
                            self.engine.finalize(ctx, &draft)?;
                        }
                    } else if ch.is_alphanumeric() || ch == '_' {
                        self.engine.insert_char(ctx, ch)?;
                    }
                }
            } else {
                // Full document replacement
                self.update_scopes(uri, &change.text)?;
            }
        }

        Ok(())
    }

    fn is_complete_declaration(&self, draft: &str) -> bool {
        // Simple heuristic: "let foo = " or "fn bar(" indicates completion
        draft.contains('=') || draft.contains('(')
    }
}
```

#### 4. Completion Request Handler

```rust
impl LspCompletionServer {
    /// Handle textDocument/completion request
    fn completion(&self, uri: &str, position: Position, query: &str)
        -> Result<Vec<CompletionItem>>
    {
        // Find context at cursor position
        let ctx = self.find_context_at_position(uri, position)?;

        // Query with distance=1 for typo tolerance
        let completions = self.engine.complete(ctx, query, 1)?;

        // Convert to LSP CompletionItems
        let items = completions.into_iter().map(|c| {
            CompletionItem {
                label: c.term,
                kind: Some(CompletionItemKind::Variable),
                detail: Some(format!("distance: {}", c.distance)),
                sort_text: Some(format!("{:02}_{}", c.distance, c.term)),
                ..Default::default()
            }
        }).collect();

        Ok(items)
    }

    fn find_context_at_position(&self, uri: &str, pos: Position)
        -> Result<ContextId>
    {
        // Find innermost scope containing position
        let mut best_ctx = self.file_contexts.get(uri)
            .ok_or_else(|| Error::UnknownFile)?;
        let mut best_range = Range::entire_file();

        for ((file, range), ctx_id) in &self.scope_map {
            if file == uri && range.contains(pos) && range.is_narrower_than(&best_range) {
                best_ctx = ctx_id;
                best_range = range.clone();
            }
        }

        Ok(*best_ctx)
    }
}
```

### Scope Tracking Strategies

Different languages require different scope tracking approaches:

#### Strategy 1: AST-Based Scoping (Recommended)

Parse source code into an Abstract Syntax Tree, then create contexts for each scope-introducing construct:

```rust
// Example for a Rust-like language
fn update_scopes(&mut self, uri: &str, text: &str) -> Result<()> {
    let ast = parse_rust(text)?;
    let file_ctx = self.file_contexts[uri];

    self.traverse_ast(&ast, file_ctx, uri)?;
    Ok(())
}

fn traverse_ast(&mut self, node: &AstNode, parent_ctx: ContextId, uri: &str)
    -> Result<()>
{
    match node {
        AstNode::Function { name, params, body, range } => {
            // Create function scope
            let fn_ctx = self.engine.create_context(parent_ctx)?;
            self.scope_map.insert((uri.to_string(), range.clone()), fn_ctx);

            // Add parameters as finalized terms
            for param in params {
                self.engine.finalize(fn_ctx, param)?;
            }

            // Recurse into body
            self.traverse_ast(body, fn_ctx, uri)?;
        }

        AstNode::Block { statements, range } => {
            // Create block scope
            let block_ctx = self.engine.create_context(parent_ctx)?;
            self.scope_map.insert((uri.to_string(), range.clone()), block_ctx);

            for stmt in statements {
                if let AstNode::LetBinding { name, .. } = stmt {
                    self.engine.finalize(block_ctx, name)?;
                }
                self.traverse_ast(stmt, block_ctx, uri)?;
            }
        }

        AstNode::Impl { type_name, methods, range } => {
            // Create impl scope
            let impl_ctx = self.engine.create_context(parent_ctx)?;
            self.scope_map.insert((uri.to_string(), range.clone()), impl_ctx);

            // 'self' is available in all methods
            self.engine.finalize(impl_ctx, "self")?;

            for method in methods {
                self.traverse_ast(method, impl_ctx, uri)?;
            }
        }

        _ => {
            // Non-scope nodes: just recurse
            for child in node.children() {
                self.traverse_ast(child, parent_ctx, uri)?;
            }
        }
    }

    Ok(())
}
```

**Pros**: Accurate scope boundaries, handles nested scopes correctly
**Cons**: Requires language-specific parser

#### Strategy 2: Regex-Based Scoping (Lightweight)

For simple languages or prototyping, use regex to detect scope keywords:

```rust
fn update_scopes_regex(&mut self, uri: &str, text: &str) -> Result<()> {
    let file_ctx = self.file_contexts[uri];
    let lines: Vec<&str> = text.lines().collect();

    let mut scope_stack = vec![file_ctx];
    let mut indent_stack = vec![0];

    for (line_num, line) in lines.iter().enumerate() {
        let indent = line.chars().take_while(|&c| c == ' ').count();

        // Pop scopes when indent decreases
        while indent < *indent_stack.last().unwrap() {
            scope_stack.pop();
            indent_stack.pop();
        }

        let parent = *scope_stack.last().unwrap();

        // Detect scope-introducing keywords
        if line.trim_start().starts_with("fn ") ||
           line.trim_start().starts_with("def ") {
            let new_ctx = self.engine.create_context(parent)?;
            scope_stack.push(new_ctx);
            indent_stack.push(indent);

            let range = Range::from_line(line_num);
            self.scope_map.insert((uri.to_string(), range), new_ctx);
        }

        // Detect variable declarations
        if let Some(var_name) = extract_variable(line) {
            self.engine.finalize(parent, var_name)?;
        }
    }

    Ok(())
}

fn extract_variable(line: &str) -> Option<&str> {
    // Match patterns like "let foo = ", "var bar = ", "const baz = "
    let re = Regex::new(r"(let|var|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=").unwrap();
    re.captures(line).map(|caps| caps.get(2).unwrap().as_str())
}
```

**Pros**: No parser required, works for many languages
**Cons**: Less accurate, struggles with complex nesting

#### Strategy 3: Hybrid (Production Quality)

Combine AST for critical scopes + regex for variables:

```rust
fn update_scopes_hybrid(&mut self, uri: &str, text: &str) -> Result<()> {
    // Use AST for scope structure
    let ast = parse_lightweight(text)?;
    self.traverse_ast(&ast, self.file_contexts[uri], uri)?;

    // Use regex for variable detection within scopes
    for ((file, range), ctx) in &self.scope_map {
        if file == uri {
            let scope_text = extract_text_in_range(text, range);
            for var in find_variables_regex(&scope_text) {
                self.engine.finalize(*ctx, var)?;
            }
        }
    }

    Ok(())
}
```

**Pros**: Balance of accuracy and performance
**Cons**: Slightly more complex

## Usage Examples

### Example 1: Simple Function Scope

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;
use libdictenstein::pathmap::PathMapDictionary;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dict = PathMapDictionary::new();
    let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

    // Root context (global scope)
    let root = engine.root_context();
    engine.finalize(root, "println")?;
    engine.finalize(root, "vec")?;

    // Function scope
    let fn_ctx = engine.create_context(root)?;
    engine.finalize(fn_ctx, "param1")?;
    engine.finalize(fn_ctx, "param2")?;

    // User types "par"
    let completions = engine.complete(fn_ctx, "par", 1)?;

    // Results: ["param1", "param2"] (exact matches)
    assert_eq!(completions.len(), 2);
    assert_eq!(completions[0].term, "param1");
    assert_eq!(completions[0].distance, 0);

    // User types "prn" (typo for "println")
    let completions = engine.complete(fn_ctx, "prn", 1)?;

    // Results: ["println"] (distance 1: insert 't', insert 'l', insert 'i')
    assert!(completions.iter().any(|c| c.term == "println" && c.distance == 1));

    Ok(())
}
```

**Use Case**: Basic function-local variable completion with typo tolerance.

### Example 2: Nested Block Scopes

```rust
fn nested_scopes_example() -> Result<()> {
    let dict = PathMapDictionary::new();
    let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

    // Simulate: fn process(input: String) { let x = 1; { let y = 2; } }
    let root = engine.root_context();

    let fn_ctx = engine.create_context(root)?;
    engine.finalize(fn_ctx, "input")?;
    engine.finalize(fn_ctx, "x")?;

    let block_ctx = engine.create_context(fn_ctx)?;
    engine.finalize(block_ctx, "y")?;

    // Inside nested block: can see y, x, input
    let completions = engine.complete(block_ctx, "", 0)?;
    let terms: Vec<_> = completions.iter().map(|c| c.term.as_str()).collect();
    assert!(terms.contains(&"y"));      // Block variable
    assert!(terms.contains(&"x"));      // Function variable
    assert!(terms.contains(&"input"));  // Parameter

    // Outside nested block: can only see x, input (not y)
    let completions = engine.complete(fn_ctx, "", 0)?;
    let terms: Vec<_> = completions.iter().map(|c| c.term.as_str()).collect();
    assert!(terms.contains(&"x"));
    assert!(terms.contains(&"input"));
    assert!(!terms.contains(&"y"));  // Not visible here!

    Ok(())
}
```

**Use Case**: Lexical scoping - inner scopes see outer variables, but not vice versa.

### Example 3: Incremental Typing with Drafts

```rust
fn incremental_typing_example() -> Result<()> {
    let dict = PathMapDictionary::new();
    let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

    let root = engine.root_context();
    let ctx = engine.create_context(root)?;

    // User types "let co"
    engine.insert_char(ctx, 'l')?;
    engine.insert_char(ctx, 'e')?;
    engine.insert_char(ctx, 't')?;
    engine.insert_char(ctx, ' ')?;
    engine.insert_char(ctx, 'c')?;
    engine.insert_char(ctx, 'o')?;

    // Draft content: "let co"
    assert_eq!(engine.draft_content(ctx)?, "let co");

    // No finalized terms yet, so query returns empty
    let completions = engine.complete(ctx, "co", 0)?;
    assert_eq!(completions.len(), 0);

    // User completes variable declaration: "let count = 0;"
    engine.insert_char(ctx, 'u')?;
    engine.insert_char(ctx, 'n')?;
    engine.insert_char(ctx, 't')?;

    // Finalize the variable
    engine.finalize(ctx, "count")?;

    // Now "count" is available for completion
    let completions = engine.complete(ctx, "cou", 0)?;
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].term, "count");

    Ok(())
}
```

**Use Case**: Real-time completion as user types variable declarations.

### Example 4: Multi-File Project with Shared Root

```rust
struct MultiFileProject {
    engine: DynamicContextualCompletionEngine<PathMapDictionary<()>>,
    file_contexts: HashMap<String, ContextId>,
}

impl MultiFileProject {
    fn new() -> Result<Self> {
        let dict = PathMapDictionary::new();
        let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

        // Root context: shared stdlib/project-wide symbols
        let root = engine.root_context();
        engine.finalize(root, "String")?;
        engine.finalize(root, "Vec")?;
        engine.finalize(root, "HashMap")?;
        // Add project-wide types, constants, etc.

        Ok(Self {
            engine,
            file_contexts: HashMap::new(),
        })
    }

    fn open_file(&mut self, path: &str) -> Result<ContextId> {
        let root = self.engine.root_context();
        let file_ctx = self.engine.create_context(root)?;
        self.file_contexts.insert(path.to_string(), file_ctx);
        Ok(file_ctx)
    }

    fn close_file(&mut self, path: &str) {
        if let Some(ctx) = self.file_contexts.remove(path) {
            self.engine.remove_context(ctx);
        }
    }

    fn complete_in_file(&self, path: &str, query: &str, distance: usize)
        -> Result<Vec<String>>
    {
        let ctx = self.file_contexts.get(path)
            .ok_or("File not open")?;

        let completions = self.engine.complete(*ctx, query, distance)?;
        Ok(completions.into_iter().map(|c| c.term).collect())
    }
}

fn main() -> Result<()> {
    let mut project = MultiFileProject::new()?;

    // Open main.rs
    let main_ctx = project.open_file("src/main.rs")?;
    project.engine.finalize(main_ctx, "main_var")?;

    // Open lib.rs
    let lib_ctx = project.open_file("src/lib.rs")?;
    project.engine.finalize(lib_ctx, "lib_var")?;

    // In main.rs: see main_var + stdlib (not lib_var)
    let results = project.complete_in_file("src/main.rs", "", 0)?;
    assert!(results.contains(&"main_var".to_string()));
    assert!(results.contains(&"String".to_string()));
    assert!(!results.contains(&"lib_var".to_string()));

    // In lib.rs: see lib_var + stdlib (not main_var)
    let results = project.complete_in_file("src/lib.rs", "", 0)?;
    assert!(results.contains(&"lib_var".to_string()));
    assert!(results.contains(&"Vec".to_string()));
    assert!(!results.contains(&"main_var".to_string()));

    // Close main.rs
    project.close_file("src/main.rs");

    Ok(())
}
```

**Use Case**: Multi-file projects with shared global scope but isolated file scopes.

### Example 5: Checkpoint-Based Undo for Refactoring

```rust
fn refactoring_with_undo() -> Result<()> {
    let dict = PathMapDictionary::new();
    let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

    let root = engine.root_context();
    let ctx = engine.create_context(root)?;

    // Original variable name: "oldName"
    engine.finalize(ctx, "oldName")?;

    // User starts renaming: types "newN"
    let cp1 = engine.checkpoint(ctx)?;
    engine.insert_char(ctx, 'n')?;
    engine.insert_char(ctx, 'e')?;
    engine.insert_char(ctx, 'w')?;
    engine.insert_char(ctx, 'N')?;

    assert_eq!(engine.draft_content(ctx)?, "newN");

    // User changes mind, undoes
    engine.restore(ctx, cp1)?;
    assert_eq!(engine.draft_content(ctx)?, "");

    // Tries different name: "betterName"
    let cp2 = engine.checkpoint(ctx)?;
    engine.insert_char(ctx, 'b')?;
    engine.insert_char(ctx, 'e')?;
    engine.insert_char(ctx, 't')?;
    engine.insert_char(ctx, 't')?;
    engine.insert_char(ctx, 'e')?;
    engine.insert_char(ctx, 'r')?;
    engine.insert_char(ctx, 'N')?;
    engine.insert_char(ctx, 'a')?;
    engine.insert_char(ctx, 'm')?;
    engine.insert_char(ctx, 'e')?;

    // Finalize rename
    engine.finalize(ctx, "betterName")?;

    // Now both names are available
    let completions = engine.complete(ctx, "", 0)?;
    let terms: Vec<_> = completions.iter().map(|c| c.term.as_str()).collect();
    assert!(terms.contains(&"oldName"));
    assert!(terms.contains(&"betterName"));

    Ok(())
}
```

**Use Case**: Undo/redo support during variable renaming or refactoring.

### Example 6: Unicode Identifier Support

```rust
fn unicode_identifiers() -> Result<()> {
    let dict = PathMapDictionary::new();
    let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

    let root = engine.root_context();
    let ctx = engine.create_context(root)?;

    // Japanese variable: "変数" (hensuu = variable)
    engine.finalize(ctx, "変数")?;

    // Chinese variable: "变量" (bianliang = variable)
    engine.finalize(ctx, "变量")?;

    // Emoji variable (valid in many languages now!)
    engine.finalize(ctx, "🚀speed")?;

    // Query with partial match
    let completions = engine.complete(ctx, "変", 0)?;
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].term, "変数");

    // Query emoji identifier
    let completions = engine.complete(ctx, "🚀", 0)?;
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].term, "🚀speed");

    // Fuzzy match works across Unicode
    let completions = engine.complete(ctx, "变", 1)?;
    // May match "变量" exactly or "變數" with distance 1
    assert!(!completions.is_empty());

    Ok(())
}
```

**Use Case**: International programming languages with non-ASCII identifiers.

## Performance Recommendations

### Query Optimization

1. **Distance Parameter**: Start with distance=1 for typical typos
   - 99% of typos are 1-2 edit distance
   - Distance=2 adds ~10x query time
   - Distance=3+ rarely needed, expensive

2. **Result Limiting**: Truncate to top 10-20 completions
   ```rust
   let mut completions = engine.complete(ctx, query, 1)?;
   completions.truncate(20);  // Most UIs show ~10 items
   ```

3. **Debouncing**: Don't query on every keystroke
   ```rust
   const DEBOUNCE_MS: u64 = 150;

   let mut last_query = Instant::now();

   fn on_keystroke(&mut self, ch: char) {
       self.engine.insert_char(self.current_ctx, ch);

       if last_query.elapsed().as_millis() >= DEBOUNCE_MS {
           self.trigger_completion();
           last_query = Instant::now();
       }
   }
   ```

### Memory Management

1. **Context Lifecycle**: Remove contexts when scopes close
   ```rust
   // When user exits function/block
   engine.remove_context(scope_ctx);
   ```

2. **Checkpoint Limits**: Bound undo history size
   ```rust
   const MAX_CHECKPOINTS: usize = 50;

   struct BoundedCheckpointStack {
       stack: Vec<Checkpoint>,
   }

   impl BoundedCheckpointStack {
       fn push(&mut self, cp: Checkpoint) {
           if self.stack.len() >= MAX_CHECKPOINTS {
               self.stack.remove(0);  // Drop oldest
           }
           self.stack.push(cp);
       }
   }
   ```

3. **Dictionary Backend Selection**:
   - **PathMapDictionary**: Best for < 100K terms, dynamic workloads
   - **DynamicDawg**: Best for 100K-1M terms, frequent inserts
   - **DoubleArrayTrie**: Best for > 1M terms, read-heavy (use StaticContextualCompletionEngine)

### Benchmarking Results

Based on real-world usage with Rust standard library identifiers (~50K terms):

| Operation | Time | Throughput |
|-----------|------|------------|
| insert_char | 3.8µs | 263K ops/sec |
| delete_char | 2.1µs | 476K ops/sec |
| checkpoint | 8ns | 125M ops/sec |
| restore | 25ns | 40M ops/sec |
| complete(dist=0) | 8.2µs | 122K queries/sec |
| complete(dist=1) | 11.5µs | 87K queries/sec |
| complete(dist=2) | 48.3µs | 21K queries/sec |
| finalize | 12.4µs | 81K terms/sec |
| create_context | 156ns | 6.4M contexts/sec |

**Real-World Target**: < 16ms total latency (60fps UI)
- Query: ~12µs
- Result processing: ~100µs
- UI render: ~15ms
- **Total: ~15.1ms** ✓

## User Experience Considerations

### 1. Ranking Strategies

Rank completions by:
1. **Edit distance** (primary) - Exact matches first
2. **Scope proximity** - Local variables before globals
3. **Frequency** - Recently used terms ranked higher
4. **Prefix match** - "sta" prefers "start" over "erase"

```rust
fn rank_completions(completions: &mut [Completion], query: &str, ctx: ContextId, tree: &ContextTree) {
    completions.sort_by(|a, b| {
        // 1. Edit distance
        a.distance.cmp(&b.distance)
            // 2. Prefix match bonus
            .then_with(|| {
                let a_prefix = a.term.starts_with(query);
                let b_prefix = b.term.starts_with(query);
                b_prefix.cmp(&a_prefix)  // True sorts first
            })
            // 3. Scope depth (local vars closer)
            .then_with(|| {
                let a_depth = tree.depth(a.context);
                let b_depth = tree.depth(b.context);
                b_depth.cmp(&a_depth)  // Deeper = more local
            })
            // 4. Alphabetical
            .then_with(|| a.term.cmp(&b.term))
    });
}
```

### 2. Visual Indicators

Show scope/distance information in UI:

```
┌─────────────────────────────────────┐
│ 📝 Completions for "cou"            │
├─────────────────────────────────────┤
│ ● count        [local, exact]       │
│ ● counter      [local, dist=1]      │
│ ○ cout         [global, dist=1]     │
│ ○ CourseID     [global, dist=2]     │
└─────────────────────────────────────┘

Legend:
  ● = current scope
  ○ = parent/global scope
  [exact] = distance 0
  [dist=N] = edit distance N
```

### 3. Context Hints

Show which scope a completion comes from:

```rust
struct EnhancedCompletion {
    term: String,
    distance: usize,
    context: ContextId,
    scope_name: String,  // "main()", "for loop", "global"
}

fn enhance_completion(c: Completion, scope_map: &HashMap<ContextId, String>)
    -> EnhancedCompletion
{
    let scope_name = scope_map.get(&c.context)
        .cloned()
        .unwrap_or_else(|| "global".to_string());

    EnhancedCompletion {
        term: c.term,
        distance: c.distance,
        context: c.context,
        scope_name,
    }
}
```

### 4. Adaptive Distance

Start with distance=0 for fast queries, increase if no results:

```rust
fn adaptive_complete(engine: &Engine, ctx: ContextId, query: &str)
    -> Result<Vec<Completion>>
{
    for distance in 0..=2 {
        let results = engine.complete(ctx, query, distance)?;
        if !results.is_empty() {
            return Ok(results);
        }
    }

    // No results even with distance=2
    Ok(vec![])
}
```

## Testing Coverage

For LSP server integration, test these scenarios:

1. **Basic Completion**: Query returns expected variables
2. **Scope Isolation**: Inner scope sees outer, not vice versa
3. **Typo Tolerance**: Distance=1 catches common typos
4. **Unicode Handling**: Emoji, CJK identifiers work correctly
5. **Context Removal**: Removing scope doesn't leak memory
6. **Concurrent Queries**: Multiple files queried simultaneously
7. **Incremental Updates**: didChange events update scopes correctly
8. **Large Projects**: 100K+ terms, 1000+ scopes, < 20ms query latency
9. **Edge Cases**: Empty query, very long query, special characters
10. **Stress Test**: 10K queries/sec, memory usage stable

See also: [../examples/lsp-completion.rs](../examples/lsp-completion.rs) for full working LSP server.

## Related Documentation

- **[Completion Engine](../implementation/completion-engine.md)** - Engine API reference
- **[Context Tree](../implementation/context-tree.md)** - Scope hierarchy implementation
- **[Draft Buffer](../implementation/draft-buffer.md)** - Character-level buffer
- **[Checkpoint System](../implementation/checkpoint-system.md)** - Undo/redo support
- **[Incremental Editing](./incremental-editing.md)** - Real-time text editing patterns
- **[LSP Example](../examples/lsp-completion.rs)** - Full LSP server implementation

## References

1. [Language Server Protocol Specification](https://microsoft.github.io/language-server-protocol/)
2. [VSCode Language Extensions Guide](https://code.visualstudio.com/api/language-extensions/language-server-extension-guide)
3. [Tree-sitter Incremental Parsing](https://tree-sitter.github.io/tree-sitter/)
4. [IntelliJ Platform SDK - Code Completion](https://plugins.jetbrains.com/docs/intellij/code-completion.html)
5. Benchmark data from `benches/contextual_benchmarks.rs`
