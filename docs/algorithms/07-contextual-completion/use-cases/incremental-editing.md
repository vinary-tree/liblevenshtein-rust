# Incremental Editing Use Case

## Overview

Incremental editing is the process of tracking text changes character-by-character in real-time, enabling responsive completion suggestions as users type. This use case focuses on editor integration, performance optimization, and handling complex editing operations.

## Key Features

1. **Character-Level Tracking** - Single-character insert/delete operations
2. **Real-Time Updates** - Sub-millisecond response to keystrokes
3. **Unicode Correctness** - Proper handling of multi-byte characters
4. **Backspace/Delete** - Efficient undo of partial input
5. **Word Boundaries** - Detecting when to finalize terms
6. **Multi-Line Support** - Newline handling for variable declarations
7. **Checkpoint Integration** - Undo/redo for editor integration

## Architecture

### Editor Event Flow

```
┌─────────────────────────────────────────────────────────┐
│                     Text Editor                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Text Buffer: "let cou|"                         │   │
│  │  Cursor Position: 7                              │   │
│  └────────────┬─────────────────────────────────────┘   │
│               │ onKeyPress('n')                         │
└───────────────┼─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│             Editing Bridge / Event Handler              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  1. Detect keystroke type (insert/delete/word)   │   │
│  │  2. Map cursor → ContextId                       │   │
│  │  3. Call engine.insert_char() / delete_char()    │   │
│  │  4. Trigger completion query if appropriate      │   │
│  └────────────┬─────────────────────────────────────┘   │
└───────────────┼─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│       DynamicContextualCompletionEngine                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │  DraftBuffer: "cou" → "coun"                     │   │
│  │  Checkpoint: created before 'n' insertion        │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Query: "coun" distance=1 → ["count", "counter"] │   │
│  └────────────┬─────────────────────────────────────┘   │
└───────────────┼─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│                  Completion UI                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  ● count      [local, exact]                     │   │
│  │  ● counter    [local, dist=1]                    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Core Editing Operations

### Insert Operations

#### Single Character Insert

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;
use libdictenstein::pathmap::PathMapDictionary;

fn handle_char_insert(
    engine: &mut DynamicContextualCompletionEngine<PathMapDictionary<()>>,
    ctx: ContextId,
    ch: char
) -> Result<()> {
    // Filter out non-identifier characters
    if ch.is_alphanumeric() || ch == '_' {
        engine.insert_char(ctx, ch)?;

        // Optionally trigger completion after each char
        let draft = engine.draft_content(ctx)?;
        if draft.len() >= 2 {  // Minimum prefix length
            trigger_completion_popup(engine, ctx, &draft)?;
        }
    } else if ch == '\n' || ch == ';' {
        // Word boundary: finalize current draft
        let draft = engine.draft_content(ctx)?;
        if !draft.is_empty() {
            engine.finalize(ctx, &draft)?;
        }
    }

    Ok(())
}
```

#### Batch Insert (Paste Operation)

```rust
fn handle_paste(
    engine: &mut DynamicContextualCompletionEngine<PathMapDictionary<()>>,
    ctx: ContextId,
    text: &str
) -> Result<()> {
    // Create checkpoint before paste for undo
    let checkpoint = engine.checkpoint(ctx)?;

    // Insert each character
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            engine.insert_char(ctx, ch)?;
        } else if ch == '\n' || ch == ';' {
            // Word boundary: finalize and clear
            let draft = engine.draft_content(ctx)?;
            if !draft.is_empty() {
                engine.finalize(ctx, &draft)?;
            }
        }
    }

    Ok(())
}
```

### Delete Operations

#### Backspace (Delete Before Cursor)

```rust
fn handle_backspace(
    engine: &mut DynamicContextualCompletionEngine<PathMapDictionary<()>>,
    ctx: ContextId
) -> Result<()> {
    let draft = engine.draft_content(ctx)?;

    if !draft.is_empty() {
        // Remove last character from draft
        engine.delete(ctx)?;

        // Update completions with shorter query
        let new_draft = engine.draft_content(ctx)?;
        if !new_draft.is_empty() {
            trigger_completion_popup(engine, ctx, &new_draft)?;
        } else {
            hide_completion_popup();
        }
    } else {
        // Draft empty: pass backspace to editor's text buffer
        editor_delete_char();
    }

    Ok(())
}
```

#### Delete (Delete After Cursor)

```rust
// Most editors handle forward delete in the text buffer, not draft
// Draft only tracks unfinalized text before cursor

fn handle_delete_key(
    engine: &mut DynamicContextualCompletionEngine<PathMapDictionary<()>>,
    ctx: ContextId
) -> Result<()> {
    // Check if cursor is at end of draft
    let draft = engine.draft_content(ctx)?;
    let cursor_at_end = editor_cursor_position() == editor_text_length();

    if cursor_at_end && !draft.is_empty() {
        // Rare case: delete at end of draft
        // Most editors won't trigger this; they delete ahead in text buffer
    }

    // Usually: editor handles delete key directly
    editor_delete_forward_char();

    Ok(())
}
```

### Word Completion

#### Accept Completion Suggestion

```rust
fn accept_completion(
    engine: &mut DynamicContextualCompletionEngine<PathMapDictionary<()>>,
    ctx: ContextId,
    selected_term: &str
) -> Result<()> {
    // Clear draft buffer
    let draft = engine.draft_content(ctx)?;
    for _ in 0..draft.chars().count() {
        engine.delete(ctx)?;
    }

    // Insert accepted completion into editor text buffer
    editor_insert_text(selected_term);

    // Finalize the term if it's new
    engine.finalize(ctx, selected_term)?;

    Ok(())
}
```

#### Tab Completion (Partial Acceptance)

```rust
fn tab_complete(
    engine: &mut DynamicContextualCompletionEngine<PathMapDictionary<()>>,
    ctx: ContextId
) -> Result<()> {
    let draft = engine.draft_content(ctx)?;
    let completions = engine.complete(ctx, &draft, 1)?;

    if let Some(first) = completions.first() {
        // Find longest common prefix among completions
        let common_prefix = longest_common_prefix(&completions);

        if common_prefix.len() > draft.len() {
            // Insert additional characters into draft
            let new_chars = &common_prefix[draft.len()..];
            for ch in new_chars.chars() {
                engine.insert_char(ctx, ch)?;
                editor_insert_char(ch);
            }

            // Update completion popup with narrowed results
            trigger_completion_popup(engine, ctx, &common_prefix)?;
        }
    }

    Ok(())
}

fn longest_common_prefix(completions: &[Completion]) -> String {
    if completions.is_empty() {
        return String::new();
    }

    let first = &completions[0].term;
    let mut prefix = String::new();

    for (i, ch) in first.chars().enumerate() {
        if completions.iter().all(|c| c.term.chars().nth(i) == Some(ch)) {
            prefix.push(ch);
        } else {
            break;
        }
    }

    prefix
}
```

## Usage Examples

### Example 1: Basic Incremental Typing

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;
use libdictenstein::pathmap::PathMapDictionary;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dict = PathMapDictionary::new();
    let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

    let root = engine.root_context();
    let ctx = engine.create_context(root)?;

    // Pre-populate with known variables
    engine.finalize(ctx, "counter")?;
    engine.finalize(ctx, "count")?;

    // User types "c"
    engine.insert_char(ctx, 'c')?;
    let completions = engine.complete(ctx, "c", 0)?;
    assert_eq!(completions.len(), 2);  // ["counter", "count"]

    // User types "o"
    engine.insert_char(ctx, 'o')?;
    let completions = engine.complete(ctx, "co", 0)?;
    assert_eq!(completions.len(), 2);  // Still ["counter", "count"]

    // User types "u"
    engine.insert_char(ctx, 'u')?;
    let completions = engine.complete(ctx, "cou", 0)?;
    assert_eq!(completions.len(), 2);  // Still ["counter", "count"]

    // User types "n"
    engine.insert_char(ctx, 'n')?;
    let completions = engine.complete(ctx, "coun", 0)?;
    assert_eq!(completions.len(), 2);  // Still ["counter", "count"]

    // User types "t"
    engine.insert_char(ctx, 't')?;
    let completions = engine.complete(ctx, "count", 0)?;
    assert_eq!(completions.len(), 2);  // ["count" (exact), "counter" (prefix)]

    Ok(())
}
```

**Use Case**: Real-time completion narrowing as user types.

### Example 2: Backspace Correction

```rust
fn backspace_correction() -> Result<()> {
    let dict = PathMapDictionary::new();
    let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

    let root = engine.root_context();
    let ctx = engine.create_context(root)?;

    engine.finalize(ctx, "variable")?;

    // User types "varaible" (typo)
    for ch in "varaible".chars() {
        engine.insert_char(ctx, ch)?;
    }

    let draft = engine.draft_content(ctx)?;
    assert_eq!(draft, "varaible");

    // No exact match due to typo
    let completions = engine.complete(ctx, &draft, 0)?;
    assert_eq!(completions.len(), 0);

    // But distance=1 finds it (1 substitution: 'a' → 'i')
    let completions = engine.complete(ctx, &draft, 1)?;
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].term, "variable");

    // User realizes typo, backspaces "ible"
    engine.delete(ctx)?;  // Remove 'e'
    engine.delete(ctx)?;  // Remove 'l'
    engine.delete(ctx)?;  // Remove 'b'
    engine.delete(ctx)?;  // Remove 'i'

    assert_eq!(engine.draft_content(ctx)?, "vara");

    // User types correct spelling "riable"
    for ch in "iable".chars() {
        engine.insert_char(ctx, ch)?;
    }

    assert_eq!(engine.draft_content(ctx)?, "variable");

    // Now exact match
    let completions = engine.complete(ctx, "variable", 0)?;
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].distance, 0);

    Ok(())
}
```

**Use Case**: Correcting typos with backspace.

### Example 3: Word Boundary Detection

```rust
fn word_boundary_finalization() -> Result<()> {
    let dict = PathMapDictionary::new();
    let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

    let root = engine.root_context();
    let ctx = engine.create_context(root)?;

    // Simulate typing: "let x = 10;"
    for ch in "x".chars() {
        engine.insert_char(ctx, ch)?;
    }

    assert_eq!(engine.draft_content(ctx)?, "x");

    // User types '=' (word boundary)
    // Finalize 'x' as a variable
    let draft = engine.draft_content(ctx)?;
    engine.finalize(ctx, &draft)?;

    // Draft should be cleared after finalization (in typical editor integration)
    // Note: engine.finalize() doesn't auto-clear; that's editor's responsibility
    // For demo, manually clear:
    engine.delete(ctx)?;

    assert_eq!(engine.draft_content(ctx)?, "");

    // Now 'x' is available for completion
    let completions = engine.complete(ctx, "x", 0)?;
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].term, "x");

    Ok(())
}
```

**Use Case**: Detecting variable declaration boundaries (spaces, '=', ';').

### Example 4: Multi-Line Editing

```rust
fn multiline_editing() -> Result<()> {
    let dict = PathMapDictionary::new();
    let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

    let root = engine.root_context();
    let fn_ctx = engine.create_context(root)?;

    // Line 1: "let alpha = 1;"
    for ch in "alpha".chars() {
        engine.insert_char(fn_ctx, ch)?;
    }
    engine.finalize(fn_ctx, "alpha")?;

    // Clear draft for next line
    for _ in 0..engine.draft_content(fn_ctx)?.len() {
        engine.delete(fn_ctx)?;
    }

    // Line 2: "let beta = 2;"
    for ch in "beta".chars() {
        engine.insert_char(fn_ctx, ch)?;
    }
    engine.finalize(fn_ctx, "beta")?;

    for _ in 0..engine.draft_content(fn_ctx)?.len() {
        engine.delete(fn_ctx)?;
    }

    // Line 3: User types "al" (should complete to "alpha")
    engine.insert_char(fn_ctx, 'a')?;
    engine.insert_char(fn_ctx, 'l')?;

    let completions = engine.complete(fn_ctx, "al", 0)?;
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].term, "alpha");

    // Line 3: User types "be" (should complete to "beta")
    engine.delete(fn_ctx)?;  // Remove 'l'
    engine.delete(fn_ctx)?;  // Remove 'a'

    engine.insert_char(fn_ctx, 'b')?;
    engine.insert_char(fn_ctx, 'e')?;

    let completions = engine.complete(fn_ctx, "be", 0)?;
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].term, "beta");

    Ok(())
}
```

**Use Case**: Tracking variables across multiple lines in the same scope.

### Example 5: Checkpoint-Based Undo

```rust
fn checkpoint_undo() -> Result<()> {
    let dict = PathMapDictionary::new();
    let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

    let root = engine.root_context();
    let ctx = engine.create_context(root)?;

    // Checkpoint before typing
    let cp1 = engine.checkpoint(ctx)?;

    // User types "wrong"
    for ch in "wrong".chars() {
        engine.insert_char(ctx, ch)?;
    }

    assert_eq!(engine.draft_content(ctx)?, "wrong");

    // Checkpoint after "wrong"
    let cp2 = engine.checkpoint(ctx)?;

    // User types "word"
    for ch in "word".chars() {
        engine.insert_char(ctx, ch)?;
    }

    assert_eq!(engine.draft_content(ctx)?, "wrongword");

    // Undo to cp2 (removes "word")
    engine.restore(ctx, cp2)?;
    assert_eq!(engine.draft_content(ctx)?, "wrong");

    // Undo to cp1 (removes "wrong")
    engine.restore(ctx, cp1)?;
    assert_eq!(engine.draft_content(ctx)?, "");

    Ok(())
}
```

**Use Case**: Editor's undo/redo integration using checkpoints.

### Example 6: Unicode Incremental Editing

```rust
fn unicode_incremental() -> Result<()> {
    let dict = PathMapDictionary::new();
    let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

    let root = engine.root_context();
    let ctx = engine.create_context(root)?;

    // Pre-finalize Japanese variable: "変数"
    engine.finalize(ctx, "変数")?;

    // User types first character: "変"
    engine.insert_char(ctx, '変')?;
    assert_eq!(engine.draft_content(ctx)?, "変");

    // Query with partial Japanese input
    let completions = engine.complete(ctx, "変", 0)?;
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].term, "変数");

    // User types second character: "数"
    engine.insert_char(ctx, '数')?;
    assert_eq!(engine.draft_content(ctx)?, "変数");

    // Exact match now
    let completions = engine.complete(ctx, "変数", 0)?;
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].distance, 0);

    // Backspace removes "数" (single character, not bytes!)
    engine.delete(ctx)?;
    assert_eq!(engine.draft_content(ctx)?, "変");

    Ok(())
}
```

**Use Case**: Correct Unicode character handling (not byte-level operations).

### Example 7: Tab Completion with Common Prefix

```rust
fn tab_completion_example() -> Result<()> {
    let dict = PathMapDictionary::new();
    let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);

    let root = engine.root_context();
    let ctx = engine.create_context(root)?;

    engine.finalize(ctx, "getUserName")?;
    engine.finalize(ctx, "getUserId")?;
    engine.finalize(ctx, "getUserEmail")?;

    // User types "getU"
    for ch in "getU".chars() {
        engine.insert_char(ctx, ch)?;
    }

    let draft = engine.draft_content(ctx)?;
    let completions = engine.complete(ctx, &draft, 0)?;
    assert_eq!(completions.len(), 3);

    // Find longest common prefix: "getUser"
    let common = longest_common_prefix(&completions);
    assert_eq!(common, "getUser");

    // Tab completion: insert "ser" to complete common prefix
    for ch in "ser".chars() {
        engine.insert_char(ctx, ch)?;
    }

    assert_eq!(engine.draft_content(ctx)?, "getUser");

    // Now only 3 results remain (still all match)
    let completions = engine.complete(ctx, "getUser", 0)?;
    assert_eq!(completions.len(), 3);

    // User types "N" to disambiguate
    engine.insert_char(ctx, 'N')?;

    let completions = engine.complete(ctx, "getUserN", 0)?;
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].term, "getUserName");

    Ok(())
}

fn longest_common_prefix(completions: &[Completion]) -> String {
    if completions.is_empty() {
        return String::new();
    }

    let first = &completions[0].term;
    let mut prefix = String::new();

    for (i, ch) in first.chars().enumerate() {
        if completions.iter().all(|c| c.term.chars().nth(i) == Some(ch)) {
            prefix.push(ch);
        } else {
            break;
        }
    }

    prefix
}
```

**Use Case**: Tab key expanding to longest common prefix among completions.

### Example 8: Rapid Typing with Debouncing

```rust
use std::time::{Duration, Instant};

struct DebouncedEditor {
    engine: DynamicContextualCompletionEngine<PathMapDictionary<()>>,
    ctx: ContextId,
    last_query: Instant,
    debounce_duration: Duration,
    pending_query: bool,
}

impl DebouncedEditor {
    fn new() -> Result<Self> {
        let dict = PathMapDictionary::new();
        let mut engine = DynamicContextualCompletionEngine::from_pathmap_dictionary(dict);
        let ctx = engine.create_context(engine.root_context())?;

        Ok(Self {
            engine,
            ctx,
            last_query: Instant::now(),
            debounce_duration: Duration::from_millis(150),
            pending_query: false,
        })
    }

    fn on_keystroke(&mut self, ch: char) -> Result<()> {
        // Always insert immediately (no lag in text buffer)
        self.engine.insert_char(self.ctx, ch)?;

        // Mark that a query is pending
        self.pending_query = true;

        Ok(())
    }

    fn tick(&mut self) -> Result<Option<Vec<Completion>>> {
        if self.pending_query &&
           self.last_query.elapsed() >= self.debounce_duration
        {
            // Debounce period elapsed: execute query
            let draft = self.engine.draft_content(self.ctx)?;
            let completions = self.engine.complete(self.ctx, &draft, 1)?;

            self.last_query = Instant::now();
            self.pending_query = false;

            Ok(Some(completions))
        } else {
            Ok(None)
        }
    }
}

fn main() -> Result<()> {
    let mut editor = DebouncedEditor::new()?;

    // Pre-populate
    editor.engine.finalize(editor.ctx, "variable")?;

    // Simulate rapid typing: "var"
    editor.on_keystroke('v')?;
    editor.on_keystroke('a')?;
    editor.on_keystroke('r')?;

    // Immediately after typing: no query yet (debouncing)
    let result = editor.tick()?;
    assert!(result.is_none());

    // Wait 150ms
    std::thread::sleep(Duration::from_millis(150));

    // Now query executes
    let result = editor.tick()?;
    assert!(result.is_some());

    let completions = result.unwrap();
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].term, "variable");

    Ok(())
}
```

**Use Case**: Debouncing queries to avoid excessive computation during rapid typing.

## Performance Optimization

### 1. Query Throttling

Limit completion queries to reduce CPU usage:

```rust
struct ThrottledEngine {
    last_query: Instant,
    min_interval: Duration,
}

impl ThrottledEngine {
    fn should_query(&mut self) -> bool {
        if self.last_query.elapsed() >= self.min_interval {
            self.last_query = Instant::now();
            true
        } else {
            false
        }
    }
}
```

### 2. Minimum Prefix Length

Don't query until user has typed enough characters:

```rust
fn should_trigger_completion(draft: &str) -> bool {
    draft.len() >= 2  // Minimum 2 characters
}
```

### 3. Caching Previous Results

Cache results for identical queries:

```rust
struct CachedCompletionEngine {
    engine: DynamicContextualCompletionEngine<PathMapDictionary<()>>,
    cache: HashMap<(ContextId, String, usize), Vec<Completion>>,
}

impl CachedCompletionEngine {
    fn complete_cached(&mut self, ctx: ContextId, query: &str, dist: usize)
        -> Result<Vec<Completion>>
    {
        let key = (ctx, query.to_string(), dist);

        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached.clone());
        }

        let results = self.engine.complete(ctx, query, dist)?;
        self.cache.insert(key, results.clone());

        Ok(results)
    }
}
```

### 4. Incremental Result Filtering

Filter previous results instead of re-querying:

```rust
struct IncrementalFilterEngine {
    last_results: Vec<Completion>,
    last_query: String,
}

impl IncrementalFilterEngine {
    fn complete_incremental(&mut self, engine: &Engine, ctx: ContextId,
                            query: &str, dist: usize)
        -> Result<Vec<Completion>>
    {
        if query.starts_with(&self.last_query) {
            // New query is extension of previous: filter results
            let filtered: Vec<_> = self.last_results.iter()
                .filter(|c| {
                    let distance = levenshtein(&c.term, query);
                    distance <= dist
                })
                .cloned()
                .collect();

            self.last_results = filtered.clone();
            self.last_query = query.to_string();

            Ok(filtered)
        } else {
            // New query unrelated: full query
            let results = engine.complete(ctx, query, dist)?;
            self.last_results = results.clone();
            self.last_query = query.to_string();

            Ok(results)
        }
    }
}
```

## Editor Integration Patterns

### VSCode Extension

```typescript
import * as vscode from 'vscode';

// Rust engine bindings (via NAPI-rs, WebAssembly, etc.)
const engine = require('./liblevenshtein_bindings');

export function activate(context: vscode.ExtensionContext) {
    const provider = new ContextualCompletionProvider();

    const registration = vscode.languages.registerCompletionItemProvider(
        { scheme: 'file', language: 'rust' },
        provider,
        '.'  // Trigger on '.'
    );

    context.subscriptions.push(registration);
}

class ContextualCompletionProvider implements vscode.CompletionItemProvider {
    async provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position
    ): Promise<vscode.CompletionItem[]> {
        const ctx = getContextAtPosition(document, position);
        const linePrefix = document.lineAt(position).text.substr(0, position.character);
        const query = extractCurrentWord(linePrefix);

        const completions = engine.complete(ctx, query, 1);

        return completions.map(c => {
            const item = new vscode.CompletionItem(c.term, vscode.CompletionItemKind.Variable);
            item.detail = `distance: ${c.distance}`;
            item.sortText = `${c.distance}_${c.term}`;
            return item;
        });
    }
}
```

### Vim Plugin (Lua)

```lua
local M = {}
local ffi = require('ffi')

-- Load Rust library (via FFI)
local lib = ffi.load('liblevenshtein.so')

ffi.cdef[[
    typedef struct Engine Engine;
    Engine* engine_new();
    void engine_insert_char(Engine* e, uint32_t ctx, char ch);
    char** engine_complete(Engine* e, uint32_t ctx, const char* query, size_t dist);
]]

function M.setup()
    local engine = lib.engine_new()

    -- Hook into insert mode
    vim.api.nvim_create_autocmd("TextChangedI", {
        callback = function()
            local ctx = get_current_context()
            local query = vim.fn.expand('<cWORD>')

            local results = lib.engine_complete(engine, ctx, query, 1)
            show_completion_popup(results)
        end
    })
end

return M
```

### Emacs Mode (elisp)

```elisp
(require 'company)

(defun liblevenshtein-complete (prefix)
  "Complete PREFIX using liblevenshtein engine."
  (let* ((ctx (liblevenshtein--current-context))
         (results (liblevenshtein--ffi-complete ctx prefix 1)))
    results))

(defun liblevenshtein-company-backend (command &optional arg &rest ignored)
  "Company backend for liblevenshtein completion."
  (interactive (list 'interactive))
  (case command
    (interactive (company-begin-backend 'liblevenshtein-company-backend))
    (prefix (company-grab-symbol))
    (candidates (liblevenshtein-complete arg))))

(add-to-list 'company-backends 'liblevenshtein-company-backend)
```

## Benchmarking Results

Based on real-world typing patterns (180 WPM = ~15 chars/sec):

| Operation | Time | Budget (16ms frame) | % Used |
|-----------|------|---------------------|--------|
| insert_char | 3.8µs | 16ms | 0.02% |
| delete_char | 2.1µs | 16ms | 0.01% |
| complete(dist=1) | 11.5µs | 16ms | 0.07% |
| UI render | ~15ms | 16ms | 93.75% |
| **Total** | ~15.02ms | 16ms | 93.85% ✓ |

**Conclusion**: Engine operations are negligible (<1% of frame budget). UI rendering dominates performance.

## Testing Coverage

For incremental editing integration, test:

1. **Single Character**: Insert/delete works correctly
2. **Rapid Typing**: 15+ chars/sec handled without lag
3. **Backspace**: Deleting characters updates drafts correctly
4. **Unicode**: Multi-byte characters (emoji, CJK) handled correctly
5. **Word Boundaries**: Space, '=', ';', '\n' trigger finalization
6. **Tab Completion**: Longest common prefix expansion
7. **Checkpoints**: Undo/redo restores correct state
8. **Multi-Line**: Variables tracked across lines in same scope
9. **Debouncing**: Queries throttled during rapid typing
10. **Memory**: No leaks after 10K insert/delete operations

See also: [../examples/lsp-completion.rs](../examples/lsp-completion.rs) for full editor integration.

## Related Documentation

- **[Code Completion](./code-completion.md)** - IDE integration patterns, LSP design
- **[Completion Engine](../implementation/completion-engine.md)** - Engine API reference
- **[Draft Buffer](../implementation/draft-buffer.md)** - Character-level buffer implementation
- **[Checkpoint System](../implementation/checkpoint-system.md)** - Undo/redo support
- **[Context Tree](../implementation/context-tree.md)** - Scope hierarchy

## References

1. [VSCode Extension API - Completion Provider](https://code.visualstudio.com/api/references/vscode-api#CompletionItemProvider)
2. [Vim's Complete-functions](https://vimhelp.org/insert.txt.html#complete-functions)
3. [Emacs Company Mode](https://company-mode.github.io/)
4. [Unicode Text Segmentation (UAX #29)](https://unicode.org/reports/tr29/)
5. Benchmark data from `benches/contextual_benchmarks.rs`
