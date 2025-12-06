# WFST-Based Text Normalization Documentation

**Last Updated**: 2025-11-21
**Status**: Design documentation for extending liblevenshtein-rust with WFST capabilities

---

## Overview

This documentation describes a **three-tier hybrid architecture** (FST + CFG + Neural) for text normalization, specifically designed to leverage liblevenshtein-rust's unique planned capabilities:

1. **NFA phonetic regular expressions** - Express complex phonetic patterns
2. **CFG grammar correction** - Handle syntax errors beyond FST capability
3. **Formally verified phonetic rules** - Coq-proven orthography transformations

**Key Differentiator**: Unlike industry systems (NVIDIA NeMo, Google Sparrowhawk) that use only FST + Neural, liblevenshtein-rust adds a deterministic CFG layer for grammar correction without neural networks.

---

## Documents in This Collection

### 1. [architecture.md](architecture.md) (~1200 lines)
**Main design document** covering the complete system architecture.

**Contents**:
- Executive summary of three-tier approach
- Six-layer pipeline design (Tokenization → Phonetic → Levenshtein → CFG → Neural → Post-processing)
- Composition operators (FST ∘ FST, NFA ∩ FST, CFG × FST, Lattice parsing)
- Weight schemes (tropical semiring)
- Lattice representation
- Integration with liblevenshtein-rust (Rust code examples)
- Comparison with industry systems (NVIDIA, Google, MoNoise)
- Performance characteristics (latency, memory, scalability)
- Deployment modes (Fast/Balanced/Accurate)

**Target Audience**: System architects, developers implementing WFST pipelines

---

### 2. [cfg_grammar_correction.md](cfg_grammar_correction.md) (~1400 lines)
**Deep dive into CFG-based grammatical error correction**.

**Contents**:
- Theoretical foundation (Chomsky hierarchy, formal CFG definitions)
- Why CFG is necessary (pumping lemma proof, linguistic structures)
- Error grammar formalism (well-formed vs error productions)
- Grammar categories (morphosyntax, phrase structure, articles, auxiliaries, negation)
- Probabilistic CFG for disambiguation (PCFG, MLE, Viterbi parsing)
- Chart parsing algorithms (CYK O(n³), Earley O(n²) average)
- Integration with FST lattices (lattice-aware parsing)
- **Lattice parsing efficiency analysis** (exponential candidate problem, complexity comparison)
- **Performance measurements** (6× speedup, 4× memory reduction)
- Implementation strategy (Rust data structures, Earley parser code)
- Advanced topics (term-level vs character-level lattices, k-best extraction)
- Benchmarks (CoNLL-2014, JFLEG, BEA-2019)

**Target Audience**: Developers implementing CFG parsers, researchers in GEC

---

### 3. [lattice_parsing.md](lattice_parsing.md) (~1050 lines)
**Complete pedagogical guide to lattice parsing for efficient CFG integration**.

**Contents**:
- Introduction (what is lattice parsing, why it matters, historical context)
- The exponential candidate explosion problem
- Compact lattice representation (DAGs, mathematical foundations)
- From strings to lattices (progressive examples)
- Lattice parsing algorithm (modified Earley parser)
- Parse forest output (shared structure representation)
- Integration with liblevenshtein-rust (Rust API examples)
- Performance analysis (benchmarks, complexity, speedup factors)
- Worked examples (complete parse traces, comparisons)
- Implementation details (topological sort, path counting, builders)

**Key Features**:
- **20+ ASCII diagrams** illustrating lattice structures and parse forests
- **15+ code examples** showing Rust implementation
- **Complete worked examples** with step-by-step Earley traces
- **Benchmark data** from 1000 real user queries

**Target Audience**: Developers implementing lattice parsers, anyone wanting deep understanding of the technique

---

### 4. [lattice_data_structures.md](lattice_data_structures.md) (~550 lines)
**Technical reference for data structures used in lattice parsing**.

**Contents**:
- Core data structures (Lattice, Node, Edge with complete Rust definitions)
- Graph representation (adjacency list vs. adjacency matrix trade-offs)
- Chart data structures (EarleyChart, EarleyState)
- Parse forest representation (packed forests, k-best extraction)
- Memory layout and optimization (SmallVec, Arc sharing, bit packing)
- Algorithms (topological sort, path counting, cycle detection)
- API reference (LatticeBuilder, PathIterator)

**Key Features**:
- **Complete Rust type definitions** ready for implementation
- **Memory optimization techniques** (SmallVec, Arc, bit packing)
- **Complexity analysis** for all data structures
- **Algorithm pseudocode** with Rust implementations

**Target Audience**: Systems programmers, performance engineers, implementers

---

### 5. [references/papers.md](references/papers.md) (~600 lines)
**Comprehensive bibliography with 35+ cited papers**.

**Contents**:
- Hybrid WFST + Neural architectures (8 papers)
- CFG-based grammatical error correction (4 papers)
- Phonetic normalization and Levenshtein automata (3 papers)
- Neural language models for text normalization (8 papers)
- Noisy text normalization (SMS/social media) (5 papers)
- Lattice rescoring and n-best reranking (3 papers)
- Production systems and tools (4 papers)
- Benchmarks and shared tasks (6 datasets)
- Theoretical foundations (2 papers)

**Key Features**:
- Full BibTeX citations
- arXiv IDs with direct PDF links
- ACL Anthology IDs for conference papers
- Open access information
- Recommended reading order

**Target Audience**: Researchers, graduate students, anyone wanting deep theoretical understanding

---

## Key Research Findings

### What Replaced N-grams?
**Answer**: Transformers (BERT, T5, GPT) for contextual understanding.

**BUT**: N-grams still used in hybrid systems for efficiency.

**Production Reality**: WFST + neural LM hybrid (not pure neural).

**Quote from NVIDIA NeMo (arXiv:2104.05055)**:
> "Low tolerance towards unrecoverable errors is the main reason why most ITN systems in production are still largely rule-based using WFSTs"

---

### Why WFSTs Still Matter (2025)
1. **Unrecoverable errors**: Neural models hallucinate → FSTs enforce hard constraints
2. **Latency**: WFST compilation enables fast inference (<20ms)
3. **Interpretability**: Rule-based logic is debuggable, neural is black-box
4. **Data efficiency**: WFSTs don't require labeled training data
5. **Deterministic**: Same input always yields same output

---

### Grammar Correction: FST vs CFG vs Neural

| Error Type | FST | CFG | Neural | Example |
|------------|-----|-----|--------|---------|
| Spelling | ✅ O(n) | - | - | "teh" → "the" |
| Phonetic | ✅ O(n) | - | - | "fone" → "phone" |
| Article selection | ❌ | ✅ O(n³) | ✅ O(n²) | "a apple" → "an apple" |
| Subject-verb agreement | ❌ | ✅ O(n³) | ✅ O(n²) | "they was" → "they were" |
| Nested phrases | ❌ | ✅ O(n³) | ✅ O(n²) | Balanced parentheses |
| Semantic disambiguation | ❌ | ❌ | ✅ O(n²) | "bank" = river vs financial |

**Key Insight**: FSTs cannot handle grammar (require counting, nesting). CFG fills the gap between FST and Neural.

---

## liblevenshtein-rust's Unique Advantage

### Industry Systems (2025)

**NVIDIA NeMo**:
- Architecture: FST + Neural
- Limitation: No CFG layer (must use neural for grammar)
- Open Source: ✅ (Python/C++)

**Google Sparrowhawk**:
- Architecture: FST only
- Limitation: Text normalization only, no grammar correction
- Open Source: ✅ (C++)

**MoNoise** (SOTA Lexical Normalization):
- Architecture: Pure Neural (seq2seq)
- Limitation: Prone to hallucination, requires large training data
- Open Source: ✅ (Python)

### liblevenshtein-rust (Planned)

**Architecture**: FST + NFA + CFG + Neural
- ✅ **Regular layer** (FST/NFA): Spelling, phonetic, morphology
- ✅ **Context-free layer** (CFG): Grammar correction, syntax
- ✅ **Neural layer** (optional): Semantic disambiguation

**Unique Features**:
1. **Only system with FST + CFG + Neural three-tier architecture**
2. **Formally verified phonetic rules** (Coq proofs)
3. **Deterministic symbolic layers** (Tiers 1-2 reproducible)
4. **Rust native performance** (zero-cost abstractions, memory safety)
5. **Composable architecture** (NFA ∩ FST ∩ CFG)

**Target Applications**:
- **Deterministic** corrections required (medical, legal)
- **Low-latency** systems (mobile, embedded)
- **Resource-constrained** environments (no GPU)
- **Safety-critical** systems (formal verification)
- **LLM preprocessing**: Clean user input before GPT/Claude/Llama (15-50ms overhead)
- **LLM postprocessing**: Validate generated text for grammar (5-20ms overhead)
- **Hybrid workflows**: Symbolic-first with neural fallback (95% symbolic, 5% LLM)

---

## Application to Programming Languages

While this design focuses on **natural language** text normalization (SMS, chat, search queries), the three-tier hybrid architecture (FST → CFG → Neural) is **equally applicable to programming language correction**.

### Relationship to Grammar Correction Design

This WFST design shares the same architectural foundation with the **Programming Language Grammar Correction** design ([`../design/grammar-correction/`](../design/grammar-correction/)), which targets Rholang and other process calculus languages.

**Architectural Mapping**:
```
WFST Text Normalization          Programming Language Correction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tier 1: FST/NFA (Phonetic)  ←→  Layer 1: Levenshtein (Lexical)
Tier 2: CFG (Grammar)        ←→  Layer 2: Tree-sitter (Syntax)
Tier 3: Neural/LLM          ←→  Layers 3-5: Type + SMT + Process
```

### Code Correction Use Cases

**1. LLM-Assisted Code Generation** (preprocessing + postprocessing)
```rust
// Preprocessing: Clean user's buggy code before Copilot
User input: "func prnt_msg(x string) { fmt.Prntln(x) }"
                 ^^^^typo                   ^^^^^typo
Corrected:  "func print_message(x string) { fmt.Println(x) }"
→ Send to Copilot → Better suggestions

// Postprocessing: Validate Copilot's generated code
Copilot output: "func calculate() { return x + y }"  // x, y undefined
CFG parse: ✅ Syntactically correct
Type check: ❌ Undefined variables
→ Apply semantic repair (suggest similar names in scope)
```

**2. IDE Real-Time Feedback** (deployment modes)
- **Fast Mode** (<20ms): Keystroke feedback (Lexical typos only)
- **Balanced Mode** (<200ms): Save action (Lexical + Syntax + Type check)
- **Accurate Mode** (<2s): Fix All (Full pipeline + process verification)

**3. SQL/JSON/YAML Correction**
- Apply CFG validation to structured data formats
- Catch syntax errors in configuration files (mismatched braces, trailing commas)
- Validate generated code from LLMs

### What Programming Languages Offer WFST

**1. Process Calculus Verification** (from Layer 5)
- Session type checking for distributed systems
- Deadlock detection for concurrent protocols
- Race condition analysis

**2. SMT-Based Semantic Repair** (from Layer 4)
- Constraint solving for complex semantic disambiguation
- Z3 integration for natural language ambiguity resolution

**3. Incremental Parsing** (Tree-sitter)
- Real-time text correction in editors
- Faster feedback for interactive applications

### See Also

For comprehensive guide on applying WFST to programming languages:
- [`programming_language_applications.md`](programming_language_applications.md) - Detailed guide with examples (Python, JavaScript, Rholang, SQL, YAML)
- [`../design/grammar-correction/README.md`](../design/grammar-correction/README.md) - Programming language design overview
- [`../design/grammar-correction/MAIN_DESIGN.md`](../design/grammar-correction/MAIN_DESIGN.md) - Complete system design for code correction

---

## Implementation Roadmap

### Phase 1: Lattice Output (Foundation)
- Modify transducer to return lattices instead of strings
- Implement n-best candidate extraction
- Add scoring framework (edit distance, phonetic, total)

### Phase 2: NFA Phonetic Regex
- Implement Thompson's construction for regex → NFA
- Add intersection operator (NFA × FST)
- Integrate with existing phonetic rules

### Phase 3: Weighted Transitions
- Add configurable cost functions
- Implement phonetic-aware substitution costs
- Benchmark performance impact

### Phase 4: CFG Grammar Correction
- Implement Earley parser
- Define error grammar formalism
- Build example grammar for common errors

### Phase 5: Neural LM Integration
- Define LanguageModel trait
- Add BERT integration (via ONNX or Python binding)
- Implement lattice rescoring

### Phase 6: Production Deployment
- Export to OpenFST FAR format
- Optimize for latency (caching, lazy evaluation)
- Add deployment modes (Fast/Balanced/Accurate)

---

## Extended Architecture

This WFST design is extended by the **MeTTaIL correction architecture**, which adds additional layers for conversational systems and LLM agent integration:

### Extended Layers

| Layer | Components | Purpose |
|-------|------------|---------|
| **Dialogue Context** | Turn Tracker, Entity Registry, Topic Graph | Multi-turn conversation tracking |
| **Pragmatic Reasoning** | Speech Act Classifier, Implicature Resolver | Intent understanding |
| **LLM Integration** | Preprocessor, Postprocessor, Hallucination Detector | Agent support |
| **Agent Learning** | Feedback Collection, Pattern Learning | Adaptive personalization |

### Related Documentation

**Extended Correction Architecture**:
- [Architecture Overview](../mettail/correction-wfst/01-architecture-overview.md) - Extended 6-layer architecture
- [Tier 1: Lexical Correction](../mettail/correction-wfst/02-tier1-lexical-correction.md) - liblevenshtein integration
- [Tier 2: Syntactic Validation](../mettail/correction-wfst/03-tier2-syntactic-validation.md) - MORK CFG validation
- [Tier 3: Semantic Type Checking](../mettail/correction-wfst/04-tier3-semantic-type-checking.md) - MeTTaIL/OSLF types
- [Data Flow](../mettail/correction-wfst/05-data-flow.md) - End-to-end pipeline
- [Integration Possibilities](../mettail/correction-wfst/06-integration-possibilities.md) - Use cases

**Extended Layer Documentation**:
- [Dialogue Context Layer](../mettail/dialogue/README.md) - Coreference resolution and topic tracking
- [LLM Integration Layer](../mettail/llm-integration/README.md) - Prompt preprocessing and response validation
- [Agent Learning Layer](../mettail/agent-learning/README.md) - Feedback integration and personalization

**Integration Implementation**:
- [MORK Integration](../integration/mork/README.md) - Phases A-D implementation details
- [PathMap Integration](../integration/pathmap/README.md) - Shared storage layer and schemas

---

## Benchmarks and Evaluation

### Datasets
- **LexNorm**: Standard lexical normalization benchmark
- **W-NUT 2015**: Twitter normalization (2,577 tweets)
- **W-NUT 2021**: Multilingual (12 languages)
- **CoNLL-2014**: Grammatical error correction (62 essays)
- **JFLEG**: Fluency-extended GEC (1,511 sentences)
- **BEA-2019**: GEC shared task (Write & Improve, LOCNESS)

### Metrics
- **Accuracy**: Percentage of correctly normalized tokens
- **Precision/Recall/F0.5**: For error detection/correction
- **GLEU**: Generalized Language Evaluation Understanding
- **Latency**: End-to-end processing time (ms)

### Baselines
- Pure Levenshtein (edit distance ≤ n)
- Pure BERT (masked LM)
- NVIDIA NeMo (FST + Neural)
- MoNoise (Neural seq2seq, SOTA on LexNorm)

---

## Tools and Resources

### WFST Libraries
- **OpenFST**: http://www.openfst.org/
- **Pynini**: https://www.opengrm.org/twiki/bin/view/GRM/Pynini
- **Sparrowhawk**: https://github.com/google/sparrowhawk
- **NVIDIA NeMo**: https://github.com/NVIDIA/NeMo-text-processing

### Parsing Libraries
- **NLTK**: Python library with CYK, Earley parsers
- **spaCy**: Dependency parsing
- **Stanford CoreNLP**: Java-based CFG parsing

### Benchmarks
- **LexNorm**: https://paperswithcode.com/task/lexical-normalization
- **W-NUT**: https://noisy-text.github.io/
- **CoNLL-2014 GEC**: https://www.comp.nus.edu.sg/~nlp/conll14st.html
- **BEA-2019 GEC**: https://www.cl.cam.ac.uk/research/nl/bea2019st/
- **JFLEG**: https://github.com/keisks/jfleg

---

## Quick Start Guide

### For Researchers
1. Read [architecture.md](architecture.md) for system overview
2. Read [lattice_parsing.md](lattice_parsing.md) for complete pedagogical guide
3. Read [references/papers.md](references/papers.md) for theoretical foundations
4. Read [cfg_grammar_correction.md](cfg_grammar_correction.md) for CFG deep dive

**Estimated time**: 8-10 hours for thorough understanding

### For Developers
1. Read [architecture.md](architecture.md), sections:
   - Integration with liblevenshtein-rust
   - Implementation Roadmap
   - Lattice Parsing: Efficient CFG Integration
   - Rust code examples
2. Study [lattice_parsing.md](lattice_parsing.md), sections:
   - Lattice Parsing Algorithm
   - Integration with liblevenshtein-rust
   - Implementation Details
3. Reference [lattice_data_structures.md](lattice_data_structures.md) for:
   - Complete Rust type definitions
   - Memory optimization techniques
   - Algorithm implementations
4. Study [cfg_grammar_correction.md](cfg_grammar_correction.md), section:
   - Implementation Strategy (Rust data structures)
   - Lattice Parsing Efficiency Analysis

**Estimated time**: 5-6 hours to get started

### For Product Managers
1. Read [architecture.md](architecture.md), sections:
   - Executive Summary
   - Comparison with Industry Systems
   - Deployment Modes
2. Skim [references/papers.md](references/papers.md) for competitive landscape

**Estimated time**: 1-2 hours for high-level understanding

---

## Document Statistics

**Total Documentation**: ~6,000 lines across 5 main documents + 1 README

**Line Counts**:
- architecture.md: ~2,050 lines (updated with lattice + LLM integration sections)
- cfg_grammar_correction.md: ~1,900 lines (updated with efficiency analysis + LLM use cases)
- lattice_parsing.md: ~1,050 lines (NEW - pedagogical guide)
- lattice_data_structures.md: ~550 lines (NEW - technical reference)
- references/papers.md: ~600 lines
- README.md: ~450 lines (this file, updated with LLM applications)

**Papers Cited**: 35+ open access papers (2002-2025)
- arXiv preprints: 20+
- ACL Anthology: 10+
- Journals: 5+
- Patents: 1

**Code Examples**: 75+ Rust pseudocode snippets (lattice parsing + LLM integration)

**Diagrams**: 40+ ASCII diagrams and flowcharts (doubled with lattice docs)

**Tables**: 30+ comparison tables (complexity, performance, LLM integration patterns)

---

## Contributing

### Missing Documents (To Be Created)

#### nfa_phonetic_regex.md (~400-500 lines)
**Contents**:
- Regex syntax for phonetic patterns
- NFA construction (Thompson's algorithm)
- Composition with Levenshtein automata
- Performance considerations

#### implementation_guide.md (~600-700 lines)
**Contents**:
- Levenshtein automaton enhancements
- NFA phonetic regex compiler
- Composition strategies (sequential, parallel, guided)
- Deployment modes (Fast/Balanced/Accurate)
- Benchmarking methodology

#### limitations.md (~400-500 lines)
**Contents**:
- What Levenshtein automata CAN do
- What CFG adds (new capabilities)
- What still requires Neural models
- FST vs CFG vs CSG hierarchy
- Parsing complexity trade-offs
- When to use each layer

### How to Contribute

1. **Feedback**: Open issues on GitHub for errors, omissions, or suggestions
2. **Examples**: Contribute Rust implementations of algorithms
3. **Benchmarks**: Run experiments and share results
4. **Use Cases**: Document real-world applications

---

## License

This documentation is part of the liblevenshtein-rust project.

**License**: Apache 2.0 (same as liblevenshtein-rust)

**Citation**: If you use this research in academic work, please cite:
```bibtex
@misc{liblevenshtein-rust-wfst-docs,
  title={WFST-Based Text Normalization Architecture for liblevenshtein-rust},
  year={2025},
  howpublished={\url{https://github.com/universal-automata/liblevenshtein-rust/docs/wfst/}}
}
```

---

## Acknowledgments

This documentation synthesizes research from:
- **NVIDIA NeMo team**: Hybrid WFST + Neural architecture
- **Google Sparrowhawk**: Production FST text normalization
- **Academic community**: 35+ cited papers from ACL, NAACL, EMNLP, Interspeech, etc.

Special thanks to the open access movement (arXiv, ACL Anthology) for making research freely available.

---

## Contact

For questions about this documentation:
- **GitHub Issues**: https://github.com/universal-automata/liblevenshtein-rust/issues
- **Discussions**: https://github.com/universal-automata/liblevenshtein-rust/discussions

For questions about liblevenshtein-rust implementation:
- See main project README

---

**Last Updated**: 2025-12-06
**Recent Updates**:
- **NEW**: Added Extended Architecture section with cross-references to MeTTaIL correction docs
- **NEW**: Added links to extended layers (Dialogue, LLM Integration, Agent Learning)
- **NEW**: Added cross-references to MORK/PathMap integration documentation
- Added comprehensive lattice parsing documentation (lattice_parsing.md, lattice_data_structures.md)
- Expanded architecture.md with lattice parsing integration section
- Expanded cfg_grammar_correction.md with efficiency analysis and benchmarks
- Added LLM integration section to architecture.md (~640 lines)
- Added LLM use cases to cfg_grammar_correction.md (~340 lines)
- Updated target applications with LLM preprocessing/postprocessing/hybrid workflows

**Next Update**: After implementing Phase 1 (Lattice Output) and example code (lattice_parsing_demo.rs)
