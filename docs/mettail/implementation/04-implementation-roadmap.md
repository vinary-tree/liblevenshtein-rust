# Implementation Roadmap

This document outlines the recommended path for implementing full semantic type
checking in MeTTaIL, progressing from the current foundation to complete OSLF
support.

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Gph-enriched Lawvere Foundation](#phase-1-gph-enriched-lawvere-foundation)
3. [Phase 2: Predicate Infrastructure](#phase-2-predicate-infrastructure)
4. [Phase 3: Behavioral Types](#phase-3-behavioral-types)
5. [Phase 4: Full OSLF](#phase-4-full-oslf)
6. [Phase 5: WFST Integration](#phase-5-wfst-integration)
7. [Phase 6: Rholang Integration](#phase-6-rholang-integration)
8. [Phase 7: Dialogue Context](#phase-7-dialogue-context)
9. [Phase 8: LLM Integration](#phase-8-llm-integration)
10. [Phase 9: Agent Learning](#phase-9-agent-learning)
11. [Dependencies](#dependencies)
12. [Validation Criteria](#validation-criteria)

---

## Overview

### The Layered Approach

We recommend a **layered implementation** that builds incrementally:

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 9: Agent Learning                                        │
│  - Feedback collection and signal detection                     │
│  - Pattern learning and generalization                          │
│  - Online weight updates and threshold adaptation               │
├─────────────────────────────────────────────────────────────────┤
│  Phase 8: LLM Integration                                       │
│  - Prompt preprocessing (correction + context injection)        │
│  - Response postprocessing (validation + hallucination detect)  │
│  - Knowledge base verification and coherence checking           │
├─────────────────────────────────────────────────────────────────┤
│  Phase 7: Dialogue Context                                      │
│  - DialogueState and turn tracking                              │
│  - Entity registry and coreference resolution                   │
│  - Topic graph and discourse structure                          │
├─────────────────────────────────────────────────────────────────┤
│  Phase 6: Rholang Integration                                   │
│  - Type checker in compiler pipeline                            │
│  - Behavioral type enforcement                                  │
│  - Gradual typing migration                                     │
├─────────────────────────────────────────────────────────────────┤
│  Phase 5: WFST Integration                                      │
│  - Three-tier correction pipeline                               │
│  - FuzzySource trait for PathMap/MORK                           │
│  - Semantic type filtering for corrections                      │
├─────────────────────────────────────────────────────────────────┤
│  Phase 4: Full OSLF                                             │
│  - Presheaf construction                                        │
│  - Internal language extraction                                 │
│  - Refined binding types                                        │
├─────────────────────────────────────────────────────────────────┤
│  Phase 3: Behavioral Types                                      │
│  - Internal graph representation                                │
│  - Step modalities (F!, F*)                                     │
│  - Reachability (◇, □)                                          │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2: Predicate Infrastructure                              │
│  - Predicate language                                           │
│  - Substitution and quantification                              │
│  - Datalog encoding                                             │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Gph-enriched Lawvere Foundation                       │
│  - Graph-enriched hom-sets                                      │
│  - Reduction rules as edges                                     │
│  - Strategy control                                             │
├─────────────────────────────────────────────────────────────────┤
│  Current: Basic Type Checking                                   │
│  - Sort validation                                              │
│  - Constructor checking                                         │
│  - Rewrite engine                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Order?

1. **Incremental validation**: Each phase is testable independently
2. **Practical value early**: Gph-theories provide useful semantics quickly
3. **Complexity management**: Add features as foundations solidify
4. **Fallback positions**: Can stop at any phase with working system

---

## Phase 1: Gph-enriched Lawvere Foundation

### Goal

Represent operational semantics directly using graph-enriched hom-sets.

### Deliverables

1. **Gph-theory representation**
   ```rust
   pub struct GphTheory {
       pub sorts: Vec<Sort>,
       pub constructors: Vec<Constructor>,
       pub reductions: Vec<Reduction>,  // Edges in hom-graphs
   }

   pub struct Reduction {
       pub name: String,
       pub source: Term,   // LHS pattern
       pub target: Term,   // RHS term
       pub context: ReductionContext,
   }
   ```

2. **Reduction context markers**
   ```rust
   pub struct ReductionContext {
       pub position: Position,      // Where reduction can occur
       pub strategy: Strategy,      // Evaluation strategy
       pub resource: Option<Gas>,   // Optional gas consumption
   }
   ```

3. **Graph operations**
   ```rust
   impl GphTheory {
       /// Get all possible reductions from a term
       pub fn successors(&self, term: &Term) -> Vec<(Reduction, Term)>;

       /// Check if a reduction sequence exists
       pub fn reachable(&self, from: &Term, to: &Term) -> bool;

       /// Normalize using strategy
       pub fn normalize(&self, term: Term, strategy: Strategy) -> Term;
   }
   ```

### Validation

- [ ] SKI combinator calculus example works
- [ ] MeTTa multiset rewriting works
- [ ] Gas consumption tracks correctly
- [ ] Strategies (innermost, outermost) produce correct results

---

## Phase 2: Predicate Infrastructure

### Goal

Add a predicate language for expressing properties of terms.

### Deliverables

1. **Predicate types**
   ```rust
   pub enum Predicate {
       // Atomic predicates
       Atom(String),              // Named predicate
       Constructor(String),       // Term uses constructor
       Equals(Term, Term),        // Term equality

       // Logical connectives
       And(Box<Predicate>, Box<Predicate>),
       Or(Box<Predicate>, Box<Predicate>),
       Not(Box<Predicate>),
       Implies(Box<Predicate>, Box<Predicate>),

       // Quantification
       Forall(Var, Sort, Box<Predicate>),
       Exists(Var, Sort, Box<Predicate>),

       // Substitution
       Subst(Box<Predicate>, Substitution),
   }
   ```

2. **Predicate evaluation**
   ```rust
   impl Predicate {
       /// Evaluate predicate on a term
       pub fn eval(&self, term: &Term, env: &Env) -> bool;

       /// Check satisfiability
       pub fn satisfiable(&self, theory: &Theory) -> Option<Substitution>;
   }
   ```

3. **Datalog encoding**
   ```rust
   ascent! {
       // Encode predicates as Datalog relations
       relation holds(Predicate, Term);
       relation constructor_of(Term, String);
       relation subterm(Term, Term);

       // Derived rules
       holds(And(p, q), t) <-- holds(p, t), holds(q, t);
       holds(Or(p, q), t) <-- holds(p, t);
       holds(Or(p, q), t) <-- holds(q, t);
       holds(Exists(x, s, p), t) <--
           of_sort(witness, s),
           holds(p.subst(x, witness), t);
   }
   ```

### Validation

- [ ] Predicate evaluation is correct
- [ ] Quantifier handling works
- [ ] Datalog encoding matches direct evaluation
- [ ] Performance acceptable for realistic predicates

---

## Phase 3: Behavioral Types

### Goal

Express and check properties about computation steps.

### Deliverables

1. **Internal graph representation**
   ```rust
   pub struct InternalGraph<S> {
       pub states: Set<S>,
       pub edges: Vec<Edge<S>>,
   }

   pub struct Edge<S> {
       pub source: S,
       pub target: S,
       pub label: ReductionLabel,
   }
   ```

2. **Step modalities**
   ```rust
   impl InternalGraph<Term> {
       /// F!(φ): Exists a successor satisfying φ
       pub fn possible(&self, phi: &Predicate) -> Predicate {
           Predicate::Exists(
               Var::fresh("succ"),
               Sort::term(),
               Box::new(Predicate::And(
                   Box::new(Predicate::Successor(Var::current(), Var::fresh("succ"))),
                   Box::new(phi.subst(Var::current(), Var::fresh("succ"))),
               ))
           )
       }

       /// F*(φ): All successors satisfy φ
       pub fn necessary(&self, phi: &Predicate) -> Predicate {
           Predicate::Forall(
               Var::fresh("succ"),
               Sort::term(),
               Box::new(Predicate::Implies(
                   Box::new(Predicate::Successor(Var::current(), Var::fresh("succ"))),
                   Box::new(phi.subst(Var::current(), Var::fresh("succ"))),
               ))
           )
       }
   }
   ```

3. **Reachability modalities**
   ```rust
   impl InternalGraph<Term> {
       /// ◇φ: Eventually φ (least fixed point)
       pub fn eventually(&self, phi: &Predicate) -> Predicate {
           // μX. φ ∨ F!(X)
           self.lfp(|x| Predicate::Or(
               Box::new(phi.clone()),
               Box::new(self.possible(&x)),
           ))
       }

       /// □φ: Always φ (greatest fixed point)
       pub fn always(&self, phi: &Predicate) -> Predicate {
           // νX. φ ∧ F*(X)
           self.gfp(|x| Predicate::And(
               Box::new(phi.clone()),
               Box::new(self.necessary(&x)),
           ))
       }
   }
   ```

4. **Behavioral type examples**
   ```rust
   // Termination: eventually reach normal form
   let terminates = graph.eventually(&Predicate::NormalForm);

   // Progress: always can make progress or is done
   let progress = graph.always(&Predicate::Or(
       Box::new(Predicate::NormalForm),
       Box::new(graph.possible(&Predicate::True)),
   ));

   // Safety: never enter bad state
   let safe = graph.always(&Predicate::Not(Box::new(bad_state)));
   ```

### Validation

- [ ] Step modalities compute correctly
- [ ] Fixed points terminate (with bounded depth)
- [ ] Behavioral properties match expected semantics
- [ ] Integration with Gph-theories works

---

## Phase 4: Full OSLF

### Goal

Implement the complete presheaf construction for maximum expressiveness.

### Deliverables

1. **Presheaf representation**
   ```rust
   pub struct Presheaf<T: Theory> {
       pub base: T,
       pub object: Box<dyn Fn(&Object<T>) -> Set>,
       pub morphism: Box<dyn Fn(&Morphism<T>) -> Function>,
   }

   impl<T: Theory> Presheaf<T> {
       /// Yoneda embedding
       pub fn yoneda(obj: &Object<T>) -> Self {
           Presheaf {
               base: obj.theory(),
               object: Box::new(|c| T::hom(c, obj)),
               morphism: Box::new(|f| /* precomposition with f */),
           }
       }
   }
   ```

2. **Internal hom**
   ```rust
   impl<T: Theory> Presheaf<T> {
       /// Internal hom [P, Q]
       pub fn hom(p: &Self, q: &Self) -> Self {
           Presheaf {
               base: p.base.clone(),
               object: Box::new(|c| {
                   // Natural transformations y(c) × P → Q
                   NatTrans::all(&Presheaf::product(&Presheaf::yoneda(c), p), q)
               }),
               morphism: /* ... */,
           }
       }
   }
   ```

3. **Subobject classifier**
   ```rust
   pub struct Omega<T: Theory>(Presheaf<T>);

   impl<T: Theory> Omega<T> {
       /// The subobject classifier
       pub fn new(theory: &T) -> Self {
           Omega(Presheaf {
               base: theory.clone(),
               object: Box::new(|c| {
                   // Sieves on c
                   theory.sieves(c)
               }),
               morphism: /* pullback of sieves */,
           })
       }

       /// Predicate as subobject
       pub fn classify<P: Predicate>(&self, pred: &P) -> Morphism<Presheaf<T>, Omega<T>> {
           // Characteristic morphism of predicate
       }
   }
   ```

4. **Internal language extraction**
   ```rust
   pub struct TypeTheory<T: Theory> {
       pub types: Vec<Type>,
       pub terms: Vec<TypedTerm>,
       pub judgments: Vec<Judgment>,
   }

   impl<T: Theory> TypeTheory<T> {
       /// Extract type theory from presheaf topos
       pub fn from_presheaf(p: &Presheaf<T>) -> Self {
           // L functor application
       }
   }
   ```

### Validation

- [ ] Yoneda embedding preserves structure
- [ ] Internal hom computes correctly
- [ ] Predicates correspond to subobjects
- [ ] Type theory extraction produces valid rules

---

## Phase 5: WFST Integration

### Goal

Integrate MeTTaIL semantic type checking with liblevenshtein's correction WFST
to create a three-tier correction pipeline.

### Deliverables

1. **FuzzySource trait for PathMap/MORK**
   ```rust
   /// Trait for fuzzy dictionary lookups
   pub trait FuzzySource {
       /// Query with fuzzy matching, returns (candidate, distance) pairs
       fn fuzzy_lookup(&self, query: &[u8], max_distance: u8)
           -> impl Iterator<Item = (Vec<u8>, u8)>;

       /// Prefix search for completion
       fn prefix_search(&self, prefix: &[u8])
           -> impl Iterator<Item = Vec<u8>>;
   }

   impl FuzzySource for PathMap {
       fn fuzzy_lookup(&self, query: &[u8], max_distance: u8)
           -> impl Iterator<Item = (Vec<u8>, u8)>
       {
           // Use liblevenshtein automaton with PathMap traversal
           let automaton = LevenshteinAutomaton::new(query, max_distance);
           self.traverse_with_automaton(&automaton)
       }
   }

   impl FuzzySource for MorkSpace {
       fn fuzzy_lookup(&self, query: &[u8], max_distance: u8)
           -> impl Iterator<Item = (Vec<u8>, u8)>
       {
           // Bloom filter check + PathMap lookup
           if self.bloom.may_contain_within(query, max_distance) {
               self.pathmap.fuzzy_lookup(query, max_distance)
           } else {
               std::iter::empty()
           }
       }
   }
   ```

2. **Candidate lattice generation**
   ```rust
   /// Weighted finite-state transducer for correction lattice
   pub struct CandidateLattice<W: Semiring> {
       states: Vec<LatticeState>,
       arcs: Vec<Arc<W>>,
   }

   pub struct Arc<W: Semiring> {
       source: StateId,
       target: StateId,
       input: Option<Symbol>,   // Input symbol (or epsilon)
       output: Option<Symbol>,  // Output symbol (or epsilon)
       weight: W,               // Semiring weight
   }

   impl<W: Semiring> CandidateLattice<W> {
       /// Build lattice from fuzzy candidates
       pub fn from_candidates<S: FuzzySource>(
           source: &S,
           query: &[u8],
           max_distance: u8,
       ) -> Self {
           let candidates = source.fuzzy_lookup(query, max_distance);
           Self::build_lattice(candidates)
       }

       /// Compose with grammar transducer
       pub fn compose_with_grammar(&self, grammar: &GrammarFst<W>) -> Self {
           // WFST composition
       }

       /// Filter by semantic type predicates
       pub fn filter_by_type<C: TypeChecker>(
           &self,
           checker: &C,
           predicates: &[Predicate],
       ) -> Self {
           // Keep only candidates satisfying type predicates
       }
   }
   ```

3. **Three-tier pipeline**
   ```rust
   /// Unified correction pipeline
   pub struct CorrectionPipeline<W: Semiring> {
       tier1: Tier1LexicalCorrector<W>,
       tier2: Tier2SyntacticValidator<W>,
       tier3: Tier3SemanticChecker<W>,
   }

   impl<W: Semiring> CorrectionPipeline<W> {
       /// Run full correction pipeline
       pub fn correct(&self, input: &str, context: &Context) -> Vec<Correction<W>> {
           // Tier 1: Generate lexical candidates
           let lattice = self.tier1.generate_lattice(input);

           // Tier 2: Filter by grammar
           let syntactic = self.tier2.filter(&lattice, &context.grammar);

           // Tier 3: Filter by semantic types
           let semantic = self.tier3.filter(&syntactic, &context.type_env);

           // Return ranked corrections
           semantic.best_paths(context.top_k)
       }
   }

   /// Tier 1: Lexical correction (liblevenshtein)
   pub struct Tier1LexicalCorrector<W: Semiring> {
       dictionary: Box<dyn FuzzySource>,
       phonetic_rules: Vec<PhoneticRule>,
       semiring: PhantomData<W>,
   }

   /// Tier 2: Syntactic validation (MORK/CFG)
   pub struct Tier2SyntacticValidator<W: Semiring> {
       mork_space: MorkSpace,
       grammar: GrammarFst<W>,
   }

   /// Tier 3: Semantic type checking (MeTTaIL)
   pub struct Tier3SemanticChecker<W: Semiring> {
       theory: GphTheory,
       predicates: PredicateEnv,
       checker: MettailTypeChecker,
   }
   ```

4. **Grammar storage in MORK**
   ```rust
   /// Store grammar rules in MORK Space for efficient validation
   pub struct GrammarMorkSpace {
       mork: MorkSpace,
       productions: Vec<ProductionRule>,
   }

   impl GrammarMorkSpace {
       /// Load grammar from file
       pub fn load(path: &Path) -> Result<Self, Error> {
           let grammar = Grammar::parse(std::fs::read_to_string(path)?)?;
           let mork = MorkSpace::new();

           for rule in grammar.productions() {
               mork.insert_pattern(rule.lhs(), rule.rhs())?;
           }

           Ok(Self { mork, productions: grammar.productions() })
       }

       /// Check if symbol sequence is valid
       pub fn is_valid(&self, symbols: &[Symbol]) -> bool {
           self.mork.matches_any_pattern(symbols)
       }
   }
   ```

5. **Type predicate storage integration**
   ```rust
   /// Store type predicates in MORK for efficient lookup
   pub struct TypePredicateMork {
       mork: MorkSpace,
       predicates: HashMap<PredicateId, Predicate>,
   }

   impl TypePredicateMork {
       /// Store predicate indexed by applicable sorts
       pub fn insert(&mut self, pred: Predicate) -> PredicateId {
           let id = PredicateId::new();
           let key = self.predicate_key(&pred);
           self.mork.insert(&key, &pred.encode())?;
           self.predicates.insert(id, pred);
           id
       }

       /// Find predicates applicable to a term
       pub fn find_applicable(&self, term: &Term) -> Vec<&Predicate> {
           let sort = term.sort();
           let pattern = self.sort_pattern(&sort);
           self.mork.match_pattern(&pattern)
               .filter_map(|id| self.predicates.get(&id))
               .collect()
       }
   }
   ```

### Validation

- [ ] FuzzySource trait works with PathMap and MORK
- [ ] Candidate lattice correctly represents alternatives
- [ ] Grammar validation filters invalid candidates
- [ ] Type predicates correctly reject ill-typed corrections
- [ ] Three-tier pipeline composes weights correctly
- [ ] Performance acceptable for interactive use (< 100ms)

---

## Phase 6: Rholang Integration

### Goal

Connect MeTTaIL type checking to the Rholang compiler.

### Deliverables

1. **Type checker API**
   ```rust
   pub trait TypeChecker {
       type Error;

       /// Type check an AST
       fn check(&self, ast: &Proc) -> Result<TypedProc, Self::Error>;

       /// Infer types where not annotated
       fn infer(&self, ast: &Proc) -> Result<TypedProc, Self::Error>;
   }

   pub struct MettailTypeChecker {
       theory: GphTheory,
       predicates: PredicateEnv,
       behavioral: BehavioralChecker,
   }
   ```

2. **Compiler integration**
   ```rust
   // In f1r3node/rholang/src/rust/interpreter/compiler/compiler.rs

   pub fn compile_with_types(source: &str) -> Result<Par, CompileError> {
       let ast = parse(source)?;

       // Type check with MeTTaIL
       let checker = MettailTypeChecker::new(rholang_theory());
       let typed = checker.check(&ast)?;

       // Normalize type-checked AST
       normalize(&typed)
   }
   ```

3. **Behavioral type enforcement**
   ```rust
   impl MettailTypeChecker {
       /// Check behavioral type annotations
       fn check_behavioral(&self, proc: &Proc, annotation: &BehavioralType) -> Result<(), TypeError> {
           match annotation {
               BehavioralType::Terminating => {
                   let graph = self.build_graph(proc);
                   if !graph.satisfies(&self.predicates.terminates()) {
                       return Err(TypeError::NonTerminating(proc.span()));
                   }
               }
               BehavioralType::DeadlockFree => {
                   // Check progress property
               }
               BehavioralType::Responsive(channel) => {
                   // Check liveness on channel
               }
           }
           Ok(())
       }
   }
   ```

4. **Gradual typing support**
   ```rust
   pub enum TypingMode {
       Full,      // All code must type check
       Gradual,   // Annotated code checked, rest dynamic
       Infer,     // Infer where possible, dynamic otherwise
       Off,       // No type checking (current behavior)
   }

   impl MettailTypeChecker {
       pub fn with_mode(mode: TypingMode) -> Self {
           // Configure based on mode
       }
   }
   ```

### Validation

- [ ] Existing Rholang code compiles unchanged (mode: Off)
- [ ] Type annotations are parsed correctly
- [ ] Type errors produce clear messages
- [ ] Behavioral types detect violations
- [ ] Performance overhead acceptable

---

## Phase 7: Dialogue Context

### Goal

Add multi-turn conversation tracking to enable context-aware correction for
human dialogues and conversational agents.

See [Dialogue Context Layer](../dialogue/README.md) for detailed documentation.

### Deliverables

1. **DialogueState with PathMap backing**
   ```rust
   /// Dialogue state backed by PathMap for persistence
   pub struct DialogueState {
       pathmap: PathMap,
       turns: VecDeque<Turn>,              // Ordered conversation history
       max_history: usize,                  // Sliding window size
       entity_registry: EntityRegistry,     // Coreference tracking
       topic_graph: TopicGraph,            // Discourse structure
       speaker_models: HashMap<ParticipantId, SpeakerModel>,
   }

   impl DialogueState {
       /// Create new dialogue state with PathMap persistence
       pub fn new(dialogue_id: &DialogueId, max_history: usize) -> Self {
           let pathmap = PathMap::new();
           Self {
               pathmap,
               turns: VecDeque::with_capacity(max_history),
               max_history,
               entity_registry: EntityRegistry::new(),
               topic_graph: TopicGraph::new(),
               speaker_models: HashMap::new(),
           }
       }

       /// Add a new turn to the dialogue
       pub fn add_turn(&mut self, turn: Turn) -> Result<(), Error> {
           // Update sliding window
           if self.turns.len() >= self.max_history {
               self.turns.pop_front();
           }
           self.turns.push_back(turn.clone());

           // Update entity registry with new mentions
           self.entity_registry.update(&turn)?;

           // Update topic graph
           self.topic_graph.update(&turn)?;

           // Persist to PathMap
           self.persist_turn(&turn)?;
           Ok(())
       }
   }
   ```

2. **Turn structure with speech act classification**
   ```rust
   /// Single dialogue turn
   pub struct Turn {
       turn_id: TurnId,
       speaker: ParticipantId,
       timestamp: Timestamp,
       raw_text: String,
       corrected_text: Option<String>,
       parsed: Vec<MettaValue>,
       speech_act: SpeechAct,
       entities: Vec<EntityMention>,
       topics: Vec<TopicRef>,
   }

   /// Speech act classification
   pub enum SpeechAct {
       Assert { content: MettaValue, confidence: f64 },
       Question { q_type: QuestionType, focus: MettaValue },
       Directive { action: MettaValue, addressee: Option<ParticipantId> },
       Commissive { commitment: MettaValue },
       Expressive { attitude: String, target: Option<MettaValue> },
       Backchannel { signal_type: BackchannelType },
   }
   ```

3. **Entity registry for coreference**
   ```rust
   /// Entity registry for coreference chains
   pub struct EntityRegistry {
       entities: HashMap<EntityId, Entity>,
       coreference_chains: HashMap<EntityId, Vec<(TurnId, usize)>>,
       salience_scores: HashMap<EntityId, f64>,
   }

   impl EntityRegistry {
       /// Resolve a reference to an entity
       pub fn resolve(&self, mention: &EntityMention) -> Option<EntityId> {
           match mention.mention_type {
               MentionType::ProperName => self.find_by_name(&mention.surface),
               MentionType::Pronoun => self.resolve_pronoun(mention),
               MentionType::DefiniteDesc => self.find_by_description(&mention.surface),
           }
       }

       /// Update salience scores based on recency and frequency
       pub fn update_salience(&mut self, current_turn: TurnId) {
           for (entity_id, chain) in &self.coreference_chains {
               let recency = self.calculate_recency(chain, current_turn);
               let frequency = chain.len() as f64;
               let salience = 0.7 * recency + 0.3 * frequency.ln();
               self.salience_scores.insert(*entity_id, salience);
           }
       }
   }
   ```

4. **Topic graph for discourse structure**
   ```rust
   /// Topic graph tracking discourse structure
   pub struct TopicGraph {
       topics: HashMap<TopicId, Topic>,
       edges: Vec<(TopicId, TopicId, TopicRelation)>,
       active_topics: Vec<TopicId>,
   }

   impl TopicGraph {
       /// Check if topic shift occurred
       pub fn detect_shift(&self, turn: &Turn) -> Option<TopicShift> {
           let turn_topics = self.extract_topics(turn);
           let active_similarity = self.similarity_to_active(&turn_topics);

           if active_similarity < TOPIC_SHIFT_THRESHOLD {
               Some(TopicShift {
                   from: self.active_topics.clone(),
                   to: turn_topics,
                   confidence: 1.0 - active_similarity,
               })
           } else {
               None
           }
       }
   }
   ```

5. **MeTTa predicates for dialogue**
   ```metta
   ; Turn and dialogue structure
   (: Turn Type)
   (: turn-speaker (-> Turn ParticipantId))
   (: turn-text (-> Turn String))
   (: turn-speech-act (-> Turn SpeechAct))

   ; Coreference resolution
   (: resolve-reference (-> String DialogueState (Maybe Entity)))
   (: coreference-chain (-> Entity DialogueState (List Mention)))
   (: entity-salience (-> Entity DialogueState Float))

   ; Topic tracking
   (: topic-similarity (-> Topic Topic Float))
   (: topic-shift (-> Turn Turn Bool))
   (: active-topics (-> DialogueState (List Topic)))
   ```

### Validation

- [ ] DialogueState correctly tracks turn history
- [ ] Entity registry resolves pronouns accurately (> 80% on test set)
- [ ] Topic graph detects topic shifts correctly
- [ ] PathMap persistence works across sessions
- [ ] Speaker models capture vocabulary preferences

---

## Phase 8: LLM Integration

### Goal

Add preprocessing and postprocessing pipelines for LLM agent input/output
correction, including hallucination detection.

See [LLM Integration Layer](../llm-integration/README.md) for detailed documentation.

### Deliverables

1. **Prompt preprocessor pipeline**
   ```rust
   /// Preprocessor for LLM prompts
   pub struct PromptPreprocessor {
       corrector: CorrectionPipeline,
       context_injector: ContextInjector,
       entity_resolver: EntityResolver,
   }

   impl PromptPreprocessor {
       /// Preprocess user input before sending to LLM
       pub fn preprocess(
           &self,
           input: &str,
           dialogue: &DialogueState,
       ) -> Result<PreprocessedPrompt, Error> {
           // Step 1: Correct user input
           let corrections = self.corrector.correct(input, &dialogue.context())?;
           let corrected = corrections.best().unwrap_or(input.to_string());

           // Step 2: Resolve entities
           let resolved_entities = self.entity_resolver.resolve(&corrected, dialogue)?;

           // Step 3: Inject dialogue context
           let context_injection = self.context_injector.format(dialogue)?;

           Ok(PreprocessedPrompt {
               corrected_input: corrected,
               corrections: corrections.all(),
               resolved_entities,
               context_injection,
               confidence: corrections.confidence(),
           })
       }
   }
   ```

2. **Context injection for dialogue history**
   ```rust
   /// Inject dialogue context into LLM prompts
   pub struct ContextInjector {
       format: ContextFormat,
       max_tokens: usize,
       rag_retriever: Option<RagRetriever>,
   }

   impl ContextInjector {
       /// Format dialogue history for injection
       pub fn format(&self, dialogue: &DialogueState) -> Result<String, Error> {
           let mut context = String::new();

           // Add relevant history
           for turn in dialogue.recent_turns(self.max_history) {
               context.push_str(&self.format_turn(turn)?);
           }

           // Add active entities
           context.push_str(&self.format_entities(dialogue)?);

           // Add RAG context if available
           if let Some(rag) = &self.rag_retriever {
               let relevant = rag.retrieve(&dialogue.current_query())?;
               context.push_str(&self.format_rag_context(&relevant)?);
           }

           Ok(context)
       }
   }
   ```

3. **Response postprocessor**
   ```rust
   /// Postprocessor for LLM responses
   pub struct ResponsePostprocessor {
       corrector: CorrectionPipeline,
       hallucination_detector: HallucinationDetector,
       coherence_checker: CoherenceChecker,
   }

   impl ResponsePostprocessor {
       /// Postprocess LLM response
       pub fn postprocess(
           &self,
           response: &str,
           dialogue: &DialogueState,
       ) -> Result<PostprocessedResponse, Error> {
           // Step 1: Correct grammar and spelling
           let corrections = self.corrector.correct(response, &dialogue.context())?;
           let corrected = corrections.best().unwrap_or(response.to_string());

           // Step 2: Detect hallucinations
           let hallucination_flags = self.hallucination_detector.detect(
               &corrected,
               &dialogue.knowledge_base(),
           )?;

           // Step 3: Check coherence with dialogue
           let coherence_score = self.coherence_checker.check(
               &corrected,
               dialogue,
           )?;

           Ok(PostprocessedResponse {
               original: response.to_string(),
               corrected,
               factual_issues: self.extract_factual_issues(&hallucination_flags),
               hallucination_flags,
               coherence_score,
               confidence: 1.0 - hallucination_flags.severity(),
           })
       }
   }
   ```

4. **Hallucination detector**
   ```rust
   /// Detect hallucinated content in LLM responses
   pub struct HallucinationDetector {
       claim_extractor: ClaimExtractor,
       verifier: KnowledgeBaseVerifier,
       entity_checker: EntityExistenceChecker,
   }

   impl HallucinationDetector {
       /// Detect hallucinations in text
       pub fn detect(
           &self,
           text: &str,
           kb: &KnowledgeBase,
       ) -> Result<Vec<HallucinationFlag>, Error> {
           let mut flags = Vec::new();

           // Extract claims from text
           let claims = self.claim_extractor.extract(text)?;

           for claim in claims {
               // Check if claim is verifiable
               let verification = self.verifier.verify(&claim, kb)?;

               match verification {
                   Verification::Supported(evidence) => {
                       // Claim is supported - no flag
                   }
                   Verification::Contradicted(evidence) => {
                       flags.push(HallucinationFlag {
                           span: claim.span,
                           content: claim.text.clone(),
                           hallucination_type: HallucinationType::ContradictedFact,
                           confidence: verification.confidence(),
                           suggestion: Some(evidence.correct_statement()),
                       });
                   }
                   Verification::Unsupported => {
                       flags.push(HallucinationFlag {
                           span: claim.span,
                           content: claim.text.clone(),
                           hallucination_type: HallucinationType::UnsupportedClaim,
                           confidence: 0.5,  // Lower confidence for unsupported
                           suggestion: None,
                       });
                   }
               }
           }

           // Check for non-existent entities
           let entity_flags = self.entity_checker.check(text, kb)?;
           flags.extend(entity_flags);

           Ok(flags)
       }
   }
   ```

5. **MeTTa predicates for LLM integration**
   ```metta
   ; Prompt preprocessing
   (: preprocess-prompt (-> String DialogueState PreprocessedPrompt))
   (: correct-input (-> String Context CorrectedInput))
   (: inject-context (-> DialogueState String))

   ; Response postprocessing
   (: postprocess-response (-> String DialogueState PostprocessedResponse))
   (: detect-hallucination (-> String KnowledgeBase (List HallucinationFlag)))
   (: check-coherence (-> String DialogueState Float))

   ; Fact checking
   (: extract-claims (-> String (List Claim)))
   (: verify-claim (-> Claim KnowledgeBase VerificationStatus))
   (: entity-exists (-> String KnowledgeBase Bool))
   ```

### Validation

- [ ] PromptPreprocessor correctly chains correction and context
- [ ] Context injection fits within token limits
- [ ] HallucinationDetector identifies fabricated facts (> 70% precision)
- [ ] Coherence checker correlates with human judgment (> 0.6 correlation)
- [ ] End-to-end pipeline latency < 500ms

---

## Phase 9: Agent Learning

### Goal

Add feedback-driven learning to continuously improve correction quality based
on user interactions.

See [Agent Learning Layer](../agent-learning/README.md) for detailed documentation.

### Deliverables

1. **Feedback collector**
   ```rust
   /// Collect feedback from user interactions
   pub struct FeedbackCollector {
       signal_detector: SignalDetector,
       feedback_store: PathMap,
       session_tracker: SessionTracker,
   }

   impl FeedbackCollector {
       /// Collect feedback from user action
       pub fn collect(
           &mut self,
           original: &str,
           correction: &str,
           user_action: &UserAction,
       ) -> Result<FeedbackRecord, Error> {
           // Detect signal type
           let signal = self.signal_detector.detect(user_action)?;

           let record = FeedbackRecord {
               id: FeedbackId::new(),
               original: original.to_string(),
               correction: correction.to_string(),
               signal,
               timestamp: Utc::now(),
               session_id: self.session_tracker.current(),
               user_id: self.session_tracker.user(),
           };

           // Store feedback
           self.feedback_store.insert(&record.key(), &record.encode())?;

           Ok(record)
       }
   }

   /// Detect implicit and explicit feedback signals
   pub struct SignalDetector {
       timing_threshold: Duration,
       edit_distance_threshold: f64,
   }

   impl SignalDetector {
       pub fn detect(&self, action: &UserAction) -> Result<FeedbackSignal, Error> {
           match action {
               UserAction::Accept { time_to_accept } => {
                   if *time_to_accept < self.timing_threshold {
                       Ok(FeedbackSignal::StrongAccept)
                   } else {
                       Ok(FeedbackSignal::Accept)
                   }
               }
               UserAction::Edit { edited_text, original } => {
                   let distance = edit_distance(original, edited_text);
                   if distance < self.edit_distance_threshold {
                       Ok(FeedbackSignal::MinorCorrection)
                   } else {
                       Ok(FeedbackSignal::Reject)
                   }
               }
               UserAction::ExplicitRating { rating } => {
                   Ok(FeedbackSignal::Explicit(*rating))
               }
           }
       }
   }
   ```

2. **Pattern learner**
   ```rust
   /// Learn error patterns from feedback
   pub struct PatternLearner {
       pattern_extractor: PatternExtractor,
       cluster_engine: PatternClusterEngine,
       rule_generator: RuleGenerator,
   }

   impl PatternLearner {
       /// Learn patterns from accumulated feedback
       pub fn learn(&mut self, feedback: &[FeedbackRecord]) -> Result<Vec<LearnedPattern>, Error> {
           // Extract patterns from positive feedback
           let positive = feedback.iter()
               .filter(|f| f.signal.is_positive())
               .collect::<Vec<_>>();

           let mut patterns = Vec::new();

           for record in positive {
               // Extract error pattern
               let extracted = self.pattern_extractor.extract(
                   &record.original,
                   &record.correction,
               )?;
               patterns.extend(extracted);
           }

           // Cluster similar patterns
           let clusters = self.cluster_engine.cluster(&patterns)?;

           // Generate rules from clusters
           let mut learned = Vec::new();
           for cluster in clusters {
               if cluster.support() >= MIN_SUPPORT {
                   let rule = self.rule_generator.generate(&cluster)?;
                   learned.push(LearnedPattern {
                       pattern: rule.pattern,
                       correction: rule.correction,
                       confidence: cluster.confidence(),
                       support: cluster.support(),
                   });
               }
           }

           Ok(learned)
       }
   }
   ```

3. **User preference modeler**
   ```rust
   /// Model user preferences for personalized correction
   pub struct UserPreferenceModeler {
       vocabulary_profiler: VocabularyProfiler,
       style_profiler: StyleProfiler,
       preference_store: PathMap,
   }

   impl UserPreferenceModeler {
       /// Update user preference model
       pub fn update(
           &mut self,
           user_id: &UserId,
           feedback: &FeedbackRecord,
       ) -> Result<(), Error> {
           // Update vocabulary profile
           self.vocabulary_profiler.update(user_id, &feedback.original)?;

           // Update style profile
           self.style_profiler.update(user_id, feedback)?;

           // Persist to PathMap
           self.persist(user_id)?;

           Ok(())
       }

       /// Get personalized correction weights
       pub fn get_weights(&self, user_id: &UserId) -> Result<CorrectionWeights, Error> {
           let vocab = self.vocabulary_profiler.get(user_id)?;
           let style = self.style_profiler.get(user_id)?;

           Ok(CorrectionWeights {
               vocabulary_weight: vocab.known_words_weight(),
               formality_level: style.formality_level(),
               aggressiveness: style.correction_aggressiveness(),
           })
       }
   }
   ```

4. **Online weight updater**
   ```rust
   /// Online learning for correction weights
   pub struct OnlineWeightUpdater {
       learning_rate: f64,
       regularization: f64,
       weight_store: PathMap,
   }

   impl OnlineWeightUpdater {
       /// Update weights based on feedback
       pub fn update(
           &mut self,
           model_id: &ModelId,
           feedback: &FeedbackRecord,
           prediction: &Prediction,
       ) -> Result<(), Error> {
           let current_weights = self.weight_store.get_weights(model_id)?;

           // Compute gradient
           let error = feedback.signal.as_value() - prediction.confidence;
           let gradient = self.compute_gradient(&current_weights, feedback, error);

           // SGD update with regularization
           let new_weights = current_weights.iter()
               .zip(gradient.iter())
               .map(|(w, g)| {
                   w - self.learning_rate * (g + self.regularization * w)
               })
               .collect();

           // Store updated weights
           self.weight_store.set_weights(model_id, &new_weights)?;

           Ok(())
       }
   }
   ```

5. **MeTTa predicates for learning**
   ```metta
   ; Feedback collection
   (: collect-feedback (-> String String UserAction FeedbackRecord))
   (: detect-signal (-> UserAction FeedbackSignal))
   (: store-feedback (-> FeedbackRecord Unit))

   ; Pattern learning
   (: extract-pattern (-> String String Pattern))
   (: cluster-patterns (-> (List Pattern) (List PatternCluster)))
   (: generate-rule (-> PatternCluster LearnedRule))

   ; User preferences
   (: update-preference (-> UserId FeedbackRecord Unit))
   (: get-weights (-> UserId CorrectionWeights))
   (: personalize (-> Correction UserId Correction))

   ; Online learning
   (: update-weights (-> ModelId FeedbackRecord Prediction Unit))
   (: adapt-threshold (-> UserId Domain Float))
   ```

### Validation

- [ ] FeedbackCollector captures > 95% of user interactions
- [ ] PatternLearner generates valid rules (> 90% precision on held-out data)
- [ ] UserPreferenceModeler improves personalization (> 10% accuracy gain)
- [ ] OnlineWeightUpdater converges within acceptable bounds
- [ ] A/B testing framework correctly compares model versions

---

## Dependencies

### Phase Dependencies

```
┌───────────────────────────────────────────────────────────────────────┐
│                      PHASE DEPENDENCY GRAPH                           │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Phase 6 (Rholang Integration)                                       │
│      └── Phase 5 (WFST Integration) [optional]                       │
│      └── Phase 4 (Full OSLF) [optional]                              │
│      └── Phase 3 (Behavioral Types)                                  │
│              └── Phase 2 (Predicates)                                │
│                      └── Phase 1 (Gph-theories)                      │
│                              └── Current (Basic Types)               │
│                                                                       │
│  ═══════════════════════════════════════════════════════════════════ │
│  Parallel development track for conversational correction             │
│  ═══════════════════════════════════════════════════════════════════ │
│                                                                       │
│  Phase 5 (WFST Integration)                                          │
│      └── Phase 3 (Behavioral Types) [for semantic filtering]         │
│      └── liblevenshtein [external]                                   │
│      └── MORK/PathMap [external]                                     │
│              │                                                        │
│              ▼                                                        │
│  Phase 7 (Dialogue Context)                                          │
│      └── Phase 5 (WFST Integration)                                  │
│      └── PathMap [external]                                          │
│              │                                                        │
│              ▼                                                        │
│  Phase 8 (LLM Integration)                                           │
│      └── Phase 7 (Dialogue Context)                                  │
│      └── External LLM APIs [external]                                │
│              │                                                        │
│              ▼                                                        │
│  Phase 9 (Agent Learning)                                            │
│      └── Phase 8 (LLM Integration)                                   │
│      └── PathMap [external] (for persistent storage)                 │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### External Dependencies

| Dependency | Version | Used By |
|------------|---------|---------|
| Rust | 1.70+ | All phases |
| Ascent | Latest | Phases 2-4 |
| LALRPOP | Latest | Phase 1+ |
| liblevenshtein | Latest | Phase 5+ |
| MORK | Latest | Phase 5+ |
| PathMap | Latest | Phase 5-9 |
| f1r3node | main | Phase 6 |
| rholang-parser | main | Phase 6 |
| External LLM APIs | Various | Phase 8 |
| NLP libraries | Various | Phases 7-8 |

---

## Validation Criteria

### Per-Phase Criteria

| Phase | Success Criteria |
|-------|------------------|
| 1 | SKI and MeTTa examples work, gas tracking correct |
| 2 | Predicates evaluate correctly, Datalog matches direct eval |
| 3 | Behavioral properties detect termination/deadlock |
| 4 | Type theory extraction produces valid rules |
| 5 | WFST pipeline produces correct, typed corrections in < 100ms |
| 6 | Rholang compiles with type checking, no regressions |
| 7 | Dialogue state persists, coreference resolution > 80% accuracy |
| 8 | LLM preprocessing/postprocessing < 500ms, hallucination detection > 70% precision |
| 9 | Pattern learning > 90% precision, personalization > 10% accuracy improvement |

### Overall Success Metrics

1. **Correctness**: No false positives or negatives in type checking
2. **Completeness**: All OSLF features implementable
3. **Performance**: < 10% overhead vs untyped compilation
4. **Usability**: Clear error messages, gradual adoption path
5. **Integration**: Seamless Rholang compiler integration
6. **Correction Quality**: Semantic types improve correction accuracy by > 20%
7. **Dialogue Coherence**: Context-aware corrections improve user satisfaction by > 15%
8. **LLM Quality**: Reduced hallucination rate and improved response coherence
9. **Learning Effectiveness**: Continuous improvement from user feedback

---

## Summary

The implementation roadmap provides:

1. **Layered progression** from simple to complex
2. **Validation checkpoints** at each phase
3. **Clear dependencies** between components
4. **Flexibility** to stop at useful intermediate points
5. **WFST integration** for semantic-aware correction
6. **Rholang integration** as the final compilation target
7. **Dialogue context** for multi-turn conversational correction
8. **LLM integration** for agent input/output validation
9. **Agent learning** for continuous improvement from feedback

### Recommended Implementation Order

**Core Path**: Phase 1 → Phase 2 → Phase 3 → Phase 6

This provides behavioral type checking in Rholang without full OSLF complexity.

**With Correction**: Phase 1 → Phase 2 → Phase 3 → Phase 5 → Phase 6

This adds semantic type-aware correction before Rholang integration.

**With Conversational Correction**: Phase 1 → Phase 2 → Phase 3 → Phase 5 → Phase 7 → Phase 8 → Phase 9

This adds dialogue context, LLM integration, and learning for conversational agents.

**Full Implementation**: All phases in order

This provides maximum expressiveness including presheaf construction and full
conversational correction capabilities.

### Parallel Development Tracks

The roadmap supports two parallel development tracks:

1. **OSLF Track** (Phases 1-4, 6): Semantic type checking for Rholang
2. **Conversational Track** (Phases 5, 7-9): Context-aware correction for dialogues

These tracks can be developed independently, with Phase 5 (WFST Integration)
serving as the bridge point.

The recommended starting point is Phase 1 (Gph-theories) in mettail-rust, building
toward WFST integration in Phase 5. From there, either proceed to Rholang
integration (Phase 6) or conversational correction (Phases 7-9) based on
project priorities.

---

## References

- See [gap-analysis.md](../reference/gap-analysis.md) for detailed gap analysis
- See [use-cases.md](../reference/use-cases.md) for validation examples
- See [bibliography.md](../reference/bibliography.md) for complete references
- See [correction-wfst/](../correction-wfst/README.md) for WFST architecture details
- See [dialogue/](../dialogue/README.md) for dialogue context layer details
- See [llm-integration/](../llm-integration/README.md) for LLM integration details
- See [agent-learning/](../agent-learning/README.md) for agent learning details
