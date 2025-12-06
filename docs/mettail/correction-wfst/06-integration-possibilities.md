# Integration Possibilities

This document describes additional integration possibilities for the unified
correction WFST architecture beyond the core three-tier system. It includes
use cases for conversational systems, LLM agent integration, and traditional
programming language tooling.

**Sources**:
- All project repositories in `/home/dylon/Workspace/f1r3fly.io/`

**Related Documentation**:
- [Dialogue Context Layer](../dialogue/README.md) - Conversation tracking
- [LLM Integration Layer](../llm-integration/README.md) - Agent preprocessing/postprocessing
- [Agent Learning Layer](../agent-learning/README.md) - Adaptive correction

---

## Table of Contents

### Conversational Systems
1. [Human Dialogue Correction](#human-dialogue-correction)
2. [LLM Agent Integration](#llm-agent-integration)
3. [Chatbot Quality Assurance](#chatbot-quality-assurance)
4. [Customer Support Correction](#customer-support-correction)

### Programming Languages
5. [Cross-Language Correction](#cross-language-correction)
6. [ASR Error Correction](#asr-error-correction)
7. [Type-Aware Code Completion](#type-aware-code-completion)
8. [Smart Contract Verification](#smart-contract-verification)
9. [Gradual Type Migration](#gradual-type-migration)
10. [IDE/LSP Integration](#idelsp-integration)
11. [Distributed Correction](#distributed-correction)

---

## Human Dialogue Correction

Correct written text in human-to-human conversations while preserving context,
speaker identity, and dialogue coherence.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Human Dialogue Correction                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Message: "Did you recieve teh document I sent yestarday?"  │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Dialogue Context Retrieval                      ││
│  │  • Previous turns with document references                   ││
│  │  • Entity: "teh document" → "quarterly_report.pdf"          ││
│  │  • Temporal: "yestarday" → relative date context            ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Three-Tier WFST Correction                      ││
│  │  Tier 1: "recieve"→"receive", "teh"→"the", "yestarday"...  ││
│  │  Tier 2: Grammar validation (past tense consistency)        ││
│  │  Tier 3: Semantic coherence with dialogue context           ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Coreference Validation                          ││
│  │  • "teh document" → verify document entity exists           ││
│  │  • "I sent" → speaker consistency check                     ││
│  │  • "yestarday" → temporal coherence with conversation       ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  Corrected: "Did you receive the document I sent yesterday?"     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
/// Dialogue-aware corrector for human conversations
pub struct DialogueCorrector {
    /// Core correction engine
    corrector: CorrectionEngine,
    /// Dialogue context manager
    dialogue_context: DialogueContext,
    /// Coreference resolver
    coreference: CoreferenceResolver,
}

impl DialogueCorrector {
    /// Correct message with full dialogue context
    pub fn correct_message(
        &self,
        message: &str,
        speaker: &ParticipantId,
        dialogue_id: &DialogueId,
    ) -> Result<CorrectedMessage, Error> {
        // Retrieve dialogue context
        let context = self.dialogue_context.get(dialogue_id)?;

        // Extract entities from message
        let entities = self.coreference.extract_mentions(message)?;

        // Resolve coreferences using context
        let resolved = self.coreference.resolve(&entities, &context)?;

        // Correct with context-aware ranking
        let corrections = self.corrector
            .with_context(&context)
            .with_entities(&resolved)
            .correct(message)?;

        // Validate coherence with dialogue
        let validated = corrections.into_iter()
            .filter(|c| self.validate_coherence(c, &context))
            .collect::<Vec<_>>();

        // Update dialogue state
        self.dialogue_context.add_turn(
            dialogue_id,
            Turn::new(speaker.clone(), message, &validated),
        )?;

        Ok(CorrectedMessage {
            original: message.to_string(),
            corrected: validated.first().map(|c| c.text.clone()),
            corrections: validated,
            entities: resolved,
        })
    }

    /// Validate correction maintains dialogue coherence
    fn validate_coherence(&self, correction: &Correction, context: &DialogueState) -> bool {
        // Check entity references still valid
        let entity_check = self.coreference
            .validate_entities(&correction.text, context);

        // Check temporal consistency
        let temporal_check = self.validate_temporal_refs(
            &correction.text,
            context,
        );

        // Check speaker consistency
        let speaker_check = self.validate_speaker_refs(
            &correction.text,
            context,
        );

        entity_check && temporal_check && speaker_check
    }
}
```

### Use Cases

| Scenario | Context Required | Correction Type |
|----------|------------------|-----------------|
| **Chat apps** | Message history, participant names | Typos, autocorrect |
| **Email threads** | Reply chain, attachments | Grammar, formality |
| **Forum posts** | Thread context, quoted text | Style consistency |
| **Comments** | Parent post, mentioned users | Reference validation |

**Documentation**: [Dialogue Context Layer](../dialogue/README.md)

---

## LLM Agent Integration

Integrate correction with LLM-based agents for improved input processing
and output quality assurance.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   LLM Agent Integration                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                      ┌────────────────────┐                     │
│                      │    User Input      │                     │
│                      └─────────┬──────────┘                     │
│                                │                                 │
│  ┌─────────────────────────────┼─────────────────────────────┐  │
│  │                             ▼                              │  │
│  │                   PREPROCESSING                            │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ 1. Three-Tier Correction                            │  │  │
│  │  │    Fix typos, grammar, semantic errors              │  │  │
│  │  │                                                      │  │  │
│  │  │ 2. Coreference Resolution                           │  │  │
│  │  │    Resolve "it", "this", "the file" from context    │  │  │
│  │  │                                                      │  │  │
│  │  │ 3. Context Injection                                │  │  │
│  │  │    Dialogue history, relevant documents, RAG        │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────┬─────────────────────────────┘  │
│                                │                                 │
│                                ▼                                 │
│                      ┌────────────────────┐                     │
│                      │      LLM API       │                     │
│                      └─────────┬──────────┘                     │
│                                │                                 │
│  ┌─────────────────────────────┼─────────────────────────────┐  │
│  │                             ▼                              │  │
│  │                   POSTPROCESSING                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ 1. Coherence Check                                  │  │  │
│  │  │    Does response address the question?              │  │  │
│  │  │                                                      │  │  │
│  │  │ 2. Fact Verification                                │  │  │
│  │  │    Check claims against knowledge base              │  │  │
│  │  │                                                      │  │  │
│  │  │ 3. Hallucination Detection                          │  │  │
│  │  │    Flag fabricated facts, nonexistent entities      │  │  │
│  │  │                                                      │  │  │
│  │  │ 4. Correction                                       │  │  │
│  │  │    Fix any errors in LLM output                     │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────┬─────────────────────────────┘  │
│                                │                                 │
│                                ▼                                 │
│                      ┌────────────────────┐                     │
│                      │   Final Response   │                     │
│                      └────────────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
/// LLM agent with integrated correction
pub struct CorrectedLlmAgent {
    /// Preprocessing pipeline
    preprocessor: PromptPreprocessor,
    /// LLM client
    llm: LlmClient,
    /// Postprocessing pipeline
    postprocessor: ResponsePostprocessor,
    /// Dialogue context
    context: DialogueContext,
}

impl CorrectedLlmAgent {
    /// Process user query with full correction pipeline
    pub async fn query(
        &self,
        user_input: &str,
        session_id: &SessionId,
    ) -> Result<AgentResponse, Error> {
        // Get dialogue context
        let dialogue = self.context.get_or_create(session_id)?;

        // PREPROCESSING
        let preprocessed = self.preprocessor.process(
            user_input,
            &dialogue,
        )?;

        // Log preprocessing results
        let preprocess_record = PreprocessRecord {
            original: user_input.to_string(),
            corrections: preprocessed.corrections.clone(),
            resolved_refs: preprocessed.resolved_entities.clone(),
        };

        // Call LLM
        let prompt = preprocessed.to_prompt();
        let raw_response = self.llm.complete(&prompt).await?;

        // POSTPROCESSING
        let postprocessed = self.postprocessor.process(
            &raw_response,
            &dialogue,
            &preprocessed,
        )?;

        // Update dialogue context
        dialogue.add_exchange(
            user_input,
            &postprocessed.corrected,
            &preprocessed,
            &postprocessed,
        )?;

        Ok(AgentResponse {
            response: postprocessed.corrected,
            original_response: raw_response,
            preprocessing: preprocess_record,
            postprocessing: PostprocessRecord {
                coherence_score: postprocessed.coherence_score,
                hallucinations: postprocessed.hallucination_flags.clone(),
                corrections: postprocessed.corrections.clone(),
            },
            confidence: postprocessed.confidence,
        })
    }
}
```

### Preprocessing Details

```rust
impl PromptPreprocessor {
    pub fn process(
        &self,
        input: &str,
        context: &DialogueState,
    ) -> Result<PreprocessedPrompt, Error> {
        // Step 1: Correct user input
        let corrections = self.corrector.correct(input)?;
        let corrected = corrections.first()
            .map(|c| c.text.clone())
            .unwrap_or_else(|| input.to_string());

        // Step 2: Resolve coreferences
        let mentions = self.coreference.extract(&corrected)?;
        let resolved = self.coreference.resolve(&mentions, context)?;
        let with_refs = self.expand_references(&corrected, &resolved);

        // Step 3: Inject context
        let context_str = self.format_context(context)?;

        // Step 4: RAG retrieval if needed
        let retrieved = self.retrieve_relevant_docs(&with_refs)?;

        Ok(PreprocessedPrompt {
            corrected_input: with_refs,
            corrections,
            resolved_entities: resolved,
            context_injection: context_str,
            retrieved_docs: retrieved,
            confidence: self.calculate_confidence(&corrections),
        })
    }
}
```

### Hallucination Detection

```rust
impl ResponsePostprocessor {
    /// Detect hallucinations in LLM response
    fn detect_hallucinations(
        &self,
        response: &str,
        context: &DialogueState,
    ) -> Vec<HallucinationFlag> {
        let mut flags = Vec::new();

        // Extract claims from response
        let claims = self.extract_claims(response);

        for claim in claims {
            // Check against knowledge base
            if !self.knowledge_base.supports(&claim) {
                flags.push(HallucinationFlag {
                    span: claim.span.clone(),
                    content: claim.text.clone(),
                    hallucination_type: HallucinationType::UnsupportedClaim,
                    confidence: 0.8,
                    suggestion: self.suggest_correction(&claim),
                });
            }

            // Check entity existence
            for entity in &claim.entities {
                if !self.entity_exists(entity, context) {
                    flags.push(HallucinationFlag {
                        span: entity.span.clone(),
                        content: entity.name.clone(),
                        hallucination_type: HallucinationType::NonexistentEntity,
                        confidence: 0.9,
                        suggestion: self.suggest_entity(entity, context),
                    });
                }
            }

            // Check temporal consistency
            if let Some(temporal) = &claim.temporal {
                if !self.temporal_consistent(temporal, context) {
                    flags.push(HallucinationFlag {
                        span: temporal.span.clone(),
                        content: temporal.text.clone(),
                        hallucination_type: HallucinationType::TemporalError,
                        confidence: 0.7,
                        suggestion: None,
                    });
                }
            }
        }

        flags
    }
}
```

**Documentation**: [LLM Integration Layer](../llm-integration/README.md)

---

## Chatbot Quality Assurance

Apply correction to improve chatbot output quality and consistency.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Chatbot Quality Assurance                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Customer Query                                                  │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Intent Classification                        ││
│  │  + Correction to normalize noisy input                      ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Response Generation                         ││
│  │  (Rule-based, Retrieval, or LLM)                            ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Quality Assurance Layer                      ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  Grammar Check                                      │   ││
│  │  │  Style Consistency (brand voice)                    │   ││
│  │  │  Factual Accuracy (against product DB)              │   ││
│  │  │  Policy Compliance (no prohibited content)          │   ││
│  │  │  Tone Appropriateness (formality level)             │   ││
│  │  └─────────────────────────────────────────────────────┘   ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  Approved Response                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
/// Chatbot QA system
pub struct ChatbotQA {
    /// Grammar and spelling corrector
    corrector: CorrectionEngine,
    /// Brand voice style guide
    style_guide: StyleGuide,
    /// Product knowledge base
    product_kb: ProductKnowledgeBase,
    /// Policy rules
    policy_rules: PolicyRules,
}

impl ChatbotQA {
    /// Validate and correct chatbot response
    pub fn validate_response(
        &self,
        response: &str,
        context: &ConversationContext,
    ) -> QAResult {
        let mut issues = Vec::new();
        let mut corrected = response.to_string();

        // 1. Grammar and spelling
        let grammar_issues = self.corrector.check(&corrected)?;
        for issue in &grammar_issues {
            corrected = self.apply_correction(&corrected, issue);
            issues.push(QAIssue::Grammar(issue.clone()));
        }

        // 2. Style consistency
        let style_issues = self.style_guide.check(&corrected)?;
        for issue in &style_issues {
            corrected = self.apply_style_fix(&corrected, issue);
            issues.push(QAIssue::Style(issue.clone()));
        }

        // 3. Factual accuracy
        let facts = self.extract_product_claims(&corrected);
        for fact in facts {
            if !self.product_kb.verify(&fact) {
                issues.push(QAIssue::FactualError(fact.clone()));
                corrected = self.remove_claim(&corrected, &fact);
            }
        }

        // 4. Policy compliance
        let policy_issues = self.policy_rules.check(&corrected)?;
        for issue in &policy_issues {
            issues.push(QAIssue::Policy(issue.clone()));
            corrected = self.redact(&corrected, issue);
        }

        // 5. Tone appropriateness
        let tone = self.analyze_tone(&corrected);
        let expected = self.expected_tone(context);
        if tone != expected {
            corrected = self.adjust_tone(&corrected, &expected);
            issues.push(QAIssue::Tone { actual: tone, expected });
        }

        QAResult {
            original: response.to_string(),
            corrected,
            issues,
            approved: issues.iter().all(|i| i.is_minor()),
        }
    }
}
```

### Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Grammar Score** | % of responses without grammar errors | >99% |
| **Style Consistency** | Adherence to brand voice | >95% |
| **Factual Accuracy** | Correct product information | 100% |
| **Policy Compliance** | No prohibited content | 100% |
| **Tone Match** | Appropriate formality | >90% |

---

## Customer Support Correction

Specialized correction for customer support interactions with domain-specific
terminology and escalation awareness.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Customer Support Correction                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Support Ticket / Chat Message                                   │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Domain-Specific Correction                      ││
│  │  • Product name normalization ("iphone" → "iPhone")         ││
│  │  • Ticket ID validation (#12345 format)                     ││
│  │  • Technical term correction (domain dictionary)            ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Customer Context Retrieval                      ││
│  │  • Previous tickets                                         ││
│  │  • Purchase history                                         ││
│  │  • Account status                                           ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Intent + Sentiment Analysis                     ││
│  │  • Issue category classification                            ││
│  │  • Urgency detection                                        ││
│  │  • Frustration level assessment                             ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  Corrected + Enriched Ticket                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
/// Customer support correction with domain awareness
pub struct SupportCorrector {
    /// Base corrector
    corrector: CorrectionEngine,
    /// Product dictionary
    product_dict: ProductDictionary,
    /// Customer database
    customer_db: CustomerDatabase,
    /// Sentiment analyzer
    sentiment: SentimentAnalyzer,
}

impl SupportCorrector {
    /// Process support ticket with full context
    pub fn process_ticket(
        &self,
        message: &str,
        customer_id: &CustomerId,
    ) -> Result<ProcessedTicket, Error> {
        // Get customer context
        let customer = self.customer_db.get(customer_id)?;
        let history = self.customer_db.get_ticket_history(customer_id)?;

        // Domain-specific correction
        let corrections = self.correct_with_domain(message)?;
        let corrected = self.apply_corrections(message, &corrections);

        // Extract and validate references
        let ticket_refs = self.extract_ticket_refs(&corrected);
        let product_refs = self.extract_product_refs(&corrected);

        // Validate references against customer history
        let validated_tickets = self.validate_ticket_refs(&ticket_refs, &history);
        let validated_products = self.validate_product_refs(&product_refs, &customer);

        // Sentiment and urgency analysis
        let sentiment = self.sentiment.analyze(&corrected)?;
        let urgency = self.assess_urgency(&corrected, &sentiment)?;

        // Intent classification
        let intent = self.classify_intent(&corrected, &history)?;

        Ok(ProcessedTicket {
            original: message.to_string(),
            corrected,
            corrections,
            ticket_refs: validated_tickets,
            product_refs: validated_products,
            sentiment,
            urgency,
            intent,
            customer_context: CustomerContext {
                tier: customer.tier,
                tenure: customer.tenure(),
                recent_issues: history.recent(5),
            },
        })
    }

    /// Correct with domain-specific dictionary
    fn correct_with_domain(&self, text: &str) -> Result<Vec<Correction>, Error> {
        // Use product dictionary for Tier 1
        let mut corrections = self.corrector
            .with_dictionary(&self.product_dict)
            .correct(text)?;

        // Add product name normalizations
        let product_names = self.product_dict.find_mentions(text);
        for name in product_names {
            if let Some(canonical) = self.product_dict.canonical(&name) {
                if canonical != name {
                    corrections.push(Correction {
                        span: name.span.clone(),
                        original: name.text.clone(),
                        text: canonical.clone(),
                        correction_type: CorrectionType::ProductNormalization,
                        confidence: 1.0,
                    });
                }
            }
        }

        Ok(corrections)
    }
}
```

### Domain Dictionaries

| Domain | Dictionary Size | Examples |
|--------|-----------------|----------|
| **Products** | 5,000+ | Product names, model numbers |
| **Technical** | 10,000+ | Error codes, features |
| **Jargon** | 2,000+ | Internal terms, abbreviations |
| **Competitors** | 1,000+ | Alternative products |

---

## Cross-Language Correction

The unified architecture enables correction across language boundaries using
MeTTa as a universal intermediate representation.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Cross-Language Correction                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input (any language)                                           │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │ Language A      │     │ Language B      │                   │
│  │ (e.g., Python)  │     │ (e.g., Rholang) │                   │
│  └────────┬────────┘     └────────┬────────┘                   │
│           │                       │                             │
│           └───────────┬───────────┘                             │
│                       ▼                                          │
│              ┌─────────────────┐                                │
│              │    MeTTa AST    │  ← Universal representation    │
│              │  (via PathMap)  │                                │
│              └────────┬────────┘                                │
│                       │                                          │
│                       ▼                                          │
│              ┌─────────────────┐                                │
│              │  Type Checking  │  ← Language-agnostic           │
│              │    (OSLF)       │                                │
│              └────────┬────────┘                                │
│                       │                                          │
│                       ▼                                          │
│  Output (corrected in original language)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
/// Cross-language correction engine
pub struct CrossLanguageCorrector {
    /// Language-specific parsers
    parsers: HashMap<String, Box<dyn Parser>>,
    /// MeTTa type checker
    type_checker: TypeChecker,
    /// PathMap for shared storage
    pathmap: PathMap,
}

impl CrossLanguageCorrector {
    /// Correct code in any supported language
    pub fn correct(
        &self,
        code: &str,
        language: &str,
    ) -> Result<Vec<Correction>, Error> {
        // Parse to language-specific AST
        let parser = self.parsers.get(language)
            .ok_or(Error::UnsupportedLanguage(language.to_string()))?;
        let ast = parser.parse(code)?;

        // Convert to MeTTa representation
        let metta_state = ast.to_metta_state(&self.pathmap)?;

        // Type check in MeTTa
        let typed = self.type_checker.check(&metta_state)?;

        // Convert back to original language
        let corrected_ast = typed.to_language_ast(language)?;

        // Generate corrections
        corrected_ast.diff(&ast)
    }
}
```

### Supported Language Pairs

| From | To | Via | Use Case |
|------|-----|-----|----------|
| Python | Rholang | MeTTa | Smart contract migration |
| Rholang | MeTTa | Direct | Knowledge integration |
| MeTTa | Rholang | PathMap | Blockchain execution |
| Natural Language | MeTTa | NLU | Knowledge extraction |

---

## ASR Error Correction

For spoken language, integrate phonetic lattices with semantic validation.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   ASR Error Correction                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Audio Input                                                     │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 ASR Decoder                                  ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         ││
│  │  │  Acoustic   │  │  Phoneme    │  │  Language   │         ││
│  │  │   Model     │→ │  Lattice    │→ │   Model     │         ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘         ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Phonetic Expansion (Tier 1)                    ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  Metaphone expansion                                │   ││
│  │  │  Soundex matching                                   │   ││
│  │  │  Phonetic similarity scoring                        │   ││
│  │  └─────────────────────────────────────────────────────┘   ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           Semantic Coherence (Tier 3)                       ││
│  │  MeTTa knowledge base for domain-specific validation       ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  Transcription Output                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
/// ASR correction with semantic validation
pub struct AsrCorrector {
    /// Phonetic rules
    phonetic_rules: Vec<PhoneticRule>,
    /// MeTTa knowledge base for domain
    knowledge_base: MettaSpace,
    /// liblevenshtein dictionary
    dictionary: DynamicDawg,
}

impl AsrCorrector {
    /// Correct ASR output with semantic awareness
    pub fn correct(
        &self,
        phoneme_lattice: &PhonemeLattice,
        domain: &str,
    ) -> Result<String, Error> {
        // Expand phoneme lattice with phonetic rules
        let expanded = self.expand_phonetic(phoneme_lattice);

        // Convert to character candidates
        let candidates: Vec<_> = expanded
            .best_paths(100)
            .map(|path| phonemes_to_text(&path))
            .collect();

        // Filter by semantic coherence
        let coherent: Vec<_> = candidates.into_iter()
            .filter(|text| self.is_semantically_coherent(text, domain))
            .collect();

        // Rank by combined score
        coherent.into_iter()
            .min_by_key(|text| self.combined_score(text))
            .ok_or(Error::NoValidCorrection)
    }

    /// Check semantic coherence against knowledge base
    fn is_semantically_coherent(&self, text: &str, domain: &str) -> bool {
        let query = format!("(coherent \"{}\" {})", text, domain);
        let result = self.knowledge_base.query(&query);
        !result.is_empty()
    }
}
```

---

## Type-Aware Code Completion

IDE integration with type-aware completions using the correction stack.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Type-Aware Code Completion                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User types: "let x: In"                                        │
│                      │ cursor                                    │
│                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Prefix Extraction                          ││
│  │  prefix = "In", context = type annotation position          ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Tier 1: Lexical Candidates                     ││
│  │  ["Int", "Integer", "Input", "Index", "Into", ...]         ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           Tier 2: Syntactic Filtering                       ││
│  │  Filter: expecting type name in annotation                  ││
│  │  ["Int", "Integer", "Input"]                               ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │             Tier 3: Type Compatibility                      ││
│  │  Filter: valid types in current scope                       ││
│  │  ["Int", "Integer"]                                        ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  Completion Menu:                                               │
│  ┌───────────────────┐                                          │
│  │ Int      (i32)    │                                          │
│  │ Integer  (num)    │                                          │
│  └───────────────────┘                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
/// Type-aware completion provider
pub struct TypeAwareCompleter {
    /// Correction engine
    corrector: CorrectionEngine,
    /// Type environment
    type_env: TypeEnvironment,
}

impl CompletionProvider for TypeAwareCompleter {
    fn provide_completions(
        &self,
        document: &Document,
        position: Position,
    ) -> Vec<CompletionItem> {
        // Extract prefix and context
        let prefix = document.prefix_at(position);
        let context = document.syntax_context_at(position);

        // Tier 1: Lexical candidates via prefix search
        let lexical = self.corrector.dictionary
            .prefix_search(prefix.as_bytes())
            .take(100)
            .collect::<Vec<_>>();

        // Tier 2: Syntactic filtering
        let expected_kinds = context.expected_symbol_kinds();
        let syntactic: Vec<_> = lexical.into_iter()
            .filter(|c| {
                let kind = self.symbol_kind(c);
                expected_kinds.contains(&kind)
            })
            .collect();

        // Tier 3: Type compatibility
        let typed: Vec<_> = syntactic.into_iter()
            .filter(|c| {
                let sym_type = self.type_env.type_of(c);
                context.is_type_compatible(&sym_type)
            })
            .map(|c| self.to_completion_item(c, &context))
            .collect();

        typed
    }
}
```

---

## Smart Contract Verification

For Rholang smart contracts, combine correction with formal verification.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               Smart Contract Verification                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Contract Source (Rholang)                                      │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Syntax Checking                             ││
│  │  (Tier 2: Tree-sitter Rholang grammar)                     ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Type Checking                               ││
│  │  (Tier 3: MeTTaIL behavioral types)                        ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  @safe         - No exceptions                      │   ││
│  │  │  @terminating  - Always completes                   │   ││
│  │  │  @isolated(ns) - Namespace isolation                │   ││
│  │  │  @linear       - Resource linearity                 │   ││
│  │  └─────────────────────────────────────────────────────┘   ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Property Verification                          ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  Balance invariants                                 │   ││
│  │  │  Access control                                     │   ││
│  │  │  Reentrancy freedom                                 │   ││
│  │  │  Deadlock freedom                                   │   ││
│  │  └─────────────────────────────────────────────────────┘   ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  Verified Contract + Proof Certificates                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Behavioral Type Annotations

```rholang
// Full contract type specification
contract Token implements ERC20 {
  @total    // Always returns
  @pure     // No side effects
  def balanceOf(@address: Address): Nat

  @total
  @safe     // No exceptions
  def transfer(@to: Address, @amount: Nat, return: Name[Bool]): Unit

  @terminating
  @isolated(internal)  // Only internal namespace access
  def _updateBalance(@addr: Address, @delta: Int): Unit
}
```

### Implementation

```rust
/// Smart contract verifier
pub struct ContractVerifier {
    /// Type checker
    type_checker: RholangTypeChecker,
    /// Property verifier
    property_verifier: PropertyVerifier,
}

impl ContractVerifier {
    /// Verify contract with all annotations
    pub fn verify(&self, contract: &Contract) -> Result<VerificationResult, VerifyError> {
        let mut results = Vec::new();

        // Check each method
        for method in &contract.methods {
            // Verify behavioral annotations
            for annotation in &method.annotations {
                let check = match annotation {
                    Annotation::Total => self.check_totality(method),
                    Annotation::Pure => self.check_purity(method),
                    Annotation::Safe => self.check_safety(method),
                    Annotation::Terminating => self.check_termination(method),
                    Annotation::Isolated(ns) => self.check_isolation(method, ns),
                    Annotation::Linear => self.check_linearity(method),
                };
                results.push(check?);
            }
        }

        // Verify contract-level properties
        let invariants = self.property_verifier.check_invariants(contract)?;

        Ok(VerificationResult { method_results: results, invariants })
    }
}
```

---

## Gradual Type Migration

Support incremental typing for migrating legacy codebases.

### Migration Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                  Gradual Type Migration                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: Analysis                                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  • Identify untyped code regions                           ││
│  │  • Infer types where possible                              ││
│  │  • Prioritize high-impact areas                            ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Phase 2: Incremental Typing                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  • Add types to critical functions first                   ││
│  │  • Use Dynamic for untyped boundaries                      ││
│  │  • Validate incrementally                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Phase 3: Full Typing                                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  • Replace Dynamic with concrete types                     ││
│  │  • Add behavioral annotations                              ││
│  │  • Enable strict mode                                      ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
/// Gradual typing migration tool
pub struct MigrationTool {
    /// Type inference engine
    inferrer: TypeInferrer,
    /// Correction engine
    corrector: CorrectionEngine,
}

impl MigrationTool {
    /// Analyze codebase for typing opportunities
    pub fn analyze(&self, codebase: &Codebase) -> MigrationPlan {
        let mut plan = MigrationPlan::new();

        for file in codebase.files() {
            let ast = self.parse(file)?;

            // Find untyped definitions
            for def in ast.definitions() {
                if def.is_untyped() {
                    // Try to infer type
                    let inferred = self.inferrer.infer(&def);

                    plan.add_suggestion(TypeSuggestion {
                        location: def.span(),
                        inferred_type: inferred,
                        confidence: inferred.confidence(),
                        impact: self.estimate_impact(&def, &codebase),
                    });
                }
            }
        }

        // Prioritize by impact and confidence
        plan.prioritize();
        plan
    }

    /// Apply migration step
    pub fn apply_step(&self, step: &MigrationStep, file: &mut File) -> Result<(), Error> {
        // Add type annotation
        let edit = step.to_edit();
        file.apply_edit(&edit)?;

        // Validate with type checker
        let typed = self.type_check(file)?;

        // Run correction for any type errors
        if !typed.errors.is_empty() {
            let corrections = self.corrector.correct(&typed.errors)?;
            for correction in corrections {
                file.apply_edit(&correction.to_edit())?;
            }
        }

        Ok(())
    }
}
```

---

## IDE/LSP Integration

Full Language Server Protocol integration for IDE support.

### LSP Features

```rust
/// Correction-aware LSP server
pub struct CorrectionLanguageServer {
    /// Correction engine
    corrector: CorrectionEngine,
    /// Document manager
    documents: DocumentManager,
}

impl LanguageServer for CorrectionLanguageServer {
    /// Provide diagnostics with correction suggestions
    fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let text = params.text_document.text;

        // Parse and type check
        let result = self.analyze(&text);

        // Generate diagnostics with corrections
        let diagnostics: Vec<_> = result.errors.iter()
            .map(|error| {
                let corrections = self.corrector.correct(error);

                Diagnostic {
                    range: error.span().to_lsp_range(),
                    severity: Some(DiagnosticSeverity::ERROR),
                    message: error.message(),
                    related_information: corrections.iter()
                        .map(|c| RelatedInformation {
                            location: Location { uri: uri.clone(), range: c.range() },
                            message: format!("Did you mean '{}'?", c.text),
                        })
                        .collect(),
                }
            })
            .collect();

        self.client.publish_diagnostics(uri, diagnostics);
    }

    /// Provide code actions for corrections
    fn code_action(&self, params: CodeActionParams) -> Vec<CodeAction> {
        let uri = &params.text_document.uri;
        let range = params.range;

        // Get corrections for this range
        let text = self.documents.get(uri);
        let error_region = &text[range.to_byte_range()];
        let corrections = self.corrector.correct_region(error_region, &text);

        corrections.iter()
            .map(|c| CodeAction {
                title: format!("Replace with '{}'", c.text),
                kind: Some(CodeActionKind::QUICKFIX),
                edit: Some(WorkspaceEdit {
                    changes: Some(hashmap! {
                        uri.clone() => vec![TextEdit { range, new_text: c.text.clone() }]
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .collect()
    }
}
```

---

## Distributed Correction

For large-scale correction across distributed systems.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Distributed Correction                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Client    │  │   Client    │  │   Client    │             │
│  │     1       │  │     2       │  │     3       │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Load Balancer                             ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  Tier 1     │     │  Tier 2     │     │  Tier 3     │       │
│  │  Workers    │     │  Workers    │     │  Workers    │       │
│  │  (Lexical)  │     │ (Syntactic) │     │ (Semantic)  │       │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘       │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │               Distributed PathMap (DAS)                     ││
│  │  • Shared dictionary                                        ││
│  │  • Grammar rules                                            ││
│  │  • Type predicates                                          ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
/// Distributed correction service
pub struct DistributedCorrector {
    /// Tier 1 worker pool
    tier1_pool: WorkerPool<Tier1Worker>,
    /// Tier 2 worker pool
    tier2_pool: WorkerPool<Tier2Worker>,
    /// Tier 3 worker pool
    tier3_pool: WorkerPool<Tier3Worker>,
    /// Distributed PathMap
    pathmap: DistributedPathMap,
}

impl DistributedCorrector {
    /// Process correction request
    pub async fn correct(&self, request: CorrectionRequest) -> CorrectionResponse {
        // Tier 1: Dispatch to lexical workers
        let tier1_result = self.tier1_pool
            .dispatch(request.clone())
            .await?;

        // Tier 2: Dispatch lattice to syntactic workers
        let tier2_result = self.tier2_pool
            .dispatch(tier1_result)
            .await?;

        // Tier 3: Dispatch to semantic workers
        let tier3_result = self.tier3_pool
            .dispatch(tier2_result)
            .await?;

        tier3_result
    }
}
```

---

## Industry Comparison

The unified correction architecture compares favorably with industry systems:

### Comparison Table

| Feature | NVIDIA NeMo | Google Sparrowhawk | MoNoise | liblevenshtein-rust |
|---------|-------------|-------------------|---------|---------------------|
| **Architecture** | FST + Neural | FST only | Pure Neural | FST + CFG + Neural |
| **Spelling correction** | ✅ FST | ✅ FST | ✅ seq2seq | ✅ Levenshtein FST |
| **Phonetic normalization** | ✅ FST rules | ✅ FST | ⚠️ Learned | ✅ NFA + verified rules |
| **Grammar correction** | ⚠️ Neural only | ❌ Not supported | ⚠️ Learned | ✅ CFG + Neural |
| **Formal verification** | ❌ None | ❌ None | ❌ None | ✅ Coq proofs |
| **Deterministic output** | ⚠️ Neural layer | ✅ Yes | ❌ No | ✅ Tiers 1-2 |
| **Latency (p50)** | ~100ms | <10ms | 100-500ms | <50ms (symbolic) |
| **Training data needed** | 10,000+ | None | 10,000+ | None (rules) |

### Unique Value Proposition

**liblevenshtein-rust** is the only system with:
1. **FST + CFG + Neural three-tier architecture**
2. **Formally verified phonetic rules** (Coq proofs)
3. **Deterministic symbolic layers** (Tiers 1-2 reproducible)
4. **Rust native performance** (zero-cost abstractions)
5. **Composable architecture** (NFA ∩ FST ∩ CFG)

**See**: [WFST Architecture - Industry Comparison](../../wfst/architecture.md#comparison-with-industry-systems) for detailed analysis.

---

## Deployment Modes

The correction architecture supports three deployment modes for different latency/accuracy trade-offs:

### Mode Configurations

| Mode | Tiers | Latency | Accuracy | Memory |
|------|-------|---------|----------|--------|
| **Fast** | FST + NFA | <20ms | ~85% | <100 MB |
| **Balanced** | FST + NFA + CFG | <200ms | ~90% | <200 MB |
| **Accurate** | FST + NFA + CFG + Neural | <500ms | ~95% | 0.5-2 GB |

### Fast Mode (Real-time)

```rust
let config = PipelineConfig {
    tiers: vec![Tier::FST, Tier::NFA],
    max_edit_distance: 2,
    phonetic_regex: Some("(ph|f)(ough|uff)..."),
    grammar: None,
    neural_lm: None,
};
```

**Use Cases**: Mobile keyboards, real-time chat, embedded devices

### Balanced Mode (Production)

```rust
let config = PipelineConfig {
    tiers: vec![Tier::FST, Tier::NFA, Tier::CFG],
    max_edit_distance: 2,
    grammar: Some(load_error_grammar("grammar.cfg")?),
    neural_lm: None,
};
```

**Use Cases**: Desktop applications, server-side normalization, document processing

### Accurate Mode (High-quality)

```rust
let config = PipelineConfig {
    tiers: vec![Tier::FST, Tier::NFA, Tier::CFG, Tier::Neural],
    grammar: Some(load_error_grammar("grammar.cfg")?),
    neural_lm: Some(BertLanguageModel::load("bert-base-uncased")?),
    neural_weight: 0.3,
};
```

**Use Cases**: Professional writing tools, academic paper correction, high-quality editing

**See**: [WFST Architecture - Deployment Modes](../../wfst/architecture.md#deployment-modes) for detailed configurations.

---

## Summary

Integration possibilities organized by domain:

### Conversational Systems

| Use Case | Key Feature | Documentation |
|----------|-------------|---------------|
| **Human Dialogue** | Context-aware correction with coreference | [Dialogue Layer](../dialogue/README.md) |
| **LLM Agent** | Preprocessing + postprocessing pipeline | [LLM Integration](../llm-integration/README.md) |
| **Chatbot QA** | Brand voice, factual accuracy, policy compliance | - |
| **Customer Support** | Domain dictionaries, sentiment, urgency | - |

### Programming Languages

| Use Case | Key Feature | Documentation |
|----------|-------------|---------------|
| **Cross-Language** | MeTTa as universal IR | - |
| **ASR Error** | Phonetic + semantic validation | - |
| **Code Completion** | Type-aware IDE integration | - |
| **Smart Contracts** | Rholang behavioral types | - |
| **Type Migration** | Incremental typing adoption | - |
| **IDE/LSP** | Full editor support | - |
| **Distributed** | Scalable architecture | - |

### Common Foundation

All integration possibilities build on the three-tier WFST architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Components                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │    PathMap      │  │   MeTTaIL       │                   │
│  │ (Shared Storage)│  │ (Type Predicates)                   │
│  └────────┬────────┘  └────────┬────────┘                   │
│           │                    │                             │
│           └──────────┬─────────┘                             │
│                      │                                       │
│                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │               Three-Tier WFST Core                       ││
│  │  Lexical (liblevenshtein) → Syntactic (MORK) → Semantic ││
│  └─────────────────────────────────────────────────────────┘│
│                      │                                       │
│       ┌──────────────┼──────────────┐                       │
│       │              │              │                        │
│       ▼              ▼              ▼                        │
│  Conversational  Programming   Distributed                   │
│    Systems        Languages      Systems                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

Key enabling technologies:
- **PathMap**: Shared storage for dictionaries, grammars, entities, dialogue state
- **MeTTaIL**: Type predicates for semantic validation
- **OSLF**: Behavioral reasoning for correctness properties
- **MORK**: Pattern matching for grammar rules
- **Dialogue Layer**: Multi-turn context management
- **Agent Learning**: Adaptive personalization

---

## References

### Core Architecture

- See [01-architecture-overview.md](./01-architecture-overview.md) for extended architecture
- See [04-tier3-semantic-type-checking.md](./04-tier3-semantic-type-checking.md) for type system
- See [05-data-flow.md](./05-data-flow.md) for complete data flow

### Extended Layers

- See [Dialogue Context Layer](../dialogue/README.md) for conversation tracking
- See [LLM Integration Layer](../llm-integration/README.md) for agent support
- See [Agent Learning Layer](../agent-learning/README.md) for adaptive correction

### Reference Materials

- See [bibliography.md](../reference/bibliography.md) for complete references
- See [gap-analysis.md](../reference/gap-analysis.md) for implementation gaps
