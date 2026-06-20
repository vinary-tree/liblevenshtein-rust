[← Documentation Index](../../README.md)

# Bibliography

Complete reference list for MeTTaIL semantic type checking documentation.

---

## Primary Sources

### Meta-MeTTa: Operational Semantics

**Meredith, L. G., Radestock, M., et al.** (2023).
*Meta-MeTTa: an operational semantics for MeTTa*.
arXiv:2305.17218.

- **Summary**: Defines MeTTa as a state machine with rewrite rules
- **Key contribution**: State tuple ⟨i, k, w, o⟩ and five core rules
- **Relevance**: Foundation for formalizing MeTTa as a λ-theory
- **Section 8**: Explicitly recommends OSLF for deriving type systems
- **DOI**: [10.48550/arXiv.2305.17218](https://doi.org/10.48550/arXiv.2305.17218)

```bibtex
@article{meredith2023metametta,
  title={Meta-MeTTa: an operational semantics for MeTTa},
  author={Meredith, L. G. and others},
  journal={arXiv preprint arXiv:2305.17218},
  year={2023}
}
```

---

### Native Type Theory (OSLF)

**Williams, P. & Stay, M.** (2022).
*Native Type Theory*.
In: Applied Category Theory (ACT 2021).
EPTCS 372, pp. 116-132.

- **Summary**: Derives type systems from operational semantics via 2-functor composition
- **Key contribution**: λ-theory →[P]→ presheaf topos →[L]→ type system
- **Relevance**: Mathematical foundation for semantic type checking
- **Key concepts**: Presheaf construction, internal language, behavioral types
- **DOI**: [10.4204/EPTCS.372.9](https://doi.org/10.4204/EPTCS.372.9) (arXiv:2102.04672)

```bibtex
@inproceedings{williams2022native,
  title={Native Type Theory},
  author={Williams, P. and Stay, M.},
  booktitle={Applied Category Theory (ACT 2021)},
  series={EPTCS},
  volume={372},
  pages={116--132},
  year={2022}
}
```

---

### Gph-enriched Lawvere Theories

**Stay, M. & Meredith, L. G.** (2017).
*Representing operational semantics with enriched Lawvere theories*.
arXiv:1704.03080.

- **Summary**: Graph-enriched Lawvere theories for operational semantics
- **Key contribution**: Reductions as edges in hom-graphs
- **Relevance**: Simpler alternative when binding eliminated via reflection
- **Key concepts**: Gph-theory, reduction contexts, gas, RHO combinators
- **DOI**: [10.48550/arXiv.1704.03080](https://doi.org/10.48550/arXiv.1704.03080)

```bibtex
@article{stay2017gph,
  title={Representing operational semantics with enriched Lawvere theories},
  author={Stay, M. and Meredith, L. G.},
  journal={arXiv preprint arXiv:1704.03080},
  year={2017}
}
```

---

### RHO Calculus

**Meredith, L. G. & Radestock, M.** (2005).
*A Reflective Higher-order Calculus*.
In: Theoretical Aspects of Computing (ICTAC 2004).
ENTCS 141(5), pp. 49-67.

- **Summary**: Reflective higher-order process calculus
- **Key contribution**: Names as quoted processes (closed theory)
- **Relevance**: Theoretical foundation of Rholang
- **Key concepts**: Quote/drop reflection, N-barbed bisimulation
- **DOI**: [10.1016/j.entcs.2005.05.016](https://doi.org/10.1016/j.entcs.2005.05.016)

```bibtex
@inproceedings{meredith2005rho,
  title={A Reflective Higher-order Calculus},
  author={Meredith, L. G. and Radestock, M.},
  booktitle={ENTCS},
  volume={141},
  number={5},
  pages={49--67},
  year={2005}
}
```

---

### OpenCog Hyperon

**Goertzel, B., et al.** (2023).
*OpenCog Hyperon: A Framework for AGI at the Human Level and Beyond*.
arXiv.

- **Summary**: Comprehensive overview of OpenCog Hyperon architecture
- **Key contribution**: Atomspace metagraph, four atom meta-types, two-layer type system
- **Relevance**: Foundation of MeTTa and Atomese 2
- **Key concepts**: Gradual typing, paraconsistent logic, cognitive synergy

```bibtex
@article{goertzel2023hyperon,
  title={OpenCog Hyperon: A Framework for AGI at the Human Level and Beyond},
  author={Goertzel, B. and others},
  journal={arXiv preprint},
  year={2023}
}
```

---

## Foundational References

### Categorical Logic and Type Theory

**Jacobs, B.** (1998).
*Categorical Logic and Type Theory*.
Elsevier.

- **Summary**: Comprehensive treatment of fibrations and type theory
- **Relevance**: Mathematical background for OSLF
- **Key chapters**: Fibrations, internal language, topos theory

```bibtex
@book{jacobs1998categorical,
  title={Categorical Logic and Type Theory},
  author={Jacobs, B.},
  publisher={Elsevier},
  year={1998}
}
```

---

### Lawvere Theories

**Lawvere, F. W.** (1963).
*Functorial Semantics of Algebraic Theories*.
PhD thesis, Columbia University.
(Reprinted in: Reprints in Theory and Applications of Categories, No. 5, 2004, pp. 1-121)

- **Summary**: Original introduction of Lawvere theories
- **Relevance**: Foundation for multi-sorted theories
- **Key concept**: Categories as algebraic theories

```bibtex
@phdthesis{lawvere1963functorial,
  title={Functorial Semantics of Algebraic Theories},
  author={Lawvere, F. W.},
  school={Columbia University},
  year={1963}
}
```

---

### Pi-Calculus

**Sangiorgi, D. & Walker, D.** (2001).
*The π-calculus: A Theory of Mobile Processes*.
Cambridge University Press.

- **Summary**: Comprehensive treatment of π-calculus
- **Relevance**: Background for understanding RHO calculus
- **Key concepts**: Bisimulation, behavioral equivalence, mobility

```bibtex
@book{sangiorgi2001pi,
  title={The $\pi$-calculus: A Theory of Mobile Processes},
  author={Sangiorgi, D. and Walker, D.},
  publisher={Cambridge University Press},
  year={2001}
}
```

---

### Modal Logic

**Blackburn, P., de Rijke, M., & Venema, Y.** (2001).
*Modal Logic*.
Cambridge University Press.

- **Summary**: Comprehensive introduction to modal logic
- **Relevance**: Background for behavioral modalities (◇, □)
- **Key concepts**: Kripke semantics, fixed points, temporal logic

```bibtex
@book{blackburn2001modal,
  title={Modal Logic},
  author={Blackburn, P. and de Rijke, M. and Venema, Y.},
  publisher={Cambridge University Press},
  year={2001}
}
```

---

### Presheaf Models

**Mac Lane, S. & Moerdijk, I.** (1992).
*Sheaves in Geometry and Logic: A First Introduction to Topos Theory*.
Springer.

- **Summary**: Introduction to topos theory and presheaves
- **Relevance**: Mathematical background for presheaf construction P
- **Key chapters**: Presheaves, Yoneda lemma, subobject classifier

```bibtex
@book{maclane1992sheaves,
  title={Sheaves in Geometry and Logic},
  author={Mac Lane, S. and Moerdijk, I.},
  publisher={Springer},
  year={1992}
}
```

---

## Implementation References

### BNFC

**Forsberg, M. & Ranta, A.** (2004).
*The BNF Converter: A High-Level Tool for Implementing Well-Behaved Parsers*.

- **Website**: https://bnfc.digitalgrammars.com/
- **Relevance**: Parser generation in MeTTaIL Scala

```bibtex
@article{forsberg2004bnfc,
  title={The {BNF} Converter},
  author={Forsberg, M. and Ranta, A.},
  year={2004}
}
```

---

### Ascent Datalog

**Ascent: Logic programming embedded in Rust**.

- **Repository**: https://github.com/s-arash/ascent
- **Relevance**: Datalog engine for mettail-rust

```bibtex
@software{ascent,
  title={Ascent: Logic programming embedded in Rust},
  url={https://github.com/s-arash/ascent}
}
```

---

### LALRPOP

**LALRPOP: LR(1) parser generator for Rust**.

- **Repository**: https://github.com/lalrpop/lalrpop
- **Relevance**: Parser implementation in mettail-rust

```bibtex
@software{lalrpop,
  title={LALRPOP: LR(1) parser generator for Rust},
  url={https://github.com/lalrpop/lalrpop}
}
```

---

### Tree-sitter

**Tree-sitter: An incremental parsing system**.

- **Website**: https://tree-sitter.github.io/tree-sitter/
- **Relevance**: Rholang parser implementation

```bibtex
@software{treesitter,
  title={Tree-sitter},
  url={https://tree-sitter.github.io/tree-sitter/}
}
```

---

## Project Sources

### MeTTaIL Scala Prototype

- **Location**: `/home/dylon/Workspace/f1r3fly.io/MeTTaIL/`
- **Description**: Theory transformation and BNFC generation

### mettail-rust Prototype

- **Location**: `/home/dylon/Workspace/f1r3fly.io/mettail-rust/`
- **Description**: High-performance rewrite engine and type checking

### Rholang (f1r3node)

- **Location**: `/home/dylon/Workspace/f1r3fly.io/f1r3node/`
- **Branches**: `new_parser`, `dylon/mettatron`
- **Description**: Rholang implementation with MeTTa integration

### rholang-rs Parser

- **Location**: `/home/dylon/Workspace/f1r3fly.io/rholang-rs/rholang-parser/`
- **Description**: Tree-sitter based Rholang parser

### PathMap

- **Location**: `/home/dylon/Workspace/f1r3fly.io/PathMap/`
- **Description**: Trie-based storage with zipper navigation and ring module

### MORK (MeTTa Optimal Reduction Kernel)

- **Location**: `/home/dylon/Workspace/f1r3fly.io/MORK/`
- **Description**: Pattern matching kernel using PathMap backend

### MeTTaTron

- **Location**: `/home/dylon/Workspace/f1r3fly.io/MeTTa-Compiler/`
- **Description**: F1R3FLY.io's MeTTa compiler with MORK/PathMap integration

### hyperon-experimental

- **Location**: `/home/dylon/Workspace/f1r3fly.io/hyperon-experimental/`
- **Description**: Official OpenCog Hyperon MeTTa implementation

### liblevenshtein

- **Location**: `/home/dylon/Workspace/f1r3fly.io/liblevenshtein-rust/`
- **Description**: Edit distance automata, phonetic rules, fuzzy dictionaries

---

## rho4u Research Papers

Local research papers and working documents providing theoretical foundations.

### OSLF Papers

**Native Type Theory / OSLF**

- **Location**: `/home/dylon/Workspace/f1r3fly.io/rho4u/oslf/oslf.pdf`
- **Description**: Full OSLF paper with 2-functor construction
- **Key contribution**: λ-theory →[P]→ presheaf topos →[L]→ type system

**OSLF to Native Types Clean Type System**

- **Location**: `/home/dylon/Workspace/f1r3fly.io/rho4u/oslf/OSLF2NativeTypesCleanTypeSystem (1).pdf`
- **Description**: Detailed derivation of clean type systems via OSLF algorithm
- **Key contribution**: Step-by-step type system generation from operational semantics

### CALCO 2017 Paper

**Representing Operational Semantics with Enriched Lawvere Theories**

- **Location**: `/home/dylon/Workspace/f1r3fly.io/rho4u/Representing operational semantics with enriched Lawvere theories calco2017/calco.pdf`
- **Description**: Full CALCO conference paper on Gph-enriched Lawvere theories
- **Key contribution**: Graph-enriched theories for operational semantics without binding
- **Key concepts**: Reduction contexts, gas consumption, RHO combinators

### Hypercube / GSLTs

**Graph-Structured Lambda Theories**

- **Location (Markdown)**: `/home/dylon/Workspace/f1r3fly.io/rho4u/hypercube/map.md`
- **Location (PDF)**: `/home/dylon/Workspace/f1r3fly.io/rho4u/hypercube/map.pdf`
- **Description**: Complete GSLT definition with magmal structure
- **Key contribution**: Multi-sorted, graph-structured, magmal, lambda theories
- **Key concepts**:
  - Generating sorts: List of sort names
  - Interaction: Magmal operation F^T on sorts
  - Morphisms: Function symbol (name, domain sorts, codomain sort)
  - Equations: Equivalence classes on terms
- **Relevance**: Simpler path to semantic types when binding eliminated via reflection

### TOGL (Theory of Graphs)

**Algebraic Theory of Graphs**

- **Location (Markdown)**: `/home/dylon/Workspace/f1r3fly.io/rho4u/togl/togl.md`
- **Location (PDF)**: `/home/dylon/Workspace/f1r3fly.io/rho4u/togl/togl.pdf`
- **Description**: Algebraic theory of graphs G[X,V] with OSLF application
- **Key contribution**: Well-formedness judgments, recursive domain definition
- **Key concepts**:
  - Graph functor: G[X,V] = X + V × G[X,V] × G[X,V]
  - Recursive domain: D = G[1 + D, 1 + D]
  - OSLF application to graph theory
- **Relevance**: Foundation for graph-based semantic reasoning

### RHOcube

**Spatial-Behavioral Types for RHO Calculus**

- **Location (PDF)**: `/home/dylon/Workspace/f1r3fly.io/rho4u/rhocube/rhocube.pdf`
- **Location (LaTeX)**: `/home/dylon/Workspace/f1r3fly.io/rho4u/rhocube/rhocube.spatial-behavioral.types.tex`
- **Description**: Concrete type inference rules generated by OSLF algorithm
- **Key contribution**: Complete spatial-behavioral type system for RHO calculus
- **Type syntax**:
  ```
  T,U ::= 0 | GT | ⟨(TT → N)⟩T | ⟨x?⟩T | *N | T|U
  N ::= @T
  GT ::= Bool | String | Int | C
  ```
- **Key inference rules**:
  - `for-comprehension`: Γ ⊢ for(t ← x)P : ⟨(TT → V)⟩T
  - `parallel`: Γ ⊢ P|Q : T | U
  - `quote/deref`: Γ ⊢ @P : @T and Γ ⊢ *x : *V
- **Relevance**: Actionable type rules for Rholang semantic checking

### MeTTa Calculus

**MeTTa as a Formal Calculus**

- **Location**: `/home/dylon/Workspace/f1r3fly.io/rho4u/metta-calculus/metta-calculus.pdf`
- **Description**: Formal calculus treatment of MeTTa
- **Key contribution**: Rigorous mathematical treatment of MeTTa rewriting
- **Relevance**: Foundation for formalizing MeTTa as λ-theory for OSLF

### Space Calculus

**Spatial Calculus Foundations**

- **Location**: `/home/dylon/Workspace/f1r3fly.io/rho4u/space-calc/space-calc.pdf`
- **Description**: Spatial calculus with location-aware computation
- **Key contribution**: Spatial reasoning for distributed processes
- **Relevance**: Foundation for spatial types in correction WFST Tier 3

---

## Related Work

### Session Types

**Honda, K., Vasconcelos, V., & Kubo, M.** (1998).
*Language Primitives and Type Discipline for Structured Communication-Based Programming*.
ESOP.

- **Relevance**: Alternative approach to behavioral types for processes

### Spatial Types

**Cardelli, L. & Gordon, A. D.** (2000).
*Anytime, Anywhere: Modal Logics for Mobile Ambients*.
POPL.

- **Relevance**: Spatial reasoning for process calculi

### Linear Types

**Wadler, P.** (2012).
*Propositions as Sessions*.
ICFP.

- **Relevance**: Linear types for session-typed processes

---

## Reading Order

For understanding the full theory:

1. **Start**: Meta-MeTTa (Meredith et al., 2023) - understand MeTTa
2. **Theory**: Native Type Theory (Williams & Stay, 2022) - OSLF construction
3. **Alternative**: Gph-enriched Lawvere (Stay & Meredith, 2017) - simpler path
4. **Integration**: RHO Calculus (Meredith & Radestock, 2005) - Rholang foundation
5. **Background**: Jacobs (1998) for categorical logic as needed

For understanding the ecosystem:

1. **Foundation**: OpenCog Hyperon (Goertzel et al., 2023) - architecture overview
2. **Reference Implementation**: hyperon-experimental - official MeTTa
3. **F1R3FLY Stack**: MeTTaTron → MORK → PathMap integration chain

For correction WFST integration:

1. **Tier 1**: liblevenshtein - edit distance and phonetic rules
2. **Tier 2**: MORK/PathMap - syntactic validation
3. **Tier 3**: MeTTaIL/MeTTaTron - semantic type checking

For implementation:

1. **Current**: Review mettail-rust source
2. **Integration**: Study f1r3node mettatron branch
3. **Tools**: BNFC, Ascent, LALRPOP, Tree-sitter documentation
