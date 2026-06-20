/*
 * c4-container.dsl — C4 Level 2 (Container) for liblevenshtein.
 * The internal "containers" are the major subsystems; external systems are the
 * libdictenstein backends and optional duallity WFST. Colours follow the
 * liblevenshtein diagram legend. Tags are positional; relationships are declared
 * after the model elements (identifiers are in global scope).
 */
workspace "liblevenshtein containers" "Major subsystems of liblevenshtein" {

  model {
    user = person "Application / Developer" ""
    dict = softwareSystem "libdictenstein" "Dictionary backends" "Dict"
    wfst = softwareSystem "duallity (optional)" "WFST composition" "Wfst"

    lev = softwareSystem "liblevenshtein" "Approximate matching library" {
      core       = container "Transducer core"       "Lazy / universal / generalized Levenshtein automata, subsumption" "src/transducer"  "Core"
      distance   = container "Distance"              "Myers bit-parallel + AVX2/SSE4.1 SIMD edit distance"              "src/distance"    "Dist"
      phonetic   = container "Phonetic engine"       "NFA product, .llev/.llre DSLs, articulatory features"             "src/phonetic"    "Phon"
      timeser    = container "Time-series (MSM)"     "Move-Split-Merge metric, automaton, trie index"                   "src/time_series" "Msm"
      wallbrk    = container "WallBreaker"           "SCDAWG + pigeonhole for large error bounds"                       "src/wallbreaker" "Wall"
      contextual = container "Contextual completion" "Hierarchical scopes, draft buffers, checkpoints"                  "src/contextual"  "Ctx"
      cache      = container "Fuzzy cache"           "FuzzyMultiMap + eviction wrappers"                                "src/cache"       "Cache"
      grep       = container "Grep"                  "Streaming decompress / archive / document fuzzy search"           "src/grep"        "Grep"
      surfaces   = container "Surfaces"              "CLI, REPL, WASM, FFI, serialization"                              "src/cli,repl,…"  "Surf"
    }

    user       -> surfaces "Invokes"
    user       -> core     "Queries"
    surfaces   -> core     "Drives queries"
    phonetic   -> core     "Composes pattern NFA with Levenshtein automaton"
    timeser    -> core     "Reuses position-set machinery"
    wallbrk    -> core     "Splits query, verifies pieces"
    contextual -> core     "Incremental fuzzy completion"
    grep       -> phonetic "Per-token phonetic match"
    core       -> distance "Exact distance & affix stripping"
    core       -> dict     "Lock-step traversal"
    contextual -> dict     "Draft dictionaries"
    cache      -> dict     "Wraps a dictionary"
    core       -> wfst     "Optional composition"
  }

  views {
    container lev "containers" {
      include *
      autolayout lr
    }

    styles {
      element "Person" {
        background #B2EBF2
        color #102027
        shape Person
      }
      element "Container" {
        color #102027
      }
      element "Core" {
        background #BBDEFB
      }
      element "Dist" {
        background #C5CAE9
      }
      element "Phon" {
        background #F8BBD0
      }
      element "Msm" {
        background #B2DFDB
      }
      element "Wall" {
        background #FFCCBC
      }
      element "Ctx" {
        background #D1C4E9
      }
      element "Cache" {
        background #FFF9C4
      }
      element "Grep" {
        background #FFE0B2
      }
      element "Surf" {
        background #F0F4C3
      }
      element "Dict" {
        background #C8E6C9
      }
      element "Wfst" {
        background #B3E5FC
      }
    }
  }
}
