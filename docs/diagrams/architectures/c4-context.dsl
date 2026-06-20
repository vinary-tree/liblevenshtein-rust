/*
 * c4-context.dsl — C4 Level 1 (System Context) for liblevenshtein.
 * The embedding application uses liblevenshtein, which traverses libdictenstein
 * dictionaries and may optionally compose with duallity (WFST). Colours follow
 * the liblevenshtein diagram legend. Tags are positional (Structurizr DSL does
 * not accept an inline `{ tags ... }` block after a description).
 */
workspace "liblevenshtein" "Approximate string-matching automata library" {

  model {
    user = person "Application / Developer" "Embeds the library for fuzzy matching, spell-checking, completion"
    lev  = softwareSystem "liblevenshtein" "Levenshtein & phonetic automata, fuzzy maps, completion (v0.9.1)" "Core"
    dict = softwareSystem "libdictenstein" "Dictionary backends: tries, DAWGs, suffix automata, ART" "Dict"
    wfst = softwareSystem "duallity (optional)" "Weighted finite-state transducer / language-model composition" "Wfst"

    user -> lev  "Queries terms within edit distance k; builds dictionaries"
    lev  -> dict "Traverses dictionaries in lock-step during a query"
    lev  -> wfst "Optionally composes automata with WFSTs"
  }

  views {
    systemContext lev "context" {
      include *
      autolayout lr
    }

    styles {
      element "Person" {
        background #B2EBF2
        color #102027
        shape Person
      }
      element "Core" {
        background #BBDEFB
        color #102027
      }
      element "Dict" {
        background #C8E6C9
        color #102027
      }
      element "Wfst" {
        background #B3E5FC
        color #102027
      }
    }
  }
}
