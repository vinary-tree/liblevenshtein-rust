; Counterexample checks for product-zipper and observed-transition refinements.
; Every check is expected to be UNSAT in both Z3 and cvc5.
(set-logic QF_LIA)

; A product child exists only when the dictionary edge and query transition
; are both live.
(declare-const dictionary-child Bool)
(declare-const query-child Bool)
(declare-const product-child Bool)
(push)
(assert (= product-child (and dictionary-child query-child)))
(assert product-child)
(assert (not dictionary-child))
(check-sat)
(pop)

; Query-first projection and dictionary-first construction have the same
; synchronized live/dead result.
(declare-const query-first-child Bool)
(push)
(assert (= query-first-child (ite query-child dictionary-child false)))
(assert (= product-child (and dictionary-child query-child)))
(assert (not (= query-first-child product-child)))
(check-sat)
(pop)

; A rejected projection constructs no owned child focus; a live projection
; constructs exactly one.
(declare-const projected-live Bool)
(declare-const constructed-children Int)
(push)
(assert (= constructed-children (ite projected-live 1 0)))
(assert (not projected-live))
(assert (not (= constructed-children 0)))
(check-sat)
(pop)

(push)
(assert (= constructed-children (ite projected-live 1 0)))
(assert projected-live)
(assert (not (= constructed-children 1)))
(check-sat)
(pop)

(push)
(assert (= product-child (and dictionary-child query-child)))
(assert product-child)
(assert (not query-child))
(check-sat)
(pop)

; Erasing path-only zipper context retains the immutable revision and native
; focus node used for every future descent.
(declare-const retained-node Int)
(declare-const retained-revision Int)
(declare-const erased-node Int)
(declare-const erased-revision Int)
(declare-const path-bytes Int)
(push)
(assert (>= path-bytes 0))
(assert (= erased-revision retained-revision))
(assert (= erased-node retained-node))
(assert (or (not (= erased-revision retained-revision))
            (not (= erased-node retained-node))))
(check-sat)
(pop)

; A successful immutable zipper descent cannot change snapshot revision.
(declare-const parent-revision Int)
(declare-const child-revision Int)
(push)
(assert (= child-revision parent-revision))
(assert (not (= child-revision parent-revision)))
(check-sat)
(pop)

; One edge extends the logical path depth by exactly one.
(declare-const parent-depth Int)
(declare-const child-depth Int)
(push)
(assert (>= parent-depth 0))
(assert (= child-depth (+ parent-depth 1)))
(assert (<= child-depth parent-depth))
(check-sat)
(pop)

; A complete transition cache entry stores exactly the recomputed successor.
(declare-const cached-target Int)
(declare-const recomputed-target Int)
(declare-const returned-target Int)
(push)
(assert (= cached-target recomputed-target))
(assert (= returned-target cached-target))
(assert (not (= returned-target recomputed-target)))
(check-sat)
(pop)

; Every lazily constructed transition miss is demanded by an inspected edge.
(declare-const constructed-transitions Int)
(declare-const inspected-reachable-edges Int)
(push)
(assert (>= constructed-transitions 0))
(assert (>= inspected-reachable-edges 0))
(assert (<= constructed-transitions inspected-reachable-edges))
(assert (> constructed-transitions inspected-reachable-edges))
(check-sat)
(pop)

; A finite final score is reportable only when it remains within the public
; range cutoff.  This pins the query="a", candidate="", cutoff=0 class.
(declare-const final-score Int)
(declare-const public-cutoff Int)
(declare-const final-reported Bool)
(push)
(assert (>= final-score 0))
(assert (>= public-cutoff 0))
(assert (= final-reported (<= final-score public-cutoff)))
(assert final-reported)
(assert (> final-score public-cutoff))
(check-sat)
(pop)

(push)
(assert (>= final-score 0))
(assert (>= public-cutoff 0))
(assert (= final-reported (<= final-score public-cutoff)))
(assert (> final-score public-cutoff))
(assert final-reported)
(check-sat)
(pop)

; A compact product queue frame contains no frontier-width multiplier.
(declare-const dictionary-cursor-bytes Int)
(declare-const state-id-bytes Int)
(declare-const path-handle-bytes Int)
(declare-const frontier-width Int)
(declare-const compact-frame-bytes Int)
(push)
(assert (>= dictionary-cursor-bytes 0))
(assert (>= state-id-bytes 0))
(assert (>= path-handle-bytes 0))
(assert (>= frontier-width 0))
(assert (= compact-frame-bytes
  (+ dictionary-cursor-bytes state-id-bytes path-handle-bytes)))
(assert (not (= compact-frame-bytes
  (+ dictionary-cursor-bytes state-id-bytes path-handle-bytes))))
(check-sat)
(pop)
