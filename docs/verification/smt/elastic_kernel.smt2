; Cross-solver counterexample checks for ElasticKernel K1--K4 and interval geometry.
; Every check is expected to be UNSAT in both Z3 and cvc5.
(set-logic QF_LIA)

(declare-const node-bound Int)
(declare-const candidate-bound Int)
(declare-const exact Int)
(declare-const cutoff Int)
(declare-const popped-bound Int)
(declare-const queued-bound Int)

; K1: pruning a subtree bound cannot discard an in-cutoff exact candidate.
(push)
(assert (<= node-bound exact))
(assert (> node-bound cutoff))
(assert (<= exact cutoff))
(check-sat)
(pop)

; A prefix-prune observation preserves edges = prefix-pruned + columns-built.
(declare-const edges Int)
(declare-const prefix-pruned Int)
(declare-const columns-built Int)
(push)
(assert (>= edges 0))
(assert (>= prefix-pruned 0))
(assert (>= columns-built 0))
(assert (= edges (+ prefix-pruned columns-built)))
(assert (not (= (+ edges 1) (+ (+ prefix-pruned 1) columns-built))))
(check-sat)
(pop)

; A column-build observation preserves the same exclusive partition.
(push)
(assert (>= edges 0))
(assert (>= prefix-pruned 0))
(assert (>= columns-built 0))
(assert (= edges (+ prefix-pruned columns-built)))
(assert (not (= (+ edges 1) (+ prefix-pruned (+ columns-built 1)))))
(check-sat)
(pop)

; Candidate-bound and exact decisions partition admitted final candidates.
(declare-const candidates Int)
(declare-const candidate-pruned Int)
(declare-const exact-evaluations Int)
(push)
(assert (>= candidates 0))
(assert (>= candidate-pruned 0))
(assert (>= exact-evaluations 0))
(assert (= candidates (+ candidate-pruned exact-evaluations)))
(assert (not (= (+ candidates 1) (+ (+ candidate-pruned 1) exact-evaluations))))
(check-sat)
(pop)

(push)
(assert (>= candidates 0))
(assert (>= candidate-pruned 0))
(assert (>= exact-evaluations 0))
(assert (= candidates (+ candidate-pruned exact-evaluations)))
(assert (not (= (+ candidates 1) (+ candidate-pruned (+ exact-evaluations 1)))))
(check-sat)
(pop)

; A reported cutoff abandonment cannot exceed exact evaluations.
(declare-const cutoff-abandoned Int)
(push)
(assert (>= exact-evaluations 0))
(assert (>= cutoff-abandoned 0))
(assert (<= cutoff-abandoned exact-evaluations))
(assert (> cutoff-abandoned exact-evaluations))
(check-sat)
(pop)

; K4: the candidate-level lower bound obeys the same rule.
(push)
(assert (<= candidate-bound exact))
(assert (> candidate-bound cutoff))
(assert (<= exact cutoff))
(check-sat)
(pop)

; Best-first termination: the popped minimum dominates all queued bounds.
(push)
(assert (<= popped-bound queued-bound))
(assert (<= queued-bound exact))
(assert (> popped-bound cutoff))
(assert (<= exact cutoff))
(check-sat)
(pop)

; K2 for additive non-negative steps.
(declare-const accumulated Int)
(declare-const step Int)
(push)
(assert (>= accumulated 0))
(assert (>= step 0))
(assert (< (+ accumulated step) accumulated))
(check-sat)
(pop)

; K2 for bottleneck/max combination.
(define-fun imax ((a Int) (b Int)) Int (ite (<= a b) b a))
(push)
(assert (< (imax accumulated step) accumulated))
(check-sat)
(pop)

; The closed-form interval gap is symmetric.
(declare-const al Int)
(declare-const ah Int)
(declare-const bl Int)
(declare-const bh Int)
(define-fun gap ((xl Int) (xh Int) (yl Int) (yh Int)) Int
  (imax 0 (imax (- xl yh) (- yl xh))))
(push)
(assert (<= al ah))
(assert (<= bl bh))
(assert (not (= (gap al ah bl bh) (gap bl bh al ah))))
(check-sat)
(pop)

; Degenerate intervals reproduce scalar absolute distance exactly.
(define-fun iabs ((x Int)) Int (ite (>= x 0) x (- x)))
(push)
(assert (not (= (gap al al bl bl) (iabs (- al bl)))))
(check-sat)
(pop)
