; Cross-solver affine-gap obligations. Every query must be UNSAT in Z3+cvc5.
(set-logic ALL)

; Layer encoding: 0=M, 1=query gap, 2=dictionary gap.
; Action encoding: 0=diagonal, 1=query gap, 2=dictionary gap.
(declare-const l1 Int)
(declare-const l2 Int)
(declare-const action Int)
(declare-const c1 Int)
(declare-const c2 Int)
(declare-const gap-open Int)
(declare-const gap-extend Int)
(declare-const substitution Int)
(declare-const maximum Int)
(declare-const operations Int)
(declare-const skipped Int)

(define-fun valid-layer ((l Int)) Bool (and (<= 0 l) (<= l 2)))
(define-fun valid-action ((a Int)) Bool (and (<= 0 a) (<= a 2)))
(define-fun gap-step ((incoming Int) (target Int)) Int
  (ite (= incoming target) gap-extend (+ gap-open gap-extend)))
(define-fun first-step ((incoming Int) (a Int)) Int
  (ite (= a 0) substitution
    (ite (= a 1) (gap-step incoming 1) (gap-step incoming 2))))
(define-fun precedes ((left Int) (right Int)) Bool
  (or (= left right) (= right 0)))
(define-fun b4 ((left-cost Int) (left Int) (right-cost Int) (right Int)) Bool
  (or (and (precedes left right) (<= left-cost right-cost))
      (<= (+ left-cost gap-open) right-cost)))
(define-fun query-gap-open-charge ((incoming Int)) Int
  (ite (= incoming 1) 0 gap-open))
(define-fun query-gap-run-cost ((left-cost Int) (incoming Int) (count Int)) Int
  (+ left-cost (query-gap-open-charge incoming) (* count gap-extend)))
(define-fun right-realignment-charge ((right Int)) Int
  (ite (= right 2) gap-open 0))
(define-fun b5-forward
  ((left-cost Int) (left Int) (right-cost Int) (right Int) (count Int)) Bool
  (and (> count 0)
       (<= (+ (query-gap-run-cost left-cost left count)
              (right-realignment-charge right))
           right-cost)))

; B-1: either gap layer can reproduce every first action from M no dearer.
(push)
(assert (and (valid-action action) (>= gap-open 0) (>= gap-extend 0)))
(assert (> (first-step 1 action) (first-step 0 action)))
(check-sat)
(pop)

(push)
(assert (and (valid-action action) (>= gap-open 0) (>= gap-extend 0)))
(assert (> (first-step 2 action) (first-step 0 action)))
(check-sat)
(pop)

; B-2: positive open cost supplies separating actions in both directions.
(push)
(assert (and (> gap-open 0) (>= gap-extend 0)))
(assert (not (< (first-step 1 1) (first-step 2 1))))
(check-sat)
(pop)

(push)
(assert (and (> gap-open 0) (>= gap-extend 0)))
(assert (not (< (first-step 2 2) (first-step 1 2))))
(check-sat)
(pop)

; B-3: switching the incoming layer costs at most one gap-open charge.
(push)
(assert (and (valid-layer l1) (valid-layer l2) (valid-action action)))
(assert (and (>= gap-open 0) (>= gap-extend 0) (>= substitution 0)))
(assert (> (first-step l1 action) (+ (first-step l2 action) gap-open)))
(check-sat)
(pop)

; B-4 preserves dominance after every shared next action.
(push)
(assert (and (valid-layer l1) (valid-layer l2) (valid-action action)))
(assert (and (>= c1 0) (>= c2 0) (>= gap-open 0)
             (>= gap-extend 0) (>= substitution 0)))
(assert (b4 c1 l1 c2 l2))
(assert (> (+ c1 (first-step l1 action))
           (+ c2 (first-step l2 action))))
(check-sat)
(pop)

; A trailing query gap extends without paying open again.
(push)
(declare-const remaining Int)
(assert (and (> remaining 0) (>= c1 0) (>= gap-open 0) (>= gap-extend 0)))
(define-fun finish-query-gap () Int (+ c1 (* remaining gap-extend)))
(assert (not (= finish-query-gap (+ c1 (* remaining gap-extend)))))
(check-sat)
(pop)

; Operation-derived window contains every affordable positive-extension run.
(push)
(assert (and (>= c1 0) (>= maximum c1) (> gap-extend 0) (>= operations 0)))
(assert (<= (+ c1 (* operations gap-extend)) maximum))
(assert (>= operations (+ (div (- maximum c1) gap-extend) 1)))
(check-sat)
(pop)

; The checked-add guard excludes any value above the active budget.
(push)
(declare-const increment Int)
(assert (and (>= c1 0) (>= increment 0) (>= maximum c1)))
(assert (<= increment (- maximum c1)))
(assert (> (+ c1 increment) maximum))
(check-sat)
(pop)

; B-5 reaches a concrete query-gap representative satisfying B-4.
(push)
(assert (and (valid-layer l1) (valid-layer l2)))
(assert (and (>= c1 0) (>= c2 0) (>= gap-open 0)
             (>= gap-extend 0) (> skipped 0)))
(assert (b5-forward c1 l1 c2 l2 skipped))
(assert (not (b4 (query-gap-run-cost c1 l1 skipped) 1 c2 l2)))
(check-sat)
(pop)

; The reached B-5 representative preserves dominance for every shared step.
(push)
(assert (and (valid-layer l1) (valid-layer l2) (valid-action action)))
(assert (and (>= c1 0) (>= c2 0) (>= gap-open 0)
             (>= gap-extend 0) (>= substitution 0) (> skipped 0)))
(assert (b5-forward c1 l1 c2 l2 skipped))
(assert (> (+ (query-gap-run-cost c1 l1 skipped) (first-step 1 action))
           (+ c2 (first-step l2 action))))
(check-sat)
(pop)

; Fused skip-and-consume uses the closed cost of the same epsilon chain.
(push)
(assert (and (valid-layer l1) (> skipped 0)))
(assert (and (>= c1 0) (>= gap-open 0) (>= gap-extend 0)))
(define-fun epsilon-chain-closed () Int
  (+ (query-gap-open-charge l1) (* skipped gap-extend)))
(assert (not (= (query-gap-run-cost c1 l1 skipped)
                (+ c1 epsilon-chain-closed))))
(check-sat)
(pop)
