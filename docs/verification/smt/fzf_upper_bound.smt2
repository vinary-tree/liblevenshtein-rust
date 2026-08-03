; Recurrence-derived capacity-sensitive fzf bound checks.
(set-logic QF_NIA)
(define-fun max2 ((a Int) (b Int)) Int (ite (>= a b) a b))
(define-fun feasible ((ok Bool) (completed Int) (term Int)) Int
  (ite ok term completed))
(define-fun bound ((completed Int) (unstarted Int) (active Int)
                   (query-len Int) (active-remaining Int)
                   (capacity Int) (beta Int)) Int
  (max2 completed
    (max2 (feasible (<= query-len capacity) completed unstarted)
          (feasible (<= active-remaining capacity) completed
                    (+ active (* active-remaining beta))))))

; A gap cannot increase the active score.
(push)
(declare-const c Int) (declare-const u Int) (declare-const a Int)
(declare-const q Int) (declare-const r Int) (declare-const cap Int)
(declare-const beta Int) (declare-const child Int)
(assert (>= beta 0)) (assert (<= r cap)) (assert (<= child a))
(assert (> (+ child (* r beta)) (bound c u a q r cap beta)))
(check-sat)
(pop)

; A match consumes one remaining query unit and gains at most beta.
(push)
(declare-const c Int) (declare-const u Int) (declare-const a Int)
(declare-const q Int) (declare-const r Int) (declare-const cap Int)
(declare-const beta Int) (declare-const child Int)
(assert (>= beta 0)) (assert (<= (+ r 1) cap))
(assert (<= child (+ a beta)))
(assert (> (+ child (* r beta)) (bound c u a q (+ r 1) cap beta)))
(check-sat)
(pop)

; A newly started child is covered only when the whole query still fits.
(push)
(declare-const c Int) (declare-const u Int) (declare-const a Int)
(declare-const q Int) (declare-const r Int) (declare-const cap Int)
(declare-const beta Int) (declare-const child Int)
(assert (<= q cap)) (assert (<= child u))
(assert (> child (bound c u a q r cap beta)))
(check-sat)
(pop)

; Derived pruning consequence.
(push)
(declare-const score Int) (declare-const upper Int) (declare-const cutoff Int)
(assert (<= score upper)) (assert (< upper cutoff)) (assert (>= score cutoff))
(check-sat)
(pop)

(push)
(declare-const initial Int) (declare-const middle Int) (declare-const final-score Int)
(assert (not (= (+ initial (- middle initial) (- final-score middle)) final-score)))
(check-sat)
(pop)
