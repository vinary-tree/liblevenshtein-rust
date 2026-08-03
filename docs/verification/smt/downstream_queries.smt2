; Cross-solver counterexample checks for Phase-9 downstream query surfaces.
; Every check is expected to be UNSAT in both Z3 and cvc5.
(set-logic QF_NIA)

(declare-const kinds Int)
(declare-const left-token Int)
(declare-const right-token Int)
(declare-const enters Int)
(declare-const leaves Int)
(declare-const ld Int)
(declare-const lc Int)
(declare-const lt Int)
(declare-const rd Int)
(declare-const rc Int)
(declare-const rt Int)
(declare-const left-index Int)
(declare-const right-index Int)
(declare-const minimum Int)
(declare-const slack Int)
(declare-const mode-minimum Int)
(declare-const mode-maximum Int)
(declare-const candidate-distance Int)

(define-fun subst-cost ((left Int) (right Int)) Int
  (ite (= left right) 0 1))
(define-fun erase-kind ((k Int) (token Int)) Int
  (ite (< token k) 0 1))
(define-fun projected-cost ((k Int) (left Int) (right Int)) Int
  (subst-cost (erase-kind k left) (erase-kind k right)))
(define-fun ranked-before
  ((d1 Int) (c1 Int) (t1 Int) (d2 Int) (c2 Int) (t2 Int)) Bool
  (or (< d1 d2)
      (and (= d1 d2)
           (or (> c1 c2) (and (= c1 c2) (<= t1 t2))))))
(define-fun offset ((left Int) (right Int)) Int
  (ite (<= left right) (- right left) (- left right)))
(define-fun safe
  ((left Int) (right Int) (min-cost Int) (available Int)) Bool
  (<= (* (offset left right) min-cost) available))
(define-fun mode-accepts
  ((minimum Int) (maximum Int) (distance Int)) Bool
  (and (<= minimum distance) (<= distance maximum)))

; Kind erasure cannot increase an aligned substitution cost.
(push)
(assert (> (projected-cost kinds left-token right-token)
           (subst-cost left-token right-token)))
(check-sat)
(pop)

; Exact mode accepts exactly the requested completed-candidate distance.
(push)
(assert (and (>= candidate-distance 0) (>= mode-minimum 0)))
(assert (not (= (mode-accepts mode-minimum mode-minimum candidate-distance)
                (= candidate-distance mode-minimum))))
(check-sat)
(pop)

; Every range-mode result fits the maximum used as the automaton budget.
(push)
(assert (and (>= mode-minimum 0) (>= mode-maximum 0)
             (>= candidate-distance 0)))
(assert (mode-accepts mode-minimum mode-maximum candidate-distance))
(assert (> candidate-distance mode-maximum))
(check-sat)
(pop)

; A balanced visitor remains balanced after enter plus leave, including reject.
(push)
(assert (= enters leaves))
(assert (not (= (+ enters 1) (+ leaves 1))))
(check-sat)
(pop)

; The geometric stack-state count for k=3, D=10 is exactly 88,573.
(push)
(assert (not (= (+ 1 3 9 27 81 243 729 2187 6561 19683 59049) 88573)))
(check-sat)
(pop)

; The same request cannot fit under the public 4,096-state guard.
(push)
(assert (<= (+ 1 3 9 27 81 243 729 2187 6561 19683 59049) 4096))
(check-sat)
(pop)

; Mutual rank precedence implies equal distance, confidence, and lexical key.
(push)
(assert (and (>= ld 0) (>= lc 0) (>= lt 0)
             (>= rd 0) (>= rc 0) (>= rt 0)))
(assert (ranked-before ld lc lt rd rc rt))
(assert (ranked-before rd rc rt ld lc lt))
(assert (or (not (= ld rd)) (not (= lc rc)) (not (= lt rt))))
(check-sat)
(pop)

; Contextual realignment is symmetric in the two DP indices.
(push)
(assert (and (>= left-index 0) (>= right-index 0)
             (> minimum 0) (>= slack 0)))
(assert (not (= (safe left-index right-index minimum slack)
                (safe right-index left-index minimum slack))))
(check-sat)
(pop)

; With a positive minimum edit cost, zero slack permits no displacement.
(push)
(assert (and (>= left-index 0) (>= right-index 0) (> minimum 0)))
(assert (safe left-index right-index minimum 0))
(assert (not (= left-index right-index)))
(check-sat)
(pop)
