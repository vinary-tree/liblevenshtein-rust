; Refactoring-equivalence checks for the Phase 11 subsumption fallback.
; Each query asks for a counterexample and must be UNSAT in Z3 and cvc5.
(set-logic ALL)

(define-fun abs-int ((x Int)) Int (ite (< x 0) (- x) x))

(define-fun shared-unit
  ((mode Int) (li Int) (lc Int) (ls Bool)
   (ri Int) (rc Int) (rs Bool) (ql Int) (step Int)) Bool
  (and (<= lc rc)
    (ite (= mode 0)
      (<= (* (abs-int (- li ri)) step) (- rc lc))
      (ite (= mode 1)
        (ite (and ls rs) (= li ri)
          (ite (or ls rs) false
            (<= (* (abs-int (- li ri)) step) (- rc lc))))
        (and (= ls rs) (<= li ql)
          (not (and ls (>= li ql) (< ri ql)))
          (< lc rc) (= li ri))))))

(define-fun legacy-unit
  ((mode Int) (li Int) (lc Int) (ls Bool)
   (ri Int) (rc Int) (rs Bool) (ql Int) (step Int)) Bool
  (ite (= mode 0)
    (and (<= lc rc) (<= (* (abs-int (- li ri)) step) (- rc lc)))
    (ite (= mode 1)
      (ite (and ls rs) (and (<= lc rc) (= li ri))
        (ite (or ls rs) false
          (and (<= lc rc) (<= (* (abs-int (- li ri)) step) (- rc lc)))))
      (and (<= lc rc) (= ls rs) (<= li ql)
        (not (and ls (>= li ql) (< ri ql)))
        (< lc rc) (= li ri)))))

(declare-const mode Int)
(declare-const li Int)
(declare-const lc Int)
(declare-const ls Bool)
(declare-const ri Int)
(declare-const rc Int)
(declare-const rs Bool)
(declare-const ql Int)
(declare-const step Int)
(push)
(assert (and (<= 0 mode) (<= mode 2)
             (<= 0 li) (<= 0 lc) (<= 0 ri) (<= 0 rc)
             (<= 0 ql) (<= 0 step)))
(assert (not (= (shared-unit mode li lc ls ri rc rs ql step)
                (legacy-unit mode li lc ls ri rc rs ql step))))
(check-sat)
(pop)

(define-fun shared-weighted
  ((mode Int) (li Int) (lc Real) (ls Bool)
   (ri Int) (rc Real) (rs Bool) (ql Int) (step Real)) Bool
  (and (<= lc rc)
    (ite (= mode 0)
      (<= (* (to_real (abs-int (- li ri))) step)
          (- rc lc))
      (ite (= mode 1)
        (ite (and ls rs) (= li ri)
          (ite (or ls rs) false
            (<= (* (to_real (abs-int (- li ri))) step)
                (- rc lc))))
        (and (= ls rs) (<= li ql)
          (not (and ls (>= li ql) (< ri ql)))
          (< lc rc) (= li ri))))))

(define-fun legacy-weighted
  ((mode Int) (li Int) (lc Real) (ls Bool)
   (ri Int) (rc Real) (rs Bool) (ql Int) (step Real)) Bool
  (ite (= mode 0)
    (and (<= lc rc)
      (<= (* (to_real (abs-int (- li ri))) step)
          (- rc lc)))
    (ite (= mode 1)
      (ite (and ls rs)
        (and (<= lc rc) (= li ri))
        (ite (or ls rs) false
          (and (<= lc rc)
            (<= (* (to_real (abs-int (- li ri))) step)
                (- rc lc)))))
      (and (<= lc rc)
        (= ls rs) (<= li ql)
        (not (and ls (>= li ql) (< ri ql)))
        (< lc rc) (= li ri)))))

(declare-const lcr Real)
(declare-const rcr Real)
(declare-const stepr Real)
(push)
(assert (and (<= 0 mode) (<= mode 2)
             (<= 0 li) (<= 0 lcr) (<= 0 ri) (<= 0 rcr)
             (<= 0 ql) (<= 0 stepr)))
(assert (not (= (shared-weighted mode li lcr ls ri rcr rs ql stepr)
                (legacy-weighted mode li lcr ls ri rcr rs ql stepr))))
(check-sat)
(pop)

; Exact weighted dominance can never accept a more expensive representative,
; including differences smaller than the former epsilon tolerance.
(push)
(assert (> lcr rcr))
(assert (shared-weighted 0 li lcr false li rcr false ql stepr))
(check-sat)
(pop)
