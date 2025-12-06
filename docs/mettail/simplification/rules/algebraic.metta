; Algebraic Simplification Rules
; These rules implement standard algebraic identities for arithmetic and boolean operations.
;
; Status: Design Documentation
; Last Updated: 2025-12-06

; =============================================================================
; Additive Identity Rules
; =============================================================================

(simplification-rule
    (id add-zero-right)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Plus $x (Num 0)))
    (template $x)

    (soundness "x + 0 = x by additive identity")
    (termination-weight -1))

(simplification-rule
    (id add-zero-left)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Plus (Num 0) $x))
    (template $x)

    (soundness "0 + x = x by additive identity")
    (termination-weight -1))

; =============================================================================
; Multiplicative Identity Rules
; =============================================================================

(simplification-rule
    (id mul-one-right)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Mult $x (Num 1)))
    (template $x)

    (soundness "x * 1 = x by multiplicative identity")
    (termination-weight -1))

(simplification-rule
    (id mul-one-left)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Mult (Num 1) $x))
    (template $x)

    (soundness "1 * x = x by multiplicative identity")
    (termination-weight -1))

; =============================================================================
; Multiplicative Zero Rules
; =============================================================================

(simplification-rule
    (id mul-zero-right)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Mult $x (Num 0)))
    (template (Num 0))

    (soundness "x * 0 = 0 by multiplicative zero")
    (termination-weight -2))  ; Eliminates entire subterm

(simplification-rule
    (id mul-zero-left)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Mult (Num 0) $x))
    (template (Num 0))

    (soundness "0 * x = 0 by multiplicative zero")
    (termination-weight -2))

; =============================================================================
; Double Negation
; =============================================================================

(simplification-rule
    (id double-neg-arith)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Neg (Neg $x)))
    (template $x)

    (soundness "--x = x by double negation")
    (termination-weight -1))

(simplification-rule
    (id double-neg-bool)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Not (Not $x)))
    (template $x)

    (soundness "!!x = x by double negation")
    (termination-weight -1))

; =============================================================================
; Idempotent Operations
; =============================================================================

(simplification-rule
    (id idempotent-and)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (And $x $x))
    (template $x)

    (soundness "x && x = x by idempotence of AND")
    (termination-weight -1))

(simplification-rule
    (id idempotent-or)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Or $x $x))
    (template $x)

    (soundness "x || x = x by idempotence of OR")
    (termination-weight -1))

; =============================================================================
; Absorption Laws
; =============================================================================

(simplification-rule
    (id and-true-right)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (And $x (Bool true)))
    (template $x)

    (soundness "x && true = x")
    (termination-weight -1))

(simplification-rule
    (id and-true-left)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (And (Bool true) $x))
    (template $x)

    (soundness "true && x = x")
    (termination-weight -1))

(simplification-rule
    (id and-false-right)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (And $x (Bool false)))
    (template (Bool false))

    (soundness "x && false = false")
    (termination-weight -2))

(simplification-rule
    (id and-false-left)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (And (Bool false) $x))
    (template (Bool false))

    (soundness "false && x = false")
    (termination-weight -2))

(simplification-rule
    (id or-true-right)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Or $x (Bool true)))
    (template (Bool true))

    (soundness "x || true = true")
    (termination-weight -2))

(simplification-rule
    (id or-true-left)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Or (Bool true) $x))
    (template (Bool true))

    (soundness "true || x = true")
    (termination-weight -2))

(simplification-rule
    (id or-false-right)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Or $x (Bool false)))
    (template $x)

    (soundness "x || false = x")
    (termination-weight -1))

(simplification-rule
    (id or-false-left)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Or (Bool false) $x))
    (template $x)

    (soundness "false || x = x")
    (termination-weight -1))

; =============================================================================
; Constant Folding (compile-time evaluation)
; =============================================================================

(simplification-rule
    (id const-fold-plus)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Plus (Num $a) (Num $b)))
    (template (Num (+ $a $b)))

    (soundness "Compile-time addition")
    (termination-weight -1))

(simplification-rule
    (id const-fold-mult)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Mult (Num $a) (Num $b)))
    (template (Num (* $a $b)))

    (soundness "Compile-time multiplication")
    (termination-weight -1))

(simplification-rule
    (id const-fold-minus)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Minus (Num $a) (Num $b)))
    (template (Num (- $a $b)))

    (soundness "Compile-time subtraction")
    (termination-weight -1))

(simplification-rule
    (id const-fold-and)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (And (Bool $a) (Bool $b)))
    (template (Bool (and $a $b)))

    (soundness "Compile-time AND")
    (termination-weight -1))

(simplification-rule
    (id const-fold-or)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Or (Bool $a) (Bool $b)))
    (template (Bool (or $a $b)))

    (soundness "Compile-time OR")
    (termination-weight -1))

; =============================================================================
; Self-Inverse Operations
; =============================================================================

(simplification-rule
    (id sub-self)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Minus $x $x))
    (template (Num 0))

    (soundness "x - x = 0")
    (termination-weight -1))

(simplification-rule
    (id xor-self)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Xor $x $x))
    (template (Num 0))

    (soundness "x ^ x = 0 (bitwise XOR)")
    (termination-weight -1))
