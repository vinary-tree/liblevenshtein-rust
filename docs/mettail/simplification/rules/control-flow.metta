; Control Flow Simplification Rules
; These rules simplify conditional expressions and eliminate dead code.
;
; Status: Design Documentation
; Last Updated: 2025-12-06

; =============================================================================
; Conditional with Constant Condition
; =============================================================================

(simplification-rule
    (id if-true)
    (category control-flow)
    (phase LocalSimplification)

    (pattern (If (Bool true) $then $else))
    (template $then)

    (soundness "if true then P else Q = P")
    (termination-weight -3))  ; Eliminates condition + else branch

(simplification-rule
    (id if-false)
    (category control-flow)
    (phase LocalSimplification)

    (pattern (If (Bool false) $then $else))
    (template $else)

    (soundness "if false then P else Q = Q")
    (termination-weight -3))

; =============================================================================
; Same Branch Elimination
; =============================================================================

(simplification-rule
    (id if-same-branch)
    (category control-flow)
    (phase LocalSimplification)

    (pattern (If $cond $P $P))
    (template $P)

    (soundness "if C then P else P = P (condition irrelevant)")
    (termination-weight -2))

; =============================================================================
; Match with Single Pattern
; =============================================================================

(simplification-rule
    (id match-wildcard)
    (category control-flow)
    (phase LocalSimplification)

    (pattern (Match $scrutinee ((Wildcard) $body)))
    (template $body)

    (soundness "match x with _ => body = body")
    (termination-weight -2))

; =============================================================================
; Dead Code After Return/Throw
; =============================================================================

(simplification-rule
    (id dead-after-return)
    (category control-flow)
    (phase AnalysisDriven)

    (pattern (Seq (Return $e) $rest))
    (template (Return $e))

    (guard (is-dead-code $rest))

    (soundness "return e; rest = return e (rest is unreachable)")
    (termination-weight -1))

(simplification-rule
    (id dead-after-throw)
    (category control-flow)
    (phase AnalysisDriven)

    (pattern (Seq (Throw $e) $rest))
    (template (Throw $e))

    (guard (is-dead-code $rest))

    (soundness "throw e; rest = throw e (rest is unreachable)")
    (termination-weight -1))

; =============================================================================
; Loop Optimization
; =============================================================================

(simplification-rule
    (id while-false)
    (category control-flow)
    (phase LocalSimplification)

    (pattern (While (Bool false) $body))
    (template (Nil))

    (soundness "while false do P = nil (never executes)")
    (termination-weight -2))

(simplification-rule
    (id for-empty-range)
    (category control-flow)
    (phase LocalSimplification)

    (pattern (For $var (Range $n $n) $body))
    (template (Nil))

    (soundness "for x in n..n do P = nil (empty range)")
    (termination-weight -2))

; =============================================================================
; Negated Conditions
; =============================================================================

(simplification-rule
    (id if-not)
    (category control-flow)
    (phase LocalSimplification)

    (pattern (If (Not $cond) $then $else))
    (template (If $cond $else $then))

    (soundness "if !C then P else Q = if C then Q else P")
    (termination-weight -1))

; =============================================================================
; Nested Conditionals
; =============================================================================

(simplification-rule
    (id if-if-same-cond)
    (category control-flow)
    (phase LocalSimplification)

    (pattern (If $cond (If $cond $then $else1) $else2))
    (template (If $cond $then $else2))

    (soundness "Nested if with same condition simplifies")
    (termination-weight -2))

; =============================================================================
; Short-Circuit Evaluation
; =============================================================================

(simplification-rule
    (id and-if-fold)
    (category control-flow)
    (phase LocalSimplification)

    (pattern (If (And $a $b) $then $else))
    (template (If $a (If $b $then $else) $else))

    (guard (is-cheap $a))  ; Only unfold if first condition is cheap

    (soundness "Short-circuit AND into nested if")
    (termination-weight 1))  ; Increases size but may enable other opts

(simplification-rule
    (id or-if-fold)
    (category control-flow)
    (phase LocalSimplification)

    (pattern (If (Or $a $b) $then $else))
    (template (If $a $then (If $b $then $else)))

    (guard (is-cheap $a))

    (soundness "Short-circuit OR into nested if")
    (termination-weight 1))

; =============================================================================
; Conditional Expression to Boolean
; =============================================================================

(simplification-rule
    (id if-to-and)
    (category control-flow)
    (phase LocalSimplification)

    (pattern (If $cond (Bool true) (Bool false)))
    (template $cond)

    (soundness "if C then true else false = C")
    (termination-weight -2))

(simplification-rule
    (id if-to-or)
    (category control-flow)
    (phase LocalSimplification)

    (pattern (If $cond (Bool true) $x))
    (template (Or $cond $x))

    (soundness "if C then true else X = C || X")
    (termination-weight -1))

(simplification-rule
    (id if-to-not)
    (category control-flow)
    (phase LocalSimplification)

    (pattern (If $cond (Bool false) (Bool true)))
    (template (Not $cond))

    (soundness "if C then false else true = !C")
    (termination-weight -2))

; =============================================================================
; Try-Catch Simplification
; =============================================================================

(simplification-rule
    (id try-pure)
    (category control-flow)
    (phase AnalysisDriven)

    (pattern (Try $body $catch))
    (template $body)

    (guard (and (is-pure $body) (cannot-throw $body)))

    (soundness "try { pure-non-throwing } catch { ... } = pure body")
    (termination-weight -1))
