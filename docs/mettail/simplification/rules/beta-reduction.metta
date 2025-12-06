; Beta Reduction and Inlining Rules
; These rules handle lambda application, let binding simplification, and function inlining.
;
; Status: Design Documentation
; Last Updated: 2025-12-06

; =============================================================================
; Beta Reduction (Lambda Application)
; =============================================================================

(simplification-rule
    (id beta-reduction)
    (category beta-reduction)
    (phase LocalSimplification)

    (pattern (App (Lambda $x $body) $arg))
    (template (Subst $body $x $arg))

    (guard (or (is-small $arg) (linear-use $x $body)))

    (soundness "(lambda x. body)(arg) = body[arg/x]")
    (termination-weight -1))

(simplification-rule
    (id beta-reduction-value)
    (category beta-reduction)
    (phase LocalSimplification)

    (pattern (App (Lambda $x $body) $v))
    (template (Subst $body $x $v))

    (guard (is-value $v))

    (soundness "(lambda x. body)(v) = body[v/x] for values")
    (termination-weight -1))

; =============================================================================
; Eta Reduction
; =============================================================================

(simplification-rule
    (id eta-reduction)
    (category beta-reduction)
    (phase LocalSimplification)

    (pattern (Lambda $x (App $f $x)))
    (template $f)

    (guard (not (free-in $x $f)))

    (soundness "lambda x. f(x) = f when x not in FV(f)")
    (termination-weight -1))

; =============================================================================
; Let Binding Simplification
; =============================================================================

(simplification-rule
    (id let-unused)
    (category inlining)
    (phase AnalysisDriven)

    (pattern (Let $x $v $body))
    (template $body)

    (guard (not (free-in $x $body)))

    (soundness "let x = v in body = body when x not in FV(body)")
    (termination-weight -1))

(simplification-rule
    (id let-inline-value)
    (category inlining)
    (phase LocalSimplification)

    (pattern (Let $x $v $body))
    (template (Subst $body $x $v))

    (guard (is-value $v))

    (soundness "let x = v in body = body[v/x] for values")
    (termination-weight -1))

(simplification-rule
    (id let-inline-small)
    (category inlining)
    (phase AnalysisDriven)

    (pattern (Let $x $e $body))
    (template (Subst $body $x $e))

    (guard (and (is-small $e) (use-count $x $body <= 3)))

    (soundness "let x = small in body = body[small/x]")
    (termination-weight -1))

(simplification-rule
    (id let-inline-once)
    (category inlining)
    (phase AnalysisDriven)

    (pattern (Let $x $e $body))
    (template (Subst $body $x $e))

    (guard (= (use-count $x $body) 1))

    (soundness "let x = e in body = body[e/x] when used once")
    (termination-weight -1))

; =============================================================================
; Let Binding Flattening
; =============================================================================

(simplification-rule
    (id let-let-flatten)
    (category inlining)
    (phase LocalSimplification)

    (pattern (Let $x $v1 (Let $y $v2 $body)))
    (template (Let $y (Subst $v2 $x $v1) (Subst $body $x $v1)))

    (guard (and (is-value $v1) (not (free-in $x $body))))

    (soundness "Flatten nested let when outer binding is value")
    (termination-weight 0))

(simplification-rule
    (id let-swap)
    (category inlining)
    (phase LocalSimplification)

    (pattern (Let $x $v1 (Let $y $v2 $body)))
    (template (Let $y $v2 (Let $x $v1 $body)))

    (guard (and (not (free-in $x $v2)) (order $x > order $y)))

    (soundness "Canonical ordering of independent let bindings")
    (termination-weight 0))

; =============================================================================
; Function Inlining
; =============================================================================

(simplification-rule
    (id inline-trivial)
    (category inlining)
    (phase AnalysisDriven)

    (pattern (Call $f $args))
    (template (Subst $body $params $args))

    (guard (and (is-trivial-function $f)
                (lookup-body $f $params $body)))

    (soundness "Inline trivial functions (single expression body)")
    (termination-weight -1))

(simplification-rule
    (id inline-once-used)
    (category inlining)
    (phase AnalysisDriven)

    (pattern (Call $f $args))
    (template (Subst $body $params $args))

    (guard (and (= (call-count $f) 1)
                (lookup-body $f $params $body)))

    (soundness "Inline functions called exactly once")
    (termination-weight -1))

; =============================================================================
; Closure Simplification
; =============================================================================

(simplification-rule
    (id closure-no-capture)
    (category inlining)
    (phase AnalysisDriven)

    (pattern (Closure $env (Lambda $x $body)))
    (template (Lambda $x $body))

    (guard (empty? $env))

    (soundness "Closure with empty environment = plain lambda")
    (termination-weight -1))

(simplification-rule
    (id closure-inline-env)
    (category inlining)
    (phase AnalysisDriven)

    (pattern (App (Closure $env (Lambda $x $body)) $arg))
    (template (Subst (Subst $body $env-vars $env-vals) $x $arg))

    (guard (and (is-small-env $env)
                (split-env $env $env-vars $env-vals)))

    (soundness "Apply closure by substituting environment")
    (termination-weight -1))

; =============================================================================
; Partial Application
; =============================================================================

(simplification-rule
    (id partial-app-full)
    (category beta-reduction)
    (phase LocalSimplification)

    (pattern (App (PartialApp $f $partial-args) $remaining-args))
    (template (Call $f (concat $partial-args $remaining-args)))

    (guard (= (+ (len $partial-args) (len $remaining-args))
              (arity $f)))

    (soundness "Complete partial application")
    (termination-weight -1))

; =============================================================================
; Recursive Binding Simplification
; =============================================================================

(simplification-rule
    (id letrec-to-let)
    (category inlining)
    (phase AnalysisDriven)

    (pattern (LetRec $x $v $body))
    (template (Let $x $v $body))

    (guard (not (free-in $x $v)))

    (soundness "letrec x = v in body = let x = v in body when x not in FV(v)")
    (termination-weight 0))

(simplification-rule
    (id letrec-unused)
    (category inlining)
    (phase AnalysisDriven)

    (pattern (LetRec $x $v $body))
    (template $body)

    (guard (not (free-in $x $body)))

    (soundness "letrec x = v in body = body when x not used")
    (termination-weight -1))
