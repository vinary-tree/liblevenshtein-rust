; Type-Aware Simplification Rules
; These rules require type information and eliminate redundant type operations.
;
; Status: Design Documentation
; Last Updated: 2025-12-06

; =============================================================================
; Redundant Cast Elimination
; =============================================================================

(simplification-rule
    (id redundant-cast)
    (category type-aware)
    (phase TypeDirected)

    (pattern (Cast $T $x))
    (template $x)

    (guard (has-type $x $T))

    (soundness "(T)x = x when x : T")
    (termination-weight -1))

(simplification-rule
    (id double-cast)
    (category type-aware)
    (phase TypeDirected)

    (pattern (Cast $T (Cast $U $x)))
    (template (Cast $T $x))

    (soundness "(T)(U)x = (T)x (intermediate cast unnecessary)")
    (termination-weight -1))

(simplification-rule
    (id upcast-downcast)
    (category type-aware)
    (phase TypeDirected)

    (pattern (Cast $T (Cast $U $x)))
    (template $x)

    (guard (and (has-type $x $T) (subtype $T $U)))

    (soundness "(T)(U)x = x when x : T and T <: U")
    (termination-weight -2))

; =============================================================================
; Identity Conversions
; =============================================================================

(simplification-rule
    (id identity-into)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall $x into () $T))
    (template $x)

    (guard (has-type $x $T))

    (soundness "x.into::<T>() = x when x : T")
    (termination-weight -1))

(simplification-rule
    (id identity-from)
    (category type-aware)
    (phase TypeDirected)

    (pattern (Call (TypePath $T from) ($x)))
    (template $x)

    (guard (has-type $x $T))

    (soundness "T::from(x) = x when x : T")
    (termination-weight -1))

; =============================================================================
; Option/Result Simplification
; =============================================================================

(simplification-rule
    (id some-unwrap)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall (Some $x) unwrap ()))
    (template $x)

    (soundness "Some(x).unwrap() = x")
    (termination-weight -1))

(simplification-rule
    (id ok-unwrap)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall (Ok $x) unwrap ()))
    (template $x)

    (soundness "Ok(x).unwrap() = x")
    (termination-weight -1))

(simplification-rule
    (id some-map)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall (Some $x) map ($f)))
    (template (Some (Call $f ($x))))

    (soundness "Some(x).map(f) = Some(f(x))")
    (termination-weight 0))

(simplification-rule
    (id none-map)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall None map ($f)))
    (template None)

    (soundness "None.map(f) = None")
    (termination-weight -1))

(simplification-rule
    (id some-and-then)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall (Some $x) and_then ($f)))
    (template (Call $f ($x)))

    (soundness "Some(x).and_then(f) = f(x)")
    (termination-weight -1))

(simplification-rule
    (id none-and-then)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall None and_then ($f)))
    (template None)

    (soundness "None.and_then(f) = None")
    (termination-weight -1))

(simplification-rule
    (id some-or)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall (Some $x) or ($other)))
    (template (Some $x))

    (soundness "Some(x).or(other) = Some(x)")
    (termination-weight -1))

(simplification-rule
    (id none-or)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall None or ($other)))
    (template $other)

    (soundness "None.or(other) = other")
    (termination-weight -1))

; =============================================================================
; Collection Simplification
; =============================================================================

(simplification-rule
    (id empty-iter-collect)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall (MethodCall (Vec) iter ()) collect ()))
    (template (Vec))

    (soundness "vec![].iter().collect() = vec![]")
    (termination-weight -2))

(simplification-rule
    (id filter-true)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall $iter filter ((Lambda $x (Bool true)))))
    (template $iter)

    (soundness "iter.filter(|_| true) = iter")
    (termination-weight -1))

(simplification-rule
    (id filter-false)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall $iter filter ((Lambda $x (Bool false)))))
    (template (Call empty ()))

    (soundness "iter.filter(|_| false) = empty()")
    (termination-weight -1))

(simplification-rule
    (id map-identity)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall $iter map ((Lambda $x $x))))
    (template $iter)

    (soundness "iter.map(|x| x) = iter")
    (termination-weight -1))

; =============================================================================
; Reference/Dereference Cancellation
; =============================================================================

(simplification-rule
    (id ref-deref)
    (category type-aware)
    (phase TypeDirected)

    (pattern (Deref (Ref $x)))
    (template $x)

    (guard (is-copy-type (type-of $x)))

    (soundness "*&x = x for Copy types")
    (termination-weight -1))

(simplification-rule
    (id deref-ref)
    (category type-aware)
    (phase TypeDirected)

    (pattern (Ref (Deref $x)))
    (template $x)

    (guard (is-ref-type (type-of $x)))

    (soundness "&*x = x for reference types")
    (termination-weight -1))

; =============================================================================
; Clone Elimination
; =============================================================================

(simplification-rule
    (id clone-copy)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall $x clone ()))
    (template $x)

    (guard (is-copy-type (type-of $x)))

    (soundness "x.clone() = x for Copy types")
    (termination-weight -1))

(simplification-rule
    (id double-clone)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall (MethodCall $x clone ()) clone ()))
    (template (MethodCall $x clone ()))

    (soundness "x.clone().clone() = x.clone()")
    (termination-weight -1))

; =============================================================================
; String Conversions
; =============================================================================

(simplification-rule
    (id string-to-string)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall $x to_string ()))
    (template $x)

    (guard (has-type $x String))

    (soundness "s.to_string() = s for String")
    (termination-weight -1))

(simplification-rule
    (id str-to-owned-to-string)
    (category type-aware)
    (phase TypeDirected)

    (pattern (MethodCall (MethodCall $x to_owned ()) to_string ()))
    (template (MethodCall $x to_string ()))

    (soundness "s.to_owned().to_string() = s.to_string()")
    (termination-weight -1))
