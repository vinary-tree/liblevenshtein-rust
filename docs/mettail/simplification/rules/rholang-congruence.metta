; Rholang Structural Congruence Rules
; These rules implement the structural congruence laws of the Rholang process calculus.
;
; Status: Design Documentation
; Last Updated: 2025-12-06

; =============================================================================
; Nil Identity Laws
; P | 0 = P and 0 | P = P
; =============================================================================

(simplification-rule
    (id nil-identity-right)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Par $P (Nil)))
    (template $P)

    (soundness "P | 0 = P by nil identity law")
    (termination-weight -1))

(simplification-rule
    (id nil-identity-left)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Par (Nil) $P))
    (template $P)

    (soundness "0 | P = P by nil identity law")
    (termination-weight -1))

; =============================================================================
; Commutativity
; P | Q = Q | P
; =============================================================================

(simplification-rule
    (id par-commute)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Par $P $Q))
    (template (Par $Q $P))

    (guard (> (canonical-order $P) (canonical-order $Q)))

    (soundness "P | Q = Q | P by commutativity (toward canonical order)")
    (termination-weight 0))

; =============================================================================
; Associativity
; (P | Q) | R = P | (Q | R)
; =============================================================================

(simplification-rule
    (id par-assoc-right)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Par (Par $P $Q) $R))
    (template (Par $P (Par $Q $R)))

    (soundness "(P | Q) | R = P | (Q | R) by associativity (right-normalize)")
    (termination-weight 0))

; =============================================================================
; Scope Extrusion Laws
; new x in (P | Q) = (new x in P) | Q  when x not free in Q
; new x in (P | Q) = P | (new x in Q)  when x not free in P
; =============================================================================

(simplification-rule
    (id scope-extrude-right)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (New $x (Par $P $Q)))
    (template (Par (New $x $P) $Q))

    (guard (not (free-in $x $Q)))

    (soundness "new x.(P|Q) = (new x.P)|Q when x not in FV(Q)")
    (termination-weight 0))

(simplification-rule
    (id scope-extrude-left)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (New $x (Par $P $Q)))
    (template (Par $P (New $x $Q)))

    (guard (not (free-in $x $P)))

    (soundness "new x.(P|Q) = P|(new x.Q) when x not in FV(P)")
    (termination-weight 0))

; =============================================================================
; Dead Scope Elimination
; new x in P = P  when x not free in P
; =============================================================================

(simplification-rule
    (id dead-scope-elim)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (New $x $P))
    (template $P)

    (guard (not (free-in $x $P)))

    (soundness "new x.P = P when x not in FV(P)")
    (termination-weight -1))

; =============================================================================
; Scope Fusion
; new x in (new x in P) = new x in P
; =============================================================================

(simplification-rule
    (id scope-fusion)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (New $x (New $x $P)))
    (template (New $x $P))

    (soundness "new x.(new x.P) = new x.P (shadowed binding)")
    (termination-weight -1))

; =============================================================================
; Replication Laws
; *0 = 0
; =============================================================================

(simplification-rule
    (id replicate-nil)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Replicate (Nil)))
    (template (Nil))

    (soundness "*0 = 0 (replicating nil is nil)")
    (termination-weight -1))

; =============================================================================
; Send/Receive Simplification
; =============================================================================

(simplification-rule
    (id send-nil-cont)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Send $chan $data (Nil)))
    (template (Send $chan $data))

    (soundness "x!(data).0 = x!(data) (nil continuation)")
    (termination-weight -1))

(simplification-rule
    (id for-nil-body)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Receive $pat $chan (Nil)))
    (template (Nil))

    (guard (not (persisted-receive?)))

    (soundness "for(@pat <- chan){0} = 0 (consuming but doing nothing)")
    (termination-weight -1))

; =============================================================================
; Match Simplification
; =============================================================================

(simplification-rule
    (id match-single-wildcard)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Match $scrutinee ((Wildcard $body))))
    (template $body)

    (soundness "match x { _ => body } = body")
    (termination-weight -1))

(simplification-rule
    (id match-known-variant)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Match (Variant $tag $data) $cases))
    (template (lookup-case $tag $data $cases))

    (guard (has-matching-case $tag $cases))

    (soundness "match Tag(data) with cases = matching case body")
    (termination-weight -1))

; =============================================================================
; Contract Simplification
; =============================================================================

(simplification-rule
    (id contract-nil-body)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Contract $name $params (Nil)))
    (template (Nil))

    (soundness "contract Name(params) = { 0 } = 0 (no-op contract)")
    (termination-weight -1))

; =============================================================================
; Channel Equivalence
; =============================================================================

(simplification-rule
    (id quote-unquote)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Quote (Unquote $chan)))
    (template $chan)

    (soundness "@(*chan) = chan")
    (termination-weight -1))

(simplification-rule
    (id unquote-quote)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Unquote (Quote $proc)))
    (template $proc)

    (soundness "*(@proc) = proc")
    (termination-weight -1))

; =============================================================================
; Canonical Ordering Function
; (Used by commutativity rule to ensure termination)
; =============================================================================

(: canonical-order (-> Process Int))

; Nil has highest order (will be eliminated)
(= (canonical-order (Nil)) 9999999)

; Send ordered by channel hash
(= (canonical-order (Send $chan $_ $_))
   (hash $chan))

; Receive ordered by channel hash
(= (canonical-order (Receive $_ $chan $_))
   (hash $chan))

; New ordered by variable name
(= (canonical-order (New $x $_))
   (hash $x))

; Par ordered by minimum of children
(= (canonical-order (Par $P $Q))
   (min (canonical-order $P) (canonical-order $Q)))

; Default: content hash
(= (canonical-order $P)
   (content-hash $P))

; =============================================================================
; Free Variable Checking
; =============================================================================

(: free-in (-> Var Process Bool))

(= (free-in $x (Nil)) False)

(= (free-in $x (Var $y))
   (= $x $y))

(= (free-in $x (Par $P $Q))
   (or (free-in $x $P) (free-in $x $Q)))

(= (free-in $x (New $y $P))
   (and (not (= $x $y)) (free-in $x $P)))

(= (free-in $x (Send $chan $data $cont))
   (or (free-in $x $chan)
       (free-in $x $data)
       (free-in $x $cont)))

(= (free-in $x (Receive $pat $chan $body))
   (or (free-in $x $chan)
       (and (not (binds $pat $x))
            (free-in $x $body))))
