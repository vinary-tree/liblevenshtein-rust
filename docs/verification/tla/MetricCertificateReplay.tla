--------------------- MODULE MetricCertificateReplay ---------------------
EXTENDS Sequences, FiniteSets

\* Finite abstraction of a future CBC certificate acceptance boundary.
\* Strings stand for a two-rule expression fragment, not Rust terms or
\* arithmetic. Universal rule soundness belongs to the Rocq proof kernel.
\* The witness marker checks presence only, not witness correctness.
CONSTANT BypassScopeCheck

Requested == [domain |-> "d", query |-> "q", gap |-> "g",
              stiffness |-> "s", arithmetic |-> "nat",
              revision |-> "r1", cutoff |-> "c1",
              observation |-> "score"]
ChangedDomain == [Requested EXCEPT !.domain = "other"]
ChangedQuery == [Requested EXCEPT !.query = "other"]
ChangedGap == [Requested EXCEPT !.gap = "other"]
ChangedStiffness == [Requested EXCEPT !.stiffness = "other"]
ChangedRevision == [Requested EXCEPT !.revision = "r2"]
ChangedCutoff == [Requested EXCEPT !.cutoff = "c2"]
ChangedArithmetic == [Requested EXCEPT !.arithmetic = "rounded"]
WitnessRequested == [Requested EXCEPT !.observation = "witness"]

OuterStep == [rule |-> "add_zero_outer",
              source |-> "x_plus_zero_plus_zero",
              target |-> "x_plus_zero", premise |-> TRUE]
InnerStep == [rule |-> "add_zero_right", source |-> "x_plus_zero",
              target |-> "x", premise |-> TRUE]
WrongSourceStep == [InnerStep EXCEPT !.source = "another_term"]
FabricatedStep == [OuterStep EXCEPT !.target = "zero"]
MissingPremiseStep == [InnerStep EXCEPT !.premise = FALSE]
UnknownRuleStep == [InnerStep EXCEPT !.rule = "unsupported"]

RuleTarget(step) ==
    IF step.rule = "add_zero_outer"
       /\ step.source = "x_plus_zero_plus_zero"
    THEN "x_plus_zero"
    ELSE IF step.rule = "add_zero_right"
            /\ step.source = "x_plus_zero"
         THEN "x"
         ELSE "invalid"

ReplayFailure(reason) == [reason |-> reason, term |-> "invalid"]

RECURSIVE Replay(_,_)
Replay(source, chain) ==
    IF Len(chain) = 0 THEN [reason |-> "accepted", term |-> source]
    ELSE LET step == Head(chain)
         IN IF source # step.source
            THEN ReplayFailure("step_source_mismatch")
            ELSE IF step.rule \notin {"add_zero_outer", "add_zero_right"}
                 THEN ReplayFailure("unsupported_rule")
                 ELSE IF ~step.premise
                      THEN ReplayFailure("missing_premise")
                      ELSE IF RuleTarget(step) = "invalid"
                           THEN ReplayFailure("rule_pattern_mismatch")
                           ELSE IF RuleTarget(step) # step.target
                                THEN ReplayFailure("rule_target_mismatch")
                                ELSE Replay(step.target, Tail(chain))

BaseExact == [kind |-> "exact", scope |-> Requested,
              source |-> "x_plus_zero_plus_zero", target |-> "x",
              steps |-> <<OuterStep, InnerStep>>, witness |-> FALSE,
              lowerRule |-> "none"]
BaseLower == [kind |-> "lower", scope |-> Requested,
              source |-> "x_plus_zero_plus_zero", target |-> "zero",
              steps |-> <<>>, witness |-> FALSE,
              lowerRule |-> "zero_floor"]

\* Expected values name the intended rejection reason, not just a Boolean.
GoodExact == [certificate |-> BaseExact, requested |-> Requested,
              expected |-> "accepted"]
GoodLower == [certificate |-> BaseLower, requested |-> Requested,
              expected |-> "wrong_kind"]
StaleRevision == [GoodExact EXCEPT !.certificate.scope = ChangedRevision,
                                  !.expected = "scope_mismatch"]
WrongArithmetic == [GoodExact EXCEPT
                       !.requested = ChangedArithmetic,
                       !.certificate.scope = ChangedArithmetic,
                       !.expected = "unsupported_arithmetic"]
WrongRequestArithmetic == [GoodExact EXCEPT !.requested = ChangedArithmetic,
                                           !.expected = "unsupported_arithmetic"]
ValidWitness == [GoodExact EXCEPT !.requested = WitnessRequested,
                                 !.certificate.scope = WitnessRequested,
                                 !.certificate.witness = TRUE]
MissingWitness == [ValidWitness EXCEPT !.certificate.witness = FALSE,
                                       !.expected = "missing_witness"]
WrongSource == [GoodExact EXCEPT
                  !.certificate.steps = <<OuterStep, WrongSourceStep>>,
                  !.expected = "step_source_mismatch"]
WrongTarget == [GoodExact EXCEPT
                  !.certificate.steps = <<FabricatedStep, InnerStep>>,
                  !.expected = "rule_target_mismatch"]
MissingPremise == [GoodExact EXCEPT
                    !.certificate.steps = <<OuterStep, MissingPremiseStep>>,
                    !.expected = "missing_premise"]
UnknownRule == [GoodExact EXCEPT
                 !.certificate.steps = <<OuterStep, UnknownRuleStep>>,
                 !.expected = "unsupported_rule"]
TruncatedChain == [GoodExact EXCEPT
                     !.certificate.steps = <<OuterStep>>,
                     !.expected = "final_target_mismatch"]
ChangedClaim == [GoodExact EXCEPT !.certificate.target = "zero",
                                  !.expected = "final_target_mismatch"]
ChangedDomainCase == [GoodExact EXCEPT
                        !.certificate.scope = ChangedDomain,
                        !.expected = "scope_mismatch"]
ChangedQueryCase == [GoodExact EXCEPT
                       !.certificate.scope = ChangedQuery,
                       !.expected = "scope_mismatch"]
ChangedGapCase == [GoodExact EXCEPT !.certificate.scope = ChangedGap,
                                     !.expected = "scope_mismatch"]
ChangedStiffnessCase == [GoodExact EXCEPT
                           !.certificate.scope = ChangedStiffness,
                           !.expected = "scope_mismatch"]
ChangedCutoffCase == [GoodExact EXCEPT !.certificate.scope = ChangedCutoff,
                                        !.expected = "scope_mismatch"]
FilterPretendingExact == [GoodExact EXCEPT
                            !.certificate.steps = <<>>,
                            !.certificate.target = "zero",
                            !.expected = "final_target_mismatch"]

Cases == {GoodExact, GoodLower, StaleRevision, WrongArithmetic,
          WrongRequestArithmetic, MissingWitness, ValidWitness,
          WrongSource, WrongTarget, MissingPremise, UnknownRule,
          TruncatedChain, ChangedClaim, ChangedDomainCase,
          ChangedQueryCase, ChangedGapCase, ChangedStiffnessCase,
          ChangedCutoffCase, FilterPretendingExact}

ScopeAccepted(requested, certificate) ==
    BypassScopeCheck \/ requested = certificate.scope

CheckExact(scenario) ==
    LET cert == scenario.certificate
        requested == scenario.requested
    IN IF cert.kind # "exact" THEN "wrong_kind"
       ELSE IF requested.arithmetic # "nat"
               \/ cert.scope.arithmetic # "nat"
            THEN "unsupported_arithmetic"
            ELSE IF requested.observation = "witness" /\ ~cert.witness
                 THEN "missing_witness"
                 ELSE IF requested.observation \notin {"score", "witness"}
                      THEN "unsupported_observation"
                      ELSE IF ~ScopeAccepted(requested, cert)
                           THEN "scope_mismatch"
                           ELSE LET result == Replay(cert.source, cert.steps)
                                IN IF result.reason # "accepted"
                                   THEN result.reason
                                   ELSE IF result.term # cert.target
                                        THEN "final_target_mismatch"
                                        ELSE "accepted"

CheckLower(certificate) ==
    IF certificate.kind # "lower" THEN "wrong_kind"
    ELSE IF certificate.scope.arithmetic # "nat"
         THEN "unsupported_arithmetic"
         ELSE IF certificate.scope.observation # "score"
              THEN "unsupported_observation"
              ELSE IF ~ScopeAccepted(Requested, certificate)
                   THEN "scope_mismatch"
                   ELSE IF certificate.lowerRule # "zero_floor"
                        THEN "unsupported_lower_rule"
                        ELSE IF certificate.source # "x_plus_zero_plus_zero"
                             \/ certificate.target # "zero"
                             THEN "lower_direction_mismatch"
                             ELSE "accepted"

ReversedLower == [BaseLower EXCEPT !.source = "zero",
                                   !.target = "x_plus_zero_plus_zero"]
UnsupportedLower == [BaseLower EXCEPT !.lowerRule = "fabricated"]
RoundedLower == [BaseLower EXCEPT !.scope = ChangedArithmetic]
WitnessLower == [BaseLower EXCEPT !.scope = WitnessRequested]

VARIABLE selected
Init == selected \in Cases
Next == UNCHANGED selected
Spec == Init /\ [][Next]_selected

ReasonClassification == CheckExact(selected) = selected.expected
ExactAndLowerAreDistinct == CheckExact(GoodLower) = "wrong_kind"
LowerRuleDirectionIsChecked ==
    CheckLower(BaseLower) = "accepted"
    /\ CheckLower(ReversedLower) = "lower_direction_mismatch"
    /\ CheckLower(UnsupportedLower) = "unsupported_lower_rule"
LowerProfileIsChecked ==
    CheckLower(RoundedLower) = "unsupported_arithmetic"
    /\ CheckLower(WitnessLower) = "unsupported_observation"
ScopeMutationDetected == CheckExact(StaleRevision) = "scope_mismatch"
PositiveControlsAccepted ==
    CheckExact(GoodExact) = "accepted"
    /\ CheckExact(ValidWitness) = "accepted"
    /\ CheckLower(BaseLower) = "accepted"
FilterCannotClaimExact == CheckExact(FilterPretendingExact) # "accepted"

=============================================================================
