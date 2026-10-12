-------------------------- MODULE RakuAutomataBoundary --------------------------
\* Finite abstraction of the Raku generalized/universal NativeCall boundary.
\* The concrete Rust distance algorithms are checked separately.  Here a
\* small set of representative operation/policy traces makes directional
\* semantics, prefix observations, limits, and independent ownership explicit.
EXTENDS Naturals, Sequences, TLC

Scenarios == {
  "g_any", "g_equal", "g_equal_match", "g_listed",
  "g_listed_reverse", "g_bridge", "u_plain", "u_plain_match",
  "u_directed", "u_reverse"
}
Variants == {"standard", "transposition", "merge_split"}
Domains == {"unicode", "byte", "u64"}
Generalized == {"g_any", "g_equal", "g_equal_match", "g_listed",
               "g_listed_reverse", "g_bridge"}
Universal == Scenarios \ Generalized

\* g_bridge is a 2:2 EQUAL operation: its first target row is empty, but
\* the retained source row reconnects when the second target unit arrives.
Target(s) == CASE s \in {"g_bridge", "g_equal_match", "u_plain_match"}
                -> <<"a", "b">>
              [] OTHER -> <<"f", "x">>

\* One-unit representatives distinguish ANY, EQUAL, LISTED and the
\* directional universal substitution policy.  The three universal variants
\* agree on these one-unit cases; the Raku property suite covers longer cases.
Accept(s, n) ==
  CASE s = "g_bridge" -> n = 2
    [] s = "g_any" -> n = 1
    [] s = "g_equal" -> FALSE
    [] s = "g_equal_match" -> n = 1
    [] s = "g_listed" -> n = 1
    [] s = "g_listed_reverse" -> FALSE
    [] s = "u_plain" -> FALSE
    [] s = "u_plain_match" -> n = 1
    [] s = "u_directed" -> n = 1
    [] s = "u_reverse" -> FALSE

RowNonempty(s, n) == IF s = "g_bridge" THEN n # 1 ELSE n <= 1
FrontierAlive(s, n) == IF s \in Universal THEN n = 0 \/ Accept(s, n)
                       ELSE TRUE

VARIABLES scenario, variant, domain, targetLimit, configuration, online,
          ownsConfiguration, prefix, accepted, rowNonempty, alive,
          configurationFrees, onlineFrees, lastAction, prefixBeforeAction
vars == <<scenario, variant, domain, targetLimit, configuration, online,
          ownsConfiguration, prefix, accepted, rowNonempty, alive,
          configurationFrees, onlineFrees, lastAction, prefixBeforeAction>>

Init ==
  /\ scenario \in Scenarios
  /\ variant \in Variants
  /\ domain \in Domains
  /\ scenario \in Generalized => domain = "unicode"
  /\ targetLimit \in 0..2
  /\ configuration = "open"
  /\ online = "absent"
  /\ ownsConfiguration = FALSE
  /\ prefix = 0
  /\ accepted = Accept(scenario, 0)
  /\ rowNonempty = RowNonempty(scenario, 0)
  /\ alive = FrontierAlive(scenario, 0)
  /\ configurationFrees = 0
  /\ onlineFrees = 0
  /\ lastAction = "init"
  /\ prefixBeforeAction = 0

OpenOnline ==
  /\ configuration = "open"
  /\ online = "absent"
  /\ online' = "open"
  /\ ownsConfiguration' = TRUE
  /\ lastAction' = "open"
  /\ prefixBeforeAction' = prefix
  /\ UNCHANGED <<scenario, variant, domain, targetLimit, configuration,
                 prefix, accepted, rowNonempty, alive,
                 configurationFrees, onlineFrees>>

Advance ==
  /\ online = "open"
  /\ prefix < Len(Target(scenario))
  /\ prefix < targetLimit
  /\ prefix' = prefix + 1
  /\ accepted' = Accept(scenario, prefix')
  /\ rowNonempty' = RowNonempty(scenario, prefix')
  /\ alive' = FrontierAlive(scenario, prefix')
  /\ lastAction' = "advance"
  /\ prefixBeforeAction' = prefix
  /\ UNCHANGED <<scenario, variant, domain, targetLimit, configuration,
                 online, ownsConfiguration, configurationFrees, onlineFrees>>

\* A hard target limit or malformed unit does not commit the attempted step.
RejectLimit ==
  /\ online = "open"
  /\ prefix < Len(Target(scenario))
  /\ prefix >= targetLimit
  /\ lastAction' = "reject_limit"
  /\ prefixBeforeAction' = prefix
  /\ UNCHANGED <<scenario, variant, domain, targetLimit, configuration,
                 online, ownsConfiguration, prefix, accepted, rowNonempty,
                 alive, configurationFrees, onlineFrees>>

RejectInvalidUnit ==
  /\ online = "open"
  /\ lastAction' = "reject_unit"
  /\ prefixBeforeAction' = prefix
  /\ UNCHANGED <<scenario, variant, domain, targetLimit, configuration,
                 online, ownsConfiguration, prefix, accepted, rowNonempty,
                 alive, configurationFrees, onlineFrees>>

Observe ==
  /\ online = "open"
  /\ lastAction' = "observe"
  /\ prefixBeforeAction' = prefix
  /\ UNCHANGED <<scenario, variant, domain, targetLimit, configuration,
                 online, ownsConfiguration, prefix, accepted, rowNonempty,
                 alive, configurationFrees, onlineFrees>>

CloseConfiguration ==
  /\ configuration = "open"
  /\ configuration' = "closed"
  /\ configurationFrees' = configurationFrees + 1
  /\ lastAction' = "close_configuration"
  /\ prefixBeforeAction' = prefix
  /\ UNCHANGED <<scenario, variant, domain, targetLimit, online,
                 ownsConfiguration, prefix, accepted, rowNonempty, alive,
                 onlineFrees>>

CloseOnline ==
  /\ online = "open"
  /\ online' = "closed"
  /\ ownsConfiguration' = FALSE
  /\ onlineFrees' = onlineFrees + 1
  /\ lastAction' = "close_online"
  /\ prefixBeforeAction' = prefix
  /\ UNCHANGED <<scenario, variant, domain, targetLimit, configuration,
                 prefix, accepted, rowNonempty, alive, configurationFrees>>

Next == OpenOnline \/ Advance \/ RejectLimit \/ RejectInvalidUnit \/ Observe
        \/ CloseConfiguration \/ CloseOnline
Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ scenario \in Scenarios
  /\ variant \in Variants
  /\ domain \in Domains
  /\ targetLimit \in 0..2
  /\ configuration \in {"open", "closed"}
  /\ online \in {"absent", "open", "closed"}
  /\ ownsConfiguration \in BOOLEAN
  /\ prefix \in 0..2
  /\ accepted \in BOOLEAN
  /\ rowNonempty \in BOOLEAN
  /\ alive \in BOOLEAN
  /\ configurationFrees \in 0..1
  /\ onlineFrees \in 0..1
  /\ prefixBeforeAction \in 0..2

RakuDomainPolicy == scenario \in Generalized => domain = "unicode"
PrefixBounded == prefix <= Len(Target(scenario)) /\ prefix <= targetLimit
PrefixObservation == accepted = Accept(scenario, prefix)
RowObservation == rowNonempty = RowNonempty(scenario, prefix)
FrontierObservation == alive = FrontierAlive(scenario, prefix)
RejectedStepIsAtomic ==
  lastAction \in {"reject_limit", "reject_unit"} => prefix = prefixBeforeAction
OwnedOnlineOutlivesConfiguration == online = "open" => ownsConfiguration
ExactlyOneNativeFree == configurationFrees <= 1 /\ onlineFrees <= 1
NoUseAfterOnlineClose ==
  lastAction \in {"advance", "observe", "reject_limit", "reject_unit"}
    => online = "open"
DirectionalPolicy ==
  /\ (scenario \in {"g_listed_reverse", "u_reverse"} /\ prefix = 1
        => ~accepted)
  /\ (scenario \in {"g_listed", "u_directed"} /\ prefix = 1
        => accepted)
GeneralizedReconnects ==
  scenario = "g_bridge" =>
    ((prefix = 1 => ~rowNonempty /\ ~accepted)
      /\ (prefix = 2 => rowNonempty /\ accepted))
UniversalDeadCannotAccept == scenario \in Universal => (~alive => ~accepted)

=============================================================================
