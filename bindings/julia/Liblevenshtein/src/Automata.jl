const DEFAULT_AUTOMATON_MAX_SOURCE_UNITS = 1_000_000
const DEFAULT_AUTOMATON_MAX_TARGET_UNITS = 1_000_000
const DEFAULT_AUTOMATON_MAX_RETAINED_CELLS = 1_000_000
const DEFAULT_AUTOMATON_MAX_STEP_WORK_UNITS = 100_000_000

function checked_csize(value::Integer, name::AbstractString)::Csize_t
    value >= 0 || throw(ArgumentError("$name must be nonnegative"))
    value <= typemax(Csize_t) || throw(OverflowError("$name exceeds Csize_t"))
    Csize_t(value)
end

function checked_u8(value::Integer, name::AbstractString)::UInt8
    value >= 0 || throw(ArgumentError("$name must be nonnegative"))
    value <= typemax(UInt8) || throw(OverflowError("$name exceeds UInt8"))
    UInt8(value)
end

"""Explicit hard limits for standalone complete and online automata evaluation."""
struct AutomatonLimits
    max_source_units::Csize_t
    max_target_units::Csize_t
    max_retained_cells::Csize_t
    max_step_work_units::Csize_t
end

function AutomatonLimits(;
    max_source_units::Integer=DEFAULT_AUTOMATON_MAX_SOURCE_UNITS,
    max_target_units::Integer=DEFAULT_AUTOMATON_MAX_TARGET_UNITS,
    max_retained_cells::Integer=DEFAULT_AUTOMATON_MAX_RETAINED_CELLS,
    max_step_work_units::Integer=DEFAULT_AUTOMATON_MAX_STEP_WORK_UNITS)
    AutomatonLimits(
        checked_csize(max_source_units, "max_source_units"),
        checked_csize(max_target_units, "max_target_units"),
        checked_csize(max_retained_cells, "max_retained_cells"),
        checked_csize(max_step_work_units, "max_step_work_units"),
    )
end

struct RawAutomatonLimits
    max_source_units::Csize_t
    max_target_units::Csize_t
    max_retained_cells::Csize_t
    max_step_work_units::Csize_t
end

RawAutomatonLimits(value::AutomatonLimits) = RawAutomatonLimits(
    value.max_source_units,
    value.max_target_units,
    value.max_retained_cells,
    value.max_step_work_units,
)

function limit_storage(limits::Union{Nothing,AutomatonLimits})
    if limits === nothing
        storage = Ref(RawAutomatonLimits(0, 0, 0, 0))
        return storage, Ptr{RawAutomatonLimits}(C_NULL)
    end
    storage = Ref(RawAutomatonLimits(limits))
    storage, Base.unsafe_convert(Ptr{RawAutomatonLimits}, storage)
end

"""One directional source-to-target restriction for a listed generalized edit."""
struct GeneralizedRestriction
    source::String
    target::String
end

GeneralizedRestriction(source::AbstractString, target::AbstractString) =
    GeneralizedRestriction(String(source), String(target))
GeneralizedRestriction(value::Pair{<:AbstractString,<:AbstractString}) =
    GeneralizedRestriction(value.first, value.second)

function normalize_restriction(value)::GeneralizedRestriction
    value isa GeneralizedRestriction && return value
    value isa Pair{<:AbstractString,<:AbstractString} &&
        return GeneralizedRestriction(value)
    throw(ArgumentError("a generalized restriction must be GeneralizedRestriction or String => String"))
end

"""An immutable runtime generalized edit-operation description."""
struct GeneralizedOperation
    consume_source::Csize_t
    consume_target::Csize_t
    weight::Float64
    name::String
    applicability::OperationApplicability
    restrictions::Tuple{Vararg{GeneralizedRestriction}}
end

function GeneralizedOperation(consume_source::Integer, consume_target::Integer,
    weight::Real, name::Union{AbstractString,Symbol};
    applicability::Union{Nothing,OperationApplicability}=nothing,
    restrictions=())
    owned_restrictions = Tuple(normalize_restriction(value) for value in restrictions)
    resolved_applicability = applicability === nothing ?
        (isempty(owned_restrictions) ? APPLICABILITY_ANY : APPLICABILITY_LISTED) :
        applicability
    GeneralizedOperation(
        checked_csize(consume_source, "consume_source"),
        checked_csize(consume_target, "consume_target"),
        Float64(weight),
        string(name),
        resolved_applicability,
        owned_restrictions,
    )
end

"""An immutable, typed collection of generalized edit operations."""
struct GeneralizedOperationSet
    operations::Tuple{Vararg{GeneralizedOperation}}
end

GeneralizedOperationSet(operations::AbstractVector{<:GeneralizedOperation}) =
    GeneralizedOperationSet(Tuple(operations))
GeneralizedOperationSet(first::GeneralizedOperation, rest::GeneralizedOperation...) =
    GeneralizedOperationSet((first, rest...))

Base.length(value::GeneralizedOperationSet) = length(value.operations)
Base.eltype(::Type{GeneralizedOperationSet}) = GeneralizedOperation
Base.iterate(value::GeneralizedOperationSet, state...) = iterate(value.operations, state...)

struct RawGeneralizedRestriction
    source_data::Ptr{UInt8}
    source_len::Csize_t
    target_data::Ptr{UInt8}
    target_len::Csize_t
end

struct RawGeneralizedOperation
    consume_source::Csize_t
    consume_target::Csize_t
    weight::Cdouble
    name_data::Ptr{UInt8}
    name_len::Csize_t
    applicability::UInt32
    reserved::UInt32
    restrictions::Ptr{RawGeneralizedRestriction}
    restriction_count::Csize_t
end

struct RawGeneralizedObservation
    consumed_target_len::Csize_t
    active_positions::Csize_t
    scaled_distance::Csize_t
    scale_denominator::UInt32
    current_row_nonempty::UInt8
    accepting::UInt8
    has_distance::UInt8
    reserved::UInt8
end

RawGeneralizedObservation() = RawGeneralizedObservation(0, 0, 0, 0, 0, 0, 0, 0)

"""A precise observation of one generalized target prefix."""
struct GeneralizedObservation
    consumed_target_length::Int
    active_positions::Int
    scaled_distance::Union{Nothing,Int}
    scale_denominator::Int
    distance::Union{Nothing,Rational{Int}}
    current_row_nonempty::Bool
    accepting::Bool
end

function GeneralizedObservation(raw::RawGeneralizedObservation)
    denominator = Int(raw.scale_denominator)
    numerator = raw.has_distance == 0 ? nothing : Int(raw.scaled_distance)
    numerator === nothing || denominator > 0 ||
        throw(ArgumentError("native generalized observation has a zero denominator"))
    exact = numerator === nothing ? nothing : numerator // denominator
    GeneralizedObservation(
        Int(raw.consumed_target_len),
        Int(raw.active_positions),
        numerator,
        denominator,
        exact,
        raw.current_row_nonempty != 0,
        raw.accepting != 0,
    )
end

data_pointer(value::Vector{T}) where {T} =
    isempty(value) ? Ptr{T}(C_NULL) : pointer(value)

function marshal_operations(value::GeneralizedOperationSet)
    count = length(value)
    names = Vector{Vector{UInt8}}(undef, count)
    restriction_sources = Vector{Vector{Vector{UInt8}}}(undef, count)
    restriction_targets = Vector{Vector{Vector{UInt8}}}(undef, count)
    raw_restrictions = Vector{Vector{RawGeneralizedRestriction}}(undef, count)

    for (index, operation) in enumerate(value.operations)
        names[index] = text_bytes(operation.name)
        sources = [text_bytes(restriction.source) for restriction in operation.restrictions]
        targets = [text_bytes(restriction.target) for restriction in operation.restrictions]
        restriction_sources[index] = sources
        restriction_targets[index] = targets
        raw_restrictions[index] = [
            RawGeneralizedRestriction(
                data_pointer(sources[pair_index]),
                length(sources[pair_index]),
                data_pointer(targets[pair_index]),
                length(targets[pair_index]),
            ) for pair_index in eachindex(sources)
        ]
    end

    raw_operations = [
        RawGeneralizedOperation(
            operation.consume_source,
            operation.consume_target,
            operation.weight,
            data_pointer(names[index]),
            length(names[index]),
            UInt32(operation.applicability),
            0,
            data_pointer(raw_restrictions[index]),
            length(raw_restrictions[index]),
        ) for (index, operation) in enumerate(value.operations)
    ]
    raw_operations, names, restriction_sources, restriction_targets, raw_restrictions
end

"""An immutable native generalized-automaton configuration."""
mutable struct GeneralizedAutomaton
    handle::Ptr{Cvoid}
    closed::Bool
end

function GeneralizedAutomaton(maximum_distance::Integer,
    operation_set::GeneralizedOperationSet)
    raw_operations, names, sources, targets, raw_restrictions =
        marshal_operations(operation_set)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve raw_operations names sources targets raw_restrictions begin
        checked(ccall(native(:llev_generalized_automaton_new), Cint,
            (UInt8, Ptr{RawGeneralizedOperation}, Csize_t, Ref{Ptr{Cvoid}}),
            checked_u8(maximum_distance, "maximum_distance"),
            data_pointer(raw_operations), length(raw_operations), output),
            :llev_generalized_automaton_new)
    end
    value = GeneralizedAutomaton(output[], false)
    finalizer(close!, value)
    value
end

GeneralizedAutomaton(maximum_distance::Integer, operations) =
    GeneralizedAutomaton(maximum_distance, GeneralizedOperationSet(operations))

function require_open(automaton::GeneralizedAutomaton)
    automaton.closed && throw(NativeError(Int32(STATUS_CLOSED),
        :generalized_automaton, "generalized automaton is closed"))
    automaton.handle
end

function evaluate(automaton::GeneralizedAutomaton, source::AbstractString,
    target::AbstractString; limits::Union{Nothing,AutomatonLimits}=nothing)
    source_bytes = text_bytes(source)
    target_bytes = text_bytes(target)
    limit_ref, limit_pointer = limit_storage(limits)
    output = Ref(RawGeneralizedObservation())
    GC.@preserve source_bytes target_bytes limit_ref begin
        checked(ccall(native(:llev_generalized_automaton_evaluate_utf8), Cint,
            (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t,
                Ptr{RawAutomatonLimits}, Ref{RawGeneralizedObservation}),
            require_open(automaton), data_pointer(source_bytes), length(source_bytes),
            data_pointer(target_bytes), length(target_bytes), limit_pointer, output),
            :llev_generalized_automaton_evaluate_utf8)
    end
    GeneralizedObservation(output[])
end

"""An exclusive generalized online-prefix state."""
mutable struct GeneralizedOnlineAutomaton
    handle::Ptr{Cvoid}
    closed::Bool
end

function online(automaton::GeneralizedAutomaton, source::AbstractString;
    limits::Union{Nothing,AutomatonLimits}=nothing)
    source_bytes = text_bytes(source)
    limit_ref, limit_pointer = limit_storage(limits)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve source_bytes limit_ref begin
        checked(ccall(native(:llev_generalized_online_new_utf8), Cint,
            (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ptr{RawAutomatonLimits}, Ref{Ptr{Cvoid}}),
            require_open(automaton), data_pointer(source_bytes), length(source_bytes),
            limit_pointer, output), :llev_generalized_online_new_utf8)
    end
    value = GeneralizedOnlineAutomaton(output[], false)
    finalizer(close!, value)
    value
end

function require_open(online::GeneralizedOnlineAutomaton)
    online.closed && throw(NativeError(Int32(STATUS_CLOSED),
        :generalized_online, "generalized online automaton is closed"))
    online.handle
end

function observation(online::GeneralizedOnlineAutomaton)
    output = Ref(RawGeneralizedObservation())
    checked(ccall(native(:llev_generalized_online_observation), Cint,
        (Ptr{Cvoid}, Ref{RawGeneralizedObservation}), require_open(online), output),
        :llev_generalized_online_observation)
    GeneralizedObservation(output[])
end

function advance!(online::GeneralizedOnlineAutomaton, unit::Char)
    output = Ref(RawGeneralizedObservation())
    checked(ccall(native(:llev_generalized_online_advance), Cint,
        (Ptr{Cvoid}, UInt32, Ref{RawGeneralizedObservation}), require_open(online),
        UInt32(unit), output), :llev_generalized_online_advance)
    GeneralizedObservation(output[])
end

function close!(automaton::GeneralizedAutomaton)
    automaton.closed && return nothing
    handle = automaton.handle
    automaton.handle = C_NULL
    automaton.closed = true
    handle == C_NULL || ccall(native(:llev_generalized_automaton_free), Cvoid,
        (Ptr{Cvoid},), handle)
    nothing
end

function close!(online::GeneralizedOnlineAutomaton)
    online.closed && return nothing
    handle = online.handle
    online.handle = C_NULL
    online.closed = true
    handle == C_NULL || ccall(native(:llev_generalized_online_free), Cvoid,
        (Ptr{Cvoid},), handle)
    nothing
end

Base.close(value::Union{GeneralizedAutomaton,GeneralizedOnlineAutomaton}) = close!(value)
Base.isopen(value::Union{GeneralizedAutomaton,GeneralizedOnlineAutomaton}) = !value.closed

abstract type AbstractUniversalPolicy end

"""The allocation-free universal substitution policy."""
struct UnrestrictedPolicy <: AbstractUniversalPolicy end
const UNRESTRICTED_POLICY = UnrestrictedPolicy()

"""One directional zero-cost source-to-target universal equivalence."""
struct UniversalEquivalence{U}
    source::U
    target::U
    function UniversalEquivalence(source::U, target::U) where
        {U<:Union{UInt8,UInt64,Char}}
        new{U}(source, target)
    end
end

UniversalEquivalence(value::Pair{U,U}) where {U<:Union{UInt8,UInt64,Char}} =
    UniversalEquivalence(value.first, value.second)

normalize_equivalence(value::UniversalEquivalence) = value

function normalize_equivalence(value::Pair)
    allowed_unit = Union{UInt8,UInt64,Char}
    value.first isa allowed_unit && value.second isa allowed_unit ||
        throw(ArgumentError(
            "universal-equivalence units must be UInt8, UInt64, or Char"))
    typeof(value.first) === typeof(value.second) ||
        throw(ArgumentError("a universal equivalence must use one unit domain"))
    UniversalEquivalence(value.first, value.second)
end

"""An immutable non-empty directional universal-equivalence policy."""
struct UniversalPolicy{U<:Union{UInt8,UInt64,Char}} <: AbstractUniversalPolicy
    equivalences::Tuple{Vararg{UniversalEquivalence{U}}}

    function UniversalPolicy{U}(
        equivalences::Tuple{Vararg{UniversalEquivalence{U}}}) where
        {U<:Union{UInt8,UInt64,Char}}
        isempty(equivalences) && throw(ArgumentError(
            "a UniversalPolicy must be non-empty; use UNRESTRICTED_POLICY otherwise"))
        new{U}(equivalences)
    end
end

function build_universal_policy(values)
    equivalences = Tuple(normalize_equivalence(value) for value in values)
    isempty(equivalences) && throw(ArgumentError(
        "a UniversalPolicy must be non-empty; use UNRESTRICTED_POLICY otherwise"))
    unit_type = typeof(equivalences[1].source)
    all(value -> typeof(value.source) === unit_type &&
        typeof(value.target) === unit_type, equivalences) ||
        throw(ArgumentError("universal equivalences must use one unit domain"))
    UniversalPolicy{unit_type}(equivalences)
end

UniversalPolicy(values::AbstractVector) = build_universal_policy(values)
UniversalPolicy(values::Tuple) = build_universal_policy(values)
UniversalPolicy(first, rest...) = build_universal_policy((first, rest...))

struct RawUniversalEquivalence
    source::UInt64
    target::UInt64
end

struct RawUniversalObservation
    consumed_target_len::Csize_t
    source_len::Csize_t
    alive::UInt8
    accepting::UInt8
    reserved::NTuple{6,UInt8}
end

RawUniversalObservation() = RawUniversalObservation(0, 0, 0, 0, ntuple(_ -> 0x00, 6))

"""A precise observation of one universal target prefix."""
struct UniversalObservation
    consumed_target_length::Int
    source_length::Int
    alive::Bool
    accepting::Bool
end

UniversalObservation(raw::RawUniversalObservation) = UniversalObservation(
    Int(raw.consumed_target_len),
    Int(raw.source_len),
    raw.alive != 0,
    raw.accepting != 0,
)

universal_value(value::UInt8) = UInt64(value)
universal_value(value::UInt64) = value
universal_value(value::Char) = UInt64(UInt32(value))

policy_domain(::UnrestrictedPolicy) = UInt32(0)
policy_domain(::UniversalPolicy{UInt8}) = UInt32(VTI.UNIT_BYTE)
policy_domain(::UniversalPolicy{Char}) = UInt32(VTI.UNIT_UNICODE_SCALAR)
policy_domain(::UniversalPolicy{UInt64}) = UInt32(VTI.UNIT_U64)

raw_equivalences(::UnrestrictedPolicy) = RawUniversalEquivalence[]
raw_equivalences(policy::UniversalPolicy) = [
    RawUniversalEquivalence(universal_value(value.source), universal_value(value.target))
    for value in policy.equivalences
]

"""An immutable native universal-automaton configuration."""
mutable struct UniversalAutomaton
    handle::Ptr{Cvoid}
    variant::UniversalVariant
    policy_unit_domain::UInt32
    closed::Bool
end

function UniversalAutomaton(maximum_distance::Integer,
    policy::AbstractUniversalPolicy=UNRESTRICTED_POLICY;
    variant::UniversalVariant=UNIVERSAL_STANDARD)
    equivalences = raw_equivalences(policy)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve equivalences begin
        checked(ccall(native(:llev_universal_automaton_new), Cint,
            (UInt8, UInt32, UInt32, Ptr{RawUniversalEquivalence}, Csize_t,
                Ref{Ptr{Cvoid}}),
            checked_u8(maximum_distance, "maximum_distance"), UInt32(variant),
            policy_domain(policy), data_pointer(equivalences), length(equivalences), output),
            :llev_universal_automaton_new)
    end
    value = UniversalAutomaton(output[], variant, policy_domain(policy), false)
    finalizer(close!, value)
    value
end

function require_open(automaton::UniversalAutomaton)
    automaton.closed && throw(NativeError(Int32(STATUS_CLOSED),
        :universal_automaton, "universal automaton is closed"))
    automaton.handle
end

function byte_units(value::AbstractVector{UInt8})
    value isa Vector{UInt8} ? value : collect(UInt8, value)
end

function u64_units(value::AbstractVector{<:Integer})
    output = Vector{UInt64}(undef, length(value))
    for (index, unit) in enumerate(value)
        unit >= 0 || throw(ArgumentError("unit $index is negative"))
        unit <= typemax(UInt64) || throw(OverflowError("unit $index exceeds UInt64"))
        output[index] = UInt64(unit)
    end
    output
end

void_pointer(value::Vector) =
    isempty(value) ? Ptr{Cvoid}(C_NULL) : Ptr{Cvoid}(pointer(value))

function universal_evaluate(automaton::UniversalAutomaton, domain::VTI.UnitDomain,
    source::Vector, source_length::Integer, target::Vector, target_length::Integer,
    limits::Union{Nothing,AutomatonLimits})
    limit_ref, limit_pointer = limit_storage(limits)
    output = Ref(RawUniversalObservation())
    GC.@preserve source target limit_ref begin
        checked(ccall(native(:llev_universal_automaton_evaluate), Cint,
            (Ptr{Cvoid}, UInt32, Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t,
                Ptr{RawAutomatonLimits}, Ref{RawUniversalObservation}),
            require_open(automaton), UInt32(domain), void_pointer(source), source_length,
            void_pointer(target), target_length, limit_pointer, output),
            :llev_universal_automaton_evaluate)
    end
    UniversalObservation(output[])
end

function evaluate(automaton::UniversalAutomaton, source::AbstractString,
    target::AbstractString; limits::Union{Nothing,AutomatonLimits}=nothing)
    source_bytes = text_bytes(source)
    target_bytes = text_bytes(target)
    universal_evaluate(automaton, VTI.UNIT_UNICODE_SCALAR,
        source_bytes, length(source_bytes), target_bytes, length(target_bytes), limits)
end

function evaluate(automaton::UniversalAutomaton, source::AbstractVector{UInt8},
    target::AbstractVector{UInt8}; limits::Union{Nothing,AutomatonLimits}=nothing)
    source_bytes = byte_units(source)
    target_bytes = byte_units(target)
    universal_evaluate(automaton, VTI.UNIT_BYTE,
        source_bytes, length(source_bytes), target_bytes, length(target_bytes), limits)
end

function evaluate(automaton::UniversalAutomaton, source::AbstractVector{<:Integer},
    target::AbstractVector{<:Integer}; limits::Union{Nothing,AutomatonLimits}=nothing)
    source_units = u64_units(source)
    target_units = u64_units(target)
    universal_evaluate(automaton, VTI.UNIT_U64,
        source_units, length(source_units), target_units, length(target_units), limits)
end

"""An exclusive universal online-prefix state in one fixed unit domain."""
mutable struct UniversalOnlineAutomaton
    handle::Ptr{Cvoid}
    unit_domain::VTI.UnitDomain
    closed::Bool
end

function universal_online(automaton::UniversalAutomaton, domain::VTI.UnitDomain,
    source::Vector, source_length::Integer,
    limits::Union{Nothing,AutomatonLimits})
    limit_ref, limit_pointer = limit_storage(limits)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve source limit_ref begin
        checked(ccall(native(:llev_universal_online_new), Cint,
            (Ptr{Cvoid}, UInt32, Ptr{Cvoid}, Csize_t, Ptr{RawAutomatonLimits},
                Ref{Ptr{Cvoid}}),
            require_open(automaton), UInt32(domain), void_pointer(source), source_length,
            limit_pointer, output), :llev_universal_online_new)
    end
    value = UniversalOnlineAutomaton(output[], domain, false)
    finalizer(close!, value)
    value
end

function online(automaton::UniversalAutomaton, source::AbstractString;
    limits::Union{Nothing,AutomatonLimits}=nothing)
    bytes = text_bytes(source)
    universal_online(automaton, VTI.UNIT_UNICODE_SCALAR, bytes, length(bytes), limits)
end

function online(automaton::UniversalAutomaton, source::AbstractVector{UInt8};
    limits::Union{Nothing,AutomatonLimits}=nothing)
    bytes = byte_units(source)
    universal_online(automaton, VTI.UNIT_BYTE, bytes, length(bytes), limits)
end

function online(automaton::UniversalAutomaton, source::AbstractVector{<:Integer};
    limits::Union{Nothing,AutomatonLimits}=nothing)
    units = u64_units(source)
    universal_online(automaton, VTI.UNIT_U64, units, length(units), limits)
end

function require_open(online::UniversalOnlineAutomaton)
    online.closed && throw(NativeError(Int32(STATUS_CLOSED),
        :universal_online, "universal online automaton is closed"))
    online.handle
end

function observation(online::UniversalOnlineAutomaton)
    output = Ref(RawUniversalObservation())
    checked(ccall(native(:llev_universal_online_observation), Cint,
        (Ptr{Cvoid}, Ref{RawUniversalObservation}), require_open(online), output),
        :llev_universal_online_observation)
    UniversalObservation(output[])
end

function universal_advance(online::UniversalOnlineAutomaton, unit::UInt64)
    output = Ref(RawUniversalObservation())
    checked(ccall(native(:llev_universal_online_advance), Cint,
        (Ptr{Cvoid}, UInt64, Ref{RawUniversalObservation}), require_open(online),
        unit, output), :llev_universal_online_advance)
    UniversalObservation(output[])
end

function advance!(online::UniversalOnlineAutomaton, unit::Char)
    online.unit_domain == VTI.UNIT_UNICODE_SCALAR ||
        throw(ArgumentError("Char advances require a Unicode-scalar online automaton"))
    universal_advance(online, UInt64(UInt32(unit)))
end

function advance!(online::UniversalOnlineAutomaton, unit::UInt8)
    online.unit_domain == VTI.UNIT_BYTE ||
        throw(ArgumentError("UInt8 advances require a byte online automaton"))
    universal_advance(online, UInt64(unit))
end

function advance!(online::UniversalOnlineAutomaton, unit::Integer)
    online.unit_domain == VTI.UNIT_U64 ||
        throw(ArgumentError("integer advances require a u64 online automaton"))
    unit >= 0 || throw(ArgumentError("target unit is negative"))
    unit <= typemax(UInt64) || throw(OverflowError("target unit exceeds UInt64"))
    universal_advance(online, UInt64(unit))
end

function close!(automaton::UniversalAutomaton)
    automaton.closed && return nothing
    handle = automaton.handle
    automaton.handle = C_NULL
    automaton.closed = true
    handle == C_NULL || ccall(native(:llev_universal_automaton_free), Cvoid,
        (Ptr{Cvoid},), handle)
    nothing
end

function close!(online::UniversalOnlineAutomaton)
    online.closed && return nothing
    handle = online.handle
    online.handle = C_NULL
    online.closed = true
    handle == C_NULL || ccall(native(:llev_universal_online_free), Cvoid,
        (Ptr{Cvoid},), handle)
    nothing
end

Base.close(value::Union{UniversalAutomaton,UniversalOnlineAutomaton}) = close!(value)
Base.isopen(value::Union{UniversalAutomaton,UniversalOnlineAutomaton}) = !value.closed

accepts(automaton::Union{GeneralizedAutomaton,UniversalAutomaton}, source, target;
    limits::Union{Nothing,AutomatonLimits}=nothing) =
    evaluate(automaton, source, target; limits=limits).accepting

"""A finite exclusive stream of online-prefix observations."""
mutable struct PrefixObservations{O,T,R}
    online::O
    target::T
    offset::Int
    closed::Bool
end

Base.IteratorSize(::Type{<:PrefixObservations}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:PrefixObservations}) = Base.HasEltype()
Base.eltype(::Type{PrefixObservations{O,T,R}}) where {O,T,R} = R

function prefix_stream(online_state::O, target::T, ::Type{R}) where {O,T,R}
    value = PrefixObservations{O,T,R}(online_state, target, 1, false)
    finalizer(close!, value)
    value
end

function prefix_observations(automaton::GeneralizedAutomaton,
    source::AbstractString, target::AbstractString;
    limits::Union{Nothing,AutomatonLimits}=nothing)
    prefix_stream(online(automaton, source; limits=limits),
        collect(String(target)), GeneralizedObservation)
end

function prefix_observations(automaton::UniversalAutomaton,
    source::AbstractString, target::AbstractString;
    limits::Union{Nothing,AutomatonLimits}=nothing)
    prefix_stream(online(automaton, source; limits=limits),
        collect(String(target)), UniversalObservation)
end

function prefix_observations(automaton::UniversalAutomaton,
    source::AbstractVector{UInt8}, target::AbstractVector{UInt8};
    limits::Union{Nothing,AutomatonLimits}=nothing)
    prefix_stream(online(automaton, source; limits=limits),
        byte_units(target), UniversalObservation)
end

function prefix_observations(automaton::UniversalAutomaton,
    source::AbstractVector{<:Integer}, target::AbstractVector{<:Integer};
    limits::Union{Nothing,AutomatonLimits}=nothing)
    prefix_stream(online(automaton, source; limits=limits),
        u64_units(target), UniversalObservation)
end

function Base.iterate(value::PrefixObservations, state=nothing)
    value.closed && return nothing
    if value.offset > length(value.target)
        close!(value)
        return nothing
    end
    try
        current = advance!(value.online, value.target[value.offset])
        value.offset += 1
        (current, nothing)
    catch
        close!(value)
        rethrow()
    end
end

function close!(value::PrefixObservations)
    value.closed && return nothing
    close!(value.online)
    value.closed = true
    nothing
end

Base.close(value::PrefixObservations) = close!(value)
Base.isopen(value::PrefixObservations) = !value.closed

function prefix_observations(function_value::Function,
    automaton::Union{GeneralizedAutomaton,UniversalAutomaton}, source, target;
    limits::Union{Nothing,AutomatonLimits}=nothing)
    values = prefix_observations(automaton, source, target; limits=limits)
    try
        function_value(values)
    finally
        close!(values)
    end
end

@doc "Evaluate a complete source/target pair and return its exact final observation." evaluate
@doc "Return whether a complete source/target pair is accepted within its configured budget." accepts
@doc "Bind an immutable standalone automaton to one source for exclusive prefix processing." online
@doc "Read the last committed online-prefix observation without advancing." observation
@doc "Transactionally commit one target unit and return the resulting observation." advance!
@doc "Create or scope a finite prefix-observation stream that closes on exhaustion or failure." prefix_observations
