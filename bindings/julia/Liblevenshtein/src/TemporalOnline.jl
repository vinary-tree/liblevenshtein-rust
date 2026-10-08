"""Hard construction and per-target-sample limits for an online machine."""
struct TemporalOnlineLimits
    max_query_len::Csize_t
    max_frontier_positions::Csize_t
    max_step_work_units::Csize_t
    max_scratch_bytes::Csize_t
end

TemporalOnlineLimits(; max_query_len::Integer=1_000_000,
    max_frontier_positions::Integer=1_000_001,
    max_step_work_units::Integer=100_000_000,
    max_scratch_bytes::Integer=256 * 1024 * 1024) =
    TemporalOnlineLimits(checked_threshold(max_query_len),
        checked_threshold(max_frontier_positions),
        checked_threshold(max_step_work_units),
        checked_threshold(max_scratch_bytes))

struct RawTemporalOnlineObservation
    consumed_target_len::Csize_t
    active_positions::Csize_t
    distance_within_cutoff::Float64
    minimum_active_cost::Float64
    has_distance::UInt8
    has_minimum::UInt8
    reserved::NTuple{6,UInt8}
end

struct RawTemporalOnlineStep
    observation::RawTemporalOnlineObservation
    kind::UInt32
    reason::UInt32
    dp_cells::Csize_t
    work_units::Csize_t
    scratch_bytes::Csize_t
    queue_entries::Csize_t
end

function empty_online_observation()
    RawTemporalOnlineObservation(0, 0, 0.0, 0.0, 0, 0, ntuple(_ -> 0x00, 6))
end

"""Exact native observation after a committed target prefix.

`distance_within_cutoff` is `nothing` when the final row exceeds the
construction cutoff. DTW uses squared-distance units, including its cutoff.
"""
struct TemporalOnlineObservation
    consumed_target_len::Int
    active_positions::Int
    distance_within_cutoff::Union{Nothing,Float64}
    minimum_active_cost::Union{Nothing,Float64}
end

function online_observation(raw::RawTemporalOnlineObservation)
    raw.has_distance <= 1 && raw.has_minimum <= 1 ||
        throw(ArgumentError("invalid native online observation flags"))
    TemporalOnlineObservation(Int(raw.consumed_target_len),
        Int(raw.active_positions),
        raw.has_distance == 1 ? raw.distance_within_cutoff : nothing,
        raw.has_minimum == 1 ? raw.minimum_active_cost : nothing)
end

"""Committed online transition or an exact resource-incomplete stop."""
struct TemporalOnlineStep
    kind::Symbol
    observation::Union{Nothing,TemporalOnlineObservation}
    reason::Union{Nothing,Symbol}
    dp_cells::Int
    work_units::Int
    scratch_bytes::Int
    queue_entries::Int
end

function online_step(raw::RawTemporalOnlineStep)
    kind = raw.kind == 0 ? :advanced :
        raw.kind == 1 ? :incomplete :
        throw(ArgumentError("invalid native online step kind"))
    reason = kind === :advanced ? nothing : native_index_reason(raw.reason)
    TemporalOnlineStep(kind,
        kind === :advanced ? online_observation(raw.observation) : nothing,
        reason, Int(raw.dp_cells), Int(raw.work_units),
        Int(raw.scratch_bytes), Int(raw.queue_entries))
end

"""Fixed-query native online temporal automaton with serialized handle calls."""
mutable struct TemporalOnlineAutomaton
    handle::Ptr{Cvoid}
    lock::ReentrantLock
    closed::Bool
end

"""Construct a bounded online MSM, ERP, TWED, DTW, or Fréchet machine.

The fixed finite query is copied. Non-ERP kernels require a finite inclusive
cutoff. DTW uses squared-distance units for its cutoff and observations.
Soft-DTW has no online elastic automaton.
"""
function TemporalOnlineAutomaton(kind::Symbol,
    query::AbstractVector{<:Real}; cutoff::Real,
    parameter0::Real=0.0, parameter1::Real=0.0,
    band::Integer=0, limits::TemporalOnlineLimits=TemporalOnlineLimits())
    api_revision() >= UInt32(14) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_temporal_online_new,
            "online temporal machines require API revision 14"))
    length(query) <= limits.max_query_len ||
        throw(ArgumentError("online query exceeds max_query_len"))
    config = RawTemporalConfig(temporal_algorithm(kind), 0,
        Float64(parameter0), Float64(parameter1),
        checked_threshold(band), Float64(cutoff))
    values = Vector{Float64}(query)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve values ccall(native(:llev_temporal_online_new),
        Cint, (Ptr{Float64}, Csize_t, Ref{RawTemporalConfig},
            Ref{TemporalOnlineLimits}, Ref{Ptr{Cvoid}}),
        isempty(values) ? C_NULL : pointer(values), length(values),
        Ref(config), Ref(limits), output)
    checked(status, :llev_temporal_online_new)
    machine = TemporalOnlineAutomaton(output[], ReentrantLock(), false)
    finalizer(close!, machine)
    machine
end

"""Read the exact observation for the already committed prefix."""
function observation(machine::TemporalOnlineAutomaton)
    lock(machine.lock) do
        machine.closed && throw(ArgumentError("online temporal machine is closed"))
        output = Ref(empty_online_observation())
        status = ccall(native(:llev_temporal_online_observation), Cint,
            (Ptr{Cvoid}, Ref{RawTemporalOnlineObservation}),
            machine.handle, output)
        checked(status, :llev_temporal_online_observation)
        online_observation(output[])
    end
end

"""Consume one finite target sample; incomplete steps do not advance."""
function advance!(machine::TemporalOnlineAutomaton, sample::Real)
    lock(machine.lock) do
        machine.closed && throw(ArgumentError("online temporal machine is closed"))
        output = Ref(RawTemporalOnlineStep(empty_online_observation(),
            0, 0, 0, 0, 0, 0))
        status = ccall(native(:llev_temporal_online_advance), Cint,
            (Ptr{Cvoid}, Float64, Ref{RawTemporalOnlineStep}),
            machine.handle, Float64(sample), output)
        checked(status, :llev_temporal_online_advance)
        online_step(output[])
    end
end

"""Fixed native scratch bytes retained independently of target stream length."""
function scratch_bytes(machine::TemporalOnlineAutomaton)
    lock(machine.lock) do
        machine.closed && throw(ArgumentError("online temporal machine is closed"))
        output = Ref{Csize_t}(0)
        status = ccall(native(:llev_temporal_online_scratch_bytes), Cint,
            (Ptr{Cvoid}, Ref{Csize_t}), machine.handle, output)
        checked(status, :llev_temporal_online_scratch_bytes)
        Int(output[])
    end
end

function close!(machine::TemporalOnlineAutomaton)
    lock(machine.lock) do
        machine.closed && return nothing
        handle = machine.handle
        machine.handle = C_NULL
        machine.closed = true
        handle == C_NULL || ccall(native(:llev_temporal_online_free),
            Cvoid, (Ptr{Cvoid},), handle)
        nothing
    end
end
Base.close(machine::TemporalOnlineAutomaton) = close!(machine)
Base.isopen(machine::TemporalOnlineAutomaton) =
    lock(machine.lock) do
        !machine.closed
    end

"""A one-shot online stream stopped by a per-step native resource limit."""
struct TemporalOnlineIncomplete <: Exception
    reason::Symbol
end
Base.showerror(io::IO, error::TemporalOnlineIncomplete) =
    print(io, "online temporal stream incomplete: ", error.reason)

"""Lazy sequence of exact prefix observations from an owned online machine."""
mutable struct TemporalOnlineStream
    source::Any
    machine::TemporalOnlineAutomaton
    source_state::Any
    started::Bool
    closed::Bool
end

Base.IteratorSize(::Type{TemporalOnlineStream}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TemporalOnlineStream}) = Base.HasEltype()
Base.eltype(::Type{TemporalOnlineStream}) = TemporalOnlineObservation

function online_observations(kind::Symbol,
    query::AbstractVector{<:Real}, source; kwargs...)
    machine = TemporalOnlineAutomaton(kind, query; kwargs...)
    TemporalOnlineStream(source, machine, nothing, false, false)
end

function Base.iterate(stream::TemporalOnlineStream, state=nothing)
    stream.closed && return nothing
    try
        item = stream.started ?
            iterate(stream.source, stream.source_state) :
            iterate(stream.source)
        if item === nothing
            close!(stream)
            return nothing
        end
        sample, stream.source_state = item
        stream.started = true
        sample isa Real ||
            throw(ArgumentError("online target samples must be real"))
        step = advance!(stream.machine, sample)
        step.kind === :incomplete &&
            throw(TemporalOnlineIncomplete(step.reason::Symbol))
        (step.observation::TemporalOnlineObservation, nothing)
    catch
        close!(stream)
        rethrow()
    end
end

function close!(stream::TemporalOnlineStream)
    stream.closed && return nothing
    stream.closed = true
    stream.source = nothing
    stream.source_state = nothing
    close!(stream.machine)
end
Base.close(stream::TemporalOnlineStream) = close!(stream)
Base.isopen(stream::TemporalOnlineStream) = !stream.closed
cancel!(stream::TemporalOnlineStream) = close!(stream)

"""Reduce exact online observations and close the stream on every exit."""
function reduce_observations!(function_value, initial,
    stream::TemporalOnlineStream)
    accumulator = initial
    try
        for observed in stream
            accumulator = function_value(accumulator, observed)
        end
        accumulator
    finally
        close!(stream)
    end
end
