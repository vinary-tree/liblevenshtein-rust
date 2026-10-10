"""Fixed-query native vector Fréchet automaton with serialized handle calls."""
mutable struct VectorFrechetOnlineAutomaton
    handle::Ptr{Cvoid}
    dimension::Int
    lock::ReentrantLock
    closed::Bool
end

"""Copy a typed path into a bounded native online vector Fréchet machine.

The machine retains a fixed query and two query-width frontier generations.
It consumes whole vector points and retains no target history. The metric
handle may close after construction; the machine owns its native metric copy.
"""
function VectorFrechetOnlineAutomaton(metric::FixedChannelMetric,
    query::VectorTemporalSeries; cutoff::Real,
    limits::TemporalOnlineLimits=TemporalOnlineLimits())
    api_revision() >= UInt32(23) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_vector_frechet_online_new,
            "vector online Fréchet requires native API revision 23"))
    query.timestamps === nothing ||
        throw(ArgumentError("vector Fréchet query must have no timestamps"))
    size(query.samples, 1) == metric.dimension ||
        throw(ArgumentError("vector query dimension differs from metric"))
    size(query.samples, 2) <= limits.max_query_len ||
        throw(ArgumentError("vector query exceeds max_query_len"))
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = lock(metric.lock) do
        metric.closed && throw(ArgumentError("vector metric is closed"))
        GC.@preserve metric query ccall(
            native(:llev_vector_frechet_online_new), Cint,
            (Ptr{Cvoid}, Ref{RawVectorSeriesView}, Float64,
                Ref{TemporalOnlineLimits}, Ref{Ptr{Cvoid}}),
            metric.handle, Ref(raw_vector_series(query)), Float64(cutoff),
            Ref(limits), output)
    end
    checked(status, :llev_vector_frechet_online_new)
    machine = VectorFrechetOnlineAutomaton(output[], metric.dimension,
        ReentrantLock(), false)
    finalizer(close!, machine)
    machine
end

"""Construct an online Fréchet machine with L1, L2, or L∞ point distance."""
function VectorFrechetOnlineAutomaton(ground::Symbol,
    query::VectorTemporalSeries; cutoff::Real,
    limits::TemporalOnlineLimits=TemporalOnlineLimits())
    api_revision() >= UInt32(27) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_vector_frechet_ground_online_new,
            "ground-metric vector Fréchet requires API revision 27"))
    ground_code = vector_frechet_ground_code(ground)
    query.timestamps === nothing ||
        throw(ArgumentError("vector Fréchet query must have no timestamps"))
    dimension = size(query.samples, 1)
    dimension > 0 || throw(ArgumentError("vector query dimension must be positive"))
    size(query.samples, 2) <= limits.max_query_len ||
        throw(ArgumentError("vector query exceeds max_query_len"))
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve query ccall(
        native(:llev_vector_frechet_ground_online_new), Cint,
        (UInt32, Ref{RawVectorSeriesView}, Float64,
            Ref{TemporalOnlineLimits}, Ref{Ptr{Cvoid}}),
        ground_code, Ref(raw_vector_series(query)), Float64(cutoff),
        Ref(limits), output)
    checked(status, :llev_vector_frechet_ground_online_new)
    machine = VectorFrechetOnlineAutomaton(output[], dimension,
        ReentrantLock(), false)
    finalizer(close!, machine)
    machine
end

"""Read the exact already committed vector-target prefix observation."""
function observation(machine::VectorFrechetOnlineAutomaton)
    lock(machine.lock) do
        machine.closed && throw(ArgumentError("vector online machine is closed"))
        output = Ref(empty_online_observation())
        status = ccall(native(:llev_vector_frechet_online_observation),
            Cint, (Ptr{Cvoid}, Ref{RawTemporalOnlineObservation}),
            machine.handle, output)
        checked(status, :llev_vector_frechet_online_observation)
        online_observation(output[])
    end
end

"""Consume one point; incomplete work leaves the prior prefix unchanged."""
function advance!(machine::VectorFrechetOnlineAutomaton,
    point::AbstractVector{<:Real})
    length(point) == machine.dimension ||
        throw(ArgumentError("vector online point dimension mismatch"))
    copied = point isa Vector{Float64} ? point : Vector{Float64}(point)
    lock(machine.lock) do
        machine.closed && throw(ArgumentError("vector online machine is closed"))
        output = Ref(RawTemporalOnlineStep(empty_online_observation(),
            0, 0, 0, 0, 0, 0))
        status = GC.@preserve copied ccall(
            native(:llev_vector_frechet_online_advance), Cint,
            (Ptr{Cvoid}, Ptr{Float64}, Csize_t,
                Ref{RawTemporalOnlineStep}),
            machine.handle, pointer(copied), length(copied), output)
        checked(status, :llev_vector_frechet_online_advance)
        online_step(output[])
    end
end

"""Fixed native scratch bytes, independent of target-prefix length."""
function scratch_bytes(machine::VectorFrechetOnlineAutomaton)
    lock(machine.lock) do
        machine.closed && throw(ArgumentError("vector online machine is closed"))
        output = Ref{Csize_t}(0)
        status = ccall(native(:llev_vector_frechet_online_scratch_bytes),
            Cint, (Ptr{Cvoid}, Ref{Csize_t}), machine.handle, output)
        checked(status, :llev_vector_frechet_online_scratch_bytes)
        Int(output[])
    end
end

function close!(machine::VectorFrechetOnlineAutomaton)
    lock(machine.lock) do
        machine.closed && return nothing
        handle = machine.handle
        machine.handle = C_NULL
        machine.closed = true
        handle == C_NULL || ccall(native(:llev_vector_frechet_online_free),
            Cvoid, (Ptr{Cvoid},), handle)
        nothing
    end
end
Base.close(machine::VectorFrechetOnlineAutomaton) = close!(machine)
Base.isopen(machine::VectorFrechetOnlineAutomaton) =
    lock(machine.lock) do
        !machine.closed
    end

"""Lazy one-shot stream of exact vector Fréchet prefix observations."""
mutable struct VectorFrechetOnlineStream
    source::Any
    machine::VectorFrechetOnlineAutomaton
    source_state::Any
    started::Bool
    closed::Bool
end

Base.IteratorSize(::Type{VectorFrechetOnlineStream}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{VectorFrechetOnlineStream}) = Base.HasEltype()
Base.eltype(::Type{VectorFrechetOnlineStream}) = TemporalOnlineObservation

function vector_frechet_online_observations(metric::FixedChannelMetric,
    query::VectorTemporalSeries, source; kwargs...)
    machine = VectorFrechetOnlineAutomaton(metric, query; kwargs...)
    VectorFrechetOnlineStream(source, machine, nothing, false, false)
end

function vector_frechet_online_observations(ground::Symbol,
    query::VectorTemporalSeries, source; kwargs...)
    machine = VectorFrechetOnlineAutomaton(ground, query; kwargs...)
    VectorFrechetOnlineStream(source, machine, nothing, false, false)
end

function Base.iterate(stream::VectorFrechetOnlineStream, state=nothing)
    stream.closed && return nothing
    try
        item = stream.started ?
            iterate(stream.source, stream.source_state) :
            iterate(stream.source)
        if item === nothing
            close!(stream)
            return nothing
        end
        point, stream.source_state = item
        stream.started = true
        point isa AbstractVector{<:Real} ||
            throw(ArgumentError("vector online target points must be real vectors"))
        step = advance!(stream.machine, point)
        step.kind === :incomplete &&
            throw(TemporalOnlineIncomplete(step.reason::Symbol))
        (step.observation::TemporalOnlineObservation, nothing)
    catch
        close!(stream)
        rethrow()
    end
end

function close!(stream::VectorFrechetOnlineStream)
    stream.closed && return nothing
    stream.closed = true
    stream.source = nothing
    stream.source_state = nothing
    close!(stream.machine)
end
Base.close(stream::VectorFrechetOnlineStream) = close!(stream)
Base.isopen(stream::VectorFrechetOnlineStream) = !stream.closed
cancel!(stream::VectorFrechetOnlineStream) = close!(stream)

function reduce_observations!(function_value, initial,
    stream::VectorFrechetOnlineStream)
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
