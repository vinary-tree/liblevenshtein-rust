"""Owned chronological samples emitted by a fixed-width rolling stream."""
struct RollingWindowSnapshot
    window_id::UInt64
    start_offset::UInt64
    end_offset::UInt64
    _values::Vector{Float64}
end

Base.length(snapshot::RollingWindowSnapshot) = length(getfield(snapshot, :_values))
Base.getproperty(snapshot::RollingWindowSnapshot, name::Symbol) =
    name === :values ? copy(getfield(snapshot, :_values)) :
    getfield(snapshot, name)
Base.propertynames(::RollingWindowSnapshot, private::Bool=false) =
    private ? (:window_id, :start_offset, :end_offset, :values, :_values) :
        (:window_id, :start_offset, :end_offset, :values)

"""Logical resources charged by one rolling input step."""
struct RollingWindowUsage
    work_units::UInt
    scratch_bytes::UInt
    snapshot_bytes::UInt
    queue_entries::UInt
end

"""A committed rolling input or an incomplete, uncommitted input."""
struct RollingWindowStep
    kind::Symbol
    snapshot::Union{Nothing,RollingWindowSnapshot}
    reason::Union{Nothing,Symbol}
    usage::RollingWindowUsage
end

"""A rolling stream stopped before its next sample was committed."""
struct RollingWindowIncomplete <: Exception
    reason::Symbol
end
Base.showerror(io::IO, error::RollingWindowIncomplete) =
    print(io, "rolling window incomplete: ", error.reason)

"""Fixed-storage rolling query machine with native width/stride semantics."""
mutable struct BoundedRollingWindow
    storage::Vector{Float64}
    width::Int
    stride::UInt
    write_index::Int
    retained::Int
    consumed::UInt64
    emitted::UInt64
    since_emit::UInt
    scratch_bytes::UInt
    closed::Bool
end

function BoundedRollingWindow(window_len::Integer, stride::Integer;
    max_series_len::Integer=1_000_000,
    max_scratch_bytes::Integer=512 * 1024 * 1024,
    max_snapshot_bytes::Integer=1024 * 1024 * 1024)
    width = checked_threshold(window_len)
    step = checked_threshold(stride)
    width > 0 || throw(ArgumentError("rolling window length must be positive"))
    step > 0 || throw(ArgumentError("rolling window stride must be positive"))
    width <= checked_threshold(max_series_len) ||
        throw(ArgumentError("rolling window exceeds max_series_len"))
    width <= checked_threshold(max_scratch_bytes) ÷ sizeof(Float64) ||
        throw(ArgumentError("rolling window exceeds max_scratch_bytes"))
    width <= checked_threshold(max_snapshot_bytes) ÷ sizeof(Float64) ||
        throw(ArgumentError("rolling window exceeds max_snapshot_bytes"))
    width <= typemax(Int) ||
        throw(ArgumentError("rolling window length exceeds Int"))
    storage = try
        zeros(Float64, Int(width))
    catch error
        error isa OutOfMemoryError || rethrow()
        throw(RollingWindowIncomplete(:allocation))
    end
    BoundedRollingWindow(storage, Int(width), UInt(step),
        1, 0, 0, 0, 0, UInt(width * sizeof(Float64)), false)
end

function rolling_usage(machine::BoundedRollingWindow,
    emitted::Bool, work::UInt)
    RollingWindowUsage(work, machine.scratch_bytes,
        emitted ? machine.scratch_bytes : UInt(0),
        UInt(machine.retained))
end

"""Consume one finite sample transactionally and emit at width/stride boundaries."""
function advance!(machine::BoundedRollingWindow, sample::Real)
    machine.closed && throw(ArgumentError("rolling window is closed"))
    value = Float64(sample)
    isfinite(value) || throw(ArgumentError("rolling sample must be finite"))
    machine.consumed == typemax(UInt64) &&
        return RollingWindowStep(:incomplete, nothing, :series_length,
            rolling_usage(machine, false, UInt(0)))
    next_consumed = machine.consumed + UInt64(1)
    will_emit = (machine.retained + 1 == machine.width &&
        machine.emitted == 0) ||
        (machine.emitted > 0 && machine.since_emit + UInt(1) ==
            machine.stride)
    if will_emit && machine.emitted == typemax(UInt64)
        return RollingWindowStep(:incomplete, nothing, :results,
            rolling_usage(machine, false, UInt(0)))
    end

    snapshot = nothing
    if will_emit
        values = try
            Vector{Float64}(undef, machine.width)
        catch error
            error isa OutOfMemoryError || rethrow()
            return RollingWindowStep(:incomplete, nothing, :allocation,
                rolling_usage(machine, false, UInt(0)))
        end
        if machine.retained < machine.width
            copyto!(values, 1, machine.storage, 1, machine.retained)
        else
            for offset in 1:(machine.width - 1)
                values[offset] = machine.storage[
                    mod1(machine.write_index + offset, machine.width)]
            end
        end
        values[end] = value
        snapshot = RollingWindowSnapshot(machine.emitted,
            next_consumed - UInt64(machine.width), next_consumed,
            values)
    end

    machine.storage[machine.write_index] = value
    machine.write_index = mod1(machine.write_index + 1, machine.width)
    machine.retained = min(machine.retained + 1, machine.width)
    machine.consumed = next_consumed
    machine.since_emit = will_emit ? UInt(0) :
        (machine.emitted > 0 ? machine.since_emit + UInt(1) : UInt(0))
    machine.emitted += UInt64(will_emit)
    RollingWindowStep(:advanced, snapshot, nothing,
        rolling_usage(machine, will_emit, UInt(1)))
end

function close!(machine::BoundedRollingWindow)
    machine.closed && return nothing
    machine.closed = true
    machine.storage = Float64[]
    nothing
end
Base.close(machine::BoundedRollingWindow) = close!(machine)
Base.isopen(machine::BoundedRollingWindow) = !machine.closed

"""One-shot lazy iterator over snapshots from an unknown-length input stream."""
mutable struct RollingWindowStream
    source::Any
    machine::BoundedRollingWindow
    source_state::Any
    started::Bool
    closed::Bool
end

Base.IteratorSize(::Type{RollingWindowStream}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{RollingWindowStream}) = Base.HasEltype()
Base.eltype(::Type{RollingWindowStream}) = RollingWindowSnapshot

function rolling_windows(source, window_len::Integer, stride::Integer; kwargs...)
    machine = BoundedRollingWindow(window_len, stride; kwargs...)
    RollingWindowStream(source, machine, nothing, false, false)
end

function Base.iterate(stream::RollingWindowStream, state=nothing)
    stream.closed && return nothing
    try
        while true
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
                throw(ArgumentError("rolling stream samples must be real"))
            step = advance!(stream.machine, sample)
            step.kind === :incomplete &&
                throw(RollingWindowIncomplete(step.reason::Symbol))
            step.snapshot === nothing || return (step.snapshot, nothing)
        end
    catch
        close!(stream)
        rethrow()
    end
end

function close!(stream::RollingWindowStream)
    stream.closed && return nothing
    stream.closed = true
    stream.source = nothing
    stream.source_state = nothing
    close!(stream.machine)
end
Base.close(stream::RollingWindowStream) = close!(stream)
Base.isopen(stream::RollingWindowStream) = !stream.closed
cancel!(stream::RollingWindowStream) = close!(stream)

"""Reduce emitted snapshots and close a rolling stream on every exit path."""
function reduce_windows!(function_value, initial, stream::RollingWindowStream)
    accumulator = initial
    try
        for snapshot in stream
            accumulator = function_value(accumulator, snapshot)
        end
        accumulator
    finally
        close!(stream)
    end
end

"""Start an exact native indexed range query from an immutable rolling window."""
function query_index_range(index::TemporalIndex,
    snapshot::RollingWindowSnapshot; kwargs...)
    query_index_range(index, getfield(snapshot, :_values); kwargs...)
end
