struct RawTemporalAlignmentLimits
    temporal::TemporalLimits
    max_witness_bytes::Csize_t
end

struct RawTemporalAlignmentStep
    operation::UInt32
    flags::UInt32
    query_endpoint::UInt64
    candidate_endpoint::UInt64
    local_cost_bits::UInt64
end

struct RawTemporalAlignmentOutcome
    kind::UInt32
    reason::UInt32
    distance::Float64
    step_count::Csize_t
    dp_cells::Csize_t
    work_units::Csize_t
    scratch_bytes::Csize_t
    witness_bytes::Csize_t
end

empty_alignment_outcome() = RawTemporalAlignmentOutcome(0, 0, 0.0, 0, 0, 0, 0, 0)

"""One forward operation from a native alignment witness.

Endpoints are Julia one-based indices when present. MSM steps contain only
the operation tag; their endpoints and local cost are `nothing`.
"""
struct TemporalAlignmentStep
    operation::Symbol
    query_endpoint::Union{Nothing,Int}
    candidate_endpoint::Union{Nothing,Int}
    local_cost::Union{Nothing,Float64}
end

"""Owned, immutable native alignment witness with bounded page iteration."""
mutable struct TemporalAlignmentWitness
    handle::Ptr{Cvoid}
    lock::ReentrantLock
    kind::Symbol
    step_count::Int
    closed::Bool
end

Base.length(witness::TemporalAlignmentWitness) = witness.step_count
Base.isempty(witness::TemporalAlignmentWitness) = witness.step_count == 0
Base.IteratorSize(::Type{TemporalAlignmentWitness}) = Base.HasLength()
Base.IteratorEltype(::Type{TemporalAlignmentWitness}) = Base.HasEltype()
Base.eltype(::Type{TemporalAlignmentWitness}) = TemporalAlignmentStep

function close!(witness::TemporalAlignmentWitness)
    lock(witness.lock) do
        witness.closed && return nothing
        handle = witness.handle
        witness.handle = C_NULL
        witness.closed = true
        handle == C_NULL || ccall(native(:llev_temporal_alignment_free),
            Cvoid, (Ptr{Cvoid},), handle)
        nothing
    end
end
Base.close(witness::TemporalAlignmentWitness) = close!(witness)
Base.isopen(witness::TemporalAlignmentWitness) = lock(witness.lock) do
    !witness.closed
end

function alignment_step(raw::RawTemporalAlignmentStep, kind::Symbol)
    raw.operation in UInt32(1):UInt32(3) ||
        throw(ArgumentError("native alignment operation is invalid"))
    operation = kind === :msm ?
        (:move, :merge, :split)[Int(raw.operation)] :
        (:align, :advance_query, :advance_candidate)[Int(raw.operation)]
    if kind === :msm
        raw.flags == 0 || throw(ArgumentError("native MSM step flags are invalid"))
        return TemporalAlignmentStep(operation, nothing, nothing, nothing)
    end
    raw.flags <= 3 || throw(ArgumentError("native alignment flags are invalid"))
    TemporalAlignmentStep(operation,
        raw.flags & 1 == 0 ? nothing : Int(raw.query_endpoint) + 1,
        raw.flags & 2 == 0 ? nothing : Int(raw.candidate_endpoint) + 1,
        reinterpret(Float64, raw.local_cost_bits))
end

"""Copy one bounded page from a native witness, starting at a one-based step."""
function alignment_page(witness::TemporalAlignmentWitness,
    first::Integer=1; page_size::Integer=256)
    start = checked_threshold(first)
    start >= 1 || throw(ArgumentError("first must be one-based"))
    capacity = checked_threshold(page_size)
    0 < capacity <= 65_536 ||
        throw(ArgumentError("page_size must be 1..65,536"))
    start <= witness.step_count + 1 ||
        throw(ArgumentError("first exceeds alignment length"))
    lock(witness.lock) do
        witness.closed && throw(ArgumentError("alignment witness is closed"))
        available = min(Int(capacity), witness.step_count - Int(start) + 1)
        raw_steps = Vector{RawTemporalAlignmentStep}(undef, available)
        written = Ref{Csize_t}(0)
        status = GC.@preserve raw_steps ccall(
            native(:llev_temporal_alignment_page), Cint,
            (Ptr{Cvoid}, Csize_t, Ptr{RawTemporalAlignmentStep},
                Csize_t, Ref{Csize_t}),
            witness.handle, start - 1,
            isempty(raw_steps) ? C_NULL : pointer(raw_steps),
            length(raw_steps), written)
        checked(status, :llev_temporal_alignment_page)
        written[] <= length(raw_steps) ||
            error("native alignment page exceeds capacity")
        [alignment_step(raw_steps[i], witness.kind) for i in 1:Int(written[])]
    end
end

function Base.iterate(witness::TemporalAlignmentWitness,
    state=(0, TemporalAlignmentStep[], 1))
    loaded, page, offset = state
    if offset > length(page)
        loaded >= length(witness) && return nothing
        page = alignment_page(witness, loaded + 1)
        isempty(page) && error("native alignment page ended early")
        loaded += length(page)
        offset = 1
    end
    (page[offset], (loaded, page, offset + 1))
end

"""Fold bounded pages without materializing an entire native witness."""
function reduce_batches!(function_value, initial,
    witness::TemporalAlignmentWitness; batch_size::Integer=256)
    size = checked_threshold(batch_size)
    0 < size <= 65_536 || throw(ArgumentError("batch_size must be 1..65,536"))
    accumulator = initial
    first = 1
    while first <= length(witness)
        page = alignment_page(witness, first; page_size=size)
        isempty(page) && error("native alignment page ended early")
        accumulator = function_value(accumulator, page)
        first += length(page)
    end
    accumulator
end

"""Tagged bounded alignment result. Only `:finite` owns a witness."""
struct TemporalAlignmentOutcome
    kind::Symbol
    distance::Union{Nothing,Float64}
    reason::Union{Nothing,Symbol}
    witness::Union{Nothing,TemporalAlignmentWitness}
    usage::NamedTuple
end

function alignment_outcome(raw::RawTemporalAlignmentOutcome,
    handle::Ptr{Cvoid}, kind::Symbol)
    result_kind = raw.kind == 0 ? :finite :
        raw.kind == 1 ? :above_cutoff :
        raw.kind == 2 ? :no_alignment :
        raw.kind == 3 ? :incomplete :
        error("unknown native alignment outcome")
    (result_kind === :finite) == (handle != C_NULL) ||
        error("native alignment handle and outcome disagree")
    reason = result_kind === :incomplete ?
        raw.reason == 5 ? :witness_bytes : native_index_reason(raw.reason) :
        nothing
    witness = if handle == C_NULL
        nothing
    else
        owned = TemporalAlignmentWitness(handle, ReentrantLock(), kind,
            Int(raw.step_count), false)
        finalizer(close!, owned)
        owned
    end
    usage = (dp_cells=raw.dp_cells, work_units=raw.work_units,
        scratch_bytes=raw.scratch_bytes, witness_bytes=raw.witness_bytes)
    TemporalAlignmentOutcome(result_kind,
        result_kind === :finite ? raw.distance : nothing,
        reason, witness, usage)
end

"""Extract a replayable native MSM, ERP, TWED, DTW, or Fréchet alignment.

The witness stays in native bounded storage and Julia reads it by pages.
Positive infinity requests an unthresholded score. Soft-DTW has no witness.
"""
function temporal_alignment(kind::Symbol,
    query::AbstractVector{<:Real}, candidate::AbstractVector{<:Real};
    parameter0::Real=0.0, parameter1::Real=0.0,
    band::Integer=0, cutoff::Real=Inf,
    limits::TemporalLimits=TemporalLimits(),
    max_witness_bytes::Integer=64 * 1024 * 1024)
    api_revision() >= UInt32(21) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_temporal_alignment_new,
            "alignment witnesses require native API revision 21"))
    kind === :soft_dtw &&
        throw(ArgumentError("Soft-DTW has no alignment witness"))
    length(query) <= limits.max_series_len &&
        length(candidate) <= limits.max_series_len ||
        throw(ArgumentError("alignment series exceeds max_series_len"))
    config = RawTemporalConfig(temporal_algorithm(kind), 0,
        Float64(parameter0), Float64(parameter1), checked_threshold(band),
        Float64(cutoff))
    bound = RawTemporalAlignmentLimits(limits,
        checked_threshold(max_witness_bytes))
    x = query isa Vector{Float64} ? query : Vector{Float64}(query)
    y = candidate isa Vector{Float64} ? candidate : Vector{Float64}(candidate)
    handle = Ref{Ptr{Cvoid}}(C_NULL)
    raw = Ref(empty_alignment_outcome())
    status = GC.@preserve x y ccall(native(:llev_temporal_alignment_new),
        Cint,
        (Ptr{Float64}, Csize_t, Ptr{Float64}, Csize_t,
            Ref{RawTemporalConfig}, Ref{RawTemporalAlignmentLimits},
            Ref{Ptr{Cvoid}}, Ref{RawTemporalAlignmentOutcome}),
        isempty(x) ? C_NULL : pointer(x), length(x),
        isempty(y) ? C_NULL : pointer(y), length(y),
        Ref(config), Ref(bound), handle, raw)
    checked(status, :llev_temporal_alignment_new)
    alignment_outcome(raw[], handle[], kind)
end

"""Extract a replayable metric TWED witness over physical timestamps."""
function timestamped_twed_alignment(config::MetricTimestampedTwedConfig,
    query::TimestampedSeries, candidate::TimestampedSeries;
    cutoff::Real=Inf, limits::TemporalLimits=TemporalLimits(),
    max_witness_bytes::Integer=64 * 1024 * 1024)
    api_revision() >= UInt32(21) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_timestamped_twed_alignment_new,
            "timestamped witnesses require native API revision 21"))
    query.unit === candidate.unit && query.origin == candidate.origin ||
        throw(ArgumentError("timestamped series unit or origin differs"))
    x = getfield(query, :_values)
    xt = getfield(query, :_timestamps)
    y = getfield(candidate, :_values)
    yt = getfield(candidate, :_timestamps)
    length(x) <= limits.max_series_len && length(y) <= limits.max_series_len ||
        throw(ArgumentError("timestamped alignment exceeds max_series_len"))
    first = RawTimestampedSeriesView(pointer(x), pointer(xt), length(x),
        timestamp_unit_code(query.unit), 0, query.origin)
    second = RawTimestampedSeriesView(pointer(y), pointer(yt), length(y),
        timestamp_unit_code(candidate.unit), 0, candidate.origin)
    bound = RawTemporalAlignmentLimits(limits,
        checked_threshold(max_witness_bytes))
    handle = Ref{Ptr{Cvoid}}(C_NULL)
    raw = Ref(empty_alignment_outcome())
    status = GC.@preserve x xt y yt ccall(
        native(:llev_timestamped_twed_alignment_new), Cint,
        (Ref{RawTimestampedSeriesView}, Ref{RawTimestampedSeriesView},
            Float64, Float64, Float64, Ref{RawTemporalAlignmentLimits},
            Ref{Ptr{Cvoid}}, Ref{RawTemporalAlignmentOutcome}),
        Ref(first), Ref(second), config.stiffness, config.gap_penalty,
        Float64(cutoff), Ref(bound), handle, raw)
    checked(status, :llev_timestamped_twed_alignment_new)
    alignment_outcome(raw[], handle[], :timestamped_twed)
end

"""Replay a scalar witness against supplied operands and its captured config."""
function replay_alignment(witness::TemporalAlignmentWitness,
    query::AbstractVector{<:Real}, candidate::AbstractVector{<:Real})
    witness.kind === :timestamped_twed &&
        throw(ArgumentError("timestamped witness needs timestamped operands"))
    x = query isa Vector{Float64} ? query : Vector{Float64}(query)
    y = candidate isa Vector{Float64} ? candidate : Vector{Float64}(candidate)
    lock(witness.lock) do
        witness.closed && throw(ArgumentError("alignment witness is closed"))
        distance = Ref{Float64}(0.0)
        status = GC.@preserve x y ccall(
            native(:llev_temporal_alignment_replay), Cint,
            (Ptr{Cvoid}, Ptr{Float64}, Csize_t,
                Ptr{Float64}, Csize_t, Ref{Float64}),
            witness.handle, isempty(x) ? C_NULL : pointer(x), length(x),
            isempty(y) ? C_NULL : pointer(y), length(y), distance)
        checked(status, :llev_temporal_alignment_replay)
        distance[]
    end
end

"""Replay a physical-time TWED witness against supplied timestamped series."""
function replay_alignment(witness::TemporalAlignmentWitness,
    query::TimestampedSeries, candidate::TimestampedSeries)
    witness.kind === :timestamped_twed ||
        throw(ArgumentError("scalar witness needs scalar operands"))
    x = getfield(query, :_values)
    xt = getfield(query, :_timestamps)
    y = getfield(candidate, :_values)
    yt = getfield(candidate, :_timestamps)
    first = RawTimestampedSeriesView(pointer(x), pointer(xt), length(x),
        timestamp_unit_code(query.unit), 0, query.origin)
    second = RawTimestampedSeriesView(pointer(y), pointer(yt), length(y),
        timestamp_unit_code(candidate.unit), 0, candidate.origin)
    lock(witness.lock) do
        witness.closed && throw(ArgumentError("alignment witness is closed"))
        distance = Ref{Float64}(0.0)
        status = GC.@preserve x xt y yt ccall(
            native(:llev_timestamped_twed_alignment_replay), Cint,
            (Ptr{Cvoid}, Ref{RawTimestampedSeriesView},
                Ref{RawTimestampedSeriesView}, Ref{Float64}),
            witness.handle, Ref(first), Ref(second), distance)
        checked(status, :llev_timestamped_twed_alignment_replay)
        distance[]
    end
end
