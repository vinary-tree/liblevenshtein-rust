"""Cumulative limits for one lazy native MSM candidate prefilter."""
struct MsmPrefilterLimits
    max_candidates::Csize_t
    max_results::Csize_t
    max_total_work_units::Csize_t
end

MsmPrefilterLimits(; max_candidates::Integer=100_000,
    max_results::Integer=100_000,
    max_total_work_units::Integer=100_000_000) =
    MsmPrefilterLimits(checked_threshold(max_candidates),
        checked_threshold(max_results),
        checked_threshold(max_total_work_units))

"""A copied candidate retained by a native MSM bound or heuristic score."""
struct MsmPrefilterCandidate
    id::UInt64
    samples::Vector{Float64}
    score::Float64
end

"""Closeable one-shot MSM prefilter over a copied temporal source."""
mutable struct MsmPrefilterCursor
    source::TemporalSeriesSource
    query::Vector{Float64}
    mode::Symbol
    split_merge_cost::Float64
    threshold::Float64
    limits::TemporalLimits
    query_limits::MsmPrefilterLimits
    next_index::Int
    visited::Csize_t
    emitted::Csize_t
    work_units::Csize_t
    closed::Bool
    lock::ReentrantLock
end

Base.IteratorSize(::Type{MsmPrefilterCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{MsmPrefilterCursor}) = Base.HasEltype()
Base.eltype(::Type{MsmPrefilterCursor}) = MsmPrefilterCandidate

function msm_prefilter_mode(mode::Symbol)
    mode === :length && return :msm_length
    mode === :euclidean && return :msm_euclidean_heuristic
    mode === :l1 && return :msm_l1_heuristic
    mode === :combined && return :msm_combined_heuristic
    throw(ArgumentError("unknown MSM prefilter mode"))
end

"""Filter a copied source lazily through the native MSM prefilter selector.

`:length` is admissible for exact pruning. `:euclidean`, `:l1`, and
`:combined` can exceed true MSM distance and require
`allow_false_negatives=true`. Every emitted series is copied so callers can
mutate it without changing the retained source. Native per-candidate work
and cumulative candidate, result, and work ceilings are enforced before
each result is published.
"""
function filter_msm_source(source::TemporalSeriesSource,
    query::AbstractVector{<:Real}; mode::Symbol=:length,
    split_merge_cost::Real=1.0, threshold::Real=Inf,
    limits::TemporalLimits=TemporalLimits(),
    query_limits::MsmPrefilterLimits=MsmPrefilterLimits(),
    allow_false_negatives::Bool=false)
    selected = msm_prefilter_mode(mode)
    mode === :length || allow_false_negatives ||
        throw(ArgumentError("heuristic MSM filtering needs allow_false_negatives=true"))
    api_revision() >= UInt32(25) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_temporal_lower_bound,
            "MSM prefilters require native API revision 25"))
    length(query) <= limits.max_series_len ||
        throw(ArgumentError("query exceeds max_series_len"))
    values = Vector{Float64}(query)
    all(isfinite, values) ||
        throw(ArgumentError("MSM prefilter query must be finite"))
    cost = Float64(split_merge_cost)
    isfinite(cost) && cost >= 0.0 ||
        throw(ArgumentError("MSM split/merge cost must be finite and nonnegative"))
    tau = Float64(threshold)
    !isnan(tau) && tau >= 0.0 ||
        throw(ArgumentError("MSM prefilter threshold must be nonnegative"))
    cursor = MsmPrefilterCursor(source, values, selected, cost, tau,
        limits, query_limits, 1, 0, 0, 0, false, ReentrantLock())
    finalizer(close!, cursor)
    cursor
end

function next_msm_prefilter_candidate!(cursor::MsmPrefilterCursor)
    while cursor.next_index <= length(cursor.source.entries)
        cursor.visited < cursor.query_limits.max_candidates ||
            throw(TemporalQueryIncomplete(:candidates, nothing, nothing))
        entry = cursor.source.entries[cursor.next_index]
        length(entry.samples) <= cursor.limits.max_series_len ||
            throw(TemporalQueryIncomplete(:series_len, entry.id, nothing))
        cost = cursor.mode in (:msm_length, :msm_combined_heuristic) ?
            cursor.split_merge_cost : 0.0
        outcome = temporal_lower_bound(cursor.mode, cursor.query,
            entry.samples; parameter0=cost, limits=cursor.limits)
        outcome.kind === :incomplete &&
            throw(TemporalQueryIncomplete(:prefilter, entry.id, outcome.reason))
        outcome.work_units <=
            cursor.query_limits.max_total_work_units - cursor.work_units ||
            throw(TemporalQueryIncomplete(:work_units, entry.id, nothing))
        cursor.work_units += outcome.work_units
        cursor.visited += 1
        cursor.next_index += 1
        if outcome.kind === :finite && outcome.value <= cursor.threshold
            cursor.emitted < cursor.query_limits.max_results ||
                throw(TemporalQueryIncomplete(:results, entry.id, nothing))
            copied = copy(entry.samples)
            cursor.emitted += 1
            return MsmPrefilterCandidate(entry.id, copied,
                outcome.value::Float64)
        end
    end
    nothing
end

function next_batch!(cursor::MsmPrefilterCursor,
    maximum::Integer=DEFAULT_MATCH_BATCH)
    lock(cursor.lock) do
        cursor.closed && return nothing
        0 < maximum <= 65_536 ||
            throw(ArgumentError("batch maximum must be 1..65,536"))
        batch = MsmPrefilterCandidate[]
        try
            while length(batch) < maximum
                candidate = next_msm_prefilter_candidate!(cursor)
                if candidate === nothing
                    close!(cursor)
                    return isempty(batch) ? nothing : batch
                end
                push!(batch, candidate)
            end
            batch
        catch
            close!(cursor)
            rethrow()
        end
    end
end

function Base.iterate(cursor::MsmPrefilterCursor, state=nothing)
    while isopen(cursor)
        batch = next_batch!(cursor, 1)
        batch === nothing && return nothing
        isempty(batch) || return (batch[1], nothing)
    end
    nothing
end

"""Reduce filtered MSM candidate batches and always close the cursor."""
function reduce_batches!(function_value, initial,
    cursor::MsmPrefilterCursor; batch_size::Integer=DEFAULT_MATCH_BATCH)
    accumulator = initial
    try
        while (batch = next_batch!(cursor, batch_size)) !== nothing
            accumulator = function_value(accumulator, batch)
        end
        accumulator
    finally
        close!(cursor)
    end
end

function close!(cursor::MsmPrefilterCursor)
    lock(cursor.lock) do
        if !cursor.closed
            cursor.closed = true
            cursor.source = TemporalSeriesSource(TemporalEntry[], Val(:owned))
            cursor.query = Float64[]
        end
        nothing
    end
end
Base.close(cursor::MsmPrefilterCursor) = close!(cursor)
Base.isopen(cursor::MsmPrefilterCursor) = lock(cursor.lock) do
    !cursor.closed
end
cancel!(cursor::MsmPrefilterCursor) = close!(cursor)
