"""One owned series and its unsigned source ID in a temporal snapshot."""
struct TemporalEntry
    id::UInt64
    samples::Vector{Float64}
end

"""A bounded, copied source snapshot for lazy temporal range queries.

Construct from an iterable of `(id, samples)` pairs. IDs are unique unsigned
integers; samples are finite real vectors. Source construction is bounded by
entry count, per-series length, and total sample storage. Later changes to
the caller's vectors cannot alter an active query.
"""
struct TemporalSeriesSource
    entries::Vector{TemporalEntry}
    TemporalSeriesSource(entries::Vector{TemporalEntry}, ::Val{:owned}) = new(entries)
end

function TemporalSeriesSource(entries; max_entries::Integer=100_000,
    max_series_len::Integer=1_000_000,
    max_source_bytes::Integer=64 * 1024 * 1024)
    entry_limit = checked_threshold(max_entries)
    series_limit = checked_threshold(max_series_len)
    byte_limit = checked_threshold(max_source_bytes)
    stored = TemporalEntry[]
    seen = Set{UInt64}()
    bytes = UInt(0)
    for (id, samples) in entries
        length(stored) < entry_limit ||
            throw(ArgumentError("temporal source exceeds max_entries"))
        id isa Integer || throw(ArgumentError("temporal IDs must be integers"))
        0 <= id <= typemax(UInt64) ||
            throw(ArgumentError("temporal ID must fit UInt64"))
        key = UInt64(id)
        key in seen && throw(ArgumentError("duplicate temporal ID $key"))
        samples isa AbstractVector{<:Real} ||
            throw(ArgumentError("temporal samples must be real vectors"))
        length(samples) <= series_limit ||
            throw(ArgumentError("temporal series exceeds max_series_len"))
        length(samples) <= (byte_limit - bytes) ÷ sizeof(Float64) ||
            throw(ArgumentError("temporal source exceeds max_source_bytes"))
        copied = Vector{Float64}(samples)
        all(isfinite, copied) ||
            throw(ArgumentError("temporal source samples must be finite"))
        bytes += UInt(length(copied)) * UInt(sizeof(Float64))
        push!(stored, TemporalEntry(key, copied))
        push!(seen, key)
    end
    TemporalSeriesSource(stored, Val(:owned))
end

Base.length(source::TemporalSeriesSource) = length(source.entries)
Base.isempty(source::TemporalSeriesSource) = isempty(source.entries)

"""Hard cumulative ceilings for one lazy temporal range query."""
struct TemporalQueryLimits
    max_candidates::Csize_t
    max_results::Csize_t
    max_total_dp_cells::Csize_t
end

TemporalQueryLimits(; max_candidates::Integer=100_000,
    max_results::Integer=100_000,
    max_total_dp_cells::Integer=100_000_000) =
    TemporalQueryLimits(checked_threshold(max_candidates),
        checked_threshold(max_results), checked_threshold(max_total_dp_cells))

"""An exact native temporal score for one source ID."""
struct TemporalMatch
    id::UInt64
    distance::Float64
end

"""A query stopped by a declared hard limit before exhaustion was proved."""
struct TemporalQueryIncomplete <: Exception
    reason::Symbol
    candidate_id::Union{Nothing,UInt64}
    detail::Union{Nothing,Symbol}
end

function Base.showerror(io::IO, error::TemporalQueryIncomplete)
    print(io, "temporal range query incomplete: ", error.reason)
    error.candidate_id === nothing || print(io, " at source ID ", error.candidate_id)
    error.detail === nothing || print(io, " (", error.detail, ")")
end

"""Closeable lazy scan of one copied temporal source snapshot."""
mutable struct TemporalRangeCursor
    source::TemporalSeriesSource
    query::Vector{Float64}
    algorithm::Symbol
    parameter0::Float64
    parameter1::Float64
    band::Csize_t
    cutoff::Float64
    limits::TemporalLimits
    query_limits::TemporalQueryLimits
    next_index::Int
    visited::Csize_t
    emitted::Csize_t
    dp_cells::Csize_t
    closed::Bool
end

Base.IteratorSize(::Type{TemporalRangeCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TemporalRangeCursor}) = Base.HasEltype()
Base.eltype(::Type{TemporalRangeCursor}) = TemporalMatch

"""Search a copied temporal source lazily with native scalar kernels.

Each candidate is compared only when the cursor advances. Per-comparison
`TemporalLimits` and cumulative `TemporalQueryLimits` bound the operation.
Results preserve source order. An exhausted budget raises
`TemporalQueryIncomplete`; it never appears as an empty complete result.
"""
function query_temporal_range(source::TemporalSeriesSource, kind::Symbol,
    query::AbstractVector{<:Real}; parameter0::Real=0.0,
    parameter1::Real=0.0, band::Integer=0, cutoff::Real=Inf,
    limits::TemporalLimits=TemporalLimits(),
    query_limits::TemporalQueryLimits=TemporalQueryLimits())
    length(query) <= limits.max_series_len ||
        throw(ArgumentError("query exceeds max_series_len"))
    q = Vector{Float64}(query)
    all(isfinite, q) ||
        throw(ArgumentError("query samples must be finite"))
    p0 = Float64(parameter0)
    p1 = Float64(parameter1)
    width = checked_threshold(band)
    tau = Float64(cutoff)
    temporal_distance(kind, Float64[], Float64[];
        parameter0=p0, parameter1=p1, band=width, cutoff=tau, limits)
    TemporalRangeCursor(source, q, kind, p0, p1, width, tau, limits,
        query_limits, 1, 0, 0, 0, false)
end

function next_temporal_match!(cursor::TemporalRangeCursor)
    cursor.closed && throw(ArgumentError("temporal cursor is closed"))
    while cursor.next_index <= length(cursor.source.entries)
        cursor.visited < cursor.query_limits.max_candidates ||
            throw(TemporalQueryIncomplete(:candidates, nothing, nothing))
        entry = cursor.source.entries[cursor.next_index]
        cells = UInt(length(cursor.query)) * UInt(length(entry.samples))
        cells <= cursor.query_limits.max_total_dp_cells - cursor.dp_cells ||
            throw(TemporalQueryIncomplete(:dp_cells, entry.id, nothing))
        cursor.next_index += 1
        cursor.visited += 1
        cursor.dp_cells += cells
        outcome = temporal_distance(cursor.algorithm, cursor.query, entry.samples;
            parameter0=cursor.parameter0, parameter1=cursor.parameter1,
            band=cursor.band, cutoff=cursor.cutoff, limits=cursor.limits)
        if outcome.kind === :incomplete
            throw(TemporalQueryIncomplete(:kernel, entry.id, outcome.reason))
        elseif outcome.kind === :finite
            cursor.emitted < cursor.query_limits.max_results ||
                throw(TemporalQueryIncomplete(:results, entry.id, nothing))
            cursor.emitted += 1
            return TemporalMatch(entry.id, outcome.value::Float64)
        end
    end
    nothing
end

function next_batch!(cursor::TemporalRangeCursor, maximum::Integer=DEFAULT_MATCH_BATCH)
    cursor.closed && return nothing
    0 < maximum <= typemax(Int) ||
        throw(ArgumentError("batch maximum must be positive and fit Int"))
    batch = TemporalMatch[]
    try
        while length(batch) < maximum
            match = next_temporal_match!(cursor)
            if match === nothing
                close!(cursor)
                return isempty(batch) ? nothing : batch
            end
            push!(batch, match)
        end
        batch
    catch
        close!(cursor)
        rethrow()
    end
end

function Base.iterate(cursor::TemporalRangeCursor, state=nothing)
    cursor.closed && return nothing
    batch = next_batch!(cursor, 1)
    batch === nothing ? nothing : (batch[1], nothing)
end

"""Reduce temporal result batches and close the cursor on every exit path."""
function reduce_batches!(function_value, initial, cursor::TemporalRangeCursor;
    batch_size::Integer=DEFAULT_MATCH_BATCH)
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

function close!(cursor::TemporalRangeCursor)
    cursor.closed && return nothing
    cursor.closed = true
    cursor.source = TemporalSeriesSource(TemporalEntry[], Val(:owned))
    cursor.query = Float64[]
    nothing
end
Base.close(cursor::TemporalRangeCursor) = close!(cursor)
Base.isopen(cursor::TemporalRangeCursor) = !cursor.closed
cancel!(cursor::TemporalRangeCursor) = close!(cursor)
