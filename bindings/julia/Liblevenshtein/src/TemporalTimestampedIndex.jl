struct RawTimestampedTwedIndexConfig
    unit::UInt32
    reserved::UInt32
    origin::Float64
    value_min::Float64
    value_max::Float64
    time_min::Float64
    time_max::Float64
    value_bins::UInt32
    time_bins::UInt32
    stiffness::Float64
    gap_penalty::Float64
    max_entries::Csize_t
    max_total_samples::Csize_t
    max_series_len::Csize_t
end

"""Cumulative native ceilings for physical-time TWED range products."""
struct TimestampedTwedSearchLimits
    common::TemporalSearchLimits
    max_product_states::Csize_t
    max_product_positions::Csize_t
    max_transition_cache_entries::Csize_t
end

TimestampedTwedSearchLimits(;
    common::TemporalSearchLimits=TemporalSearchLimits(),
    max_product_states::Integer=1_000_000,
    max_product_positions::Integer=8_000_000,
    max_transition_cache_entries::Integer=2_000_000) =
    TimestampedTwedSearchLimits(common,
        checked_threshold(max_product_states),
        checked_threshold(max_product_positions),
        checked_threshold(max_transition_cache_entries))

struct RawTimestampedTwedMatch
    id::UInt64
    episode_id::UInt64
    distance::Float64
end

"""One exact physical-time match with caller ID and stable episode ID."""
struct TimestampedTwedMatch
    id::UInt64
    episode_id::UInt64
    distance::Float64
end

"""Freezable native index over one canonical physical-time TWED domain."""
mutable struct TimestampedTwedIndex
    handle::Ptr{Cvoid}
    unit::Symbol
    origin::Float64
    max_series_len::Csize_t
    frozen::Bool
    closed::Bool
end

function TimestampedTwedIndex(config::MetricTimestampedTwedConfig;
    unit::Symbol=:seconds, origin::Real=0.0,
    value_min::Real, value_max::Real,
    time_min::Real, time_max::Real,
    value_bins::Integer=256, time_bins::Integer=256,
    max_entries::Integer=100_000,
    max_total_samples::Integer=1_000_000,
    max_series_len::Integer=1_000_000)
    api_revision() >= UInt32(17) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_timestamped_twed_index_new,
            "timestamped TWED indexes require native API revision 17"))
    vb = checked_threshold(value_bins)
    tb = checked_threshold(time_bins)
    1 <= vb <= 2^31 && 1 <= tb <= 2^31 ||
        throw(ArgumentError("timestamped bin counts must be in 1..2^31"))
    raw = RawTimestampedTwedIndexConfig(timestamp_unit_code(unit), 0,
        Float64(origin), Float64(value_min), Float64(value_max),
        Float64(time_min), Float64(time_max), UInt32(vb), UInt32(tb),
        config.stiffness, config.gap_penalty,
        checked_threshold(max_entries), checked_threshold(max_total_samples),
        checked_threshold(max_series_len))
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(native(:llev_timestamped_twed_index_new), Cint,
        (Ref{RawTimestampedTwedIndexConfig}, Ref{Ptr{Cvoid}}),
        Ref(raw), output)
    checked(status, :llev_timestamped_twed_index_new)
    index = TimestampedTwedIndex(output[], unit, raw.origin,
        raw.max_series_len, false, false)
    finalizer(close!, index)
    index
end

function timestamped_index_identity(index::TimestampedTwedIndex,
    series::TimestampedSeries)
    index.unit === series.unit ||
        throw(ArgumentError("timestamped index and series units differ"))
    reinterpret(UInt64, index.origin) ==
        reinterpret(UInt64, series.origin) ||
        throw(ArgumentError("timestamped index and series origins differ"))
    nothing
end

"""Insert a copied episode and return its stable native episode ID."""
function insert_episode!(index::TimestampedTwedIndex, id::Integer,
    series::TimestampedSeries)
    index.closed && throw(ArgumentError("timestamped index is closed"))
    index.frozen && throw(ArgumentError("timestamped index is frozen"))
    0 <= id <= typemax(UInt64) || throw(ArgumentError("ID must fit UInt64"))
    length(getfield(series, :_values)) <= index.max_series_len ||
        throw(ArgumentError("timestamped episode exceeds max_series_len"))
    timestamped_index_identity(index, series)
    values = getfield(series, :_values)
    times = getfield(series, :_timestamps)
    view = RawTimestampedSeriesView(pointer(values), pointer(times),
        length(values), timestamp_unit_code(series.unit), 0, series.origin)
    episode = Ref{UInt64}(0)
    status = GC.@preserve values times ccall(
        native(:llev_timestamped_twed_index_insert), Cint,
        (Ptr{Cvoid}, UInt64, Ref{RawTimestampedSeriesView}, Ref{UInt64}),
        index.handle, UInt64(id), Ref(view), episode)
    checked(status, :llev_timestamped_twed_index_insert)
    episode[]
end

function insert!(index::TimestampedTwedIndex, id::Integer,
    series::TimestampedSeries)
    insert_episode!(index, id, series)
    index
end

function freeze!(index::TimestampedTwedIndex)
    index.closed && throw(ArgumentError("timestamped index is closed"))
    index.frozen && return index
    status = ccall(native(:llev_timestamped_twed_index_freeze), Cint,
        (Ptr{Cvoid},), index.handle)
    checked(status, :llev_timestamped_twed_index_freeze)
    index.frozen = true
    index
end

function close!(index::TimestampedTwedIndex)
    index.closed && return nothing
    handle = index.handle
    index.handle = C_NULL
    index.closed = true
    handle == C_NULL || ccall(native(:llev_timestamped_twed_index_free),
        Cvoid, (Ptr{Cvoid},), handle)
    nothing
end
Base.close(index::TimestampedTwedIndex) = close!(index)
Base.isopen(index::TimestampedTwedIndex) = !index.closed

"""Lazy exact cursor retaining a frozen physical-time index revision."""
mutable struct TimestampedTwedIndexCursor
    handle::Ptr{Cvoid}
    page_work_units::Csize_t
    page_results::Csize_t
    closed::Bool
end

Base.IteratorSize(::Type{TimestampedTwedIndexCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TimestampedTwedIndexCursor}) = Base.HasEltype()
Base.eltype(::Type{TimestampedTwedIndexCursor}) = TimestampedTwedMatch

function query_metric_range(index::TimestampedTwedIndex,
    series::TimestampedSeries; cutoff::Real=Inf,
    limits::TimestampedTwedSearchLimits=TimestampedTwedSearchLimits(),
    page_work_units::Integer=100_000, page_results::Integer=256)
    index.closed && throw(ArgumentError("timestamped index is closed"))
    index.frozen || throw(ArgumentError("timestamped index must be frozen"))
    timestamped_index_identity(index, series)
    length(getfield(series, :_values)) <= limits.common.max_series_len ||
        throw(ArgumentError("timestamped query exceeds max_series_len"))
    work = checked_threshold(page_work_units)
    results = checked_threshold(page_results)
    work > 0 && results > 0 ||
        throw(ArgumentError("timestamped page limits must be positive"))
    values = getfield(series, :_values)
    times = getfield(series, :_timestamps)
    view = RawTimestampedSeriesView(pointer(values), pointer(times),
        length(values), timestamp_unit_code(series.unit), 0, series.origin)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve values times ccall(
        native(:llev_timestamped_twed_index_query_range), Cint,
        (Ptr{Cvoid}, Ref{RawTimestampedSeriesView}, Float64,
            Ref{TimestampedTwedSearchLimits}, Ref{Ptr{Cvoid}}),
        index.handle, Ref(view), Float64(cutoff), Ref(limits), output)
    checked(status, :llev_timestamped_twed_index_query_range)
    cursor = TimestampedTwedIndexCursor(output[], work, results, false)
    finalizer(close!, cursor)
    cursor
end

function next_batch!(cursor::TimestampedTwedIndexCursor,
    maximum::Integer=DEFAULT_MATCH_BATCH)
    cursor.closed && return nothing
    0 < maximum <= 65_536 ||
        throw(ArgumentError("batch maximum must be 1..65,536"))
    raw = Vector{RawTimestampedTwedMatch}(undef, maximum)
    written = Ref{Csize_t}(0)
    done = Ref{UInt8}(0)
    reason = Ref{UInt32}(0)
    try
        status = GC.@preserve raw ccall(
            native(:llev_timestamped_twed_cursor_next_batch), Cint,
            (Ptr{Cvoid}, Ptr{RawTimestampedTwedMatch}, Csize_t,
                Csize_t, Csize_t, Ref{Csize_t}, Ref{UInt8}, Ref{UInt32}),
            cursor.handle, pointer(raw), length(raw), cursor.page_work_units,
            cursor.page_results, written, done, reason)
        if status == Int32(STATUS_LIMIT_EXCEEDED)
            throw(TemporalQueryIncomplete(:timestamped_twed_index, nothing,
                native_index_reason(reason[])))
        end
        checked(status, :llev_timestamped_twed_cursor_next_batch)
        if written[] > 0
            batch = [TimestampedTwedMatch(raw[i].id, raw[i].episode_id,
                raw[i].distance) for i in 1:Int(written[])]
            done[] != 0 && close!(cursor)
            return batch
        end
        if done[] != 0
            close!(cursor)
            return nothing
        end
        TimestampedTwedMatch[]
    catch
        close!(cursor)
        rethrow()
    end
end

function Base.iterate(cursor::TimestampedTwedIndexCursor, state=nothing)
    while !cursor.closed
        batch = next_batch!(cursor, 1)
        batch === nothing && return nothing
        isempty(batch) || return (batch[1], nothing)
    end
    nothing
end

function reduce_batches!(function_value, initial,
    cursor::TimestampedTwedIndexCursor;
    batch_size::Integer=DEFAULT_MATCH_BATCH)
    accumulator = initial
    try
        while (batch = next_batch!(cursor, batch_size)) !== nothing
            isempty(batch) && continue
            accumulator = function_value(accumulator, batch)
        end
        accumulator
    finally
        close!(cursor)
    end
end

function close!(cursor::TimestampedTwedIndexCursor)
    cursor.closed && return nothing
    handle = cursor.handle
    cursor.handle = C_NULL
    cursor.closed = true
    handle == C_NULL || ccall(native(:llev_timestamped_twed_cursor_free),
        Cvoid, (Ptr{Cvoid},), handle)
    nothing
end
Base.close(cursor::TimestampedTwedIndexCursor) = close!(cursor)
Base.isopen(cursor::TimestampedTwedIndexCursor) = !cursor.closed
cancel!(cursor::TimestampedTwedIndexCursor) = close!(cursor)
