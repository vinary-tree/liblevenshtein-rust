struct RawTemporalIndexConfig
    temporal::RawTemporalConfig
    quant_min::Float64
    quant_max::Float64
    quant_bins::UInt32
    reserved::UInt32
    max_entries::Csize_t
    max_total_samples::Csize_t
    max_series_len::Csize_t
end

"""Cumulative native ceilings for one resumable indexed range query."""
struct TemporalSearchLimits
    max_series_len::Csize_t
    max_dp_cells::Csize_t
    max_work_units::Csize_t
    max_scratch_bytes::Csize_t
    max_trie_nodes::Csize_t
    max_trie_edges::Csize_t
    max_candidates::Csize_t
    max_results::Csize_t
    max_queue_entries::Csize_t
    max_continuation_bytes::Csize_t
end

function TemporalSearchLimits(; max_series_len::Integer=1_000_000,
    max_dp_cells::Integer=100_000_000, max_work_units::Integer=200_000_000,
    max_scratch_bytes::Integer=512 * 1024 * 1024,
    max_trie_nodes::Integer=10_000_000, max_trie_edges::Integer=20_000_000,
    max_candidates::Integer=1_000_000, max_results::Integer=100_000,
    max_queue_entries::Integer=1_000_000,
    max_continuation_bytes::Integer=64 * 1024 * 1024)
    TemporalSearchLimits(
        checked_threshold(max_series_len), checked_threshold(max_dp_cells),
        checked_threshold(max_work_units), checked_threshold(max_scratch_bytes),
        checked_threshold(max_trie_nodes), checked_threshold(max_trie_edges),
        checked_threshold(max_candidates), checked_threshold(max_results),
        checked_threshold(max_queue_entries),
        checked_threshold(max_continuation_bytes))
end

struct RawTemporalIndexMatch
    id::UInt64
    distance::Float64
end

function native_index_reason(code::UInt32)
    reasons = (:unknown, :dp_cells, :work_units, :scratch_bytes,
        :trie_nodes, :trie_edges, :candidates, :results, :queue_entries,
        :continuation_bytes, :overflow, :invalid_stored_data,
        :unsupported, :allocation, :page_work_units, :cancelled,
        :series_length)
    Int(code) + 1 <= length(reasons) ? reasons[Int(code) + 1] : :unknown
end

"""Native quantized temporal index with a frozen snapshot after construction."""
mutable struct TemporalIndex
    handle::Ptr{Cvoid}
    max_series_len::Csize_t
    frozen::Bool
    closed::Bool
    lock::ReentrantLock
end

"""Build a bounded native temporal index for MSM, ERP, TWED, DTW, or Fréchet.

The quantization interval must be finite and increasing, with 1–256 bins.
Stored samples are checked and copied by native construction. Freeze the index
before searching; active cursors keep its snapshot after the handle closes.
Insert, freeze, query creation, and close serialize on the index handle.
"""
function TemporalIndex(kind::Symbol; quant_min::Real, quant_max::Real,
    quant_bins::Integer=256, parameter0::Real=0.0, parameter1::Real=0.0,
    band::Integer=0, max_entries::Integer=100_000,
    max_total_samples::Integer=1_000_000,
    max_series_len::Integer=1_000_000)
    api_revision() >= UInt32(12) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED), :llev_temporal_index_new,
            "native temporal indexes require API revision 12"))
    bins = checked_threshold(quant_bins)
    1 <= bins <= 256 || throw(ArgumentError("quant_bins must be 1..256"))
    raw = RawTemporalIndexConfig(
        RawTemporalConfig(temporal_algorithm(kind), 0, Float64(parameter0),
            Float64(parameter1), checked_threshold(band), Inf),
        Float64(quant_min), Float64(quant_max), UInt32(bins), 0,
        checked_threshold(max_entries), checked_threshold(max_total_samples),
        checked_threshold(max_series_len))
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(native(:llev_temporal_index_new), Cint,
        (Ref{RawTemporalIndexConfig}, Ref{Ptr{Cvoid}}), Ref(raw), output)
    checked(status, :llev_temporal_index_new)
    index = TemporalIndex(output[], raw.max_series_len, false, false,
        ReentrantLock())
    finalizer(close!, index)
    index
end

function insert!(index::TemporalIndex, id::Integer, samples::AbstractVector{<:Real})
    0 <= id <= typemax(UInt64) || throw(ArgumentError("ID must fit UInt64"))
    length(samples) <= index.max_series_len ||
        throw(ArgumentError("temporal series exceeds max_series_len"))
    values = Vector{Float64}(samples)
    status = lock(index.lock) do
        index.closed && throw(ArgumentError("temporal index is closed"))
        index.frozen && throw(ArgumentError("temporal index is frozen"))
        GC.@preserve values ccall(native(:llev_temporal_index_insert), Cint,
            (Ptr{Cvoid}, UInt64, Ptr{Float64}, Csize_t), index.handle,
            UInt64(id), isempty(values) ? C_NULL : pointer(values),
            length(values))
    end
    checked(status, :llev_temporal_index_insert)
    index
end

"""Finish construction; later mutations are rejected."""
function freeze!(index::TemporalIndex)
    lock(index.lock) do
        index.closed && throw(ArgumentError("temporal index is closed"))
        if !index.frozen
            status = ccall(native(:llev_temporal_index_freeze), Cint,
                (Ptr{Cvoid},), index.handle)
            checked(status, :llev_temporal_index_freeze)
            index.frozen = true
        end
        index
    end
end

function close!(index::TemporalIndex)
    lock(index.lock) do
        if !index.closed
            handle = index.handle
            index.handle = C_NULL
            index.closed = true
            handle == C_NULL || ccall(native(:llev_temporal_index_free),
                Cvoid, (Ptr{Cvoid},), handle)
        end
        nothing
    end
end
Base.close(index::TemporalIndex) = close!(index)
Base.isopen(index::TemporalIndex) = lock(index.lock) do
    !index.closed
end

"""Lazy native page cursor over one frozen indexed temporal snapshot."""
mutable struct TemporalIndexCursor
    handle::Ptr{Cvoid}
    page_work_units::Csize_t
    page_results::Csize_t
    closed::Bool
    lock::ReentrantLock
end

Base.IteratorSize(::Type{TemporalIndexCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TemporalIndexCursor}) = Base.HasEltype()
Base.eltype(::Type{TemporalIndexCursor}) = TemporalMatch

"""Open a bounded, resumable exact native range query.

Each cursor page charges at most `page_work_units` work and accepts at most
`page_results` new matches. Session limits remain cumulative. A complete
empty result proves absence; exhausted cumulative limits raise
`TemporalQueryIncomplete` when the cursor advances beyond its exact subset.
Concurrent page reads and close serialize on the cursor handle.
"""
function query_index_range(index::TemporalIndex, query::AbstractVector{<:Real};
    cutoff::Real=Inf, limits::TemporalSearchLimits=TemporalSearchLimits(),
    page_work_units::Integer=100_000, page_results::Integer=256)
    length(query) <= limits.max_series_len ||
        throw(ArgumentError("query exceeds max_series_len"))
    work = checked_threshold(page_work_units)
    results = checked_threshold(page_results)
    work > 0 && results > 0 ||
        throw(ArgumentError("temporal page limits must be positive"))
    values = Vector{Float64}(query)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = lock(index.lock) do
        index.closed && throw(ArgumentError("temporal index is closed"))
        index.frozen || throw(ArgumentError("temporal index must be frozen"))
        GC.@preserve values ccall(native(:llev_temporal_index_query_range),
            Cint, (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Float64,
                Ref{TemporalSearchLimits}, Ref{Ptr{Cvoid}}),
            index.handle, isempty(values) ? C_NULL : pointer(values),
            length(values), Float64(cutoff), Ref(limits), output)
    end
    checked(status, :llev_temporal_index_query_range)
    cursor = TemporalIndexCursor(output[], work, results, false,
        ReentrantLock())
    finalizer(close!, cursor)
    cursor
end

"""Open the bounded canonical ERP automaton product on a frozen ERP index.

This uses reachable antichain states and verifies full-precision candidates.
The returned cursor has the same cumulative limits, page behavior, and
snapshot lifetime as `query_index_range`. A non-ERP index is rejected.
"""
function query_index_erp_automaton_range(index::TemporalIndex,
    query::AbstractVector{<:Real}; cutoff::Real=Inf,
    limits::TemporalSearchLimits=TemporalSearchLimits(),
    page_work_units::Integer=100_000, page_results::Integer=256)
    api_revision() >= UInt32(29) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_temporal_index_query_erp_automaton_range,
            "native ERP automaton range requires API revision 29"))
    length(query) <= limits.max_series_len ||
        throw(ArgumentError("query exceeds max_series_len"))
    work = checked_threshold(page_work_units)
    results = checked_threshold(page_results)
    work > 0 && results > 0 ||
        throw(ArgumentError("temporal page limits must be positive"))
    values = Vector{Float64}(query)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    reason = Ref{UInt32}(0)
    status = lock(index.lock) do
        index.closed && throw(ArgumentError("temporal index is closed"))
        index.frozen || throw(ArgumentError("temporal index must be frozen"))
        GC.@preserve values ccall(
            native(:llev_temporal_index_query_erp_automaton_range), Cint,
            (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Float64,
                Ref{TemporalSearchLimits}, Ref{Ptr{Cvoid}}, Ref{UInt32}),
            index.handle, isempty(values) ? C_NULL : pointer(values),
            length(values), Float64(cutoff), Ref(limits), output, reason)
    end
    status == Int32(STATUS_LIMIT_EXCEEDED) &&
        throw(TemporalQueryIncomplete(:native_erp_automaton, nothing,
            native_index_reason(reason[])))
    checked(status, :llev_temporal_index_query_erp_automaton_range)
    cursor = TemporalIndexCursor(output[], work, results, false,
        ReentrantLock())
    finalizer(close!, cursor)
    cursor
end

function next_batch!(cursor::TemporalIndexCursor, maximum::Integer=DEFAULT_MATCH_BATCH)
    lock(cursor.lock) do
        cursor.closed && return nothing
        0 < maximum <= 65_536 ||
            throw(ArgumentError("batch maximum must be 1..65,536"))
        raw = Vector{RawTemporalIndexMatch}(undef, maximum)
        written = Ref{Csize_t}(0)
        done = Ref{UInt8}(0)
        reason = Ref{UInt32}(0)
        try
            status = GC.@preserve raw ccall(
                native(:llev_temporal_index_cursor_next_batch), Cint,
                (Ptr{Cvoid}, Ptr{RawTemporalIndexMatch}, Csize_t, Csize_t,
                    Csize_t, Ref{Csize_t}, Ref{UInt8}, Ref{UInt32}),
                cursor.handle, pointer(raw), length(raw),
                cursor.page_work_units, cursor.page_results, written, done,
                reason)
            if status == Int32(STATUS_LIMIT_EXCEEDED)
                throw(TemporalQueryIncomplete(:native_index, nothing,
                    native_index_reason(reason[])))
            end
            checked(status, :llev_temporal_index_cursor_next_batch)
            if written[] > 0
                batch = [TemporalMatch(raw[i].id, raw[i].distance)
                    for i in 1:Int(written[])]
                done[] != 0 && close!(cursor)
                return batch
            end
            if done[] != 0
                close!(cursor)
                return nothing
            end
            TemporalMatch[]
        catch
            close!(cursor)
            rethrow()
        end
    end
end

function Base.iterate(cursor::TemporalIndexCursor, state=nothing)
    while isopen(cursor)
        batch = next_batch!(cursor, 1)
        batch === nothing && return nothing
        isempty(batch) || return (batch[1], nothing)
    end
    nothing
end

function reduce_batches!(function_value, initial, cursor::TemporalIndexCursor;
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

function close!(cursor::TemporalIndexCursor)
    lock(cursor.lock) do
        if !cursor.closed
            handle = cursor.handle
            cursor.handle = C_NULL
            cursor.closed = true
            handle == C_NULL ||
                ccall(native(:llev_temporal_index_cursor_free),
                    Cvoid, (Ptr{Cvoid},), handle)
        end
        nothing
    end
end
Base.close(cursor::TemporalIndexCursor) = close!(cursor)
Base.isopen(cursor::TemporalIndexCursor) = lock(cursor.lock) do
    !cursor.closed
end
cancel!(cursor::TemporalIndexCursor) = close!(cursor)

"""One complete exact top-k result, copied in bounded pages after native scan."""
mutable struct TemporalKnnCursor
    handle::Ptr{Cvoid}
    page_results::Csize_t
    closed::Bool
    lock::ReentrantLock
end

Base.IteratorSize(::Type{TemporalKnnCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TemporalKnnCursor}) = Base.HasEltype()
Base.eltype(::Type{TemporalKnnCursor}) = TemporalMatch

"""Search a frozen scalar index for exact k nearest neighbors.

Native search scans the full frozen source under cumulative limits before
returning a cursor. Incomplete work raises `TemporalQueryIncomplete` without
publishing a partial top-k. Result pages remain valid after the index closes.
"""
function query_index_knn(index::TemporalIndex,
    query::AbstractVector{<:Real}, k::Integer;
    limits::TemporalSearchLimits=TemporalSearchLimits(),
    page_results::Integer=DEFAULT_MATCH_BATCH)
    api_revision() >= UInt32(28) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_temporal_index_query_knn,
            "native temporal kNN requires API revision 28"))
    length(query) <= limits.max_series_len ||
        throw(ArgumentError("query exceeds max_series_len"))
    count = checked_threshold(k)
    page = checked_threshold(page_results)
    0 < page <= 65_536 ||
        throw(ArgumentError("kNN page results must be 1..65,536"))
    values = Vector{Float64}(query)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    reason = Ref{UInt32}(0)
    status = lock(index.lock) do
        index.closed && throw(ArgumentError("temporal index is closed"))
        index.frozen || throw(ArgumentError("temporal index must be frozen"))
        GC.@preserve values ccall(native(:llev_temporal_index_query_knn),
            Cint, (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Csize_t,
                Ref{TemporalSearchLimits}, Ref{Ptr{Cvoid}}, Ref{UInt32}),
            index.handle, isempty(values) ? C_NULL : pointer(values),
            length(values), count, Ref(limits), output, reason)
    end
    status == Int32(STATUS_LIMIT_EXCEEDED) &&
        throw(TemporalQueryIncomplete(:native_index_knn, nothing,
            native_index_reason(reason[])))
    checked(status, :llev_temporal_index_query_knn)
    cursor = TemporalKnnCursor(output[], page, false, ReentrantLock())
    finalizer(close!, cursor)
    cursor
end

function next_batch!(cursor::TemporalKnnCursor,
    maximum::Integer=DEFAULT_MATCH_BATCH)
    lock(cursor.lock) do
        cursor.closed && return nothing
        0 < maximum <= 65_536 ||
            throw(ArgumentError("batch maximum must be 1..65,536"))
        capacity = min(maximum, cursor.page_results)
        raw = Vector{RawTemporalIndexMatch}(undef, capacity)
        written = Ref{Csize_t}(0)
        done = Ref{UInt8}(0)
        try
            status = GC.@preserve raw ccall(
                native(:llev_temporal_knn_cursor_next_batch), Cint,
                (Ptr{Cvoid}, Ptr{RawTemporalIndexMatch}, Csize_t,
                    Ref{Csize_t}, Ref{UInt8}),
                cursor.handle, pointer(raw), length(raw), written, done)
            checked(status, :llev_temporal_knn_cursor_next_batch)
            if written[] > 0
                batch = [TemporalMatch(raw[i].id, raw[i].distance)
                    for i in 1:Int(written[])]
                done[] != 0 && close!(cursor)
                return batch
            end
            done[] != 0 && (close!(cursor); return nothing)
            error("native kNN cursor made no progress")
        catch
            close!(cursor)
            rethrow()
        end
    end
end

function Base.iterate(cursor::TemporalKnnCursor, state=nothing)
    batch = next_batch!(cursor, 1)
    batch === nothing ? nothing : (batch[1], nothing)
end

function reduce_batches!(function_value, initial, cursor::TemporalKnnCursor;
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

function close!(cursor::TemporalKnnCursor)
    lock(cursor.lock) do
        if !cursor.closed
            handle = cursor.handle
            cursor.handle = C_NULL
            cursor.closed = true
            handle == C_NULL || ccall(native(:llev_temporal_knn_cursor_free),
                Cvoid, (Ptr{Cvoid},), handle)
        end
        nothing
    end
end
Base.close(cursor::TemporalKnnCursor) = close!(cursor)
Base.isopen(cursor::TemporalKnnCursor) = lock(cursor.lock) do
    !cursor.closed
end
cancel!(cursor::TemporalKnnCursor) = close!(cursor)
