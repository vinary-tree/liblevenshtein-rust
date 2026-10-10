struct RawHybridIndexConfig
    source::RawQuantizedIndexConfig
    msm_cost::Float64
    trie_threshold_multiplier::Float64
    lower_bound_type::UInt32
    use_lower_bounds::UInt32
end

"""Verified MSM result from an advisory quantized candidate pipeline."""
struct HybridMsmMatch
    id::UInt64
    distance::Float64
end

"""Mutable-then-frozen native hybrid quantized-candidate / MSM index.

The MSM scores are exact for emitted candidates. Quantized candidate filtering
can miss true MSM neighbors. Optional prefix Euclidean, L1, and combined
heuristics can miss additional neighbors. Even a complete query cannot prove
full MSM recall or absence.
"""
mutable struct NativeHybridMsmIndex
    handle::Ptr{Cvoid}
    max_series_len::Csize_t
    frozen::Bool
    closed::Bool
    lock::ReentrantLock
end

function hybrid_bound_type(kind::Symbol)
    kind === :length && return UInt32(0)
    kind === :euclidean && return UInt32(1)
    kind === :l1 && return UInt32(2)
    kind === :combined && return UInt32(3)
    throw(ArgumentError("hybrid bound must be :length, :euclidean, :l1, or :combined"))
end

function NativeHybridMsmIndex(; quant_min::Real, quant_max::Real,
    quant_bins::Integer=256, msm_cost::Real=1.0,
    trie_threshold_multiplier::Real=2.0,
    lower_bound::Symbol=:length, use_lower_bounds::Bool=true,
    max_entries::Integer=100_000, max_total_samples::Integer=1_000_000,
    max_series_len::Integer=1_000_000)
    api_revision() >= UInt32(31) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_hybrid_index_new,
            "native hybrid MSM indexes require API revision 31"))
    bins = checked_threshold(quant_bins)
    1 <= bins <= 256 || throw(ArgumentError("quant_bins must be 1..256"))
    source = RawQuantizedIndexConfig(Float64(quant_min),
        Float64(quant_max), UInt32(bins), 0,
        checked_threshold(max_entries),
        checked_threshold(max_total_samples),
        checked_threshold(max_series_len))
    raw = RawHybridIndexConfig(source, Float64(msm_cost),
        Float64(trie_threshold_multiplier),
        hybrid_bound_type(lower_bound),
        UInt32(use_lower_bounds))
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(native(:llev_hybrid_index_new), Cint,
        (Ref{RawHybridIndexConfig}, Ref{Ptr{Cvoid}}), Ref(raw), output)
    checked(status, :llev_hybrid_index_new)
    index = NativeHybridMsmIndex(output[], source.max_series_len,
        false, false, ReentrantLock())
    finalizer(close!, index)
    index
end

function insert!(index::NativeHybridMsmIndex, id::Integer,
    samples::AbstractVector{<:Real})
    0 <= id <= typemax(UInt64) || throw(ArgumentError("ID must fit UInt64"))
    length(samples) <= index.max_series_len ||
        throw(ArgumentError("hybrid series exceeds max_series_len"))
    values = Vector{Float64}(samples)
    all(isfinite, values) ||
        throw(ArgumentError("hybrid MSM source series must be finite"))
    status = lock(index.lock) do
        index.closed && throw(ArgumentError("hybrid index is closed"))
        index.frozen && throw(ArgumentError("hybrid index is frozen"))
        GC.@preserve values ccall(native(:llev_hybrid_index_insert),
            Cint, (Ptr{Cvoid}, UInt64, Ptr{Float64}, Csize_t),
            index.handle, UInt64(id),
            isempty(values) ? C_NULL : pointer(values), length(values))
    end
    checked(status, :llev_hybrid_index_insert)
    index
end

function freeze!(index::NativeHybridMsmIndex)
    lock(index.lock) do
        index.closed && throw(ArgumentError("hybrid index is closed"))
        if !index.frozen
            status = ccall(native(:llev_hybrid_index_freeze), Cint,
                (Ptr{Cvoid},), index.handle)
            checked(status, :llev_hybrid_index_freeze)
            index.frozen = true
        end
        index
    end
end

function close!(index::NativeHybridMsmIndex)
    lock(index.lock) do
        if !index.closed
            handle = index.handle
            index.handle = C_NULL
            index.closed = true
            handle == C_NULL || ccall(native(:llev_hybrid_index_free),
                Cvoid, (Ptr{Cvoid},), handle)
        end
        nothing
    end
end
Base.close(index::NativeHybridMsmIndex) = close!(index)
Base.isopen(index::NativeHybridMsmIndex) = lock(index.lock) do
    !index.closed
end

mutable struct HybridMsmCursor
    handle::Ptr{Cvoid}
    page_work_units::Csize_t
    page_results::Csize_t
    done::Bool
    closed::Bool
    lock::ReentrantLock
end

Base.IteratorSize(::Type{HybridMsmCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{HybridMsmCursor}) = Base.HasEltype()
Base.eltype(::Type{HybridMsmCursor}) = HybridMsmMatch

"""Open a bounded lazy hybrid range query on one frozen native snapshot.

Results have exact MSM scores but follow source bucket order. A complete
cursor proves only completion of the configured advisory candidate pipeline.
On exhaustion, `TemporalQueryIncomplete` never asserts a complete set.
"""
function query_hybrid_msm_range(index::NativeHybridMsmIndex,
    query::AbstractVector{<:Real}; cutoff::Real,
    limits::TemporalSearchLimits=TemporalSearchLimits(),
    page_work_units::Integer=100_000, page_results::Integer=256)
    length(query) <= limits.max_series_len ||
        throw(ArgumentError("query exceeds max_series_len"))
    work = checked_threshold(page_work_units)
    results = checked_threshold(page_results)
    work > 0 && 0 < results <= 65_536 ||
        throw(ArgumentError("hybrid page limits must be positive and results <=65,536"))
    bound = Float64(cutoff)
    !isnan(bound) && bound >= 0 ||
        throw(ArgumentError("hybrid MSM cutoff must be nonnegative"))
    values = Vector{Float64}(query)
    all(isfinite, values) ||
        throw(ArgumentError("hybrid MSM query must be finite"))
    output = Ref{Ptr{Cvoid}}(C_NULL)
    reason = Ref{UInt32}(0)
    status = lock(index.lock) do
        index.closed && throw(ArgumentError("hybrid index is closed"))
        index.frozen || throw(ArgumentError("hybrid index must be frozen"))
        GC.@preserve values ccall(native(:llev_hybrid_index_query_range),
            Cint, (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Float64,
                Ref{TemporalSearchLimits}, Ref{Ptr{Cvoid}}, Ref{UInt32}),
            index.handle, isempty(values) ? C_NULL : pointer(values),
            length(values), bound, Ref(limits), output, reason)
    end
    status == Int32(STATUS_LIMIT_EXCEEDED) &&
        throw(TemporalQueryIncomplete(:hybrid_msm, nothing,
            native_index_reason(reason[])))
    checked(status, :llev_hybrid_index_query_range)
    cursor = HybridMsmCursor(output[], work, results, false, false,
        ReentrantLock())
    finalizer(close!, cursor)
    cursor
end

"""Return bounded advisory hybrid MSM nearest neighbors.

The native threshold-expanding scan completes before this result cursor is
returned. It retains at most `k` exact MSM scores. Resource exhaustion
raises `TemporalQueryIncomplete` without publishing partial neighbors.
Quantized filtering and optional heuristics still prevent a recall proof.
"""
function query_hybrid_msm_knn(index::NativeHybridMsmIndex,
    query::AbstractVector{<:Real}, k::Integer;
    initial_threshold::Real=0.0,
    limits::TemporalSearchLimits=TemporalSearchLimits(),
    page_work_units::Integer=100_000, page_results::Integer=256)
    length(query) <= limits.max_series_len ||
        throw(ArgumentError("query exceeds max_series_len"))
    neighbors = checked_threshold(k)
    work = checked_threshold(page_work_units)
    results = checked_threshold(page_results)
    work > 0 && 0 < results <= 65_536 ||
        throw(ArgumentError("hybrid page limits must be positive and results <=65,536"))
    values = Vector{Float64}(query)
    all(isfinite, values) ||
        throw(ArgumentError("hybrid MSM query must be finite"))
    output = Ref{Ptr{Cvoid}}(C_NULL)
    reason = Ref{UInt32}(0)
    status = lock(index.lock) do
        index.closed && throw(ArgumentError("hybrid index is closed"))
        index.frozen || throw(ArgumentError("hybrid index must be frozen"))
        GC.@preserve values ccall(native(:llev_hybrid_index_query_knn),
            Cint, (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Csize_t, Float64,
                Ref{TemporalSearchLimits}, Ref{Ptr{Cvoid}}, Ref{UInt32}),
            index.handle, isempty(values) ? C_NULL : pointer(values),
            length(values), neighbors, Float64(initial_threshold),
            Ref(limits), output, reason)
    end
    status == Int32(STATUS_LIMIT_EXCEEDED) &&
        throw(TemporalQueryIncomplete(:hybrid_msm_knn, nothing,
            native_index_reason(reason[])))
    checked(status, :llev_hybrid_index_query_knn)
    cursor = HybridMsmCursor(output[], work, results, false, false,
        ReentrantLock())
    finalizer(close!, cursor)
    cursor
end

function next_batch!(cursor::HybridMsmCursor,
    maximum::Integer=DEFAULT_MATCH_BATCH)
    lock(cursor.lock) do
        cursor.closed && return nothing
        if cursor.done
            close!(cursor)
            return nothing
        end
        0 < maximum <= 65_536 ||
            throw(ArgumentError("batch maximum must be 1..65,536"))
        raw = Vector{RawTemporalIndexMatch}(undef,
            min(maximum, cursor.page_results))
        written = Ref{Csize_t}(0)
        done = Ref{UInt8}(0)
        reason = Ref{UInt32}(0)
        try
            status = GC.@preserve raw ccall(
                native(:llev_hybrid_cursor_next_batch), Cint,
                (Ptr{Cvoid}, Ptr{RawTemporalIndexMatch}, Csize_t, Csize_t,
                    Csize_t, Ref{Csize_t}, Ref{UInt8}, Ref{UInt32}),
                cursor.handle, pointer(raw), length(raw),
                cursor.page_work_units, cursor.page_results,
                written, done, reason)
            status == Int32(STATUS_LIMIT_EXCEEDED) &&
                throw(TemporalQueryIncomplete(:hybrid_msm, nothing,
                    native_index_reason(reason[])))
            checked(status, :llev_hybrid_cursor_next_batch)
            if written[] > 0
                cursor.done = done[] != 0
                return [HybridMsmMatch(raw[i].id, raw[i].distance)
                    for i in 1:Int(written[])]
            end
            done[] != 0 && (close!(cursor); return nothing)
            HybridMsmMatch[]
        catch
            close!(cursor)
            rethrow()
        end
    end
end

function Base.iterate(cursor::HybridMsmCursor,
    state=(HybridMsmMatch[], 1))
    while isopen(cursor)
        batch, position = state
        position <= length(batch) &&
            return (batch[position], (batch, position + 1))
        batch = next_batch!(cursor, DEFAULT_MATCH_BATCH)
        batch === nothing && return nothing
        state = (batch, 1)
    end
    nothing
end

function reduce_batches!(function_value, initial, cursor::HybridMsmCursor;
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

function close!(cursor::HybridMsmCursor)
    lock(cursor.lock) do
        if !cursor.closed
            handle = cursor.handle
            cursor.handle = C_NULL
            cursor.closed = true
            handle == C_NULL || ccall(native(:llev_hybrid_cursor_free),
                Cvoid, (Ptr{Cvoid},), handle)
        end
        nothing
    end
end
Base.close(cursor::HybridMsmCursor) = close!(cursor)
Base.isopen(cursor::HybridMsmCursor) = lock(cursor.lock) do
    !cursor.closed
end
cancel!(cursor::HybridMsmCursor) = close!(cursor)
