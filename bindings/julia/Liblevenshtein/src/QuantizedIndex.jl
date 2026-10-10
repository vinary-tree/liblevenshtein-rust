struct RawQuantizedIndexConfig
    quant_min::Float64
    quant_max::Float64
    quant_bins::UInt32
    reserved::UInt32
    max_entries::Csize_t
    max_total_samples::Csize_t
    max_series_len::Csize_t
end

struct RawQuantizedMatch
    id::UInt64
    edit_distance::Csize_t
end

"""Advisory quantized-byte edit candidate, with exact byte edit distance."""
struct QuantizedMatch
    id::UInt64
    edit_distance::Int
end

"""Mutable-then-frozen native quantized temporal candidate index.

Quantized edit candidates can include false positives and miss full-precision
temporal neighbors. They cannot prove absence under MSM, ERP, TWED, DTW, or
Fréchet. Stored series and queries follow the native quantizer's handling of
NaN and infinities. Construction has explicit source ceilings.
"""
mutable struct NativeQuantizedIndex
    handle::Ptr{Cvoid}
    max_series_len::Csize_t
    frozen::Bool
    closed::Bool
    lock::ReentrantLock
end

function NativeQuantizedIndex(; quant_min::Real, quant_max::Real,
    quant_bins::Integer=256, max_entries::Integer=100_000,
    max_total_samples::Integer=1_000_000,
    max_series_len::Integer=1_000_000)
    api_revision() >= UInt32(30) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_quantized_index_new,
            "native quantized candidate indexes require API revision 30"))
    bins = checked_threshold(quant_bins)
    1 <= bins <= 256 || throw(ArgumentError("quant_bins must be 1..256"))
    raw = RawQuantizedIndexConfig(Float64(quant_min), Float64(quant_max),
        UInt32(bins), 0, checked_threshold(max_entries),
        checked_threshold(max_total_samples), checked_threshold(max_series_len))
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(native(:llev_quantized_index_new), Cint,
        (Ref{RawQuantizedIndexConfig}, Ref{Ptr{Cvoid}}), Ref(raw), output)
    checked(status, :llev_quantized_index_new)
    index = NativeQuantizedIndex(output[], raw.max_series_len, false,
        false, ReentrantLock())
    finalizer(close!, index)
    index
end

function insert!(index::NativeQuantizedIndex, id::Integer,
    samples::AbstractVector{<:Real})
    0 <= id <= typemax(UInt64) || throw(ArgumentError("ID must fit UInt64"))
    length(samples) <= index.max_series_len ||
        throw(ArgumentError("quantized series exceeds max_series_len"))
    values = Vector{Float64}(samples)
    status = lock(index.lock) do
        index.closed && throw(ArgumentError("quantized index is closed"))
        index.frozen && throw(ArgumentError("quantized index is frozen"))
        GC.@preserve values ccall(native(:llev_quantized_index_insert),
            Cint, (Ptr{Cvoid}, UInt64, Ptr{Float64}, Csize_t),
            index.handle, UInt64(id),
            isempty(values) ? C_NULL : pointer(values), length(values))
    end
    checked(status, :llev_quantized_index_insert)
    index
end

function freeze!(index::NativeQuantizedIndex)
    lock(index.lock) do
        index.closed && throw(ArgumentError("quantized index is closed"))
        if !index.frozen
            status = ccall(native(:llev_quantized_index_freeze), Cint,
                (Ptr{Cvoid},), index.handle)
            checked(status, :llev_quantized_index_freeze)
            index.frozen = true
        end
        index
    end
end

function close!(index::NativeQuantizedIndex)
    lock(index.lock) do
        if !index.closed
            handle = index.handle
            index.handle = C_NULL
            index.closed = true
            handle == C_NULL || ccall(native(:llev_quantized_index_free),
                Cvoid, (Ptr{Cvoid},), handle)
        end
        nothing
    end
end
Base.close(index::NativeQuantizedIndex) = close!(index)
Base.isopen(index::NativeQuantizedIndex) = lock(index.lock) do
    !index.closed
end

mutable struct QuantizedCursor
    handle::Ptr{Cvoid}
    page_work_units::Csize_t
    page_results::Csize_t
    done::Bool
    closed::Bool
    lock::ReentrantLock
end

Base.IteratorSize(::Type{QuantizedCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{QuantizedCursor}) = Base.HasEltype()
Base.eltype(::Type{QuantizedCursor}) = QuantizedMatch

function quantized_algorithm(kind::Symbol)
    kind === :standard && return UInt32(0)
    kind === :transposition && return UInt32(1)
    kind === :merge_split && return UInt32(2)
    throw(ArgumentError("quantized algorithm must be :standard, :transposition, or :merge_split"))
end

"""Open a bounded lazy quantized-byte candidate query.

The cursor retains one frozen snapshot after the index closes. Page results
are exact for the byte edit query and advisory for original samples. An empty
complete cursor proves only that no quantized key matched. Resource exhaustion
raises `TemporalQueryIncomplete` without asserting a complete candidate set.
"""
function query_quantized(index::NativeQuantizedIndex,
    query::AbstractVector{<:Real}, max_distance::Integer;
    algorithm::Symbol=:standard,
    limits::TemporalSearchLimits=TemporalSearchLimits(),
    page_work_units::Integer=100_000, page_results::Integer=256)
    length(query) <= limits.max_series_len ||
        throw(ArgumentError("query exceeds max_series_len"))
    distance = checked_threshold(max_distance)
    work = checked_threshold(page_work_units)
    results = checked_threshold(page_results)
    work > 0 && 0 < results <= 65_536 ||
        throw(ArgumentError("quantized page limits must be positive and results <=65,536"))
    variant = quantized_algorithm(algorithm)
    values = Vector{Float64}(query)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    reason = Ref{UInt32}(0)
    status = lock(index.lock) do
        index.closed && throw(ArgumentError("quantized index is closed"))
        index.frozen || throw(ArgumentError("quantized index must be frozen"))
        GC.@preserve values ccall(native(:llev_quantized_index_query),
            Cint, (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Csize_t,
                UInt32, Ref{TemporalSearchLimits}, Ref{Ptr{Cvoid}}, Ref{UInt32}),
            index.handle, isempty(values) ? C_NULL : pointer(values),
            length(values), distance, variant, Ref(limits), output, reason)
    end
    status == Int32(STATUS_LIMIT_EXCEEDED) &&
        throw(TemporalQueryIncomplete(:quantized_candidates, nothing,
            native_index_reason(reason[])))
    checked(status, :llev_quantized_index_query)
    cursor = QuantizedCursor(output[], work, results, false, false,
        ReentrantLock())
    finalizer(close!, cursor)
    cursor
end

function next_batch!(cursor::QuantizedCursor,
    maximum::Integer=DEFAULT_MATCH_BATCH)
    lock(cursor.lock) do
        cursor.closed && return nothing
        if cursor.done
            close!(cursor)
            return nothing
        end
        0 < maximum <= 65_536 ||
            throw(ArgumentError("batch maximum must be 1..65,536"))
        raw = Vector{RawQuantizedMatch}(undef, min(maximum, cursor.page_results))
        written = Ref{Csize_t}(0)
        done = Ref{UInt8}(0)
        reason = Ref{UInt32}(0)
        try
            status = GC.@preserve raw ccall(
                native(:llev_quantized_cursor_next_batch), Cint,
                (Ptr{Cvoid}, Ptr{RawQuantizedMatch}, Csize_t, Csize_t,
                    Csize_t, Ref{Csize_t}, Ref{UInt8}, Ref{UInt32}),
                cursor.handle, pointer(raw), length(raw),
                cursor.page_work_units, cursor.page_results, written,
                done, reason)
            status == Int32(STATUS_LIMIT_EXCEEDED) &&
                throw(TemporalQueryIncomplete(:quantized_candidates,
                    nothing, native_index_reason(reason[])))
            checked(status, :llev_quantized_cursor_next_batch)
            if written[] > 0
                batch = [QuantizedMatch(raw[i].id,
                    Int(raw[i].edit_distance)) for i in 1:Int(written[])]
                cursor.done = done[] != 0
                return batch
            end
            done[] != 0 && (close!(cursor); return nothing)
            QuantizedMatch[]
        catch
            close!(cursor)
            rethrow()
        end
    end
end

"""Copy one full-precision source series from the cursor's frozen snapshot.

Use an identifier yielded by `next_batch!` before requesting the next page.
The copied vector stays valid after the cursor closes.
"""
function original_samples(cursor::QuantizedCursor, id::Integer)
    0 <= id <= typemax(UInt64) || throw(ArgumentError("ID must fit UInt64"))
    lock(cursor.lock) do
        cursor.closed && throw(ArgumentError("quantized cursor is closed"))
        count = Ref{Csize_t}(0)
        status = ccall(native(:llev_quantized_cursor_original), Cint,
            (Ptr{Cvoid}, UInt64, Ptr{Float64}, Csize_t, Ref{Csize_t}),
            cursor.handle, UInt64(id), C_NULL, 0, count)
        checked(status, :llev_quantized_cursor_original)
        samples = Vector{Float64}(undef, Int(count[]))
        status = GC.@preserve samples ccall(
            native(:llev_quantized_cursor_original), Cint,
            (Ptr{Cvoid}, UInt64, Ptr{Float64}, Csize_t, Ref{Csize_t}),
            cursor.handle, UInt64(id),
            isempty(samples) ? C_NULL : pointer(samples),
            length(samples), count)
        checked(status, :llev_quantized_cursor_original)
        samples
    end
end

function Base.iterate(cursor::QuantizedCursor,
    state=(QuantizedMatch[], 1))
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

function reduce_batches!(function_value, initial, cursor::QuantizedCursor;
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

function close!(cursor::QuantizedCursor)
    lock(cursor.lock) do
        if !cursor.closed
            handle = cursor.handle
            cursor.handle = C_NULL
            cursor.closed = true
            handle == C_NULL || ccall(native(:llev_quantized_cursor_free),
                Cvoid, (Ptr{Cvoid},), handle)
        end
        nothing
    end
end
Base.close(cursor::QuantizedCursor) = close!(cursor)
Base.isopen(cursor::QuantizedCursor) = lock(cursor.lock) do
    !cursor.closed
end
cancel!(cursor::QuantizedCursor) = close!(cursor)
