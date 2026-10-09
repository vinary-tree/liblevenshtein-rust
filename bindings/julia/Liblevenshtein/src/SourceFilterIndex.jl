struct RawSourceFilterIndexConfig
    mode::UInt32
    reserved::UInt32
    ngram_size::Csize_t
    jaro_threshold::Float64
    max_terms::Csize_t
    max_term_bytes::Csize_t
    max_source_bytes::Csize_t
    max_query_bytes::Csize_t
end

"""Frozen native postings index over a copied, deduplicated Unicode source.

`:ngram` returns the native conservative candidate set. `:hybrid` additionally
applies native Jaro-Winkler refinement. The index owns all native term bytes;
query cursors copy insertion IDs and retain the Julia source snapshot.
"""
mutable struct NativeSourceFilterIndex
    handle::Ptr{Cvoid}
    source::SourceFilterSource
    mode::Symbol
    max_query_bytes::Csize_t
    closed::Bool
end

function NativeSourceFilterIndex(terms; mode::Symbol=:ngram,
    ngram_size::Integer=2, jaro_threshold::Union{Nothing,Real}=nothing,
    max_terms::Integer=100_000, max_term_bytes::Integer=4096,
    max_source_bytes::Integer=64 * 1024 * 1024,
    max_query_bytes::Integer=4096)
    api_revision() >= UInt32(19) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_source_filter_index_new,
            "persistent source-filter indexes require native API revision 19"))
    mode in (:ngram, :hybrid) ||
        throw(ArgumentError("source-filter mode must be :ngram or :hybrid"))
    n = checked_threshold(ngram_size)
    n > 0 || throw(ArgumentError("ngram_size must be positive"))
    threshold = jaro_threshold === nothing ?
        (mode === :ngram ? 0.0 : 0.7) : Float64(jaro_threshold)
    isfinite(threshold) && 0.0 <= threshold <= 1.0 ||
        throw(ArgumentError("Jaro threshold must be finite in [0,1]"))
    mode === :ngram && threshold != 0.0 &&
        throw(ArgumentError("Jaro threshold must be zero for :ngram mode"))
    term_limit = checked_threshold(max_terms)
    per_term = checked_threshold(max_term_bytes)
    source_limit = checked_threshold(max_source_bytes)
    query_limit = checked_threshold(max_query_bytes)
    input_terms = terms isa SourceFilterSource ? terms.terms : terms
    source = SourceFilterSource(input_terms; max_terms=term_limit,
        max_term_bytes=per_term, max_source_bytes=source_limit)
    raw = RawSourceFilterIndexConfig(mode === :ngram ? UInt32(1) : UInt32(2),
        0, n, threshold, term_limit, per_term, source_limit, query_limit)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(native(:llev_source_filter_index_new), Cint,
        (Ref{RawSourceFilterIndexConfig}, Ref{Ptr{Cvoid}}),
        Ref(raw), output)
    checked(status, :llev_source_filter_index_new)
    index = NativeSourceFilterIndex(output[], source, mode, query_limit, false)
    finalizer(close!, index)
    try
        for (position, term) in enumerate(source.terms)
            bytes = text_bytes(term)
            id = Ref{Csize_t}(0)
            status = GC.@preserve bytes ccall(
                native(:llev_source_filter_index_insert), Cint,
                (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{Csize_t}),
                index.handle, isempty(bytes) ? C_NULL : pointer(bytes),
                length(bytes), id)
            checked(status, :llev_source_filter_index_insert)
            id[] == position - 1 ||
                error("native source-filter insertion ID differs from source order")
        end
        status = ccall(native(:llev_source_filter_index_freeze), Cint,
            (Ptr{Cvoid},), index.handle)
        checked(status, :llev_source_filter_index_freeze)
        index
    catch
        close!(index)
        rethrow()
    end
end

Base.length(index::NativeSourceFilterIndex) = length(index.source)
Base.isempty(index::NativeSourceFilterIndex) = isempty(index.source)

function close!(index::NativeSourceFilterIndex)
    index.closed && return nothing
    handle = index.handle
    index.handle = C_NULL
    index.closed = true
    handle == C_NULL || ccall(native(:llev_source_filter_index_free),
        Cvoid, (Ptr{Cvoid},), handle)
    nothing
end
Base.close(index::NativeSourceFilterIndex) = close!(index)
Base.isopen(index::NativeSourceFilterIndex) = !index.closed

"""Closeable page iterator over a copied native candidate-ID snapshot."""
mutable struct NativeSourceFilterCursor
    source::SourceFilterSource
    ids::Vector{Csize_t}
    next_index::Int
    page_results::Csize_t
    closed::Bool
end

Base.IteratorSize(::Type{NativeSourceFilterCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{NativeSourceFilterCursor}) = Base.HasEltype()
Base.eltype(::Type{NativeSourceFilterCursor}) = String

"""Open a bounded native postings-index query with lazy result iteration."""
function query_filter_index(index::NativeSourceFilterIndex,
    input::AbstractString, max_distance::Integer;
    limits::SourceFilterLimits=SourceFilterLimits(),
    page_results::Integer=256)
    index.closed && throw(ArgumentError("source-filter index is closed"))
    ncodeunits(input) <= index.max_query_bytes ||
        throw(SourceFilterIncomplete(:query_bytes))
    page = checked_threshold(page_results)
    page > 0 || throw(ArgumentError("page_results must be positive"))
    query = text_bytes(input)
    distance = checked_threshold(max_distance)
    capacity = min(length(index.source), limits.max_results)
    ids = Vector{Csize_t}(undef, capacity)
    output = isempty(ids) ? Ptr{Csize_t}(C_NULL) : pointer(ids)
    written = Ref{Csize_t}(0)
    reason = Ref{UInt32}(0)
    status = GC.@preserve query ids ccall(
        native(:llev_source_filter_index_query), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Csize_t,
            Ref{SourceFilterLimits}, Ptr{Csize_t}, Csize_t,
            Ref{Csize_t}, Ref{UInt32}),
        index.handle, isempty(query) ? C_NULL : pointer(query),
        length(query), distance,
        Ref(limits), output, capacity, written, reason)
    if status == Int32(STATUS_LIMIT_EXCEEDED)
        reasons = (:unknown, :candidates, :results, :capacity, :comparisons)
        code = Int(reason[]) + 1
        throw(SourceFilterIncomplete(
            code <= length(reasons) ? reasons[code] : :unknown))
    end
    checked(status, :llev_source_filter_index_query)
    resize!(ids, Int(written[]))
    captured = SourceFilterSource(copy(index.source.terms), Val(:owned))
    NativeSourceFilterCursor(captured, ids, 1, page, false)
end

"""Open a native n-gram postings query over a frozen index."""
function query_ngram(index::NativeSourceFilterIndex,
    input::AbstractString, max_distance::Integer; kwargs...)
    index.mode === :ngram ||
        throw(ArgumentError("source-filter index mode is not :ngram"))
    query_filter_index(index, input, max_distance; kwargs...)
end

"""Open a native n-gram/Jaro-Winkler postings query over a frozen index."""
function query_hybrid(index::NativeSourceFilterIndex,
    input::AbstractString, max_distance::Integer; kwargs...)
    index.mode === :hybrid ||
        throw(ArgumentError("source-filter index mode is not :hybrid"))
    query_filter_index(index, input, max_distance; kwargs...)
end

function next_batch!(cursor::NativeSourceFilterCursor,
    maximum::Integer=DEFAULT_MATCH_BATCH)
    cursor.closed && return nothing
    0 < maximum <= 65_536 ||
        throw(ArgumentError("batch maximum must be 1..65,536"))
    remaining = length(cursor.ids) - cursor.next_index + 1
    if remaining <= 0
        close!(cursor)
        return nothing
    end
    count = min(remaining, maximum, cursor.page_results)
    result = [cursor.source.terms[Int(cursor.ids[i]) + 1]
        for i in cursor.next_index:(cursor.next_index + count - 1)]
    cursor.next_index += count
    cursor.next_index > length(cursor.ids) && close!(cursor)
    result
end

function Base.iterate(cursor::NativeSourceFilterCursor, state=nothing)
    batch = next_batch!(cursor, 1)
    batch === nothing ? nothing : (batch[1], nothing)
end

function reduce_batches!(function_value, initial,
    cursor::NativeSourceFilterCursor;
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

function close!(cursor::NativeSourceFilterCursor)
    cursor.closed && return nothing
    cursor.closed = true
    cursor.source = SourceFilterSource(String[], Val(:owned))
    empty!(cursor.ids)
    nothing
end
Base.close(cursor::NativeSourceFilterCursor) = close!(cursor)
Base.isopen(cursor::NativeSourceFilterCursor) = !cursor.closed
cancel!(cursor::NativeSourceFilterCursor) = close!(cursor)
