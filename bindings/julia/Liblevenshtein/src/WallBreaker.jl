"""Finite owned Unicode term matcher; this does not accept a `vt.dictionary.v1` resource."""
mutable struct WallBreakerMatcher
    handle::Ptr{Cvoid}
    closed::Bool
end

"""Copied Unicode WallBreaker result with exact selected-algorithm distance."""
struct WallBreakerMatch
    term::String
    distance::Int
end

Base.:(==)(a::WallBreakerMatch, b::WallBreakerMatch) =
    a.term == b.term && a.distance == b.distance
Base.hash(a::WallBreakerMatch, seed::UInt) = hash((a.term, a.distance), seed)

struct RawWallBreakerTerm
    data::Ptr{UInt8}
    byte_len::Csize_t
end

struct RawWallBreakerLimits
    max_terms::Csize_t
    max_total_term_bytes::Csize_t
    max_term_scalars::Csize_t
    max_query_scalars::Csize_t
    max_candidate_clone_bytes::Csize_t
    max_results::Csize_t
    max_result_bytes::Csize_t
end

"""Logical matcher limits, with implementation hard ceilings; not an RSS budget."""
struct WallBreakerLimits
    max_terms::Int
    max_total_term_bytes::Int
    max_term_scalars::Int
    max_query_scalars::Int
    max_candidate_clone_bytes::Int
    max_results::Int
    max_result_bytes::Int
    function WallBreakerLimits(; max_terms=4096, max_total_term_bytes=1 << 20,
        max_term_scalars=256, max_query_scalars=256,
        max_candidate_clone_bytes=16 << 20, max_results=4096,
        max_result_bytes=1 << 20)
        given = (max_terms, max_total_term_bytes, max_term_scalars,
            max_query_scalars, max_candidate_clone_bytes, max_results,
            max_result_bytes)
        hard = (4096, 1 << 20, 256, 256, 16 << 20, 4096, 1 << 20)
        all(0 < value <= ceiling for (value, ceiling) in zip(given, hard)) ||
            throw(ArgumentError("WallBreaker limits must be positive and within native hard ceilings"))
        new(given...)
    end
end

raw_limits(value::WallBreakerLimits) = RawWallBreakerLimits(
    Csize_t(value.max_terms), Csize_t(value.max_total_term_bytes),
    Csize_t(value.max_term_scalars), Csize_t(value.max_query_scalars),
    Csize_t(value.max_candidate_clone_bytes), Csize_t(value.max_results),
    Csize_t(value.max_result_bytes))

struct RawWallBreakerResult
    term_data::Ptr{UInt8}
    byte_len::Csize_t
    distance::Csize_t
end

struct RawWallBreakerBatch
    results::Ptr{RawWallBreakerResult}
    len::Csize_t
    generation::UInt64
end

mutable struct WallBreakerCursor
    handle::Ptr{Cvoid}
    pending::Vector{WallBreakerMatch}
    offset::Int
    closed::Bool
end

Base.IteratorSize(::Type{WallBreakerCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{WallBreakerCursor}) = Base.HasEltype()
Base.eltype(::Type{WallBreakerCursor}) = WallBreakerMatch

function WallBreakerMatcher(terms::AbstractVector{<:AbstractString};
    max_distance::Integer=1, algorithm::Algorithm=ALGORITHM_STANDARD,
    limits::WallBreakerLimits=WallBreakerLimits())
    0 <= max_distance <= 8 || throw(ArgumentError("max_distance must be in 0:8"))
    bytes = [text_bytes(term) for term in terms]
    raw = [RawWallBreakerTerm(data_pointer(term), length(term)) for term in bytes]
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve bytes raw begin
        checked(ccall(native(:llev_wallbreaker_new_utf8), Cint,
            (Ptr{RawWallBreakerTerm}, Csize_t, RawWallBreakerLimits, UInt32,
                Csize_t, Ref{Ptr{Cvoid}}),
            data_pointer(raw), length(raw), raw_limits(limits), UInt32(algorithm),
            max_distance, output), :llev_wallbreaker_new_utf8)
    end
    matcher = WallBreakerMatcher(output[], false)
    finalizer(close!, matcher)
    matcher
end

function close!(value::WallBreakerMatcher)
    value.closed && return nothing
    ccall(native(:llev_wallbreaker_free), Cvoid, (Ptr{Cvoid},), value.handle)
    value.handle = C_NULL
    value.closed = true
    nothing
end

Base.close(value::WallBreakerMatcher) = close!(value)
Base.isopen(value::WallBreakerMatcher) = !value.closed

function query(matcher::WallBreakerMatcher, source::AbstractString)
    matcher.closed && throw(NativeError(Int32(STATUS_CLOSED), :query, "matcher is closed"))
    bytes = text_bytes(source)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve bytes matcher begin
        checked(ccall(native(:llev_wallbreaker_query_utf8), Cint,
            (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{Ptr{Cvoid}}),
            matcher.handle, data_pointer(bytes), length(bytes), output),
            :llev_wallbreaker_query_utf8)
    end
    cursor = WallBreakerCursor(output[], WallBreakerMatch[], 1, false)
    finalizer(close!, cursor)
    cursor
end

function next_batch!(cursor::WallBreakerCursor, maximum::Integer=DEFAULT_MATCH_BATCH;
    max_bytes::Integer=1 << 20)
    cursor.closed && throw(NativeError(Int32(STATUS_CLOSED), :next_batch!, "cursor is closed"))
    maximum > 0 && max_bytes > 0 ||
        throw(ArgumentError("batch entry and byte limits must be positive"))
    view = Ref(RawWallBreakerBatch(Ptr{RawWallBreakerResult}(C_NULL), 0, 0))
    status = ccall(native(:llev_wallbreaker_cursor_next_batch), Cint,
        (Ptr{Cvoid}, Csize_t, Csize_t, Ref{RawWallBreakerBatch}),
        cursor.handle, checked_csize(maximum, "maximum"),
        checked_csize(max_bytes, "max_bytes"), view)
    checked(status, :llev_wallbreaker_cursor_next_batch; allow_end=true) || return nothing
    batch = view[]
    copied = WallBreakerMatch[]
    try
        sizehint!(copied, batch.len)
        for index in 1:Int(batch.len)
            item = unsafe_load(batch.results, index)
            term = item.byte_len == 0 ? "" : unsafe_string(item.term_data, item.byte_len)
            push!(copied, WallBreakerMatch(term, Int(item.distance)))
        end
    finally
        checked(ccall(native(:llev_wallbreaker_cursor_release_batch), Cint,
            (Ptr{Cvoid}, UInt64), cursor.handle, batch.generation),
            :llev_wallbreaker_cursor_release_batch)
    end
    copied
end

function Base.iterate(cursor::WallBreakerCursor, state=nothing)
    cursor.closed && return nothing
    if cursor.offset > length(cursor.pending)
        batch = next_batch!(cursor)
        if batch === nothing
            close!(cursor)
            return nothing
        end
        cursor.pending = batch
        cursor.offset = 1
    end
    value = cursor.pending[cursor.offset]
    cursor.offset += 1
    (value, nothing)
end

function cancel!(cursor::WallBreakerCursor)
    cursor.closed && return nothing
    checked(ccall(native(:llev_wallbreaker_cursor_cancel), Cint,
        (Ptr{Cvoid},), cursor.handle), :llev_wallbreaker_cursor_cancel)
    nothing
end

function close!(cursor::WallBreakerCursor)
    cursor.closed && return nothing
    ccall(native(:llev_wallbreaker_cursor_free), Cvoid, (Ptr{Cvoid},), cursor.handle)
    cursor.handle = C_NULL
    cursor.closed = true
    nothing
end

Base.close(cursor::WallBreakerCursor) = close!(cursor)
Base.isopen(cursor::WallBreakerCursor) = !cursor.closed

struct RawPatternPiece
    byte_offset::Csize_t
    byte_len::Csize_t
    start_scalar::Csize_t
    end_scalar::Csize_t
    piece_index::Csize_t
end

"""Native splitter projection; offsets and indices are zero-based."""
struct WallBreakerPatternPiece
    content::String
    byte_offset::Int
    byte_len::Int
    start_scalar::Int
    end_scalar::Int
    piece_index::Int
end

function pattern_pieces(source::AbstractString, max_distance::Integer;
    algorithm::Algorithm=ALGORITHM_STANDARD)
    0 <= max_distance <= 8 || throw(ArgumentError("max_distance must be in 0:8"))
    bytes = text_bytes(source)
    required = Ref{Csize_t}(0)
    GC.@preserve bytes begin
        status = ccall(native(:llev_wallbreaker_split_utf8), Cint,
            (Ptr{UInt8}, Csize_t, UInt32, Csize_t, Ptr{RawPatternPiece}, Csize_t,
                Ref{Csize_t}), data_pointer(bytes), length(bytes), UInt32(algorithm),
            max_distance, Ptr{RawPatternPiece}(C_NULL), 0, required)
        status == Int32(STATUS_LIMIT_EXCEEDED) || checked(status, :llev_wallbreaker_split_utf8)
    end
    raw = Vector{RawPatternPiece}(undef, Int(required[]))
    GC.@preserve bytes raw begin
        checked(ccall(native(:llev_wallbreaker_split_utf8), Cint,
            (Ptr{UInt8}, Csize_t, UInt32, Csize_t, Ptr{RawPatternPiece}, Csize_t,
                Ref{Csize_t}), data_pointer(bytes), length(bytes), UInt32(algorithm),
            max_distance, data_pointer(raw), length(raw), required),
            :llev_wallbreaker_split_utf8)
    end
    [WallBreakerPatternPiece(piece.byte_len == 0 ? "" : String(bytes[
        Int(piece.byte_offset)+1:Int(piece.byte_offset + piece.byte_len)]),
        Int(piece.byte_offset), Int(piece.byte_len), Int(piece.start_scalar),
        Int(piece.end_scalar), Int(piece.piece_index)) for piece in raw]
end
