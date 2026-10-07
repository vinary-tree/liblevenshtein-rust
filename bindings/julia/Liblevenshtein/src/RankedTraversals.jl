"""Start a native distance-then-term query over one captured Unicode revision.

The native cursor buffers at most one edit-distance layer. Close the returned
cursor when stopping before exhaustion, or call `cancel!`.
"""
query_ranked(transducer::Transducer, input::AbstractString,
    maximum_distance::Integer) = query(transducer, input, maximum_distance;
    order=ORDER_DISTANCE_THEN_TERM)

"""An inclusive distance interval over a native ranked cursor."""
mutable struct DistanceRangeCursor
    inner::QueryCursor
    minimum_distance::Int
    maximum_distance::Int
    closed::Bool
    exhausted::Bool
end

Base.IteratorSize(::Type{DistanceRangeCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{DistanceRangeCursor}) = Base.HasEltype()
Base.eltype(::Type{DistanceRangeCursor}) = Match

"""Stream matches within an inclusive distance interval in native rank order.

`minimum_distance` is applied to completed matches. The native automaton uses
`maximum_distance` to prune unreachable dictionary subtrees. An exact-distance
query sets both bounds to the same value.
"""
function query_mode(transducer::Transducer, input::AbstractString;
    minimum_distance::Integer=0, maximum_distance::Integer)
    0 <= minimum_distance <= maximum_distance <= typemax(Int) ||
        throw(ArgumentError("distance interval must be ordered, nonnegative, and fit Int"))
    cursor = DistanceRangeCursor(query_ranked(transducer, input, maximum_distance),
        Int(minimum_distance), Int(maximum_distance), false, false)
    finalizer(close!, cursor)
    cursor
end

function next_batch!(cursor::DistanceRangeCursor,
    maximum::Integer=DEFAULT_MATCH_BATCH)
    cursor.exhausted && return nothing
    cursor.closed && throw(NativeError(Int32(STATUS_CLOSED), :query_mode,
        "cursor is closed"))
    maximum > 0 || throw(ArgumentError("maximum batch size must be positive"))
    result = Match[]
    sizehint!(result, min(maximum, DEFAULT_MATCH_BATCH))
    try
        while length(result) < maximum
            next = iterate(cursor.inner)
            if next === nothing
                cursor.exhausted = true
                close!(cursor)
                break
            end
            match = next[1]
            if match.distance > cursor.maximum_distance
                cursor.exhausted = true
                close!(cursor)
                break
            end
            match.distance >= cursor.minimum_distance && push!(result, match)
        end
    catch
        close!(cursor)
        rethrow()
    end
    isempty(result) ? nothing : result
end

function Base.iterate(cursor::DistanceRangeCursor, state=nothing)
    cursor.closed && return nothing
    batch = next_batch!(cursor, 1)
    batch === nothing ? nothing : (batch[1], nothing)
end

function close!(cursor::DistanceRangeCursor)
    cursor.closed && return nothing
    cursor.closed = true
    close!(cursor.inner)
end

Base.close(cursor::DistanceRangeCursor) = close!(cursor)
Base.isopen(cursor::DistanceRangeCursor) = !cursor.closed
cancel!(cursor::Union{QueryCursor,DistanceRangeCursor}) = close!(cursor)

"""A copied match with a host-computed score, ordered within its distance layer."""
struct ScoredMatch
    term::String
    distance::Int
    id::Union{Nothing,UInt64}
    confidence::Float64
end

"""A closeable cursor that scores and sorts one native distance layer at a time."""
mutable struct ScoredCursor
    inner::QueryCursor
    scorer::Any
    layer::Vector{ScoredMatch}
    offset::Int
    lookahead::Union{Nothing,Match}
    closed::Bool
    exhausted::Bool
end

Base.IteratorSize(::Type{ScoredCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{ScoredCursor}) = Base.HasEltype()
Base.eltype(::Type{ScoredCursor}) = ScoredMatch

"""Rank Unicode suggestions by distance, descending score, then ascending term.

`scorer(term, distance, id)` is called once for every candidate in the current
distance layer. A non-finite score ranks last. The native cursor owns the
dictionary snapshot; this adapter retains only one scored layer and the native
cursor's bounded batch. Close or cancel when stopping early.
"""
function query_suggestions(transducer::Transducer, input::AbstractString,
    maximum_distance::Integer, scorer)
    cursor = ScoredCursor(query_ranked(transducer, input, maximum_distance),
        scorer, ScoredMatch[], 1, nothing, false, false)
    finalizer(close!, cursor)
    cursor
end

function fill_layer!(cursor::ScoredCursor)
    first_match = cursor.lookahead
    cursor.lookahead = nothing
    if first_match === nothing
        next = iterate(cursor.inner)
        next === nothing && return false
        first_match = next[1]
    end
    distance = first_match.distance
    empty!(cursor.layer)
    while true
        raw_score = Float64(cursor.scorer(first_match.term,
            first_match.distance, first_match.id))
        score = isfinite(raw_score) ? raw_score : -Inf
        push!(cursor.layer, ScoredMatch(first_match.term, distance,
            first_match.id, score))
        next = iterate(cursor.inner)
        if next === nothing
            break
        end
        first_match = next[1]
        if first_match.distance != distance
            cursor.lookahead = first_match
            break
        end
    end
    sort!(cursor.layer; lt=(left, right) ->
        left.confidence > right.confidence ||
        (left.confidence == right.confidence && left.term < right.term))
    cursor.offset = 1
    true
end

function next_batch!(cursor::ScoredCursor,
    maximum::Integer=DEFAULT_MATCH_BATCH)
    cursor.exhausted && return nothing
    cursor.closed && throw(NativeError(Int32(STATUS_CLOSED), :query_suggestions,
        "cursor is closed"))
    maximum > 0 || throw(ArgumentError("maximum batch size must be positive"))
    result = ScoredMatch[]
    sizehint!(result, min(maximum, DEFAULT_MATCH_BATCH))
    try
        while length(result) < maximum
            if cursor.offset > length(cursor.layer)
                if !fill_layer!(cursor)
                    cursor.exhausted = true
                    close!(cursor)
                    break
                end
            end
            push!(result, cursor.layer[cursor.offset])
            cursor.offset += 1
        end
    catch
        close!(cursor)
        rethrow()
    end
    isempty(result) ? nothing : result
end

function Base.iterate(cursor::ScoredCursor, state=nothing)
    cursor.closed && return nothing
    batch = next_batch!(cursor, 1)
    batch === nothing ? nothing : (batch[1], nothing)
end

function close!(cursor::ScoredCursor)
    cursor.closed && return nothing
    cursor.closed = true
    empty!(cursor.layer)
    cursor.lookahead = nothing
    close!(cursor.inner)
end

Base.close(cursor::ScoredCursor) = close!(cursor)
Base.isopen(cursor::ScoredCursor) = !cursor.closed
cancel!(cursor::ScoredCursor) = close!(cursor)

"""Reduce owned batches from a ranked adapter and consume it on every exit."""
function reduce_batches!(function_value, initial,
    cursor::Union{DistanceRangeCursor,ScoredCursor};
    batch_size::Integer=DEFAULT_MATCH_BATCH)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    accumulator = initial
    try
        while true
            batch = next_batch!(cursor, batch_size)
            batch === nothing && return accumulator
            accumulator = function_value(accumulator, batch)
        end
    finally
        close!(cursor)
    end
end
