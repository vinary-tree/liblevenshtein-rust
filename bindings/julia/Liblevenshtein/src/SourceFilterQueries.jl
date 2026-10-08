"""Copied, bounded Unicode source terms for lazy native source filtering.

Terms are deduplicated like the native n-gram and hybrid indexes. Source order
is retained for deterministic Julia iteration. Later changes to the caller's
container do not affect this snapshot.
"""
struct SourceFilterSource
    terms::Vector{String}
    SourceFilterSource(terms::Vector{String}, ::Val{:owned}) = new(terms)
end

function SourceFilterSource(terms; max_terms::Integer=100_000,
    max_term_bytes::Integer=4096, max_source_bytes::Integer=64 * 1024 * 1024)
    term_limit = checked_threshold(max_terms)
    per_term = checked_threshold(max_term_bytes)
    byte_limit = checked_threshold(max_source_bytes)
    stored = String[]
    seen = Set{String}()
    bytes = UInt(0)
    for term in terms
        term isa AbstractString ||
            throw(ArgumentError("source terms must be strings"))
        ncodeunits(term) <= per_term ||
            throw(ArgumentError("source term exceeds max_term_bytes"))
        copied = String(term)
        copied in seen && continue
        length(stored) < term_limit ||
            throw(ArgumentError("source exceeds max_terms"))
        size = UInt(length(codeunits(copied)))
        size <= byte_limit - bytes ||
            throw(ArgumentError("source exceeds max_source_bytes"))
        push!(stored, copied)
        push!(seen, copied)
        bytes += size
    end
    SourceFilterSource(stored, Val(:owned))
end

Base.length(source::SourceFilterSource) = length(source.terms)
Base.isempty(source::SourceFilterSource) = isempty(source.terms)

"""Hard cumulative ceilings for one source-filter query."""
struct SourceFilterLimits
    max_candidates::Csize_t
    max_results::Csize_t
    max_comparisons::Csize_t
end

SourceFilterLimits(; max_candidates::Integer=100_000,
    max_results::Integer=100_000,
    max_comparisons::Integer=100_000_000) =
    SourceFilterLimits(checked_threshold(max_candidates),
        checked_threshold(max_results), checked_threshold(max_comparisons))

"""The exact source subset is incomplete because a hard limit was reached."""
struct SourceFilterIncomplete <: Exception
    reason::Symbol
end

Base.showerror(io::IO, error::SourceFilterIncomplete) =
    print(io, "source filter query incomplete: ", error.reason)

"""One closeable, page-bounded native n-gram or hybrid source-filter cursor."""
mutable struct SourceFilterCursor
    source::SourceFilterSource
    query::String
    mode::UInt32
    max_distance::Csize_t
    ngram_size::Csize_t
    jaro_threshold::Float64
    max_input_bytes::Csize_t
    limits::SourceFilterLimits
    page_candidates::Csize_t
    next_index::Int
    visited::Csize_t
    emitted::Csize_t
    comparisons::Csize_t
    closed::Bool
end

Base.IteratorSize(::Type{SourceFilterCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{SourceFilterCursor}) = Base.HasEltype()
Base.eltype(::Type{SourceFilterCursor}) = String

"""Open a lazy native n-gram query over a copied source snapshot."""
query_ngram(source::SourceFilterSource, input::AbstractString,
    max_distance::Integer; ngram_size::Integer=2,
    max_input_bytes::Integer=4096,
    limits::SourceFilterLimits=SourceFilterLimits(),
    page_candidates::Integer=256) =
    query_source_snapshot(source, input, max_distance, UInt32(1);
        ngram_size, jaro_threshold=0.0, max_input_bytes, limits,
        page_candidates)

"""Open a lazy native n-gram/Jaro-Winkler query over a copied source."""
query_hybrid(source::SourceFilterSource, input::AbstractString,
    max_distance::Integer; ngram_size::Integer=2,
    jaro_threshold::Real=0.7, max_input_bytes::Integer=4096,
    limits::SourceFilterLimits=SourceFilterLimits(),
    page_candidates::Integer=256) =
    query_source_snapshot(source, input, max_distance, UInt32(2);
        ngram_size, jaro_threshold, max_input_bytes, limits, page_candidates)

function query_source_snapshot(source::SourceFilterSource, input::AbstractString,
    max_distance::Integer, mode::UInt32; ngram_size::Integer,
    jaro_threshold::Real, max_input_bytes::Integer,
    limits::SourceFilterLimits, page_candidates::Integer)
    page = checked_threshold(page_candidates)
    page > 0 || throw(ArgumentError("page_candidates must be positive"))
    query = String(input)
    distance = checked_threshold(max_distance)
    n = checked_threshold(ngram_size)
    byte_limit = checked_threshold(max_input_bytes)
    threshold = Float64(jaro_threshold)
    source_candidate(mode, query, "", distance;
        ngram_size=n, jaro_threshold=threshold,
        max_input_bytes=byte_limit,
        max_comparisons=limits.max_comparisons)
    SourceFilterCursor(source, query, mode, distance, n, threshold,
        byte_limit, limits, page, 1, 0, 0, 0, false)
end

"""Return at most one source page of owned terms; empty means paused."""
function next_batch!(cursor::SourceFilterCursor, maximum::Integer=DEFAULT_MATCH_BATCH)
    cursor.closed && return nothing
    0 < maximum <= 65_536 ||
        throw(ArgumentError("batch maximum must be 1..65,536"))
    batch = String[]
    inspected = UInt(0)
    try
        while length(batch) < maximum &&
            inspected < cursor.page_candidates &&
            cursor.next_index <= length(cursor.source.terms)
            cursor.visited < cursor.limits.max_candidates ||
                throw(SourceFilterIncomplete(:candidates))
            term = cursor.source.terms[cursor.next_index]
            cursor.next_index += 1
            cursor.visited += 1
            inspected += 1
            remaining = cursor.limits.max_comparisons - cursor.comparisons
            work = if cursor.mode == UInt32(2) && cursor.jaro_threshold > 0
                qchars = UInt(length(cursor.query))
                tchars = UInt(length(term))
                tchars == 0 ? UInt(0) :
                    (qchars <= remaining ÷ tchars ? qchars * tchars :
                        throw(SourceFilterIncomplete(:comparisons)))
            else
                UInt(0)
            end
            cursor.comparisons += work
            accepted = source_candidate(cursor.mode, cursor.query, term,
                cursor.max_distance; ngram_size=cursor.ngram_size,
                jaro_threshold=cursor.jaro_threshold,
                max_input_bytes=cursor.max_input_bytes,
                max_comparisons=remaining)
            if accepted
                cursor.emitted < cursor.limits.max_results ||
                    throw(SourceFilterIncomplete(:results))
                cursor.emitted += 1
                push!(batch, term)
            end
        end
        if cursor.next_index > length(cursor.source.terms)
            close!(cursor)
            return isempty(batch) ? nothing : batch
        end
        batch
    catch
        close!(cursor)
        rethrow()
    end
end

function Base.iterate(cursor::SourceFilterCursor, state=nothing)
    while !cursor.closed
        batch = next_batch!(cursor, 1)
        batch === nothing && return nothing
        isempty(batch) || return (batch[1], nothing)
    end
    nothing
end

function reduce_batches!(function_value, initial, cursor::SourceFilterCursor;
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

function close!(cursor::SourceFilterCursor)
    cursor.closed && return nothing
    cursor.closed = true
    cursor.source = SourceFilterSource(String[], Val(:owned))
    cursor.query = ""
    nothing
end
Base.close(cursor::SourceFilterCursor) = close!(cursor)
Base.isopen(cursor::SourceFilterCursor) = !cursor.closed
cancel!(cursor::SourceFilterCursor) = close!(cursor)
