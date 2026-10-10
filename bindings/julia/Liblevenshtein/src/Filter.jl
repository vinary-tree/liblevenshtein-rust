"""Compare two Unicode strings with the native Jaro scorer.

The input byte and worst-case scalar-comparison limits are checked before
native scoring. A zero prefix scale uses Jaro; a positive scale up to 0.25
uses Jaro-Winkler. The default limits can be lowered for untrusted inputs.
"""
function jaro_similarity(left::AbstractString, right::AbstractString;
    prefix_scale::Real=0.0, max_input_bytes::Integer=1_000_000,
    max_comparisons::Integer=100_000_000)
    api_revision() >= UInt32(11) || throw(NativeError(Int32(STATUS_UNSUPPORTED),
        :llev_jaro_similarity_utf8, "Jaro scoring requires native API revision 11"))
    x = text_bytes(left)
    y = text_bytes(right)
    score = Ref{Cdouble}(0.0)
    status = GC.@preserve x y ccall(native(:llev_jaro_similarity_utf8), Cint,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, Cdouble, Csize_t,
            Csize_t, Ref{Cdouble}),
        isempty(x) ? C_NULL : pointer(x), length(x),
        isempty(y) ? C_NULL : pointer(y), length(y),
        Float64(prefix_scale), checked_threshold(max_input_bytes),
        checked_threshold(max_comparisons), score)
    checked(status, :llev_jaro_similarity_utf8)
    Float64(score[])
end

"""Native Jaro-Winkler similarity with the conventional 0.1 prefix scale."""
jaro_winkler_similarity(left::AbstractString, right::AbstractString; kwargs...) =
    jaro_similarity(left, right; prefix_scale=0.1, kwargs...)

"""Native Jaro-Winkler similarity with a caller-chosen prefix scale."""
jaro_winkler_similarity_scaled(left::AbstractString, right::AbstractString,
    prefix_scale::Real; kwargs...) =
    jaro_similarity(left, right; prefix_scale, kwargs...)

"""Test a Jaro-Winkler similarity threshold with bounded native scoring."""
is_similar(left::AbstractString, right::AbstractString, threshold::Real;
    kwargs...) = jaro_winkler_similarity(left, right; kwargs...) >= threshold

function source_candidate(mode::UInt32, query::AbstractString,
    candidate::AbstractString, max_distance::Integer;
    ngram_size::Integer=2, jaro_threshold::Real=0.0,
    max_input_bytes::Integer=4096, max_comparisons::Integer=16_777_216)
    api_revision() >= UInt32(11) || throw(NativeError(Int32(STATUS_UNSUPPORTED),
        :llev_source_filter_utf8,
        "source filters require native API revision 11"))
    q = text_bytes(query)
    c = text_bytes(candidate)
    accepted = Ref{UInt8}(0)
    status = GC.@preserve q c ccall(native(:llev_source_filter_utf8), Cint,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, UInt32, Csize_t,
            Csize_t, Cdouble, Csize_t, Csize_t, Ref{UInt8}),
        isempty(q) ? C_NULL : pointer(q), length(q),
        isempty(c) ? C_NULL : pointer(c), length(c), mode,
        checked_threshold(ngram_size), checked_threshold(max_distance),
        Float64(jaro_threshold), checked_threshold(max_input_bytes),
        checked_threshold(max_comparisons), accepted)
    checked(status, :llev_source_filter_utf8)
    accepted[] != 0
end

"""Test one Unicode candidate against native n-gram overlap filtering."""
ngram_candidate(query::AbstractString, candidate::AbstractString,
    max_distance::Integer; ngram_size::Integer=2,
    max_input_bytes::Integer=4096) =
    source_candidate(UInt32(1), query, candidate, max_distance;
        ngram_size, max_input_bytes, max_comparisons=0)

"""Test one Unicode candidate against native n-gram/Jaro hybrid filtering."""
hybrid_candidate(query::AbstractString, candidate::AbstractString,
    max_distance::Integer; ngram_size::Integer=2,
    jaro_threshold::Real=0.7, max_input_bytes::Integer=4096,
    max_comparisons::Integer=16_777_216) =
    source_candidate(UInt32(2), query, candidate, max_distance;
        ngram_size, jaro_threshold, max_input_bytes, max_comparisons)

function query_source_filter(transducer::Transducer, input::AbstractString,
    maximum_distance::Integer, mode::UInt32;
    ngram_size::Integer=2, jaro_threshold::Real=0.0,
    max_input_bytes::Integer=4096,
    max_comparisons::Integer=16_777_216)
    query = String(input)
    visitor = PrefixVisitor((_unit, _depth) -> true,
        (_unit, _depth) -> nothing;
        permits=prefix -> source_candidate(mode, query, join(prefix),
            maximum_distance; ngram_size, jaro_threshold, max_input_bytes,
            max_comparisons))
    query_pruned(transducer, query, maximum_distance, visitor)
end

"""Stream Unicode fuzzy matches admitted by native n-gram filtering.

Each final source prefix is tested before a result is emitted. The cursor
captures one native dictionary snapshot and yields `SpecializedMatch` values.
"""
query_ngram(transducer::Transducer, input::AbstractString,
    maximum_distance::Integer; ngram_size::Integer=2,
    max_input_bytes::Integer=4096) =
    query_source_filter(transducer, input, maximum_distance, UInt32(1);
        ngram_size, max_input_bytes, max_comparisons=0)

"""Stream Unicode fuzzy matches admitted by native n-gram/Jaro filtering."""
query_hybrid(transducer::Transducer, input::AbstractString,
    maximum_distance::Integer; ngram_size::Integer=2,
    jaro_threshold::Real=0.7, max_input_bytes::Integer=4096,
    max_comparisons::Integer=16_777_216) =
    query_source_filter(transducer, input, maximum_distance, UInt32(2);
        ngram_size, jaro_threshold, max_input_bytes, max_comparisons)

"""Filter a Unicode fuzzy query by native Jaro-Winkler similarity.

The source prefix is tested when it becomes a final dictionary node. The
cursor keeps the native query snapshot, streams bounded batches, and returns
`SpecializedMatch` values in dictionary traversal order. Close or cancel it
when iteration stops early.
"""
function query_jaro(transducer::Transducer, input::AbstractString,
    maximum_distance::Integer; minimum_similarity::Real=0.8,
    prefix_scale::Real=0.1, max_input_bytes::Integer=1_000_000,
    max_comparisons::Integer=100_000_000)
    threshold = Float64(minimum_similarity)
    isfinite(threshold) && 0.0 <= threshold <= 1.0 ||
        throw(ArgumentError("minimum_similarity must be finite and within [0, 1]"))
    query = String(input)
    visitor = PrefixVisitor((_unit, _depth) -> true,
        (_unit, _depth) -> nothing;
        permits=prefix -> jaro_similarity(join(prefix), query;
            prefix_scale, max_input_bytes, max_comparisons) >= threshold)
    query_pruned(transducer, query, maximum_distance, visitor)
end
