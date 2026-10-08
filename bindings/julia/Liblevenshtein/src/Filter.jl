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
