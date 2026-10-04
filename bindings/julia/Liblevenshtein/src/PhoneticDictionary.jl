"""One copied native candidate; `distance` is measured in normalized space."""
struct PhoneticCandidate
    term::String
    distance::Int
    normalized_form::String
end

struct RawUtf8View
    data::Ptr{UInt8}
    len::Csize_t
end

struct RawOwnedString
    data::Ptr{UInt8}
    len::Csize_t
end

struct RawPhoneticCandidate
    term::RawOwnedString
    distance::Csize_t
    normalized_form::RawOwnedString
end

"""An owned phonetic-normalized dictionary.

The default backend is mutable; `compact=true` chooses native term-ID payloads
and is immutable. Both preserve native normalized-space relevance ordering.
"""
mutable struct PhoneticNormalizedDictionary
    handle::Ptr{Cvoid}
    compact::Bool
    closed::Bool
end

function PhoneticNormalizedDictionary(terms;
    rules::Union{Nothing,PhoneticRuleSet}=nothing,
    algorithm::Algorithm=ALGORITHM_STANDARD,
    compact::Bool=false,
    max_terms::Integer=100_000,
    max_total_bytes::Integer=1_000_000)
    max_terms > 0 || throw(ArgumentError("max_terms must be positive"))
    max_total_bytes > 0 || throw(ArgumentError("max_total_bytes must be positive"))
    terms isa AbstractString && throw(ArgumentError("terms must be an iterable of strings"))
    encoded = Vector{UInt8}[]
    for term in terms
        term isa AbstractString || throw(ArgumentError("each term must be a string"))
        push!(encoded, text_bytes(term))
        length(encoded) <= max_terms ||
            throw(ArgumentError("term count exceeds max_terms"))
    end
    views = RawUtf8View[RawUtf8View(isempty(bytes) ? C_NULL : pointer(bytes),
        length(bytes)) for bytes in encoded]
    output = Ref{Ptr{Cvoid}}(C_NULL)
    rules_handle = rules === nothing ? Ptr{Cvoid}(C_NULL) : require_open(rules)
    GC.@preserve encoded views checked(ccall(native(:llev_phonetic_dictionary_new), Cint,
        (Ptr{RawUtf8View}, Csize_t, Ptr{Cvoid}, UInt32, UInt8,
            Csize_t, Csize_t, Ref{Ptr{Cvoid}}),
        isempty(views) ? C_NULL : pointer(views), length(views), rules_handle,
        UInt32(algorithm), UInt8(compact),
        checked_csize(max_terms, "max_terms"),
        checked_csize(max_total_bytes, "max_total_bytes"), output),
        :llev_phonetic_dictionary_new)
    dictionary = PhoneticNormalizedDictionary(output[], compact, false)
    finalizer(close!, dictionary)
    dictionary
end

function require_open(dictionary::PhoneticNormalizedDictionary)
    dictionary.closed && throw(NativeError(Int32(STATUS_CLOSED),
        :phonetic_dictionary, "dictionary is closed"))
    dictionary.handle
end

function close!(dictionary::PhoneticNormalizedDictionary)
    dictionary.closed && return nothing
    handle = dictionary.handle
    dictionary.handle = C_NULL
    dictionary.closed = true
    handle == C_NULL || ccall(native(:llev_phonetic_dictionary_free), Cvoid,
        (Ptr{Cvoid},), handle)
    nothing
end

Base.close(dictionary::PhoneticNormalizedDictionary) = close!(dictionary)
Base.isopen(dictionary::PhoneticNormalizedDictionary) = !dictionary.closed

function copied_owned_string(raw::RawOwnedString)
    raw.len == 0 && return ""
    raw.data == C_NULL && throw(ArgumentError("native phonetic result has null string data"))
    String(copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(raw.data), Int(raw.len))))
end

"""Return copied native candidates in native relevance order."""
function query(dictionary::PhoneticNormalizedDictionary, text::AbstractString;
    max_distance::Integer=0, max_query_scalars::Integer=1_000_000,
    max_results::Integer=100_000)
    max_distance >= 0 || throw(ArgumentError("max_distance must be nonnegative"))
    max_query_scalars > 0 || throw(ArgumentError("max_query_scalars must be positive"))
    max_results > 0 || throw(ArgumentError("max_results must be positive"))
    bytes = text_bytes(text)
    output = Ref{Ptr{RawPhoneticCandidate}}(C_NULL)
    count = Ref{Csize_t}(0)
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_dictionary_query), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Csize_t, Csize_t, Csize_t,
            Ref{Ptr{RawPhoneticCandidate}}, Ref{Csize_t}),
        require_open(dictionary), isempty(bytes) ? C_NULL : pointer(bytes),
        length(bytes), checked_csize(max_distance, "max_distance"),
        checked_csize(max_query_scalars, "max_query_scalars"),
        checked_csize(max_results, "max_results"), output, count),
        :llev_phonetic_dictionary_query)
    try
        results = PhoneticCandidate[]
        sizehint!(results, Int(count[]))
        for index in 1:Int(count[])
            item = unsafe_load(output[], index)
            push!(results, PhoneticCandidate(copied_owned_string(item.term),
                Int(item.distance), copied_owned_string(item.normalized_form)))
        end
        results
    finally
        ccall(native(:llev_phonetic_candidates_free), Cvoid,
            (Ptr{RawPhoneticCandidate}, Csize_t), output[], count[])
    end
end

function phonetic_update!(dictionary::PhoneticNormalizedDictionary,
    term::AbstractString, remove::Bool; max_term_scalars::Integer=1_000_000)
    max_term_scalars > 0 || throw(ArgumentError("max_term_scalars must be positive"))
    bytes = text_bytes(term)
    changed = Ref{UInt8}(0)
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_dictionary_update), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, UInt8, Csize_t, Ref{UInt8}),
        require_open(dictionary), isempty(bytes) ? C_NULL : pointer(bytes), length(bytes),
        UInt8(remove), checked_csize(max_term_scalars, "max_term_scalars"), changed),
        :llev_phonetic_dictionary_update)
    changed[] != 0
end

"""Insert a spelling; return whether it was not already present."""
insert!(dictionary::PhoneticNormalizedDictionary, term::AbstractString; kwargs...) =
    phonetic_update!(dictionary, term, false; kwargs...)

"""Remove a spelling; return whether it existed in the normalized index."""
remove!(dictionary::PhoneticNormalizedDictionary, term::AbstractString; kwargs...) =
    phonetic_update!(dictionary, term, true; kwargs...)

Base.push!(dictionary::PhoneticNormalizedDictionary, term::AbstractString) =
    (insert!(dictionary, term); dictionary)
Base.delete!(dictionary::PhoneticNormalizedDictionary, term::AbstractString) =
    (remove!(dictionary, term); dictionary)
