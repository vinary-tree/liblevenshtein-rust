module Liblevenshtein

using Libdl
import VinaryTreeInterop

const VTI = VinaryTreeInterop

include("GeneratedAbi.jl")

export ABI_VERSION,
    API_REVISION,
    DEFAULT_MATCH_BATCH,
    BUILD_FEATURE_CORE,
    BUILD_FEATURE_PHONETIC,
    Status,
    Algorithm,
    QueryOrder,
    PhoneticRuleSetKind,
    NativeError,
    Match,
    BorrowedMatch,
    BorrowedBatch,
    Transducer,
    QueryCursor,
    PhoneticPattern,
    PhoneticRuleSet,
    abi_version,
    api_revision,
    build_features,
    distance,
    damerau_distance,
    true_damerau_distance,
    snapshot,
    unit_domain,
    query,
    next_batch!,
    reduce_batches!,
    materialize,
    close!,
    ALGORITHM_STANDARD,
    ALGORITHM_TRANSPOSITION,
    ALGORITHM_MERGE_AND_SPLIT,
    ALGORITHM_DAMERAU_LEVENSHTEIN,
    ORDER_TRAVERSAL,
    ORDER_DISTANCE_THEN_TERM,
    RULES_ENGLISH_ORTHOGRAPHY,
    RULES_ENGLISH_PHONETIC

"""A copied native failure with its stable status, operation, and diagnostic."""
struct NativeError <: Exception
    status::Int32
    operation::Symbol
    message::String
end

function Base.showerror(io::IO, error::NativeError)
    print(io, error.operation, " failed with native status ", error.status)
    isempty(error.message) || print(io, ": ", error.message)
end

struct RawMatch
    term_data::Ptr{Cvoid}
    term_len::Csize_t
    byte_len::Csize_t
    distance::Csize_t
    id::UInt64
    unit_domain::UInt32
    has_id::UInt8
    reserved::NTuple{3,UInt8}
end

struct RawBatch
    matches::Ptr{RawMatch}
    len::Csize_t
    generation::UInt64
end

struct OwnedString
    data::Ptr{UInt8}
    len::Csize_t
end

"""One independently owned fuzzy match whose term type preserves its unit domain."""
struct Match{T}
    term::T
    distance::Int
    id::Union{Nothing,UInt64}
    unit_domain::VTI.UnitDomain
end

mutable struct BorrowedBatch
    matches::Ptr{RawMatch}
    length::Int
    active::Bool
end

struct BorrowedMatch
    batch::BorrowedBatch
    index::Int
end

Base.length(batch::BorrowedBatch) = batch.active ? batch.length : 0
Base.eltype(::Type{BorrowedBatch}) = BorrowedMatch

function Base.getindex(batch::BorrowedBatch, index::Integer)
    batch.active || throw(ArgumentError("borrowed match batch has expired"))
    checkbounds(1:batch.length, index)
    BorrowedMatch(batch, Int(index))
end

function Base.iterate(batch::BorrowedBatch, state::Int=1)
    state > length(batch) && return nothing
    (batch[state], state + 1)
end

function raw(match::BorrowedMatch)
    match.batch.active || throw(ArgumentError("borrowed match has expired"))
    unsafe_load(match.batch.matches, match.index)
end

Base.getproperty(match::BorrowedMatch, name::Symbol) = if name === :distance
    Int(raw(match).distance)
elseif name === :id
    value = raw(match)
    value.has_id == 0 ? nothing : value.id
elseif name === :unit_domain
    VTI.UnitDomain(raw(match).unit_domain)
else
    getfield(match, name)
end

const LIBRARY_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function library_candidates()
    names = Sys.iswindows() ? ["liblevenshtein.dll"] :
        Sys.isapple() ? ["libliblevenshtein.dylib"] : ["libliblevenshtein.so"]
    explicit = get(ENV, "LIBLEVENSHTEIN_LIBRARY", "")
    isempty(explicit) ? names : vcat([explicit], names)
end

function library_handle()
    LIBRARY_HANDLE[] != C_NULL && return LIBRARY_HANDLE[]
    failures = String[]
    for candidate in library_candidates()
        try
            LIBRARY_HANDLE[] = Libdl.dlopen(candidate)
            return LIBRARY_HANDLE[]
        catch error
            push!(failures, "$candidate: $(sprint(showerror, error))")
        end
    end
    error("could not load liblevenshtein; set LIBLEVENSHTEIN_LIBRARY\n" *
        join(failures, "\n"))
end

native(name::Symbol) = Libdl.dlsym(library_handle(), name)

abi_version() = UInt32(ccall(native(:llev_abi_version), UInt32, ()))
api_revision() = UInt32(ccall(native(:llev_api_revision), UInt32, ()))
build_features() = UInt64(ccall(native(:llev_build_features), UInt64, ()))

function last_error_message()
    pointer = ccall(native(:llev_last_error_message), Cstring, ())
    pointer == C_NULL ? "" : unsafe_string(pointer)
end

function checked(code::Integer, operation::Symbol; allow_end::Bool=false)
    code == Int32(STATUS_OK) && return true
    allow_end && code == Int32(STATUS_END) && return false
    throw(NativeError(Int32(code), operation, last_error_message()))
end

function text_bytes(value::AbstractString)
    Vector{UInt8}(codeunits(String(value)))
end

function exact_distance_call(symbol::Symbol, source::AbstractString,
    target::AbstractString)::Int
    left = text_bytes(source)
    right = text_bytes(target)
    result = GC.@preserve left right begin
        ccall(native(symbol), Csize_t,
            (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t),
            isempty(left) ? C_NULL : pointer(left), length(left),
            isempty(right) ? C_NULL : pointer(right), length(right))
    end
    result == typemax(Csize_t) &&
        throw(NativeError(Int32(STATUS_INVALID_UTF8), symbol, last_error_message()))
    Int(result)
end

function threshold_distance_call(symbol::Symbol, source::AbstractString,
    target::AbstractString, threshold::Integer)::Union{Nothing,Int}
    threshold >= 0 || throw(ArgumentError("threshold must be nonnegative"))
    left = text_bytes(source)
    right = text_bytes(target)
    result = GC.@preserve left right ccall(native(symbol), Csize_t,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, Csize_t),
        isempty(left) ? C_NULL : pointer(left), length(left),
        isempty(right) ? C_NULL : pointer(right), length(right), threshold)
    result == typemax(Csize_t) &&
        throw(NativeError(Int32(STATUS_INVALID_UTF8), symbol, last_error_message()))
    result == typemax(Csize_t) - 1 && return nothing
    Int(result)
end

_distance(source, target, ::Nothing) =
    exact_distance_call(:llev_distance, source, target)
_distance(source, target, threshold::Integer) =
    threshold_distance_call(:llev_distance_threshold, source, target, threshold)
_damerau_distance(source, target, ::Nothing) =
    exact_distance_call(:llev_damerau_distance, source, target)
_damerau_distance(source, target, threshold::Integer) =
    threshold_distance_call(:llev_damerau_distance_threshold, source, target, threshold)
_true_damerau_distance(source, target, ::Nothing) =
    exact_distance_call(:llev_true_damerau_distance, source, target)
_true_damerau_distance(source, target, threshold::Integer) =
    threshold_distance_call(:llev_true_damerau_distance_threshold, source, target, threshold)

"""Compute Unicode-scalar Levenshtein distance, or `nothing` above `threshold`."""
distance(source::AbstractString, target::AbstractString; threshold=nothing) =
    _distance(source, target, threshold)

"""Compute optimal-string-alignment distance, or `nothing` above `threshold`."""
damerau_distance(source::AbstractString, target::AbstractString; threshold=nothing) =
    _damerau_distance(source, target, threshold)

"""Compute unrestricted Damerau-Levenshtein distance, or `nothing` above `threshold`."""
true_damerau_distance(source::AbstractString, target::AbstractString; threshold=nothing) =
    _true_damerau_distance(source, target, threshold)

function materialize(value::RawMatch)
    domain = VTI.UnitDomain(value.unit_domain)
    term = if domain == VTI.UNIT_UNICODE_SCALAR
        value.byte_len == 0 ? "" : unsafe_string(Ptr{UInt8}(value.term_data), value.byte_len)
    elseif domain == VTI.UNIT_BYTE
        value.byte_len == 0 ? UInt8[] : copy(unsafe_wrap(Vector{UInt8},
            Ptr{UInt8}(value.term_data), value.byte_len; own=false))
    elseif domain == VTI.UNIT_U64
        value.term_len == 0 ? UInt64[] : copy(unsafe_wrap(Vector{UInt64},
            Ptr{UInt64}(value.term_data), value.term_len; own=false))
    else
        throw(ArgumentError("unknown native unit domain $(value.unit_domain)"))
    end
    Match(term, Int(value.distance), value.has_id == 0 ? nothing : value.id, domain)
end

materialize(match::BorrowedMatch) = materialize(raw(match))

mutable struct QueryCursor
    handle::Ptr{Cvoid}
    pending::Vector{Match}
    offset::Int
    closed::Bool
end

Base.IteratorSize(::Type{QueryCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{QueryCursor}) = Base.HasEltype()
Base.eltype(::Type{QueryCursor}) = Match

function QueryCursor(handle::Ptr{Cvoid})
    handle == C_NULL && throw(ArgumentError("native cursor handle is null"))
    cursor = QueryCursor(handle, Match[], 1, false)
    finalizer(close!, cursor)
    cursor
end

function require_open(cursor::QueryCursor)
    cursor.closed && throw(NativeError(Int32(STATUS_CLOSED), :cursor, "cursor is closed"))
    cursor.handle
end

function next_batch!(cursor::QueryCursor, maximum::Integer=DEFAULT_MATCH_BATCH)
    maximum > 0 || throw(ArgumentError("maximum batch size must be positive"))
    view = Ref(RawBatch(Ptr{RawMatch}(C_NULL), 0, 0))
    status = ccall(native(:llev_query_cursor_next_batch), Cint,
        (Ptr{Cvoid}, Csize_t, Ref{RawBatch}), require_open(cursor), maximum, view)
    checked(status, :llev_query_cursor_next_batch; allow_end=true) || return nothing
    batch = view[]
    output = Match[]
    try
        sizehint!(output, batch.len)
        for index in 1:Int(batch.len)
            push!(output, materialize(unsafe_load(batch.matches, index)))
        end
    finally
        checked(ccall(native(:llev_query_cursor_release_batch), Cint,
            (Ptr{Cvoid}, UInt64), cursor.handle, batch.generation),
            :llev_query_cursor_release_batch)
    end
    output
end

function Base.iterate(cursor::QueryCursor, state=nothing)
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

mutable struct ReducerState
    function_value::Any
    accumulator::Any
    failure::Any
end

function reducer_callback(context::Ptr{Cvoid}, matches::Ptr{RawMatch}, len::Csize_t)::Cint
    state = unsafe_pointer_to_objref(context)::ReducerState
    batch = BorrowedBatch(matches, Int(len), true)
    try
        state.accumulator = state.function_value(state.accumulator, batch)
        return Cint(STATUS_OK)
    catch error
        state.failure = (error, catch_backtrace())
        return Cint(STATUS_END)
    finally
        batch.active = false
    end
end

const REDUCER_CALLBACK = Ref{Ptr{Cvoid}}(C_NULL)

"""Reduce bounded zero-copy native batches and always consume the cursor."""
function reduce_batches!(function_value, initial, cursor::QueryCursor;
    batch_size::Integer=DEFAULT_MATCH_BATCH)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    state = ReducerState(function_value, initial, nothing)
    count = Ref{Csize_t}(0)
    try
        GC.@preserve state begin
            checked(ccall(native(:llev_query_cursor_reduce), Cint,
                (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}, Ref{Csize_t}),
                require_open(cursor), batch_size, REDUCER_CALLBACK[],
                pointer_from_objref(state), count), :llev_query_cursor_reduce)
        end
    finally
        close!(cursor)
    end
    state.failure === nothing || Base.throw(state.failure[1])
    state.accumulator
end

function close!(cursor::QueryCursor)
    cursor.closed && return nothing
    handle = cursor.handle
    status = ccall(native(:llev_query_cursor_free), Cint, (Ptr{Cvoid},), handle)
    checked(status, :llev_query_cursor_free)
    cursor.handle = C_NULL
    cursor.closed = true
    empty!(cursor.pending)
    nothing
end

Base.close(cursor::QueryCursor) = close!(cursor)
Base.isopen(cursor::QueryCursor) = !cursor.closed

mutable struct Transducer
    handle::Ptr{Cvoid}
    closed::Bool
end

function resource_raw(source)
    source isa VTI.Resource && return VTI.raw_resource(source)
    source isa VTI.Dictionary && return VTI.raw_resource(source.resource)
    throw(ArgumentError("source must be a VinaryTreeInterop Resource or Dictionary"))
end

function Transducer(source, algorithm::Algorithm=ALGORITHM_STANDARD)
    input = Ref(resource_raw(source))
    output = Ref{Ptr{Cvoid}}(C_NULL)
    checked(ccall(native(:llev_transducer_new), Cint,
        (Ref{VTI.VtResourceRaw}, UInt32, Ref{Ptr{Cvoid}}), input,
        UInt32(algorithm), output), :llev_transducer_new)
    value = Transducer(output[], false)
    finalizer(close!, value)
    value
end

function require_open(transducer::Transducer)
    transducer.closed &&
        throw(NativeError(Int32(STATUS_CLOSED), :transducer, "transducer is closed"))
    transducer.handle
end

function snapshot(transducer::Transducer)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    checked(ccall(native(:llev_transducer_snapshot), Cint,
        (Ptr{Cvoid}, Ref{Ptr{Cvoid}}), require_open(transducer), output),
        :llev_transducer_snapshot)
    value = Transducer(output[], false)
    finalizer(close!, value)
    value
end

function unit_domain(transducer::Transducer)
    output = Ref{UInt32}(0)
    checked(ccall(native(:llev_transducer_unit_domain), Cint,
        (Ptr{Cvoid}, Ref{UInt32}), require_open(transducer), output),
        :llev_transducer_unit_domain)
    VTI.UnitDomain(output[])
end

function query(transducer::Transducer, input::AbstractString, maximum_distance::Integer;
    order::QueryOrder=ORDER_TRAVERSAL)
    bytes = text_bytes(input)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve bytes checked(ccall(native(:llev_transducer_query_utf8), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Csize_t, UInt32, Ref{Ptr{Cvoid}}),
        require_open(transducer), isempty(bytes) ? C_NULL : pointer(bytes), length(bytes),
        maximum_distance, UInt32(order), output), :llev_transducer_query_utf8)
    QueryCursor(output[])
end

function query(transducer::Transducer, input::AbstractVector{UInt8},
    maximum_distance::Integer; order::QueryOrder=ORDER_TRAVERSAL)
    bytes = Vector{UInt8}(input)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve bytes checked(ccall(native(:llev_transducer_query_bytes), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Csize_t, UInt32, Ref{Ptr{Cvoid}}),
        require_open(transducer), isempty(bytes) ? C_NULL : pointer(bytes), length(bytes),
        maximum_distance, UInt32(order), output), :llev_transducer_query_bytes)
    QueryCursor(output[])
end

function query(transducer::Transducer, input::AbstractVector{<:Integer},
    maximum_distance::Integer; order::QueryOrder=ORDER_TRAVERSAL)
    tokens = UInt64[]
    sizehint!(tokens, length(input))
    for token in input
        0 <= token <= typemax(UInt64) || throw(ArgumentError("query token is outside UInt64"))
        push!(tokens, UInt64(token))
    end
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve tokens checked(ccall(native(:llev_transducer_query_u64), Cint,
        (Ptr{Cvoid}, Ptr{UInt64}, Csize_t, Csize_t, UInt32, Ref{Ptr{Cvoid}}),
        require_open(transducer), isempty(tokens) ? C_NULL : pointer(tokens), length(tokens),
        maximum_distance, UInt32(order), output), :llev_transducer_query_u64)
    QueryCursor(output[])
end

function close!(transducer::Transducer)
    transducer.closed && return nothing
    handle = transducer.handle
    transducer.handle = C_NULL
    transducer.closed = true
    handle == C_NULL || ccall(native(:llev_transducer_free), Cvoid, (Ptr{Cvoid},), handle)
    nothing
end

Base.close(transducer::Transducer) = close!(transducer)
Base.isopen(transducer::Transducer) = !transducer.closed

mutable struct PhoneticPattern
    handle::Ptr{Cvoid}
    closed::Bool
end

function compile_pattern(source::AbstractString, symbol::Symbol)
    bytes = text_bytes(source)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve bytes checked(ccall(native(symbol), Cint,
        (Ptr{UInt8}, Csize_t, Ref{Ptr{Cvoid}}),
        isempty(bytes) ? C_NULL : pointer(bytes), length(bytes), output), symbol)
    value = PhoneticPattern(output[], false)
    finalizer(close!, value)
    value
end

PhoneticPattern(source::AbstractString; llre::Bool=false) =
    compile_pattern(source, llre ? :llev_phonetic_pattern_compile_llre :
        :llev_phonetic_pattern_compile_regex)

function require_open(pattern::PhoneticPattern)
    pattern.closed &&
        throw(NativeError(Int32(STATUS_CLOSED), :phonetic_pattern, "pattern is closed"))
    pattern.handle
end

function Base.size(pattern::PhoneticPattern)
    states = Ref{Csize_t}(0)
    transitions = Ref{Csize_t}(0)
    checked(ccall(native(:llev_phonetic_pattern_size), Cint,
        (Ptr{Cvoid}, Ref{Csize_t}, Ref{Csize_t}), require_open(pattern), states,
        transitions), :llev_phonetic_pattern_size)
    (Int(states[]), Int(transitions[]))
end

function Base.in(input::AbstractString, pattern::PhoneticPattern)
    bytes = text_bytes(input)
    output = Ref{UInt8}(0)
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_pattern_matches), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{UInt8}), require_open(pattern),
        isempty(bytes) ? C_NULL : pointer(bytes), length(bytes), output),
        :llev_phonetic_pattern_matches)
    output[] != 0
end

function query(transducer::Transducer, pattern::PhoneticPattern,
    maximum_distance::Integer)
    0 <= maximum_distance <= typemax(UInt8) ||
        throw(ArgumentError("phonetic maximum distance must fit UInt8"))
    output = Ref{Ptr{Cvoid}}(C_NULL)
    checked(ccall(native(:llev_transducer_query_pattern), Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, UInt8, Ref{Ptr{Cvoid}}), require_open(transducer),
        require_open(pattern), maximum_distance, output), :llev_transducer_query_pattern)
    QueryCursor(output[])
end

function close!(pattern::PhoneticPattern)
    pattern.closed && return nothing
    handle = pattern.handle
    pattern.handle = C_NULL
    pattern.closed = true
    handle == C_NULL || ccall(native(:llev_phonetic_pattern_free), Cvoid,
        (Ptr{Cvoid},), handle)
    nothing
end

Base.close(pattern::PhoneticPattern) = close!(pattern)
Base.isopen(pattern::PhoneticPattern) = !pattern.closed

mutable struct PhoneticRuleSet
    handle::Ptr{Cvoid}
    closed::Bool
end

@doc "Return the stable native ABI generation." abi_version
@doc "Return the additive native API revision." api_revision
@doc "Return the compiled native feature-bit set." build_features
@doc "Capture an independently owned immutable transducer revision." snapshot
@doc "Return a transducer's byte, Unicode-scalar, or u64 unit domain." unit_domain
@doc "Start a lazy snapshot-consistent query selected by the input's Julia type." query
@doc "Copy and settle one bounded native cursor batch." next_batch!
@doc "Consume callback-scoped zero-copy batches and close the cursor." reduce_batches!
@doc "Copy a callback-scoped borrowed match into an independently owned Match." materialize
@doc "Deterministically release an owned native handle; repeated calls are harmless." close!
@doc "A lexical zero-copy descriptor valid only during one reducer callback." BorrowedMatch
@doc "A lexical read-only sequence of borrowed matches for one reducer callback." BorrowedBatch
@doc "An exclusive one-shot fuzzy result iterator retaining its query-start snapshot." QueryCursor
@doc "A shareable algorithm configuration retaining a versioned dictionary provider." Transducer
@doc "An immutable compiled phonetic-language automaton." PhoneticPattern
@doc "An immutable parsed or built-in phonetic rewrite-rule set." PhoneticRuleSet

function PhoneticRuleSet(source::AbstractString)
    bytes = text_bytes(source)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_rules_parse), Cint,
        (Ptr{UInt8}, Csize_t, Ref{Ptr{Cvoid}}),
        isempty(bytes) ? C_NULL : pointer(bytes), length(bytes), output),
        :llev_phonetic_rules_parse)
    value = PhoneticRuleSet(output[], false)
    finalizer(close!, value)
    value
end

function PhoneticRuleSet(kind::PhoneticRuleSetKind)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    checked(ccall(native(:llev_phonetic_rules_builtin), Cint,
        (UInt32, Ref{Ptr{Cvoid}}), UInt32(kind), output), :llev_phonetic_rules_builtin)
    value = PhoneticRuleSet(output[], false)
    finalizer(close!, value)
    value
end

function require_open(rules::PhoneticRuleSet)
    rules.closed &&
        throw(NativeError(Int32(STATUS_CLOSED), :phonetic_rules, "rule set is closed"))
    rules.handle
end

function Base.length(rules::PhoneticRuleSet)
    output = Ref{Csize_t}(0)
    checked(ccall(native(:llev_phonetic_rules_len), Cint,
        (Ptr{Cvoid}, Ref{Csize_t}), require_open(rules), output),
        :llev_phonetic_rules_len)
    Int(output[])
end

function (rules::PhoneticRuleSet)(input::AbstractString)
    bytes = text_bytes(input)
    output = Ref(OwnedString(C_NULL, 0))
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_rules_apply), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ref{OwnedString}), require_open(rules),
        isempty(bytes) ? C_NULL : pointer(bytes), length(bytes), output),
        :llev_phonetic_rules_apply)
    try
        output[].len == 0 ? "" : unsafe_string(output[].data, output[].len)
    finally
        ccall(native(:llev_owned_string_free), Cvoid, (Ref{OwnedString},), output)
    end
end

function close!(rules::PhoneticRuleSet)
    rules.closed && return nothing
    handle = rules.handle
    rules.handle = C_NULL
    rules.closed = true
    handle == C_NULL || ccall(native(:llev_phonetic_rules_free), Cvoid,
        (Ptr{Cvoid},), handle)
    nothing
end

Base.close(rules::PhoneticRuleSet) = close!(rules)
Base.isopen(rules::PhoneticRuleSet) = !rules.closed

function __init__()
    REDUCER_CALLBACK[] = @cfunction(reducer_callback, Cint,
        (Ptr{Cvoid}, Ptr{RawMatch}, Csize_t))
    abi_version() == ABI_VERSION || error("liblevenshtein native ABI version mismatch")
    api_revision() >= API_REVISION || error("liblevenshtein native API revision is too old")
end

end
