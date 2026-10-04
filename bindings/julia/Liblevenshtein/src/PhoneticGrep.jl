"""An owned copied native word-boundary phonetic grep result.

`line_number` is one-based. `start_byte` and `end_byte` are zero-based,
end-exclusive offsets within that line; `text` is an independent Julia copy.
"""
struct PhoneticGrepMatch
    text::String
    line_number::Int
    start_byte::Int
    end_byte::Int
    distance::Int
end

struct RawPhoneticGrepMatch
    line_number::Csize_t
    start_byte::Csize_t
    end_byte::Csize_t
    distance::UInt8
    reserved::NTuple{7,UInt8}
end

"""Reusable native phonetic word-boundary matcher with optional rewrite rules."""
mutable struct PhoneticGrep
    handle::Ptr{Cvoid}
    closed::Bool
end

function PhoneticGrep(pattern::AbstractString;
    rules::Union{Nothing,PhoneticRuleSet}=nothing,
    max_distance::Integer=0,
    algorithm::Algorithm=ALGORITHM_STANDARD,
    case_insensitive::Bool=false)
    0 <= max_distance <= typemax(UInt8) ||
        throw(ArgumentError("max_distance must fit UInt8"))
    bytes = text_bytes(pattern)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    rules_handle = rules === nothing ? Ptr{Cvoid}(C_NULL) : require_open(rules)
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_grep_new), Cint,
        (Ptr{UInt8}, Csize_t, Ptr{Cvoid}, UInt8, UInt32, UInt8, Ref{Ptr{Cvoid}}),
        isempty(bytes) ? C_NULL : pointer(bytes), length(bytes), rules_handle,
        UInt8(max_distance), UInt32(algorithm), UInt8(case_insensitive), output),
        :llev_phonetic_grep_new)
    value = PhoneticGrep(output[], false)
    finalizer(close!, value)
    value
end

function require_open(grep::PhoneticGrep)
    grep.closed &&
        throw(NativeError(Int32(STATUS_CLOSED), :phonetic_grep, "grep matcher is closed"))
    grep.handle
end

function close!(grep::PhoneticGrep)
    grep.closed && return nothing
    handle = grep.handle
    grep.handle = C_NULL
    grep.closed = true
    handle == C_NULL || ccall(native(:llev_phonetic_grep_free), Cvoid, (Ptr{Cvoid},), handle)
    nothing
end

Base.close(grep::PhoneticGrep) = close!(grep)
Base.isopen(grep::PhoneticGrep) = !grep.closed

"""Return `(effective, local_override)` edit bounds; the override is `nothing` when absent."""
function distance_config(grep::PhoneticGrep)
    effective = Ref{UInt8}(0)
    local_distance = Ref{UInt8}(0)
    has_local = Ref{UInt8}(0)
    checked(ccall(native(:llev_phonetic_grep_distance_config), Cint,
        (Ptr{Cvoid}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}),
        require_open(grep), effective, local_distance, has_local),
        :llev_phonetic_grep_distance_config)
    (effective=Int(effective[]), local_override=has_local[] == 0 ? nothing : Int(local_distance[]))
end

"""Return a candidate's native edit distance, or `nothing` outside the bound."""
function match_distance(grep::PhoneticGrep, candidate::AbstractString;
    max_candidate_bytes::Integer=1_000_000)
    max_candidate_bytes > 0 || throw(ArgumentError("max_candidate_bytes must be positive"))
    ceiling = checked_csize(max_candidate_bytes, "max_candidate_bytes")
    bytes = text_bytes(candidate)
    distance = Ref{UInt8}(0)
    found = Ref{UInt8}(0)
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_grep_matches), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Csize_t, Ref{UInt8}, Ref{UInt8}),
        require_open(grep), isempty(bytes) ? C_NULL : pointer(bytes), length(bytes),
        ceiling, distance, found), :llev_phonetic_grep_matches)
    found[] == 0 ? nothing : Int(distance[])
end

Base.in(candidate::AbstractString, grep::PhoneticGrep) =
    match_distance(grep, candidate) !== nothing

function native_grep_scan(grep::PhoneticGrep, text::AbstractString, symbol::Symbol;
    max_input_bytes::Integer=1_000_000, max_matches::Integer=1_000_000)
    max_input_bytes > 0 || throw(ArgumentError("max_input_bytes must be positive"))
    max_matches >= 0 || throw(ArgumentError("max_matches must be nonnegative"))
    max_matches <= typemax(Int) || throw(OverflowError("max_matches exceeds Int"))
    ceiling = checked_csize(max_input_bytes, "max_input_bytes")
    match_limit = checked_csize(max_matches, "max_matches")
    bytes = text_bytes(text)
    slots = min(length(bytes), Int(match_limit))
    output = Vector{RawPhoneticGrepMatch}(undef, slots)
    count = Ref{Csize_t}(0)
    GC.@preserve bytes output checked(ccall(native(symbol), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Csize_t,
            Ptr{RawPhoneticGrepMatch}, Csize_t, Ref{Csize_t}),
        require_open(grep), isempty(bytes) ? C_NULL : pointer(bytes), length(bytes),
        ceiling, isempty(output) ? C_NULL : pointer(output), length(output), count), symbol)
    resize!(output, Int(count[]))
    output
end

function copied_grep_match(raw::RawPhoneticGrepMatch, line_bytes::Vector{UInt8})
    first = Int(raw.start_byte) + 1
    last = Int(raw.end_byte)
    1 <= first <= last + 1 <= length(line_bytes) + 1 ||
        throw(ArgumentError("native phonetic grep returned an invalid byte range"))
    PhoneticGrepMatch(String(copy(@view line_bytes[first:last])),
        Int(raw.line_number), Int(raw.start_byte), last, Int(raw.distance))
end

"""Copy all non-overlapping matches in one UTF-8 line with native byte offsets."""
function scan_line(grep::PhoneticGrep, line::AbstractString; kwargs...)
    raw = native_grep_scan(grep, line, :llev_phonetic_grep_scan_line; kwargs...)
    bytes = text_bytes(line)
    [copied_grep_match(match, bytes) for match in raw]
end

"""Copy native word-boundary matches in all logical lines of a UTF-8 document."""
function scan_text(grep::PhoneticGrep, document::AbstractString; kwargs...)
    raw = native_grep_scan(grep, document, :llev_phonetic_grep_scan_text; kwargs...)
    lines = [Vector{UInt8}(codeunits(line)) for line in split(String(document), '\n'; keepempty=true)]
    result = PhoneticGrepMatch[]
    sizehint!(result, length(raw))
    for match in raw
        index = Int(match.line_number)
        1 <= index <= length(lines) ||
            throw(ArgumentError("native phonetic grep returned an invalid line number"))
        push!(result, copied_grep_match(match, lines[index]))
    end
    result
end
