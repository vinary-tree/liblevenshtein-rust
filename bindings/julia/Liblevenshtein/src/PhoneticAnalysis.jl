"""Native IPA feature-dimension costs. Each field must be finite and nonnegative."""
struct PhoneticFeatureWeights
    voicing::Cdouble
    place_step::Cdouble
    manner_default::Cdouble
    manner_table_scale::Cdouble
    vowel_height_step::Cdouble
    vowel_backness_step::Cdouble
    vowel_rounding::Cdouble
end

function PhoneticFeatureWeights(;
    voicing::Real=0.1,
    place_step::Real=0.15,
    manner_default::Real=0.5,
    manner_table_scale::Real=1.0,
    vowel_height_step::Real=0.15,
    vowel_backness_step::Real=0.15,
    vowel_rounding::Real=0.1)
    values = (voicing, place_step, manner_default, manner_table_scale,
        vowel_height_step, vowel_backness_step, vowel_rounding)
    all(value -> isfinite(value) && value >= 0, values) ||
        throw(ArgumentError("phonetic feature weights must be finite and nonnegative"))
    converted = map(Cdouble, values)
    all(isfinite, converted) ||
        throw(ArgumentError("phonetic feature weights exceed Float64 range"))
    PhoneticFeatureWeights(converted...)
end

function phonetic_weights_pointer(weights::Union{Nothing,PhoneticFeatureWeights})
    storage = Ref(weights === nothing ? PhoneticFeatureWeights() : weights)
    address = weights === nothing ? Ptr{PhoneticFeatureWeights}(C_NULL) :
        Base.unsafe_convert(Ptr{PhoneticFeatureWeights}, storage)
    storage, address
end

"""Native articulatory distance between two Unicode scalars, optionally reweighted."""
function articulatory_distance(source::Char, target::Char;
    weights::Union{Nothing,PhoneticFeatureWeights}=nothing)
    storage, address = phonetic_weights_pointer(weights)
    output = Ref{Cdouble}(0)
    GC.@preserve storage checked(ccall(native(:llev_phonetic_articulatory_distance), Cint,
        (UInt32, UInt32, Ptr{PhoneticFeatureWeights}, Ref{Cdouble}),
        UInt32(source), UInt32(target), address, output),
        :llev_phonetic_articulatory_distance)
    Float64(output[])
end

"""Native articulatory edit distance with exact source/target scalar work ceiling."""
function articulatory_edit_distance(source::AbstractString, target::AbstractString;
    weights::Union{Nothing,PhoneticFeatureWeights}=nothing,
    max_cells::Integer=4_000_000)
    max_cells > 0 || throw(ArgumentError("max_cells must be positive"))
    ceiling = checked_csize(max_cells, "max_cells")
    source_bytes = text_bytes(source)
    target_bytes = text_bytes(target)
    storage, address = phonetic_weights_pointer(weights)
    output = Ref{Cdouble}(0)
    GC.@preserve source_bytes target_bytes storage checked(ccall(
        native(:llev_phonetic_articulatory_edit_distance), Cint,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t,
            Ptr{PhoneticFeatureWeights}, Csize_t, Ref{Cdouble}),
        isempty(source_bytes) ? C_NULL : pointer(source_bytes), length(source_bytes),
        isempty(target_bytes) ? C_NULL : pointer(target_bytes), length(target_bytes),
        address, ceiling, output), :llev_phonetic_articulatory_edit_distance)
    Float64(output[])
end

"""Count syllables using English spelling heuristics or IPA transcription rules."""
function syllable_count(input::AbstractString; ipa::Bool=false,
    max_input_scalars::Integer=1_000_000)
    max_input_scalars > 0 || throw(ArgumentError("max_input_scalars must be positive"))
    ceiling = checked_csize(max_input_scalars, "max_input_scalars")
    bytes = text_bytes(input)
    output = Ref{Csize_t}(0)
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_syllable_count), Cint,
        (Ptr{UInt8}, Csize_t, UInt8, Csize_t, Ref{Csize_t}),
        isempty(bytes) ? C_NULL : pointer(bytes), length(bytes),
        ipa ? UInt8(1) : UInt8(0), ceiling, output),
        :llev_phonetic_syllable_count)
    Int(output[])
end

"""Return native syllable starts as zero-based Unicode-scalar offsets.

The boundary vector can differ in length from `syllable_count` because the two
native heuristics expose distinct structural and count observations.
"""
function syllable_boundaries(input::AbstractString; ipa::Bool=false,
    max_input_scalars::Integer=1_000_000)
    max_input_scalars > 0 || throw(ArgumentError("max_input_scalars must be positive"))
    ceiling = checked_csize(max_input_scalars, "max_input_scalars")
    bytes = text_bytes(input)
    count = length(input)
    count <= max_input_scalars || throw(NativeError(Int32(STATUS_LIMIT_EXCEEDED),
        :llev_phonetic_syllable_boundaries,
        "syllable input exceeds max_input_scalars"))
    offsets = Vector{Csize_t}(undef, count)
    output = Ref{Csize_t}(0)
    GC.@preserve bytes offsets checked(ccall(native(:llev_phonetic_syllable_boundaries), Cint,
        (Ptr{UInt8}, Csize_t, UInt8, Csize_t, Ptr{Csize_t}, Csize_t, Ref{Csize_t}),
        isempty(bytes) ? C_NULL : pointer(bytes), length(bytes),
        ipa ? UInt8(1) : UInt8(0), ceiling,
        isempty(offsets) ? C_NULL : pointer(offsets), length(offsets), output),
        :llev_phonetic_syllable_boundaries)
    Int.(view(offsets, 1:Int(output[])))
end
