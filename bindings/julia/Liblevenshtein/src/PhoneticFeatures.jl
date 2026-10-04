"""Stable names for native IPA feature bits, in published C ABI order."""
const PHONETIC_FEATURE_NAMES = (
    :Voiced, :Voiceless, :Stop, :Fricative, :Affricate, :Nasal,
    :Approximant, :Lateral, :Rhotic, :Bilabial, :Labiodental,
    :Dental, :Alveolar, :PostAlveolar, :Palatal, :Velar, :Glottal,
    :Vowel, :Consonant, :High, :Mid, :Low, :Front, :Central,
    :Back, :Rounded, :Unrounded, :Sibilant, :Aspirated, :Tense,
    :Pharyngealized, :Labialized, :Velarized, :Retroflex, :Uvular,
    :Pharyngeal, :Epiglottal, :Tap, :Trill, :Ejective, :Implosive,
    :Click)

function feature_mask(features)
    bits = UInt64(0)
    for feature in features
        feature isa Symbol || throw(ArgumentError("phonetic feature names must be Symbols"))
        index = findfirst(==(feature), PHONETIC_FEATURE_NAMES)
        index === nothing && throw(ArgumentError("unknown phonetic feature: $feature"))
        bits |= UInt64(1) << (index - 1)
    end
    bits
end

"""Return native IPA feature names for one Unicode scalar as a `Set{Symbol}`."""
function phonetic_features(character::Char)
    bits = Ref{UInt64}(0)
    checked(ccall(native(:llev_phonetic_features), Cint,
        (UInt32, Ref{UInt64}), UInt32(character), bits),
        :llev_phonetic_features)
    Set(PHONETIC_FEATURE_NAMES[index] for index in eachindex(PHONETIC_FEATURE_NAMES)
        if bits[] & (UInt64(1) << (index - 1)) != 0)
end

function copied_native_chars(symbol::Symbol, first::UInt64, selector::UInt8=UInt8(0))
    count = Ref{Csize_t}(0)
    status = if symbol === :llev_phonetic_chars_with_features
        ccall(native(symbol), Cint, (UInt64, UInt8, Ptr{UInt32}, Csize_t,
            Ref{Csize_t}), first, selector, C_NULL, 0, count)
    else
        ccall(native(symbol), Cint, (UInt32, Ptr{UInt32}, Csize_t,
            Ref{Csize_t}), UInt32(first), C_NULL, 0, count)
    end
    if status != Int32(STATUS_LIMIT_EXCEEDED)
        checked(status, symbol)
    end
    count[] == 0 && return Char[]
    output = Vector{UInt32}(undef, Int(count[]))
    checked(if symbol === :llev_phonetic_chars_with_features
        ccall(native(symbol), Cint, (UInt64, UInt8, Ptr{UInt32}, Csize_t,
            Ref{Csize_t}), first, selector, output, length(output), count)
    else
        ccall(native(symbol), Cint, (UInt32, Ptr{UInt32}, Csize_t,
            Ref{Csize_t}), UInt32(first), output, length(output), count)
    end, symbol)
    Char.(view(output, 1:Int(count[])))
end

"""Find native-table characters with all or any selected IPA features."""
function characters_with_features(features; any::Bool=false)
    copied_native_chars(:llev_phonetic_chars_with_features,
        feature_mask(features), UInt8(any))
end

"""Return native feature-similar characters, excluding the input scalar."""
similar_phonetic_chars(character::Char) =
    copied_native_chars(:llev_phonetic_similar_chars, UInt64(character))

"""Return an optional native same-place-and-manner voicing counterpart."""
function voicing_pair(character::Char)
    result = Ref{UInt32}(0)
    found = Ref{UInt8}(0)
    checked(ccall(native(:llev_phonetic_voicing_pair), Cint,
        (UInt32, Ref{UInt32}, Ref{UInt8}), UInt32(character), result, found),
        :llev_phonetic_voicing_pair)
    found[] == 0 ? nothing : Char(result[])
end

function native_phonetic_relation(source::Char, target::Char, selector::UInt8)
    result = Ref{UInt8}(0)
    checked(ccall(native(:llev_phonetic_feature_relation), Cint,
        (UInt32, UInt32, UInt8, Ref{UInt8}), UInt32(source), UInt32(target),
        selector, result), :llev_phonetic_feature_relation)
    result[] != 0
end

"""Native shared-feature similarity (distinct from edit substitution cost)."""
are_phonetically_similar(source::Char, target::Char) =
    native_phonetic_relation(source, target, 0x00)

"""Native zero-cost substitution relation used by phonetic matching."""
is_free_phonetic_substitution(source::Char, target::Char) =
    native_phonetic_relation(source, target, 0x01)

"""Native feature-based character expansion in its native order."""
expand_feature_based(character::Char) =
    copied_native_chars(:llev_phonetic_expand_feature_based, UInt64(character))

"""Feature-weighted distance between two explicit IPA feature sets."""
function feature_set_distance(source, target;
    weights::Union{Nothing,PhoneticFeatureWeights}=nothing)
    source_bits = feature_mask(source)
    target_bits = feature_mask(target)
    storage, address = phonetic_weights_pointer(weights)
    output = Ref{Cdouble}(0)
    GC.@preserve storage checked(ccall(native(:llev_phonetic_feature_set_distance), Cint,
        (UInt64, UInt64, Ptr{PhoneticFeatureWeights}, Ref{Cdouble}),
        source_bits, target_bits, address, output),
        :llev_phonetic_feature_set_distance)
    Float64(output[])
end
