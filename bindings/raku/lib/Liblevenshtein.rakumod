unit module Liblevenshtein;

use NativeCall;
need Liblevenshtein::GeneratedAbi;

# Generated ABI declarations live in a separately auditable module. Raku does
# not re-export names imported with `use`, so the public facade deliberately
# aliases each generated type and value into its own default export set.
our constant ABI-VERSION is export = Liblevenshtein::GeneratedAbi::ABI-VERSION;
our constant API-REVISION is export = Liblevenshtein::GeneratedAbi::API-REVISION;
our constant DEFAULT-MATCH-BATCH is export =
    Liblevenshtein::GeneratedAbi::DEFAULT-MATCH-BATCH;
our constant BUILD-FEATURE-CORE is export =
    Liblevenshtein::GeneratedAbi::BUILD-FEATURE-CORE;
our constant BUILD-FEATURE-PHONETIC is export =
    Liblevenshtein::GeneratedAbi::BUILD-FEATURE-PHONETIC;

our constant Status is export = Liblevenshtein::GeneratedAbi::Status;
our constant OK is export = Liblevenshtein::GeneratedAbi::OK;
our constant END is export = Liblevenshtein::GeneratedAbi::END;
our constant INVALID-ARGUMENT is export =
    Liblevenshtein::GeneratedAbi::INVALID-ARGUMENT;
our constant INVALID-UTF8 is export = Liblevenshtein::GeneratedAbi::INVALID-UTF8;
our constant NULL-POINTER is export = Liblevenshtein::GeneratedAbi::NULL-POINTER;
our constant PANIC is export = Liblevenshtein::GeneratedAbi::PANIC;
our constant UNSUPPORTED is export = Liblevenshtein::GeneratedAbi::UNSUPPORTED;
our constant IO-ERROR is export = Liblevenshtein::GeneratedAbi::IO-ERROR;
our constant CLOSED is export = Liblevenshtein::GeneratedAbi::CLOSED;
our constant LIMIT-EXCEEDED is export =
    Liblevenshtein::GeneratedAbi::LIMIT-EXCEEDED;
our constant PROVIDER-ERROR is export =
    Liblevenshtein::GeneratedAbi::PROVIDER-ERROR;
our constant BATCH-IN-USE is export =
    Liblevenshtein::GeneratedAbi::BATCH-IN-USE;
our constant DOMAIN-MISMATCH is export =
    Liblevenshtein::GeneratedAbi::DOMAIN-MISMATCH;

our constant Algorithm is export = Liblevenshtein::GeneratedAbi::Algorithm;
our constant STANDARD is export = Liblevenshtein::GeneratedAbi::STANDARD;
our constant TRANSPOSITION is export = Liblevenshtein::GeneratedAbi::TRANSPOSITION;
our constant MERGE-AND-SPLIT is export =
    Liblevenshtein::GeneratedAbi::MERGE-AND-SPLIT;
our constant DAMERAU-LEVENSHTEIN is export =
    Liblevenshtein::GeneratedAbi::DAMERAU-LEVENSHTEIN;

our constant QueryOrder is export = Liblevenshtein::GeneratedAbi::QueryOrder;
our constant TRAVERSAL is export = Liblevenshtein::GeneratedAbi::TRAVERSAL;
our constant DISTANCE-THEN-TERM is export =
    Liblevenshtein::GeneratedAbi::DISTANCE-THEN-TERM;

our constant PhoneticRuleSetKind is export =
    Liblevenshtein::GeneratedAbi::PhoneticRuleSetKind;
our constant ENGLISH-ORTHOGRAPHY is export =
    Liblevenshtein::GeneratedAbi::ENGLISH-ORTHOGRAPHY;
our constant ENGLISH-PHONETIC is export =
    Liblevenshtein::GeneratedAbi::ENGLISH-PHONETIC;

our constant OperationApplicability is export =
    Liblevenshtein::GeneratedAbi::OperationApplicability;
our constant APPLICABILITY-ANY is export =
    Liblevenshtein::GeneratedAbi::APPLICABILITY-ANY;
our constant APPLICABILITY-EQUAL is export =
    Liblevenshtein::GeneratedAbi::APPLICABILITY-EQUAL;
our constant APPLICABILITY-ADJACENT-TRANSPOSE is export =
    Liblevenshtein::GeneratedAbi::APPLICABILITY-ADJACENT-TRANSPOSE;
our constant APPLICABILITY-LISTED is export =
    Liblevenshtein::GeneratedAbi::APPLICABILITY-LISTED;

our constant UniversalVariant is export =
    Liblevenshtein::GeneratedAbi::UniversalVariant;
our constant UNIVERSAL-STANDARD is export =
    Liblevenshtein::GeneratedAbi::UNIVERSAL-STANDARD;
our constant UNIVERSAL-TRANSPOSITION is export =
    Liblevenshtein::GeneratedAbi::UNIVERSAL-TRANSPOSITION;
our constant UNIVERSAL-MERGE-AND-SPLIT is export =
    Liblevenshtein::GeneratedAbi::UNIVERSAL-MERGE-AND-SPLIT;

module InteropAccess {
    use Vinary::Tree::Interop;

    our constant ResourceType = Resource;
    our constant DictionaryType = Dictionary;
    our constant RawResourceType = RawResource;
    our constant UnitDomainType = UnitDomain;
    our constant ByteDomain = BYTE;
    our constant UnicodeDomain = UNICODE-SCALAR;
    our constant U64Domain = U64;
}

our constant UnitDomain is export = InteropAccess::UnitDomainType;
our constant BYTE is export = InteropAccess::ByteDomain;
our constant UNICODE-SCALAR is export = InteropAccess::UnicodeDomain;
our constant U64 is export = InteropAccess::U64Domain;

class X::Liblevenshtein is Exception {
    has Int:D $.status is required;
    has Str:D $.operation is required;
    has Str:D $.detail = '';

    method message(--> Str:D) {
        my $base = "liblevenshtein operation '$!operation' failed with status $!status";
        $!detail.chars ?? "$base: $!detail" !! $base
    }
}

class RawMatch is repr('CStruct') is export {
    has Pointer $.term-data;
    has size_t $.term-len;
    has size_t $.byte-len;
    has size_t $.distance;
    has uint64 $.id;
    has uint32 $.unit-domain;
    has uint8 $.has-id;
    has uint8 $.reserved0;
    has uint8 $.reserved1;
    has uint8 $.reserved2;
}

class RawBatch is repr('CStruct') is export {
    has Pointer $.matches;
    has size_t $.len;
    has uint64 $.generation;
}

class RawQueryCacheStats is repr('CStruct') is export {
    has uint64 $.requests;
    has uint64 $.hits;
    has uint64 $.misses;
    has uint64 $.admissions;
    has uint64 $.rejections;
    has uint64 $.evictions;
    has size_t $.resident-entries;
    has size_t $.resident-weight;
}

class QueryCacheStats is export {
    has UInt:D $.requests is required;
    has UInt:D $.hits is required;
    has UInt:D $.misses is required;
    has UInt:D $.admissions is required;
    has UInt:D $.rejections is required;
    has UInt:D $.evictions is required;
    has Int:D $.resident-entries is required;
    has Int:D $.resident-weight is required;
}

class OwnedString is repr('CStruct') is export {
    has Pointer $.data;
    has size_t $.len;
}

class PhoneticFeatureWeights is repr('CStruct') is export {
    has num64 $.voicing;
    has num64 $.place-step;
    has num64 $.manner-default;
    has num64 $.manner-table-scale;
    has num64 $.vowel-height-step;
    has num64 $.vowel-backness-step;
    has num64 $.vowel-rounding;
}

class RawPhoneticGrepMatch is repr('CStruct') {
    has size_t $.line-number;
    has size_t $.start-byte;
    has size_t $.end-byte;
    has uint8 $.distance;
    has uint8 $.reserved0;
    has uint8 $.reserved1;
    has uint8 $.reserved2;
    has uint8 $.reserved3;
    has uint8 $.reserved4;
    has uint8 $.reserved5;
    has uint8 $.reserved6;
}

class PhoneticGrepHit is export {
    has Int:D $.line-number is required;
    has Int:D $.start-byte is required;
    has Int:D $.end-byte is required;
    has Int:D $.distance is required;
    has Str:D $.text is required;
}

class RawPhoneticOnlineMatch is repr('CStruct') {
    has size_t $.byte-start;
    has size_t $.byte-end;
    has size_t $.char-start;
    has size_t $.char-end;
    has Pointer $.original-data;
    has size_t $.original-len;
    has Pointer $.normalized-data;
    has size_t $.normalized-len;
    has uint8 $.distance;
    has uint8 $.reserved0;
    has uint8 $.reserved1;
    has uint8 $.reserved2;
    has uint8 $.reserved3;
    has uint8 $.reserved4;
    has uint8 $.reserved5;
    has uint8 $.reserved6;
}

class PhoneticOnlineHit is export {
    has Int:D $.byte-start is required;
    has Int:D $.byte-end is required;
    has Int:D $.char-start is required;
    has Int:D $.char-end is required;
    has Str:D $.original-text is required;
    has Str:D $.normalized-text is required;
    has Int:D $.distance is required;
}

class RawPhoneticTokenDetail is repr('CStruct') {
    has size_t $.token-index;
    has size_t $.byte-start;
    has size_t $.byte-end;
    has Pointer $.original-data;
    has size_t $.original-len;
    has Pointer $.normalized-data;
    has size_t $.normalized-len;
    has uint8 $.distance;
    has uint8 $.reserved0;
    has uint8 $.reserved1;
    has uint8 $.reserved2;
    has uint8 $.reserved3;
    has uint8 $.reserved4;
    has uint8 $.reserved5;
    has uint8 $.reserved6;
}

class RawPhoneticTokenMatch is repr('CStruct') {
    has size_t $.byte-start;
    has size_t $.byte-end;
    has uint8 $.total-distance;
    has uint8 $.reserved0;
    has uint8 $.reserved1;
    has uint8 $.reserved2;
    has uint8 $.reserved3;
    has uint8 $.reserved4;
    has uint8 $.reserved5;
    has uint8 $.reserved6;
    has Pointer $.matched-data;
    has size_t $.matched-len;
    has Pointer $.details;
    has size_t $.detail-count;
}

class PhoneticTokenDetail is export {
    has Int:D $.token-index is required;
    has Int:D $.byte-start is required;
    has Int:D $.byte-end is required;
    has Str:D $.original-text is required;
    has Str:D $.normalized-text is required;
    has Int:D $.distance is required;
}

class PhoneticTokenHit is export {
    has Int:D $.byte-start is required;
    has Int:D $.byte-end is required;
    has Int:D $.total-distance is required;
    has Str:D $.matched-text is required;
    has List:D $.details is required;
}

class RawPhoneticUtf8View is repr('CStruct') {
    has size_t $.data;
    has size_t $.len;
}

class RawPhoneticCandidate is repr('CStruct') {
    has Pointer $.term-data;
    has size_t $.term-len;
    has size_t $.distance;
    has Pointer $.normalized-data;
    has size_t $.normalized-len;
}

class PhoneticCandidate is export {
    has Str:D $.term is required;
    has Int:D $.distance is required;
    has Str:D $.normalized-form is required;
}

class PhoneticExpansionLimits is repr('CStruct') is export {
    has size_t $.max-input-scalars;
    has size_t $.max-rules;
    has size_t $.max-rule-units;
    has size_t $.max-nodes;
    has size_t $.max-output-bytes;
}

class Match is export {
    has Mu $.term is required;
    has Int:D $.distance is required;
    has Mu $.id;
    has UnitDomain:D $.unit-domain is required;
}

# These layouts mirror liblevenshtein_abi.h; the public enum values above are
# generated from api.json. Pointer slots use pointer-width size_t because
# Rakudo cannot initialize a Pointer attribute in a CStruct constructor.
# The exact owner of each address remains live until the native constructor
# has copied the borrowed descriptor tree.
class RawAutomatonLimits is repr('CStruct') is export {
    has size_t $.max-source-units;
    has size_t $.max-target-units;
    has size_t $.max-retained-cells;
    has size_t $.max-step-work-units;
}

class RawGeneralizedRestriction is repr('CStruct') is export {
    has size_t $.source-data;
    has size_t $.source-len;
    has size_t $.target-data;
    has size_t $.target-len;
}

class RawGeneralizedOperation is repr('CStruct') is export {
    has size_t $.consume-source;
    has size_t $.consume-target;
    has num64 $.weight;
    has size_t $.name-data;
    has size_t $.name-len;
    has uint32 $.applicability;
    has uint32 $.reserved;
    has size_t $.restrictions;
    has size_t $.restriction-count;
}

class RawUniversalEquivalence is repr('CStruct') is export {
    has uint64 $.source;
    has uint64 $.target;
}

class RawGeneralizedObservation is repr('CStruct') is export {
    has size_t $.consumed-target-len;
    has size_t $.active-positions;
    has size_t $.scaled-distance;
    has uint32 $.scale-denominator;
    has uint8 $.current-row-nonempty;
    has uint8 $.accepting;
    has uint8 $.has-distance;
    has uint8 $.reserved;
}

class RawUniversalObservation is repr('CStruct') is export {
    has size_t $.consumed-target-len;
    has size_t $.source-len;
    has uint8 $.alive;
    has uint8 $.accepting;
    has uint8 $.reserved0;
    has uint8 $.reserved1;
    has uint8 $.reserved2;
    has uint8 $.reserved3;
    has uint8 $.reserved4;
    has uint8 $.reserved5;
}

sub native-library(--> Str:D) {
    return %*ENV<LIBLEVENSHTEIN_LIBRARY>
        if %*ENV<LIBLEVENSHTEIN_LIBRARY>:exists;
    $*DISTRO.is-win ?? 'liblevenshtein.dll' !!
        $*KERNEL.name eq 'darwin' ?? 'libliblevenshtein.dylib' !!
        'libliblevenshtein.so'
}

sub llev-abi-version(--> uint32)
    is native(&native-library) is symbol('llev_abi_version') { * }
sub llev-api-revision(--> uint32)
    is native(&native-library) is symbol('llev_api_revision') { * }
sub llev-build-features(--> uint64)
    is native(&native-library) is symbol('llev_build_features') { * }
sub llev-last-error-message(--> Str)
    is native(&native-library) is symbol('llev_last_error_message') { * }
sub llev-distance(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_distance') { * }
sub llev-distance-threshold(Pointer, size_t, Pointer, size_t, size_t --> size_t)
    is native(&native-library) is symbol('llev_distance_threshold') { * }
sub llev-damerau-distance(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_damerau_distance') { * }
sub llev-damerau-distance-threshold(Pointer, size_t, Pointer, size_t, size_t --> size_t)
    is native(&native-library) is symbol('llev_damerau_distance_threshold') { * }
sub llev-true-damerau-distance(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_true_damerau_distance') { * }
sub llev-true-damerau-distance-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_true_damerau_distance_threshold') { * }
sub llev-merge-and-split-distance(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_merge_and_split_distance') { * }
sub llev-merge-and-split-distance-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_merge_and_split_distance_threshold') { * }
sub llev-hamming-distance(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_hamming_distance') { * }
sub llev-hamming-distance-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_hamming_distance_threshold') { * }
sub llev-indel-distance(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_indel_distance') { * }
sub llev-indel-distance-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_indel_distance_threshold') { * }
sub llev-affine-gap-distance(
    Pointer, size_t, Pointer, size_t, size_t, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_affine_gap_distance') { * }
sub llev-affine-gap-distance-threshold(
    Pointer, size_t, Pointer, size_t, size_t, size_t, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_affine_gap_distance_threshold') { * }

sub llev-distance-bytes(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_distance_bytes') { * }
sub llev-distance-bytes-threshold(Pointer, size_t, Pointer, size_t, size_t --> size_t)
    is native(&native-library) is symbol('llev_distance_bytes_threshold') { * }
sub llev-distance-u64(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_distance_u64') { * }
sub llev-distance-u64-threshold(Pointer, size_t, Pointer, size_t, size_t --> size_t)
    is native(&native-library) is symbol('llev_distance_u64_threshold') { * }
sub llev-damerau-distance-bytes(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_damerau_distance_bytes') { * }
sub llev-damerau-distance-bytes-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_damerau_distance_bytes_threshold') { * }
sub llev-damerau-distance-u64(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_damerau_distance_u64') { * }
sub llev-damerau-distance-u64-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_damerau_distance_u64_threshold') { * }
sub llev-true-damerau-distance-bytes(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_true_damerau_distance_bytes') { * }
sub llev-true-damerau-distance-bytes-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_true_damerau_distance_bytes_threshold') { * }
sub llev-true-damerau-distance-u64(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_true_damerau_distance_u64') { * }
sub llev-true-damerau-distance-u64-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_true_damerau_distance_u64_threshold') { * }
sub llev-merge-and-split-distance-bytes(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_merge_and_split_distance_bytes') { * }
sub llev-merge-and-split-distance-bytes-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_merge_and_split_distance_bytes_threshold') { * }
sub llev-merge-and-split-distance-u64(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_merge_and_split_distance_u64') { * }
sub llev-merge-and-split-distance-u64-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_merge_and_split_distance_u64_threshold') { * }
sub llev-hamming-distance-bytes(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_hamming_distance_bytes') { * }
sub llev-hamming-distance-bytes-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_hamming_distance_bytes_threshold') { * }
sub llev-hamming-distance-u64(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_hamming_distance_u64') { * }
sub llev-hamming-distance-u64-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_hamming_distance_u64_threshold') { * }
sub llev-indel-distance-bytes(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_indel_distance_bytes') { * }
sub llev-indel-distance-bytes-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_indel_distance_bytes_threshold') { * }
sub llev-indel-distance-u64(Pointer, size_t, Pointer, size_t --> size_t)
    is native(&native-library) is symbol('llev_indel_distance_u64') { * }
sub llev-indel-distance-u64-threshold(
    Pointer, size_t, Pointer, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_indel_distance_u64_threshold') { * }
sub llev-affine-gap-distance-bytes(
    Pointer, size_t, Pointer, size_t, size_t, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_affine_gap_distance_bytes') { * }
sub llev-affine-gap-distance-bytes-threshold(
    Pointer, size_t, Pointer, size_t, size_t, size_t, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_affine_gap_distance_bytes_threshold') { * }
sub llev-affine-gap-distance-u64(
    Pointer, size_t, Pointer, size_t, size_t, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_affine_gap_distance_u64') { * }
sub llev-affine-gap-distance-u64-threshold(
    Pointer, size_t, Pointer, size_t, size_t, size_t, size_t, size_t --> size_t
) is native(&native-library) is symbol('llev_affine_gap_distance_u64_threshold') { * }
sub llev-transducer-new(InteropAccess::RawResourceType, uint32, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_transducer_new') { * }
sub llev-transducer-snapshot(Pointer, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_transducer_snapshot') { * }
sub llev-transducer-free(Pointer)
    is native(&native-library) is symbol('llev_transducer_free') { * }
sub llev-transducer-unit-domain(Pointer, uint32 is rw --> int32)
    is native(&native-library) is symbol('llev_transducer_unit_domain') { * }
sub llev-query-cache-new(Pointer, size_t, size_t, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_query_cache_new') { * }
sub llev-query-cache-clear(Pointer --> int32)
    is native(&native-library) is symbol('llev_query_cache_clear') { * }
sub llev-query-cache-reset-stats(Pointer --> int32)
    is native(&native-library) is symbol('llev_query_cache_reset_stats') { * }
sub llev-query-cache-stats(Pointer, RawQueryCacheStats --> int32)
    is native(&native-library) is symbol('llev_query_cache_stats') { * }
sub llev-query-cache-free(Pointer)
    is native(&native-library) is symbol('llev_query_cache_free') { * }
sub llev-transducer-query-utf8(
    Pointer, Pointer, size_t, size_t, uint32, Pointer is rw --> int32
) is native(&native-library) is symbol('llev_transducer_query_utf8') { * }
sub llev-transducer-query-bytes(
    Pointer, Pointer, size_t, size_t, uint32, Pointer is rw --> int32
) is native(&native-library) is symbol('llev_transducer_query_bytes') { * }
sub llev-transducer-query-u64(
    Pointer, Pointer, size_t, size_t, uint32, Pointer is rw --> int32
) is native(&native-library) is symbol('llev_transducer_query_u64') { * }
sub llev-query-cache-query-utf8(
    Pointer, Pointer, size_t, size_t, uint32, Pointer is rw --> int32
) is native(&native-library) is symbol('llev_query_cache_query_utf8') { * }
sub llev-query-cache-query-bytes(
    Pointer, Pointer, size_t, size_t, uint32, Pointer is rw --> int32
) is native(&native-library) is symbol('llev_query_cache_query_bytes') { * }
sub llev-query-cache-query-u64(
    Pointer, Pointer, size_t, size_t, uint32, Pointer is rw --> int32
) is native(&native-library) is symbol('llev_query_cache_query_u64') { * }
sub llev-query-cursor-next-batch(Pointer, size_t, RawBatch --> int32)
    is native(&native-library) is symbol('llev_query_cursor_next_batch') { * }
sub llev-query-cursor-release-batch(Pointer, uint64 --> int32)
    is native(&native-library) is symbol('llev_query_cursor_release_batch') { * }
sub llev-query-cursor-free(Pointer --> int32)
    is native(&native-library) is symbol('llev_query_cursor_free') { * }
sub llev-generalized-automaton-new(uint8, Pointer, size_t, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_generalized_automaton_new') { * }
sub llev-generalized-automaton-free(Pointer)
    is native(&native-library) is symbol('llev_generalized_automaton_free') { * }
sub llev-generalized-automaton-evaluate-utf8(
    Pointer, Pointer, size_t, Pointer, size_t, RawAutomatonLimits,
    RawGeneralizedObservation --> int32
) is native(&native-library) is symbol('llev_generalized_automaton_evaluate_utf8') { * }
sub llev-generalized-online-new-utf8(
    Pointer, Pointer, size_t, RawAutomatonLimits, Pointer is rw --> int32
) is native(&native-library) is symbol('llev_generalized_online_new_utf8') { * }
sub llev-generalized-online-advance(
    Pointer, uint32, RawGeneralizedObservation --> int32
) is native(&native-library) is symbol('llev_generalized_online_advance') { * }
sub llev-generalized-online-observation(
    Pointer, RawGeneralizedObservation --> int32
) is native(&native-library) is symbol('llev_generalized_online_observation') { * }
sub llev-generalized-online-free(Pointer)
    is native(&native-library) is symbol('llev_generalized_online_free') { * }
sub llev-universal-automaton-new(
    uint8, uint32, uint32, Pointer, size_t, Pointer is rw --> int32
) is native(&native-library) is symbol('llev_universal_automaton_new') { * }
sub llev-universal-automaton-free(Pointer)
    is native(&native-library) is symbol('llev_universal_automaton_free') { * }
sub llev-universal-automaton-evaluate(
    Pointer, uint32, Pointer, size_t, Pointer, size_t, RawAutomatonLimits,
    RawUniversalObservation --> int32
) is native(&native-library) is symbol('llev_universal_automaton_evaluate') { * }
sub llev-universal-online-new(
    Pointer, uint32, Pointer, size_t, RawAutomatonLimits, Pointer is rw --> int32
) is native(&native-library) is symbol('llev_universal_online_new') { * }
sub llev-universal-online-advance(
    Pointer, uint64, RawUniversalObservation --> int32
) is native(&native-library) is symbol('llev_universal_online_advance') { * }
sub llev-universal-online-observation(
    Pointer, RawUniversalObservation --> int32
) is native(&native-library) is symbol('llev_universal_online_observation') { * }
sub llev-universal-online-free(Pointer)
    is native(&native-library) is symbol('llev_universal_online_free') { * }
sub llev-phonetic-pattern-compile-regex(Pointer, size_t, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_pattern_compile_regex') { * }
sub llev-phonetic-pattern-compile-llre(Pointer, size_t, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_pattern_compile_llre') { * }
sub llev-phonetic-pattern-free(Pointer)
    is native(&native-library) is symbol('llev_phonetic_pattern_free') { * }
sub llev-phonetic-pattern-size(Pointer, size_t is rw, size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_pattern_size') { * }
sub llev-phonetic-pattern-matches(Pointer, Pointer, size_t, uint8 is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_pattern_matches') { * }
sub llev-transducer-query-pattern(Pointer, Pointer, uint8, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_transducer_query_pattern') { * }
sub llev-phonetic-rules-parse(Pointer, size_t, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_rules_parse') { * }
sub llev-phonetic-rules-load-file(Pointer, size_t, Pointer, size_t, size_t,
    Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_rules_load_file') { * }
sub llev-phonetic-rules-builtin(uint32, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_rules_builtin') { * }
sub llev-phonetic-rules-language(Pointer, size_t, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_rules_language') { * }
sub llev-phonetic-rules-free(Pointer)
    is native(&native-library) is symbol('llev_phonetic_rules_free') { * }
sub llev-phonetic-rules-len(Pointer, size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_rules_len') { * }
sub llev-phonetic-rules-apply(Pointer, Pointer, size_t, OwnedString --> int32)
    is native(&native-library) is symbol('llev_phonetic_rules_apply') { * }
sub llev-phonetic-articulatory-distance(uint32, uint32, Pointer, num64 is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_articulatory_distance') { * }
sub llev-phonetic-articulatory-edit-distance(Pointer, size_t, Pointer, size_t,
    Pointer, size_t, num64 is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_articulatory_edit_distance') { * }
sub llev-phonetic-syllable-count(Pointer, size_t, uint8, size_t,
    size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_syllable_count') { * }
sub llev-phonetic-syllable-boundaries(Pointer, size_t, uint8, size_t,
    Pointer, size_t, size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_syllable_boundaries') { * }
sub llev-phonetic-features(uint32, uint64 is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_features') { * }
sub llev-phonetic-chars-with-features(uint64, uint8, Pointer, size_t,
    size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_chars_with_features') { * }
sub llev-phonetic-similar-chars(uint32, Pointer, size_t, size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_similar_chars') { * }
sub llev-phonetic-voicing-pair(uint32, uint32 is rw, uint8 is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_voicing_pair') { * }
sub llev-phonetic-feature-relation(uint32, uint32, uint8, uint8 is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_feature_relation') { * }
sub llev-phonetic-expand-feature-based(uint32, Pointer, size_t,
    size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_expand_feature_based') { * }
sub llev-phonetic-feature-set-distance(uint64, uint64, Pointer, num64 is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_feature_set_distance') { * }
sub llev-phonetic-grep-new(Pointer, size_t, Pointer, uint8, uint32, uint8,
    Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_grep_new') { * }
sub llev-phonetic-grep-free(Pointer)
    is native(&native-library) is symbol('llev_phonetic_grep_free') { * }
sub llev-phonetic-grep-distance-config(Pointer, uint8 is rw, uint8 is rw,
    uint8 is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_grep_distance_config') { * }
sub llev-phonetic-grep-matches(Pointer, Pointer, size_t, size_t, uint8 is rw,
    uint8 is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_grep_matches') { * }
sub llev-phonetic-grep-scan-line(Pointer, Pointer, size_t, size_t, Pointer,
    size_t, size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_grep_scan_line') { * }
sub llev-phonetic-grep-scan-text(Pointer, Pointer, size_t, size_t, Pointer,
    size_t, size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_grep_scan_text') { * }
sub llev-phonetic-transducer-new(Pointer, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_transducer_new') { * }
sub llev-phonetic-transducer-free(Pointer)
    is native(&native-library) is symbol('llev_phonetic_transducer_free') { * }
sub llev-phonetic-transducer-feed(Pointer, Pointer, size_t, size_t, size_t,
    OwnedString --> int32)
    is native(&native-library) is symbol('llev_phonetic_transducer_feed') { * }
sub llev-phonetic-transducer-finish(Pointer, size_t, OwnedString --> int32)
    is native(&native-library) is symbol('llev_phonetic_transducer_finish') { * }
sub llev-phonetic-transducer-reset(Pointer --> int32)
    is native(&native-library) is symbol('llev_phonetic_transducer_reset') { * }
sub llev-phonetic-transducer-normalize(Pointer, Pointer, size_t, size_t,
    size_t, OwnedString --> int32)
    is native(&native-library) is symbol('llev_phonetic_transducer_normalize') { * }
sub llev-phonetic-expand(Pointer, size_t, Pointer, Pointer,
    OwnedString --> int32)
    is native(&native-library) is symbol('llev_phonetic_expand') { * }
sub llev-phonetic-expand-with-costs(Pointer, size_t, Pointer, Pointer,
    OwnedString, num64 is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_expand_with_costs') { * }
sub llev-phonetic-online-new(Pointer, size_t, Pointer, uint8, uint8, size_t,
    Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_online_new') { * }
sub llev-phonetic-online-free(Pointer)
    is native(&native-library) is symbol('llev_phonetic_online_free') { * }
sub llev-phonetic-online-normalized-query(Pointer, OwnedString --> int32)
    is native(&native-library) is symbol('llev_phonetic_online_normalized_query') { * }
sub llev-phonetic-online-scan(Pointer, Pointer, size_t, size_t, size_t,
    Pointer is rw, size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_online_scan') { * }
sub llev-phonetic-online-matches-free(Pointer, size_t)
    is native(&native-library) is symbol('llev_phonetic_online_matches_free') { * }
sub llev-phonetic-online-stream-new(Pointer, size_t, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_online_stream_new') { * }
sub llev-phonetic-online-stream-free(Pointer)
    is native(&native-library) is symbol('llev_phonetic_online_stream_free') { * }
sub llev-phonetic-online-stream-feed(Pointer, Pointer, size_t --> int32)
    is native(&native-library) is symbol('llev_phonetic_online_stream_feed') { * }
sub llev-phonetic-online-stream-finish(Pointer, size_t, Pointer is rw,
    size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_online_stream_finish') { * }
sub llev-phonetic-token-new(Pointer, size_t, Pointer, uint8, size_t,
    Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_token_new') { * }
sub llev-phonetic-token-free(Pointer)
    is native(&native-library) is symbol('llev_phonetic_token_free') { * }
sub llev-phonetic-token-scan(Pointer, Pointer, size_t, size_t, size_t,
    size_t, Pointer is rw, size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_token_scan') { * }
sub llev-phonetic-token-matches-free(Pointer, size_t)
    is native(&native-library) is symbol('llev_phonetic_token_matches_free') { * }
sub llev-phonetic-dictionary-new(Pointer, size_t, Pointer, uint32, uint8,
    size_t, size_t, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_dictionary_new') { * }
sub llev-phonetic-dictionary-free(Pointer)
    is native(&native-library) is symbol('llev_phonetic_dictionary_free') { * }
sub llev-phonetic-dictionary-query(Pointer, Pointer, size_t, size_t, size_t,
    size_t, Pointer is rw, size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_dictionary_query') { * }
sub llev-phonetic-candidates-free(Pointer, size_t)
    is native(&native-library) is symbol('llev_phonetic_candidates_free') { * }
sub llev-phonetic-dictionary-update(Pointer, Pointer, size_t, uint8, size_t,
    uint8 is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_dictionary_update') { * }
sub llev-owned-string-free(OwnedString)
    is native(&native-library) is symbol('llev_owned_string_free') { * }
sub memcpy(Pointer, Pointer, size_t --> Pointer) is native { * }

sub take-owned-string(OwnedString:D $output --> Str:D) {
    LEAVE llev-owned-string-free($output);
    my $result = buf8.allocate($output.len);
    memcpy(raw-pointer($result), $output.data, $output.len) if $output.len;
    $result.decode('utf8')
}

sub copy-native-utf8(Pointer $data, Int:D $length --> Str:D) {
    die 'native result has a null pointer with nonzero length'
        if $length && !$data;
    my $result = buf8.allocate($length);
    memcpy(raw-pointer($result), $data, $length) if $length;
    $result.decode('utf8')
}

sub copy-online-matches(Pointer $matches, Int:D $count --> List:D) {
    LEAVE llev-phonetic-online-matches-free($matches, $count);
    return ().List unless $count;
    die 'native online scan returned null matches' unless $matches;
    my $width = nativesizeof(RawPhoneticOnlineMatch.new);
    my @result;
    for ^$count -> $index {
        my $raw = copy-cstruct(RawPhoneticOnlineMatch,
            Pointer.new($matches.Int + $index * $width));
        die 'native online scan returned inverted spans'
            unless $raw.byte-start <= $raw.byte-end &&
                $raw.char-start <= $raw.char-end;
        @result.push(PhoneticOnlineHit.new(
            byte-start => $raw.byte-start.Int,
            byte-end => $raw.byte-end.Int,
            char-start => $raw.char-start.Int,
            char-end => $raw.char-end.Int,
            original-text => copy-native-utf8($raw.original-data,
                $raw.original-len),
            normalized-text => copy-native-utf8($raw.normalized-data,
                $raw.normalized-len),
            distance => $raw.distance.Int));
    }
    @result.List
}

sub copy-token-matches(Pointer $matches, Int:D $count --> List:D) {
    LEAVE llev-phonetic-token-matches-free($matches, $count);
    return ().List unless $count;
    die 'native token scan returned null matches' unless $matches;
    my $match-width = nativesizeof(RawPhoneticTokenMatch.new);
    my $detail-width = nativesizeof(RawPhoneticTokenDetail.new);
    my @result;
    for ^$count -> $index {
        my $raw = copy-cstruct(RawPhoneticTokenMatch,
            Pointer.new($matches.Int + $index * $match-width));
        die 'native token scan returned an inverted byte span'
            unless $raw.byte-start <= $raw.byte-end;
        die 'native token scan returned null details'
            if $raw.detail-count && !$raw.details;
        my @details;
        for ^$raw.detail-count -> $detail-index {
            my $detail = copy-cstruct(RawPhoneticTokenDetail,
                Pointer.new($raw.details.Int + $detail-index * $detail-width));
            die 'native token scan returned an inverted detail span'
                unless $detail.byte-start <= $detail.byte-end;
            @details.push(PhoneticTokenDetail.new(
                token-index => $detail.token-index.Int,
                byte-start => $detail.byte-start.Int,
                byte-end => $detail.byte-end.Int,
                original-text => copy-native-utf8($detail.original-data,
                    $detail.original-len),
                normalized-text => copy-native-utf8($detail.normalized-data,
                    $detail.normalized-len),
                distance => $detail.distance.Int));
        }
        @result.push(PhoneticTokenHit.new(
            byte-start => $raw.byte-start.Int,
            byte-end => $raw.byte-end.Int,
            total-distance => $raw.total-distance.Int,
            matched-text => copy-native-utf8($raw.matched-data,
                $raw.matched-len),
            details => @details.List));
    }
    @result.List
}

sub copy-phonetic-candidates(Pointer $candidates, Int:D $count --> List:D) {
    LEAVE llev-phonetic-candidates-free($candidates, $count);
    return ().List unless $count;
    die 'native dictionary returned null candidates' unless $candidates;
    my $width = nativesizeof(RawPhoneticCandidate.new);
    my @result;
    for ^$count -> $index {
        my $raw = copy-cstruct(RawPhoneticCandidate,
            Pointer.new($candidates.Int + $index * $width));
        @result.push(PhoneticCandidate.new(
            term => copy-native-utf8($raw.term-data, $raw.term-len),
            distance => $raw.distance.Int,
            normalized-form => copy-native-utf8($raw.normalized-data,
                $raw.normalized-len)));
    }
    @result.List
}

sub abi-version(--> UInt:D) is export { llev-abi-version().UInt }
sub api-revision(--> UInt:D) is export { llev-api-revision().UInt }
sub build-features(--> UInt:D) is export { llev-build-features().UInt }

sub check-status(Int:D $status, Str:D $operation, Bool :$allow-end = False --> Bool:D) {
    return True if $status == OK;
    return False if $allow-end && $status == END;
    X::Liblevenshtein.new(
        :$status,
        :$operation,
        detail => (try llev-last-error-message) // '',
    ).throw
}

sub raw-pointer(Blob:D $buffer --> Pointer) {
    return Pointer unless $buffer.elems;
    nativecast(Pointer, $buffer)
}

class NativeByteArgument {
    has CArray[uint8] $.storage is required;
    has Int:D $.length is required;

    method pointer(--> Pointer:D) { nativecast(Pointer, $!storage) }
}

sub byte-argument(Blob:D $buffer --> NativeByteArgument:D) {
    # Repeated empty/nonempty Rakudo calls using a zero-sized CArray or null
    # pointer produced invalid-input results or an allocator abort. Supply a
    # stable owned non-null address while retaining logical length zero; the
    # extra slot is never part of the C input slice.
    my $units = CArray[uint8].allocate($buffer.elems max 1);
    for $buffer.list.kv -> $index, $value {
        $units[$index] = $value;
    }
    NativeByteArgument.new(storage => $units, length => $buffer.elems)
}

sub require-phonetic(--> Nil) {
    X::Liblevenshtein.new(
        status => UNSUPPORTED,
        operation => 'phonetic',
        detail => 'native library lacks bindings-phonetic',
    ).throw unless build-features() +& BUILD-FEATURE-PHONETIC;
}

sub phonetic-scalar(Str:D $character --> Int:D) {
    my @scalars = $character.ords;
    die 'phonetic character must contain exactly one Unicode scalar'
        unless @scalars.elems == 1;
    @scalars[0]
}

sub phonetic-limit(Int:D $limit, Str:D $name --> Int:D) {
    die "$name must be positive and fit size_t"
        unless 0 < $limit < 2 ** (nativesizeof(size_t) * 8);
    $limit
}

sub phonetic-mask(Int:D $mask --> UInt:D) {
    die 'phonetic feature mask must use only published bits 0..41'
        unless 0 <= $mask < 2**42;
    $mask.UInt
}

sub copy-cstruct(::T, Pointer:D $source --> T:D) {
    my $copy = T.new;
    memcpy(nativecast(Pointer, $copy), $source, nativesizeof($copy));
    $copy
}

my constant SIZE-MAX = 2 ** (nativesizeof(size_t) * 8) - 1;
my constant U64-MAX = 2**64 - 1;

class AffineGapCosts is export {
    has UInt:D $.gap-open is required;
    has UInt:D $.gap-extend is required;
    has UInt:D $.substitution is required;
}

sub checked-native-cost(Int:D $cost, Str:D $name --> Int:D) {
    die "$name must fit the native nonnegative distance range"
        unless 0 <= $cost < SIZE-MAX - 2;
    $cost
}

sub checked-threshold(Mu $threshold --> Mu) {
    $threshold.defined
        ?? checked-native-cost($threshold, 'threshold')
        !! Nil
}

sub u64-tokens(Positional:D $input --> CArray[uint64]) {
    # Rakudo's zero-element CArray can corrupt its allocator on repeated
    # NativeCall. Keep one backing slot but pass the actual logical length.
    my $tokens = CArray[uint64].allocate($input.elems max 1);
    for $input.list.kv -> $index, $token {
        die 'distance token is outside uint64'
            unless $token ~~ Int && 0 <= $token <= U64-MAX;
        $tokens[$index] = $token;
    }
    $tokens
}

sub distance-result(Mu $result, Str:D $operation --> Mu) {
    if $result == SIZE-MAX || $result == -1 {
        X::Liblevenshtein.new(
            status => INVALID-ARGUMENT,
            :$operation,
            detail => 'native distance inputs were rejected',
        ).throw;
    }
    # Above-threshold and mathematically undefined/unrepresentable results
    # are distinct native sentinels but both are absent optional distances.
    return Nil if $result == SIZE-MAX - 1 || $result == -2
        || $result == SIZE-MAX - 2 || $result == -3;
    $result.Int
}

sub retained-distance-result(Mu $result, Str:D $operation, $left-storage,
    $right-storage --> Mu) {
    # Keep both native allocations live until the NativeCall has returned.
    die 'invalid distance backing storage'
        unless $left-storage.defined && $right-storage.defined;
    distance-result($result, $operation)
}

multi sub distance-call(Str:D $kind, Str:D $source, Str:D $target, Mu $threshold --> Mu) {
    my $left-bytes = $source.encode('utf8');
    my $right-bytes = $target.encode('utf8');
    my $left-size = $left-bytes.elems;
    my $right-size = $right-bytes.elems;
    my $left = byte-argument($left-bytes);
    my $right = byte-argument($right-bytes);
    my $left-pointer = $left.pointer;
    my $right-pointer = $right.pointer;
    my $result = do given $kind {
        when 'standard' {
            $threshold.defined
                ?? llev-distance-threshold($left-pointer, $left-size,
                    $right-pointer, $right-size, $threshold)
                !! llev-distance($left-pointer, $left-size,
                    $right-pointer, $right-size)
        }
        when 'osa' {
            $threshold.defined
                ?? llev-damerau-distance-threshold($left-pointer, $left-size,
                    $right-pointer, $right-size, $threshold)
                !! llev-damerau-distance($left-pointer, $left-size,
                    $right-pointer, $right-size)
        }
        when 'true' {
            $threshold.defined
                ?? llev-true-damerau-distance-threshold($left-pointer, $left-size,
                    $right-pointer, $right-size, $threshold)
                !! llev-true-damerau-distance($left-pointer, $left-size,
                    $right-pointer, $right-size)
        }
        when 'merge' {
            $threshold.defined
                ?? llev-merge-and-split-distance-threshold(
                    $left-pointer, $left-size,
                    $right-pointer, $right-size, $threshold)
                !! llev-merge-and-split-distance(
                    $left-pointer, $left-size,
                    $right-pointer, $right-size)
        }
        when 'hamming' {
            $threshold.defined
                ?? llev-hamming-distance-threshold(
                    $left-pointer, $left-size,
                    $right-pointer, $right-size, $threshold)
                !! llev-hamming-distance($left-pointer, $left-size,
                    $right-pointer, $right-size)
        }
        when 'indel' {
            $threshold.defined
                ?? llev-indel-distance-threshold(
                    $left-pointer, $left-size,
                    $right-pointer, $right-size, $threshold)
                !! llev-indel-distance($left-pointer, $left-size,
                    $right-pointer, $right-size)
        }
    };
    retained-distance-result($result, "{$kind}-distance", $left, $right)
}

multi sub distance-call(
    Str:D $kind, Blob:D $source, Blob:D $target, Mu $threshold --> Mu
) {
    my $left-storage = byte-argument($source);
    my $right-storage = byte-argument($target);
    my $left = $left-storage.pointer;
    my $right = $right-storage.pointer;
    my $result = do given $kind {
        when 'standard' {
            $threshold.defined
                ?? llev-distance-bytes-threshold(
                    $left, $source.elems, $right, $target.elems, $threshold)
                !! llev-distance-bytes($left, $source.elems, $right, $target.elems)
        }
        when 'osa' {
            $threshold.defined
                ?? llev-damerau-distance-bytes-threshold(
                    $left, $source.elems, $right, $target.elems, $threshold)
                !! llev-damerau-distance-bytes($left, $source.elems, $right, $target.elems)
        }
        when 'true' {
            $threshold.defined
                ?? llev-true-damerau-distance-bytes-threshold(
                    $left, $source.elems, $right, $target.elems, $threshold)
                !! llev-true-damerau-distance-bytes(
                    $left, $source.elems, $right, $target.elems)
        }
        when 'merge' {
            $threshold.defined
                ?? llev-merge-and-split-distance-bytes-threshold(
                    $left, $source.elems, $right, $target.elems, $threshold)
                !! llev-merge-and-split-distance-bytes(
                    $left, $source.elems, $right, $target.elems)
        }
        when 'hamming' {
            $threshold.defined
                ?? llev-hamming-distance-bytes-threshold(
                    $left, $source.elems, $right, $target.elems, $threshold)
                !! llev-hamming-distance-bytes(
                    $left, $source.elems, $right, $target.elems)
        }
        when 'indel' {
            $threshold.defined
                ?? llev-indel-distance-bytes-threshold(
                    $left, $source.elems, $right, $target.elems, $threshold)
                !! llev-indel-distance-bytes($left, $source.elems, $right, $target.elems)
        }
    };
    retained-distance-result($result, "{$kind}-distance-bytes",
        $left-storage, $right-storage)
}

multi sub distance-call(
    Str:D $kind, Positional:D $source, Positional:D $target, Mu $threshold --> Mu
) {
    my $left = u64-tokens($source);
    my $right = u64-tokens($target);
    my $left-pointer = nativecast(Pointer, $left);
    my $right-pointer = nativecast(Pointer, $right);
    my $result = do given $kind {
        when 'standard' {
            $threshold.defined
                ?? llev-distance-u64-threshold(
                    $left-pointer, $source.elems,
                    $right-pointer, $target.elems, $threshold)
                !! llev-distance-u64(
                    $left-pointer, $source.elems, $right-pointer, $target.elems)
        }
        when 'osa' {
            $threshold.defined
                ?? llev-damerau-distance-u64-threshold(
                    $left-pointer, $source.elems,
                    $right-pointer, $target.elems, $threshold)
                !! llev-damerau-distance-u64(
                    $left-pointer, $source.elems, $right-pointer, $target.elems)
        }
        when 'true' {
            $threshold.defined
                ?? llev-true-damerau-distance-u64-threshold(
                    $left-pointer, $source.elems,
                    $right-pointer, $target.elems, $threshold)
                !! llev-true-damerau-distance-u64(
                    $left-pointer, $source.elems, $right-pointer, $target.elems)
        }
        when 'merge' {
            $threshold.defined
                ?? llev-merge-and-split-distance-u64-threshold(
                    $left-pointer, $source.elems,
                    $right-pointer, $target.elems, $threshold)
                !! llev-merge-and-split-distance-u64(
                    $left-pointer, $source.elems, $right-pointer, $target.elems)
        }
        when 'hamming' {
            $threshold.defined
                ?? llev-hamming-distance-u64-threshold(
                    $left-pointer, $source.elems,
                    $right-pointer, $target.elems, $threshold)
                !! llev-hamming-distance-u64(
                    $left-pointer, $source.elems, $right-pointer, $target.elems)
        }
        when 'indel' {
            $threshold.defined
                ?? llev-indel-distance-u64-threshold(
                    $left-pointer, $source.elems,
                    $right-pointer, $target.elems, $threshold)
                !! llev-indel-distance-u64(
                    $left-pointer, $source.elems, $right-pointer, $target.elems)
        }
    };
    retained-distance-result($result, "{$kind}-distance-u64", $left, $right)
}

multi sub distance(Str:D $source, Str:D $target --> Int:D) is export {
    distance-call('standard', $source, $target, Nil)
}
multi sub distance(Str:D $source, Str:D $target, Int:D :$threshold! --> Mu) is export {
    distance-call('standard', $source, $target, checked-threshold($threshold))
}
multi sub damerau-distance(Str:D $source, Str:D $target --> Int:D) is export {
    distance-call('osa', $source, $target, Nil)
}
multi sub damerau-distance(
    Str:D $source, Str:D $target, Int:D :$threshold! --> Mu
) is export {
    distance-call('osa', $source, $target, checked-threshold($threshold))
}
multi sub true-damerau-distance(Str:D $source, Str:D $target --> Int:D) is export {
    distance-call('true', $source, $target, Nil)
}
multi sub true-damerau-distance(
    Str:D $source, Str:D $target, Int:D :$threshold! --> Mu
) is export {
    distance-call('true', $source, $target, checked-threshold($threshold))
}

# Blob and u64-token overloads preserve the native unit domain. In particular,
# arbitrary bytes are never decoded as UTF-8 and tokens are not narrowed.
multi sub distance(Blob:D $source, Blob:D $target, Int :$threshold --> Mu) is export {
    distance-call('standard', $source, $target, checked-threshold($threshold))
}
multi sub distance(
    Positional:D $source, Positional:D $target, Int :$threshold --> Mu
) is export {
    distance-call('standard', $source, $target, checked-threshold($threshold))
}
multi sub damerau-distance(
    Blob:D $source, Blob:D $target, Int :$threshold --> Mu
) is export {
    distance-call('osa', $source, $target, checked-threshold($threshold))
}
multi sub damerau-distance(
    Positional:D $source, Positional:D $target, Int :$threshold --> Mu
) is export {
    distance-call('osa', $source, $target, checked-threshold($threshold))
}
multi sub true-damerau-distance(
    Blob:D $source, Blob:D $target, Int :$threshold --> Mu
) is export {
    distance-call('true', $source, $target, checked-threshold($threshold))
}
multi sub true-damerau-distance(
    Positional:D $source, Positional:D $target, Int :$threshold --> Mu
) is export {
    distance-call('true', $source, $target, checked-threshold($threshold))
}

multi sub merge-and-split-distance(
    Str:D $source, Str:D $target, Int :$threshold --> Mu
) is export {
    distance-call('merge', $source, $target, checked-threshold($threshold))
}
multi sub merge-and-split-distance(
    Blob:D $source, Blob:D $target, Int :$threshold --> Mu
) is export {
    distance-call('merge', $source, $target, checked-threshold($threshold))
}
multi sub merge-and-split-distance(
    Positional:D $source, Positional:D $target, Int :$threshold --> Mu
) is export {
    distance-call('merge', $source, $target, checked-threshold($threshold))
}
multi sub hamming-distance(
    Str:D $source, Str:D $target, Int :$threshold --> Mu
) is export {
    distance-call('hamming', $source, $target, checked-threshold($threshold))
}
multi sub hamming-distance(
    Blob:D $source, Blob:D $target, Int :$threshold --> Mu
) is export {
    distance-call('hamming', $source, $target, checked-threshold($threshold))
}
multi sub hamming-distance(
    Positional:D $source, Positional:D $target, Int :$threshold --> Mu
) is export {
    distance-call('hamming', $source, $target, checked-threshold($threshold))
}
multi sub indel-distance(
    Str:D $source, Str:D $target, Int :$threshold --> Mu
) is export {
    distance-call('indel', $source, $target, checked-threshold($threshold))
}
multi sub indel-distance(
    Blob:D $source, Blob:D $target, Int :$threshold --> Mu
) is export {
    distance-call('indel', $source, $target, checked-threshold($threshold))
}
multi sub indel-distance(
    Positional:D $source, Positional:D $target, Int :$threshold --> Mu
) is export {
    distance-call('indel', $source, $target, checked-threshold($threshold))
}

multi sub affine-call(
    Str:D $source, Str:D $target, AffineGapCosts:D $costs, Mu $threshold --> Mu
) {
    my $left-bytes = $source.encode('utf8');
    my $right-bytes = $target.encode('utf8');
    my $left-size = $left-bytes.elems;
    my $right-size = $right-bytes.elems;
    my $left = byte-argument($left-bytes);
    my $right = byte-argument($right-bytes);
    my $left-pointer = $left.pointer;
    my $right-pointer = $right.pointer;
    my $open = checked-native-cost($costs.gap-open, 'gap-open');
    my $extend = checked-native-cost($costs.gap-extend, 'gap-extend');
    my $substitute = checked-native-cost($costs.substitution, 'substitution');
    my $result = $threshold.defined
        ?? llev-affine-gap-distance-threshold(
            $left-pointer, $left-size, $right-pointer, $right-size,
            $open, $extend, $substitute, $threshold)
        !! llev-affine-gap-distance(
            $left-pointer, $left-size, $right-pointer, $right-size,
            $open, $extend, $substitute);
    retained-distance-result($result, 'affine-gap-distance', $left, $right)
}

multi sub affine-call(
    Blob:D $source, Blob:D $target, AffineGapCosts:D $costs, Mu $threshold --> Mu
) {
    my $left = byte-argument($source);
    my $right = byte-argument($target);
    my $open = checked-native-cost($costs.gap-open, 'gap-open');
    my $extend = checked-native-cost($costs.gap-extend, 'gap-extend');
    my $substitute = checked-native-cost($costs.substitution, 'substitution');
    my $result = $threshold.defined
        ?? llev-affine-gap-distance-bytes-threshold(
            $left.pointer, $source.elems,
            $right.pointer, $target.elems,
            $open, $extend, $substitute, $threshold)
        !! llev-affine-gap-distance-bytes(
            $left.pointer, $source.elems,
            $right.pointer, $target.elems,
            $open, $extend, $substitute);
    retained-distance-result($result, 'affine-gap-distance-bytes', $left, $right)
}

multi sub affine-call(
    Positional:D $source, Positional:D $target,
    AffineGapCosts:D $costs, Mu $threshold --> Mu
) {
    my $left = u64-tokens($source);
    my $right = u64-tokens($target);
    my $open = checked-native-cost($costs.gap-open, 'gap-open');
    my $extend = checked-native-cost($costs.gap-extend, 'gap-extend');
    my $substitute = checked-native-cost($costs.substitution, 'substitution');
    my $result = $threshold.defined
        ?? llev-affine-gap-distance-u64-threshold(
            nativecast(Pointer, $left), $source.elems,
            nativecast(Pointer, $right), $target.elems,
            $open, $extend, $substitute, $threshold)
        !! llev-affine-gap-distance-u64(
            nativecast(Pointer, $left), $source.elems,
            nativecast(Pointer, $right), $target.elems,
            $open, $extend, $substitute);
    retained-distance-result($result, 'affine-gap-distance-u64', $left, $right)
}

multi sub affine-gap-distance(
    Str:D $source, Str:D $target, AffineGapCosts:D $costs,
    Int :$threshold --> Mu
) is export {
    affine-call($source, $target, $costs, checked-threshold($threshold))
}
multi sub affine-gap-distance(
    Blob:D $source, Blob:D $target, AffineGapCosts:D $costs,
    Int :$threshold --> Mu
) is export {
    affine-call($source, $target, $costs, checked-threshold($threshold))
}
multi sub affine-gap-distance(
    Positional:D $source, Positional:D $target, AffineGapCosts:D $costs,
    Int :$threshold --> Mu
) is export {
    affine-call($source, $target, $costs, checked-threshold($threshold))
}

sub materialize(RawMatch:D $raw --> Match:D) {
    my $domain = UnitDomain($raw.unit-domain);
    my $term = do given $domain {
        when UNICODE-SCALAR {
            my $bytes = buf8.allocate($raw.byte-len);
            memcpy(raw-pointer($bytes), $raw.term-data, $raw.byte-len)
                if $raw.byte-len;
            $bytes.decode('utf8')
        }
        when BYTE {
            my $bytes = buf8.allocate($raw.byte-len);
            memcpy(raw-pointer($bytes), $raw.term-data, $raw.byte-len)
                if $raw.byte-len;
            Buf.new($bytes.list)
        }
        when U64 {
            my $values = CArray[uint64].allocate($raw.term-len);
            memcpy(nativecast(Pointer, $values), $raw.term-data,
                $raw.term-len * nativesizeof(uint64)) if $raw.term-len;
            (0 ..^ $raw.term-len).map({ $values[$_].UInt }).Array
        }
    };
    Match.new(
        :$term,
        distance => $raw.distance.Int,
        id => ($raw.has-id ?? $raw.id.UInt !! Nil),
        unit-domain => $domain,
    )
}

class QueryCursor does Iterable is export {
    has Pointer $!handle is required;
    has Bool $!closed = False;
    has Bool $!claimed = False;

    submethod BUILD(Pointer:D :$handle!) { $!handle = $handle }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(
            status => CLOSED,
            operation => 'query-cursor',
            detail => 'cursor is closed',
        ).throw if $!closed;
        $!handle
    }

    method next-batch(Int:D $maximum = DEFAULT-MATCH-BATCH --> Mu) {
        die 'maximum batch size must be positive' unless $maximum > 0;
        my $view = RawBatch.new;
        my $status = llev-query-cursor-next-batch(self!handle, $maximum, $view);
        return Nil unless check-status($status, 'query-cursor-next-batch', :allow-end);
        my @matches;
        my UInt:D $generation = $view.generation.UInt;
        my $failure;
        try {
            for 0 ..^ $view.len -> $index {
                my $address = Pointer.new(
                    $view.matches.Int + $index * nativesizeof(RawMatch)
                );
                @matches.push(materialize(copy-cstruct(RawMatch, $address)));
            }
            CATCH { default { $failure = $_ } }
        }
        my $release-status = llev-query-cursor-release-batch($!handle, $generation);
        $failure.rethrow if $failure.defined;
        check-status($release-status, 'query-cursor-release-batch');
        @matches.Array
    }

    method iterator(--> Iterator:D) {
        die 'query cursor is one-shot' if $!claimed;
        $!claimed = True;
        my $cursor = self;
        class :: does Iterator {
            has QueryCursor:D $.cursor is required;
            has @.pending is rw;
            method pull-one() {
                if @!pending.elems == 0 {
                    my $batch = $!cursor.next-batch;
                    unless $batch.defined {
                        $!cursor.close;
                        return IterationEnd;
                    }
                    @!pending = $batch.list;
                }
                @!pending.shift
            }
            submethod DESTROY { try $!cursor.close }
        }.new(:$cursor)
    }

    method Seq(--> Seq:D) { Seq.new(self.iterator) }
    method list(--> List:D) { self.Seq.list }

    method reduce-batches(
        &operation, Mu $initial, Int:D :$batch-size = DEFAULT-MATCH-BATCH --> Mu
    ) {
        my $accumulator = $initial;
        LEAVE self.close;
        loop {
            my $batch = self.next-batch($batch-size);
            last unless $batch.defined;
            $accumulator = operation($accumulator, $batch);
        }
        $accumulator
    }

    method close(--> Nil) {
        return if $!closed;
        check-status(llev-query-cursor-free($!handle), 'query-cursor-free');
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

class PhoneticPattern is export {
    has Pointer $!handle is required;
    has Bool $!closed = False;

    submethod BUILD(Pointer:D :$handle!) { $!handle = $handle }

    multi method new(Str:D :$regex!) { self!compile($regex, False) }
    multi method new(Str:D :$llre!) { self!compile($llre, True) }

    method !compile(Str:D $source, Bool:D $llre --> PhoneticPattern:D) {
        my $bytes = byte-argument($source.encode('utf8'));
        my Pointer $output .= new;
        my $status = $llre
            ?? llev-phonetic-pattern-compile-llre(
                $bytes.pointer, $bytes.length, $output)
            !! llev-phonetic-pattern-compile-regex(
                $bytes.pointer, $bytes.length, $output);
        check-status($status, $llre ?? 'phonetic-pattern-llre' !!
            'phonetic-pattern-regex');
        self.bless(handle => $output)
    }

    method native-handle(--> Pointer:D) {
        X::Liblevenshtein.new(
            status => CLOSED,
            operation => 'phonetic-pattern',
            detail => 'pattern is closed',
        ).throw if $!closed;
        $!handle
    }

    method size(--> List:D) {
        my size_t $states = 0;
        my size_t $transitions = 0;
        check-status(llev-phonetic-pattern-size(
            self.native-handle, $states, $transitions,
        ), 'phonetic-pattern-size');
        ($states.Int, $transitions.Int)
    }

    method accepts(Str:D $input --> Bool:D) {
        my $bytes = byte-argument($input.encode('utf8'));
        my uint8 $output = 0;
        check-status(llev-phonetic-pattern-matches(
            self.native-handle, $bytes.pointer, $bytes.length, $output,
        ), 'phonetic-pattern-matches');
        so $output
    }

    method close(--> Nil) {
        return if $!closed;
        llev-phonetic-pattern-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

sub phonetic-named-class(Str:D $name --> PhoneticPattern:D) is export {
    die 'named phonetic class must be a short identifier'
        unless 0 < $name.chars <= 64 && $name ~~ /^ \w+ $/;
    PhoneticPattern.new(regex => '[:' ~ $name ~ ':]')
}

class Transducer is export {
    has Pointer $!handle is required;
    has Bool $!closed = False;

    submethod BUILD(Pointer:D :$handle!) { $!handle = $handle }

    multi method new(
        InteropAccess::ResourceType:D :$resource!,
        Algorithm:D :$algorithm = STANDARD,
    ) {
        my Pointer $output .= new;
        check-status(llev-transducer-new($resource.raw, $algorithm, $output),
            'transducer-new');
        self.bless(handle => $output)
    }

    multi method new(
        InteropAccess::DictionaryType:D :$dictionary!,
        Algorithm:D :$algorithm = STANDARD,
    ) {
        self.new(resource => $dictionary.resource, :$algorithm)
    }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(
            status => CLOSED,
            operation => 'transducer',
            detail => 'transducer is closed',
        ).throw if $!closed;
        $!handle
    }

    method native-handle(--> Pointer:D) { self!handle }

    method snapshot(--> Transducer:D) {
        my Pointer $output .= new;
        check-status(llev-transducer-snapshot(self!handle, $output),
            'transducer-snapshot');
        Transducer.bless(handle => $output)
    }

    method unit-domain(--> UnitDomain:D) {
        my uint32 $output = 0;
        check-status(llev-transducer-unit-domain(self!handle, $output),
            'transducer-unit-domain');
        UnitDomain($output)
    }

    multi method query(
        Str:D $input, Int:D $maximum-distance,
        QueryOrder:D :$order = TRAVERSAL,
    --> QueryCursor:D) {
        die 'maximum distance must be nonnegative' if $maximum-distance < 0;
        my $bytes = byte-argument($input.encode('utf8'));
        my Pointer $output .= new;
        check-status(llev-transducer-query-utf8(
            self!handle, $bytes.pointer, $bytes.length, $maximum-distance,
            $order, $output,
        ), 'transducer-query-utf8');
        QueryCursor.new(handle => $output)
    }

    multi method query(
        Blob:D $input, Int:D $maximum-distance,
        QueryOrder:D :$order = TRAVERSAL,
    --> QueryCursor:D) {
        die 'maximum distance must be nonnegative' if $maximum-distance < 0;
        my $bytes = byte-argument($input);
        my Pointer $output .= new;
        check-status(llev-transducer-query-bytes(
            self!handle, $bytes.pointer, $bytes.length, $maximum-distance,
            $order, $output,
        ), 'transducer-query-bytes');
        QueryCursor.new(handle => $output)
    }

    multi method query(
        Positional:D $input, Int:D $maximum-distance,
        QueryOrder:D :$order = TRAVERSAL,
    --> QueryCursor:D) {
        die 'maximum distance must be nonnegative' if $maximum-distance < 0;
        my $tokens = u64-tokens($input);
        my Pointer $output .= new;
        check-status(llev-transducer-query-u64(
            self!handle, nativecast(Pointer, $tokens), $input.elems,
            $maximum-distance, $order, $output,
        ), 'transducer-query-u64');
        QueryCursor.new(handle => $output)
    }

    multi method query(
        PhoneticPattern:D $pattern, Int:D $maximum-distance,
    --> QueryCursor:D) {
        die 'phonetic maximum distance must fit uint8'
            unless 0 <= $maximum-distance <= 255;
        my Pointer $output .= new;
        check-status(llev-transducer-query-pattern(
            self!handle, $pattern.native-handle, $maximum-distance, $output,
        ), 'transducer-query-pattern');
        QueryCursor.new(handle => $output)
    }

    method close(--> Nil) {
        return if $!closed;
        llev-transducer-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

class QueryCache is export {
    has Pointer $!handle is required;
    has Bool $!closed = False;

    submethod BUILD(Pointer:D :$handle!) { $!handle = $handle }

    multi method new(
        Transducer:D :$transducer!,
        Int:D :$max-entries = 1024,
        Int:D :$max-weight = 64 * 1024 * 1024,
    ) {
        die 'max-entries must be nonnegative' if $max-entries < 0;
        die 'max-weight must be nonnegative' if $max-weight < 0;
        my Pointer $output .= new;
        check-status(llev-query-cache-new(
            $transducer.native-handle, $max-entries, $max-weight, $output,
        ), 'query-cache-new');
        self.bless(handle => $output)
    }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(
            status => CLOSED,
            operation => 'query-cache',
            detail => 'query cache is closed',
        ).throw if $!closed;
        $!handle
    }

    method stats(--> QueryCacheStats:D) {
        my $raw = RawQueryCacheStats.new;
        check-status(llev-query-cache-stats(self!handle, $raw),
            'query-cache-stats');
        QueryCacheStats.new(
            requests => $raw.requests.UInt,
            hits => $raw.hits.UInt,
            misses => $raw.misses.UInt,
            admissions => $raw.admissions.UInt,
            rejections => $raw.rejections.UInt,
            evictions => $raw.evictions.UInt,
            resident-entries => $raw.resident-entries.Int,
            resident-weight => $raw.resident-weight.Int,
        )
    }

    method elems(--> Int:D) { self.stats.resident-entries }
    method Bool(--> Bool:D) { so self.elems }

    method clear(--> QueryCache:D) {
        check-status(llev-query-cache-clear(self!handle), 'query-cache-clear');
        self
    }

    method reset-stats(--> QueryCache:D) {
        check-status(llev-query-cache-reset-stats(self!handle),
            'query-cache-reset-stats');
        self
    }

    multi method query(
        Str:D $input, Int:D $maximum-distance,
        QueryOrder:D :$order = TRAVERSAL,
    --> QueryCursor:D) {
        die 'maximum distance must be nonnegative' if $maximum-distance < 0;
        my $bytes = byte-argument($input.encode('utf8'));
        my Pointer $output .= new;
        check-status(llev-query-cache-query-utf8(
            self!handle, $bytes.pointer, $bytes.length, $maximum-distance,
            $order, $output,
        ), 'query-cache-query-utf8');
        QueryCursor.new(handle => $output)
    }

    multi method query(
        Blob:D $input, Int:D $maximum-distance,
        QueryOrder:D :$order = TRAVERSAL,
    --> QueryCursor:D) {
        die 'maximum distance must be nonnegative' if $maximum-distance < 0;
        my $bytes = byte-argument($input);
        my Pointer $output .= new;
        check-status(llev-query-cache-query-bytes(
            self!handle, $bytes.pointer, $bytes.length, $maximum-distance,
            $order, $output,
        ), 'query-cache-query-bytes');
        QueryCursor.new(handle => $output)
    }

    multi method query(
        Positional:D $input, Int:D $maximum-distance,
        QueryOrder:D :$order = TRAVERSAL,
    --> QueryCursor:D) {
        die 'maximum distance must be nonnegative' if $maximum-distance < 0;
        my $tokens = u64-tokens($input);
        my Pointer $output .= new;
        check-status(llev-query-cache-query-u64(
            self!handle, nativecast(Pointer, $tokens), $input.elems,
            $maximum-distance, $order, $output,
        ), 'query-cache-query-u64');
        QueryCursor.new(handle => $output)
    }

    method close(--> Nil) {
        return if $!closed;
        llev-query-cache-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

class PhoneticRuleSet is export {
    has Pointer $!handle is required;
    has Bool $!closed = False;

    submethod BUILD(Pointer:D :$handle!) { $!handle = $handle }

    multi method new(Str:D :$source!) {
        my $bytes = byte-argument($source.encode('utf8'));
        my Pointer $output .= new;
        check-status(llev-phonetic-rules-parse(
            $bytes.pointer, $bytes.length, $output,
        ), 'phonetic-rules-parse');
        self.bless(handle => $output)
    }

    multi method new(PhoneticRuleSetKind:D :$builtin!) {
        my Pointer $output .= new;
        check-status(llev-phonetic-rules-builtin($builtin, $output),
            'phonetic-rules-builtin');
        self.bless(handle => $output)
    }

    multi method new(Str:D :$language!) {
        require-phonetic;
        die 'language selection needs native API revision 11'
            if api-revision() < 11;
        my $bytes = byte-argument($language.encode('utf8'));
        die 'language tag must contain 1..64 UTF-8 bytes'
            unless 0 < $bytes.length <= 64;
        my Pointer $output .= new;
        check-status(llev-phonetic-rules-language($bytes.pointer,
            $bytes.length, $output), 'phonetic-rules-language');
        self.bless(handle => $output)
    }

    multi method new(Str:D :$file!, Positional:D :$search-paths = [],
        Int:D :$max-total-path-bytes = 4096) {
        require-phonetic;
        phonetic-limit($max-total-path-bytes, 'max-total-path-bytes');
        die 'at most 64 phonetic search paths are allowed'
            if $search-paths.elems > 64;
        my $path = byte-argument($file.encode('utf8'));
        my $total = $path.length;
        my @owners;
        my @views;
        for $search-paths.list -> $entry {
            die 'phonetic search path must be nonempty Str'
                unless $entry ~~ Str:D && $entry.chars;
            my $bytes = byte-argument($entry.encode('utf8'));
            $total += $bytes.length;
            @owners.push($bytes);
            @views.push(RawPhoneticUtf8View.new(
                data => $bytes.pointer.Int, len => $bytes.length));
        }
        die 'phonetic file paths exceed max-total-path-bytes'
            if $total > $max-total-path-bytes;
        my $storage = contiguous-structs(@views,
            nativesizeof(RawPhoneticUtf8View.new));
        my Pointer $output .= new;
        check-status(llev-phonetic-rules-load-file($path.pointer,
            $path.length, nativecast(Pointer, $storage), @views.elems,
            $max-total-path-bytes, $output), 'phonetic-rules-load-file');
        self.bless(handle => $output)
    }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(
            status => CLOSED,
            operation => 'phonetic-rules',
            detail => 'rule set is closed',
        ).throw if $!closed;
        $!handle
    }

    method native-handle(--> Pointer:D) { self!handle }

    method elems(--> Int:D) {
        my size_t $output = 0;
        check-status(llev-phonetic-rules-len(self!handle, $output),
            'phonetic-rules-len');
        $output.Int
    }

    method apply(Str:D $input --> Str:D) {
        my $bytes = byte-argument($input.encode('utf8'));
        my $output = OwnedString.new;
        check-status(llev-phonetic-rules-apply(
            self!handle, $bytes.pointer, $bytes.length, $output,
        ), 'phonetic-rules-apply');
        LEAVE llev-owned-string-free($output);
        my $result = buf8.allocate($output.len);
        memcpy(raw-pointer($result), $output.data, $output.len) if $output.len;
        $result.decode('utf8')
    }

    method close(--> Nil) {
        return if $!closed;
        llev-phonetic-rules-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

# The feature mask uses the fixed bit indices in liblevenshtein.h; callers
# may combine bits without depending on Rust enum discriminants.
sub phonetic-features(Str:D $character --> UInt:D) is export {
    require-phonetic;
    my uint64 $mask = 0;
    check-status(llev-phonetic-features(phonetic-scalar($character), $mask),
        'phonetic-features');
    $mask.UInt
}

sub phonetic-feature-characters(Int:D $selector, Int:D $value,
    Int:D $max-chars --> List:D) {
    require-phonetic;
    phonetic-limit($max-chars, 'max-chars');
    my size_t $required = 0;
    my $status = do given $selector {
        when 0 { llev-phonetic-chars-with-features(
            phonetic-mask($value), 0, Pointer, 0, $required) }
        when 1 { llev-phonetic-chars-with-features(
            phonetic-mask($value), 1, Pointer, 0, $required) }
        when 2 { llev-phonetic-similar-chars($value, Pointer, 0, $required) }
        when 3 { llev-phonetic-expand-feature-based(
            $value, Pointer, 0, $required) }
    };
    check-status($status, 'phonetic-feature-characters')
        unless $status == LIMIT-EXCEEDED;
    die 'phonetic character result exceeds max-chars'
        if $required > $max-chars;
    return ().List unless $required;
    my $chars = CArray[uint32].allocate($required);
    $status = do given $selector {
        when 0 { llev-phonetic-chars-with-features(
            $value, 0, nativecast(Pointer, $chars), $required, $required) }
        when 1 { llev-phonetic-chars-with-features(
            $value, 1, nativecast(Pointer, $chars), $required, $required) }
        when 2 { llev-phonetic-similar-chars($value,
            nativecast(Pointer, $chars), $required, $required) }
        when 3 { llev-phonetic-expand-feature-based($value,
            nativecast(Pointer, $chars), $required, $required) }
    };
    check-status($status, 'phonetic-feature-characters');
    (^$required).map({ $chars[$_].chr }).List
}

sub phonetic-chars-with-features(Int:D $mask, Bool:D :$any = False,
    Int:D :$max-chars = 4096 --> List:D) is export {
    phonetic-feature-characters($any ?? 1 !! 0, $mask, $max-chars)
}

sub phonetic-similar-chars(Str:D $character,
    Int:D :$max-chars = 4096 --> List:D) is export {
    phonetic-feature-characters(2, phonetic-scalar($character), $max-chars)
}

sub phonetic-expand-feature-based(Str:D $character,
    Int:D :$max-chars = 4096 --> List:D) is export {
    phonetic-feature-characters(3, phonetic-scalar($character), $max-chars)
}

sub phonetic-voicing-pair(Str:D $character --> Mu) is export {
    require-phonetic;
    my uint32 $partner = 0;
    my uint8 $found = 0;
    check-status(llev-phonetic-voicing-pair(
        phonetic-scalar($character), $partner, $found), 'phonetic-voicing-pair');
    $found ?? $partner.chr !! Nil
}

sub phonetic-feature-relation(Str:D $source, Str:D $target,
    Bool:D :$free-substitution = False --> Bool:D) is export {
    require-phonetic;
    my uint8 $matches = 0;
    check-status(llev-phonetic-feature-relation(
        phonetic-scalar($source), phonetic-scalar($target),
        $free-substitution ?? 1 !! 0, $matches), 'phonetic-feature-relation');
    so $matches
}

sub phonetic-feature-set-distance(Int:D $source-mask, Int:D $target-mask,
    PhoneticFeatureWeights :$weights --> Num:D) is export {
    require-phonetic;
    my num64 $distance = 0e0;
    check-status(llev-phonetic-feature-set-distance(
        phonetic-mask($source-mask), phonetic-mask($target-mask),
        $weights.defined ?? nativecast(Pointer, $weights) !! Pointer,
        $distance), 'phonetic-feature-set-distance');
    $distance.Num
}

sub phonetic-articulatory-distance(Str:D $source, Str:D $target,
    PhoneticFeatureWeights :$weights --> Num:D) is export {
    require-phonetic;
    my num64 $distance = 0e0;
    check-status(llev-phonetic-articulatory-distance(
        phonetic-scalar($source), phonetic-scalar($target),
        $weights.defined ?? nativecast(Pointer, $weights) !! Pointer,
        $distance), 'phonetic-articulatory-distance');
    $distance.Num
}

sub phonetic-articulatory-edit-distance(Str:D $source, Str:D $target,
    Int:D :$max-cells = 1_000_000, PhoneticFeatureWeights :$weights
    --> Num:D) is export {
    require-phonetic;
    phonetic-limit($max-cells, 'max-cells');
    my $left = byte-argument($source.encode('utf8'));
    my $right = byte-argument($target.encode('utf8'));
    my num64 $distance = 0e0;
    check-status(llev-phonetic-articulatory-edit-distance(
        $left.pointer, $left.length, $right.pointer, $right.length,
        $weights.defined ?? nativecast(Pointer, $weights) !! Pointer,
        $max-cells, $distance), 'phonetic-articulatory-edit-distance');
    $distance.Num
}

sub phonetic-syllable-count(Str:D $input, Bool:D :$ipa = False,
    Int:D :$max-input-scalars = 4096 --> Int:D) is export {
    require-phonetic;
    phonetic-limit($max-input-scalars, 'max-input-scalars');
    die 'phonetic input exceeds max-input-scalars'
        if $input.ords.elems > $max-input-scalars;
    my $bytes = byte-argument($input.encode('utf8'));
    my size_t $count = 0;
    check-status(llev-phonetic-syllable-count($bytes.pointer,
        $bytes.length, $ipa ?? 1 !! 0, $max-input-scalars, $count),
        'phonetic-syllable-count');
    $count.Int
}

sub phonetic-syllable-boundaries(Str:D $input, Bool:D :$ipa = False,
    Int:D :$max-input-scalars = 4096,
    Int:D :$max-boundaries = 4096 --> List:D) is export {
    require-phonetic;
    phonetic-limit($max-input-scalars, 'max-input-scalars');
    phonetic-limit($max-boundaries, 'max-boundaries');
    die 'phonetic input exceeds max-input-scalars'
        if $input.ords.elems > $max-input-scalars;
    my $bytes = byte-argument($input.encode('utf8'));
    my size_t $required = 0;
    my $status = llev-phonetic-syllable-boundaries($bytes.pointer,
        $bytes.length, $ipa ?? 1 !! 0, $max-input-scalars,
        Pointer, 0, $required);
    check-status($status, 'phonetic-syllable-boundaries')
        unless $status == LIMIT-EXCEEDED;
    die 'phonetic syllable result exceeds max-boundaries'
        if $required > $max-boundaries;
    return ().List unless $required;
    my $positions = CArray[size_t].allocate($required);
    check-status(llev-phonetic-syllable-boundaries($bytes.pointer,
        $bytes.length, $ipa ?? 1 !! 0, $max-input-scalars,
        nativecast(Pointer, $positions), $required, $required),
        'phonetic-syllable-boundaries');
    (^$required).map({ $positions[$_].Int }).List
}

class PhoneticGrep is export {
    has Pointer $!handle is required;
    has Bool $!closed = False;

    submethod BUILD(Pointer:D :$handle!) { $!handle = $handle }

    multi method new(Str:D :$pattern!, PhoneticRuleSet :$rules,
        Int:D :$max-distance = 0, Algorithm:D :$algorithm = STANDARD,
        Bool:D :$case-insensitive = False) {
        require-phonetic;
        die 'max-distance must fit uint8'
            unless 0 <= $max-distance <= 255;
        my $bytes = byte-argument($pattern.encode('utf8'));
        my Pointer $output .= new;
        check-status(llev-phonetic-grep-new(
            $bytes.pointer, $bytes.length,
            $rules.defined ?? $rules.native-handle !! Pointer,
            $max-distance, $algorithm.Int, $case-insensitive ?? 1 !! 0,
            $output), 'phonetic-grep-new');
        self.bless(handle => $output)
    }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(status => CLOSED,
            operation => 'phonetic-grep', detail => 'grep matcher is closed').throw
            if $!closed;
        $!handle
    }

    method distance-config(--> Map:D) {
        my uint8 $effective = 0;
        my uint8 $local = 0;
        my uint8 $has-local = 0;
        check-status(llev-phonetic-grep-distance-config(self!handle,
            $effective, $local, $has-local), 'phonetic-grep-distance-config');
        Map.new(effective => $effective.Int,
            local-override => ($has-local ?? $local.Int !! Nil))
    }

    method match-distance(Str:D $candidate,
        Int:D :$max-candidate-bytes = 1_000_000 --> Mu) {
        phonetic-limit($max-candidate-bytes, 'max-candidate-bytes');
        my $bytes = byte-argument($candidate.encode('utf8'));
        my uint8 $distance = 0;
        my uint8 $found = 0;
        check-status(llev-phonetic-grep-matches(self!handle,
            $bytes.pointer, $bytes.length, $max-candidate-bytes,
            $distance, $found), 'phonetic-grep-matches');
        $found ?? $distance.Int !! Nil
    }

    method !scan(Str:D $source, Bool:D $document, Int:D $max-input-bytes,
        Int:D $max-matches --> List:D) {
        phonetic-limit($max-input-bytes, 'max-input-bytes');
        phonetic-limit($max-matches, 'max-matches');
        my $bytes = byte-argument($source.encode('utf8'));
        die 'phonetic grep input exceeds max-input-bytes'
            if $bytes.length > $max-input-bytes;
        my size_t $required = 0;
        my $status = $document
            ?? llev-phonetic-grep-scan-text(self!handle,
                $bytes.pointer, $bytes.length, $max-input-bytes,
                Pointer, 0, $required)
            !! llev-phonetic-grep-scan-line(self!handle,
                $bytes.pointer, $bytes.length, $max-input-bytes,
                Pointer, 0, $required);
        check-status($status, 'phonetic-grep-scan')
            unless $status == LIMIT-EXCEEDED;
        die 'phonetic grep result exceeds max-matches'
            if $required > $max-matches;
        return ().List unless $required;
        my $width = nativesizeof(RawPhoneticGrepMatch.new);
        my $output = CArray[uint8].allocate($required * $width);
        $status = $document
            ?? llev-phonetic-grep-scan-text(self!handle,
                $bytes.pointer, $bytes.length, $max-input-bytes,
                nativecast(Pointer, $output), $required, $required)
            !! llev-phonetic-grep-scan-line(self!handle,
                $bytes.pointer, $bytes.length, $max-input-bytes,
                nativecast(Pointer, $output), $required, $required);
        check-status($status, 'phonetic-grep-scan');
        my @lines = $document ?? $source.lines.list !! ($source,);
        my @hits;
        for ^$required -> $index {
            my $address = Pointer.new(nativecast(Pointer, $output).Int +
                $index * $width);
            my $raw = copy-cstruct(RawPhoneticGrepMatch, $address);
            my $line-index = $raw.line-number.Int - 1;
            die 'native grep returned invalid line number'
                unless 0 <= $line-index < @lines.elems;
            my $line = @lines[$line-index].encode('utf8');
            die 'native grep returned invalid byte span'
                unless 0 <= $raw.start-byte <= $raw.end-byte <= $line.elems;
            @hits.push(PhoneticGrepHit.new(
                line-number => $raw.line-number.Int,
                start-byte => $raw.start-byte.Int,
                end-byte => $raw.end-byte.Int,
                distance => $raw.distance.Int,
                text => $line.subbuf($raw.start-byte,
                    $raw.end-byte - $raw.start-byte).decode('utf8')));
        }
        @hits.List
    }

    method scan-line(Str:D $line, Int:D :$max-input-bytes = 1_000_000,
        Int:D :$max-matches = 4096 --> List:D) {
        self!scan($line, False, $max-input-bytes, $max-matches)
    }

    method scan-text(Str:D $document, Int:D :$max-input-bytes = 1_000_000,
        Int:D :$max-matches = 4096 --> List:D) {
        self!scan($document, True, $max-input-bytes, $max-matches)
    }

    method close(--> Nil) {
        return if $!closed;
        llev-phonetic-grep-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

sub phonetic-expansion-limits(
    Int:D :$max-input-scalars = 4096,
    Int:D :$max-rules = 4096,
    Int:D :$max-rule-units = 100_000,
    Int:D :$max-nodes = 100_000,
    Int:D :$max-output-bytes = 1_000_000,
    --> PhoneticExpansionLimits:D) is export {
    PhoneticExpansionLimits.new(
        max-input-scalars => phonetic-limit($max-input-scalars,
            'max-input-scalars'),
        max-rules => phonetic-limit($max-rules, 'max-rules'),
        max-rule-units => phonetic-limit($max-rule-units, 'max-rule-units'),
        max-nodes => phonetic-limit($max-nodes, 'max-nodes'),
        max-output-bytes => phonetic-limit($max-output-bytes,
            'max-output-bytes'))
}

sub phonetic-expand(Str:D $input, PhoneticRuleSet :$rules,
    PhoneticExpansionLimits :$limits = phonetic-expansion-limits()
    --> Str:D) is export {
    require-phonetic;
    my $bytes = byte-argument($input.encode('utf8'));
    my $output = OwnedString.new;
    check-status(llev-phonetic-expand($bytes.pointer, $bytes.length,
        $rules.defined ?? $rules.native-handle !! Pointer,
        nativecast(Pointer, $limits), $output), 'phonetic-expand');
    take-owned-string($output)
}

sub phonetic-expand-with-costs(Str:D $input, PhoneticRuleSet :$rules,
    PhoneticExpansionLimits :$limits = phonetic-expansion-limits()
    --> Map:D) is export {
    require-phonetic;
    my $bytes = byte-argument($input.encode('utf8'));
    my $output = OwnedString.new;
    my num64 $cost = 0e0;
    check-status(llev-phonetic-expand-with-costs(
        $bytes.pointer, $bytes.length,
        $rules.defined ?? $rules.native-handle !! Pointer,
        nativecast(Pointer, $limits), $output, $cost),
        'phonetic-expand-with-costs');
    Map.new(pattern => take-owned-string($output), cost => $cost.Num)
}

class PhoneticTransducer is export {
    has Pointer $!handle is required;
    has Bool $!closed = False;

    submethod BUILD(Pointer:D :$handle!) { $!handle = $handle }

    multi method new(PhoneticRuleSet :$rules) {
        require-phonetic;
        my Pointer $output .= new;
        check-status(llev-phonetic-transducer-new(
            $rules.defined ?? $rules.native-handle !! Pointer,
            $output), 'phonetic-transducer-new');
        self.bless(handle => $output)
    }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(status => CLOSED,
            operation => 'phonetic-transducer',
            detail => 'transducer is closed').throw if $!closed;
        $!handle
    }

    method feed(Str:D $chunk, Int:D :$max-input-scalars = 4096,
        Int:D :$max-output-bytes = 1_000_000 --> Str:D) {
        phonetic-limit($max-input-scalars, 'max-input-scalars');
        phonetic-limit($max-output-bytes, 'max-output-bytes');
        my $bytes = byte-argument($chunk.encode('utf8'));
        my $output = OwnedString.new;
        check-status(llev-phonetic-transducer-feed(self!handle,
            $bytes.pointer, $bytes.length, $max-input-scalars,
            $max-output-bytes, $output), 'phonetic-transducer-feed');
        take-owned-string($output)
    }

    method finish(Int:D :$max-output-bytes = 1_000_000 --> Str:D) {
        phonetic-limit($max-output-bytes, 'max-output-bytes');
        my $output = OwnedString.new;
        check-status(llev-phonetic-transducer-finish(self!handle,
            $max-output-bytes, $output), 'phonetic-transducer-finish');
        take-owned-string($output)
    }

    method reset(--> PhoneticTransducer:D) {
        check-status(llev-phonetic-transducer-reset(self!handle),
            'phonetic-transducer-reset');
        self
    }

    method normalize(Str:D $input, Int:D :$max-input-scalars = 4096,
        Int:D :$max-output-bytes = 1_000_000 --> Str:D) {
        phonetic-limit($max-input-scalars, 'max-input-scalars');
        phonetic-limit($max-output-bytes, 'max-output-bytes');
        my $bytes = byte-argument($input.encode('utf8'));
        my $output = OwnedString.new;
        check-status(llev-phonetic-transducer-normalize(self!handle,
            $bytes.pointer, $bytes.length, $max-input-scalars,
            $max-output-bytes, $output), 'phonetic-transducer-normalize');
        take-owned-string($output)
    }

    method close(--> Nil) {
        return if $!closed;
        llev-phonetic-transducer-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

class PhoneticOnlineStream is export {
    has Pointer $!handle is required;
    has Bool $!closed = False;

    submethod BUILD(Pointer:D :$handle!) { $!handle = $handle }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(status => CLOSED,
            operation => 'phonetic-online-stream',
            detail => 'stream is closed').throw if $!closed;
        $!handle
    }

    method feed(Str:D $chunk --> PhoneticOnlineStream:D) {
        my $bytes = byte-argument($chunk.encode('utf8'));
        check-status(llev-phonetic-online-stream-feed(self!handle,
            $bytes.pointer, $bytes.length), 'phonetic-online-stream-feed');
        self
    }

    method finish(Int:D :$max-matches = 4096 --> List:D) {
        phonetic-limit($max-matches, 'max-matches');
        my Pointer $output .= new;
        my size_t $count = 0;
        check-status(llev-phonetic-online-stream-finish(self!handle,
            $max-matches, $output, $count), 'phonetic-online-stream-finish');
        copy-online-matches($output, $count)
    }

    method close(--> Nil) {
        return if $!closed;
        llev-phonetic-online-stream-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

class PhoneticOnlineGrep is export {
    has Pointer $!handle is required;
    has Bool $!closed = False;

    submethod BUILD(Pointer:D :$handle!) { $!handle = $handle }

    multi method new(Str:D :$pattern!, PhoneticRuleSet :$rules,
        Int:D :$max-distance = 0, Bool:D :$case-insensitive = False,
        Int:D :$max-pattern-scalars = 4096) {
        require-phonetic;
        die 'max-distance must fit uint8'
            unless 0 <= $max-distance <= 255;
        phonetic-limit($max-pattern-scalars, 'max-pattern-scalars');
        my $bytes = byte-argument($pattern.encode('utf8'));
        die 'phonetic pattern exceeds max-pattern-scalars'
            if $pattern.ords.elems > $max-pattern-scalars;
        my Pointer $output .= new;
        check-status(llev-phonetic-online-new($bytes.pointer, $bytes.length,
            $rules.defined ?? $rules.native-handle !! Pointer,
            $max-distance, $case-insensitive ?? 1 !! 0,
            $max-pattern-scalars, $output), 'phonetic-online-new');
        self.bless(handle => $output)
    }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(status => CLOSED,
            operation => 'phonetic-online',
            detail => 'matcher is closed').throw if $!closed;
        $!handle
    }

    method normalized-query(--> Str:D) {
        my $output = OwnedString.new;
        check-status(llev-phonetic-online-normalized-query(self!handle,
            $output), 'phonetic-online-normalized-query');
        take-owned-string($output)
    }

    method scan(Str:D $document, Int:D :$max-input-bytes = 1_000_000,
        Int:D :$max-matches = 4096 --> List:D) {
        phonetic-limit($max-input-bytes, 'max-input-bytes');
        phonetic-limit($max-matches, 'max-matches');
        my $bytes = byte-argument($document.encode('utf8'));
        die 'phonetic online input exceeds max-input-bytes'
            if $bytes.length > $max-input-bytes;
        my Pointer $output .= new;
        my size_t $count = 0;
        check-status(llev-phonetic-online-scan(self!handle,
            $bytes.pointer, $bytes.length, $max-input-bytes,
            $max-matches, $output, $count), 'phonetic-online-scan');
        copy-online-matches($output, $count)
    }

    method stream(Int:D :$max-total-bytes = 1_000_000
        --> PhoneticOnlineStream:D) {
        phonetic-limit($max-total-bytes, 'max-total-bytes');
        my Pointer $output .= new;
        check-status(llev-phonetic-online-stream-new(self!handle,
            $max-total-bytes, $output), 'phonetic-online-stream-new');
        PhoneticOnlineStream.new(handle => $output)
    }

    method close(--> Nil) {
        return if $!closed;
        llev-phonetic-online-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

class PhoneticTokenGrep is export {
    has Pointer $!handle is required;
    has Bool $!closed = False;

    submethod BUILD(Pointer:D :$handle!) { $!handle = $handle }

    multi method new(Str:D :$query!, PhoneticRuleSet :$rules,
        Int:D :$default-distance = 0,
        Int:D :$max-query-bytes = 4096) {
        require-phonetic;
        die 'default-distance must fit uint8'
            unless 0 <= $default-distance <= 255;
        phonetic-limit($max-query-bytes, 'max-query-bytes');
        my $bytes = byte-argument($query.encode('utf8'));
        die 'phonetic token query exceeds max-query-bytes'
            if $bytes.length > $max-query-bytes;
        my Pointer $output .= new;
        check-status(llev-phonetic-token-new($bytes.pointer, $bytes.length,
            $rules.defined ?? $rules.native-handle !! Pointer,
            $default-distance, $max-query-bytes, $output),
            'phonetic-token-new');
        self.bless(handle => $output)
    }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(status => CLOSED,
            operation => 'phonetic-token',
            detail => 'matcher is closed').throw if $!closed;
        $!handle
    }

    method scan(Str:D $document, Int:D :$max-input-bytes = 1_000_000,
        Int:D :$max-matches = 4096, Int:D :$max-details = 65_536
        --> List:D) {
        phonetic-limit($max-input-bytes, 'max-input-bytes');
        phonetic-limit($max-matches, 'max-matches');
        phonetic-limit($max-details, 'max-details');
        my $bytes = byte-argument($document.encode('utf8'));
        die 'phonetic token input exceeds max-input-bytes'
            if $bytes.length > $max-input-bytes;
        my Pointer $output .= new;
        my size_t $count = 0;
        check-status(llev-phonetic-token-scan(self!handle,
            $bytes.pointer, $bytes.length, $max-input-bytes,
            $max-matches, $max-details, $output, $count),
            'phonetic-token-scan');
        copy-token-matches($output, $count)
    }

    method close(--> Nil) {
        return if $!closed;
        llev-phonetic-token-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

class PhoneticNormalizedDictionary is export {
    has Pointer $!handle is required;
    has Bool:D $.compact is required;
    has Bool $!closed = False;

    submethod BUILD(Pointer:D :$handle!, Bool:D :$compact!) {
        $!handle = $handle;
        $!compact = $compact;
    }

    multi method new(Positional:D :$terms!, PhoneticRuleSet :$rules,
        Algorithm:D :$algorithm = STANDARD, Bool:D :$compact = False,
        Int:D :$max-terms = 100_000,
        Int:D :$max-total-bytes = 16_000_000) {
        require-phonetic;
        phonetic-limit($max-terms, 'max-terms');
        phonetic-limit($max-total-bytes, 'max-total-bytes');
        die 'phonetic dictionary exceeds max-terms'
            if $terms.elems > $max-terms;
        my @owners;
        my @views;
        my $total = 0;
        for $terms.list -> $term {
            die 'phonetic dictionary terms must be Str'
                unless $term ~~ Str:D;
            my $bytes = byte-argument($term.encode('utf8'));
            $total += $bytes.length;
            die 'phonetic dictionary exceeds max-total-bytes'
                if $total > $max-total-bytes;
            @owners.push($bytes);
            @views.push(RawPhoneticUtf8View.new(
                data => $bytes.pointer.Int, len => $bytes.length));
        }
        my $storage = contiguous-structs(@views,
            nativesizeof(RawPhoneticUtf8View.new));
        my Pointer $output .= new;
        check-status(llev-phonetic-dictionary-new(
            nativecast(Pointer, $storage), @views.elems,
            $rules.defined ?? $rules.native-handle !! Pointer,
            $algorithm.Int, $compact ?? 1 !! 0, $max-terms,
            $max-total-bytes, $output), 'phonetic-dictionary-new');
        self.bless(handle => $output, :$compact)
    }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(status => CLOSED,
            operation => 'phonetic-dictionary',
            detail => 'dictionary is closed').throw if $!closed;
        $!handle
    }

    method query(Str:D $term, Int:D :$max-distance = 2,
        Int:D :$max-query-scalars = 4096,
        Int:D :$max-results = 4096 --> List:D) {
        die 'max-distance must be nonnegative and fit size_t'
            unless 0 <= $max-distance < 2 ** (nativesizeof(size_t) * 8);
        phonetic-limit($max-query-scalars, 'max-query-scalars');
        phonetic-limit($max-results, 'max-results');
        die 'phonetic query exceeds max-query-scalars'
            if $term.ords.elems > $max-query-scalars;
        my $bytes = byte-argument($term.encode('utf8'));
        my Pointer $output .= new;
        my size_t $count = 0;
        check-status(llev-phonetic-dictionary-query(self!handle,
            $bytes.pointer, $bytes.length, $max-distance,
            $max-query-scalars, $max-results, $output, $count),
            'phonetic-dictionary-query');
        copy-phonetic-candidates($output, $count)
    }

    method !update(Str:D $term, Bool:D $remove,
        Int:D $max-term-scalars --> Bool:D) {
        die 'compact phonetic dictionary is immutable' if $!compact;
        phonetic-limit($max-term-scalars, 'max-term-scalars');
        die 'phonetic term exceeds max-term-scalars'
            if $term.ords.elems > $max-term-scalars;
        my $bytes = byte-argument($term.encode('utf8'));
        my uint8 $changed = 0;
        check-status(llev-phonetic-dictionary-update(self!handle,
            $bytes.pointer, $bytes.length, $remove ?? 1 !! 0,
            $max-term-scalars, $changed), 'phonetic-dictionary-update');
        so $changed
    }

    method insert(Str:D $term, Int:D :$max-term-scalars = 4096 --> Bool:D) {
        self!update($term, False, $max-term-scalars)
    }

    method remove(Str:D $term, Int:D :$max-term-scalars = 4096 --> Bool:D) {
        self!update($term, True, $max-term-scalars)
    }

    method close(--> Nil) {
        return if $!closed;
        llev-phonetic-dictionary-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

# Standalone automata have no dictionary dependency. The native implementation
# owns the validated operation set or substitution policy after construction.
class AutomatonLimits is export {
    has Int:D $.max-source-units = 1_000_000;
    has Int:D $.max-target-units = 1_000_000;
    has Int:D $.max-retained-cells = 1_000_000;
    has Int:D $.max-step-work-units = 100_000_000;

    method raw(--> RawAutomatonLimits:D) {
        for ($!max-source-units, $!max-target-units,
            $!max-retained-cells, $!max-step-work-units) -> $limit {
            die 'automaton limits must fit nonnegative native size_t'
                unless 0 <= $limit <= SIZE-MAX;
        }
        RawAutomatonLimits.new(
            max-source-units => $!max-source-units,
            max-target-units => $!max-target-units,
            max-retained-cells => $!max-retained-cells,
            max-step-work-units => $!max-step-work-units,
        )
    }
}

class GeneralizedRestriction is export {
    has Str:D $.source is required;
    has Str:D $.target is required;
}

class GeneralizedOperation is export {
    has Int:D $.consume-source is required;
    has Int:D $.consume-target is required;
    has Real:D $.weight is required;
    has Str:D $.name is required;
    has OperationApplicability:D $.applicability = APPLICABILITY-ANY;
    has Positional:D $.restrictions = [];
}

class GeneralizedOperationSet is export {
    has Positional:D $.operations is required;
    method elems(--> Int:D) { $!operations.elems }
    method list(--> List:D) { $!operations.list }
}

class GeneralizedObservation is export {
    has Int:D $.consumed-target-length is required;
    has Int:D $.active-positions is required;
    has Mu $.scaled-distance;
    has Int:D $.scale-denominator is required;
    has Mu $.distance;
    has Bool:D $.current-row-nonempty is required;
    has Bool:D $.accepting is required;
}

sub generalized-observation(RawGeneralizedObservation:D $raw
    --> GeneralizedObservation:D) {
    my $numerator = $raw.has-distance ?? $raw.scaled-distance.Int !! Nil;
    my $denominator = $raw.scale-denominator.Int;
    die 'native generalized observation has zero cost denominator'
        if $numerator.defined && $denominator == 0;
    GeneralizedObservation.new(
        consumed-target-length => $raw.consumed-target-len.Int,
        active-positions => $raw.active-positions.Int,
        scaled-distance => $numerator,
        scale-denominator => $denominator,
        distance => $numerator.defined ?? $numerator / $denominator !! Nil,
        current-row-nonempty => so $raw.current-row-nonempty,
        accepting => so $raw.accepting,
    )
}

sub checked-maximum-distance(Int:D $value --> Int:D) {
    die 'maximum distance must fit uint8' unless 0 <= $value <= 255;
    $value
}

sub scalar-value(Str:D $value --> Int:D) {
    die 'a Unicode unit must contain exactly one scalar' unless $value.chars == 1;
    $value.ord
}

sub contiguous-structs(Positional:D $values, Int:D $width --> CArray[uint8]) {
    # NativeCall's CArray[CStruct] stores *pointers* to CStruct instances.
    # The ABI expects structs inline, so copy their exact native layouts into
    # one byte allocation. A single backing byte keeps an empty slice stable.
    my $bytes = CArray[uint8].allocate(($values.elems * $width) max 1);
    my $base = nativecast(Pointer, $bytes).Int;
    for $values.list.kv -> $index, $value {
        memcpy(Pointer.new($base + $index * $width),
            nativecast(Pointer, $value), $width);
    }
    $bytes
}

sub marshal-generalized-operations(Positional:D $operations, @owners
    --> CArray[uint8]) {
    die 'generalized operations must be nonempty' unless $operations.elems;
    my @raw;
    for $operations.list.kv -> $index, $operation {
        die 'each generalized operation must be a GeneralizedOperation'
            unless $operation ~~ GeneralizedOperation;
        for ($operation.consume-source, $operation.consume-target) -> $arity {
            die 'operation arity must fit nonnegative native size_t'
                unless 0 <= $arity <= SIZE-MAX;
        }
        my $name = byte-argument($operation.name.encode('utf8'));
        @owners.push($name);
        my @pairs = $operation.restrictions.list;
        my @raw-pairs;
        for @pairs.kv -> $pair-index, $restriction {
            my $value = $restriction ~~ Pair
                ?? GeneralizedRestriction.new(
                    source => $restriction.key,
                    target => $restriction.value,
                ) !! $restriction;
            die 'a listed restriction must be a source/target text pair'
                unless $value ~~ GeneralizedRestriction;
            my $source = byte-argument($value.source.encode('utf8'));
            my $target = byte-argument($value.target.encode('utf8'));
            @owners.append($source, $target);
            @raw-pairs.push(RawGeneralizedRestriction.new(
                source-data => $source.pointer.Int,
                source-len => $source.length,
                target-data => $target.pointer.Int,
                target-len => $target.length,
            ));
        }
        my $pairs = contiguous-structs(
            @raw-pairs, nativesizeof(RawGeneralizedRestriction));
        @owners.push($pairs);
        @raw.push(RawGeneralizedOperation.new(
            consume-source => $operation.consume-source,
            consume-target => $operation.consume-target,
            weight => $operation.weight.Num,
            name-data => $name.pointer.Int,
            name-len => $name.length,
            applicability => $operation.applicability.Int,
            reserved => 0,
            restrictions => nativecast(Pointer, $pairs).Int,
            restriction-count => @pairs.elems,
        ));
    }
    contiguous-structs(@raw, nativesizeof(RawGeneralizedOperation))
}

class GeneralizedOnlineAutomaton is export {
    has Pointer $!handle is required;
    has Bool $!closed = False;
    submethod BUILD(Pointer:D :$handle!) { $!handle = $handle }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(status => CLOSED,
            operation => 'generalized-online', detail => 'online state is closed').throw
            if $!closed;
        $!handle
    }

    method observation(--> GeneralizedObservation:D) {
        my $output = RawGeneralizedObservation.new;
        check-status(llev-generalized-online-observation(self!handle, $output),
            'generalized-online-observation');
        generalized-observation($output)
    }

    method advance(Str:D $unit --> GeneralizedObservation:D) {
        my $output = RawGeneralizedObservation.new;
        check-status(llev-generalized-online-advance(
            self!handle, scalar-value($unit), $output,
        ), 'generalized-online-advance');
        generalized-observation($output)
    }

    method close(--> Nil) {
        return if $!closed;
        llev-generalized-online-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

class GeneralizedAutomaton is export {
    has Pointer $!handle is required;
    has Bool $!closed = False;
    submethod BUILD(Pointer:D :$handle!) { $!handle = $handle }

    multi method new(Int:D $maximum-distance, GeneralizedOperationSet:D $set) {
        my @owners;
        my $operations = marshal-generalized-operations($set.operations, @owners);
        my Pointer $output .= new;
        my $status = llev-generalized-automaton-new(
            checked-maximum-distance($maximum-distance),
            nativecast(Pointer, $operations), $set.elems, $output,
        );
        die 'generalized operation backing storage was lost' unless @owners.elems;
        check-status($status, 'generalized-automaton-new');
        self.bless(handle => $output)
    }

    multi method new(Int:D $maximum-distance, Positional:D $operations) {
        self.new($maximum-distance,
            GeneralizedOperationSet.new(operations => $operations))
    }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(status => CLOSED,
            operation => 'generalized-automaton', detail => 'automaton is closed').throw
            if $!closed;
        $!handle
    }

    method evaluate(Str:D $source, Str:D $target,
        AutomatonLimits :$limits --> GeneralizedObservation:D) {
        my $left = byte-argument($source.encode('utf8'));
        my $right = byte-argument($target.encode('utf8'));
        my $raw-limits = $limits.defined ?? $limits.raw !! Nil;
        my $limit-argument = $raw-limits.defined
            ?? $raw-limits !! RawAutomatonLimits;
        my $output = RawGeneralizedObservation.new;
        my $retained = [$left, $right, $raw-limits];
        my $status = llev-generalized-automaton-evaluate-utf8(
            self!handle, $left.pointer, $left.length,
            $right.pointer, $right.length, $limit-argument, $output,
        );
        die 'generalized evaluation backing storage was lost'
            unless $retained.elems == 3;
        check-status($status, 'generalized-automaton-evaluate-utf8');
        generalized-observation($output)
    }

    method accepts(Str:D $source, Str:D $target,
        AutomatonLimits :$limits --> Bool:D) {
        self.evaluate($source, $target, :$limits).accepting
    }

    method online(Str:D $source,
        AutomatonLimits :$limits --> GeneralizedOnlineAutomaton:D) {
        my $bytes = byte-argument($source.encode('utf8'));
        my $raw-limits = $limits.defined ?? $limits.raw !! Nil;
        my $limit-argument = $raw-limits.defined
            ?? $raw-limits !! RawAutomatonLimits;
        my Pointer $output .= new;
        my $retained = [$bytes, $raw-limits];
        my $status = llev-generalized-online-new-utf8(
            self!handle, $bytes.pointer, $bytes.length, $limit-argument, $output,
        );
        die 'generalized online backing storage was lost'
            unless $retained.elems == 2;
        check-status($status, 'generalized-online-new-utf8');
        GeneralizedOnlineAutomaton.bless(handle => $output)
    }

    method close(--> Nil) {
        return if $!closed;
        llev-generalized-automaton-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

# Omitting a policy selects the native unrestricted specialization. A policy
# with directional pairs is domain-specific and must contain at least one pair.
class UniversalEquivalence is export {
    has Mu $.source is required;
    has Mu $.target is required;
}

class UniversalPolicy is export {
    has UnitDomain:D $.domain is required;
    has Positional:D $.equivalences is required;
    method elems(--> Int:D) { $!equivalences.elems }
}

class UniversalObservation is export {
    has Int:D $.consumed-target-length is required;
    has Int:D $.source-length is required;
    has Bool:D $.alive is required;
    has Bool:D $.accepting is required;
}

sub universal-observation(RawUniversalObservation:D $raw
    --> UniversalObservation:D) {
    UniversalObservation.new(
        consumed-target-length => $raw.consumed-target-len.Int,
        source-length => $raw.source-len.Int,
        alive => so $raw.alive,
        accepting => so $raw.accepting,
    )
}

sub checked-universal-unit(Mu $value, UnitDomain:D $domain --> Int:D) {
    if $domain == UNICODE-SCALAR {
        return scalar-value($value) if $value ~~ Str;
        die 'Unicode equivalence must be a scalar string or scalar number'
            unless $value ~~ Int;
        die 'Unicode scalar is outside the valid scalar range'
            unless 0 <= $value <= 0x10ffff && !($value >= 0xd800 && $value <= 0xdfff);
        return $value;
    }
    die 'byte/u64 equivalence must contain integer units' unless $value ~~ Int;
    my $maximum = $domain == BYTE ?? 255 !! U64-MAX;
    die 'universal-equivalence unit is outside its domain'
        unless 0 <= $value <= $maximum;
    $value
}

class NativeUniversalInput {
    has UnitDomain:D $.domain is required;
    has Pointer:D $.pointer is required;
    has Int:D $.length is required;
    has Mu $.owner is required;
}

multi sub universal-input(Str:D $source --> NativeUniversalInput:D) {
    my $bytes = byte-argument($source.encode('utf8'));
    NativeUniversalInput.new(
        domain => UNICODE-SCALAR, pointer => $bytes.pointer,
        length => $bytes.length, owner => $bytes,
    )
}

multi sub universal-input(Blob:D $source --> NativeUniversalInput:D) {
    my $bytes = byte-argument($source);
    NativeUniversalInput.new(
        domain => BYTE, pointer => $bytes.pointer,
        length => $bytes.length, owner => $bytes,
    )
}

multi sub universal-input(Positional:D $source --> NativeUniversalInput:D) {
    my $tokens = u64-tokens($source);
    NativeUniversalInput.new(
        domain => U64, pointer => nativecast(Pointer, $tokens),
        length => $source.elems, owner => $tokens,
    )
}

class UniversalOnlineAutomaton is export {
    has Pointer $!handle is required;
    has UnitDomain:D $.unit-domain is required;
    has Bool $!closed = False;
    submethod BUILD(Pointer:D :$handle!, UnitDomain:D :$unit-domain!) {
        $!handle = $handle;
        $!unit-domain = $unit-domain;
    }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(status => CLOSED,
            operation => 'universal-online', detail => 'online state is closed').throw
            if $!closed;
        $!handle
    }

    method observation(--> UniversalObservation:D) {
        my $output = RawUniversalObservation.new;
        check-status(llev-universal-online-observation(self!handle, $output),
            'universal-online-observation');
        universal-observation($output)
    }

    method advance(Mu $unit --> UniversalObservation:D) {
        my $handle = self!handle;
        die 'a Unicode prefix advances by one scalar string'
            if $!unit-domain == UNICODE-SCALAR && $unit !~~ Str;
        die 'a byte/u64 prefix advances by one integer'
            if $!unit-domain != UNICODE-SCALAR && $unit !~~ Int;
        my $value = checked-universal-unit($unit, $!unit-domain);
        # Rakudo NativeCall cannot unbox a boxed Int above signed i64 for a
        # by-value uint64 argument. Pass the equivalent signed two's-complement
        # bit pattern; the ABI receives the exact unsigned 64-bit token.
        my $native-unit = $value > 2**63 - 1 ?? $value - 2**64 !! $value;
        my $output = RawUniversalObservation.new;
        check-status(llev-universal-online-advance($handle, $native-unit, $output),
            'universal-online-advance');
        universal-observation($output)
    }

    method close(--> Nil) {
        return if $!closed;
        llev-universal-online-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

class UniversalAutomaton is export {
    has Pointer $!handle is required;
    has Bool $!closed = False;
    submethod BUILD(Pointer:D :$handle!) { $!handle = $handle }

    multi method new(Int:D $maximum-distance,
        UniversalVariant:D :$variant = UNIVERSAL-STANDARD,
        UniversalPolicy :$policy) {
        my $count = $policy.defined ?? $policy.elems !! 0;
        die 'a directional universal policy must have at least one equivalence'
            if $policy.defined && !$count;
        my @raw-pairs;
        if $policy.defined {
            die 'universal policy domain must be BYTE, UNICODE-SCALAR, or U64'
                unless $policy.domain == BYTE || $policy.domain == UNICODE-SCALAR
                    || $policy.domain == U64;
            for $policy.equivalences.list.kv -> $index, $equivalence {
                my $value = $equivalence ~~ Pair
                    ?? UniversalEquivalence.new(
                        source => $equivalence.key,
                        target => $equivalence.value,
                    ) !! $equivalence;
                die 'universal equivalence must be a directional pair'
                    unless $value ~~ UniversalEquivalence;
                @raw-pairs.push(RawUniversalEquivalence.new(
                    source => checked-universal-unit($value.source, $policy.domain),
                    target => checked-universal-unit($value.target, $policy.domain),
                ));
            }
        }
        my $pairs = contiguous-structs(
            @raw-pairs, nativesizeof(RawUniversalEquivalence));
        my Pointer $output .= new;
        my $status = llev-universal-automaton-new(
            checked-maximum-distance($maximum-distance), $variant.Int,
            $policy.defined ?? $policy.domain.Int !! 0,
            $count ?? nativecast(Pointer, $pairs) !! Pointer,
            $count, $output,
        );
        die 'universal policy backing storage was lost' unless $pairs.defined;
        check-status($status, 'universal-automaton-new');
        self.bless(handle => $output)
    }

    method !handle(--> Pointer:D) {
        X::Liblevenshtein.new(status => CLOSED,
            operation => 'universal-automaton', detail => 'automaton is closed').throw
            if $!closed;
        $!handle
    }

    method evaluate(Mu $source, Mu $target,
        AutomatonLimits :$limits --> UniversalObservation:D) {
        my $left = universal-input($source);
        my $right = universal-input($target);
        die 'universal source and target must share a unit domain'
            unless $left.domain == $right.domain;
        my $raw-limits = $limits.defined ?? $limits.raw !! Nil;
        my $limit-argument = $raw-limits.defined
            ?? $raw-limits !! RawAutomatonLimits;
        my $output = RawUniversalObservation.new;
        my $retained = [$left, $right, $raw-limits];
        my $status = llev-universal-automaton-evaluate(
            self!handle, $left.domain.Int, $left.pointer, $left.length,
            $right.pointer, $right.length, $limit-argument, $output,
        );
        die 'universal evaluation backing storage was lost'
            unless $retained.elems == 3;
        check-status($status, 'universal-automaton-evaluate');
        universal-observation($output)
    }

    method accepts(Mu $source, Mu $target,
        AutomatonLimits :$limits --> Bool:D) {
        self.evaluate($source, $target, :$limits).accepting
    }

    method online(Mu $source,
        AutomatonLimits :$limits --> UniversalOnlineAutomaton:D) {
        my $input = universal-input($source);
        my $raw-limits = $limits.defined ?? $limits.raw !! Nil;
        my $limit-argument = $raw-limits.defined
            ?? $raw-limits !! RawAutomatonLimits;
        my Pointer $output .= new;
        my $retained = [$input, $raw-limits];
        my $status = llev-universal-online-new(
            self!handle, $input.domain.Int, $input.pointer, $input.length,
            $limit-argument, $output,
        );
        die 'universal online backing storage was lost'
            unless $retained.elems == 2;
        check-status($status, 'universal-online-new');
        UniversalOnlineAutomaton.bless(
            handle => $output, unit-domain => $input.domain,
        )
    }

    method close(--> Nil) {
        return if $!closed;
        llev-universal-automaton-free($!handle);
        $!handle = Pointer;
        $!closed = True;
    }

    method opened(--> Bool:D) { !$!closed }
    submethod DESTROY { try self.close }
}

INIT {
    die "liblevenshtein native ABI version mismatch"
        unless abi-version() == ABI-VERSION;
    die "liblevenshtein native API revision is too old"
        unless api-revision() >= API-REVISION;
}
