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

class Match is export {
    has Mu $.term is required;
    has Int:D $.distance is required;
    has Mu $.id;
    has UnitDomain:D $.unit-domain is required;
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
sub llev-phonetic-rules-builtin(uint32, Pointer is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_rules_builtin') { * }
sub llev-phonetic-rules-free(Pointer)
    is native(&native-library) is symbol('llev_phonetic_rules_free') { * }
sub llev-phonetic-rules-len(Pointer, size_t is rw --> int32)
    is native(&native-library) is symbol('llev_phonetic_rules_len') { * }
sub llev-phonetic-rules-apply(Pointer, Pointer, size_t, OwnedString --> int32)
    is native(&native-library) is symbol('llev_phonetic_rules_apply') { * }
sub llev-owned-string-free(OwnedString)
    is native(&native-library) is symbol('llev_owned_string_free') { * }
sub memcpy(Pointer, Pointer, size_t --> Pointer) is native { * }

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

sub raw-pointer(Blob:D $buffer --> Pointer:D) {
    nativecast(Pointer, $buffer)
}

sub copy-cstruct(::T, Pointer:D $source --> T:D) {
    my $copy = T.new;
    memcpy(nativecast(Pointer, $copy), $source, nativesizeof($copy));
    $copy
}

my constant SIZE-MAX = 2 ** (nativesizeof(size_t) * 8) - 1;

sub distance-call(Str:D $kind, Str:D $source, Str:D $target, Mu $threshold --> Mu) {
    my $left = $source.encode('utf8');
    my $right = $target.encode('utf8');
    my $result = do given $kind {
        when 'standard' {
            $threshold.defined
                ?? llev-distance-threshold(raw-pointer($left), $left.elems,
                    raw-pointer($right), $right.elems, $threshold)
                !! llev-distance(raw-pointer($left), $left.elems,
                    raw-pointer($right), $right.elems)
        }
        when 'osa' {
            $threshold.defined
                ?? llev-damerau-distance-threshold(raw-pointer($left), $left.elems,
                    raw-pointer($right), $right.elems, $threshold)
                !! llev-damerau-distance(raw-pointer($left), $left.elems,
                    raw-pointer($right), $right.elems)
        }
        when 'true' {
            $threshold.defined
                ?? llev-true-damerau-distance-threshold(raw-pointer($left), $left.elems,
                    raw-pointer($right), $right.elems, $threshold)
                !! llev-true-damerau-distance(raw-pointer($left), $left.elems,
                    raw-pointer($right), $right.elems)
        }
    };
    X::Liblevenshtein.new(
        status => INVALID-UTF8,
        operation => "{$kind}-distance",
        detail => (try llev-last-error-message) // '',
    ).throw if $result == SIZE-MAX || $result == -1;
    return Nil if $result == SIZE-MAX - 1 || $result == -2;
    $result.Int
}

multi sub distance(Str:D $source, Str:D $target --> Int:D) is export {
    distance-call('standard', $source, $target, Nil)
}
multi sub distance(Str:D $source, Str:D $target, Int:D :$threshold! --> Mu) is export {
    die 'threshold must be nonnegative' if $threshold < 0;
    distance-call('standard', $source, $target, $threshold)
}
multi sub damerau-distance(Str:D $source, Str:D $target --> Int:D) is export {
    distance-call('osa', $source, $target, Nil)
}
multi sub damerau-distance(
    Str:D $source, Str:D $target, Int:D :$threshold! --> Mu
) is export {
    die 'threshold must be nonnegative' if $threshold < 0;
    distance-call('osa', $source, $target, $threshold)
}
multi sub true-damerau-distance(Str:D $source, Str:D $target --> Int:D) is export {
    distance-call('true', $source, $target, Nil)
}
multi sub true-damerau-distance(
    Str:D $source, Str:D $target, Int:D :$threshold! --> Mu
) is export {
    die 'threshold must be nonnegative' if $threshold < 0;
    distance-call('true', $source, $target, $threshold)
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
        my $bytes = $source.encode('utf8');
        my Pointer $output .= new;
        my $status = $llre
            ?? llev-phonetic-pattern-compile-llre(
                raw-pointer($bytes), $bytes.elems, $output)
            !! llev-phonetic-pattern-compile-regex(
                raw-pointer($bytes), $bytes.elems, $output);
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
        my $bytes = $input.encode('utf8');
        my uint8 $output = 0;
        check-status(llev-phonetic-pattern-matches(
            self.native-handle, raw-pointer($bytes), $bytes.elems, $output,
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
        my $bytes = $input.encode('utf8');
        my Pointer $output .= new;
        check-status(llev-transducer-query-utf8(
            self!handle, raw-pointer($bytes), $bytes.elems, $maximum-distance,
            $order, $output,
        ), 'transducer-query-utf8');
        QueryCursor.new(handle => $output)
    }

    multi method query(
        Blob:D $input, Int:D $maximum-distance,
        QueryOrder:D :$order = TRAVERSAL,
    --> QueryCursor:D) {
        die 'maximum distance must be nonnegative' if $maximum-distance < 0;
        my Pointer $output .= new;
        check-status(llev-transducer-query-bytes(
            self!handle, raw-pointer($input), $input.elems, $maximum-distance,
            $order, $output,
        ), 'transducer-query-bytes');
        QueryCursor.new(handle => $output)
    }

    multi method query(
        Positional:D $input, Int:D $maximum-distance,
        QueryOrder:D :$order = TRAVERSAL,
    --> QueryCursor:D) {
        die 'maximum distance must be nonnegative' if $maximum-distance < 0;
        my $tokens = CArray[uint64].allocate($input.elems);
        for $input.list.kv -> $index, $token {
            die 'query token is outside uint64'
                unless $token ~~ Int && 0 <= $token <= 2**64 - 1;
            $tokens[$index] = $token;
        }
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
        my $bytes = $input.encode('utf8');
        my Pointer $output .= new;
        check-status(llev-query-cache-query-utf8(
            self!handle, raw-pointer($bytes), $bytes.elems, $maximum-distance,
            $order, $output,
        ), 'query-cache-query-utf8');
        QueryCursor.new(handle => $output)
    }

    multi method query(
        Blob:D $input, Int:D $maximum-distance,
        QueryOrder:D :$order = TRAVERSAL,
    --> QueryCursor:D) {
        die 'maximum distance must be nonnegative' if $maximum-distance < 0;
        my Pointer $output .= new;
        check-status(llev-query-cache-query-bytes(
            self!handle, raw-pointer($input), $input.elems, $maximum-distance,
            $order, $output,
        ), 'query-cache-query-bytes');
        QueryCursor.new(handle => $output)
    }

    multi method query(
        Positional:D $input, Int:D $maximum-distance,
        QueryOrder:D :$order = TRAVERSAL,
    --> QueryCursor:D) {
        die 'maximum distance must be nonnegative' if $maximum-distance < 0;
        my $tokens = CArray[uint64].allocate($input.elems);
        for $input.list.kv -> $index, $token {
            die 'query token is outside uint64'
                unless $token ~~ Int && 0 <= $token <= 2**64 - 1;
            $tokens[$index] = $token;
        }
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
        my $bytes = $source.encode('utf8');
        my Pointer $output .= new;
        check-status(llev-phonetic-rules-parse(
            raw-pointer($bytes), $bytes.elems, $output,
        ), 'phonetic-rules-parse');
        self.bless(handle => $output)
    }

    multi method new(PhoneticRuleSetKind:D :$builtin!) {
        my Pointer $output .= new;
        check-status(llev-phonetic-rules-builtin($builtin, $output),
            'phonetic-rules-builtin');
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

    method elems(--> Int:D) {
        my size_t $output = 0;
        check-status(llev-phonetic-rules-len(self!handle, $output),
            'phonetic-rules-len');
        $output.Int
    }

    method apply(Str:D $input --> Str:D) {
        my $bytes = $input.encode('utf8');
        my $output = OwnedString.new;
        check-status(llev-phonetic-rules-apply(
            self!handle, raw-pointer($bytes), $bytes.elems, $output,
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

INIT {
    die "liblevenshtein native ABI version mismatch"
        unless abi-version() == ABI-VERSION;
    die "liblevenshtein native API revision is too old"
        unless api-revision() >= API-REVISION;
}
