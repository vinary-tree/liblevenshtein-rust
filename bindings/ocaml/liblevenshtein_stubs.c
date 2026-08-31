#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "liblevenshtein.h"
#include "vinary_tree_ocaml.h"

typedef struct { LlevTransducer* value; } OcamlTransducer;
typedef struct { LlevQueryCache* value; } OcamlQueryCache;
typedef struct {
    LlevQueryCursor* value;
    LlevMatchBatchView batch;
    size_t index;
    int leased;
} OcamlCursor;
typedef struct { LlevPhoneticPattern* value; } OcamlPattern;
typedef struct { LlevPhoneticRuleSet* value; } OcamlRules;

static void check_status(LlevStatus status) {
    if (status == LLEV_STATUS_OK) return;
    const char* message = llev_last_error_message();
    caml_failwith(message && *message ? message : "liblevenshtein error");
}

static void transducer_finalize(value block) {
    OcamlTransducer* handle = (OcamlTransducer*)Data_custom_val(block);
    if (handle->value) { llev_transducer_free(handle->value); handle->value = NULL; }
}
static void cursor_finalize(value block) {
    OcamlCursor* handle = (OcamlCursor*)Data_custom_val(block);
    if (!handle->value) return;
    if (handle->leased) {
        (void)llev_query_cursor_release_batch(handle->value, handle->batch.generation);
        handle->leased = 0;
    }
    (void)llev_query_cursor_free(handle->value);
    handle->value = NULL;
}
static void query_cache_finalize(value block) {
    OcamlQueryCache* handle = (OcamlQueryCache*)Data_custom_val(block);
    if (handle->value) { llev_query_cache_free(handle->value); handle->value = NULL; }
}
static void pattern_finalize(value block) {
    OcamlPattern* handle = (OcamlPattern*)Data_custom_val(block);
    if (handle->value) { llev_phonetic_pattern_free(handle->value); handle->value = NULL; }
}
static void rules_finalize(value block) {
    OcamlRules* handle = (OcamlRules*)Data_custom_val(block);
    if (handle->value) { llev_phonetic_rules_free(handle->value); handle->value = NULL; }
}

#define CUSTOM_OPERATIONS(symbol, identifier, finalizer) \
static struct custom_operations symbol = { \
    identifier, finalizer, custom_compare_default, custom_hash_default, \
    custom_serialize_default, custom_deserialize_default, \
    custom_compare_ext_default, custom_fixed_length_default \
}
CUSTOM_OPERATIONS(transducer_operations,
                  "io.vinarytree.liblevenshtein.transducer.v1", transducer_finalize);
CUSTOM_OPERATIONS(cursor_operations,
                  "io.vinarytree.liblevenshtein.cursor.v1", cursor_finalize);
CUSTOM_OPERATIONS(query_cache_operations,
                  "io.vinarytree.liblevenshtein.query-cache.v1", query_cache_finalize);
CUSTOM_OPERATIONS(pattern_operations,
                  "io.vinarytree.liblevenshtein.pattern.v1", pattern_finalize);
CUSTOM_OPERATIONS(rules_operations,
                  "io.vinarytree.liblevenshtein.rules.v1", rules_finalize);

static LlevTransducer* transducer_val(value block) {
    OcamlTransducer* result = (OcamlTransducer*)Data_custom_val(block);
    if (!result->value) caml_invalid_argument("transducer is closed");
    return result->value;
}
static OcamlCursor* cursor_val(value block) {
    OcamlCursor* result = (OcamlCursor*)Data_custom_val(block);
    if (!result->value) caml_invalid_argument("cursor is closed");
    return result;
}
static LlevQueryCache* query_cache_val(value block) {
    OcamlQueryCache* result = (OcamlQueryCache*)Data_custom_val(block);
    if (!result->value) caml_invalid_argument("query cache is closed");
    return result->value;
}
static LlevPhoneticPattern* pattern_val(value block) {
    OcamlPattern* result = (OcamlPattern*)Data_custom_val(block);
    if (!result->value) caml_invalid_argument("phonetic pattern is closed");
    return result->value;
}
static LlevPhoneticRuleSet* rules_val(value block) {
    OcamlRules* result = (OcamlRules*)Data_custom_val(block);
    if (!result->value) caml_invalid_argument("phonetic rules are closed");
    return result->value;
}

static value copy_transducer(LlevTransducer* raw) {
    value block = caml_alloc_custom(&transducer_operations,
                                    sizeof(OcamlTransducer), 0, 1);
    ((OcamlTransducer*)Data_custom_val(block))->value = raw;
    return block;
}
static value copy_cursor(LlevQueryCursor* raw) {
    value block = caml_alloc_custom(&cursor_operations, sizeof(OcamlCursor), 0, 1);
    OcamlCursor* cursor = (OcamlCursor*)Data_custom_val(block);
    memset(cursor, 0, sizeof(*cursor)); cursor->value = raw;
    return block;
}
static value copy_query_cache(LlevQueryCache* raw) {
    value block = caml_alloc_custom(&query_cache_operations,
                                    sizeof(OcamlQueryCache), 0, 1);
    ((OcamlQueryCache*)Data_custom_val(block))->value = raw;
    return block;
}
static value copy_pattern(LlevPhoneticPattern* raw) {
    value block = caml_alloc_custom(&pattern_operations, sizeof(OcamlPattern), 0, 1);
    ((OcamlPattern*)Data_custom_val(block))->value = raw; return block;
}
static value copy_rules(LlevPhoneticRuleSet* raw) {
    value block = caml_alloc_custom(&rules_operations, sizeof(OcamlRules), 0, 1);
    ((OcamlRules*)Data_custom_val(block))->value = raw; return block;
}

static value copy_optional_id(const LlevMatch* input) {
    CAMLparam0(); CAMLlocal2(result, payload);
    if (!input->has_id) CAMLreturn(Val_int(0));
    payload = caml_copy_int64((int64_t)input->id);
    result = caml_alloc(1, 0); Store_field(result, 0, payload); CAMLreturn(result);
}

static value copy_term(const LlevMatch* input) {
    CAMLparam0(); CAMLlocal3(result, payload, token);
    if (input->unit_domain == VT_UNIT_DOMAIN_U64) {
        payload = caml_alloc(input->term_len, 0);
        const uint64_t* tokens = (const uint64_t*)input->term_data;
        for (size_t index = 0; index < input->term_len; ++index) {
            token = caml_copy_int64((int64_t)tokens[index]);
            Store_field(payload, index, token);
        }
        result = caml_alloc(1, 1);
    } else {
        payload = caml_alloc_initialized_string(
            input->byte_len, (const char*)input->term_data);
        result = caml_alloc(1, 0);
    }
    Store_field(result, 0, payload);
    CAMLreturn(result);
}

static value copy_match(const LlevMatch* input) {
    CAMLparam0(); CAMLlocal3(result, term, identifier);
    term = copy_term(input); identifier = copy_optional_id(input);
    result = caml_alloc_tuple(3);
    Store_field(result, 0, term);
    Store_field(result, 1, Val_long(input->distance));
    Store_field(result, 2, identifier);
    CAMLreturn(result);
}

static int refill(OcamlCursor* cursor, size_t maximum) {
    if (cursor->leased) {
        check_status(llev_query_cursor_release_batch(
            cursor->value, cursor->batch.generation));
        cursor->leased = 0;
    }
    memset(&cursor->batch, 0, sizeof(cursor->batch));
    LlevStatus status = llev_query_cursor_next_batch(
        cursor->value, maximum, &cursor->batch);
    if (status == LLEV_STATUS_END) return 0;
    check_status(status);
    if (cursor->batch.len == 0) return 0;
    cursor->leased = 1; cursor->index = 0; return 1;
}

CAMLprim value ocaml_llev_transducer(value resource, value algorithm) {
    CAMLparam2(resource, algorithm);
    LlevTransducer* output = NULL;
    check_status(llev_transducer_new(vt_ocaml_get_resource(resource),
        (uint32_t)Int_val(algorithm), &output));
    CAMLreturn(copy_transducer(output));
}
CAMLprim value ocaml_llev_transducer_close(value block) {
    CAMLparam1(block); transducer_finalize(block); CAMLreturn(Val_unit);
}

CAMLprim value ocaml_llev_query_cache_new(value transducer, value entries,
                                           value weight) {
    CAMLparam3(transducer, entries, weight);
    if (Long_val(entries) < 0 || Long_val(weight) < 0)
        caml_invalid_argument("negative query-cache limit");
    LlevQueryCache* output = NULL;
    check_status(llev_query_cache_new(transducer_val(transducer),
        Long_val(entries), Long_val(weight), &output));
    CAMLreturn(copy_query_cache(output));
}
CAMLprim value ocaml_llev_query_cache_close(value block) {
    CAMLparam1(block); query_cache_finalize(block); CAMLreturn(Val_unit);
}
CAMLprim value ocaml_llev_query_cache_clear(value block) {
    CAMLparam1(block);
    check_status(llev_query_cache_clear(query_cache_val(block)));
    CAMLreturn(Val_unit);
}
CAMLprim value ocaml_llev_query_cache_reset_stats(value block) {
    CAMLparam1(block);
    check_status(llev_query_cache_reset_stats(query_cache_val(block)));
    CAMLreturn(Val_unit);
}
CAMLprim value ocaml_llev_query_cache_stats(value block) {
    CAMLparam1(block); CAMLlocal2(result, counter);
    LlevQueryCacheStats stats = {0};
    check_status(llev_query_cache_stats(query_cache_val(block), &stats));
    result = caml_alloc_tuple(8);
#define STORE_COUNTER(index, field) \
    counter = caml_copy_int64((int64_t)stats.field); Store_field(result, index, counter)
    STORE_COUNTER(0, requests);
    STORE_COUNTER(1, hits);
    STORE_COUNTER(2, misses);
    STORE_COUNTER(3, admissions);
    STORE_COUNTER(4, rejections);
    STORE_COUNTER(5, evictions);
#undef STORE_COUNTER
    Store_field(result, 6, Val_long(stats.resident_entries));
    Store_field(result, 7, Val_long(stats.resident_weight));
    CAMLreturn(result);
}

static value query_cache_text(value cache, value input, value maximum,
                              value order, int bytes) {
    CAMLparam4(cache, input, maximum, order);
    if (Long_val(maximum) < 0) caml_invalid_argument("negative maximum distance");
    LlevQueryCursor* output = NULL;
    LlevStatus status = bytes
        ? llev_query_cache_query_bytes(query_cache_val(cache),
            (const uint8_t*)String_val(input), caml_string_length(input),
            Long_val(maximum), (uint32_t)Int_val(order), &output)
        : llev_query_cache_query_utf8(query_cache_val(cache), String_val(input),
            caml_string_length(input), Long_val(maximum),
            (uint32_t)Int_val(order), &output);
    check_status(status); CAMLreturn(copy_cursor(output));
}
CAMLprim value ocaml_llev_query_cache_query(value cache, value input,
                                             value maximum, value order) {
    return query_cache_text(cache, input, maximum, order, 0);
}
CAMLprim value ocaml_llev_query_cache_query_bytes(value cache, value input,
                                                   value maximum, value order) {
    return query_cache_text(cache, input, maximum, order, 1);
}
CAMLprim value ocaml_llev_query_cache_query_u64(value cache, value input,
                                                 value maximum, value order) {
    CAMLparam4(cache, input, maximum, order);
    if (Long_val(maximum) < 0) caml_invalid_argument("negative maximum distance");
    size_t length = Wosize_val(input);
    uint64_t* tokens = length ? malloc(length * sizeof(uint64_t)) : NULL;
    if (length && !tokens) caml_raise_out_of_memory();
    for (size_t index = 0; index < length; ++index)
        tokens[index] = (uint64_t)Int64_val(Field(input, index));
    LlevQueryCursor* output = NULL;
    LlevStatus status = llev_query_cache_query_u64(query_cache_val(cache),
        tokens, length, Long_val(maximum), (uint32_t)Int_val(order), &output);
    free(tokens); check_status(status); CAMLreturn(copy_cursor(output));
}

static value query_text(value transducer, value input, value maximum,
                        value order, int bytes) {
    CAMLparam4(transducer, input, maximum, order);
    if (Long_val(maximum) < 0) caml_invalid_argument("negative maximum distance");
    LlevQueryCursor* output = NULL;
    LlevStatus status = bytes
        ? llev_transducer_query_bytes(transducer_val(transducer),
            (const uint8_t*)String_val(input), caml_string_length(input),
            Long_val(maximum), (uint32_t)Int_val(order), &output)
        : llev_transducer_query_utf8(transducer_val(transducer), String_val(input),
            caml_string_length(input), Long_val(maximum),
            (uint32_t)Int_val(order), &output);
    check_status(status); CAMLreturn(copy_cursor(output));
}
CAMLprim value ocaml_llev_query(value transducer, value input,
                                value maximum, value order) {
    return query_text(transducer, input, maximum, order, 0);
}
CAMLprim value ocaml_llev_query_bytes(value transducer, value input,
                                      value maximum, value order) {
    return query_text(transducer, input, maximum, order, 1);
}

CAMLprim value ocaml_llev_query_u64(value transducer, value input,
                                    value maximum, value order) {
    CAMLparam4(transducer, input, maximum, order);
    if (Long_val(maximum) < 0) caml_invalid_argument("negative maximum distance");
    size_t length = Wosize_val(input);
    uint64_t* tokens = length ? malloc(length * sizeof(uint64_t)) : NULL;
    if (length && !tokens) caml_raise_out_of_memory();
    for (size_t index = 0; index < length; ++index)
        tokens[index] = (uint64_t)Int64_val(Field(input, index));
    LlevQueryCursor* output = NULL;
    LlevStatus status = llev_transducer_query_u64(transducer_val(transducer),
        tokens, length, Long_val(maximum), (uint32_t)Int_val(order), &output);
    free(tokens); check_status(status); CAMLreturn(copy_cursor(output));
}

CAMLprim value ocaml_llev_query_pattern(value transducer, value pattern,
                                        value maximum) {
    CAMLparam3(transducer, pattern, maximum);
    long distance = Long_val(maximum);
    if (distance < 0 || distance > UINT8_MAX)
        caml_invalid_argument("pattern maximum distance is outside u8");
    LlevQueryCursor* output = NULL;
    check_status(llev_transducer_query_pattern(transducer_val(transducer),
        pattern_val(pattern), (uint8_t)distance, &output));
    CAMLreturn(copy_cursor(output));
}

CAMLprim value ocaml_llev_cursor_close(value block) {
    CAMLparam1(block);
    OcamlCursor* cursor = cursor_val(block);
    if (cursor->leased) {
        check_status(llev_query_cursor_release_batch(
            cursor->value, cursor->batch.generation));
        cursor->leased = 0;
    }
    check_status(llev_query_cursor_free(cursor->value)); cursor->value = NULL;
    CAMLreturn(Val_unit);
}

CAMLprim value ocaml_llev_cursor_next(value block) {
    CAMLparam1(block); CAMLlocal2(result, item);
    OcamlCursor* cursor = cursor_val(block);
    if ((!cursor->leased || cursor->index == cursor->batch.len) &&
        !refill(cursor, LLEV_DEFAULT_MATCH_BATCH)) CAMLreturn(Val_int(0));
    item = copy_match(&cursor->batch.matches[cursor->index++]);
    result = caml_alloc(1, 0); Store_field(result, 0, item); CAMLreturn(result);
}

CAMLprim value ocaml_llev_cursor_next_batch(value block, value maximum) {
    CAMLparam2(block, maximum); CAMLlocal3(result, array, item);
    long requested = Long_val(maximum);
    if (requested <= 0) caml_invalid_argument("batch size must be positive");
    OcamlCursor* cursor = cursor_val(block);
    if (!refill(cursor, (size_t)requested)) CAMLreturn(Val_int(0));
    array = caml_alloc(cursor->batch.len, 0);
    for (size_t index = 0; index < cursor->batch.len; ++index) {
        item = copy_match(&cursor->batch.matches[index]);
        Store_field(array, index, item);
    }
    cursor->index = cursor->batch.len;
    result = caml_alloc(1, 0); Store_field(result, 0, array); CAMLreturn(result);
}

static value compile_pattern(value source, int llre) {
    CAMLparam1(source);
    LlevPhoneticPattern* output = NULL;
    LlevStatus status = llre
        ? llev_phonetic_pattern_compile_llre(String_val(source),
            caml_string_length(source), &output)
        : llev_phonetic_pattern_compile_regex(String_val(source),
            caml_string_length(source), &output);
    check_status(status); CAMLreturn(copy_pattern(output));
}
CAMLprim value ocaml_llev_regex_pattern(value source) {
    return compile_pattern(source, 0);
}
CAMLprim value ocaml_llev_llre_pattern(value source) {
    return compile_pattern(source, 1);
}
CAMLprim value ocaml_llev_pattern_matches(value pattern, value input) {
    uint8_t output = 0;
    check_status(llev_phonetic_pattern_matches(pattern_val(pattern),
        String_val(input), caml_string_length(input), &output));
    return Val_bool(output);
}
CAMLprim value ocaml_llev_pattern_size(value pattern) {
    CAMLparam1(pattern); CAMLlocal1(result);
    size_t states = 0, transitions = 0;
    check_status(llev_phonetic_pattern_size(pattern_val(pattern),
        &states, &transitions));
    result = caml_alloc_tuple(2);
    Store_field(result, 0, Val_long(states));
    Store_field(result, 1, Val_long(transitions)); CAMLreturn(result);
}
CAMLprim value ocaml_llev_pattern_close(value block) {
    CAMLparam1(block); pattern_finalize(block); CAMLreturn(Val_unit);
}

CAMLprim value ocaml_llev_phonetic_rules(value source) {
    CAMLparam1(source);
    LlevPhoneticRuleSet* output = NULL; LlevStatus status;
    size_t length = caml_string_length(source);
    if (length == 19 && memcmp(String_val(source), "english-orthography", 19) == 0)
        status = llev_phonetic_rules_builtin(
            LLEV_PHONETIC_RULE_SET_ENGLISH_ORTHOGRAPHY, &output);
    else if (length == 16 && memcmp(String_val(source), "english-phonetic", 16) == 0)
        status = llev_phonetic_rules_builtin(
            LLEV_PHONETIC_RULE_SET_ENGLISH_PHONETIC, &output);
    else status = llev_phonetic_rules_parse(String_val(source), length, &output);
    check_status(status); CAMLreturn(copy_rules(output));
}
CAMLprim value ocaml_llev_rules_length(value rules) {
    size_t output = 0;
    check_status(llev_phonetic_rules_len(rules_val(rules), &output));
    return Val_long(output);
}
CAMLprim value ocaml_llev_apply_rules(value rules, value input) {
    CAMLparam2(rules, input); CAMLlocal1(result);
    LlevOwnedString output = {0};
    check_status(llev_phonetic_rules_apply(rules_val(rules), String_val(input),
        caml_string_length(input), &output));
    result = caml_alloc_initialized_string(output.len, output.data);
    llev_owned_string_free(&output); CAMLreturn(result);
}
CAMLprim value ocaml_llev_rules_close(value block) {
    CAMLparam1(block); rules_finalize(block); CAMLreturn(Val_unit);
}

#define DISTANCE_STUB(name, function) \
CAMLprim value name(value source, value target) { \
    return Val_long(function(String_val(source), caml_string_length(source), \
        String_val(target), caml_string_length(target))); \
}
#define THRESHOLD_STUB(name, function) \
CAMLprim value name(value source, value target, value threshold) { \
    if (Long_val(threshold) < 0) caml_invalid_argument("negative threshold"); \
    return Val_long(function(String_val(source), caml_string_length(source), \
        String_val(target), caml_string_length(target), Long_val(threshold))); \
}
DISTANCE_STUB(ocaml_llev_distance, llev_distance)
THRESHOLD_STUB(ocaml_llev_distance_threshold, llev_distance_threshold)
DISTANCE_STUB(ocaml_llev_damerau_distance, llev_damerau_distance)
THRESHOLD_STUB(ocaml_llev_damerau_distance_threshold,
               llev_damerau_distance_threshold)
DISTANCE_STUB(ocaml_llev_true_damerau_distance, llev_true_damerau_distance)
THRESHOLD_STUB(ocaml_llev_true_damerau_distance_threshold,
               llev_true_damerau_distance_threshold)
