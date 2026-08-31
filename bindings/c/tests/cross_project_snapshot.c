#include "libdictenstein.h"
#include "liblevenshtein.h"

#include <assert.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * C1 (reduced interop contract): layout and discriminant mirror pins. Every
 * facade in every language decodes these records by offset and branches on
 * these discriminants, so the C oracle pins both here. A silent renumbering or
 * field reorder in the shared ABI must break this translation unit at compile
 * time, before any downstream binding mistranslates a status or a match.
 */
_Static_assert(LLEV_STATUS_OK == 0, "LlevStatus OK discriminant moved");
_Static_assert(LLEV_STATUS_END == 1, "LlevStatus END discriminant moved");
_Static_assert(LLEV_STATUS_BATCH_IN_USE == 11, "LlevStatus BATCH_IN_USE moved");
_Static_assert(LLEV_STATUS_DOMAIN_MISMATCH == 12, "LlevStatus DOMAIN_MISMATCH moved");
_Static_assert(LLEV_ALGORITHM_STANDARD == 0, "LlevAlgorithm STANDARD moved");
_Static_assert(LLEV_ALGORITHM_TRANSPOSITION == 1,
               "LlevAlgorithm TRANSPOSITION moved");
_Static_assert(LLEV_ALGORITHM_MERGE_AND_SPLIT == 2,
               "LlevAlgorithm MERGE_AND_SPLIT moved");
_Static_assert(LLEV_ALGORITHM_DAMERAU_LEVENSHTEIN == 3,
               "LlevAlgorithm DAMERAU_LEVENSHTEIN moved");
_Static_assert(LLEV_QUERY_ORDER_TRAVERSAL == 0, "LlevQueryOrder TRAVERSAL moved");
_Static_assert(LLEV_QUERY_ORDER_DISTANCE_THEN_TERM == 1,
               "LlevQueryOrder DISTANCE_THEN_TERM moved");
_Static_assert(LLEV_PHONETIC_RULE_SET_ENGLISH_ORTHOGRAPHY == 0,
               "LlevPhoneticRuleSetKind ENGLISH_ORTHOGRAPHY moved");
_Static_assert(LLEV_PHONETIC_RULE_SET_ENGLISH_PHONETIC == 1,
               "LlevPhoneticRuleSetKind ENGLISH_PHONETIC moved");
_Static_assert(VT_STATUS_OK == 0, "VtStatus OK discriminant moved");
_Static_assert(VT_UNIT_DOMAIN_UNICODE_SCALAR == 2, "VtUnitDomain scalar moved");
_Static_assert(offsetof(LlevMatch, term_data) == 0, "LlevMatch.term_data must lead");
_Static_assert(offsetof(LlevMatch, term_len) == sizeof(const void*),
               "LlevMatch.term_len must follow the term pointer");
_Static_assert(offsetof(LlevMatchBatchView, matches) == 0,
               "LlevMatchBatchView.matches must lead");
_Static_assert(offsetof(LlevMatchBatchView, len) == sizeof(const LlevMatch*),
               "LlevMatchBatchView.len must follow the match pointer");
_Static_assert(sizeof(VtResource) == 2 * sizeof(void*),
               "VtResource must remain a two-word handle");

typedef struct Seen {
    bool cat;
    bool cot;
    bool cut;
    bool scat;
    size_t count;
} Seen;

static void remember(Seen* seen, const LlevMatch* item) {
    assert(item->unit_domain == VT_UNIT_DOMAIN_UNICODE_SCALAR);
    const char* term = (const char*)item->term_data;
    if (item->byte_len == 3 && memcmp(term, "cat", 3) == 0) seen->cat = true;
    if (item->byte_len == 3 && memcmp(term, "cot", 3) == 0) seen->cot = true;
    if (item->byte_len == 3 && memcmp(term, "cut", 3) == 0) seen->cut = true;
    if (item->byte_len == 4 && memcmp(term, "scat", 4) == 0) seen->scat = true;
    ++seen->count;
}

static void insert(LdictDictionary* dictionary, const char* term, uint64_t value) {
    uint8_t inserted = 0;
    LdictOptionalU64 optional = {value, 1, {0}};
    assert(ldict_dictionary_insert_text(
               dictionary, (const uint8_t*)term, strlen(term), optional, &inserted) ==
           LDICT_STATUS_OK);
    assert(inserted == 1);
}

static bool cursor_contains_text(LlevTransducer* transducer,
                                 const char* query,
                                 size_t query_len,
                                 size_t maximum_distance,
                                 uint32_t order,
                                 const char* expected,
                                 size_t expected_distance) {
    LlevQueryCursor* cursor = NULL;
    assert(llev_transducer_query_utf8(transducer, query, query_len,
                                      maximum_distance, order, &cursor) ==
           LLEV_STATUS_OK);
    bool found = false;
    for (;;) {
        LlevMatchBatchView batch = {0};
        LlevStatus status = llev_query_cursor_next_batch(cursor, 2, &batch);
        if (status == LLEV_STATUS_END) break;
        assert(status == LLEV_STATUS_OK);
        for (size_t i = 0; i < batch.len; ++i) {
            const LlevMatch* match = &batch.matches[i];
            assert(match->unit_domain == VT_UNIT_DOMAIN_UNICODE_SCALAR);
            if (match->byte_len == strlen(expected) &&
                memcmp(match->term_data, expected, match->byte_len) == 0 &&
                match->distance == expected_distance) {
                found = true;
            }
        }
        assert(llev_query_cursor_release_batch(cursor, batch.generation) ==
               LLEV_STATUS_OK);
    }
    assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);
    return found;
}

static void assert_distance_api(void) {
    const size_t exceeded = SIZE_MAX - 1;
    assert(llev_distance("kitten", 6, "sitting", 7) == 3);
    assert(llev_distance_threshold("kitten", 6, "sitting", 7, 3) == 3);
    assert(llev_distance_threshold("kitten", 6, "sitting", 7, 2) == exceeded);
    assert(llev_damerau_distance("ab", 2, "ba", 2) == 1);
    assert(llev_damerau_distance_threshold("ab", 2, "ba", 2, 1) == 1);
    assert(llev_damerau_distance_threshold("ab", 2, "ba", 2, 0) == exceeded);
    assert(llev_true_damerau_distance("ca", 2, "abc", 3) == 2);
    assert(llev_true_damerau_distance_threshold("ca", 2, "abc", 3, 2) == 2);
    assert(llev_true_damerau_distance_threshold("ca", 2, "abc", 3, 1) ==
           exceeded);

    /* The six functions count Unicode scalar values rather than UTF-8 bytes. */
    assert(llev_distance("caf\xC3\xA9", 5, "cafe", 4) == 1);
    assert(llev_true_damerau_distance("\xF0\x9F\xA6\x80", 4, "x", 1) == 1);
}

static void assert_legacy_string_api(void) {
    char* duplicate = llev_string_dup("owned by liblevenshtein");
    assert(duplicate != NULL);
    assert(strcmp(duplicate, "owned by liblevenshtein") == 0);
    llev_string_free(duplicate);
    assert(llev_string_dup(NULL) == NULL);
    llev_string_free(NULL);

    /* The array release symbol is retained for arrays produced by legacy ABI
     * revisions. Current APIs return LlevOwnedString instead; NULL is its only
     * locally constructible, allocator-correct compatibility case. */
    llev_string_array_free(NULL, 0);
}

typedef struct Reduction {
    size_t callbacks;
    size_t matches;
} Reduction;

static LlevStatus count_reduced_batch(void* context,
                                      const LlevMatch* matches,
                                      size_t len) {
    Reduction* reduction = (Reduction*)context;
    assert(reduction != NULL);
    assert(matches != NULL || len == 0);
    ++reduction->callbacks;
    reduction->matches += len;
    return LLEV_STATUS_OK;
}

static void assert_reducer_api(LlevTransducer* transducer) {
    LlevQueryCursor* cursor = NULL;
    assert(llev_transducer_query_utf8(transducer, "cat", 3, 2,
                                      LLEV_QUERY_ORDER_TRAVERSAL, &cursor) ==
           LLEV_STATUS_OK);
    Reduction reduction = {0};
    size_t reduced = 0;
    assert(llev_query_cursor_reduce(cursor, 2, count_reduced_batch, &reduction,
                                    &reduced) == LLEV_STATUS_OK);
    assert(reduction.callbacks == 2);
    assert(reduction.matches == 4);
    assert(reduced == reduction.matches);
    assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);
}

static void assert_query_cache_api(LlevTransducer* transducer) {
    LlevQueryCache* cache = NULL;
    assert(llev_query_cache_new(transducer, 8, 1024 * 1024, &cache) ==
           LLEV_STATUS_OK);
    LlevQueryCursor* cursor = NULL;
    assert(llev_query_cache_query_utf8(
               cache, "cat", 3, 2, LLEV_QUERY_ORDER_TRAVERSAL, &cursor) ==
           LLEV_STATUS_OK);
    assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);
    assert(llev_query_cache_query_utf8(
               cache, "cat", 3, 2, LLEV_QUERY_ORDER_TRAVERSAL, &cursor) ==
           LLEV_STATUS_OK);
    assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);

    const uint8_t bytes[] = {0, 1};
    assert(llev_query_cache_query_bytes(
               cache, bytes, 2, 1, LLEV_QUERY_ORDER_TRAVERSAL, &cursor) ==
           LLEV_STATUS_DOMAIN_MISMATCH);
    const uint64_t tokens[] = {0, 1};
    assert(llev_query_cache_query_u64(
               cache, tokens, 2, 1, LLEV_QUERY_ORDER_TRAVERSAL, &cursor) ==
           LLEV_STATUS_DOMAIN_MISMATCH);

    LlevQueryCacheStats stats = {0};
    assert(llev_query_cache_stats(cache, &stats) == LLEV_STATUS_OK);
    assert(stats.requests == 4 && stats.hits == 1 && stats.misses == 3);
    assert(stats.resident_entries == 1 && stats.resident_weight > 0);
    assert(llev_query_cache_clear(cache) == LLEV_STATUS_OK);
    assert(llev_query_cache_stats(cache, &stats) == LLEV_STATUS_OK);
    assert(stats.resident_entries == 0);
    assert(llev_query_cache_reset_stats(cache) == LLEV_STATUS_OK);
    assert(llev_query_cache_stats(cache, &stats) == LLEV_STATUS_OK);
    assert(stats.requests == 0);
    llev_query_cache_free(cache);
}

static void assert_phonetic_api(LlevTransducer* transducer) {
    assert((llev_build_features() & LLEV_BUILD_FEATURE_PHONETIC) != 0);

    LlevPhoneticPattern* regex = NULL;
    assert(llev_phonetic_pattern_compile_regex("c[ao]t", 6, &regex) ==
           LLEV_STATUS_OK);
    size_t states = 0;
    size_t transitions = 0;
    assert(llev_phonetic_pattern_size(regex, &states, &transitions) ==
           LLEV_STATUS_OK);
    assert(states > 0 && transitions > 0);
    uint8_t matches = 0;
    assert(llev_phonetic_pattern_matches(regex, "cat", 3, &matches) ==
           LLEV_STATUS_OK);
    assert(matches == 1);
    assert(llev_phonetic_pattern_matches(regex, "cut", 3, &matches) ==
           LLEV_STATUS_OK);
    assert(matches == 0);

    LlevQueryCursor* product = NULL;
    assert(llev_transducer_query_pattern(transducer, regex, 0, &product) ==
           LLEV_STATUS_OK);
    Reduction reduction = {0};
    size_t reduced = 0;
    assert(llev_query_cursor_reduce(product, 1, count_reduced_batch, &reduction,
                                    &reduced) == LLEV_STATUS_OK);
    assert(reduced == 2);
    assert(llev_query_cursor_free(product) == LLEV_STATUS_OK);
    llev_phonetic_pattern_free(regex);

    static const char llre[] = "@name \"Greeting\"\n^hello$";
    LlevPhoneticPattern* language = NULL;
    assert(llev_phonetic_pattern_compile_llre(llre, sizeof(llre) - 1,
                                              &language) == LLEV_STATUS_OK);
    assert(llev_phonetic_pattern_matches(language, "hello", 5, &matches) ==
           LLEV_STATUS_OK);
    assert(matches == 1);
    assert(llev_phonetic_pattern_matches(language, "world", 5, &matches) ==
           LLEV_STATUS_OK);
    assert(matches == 0);
    llev_phonetic_pattern_free(language);

    LlevPhoneticPattern* invalid = NULL;
    assert(llev_phonetic_pattern_compile_regex("(", 1, &invalid) ==
           LLEV_STATUS_INVALID_ARGUMENT);
    assert(invalid == NULL);
    const char* diagnostic = llev_last_error_message();
    assert(diagnostic != NULL && diagnostic[0] != '\0');

    static const char rules_source[] = "ph -> f\ngh ->\n";
    LlevPhoneticRuleSet* rules = NULL;
    assert(llev_phonetic_rules_parse(rules_source, sizeof(rules_source) - 1,
                                     &rules) == LLEV_STATUS_OK);
    size_t rule_count = 0;
    assert(llev_phonetic_rules_len(rules, &rule_count) == LLEV_STATUS_OK);
    assert(rule_count == 2);
    LlevOwnedString output = {0};
    assert(llev_phonetic_rules_apply(rules, "phgh", 4, &output) ==
           LLEV_STATUS_OK);
    assert(output.len == 1 && memcmp(output.data, "f", 1) == 0);
    llev_owned_string_free(&output);
    assert(output.data == NULL && output.len == 0);
    llev_phonetic_rules_free(rules);

    const uint32_t kinds[] = {
        LLEV_PHONETIC_RULE_SET_ENGLISH_ORTHOGRAPHY,
        LLEV_PHONETIC_RULE_SET_ENGLISH_PHONETIC,
    };
    for (size_t i = 0; i < sizeof(kinds) / sizeof(kinds[0]); ++i) {
        LlevPhoneticRuleSet* builtin = NULL;
        assert(llev_phonetic_rules_builtin(kinds[i], &builtin) == LLEV_STATUS_OK);
        assert(llev_phonetic_rules_len(builtin, &rule_count) == LLEV_STATUS_OK);
        assert(rule_count > 0);
        assert(llev_phonetic_rules_apply(builtin, "phone", 5, &output) ==
               LLEV_STATUS_OK);
        assert(output.data != NULL && output.len > 0);
        llev_owned_string_free(&output);
        llev_phonetic_rules_free(builtin);
    }
}

static void assert_algorithm_and_order_api(void) {
    LdictDictionary* dictionary = NULL;
    assert(ldict_dynamic_dawg_new(VT_UNIT_DOMAIN_UNICODE_SCALAR, &dictionary) ==
           LDICT_STATUS_OK);
    insert(dictionary, "ab", 1);
    insert(dictionary, "c", 2);
    insert(dictionary, "abc", 3);
    insert(dictionary, "bat", 4);
    insert(dictionary, "cat", 5);
    insert(dictionary, "cats", 6);
    VtResource resource = {0};
    assert(ldict_dictionary_resource(dictionary, &resource) == LDICT_STATUS_OK);

    LlevTransducer* standard = NULL;
    LlevTransducer* transposition = NULL;
    LlevTransducer* merge_and_split = NULL;
    LlevTransducer* damerau = NULL;
    assert(llev_transducer_new(&resource, LLEV_ALGORITHM_STANDARD, &standard) ==
           LLEV_STATUS_OK);
    assert(llev_transducer_new(&resource, LLEV_ALGORITHM_TRANSPOSITION,
                               &transposition) == LLEV_STATUS_OK);
    assert(llev_transducer_new(&resource, LLEV_ALGORITHM_MERGE_AND_SPLIT,
                               &merge_and_split) == LLEV_STATUS_OK);
    assert(llev_transducer_new(&resource, LLEV_ALGORITHM_DAMERAU_LEVENSHTEIN,
                               &damerau) == LLEV_STATUS_OK);
    assert(!cursor_contains_text(standard, "ba", 2, 1,
                                 LLEV_QUERY_ORDER_TRAVERSAL, "ab", 1));
    assert(cursor_contains_text(transposition, "ba", 2, 1,
                                LLEV_QUERY_ORDER_TRAVERSAL, "ab", 1));
    assert(cursor_contains_text(merge_and_split, "ab", 2, 1,
                                LLEV_QUERY_ORDER_TRAVERSAL, "c", 1));
    assert(cursor_contains_text(damerau, "ca", 2, 2,
                                LLEV_QUERY_ORDER_TRAVERSAL, "abc", 2));

    LlevQueryCursor* ordered = NULL;
    assert(llev_transducer_query_utf8(standard, "cat", 3, 1,
                                      LLEV_QUERY_ORDER_DISTANCE_THEN_TERM,
                                      &ordered) == LLEV_STATUS_OK);
    LlevMatchBatchView batch = {0};
    assert(llev_query_cursor_next_batch(ordered, 8, &batch) == LLEV_STATUS_OK);
    assert(batch.len == 3);
    assert(batch.matches[0].distance == 0 &&
           memcmp(batch.matches[0].term_data, "cat", 3) == 0);
    assert(batch.matches[1].distance == 1 &&
           memcmp(batch.matches[1].term_data, "bat", 3) == 0);
    assert(batch.matches[2].distance == 1 &&
           memcmp(batch.matches[2].term_data, "cats", 4) == 0);
    assert(llev_query_cursor_release_batch(ordered, batch.generation) ==
           LLEV_STATUS_OK);
    assert(llev_query_cursor_free(ordered) == LLEV_STATUS_OK);

    llev_transducer_free(damerau);
    llev_transducer_free(merge_and_split);
    llev_transducer_free(transposition);
    llev_transducer_free(standard);
    ldict_dictionary_free(dictionary);
}

static void assert_non_text_domains(void) {
    LdictDictionary* bytes = NULL;
    assert(ldict_dynamic_dawg_new(VT_UNIT_DOMAIN_BYTE, &bytes) == LDICT_STATUS_OK);
    const uint8_t byte_term[] = {0xff, 0x00, 0x7f};
    uint8_t inserted = 0;
    LdictOptionalU64 byte_value = {UINT64_MAX, 1, {0}};
    assert(ldict_dictionary_insert_text(bytes, byte_term, sizeof(byte_term),
                                        byte_value, &inserted) == LDICT_STATUS_OK);
    assert(inserted == 1);
    VtResource resource = {0};
    assert(ldict_dictionary_resource(bytes, &resource) == LDICT_STATUS_OK);
    LlevTransducer* byte_transducer = NULL;
    assert(llev_transducer_new(&resource, LLEV_ALGORITHM_STANDARD,
                               &byte_transducer) == LLEV_STATUS_OK);
    const uint8_t byte_query[] = {0xff, 0x00, 0x7e};
    LlevQueryCursor* cursor = NULL;
    assert(llev_transducer_query_bytes(byte_transducer, byte_query,
                                       sizeof(byte_query), 1,
                                       LLEV_QUERY_ORDER_TRAVERSAL, &cursor) ==
           LLEV_STATUS_OK);
    LlevMatchBatchView batch = {0};
    assert(llev_query_cursor_next_batch(cursor, 1, &batch) == LLEV_STATUS_OK);
    assert(batch.len == 1);
    assert(batch.matches[0].unit_domain == VT_UNIT_DOMAIN_BYTE);
    assert(batch.matches[0].term_len == sizeof(byte_term));
    assert(batch.matches[0].byte_len == sizeof(byte_term));
    assert(memcmp(batch.matches[0].term_data, byte_term, sizeof(byte_term)) == 0);
    assert(batch.matches[0].distance == 1);
    assert(batch.matches[0].has_id == 1 && batch.matches[0].id == UINT64_MAX);
    assert(llev_query_cursor_release_batch(cursor, batch.generation) ==
           LLEV_STATUS_OK);
    assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);
    llev_transducer_free(byte_transducer);
    ldict_dictionary_free(bytes);

    LdictDictionary* tokens = NULL;
    assert(ldict_dynamic_dawg_new(VT_UNIT_DOMAIN_U64, &tokens) == LDICT_STATUS_OK);
    const uint64_t token_term[] = {0, UINT64_MAX};
    LdictOptionalU64 token_value = {7, 1, {0}};
    assert(ldict_dictionary_insert_u64(tokens, token_term, 2, token_value,
                                       &inserted) == LDICT_STATUS_OK);
    assert(inserted == 1);
    assert(ldict_dictionary_resource(tokens, &resource) == LDICT_STATUS_OK);
    LlevTransducer* token_transducer = NULL;
    assert(llev_transducer_new(&resource, LLEV_ALGORITHM_STANDARD,
                               &token_transducer) == LLEV_STATUS_OK);
    const uint64_t token_query[] = {0, UINT64_MAX - 1};
    assert(llev_transducer_query_u64(token_transducer, token_query, 2, 1,
                                     LLEV_QUERY_ORDER_TRAVERSAL, &cursor) ==
           LLEV_STATUS_OK);
    assert(llev_query_cursor_next_batch(cursor, 1, &batch) == LLEV_STATUS_OK);
    assert(batch.len == 1);
    assert(batch.matches[0].unit_domain == VT_UNIT_DOMAIN_U64);
    assert(batch.matches[0].term_len == 2);
    assert(batch.matches[0].byte_len == 2 * sizeof(uint64_t));
    assert(memcmp(batch.matches[0].term_data, token_term, sizeof(token_term)) == 0);
    assert(batch.matches[0].distance == 1);
    assert(batch.matches[0].has_id == 1 && batch.matches[0].id == 7);
    assert(llev_query_cursor_release_batch(cursor, batch.generation) ==
           LLEV_STATUS_OK);
    assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);
    llev_transducer_free(token_transducer);
    ldict_dictionary_free(tokens);
}

int main(void) {
    assert_distance_api();
    assert_legacy_string_api();
    assert_algorithm_and_order_api();
    assert_non_text_domains();

    LdictDictionary* dictionary = NULL;
    assert(ldict_dynamic_dawg_new(VT_UNIT_DOMAIN_UNICODE_SCALAR, &dictionary) ==
           LDICT_STATUS_OK);
    insert(dictionary, "cat", 1);
    insert(dictionary, "cot", 2);
    insert(dictionary, "cut", 3);
    insert(dictionary, "scat", 4);

    VtResource resource = {0};
    assert(ldict_dictionary_resource(dictionary, &resource) == LDICT_STATUS_OK);
    LlevTransducer* transducer = NULL;
    assert(llev_transducer_new(&resource, LLEV_ALGORITHM_STANDARD, &transducer) ==
           LLEV_STATUS_OK);

    assert_reducer_api(transducer);
    assert_query_cache_api(transducer);
    assert_phonetic_api(transducer);

    LlevTransducer* frozen = NULL;
    assert(llev_transducer_snapshot(transducer, &frozen) == LLEV_STATUS_OK);

    /* C1 (reduced): identity and version of the loaded shared objects, plus the
     * transducer's inherited unit domain. */
    assert(llev_abi_version() == LLEV_ABI_VERSION);
    assert(llev_api_revision() == LLEV_API_REVISION);
    assert((llev_build_features() & LLEV_BUILD_FEATURE_CORE) != 0);
    VtUnitDomain domain = VT_UNIT_DOMAIN_BYTE;
    assert(llev_transducer_unit_domain(transducer, &domain) == LLEV_STATUS_OK);
    assert(domain == VT_UNIT_DOMAIN_UNICODE_SCALAR);

    LlevQueryCursor* cursor = NULL;
    assert(llev_transducer_query_utf8(transducer, "cat", 3, 2,
                                      LLEV_QUERY_ORDER_TRAVERSAL, &cursor) ==
           LLEV_STATUS_OK);
    Seen seen = {0};
    LlevMatchBatchView batch = {0};
    assert(llev_query_cursor_next_batch(cursor, 1, &batch) == LLEV_STATUS_OK);
    assert(batch.len == 1);
    remember(&seen, &batch.matches[0]);
    /* C2/C5 (reduced): a live batch lease pins the cursor. Closing it is refused
     * with BATCH_IN_USE rather than invalidating the borrowed descriptors. */
    assert(llev_query_cursor_free(cursor) == LLEV_STATUS_BATCH_IN_USE);
    assert(llev_query_cursor_release_batch(cursor, batch.generation) == LLEV_STATUS_OK);

    uint8_t changed = 0;
    assert(ldict_dictionary_remove_text(dictionary, (const uint8_t*)"cot", 3,
                                        &changed) == LDICT_STATUS_OK);
    assert(changed == 1);
    LdictOptionalU64 updated = {30, 1, {0}};
    assert(ldict_dictionary_insert_text(dictionary, (const uint8_t*)"cut", 3,
                                        updated, &changed) == LDICT_STATUS_OK);
    assert(changed == 0);
    insert(dictionary, "cit", 5);
    size_t reclaimed = 0;
    assert(ldict_dictionary_compact(dictionary, &reclaimed) == LDICT_STATUS_OK);
    assert(ldict_dictionary_clear(dictionary) == LDICT_STATUS_OK);
    insert(dictionary, "new", 99);

    LlevQueryCursor* fresh = NULL;
    assert(llev_transducer_query_utf8(transducer, "cat", 3, 8,
                                      LLEV_QUERY_ORDER_TRAVERSAL, &fresh) ==
           LLEV_STATUS_OK);
    assert(llev_query_cursor_next_batch(fresh, 8, &batch) == LLEV_STATUS_OK);
    assert(batch.len == 1);
    assert(batch.matches[0].byte_len == 3);
    assert(memcmp(batch.matches[0].term_data, "new", 3) == 0);
    assert(llev_query_cursor_release_batch(fresh, batch.generation) == LLEV_STATUS_OK);
    assert(llev_query_cursor_free(fresh) == LLEV_STATUS_OK);

    ldict_dictionary_free(dictionary);
    llev_transducer_free(transducer);

    assert(cursor_contains_text(frozen, "cat", 3, 2,
                                LLEV_QUERY_ORDER_TRAVERSAL, "cot", 1));
    assert(!cursor_contains_text(frozen, "cat", 3, 2,
                                 LLEV_QUERY_ORDER_TRAVERSAL, "new", 3));
    llev_transducer_free(frozen);

    for (;;) {
        LlevStatus status = llev_query_cursor_next_batch(cursor, 2, &batch);
        if (status == LLEV_STATUS_END) break;
        assert(status == LLEV_STATUS_OK);
        for (size_t i = 0; i < batch.len; ++i) remember(&seen, &batch.matches[i]);
        assert(llev_query_cursor_release_batch(cursor, batch.generation) ==
               LLEV_STATUS_OK);
    }
    assert(seen.count == 4);
    assert(seen.cat && seen.cot && seen.cut && seen.scat);
    assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);
    return 0;
}
