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
_Static_assert(LLEV_QUERY_ORDER_TRAVERSAL == 0, "LlevQueryOrder TRAVERSAL moved");
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

int main(void) {
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
