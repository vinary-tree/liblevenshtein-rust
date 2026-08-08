#include "libdictenstein.h"
#include "liblevenshtein.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

    LlevQueryCursor* cursor = NULL;
    assert(llev_transducer_query_utf8(transducer, "cat", 3, 2,
                                      LLEV_QUERY_ORDER_TRAVERSAL, &cursor) ==
           LLEV_STATUS_OK);
    Seen seen = {0};
    LlevMatchBatchView batch = {0};
    assert(llev_query_cursor_next_batch(cursor, 1, &batch) == LLEV_STATUS_OK);
    assert(batch.len == 1);
    remember(&seen, &batch.matches[0]);
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
