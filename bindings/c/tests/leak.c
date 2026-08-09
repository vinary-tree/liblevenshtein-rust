/*
 * C9 leak-discipline suite for the C facade, run under AddressSanitizer +
 * LeakSanitizer. A >=10,000-cycle create/use/free loop over dictionaries,
 * transducers, query cursors, phonetic patterns, and rule sets must free every
 * handle each cycle; LSan reports any leaked allocation at process exit and
 * fails the run. This is the machine-level complement to the managed-heap
 * steady-state checks the GC'd facades use.
 */
#include "libdictenstein.h"
#include "liblevenshtein.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define CYCLES 10000

static void insert(LdictDictionary* dictionary, const char* term, size_t length, uint64_t value) {
    LdictOptionalU64 optional = {value, 1, {0}};
    uint8_t inserted = 0;
    assert(ldict_dictionary_insert_text(dictionary, (const uint8_t*)term, length, optional, &inserted) == LDICT_STATUS_OK);
}

static void transducer_cycle(void) {
    LdictDictionary* dictionary = NULL;
    assert(ldict_dynamic_dawg_new(VT_UNIT_DOMAIN_UNICODE_SCALAR, &dictionary) == LDICT_STATUS_OK);
    insert(dictionary, "cat", 3, 1);
    insert(dictionary, "cot", 3, 2);
    insert(dictionary, "cut", 3, 3);
    insert(dictionary, "scat", 4, 4);

    VtResource resource = {0};
    assert(ldict_dictionary_resource(dictionary, &resource) == LDICT_STATUS_OK);
    LlevTransducer* transducer = NULL;
    assert(llev_transducer_new(&resource, LLEV_ALGORITHM_STANDARD, &transducer) == LLEV_STATUS_OK);
    LlevQueryCursor* cursor = NULL;
    assert(llev_transducer_query_utf8(transducer, "cat", 3, 2, LLEV_QUERY_ORDER_TRAVERSAL, &cursor) == LLEV_STATUS_OK);

    LlevMatchBatchView batch = {0};
    for (;;) {
        LlevStatus status = llev_query_cursor_next_batch(cursor, 4, &batch);
        if (status == LLEV_STATUS_END) break;
        assert(status == LLEV_STATUS_OK);
        assert(llev_query_cursor_release_batch(cursor, batch.generation) == LLEV_STATUS_OK);
    }
    assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);
    llev_transducer_free(transducer);
    ldict_dictionary_free(dictionary);
}

static void phonetic_cycle(void) {
    LlevPhoneticPattern* pattern = NULL;
    assert(llev_phonetic_pattern_compile_regex("c[ao]t", 6, &pattern) == LLEV_STATUS_OK);
    uint8_t matched = 0;
    assert(llev_phonetic_pattern_matches(pattern, "cat", 3, &matched) == LLEV_STATUS_OK);
    llev_phonetic_pattern_free(pattern);

    LlevPhoneticRuleSet* rules = NULL;
    assert(llev_phonetic_rules_builtin(0 /* English orthography */, &rules) == LLEV_STATUS_OK);
    LlevOwnedString output = {0};
    assert(llev_phonetic_rules_apply(rules, "phone", 5, &output) == LLEV_STATUS_OK);
    llev_owned_string_free(&output);
    llev_phonetic_rules_free(rules);
}

int main(void) {
    for (int i = 0; i < CYCLES; ++i) {
        transducer_cycle();
        phonetic_cycle();
    }
    printf("C leak loop completed %d cycles\n", CYCLES);
    return 0;
}
