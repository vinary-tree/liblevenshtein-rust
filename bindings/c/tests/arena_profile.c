/*
 * Arena-reuse and zero-copy profile for the resource ABI (wave W8).
 *
 * The C-ABI cursor packs each batch's match terms into a per-cursor arena and
 * hands the C side descriptors (`term_data`, `byte_len`) that point *into* that
 * arena -- no per-match heap allocation, and the arena buffer is cleared and
 * refilled (capacity retained) on every batch rather than reallocated. This
 * driver verifies the two observable consequences at the C boundary:
 *
 *   (Z) zero-copy / packing: within a batch every `term_data` points into one
 *       contiguous region whose span is bounded by the packed byte total, and the
 *       descriptors are laid out in ascending address order -- i.e. the terms are
 *       packed into a single buffer, not individually allocated;
 *   (R) arena reuse: after a batch is released and the next batch is pulled, the
 *       first descriptor lands at the very same base address -- the arena buffer
 *       was reused (cleared + refilled from offset 0), not reallocated.
 *
 * A warm-up loop then drains the same query many times so a heap profiler
 * (valgrind massif / dhat via scripts/profile-ffi-arena.sh) can confirm that the
 * steady-state allocation count per warm batch is ~0. This corroborates,
 * dynamically, the LLEV-ARENA invariants proven in
 * docs/verification/verus/ffi_batch_arena.rs.
 */
#include "libdictenstein.h"
#include "liblevenshtein.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define QUERY "aaaaa"
#define QUERY_LEN 5u
#define MAX_DISTANCE 2u
#define CAPACITY 4u
#define TERM_LEN 5u
#define EXPECTED_MATCHES 16u /* length-5 strings over {a,b} with <=2 'b's: C(5,0)+C(5,1)+C(5,2) */
#define WARMUP_QUERIES 2000

/* Build a dictionary of every length-5 string over {a,b} with at most two 'b's.
 * Each is within edit distance 2 of QUERY ("aaaaa") -- there are
 * C(5,0)+C(5,1)+C(5,2) = 16 of them -- so a distance-2 query returns exactly
 * those 16, and crucially they are all the SAME length. Equal-length terms make
 * every batch pack the same byte total, so once the arena reaches its capacity in
 * the first batch it is cleared-and-refilled (never reallocated) for the rest of
 * the query -- the reuse this profile checks. */
static LlevTransducer* build_transducer(LdictDictionary** out_dictionary) {
    LdictDictionary* dictionary = NULL;
    assert(ldict_dynamic_dawg_new(VT_UNIT_DOMAIN_UNICODE_SCALAR, &dictionary) == LDICT_STATUS_OK);
    char term[TERM_LEN + 1];
    term[TERM_LEN] = '\0';
    unsigned inserted_count = 0;
    for (unsigned mask = 0; mask < (1u << TERM_LEN); ++mask) {
        if (__builtin_popcount(mask) > 2) continue; /* <=2 substitutions -> distance <=2 */
        for (unsigned bit = 0; bit < TERM_LEN; ++bit) {
            term[bit] = (mask & (1u << bit)) ? 'b' : 'a';
        }
        LdictOptionalU64 optional = {(uint64_t)mask, 1, {0}};
        uint8_t inserted = 0;
        assert(ldict_dictionary_insert_text(dictionary, (const uint8_t*)term, TERM_LEN, optional,
                                            &inserted) == LDICT_STATUS_OK);
        ++inserted_count;
    }
    assert(inserted_count == EXPECTED_MATCHES);
    VtResource resource = {0};
    assert(ldict_dictionary_resource(dictionary, &resource) == LDICT_STATUS_OK);
    LlevTransducer* transducer = NULL;
    assert(llev_transducer_new(&resource, LLEV_ALGORITHM_STANDARD, &transducer) == LLEV_STATUS_OK);
    *out_dictionary = dictionary;
    return transducer;
}

/* One drained query, checking (Z) intra-batch packing and (R) cross-batch arena
 * reuse. Returns the total matches seen. */
static size_t profile_one_query(LlevTransducer* transducer) {
    LlevQueryCursor* cursor = NULL;
    assert(llev_transducer_query_utf8(transducer, QUERY, QUERY_LEN, MAX_DISTANCE,
                                      LLEV_QUERY_ORDER_TRAVERSAL, &cursor) == LLEV_STATUS_OK);

    const uint8_t* first_base = NULL; /* arena base observed in the first batch */
    size_t total = 0;
    LlevMatchBatchView batch = {0};
    for (;;) {
        LlevStatus status = llev_query_cursor_next_batch(cursor, CAPACITY, &batch);
        if (status == LLEV_STATUS_END) break;
        assert(status == LLEV_STATUS_OK);
        if (batch.len == 0) break;

        /* (Z) packing: descriptors ascend in address order and every term lies
         * within [base, base + packed_span), where packed_span is the sum of the
         * batch's byte lengths -- a single contiguous buffer, not scattered
         * allocations. `term_data` is `const void*`, so cast to a byte pointer
         * for address arithmetic. */
        const uint8_t* base = (const uint8_t*)batch.matches[0].term_data;
        const uint8_t* previous = NULL;
        size_t packed_span = 0;
        for (size_t m = 0; m < batch.len; ++m) {
            const LlevMatch* item = &batch.matches[m];
            const uint8_t* term = (const uint8_t*)item->term_data;
            packed_span += item->byte_len;
            if (item->byte_len > 0) {
                assert(term >= base); /* nothing precedes the base */
                if (previous != NULL) {
                    assert(term >= previous); /* ascending layout */
                }
                previous = term;
            }
        }
        /* The furthest term end stays within the packed span from the base: the
         * whole batch occupies one contiguous arena region. */
        for (size_t m = 0; m < batch.len; ++m) {
            const LlevMatch* item = &batch.matches[m];
            const uint8_t* term = (const uint8_t*)item->term_data;
            assert(term + item->byte_len <= base + packed_span);
        }

        /* (R) reuse: the first descriptor of every batch lands at the same base
         * address -- the arena was cleared and refilled, not reallocated. */
        if (first_base == NULL) {
            first_base = base;
        } else {
            assert(base == first_base);
        }

        total += batch.len;
        assert(llev_query_cursor_release_batch(cursor, batch.generation) == LLEV_STATUS_OK);
    }
    assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);
    return total;
}

int main(void) {
    LdictDictionary* dictionary = NULL;
    LlevTransducer* transducer = build_transducer(&dictionary);

    size_t matches = profile_one_query(transducer);
    assert(matches == EXPECTED_MATCHES); /* the 16 length-5 {a,b} terms within distance 2 of "aaaaa" */

    /* Warm-up: repeatedly drain the same query. After the first drain the arena
     * has reached its steady-state capacity, so a heap profiler should record ~0
     * allocations per subsequent batch. */
    for (int i = 0; i < WARMUP_QUERIES; ++i) {
        size_t again = profile_one_query(transducer);
        assert(again == matches);
    }

    llev_transducer_free(transducer);
    ldict_dictionary_free(dictionary);
    printf("C arena profile passed (%d warm queries, matches=%zu)\n", WARMUP_QUERIES, matches);
    return 0;
}
