/*
 * C8 native property-based tests for the C facade: a seeded pseudo-PBT loop that
 * checks each property against an in-language brute-force Levenshtein oracle.
 *
 *   (a) distance symmetry d(a,b)==d(b,a), identity d(a,a)==0, threshold
 *       consistency dThreshold(a,b,k) == (d<=k ? d : SIZE_MAX-1) across the
 *       Levenshtein/Damerau/true-Damerau variants, plus standard oracle
 *       agreement;
 *   (b) a query's result set at distance k equals {t in dict : lev(query,t)<=k}
 *       over a random dictionary built with the libdictenstein DynamicDawg,
 *       with exact distances and value round-trips;
 *   (c) u64 value round-trips, with 0 and 2^64-1 pinned.
 *
 * The PRNG is seeded from a constant so failures reproduce. The C facade passes
 * real (non-null) buffers, so empty operands are exercised directly. The
 * empty-operand characterization below also confirms, at the C level, the fix
 * for LLEV-B18: the native distance entry points now accept a NULL data pointer
 * for a zero-length string (some host runtimes -- .NET, Go -- marshal an empty
 * string as NULL+0), returning the true distance rather than the SIZE_MAX
 * invalid-input sentinel, while still rejecting NULL with a non-zero length.
 */
#include "libdictenstein.h"
#include "liblevenshtein.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ALPHABET "abcd"
#define ALPHABET_SIZE 4u
#define MAX_LEN 6u
#define MAX_ENTRIES 8u
#define SENTINEL ((size_t)-1 - 1) /* threshold over-bound: SIZE_MAX - 1 */

static uint64_t rng_state;

static uint64_t next_random(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static size_t generate_string(char* buffer, size_t max_length) {
    size_t length = (size_t)(next_random() % (max_length + 1));
    for (size_t i = 0; i < length; ++i) {
        buffer[i] = ALPHABET[next_random() % ALPHABET_SIZE];
    }
    buffer[length] = '\0';
    return length;
}

static size_t oracle(const char* a, size_t la, const char* b, size_t lb) {
    if (la == 0) return lb;
    if (lb == 0) return la;
    size_t previous[MAX_LEN + 1];
    size_t current[MAX_LEN + 1];
    for (size_t j = 0; j <= lb; ++j) previous[j] = j;
    for (size_t i = 1; i <= la; ++i) {
        current[0] = i;
        for (size_t j = 1; j <= lb; ++j) {
            size_t cost = a[i - 1] == b[j - 1] ? 0 : 1;
            size_t best = previous[j] + 1;
            if (current[j - 1] + 1 < best) best = current[j - 1] + 1;
            if (previous[j - 1] + cost < best) best = previous[j - 1] + cost;
            current[j] = best;
        }
        for (size_t j = 0; j <= lb; ++j) previous[j] = current[j];
    }
    return previous[lb];
}

static void characterize_empty_operands(void) {
    char empty[1] = {0};
    /* A non-null zero-length buffer is a valid empty string. */
    assert(llev_distance(empty, 0, empty, 0) == 0);
    assert(llev_distance(empty, 0, "ab", 2) == 2);
    assert(llev_distance("ab", 2, empty, 0) == 2);
    /* LLEV-B18 fix: a NULL data pointer with length 0 denotes the empty string,
     * so a facade that marshals an empty string as NULL+0 (.NET's
     * `fixed (byte* p = new byte[0])`, Go's `&slice[0]` on an empty slice) now
     * gets the true distance, matching the transducer query path. */
    assert(llev_distance(NULL, 0, NULL, 0) == 0);
    assert(llev_distance(NULL, 0, "ab", 2) == 2);
    assert(llev_distance("ab", 2, NULL, 0) == 2);
    /* A NULL pointer with a non-zero length remains an invalid operand and still
     * returns the invalid-input sentinel (the fix does not weaken this guard). */
    assert(llev_distance(NULL, 2, "ab", 2) == (size_t)-1);
}

static void distance_properties(void) {
    for (int iteration = 0; iteration < 4000; ++iteration) {
        char a[MAX_LEN + 1];
        char b[MAX_LEN + 1];
        size_t la = generate_string(a, MAX_LEN);
        size_t lb = generate_string(b, MAX_LEN);
        size_t k = (size_t)(next_random() % 4);

        size_t standard = llev_distance(a, la, b, lb);
        assert(standard == llev_distance(b, lb, a, la));        /* symmetry */
        assert(llev_distance(a, la, a, la) == 0);               /* identity */
        assert(standard == oracle(a, la, b, lb));               /* oracle */
        size_t standard_bounded = llev_distance_threshold(a, la, b, lb, k);
        assert(standard_bounded == (standard <= k ? standard : SENTINEL));

        size_t damerau = llev_damerau_distance(a, la, b, lb);
        assert(damerau == llev_damerau_distance(b, lb, a, la));
        assert(llev_damerau_distance(a, la, a, la) == 0);
        size_t damerau_bounded = llev_damerau_distance_threshold(a, la, b, lb, k);
        assert(damerau_bounded == (damerau <= k ? damerau : SENTINEL));

        size_t true_damerau = llev_true_damerau_distance(a, la, b, lb);
        assert(true_damerau == llev_true_damerau_distance(b, lb, a, la));
        assert(llev_true_damerau_distance(a, la, a, la) == 0);
        size_t true_bounded = llev_true_damerau_distance_threshold(a, la, b, lb, k);
        assert(true_bounded == (true_damerau <= k ? true_damerau : SENTINEL));
    }
}

typedef struct Entry {
    char term[MAX_LEN + 1];
    size_t length;
    uint64_t value;
    uint8_t has_value;
} Entry;

/* Build a dictionary, run one query, and check the result set against the
 * oracle including exact distances and value round-trips. */
static void query_matches_oracle(void) {
    for (int iteration = 0; iteration < 400; ++iteration) {
        Entry entries[MAX_ENTRIES];
        size_t entry_count = 0;
        size_t requested = (size_t)(next_random() % (MAX_ENTRIES + 1));
        for (size_t i = 0; i < requested; ++i) {
            Entry candidate;
            candidate.length = generate_string(candidate.term, MAX_LEN);
            candidate.has_value = (uint8_t)(next_random() % 4 != 0);
            candidate.value = next_random();
            uint8_t duplicate = 0;
            for (size_t j = 0; j < entry_count; ++j) {
                if (entries[j].length == candidate.length &&
                    memcmp(entries[j].term, candidate.term, candidate.length) == 0) {
                    duplicate = 1;
                    break;
                }
            }
            if (!duplicate) entries[entry_count++] = candidate;
        }

        char query[MAX_LEN + 1];
        size_t query_length = generate_string(query, MAX_LEN);
        size_t k = (size_t)(next_random() % 4);

        LdictDictionary* dictionary = NULL;
        assert(ldict_dynamic_dawg_new(VT_UNIT_DOMAIN_UNICODE_SCALAR, &dictionary) == LDICT_STATUS_OK);
        for (size_t i = 0; i < entry_count; ++i) {
            LdictOptionalU64 optional = {entries[i].value, entries[i].has_value, {0}};
            uint8_t inserted = 0;
            assert(ldict_dictionary_insert_text(dictionary, (const uint8_t*)entries[i].term,
                                                entries[i].length, optional, &inserted) == LDICT_STATUS_OK);
        }

        VtResource resource = {0};
        assert(ldict_dictionary_resource(dictionary, &resource) == LDICT_STATUS_OK);
        LlevTransducer* transducer = NULL;
        assert(llev_transducer_new(&resource, LLEV_ALGORITHM_STANDARD, &transducer) == LLEV_STATUS_OK);
        LlevQueryCursor* cursor = NULL;
        assert(llev_transducer_query_utf8(transducer, query, query_length, k,
                                          LLEV_QUERY_ORDER_TRAVERSAL, &cursor) == LLEV_STATUS_OK);

        /* Track which oracle-expected entries the query returned. */
        uint8_t matched[MAX_ENTRIES] = {0};
        size_t returned = 0;
        LlevMatchBatchView batch = {0};
        for (;;) {
            LlevStatus status = llev_query_cursor_next_batch(cursor, 4, &batch);
            if (status == LLEV_STATUS_END) break;
            assert(status == LLEV_STATUS_OK);
            for (size_t m = 0; m < batch.len; ++m) {
                const LlevMatch* item = &batch.matches[m];
                ++returned;
                for (size_t i = 0; i < entry_count; ++i) {
                    if (entries[i].length == item->byte_len &&
                        memcmp(entries[i].term, item->term_data, item->byte_len) == 0) {
                        assert(item->distance == oracle(query, query_length, entries[i].term, entries[i].length));
                        assert((item->has_id != 0) == (entries[i].has_value != 0));
                        if (item->has_id) assert(item->id == entries[i].value);
                        matched[i] = 1;
                    }
                }
            }
            assert(llev_query_cursor_release_batch(cursor, batch.generation) == LLEV_STATUS_OK);
        }
        assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);

        size_t expected = 0;
        for (size_t i = 0; i < entry_count; ++i) {
            if (oracle(query, query_length, entries[i].term, entries[i].length) <= k) {
                ++expected;
                assert(matched[i]); /* every within-threshold entry was returned */
            }
        }
        assert(returned == expected); /* and nothing outside the threshold leaked */

        llev_transducer_free(transducer);
        ldict_dictionary_free(dictionary);
    }
}

static void u64_value_round_trip(void) {
    const uint64_t values[] = {0, 1, (uint64_t)1 << 63, ~(uint64_t)0};
    for (size_t v = 0; v < sizeof(values) / sizeof(values[0]); ++v) {
        LdictDictionary* dictionary = NULL;
        assert(ldict_dynamic_dawg_new(VT_UNIT_DOMAIN_UNICODE_SCALAR, &dictionary) == LDICT_STATUS_OK);
        LdictOptionalU64 optional = {values[v], 1, {0}};
        uint8_t inserted = 0;
        assert(ldict_dictionary_insert_text(dictionary, (const uint8_t*)"term", 4, optional, &inserted) == LDICT_STATUS_OK);
        VtResource resource = {0};
        assert(ldict_dictionary_resource(dictionary, &resource) == LDICT_STATUS_OK);
        LlevTransducer* transducer = NULL;
        assert(llev_transducer_new(&resource, LLEV_ALGORITHM_STANDARD, &transducer) == LLEV_STATUS_OK);
        LlevQueryCursor* cursor = NULL;
        assert(llev_transducer_query_utf8(transducer, "term", 4, 0,
                                          LLEV_QUERY_ORDER_TRAVERSAL, &cursor) == LLEV_STATUS_OK);
        LlevMatchBatchView batch = {0};
        assert(llev_query_cursor_next_batch(cursor, 4, &batch) == LLEV_STATUS_OK);
        assert(batch.len == 1);
        assert(batch.matches[0].has_id != 0 && batch.matches[0].id == values[v]);
        assert(llev_query_cursor_release_batch(cursor, batch.generation) == LLEV_STATUS_OK);
        assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);
        llev_transducer_free(transducer);
        ldict_dictionary_free(dictionary);
    }
}

int main(void) {
    rng_state = 20260809ULL;
    characterize_empty_operands();
    distance_properties();
    query_matches_oracle();
    u64_value_round_trip();
    printf("C property tests passed\n");
    return 0;
}
