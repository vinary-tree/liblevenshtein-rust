/*
 * C harness for the cross-language benchmark program.
 *
 * Implements harnesses/common/PROTOCOL.md over the raw llev_* / ldict_* C
 * ABI — the same boundary every language facade wraps — so this target is
 * the binding-overhead FLOOR: batch cursor drain (capacity 256), no facade.
 *
 * Call sequence mirrors bindings/c/tests/cross_project_snapshot.c:
 *   ldict_dynamic_dawg_new + ldict_dictionary_insert_text_batch
 *     (or ldict_double_array_trie_new from the pre-sorted entries)
 *   -> ldict_dictionary_resource -> llev_transducer_new
 *   -> llev_transducer_query_utf8
 *   -> llev_query_cursor_next_batch / llev_query_cursor_release_batch
 *   -> llev_query_cursor_free
 */

#include "libdictenstein.h"
#include "liblevenshtein.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FNV_OFFSET UINT64_C(0xcbf29ce484222325)
#define FNV_PRIME UINT64_C(0x100000001b3)
#define BATCH_CAPACITY 256
#define WALL_CAP_SECONDS 300.0
#define SAMPLE_DEFINITION \
    "one full pass over the query set; every cursor fully drained and " \
    "(term, distance) materialized"

#ifndef BENCH_LIBLEV_VERSION
#error "build.sh must define BENCH_LIBLEV_VERSION"
#endif

static void die(const char* message) {
    fprintf(stderr, "bench-cross-c: %s\n", message);
    exit(2);
}

static void die2(const char* message, const char* detail) {
    fprintf(stderr, "bench-cross-c: %s: %s\n", message, detail);
    exit(2);
}

/* ------------------------------------------------------------------ */
/* Checksum primitives (PROTOCOL.md section 8) + self-test (section 2) */
/* ------------------------------------------------------------------ */

static uint64_t fnv_update(uint64_t h, uint8_t b) {
    return (h ^ (uint64_t)b) * FNV_PRIME;
}

static uint64_t fnv1a64(const uint8_t* data, size_t len) {
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < len; ++i) h = fnv_update(h, data[i]);
    return h;
}

static uint64_t entry_hash(const uint8_t* term, size_t byte_len, uint64_t distance) {
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < byte_len; ++i) h = fnv_update(h, term[i]);
    h = fnv_update(h, 0x00);
    for (int i = 0; i < 8; ++i) h = fnv_update(h, (uint8_t)((distance >> (8 * i)) & 0xff));
    return h;
}

static void self_test(void) {
    if (fnv1a64((const uint8_t*)"", 0) != UINT64_C(0xcbf29ce484222325)) die("self-test: fnv empty");
    if (fnv1a64((const uint8_t*)"a", 1) != UINT64_C(0xaf63dc4c8601ec8c)) die("self-test: fnv 'a'");
    if (entry_hash((const uint8_t*)"cat", 3, 1) != UINT64_C(0x9697fa3e50464bc4))
        die("self-test: entry(cat,1)");
    if (entry_hash((const uint8_t*)"cat", 3, 0) != UINT64_C(0xb592c1475b3595e5))
        die("self-test: entry(cat,0)");
    if (entry_hash((const uint8_t*)"cot", 3, 1) != UINT64_C(0xb8acc5d3816bcdea))
        die("self-test: entry(cot,1)");
    uint64_t combined = entry_hash((const uint8_t*)"cat", 3, 0)
        + entry_hash((const uint8_t*)"cot", 3, 1);
    if (combined != UINT64_C(0x6e3f871adca163cf)) die("self-test: combined checksum");
}

/* ------------------------------------------------------------------ */
/* Timing                                                              */
/* ------------------------------------------------------------------ */

static uint64_t mono_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) die("clock_gettime failed");
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

/* ------------------------------------------------------------------ */
/* Input loading (PROTOCOL.md section 3)                               */
/* ------------------------------------------------------------------ */

typedef struct Lines {
    char* buffer;        /* owned; lines NUL-terminated in place */
    const char** ptrs;   /* owned array of line starts */
    size_t* lens;        /* owned array of line lengths */
    size_t count;
} Lines;

static Lines read_lines(const char* path) {
    FILE* fh = fopen(path, "rb");
    if (!fh) die2("cannot open", path);
    if (fseek(fh, 0, SEEK_END) != 0) die2("cannot seek", path);
    long size = ftell(fh);
    if (size < 0) die2("cannot tell", path);
    rewind(fh);
    char* buffer = (char*)malloc((size_t)size + 1);
    if (!buffer) die("out of memory reading file");
    if (fread(buffer, 1, (size_t)size, fh) != (size_t)size) die2("short read", path);
    fclose(fh);
    buffer[size] = '\0';

    size_t line_estimate = 1;
    for (long i = 0; i < size; ++i)
        if (buffer[i] == '\n') ++line_estimate;
    const char** ptrs = (const char**)malloc(line_estimate * sizeof(*ptrs));
    size_t* lens = (size_t*)malloc(line_estimate * sizeof(*lens));
    if (!ptrs || !lens) die("out of memory allocating line index");

    size_t count = 0;
    char* cursor = buffer;
    while (cursor < buffer + size) {
        char* newline = memchr(cursor, '\n', (size_t)(buffer + size - cursor));
        size_t len = newline ? (size_t)(newline - cursor) : strlen(cursor);
        if (len > 0) {
            cursor[len] = '\0';
            ptrs[count] = cursor;
            lens[count] = len;
            ++count;
        }
        if (!newline) break;
        cursor = newline + 1;
    }
    if (count == 0) die2("no lines in", path);
    Lines result = {buffer, ptrs, lens, count};
    return result;
}

static void free_lines(Lines* lines) {
    free(lines->buffer);
    free((void*)lines->ptrs);
    free(lines->lens);
    memset(lines, 0, sizeof(*lines));
}

static void assert_strictly_sorted(const Lines* lines, const char* path) {
    for (size_t i = 0; i + 1 < lines->count; ++i) {
        if (strcmp(lines->ptrs[i], lines->ptrs[i + 1]) >= 0) {
            fprintf(stderr,
                    "bench-cross-c: %s is not strictly byte-sorted at line %zu: "
                    "\"%s\" >= \"%s\"\n",
                    path, i + 1, lines->ptrs[i], lines->ptrs[i + 1]);
            exit(2);
        }
    }
}

/* ------------------------------------------------------------------ */
/* CLI (PROTOCOL.md section 1)                                         */
/* ------------------------------------------------------------------ */

typedef struct Args {
    const char* mode;
    const char* algorithm;
    long max_distance;
    const char* dictionary;
    const char* queries;
    const char* backend;
    const char* out;
    long samples;
    double warmup_seconds;
    long gate_limit;
    long reps;
    const char* cells;
} Args;

static Args parse_args(int argc, char** argv) {
    Args args = {NULL, NULL, -1, NULL, NULL, NULL, NULL, 30, 3.0, 200, 10, NULL};
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) die2("flag requires a value", argv[i]);
        const char* flag = argv[i];
        const char* value = argv[i + 1];
        if (strcmp(flag, "--mode") == 0) args.mode = value;
        else if (strcmp(flag, "--algorithm") == 0) args.algorithm = value;
        else if (strcmp(flag, "--max-distance") == 0) args.max_distance = strtol(value, NULL, 10);
        else if (strcmp(flag, "--dictionary") == 0) args.dictionary = value;
        else if (strcmp(flag, "--queries") == 0) args.queries = value;
        else if (strcmp(flag, "--backend") == 0) args.backend = value;
        else if (strcmp(flag, "--out") == 0) args.out = value;
        else if (strcmp(flag, "--samples") == 0) args.samples = strtol(value, NULL, 10);
        else if (strcmp(flag, "--warmup-seconds") == 0) args.warmup_seconds = strtod(value, NULL);
        else if (strcmp(flag, "--gate-limit") == 0) args.gate_limit = strtol(value, NULL, 10);
        else if (strcmp(flag, "--reps") == 0) args.reps = strtol(value, NULL, 10);
        else if (strcmp(flag, "--cells") == 0) args.cells = value;
        else die2("unknown flag", flag);
    }
    if (!args.mode) die("--mode is required");
    if (!args.dictionary) die("--dictionary is required");
    if (!args.backend) die("--backend is required");
    return args;
}

static LlevAlgorithm parse_algorithm(const char* name) {
    if (strcmp(name, "standard") == 0) return LLEV_ALGORITHM_STANDARD;
    if (strcmp(name, "transposition") == 0) return LLEV_ALGORITHM_TRANSPOSITION;
    if (strcmp(name, "merge_and_split") == 0) return LLEV_ALGORITHM_MERGE_AND_SPLIT;
    if (strcmp(name, "damerau_levenshtein") == 0) return LLEV_ALGORITHM_DAMERAU_LEVENSHTEIN;
    die2("unknown algorithm", name);
    return LLEV_ALGORITHM_STANDARD; /* unreachable */
}

/* ------------------------------------------------------------------ */
/* Dictionary construction                                             */
/* ------------------------------------------------------------------ */

static LdictTextEntry* build_entries(const Lines* terms) {
    LdictTextEntry* entries = (LdictTextEntry*)malloc(terms->count * sizeof(*entries));
    if (!entries) die("out of memory allocating entries");
    for (size_t i = 0; i < terms->count; ++i) {
        entries[i].data = (const uint8_t*)terms->ptrs[i];
        entries[i].len = terms->lens[i];
        entries[i].value.value = 0;
        entries[i].value.has_value = 0;
        memset(entries[i].value.reserved, 0, sizeof(entries[i].value.reserved));
    }
    return entries;
}

static LdictDictionary* build_dictionary(const char* backend,
                                         const LdictTextEntry* entries,
                                         size_t count) {
    LdictDictionary* dictionary = NULL;
    if (strcmp(backend, "dynamic_dawg") == 0) {
        if (ldict_dynamic_dawg_new(VT_UNIT_DOMAIN_UNICODE_SCALAR, &dictionary) != LDICT_STATUS_OK)
            die2("ldict_dynamic_dawg_new failed", ldict_last_error_message());
        size_t inserted = 0;
        if (ldict_dictionary_insert_text_batch(dictionary, entries, count, &inserted) !=
            LDICT_STATUS_OK)
            die2("insert_text_batch failed", ldict_last_error_message());
        if (inserted != count) die("batch insert count mismatch");
    } else if (strcmp(backend, "double_array_trie") == 0) {
        if (ldict_double_array_trie_new(VT_UNIT_DOMAIN_UNICODE_SCALAR, entries, count,
                                        &dictionary) != LDICT_STATUS_OK)
            die2("ldict_double_array_trie_new failed", ldict_last_error_message());
    } else {
        die2("unknown backend", backend);
    }
    return dictionary;
}

/* ------------------------------------------------------------------ */
/* Passes (PROTOCOL.md section 5)                                      */
/* ------------------------------------------------------------------ */

typedef struct Triple {
    uint64_t matches;
    uint64_t bytes;
    uint64_t dist;
} Triple;

static bool triple_eq(Triple a, Triple b) {
    return a.matches == b.matches && a.bytes == b.bytes && a.dist == b.dist;
}

static void drain_query(LlevTransducer* transducer,
                        const char* query,
                        size_t query_len,
                        long max_distance,
                        Triple* triple,
                        uint64_t* checksum /* nullable */) {
    LlevQueryCursor* cursor = NULL;
    LlevStatus status = llev_transducer_query_utf8(
        transducer, query, query_len, (size_t)max_distance, LLEV_QUERY_ORDER_TRAVERSAL, &cursor);
    if (status != LLEV_STATUS_OK) die2("query_utf8 failed", llev_last_error_message());
    for (;;) {
        LlevMatchBatchView batch = {0};
        status = llev_query_cursor_next_batch(cursor, BATCH_CAPACITY, &batch);
        if (status == LLEV_STATUS_END) break;
        if (status != LLEV_STATUS_OK) die2("next_batch failed", llev_last_error_message());
        for (size_t i = 0; i < batch.len; ++i) {
            const LlevMatch* match = &batch.matches[i];
            triple->matches += 1;
            triple->bytes += (uint64_t)match->byte_len;
            triple->dist += (uint64_t)match->distance;
            if (checksum) {
                *checksum += entry_hash((const uint8_t*)match->term_data, match->byte_len,
                                        (uint64_t)match->distance);
            }
        }
        if (llev_query_cursor_release_batch(cursor, batch.generation) != LLEV_STATUS_OK)
            die2("release_batch failed", llev_last_error_message());
    }
    if (llev_query_cursor_free(cursor) != LLEV_STATUS_OK)
        die2("cursor_free failed", llev_last_error_message());
}

static Triple full_pass(LlevTransducer* transducer,
                        const Lines* queries,
                        long max_distance,
                        size_t limit,
                        uint64_t* checksum /* nullable */) {
    Triple triple = {0, 0, 0};
    size_t n = limit < queries->count ? limit : queries->count;
    for (size_t i = 0; i < n; ++i)
        drain_query(transducer, queries->ptrs[i], queries->lens[i], max_distance, &triple,
                    checksum);
    return triple;
}

/* ------------------------------------------------------------------ */
/* JSON output (PROTOCOL.md section 11)                                */
/* ------------------------------------------------------------------ */

static void json_escape_to(FILE* fh, const char* s) {
    for (const char* p = s; *p; ++p) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
            case '"': fputs("\\\"", fh); break;
            case '\\': fputs("\\\\", fh); break;
            case '\n': fputs("\\n", fh); break;
            case '\r': fputs("\\r", fh); break;
            case '\t': fputs("\\t", fh); break;
            default:
                if (c < 0x20) fprintf(fh, "\\u%04x", c);
                else fputc((int)c, fh);
        }
    }
}

static const char* queryset_from_path(const char* path) {
    const char* base = strrchr(path, '/');
    base = base ? base + 1 : path;
    static char stem[128];
    size_t len = strlen(base);
    const char* dot = strrchr(base, '.');
    if (dot) len = (size_t)(dot - base);
    if (len >= sizeof(stem)) len = sizeof(stem) - 1;
    memcpy(stem, base, len);
    stem[len] = '\0';
    return stem;
}

typedef struct CellOutput {
    const char* mode;
    const char* backend_structure; /* dynamic_dawg | double_array_trie */
    const char* algorithm;
    long max_distance;
    const char* dictionary_file;
    size_t term_count;
    uint64_t construct_ns;
    bool has_construct_ns;
    const char* queries_file;
    size_t query_count;
    int warmup_passes;
    long samples_requested;
    double warmup_seconds;
    const uint64_t* samples_ns;
    size_t sample_count;
    Triple triple;
    uint64_t checksum;
    const uint64_t* construct_times;
    long construct_reps;
    const char* status;
    const char* extra_note; /* nullable */
} CellOutput;

static void write_result(const char* out_path, const CellOutput* cell) {
    FILE* fh = fopen(out_path, "wb");
    if (!fh) die2("cannot write", out_path);

    time_t now = time(NULL);
    struct tm utc;
    gmtime_r(&now, &utc);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc);

    fputs("{\n", fh);
    fputs("  \"schema_version\": \"1.0.0\",\n", fh);
    fputs("  \"suite\": \"cross-language-v1\",\n", fh);
    fprintf(fh, "  \"timestamp_utc\": \"%s\",\n", timestamp);
    fputs("  \"target\": {\n", fh);
    fputs("    \"language\": \"c\",\n", fh);
    fputs("    \"implementation\": \"vinary-tree\",\n", fh);
    fputs("    \"backend\": \"c-abi\",\n", fh);
    fprintf(fh, "    \"runtime_version\": \"cc %s\",\n", __VERSION__);
    fprintf(fh, "    \"library_version\": \"%s (abi %u rev %u)\",\n", BENCH_LIBLEV_VERSION,
            llev_abi_version(), llev_api_revision());
    fputs("    \"artifact\": { \"kind\": \"cdylib\", \"id\": \"libliblevenshtein.so\" }\n", fh);
    fputs("  },\n", fh);
    fputs("  \"dictionary\": {\n", fh);
    fputs("    \"file\": \"", fh);
    json_escape_to(fh, cell->dictionary_file);
    fputs("\",\n", fh);
    fprintf(fh, "    \"term_count\": %zu,\n", cell->term_count);
    fprintf(fh, "    \"structure\": \"%s\",\n", cell->backend_structure);
    fputs("    \"unit_domain\": \"unicode_scalar\"", fh);
    if (cell->has_construct_ns)
        fprintf(fh, ",\n    \"construct_ns\": %" PRIu64 "\n", cell->construct_ns);
    else
        fputs("\n", fh);
    fputs("  },\n", fh);
    fputs("  \"workload\": {\n", fh);
    fprintf(fh, "    \"queryset\": \"%s\",\n", queryset_from_path(cell->queries_file));
    fputs("    \"file\": \"", fh);
    json_escape_to(fh, cell->queries_file);
    fputs("\",\n", fh);
    fprintf(fh, "    \"query_count\": %zu\n", cell->query_count);
    fputs("  },\n", fh);
    fprintf(fh, "  \"algorithm\": \"%s\",\n", cell->algorithm);
    fprintf(fh, "  \"max_distance\": %ld,\n", cell->max_distance);
    fprintf(fh, "  \"mode\": \"%s\",\n", cell->mode);
    fputs("  \"protocol\": {\n", fh);
    fputs("    \"timer\": \"monotonic\",\n", fh);
    fputs("    \"harness\": \"self-timed\",\n", fh);
    fprintf(fh, "    \"warmup_seconds_min\": %g,\n", cell->warmup_seconds);
    fprintf(fh, "    \"warmup_passes\": %d,\n", cell->warmup_passes);
    fprintf(fh, "    \"samples_requested\": %ld,\n", cell->samples_requested);
    fputs("    \"sample_definition\": \"", fh);
    json_escape_to(fh, SAMPLE_DEFINITION);
    fputs("\",\n", fh);
    fprintf(fh, "    \"batch_size\": %d,\n", BATCH_CAPACITY);
    fprintf(fh, "    \"wall_cap_seconds\": %g\n", WALL_CAP_SECONDS);
    fputs("  },\n", fh);
    if (cell->construct_times) {
        fputs("  \"construct\": {\n", fh);
        fprintf(fh, "    \"reps\": %ld,\n", cell->construct_reps);
        fputs("    \"times_ns\": [", fh);
        for (long i = 0; i < cell->construct_reps; ++i)
            fprintf(fh, "%s%" PRIu64, i ? ", " : "", cell->construct_times[i]);
        fputs("],\n", fh);
        fprintf(fh, "    \"term_count\": %zu\n", cell->term_count);
        fputs("  },\n", fh);
    } else {
        fputs("  \"measurements\": {\n", fh);
        fputs("    \"samples_ns\": [", fh);
        for (size_t i = 0; i < cell->sample_count; ++i)
            fprintf(fh, "%s%" PRIu64, i ? ", " : "", cell->samples_ns[i]);
        fputs("],\n", fh);
        fprintf(fh, "    \"sample_count\": %zu,\n", cell->sample_count);
        fprintf(fh, "    \"matches_per_pass\": %" PRIu64 ",\n", cell->triple.matches);
        fprintf(fh, "    \"term_bytes_per_pass\": %" PRIu64 ",\n", cell->triple.bytes);
        fprintf(fh, "    \"distance_sum_per_pass\": %" PRIu64 ",\n", cell->triple.dist);
        fprintf(fh, "    \"checksum_hex\": \"%016" PRIx64 "\"\n", cell->checksum);
        fputs("  },\n", fh);
    }
    fprintf(fh, "  \"status\": \"%s\",\n", cell->status);
    fputs("  \"notes\": [", fh);
    fputs("\"raw C ABI, batch capacity 256: the binding-overhead floor\"", fh);
    if (cell->extra_note) {
        fputs(", \"", fh);
        json_escape_to(fh, cell->extra_note);
        fputs("\"", fh);
    }
    fputs("]\n", fh);
    fputs("}\n", fh);
    fclose(fh);
}

/* ------------------------------------------------------------------ */
/* Modes (PROTOCOL.md section 6)                                       */
/* ------------------------------------------------------------------ */

static void run_query_cell(LlevTransducer* transducer,
                           const Lines* queries,
                           const Args* args,
                           const char* algorithm,
                           long max_distance,
                           const char* queries_file,
                           const char* out_path,
                           const char* structure,
                           const Lines* terms,
                           uint64_t construct_ns) {
    Triple ref = {0, 0, 0};
    uint64_t checksum = 0;
    ref = full_pass(transducer, queries, max_distance, queries->count, &checksum);

    uint64_t warm_start = mono_ns();
    uint64_t warmup_ns = (uint64_t)(args->warmup_seconds * 1e9);
    int warmup_passes = 0;
    uint64_t last_pass_ns = 0;
    while ((mono_ns() - warm_start) < warmup_ns || warmup_passes < 2) {
        uint64_t t0 = mono_ns();
        Triple triple = full_pass(transducer, queries, max_distance, queries->count, NULL);
        last_pass_ns = mono_ns() - t0;
        if (!triple_eq(triple, ref)) die("nondeterministic result during warmup");
        ++warmup_passes;
    }

    long sample_count = args->samples;
    const char* status = "ok";
    char note[256];
    const char* extra_note = NULL;
    double last_pass_seconds = (double)last_pass_ns / 1e9;
    if ((double)sample_count * last_pass_seconds > WALL_CAP_SECONDS) {
        long reduced = (long)(WALL_CAP_SECONDS / last_pass_seconds);
        if (reduced < 10) reduced = 10;
        snprintf(note, sizeof(note),
                 "samples reduced from %ld to %ld by the %gs wall cap (estimated pass %.3fs)",
                 sample_count, reduced, WALL_CAP_SECONDS, last_pass_seconds);
        extra_note = note;
        sample_count = reduced;
        status = "degraded";
    }

    uint64_t* samples_ns = (uint64_t*)malloc((size_t)sample_count * sizeof(*samples_ns));
    if (!samples_ns) die("out of memory allocating samples");
    for (long i = 0; i < sample_count; ++i) {
        uint64_t t0 = mono_ns();
        Triple triple = full_pass(transducer, queries, max_distance, queries->count, NULL);
        samples_ns[i] = mono_ns() - t0;
        if (!triple_eq(triple, ref)) die("nondeterministic result during measurement");
    }

    CellOutput cell = {
        .mode = "query",
        .backend_structure = structure,
        .algorithm = algorithm,
        .max_distance = max_distance,
        .dictionary_file = args->dictionary,
        .term_count = terms->count,
        .construct_ns = construct_ns,
        .has_construct_ns = true,
        .queries_file = queries_file,
        .query_count = queries->count,
        .warmup_passes = warmup_passes,
        .samples_requested = args->samples,
        .warmup_seconds = args->warmup_seconds,
        .samples_ns = samples_ns,
        .sample_count = (size_t)sample_count,
        .triple = ref,
        .checksum = checksum,
        .construct_times = NULL,
        .construct_reps = 0,
        .status = status,
        .extra_note = extra_note,
    };
    write_result(out_path, &cell);
    free(samples_ns);
}

int main(int argc, char** argv) {
    self_test();
    Args args = parse_args(argc, argv);

    Lines terms = read_lines(args.dictionary);
    assert_strictly_sorted(&terms, args.dictionary);
    LdictTextEntry* entries = build_entries(&terms);

    const char* structure = args.backend;

    if (strcmp(args.mode, "construct") == 0) {
        if (!args.out) die("--out is required for construct mode");
        /* One warmup build, then reps timed builds; frees outside timing. */
        LdictDictionary* warm = build_dictionary(args.backend, entries, terms.count);
        ldict_dictionary_free(warm);
        uint64_t* times = (uint64_t*)malloc((size_t)args.reps * sizeof(*times));
        if (!times) die("out of memory");
        for (long r = 0; r < args.reps; ++r) {
            uint64_t t0 = mono_ns();
            LdictDictionary* dict = build_dictionary(args.backend, entries, terms.count);
            times[r] = mono_ns() - t0;
            ldict_dictionary_free(dict);
        }
        CellOutput cell = {
            .mode = "construct",
            .backend_structure = structure,
            .algorithm = "standard",
            .max_distance = 1,
            .dictionary_file = args.dictionary,
            .term_count = terms.count,
            .has_construct_ns = false,
            .queries_file = args.queries ? args.queries : "workload/queries/hits.txt",
            .query_count = 1,
            .warmup_passes = 1,
            .samples_requested = args.reps,
            .warmup_seconds = args.warmup_seconds,
            .samples_ns = NULL,
            .sample_count = 0,
            .triple = {0, 0, 0},
            .checksum = 0,
            .construct_times = times,
            .construct_reps = args.reps,
            .status = "ok",
            .extra_note =
                "construct mode: timed region is the build from the pre-sorted in-memory list only",
        };
        write_result(args.out, &cell);
        free(times);
        free(entries);
        free_lines(&terms);
        return 0;
    }

    /* query | verify | memory-child (and --cells batch driver) */
    uint64_t build_t0 = mono_ns();
    LdictDictionary* dictionary = build_dictionary(args.backend, entries, terms.count);
    uint64_t construct_ns = mono_ns() - build_t0;

    VtResource resource;
    memset(&resource, 0, sizeof(resource));
    if (ldict_dictionary_resource(dictionary, &resource) != LDICT_STATUS_OK)
        die2("dictionary_resource failed", ldict_last_error_message());

    if (args.cells) {
        FILE* fh = fopen(args.cells, "rb");
        if (!fh) die2("cannot open cells file", args.cells);
        char line[4096];
        while (fgets(line, sizeof(line), fh)) {
            size_t len = strlen(line);
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
            if (len == 0 || line[0] == '#') continue;
            char* fields[4] = {NULL, NULL, NULL, NULL};
            char* cursor = line;
            for (int f = 0; f < 4; ++f) {
                fields[f] = cursor;
                if (f < 3) {
                    char* tab = strchr(cursor, '\t');
                    if (!tab) die2("cells row needs 4 tab-separated fields", line);
                    *tab = '\0';
                    cursor = tab + 1;
                }
            }
            LlevAlgorithm algorithm = parse_algorithm(fields[0]);
            long max_distance = strtol(fields[1], NULL, 10);
            Lines queries = read_lines(fields[2]);
            LlevTransducer* transducer = NULL;
            if (llev_transducer_new(&resource, algorithm, &transducer) != LLEV_STATUS_OK)
                die2("transducer_new failed", llev_last_error_message());
            if (strcmp(args.mode, "verify") == 0) {
                /* Batched verify: the gate uses this to amortize one DAT
                 * build across all 27 oracle-comparison cells. */
                size_t limit = queries.count;
                if ((size_t)args.gate_limit < limit) limit = (size_t)args.gate_limit;
                uint64_t checksum = 0;
                Triple triple =
                    full_pass(transducer, &queries, max_distance, limit, &checksum);
                CellOutput cell = {
                    .mode = "verify",
                    .backend_structure = structure,
                    .algorithm = fields[0],
                    .max_distance = max_distance,
                    .dictionary_file = args.dictionary,
                    .term_count = terms.count,
                    .construct_ns = construct_ns,
                    .has_construct_ns = true,
                    .queries_file = fields[2],
                    .query_count = limit,
                    .warmup_passes = 0,
                    .samples_requested = 0,
                    .warmup_seconds = args.warmup_seconds,
                    .samples_ns = NULL,
                    .sample_count = 0,
                    .triple = triple,
                    .checksum = checksum,
                    .construct_times = NULL,
                    .construct_reps = 0,
                    .status = "ok",
                    .extra_note = NULL,
                };
                write_result(fields[3], &cell);
            } else {
                run_query_cell(transducer, &queries, &args, fields[0], max_distance, fields[2],
                               fields[3], structure, &terms, construct_ns);
            }
            llev_transducer_free(transducer);
            free_lines(&queries);
        }
        fclose(fh);
    } else {
        if (!args.algorithm) die("--algorithm is required");
        if (args.max_distance < 0) die("--max-distance is required");
        if (!args.queries) die("--queries is required");
        if (!args.out) die("--out is required");
        LlevAlgorithm algorithm = parse_algorithm(args.algorithm);
        Lines queries = read_lines(args.queries);
        LlevTransducer* transducer = NULL;
        if (llev_transducer_new(&resource, algorithm, &transducer) != LLEV_STATUS_OK)
            die2("transducer_new failed", llev_last_error_message());

        if (strcmp(args.mode, "query") == 0) {
            run_query_cell(transducer, &queries, &args, args.algorithm, args.max_distance,
                           args.queries, args.out, structure, &terms, construct_ns);
        } else if (strcmp(args.mode, "verify") == 0 || strcmp(args.mode, "memory-child") == 0) {
            size_t limit = queries.count;
            if (strcmp(args.mode, "verify") == 0 && (size_t)args.gate_limit < limit)
                limit = (size_t)args.gate_limit;
            uint64_t checksum = 0;
            Triple triple = full_pass(transducer, &queries, args.max_distance, limit, &checksum);
            CellOutput cell = {
                .mode = args.mode[0] == 'v' ? "verify" : "memory",
                .backend_structure = structure,
                .algorithm = args.algorithm,
                .max_distance = args.max_distance,
                .dictionary_file = args.dictionary,
                .term_count = terms.count,
                .construct_ns = construct_ns,
                .has_construct_ns = true,
                .queries_file = args.queries,
                .query_count = limit,
                .warmup_passes = 0,
                .samples_requested = 0,
                .warmup_seconds = args.warmup_seconds,
                .samples_ns = NULL,
                .sample_count = 0,
                .triple = triple,
                .checksum = checksum,
                .construct_times = NULL,
                .construct_reps = 0,
                .status = "ok",
                .extra_note = NULL,
            };
            write_result(args.out, &cell);
        } else {
            die2("unknown mode", args.mode);
        }
        llev_transducer_free(transducer);
        free_lines(&queries);
    }

    ldict_dictionary_free(dictionary);
    free(entries);
    free_lines(&terms);
    return 0;
}
