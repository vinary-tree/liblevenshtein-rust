// C++ harness for the cross-language benchmark program.
//
// Implements harnesses/common/PROTOCOL.md through the header-only RAII
// facades (include/liblevenshtein.hpp + libdictenstein.hpp) over the same
// C ABI the C harness calls directly — the delta between the two targets IS
// the measured cost of the C++ wrapper layer (spans, shared_ptr cursor
// state, exception checks).

#include "libdictenstein.hpp"
#include "liblevenshtein.hpp"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace levc = vinary_tree::liblevenshtein;
namespace dict = vinary_tree::libdictenstein;

namespace {

constexpr std::uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr std::uint64_t FNV_PRIME = 0x100000001b3ULL;
constexpr std::size_t BATCH_CAPACITY = 256;
constexpr double WALL_CAP_SECONDS = 300.0;
constexpr const char* SAMPLE_DEFINITION =
    "one full pass over the query set; every cursor fully drained and "
    "(term, distance) materialized";

[[noreturn]] void die(const std::string& message) {
    std::fprintf(stderr, "bench-cross-cpp: %s\n", message.c_str());
    std::exit(2);
}

std::uint64_t fnv_update(std::uint64_t hash, std::uint8_t byte) {
    return (hash ^ static_cast<std::uint64_t>(byte)) * FNV_PRIME;
}

std::uint64_t entry_hash(std::string_view term, std::uint64_t distance) {
    std::uint64_t hash = FNV_OFFSET;
    for (unsigned char c : term) {
        hash = fnv_update(hash, c);
    }
    hash = fnv_update(hash, 0x00);
    for (int i = 0; i < 8; ++i) {
        hash = fnv_update(hash, static_cast<std::uint8_t>((distance >> (8 * i)) & 0xff));
    }
    return hash;
}

void self_test() {
    if (fnv_update(FNV_OFFSET, 'a') != 0xaf63dc4c8601ec8cULL) die("self-test: fnv 'a'");
    if (entry_hash("cat", 1) != 0x9697fa3e50464bc4ULL) die("self-test: entry(cat,1)");
    if (entry_hash("cat", 0) != 0xb592c1475b3595e5ULL) die("self-test: entry(cat,0)");
    if (entry_hash("cot", 1) != 0xb8acc5d3816bcdeaULL) die("self-test: entry(cot,1)");
    if (entry_hash("cat", 0) + entry_hash("cot", 1) != 0x6e3f871adca163cfULL)
        die("self-test: combined checksum");
}

std::uint64_t mono_ns() {
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) die("clock_gettime failed");
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL
        + static_cast<std::uint64_t>(ts.tv_nsec);
}

std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) die("cannot open " + path);
    std::stringstream buffer;
    buffer << stream.rdbuf();
    std::string data = buffer.str();
    std::size_t estimate = 1 + static_cast<std::size_t>(
        std::count(data.begin(), data.end(), '\n'));
    std::vector<std::string> lines;
    lines.reserve(estimate);
    std::size_t start = 0;
    for (std::size_t i = 0; i <= data.size(); ++i) {
        if (i == data.size() || data[i] == '\n') {
            if (i > start) lines.emplace_back(data.substr(start, i - start));
            start = i + 1;
        }
    }
    if (lines.empty()) die("no lines in " + path);
    return lines;
}

void assert_strictly_sorted(const std::vector<std::string>& lines, const std::string& path) {
    for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
        if (!(lines[i] < lines[i + 1])) {
            die(path + " is not strictly byte-sorted at line " + std::to_string(i + 1));
        }
    }
}

struct Args {
    std::string mode;
    std::string algorithm;
    long max_distance = -1;
    std::string dictionary;
    std::string queries;
    std::string backend;
    std::string out;
    long samples = 30;
    double warmup_seconds = 3.0;
    long gate_limit = 200;
    long reps = 10;
    std::string cells;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string_view flag = argv[i];
        std::string value = argv[i + 1];
        if (flag == "--mode") args.mode = value;
        else if (flag == "--algorithm") args.algorithm = value;
        else if (flag == "--max-distance") args.max_distance = std::stol(value);
        else if (flag == "--dictionary") args.dictionary = value;
        else if (flag == "--queries") args.queries = value;
        else if (flag == "--backend") args.backend = value;
        else if (flag == "--out") args.out = value;
        else if (flag == "--samples") args.samples = std::stol(value);
        else if (flag == "--warmup-seconds") args.warmup_seconds = std::stod(value);
        else if (flag == "--gate-limit") args.gate_limit = std::stol(value);
        else if (flag == "--reps") args.reps = std::stol(value);
        else if (flag == "--cells") args.cells = value;
        else die("unknown flag: " + std::string(flag));
    }
    if (args.mode.empty() || args.dictionary.empty() || args.backend.empty()) {
        die("--mode, --dictionary, --backend are required");
    }
    return args;
}

levc::algorithm parse_algorithm(const std::string& name) {
    if (name == "standard") return levc::algorithm::standard;
    if (name == "transposition") return levc::algorithm::transposition;
    if (name == "merge_and_split") return levc::algorithm::merge_and_split;
    if (name == "damerau_levenshtein") return levc::algorithm::damerau_levenshtein;
    die("unknown algorithm: " + name);
}

std::unique_ptr<dict::dictionary> build_dictionary(
    const std::string& backend,
    const std::vector<std::string>& terms,
    const std::vector<std::pair<std::string_view, std::optional<std::uint64_t>>>& entries) {
    if (backend == "dynamic_dawg") {
        auto dawg = std::make_unique<dict::dynamic_dawg>();
        // One ABI crossing via the facade's batch insert.
        const std::size_t inserted = dawg->insert_all(
            std::span<const std::pair<std::string_view, std::optional<std::uint64_t>>>(entries));
        if (inserted != terms.size()) {
            die("batch insert count mismatch: " + std::to_string(inserted));
        }
        return dawg;
    }
    if (backend == "double_array_trie") {
        return std::make_unique<dict::double_array_trie>(
            std::span<const std::pair<std::string_view, std::optional<std::uint64_t>>>(entries));
    }
    die("unknown backend: " + backend);
}

struct Triple {
    std::uint64_t matches = 0;
    std::uint64_t bytes = 0;
    std::uint64_t dist = 0;
    bool operator==(const Triple&) const = default;
};

Triple full_pass(const levc::transducer& transducer,
                 const std::vector<std::string>& queries,
                 long max_distance,
                 std::size_t limit,
                 std::uint64_t* checksum) {
    Triple triple;
    const std::size_t n = std::min(limit, queries.size());
    for (std::size_t i = 0; i < n; ++i) {
        auto cursor = transducer.query(queries[i],
                                       static_cast<std::size_t>(max_distance));
        for (;;) {
            auto batch = cursor.next_batch(BATCH_CAPACITY);
            auto matches = batch.matches();
            if (matches.empty()) break;
            for (const auto& match : matches) {
                const auto term = levc::batch::utf8(match);
                triple.matches += 1;
                triple.bytes += term.size();
                triple.dist += match.distance;
                if (checksum) *checksum += entry_hash(term, match.distance);
            }
        }
    }
    return triple;
}

std::string json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string queryset_from_path(const std::string& path) {
    auto slash = path.find_last_of('/');
    std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    auto dot = base.find_last_of('.');
    return dot == std::string::npos ? base : base.substr(0, dot);
}

struct CellOutput {
    std::string mode;
    std::string structure;
    std::string algorithm;
    long max_distance = 1;
    std::string dictionary_file;
    std::size_t term_count = 0;
    std::optional<std::uint64_t> construct_ns;
    std::string queries_file;
    std::size_t query_count = 0;
    int warmup_passes = 0;
    long samples_requested = 0;
    double warmup_seconds = 3.0;
    std::vector<std::uint64_t> samples_ns;
    Triple triple;
    std::uint64_t checksum = 0;
    std::vector<std::uint64_t> construct_times;
    std::string status = "ok";
    std::vector<std::string> notes;
};

void write_result(const std::string& out_path, const CellOutput& cell) {
    std::ofstream out(out_path, std::ios::binary);
    if (!out) die("cannot write " + out_path);

    std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc);

    out << "{\n";
    out << "  \"schema_version\": \"1.0.0\",\n";
    out << "  \"suite\": \"cross-language-v1\",\n";
    out << "  \"timestamp_utc\": \"" << timestamp << "\",\n";
    out << "  \"target\": {\n";
    out << "    \"language\": \"cpp\",\n";
    out << "    \"implementation\": \"vinary-tree\",\n";
    out << "    \"backend\": \"cpp-raii\",\n";
    out << "    \"runtime_version\": \"c++ " << __VERSION__ << "\",\n";
    out << "    \"library_version\": \"" << BENCH_LIBLEV_VERSION << " (abi "
        << llev_abi_version() << " rev " << llev_api_revision() << ")\",\n";
    out << "    \"artifact\": { \"kind\": \"cdylib\", \"id\": \"libliblevenshtein.so\" }\n";
    out << "  },\n";
    out << "  \"dictionary\": {\n";
    out << "    \"file\": \"" << json_escape(cell.dictionary_file) << "\",\n";
    out << "    \"term_count\": " << cell.term_count << ",\n";
    out << "    \"structure\": \"" << cell.structure << "\",\n";
    out << "    \"unit_domain\": \"unicode_scalar\"";
    if (cell.construct_ns) {
        out << ",\n    \"construct_ns\": " << *cell.construct_ns << "\n";
    } else {
        out << "\n";
    }
    out << "  },\n";
    out << "  \"workload\": {\n";
    out << "    \"queryset\": \"" << queryset_from_path(cell.queries_file) << "\",\n";
    out << "    \"file\": \"" << json_escape(cell.queries_file) << "\",\n";
    out << "    \"query_count\": " << cell.query_count << "\n";
    out << "  },\n";
    out << "  \"algorithm\": \"" << cell.algorithm << "\",\n";
    out << "  \"max_distance\": " << cell.max_distance << ",\n";
    out << "  \"mode\": \"" << cell.mode << "\",\n";
    out << "  \"protocol\": {\n";
    out << "    \"timer\": \"monotonic\",\n";
    out << "    \"harness\": \"self-timed\",\n";
    out << "    \"warmup_seconds_min\": " << cell.warmup_seconds << ",\n";
    out << "    \"warmup_passes\": " << cell.warmup_passes << ",\n";
    out << "    \"samples_requested\": " << cell.samples_requested << ",\n";
    out << "    \"sample_definition\": \"" << json_escape(SAMPLE_DEFINITION) << "\",\n";
    out << "    \"batch_size\": " << BATCH_CAPACITY << ",\n";
    out << "    \"wall_cap_seconds\": " << static_cast<long>(WALL_CAP_SECONDS) << "\n";
    out << "  },\n";
    if (!cell.construct_times.empty()) {
        out << "  \"construct\": {\n";
        out << "    \"reps\": " << cell.construct_times.size() << ",\n";
        out << "    \"times_ns\": [";
        for (std::size_t i = 0; i < cell.construct_times.size(); ++i) {
            if (i) out << ", ";
            out << cell.construct_times[i];
        }
        out << "],\n";
        out << "    \"term_count\": " << cell.term_count << "\n";
        out << "  },\n";
    } else {
        out << "  \"measurements\": {\n";
        out << "    \"samples_ns\": [";
        for (std::size_t i = 0; i < cell.samples_ns.size(); ++i) {
            if (i) out << ", ";
            out << cell.samples_ns[i];
        }
        out << "],\n";
        out << "    \"sample_count\": " << cell.samples_ns.size() << ",\n";
        out << "    \"matches_per_pass\": " << cell.triple.matches << ",\n";
        out << "    \"term_bytes_per_pass\": " << cell.triple.bytes << ",\n";
        out << "    \"distance_sum_per_pass\": " << cell.triple.dist << ",\n";
        char checksum_hex[24];
        std::snprintf(checksum_hex, sizeof(checksum_hex), "%016" PRIx64, cell.checksum);
        out << "    \"checksum_hex\": \"" << checksum_hex << "\"\n";
        out << "  },\n";
    }
    out << "  \"status\": \"" << cell.status << "\",\n";
    out << "  \"notes\": [";
    out << "\"C++ RAII facade over the same C ABI as the c target: the delta is the wrapper cost\"";
    for (const auto& note : cell.notes) {
        out << ", \"" << json_escape(note) << "\"";
    }
    out << "]\n";
    out << "}\n";
}

void run_query_cell(const levc::transducer& transducer,
                    const std::vector<std::string>& queries,
                    const Args& args,
                    const std::string& algorithm,
                    long max_distance,
                    const std::string& queries_file,
                    const std::string& out_path,
                    const std::string& structure,
                    std::size_t term_count,
                    std::uint64_t construct_ns) {
    std::uint64_t checksum = 0;
    Triple reference = full_pass(transducer, queries, max_distance, queries.size(), &checksum);

    const std::uint64_t warm_start = mono_ns();
    const auto warmup_ns = static_cast<std::uint64_t>(args.warmup_seconds * 1e9);
    int warmup_passes = 0;
    std::uint64_t last_pass_ns = 0;
    while ((mono_ns() - warm_start) < warmup_ns || warmup_passes < 2) {
        const std::uint64_t t0 = mono_ns();
        Triple triple = full_pass(transducer, queries, max_distance, queries.size(), nullptr);
        last_pass_ns = mono_ns() - t0;
        if (!(triple == reference)) die("nondeterministic result during warmup");
        ++warmup_passes;
    }

    long sample_count = args.samples;
    std::string status = "ok";
    std::vector<std::string> notes;
    const double last_pass_seconds = static_cast<double>(last_pass_ns) / 1e9;
    if (static_cast<double>(sample_count) * last_pass_seconds > WALL_CAP_SECONDS) {
        long reduced = static_cast<long>(WALL_CAP_SECONDS / last_pass_seconds);
        if (reduced < 10) reduced = 10;
        notes.push_back("samples reduced from " + std::to_string(sample_count) + " to "
                        + std::to_string(reduced) + " by the wall cap");
        sample_count = reduced;
        status = "degraded";
    }

    std::vector<std::uint64_t> samples_ns;
    samples_ns.reserve(static_cast<std::size_t>(sample_count));
    for (long i = 0; i < sample_count; ++i) {
        const std::uint64_t t0 = mono_ns();
        Triple triple = full_pass(transducer, queries, max_distance, queries.size(), nullptr);
        samples_ns.push_back(mono_ns() - t0);
        if (!(triple == reference)) die("nondeterministic result during measurement");
    }

    CellOutput cell;
    cell.mode = "query";
    cell.structure = structure;
    cell.algorithm = algorithm;
    cell.max_distance = max_distance;
    cell.dictionary_file = args.dictionary;
    cell.term_count = term_count;
    cell.construct_ns = construct_ns;
    cell.queries_file = queries_file;
    cell.query_count = queries.size();
    cell.warmup_passes = warmup_passes;
    cell.samples_requested = args.samples;
    cell.warmup_seconds = args.warmup_seconds;
    cell.samples_ns = std::move(samples_ns);
    cell.triple = reference;
    cell.checksum = checksum;
    cell.status = status;
    cell.notes = std::move(notes);
    write_result(out_path, cell);
}

}  // namespace

int main(int argc, char** argv) {
    self_test();
    Args args = parse_args(argc, argv);

    std::vector<std::string> terms = read_lines(args.dictionary);
    assert_strictly_sorted(terms, args.dictionary);
    std::vector<std::pair<std::string_view, std::optional<std::uint64_t>>> entries;
    entries.reserve(terms.size());
    for (const auto& term : terms) {
        entries.emplace_back(term, std::nullopt);
    }

    if (args.mode == "construct") {
        if (args.out.empty()) die("--out is required for construct mode");
        build_dictionary(args.backend, terms, entries);  // warmup build
        std::vector<std::uint64_t> times;
        times.reserve(static_cast<std::size_t>(args.reps));
        for (long r = 0; r < args.reps; ++r) {
            const std::uint64_t t0 = mono_ns();
            auto dictionary = build_dictionary(args.backend, terms, entries);
            times.push_back(mono_ns() - t0);
            dictionary.reset();
        }
        CellOutput cell;
        cell.mode = "construct";
        cell.structure = args.backend;
        cell.algorithm = "standard";
        cell.max_distance = 1;
        cell.dictionary_file = args.dictionary;
        cell.term_count = terms.size();
        cell.queries_file =
            args.queries.empty() ? "workload/queries/hits.txt" : args.queries;
        cell.query_count = 1;
        cell.warmup_passes = 1;
        cell.samples_requested = args.reps;
        cell.warmup_seconds = args.warmup_seconds;
        cell.construct_times = std::move(times);
        cell.notes.push_back(
            "construct mode: timed region is the build from the pre-sorted in-memory list only");
        write_result(args.out, cell);
        return 0;
    }

    const std::uint64_t build_t0 = mono_ns();
    auto dictionary = build_dictionary(args.backend, terms, entries);
    const std::uint64_t construct_ns = mono_ns() - build_t0;
    const VtResource resource = dictionary->resource();

    auto run_single = [&](const std::string& algorithm_name, long max_distance,
                          const std::string& queries_path, const std::string& out_path) {
        levc::transducer transducer(resource, parse_algorithm(algorithm_name));
        std::vector<std::string> queries = read_lines(queries_path);
        if (args.mode == "verify" || args.mode == "memory-child") {
            std::size_t limit = queries.size();
            if (args.mode == "verify" && static_cast<std::size_t>(args.gate_limit) < limit) {
                limit = static_cast<std::size_t>(args.gate_limit);
            }
            std::uint64_t checksum = 0;
            Triple triple = full_pass(transducer, queries, max_distance, limit, &checksum);
            CellOutput cell;
            cell.mode = args.mode == "verify" ? "verify" : "memory";
            cell.structure = args.backend;
            cell.algorithm = algorithm_name;
            cell.max_distance = max_distance;
            cell.dictionary_file = args.dictionary;
            cell.term_count = terms.size();
            cell.construct_ns = construct_ns;
            cell.queries_file = queries_path;
            cell.query_count = limit;
            cell.samples_requested = 0;
            cell.warmup_seconds = args.warmup_seconds;
            cell.triple = triple;
            cell.checksum = checksum;
            write_result(out_path, cell);
        } else if (args.mode == "query") {
            run_query_cell(transducer, queries, args, algorithm_name, max_distance,
                           queries_path, out_path, args.backend, terms.size(), construct_ns);
        } else {
            die("unknown mode: " + args.mode);
        }
    };

    if (!args.cells.empty()) {
        std::ifstream cells(args.cells);
        if (!cells) die("cannot open cells file " + args.cells);
        std::string line;
        while (std::getline(cells, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> fields;
            fields.reserve(4);
            std::size_t start = 0;
            for (int f = 0; f < 3; ++f) {
                auto tab = line.find('\t', start);
                if (tab == std::string::npos) die("cells row needs 4 fields: " + line);
                fields.push_back(line.substr(start, tab - start));
                start = tab + 1;
            }
            fields.push_back(line.substr(start));
            run_single(fields[0], std::stol(fields[1]), fields[2], fields[3]);
        }
    } else {
        if (args.algorithm.empty() || args.max_distance < 0 || args.queries.empty()
            || args.out.empty()) {
            die("--algorithm, --max-distance, --queries, --out are required");
        }
        run_single(args.algorithm, args.max_distance, args.queries, args.out);
    }
    return 0;
}
