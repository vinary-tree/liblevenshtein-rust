// C8 native property-based tests for the C++ facade: a seeded pseudo-PBT loop
// checked against an in-language Levenshtein oracle.
//
//   (a) distance symmetry/identity and threshold-sentinel consistency across
//       Levenshtein/Damerau/true-Damerau (C ABI, included by the .hpp), plus
//       standard oracle agreement;
//   (b) query result set == {t in dict : lev(query,t) <= k} over a random
//       libdictenstein DynamicDawg dictionary driven through the .hpp
//       transducer/query_cursor, with exact distances and value round-trips;
//   (c) u64 value round-trips, with 0 and 2^64-1 pinned.
//
// The PRNG is seeded from a constant. std::string::data() is non-null even when
// empty, so (like the C facade) empty operands are exercised correctly.

#include "libdictenstein.h"
#include "liblevenshtein.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr char alphabet[] = "abcd";
constexpr std::size_t alphabet_size = 4;
constexpr std::size_t max_length = 6;
constexpr std::size_t sentinel = static_cast<std::size_t>(-1) - 1; // threshold over-bound

std::uint64_t rng_state = 20260809ULL;

std::uint64_t next_random() {
    std::uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::string generate_string(std::size_t maximum) {
    std::size_t length = next_random() % (maximum + 1);
    std::string result;
    result.reserve(length);
    for (std::size_t i = 0; i < length; ++i) result.push_back(alphabet[next_random() % alphabet_size]);
    return result;
}

std::size_t oracle(const std::string& a, const std::string& b) {
    if (a == b) return 0;
    if (a.empty()) return b.size();
    if (b.empty()) return a.size();
    std::vector<std::size_t> previous(b.size() + 1);
    std::vector<std::size_t> current(b.size() + 1);
    for (std::size_t j = 0; j <= b.size(); ++j) previous[j] = j;
    for (std::size_t i = 1; i <= a.size(); ++i) {
        current[0] = i;
        for (std::size_t j = 1; j <= b.size(); ++j) {
            std::size_t cost = a[i - 1] == b[j - 1] ? 0 : 1;
            current[j] = std::min({previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost});
        }
        std::swap(previous, current);
    }
    return previous[b.size()];
}

std::size_t distance_native(const std::string& a, const std::string& b) {
    return llev_distance(a.data(), a.size(), b.data(), b.size());
}

void distance_properties() {
    for (int iteration = 0; iteration < 4000; ++iteration) {
        std::string a = generate_string(max_length);
        std::string b = generate_string(max_length);
        std::size_t k = next_random() % 4;

        std::size_t standard = distance_native(a, b);
        assert(standard == distance_native(b, a));       // symmetry
        assert(distance_native(a, a) == 0);              // identity
        assert(standard == oracle(a, b));                // oracle
        std::size_t bounded = llev_distance_threshold(a.data(), a.size(), b.data(), b.size(), k);
        assert(bounded == (standard <= k ? standard : sentinel));

        std::size_t damerau = llev_damerau_distance(a.data(), a.size(), b.data(), b.size());
        assert(damerau == llev_damerau_distance(b.data(), b.size(), a.data(), a.size()));
        assert(llev_damerau_distance(a.data(), a.size(), a.data(), a.size()) == 0);
        std::size_t damerau_bounded =
            llev_damerau_distance_threshold(a.data(), a.size(), b.data(), b.size(), k);
        assert(damerau_bounded == (damerau <= k ? damerau : sentinel));

        std::size_t true_damerau = llev_true_damerau_distance(a.data(), a.size(), b.data(), b.size());
        assert(true_damerau == llev_true_damerau_distance(b.data(), b.size(), a.data(), a.size()));
        assert(llev_true_damerau_distance(a.data(), a.size(), a.data(), a.size()) == 0);
        std::size_t true_bounded =
            llev_true_damerau_distance_threshold(a.data(), a.size(), b.data(), b.size(), k);
        assert(true_bounded == (true_damerau <= k ? true_damerau : sentinel));
    }
}

// Owning RAII wrapper for a libdictenstein DynamicDawg used only as a resource
// source for the .hpp transducer.
class dawg {
public:
    dawg() { assert(ldict_dynamic_dawg_new(VT_UNIT_DOMAIN_UNICODE_SCALAR, &handle_) == LDICT_STATUS_OK); }
    ~dawg() { ldict_dictionary_free(handle_); }
    dawg(const dawg&) = delete;
    dawg& operator=(const dawg&) = delete;

    void put(const std::string& term, std::optional<std::uint64_t> value) {
        LdictOptionalU64 optional{value ? *value : 0, static_cast<std::uint8_t>(value ? 1 : 0), {0}};
        std::uint8_t inserted = 0;
        assert(ldict_dictionary_insert_text(handle_, reinterpret_cast<const std::uint8_t*>(term.data()),
                                            term.size(), optional, &inserted) == LDICT_STATUS_OK);
    }
    VtResource resource() const {
        VtResource result{};
        assert(ldict_dictionary_resource(handle_, &result) == LDICT_STATUS_OK);
        return result;
    }

private:
    LdictDictionary* handle_ = nullptr;
};

struct observed {
    std::size_t distance;
    std::optional<std::uint64_t> id;
};

std::map<std::string, observed> run_query(
    const std::map<std::string, std::optional<std::uint64_t>>& entries,
    const std::string& query, std::size_t k) {
    dawg dictionary;
    for (const auto& [term, value] : entries) dictionary.put(term, value);
    VtResource resource = dictionary.resource();
    vinary_tree::liblevenshtein::transducer automaton(resource);
    auto cursor = automaton.query(query, k);
    std::map<std::string, observed> result;
    for (;;) {
        auto values = cursor.next_batch(4);
        if (values.matches().empty()) break;
        for (const auto& item : values.matches()) {
            std::string term(vinary_tree::liblevenshtein::batch::utf8(item));
            result[term] = {item.distance, item.has_id ? std::optional<std::uint64_t>(item.id) : std::nullopt};
        }
    }
    return result;
}

void query_matches_oracle() {
    for (int iteration = 0; iteration < 400; ++iteration) {
        std::map<std::string, std::optional<std::uint64_t>> entries;
        std::size_t requested = next_random() % 9;
        for (std::size_t i = 0; i < requested; ++i) {
            std::optional<std::uint64_t> value;
            if (next_random() % 4 != 0) value = next_random();
            entries[generate_string(max_length)] = value;
        }
        std::string query = generate_string(max_length);
        std::size_t k = next_random() % 4;

        auto got = run_query(entries, query, k);
        std::map<std::string, std::optional<std::uint64_t>> expected;
        for (const auto& [term, value] : entries) {
            if (oracle(query, term) <= k) expected[term] = value;
        }
        assert(got.size() == expected.size());
        for (const auto& [term, match] : got) {
            auto it = expected.find(term);
            assert(it != expected.end());
            assert(match.distance == oracle(query, term));
            assert(match.id == it->second);
        }
    }
}

void u64_value_round_trip() {
    for (std::uint64_t value : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 63, ~std::uint64_t{0}}) {
        auto got = run_query({{"term", value}}, "term", 0);
        assert(got.size() == 1);
        assert(got.at("term").id == std::optional<std::uint64_t>(value));
    }
}

} // namespace

int main() {
    distance_properties();
    query_matches_oracle();
    u64_value_round_trip();
    return 0;
}
