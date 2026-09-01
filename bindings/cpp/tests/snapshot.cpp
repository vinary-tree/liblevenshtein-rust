#include "liblevenshtein.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

std::atomic_size_t live_contexts{0};
std::atomic_size_t retain_calls{0};
std::atomic_size_t release_calls{0};

struct node {
    std::map<std::uint64_t, std::uint64_t> edges;
    bool final = false;
    std::optional<std::uint64_t> value;
};

struct revision {
    std::vector<node> nodes{1};
    std::size_t terms = 0;

    void insert(std::string_view term, std::optional<std::uint64_t> value) {
        std::size_t current = 0;
        for (unsigned char unit : term) {
            auto [entry, inserted] = nodes[current].edges.emplace(unit, nodes.size());
            if (inserted) nodes.emplace_back();
            current = entry->second;
        }
        if (!nodes[current].final) ++terms;
        nodes[current].final = true;
        nodes[current].value = value;
    }

    void insert(std::span<const std::uint64_t> term,
                std::optional<std::uint64_t> value) {
        std::size_t current = 0;
        for (std::uint64_t unit : term) {
            auto [entry, inserted] = nodes[current].edges.emplace(unit, nodes.size());
            if (inserted) nodes.emplace_back();
            current = entry->second;
        }
        if (!nodes[current].final) ++terms;
        nodes[current].final = true;
        nodes[current].value = value;
    }
};

struct store {
    std::mutex lock;
    std::shared_ptr<const revision> current;
};

struct context {
    context(std::shared_ptr<store> source_value = {},
            std::shared_ptr<const revision> snapshot_value = {},
            const VtDictionaryVTable* mutable_vtable_value = nullptr,
            const VtDictionaryVTable* snapshot_vtable_value = nullptr)
        : source(std::move(source_value)), snapshot(std::move(snapshot_value)),
          mutable_vtable(mutable_vtable_value),
          snapshot_vtable(snapshot_vtable_value) {
        live_contexts.fetch_add(1, std::memory_order_relaxed);
    }
    ~context() { live_contexts.fetch_sub(1, std::memory_order_relaxed); }

    std::atomic_size_t references{1};
    std::shared_ptr<store> source;
    std::shared_ptr<const revision> snapshot;
    const VtDictionaryVTable* mutable_vtable;
    const VtDictionaryVTable* snapshot_vtable;
};

extern const VtResourceVTable resource_vtable;
extern const VtDictionaryVTable unicode_mutable_vtable;
extern const VtDictionaryVTable unicode_snapshot_vtable;
extern const VtDictionaryVTable byte_mutable_vtable;
extern const VtDictionaryVTable byte_snapshot_vtable;
extern const VtDictionaryVTable u64_mutable_vtable;
extern const VtDictionaryVTable u64_snapshot_vtable;

void retain(void* raw) {
    retain_calls.fetch_add(1, std::memory_order_relaxed);
    static_cast<context*>(raw)->references.fetch_add(1, std::memory_order_relaxed);
}

void release(void* raw) {
    release_calls.fetch_add(1, std::memory_order_relaxed);
    auto* value = static_cast<context*>(raw);
    if (value->references.fetch_sub(1, std::memory_order_acq_rel) == 1) delete value;
}

std::shared_ptr<const revision> captured(context* value) {
    if (value->snapshot) return value->snapshot;
    std::scoped_lock guard(value->source->lock);
    return value->source->current;
}

VtStatus query_interface(void* raw, const VtInterfaceId* id,
                         std::uint32_t minimum, const void** output) {
    if (!id || !output) return VT_STATUS_NULL_POINTER;
    if (minimum > VT_DICTIONARY_INTERFACE_VERSION ||
        std::memcmp(id->bytes, VT_DICTIONARY_INTERFACE_ID.bytes, sizeof(id->bytes)) != 0) {
        return VT_STATUS_UNSUPPORTED;
    }
    auto* value = static_cast<context*>(raw);
    *output = value->snapshot ? static_cast<const void*>(value->snapshot_vtable)
                              : static_cast<const void*>(value->mutable_vtable);
    return VT_STATUS_OK;
}

VtStatus snapshot(void* raw, VtResource* output) {
    if (!output) return VT_STATUS_NULL_POINTER;
    auto* source = static_cast<context*>(raw);
    auto* result = new context(source->source, captured(source),
                               source->mutable_vtable, source->snapshot_vtable);
    output->context = result;
    output->vtable = &resource_vtable;
    return VT_STATUS_OK;
}

VtStatus root(void*, std::uint64_t* output) {
    if (!output) return VT_STATUS_NULL_POINTER;
    *output = 0;
    return VT_STATUS_OK;
}

VtStatus length(void* raw, std::size_t* output, std::uint8_t* known) {
    if (!output || !known) return VT_STATUS_NULL_POINTER;
    *output = captured(static_cast<context*>(raw))->terms;
    *known = 1;
    return VT_STATUS_OK;
}

VtStatus is_final(void* raw, std::uint64_t id, std::uint8_t* output) {
    if (!output) return VT_STATUS_NULL_POINTER;
    auto value = captured(static_cast<context*>(raw));
    if (id >= value->nodes.size()) return VT_STATUS_INVALID_ARGUMENT;
    *output = value->nodes[id].final;
    return VT_STATUS_OK;
}

VtStatus node_value(void* raw, std::uint64_t id, VtOptionalU64* output) {
    if (!output) return VT_STATUS_NULL_POINTER;
    auto value = captured(static_cast<context*>(raw));
    if (id >= value->nodes.size() || !value->nodes[id].final) return VT_STATUS_INVALID_ARGUMENT;
    *output = {};
    if (value->nodes[id].value) {
        output->has_value = 1;
        output->value = *value->nodes[id].value;
    }
    return VT_STATUS_OK;
}

VtStatus transition(void* raw, std::uint64_t id, std::uint64_t label,
                    std::uint64_t* child, std::uint8_t* found) {
    if (!child || !found) return VT_STATUS_NULL_POINTER;
    auto value = captured(static_cast<context*>(raw));
    if (id >= value->nodes.size()) return VT_STATUS_INVALID_ARGUMENT;
    auto entry = value->nodes[id].edges.find(label);
    *found = entry != value->nodes[id].edges.end();
    *child = *found ? entry->second : 0;
    return VT_STATUS_OK;
}

VtStatus edges(void* raw, std::uint64_t id, std::size_t start,
               VtDictionaryEdge* output, std::size_t capacity,
               std::size_t* written, std::size_t* total) {
    if ((!output && capacity) || !written || !total) return VT_STATUS_NULL_POINTER;
    auto value = captured(static_cast<context*>(raw));
    if (id >= value->nodes.size()) return VT_STATUS_INVALID_ARGUMENT;
    const auto& source = value->nodes[id].edges;
    *total = source.size();
    *written = 0;
    auto entry = source.begin();
    std::advance(entry, std::min(start, source.size()));
    while (entry != source.end() && *written < capacity) {
        output[*written] = {entry->first, entry->second};
        ++*written;
        ++entry;
    }
    return VT_STATUS_OK;
}

const VtResourceVTable resource_vtable = {
    sizeof(VtResourceVTable), VT_ABI_VERSION, 0, retain, release, query_interface};

constexpr VtDictionaryVTable dictionary_vtable(VtUnitDomain domain,
                                                std::uint64_t flags) {
    return {sizeof(VtDictionaryVTable), VT_DICTIONARY_INTERFACE_VERSION,
            domain, VT_VALUE_DOMAIN_OPTIONAL_U64, flags,
            snapshot, root, length, is_final, node_value, transition, edges};
}

const VtDictionaryVTable unicode_mutable_vtable =
    dictionary_vtable(VT_UNIT_DOMAIN_UNICODE_SCALAR, 0);
const VtDictionaryVTable unicode_snapshot_vtable =
    dictionary_vtable(VT_UNIT_DOMAIN_UNICODE_SCALAR, VT_DICTIONARY_FLAG_IMMUTABLE);
const VtDictionaryVTable byte_mutable_vtable =
    dictionary_vtable(VT_UNIT_DOMAIN_BYTE, 0);
const VtDictionaryVTable byte_snapshot_vtable =
    dictionary_vtable(VT_UNIT_DOMAIN_BYTE, VT_DICTIONARY_FLAG_IMMUTABLE);
const VtDictionaryVTable u64_mutable_vtable =
    dictionary_vtable(VT_UNIT_DOMAIN_U64, 0);
const VtDictionaryVTable u64_snapshot_vtable =
    dictionary_vtable(VT_UNIT_DOMAIN_U64, VT_DICTIONARY_FLAG_IMMUTABLE);

std::pair<const VtDictionaryVTable*, const VtDictionaryVTable*>
vtables(VtUnitDomain domain) {
    switch (domain) {
    case VT_UNIT_DOMAIN_BYTE:
        return {&byte_mutable_vtable, &byte_snapshot_vtable};
    case VT_UNIT_DOMAIN_UNICODE_SCALAR:
        return {&unicode_mutable_vtable, &unicode_snapshot_vtable};
    case VT_UNIT_DOMAIN_U64:
        return {&u64_mutable_vtable, &u64_snapshot_vtable};
    default:
        throw std::invalid_argument("unsupported dictionary unit domain");
    }
}

class dictionary {
public:
    explicit dictionary(std::map<std::string, std::optional<std::uint64_t>> entries)
        : dictionary(VT_UNIT_DOMAIN_UNICODE_SCALAR, std::move(entries)) {}

    dictionary(VtUnitDomain domain,
               std::map<std::string, std::optional<std::uint64_t>> entries)
        : source_(std::make_shared<store>()) {
        const auto [mutable_vtable, snapshot_vtable] = vtables(domain);
        mutable_vtable_ = mutable_vtable;
        snapshot_vtable_ = snapshot_vtable;
        publish(std::move(entries));
        resource_.context =
            new context(source_, {}, mutable_vtable_, snapshot_vtable_);
        resource_.vtable = &resource_vtable;
    }

    explicit dictionary(
        std::vector<std::pair<std::vector<std::uint64_t>,
                             std::optional<std::uint64_t>>> entries)
        : source_(std::make_shared<store>()),
          mutable_vtable_(&u64_mutable_vtable),
          snapshot_vtable_(&u64_snapshot_vtable) {
        auto next = std::make_shared<revision>();
        for (const auto& [term, value] : entries) next->insert(term, value);
        source_->current = std::move(next);
        resource_.context =
            new context(source_, {}, mutable_vtable_, snapshot_vtable_);
        resource_.vtable = &resource_vtable;
    }
    ~dictionary() { release(resource_.context); }
    dictionary(const dictionary&) = delete;

    const VtResource& resource() const { return resource_; }
    void publish(std::map<std::string, std::optional<std::uint64_t>> entries) {
        auto next = std::make_shared<revision>();
        for (const auto& [term, value] : entries) next->insert(term, value);
        std::scoped_lock guard(source_->lock);
        source_->current = std::move(next);
    }

private:
    std::shared_ptr<store> source_;
    const VtDictionaryVTable* mutable_vtable_ = &unicode_mutable_vtable;
    const VtDictionaryVTable* snapshot_vtable_ = &unicode_snapshot_vtable;
    VtResource resource_{};
};

std::vector<std::string> drain(vinary_tree::liblevenshtein::query_cursor& cursor) {
    std::vector<std::string> result;
    for (;;) {
        auto values = cursor.next_batch(2);
        if (values.matches().empty()) break;
        for (const auto& item : values.matches()) {
            result.emplace_back(vinary_tree::liblevenshtein::batch::utf8(item));
        }
    }
    std::ranges::sort(result);
    return result;
}

bool contains(vinary_tree::liblevenshtein::transducer& transducer,
              std::string_view query, std::size_t maximum_distance,
              std::string_view expected, std::size_t distance) {
    auto cursor = transducer.query(query, maximum_distance);
    for (;;) {
        auto values = cursor.next_batch(2);
        if (values.matches().empty()) return false;
        for (const auto& item : values.matches()) {
            if (vinary_tree::liblevenshtein::batch::utf8(item) == expected &&
                item.distance == distance) {
                return true;
            }
        }
    }
}

struct reduction {
    std::size_t callbacks = 0;
    std::size_t matches = 0;
};

LlevStatus count_reduced_batch(void* raw, const LlevMatch* matches,
                               std::size_t len) {
    auto* value = static_cast<reduction*>(raw);
    assert(value != nullptr);
    assert(matches != nullptr || len == 0);
    ++value->callbacks;
    value->matches += len;
    return LLEV_STATUS_OK;
}

void assert_raw_c_passthrough(const VtResource& resource) {
    static_assert(LLEV_PHONETIC_RULE_SET_ENGLISH_ORTHOGRAPHY == 0);
    static_assert(LLEV_PHONETIC_RULE_SET_ENGLISH_PHONETIC == 1);

    assert(llev_abi_version() == LLEV_ABI_VERSION);
    assert(llev_api_revision() == LLEV_API_REVISION);
    assert((llev_build_features() & LLEV_BUILD_FEATURE_CORE) != 0);
    assert((llev_build_features() & LLEV_BUILD_FEATURE_PHONETIC) != 0);

    char* duplicate = llev_string_dup("C ABI visible through C++");
    assert(duplicate != nullptr);
    assert(std::strcmp(duplicate, "C ABI visible through C++") == 0);
    llev_string_free(duplicate);
    llev_string_free(nullptr);
    llev_string_array_free(nullptr, 0);

    const std::size_t exceeded = SIZE_MAX - 1;
    assert(llev_distance("kitten", 6, "sitting", 7) == 3);
    assert(llev_distance_threshold("kitten", 6, "sitting", 7, 2) == exceeded);
    assert(llev_damerau_distance("ab", 2, "ba", 2) == 1);
    assert(llev_damerau_distance_threshold("ab", 2, "ba", 2, 1) == 1);
    assert(llev_true_damerau_distance("ca", 2, "abc", 3) == 2);
    assert(llev_true_damerau_distance_threshold("ca", 2, "abc", 3, 1) ==
           exceeded);

    LlevTransducer* transducer = nullptr;
    assert(llev_transducer_new(&resource, LLEV_ALGORITHM_STANDARD, &transducer) ==
           LLEV_STATUS_OK);
    VtUnitDomain domain = VT_UNIT_DOMAIN_BYTE;
    assert(llev_transducer_unit_domain(transducer, &domain) == LLEV_STATUS_OK);
    assert(domain == VT_UNIT_DOMAIN_UNICODE_SCALAR);

    LlevQueryCursor* cursor = nullptr;
    assert(llev_transducer_query_utf8(transducer, "cat", 3, 2,
                                      LLEV_QUERY_ORDER_TRAVERSAL, &cursor) ==
           LLEV_STATUS_OK);
    LlevMatchBatchView batch{};
    assert(llev_query_cursor_next_batch(cursor, 1, &batch) == LLEV_STATUS_OK);
    assert(batch.len == 1);
    assert(llev_query_cursor_release_batch(cursor, batch.generation) ==
           LLEV_STATUS_OK);
    assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);

    assert(llev_transducer_query_utf8(transducer, "cat", 3, 2,
                                      LLEV_QUERY_ORDER_TRAVERSAL, &cursor) ==
           LLEV_STATUS_OK);
    reduction reduced;
    std::size_t reduced_count = 0;
    assert(llev_query_cursor_reduce(cursor, 2, count_reduced_batch, &reduced,
                                    &reduced_count) == LLEV_STATUS_OK);
    assert(reduced.callbacks == 2 && reduced.matches == 4);
    assert(reduced_count == reduced.matches);
    assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);

    LlevPhoneticPattern* regex = nullptr;
    assert(llev_phonetic_pattern_compile_regex("c[ao]t", 6, &regex) ==
           LLEV_STATUS_OK);
    std::size_t states = 0;
    std::size_t transitions = 0;
    assert(llev_phonetic_pattern_size(regex, &states, &transitions) ==
           LLEV_STATUS_OK);
    assert(states > 0 && transitions > 0);
    std::uint8_t matches = 0;
    assert(llev_phonetic_pattern_matches(regex, "cat", 3, &matches) ==
           LLEV_STATUS_OK);
    assert(matches == 1);
    assert(llev_transducer_query_pattern(transducer, regex, 0, &cursor) ==
           LLEV_STATUS_OK);
    reduced = {};
    reduced_count = 0;
    assert(llev_query_cursor_reduce(cursor, 1, count_reduced_batch, &reduced,
                                    &reduced_count) == LLEV_STATUS_OK);
    assert(reduced_count == 2);
    assert(llev_query_cursor_free(cursor) == LLEV_STATUS_OK);
    llev_phonetic_pattern_free(regex);

    static constexpr std::string_view llre = "@name \"Greeting\"\n^hello$";
    LlevPhoneticPattern* language = nullptr;
    assert(llev_phonetic_pattern_compile_llre(llre.data(), llre.size(), &language) ==
           LLEV_STATUS_OK);
    assert(llev_phonetic_pattern_matches(language, "hello", 5, &matches) ==
           LLEV_STATUS_OK);
    assert(matches == 1);
    llev_phonetic_pattern_free(language);

    LlevPhoneticPattern* invalid = nullptr;
    const auto invalid_status =
        llev_phonetic_pattern_compile_regex("(", 1, &invalid);
    assert(invalid_status == LLEV_STATUS_INVALID_ARGUMENT);
    assert(invalid == nullptr);
    assert(llev_last_error_message() != nullptr);
    assert(llev_last_error_message()[0] != '\0');
    try {
        vinary_tree::liblevenshtein::check(invalid_status);
        assert(false && "invalid regex did not throw the C++ typed error");
    } catch (const vinary_tree::liblevenshtein::error& failure) {
        assert(failure.status() == LLEV_STATUS_INVALID_ARGUMENT);
        assert(std::string_view(failure.what()).size() > 0);
    }

    static constexpr std::string_view rules_source = "ph -> f\ngh ->\n";
    LlevPhoneticRuleSet* rules = nullptr;
    assert(llev_phonetic_rules_parse(rules_source.data(), rules_source.size(),
                                     &rules) == LLEV_STATUS_OK);
    std::size_t rule_count = 0;
    assert(llev_phonetic_rules_len(rules, &rule_count) == LLEV_STATUS_OK);
    assert(rule_count == 2);
    LlevOwnedString output{};
    assert(llev_phonetic_rules_apply(rules, "phgh", 4, &output) ==
           LLEV_STATUS_OK);
    assert(output.len == 1 && std::memcmp(output.data, "f", 1) == 0);
    llev_owned_string_free(&output);
    assert(output.data == nullptr && output.len == 0);
    llev_phonetic_rules_free(rules);

    constexpr std::array<LlevPhoneticRuleSetKind, 2> kinds{
        LLEV_PHONETIC_RULE_SET_ENGLISH_ORTHOGRAPHY,
        LLEV_PHONETIC_RULE_SET_ENGLISH_PHONETIC,
    };
    for (const auto kind : kinds) {
        LlevPhoneticRuleSet* builtin = nullptr;
        assert(llev_phonetic_rules_builtin(static_cast<std::uint32_t>(kind),
                                           &builtin) == LLEV_STATUS_OK);
        assert(llev_phonetic_rules_len(builtin, &rule_count) == LLEV_STATUS_OK);
        assert(rule_count > 0);
        assert(llev_phonetic_rules_apply(builtin, "phone", 5, &output) ==
               LLEV_STATUS_OK);
        assert(output.data != nullptr && output.len > 0);
        llev_owned_string_free(&output);
        llev_phonetic_rules_free(builtin);
    }
    llev_transducer_free(transducer);
}

void assert_cpp_algorithms_order_and_lifecycle() {
    namespace ll = vinary_tree::liblevenshtein;
    dictionary entries({{"ab", 1}, {"c", 2}, {"abc", 3}, {"bat", 4},
                        {"cat", 5}, {"cats", 6}});

    ll::transducer standard(entries.resource(), ll::algorithm::standard);
    ll::transducer transposition(entries.resource(), ll::algorithm::transposition);
    ll::transducer merge(entries.resource(), ll::algorithm::merge_and_split);
    ll::transducer damerau(entries.resource(), ll::algorithm::damerau_levenshtein);
    assert(!contains(standard, "ba", 1, "ab", 1));
    assert(contains(transposition, "ba", 1, "ab", 1));
    assert(contains(merge, "ab", 1, "c", 1));
    assert(contains(damerau, "ca", 2, "abc", 2));

    auto ordered = standard.query("cat", 1, ll::query_order::distance_then_term);
    std::vector<std::string> terms;
    for (;;) {
        auto values = ordered.next_batch(8);
        if (values.matches().empty()) break;
        for (const auto& item : values.matches())
            terms.emplace_back(ll::batch::utf8(item));
    }
    assert(terms == (std::vector<std::string>{"cat", "bat", "cats"}));

    const auto retains_before = retain_calls.load();
    const auto releases_before = release_calls.load();
    {
        ll::transducer scoped(entries.resource());
        assert(retain_calls.load() == retains_before + 1);
    }
    assert(release_calls.load() == releases_before + 1);
}

void assert_cpp_non_text_domains() {
    namespace ll = vinary_tree::liblevenshtein;

    const std::string byte_term("\xff\0\x7f", 3);
    dictionary bytes(VT_UNIT_DOMAIN_BYTE, {{byte_term, UINT64_MAX}});
    ll::transducer byte_transducer(bytes.resource());
    const std::array byte_query{std::byte{0xff}, std::byte{0x00}, std::byte{0x7e}};
    auto byte_cursor = byte_transducer.query(byte_query, 1);
    auto byte_batch = byte_cursor.next_batch(1);
    assert(byte_batch.matches().size() == 1);
    const auto& byte_match = byte_batch.matches().front();
    const auto byte_result = ll::batch::bytes(byte_match);
    const std::array expected_bytes{std::byte{0xff}, std::byte{0x00}, std::byte{0x7f}};
    assert(std::ranges::equal(byte_result, expected_bytes));
    assert(byte_match.distance == 1 && byte_match.has_id == 1 &&
           byte_match.id == UINT64_MAX);
    try {
        static_cast<void>(ll::batch::utf8(byte_match));
        assert(false && "byte match was accepted as Unicode text");
    } catch (const std::invalid_argument&) {
    }

    const std::vector<std::uint64_t> token_term{0, UINT64_MAX};
    dictionary tokens({{token_term, 7}});
    ll::transducer token_transducer(tokens.resource());
    const std::array<std::uint64_t, 2> token_query{0, UINT64_MAX - 1};
    auto token_cursor = token_transducer.query(token_query, 1);
    auto token_batch = token_cursor.next_batch(1);
    assert(token_batch.matches().size() == 1);
    const auto& token_match = token_batch.matches().front();
    assert(std::ranges::equal(ll::batch::tokens(token_match), token_term));
    assert(token_match.distance == 1 && token_match.has_id == 1 &&
           token_match.id == 7);
}

} // namespace

int main() {
    assert_cpp_algorithms_order_and_lifecycle();
    assert_cpp_non_text_domains();

    dictionary words({{"cat", 1}, {"cot", 2}, {"cut", 3}, {"scat", std::nullopt}});
    assert_raw_c_passthrough(words.resource());
    vinary_tree::liblevenshtein::transducer automaton(words.resource());

    try {
        const std::array bytes{std::byte{'c'}, std::byte{'a'}, std::byte{'t'}};
        static_cast<void>(automaton.query(bytes, 0));
        assert(false && "Unicode dictionary accepted a raw-byte query");
    } catch (const vinary_tree::liblevenshtein::error& failure) {
        assert(failure.status() == LLEV_STATUS_DOMAIN_MISMATCH);
        assert(std::string_view(failure.what()).size() > 0);
    }

    auto expected_cursor = automaton.query("cat", 2);
    const auto expected = drain(expected_cursor);

    auto cursor = automaton.query("cat", 2);
    auto first = cursor.next_batch(1);
    assert(first.matches().size() == 1);
    auto moved_first = std::move(first);
    assert(first.matches().empty());
    assert(moved_first.matches().size() == 1);
    first = std::move(moved_first);
    assert(moved_first.matches().empty());
    std::vector<std::string> observed{
        std::string(vinary_tree::liblevenshtein::batch::utf8(first.matches().front()))};
    first = {};

    words.publish({{"cat", 1}, {"cit", 5}, {"cut", 30}, {"scat", std::nullopt}});
    auto suffix = drain(cursor);
    observed.insert(observed.end(), suffix.begin(), suffix.end());
    std::ranges::sort(observed);
    assert(observed == expected);

    auto fresh_cursor = automaton.query("cat", 2);
    assert(drain(fresh_cursor) != expected);

    auto cursor_before_move = automaton.query("cat", 0);
    auto cursor_after_move = std::move(cursor_before_move);
    try {
        static_cast<void>(cursor_before_move.next_batch());
        assert(false && "moved-from cursor remained callable");
    } catch (const std::logic_error&) {
    }
    assert(!cursor_after_move.next_batch().matches().empty());

    const auto contexts_before_escaped_batch = live_contexts.load();
    auto escaped_batch = [&automaton] {
        auto temporary_cursor = automaton.query("cat", 2);
        return temporary_cursor.next_batch(1);
    }();
    assert(!escaped_batch.matches().empty());
    assert(live_contexts.load() == contexts_before_escaped_batch + 1);
    escaped_batch = {};
    assert(live_contexts.load() == contexts_before_escaped_batch);

    const auto contexts_before_cursor_scope = live_contexts.load();
    {
        auto temporary_cursor = automaton.query("cat", 2);
        assert(live_contexts.load() == contexts_before_cursor_scope + 1);
    }
    assert(live_contexts.load() == contexts_before_cursor_scope);

    // C1 (identity/version): the C ABI reports its version and revision.
    assert(llev_abi_version() == LLEV_ABI_VERSION);
    assert(llev_api_revision() == LLEV_API_REVISION);

    // C6 (Unicode text domains + distance thresholds): the distance functions
    // decode UTF-8 to Unicode scalar values, so a multi-byte character counts
    // as a single edit; the threshold variants cap at the requested bound and
    // report an over-threshold sentinel beyond it.
    {
        const std::string cafe = "cafe";
        const std::string cafe_accent = "café"; // "café": é is 2 UTF-8 bytes, 1 scalar
        const std::string crab = "\U0001F980";        // 🦀: 4 UTF-8 bytes, 1 scalar
        // é -> e is one scalar substitution, not two byte edits.
        assert(llev_distance(cafe_accent.data(), cafe_accent.size(), cafe.data(),
                             cafe.size()) == 1);
        // identity holds on multi-byte input.
        assert(llev_distance(crab.data(), crab.size(), crab.data(), crab.size()) == 0);
        // a threshold at or above the true distance reports it exactly.
        assert(llev_damerau_distance_threshold(cafe_accent.data(), cafe_accent.size(),
                                               cafe.data(), cafe.size(), 2) == 1);
        // a threshold below the true distance exceeds the bound (sentinel > 1).
        assert(llev_distance_threshold(crab.data(), crab.size(), cafe.data(),
                                       cafe.size(), 1) > 1);
    }
}
