#include "liblevenshtein.hpp"

#include <algorithm>
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

struct node {
    std::map<char32_t, std::uint64_t> edges;
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
};

struct store {
    std::mutex lock;
    std::shared_ptr<const revision> current;
};

struct context {
    context(std::shared_ptr<store> source_value = {},
            std::shared_ptr<const revision> snapshot_value = {})
        : source(std::move(source_value)), snapshot(std::move(snapshot_value)) {
        live_contexts.fetch_add(1, std::memory_order_relaxed);
    }
    ~context() { live_contexts.fetch_sub(1, std::memory_order_relaxed); }

    std::atomic_size_t references{1};
    std::shared_ptr<store> source;
    std::shared_ptr<const revision> snapshot;
};

extern const VtResourceVTable resource_vtable;
extern const VtDictionaryVTable mutable_vtable;
extern const VtDictionaryVTable snapshot_vtable;

void retain(void* raw) {
    static_cast<context*>(raw)->references.fetch_add(1, std::memory_order_relaxed);
}

void release(void* raw) {
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
    *output = static_cast<context*>(raw)->snapshot
        ? static_cast<const void*>(&snapshot_vtable)
        : static_cast<const void*>(&mutable_vtable);
    return VT_STATUS_OK;
}

VtStatus snapshot(void* raw, VtResource* output) {
    if (!output) return VT_STATUS_NULL_POINTER;
    auto* source = static_cast<context*>(raw);
    auto* result = new context(source->source, captured(source));
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
    auto entry = value->nodes[id].edges.find(static_cast<char32_t>(label));
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
        output[*written] = {static_cast<std::uint64_t>(entry->first), entry->second};
        ++*written;
        ++entry;
    }
    return VT_STATUS_OK;
}

const VtResourceVTable resource_vtable = {
    sizeof(VtResourceVTable), VT_ABI_VERSION, 0, retain, release, query_interface};

constexpr VtDictionaryVTable dictionary_vtable(std::uint64_t flags) {
    return {sizeof(VtDictionaryVTable), VT_DICTIONARY_INTERFACE_VERSION,
            VT_UNIT_DOMAIN_UNICODE_SCALAR, VT_VALUE_DOMAIN_OPTIONAL_U64, flags,
            snapshot, root, length, is_final, node_value, transition, edges};
}

const VtDictionaryVTable mutable_vtable = dictionary_vtable(0);
const VtDictionaryVTable snapshot_vtable = dictionary_vtable(VT_DICTIONARY_FLAG_IMMUTABLE);

class dictionary {
public:
    explicit dictionary(std::map<std::string, std::optional<std::uint64_t>> entries)
        : source_(std::make_shared<store>()) {
        publish(std::move(entries));
        resource_.context = new context(source_);
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

} // namespace

int main() {
    dictionary words({{"cat", 1}, {"cot", 2}, {"cut", 3}, {"scat", std::nullopt}});
    vinary_tree::liblevenshtein::transducer automaton(words.resource());

    auto expected_cursor = automaton.query("cat", 2);
    const auto expected = drain(expected_cursor);

    auto cursor = automaton.query("cat", 2);
    auto first = cursor.next_batch(1);
    assert(first.matches().size() == 1);
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

    const auto contexts_before_escaped_batch = live_contexts.load();
    auto escaped_batch = [&automaton] {
        auto temporary_cursor = automaton.query("cat", 2);
        return temporary_cursor.next_batch(1);
    }();
    assert(!escaped_batch.matches().empty());
    assert(live_contexts.load() == contexts_before_escaped_batch + 1);
    escaped_batch = {};
    assert(live_contexts.load() == contexts_before_escaped_batch);

    // C1 (identity/version): the C ABI reports its version and revision.
    assert(llev_abi_version() == 1);
    assert(llev_api_revision() >= 1);

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
