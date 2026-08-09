// C9 leak-discipline suite for the C++ facade, run under AddressSanitizer +
// LeakSanitizer. A >=10,000-cycle create/use/free loop over dictionaries,
// transducers, query cursors, phonetic patterns, and rule sets must free every
// handle each cycle. The .hpp transducer/query_cursor free their handles via
// RAII; libdictenstein dictionaries and phonetic handles are freed explicitly.
// LSan fails the run on any leaked allocation at process exit.

#include "libdictenstein.h"
#include "liblevenshtein.hpp"

#include <cassert>
#include <cstdint>

namespace {

constexpr int cycles = 10000;

void insert(LdictDictionary* dictionary, const char* term, std::size_t length, std::uint64_t value) {
    LdictOptionalU64 optional{value, 1, {0}};
    std::uint8_t inserted = 0;
    assert(ldict_dictionary_insert_text(dictionary, reinterpret_cast<const std::uint8_t*>(term),
                                        length, optional, &inserted) == LDICT_STATUS_OK);
}

void transducer_cycle() {
    LdictDictionary* dictionary = nullptr;
    assert(ldict_dynamic_dawg_new(VT_UNIT_DOMAIN_UNICODE_SCALAR, &dictionary) == LDICT_STATUS_OK);
    insert(dictionary, "cat", 3, 1);
    insert(dictionary, "cot", 3, 2);
    insert(dictionary, "cut", 3, 3);
    insert(dictionary, "scat", 4, 4);
    VtResource resource{};
    assert(ldict_dictionary_resource(dictionary, &resource) == LDICT_STATUS_OK);
    {
        vinary_tree::liblevenshtein::transducer automaton(resource);
        auto cursor = automaton.query("cat", 2);
        for (;;) {
            auto values = cursor.next_batch(4);
            if (values.matches().empty()) break;
        }
    }
    ldict_dictionary_free(dictionary);
}

void phonetic_cycle() {
    LlevPhoneticPattern* pattern = nullptr;
    assert(llev_phonetic_pattern_compile_regex("c[ao]t", 6, &pattern) == LLEV_STATUS_OK);
    std::uint8_t matched = 0;
    assert(llev_phonetic_pattern_matches(pattern, "cat", 3, &matched) == LLEV_STATUS_OK);
    llev_phonetic_pattern_free(pattern);

    LlevPhoneticRuleSet* rules = nullptr;
    assert(llev_phonetic_rules_builtin(0 /* English orthography */, &rules) == LLEV_STATUS_OK);
    LlevOwnedString output{};
    assert(llev_phonetic_rules_apply(rules, "phone", 5, &output) == LLEV_STATUS_OK);
    llev_owned_string_free(&output);
    llev_phonetic_rules_free(rules);
}

} // namespace

int main() {
    for (int i = 0; i < cycles; ++i) {
        transducer_cycle();
        phonetic_cycle();
    }
    return 0;
}
