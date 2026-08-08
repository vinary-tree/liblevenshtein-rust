#include <node_api.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "libdictenstein.h"
#include "liblevenshtein.h"
#include "lling_llang.h"
#include "duallity.h"

namespace {

struct DictionaryHandle { LdictDictionary* value; };
struct TransducerHandle { LlevTransducer* value; };
struct CursorHandle { LlevQueryCursor* value; };
struct PatternHandle { LlevPhoneticPattern* value; };
struct RulesHandle { LlevPhoneticRuleSet* value; };
struct WfstBuilderHandle { LlingWfstBuilder* value; };
struct WfstHandle { VtResource value; };

napi_value fail_napi(napi_env env, const char* expression, napi_status status) {
  const napi_extended_error_info* info = nullptr;
  napi_get_last_error_info(env, &info);
  std::string message(expression);
  message += ": ";
  message += info && info->error_message ? info->error_message : "N-API failure";
  message += " (" + std::to_string(status) + ")";
  napi_throw_error(env, nullptr, message.c_str());
  return nullptr;
}

#define NAPI_OR_RETURN(env, expression)                                           \
  do {                                                                             \
    const napi_status napi_result_ = (expression);                                  \
    if (napi_result_ != napi_ok) return fail_napi((env), #expression, napi_result_); \
  } while (false)

napi_value undefined(napi_env env) {
  napi_value result;
  napi_get_undefined(env, &result);
  return result;
}

napi_value boolean(napi_env env, bool value) {
  napi_value result;
  napi_get_boolean(env, value, &result);
  return result;
}

napi_value number(napi_env env, double value) {
  napi_value result;
  napi_create_double(env, value, &result);
  return result;
}

napi_value bigint(napi_env env, uint64_t value) {
  napi_value result;
  napi_create_bigint_uint64(env, value, &result);
  return result;
}

void property(napi_env env, napi_value object, const char* name, napi_value value) {
  napi_set_named_property(env, object, name, value);
}

std::vector<napi_value> arguments(napi_env env, napi_callback_info info, size_t maximum) {
  std::vector<napi_value> result(maximum);
  size_t count = maximum;
  if (napi_get_cb_info(env, info, &count, result.data(), nullptr, nullptr) != napi_ok) return {};
  result.resize(count);
  return result;
}

bool uint32(napi_env env, napi_value value, uint32_t* output) {
  return napi_get_value_uint32(env, value, output) == napi_ok;
}

bool size_value(napi_env env, napi_value value, size_t* output) {
  uint32_t selected = 0;
  if (!uint32(env, value, &selected)) return false;
  *output = selected;
  return true;
}

bool uint64(napi_env env, napi_value value, uint64_t* output) {
  bool lossless = false;
  return napi_get_value_bigint_uint64(env, value, output, &lossless) == napi_ok && lossless;
}

bool string(napi_env env, napi_value value, std::string* output) {
  size_t length = 0;
  if (napi_get_value_string_utf8(env, value, nullptr, 0, &length) != napi_ok) return false;
  output->resize(length);
  return napi_get_value_string_utf8(env, value, output->data(), length + 1, &length) == napi_ok;
}

bool optional_u64(napi_env env, napi_value value, LdictOptionalU64* output) {
  napi_valuetype type;
  if (napi_typeof(env, value, &type) != napi_ok) return false;
  bool is_null = false;
  napi_value null_value;
  napi_get_null(env, &null_value);
  napi_strict_equals(env, value, null_value, &is_null);
  if (type == napi_undefined || is_null) {
    *output = LdictOptionalU64{0, 0, {0}};
    return true;
  }
  bool lossless = false;
  uint64_t number = 0;
  if (napi_get_value_bigint_uint64(env, value, &number, &lossless) != napi_ok || !lossless) return false;
  *output = LdictOptionalU64{number, 1, {0}};
  return true;
}

bool typed_array(napi_env env, napi_value value, napi_typedarray_type expected,
                 void** data, size_t* length) {
  napi_typedarray_type actual;
  napi_value array_buffer;
  size_t byte_offset = 0;
  return napi_get_typedarray_info(env, value, &actual, length, data,
                                  &array_buffer, &byte_offset) == napi_ok &&
         actual == expected;
}

template <typename T>
T* external(napi_env env, napi_value value) {
  void* result = nullptr;
  if (napi_get_value_external(env, value, &result) != napi_ok || result == nullptr) {
    napi_throw_type_error(env, nullptr, "invalid native handle");
    return nullptr;
  }
  return static_cast<T*>(result);
}

void dictionary_finalize(napi_env, void* data, void*) {
  auto* handle = static_cast<DictionaryHandle*>(data);
  if (handle->value) ldict_dictionary_free(handle->value);
  delete handle;
}
void transducer_finalize(napi_env, void* data, void*) {
  auto* handle = static_cast<TransducerHandle*>(data);
  if (handle->value) llev_transducer_free(handle->value);
  delete handle;
}
void cursor_finalize(napi_env, void* data, void*) {
  auto* handle = static_cast<CursorHandle*>(data);
  if (handle->value) (void)llev_query_cursor_free(handle->value);
  delete handle;
}
void pattern_finalize(napi_env, void* data, void*) {
  auto* handle = static_cast<PatternHandle*>(data);
  if (handle->value) llev_phonetic_pattern_free(handle->value);
  delete handle;
}
void rules_finalize(napi_env, void* data, void*) {
  auto* handle = static_cast<RulesHandle*>(data);
  if (handle->value) llev_phonetic_rules_free(handle->value);
  delete handle;
}
void wfst_builder_finalize(napi_env, void* data, void*) {
  auto* handle = static_cast<WfstBuilderHandle*>(data);
  if (handle->value) lling_wfst_builder_free(handle->value);
  delete handle;
}
void wfst_finalize(napi_env, void* data, void*) {
  auto* handle = static_cast<WfstHandle*>(data);
  lling_resource_release(handle->value);
  handle->value = VtResource{};
  delete handle;
}

napi_value dictionary_external(napi_env env, LdictDictionary* value) {
  napi_value result;
  auto* handle = new DictionaryHandle{value};
  const auto status = napi_create_external(env, handle, dictionary_finalize, nullptr, &result);
  if (status != napi_ok) {
    dictionary_finalize(env, handle, nullptr);
    return fail_napi(env, "napi_create_external", status);
  }
  return result;
}

napi_value ldict_error(napi_env env, LdictStatus status) {
  std::string message = ldict_last_error_message();
  if (message.empty()) message = "libdictenstein status " + std::to_string(status);
  napi_throw_error(env, nullptr, message.c_str());
  return nullptr;
}
napi_value llev_error(napi_env env, LlevStatus status) {
  std::string message = llev_last_error_message();
  if (message.empty()) message = "liblevenshtein status " + std::to_string(status);
  napi_throw_error(env, nullptr, message.c_str());
  return nullptr;
}
napi_value lling_error(napi_env env, LlingStatus status) {
  std::string message = lling_last_error_message();
  if (message.empty()) message = "lling-llang status " + std::to_string(status);
  napi_throw_error(env, nullptr, message.c_str());
  return nullptr;
}
napi_value duallity_error(napi_env env, DuallityStatus status) {
  std::string message = duallity_last_error_message();
  if (message.empty()) message = "duallity status " + std::to_string(status);
  napi_throw_error(env, nullptr, message.c_str());
  return nullptr;
}

napi_value dynamic_new(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  uint32_t domain = 0;
  if (args.size() != 1 || !uint32(env, args[0], &domain)) {
    napi_throw_type_error(env, nullptr, "dynamicDawgNew requires a numeric domain");
    return nullptr;
  }
  LdictDictionary* dictionary = nullptr;
  const auto status = ldict_dynamic_dawg_new(domain, &dictionary);
  return status == LDICT_STATUS_OK ? dictionary_external(env, dictionary) : ldict_error(env, status);
}

napi_value scdawg_new(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  uint32_t domain = 0;
  if (args.size() != 1 || !uint32(env, args[0], &domain)) {
    napi_throw_type_error(env, nullptr, "scdawgNew requires a numeric domain");
    return nullptr;
  }
  LdictDictionary* dictionary = nullptr;
  const auto status = ldict_scdawg_new(domain, &dictionary);
  return status == LDICT_STATUS_OK ? dictionary_external(env, dictionary) : ldict_error(env, status);
}

napi_value dat_new(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 2);
  uint32_t domain = 0;
  bool is_array = false;
  if (args.size() != 2 || !uint32(env, args[0], &domain) ||
      napi_is_array(env, args[1], &is_array) != napi_ok || !is_array) {
    napi_throw_type_error(env, nullptr, "doubleArrayTrieNew requires domain and entry array");
    return nullptr;
  }
  uint32_t length = 0;
  napi_get_array_length(env, args[1], &length);
  std::vector<std::string> terms(length);
  std::vector<LdictTextEntry> entries(length);
  for (uint32_t index = 0; index < length; ++index) {
    napi_value item, term, value;
    napi_get_element(env, args[1], index, &item);
    napi_get_named_property(env, item, "term", &term);
    napi_get_named_property(env, item, "value", &value);
    if (!string(env, term, &terms[index]) || !optional_u64(env, value, &entries[index].value)) {
      napi_throw_type_error(env, nullptr, "invalid DoubleArrayTrie entry");
      return nullptr;
    }
    entries[index].data = reinterpret_cast<const uint8_t*>(terms[index].data());
    entries[index].len = terms[index].size();
  }
  LdictDictionary* dictionary = nullptr;
  const auto status = ldict_double_array_trie_new(domain, entries.data(), entries.size(), &dictionary);
  return status == LDICT_STATUS_OK ? dictionary_external(env, dictionary) : ldict_error(env, status);
}

napi_value persistent(napi_env env, napi_callback_info info, bool create) {
  const auto args = arguments(env, info, 2);
  uint32_t domain = 0;
  std::string path;
  if (args.size() != 2 || !uint32(env, args[0], &domain) || !string(env, args[1], &path)) {
    napi_throw_type_error(env, nullptr, "persistent ARTrie requires domain and path");
    return nullptr;
  }
  LdictDictionary* dictionary = nullptr;
  const auto* data = reinterpret_cast<const uint8_t*>(path.data());
  const auto status = create
      ? ldict_persistent_artrie_create(domain, data, path.size(), &dictionary)
      : ldict_persistent_artrie_open(domain, data, path.size(), &dictionary);
  return status == LDICT_STATUS_OK ? dictionary_external(env, dictionary) : ldict_error(env, status);
}
napi_value persistent_create(napi_env env, napi_callback_info info) { return persistent(env, info, true); }
napi_value persistent_open(napi_env env, napi_callback_info info) { return persistent(env, info, false); }

napi_value dictionary_close(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  if (args.size() != 1) return nullptr;
  auto* handle = external<DictionaryHandle>(env, args[0]);
  if (!handle) return nullptr;
  if (handle->value) ldict_dictionary_free(std::exchange(handle->value, nullptr));
  return undefined(env);
}

napi_value dictionary_len(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  auto* handle = args.size() == 1 ? external<DictionaryHandle>(env, args[0]) : nullptr;
  if (!handle || !handle->value) return ldict_error(env, LDICT_STATUS_CLOSED);
  size_t length = 0;
  const auto status = ldict_dictionary_len(handle->value, &length);
  return status == LDICT_STATUS_OK ? number(env, static_cast<double>(length)) : ldict_error(env, status);
}

napi_value dictionary_put_text(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 3);
  auto* handle = args.size() == 3 ? external<DictionaryHandle>(env, args[0]) : nullptr;
  std::string term;
  LdictOptionalU64 value{};
  if (!handle || !handle->value || !string(env, args[1], &term) || !optional_u64(env, args[2], &value)) {
    napi_throw_type_error(env, nullptr, "putText requires dictionary, string, and bigint/null");
    return nullptr;
  }
  uint8_t inserted = 0;
  const auto status = ldict_dictionary_insert_text(handle->value,
      reinterpret_cast<const uint8_t*>(term.data()), term.size(), value, &inserted);
  return status == LDICT_STATUS_OK ? boolean(env, inserted != 0) : ldict_error(env, status);
}

napi_value dictionary_remove_text(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 2);
  auto* handle = args.size() == 2 ? external<DictionaryHandle>(env, args[0]) : nullptr;
  std::string term;
  if (!handle || !handle->value || !string(env, args[1], &term)) return nullptr;
  uint8_t removed = 0;
  const auto status = ldict_dictionary_remove_text(handle->value,
      reinterpret_cast<const uint8_t*>(term.data()), term.size(), &removed);
  return status == LDICT_STATUS_OK ? boolean(env, removed != 0) : ldict_error(env, status);
}

napi_value dictionary_get_text(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 2);
  auto* handle = args.size() == 2 ? external<DictionaryHandle>(env, args[0]) : nullptr;
  std::string term;
  if (!handle || !handle->value || !string(env, args[1], &term)) return nullptr;
  uint8_t found = 0;
  LdictOptionalU64 value{};
  const auto status = ldict_dictionary_get_text(handle->value,
      reinterpret_cast<const uint8_t*>(term.data()), term.size(), &found, &value);
  if (status != LDICT_STATUS_OK) return ldict_error(env, status);
  napi_value result, null_value;
  napi_create_object(env, &result);
  napi_get_null(env, &null_value);
  property(env, result, "found", boolean(env, found != 0));
  property(env, result, "value", value.has_value ? bigint(env, value.value) : null_value);
  return result;
}

napi_value dictionary_put_u64(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 3);
  auto* handle = args.size() == 3 ? external<DictionaryHandle>(env, args[0]) : nullptr;
  void* data = nullptr;
  size_t length = 0;
  LdictOptionalU64 value{};
  if (!handle || !handle->value ||
      !typed_array(env, args[1], napi_biguint64_array, &data, &length) ||
      !optional_u64(env, args[2], &value)) {
    napi_throw_type_error(env, nullptr, "putU64 requires dictionary, BigUint64Array, and bigint/null");
    return nullptr;
  }
  uint8_t inserted = 0;
  const auto status = ldict_dictionary_insert_u64(
      handle->value, static_cast<const uint64_t*>(data), length, value, &inserted);
  return status == LDICT_STATUS_OK ? boolean(env, inserted != 0) : ldict_error(env, status);
}

napi_value dictionary_remove_u64(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 2);
  auto* handle = args.size() == 2 ? external<DictionaryHandle>(env, args[0]) : nullptr;
  void* data = nullptr;
  size_t length = 0;
  if (!handle || !handle->value ||
      !typed_array(env, args[1], napi_biguint64_array, &data, &length)) return nullptr;
  uint8_t removed = 0;
  const auto status = ldict_dictionary_remove_u64(
      handle->value, static_cast<const uint64_t*>(data), length, &removed);
  return status == LDICT_STATUS_OK ? boolean(env, removed != 0) : ldict_error(env, status);
}

napi_value dictionary_get_u64(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 2);
  auto* handle = args.size() == 2 ? external<DictionaryHandle>(env, args[0]) : nullptr;
  void* data = nullptr;
  size_t length = 0;
  if (!handle || !handle->value ||
      !typed_array(env, args[1], napi_biguint64_array, &data, &length)) return nullptr;
  uint8_t found = 0;
  LdictOptionalU64 value{};
  const auto status = ldict_dictionary_get_u64(
      handle->value, static_cast<const uint64_t*>(data), length, &found, &value);
  if (status != LDICT_STATUS_OK) return ldict_error(env, status);
  napi_value result, null_value;
  napi_create_object(env, &result);
  napi_get_null(env, &null_value);
  property(env, result, "found", boolean(env, found != 0));
  property(env, result, "value", value.has_value ? bigint(env, value.value) : null_value);
  return result;
}

napi_value dictionary_noarg(napi_env env, napi_callback_info info, int operation) {
  const auto args = arguments(env, info, 1);
  auto* handle = args.size() == 1 ? external<DictionaryHandle>(env, args[0]) : nullptr;
  if (!handle || !handle->value) return ldict_error(env, LDICT_STATUS_CLOSED);
  if (operation == 0) {
    const auto status = ldict_dictionary_clear(handle->value);
    return status == LDICT_STATUS_OK ? undefined(env) : ldict_error(env, status);
  }
  if (operation == 1) {
    size_t reclaimed = 0;
    const auto status = ldict_dictionary_compact(handle->value, &reclaimed);
    return status == LDICT_STATUS_OK ? number(env, static_cast<double>(reclaimed)) : ldict_error(env, status);
  }
  const auto status = ldict_dictionary_checkpoint(handle->value);
  return status == LDICT_STATUS_OK ? undefined(env) : ldict_error(env, status);
}
napi_value dictionary_clear(napi_env env, napi_callback_info info) { return dictionary_noarg(env, info, 0); }
napi_value dictionary_compact(napi_env env, napi_callback_info info) { return dictionary_noarg(env, info, 1); }
napi_value dictionary_checkpoint(napi_env env, napi_callback_info info) { return dictionary_noarg(env, info, 2); }

napi_value substring_operation(napi_env env, napi_callback_info info, bool frequency) {
  const auto args = arguments(env, info, 2);
  auto* handle = args.size() == 2 ? external<DictionaryHandle>(env, args[0]) : nullptr;
  std::string term;
  if (!handle || !handle->value || !string(env, args[1], &term)) return nullptr;
  if (frequency) {
    size_t output = 0;
    const auto status = ldict_scdawg_substring_frequency(handle->value,
        reinterpret_cast<const uint8_t*>(term.data()), term.size(), &output);
    return status == LDICT_STATUS_OK ? number(env, static_cast<double>(output)) : ldict_error(env, status);
  }
  uint8_t output = 0;
  const auto status = ldict_scdawg_contains_substring(handle->value,
      reinterpret_cast<const uint8_t*>(term.data()), term.size(), &output);
  return status == LDICT_STATUS_OK ? boolean(env, output != 0) : ldict_error(env, status);
}
napi_value contains_substring(napi_env env, napi_callback_info info) { return substring_operation(env, info, false); }
napi_value substring_frequency(napi_env env, napi_callback_info info) { return substring_operation(env, info, true); }

napi_value transducer_new(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 2);
  auto* dictionary = args.size() == 2 ? external<DictionaryHandle>(env, args[0]) : nullptr;
  uint32_t algorithm = 0;
  if (!dictionary || !dictionary->value || !uint32(env, args[1], &algorithm)) return nullptr;
  VtResource resource{};
  auto status = ldict_dictionary_resource(dictionary->value, &resource);
  if (status != LDICT_STATUS_OK) return ldict_error(env, status);
  LlevTransducer* transducer = nullptr;
  const auto llev_status = llev_transducer_new(&resource, algorithm, &transducer);
  if (llev_status != LLEV_STATUS_OK) return llev_error(env, llev_status);
  napi_value result;
  auto* handle = new TransducerHandle{transducer};
  const auto napi_status = napi_create_external(env, handle, transducer_finalize, nullptr, &result);
  if (napi_status != napi_ok) {
    transducer_finalize(env, handle, nullptr);
    return fail_napi(env, "napi_create_external", napi_status);
  }
  return result;
}

napi_value transducer_close(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  auto* handle = args.size() == 1 ? external<TransducerHandle>(env, args[0]) : nullptr;
  if (!handle) return nullptr;
  if (handle->value) llev_transducer_free(std::exchange(handle->value, nullptr));
  return undefined(env);
}

napi_value cursor_external(napi_env env, LlevQueryCursor* value) {
  napi_value result;
  auto* handle = new CursorHandle{value};
  const auto status = napi_create_external(env, handle, cursor_finalize, nullptr, &result);
  if (status != napi_ok) {
    cursor_finalize(env, handle, nullptr);
    return fail_napi(env, "napi_create_external", status);
  }
  return result;
}

napi_value query_text(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 4);
  auto* handle = args.size() == 4 ? external<TransducerHandle>(env, args[0]) : nullptr;
  std::string query;
  size_t maximum = 0;
  uint32_t order = 0;
  if (!handle || !handle->value || !string(env, args[1], &query) ||
      !size_value(env, args[2], &maximum) || !uint32(env, args[3], &order)) return nullptr;
  LlevQueryCursor* cursor = nullptr;
  const auto status = llev_transducer_query_utf8(handle->value, query.data(), query.size(),
                                                  maximum, order, &cursor);
  return status == LLEV_STATUS_OK ? cursor_external(env, cursor) : llev_error(env, status);
}

napi_value query_typed(napi_env env, napi_callback_info info, bool u64) {
  const auto args = arguments(env, info, 4);
  auto* handle = args.size() == 4 ? external<TransducerHandle>(env, args[0]) : nullptr;
  void* data = nullptr;
  size_t length = 0;
  size_t maximum = 0;
  uint32_t order = 0;
  const auto type = u64 ? napi_biguint64_array : napi_uint8_array;
  if (!handle || !handle->value || !typed_array(env, args[1], type, &data, &length) ||
      !size_value(env, args[2], &maximum) || !uint32(env, args[3], &order)) return nullptr;
  LlevQueryCursor* cursor = nullptr;
  const auto status = u64
      ? llev_transducer_query_u64(handle->value, static_cast<const uint64_t*>(data),
                                  length, maximum, order, &cursor)
      : llev_transducer_query_bytes(handle->value, static_cast<const uint8_t*>(data),
                                    length, maximum, order, &cursor);
  return status == LLEV_STATUS_OK ? cursor_external(env, cursor) : llev_error(env, status);
}
napi_value query_bytes(napi_env env, napi_callback_info info) { return query_typed(env, info, false); }
napi_value query_u64(napi_env env, napi_callback_info info) { return query_typed(env, info, true); }

napi_value cursor_close(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  auto* handle = args.size() == 1 ? external<CursorHandle>(env, args[0]) : nullptr;
  if (!handle) return nullptr;
  if (handle->value) {
    const auto status = llev_query_cursor_free(handle->value);
    if (status != LLEV_STATUS_OK) return llev_error(env, status);
    handle->value = nullptr;
  }
  return undefined(env);
}

napi_value cursor_next_batch(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 2);
  auto* handle = args.size() == 2 ? external<CursorHandle>(env, args[0]) : nullptr;
  size_t maximum = 0;
  if (!handle || !handle->value || !size_value(env, args[1], &maximum)) return nullptr;
  LlevMatchBatchView batch{};
  const auto status = llev_query_cursor_next_batch(handle->value, maximum, &batch);
  if (status == LLEV_STATUS_END) {
    napi_value empty;
    napi_create_array(env, &empty);
    return empty;
  }
  if (status != LLEV_STATUS_OK) return llev_error(env, status);
  napi_value output;
  napi_create_array_with_length(env, batch.len, &output);
  for (size_t index = 0; index < batch.len; ++index) {
    const auto& match = batch.matches[index];
    napi_value item, term, domain, payload, null_value;
    napi_create_object(env, &item);
    napi_create_object(env, &term);
    napi_get_null(env, &null_value);
    if (match.unit_domain == VT_UNIT_DOMAIN_UNICODE_SCALAR) {
      napi_create_string_utf8(env, static_cast<const char*>(match.term_data), match.byte_len, &payload);
      napi_create_string_utf8(env, "unicode", NAPI_AUTO_LENGTH, &domain);
    } else {
      void* destination = nullptr;
      napi_value array_buffer;
      napi_create_arraybuffer(env, match.byte_len, &destination, &array_buffer);
      std::memcpy(destination, match.term_data, match.byte_len);
      const auto type = match.unit_domain == VT_UNIT_DOMAIN_U64 ? napi_biguint64_array : napi_uint8_array;
      const size_t length = match.unit_domain == VT_UNIT_DOMAIN_U64 ? match.term_len : match.byte_len;
      napi_create_typedarray(env, type, length, array_buffer, 0, &payload);
      napi_create_string_utf8(env, match.unit_domain == VT_UNIT_DOMAIN_U64 ? "u64" : "byte",
                              NAPI_AUTO_LENGTH, &domain);
    }
    property(env, term, "domain", domain);
    property(env, term, "value", payload);
    property(env, item, "term", term);
    property(env, item, "distance", number(env, static_cast<double>(match.distance)));
    property(env, item, "id", match.has_id ? bigint(env, match.id) : null_value);
    napi_set_element(env, output, static_cast<uint32_t>(index), item);
  }
  const auto release = llev_query_cursor_release_batch(handle->value, batch.generation);
  return release == LLEV_STATUS_OK ? output : llev_error(env, release);
}

napi_value pattern_compile(napi_env env, napi_callback_info info, bool llre) {
  const auto args = arguments(env, info, 1);
  std::string source;
  if (args.size() != 1 || !string(env, args[0], &source)) {
    napi_throw_type_error(env, nullptr, "pattern source must be a string");
    return nullptr;
  }
  LlevPhoneticPattern* pattern = nullptr;
  const auto status = llre
      ? llev_phonetic_pattern_compile_llre(source.data(), source.size(), &pattern)
      : llev_phonetic_pattern_compile_regex(source.data(), source.size(), &pattern);
  if (status != LLEV_STATUS_OK) return llev_error(env, status);
  napi_value result;
  auto* handle = new PatternHandle{pattern};
  const auto napi_status = napi_create_external(env, handle, pattern_finalize, nullptr, &result);
  if (napi_status != napi_ok) {
    pattern_finalize(env, handle, nullptr);
    return fail_napi(env, "napi_create_external", napi_status);
  }
  return result;
}
napi_value pattern_compile_regex(napi_env env, napi_callback_info info) {
  return pattern_compile(env, info, false);
}
napi_value pattern_compile_llre(napi_env env, napi_callback_info info) {
  return pattern_compile(env, info, true);
}

napi_value pattern_close(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  auto* handle = args.size() == 1 ? external<PatternHandle>(env, args[0]) : nullptr;
  if (!handle) return nullptr;
  if (handle->value) llev_phonetic_pattern_free(std::exchange(handle->value, nullptr));
  return undefined(env);
}

napi_value pattern_matches(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 2);
  auto* handle = args.size() == 2 ? external<PatternHandle>(env, args[0]) : nullptr;
  std::string input;
  if (!handle || !handle->value || !string(env, args[1], &input)) return nullptr;
  uint8_t matches = 0;
  const auto status = llev_phonetic_pattern_matches(
      handle->value, input.data(), input.size(), &matches);
  return status == LLEV_STATUS_OK ? boolean(env, matches != 0) : llev_error(env, status);
}

napi_value query_pattern(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 3);
  auto* transducer = args.size() == 3 ? external<TransducerHandle>(env, args[0]) : nullptr;
  auto* pattern = args.size() == 3 ? external<PatternHandle>(env, args[1]) : nullptr;
  uint32_t maximum = 0;
  if (!transducer || !transducer->value || !pattern || !pattern->value ||
      !uint32(env, args[2], &maximum) || maximum > UINT8_MAX) return nullptr;
  LlevQueryCursor* cursor = nullptr;
  const auto status = llev_transducer_query_pattern(
      transducer->value, pattern->value, static_cast<uint8_t>(maximum), &cursor);
  return status == LLEV_STATUS_OK ? cursor_external(env, cursor) : llev_error(env, status);
}

napi_value rules_compile(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  std::string source;
  if (args.size() != 1 || !string(env, args[0], &source)) return nullptr;
  LlevPhoneticRuleSet* rules = nullptr;
  LlevStatus status;
  if (source == "english-orthography") {
    status = llev_phonetic_rules_builtin(LLEV_PHONETIC_RULE_SET_ENGLISH_ORTHOGRAPHY, &rules);
  } else if (source == "english-phonetic") {
    status = llev_phonetic_rules_builtin(LLEV_PHONETIC_RULE_SET_ENGLISH_PHONETIC, &rules);
  } else {
    status = llev_phonetic_rules_parse(source.data(), source.size(), &rules);
  }
  if (status != LLEV_STATUS_OK) return llev_error(env, status);
  napi_value result;
  auto* handle = new RulesHandle{rules};
  const auto napi_status = napi_create_external(env, handle, rules_finalize, nullptr, &result);
  if (napi_status != napi_ok) {
    rules_finalize(env, handle, nullptr);
    return fail_napi(env, "napi_create_external", napi_status);
  }
  return result;
}

napi_value rules_close(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  auto* handle = args.size() == 1 ? external<RulesHandle>(env, args[0]) : nullptr;
  if (!handle) return nullptr;
  if (handle->value) llev_phonetic_rules_free(std::exchange(handle->value, nullptr));
  return undefined(env);
}

napi_value rules_apply(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 2);
  auto* handle = args.size() == 2 ? external<RulesHandle>(env, args[0]) : nullptr;
  std::string input;
  if (!handle || !handle->value || !string(env, args[1], &input)) return nullptr;
  LlevOwnedString output{};
  const auto status = llev_phonetic_rules_apply(
      handle->value, input.data(), input.size(), &output);
  if (status != LLEV_STATUS_OK) return llev_error(env, status);
  napi_value result;
  const auto napi_status = napi_create_string_utf8(env, output.data, output.len, &result);
  llev_owned_string_free(&output);
  return napi_status == napi_ok ? result : fail_napi(env, "napi_create_string_utf8", napi_status);
}

napi_value distance(napi_env env, napi_callback_info info, int kind) {
  const auto args = arguments(env, info, 2);
  std::string source, target;
  if (args.size() != 2 || !string(env, args[0], &source) || !string(env, args[1], &target)) return nullptr;
  size_t result = 0;
  if (kind == 0) result = llev_distance(source.data(), source.size(), target.data(), target.size());
  if (kind == 1) result = llev_damerau_distance(source.data(), source.size(), target.data(), target.size());
  if (kind == 2) result = llev_true_damerau_distance(source.data(), source.size(), target.data(), target.size());
  return number(env, static_cast<double>(result));
}
napi_value levenshtein_distance(napi_env env, napi_callback_info info) { return distance(env, info, 0); }
napi_value damerau_distance(napi_env env, napi_callback_info info) { return distance(env, info, 1); }
napi_value true_damerau_distance(napi_env env, napi_callback_info info) { return distance(env, info, 2); }

napi_value wfst_external(napi_env env, VtResource value) {
  napi_value result;
  auto* handle = new WfstHandle{value};
  const auto status = napi_create_external(env, handle, wfst_finalize, nullptr, &result);
  if (status != napi_ok) {
    wfst_finalize(env, handle, nullptr);
    return fail_napi(env, "napi_create_external", status);
  }
  return result;
}

napi_value wfst_builder_new(napi_env env, napi_callback_info) {
  LlingWfstBuilder* builder = nullptr;
  const auto status = lling_wfst_builder_new(&builder);
  if (status != LLING_STATUS_OK) return lling_error(env, status);
  napi_value result;
  auto* handle = new WfstBuilderHandle{builder};
  const auto napi_status = napi_create_external(env, handle, wfst_builder_finalize, nullptr, &result);
  if (napi_status != napi_ok) {
    wfst_builder_finalize(env, handle, nullptr);
    return fail_napi(env, "napi_create_external", napi_status);
  }
  return result;
}

napi_value wfst_builder_close(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  auto* handle = args.size() == 1 ? external<WfstBuilderHandle>(env, args[0]) : nullptr;
  if (!handle) return nullptr;
  if (handle->value) lling_wfst_builder_free(std::exchange(handle->value, nullptr));
  return undefined(env);
}

napi_value wfst_builder_add_state(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  auto* handle = args.size() == 1 ? external<WfstBuilderHandle>(env, args[0]) : nullptr;
  if (!handle || !handle->value) return lling_error(env, LLING_STATUS_CLOSED);
  uint32_t state = 0;
  const auto status = lling_wfst_builder_add_state(handle->value, &state);
  return status == LLING_STATUS_OK ? number(env, state) : lling_error(env, status);
}

napi_value wfst_builder_set_start(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 2);
  auto* handle = args.size() == 2 ? external<WfstBuilderHandle>(env, args[0]) : nullptr;
  uint32_t state = 0;
  if (!handle || !handle->value || !uint32(env, args[1], &state)) return nullptr;
  const auto status = lling_wfst_builder_set_start(handle->value, state);
  return status == LLING_STATUS_OK ? undefined(env) : lling_error(env, status);
}

napi_value wfst_builder_set_final(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 3);
  auto* handle = args.size() == 3 ? external<WfstBuilderHandle>(env, args[0]) : nullptr;
  uint32_t state = 0;
  double weight = 0;
  if (!handle || !handle->value || !uint32(env, args[1], &state) ||
      napi_get_value_double(env, args[2], &weight) != napi_ok) return nullptr;
  const auto status = lling_wfst_builder_set_final(handle->value, state, weight);
  return status == LLING_STATUS_OK ? undefined(env) : lling_error(env, status);
}

napi_value wfst_builder_add_arc(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 8);
  auto* handle = args.size() == 8 ? external<WfstBuilderHandle>(env, args[0]) : nullptr;
  uint32_t from = 0, has_input = 0, has_output = 0, to = 0;
  uint64_t input = 0, output = 0;
  double weight = 0;
  if (!handle || !handle->value || !uint32(env, args[1], &from) ||
      !uint64(env, args[2], &input) || !uint32(env, args[3], &has_input) ||
      !uint64(env, args[4], &output) || !uint32(env, args[5], &has_output) ||
      !uint32(env, args[6], &to) || napi_get_value_double(env, args[7], &weight) != napi_ok) {
    napi_throw_type_error(env, nullptr, "invalid WFST arc fields");
    return nullptr;
  }
  const auto status = lling_wfst_builder_add_arc(
      handle->value, from, input, static_cast<uint8_t>(has_input),
      output, static_cast<uint8_t>(has_output), to, weight);
  return status == LLING_STATUS_OK ? undefined(env) : lling_error(env, status);
}

napi_value wfst_builder_build(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  auto* handle = args.size() == 1 ? external<WfstBuilderHandle>(env, args[0]) : nullptr;
  if (!handle || !handle->value) return lling_error(env, LLING_STATUS_CLOSED);
  LlingWfst* wfst = nullptr;
  auto status = lling_wfst_builder_build(handle->value, &wfst);
  if (status != LLING_STATUS_OK) return lling_error(env, status);
  VtResource resource{};
  status = lling_wfst_resource(wfst, &resource);
  lling_wfst_free(wfst);
  return status == LLING_STATUS_OK ? wfst_external(env, resource) : lling_error(env, status);
}

napi_value duallity_wfst_new(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 5);
  auto* dictionary = args.size() == 5 ? external<DictionaryHandle>(env, args[0]) : nullptr;
  std::string query;
  size_t maximum = 0;
  uint32_t algorithm = 0, kind = 0;
  if (!dictionary || !dictionary->value || !string(env, args[1], &query) ||
      !size_value(env, args[2], &maximum) || !uint32(env, args[3], &algorithm) ||
      !uint32(env, args[4], &kind)) return nullptr;
  VtResource dictionary_resource{};
  const auto dictionary_status = ldict_dictionary_resource(dictionary->value, &dictionary_resource);
  if (dictionary_status != LDICT_STATUS_OK) return ldict_error(env, dictionary_status);
  DuallityWfst* wfst = nullptr;
  auto status = duallity_wfst_new(dictionary_resource,
      reinterpret_cast<const uint8_t*>(query.data()), query.size(), maximum,
      algorithm, kind, &wfst);
  if (status != DUALLITY_STATUS_OK) return duallity_error(env, status);
  VtResource resource{};
  status = duallity_wfst_resource(wfst, &resource);
  duallity_wfst_free(wfst);
  return status == DUALLITY_STATUS_OK ? wfst_external(env, resource) : duallity_error(env, status);
}

napi_value wfst_compose(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 2);
  auto* first = args.size() == 2 ? external<WfstHandle>(env, args[0]) : nullptr;
  auto* second = args.size() == 2 ? external<WfstHandle>(env, args[1]) : nullptr;
  if (!first || !second || first->value.context == nullptr || second->value.context == nullptr) return nullptr;
  LlingWfst* wfst = nullptr;
  auto status = lling_wfst_compose(first->value, second->value, &wfst);
  if (status != LLING_STATUS_OK) return lling_error(env, status);
  VtResource resource{};
  status = lling_wfst_resource(wfst, &resource);
  lling_wfst_free(wfst);
  return status == LLING_STATUS_OK ? wfst_external(env, resource) : lling_error(env, status);
}

napi_value wfst_close(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  auto* handle = args.size() == 1 ? external<WfstHandle>(env, args[0]) : nullptr;
  if (!handle) return nullptr;
  lling_resource_release(handle->value);
  handle->value = VtResource{};
  return undefined(env);
}

const VtWfstVTable* wfst_table(napi_env env, WfstHandle* handle) {
  if (!handle || handle->value.context == nullptr || handle->value.vtable == nullptr) {
    napi_throw_error(env, nullptr, "WFST is closed");
    return nullptr;
  }
  const void* interface = nullptr;
  const auto status = handle->value.vtable->query_interface(
      handle->value.context, &VT_WFST_INTERFACE_ID, VT_WFST_INTERFACE_VERSION, &interface);
  if (status != VT_STATUS_OK || interface == nullptr) {
    napi_throw_error(env, nullptr, "resource has no compatible scalar WFST interface");
    return nullptr;
  }
  return static_cast<const VtWfstVTable*>(interface);
}

napi_value wfst_start(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  auto* handle = args.size() == 1 ? external<WfstHandle>(env, args[0]) : nullptr;
  const auto* table = wfst_table(env, handle);
  if (!table) return nullptr;
  uint64_t state = 0;
  const auto status = table->start(handle->value.context, &state);
  if (status != VT_STATUS_OK) {
    napi_throw_error(env, nullptr, "WFST start callback failed");
    return nullptr;
  }
  return bigint(env, state);
}

napi_value wfst_weight_domain(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 1);
  auto* handle = args.size() == 1 ? external<WfstHandle>(env, args[0]) : nullptr;
  const auto* table = wfst_table(env, handle);
  return table ? number(env, table->weight_domain) : nullptr;
}

napi_value label_value(napi_env env, uint64_t label, bool present) {
  if (!present) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }
  if (label > 0x10ffff || (label >= 0xd800 && label <= 0xdfff)) {
    napi_throw_error(env, nullptr, "WFST provider returned an invalid Unicode scalar");
    return nullptr;
  }
  char encoded[4];
  size_t length = 0;
  if (label <= 0x7f) encoded[length++] = static_cast<char>(label);
  else if (label <= 0x7ff) {
    encoded[length++] = static_cast<char>(0xc0 | (label >> 6));
    encoded[length++] = static_cast<char>(0x80 | (label & 0x3f));
  } else if (label <= 0xffff) {
    encoded[length++] = static_cast<char>(0xe0 | (label >> 12));
    encoded[length++] = static_cast<char>(0x80 | ((label >> 6) & 0x3f));
    encoded[length++] = static_cast<char>(0x80 | (label & 0x3f));
  } else {
    encoded[length++] = static_cast<char>(0xf0 | (label >> 18));
    encoded[length++] = static_cast<char>(0x80 | ((label >> 12) & 0x3f));
    encoded[length++] = static_cast<char>(0x80 | ((label >> 6) & 0x3f));
    encoded[length++] = static_cast<char>(0x80 | (label & 0x3f));
  }
  napi_value result;
  napi_create_string_utf8(env, encoded, length, &result);
  return result;
}

napi_value wfst_state(napi_env env, napi_callback_info info) {
  const auto args = arguments(env, info, 2);
  auto* handle = args.size() == 2 ? external<WfstHandle>(env, args[0]) : nullptr;
  uint64_t state = 0;
  if (!handle || !uint64(env, args[1], &state)) return nullptr;
  const auto* table = wfst_table(env, handle);
  if (!table) return nullptr;
  uint8_t valid = 0, final_state = 0;
  double final_weight = 0;
  auto status = table->state_info(handle->value.context, state, &valid, &final_state, &final_weight);
  if (status != VT_STATUS_OK) {
    napi_throw_error(env, nullptr, "WFST state_info callback failed");
    return nullptr;
  }
  std::vector<VtWfstArc> arcs;
  if (valid) {
    size_t offset = 0;
    do {
      std::vector<VtWfstArc> page(VT_RECOMMENDED_ARC_BATCH);
      size_t written = 0, total = 0;
      status = table->state_arcs(handle->value.context, state, offset, page.data(), page.size(), &written, &total);
      if (status != VT_STATUS_OK || written > page.size() || offset + written > total ||
          (written == 0 && offset < total)) {
        napi_throw_error(env, nullptr, "WFST state_arcs callback returned invalid output");
        return nullptr;
      }
      arcs.insert(arcs.end(), page.begin(), page.begin() + static_cast<std::ptrdiff_t>(written));
      offset += written;
      if (offset == total) break;
    } while (true);
  }
  napi_value result, output;
  napi_create_object(env, &result);
  napi_create_array_with_length(env, arcs.size(), &output);
  property(env, result, "valid", boolean(env, valid != 0));
  property(env, result, "final", boolean(env, final_state != 0));
  property(env, result, "finalWeight", number(env, final_weight));
  for (size_t index = 0; index < arcs.size(); ++index) {
    napi_value arc;
    napi_create_object(env, &arc);
    property(env, arc, "input", label_value(env, arcs[index].input_label, arcs[index].has_input != 0));
    property(env, arc, "output", label_value(env, arcs[index].output_label, arcs[index].has_output != 0));
    property(env, arc, "target", bigint(env, arcs[index].target_state));
    property(env, arc, "weight", number(env, arcs[index].weight));
    napi_set_element(env, output, static_cast<uint32_t>(index), arc);
  }
  property(env, result, "arcs", output);
  return result;
}

napi_value initialize(napi_env env, napi_value exports) {
  const napi_property_descriptor properties[] = {
    {"dynamicDawgNew", nullptr, dynamic_new, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"doubleArrayTrieNew", nullptr, dat_new, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"scdawgNew", nullptr, scdawg_new, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"persistentARTrieCreate", nullptr, persistent_create, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"persistentARTrieOpen", nullptr, persistent_open, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"dictionaryClose", nullptr, dictionary_close, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"dictionaryLen", nullptr, dictionary_len, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"dictionaryPutText", nullptr, dictionary_put_text, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"dictionaryRemoveText", nullptr, dictionary_remove_text, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"dictionaryGetText", nullptr, dictionary_get_text, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"dictionaryPutU64", nullptr, dictionary_put_u64, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"dictionaryRemoveU64", nullptr, dictionary_remove_u64, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"dictionaryGetU64", nullptr, dictionary_get_u64, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"dictionaryClear", nullptr, dictionary_clear, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"dictionaryCompact", nullptr, dictionary_compact, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"dictionaryCheckpoint", nullptr, dictionary_checkpoint, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"containsSubstring", nullptr, contains_substring, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"substringFrequency", nullptr, substring_frequency, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"transducerNew", nullptr, transducer_new, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"transducerClose", nullptr, transducer_close, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"queryText", nullptr, query_text, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"queryBytes", nullptr, query_bytes, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"queryU64", nullptr, query_u64, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"cursorNextBatch", nullptr, cursor_next_batch, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"cursorClose", nullptr, cursor_close, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"patternCompileRegex", nullptr, pattern_compile_regex, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"patternCompileLlre", nullptr, pattern_compile_llre, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"patternMatches", nullptr, pattern_matches, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"patternClose", nullptr, pattern_close, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"queryPattern", nullptr, query_pattern, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"rulesCompile", nullptr, rules_compile, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"rulesApply", nullptr, rules_apply, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"rulesClose", nullptr, rules_close, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"levenshteinDistance", nullptr, levenshtein_distance, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"damerauDistance", nullptr, damerau_distance, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"trueDamerauDistance", nullptr, true_damerau_distance, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"wfstBuilderNew", nullptr, wfst_builder_new, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"wfstBuilderClose", nullptr, wfst_builder_close, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"wfstBuilderAddState", nullptr, wfst_builder_add_state, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"wfstBuilderSetStart", nullptr, wfst_builder_set_start, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"wfstBuilderSetFinal", nullptr, wfst_builder_set_final, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"wfstBuilderAddArc", nullptr, wfst_builder_add_arc, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"wfstBuilderBuild", nullptr, wfst_builder_build, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"duallityWfstNew", nullptr, duallity_wfst_new, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"wfstCompose", nullptr, wfst_compose, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"wfstClose", nullptr, wfst_close, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"wfstStart", nullptr, wfst_start, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"wfstWeightDomain", nullptr, wfst_weight_domain, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"wfstState", nullptr, wfst_state, nullptr, nullptr, nullptr, napi_default, nullptr},
  };
  NAPI_OR_RETURN(env, napi_define_properties(env, exports,
      sizeof(properties) / sizeof(properties[0]), properties));
  return exports;
}

}  // namespace

NAPI_MODULE(NODE_GYP_MODULE_NAME, initialize)
