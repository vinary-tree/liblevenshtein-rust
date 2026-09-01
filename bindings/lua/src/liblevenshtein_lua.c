#include <lua.h>
#include <lauxlib.h>

#include <stdint.h>
#include <string.h>

#include "liblevenshtein.h"
#include "vinary_tree_lua.h"

#define TRANSDUCER_MT "vinary-tree.liblevenshtein.transducer"
#define CURSOR_MT "vinary-tree.liblevenshtein.cursor"
#define PATTERN_MT "vinary-tree.liblevenshtein.pattern"
#define RULES_MT "vinary-tree.liblevenshtein.rules"

typedef struct LuaTransducer { LlevTransducer* value; } LuaTransducer;
typedef struct LuaCursor {
    LlevQueryCursor* value;
    LlevMatchBatchView batch;
    size_t index;
    uint8_t leased;
} LuaCursor;
typedef struct LuaPattern { LlevPhoneticPattern* value; } LuaPattern;
typedef struct LuaRules { LlevPhoneticRuleSet* value; } LuaRules;

static int fail(lua_State* state, LlevStatus status) {
    const char* message = llev_last_error_message();
    return luaL_error(state, "%s (status %d)", message && *message ? message : "liblevenshtein error", (int)status);
}

static uint32_t algorithm(lua_State* state, int index) {
    const char* value = luaL_optstring(state, index, "standard");
    if (strcmp(value, "standard") == 0) return LLEV_ALGORITHM_STANDARD;
    if (strcmp(value, "transposition") == 0) return LLEV_ALGORITHM_TRANSPOSITION;
    if (strcmp(value, "merge-and-split") == 0) return LLEV_ALGORITHM_MERGE_AND_SPLIT;
    if (strcmp(value, "damerau-levenshtein") == 0) return LLEV_ALGORITHM_DAMERAU_LEVENSHTEIN;
    luaL_argerror(state, index, "unknown algorithm");
    return 0;
}

static uint32_t order(lua_State* state, int index) {
    const char* value = luaL_optstring(state, index, "traversal");
    if (strcmp(value, "traversal") == 0) return LLEV_QUERY_ORDER_TRAVERSAL;
    if (strcmp(value, "distance-then-term") == 0) return LLEV_QUERY_ORDER_DISTANCE_THEN_TERM;
    luaL_argerror(state, index, "unknown query order");
    return 0;
}

static size_t maximum_distance(lua_State* state, int index) {
    lua_Integer value = luaL_checkinteger(state, index);
    luaL_argcheck(state, value >= 0, index, "maximum distance must be non-negative");
    return (size_t)value;
}

static const uint64_t* u64_sequence(lua_State* state, int index, size_t* out_length) {
    luaL_checktype(state, index, LUA_TTABLE);
    lua_Integer length = luaL_len(state, index);
    luaL_argcheck(state, length >= 0, index, "invalid token sequence length");
    uint64_t* data = (uint64_t*)lua_newuserdatauv(
        state, (size_t)length * sizeof(uint64_t), 0);
    for (lua_Integer position = 1; position <= length; ++position) {
        lua_rawgeti(state, index, position);
        lua_Integer token = luaL_checkinteger(state, -1);
        luaL_argcheck(state, token >= 0, index, "tokens must be non-negative integers");
        data[position - 1] = (uint64_t)token;
        lua_pop(state, 1);
    }
    *out_length = (size_t)length;
    return data;
}

static LuaTransducer* transducer(lua_State* state, int index) {
    LuaTransducer* value = (LuaTransducer*)luaL_checkudata(state, index, TRANSDUCER_MT);
    luaL_argcheck(state, value->value != NULL, index, "transducer is closed");
    return value;
}
static LuaCursor* cursor(lua_State* state, int index) {
    LuaCursor* value = (LuaCursor*)luaL_checkudata(state, index, CURSOR_MT);
    luaL_argcheck(state, value->value != NULL, index, "cursor is closed");
    return value;
}
static LuaPattern* pattern(lua_State* state, int index) {
    LuaPattern* value = (LuaPattern*)luaL_checkudata(state, index, PATTERN_MT);
    luaL_argcheck(state, value->value != NULL, index, "pattern is closed");
    return value;
}

static int transducer_new(lua_State* state) {
    VtLuaDictionaryResource* dictionary =
        (VtLuaDictionaryResource*)luaL_checkudata(state, 1, VT_LUA_DICTIONARY_METATABLE);
    luaL_argcheck(state, dictionary->magic == VT_LUA_DICTIONARY_MAGIC && !dictionary->closed,
                  1, "invalid or closed dictionary resource");
    LlevTransducer* output = NULL;
    LlevStatus status = llev_transducer_new(&dictionary->resource, algorithm(state, 2), &output);
    if (status != LLEV_STATUS_OK) return fail(state, status);
    LuaTransducer* value = (LuaTransducer*)lua_newuserdatauv(state, sizeof(*value), 0);
    value->value = output;
    luaL_setmetatable(state, TRANSDUCER_MT);
    return 1;
}

static int transducer_close(lua_State* state) {
    LuaTransducer* value = (LuaTransducer*)luaL_checkudata(state, 1, TRANSDUCER_MT);
    if (value->value) {
        llev_transducer_free(value->value);
        value->value = NULL;
    }
    return 0;
}

static int cursor_close(lua_State* state) {
    LuaCursor* value = (LuaCursor*)luaL_checkudata(state, 1, CURSOR_MT);
    if (value->value) {
        if (value->leased) {
            (void)llev_query_cursor_release_batch(value->value, value->batch.generation);
            value->leased = 0;
        }
        LlevStatus status = llev_query_cursor_free(value->value);
        if (status != LLEV_STATUS_OK) return fail(state, status);
        value->value = NULL;
    }
    return 0;
}

static int push_cursor(lua_State* state, LlevQueryCursor* raw) {
    LuaCursor* value = (LuaCursor*)lua_newuserdatauv(state, sizeof(*value), 0);
    memset(value, 0, sizeof(*value));
    value->value = raw;
    luaL_setmetatable(state, CURSOR_MT);
    return 1;
}

static int query(lua_State* state) {
    LuaTransducer* value = transducer(state, 1);
    size_t length = 0;
    const char* input = luaL_checklstring(state, 2, &length);
    size_t maximum = maximum_distance(state, 3);
    LlevQueryCursor* output = NULL;
    LlevStatus status = llev_transducer_query_utf8(
        value->value, input, length, maximum, order(state, 4), &output);
    return status == LLEV_STATUS_OK ? push_cursor(state, output) : fail(state, status);
}

static int query_bytes(lua_State* state) {
    LuaTransducer* value = transducer(state, 1);
    size_t length = 0;
    const char* input = luaL_checklstring(state, 2, &length);
    LlevQueryCursor* output = NULL;
    LlevStatus status = llev_transducer_query_bytes(
        value->value, (const uint8_t*)input, length, maximum_distance(state, 3),
        order(state, 4), &output);
    return status == LLEV_STATUS_OK ? push_cursor(state, output) : fail(state, status);
}

static int query_u64(lua_State* state) {
    LuaTransducer* value = transducer(state, 1);
    size_t maximum = maximum_distance(state, 3);
    uint32_t selected_order = order(state, 4);
    size_t length = 0;
    const uint64_t* input = u64_sequence(state, 2, &length);
    LlevQueryCursor* output = NULL;
    LlevStatus status = llev_transducer_query_u64(
        value->value, input, length, maximum, selected_order, &output);
    return status == LLEV_STATUS_OK ? push_cursor(state, output) : fail(state, status);
}

static int transducer_domain(lua_State* state) {
    VtUnitDomain output = VT_UNIT_DOMAIN_BYTE;
    LlevStatus status = llev_transducer_unit_domain(transducer(state, 1)->value, &output);
    if (status != LLEV_STATUS_OK) return fail(state, status);
    if (output == VT_UNIT_DOMAIN_BYTE) lua_pushliteral(state, "byte");
    else if (output == VT_UNIT_DOMAIN_UNICODE_SCALAR) lua_pushliteral(state, "unicode");
    else lua_pushliteral(state, "u64");
    return 1;
}

static int query_pattern(lua_State* state) {
    LuaTransducer* value = transducer(state, 1);
    LuaPattern* selected = pattern(state, 2);
    lua_Integer maximum = luaL_checkinteger(state, 3);
    luaL_argcheck(state, maximum >= 0 && maximum <= UINT8_MAX, 3, "distance is outside u8");
    LlevQueryCursor* output = NULL;
    LlevStatus status = llev_transducer_query_pattern(
        value->value, selected->value, (uint8_t)maximum, &output);
    return status == LLEV_STATUS_OK ? push_cursor(state, output) : fail(state, status);
}

static void push_match(lua_State* state, const LlevMatch* value) {
    lua_createtable(state, 0, 3);
    if (value->unit_domain == VT_UNIT_DOMAIN_U64) {
        const uint64_t* tokens = (const uint64_t*)value->term_data;
        lua_createtable(state, (int)value->term_len, 0);
        for (size_t index = 0; index < value->term_len; ++index) {
            lua_pushinteger(state, (lua_Integer)tokens[index]);
            lua_rawseti(state, -2, (lua_Integer)index + 1);
        }
    } else {
        lua_pushlstring(state, (const char*)value->term_data, value->byte_len);
    }
    lua_setfield(state, -2, "term");
    lua_pushinteger(state, (lua_Integer)value->distance); lua_setfield(state, -2, "distance");
    if (value->has_id) { lua_pushinteger(state, (lua_Integer)value->id); lua_setfield(state, -2, "id"); }
}

static LlevStatus refill(LuaCursor* value, size_t maximum) {
    if (value->leased) {
        LlevStatus release = llev_query_cursor_release_batch(value->value, value->batch.generation);
        if (release != LLEV_STATUS_OK) return release;
        value->leased = 0;
    }
    memset(&value->batch, 0, sizeof(value->batch));
    LlevStatus status = llev_query_cursor_next_batch(value->value, maximum, &value->batch);
    if (status == LLEV_STATUS_OK) {
        value->leased = 1;
        value->index = 0;
    }
    return status;
}

static int cursor_next(lua_State* state) {
    LuaCursor* value = cursor(state, 1);
    if (!value->leased || value->index == value->batch.len) {
        LlevStatus status = refill(value, LLEV_DEFAULT_MATCH_BATCH);
        if (status == LLEV_STATUS_END || (status == LLEV_STATUS_OK && value->batch.len == 0)) return 0;
        if (status != LLEV_STATUS_OK) return fail(state, status);
    }
    push_match(state, &value->batch.matches[value->index++]);
    return 1;
}

static int cursor_next_batch(lua_State* state) {
    LuaCursor* value = cursor(state, 1);
    size_t maximum = (size_t)luaL_optinteger(state, 2, LLEV_DEFAULT_MATCH_BATCH);
    luaL_argcheck(state, maximum > 0, 2, "batch size must be positive");
    LlevStatus status = refill(value, maximum);
    if (status == LLEV_STATUS_END || (status == LLEV_STATUS_OK && value->batch.len == 0)) {
        lua_pushnil(state);
        return 1;
    }
    if (status != LLEV_STATUS_OK) return fail(state, status);
    lua_createtable(state, (int)value->batch.len, 0);
    for (size_t index = 0; index < value->batch.len; ++index) {
        push_match(state, &value->batch.matches[index]);
        lua_rawseti(state, -2, (lua_Integer)index + 1);
    }
    value->index = value->batch.len;
    return 1;
}

static int reduce_batches(lua_State* state) {
    LuaCursor* value = cursor(state, 1);
    luaL_checkany(state, 2);
    luaL_checktype(state, 3, LUA_TFUNCTION);
    size_t maximum = (size_t)luaL_optinteger(state, 4, LLEV_DEFAULT_MATCH_BATCH);
    luaL_argcheck(state, maximum > 0, 4, "batch size must be positive");
    for (;;) {
        LlevStatus status = refill(value, maximum);
        if (status == LLEV_STATUS_END || (status == LLEV_STATUS_OK && value->batch.len == 0)) break;
        if (status != LLEV_STATUS_OK) return fail(state, status);
        lua_pushvalue(state, 3);
        lua_pushvalue(state, 2);
        lua_createtable(state, (int)value->batch.len, 0);
        for (size_t index = 0; index < value->batch.len; ++index) {
            push_match(state, &value->batch.matches[index]);
            lua_rawseti(state, -2, (lua_Integer)index + 1);
        }
        lua_call(state, 2, 1);
        lua_replace(state, 2);
    }
    lua_pushvalue(state, 2);
    return 1;
}

static int compile_pattern(lua_State* state, int llre) {
    size_t length = 0;
    const char* source = luaL_checklstring(state, 1, &length);
    LlevPhoneticPattern* output = NULL;
    LlevStatus status = llre
        ? llev_phonetic_pattern_compile_llre(source, length, &output)
        : llev_phonetic_pattern_compile_regex(source, length, &output);
    if (status != LLEV_STATUS_OK) return fail(state, status);
    LuaPattern* value = (LuaPattern*)lua_newuserdatauv(state, sizeof(*value), 0);
    value->value = output;
    luaL_setmetatable(state, PATTERN_MT);
    return 1;
}
static int regex(lua_State* state) { return compile_pattern(state, 0); }
static int llre(lua_State* state) { return compile_pattern(state, 1); }
static int pattern_matches(lua_State* state) {
    LuaPattern* value = pattern(state, 1);
    size_t length = 0;
    const char* input = luaL_checklstring(state, 2, &length);
    uint8_t output = 0;
    LlevStatus status = llev_phonetic_pattern_matches(value->value, input, length, &output);
    if (status != LLEV_STATUS_OK) return fail(state, status);
    lua_pushboolean(state, output);
    return 1;
}
static int pattern_size(lua_State* state) {
    LuaPattern* value = pattern(state, 1);
    size_t states = 0, transitions = 0;
    LlevStatus status = llev_phonetic_pattern_size(value->value, &states, &transitions);
    if (status != LLEV_STATUS_OK) return fail(state, status);
    lua_pushinteger(state, (lua_Integer)states);
    lua_pushinteger(state, (lua_Integer)transitions);
    return 2;
}
static int pattern_close(lua_State* state) {
    LuaPattern* value = (LuaPattern*)luaL_checkudata(state, 1, PATTERN_MT);
    if (value->value) { llev_phonetic_pattern_free(value->value); value->value = NULL; }
    return 0;
}

static int rules_new(lua_State* state) {
    size_t length = 0;
    const char* source = luaL_checklstring(state, 1, &length);
    LlevPhoneticRuleSet* output = NULL;
    LlevStatus status;
    if (strcmp(source, "english-orthography") == 0) {
        status = llev_phonetic_rules_builtin(LLEV_PHONETIC_RULE_SET_ENGLISH_ORTHOGRAPHY, &output);
    } else if (strcmp(source, "english-phonetic") == 0) {
        status = llev_phonetic_rules_builtin(LLEV_PHONETIC_RULE_SET_ENGLISH_PHONETIC, &output);
    } else {
        status = llev_phonetic_rules_parse(source, length, &output);
    }
    if (status != LLEV_STATUS_OK) return fail(state, status);
    LuaRules* value = (LuaRules*)lua_newuserdatauv(state, sizeof(*value), 0);
    value->value = output;
    luaL_setmetatable(state, RULES_MT);
    return 1;
}
static int rules_apply(lua_State* state) {
    LuaRules* value = (LuaRules*)luaL_checkudata(state, 1, RULES_MT);
    luaL_argcheck(state, value->value != NULL, 1, "rules are closed");
    size_t length = 0;
    const char* input = luaL_checklstring(state, 2, &length);
    LlevOwnedString output = {0};
    LlevStatus status = llev_phonetic_rules_apply(value->value, input, length, &output);
    if (status != LLEV_STATUS_OK) return fail(state, status);
    lua_pushlstring(state, output.data, output.len);
    llev_owned_string_free(&output);
    return 1;
}
static int rules_length(lua_State* state) {
    LuaRules* value = (LuaRules*)luaL_checkudata(state, 1, RULES_MT);
    luaL_argcheck(state, value->value != NULL, 1, "rules are closed");
    size_t output = 0;
    LlevStatus status = llev_phonetic_rules_len(value->value, &output);
    if (status != LLEV_STATUS_OK) return fail(state, status);
    lua_pushinteger(state, (lua_Integer)output);
    return 1;
}
static int rules_close(lua_State* state) {
    LuaRules* value = (LuaRules*)luaL_checkudata(state, 1, RULES_MT);
    if (value->value) { llev_phonetic_rules_free(value->value); value->value = NULL; }
    return 0;
}

static int distance(lua_State* state) {
    size_t source_length = 0, target_length = 0;
    const char* source = luaL_checklstring(state, 1, &source_length);
    const char* target = luaL_checklstring(state, 2, &target_length);
    lua_pushinteger(state, (lua_Integer)llev_distance(source, source_length, target, target_length));
    return 1;
}

static int distance_threshold(lua_State* state) {
    size_t source_length = 0, target_length = 0;
    const char* source = luaL_checklstring(state, 1, &source_length);
    const char* target = luaL_checklstring(state, 2, &target_length);
    lua_pushinteger(state, (lua_Integer)llev_distance_threshold(
        source, source_length, target, target_length, maximum_distance(state, 3)));
    return 1;
}

static int damerau_distance(lua_State* state) {
    size_t source_length = 0, target_length = 0;
    const char* source = luaL_checklstring(state, 1, &source_length);
    const char* target = luaL_checklstring(state, 2, &target_length);
    lua_pushinteger(state, (lua_Integer)llev_damerau_distance(
        source, source_length, target, target_length));
    return 1;
}

static int damerau_distance_threshold(lua_State* state) {
    size_t source_length = 0, target_length = 0;
    const char* source = luaL_checklstring(state, 1, &source_length);
    const char* target = luaL_checklstring(state, 2, &target_length);
    lua_pushinteger(state, (lua_Integer)llev_damerau_distance_threshold(
        source, source_length, target, target_length, maximum_distance(state, 3)));
    return 1;
}

static int true_damerau_distance(lua_State* state) {
    size_t source_length = 0, target_length = 0;
    const char* source = luaL_checklstring(state, 1, &source_length);
    const char* target = luaL_checklstring(state, 2, &target_length);
    lua_pushinteger(state, (lua_Integer)llev_true_damerau_distance(
        source, source_length, target, target_length));
    return 1;
}

static int true_damerau_distance_threshold(lua_State* state) {
    size_t source_length = 0, target_length = 0;
    const char* source = luaL_checklstring(state, 1, &source_length);
    const char* target = luaL_checklstring(state, 2, &target_length);
    lua_pushinteger(state, (lua_Integer)llev_true_damerau_distance_threshold(
        source, source_length, target, target_length, maximum_distance(state, 3)));
    return 1;
}

static void metatable(lua_State* state, const char* name, lua_CFunction close,
                      const luaL_Reg* methods, lua_CFunction call) {
    luaL_newmetatable(state, name);
    lua_pushcfunction(state, close); lua_setfield(state, -2, "__gc");
    lua_pushcfunction(state, close); lua_setfield(state, -2, "__close");
    if (call) { lua_pushcfunction(state, call); lua_setfield(state, -2, "__call"); }
    lua_newtable(state); luaL_setfuncs(state, methods, 0); lua_setfield(state, -2, "__index");
    lua_pop(state, 1);
}

static void string_constant(lua_State* state, const char* name, const char* value) {
    lua_pushstring(state, value);
    lua_setfield(state, -2, name);
}

static void status_constant(lua_State* state, const char* name, LlevStatus value) {
    lua_pushinteger(state, (lua_Integer)value);
    lua_setfield(state, -2, name);
}

int luaopen_vinary_tree_liblevenshtein(lua_State* state) {
    const luaL_Reg transducer_methods[] = {
        {"query", query}, {"query_bytes", query_bytes}, {"query_u64", query_u64},
        {"query_pattern", query_pattern}, {"domain", transducer_domain},
        {"close", transducer_close}, {NULL, NULL}};
    const luaL_Reg cursor_methods[] = {
        {"next", cursor_next}, {"next_batch", cursor_next_batch},
        {"reduce_batches", reduce_batches}, {"close", cursor_close}, {NULL, NULL}};
    const luaL_Reg pattern_methods[] = {
        {"matches", pattern_matches}, {"size", pattern_size}, {"close", pattern_close}, {NULL, NULL}};
    const luaL_Reg rules_methods[] = {
        {"apply", rules_apply}, {"len", rules_length}, {"close", rules_close}, {NULL, NULL}};
    metatable(state, TRANSDUCER_MT, transducer_close, transducer_methods, NULL);
    metatable(state, CURSOR_MT, cursor_close, cursor_methods, cursor_next);
    metatable(state, PATTERN_MT, pattern_close, pattern_methods, NULL);
    metatable(state, RULES_MT, rules_close, rules_methods, NULL);
    const luaL_Reg functions[] = {
        {"transducer", transducer_new}, {"phonetic_pattern", regex}, {"llre_pattern", llre},
        {"phonetic_rules", rules_new}, {"distance", distance},
        {"distance_threshold", distance_threshold}, {"damerau_distance", damerau_distance},
        {"damerau_distance_threshold", damerau_distance_threshold},
        {"true_damerau_distance", true_damerau_distance},
        {"true_damerau_distance_threshold", true_damerau_distance_threshold},
        {NULL, NULL}
    };
    luaL_newlib(state, functions);

    lua_createtable(state, 0, 4);
    string_constant(state, "standard", "standard");
    string_constant(state, "transposition", "transposition");
    string_constant(state, "merge_and_split", "merge-and-split");
    string_constant(state, "damerau_levenshtein", "damerau-levenshtein");
    lua_setfield(state, -2, "algorithm");

    lua_createtable(state, 0, 2);
    string_constant(state, "traversal", "traversal");
    string_constant(state, "distance_then_term", "distance-then-term");
    lua_setfield(state, -2, "order");

    lua_createtable(state, 0, 2);
    string_constant(state, "english_orthography", "english-orthography");
    string_constant(state, "english_phonetic", "english-phonetic");
    lua_setfield(state, -2, "phonetic_rule_set_kind");

    lua_createtable(state, 0, 13);
    status_constant(state, "ok", LLEV_STATUS_OK);
    status_constant(state, "end_of_stream", LLEV_STATUS_END);
    status_constant(state, "invalid_argument", LLEV_STATUS_INVALID_ARGUMENT);
    status_constant(state, "invalid_utf8", LLEV_STATUS_INVALID_UTF8);
    status_constant(state, "null_pointer", LLEV_STATUS_NULL_POINTER);
    status_constant(state, "panic", LLEV_STATUS_PANIC);
    status_constant(state, "unsupported", LLEV_STATUS_UNSUPPORTED);
    status_constant(state, "io_error", LLEV_STATUS_IO_ERROR);
    status_constant(state, "closed", LLEV_STATUS_CLOSED);
    status_constant(state, "limit_exceeded", LLEV_STATUS_LIMIT_EXCEEDED);
    status_constant(state, "provider_error", LLEV_STATUS_PROVIDER_ERROR);
    status_constant(state, "batch_in_use", LLEV_STATUS_BATCH_IN_USE);
    status_constant(state, "domain_mismatch", LLEV_STATUS_DOMAIN_MISMATCH);
    lua_setfield(state, -2, "status");

    return 1;
}
