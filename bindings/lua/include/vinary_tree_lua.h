#ifndef VINARY_TREE_LUA_H
#define VINARY_TREE_LUA_H

#include <lua.h>
#include <lauxlib.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>
#include "vinary_tree_interop.h"

#define VT_LUA_DICTIONARY_MAGIC UINT64_C(0x56544c5541444943)
#define VT_LUA_DICTIONARY_METATABLE "vinary-tree.dictionary.v1"

typedef struct VtLuaDictionaryResource {
    uint64_t magic;
    VtResource resource;
    uint32_t unit_domain;
    uint8_t closed;
} VtLuaDictionaryResource;

/* Lua integers are signed; decimal strings preserve the upper half of u64. */
static inline uint64_t vt_lua_check_u64(
    lua_State* state, int index, const char* label) {
    if (lua_isinteger(state, index)) {
        lua_Integer value = lua_tointeger(state, index);
        luaL_argcheck(state, value >= 0, index, label);
        return (uint64_t)value;
    }
    size_t length = 0;
    const char* decimal = luaL_checklstring(state, index, &length);
    luaL_argcheck(state, length != 0, index, label);
    uint64_t value = 0;
    for (size_t position = 0; position < length; ++position) {
        unsigned char digit = (unsigned char)decimal[position];
        luaL_argcheck(state, digit >= '0' && digit <= '9', index, label);
        digit -= '0';
        luaL_argcheck(
            state, value <= (UINT64_MAX - digit) / 10, index, label);
        value = value * 10 + digit;
    }
    return value;
}

static inline void vt_lua_push_u64(lua_State* state, uint64_t value) {
    if (value <= (uint64_t)LUA_MAXINTEGER) {
        lua_pushinteger(state, (lua_Integer)value);
        return;
    }
    char decimal[32];
    int length = snprintf(decimal, sizeof(decimal), "%" PRIu64, value);
    if (length < 0 || (size_t)length >= sizeof(decimal))
        luaL_error(state, "failed to format unsigned integer");
    lua_pushlstring(state, decimal, (size_t)length);
}

#endif
