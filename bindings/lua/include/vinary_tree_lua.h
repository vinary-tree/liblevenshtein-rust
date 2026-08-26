#ifndef VINARY_TREE_LUA_H
#define VINARY_TREE_LUA_H

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

#endif
