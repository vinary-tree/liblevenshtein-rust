#ifndef VINARY_TREE_OCAML_H
#define VINARY_TREE_OCAML_H

#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>
#include "vinary_tree_interop.h"

typedef struct VtOcamlResource {
    VtResource value;
    int closed;
} VtOcamlResource;

static void vt_ocaml_resource_finalize(value block) {
    VtOcamlResource* resource = (VtOcamlResource*)Data_custom_val(block);
    if (!resource->closed && resource->value.context && resource->value.vtable) {
        resource->value.vtable->release(resource->value.context);
        resource->closed = 1;
    }
}

static struct custom_operations vt_ocaml_resource_operations = {
    "io.vinarytree.resource.v1",
    vt_ocaml_resource_finalize,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default
};

static inline value vt_ocaml_copy_resource(const VtResource* source) {
    if (!source || !source->context || !source->vtable ||
        !source->vtable->retain || !source->vtable->release) {
        caml_invalid_argument("invalid vinary-tree resource");
    }
    source->vtable->retain(source->context);
    value block = caml_alloc_custom(&vt_ocaml_resource_operations,
                                    sizeof(VtOcamlResource), 0, 1);
    VtOcamlResource* target = (VtOcamlResource*)Data_custom_val(block);
    target->value = *source;
    target->closed = 0;
    return block;
}

static inline const VtResource* vt_ocaml_get_resource(value block) {
    VtOcamlResource* resource = (VtOcamlResource*)Data_custom_val(block);
    if (resource->closed) caml_invalid_argument("vinary-tree resource is closed");
    return &resource->value;
}

#endif
