#include <stdlib.h>
#include "vinary_tree_interop.h"

void vt_hs_resource_free(VtResource* resource) {
    if (!resource) return;
    if (resource->context && resource->vtable && resource->vtable->release)
        resource->vtable->release(resource->context);
    free(resource);
}
