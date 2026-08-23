#ifndef CLIBLEVENSHTEIN_SWIFT_H
#define CLIBLEVENSHTEIN_SWIFT_H

/*
 * SwiftPM imports this system-library module in isolation, so the Swift
 * VinaryTreeInterop target dependency does not contribute a C header search
 * path here. Point liblevenshtein_abi.h at the repository's governed interop
 * mirror before including the canonical public header.
 */
#ifndef VT_INTEROP_HEADER
#define VT_INTEROP_HEADER "../bindings/swift/liblevenshtein/Sources/CLiblevenshtein/vinary_tree_interop.h"
#endif

#include "../../../../../include/liblevenshtein.h"

#endif
