include_guard(GLOBAL)

get_filename_component(_VT_INTEROP_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

if(NOT TARGET vinary-tree::interop)
  add_library(vinary-tree::interop INTERFACE IMPORTED)
  set_target_properties(vinary-tree::interop PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_VT_INTEROP_PREFIX}/include"
  )
endif()

set(vinary-tree-interop_FOUND TRUE)
unset(_VT_INTEROP_PREFIX)
