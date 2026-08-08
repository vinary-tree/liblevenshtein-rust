include_guard(GLOBAL)

include(CMakeFindDependencyMacro)
find_dependency(Threads)
find_dependency(vinary-tree-interop 0.1 CONFIG)

get_filename_component(_LLEVENSHTEIN_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

if(NOT TARGET liblevenshtein::shared)
  add_library(liblevenshtein::shared SHARED IMPORTED)
  set_target_properties(liblevenshtein::shared PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_LLEVENSHTEIN_PREFIX}/include"
    INTERFACE_LINK_LIBRARIES "vinary-tree::interop"
  )

  if(WIN32)
    set_target_properties(liblevenshtein::shared PROPERTIES
      IMPORTED_LOCATION "${_LLEVENSHTEIN_PREFIX}/bin/liblevenshtein.dll"
      IMPORTED_IMPLIB "${_LLEVENSHTEIN_PREFIX}/lib/liblevenshtein.dll.lib"
    )
  elseif(APPLE)
    set_target_properties(liblevenshtein::shared PROPERTIES
      IMPORTED_LOCATION "${_LLEVENSHTEIN_PREFIX}/lib/libliblevenshtein.dylib"
    )
  else()
    set_target_properties(liblevenshtein::shared PROPERTIES
      IMPORTED_LOCATION "${_LLEVENSHTEIN_PREFIX}/lib/libliblevenshtein.so"
    )
  endif()
endif()

if(NOT TARGET liblevenshtein::static)
  add_library(liblevenshtein::static STATIC IMPORTED)
  set_target_properties(liblevenshtein::static PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_LLEVENSHTEIN_PREFIX}/include"
    INTERFACE_COMPILE_DEFINITIONS LLEV_STATIC
    INTERFACE_LINK_LIBRARIES "vinary-tree::interop"
  )

  if(WIN32)
    set_target_properties(liblevenshtein::static PROPERTIES
      IMPORTED_LOCATION "${_LLEVENSHTEIN_PREFIX}/lib/liblevenshtein.lib"
      INTERFACE_LINK_LIBRARIES "bcrypt;userenv;ws2_32;ntdll;synchronization;advapi32;Threads::Threads"
    )
  elseif(APPLE)
    find_library(_LLEVENSHTEIN_ICONV_LIBRARY NAMES iconv REQUIRED)
    find_library(_LLEVENSHTEIN_COREFOUNDATION_FRAMEWORK
      NAMES CoreFoundation REQUIRED)
    find_library(_LLEVENSHTEIN_SECURITY_FRAMEWORK NAMES Security REQUIRED)
    set_target_properties(liblevenshtein::static PROPERTIES
      IMPORTED_LOCATION "${_LLEVENSHTEIN_PREFIX}/lib/libliblevenshtein.a"
      INTERFACE_LINK_LIBRARIES
        "${CMAKE_DL_LIBS};Threads::Threads;m;${_LLEVENSHTEIN_ICONV_LIBRARY};${_LLEVENSHTEIN_COREFOUNDATION_FRAMEWORK};${_LLEVENSHTEIN_SECURITY_FRAMEWORK}"
    )
    unset(_LLEVENSHTEIN_ICONV_LIBRARY CACHE)
    unset(_LLEVENSHTEIN_COREFOUNDATION_FRAMEWORK CACHE)
    unset(_LLEVENSHTEIN_SECURITY_FRAMEWORK CACHE)
  else()
    set_target_properties(liblevenshtein::static PROPERTIES
      IMPORTED_LOCATION "${_LLEVENSHTEIN_PREFIX}/lib/libliblevenshtein.a"
      INTERFACE_LINK_LIBRARIES "${CMAKE_DL_LIBS};Threads::Threads;m"
    )
  endif()
endif()

if(NOT DEFINED LIBLEVENSHTEIN_LINKAGE)
  set(LIBLEVENSHTEIN_LINKAGE "SHARED")
endif()
string(TOUPPER "${LIBLEVENSHTEIN_LINKAGE}" _LLEVENSHTEIN_LINKAGE)
if(NOT _LLEVENSHTEIN_LINKAGE STREQUAL "SHARED" AND
   NOT _LLEVENSHTEIN_LINKAGE STREQUAL "STATIC")
  message(FATAL_ERROR "LIBLEVENSHTEIN_LINKAGE must be SHARED or STATIC")
endif()

if(NOT TARGET liblevenshtein::liblevenshtein)
  add_library(liblevenshtein::liblevenshtein INTERFACE IMPORTED)
  if(_LLEVENSHTEIN_LINKAGE STREQUAL "STATIC")
    set_property(TARGET liblevenshtein::liblevenshtein PROPERTY
      INTERFACE_LINK_LIBRARIES liblevenshtein::static)
  else()
    set_property(TARGET liblevenshtein::liblevenshtein PROPERTY
      INTERFACE_LINK_LIBRARIES liblevenshtein::shared)
  endif()
endif()

set(liblevenshtein_FOUND TRUE)
unset(_LLEVENSHTEIN_LINKAGE)
unset(_LLEVENSHTEIN_PREFIX)
