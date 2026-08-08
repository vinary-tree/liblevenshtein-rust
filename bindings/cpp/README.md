# C and C++ package

Release archives contain the stable C17 header, the header-only C++20 RAII
facade, shared and static native libraries, a relocatable CMake config package,
and `pkg-config` metadata. C23 and C++23 consumers are compile-checked as well.

```cmake
find_package(vinary-tree-interop CONFIG REQUIRED)
find_package(liblevenshtein CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE liblevenshtein::liblevenshtein)
```

`liblevenshtein::liblevenshtein` selects the shared library by default. Set
`LIBLEVENSHTEIN_LINKAGE=STATIC` before `find_package` for a fully static native
link, or name `liblevenshtein::shared` / `liblevenshtein::static` explicitly.
The static target propagates its platform system libraries. The equivalent
command-line interface is `pkg-config liblevenshtein` for shared linking and
`pkg-config --static liblevenshtein` for static linking.

The shared form must remain available to the process at runtime. The static
form has no runtime dependency on `liblevenshtein`; only ordinary operating
system libraries remain.

Construct dictionaries through libdictenstein (or implement
`vt.dictionary.v1` as a host provider), then pass the two-word `VtResource` to
the liblevenshtein transducer. The cursor retains its query-start revision and
returns leased spans backed by contiguous term arenas; release each batch before
advancing or destroying the cursor.
