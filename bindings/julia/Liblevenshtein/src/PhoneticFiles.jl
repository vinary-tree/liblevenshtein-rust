function loaded_phonetic_file(path::AbstractString, search_paths,
    max_total_path_bytes::Integer, symbol::Symbol, constructor)
    max_total_path_bytes > 0 ||
        throw(ArgumentError("max_total_path_bytes must be positive"))
    paths = Vector{UInt8}[]
    for search_path in search_paths
        search_path isa AbstractString ||
            throw(ArgumentError("search_paths must contain strings"))
        push!(paths, text_bytes(search_path))
    end
    length(paths) <= 64 || throw(ArgumentError("at most 64 search paths are allowed"))
    path_bytes = text_bytes(path)
    views = RawUtf8View[RawUtf8View(isempty(bytes) ? C_NULL : pointer(bytes),
        length(bytes)) for bytes in paths]
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve path_bytes paths views checked(ccall(native(symbol), Cint,
        (Ptr{UInt8}, Csize_t, Ptr{RawUtf8View}, Csize_t, Csize_t,
            Ref{Ptr{Cvoid}}),
        isempty(path_bytes) ? C_NULL : pointer(path_bytes), length(path_bytes),
        isempty(views) ? C_NULL : pointer(views), length(views),
        checked_csize(max_total_path_bytes, "max_total_path_bytes"), output), symbol)
    value = constructor(output[], false)
    finalizer(close!, value)
    value
end

"""Load a trusted local `.llev` file with native `@include` resolution.

`search_paths` are extra UTF-8 directories. The path ceiling bounds path
arguments, not the contents of transitively included files. Use this only for
trusted local files; use `PhoneticRuleSet(source)` for untrusted inline text.
"""
function load_phonetic_rules(path::AbstractString;
    search_paths=String[], max_total_path_bytes::Integer=4_096)
    loaded_phonetic_file(path, search_paths, max_total_path_bytes,
        :llev_phonetic_rules_load_file, PhoneticRuleSet)
end

"""Load a trusted local `.llre` file with native `@import` resolution.

Compilation uses the same NFA state ceiling as inline pattern constructors.
The path ceiling does not bound imported file contents.
"""
function load_phonetic_pattern(path::AbstractString;
    search_paths=String[], max_total_path_bytes::Integer=4_096)
    loaded_phonetic_file(path, search_paths, max_total_path_bytes,
        :llev_phonetic_pattern_load_llre_file, PhoneticPattern)
end
