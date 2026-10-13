using Liblevenshtein
using Libdictenstein
using Statistics

const LL = Liblevenshtein

function median_ns(operation; warmup=1000, iterations=5000, blocks=7)
    for _ in 1:warmup
        operation()
    end
    samples = Float64[]
    for _ in 1:blocks
        started = time_ns()
        for _ in 1:iterations
            operation()
        end
        push!(samples, (time_ns() - started) / iterations)
    end
    median(samples)
end

function main()
    dictionary = Libdictenstein.DynamicDawg()
    source = nothing
    transducer = nothing
    cache = nothing
    try
        for term in ("cat", "cot", "cut", "cart", "scat")
            dictionary[term] = nothing
        end
        source = Libdictenstein.snapshot(dictionary)
        transducer = LL.Transducer(source)
        cache = LL.QueryCache(transducer; max_entries=16, max_weight=1 << 20,
            seed=0x1234, sketch_width=128, age_after=160)
        query_bytes = Vector{UInt8}(codeunits("cat"))
        native_hit = () -> begin
            cursor = Ref{Ptr{Cvoid}}(C_NULL)
            status = GC.@preserve query_bytes ccall(
                LL.native(:llev_query_cache_query_utf8), Cint,
                (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Csize_t, UInt32, Ref{Ptr{Cvoid}}),
                LL.require_open(cache), pointer(query_bytes), length(query_bytes), 1,
                UInt32(LL.ORDER_TRAVERSAL), cursor)
            LL.checked(status, :llev_query_cache_query_utf8)
            ccall(LL.native(:llev_query_cursor_free), Cint,
                (Ptr{Cvoid},), cursor[])
        end
        julia_hit = () -> close(LL.query(cache, "cat", 1))
        native_hit()
        julia_hit()
        @assert LL.cache_stats(cache).hits >= 1
        println("scenario\tnative_ffi_ns\tjulia_facade_ns\tboundary_delta_ns")
        native_ns = median_ns(native_hit)
        julia_ns = median_ns(julia_hit)
        println("configured_cache_hit\t$(round(native_ns; digits=1))\t" *
            "$(round(julia_ns; digits=1))\t$(round(julia_ns - native_ns; digits=1))")
    finally
        cache === nothing || close(cache)
        transducer === nothing || close(transducer)
        source === nothing || close(source)
        close(dictionary)
    end
end

main()
