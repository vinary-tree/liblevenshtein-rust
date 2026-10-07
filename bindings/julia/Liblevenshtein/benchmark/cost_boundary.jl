using Liblevenshtein
using Libdictenstein
using Statistics

const LL = Liblevenshtein

function median_ns(operation; warmup=100, iterations=200, blocks=7)
    for _ in 1:warmup
        operation()
    end
    samples = Float64[]
    for _ in 1:blocks
        start = time_ns()
        for _ in 1:iterations
            operation()
        end
        push!(samples, (time_ns() - start) / iterations)
    end
    median(samples)
end

function direct_count(transducer, input, maximum_cost, costs)
    symbol = costs isa LL.AffineGapCosts ? :llev_transducer_query_affine :
        :llev_transducer_query_weighted
    raw = Ref(costs isa LL.AffineGapCosts ? LL.RawAffineCosts(costs) :
        LL.RawOperationCostsF64(costs))
    units, domain = if input isa AbstractString
        LL.text_bytes(input), UInt32(Libdictenstein.UNIT_UNICODE_SCALAR)
    elseif input isa AbstractVector{UInt8}
        LL.ffi_units(input), UInt32(Libdictenstein.UNIT_BYTE)
    else
        LL.ffi_units(input), UInt32(Libdictenstein.UNIT_U64)
    end
    cursor = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve units raw ccall(LL.native(symbol), Cint,
        (Ptr{Cvoid}, UInt32, Ptr{Cvoid}, Csize_t, Cdouble,
            Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
        LL.require_open(transducer), domain,
        isempty(units) ? C_NULL : pointer(units), length(units),
        maximum_cost,
        Ptr{Cvoid}(Base.unsafe_convert(Ptr{typeof(raw[])}, raw)), cursor)
    LL.checked(status, symbol)
    count = 0
    try
        while true
            view = Ref(LL.RawCostBatch(Ptr{LL.RawCostMatch}(C_NULL), 0, 0))
            status = ccall(LL.native(:llev_cost_cursor_next_batch), Cint,
                (Ptr{Cvoid}, Csize_t, Ref{LL.RawCostBatch}),
                cursor[], LL.DEFAULT_MATCH_BATCH, view)
            status == Cint(LL.STATUS_END) && break
            LL.checked(status, :llev_cost_cursor_next_batch)
            count += Int(view[].len)
            LL.checked(ccall(LL.native(:llev_cost_cursor_release_batch), Cint,
                (Ptr{Cvoid}, UInt64), cursor[], view[].generation),
                :llev_cost_cursor_release_batch)
        end
    finally
        LL.checked(ccall(LL.native(:llev_cost_cursor_free), Cint,
            (Ptr{Cvoid},), cursor[]), :llev_cost_cursor_free)
    end
    count
end

function run_scenario(name, domain, input, terms, affine, weighted)
    dictionary = Libdictenstein.DynamicDawg(domain)
    source = nothing
    transducer = nothing
    try
        for (index, term) in enumerate(terms)
            dictionary[term] = UInt64(index)
        end
        source = Libdictenstein.snapshot(dictionary)
        transducer = LL.Transducer(source)
        for (label, costs, query) in (
            ("affine", affine, LL.query_affine),
            ("weighted", weighted, LL.query_weighted),
        )
            direct = () -> direct_count(transducer, input, 2.0, costs)
            managed = () -> length(collect(query(transducer, input, 2.0, costs)))
            @assert direct() == managed()
            native_ns = median_ns(direct)
            julia_ns = median_ns(managed)
            # Includes materializing owned Julia keys, unlike the direct count.
            budget_ns = native_ns * 10 + 50_000
            status = julia_ns <= budget_ns ? "PASS" : "FAIL"
            println("$name/$label\t$(round(native_ns; digits=1))\t" *
                "$(round(julia_ns; digits=1))\t$(round(budget_ns; digits=1))\t$status")
            status == "PASS" || error("cost-boundary budget exceeded for $name/$label")
        end
    finally
        transducer === nothing || close(transducer)
        source === nothing || close(source)
        close(dictionary)
    end
end

function main()
    affine = LL.AffineGapCosts(0.5, 0.25, 1.0)
    weighted = LL.WeightedOperationCosts(:typo)
    println("scenario\tnative_ns\tjulia_ns\tbudget_ns\tstatus")
    run_scenario("unicode", Libdictenstein.UNIT_UNICODE_SCALAR, "cat",
        ["cat", "cut", "cot", "cart", "dog"], affine, weighted)
    run_scenario("bytes", Libdictenstein.UNIT_BYTE, UInt8[0, 255],
        [UInt8[0, 255], UInt8[0, 254], UInt8[1, 255], UInt8[2, 2]],
        affine, weighted)
    run_scenario("u64", Libdictenstein.UNIT_U64, UInt64[10, 20],
        [UInt64[10, 20], UInt64[10, 30], UInt64[10], UInt64[99, 99]],
        affine, weighted)
end

main()
