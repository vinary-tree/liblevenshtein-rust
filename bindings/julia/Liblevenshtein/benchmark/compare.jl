using Liblevenshtein
using Statistics

function sample(operation; warmup=10_000, iterations=100_000, samples=9)
    for _ in 1:warmup
        operation()
    end
    values = Float64[]
    for _ in 1:samples
        started = time_ns()
        for _ in 1:iterations
            operation()
        end
        push!(values, (time_ns() - started) / iterations)
    end
    (minimum=minimum(values), median=median(values), maximum=maximum(values))
end

println("standard distance ns/op: ",
    sample(() -> distance("levenshtein", "liblevenshtein")))
println("thresholded distance ns/op: ",
    sample(() -> distance("levenshtein", "liblevenshtein"; threshold=4)))
