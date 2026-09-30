using Liblevenshtein
using Statistics

const LL = Liblevenshtein

function sample(operation; warmup=100, iterations=200, samples=7)
    for _ in 1:warmup
        operation()
    end
    measurements = Float64[]
    for _ in 1:samples
        started = time_ns()
        for _ in 1:iterations
            operation()
        end
        push!(measurements, (time_ns() - started) / iterations)
    end
    median(measurements)
end

function native_controls(path)
    controls = Dict{String,Float64}()
    for line in split(read(Cmd([path, "--bench"]), String), '\n'; keepempty=false)
        tag, name, elapsed = split(line, '\t')
        tag == "B" || error("invalid native benchmark record")
        controls[name] = parse(Float64, elapsed)
    end
    controls
end

function main()
    control = get(ENV, "LIBLEVENSHTEIN_RUST_CONTROL", "")
    isempty(control) && error("set LIBLEVENSHTEIN_RUST_CONTROL to the built Rust example")
    native = native_controls(control)
    operations = LL.GeneralizedOperationSet(
        LL.GeneralizedOperation(0, 1, 1, :insert),
        LL.GeneralizedOperation(1, 0, 1, :delete),
        LL.GeneralizedOperation(1, 1, 0, :equal;
            applicability=LL.APPLICABILITY_EQUAL),
        LL.GeneralizedOperation(1, 1, 0.5, :substitute),
    )
    generalized = LL.GeneralizedAutomaton(2, operations)
    universal = LL.UniversalAutomaton(2)
    source = repeat("a", 32)
    complete = source
    early = repeat("z", 32)
    late = repeat("a", 27) * "zzzzz"
    targets = [iseven(index) ? complete : early for index in 0:31]
    for automaton in (generalized, universal)
        @assert LL.accepts(automaton, source, complete)
        @assert !LL.accepts(automaton, source, early)
        @assert !LL.accepts(automaton, source, late)
    end

    function traverse(automaton, target)
        state = LL.online(automaton, source)
        try
            for unit in target
                LL.advance!(state, unit)
            end
            LL.observation(state)
        finally
            close(state)
        end
    end

    scenarios = [
        ("generalized_construction", () -> close(LL.GeneralizedAutomaton(2, operations))),
        ("universal_construction", () -> close(LL.UniversalAutomaton(2))),
        ("generalized_complete", () -> LL.evaluate(generalized, source, complete)),
        ("universal_complete", () -> LL.evaluate(universal, source, complete)),
        ("generalized_early_reject", () -> LL.evaluate(generalized, source, early)),
        ("universal_early_reject", () -> LL.evaluate(universal, source, early)),
        ("generalized_late_reject", () -> LL.evaluate(generalized, source, late)),
        ("universal_late_reject", () -> LL.evaluate(universal, source, late)),
        ("generalized_traversal_32", () -> traverse(generalized, complete)),
        ("universal_traversal_32", () -> traverse(universal, complete)),
        ("generalized_batch_32", () -> foreach(target -> LL.evaluate(generalized, source, target), targets)),
        ("universal_batch_32", () -> foreach(target -> LL.evaluate(universal, source, target), targets)),
    ]

    failures = String[]
    try
        println("scenario\tnative_ns\tjulia_ns\tbudget_ns\tstatus")
        for (name, operation) in scenarios
            measured = sample(operation)
            reference = native[name]
            # The additive allowance covers dynamic-language dispatch and FFI
            # overhead on very small native operations. A tenfold native-work
            # multiplier catches large algorithmic or marshalling regressions.
            allowance = endswith(name, "batch_32") ? 500_000.0 : 50_000.0
            budget = reference * 10 + allowance
            status = measured <= budget ? "PASS" : "FAIL"
            println("$name\t$(round(reference; digits=1))\t$(round(measured; digits=1))\t$(round(budget; digits=1))\t$status")
            status == "PASS" || push!(failures, name)
        end
    finally
        close(generalized)
        close(universal)
    end
    isempty(failures) || error("automata regression budget exceeded: $(join(failures, ", "))")
end

main()
