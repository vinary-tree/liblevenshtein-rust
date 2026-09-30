using Test

function qualification_error(f)
    try
        f()
        nothing
    catch error
        error
    end
end

function generalized_fields(value)
    (value.consumed_target_length, value.active_positions,
        value.scaled_distance, value.current_row_nonempty, value.accepting)
end

function universal_fields(value)
    (value.consumed_target_length, value.source_length, value.alive, value.accepting)
end

function qualification_units(field)
    isempty(field) ? UInt64[] : parse.(UInt64, split(field, ','))
end

qualification_text(field) = join(Char.(UInt32.(qualification_units(field))))

function qualification_operation(id)
    if id == 0
        LL.GeneralizedOperation(0, 1, 1, :insert)
    elseif id == 1
        LL.GeneralizedOperation(1, 0, 1, :delete)
    elseif id == 2
        LL.GeneralizedOperation(1, 1, 0, :equal;
            applicability=LL.APPLICABILITY_EQUAL)
    elseif id == 3
        LL.GeneralizedOperation(1, 1, 0.5, :substitute)
    elseif id == 4
        LL.GeneralizedOperation(2, 2, 0.75, :transpose;
            applicability=LL.APPLICABILITY_ADJACENT_TRANSPOSE)
    elseif id == 5
        LL.GeneralizedOperation(2, 1, 0.25, :listed;
            restrictions=("éa" => "δ", "ph" => "f"))
    elseif id == 6
        LL.GeneralizedOperation(2, 2, 0, :pair;
            applicability=LL.APPLICABILITY_EQUAL)
    elseif id == 7
        LL.GeneralizedOperation(2, 1, 1.25, :merge)
    elseif id == 8
        LL.GeneralizedOperation(1, 2, 1.25, :split)
    elseif id == 9
        LL.GeneralizedOperation(1, 1, 0, :listed_zero;
            restrictions=("a" => "é",))
    else
        error("unknown Rust control operation $id")
    end
end

function expected_generalized(field)
    [begin
        fields = split(observation, ',')
        (parse(Int, fields[1]), parse(Int, fields[2]),
            fields[3] == "x" ? nothing : parse(Int, fields[3]),
            parse(Int, fields[2]) != 0, fields[4] == "1")
    end for observation in split(field, ';')]
end

function expected_universal(field)
    [begin
        fields = split(observation, ',')
        (parse(Int, fields[1]), parse(Int, fields[2]),
            fields[3] == "1", fields[4] == "1")
    end for observation in split(field, ';')]
end

function compare_generalized_control(fields, cache)
    _, ids_field, budget_field, source_field, target_field, denominator_field,
        observations_field = fields
    key = (ids_field, parse(Int, budget_field))
    automaton = get!(cache, key) do
        operations = LL.GeneralizedOperation[
            qualification_operation(Int(id)) for id in qualification_units(ids_field)]
        LL.GeneralizedAutomaton(key[2], LL.GeneralizedOperationSet(operations))
    end
    source = qualification_text(source_field)
    target = qualification_text(target_field)
    expected = expected_generalized(observations_field)
    denominator = parse(Int, denominator_field)
    @test length(expected) == length(target) + 1
    state = LL.online(automaton, source)
    try
        initial = LL.observation(state)
        @test generalized_fields(initial) == first(expected)
        @test initial.scale_denominator == denominator
        for (unit, wanted) in zip(target, expected[2:end])
            observed = LL.advance!(state, unit)
            @test generalized_fields(observed) == wanted
            @test observed.scale_denominator == denominator
            @test generalized_fields(LL.observation(state)) == wanted
        end
    finally
        close(state)
    end
    complete = LL.evaluate(automaton, source, target)
    @test generalized_fields(complete) == last(expected)
    @test complete.scale_denominator == denominator
    @test LL.accepts(automaton, source, target) == last(expected)[5]
end

function compare_universal_control(fields, cache)
    _, variant_field, policy_field, domain_field, budget_field, source_field,
        target_field, observations_field = fields
    variant_id = parse(Int, variant_field)
    policy_id = parse(Int, policy_field)
    domain = only(domain_field)
    budget = parse(Int, budget_field)
    key = (variant_id, policy_id, domain, budget)
    automaton = get!(cache, key) do
        variant = (LL.UNIVERSAL_STANDARD, LL.UNIVERSAL_TRANSPOSITION,
            LL.UNIVERSAL_MERGE_AND_SPLIT)[variant_id + 1]
        policy = if policy_id == 0
            LL.UNRESTRICTED_POLICY
        elseif domain == 'T'
            LL.UniversalPolicy('a' => 'é')
        elseif domain == 'B'
            LL.UniversalPolicy(UInt8('p') => UInt8('f'))
        else
            LL.UniversalPolicy(typemax(UInt64) => UInt64(0))
        end
        LL.UniversalAutomaton(budget, policy; variant)
    end
    source = domain == 'T' ? qualification_text(source_field) :
        domain == 'B' ? UInt8.(qualification_units(source_field)) :
        qualification_units(source_field)
    target = domain == 'T' ? qualification_text(target_field) :
        domain == 'B' ? UInt8.(qualification_units(target_field)) :
        qualification_units(target_field)
    expected = expected_universal(observations_field)
    @test length(expected) == length(target) + 1
    state = LL.online(automaton, source)
    try
        @test universal_fields(LL.observation(state)) == first(expected)
        for (unit, wanted) in zip(target, expected[2:end])
            @test universal_fields(LL.advance!(state, unit)) == wanted
            @test universal_fields(LL.observation(state)) == wanted
        end
    finally
        close(state)
    end
    @test universal_fields(LL.evaluate(automaton, source, target)) == last(expected)
    @test LL.accepts(automaton, source, target) == last(expected)[4]
end

@testset "automata invalid domains, rollback, and ownership" begin
    @test_throws ArgumentError LL.GeneralizedOperation(-1, 1, 1, :bad)
    @test_throws OverflowError LL.GeneralizedOperation(big(typemax(UInt)) + 1, 1, 1, :bad)
    @test_throws ArgumentError LL.GeneralizedAutomaton(-1, ())
    @test_throws OverflowError LL.UniversalAutomaton(256)
    @test_throws ArgumentError LL.UniversalPolicy(UInt8(1) => UInt64(1))
    @test_throws ArgumentError LL.UniversalPolicy('a' => 'b', UInt8(1) => UInt8(2))
    @test_throws ArgumentError LL.AutomatonLimits(max_step_work_units=-1)
    for operation in (
        LL.GeneralizedOperation(0, 0, 1, :stutter),
        LL.GeneralizedOperation(1, 0, 0, :free_delete),
        LL.GeneralizedOperation(1, 1, -1, :negative),
        LL.GeneralizedOperation(1, 1, Inf, :infinite),
        LL.GeneralizedOperation(1, 1, NaN, :nan),
        LL.GeneralizedOperation(1, 2, 1, :bad_equal;
            applicability=LL.APPLICABILITY_EQUAL),
        LL.GeneralizedOperation(1, 1, 1, :bad_transpose;
            applicability=LL.APPLICABILITY_ADJACENT_TRANSPOSE),
        LL.GeneralizedOperation(2, 1, 1, :bad_listed;
            restrictions=("a" => "b",)),
    )
        error = qualification_error(() -> LL.GeneralizedAutomaton(2, (operation,)))
        @test error isa LL.NativeError
        @test error.status == Int32(LL.STATUS_INVALID_ARGUMENT)
    end

    generalized = LL.GeneralizedAutomaton(1, (
        LL.GeneralizedOperation(1, 1, 0, :equal;
            applicability=LL.APPLICABILITY_EQUAL),
        LL.GeneralizedOperation(0, 1, 1, :insert),
    ))
    general_state = LL.online(generalized, "a";
        limits=LL.AutomatonLimits(max_target_units=1))
    close(generalized)
    try
        first_result = LL.advance!(general_state, 'a')
        @test qualification_error(() -> LL.advance!(general_state, 'b')).status ==
            Int32(LL.STATUS_LIMIT_EXCEEDED)
        @test generalized_fields(LL.observation(general_state)) ==
            generalized_fields(first_result)
    finally
        close(general_state)
    end
    @test !isopen(generalized)
    @test !isopen(general_state)
    @test_throws LL.NativeError LL.observation(general_state)
    @test_throws LL.NativeError LL.evaluate(generalized, "a", "a")

    invalid_scalar = Char(0xd800)
    @test !isvalid(invalid_scalar)
    scalar_parent = LL.GeneralizedAutomaton(0, (
        LL.GeneralizedOperation(1, 1, 0, :equal;
            applicability=LL.APPLICABILITY_EQUAL),
    ))
    scalar_state = LL.online(scalar_parent, "a")
    try
        before = generalized_fields(LL.observation(scalar_state))
        error = qualification_error(() -> LL.advance!(scalar_state, invalid_scalar))
        @test error isa LL.NativeError
        @test error.status == Int32(LL.STATUS_INVALID_ARGUMENT)
        @test generalized_fields(LL.observation(scalar_state)) == before
        @test LL.advance!(scalar_state, 'a').accepting
    finally
        close(scalar_state)
        close(scalar_parent)
    end
    @test qualification_error(() -> LL.UniversalAutomaton(0,
        LL.UniversalPolicy(invalid_scalar => 'a'))) isa LL.NativeError

    domain_automaton = LL.UniversalAutomaton(0)
    try
        @test_throws ArgumentError LL.evaluate(domain_automaton, Int[-1], Int[0])
        @test_throws OverflowError LL.online(domain_automaton,
            [big(typemax(UInt64)) + 1])
    finally
        close(domain_automaton)
    end

    for (policy, source, good, bad) in (
        (LL.UniversalPolicy('a' => 'é'), "a", 'é', UInt8(0x61)),
        (LL.UniversalPolicy(UInt8(0xff) => UInt8(0)), UInt8[0xff], UInt8(0), 'a'),
        (LL.UniversalPolicy(typemax(UInt64) => UInt64(0)),
            UInt64[typemax(UInt64)], UInt64(0), 'a'),
    )
        parent = LL.UniversalAutomaton(0, policy)
        state = LL.online(parent, source;
            limits=LL.AutomatonLimits(max_target_units=1))
        close(parent)
        try
            before = LL.observation(state)
            @test qualification_error(() -> LL.advance!(state, bad)) isa ArgumentError
            @test universal_fields(LL.observation(state)) == universal_fields(before)
            @test LL.advance!(state, good).accepting
            @test qualification_error(() -> LL.advance!(state, good)).status ==
                Int32(LL.STATUS_LIMIT_EXCEEDED)
            @test LL.observation(state).consumed_target_length == 1
        finally
            close(state)
        end
    end

    bridge = LL.GeneralizedAutomaton(0, (
        LL.GeneralizedOperation(2, 2, 0, :pair;
            applicability=LL.APPLICABILITY_EQUAL),
    ))
    try
        state = LL.online(bridge, "éa")
        try
            @test !LL.advance!(state, 'é').current_row_nonempty
            @test LL.advance!(state, 'a').accepting
        finally
            close(state)
        end
        @test !LL.evaluate(bridge, "éa", "é").accepting
        @test LL.evaluate(bridge, "éa", "éa").accepting
    finally
        close(bridge)
    end

    bounded = LL.UniversalAutomaton(0)
    try
        stream = LL.prefix_observations(bounded, "a", "aa";
            limits=LL.AutomatonLimits(max_target_units=1))
        @test qualification_error(() -> collect(stream)) isa LL.NativeError
        @test !isopen(stream)
        empty_stream = LL.prefix_observations(bounded, "a", "")
        @test isempty(collect(empty_stream))
        @test !isopen(empty_stream)
    finally
        close(bounded)
    end
end

control = get(ENV, "LIBLEVENSHTEIN_RUST_CONTROL", "")
if !isempty(control)
    @testset "independent public Rust automata controls" begin
        generalized_cache = Dict{Tuple{String,Int},LL.GeneralizedAutomaton}()
        universal_cache = Dict{Tuple{Int,Int,Char,Int},LL.UniversalAutomaton}()
        generalized_count = 0
        universal_count = 0
        try
            for line in split(read(Cmd([control]), String), '\n'; keepempty=false)
                fields = split(line, '\t'; keepempty=true)
                if fields[1] == "G"
                    compare_generalized_control(fields, generalized_cache)
                    generalized_count += 1
                elseif fields[1] == "U"
                    compare_universal_control(fields, universal_cache)
                    universal_count += 1
                else
                    error("unknown Rust automata control record")
                end
            end
        finally
            foreach(close, values(generalized_cache))
            foreach(close, values(universal_cache))
        end
        @test generalized_count == 1092
        @test universal_count == 3456
    end
else
    @info "Rust automata controls unavailable; set LIBLEVENSHTEIN_RUST_CONTROL to the built example"
end
