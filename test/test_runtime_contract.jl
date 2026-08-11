function _is_setprecision_callee(callee)
    callee === :setprecision && return true
    return callee isa Expr &&
           callee.head === :. &&
           length(callee.args) == 2 &&
           callee.args[2] isa QuoteNode &&
           callee.args[2].value === :setprecision
end

function _validate_setprecision_scopes!(
    violations::Vector{String}, expression, source::String)
    expression isa Expr || return 0

    if expression.head === :do &&
       !isempty(expression.args) &&
       expression.args[1] isa Expr
        call = expression.args[1]
        if call.head === :call &&
           !isempty(call.args) &&
           _is_setprecision_callee(call.args[1])
            if length(call.args) < 3 || call.args[2] !== :BigFloat
                push!(violations,
                      "$source: precision scopes must explicitly target BigFloat")
            end

            count = 1
            for argument in call.args[2:end]
                count += _validate_setprecision_scopes!(
                    violations, argument, source)
            end
            for argument in expression.args[2:end]
                count += _validate_setprecision_scopes!(
                    violations, argument, source)
            end
            return count
        end
    end

    if expression.head === :call &&
       !isempty(expression.args) &&
       _is_setprecision_callee(expression.args[1])
        push!(violations,
              "$source: setprecision must be used with a do block")
    end

    count = 0
    for argument in expression.args
        count += _validate_setprecision_scopes!(violations, argument, source)
    end
    return count
end

function _source_setprecision_contract()
    source_root = normpath(joinpath(@__DIR__, "..", "src"))
    source_files = String[]
    for (directory, _, filenames) in walkdir(source_root)
        for filename in filenames
            endswith(filename, ".jl") || continue
            push!(source_files, joinpath(directory, filename))
        end
    end
    sort!(source_files)

    violations = String[]
    scope_count = 0
    for source_file in source_files
        expression = Meta.parseall(read(source_file, String))
        scope_count += _validate_setprecision_scopes!(
            violations, expression, source_file)
    end
    return violations, scope_count
end

@testset "Julia runtime safety contract" begin
    @test VERSION >= v"1.12.0"

    good_violations = String[]
    good_count = _validate_setprecision_scopes!(
        good_violations,
        Meta.parse("setprecision(BigFloat, 128) do; precision(BigFloat); end"),
        "synthetic-good")
    @test good_count == 1
    @test isempty(good_violations)

    bad_violations = String[]
    _validate_setprecision_scopes!(
        bad_violations,
        Meta.parse("setprecision(BigFloat, 128)"),
        "synthetic-bad")
    @test length(bad_violations) == 1

    source_violations, source_scope_count = _source_setprecision_contract()
    @test source_scope_count > 0
    @test isempty(source_violations)

    baseline = precision(BigFloat)
    first_entered = Channel{Nothing}(1)
    second_entered = Channel{Nothing}(1)
    release_second = Channel{Nothing}(1)

    first_task = Threads.@spawn setprecision(BigFloat, 512) do
        put!(first_entered, nothing)
        take!(second_entered)
        return precision(BigFloat), precision(BigFloat(1))
    end
    take!(first_entered)

    second_task = Threads.@spawn setprecision(BigFloat, 64) do
        put!(second_entered, nothing)
        take!(release_second)
        return precision(BigFloat), precision(BigFloat(1))
    end

    first_observed = fetch(first_task)
    put!(release_second, nothing)
    second_observed = fetch(second_task)

    @test first_observed == (512, 512)
    @test second_observed == (64, 64)
    @test precision(BigFloat) == baseline
end
