module SonnetFloatingBudgetFullTests
using DiffMoM,Test
include(joinpath(@__DIR__,"helpers","native_floating_pad_project.jl"))

function numerical_budget(problem,frequency;kw...)
    budget=1
    for _ in 1:6
        try
            solve_planar(problem,frequency;max_bytes=budget,kw...)
            return budget
        catch error
            error isa ArgumentError || rethrow()
            required=match(r"requires (\d+) raw bytes",sprint(showerror,error))
            required===nothing && rethrow()
            next=parse(Int,required[1]);budget<next<=2*1024^2 || rethrow()
            budget=next
        end
    end
    error("tiny feed budget did not converge")
end

@testset "FLOAT pin standards share the retained source and circuit budget" begin
    project=_native_floating_pad_project(;cup=true)
    append!(project.ports[1].values,["FIX",".01"])
    frequency=50e9;limit=2*1024^2
    model=sonnet_floating_model(project,frequency;max_bytes=limit)
    geometry=DiffMoM._subdivision_retained_payload(Any[model.problem.basis,
        model.problem.sheets,model.problem.vias,model.problem.vols,model.problem.bricks,
        model.problem.stack.layers,model.sheet_zs,model.via_sigma,model.contraction,
        model.floating_common,model.source_incidence])
    @test model.payload>=geometry
    reference=solve_sonnet_project(project,frequency;mx=32,my=32,max_bytes=limit)
    retained=DiffMoM._sonnet_retained_solve_payload((reference.em,reference.circuit))
    @test retained>DiffMoM._sonnet_retained_solve_payload(reference.em)
    @test DiffMoM._sonnet_retained_solve_payload((reference.em,reference.circuit,reference.em))==retained
    n=length(model.labels);nc=size(model.floating_common,2)
    workspace=16*36*(n+nc+model.circuit.nnodes)^2+
        16*planar_basis_count(model.problem.basis)*n+8length(model.problem.ports)*(n+nc)
    reserve=model.payload+workspace
    @test DiffMoM._sonnet_child_solve_budget((reference.em,reference.circuit),reserve,
        reserve+retained+1)==1
    @test_throws ArgumentError DiffMoM._sonnet_child_solve_budget((reference.em,reference.circuit),
        reserve,reserve+retained-1)
    pin=model.pin_refplanes[1].terminal
    feed=PlanarPort(pin.level,:west,DiffMoM._sonnet_pin_edge_extent(model.problem,pin),50.)
    numerical=maximum(numerical_budget(DiffMoM._sonnet_feed_standard(model.problem,feed,nf),
        frequency;mx=2nf,my=32) for nf in (8,16))
    # The former flow let the standards consume this whole numerical
    # allowance while both parent results were still live.
    error=try
        solve_sonnet_project(project,frequency;mx=32,my=32,max_bytes=reserve+numerical)
        nothing
    catch error
        error
    end
    @test error isa ArgumentError
    @test occursin("max_bytes=$(numerical-retained)",sprint(showerror,error))
    sufficient=solve_sonnet_project(project,frequency;mx=32,my=32,
        max_bytes=reserve+retained+numerical)
    @test sufficient.s≈reference.s rtol=1e-12
    @test sufficient.y≈reference.y rtol=1e-12
end
end
