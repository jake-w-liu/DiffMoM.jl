module SonnetFloatingBudgetTests
using DiffMoM,Test

@testset "FLOAT source geometry and retained calibration budget" begin
    grid=CellGrid(.002,.002,2,2);sheet=sheet_level(1,2,2)
    sheet.mask.=true;sheet.connect_west.=true
    stack=PlanarStackup([PlanarLayer(1,1,.0005),PlanarLayer(1,1,.0005)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    problem=build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:west,1:2,50.)])
    model=(;problem,contraction=ones(1,1),floating_common=zeros(1,0),
        sheet_zs=[zeros(ComplexF64,2,2)],via_sigma=Float64[],z0=ComplexF64[50],labels=[1],
        source_incidence=ones(1,1),source_terminals=[(1,0)],circuit=PlanarCircuit(1,[1]),
        bridges=Any[],reference_paths=Any[],pin_refplanes=Dict(),
        scalar_files=nothing,model_files=nothing,variables=Dict{String,Float64}())
    geometry=DiffMoM._subdivision_retained_payload(Any[model.problem.basis,
        model.problem.sheets,model.problem.vias,model.problem.vols,model.problem.bricks,
        model.problem.stack.layers,model.sheet_zs,model.via_sigma,model.contraction,
        model.floating_common,model.source_incidence])
    @test DiffMoM._sonnet_floating_retained_payload(model)>=geometry
    # S and Y intentionally share storage: each retained allocation counts
    # once, even when the same result appears twice in the parent tuple.
    s=zeros(ComplexF64,1,1);voltage=zeros(ComplexF64,3,1)
    current=zeros(ComplexF64,1,1);refs=ComplexF64[50]
    circuit=PlanarCircuitResult(1.,s,s,voltage,current,Int[],refs)
    retained=sizeof(s)+sizeof(voltage)+sizeof(current)+sizeof(refs)
    @test DiffMoM._sonnet_retained_solve_payload((circuit,circuit))==retained
    @test DiffMoM._sonnet_child_solve_budget((circuit,circuit),100,retained+101)==1
    @test_throws ArgumentError DiffMoM._sonnet_child_solve_budget((circuit,circuit),100,retained+99)
end
end
