module PlanarLegacyCurrentTypesTests
using DiffMoM,LinearAlgebra,Test,TOML

grid=CellGrid(.002,.002,4,4)
stack=PlanarStackup([PlanarLayer(1.,1.,.001),PlanarLayer(1.,1.,.001)],
    TERM_GND,TERM_SPACE,grid.a,grid.b)
sheet=sheet_level(1,4,4);rasterize_rect!(sheet,grid,0,grid.a,0,grid.b)
prob=build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:x,2,2:2,50.)])
raw=solve_planar(prob,1e9;mx=12,my=12,surface_zs=.1)
unit=zeros(ComplexF64,size(raw.currents));unit[1]=1
expected=PlanarResult(raw.problem,raw.omega,raw.freq,raw.z_mom,raw.lu_fact,unit,raw.y,raw.s,raw.z0)
maps=planar_current_maps(expected)
field=planar_farfield(expected;theta=[.4],phi=[.2])
rows=Dict{String,Any}[]
@testset "result currents: legacy ordinary numeric types preserve consumers" begin
    for T in (Complex{Int},ComplexF32,ComplexF64,Float32,Float64)
        X=zeros(T,size(unit));X[1]=1;saved=copy(X)
        result=PlanarResult(raw.problem,raw.omega,raw.freq,raw.z_mom,raw.lu_fact,X,raw.y,raw.s,raw.z0)
        @test eltype(result.currents)===ComplexF64
        @test result.currents==unit
        @test X==saved
        actual=planar_current_maps(result)
        @test all(all(getproperty(a,k)==getproperty(b,k) for k in (:jx,:jy,:jz)) for (a,b) in zip(actual,maps))
        contracted=planar_contract_ports(result,ones(1,1);z0=result.z0)
        @test contracted.currents==unit
        pattern=planar_farfield(result;theta=[.4],phi=[.2])
        @test pattern.etheta==field.etheta && pattern.ephi==field.ephi && pattern.intensity==field.intensity
        @test (T===ComplexF64) ? result.currents===X : result.currents!==X
        push!(rows,Dict("input_type"=>string(T),"stored_type"=>string(eltype(result.currents))))
    end
end
@testset "result currents: Float64 and owned wide views preserve representation" begin
    ordinary=PlanarResult(raw.problem,raw.omega,raw.freq,raw.z_mom,raw.lu_fact,view(unit,:,:),raw.y,raw.s,raw.z0)
    @test ordinary.currents==unit && ordinary.currents!==unit
    X=Complex{BigFloat}.(unit);saved=deepcopy(X)
    wide=PlanarResult(raw.problem,raw.omega,raw.freq,raw.z_mom,raw.lu_fact,X,raw.y,raw.s,raw.z0)
    viewed=PlanarResult(raw.problem,raw.omega,raw.freq,raw.z_mom,raw.lu_fact,view(X,:,:),raw.y,raw.s,raw.z0)
    @test wide.currents===X && eltype(wide.currents)===Complex{BigFloat}
    @test viewed.currents==X && viewed.currents!==X && eltype(viewed.currents)===Complex{BigFloat}
    @test DiffMoM._planar_current_precision(viewed.currents)==precision(BigFloat)
    @test all(all(getproperty(a,k)==getproperty(b,k) for k in (:jx,:jy,:jz)) for (a,b) in zip(planar_current_maps(wide),maps))
    pattern=planar_farfield(wide;theta=[.4],phi=[.2])
    @test pattern.etheta==field.etheta && pattern.ephi==field.ephi && pattern.intensity==field.intensity
    @test X==saved
end

end
