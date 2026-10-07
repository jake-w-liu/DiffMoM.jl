module PlanarRadiationRangeTests
using DiffMoM,LinearAlgebra,Test,TOML

grid=CellGrid(.002,.002,4,4)
stack=PlanarStackup([PlanarLayer(1.,1.,.001),PlanarLayer(1.,1.,.001)],
    TERM_GND,TERM_SPACE,grid.a,grid.b)
sheet=sheet_level(1,4,4);rasterize_rect!(sheet,grid,0,grid.a,0,grid.b)
prob=build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:x,2,2:2,50.)])
raw=solve_planar(prob,1e9;mx=12,my=12,surface_zs=.1)
unit=planar_farfield(prob,raw.currents[:,1],1e9;theta=[.4],phi=[.2])
field=maximum(abs,vcat(vec(unit.etheta),vec(unit.ephi)))
large=ldexp(floatmax(Float64),-precision(Float64))
representable=2sqrt(floatmax(Float64))/field
Q=Rational{BigInt}
eta=sqrt(DiffMoM._MU0/DiffMoM._EPS0)
# Count the expression's scalar operations: scale multiplication/sqrt,
# four component divisions, four squares, and three additions. On these
# normal-range controls a gamma bound covers their rounded positive sum.
roles=(:scale_multiply,:scale_sqrt,:theta_re_divide,:theta_im_divide,
    :phi_re_divide,:phi_im_divide,:theta_re_square,:theta_im_square,
    :phi_re_square,:phi_im_square,:theta_add,:phi_add,:polarizations_add)
roundoff=Q(eps(Float64))/2
gamma=(length(roles)*roundoff)/(1-length(roles)*roundoff)
rows=Dict{String,Any}[]
@testset "radiation intensity: representable energy and output boundaries" begin
    for amplitude in (1.,representable)
        coefficients=raw.currents[:,1].*amplitude
        saved=copy(coefficients)
        pattern=planar_farfield(prob,coefficients,1e9;theta=[.4],phi=[.2])
        expected=(abs2.(Complex{Q}.(pattern.etheta))+abs2.(Complex{Q}.(pattern.ephi)))./(2Q(eta))
        @test all(isfinite,pattern.etheta) && all(isfinite,pattern.ephi)
        @test all(isfinite,pattern.intensity) && all(>(0),pattern.intensity)
        @test all(i->abs(Q(pattern.intensity[i])-expected[i])<=gamma*expected[i],eachindex(expected))
        @test coefficients==saved
        push!(rows,Dict("amplitude"=>amplitude,"reported_intensity"=>vec(pattern.intensity),
            "exact_field_intensity"=>vec(Float64.(expected))))
    end
    coefficients=raw.currents[:,1].*large
    @test all(isfinite,coefficients)
    @test_throws ArgumentError planar_farfield(prob,coefficients,1e9;theta=[.4],phi=[.2])
    @test_throws ArgumentError planar_farfield(raw;voltages=[large],theta=[.4],phi=[.2])
    @test_throws ArgumentError planar_farfield(prob,coefficients,1e9;theta=[.4],phi=[.2],max_bytes=1)
    zeros_pattern=planar_farfield(prob,zero.(coefficients),1e9;theta=[.4],phi=[.2])
    @test all(iszero,zeros_pattern.intensity)
end

end
