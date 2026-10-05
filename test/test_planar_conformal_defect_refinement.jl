using DiffMoM, Test

@testset "conformal: normalized low-frequency residual corrections" begin
    a,b=1e-3,.5e-3
    stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3),PlanarLayer(1.,1.,.5e-3)],
        TERM_GND,TERM_GND,a,b)
    mesh=PlanarConformalMesh([0. a a 0.;0. 0. b b],[1 1;2 3;3 4])
    prob=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(0.,b)),
        PlanarConformalPort(1,:east,(0.,b))])
    # Nearby representable frequencies exposed the same residual plateau as
    # the Windows 1 MHz failure. Neither extra warm retries nor an unscaled
    # correction removed it; retain both independent acceptance equations.
    for offset in -32:32
        frequency=offset<0 ? prevfloat(1e6,-offset) : nextfloat(1e6,offset)
        result=solve_planar_conformal_defect(prob,frequency;modes=32,nx=16,ny=8,surface_zs=2.)
        @test maximum(result.diagnostics.initial_projected_relative_residuals)<=1e-10
        @test maximum(result.relative_residuals)<=1e-10
        @test real(2/(result.y[1,1]-result.y[1,2]))≈2a/b rtol=2e-6
        @test all(i->i<=10000,result.iterations)
    end
end
