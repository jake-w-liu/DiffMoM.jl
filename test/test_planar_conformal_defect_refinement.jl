using DiffMoM, Test

@testset "conformal: correction updates retain product and sum roundoff" begin
    initial=ComplexF64[1e16+10im,1.1-7im]
    deltas=[ComplexF64[1+11im,1e16+1e16im],
        ComplexF64[-1e16-21im,-1e16-1e16im],ComplexF64[2+3im,.1+1.1im]]
    scales=[1.,1.,1.1]
    exact=setprecision(256) do
        result=Complex{BigFloat}.(initial)
        for (delta,scale) in zip(deltas,scales)
            result.+=BigFloat(scale).*Complex{BigFloat}.(delta)
        end
        ComplexF64.(result)
    end
    ordinary=copy(initial);actual=copy(initial);carry=zeros(ComplexF64,2)
    for (delta,scale) in zip(deltas,scales)
        ordinary.+=scale.*delta
        @test DiffMoM._planar_projection_compensated_update!(actual,delta,scale,carry)===actual
    end
    @test ordinary!=exact
    @test actual==exact
    @test (@allocated DiffMoM._planar_projection_compensated_update!(actual,deltas[1],scales[1],carry))==0
    @test_throws DimensionMismatch DiffMoM._planar_projection_compensated_update!(actual,deltas[1],1.,zeros(ComplexF64,1))
end

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
