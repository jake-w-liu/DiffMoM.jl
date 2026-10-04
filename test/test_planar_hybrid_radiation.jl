using DiffMoM,Test,LinearAlgebra,Random

@testset "hybrid radiation preserves simultaneous triangle and bulk currents" begin
    grid=CellGrid(.002,.003,4,6)
    stack=PlanarStackup([PlanarLayer(1.,1.,.0004),PlanarLayer(1.,1.,.0006)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    mesh=PlanarConformalMesh([0. .002 .002 0.;.001 .001 .002 .002],
        [1 1;2 3;3 4];interfaces=[1,1])
    conformal=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(.001,.002)),
        PlanarConformalPort(1,:east,(.001,.002))])
    via=via_level(1,4,6);via.uni[2,3]=true;via.tap[2,3]=true
    volume=vol_level(2,4,6);volume.mask[2:3,3:4].=true
    prob=PlanarHybridProblem(conformal,grid;vias=[via],vols=[volume])
    radiation=planar_radiation_stack(stack;bottom=TERM_SPACE,top=TERM_SPACE)
    nc=length(conformal.basis.width);nb=nc+planar_basis_count(prob.bulk.basis)
    rng=MersenneTwister(321);coeff=randn(rng,ComplexF64,nb)
    options=(;theta=[.2,.8,2.4,pi-.2],phi=[.1,.6],radiation_stack=radiation)
    whole=planar_farfield(prob,coeff,3e9;options...)
    sheets=planar_farfield(conformal,coeff[1:nc],3e9;options...)
    bulk=planar_farfield(prob.bulk,coeff[nc+1:end],3e9;options...)
    @test whole.etheta≈sheets.etheta+bulk.etheta rtol=1e-13
    @test whole.ephi≈sheets.ephi+bulk.ephi rtol=1e-13
    @test whole.intensity≈(abs2.(whole.etheta)+abs2.(whole.ephi))/(2sqrt(DiffMoM._MU0/DiffMoM._EPS0)) rtol=1e-13
    @test_throws ArgumentError planar_farfield(prob,coeff[1:end-1],3e9;options...)
    @test_throws ArgumentError planar_farfield(prob,coeff,3e9;max_bytes=1,options...)
    power=planar_radiated_power(prob,coeff,3e9;ntheta=6,nphi=12,refine=true,radiation_stack=radiation)
    @test power.power>0 && power.relative_change<.001
    dense=solve_planar_hybrid(prob,3e9;mx=16,my=20,surface_zs=.2,via_sigma=1e6,volume_sigma=1e6)
    fft=solve_planar_hybrid(prob,3e9;method=:ufft,mx=16,my=20,surface_zs=.2,
        via_sigma=1e6,volume_sigma=1e6,memory=100,rtol=1e-10)
    waves=ComplexF64[1.,.2im]
    for result in (dense,fft)
        voltage=sqrt.([p.z0 for p in prob.ports]).*(waves+result.s*waves)
        direct=planar_farfield(prob,result.currents*voltage,3e9;options...)
        mapped=planar_farfield(result;incident_waves=waves,options...)
        @test direct.etheta≈mapped.etheta rtol=1e-12
        @test direct.ephi≈mapped.ephi rtol=1e-12
        @test mapped.accepted_power≈.5real(dot(voltage,result.y*voltage)) rtol=1e-12
        @test_throws ArgumentError planar_farfield(result;max_bytes=1,options...)
    end
    @test planar_farfield(dense;incident_waves=waves,options...).etheta≈
        planar_farfield(fft;incident_waves=waves,options...).etheta rtol=1e-8
end
