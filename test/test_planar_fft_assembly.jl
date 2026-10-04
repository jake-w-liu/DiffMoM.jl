using DiffMoM,Test,LinearAlgebra

@testset "exact FFT retained dense fill versus independent modal assembly" begin
    for walls in (WALL_PEC,WALL_PMC)
        grid=CellGrid(.008,.006,5,4;walls)
        stack=PlanarStackup([PlanarLayer(2.2-.02im,1.,.0004),
            PlanarLayer(3.,1.1,.0006;epsr_z=4.),PlanarLayer(1.,1.,.0003)],TERM_GND,TERM_GND,grid.a,grid.b)
        sheets=[sheet_level(1,5,4),sheet_level(2,5,4)]
        for sh in sheets
            sh.mask.=true;sh.connect_west.=true;sh.connect_east.=true
            sh.connect_south.=true;sh.connect_north.=true
        end
        vl=via_level(2,5,4);vl.uni[2,2]=vl.tap[2,2]=true;vl.uni[4,3]=true
        vol=vol_level(3,5,4);vol.mask.=true
        vol.connect_west.=true;vol.connect_east.=true;vol.connect_south.=true;vol.connect_north.=true
        prob=build_planar_problem(stack,grid,sheets,[PlanarPort(1,:west,2:3,50.),
            PlanarPort(2,:north,2:4,75.)];vias=[vl],vols=[vol])
        for (mx,my) in ((7,6),(17,13))
            kw=(;mx,my,surface_zs=[.3+.2im,.5+.1im],via_sigma=5.8e7,volume_sigma=4e7)
            modal=assemble_planar_z(stack,grid,sheets,prob.basis,2pi*8e9;vias=prob.vias,vols=prob.vols,kw...)
            fft=assemble_planar_z_ufft(prob,8e9;kw...)
            @test norm(fft-modal)/norm(modal)<2e-12
            @test fft≈transpose(fft) rtol=2e-12
            direct=solve_planar(prob,8e9;kw...)
            fast=solve_planar(prob,8e9;method=:dense_fft,kw...)
            @test fast.y≈direct.y rtol=2e-10
            @test fast.s≈direct.s rtol=2e-10
            @test fast.currents≈direct.currents rtol=2e-10
            compact=solve_planar(prob,8e9;method=:dense_fft,retain_matrix=false,kw...)
            @test compact.z_mom===nothing
            @test compact.s≈fast.s rtol=1e-13
        end
        @test_throws ArgumentError assemble_planar_z_ufft(prob,8e9;max_bytes=1)
        @test_throws ArgumentError assemble_planar_z_ufft(prob,8e9;mx=typemax(Int))
        @test_throws ArgumentError solve_planar(prob,8e9;method=:dense_fft,max_bytes=1)
    end
    # Translated one-sided terminal spectra contain mixed sine/cosine
    # coefficients; nonuniform material and mutual sheet Gram are retained.
    grid=CellGrid(.006,.004,6,4)
    stack=PlanarStackup([PlanarLayer(1.,1.,.0005),PlanarLayer(1.,1.,.0005),PlanarLayer(1.,1.,.0005)],TERM_GND,TERM_GND,grid.a,grid.b)
    sheets=[sheet_level(1,6,4),sheet_level(2,6,4)]
    for sh in sheets
        sh.mask[2:5,2:3].=true
    end
    ports=[PlanarPort(1,:terminal_x,1,2:3,50.;metal_side=:positive),
        PlanarPort(2,:terminal_x,5,2:3,75.;metal_side=:negative)]
    prob=build_planar_problem(stack,grid,sheets,ports)
    maps=[fill(.1+.2im,6,4),fill(.3+.1im,6,4)];maps[1][3,2]=.9+.8im
    kw=(;mx=21,my=19,surface_zs=maps,sheet_coupling_zs=[.01+.01im .002;.002 .02+.01im])
    modal=assemble_planar_z(stack,grid,sheets,prob.basis,2pi*5e9;kw...)
    fft=assemble_planar_z_ufft(prob,5e9;kw...)
    @test norm(fft-modal)/norm(modal)<2e-12
end
