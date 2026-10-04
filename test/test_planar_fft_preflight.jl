module PlanarFFTPreflightTests
using DiffMoM, Test, LinearAlgebra

function discovery_fixture(nx,ny)
    grid=CellGrid(.01,.008,nx,ny)
    stack=PlanarStackup([PlanarLayer(2.,1.,.0005),PlanarLayer(1.,1.,.0005)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    sheet=sheet_level(1,nx,ny);sheet.mask.=true
    sheet.connect_west.=true;sheet.connect_east.=true
    return build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:west,1:ny,50.),
        PlanarPort(1,:east,1:ny,50.)])
end

function reject_discovery(prob,dense)
    try
        DiffMoM._planar_fft_workspace(prob,1e9,Val(dense);max_bytes=1)
    catch err
        err isa ArgumentError || rethrow()
        return nothing
    end
    error("an insufficient discovery budget was accepted")
end

@testset "FFT discovery rejects before linear basis allocations" begin
    # The earlier preflight created ~480KB and ~1.4MB of temporary arrays
    # for these borrowed problems before rejecting max_bytes=1.
    for (nx,ny) in ((100,100),(300,100)),dense in (false,true)
        prob=discovery_fixture(nx,ny)
        reject_discovery(prob,dense)
        @test (@allocated reject_discovery(prob,dense))<32_768
        @test_throws ArgumentError DiffMoM._planar_fft_workspace(prob,1e9,Val(dense);max_bytes=0)
        @test_throws ArgumentError DiffMoM._planar_fft_workspace(prob,1e9,Val(dense);max_bytes=-1)
        @test_throws ArgumentError DiffMoM._planar_fft_workspace(prob,1e9,Val(dense);max_bytes=BigInt(typemax(Int))+1)
    end
    prob=discovery_fixture(4,3)
    kw=(;mx=13,my=11,surface_zs=.2+.1im)
    compact=assemble_planar_z_ufft(prob,1e9;kw...)
    modal=assemble_planar_z(prob.stack,prob.grid,prob.sheets,prob.basis,2pi*1e9;kw...)
    @test compact≈modal rtol=2e-12
    operator=planar_ufft_operator(prob,1e9;kw...)
    x=ComplexF64[sin(p)+im*cos(p) for p in 1:size(operator,1)]
    @test operator*x≈modal*x rtol=2e-12
end
end
