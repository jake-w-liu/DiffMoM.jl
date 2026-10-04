using DiffMoM, Test, LinearAlgebra

@testset "dense modal contractions: contiguous and repeated interfaces" begin
    for walls in (WALL_PEC, WALL_PMC), interfaces in ([1,2], [1,2,1])
        grid = CellGrid(.006,.003,6,3; walls)
        stack = PlanarStackup([PlanarLayer(2-.02im,1.,.0004),
            PlanarLayer(3.,1.,.0006),PlanarLayer(1.,1.,.0005)],
            TERM_GND,TERM_GND,grid.a,grid.b)
        sheets = [sheet_level(level,6,3) for level in interfaces]
        sheets[1].mask[1:2,:] .= true
        sheets[2].mask .= true
        length(sheets)==3 && (sheets[3].mask[5:6,:] .= true)
        basis = build_planar_basis(grid,sheets,PlanarPort[])
        prob = PlanarProblem(stack,grid,sheets,PlanarPort[],basis)
        kwargs = (; mx=17,my=11,surface_zs=.2+.1im)
        # The exact FFT signed-kernel fill is independent of the grouped
        # matrix contraction and preserves every supplied modal index.
        expected = assemble_planar_z_ufft(prob,4e9;kwargs...)
        for block in (1,7,512)
            actual = assemble_planar_z(stack,grid,sheets,basis,2pi*4e9;
                block,kwargs...)
            @test actual ≈ expected rtol=3e-12 atol=1e-15
            @test actual ≈ transpose(actual) rtol=3e-12 atol=1e-15
        end
    end
end
