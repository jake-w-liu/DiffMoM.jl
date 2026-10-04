using DiffMoM, Test, LinearAlgebra, Random

@testset "planar FFT: complete modal operator equivalence" begin
    G = DiffMoM
    rng = MersenneTwister(3271)
    for walls in (WALL_PEC, WALL_PMC)
        grid = CellGrid(8e-3, 6e-3, 5, 4; walls)
        stack = PlanarStackup([PlanarLayer(2.2 - .02im, 1.0, .4e-3),
            PlanarLayer(3.0, 1.1, .6e-3; epsr_z=4.0),
            PlanarLayer(1.0, 1.0, .3e-3)], TERM_GND, TERM_GND, grid.a, grid.b)
        sheets = [sheet_level(1, 5, 4), sheet_level(2, 5, 4)]
        for sh in sheets
            sh.mask .= true
            sh.connect_west .= true; sh.connect_east .= true
            sh.connect_south .= true; sh.connect_north .= true
        end
        vl = via_level(2, 5, 4)
        vl.uni[2, 2] = vl.tap[2, 2] = true
        vl.uni[4, 3] = true
        vol = vol_level(3, 5, 4)
        vol.mask .= true
        vol.connect_west .= true; vol.connect_east .= true
        vol.connect_south .= true; vol.connect_north .= true
        ports = [PlanarPort(1, :west, 2:3, 50.0), PlanarPort(2, :north, 2:4, 75.0)]
        prob = build_planar_problem(stack, grid, sheets, ports; vias=[vl], vols=[vol])
        nb = planar_basis_count(prob.basis)
        xs = [randn(rng, ComplexF64, nb) for _ in 1:2]
        # Modes above 2nx/2ny verify analytic high-mode folding.  All
        # basis families including both kinds of wall half are present.
        for (mx, my) in ((7, 6), (17, 13))
            kw = (; mx, my, surface_zs=[.3+.2im, .5+.1im],
                via_sigma=5.8e7, volume_sigma=4e7)
            Z = assemble_planar_z(stack, grid, sheets, prob.basis, 2pi * 8e9;
                vias=prob.vias, vols=prob.vols, kw...)
            A = G.planar_ufft_operator(prob, 8e9; kw...)
            @test size(A) == size(Z)
            @test G._planar_ufft_diagonal(A) ≈ diag(Z) rtol=2e-12
            for x in xs
                y = A * x
                @test norm(y - Z*x) / norm(Z*x) < 2e-12
                # Complex-symmetric Galerkin reaction uses transpose,
                # including complex excitation coefficients.
                @test sum(xs[2] .* (A * xs[1])) ≈
                    sum(xs[1] .* (A * xs[2])) rtol=2e-11
                z = copy(x)
                mul!(z, A, z)  # source/output alias is safe
                @test z ≈ y rtol=2e-12
                z = copy(y)
                mul!(z, A, x, 2+.3im, -.2+.1im)
                @test z ≈ (1.8+.4im) .* y rtol=2e-12
            end
            A * xs[1]  # warm repeated matvec specialization
            @test @allocated(mul!(zeros(ComplexF64, nb), A, xs[1])) < 3000
            @test_throws DimensionMismatch mul!(zeros(ComplexF64, nb+1), A, xs[1])
        end
    end
end

@testset "planar FFT: port solve and residual gate" begin
    G = DiffMoM
    grid = CellGrid(4e-3, 3e-3, 6, 5)
    stack = PlanarStackup([PlanarLayer(2.0-.02im, 1., .4e-3),
        PlanarLayer(1.0, 1., .4e-3)], TERM_GND, TERM_GND, grid.a, grid.b)
    sheet = sheet_level(1, 6, 5)
    rasterize_rect!(sheet, grid, 0., grid.a, .6e-3, 2.4e-3)
    rows = findall(sheet.mask[1, :])
    sheet.connect_west[rows] .= true; sheet.connect_east[rows] .= true
    prob = build_planar_problem(stack, grid, [sheet],
        [PlanarPort(1,:west,first(rows):last(rows),50.),
         PlanarPort(1,:east,first(rows):last(rows),75.)])
    kw = (; mx=20, my=18, surface_zs=.1+.1im)
    dense = solve_planar(prob, 5e9; kw...)
    fast = G.solve_planar_ufft(prob, 5e9; rtol=1e-10, memory=100, kw...)
    @test fast.y ≈ dense.y rtol=1e-9
    @test fast.s ≈ dense.s rtol=1e-9
    @test fast.currents ≈ dense.currents rtol=1e-8
    @test maximum(fast.relative_residuals) < 1e-10
    @test_throws ErrorException G.solve_planar_ufft(prob, 5e9;
        maxiter=1, memory=1, rtol=1e-13, kw...)
    @test_throws ArgumentError G.planar_ufft_operator(prob,5e9; max_bytes=1)
    @test_throws ArgumentError G.planar_ufft_operator(prob,5e9; mx=typemax(Int))
    @test_throws ArgumentError size(fast.operator, 0)
    @test_throws ArgumentError G.solve_planar_ufft(prob,5e9; max_bytes=1)
end
