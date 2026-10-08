using DiffMoM, Test, LinearAlgebra, Profile

function _resource_planar_problem()
    a, b = 5e-3, 10e-3
    stack = PlanarStackup([PlanarLayer(1.0, 1.0, 0.5e-3),
        PlanarLayer(1.0, 1.0, 0.5e-3)], TERM_GND, TERM_GND, a, b)
    grid = CellGrid(a, b, 32, 40)
    sheet = sheet_level(1, grid.nx, grid.ny)
    rasterize_rect!(sheet, grid, 0.0, a, 4e-3, 6e-3; connected=true)
    return build_planar_problem(stack, grid, [sheet],
        [PlanarPort(1, :west, 17:24, 50.0),
         PlanarPort(1, :east, 17:24, 75.0)])
end

function _rejected_planar_solve_allocation(prob)
    try
        solve_planar(prob, 8e9; max_bytes=1)
    catch err
        err isa ArgumentError || rethrow()
    end
    return nothing
end

function _retained_planar_matrix_buffer_payload(prob, retain::Bool, payload::Int)
    Profile.Allocs.clear()
    try
        # A sampling probability of one records every allocation. Measure
        # whole matrix data buffers, independently of metadata and total-byte
        # counter variation, using the same required matrix payload.
        Profile.Allocs.@profile sample_rate=1.0 solve_planar(prob, 8e9; retain_matrix=retain)
        return sum(a.size for a in Profile.Allocs.fetch().allocs
            if a.type===Profile.Allocs.BufferType && a.size==payload)
    finally
        Profile.Allocs.clear()
    end
end

@testset "planar: solve resource contract" begin
    prob = _resource_planar_problem()
    nb = planar_basis_count(prob.basis)
    _rejected_planar_solve_allocation(prob)
    @test (@allocated _rejected_planar_solve_allocation(prob)) < 16nb^2
    @test_throws ArgumentError solve_planar(prob, 8e9; max_bytes=0)
    @test_throws ArgumentError solve_planar(prob, 8e9; max_bytes=-1)
    kept = solve_planar(prob, 8e9)
    compact = solve_planar(prob, 8e9; retain_matrix=false)
    @test compact.z_mom === nothing
    @test kept.z_mom !== kept.lu_fact.factors
    @test compact.y ≈ kept.y rtol=1e-12
    @test compact.s ≈ kept.s rtol=1e-12
    @test compact.currents ≈ kept.currents rtol=1e-12
    @test norm(kept.z_mom * compact.currents -
        kept.z_mom * kept.currents) < 1e-12
    matrix_payload=sizeof(kept.z_mom)
    retained_payload=_retained_planar_matrix_buffer_payload(prob,true,matrix_payload)
    compact_payload=_retained_planar_matrix_buffer_payload(prob,false,matrix_payload)
    @test retained_payload-compact_payload >= matrix_payload
end

@testset "planar: normalized admittance conversion" begin
    for n in (1, 2, 7)
        Y = ComplexF64[complex(p == q ? 0.02 : 0.001,
            0.0001 * (p + 2q)) for p in 1:n, q in 1:n]
        z0 = Float64[25 + 10p for p in 1:n]
        D = Diagonal(z0)
        R = Diagonal(sqrt.(z0))
        expected = R \ ((I - D * Y) / (I + D * Y)) * R
        @test planar_y_to_s(Y, z0) ≈ expected rtol=2e-14 atol=2e-14
    end
    @test_throws ArgumentError planar_y_to_s(zeros(ComplexF64, 0, 0), Float64[])
    @test_throws ArgumentError planar_y_to_s(reshape([ComplexF64(NaN)], 1, 1), [50.0])
    @test_throws ArgumentError planar_y_to_s(ones(ComplexF64, 1, 1), [50.0 + Inf*im])
end

@testset "planar: Touchstone validation preserves files" begin
    mktempdir() do dir
        path = joinpath(dir, "existing.s2p")
        original = "existing measurement data\n"
        write(path, original)
        S = Matrix{ComplexF64}(I, 2, 2)
        @test_throws DimensionMismatch write_touchstone(path, [1e9, 2e9],
            [S, zeros(ComplexF64, 3, 3)])
        @test read(path, String) == original
        for (freqs, matrices, z0) in ((Float64[], Matrix{ComplexF64}[], 50.0),
                ([NaN], [S], 50.0), ([-1.0], [S], 50.0),
                ([1e9], [S .* NaN], 50.0), ([1e9], [S], -50.0))
            @test_throws ArgumentError write_touchstone(path, freqs, matrices, z0)
            @test read(path, String) == original
        end
        @test write_touchstone(path, [0.0, 1e9], [S, S]) == path
    end
end
