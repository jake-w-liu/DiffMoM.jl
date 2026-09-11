using Test, LinearAlgebra

@testset "configurable exact residual work" begin
    default = DiffMoM._DEFAULT_MAX_TRUE_RESIDUAL_EXACT_TERMS
    @test DiffMoM._enforce_true_residual_exact_work(3, 1, "budget";
        max_true_residual_exact_terms=4) == 4
    @test_throws ArgumentError DiffMoM._enforce_true_residual_exact_work(3, 1, "budget";
        max_true_residual_exact_terms=3)
    @test DiffMoM._enforce_true_residual_exact_work(0, 0, "budget";
        max_true_residual_exact_terms=0) == 0
    @test_throws ArgumentError DiffMoM._enforce_true_residual_exact_work(default+1, 0, "default")
    @test DiffMoM._enforce_true_residual_exact_work(default+1, 0, "override";
        max_true_residual_exact_terms=default+1) == default+1
    @test_throws ArgumentError DiffMoM._enforce_true_residual_exact_work(0, 0, "invalid";
        max_true_residual_exact_terms=-1)

    A = ComplexF64[3.1 0.4 0.2; 0.7 2.3 0.1; 0.2 0.4 1.8]
    b = ComplexF64[1+1im, 2-0.5im, -1+0.25im]
    expected, expected_adjoint = setprecision(BigFloat, 4352) do
        high_A, high_b = Complex{BigFloat}.(A), Complex{BigFloat}.(b)
        ComplexF64.(high_A \ high_b), ComplexF64.(adjoint(high_A) \ high_b)
    end
    @test_throws ArgumentError DiffMoM._true_residual_ratio(A, expected, b, "zero budget";
        max_true_residual_exact_terms=0)
    @test DiffMoM._true_residual_ratio(A, expected, b, "bounded residual";
        max_true_residual_exact_terms=12) < 1e-14
    for solver in (solve_gmres, solve_gmres_adjoint)
        @test_throws ArgumentError solver(A, b; tol=1e-12, memory=3,
            max_true_residual_exact_terms=0)
        values, _ = solver(A, b; tol=1e-12, memory=3,
            max_true_residual_exact_terms=12)
        oracle = solver === solve_gmres ? expected : expected_adjoint
        @test isapprox(values, oracle; rtol=1e-11, atol=0)
    end
    for solver in (solve_forward, solve_system)
        @test_throws ArgumentError solver(A, b; solver=:gmres, gmres_tol=1e-12,
            gmres_memory=3, max_true_residual_exact_terms=0)
        @test isapprox(solver(A, b; solver=:gmres, gmres_tol=1e-12, gmres_memory=3,
            max_true_residual_exact_terms=12), expected; rtol=1e-11, atol=0)
        @test_throws ArgumentError solver(A, b; max_true_residual_exact_terms=-1)
    end

    mesh = make_rect_plate(0.08, 0.06, 2, 1)
    rwg = build_rwg(mesh)
    @test rwg.nedges == 3
    options = (; tol=1e-12, maxiter=200, check_gmres_convergence=true,
        check_true_residual=true, true_residual_factor=100.0,
        max_true_residual_exact_terms=0)
    state = DiffMoM._retain_scattering_state(mesh, rwg, A, lu(A), nothing,
        copy(b), copy(expected), 3e9, 299792458.0, 3, :dense_direct,
        options, 2_000_000_000)
    previous_rhs, previous_current = copy(state.rhs), copy(state.I_coeffs)
    @test_throws ArgumentError solve_prepared!(state)
    @test state.rhs == previous_rhs
    @test state.I_coeffs == previous_current
    @test state.rhs_revision == 0
    @test isapprox(solve_prepared!(state; max_true_residual_exact_terms=12),
        expected; rtol=1e-13, atol=0)
    @test state.last_solve.max_true_residual_exact_terms == 12
    @test state.solver_options.max_true_residual_exact_terms == 0
    @test isapprox(solve_prepared_adjoint!(state, b; max_true_residual_exact_terms=12),
        expected_adjoint; rtol=1e-13, atol=0)
    @test state.rhs_revision == 1

    k = 2pi * 3e9 / 299792458.0
    wave = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
    _, physical = solve_scattering(mesh, 3e9, wave; method=:dense_direct,
        return_state=true, check_resolution=false, verbose=false,
        max_true_residual_exact_terms=12)
    @test physical.solver_options.max_true_residual_exact_terms == 12
    pair = build_nested_rwg_pair(mesh)
    _, fine = solve_scattering(pair.fine_mesh, 3e9, wave; method=:dense_direct,
        return_state=true, check_resolution=false, verbose=false)
    system = prepare_galerkin_error(physical, pair, fine.operator, fine.rhs)
    @test system.coarse.solver_options.max_true_residual_exact_terms == 12
    @test system.coarse.last_solve.max_true_residual_exact_terms == 12
end
