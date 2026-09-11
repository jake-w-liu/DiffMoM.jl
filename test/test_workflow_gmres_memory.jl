using Test
using LinearAlgebra
using DiffMoM

struct GMRESMemoryRHS <: AbstractMatrix{ComplexF64}
    values::Matrix{ComplexF64}
    materializations::Base.RefValue{Int}
end
Base.size(rhs::GMRESMemoryRHS) = size(rhs.values)
Base.getindex(rhs::GMRESMemoryRHS, i::Int, j::Int) = rhs.values[i, j]
function Base.Matrix{ComplexF64}(rhs::GMRESMemoryRHS)
    rhs.materializations[] += 1
    return copy(rhs.values)
end

@testset "Workflow and retained Krylov memory" begin
    frequency = 3e9
    k = 2pi * frequency / 299792458.0
    mesh = make_rect_plate(0.03, 0.02, 4, 4)
    source = make_plane_wave(k * normalize(Vec3(-0.2, -0.3, -1.0)),
        1.0, normalize(Vec3(-0.3, 0.2, 0.0)))
    physical = (; quad_order=7, check_resolution=false, verbose=false)
    solver = (; preconditioner=:none, gmres_tol=1e-10, true_residual_factor=10.0)
    direct = solve_scattering(mesh, frequency, source;
        physical..., method=:dense_direct, return_state=true)
    baseline = solve_scattering(mesh, frequency, source;
        physical..., solver..., method=:dense_gmres, gmres_maxiter=300)
    _, default_stats = solve_gmres(direct.state.operator, direct.state.rhs;
        memory=20, maxiter=300, tol=1e-10, true_residual_factor=10.0)
    @test baseline isa ScatteringResult
    @test baseline.gmres_iters == default_stats.niter
    @test_throws r"GMRES did not converge consistently" solve_scattering(
        mesh, frequency, source; physical..., solver..., method=:dense_gmres,
        gmres_memory=1, gmres_maxiter=300)
    for invalid in (0, -1)
        @test_throws r"GMRES memory must be at least 1" solve_scattering(
            mesh, frequency, source; physical..., solver..., method=:dense_gmres,
            gmres_memory=invalid, max_dense_matrix_bytes=1)
    end

    retained = nothing
    for method in (:dense_gmres, :aca_gmres, :mlfma)
        prepared = solve_scattering(mesh, frequency, source;
            physical..., solver..., method, gmres_memory=40,
            gmres_maxiter=80, return_state=true)
        state = prepared.state
        @test state.solver_options.memory == 40
        @test prepared.result.gmres_iters <= 80
        @test norm(state.operator * prepared.result.I_coeffs - state.rhs) /
              norm(state.rhs) <= 1e-9 # tol times the declared true-residual factor.
        repeated = solve_prepared!(state)
        @test norm(state.operator * repeated - state.rhs) / norm(state.rhs) <= 1e-9
        method == :dense_gmres && (retained = state)
    end

    rhs = hcat(retained.rhs, conj.(retained.rhs) + 0.1im .* reverse(retained.rhs))
    tracked = GMRESMemoryRHS(rhs, Ref(0))
    base = DiffMoM._retained_solve_work_bytes(retained, size(rhs, 2))
    small_budget = base + DiffMoM._gmres_workspace_bytes(retained.rwg.nedges, 20)
    @test small_budget < base + DiffMoM._gmres_workspace_bytes(retained.rwg.nedges, 40)
    previous_rhs, previous_current = copy(retained.rhs), copy(retained.I_coeffs)
    @test_throws r"GMRES Krylov workspace" solve_prepared!(
        retained; rhs=tracked, max_work_bytes=small_budget)
    @test tracked.materializations[] == 0
    @test retained.rhs == previous_rhs
    @test retained.I_coeffs == previous_current
    current = solve_prepared!(retained; rhs)
    @test norm(retained.operator * current - rhs) / norm(rhs) <= 1e-9
    previous_current = copy(retained.I_coeffs)
    adjoint_current = solve_prepared_adjoint!(retained, rhs)
    @test norm(adjoint(retained.operator) * adjoint_current - rhs) / norm(rhs) <= 1e-9
    @test retained.I_coeffs == previous_current
    @test all(<=(1e-9), retained.last_solve.true_residuals)
    retained.solver_options = merge(retained.solver_options, (memory=20,))
    @test_throws r"retained scattering configuration changed" solve_prepared!(retained)

    # This modest mesh has 5208 unknowns: a full basis exceeds the 512 MiB
    # Krylov ceiling. The refusal must precede even a one-byte ACA allowance.
    large = make_rect_plate(0.3, 0.2, 42, 42)
    @test DiffMoM._gmres_workspace_bytes(build_rwg(large).nedges, 10_000) >
          DiffMoM._DEFAULT_MAX_GMRES_WORKSPACE_BYTES
    @test_throws r"GMRES Krylov workspace" solve_scattering(
        large, frequency, source; physical..., solver..., method=:aca_gmres,
        gmres_memory=10_000, max_aca_storage_bytes=1)
end
