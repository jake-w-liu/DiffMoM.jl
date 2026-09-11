using Test
using LinearAlgebra
using SparseArrays
using Random
using DiffMoM

@testset "Retained checked scattering solves" begin
    rng = MersenneTwister(17101)
    mesh = make_rect_plate(0.10, 0.08, 2, 2)
    rwg = build_rwg(mesh)
    n = rwg.nedges
    matrix = randn(rng, ComplexF64, n, n) + 4n * I
    expected = randn(rng, ComplexF64, n, 3)
    rhs = matrix * expected[:, 1]
    factor = DiffMoM._factor_dense_linear_system(matrix, ComplexF64, "retained test")
    options = (tol=1e-11, maxiter=100, check_gmres_convergence=true,
               check_true_residual=true, true_residual_factor=10.0)
    state = DiffMoM._retain_scattering_state(
        mesh, rwg, matrix, factor, nothing, rhs, expected[:, 1],
        3.0e8, 299792458.0, 3, :dense_direct, options, 10^8)
    @test state.last_solve === nothing

    # Manufacturing b=A*x supplies an expectation independent of the solver.
    # The diagonally dominant matrix has small condition number; 1e-12 allows
    # only ordinary factor/reduction rounding and rejects a missing conjugate.
    @test DiffMoM.solve_prepared!(state) ≈ expected[:, 1] rtol=1e-12
    @test state.factorization === factor
    @test state.operator === matrix
    @test state.rhs_revision == 1
    input_batch = matrix * expected
    @test DiffMoM.solve_prepared!(state; rhs=input_batch) ≈ expected rtol=1e-12
    input_batch[1] += 1
    @test state.rhs != input_batch
    saved_rhs = copy(state.rhs)
    saved_current = copy(state.I_coeffs)
    revision = state.rhs_revision
    @test DiffMoM.solve_prepared_adjoint!(state, matrix' * expected) ≈ expected rtol=1e-12
    @test state.last_solve.adjoint
    @test maximum(state.last_solve.true_residuals) < 1e-12
    @test state.rhs == saved_rhs
    @test state.I_coeffs == saved_current
    @test state.rhs_revision == revision

    @test_throws DimensionMismatch DiffMoM.solve_prepared!(state; rhs=zeros(n + 1))
    @test_throws ArgumentError DiffMoM.solve_prepared!(state; rhs=zeros(n, 0))
    @test_throws ArgumentError DiffMoM.solve_prepared!(state; rhs=fill(NaN, n))
    @test_throws ArgumentError DiffMoM.solve_prepared_adjoint!(state, fill(Inf, n))
    @test_throws ArgumentError DiffMoM.solve_prepared!(state; max_work_bytes=1)
    @test_throws ArgumentError DiffMoM.solve_prepared!(state; max_work_bytes=big(typemax(Int)) + 1)
    @test state.rhs == saved_rhs
    @test state.I_coeffs == saved_current
    @test state.rhs_revision == revision

    @test iszero(norm(DiffMoM.solve_prepared!(state; rhs=zeros(n))))
    @test state.last_solve.true_residuals == [0.0]
    matrix[1, 2] += 0.1im
    @test_throws ArgumentError DiffMoM.solve_prepared!(state)
    matrix[1, 2] -= 0.1im
    # Restore the exact original value before exercising the geometry guard.
    state.signature = DiffMoM._retained_configuration_signature(state)
    mesh.xyz[1, 1] += 0.001
    @test_throws ArgumentError DiffMoM.solve_prepared_adjoint!(state, rhs)

    # An exceptional factor token must retain scaling and adjoint semantics.
    tiny_mesh = make_rect_plate(0.1, 0.1, 1, 1)
    tiny_rwg = build_rwg(tiny_mesh)
    extreme = reshape(ComplexF64[complex(1e-290, 2e-290)], 1, 1)
    extreme_factor = DiffMoM._factor_dense_linear_system(extreme, ComplexF64, "extreme test")
    extreme_rhs = ComplexF64[complex(3e-290, -1e-290)]
    oracle = ComplexF64.(Complex{BigFloat}.(extreme) \ Complex{BigFloat}.(extreme_rhs))
    extreme_state = DiffMoM._retain_scattering_state(
        tiny_mesh, tiny_rwg, extreme, extreme_factor, nothing, extreme_rhs, oracle,
        3e8, 299792458.0, 3, :dense_direct, options, 10^8)
    @test DiffMoM.solve_prepared!(extreme_state) ≈ oracle rtol=1e-12
    adjoint_oracle = ComplexF64.(Complex{BigFloat}.(extreme)' \ Complex{BigFloat}.(extreme_rhs))
    @test DiffMoM.solve_prepared_adjoint!(extreme_state, extreme_rhs) ≈ adjoint_oracle rtol=1e-12

    # GMRES uses the same physical equations, including the adjoint path.
    iterative_matrix = randn(rng, ComplexF64, n, n) + 4n * I
    iterative_mesh = make_rect_plate(0.10, 0.08, 2, 2)
    iterative_rwg = build_rwg(iterative_mesh)
    iterative_state = DiffMoM._retain_scattering_state(
        iterative_mesh, iterative_rwg, iterative_matrix, nothing, nothing,
        iterative_matrix * expected[:, 1], expected[:, 1],
        3e8, 299792458.0, 3, :dense_gmres, options, 10^8)
    @test DiffMoM.solve_prepared!(iterative_state) ≈ expected[:, 1] rtol=1e-10
    @test DiffMoM.solve_prepared_adjoint!(iterative_state, iterative_matrix' * expected) ≈ expected rtol=1e-10
end

@testset "Retained workflow integration and shared-state adjoints" begin
    rng = MersenneTwister(17103)
    frequency = 3e8
    k = 2pi * frequency / 299792458.0
    source = make_plane_wave(Vec3(0, 0, -k), 1.0, Vec3(1, 0, 0))
    for method in (:dense_direct, :dense_gmres, :aca_gmres, :mlfma)
        mesh = make_rect_plate(0.1, 0.08, 2, 2)
        ordinary = solve_scattering(
            mesh, frequency, source; method=method,
            preconditioner=:none, gmres_tol=1e-11, verbose=false)
        prepared = solve_scattering(
            mesh, frequency, source; method=method,
            preconditioner=:none, gmres_tol=1e-11, verbose=false,
            return_state=true)
        @test ordinary isa ScatteringResult
        @test prepared.result isa ScatteringResult
        @test prepared.state isa DiffMoM.RetainedScatteringState
        state = prepared.state
        @test state.last_solve === nothing
        @test prepared.result.I_coeffs ≈ ordinary.I_coeffs rtol=1e-10
        factor_token = state.factorization
        operator_token = state.operator
        @test state.excitation === source
        @test DiffMoM.solve_prepared!(state) ≈ ordinary.I_coeffs rtol=1e-10
        @test state.factorization === factor_token
        @test state.operator === operator_token
        @test state.excitation === source
        expected = randn(rng, ComplexF64, state.rwg.nedges, 3)
        rhs = hcat((state.operator' * expected[:, j] for j in 1:3)...)
        forward_rhs = copy(state.rhs)
        revision = state.rhs_revision
        tasks = [Threads.@spawn(DiffMoM.solve_prepared_adjoint!(state, rhs[:, j])) for j in 1:3]
        @test hcat(fetch.(tasks)...) ≈ expected rtol=1e-9
        @test state.rhs == forward_rhs
        @test state.rhs_revision == revision
        new_rhs = state.operator * expected[:, 1]
        @test DiffMoM.solve_prepared!(state; rhs=new_rhs) ≈ expected[:, 1] rtol=1e-9
        @test state.excitation === nothing
        @test state.rhs_revision == revision + 1
        old_frequency = state.frequency_hz
        state.frequency_hz *= 2
        @test_throws ArgumentError DiffMoM.solve_prepared!(state)
        state.frequency_hz = old_frequency
        old_order = state.quad_order
        state.quad_order = 7
        @test_throws ArgumentError DiffMoM.solve_prepared_adjoint!(state, forward_rhs)
        state.quad_order = old_order
        if method === :aca_gmres
            state.operator.dense_blocks[1].data[1] += 1
        elseif method === :mlfma
            nonzeros(state.operator.Z_near)[1] += 1
        else
            state.operator[1] += 1
        end
        @test_throws ArgumentError DiffMoM.solve_prepared!(state)
    end
end

@testset "Fixed-facet RWG nesting" begin
    rng = MersenneTwister(17102)
    plate = make_rect_plate(0.4, 0.3, 3, 2)
    bent = make_rect_plate(0.4, 0.3, 2, 2)
    # The x=0 mesh line is a conforming seam in this even subdivision.
    for vertex in axes(bent.xyz, 2)
        x = bent.xyz[1, vertex]
        if x > 0
            bent.xyz[1, vertex] = cos(pi / 3) * x
            bent.xyz[3, vertex] = sin(pi / 3) * x
        end
    end
    # Removing a complete interior cell preserves a rectangular slot boundary.
    slot_base = make_rect_plate(0.4, 0.3, 4, 4)
    keep = [t for t in axes(slot_base.tri, 2) if !(
        0 < sum(slot_base.xyz[1, slot_base.tri[:, t]]) / 3 < 0.1 &&
        0 < sum(slot_base.xyz[2, slot_base.tri[:, t]]) / 3 < 0.075)]
    slot = TriMesh(copy(slot_base.xyz), slot_base.tri[:, keep])
    reversed = TriMesh(copy(plate.xyz), plate.tri[[1, 3, 2], end:-1:1])
    for mesh in (plate, bent, slot, reversed)
        pair = DiffMoM.build_nested_rwg_pair(mesh)
        coarse = pair.coarse_rwg
        fine = pair.fine_rwg
        @test ntriangles(pair.fine_mesh) == 4ntriangles(mesh)
        @test size(pair.P) == (fine.nedges, coarse.nedges)
        @test size(pair.Q) == (fine.nedges, fine.nedges - coarse.nedges)
        @test rank(Matrix(hcat(pair.P, pair.Q))) == fine.nedges
        @test cond(Matrix(hcat(pair.P, pair.Q))) < 100
        @test pair.pivot_ratio > sqrt(eps(Float64))
        coefficients = randn(rng, ComplexF64, coarse.nedges)
        lifted = pair.P * coefficients
        for triangle in 1:ntriangles(pair.fine_mesh)
            parent = pair.parent_triangles[triangle]
            vertices = pair.fine_mesh.xyz[:, pair.fine_mesh.tri[:, triangle]]
            for barycentric in ([1 / 3, 1 / 3, 1 / 3], [0.2, 0.3, 0.5])
                point = Vec3(vertices * barycentric)
                expected_field = sum(coefficients[e] * eval_rwg(coarse, e, point, parent)
                                     for e in 1:coarse.nedges)
                actual_field = sum(lifted[e] * eval_rwg(fine, e, point, triangle)
                                   for e in 1:fine.nedges)
                # Affine RWG reconstruction on shape-regular small meshes.
                @test actual_field ≈ expected_field rtol=1e-11 atol=1e-12
            end
            expected_divergence = sum(coefficients[e] * div_rwg(coarse, e, parent)
                                      for e in 1:coarse.nedges)
            actual_divergence = sum(lifted[e] * div_rwg(fine, e, triangle)
                                    for e in 1:fine.nedges)
            @test actual_divergence ≈ expected_divergence rtol=1e-11 atol=1e-11
        end
        # No boundary-edge unknown is introduced, so the normal trace at
        # every exterior/slot edge remains zero after injection.
        for ((first, second), adjacent) in DiffMoM._build_edge_triangle_map(pair.fine_mesh)
            length(adjacent) == 1 || continue
            triangle = adjacent[1][1]
            opposite = only(setdiff(pair.fine_mesh.tri[:, triangle], [first, second]))
            a = Vec3(pair.fine_mesh.xyz[:, first])
            b = Vec3(pair.fine_mesh.xyz[:, second])
            point = (a + b) / 2
            tangent = (b - a) / norm(b - a)
            normal = point - Vec3(pair.fine_mesh.xyz[:, opposite])
            normal -= dot(normal, tangent) * tangent
            normal /= norm(normal)
            boundary_field = sum(lifted[e] * eval_rwg(fine, e, point, triangle)
                                 for e in 1:fine.nedges)
            @test abs(sum(normal .* boundary_field)) < 1e-11 * max(1, norm(coefficients))
        end
        reversed_basis = deepcopy(coarse)
        reversed_basis.tplus[1], reversed_basis.tminus[1] =
            reversed_basis.tminus[1], reversed_basis.tplus[1]
        reversed_basis.vplus_opp[1], reversed_basis.vminus_opp[1] =
            reversed_basis.vminus_opp[1], reversed_basis.vplus_opp[1]
        reversed_basis.area_plus[1], reversed_basis.area_minus[1] =
            reversed_basis.area_minus[1], reversed_basis.area_plus[1]
        reversed_basis.evert[:, 1] .= reverse(reversed_basis.evert[:, 1])
        changed_injection = DiffMoM._midpoint_rwg_injection(
            reversed_basis, fine, pair.parent_triangles)
        @test changed_injection[:, 1] ≈ -pair.P[:, 1] atol=1e-12
        @test changed_injection[:, 2:end] ≈ pair.P[:, 2:end] atol=1e-12
    end
    @test_throws ArgumentError DiffMoM.build_nested_rwg_pair(plate; max_work_bytes=1)
    @test_throws ArgumentError DiffMoM.build_nested_rwg_pair(plate; rank_rtol=0)
    @test_throws ArgumentError DiffMoM.build_nested_rwg_pair(plate; rank_rtol=NaN)
    @test_throws ArgumentError DiffMoM.build_nested_rwg_pair(plate; rank_rtol=1)
    @test_throws ArgumentError DiffMoM.build_nested_rwg_pair(
        plate; max_work_bytes=big(typemax(Int)) + 1)
end

@testset "Nested Galerkin block and complete field identities" begin
    rng = MersenneTwister(17104)
    pair = DiffMoM.build_nested_rwg_pair(make_rect_plate(0.1, 0.1, 1, 1))
    p, q = pair.P, pair.Q
    n = pair.fine_rwg.nedges
    fine_matrix = randn(rng, ComplexF64, n, n) + 4n * I
    fine_rhs = randn(rng, ComplexF64, n)
    output = randn(rng, ComplexF64, 4, n)
    a, b = p' * fine_matrix * p, p' * fine_matrix * q
    c, d = q' * fine_matrix * p, q' * fine_matrix * q
    coarse_rhs, unresolved_rhs = p' * fine_rhs, q' * fine_rhs
    coarse_current = a \ coarse_rhs
    lift = a \ b
    unresolved_current = (d - c * lift) \ (unresolved_rhs - c * coarse_current)
    enriched = fine_matrix \ fine_rhs
    reconstructed = p * (coarse_current - lift * unresolved_current) + q * unresolved_current
    @test reconstructed ≈ enriched rtol=1e-12
    @test norm(a * (-lift * unresolved_current) + b * unresolved_current) < 1e-12
    h = output * q - output * p * lift
    @test output * p * coarse_current + h * unresolved_current ≈ output * enriched rtol=1e-12
    lambda = a' \ (output * p)'
    @test h ≈ output * q - lambda' * b rtol=1e-12
    # An omitted unresolved output term must measurably fail this oracle.
    incomplete = output * p * (coarse_current - lift * unresolved_current)
    @test norm(incomplete - output * enriched) > 1e-3 * norm(output * enriched)
end
