module FocusedScientificKernelTests

using Test
using LinearAlgebra
using Random
using SpecialFunctions: besselh
using DiffMoM

# One eight-triangle fixture covers shared-edge, vertex-neighbor and separated
# supports. Larger convergence studies belong to the full validation runner.
@testset "Focused surface geometry, fields and sensitivities" begin
    mesh = make_rect_plate(0.1, 0.08, 2, 2)
    rwg = build_rwg(mesh)
    k, eta = 2pi * 3e8 / 299792458.0, 376.730313668
    n = rwg.nedges
    rng = Xoshiro(1729)

    @testset "Geometry and polynomial quadrature" begin
        @test (nvertices(mesh), ntriangles(mesh), n) == (9, 8, 8)
        @test sum(triangle_area(mesh, t) for t in 1:8) ≈ 0.008
        @test all(t -> triangle_normal(mesh, t) ≈ Vec3(0, 0, 1), 1:8)
        for edge in 1:n
            plus, minus = basis_triangles(rwg, edge)
            @test triangle_area(mesh, plus) * div_rwg(rwg, edge, plus) ≈ rwg.len[edge]
            @test triangle_area(mesh, minus) * div_rwg(rwg, edge, minus) ≈ -rwg.len[edge]
        end
        points, weights = tri_quad_rule(3)
        # Independent exact monomial integrals on the unit reference triangle.
        for a in 0:2, b in 0:(2-a)
            integral = sum(w * p[1]^a * p[2]^b for (p, w) in zip(points, weights))
            @test integral ≈ factorial(a) * factorial(b) / factorial(a+b+2)
        end
        mapped = tri_quad_points(mesh, 1, points)
        @test sum(mapped) / length(mapped) ≈ triangle_center(mesh, 1)
        @test_throws ArgumentError make_rect_plate(0.1, 0.08, 0, 2)
    end

    dense = assemble_Z_efie(mesh, rwg, k)
    operator = matrixfree_efie_operator(mesh, rwg, k)
    x, y = randn(rng, ComplexF64, n), randn(rng, ComplexF64, n)
    @testset "EFIE forward, adjoint and solve" begin
        @test dense ≈ transpose(dense) rtol=1e-12
        @test operator * x ≈ dense * x rtol=1e-12
        @test adjoint(operator) * y ≈ adjoint(dense) * y rtol=1e-12
        @test dot(y, operator * x) ≈ dot(adjoint(operator) * y, x) rtol=1e-12
        aliased = copy(x)
        mul!(aliased, operator, aliased, 0.7+0.2im, -0.1+0.3im)
        @test aliased ≈ (0.7+0.2im) .* (dense*x) .+ (-0.1+0.3im) .* x rtol=1e-12
        incident = assemble_v_plane_wave(mesh, rwg, Vec3(0, 0, -k), 1.0, Vec3(1, 0, 0))
        direct = solve_forward(dense, incident)
        iterative = solve_forward(operator, incident;
            solver=:gmres, gmres_tol=1e-11, gmres_maxiter=30, gmres_memory=n)
        @test norm(dense*direct - incident) <= 1e-12norm(incident)
        @test iterative ≈ direct rtol=1e-9
        @test_throws DimensionMismatch mul!(zeros(ComplexF64, n-1), operator, x)
    end

    grid = make_sph_grid(3, 4)
    radiation = radiation_vectors(mesh, rwg, grid, k)
    polarization = pol_linear_x(grid)
    q = build_Q(radiation, grid, polarization)
    @testset "Far field and power use the same physical normalization" begin
        field = compute_farfield(radiation, x, length(grid.w))
        @test all(>(0), grid.w)
        @test maximum(abs(dot(grid.rhat[:, j], field[:, j])) for j in eachindex(grid.w)) <=
              1e-12norm(field)
        angular_energy = sum(grid.w[j] * sum(abs2, field[:, j]) for j in eachindex(grid.w))
        @test radiated_power(field, grid; eta0=eta) ≈ angular_energy / (2eta) rtol=1e-12
        @test ishermitian(q)
        @test minimum(eigvals(Hermitian(q))) >= -64eps(Float64)*opnorm(q)
        @test real(dot(x, q*x)) ≈ projected_power(field, grid, polarization) rtol=1e-12
        @test build_Q_operator(radiation, grid, polarization) * x ≈ q*x rtol=1e-12
    end

    @testset "Impedance adjoints agree with independent central differences" begin
        mass = [Matrix(assemble_rwg_gram(mesh, rwg))]
        rhs = assemble_v_plane_wave(mesh, rwg, Vec3(0, 0, -k), 1.0, Vec3(1, 0, 0))
        theta, step = [35.0], 1e-3
        for reactive in (false, true)
            factor = reactive ? im : 1.0
            # Assemble the perturbed equation explicitly for the FD oracle.
            objective(t) = begin
                current = (dense - factor*t*mass[1]) \ rhs
                real(dot(current, q*current))
            end
            loaded = dense - factor*theta[1]*mass[1]
            current = solve_forward(loaded, rhs)
            adjoint_current = solve_adjoint(loaded, q, current)
            derivative = only(gradient_impedance(mass, current, adjoint_current; reactive))
            finite_difference = (objective(theta[1]+step) - objective(theta[1]-step)) / (2step)
            @test derivative ≈ finite_difference rtol=2e-6
            composite = ImpedanceLoadedOperator(operator, mass, theta, reactive)
            @test composite*x ≈ loaded*x rtol=1e-12
            @test adjoint(composite)*y ≈ adjoint(loaded)*y rtol=1e-12
        end
    end
end

@testset "Focused four-cell TM volume equation" begin
    mesh = Mesh2D((-0.05, 0.05), (-0.05, 0.05), 2, 2)
    k, contrast = 12.0, [0.5, 1.0, 0.0, 0.8]
    incident = planewave_2d(mesh, k, 0.3)
    result = solve_vie_2d(mesh, k, contrast, incident)
    radius = sqrt(mesh.cell_area/pi)
    self = (-im*pi/(2k^2)) * (k*radius*besselh(1, 2, k*radius) - 2im/pi)
    # Hankel expressions independently specify the discretized integral.
    integral = ComplexF64[i == j ? self :
        (-im/4)*besselh(0, 2, k*norm(mesh.centers[i]-mesh.centers[j]))*mesh.cell_area
        for i in 1:4, j in 1:4]
    expected = Matrix{ComplexF64}(I, 4, 4) - k^2*integral*Diagonal(contrast)
    @test result.D ≈ integral rtol=1e-13
    @test result.Z ≈ expected rtol=1e-13
    @test result.E_total ≈ expected \ incident rtol=1e-12
    @test norm(expected*result.E_total - incident) < 1e-12norm(incident)
    free_space = solve_vie_2d(mesh, k, zeros(4), incident)
    @test free_space.E_total == incident
    @test_throws ArgumentError solve_vie_2d(mesh, k, contrast, incident; max_output_bytes=1)
end

@testset "Focused DDA dense, FFT and tensor contracts" begin
    grid = VoxelGrid3D((-.1, .1), (-.1, .1), (-.1, .1), 2, 2, 2)
    k, epsr = 2pi, 2.5-0.05im
    dense, _, _ = assemble_dda_3d(grid, k, epsr)
    operator = dda_operator_3d(grid, k, epsr)
    fft_operator = fft_dda_operator_3d(grid, k, epsr)
    x = randn(Xoshiro(1730), ComplexF64, size(dense, 2))
    @test operator*x ≈ dense*x rtol=1e-12
    @test adjoint(operator)*x ≈ adjoint(dense)*x rtol=1e-12
    @test fft_operator*x ≈ dense*x rtol=1e-12
    alpha, beta = 0.7+0.2im, -0.1+0.3im
    aliased = copy(x)
    mul!(aliased, fft_operator, aliased, alpha, beta)
    @test aliased ≈ alpha .* (dense*x) .+ beta .* x rtol=1e-12

    incident = planewave_dda_3d(grid, Vec3(0, 0, k), 1.0+0.2im, Vec3(1, 0.3, 0))
    direct = solve_dda_3d(grid, k, epsr, incident)
    iterative = solve_dda_3d(grid, k, epsr, incident;
        solver=:gmres, tol=1e-11, maxiter=40, memory=24)
    @test reduce(vcat, direct.E_total) ≈ dense \ reduce(vcat, incident) rtol=1e-12
    @test reduce(vcat, iterative.E_total) ≈ reduce(vcat, direct.E_total) rtol=1e-9
    # Full off-diagonal material coupling must survive both operator paths.
    tensor = ComplexF64[2.4-0.02im 0.03 0.01im; 0.03 1.7-0.01im 0; 0.01im 0 1.2]
    tensor_dense, _, _ = assemble_dda_3d(grid, k, tensor)
    tensor_operator = dda_operator_3d(grid, k, tensor)
    tensor_fft = fft_dda_operator_3d(grid, k, tensor)
    @test tensor_operator*x ≈ tensor_dense*x rtol=1e-12
    @test tensor_fft*x ≈ tensor_dense*x rtol=1e-12
    @test adjoint(tensor_operator)*x ≈ adjoint(tensor_dense)*x rtol=1e-12

    # An isolated dipole has a closed-form Rayleigh far field.
    single = VoxelGrid3D((-.05, .05), (-.05, .05), (-.05, .05), 1, 1, 1)
    dipole = solve_dda_3d(single, k, epsr, [CVec3(1, 0, 0)])
    polarizability = 3single.volumes[1]*(epsr-1)/(epsr+2)
    @test farfield_dda_3d(dipole, Vec3(0, 0, 1)) ≈
          CVec3(k^2*polarizability/(4pi), 0, 0) rtol=1e-12
    @test farfield_dda_3d(dipole, Vec3(1, 0, 0)) == CVec3(0, 0, 0)
end

@testset "Focused Gaussian conditioning and calibration boundaries" begin
    mass = ComplexF64[3 im 0; -im 4 0.2; 0 0.2 2]
    rows, observations, tau = ComplexF64[1+im 0.3im -0.2], ComplexF64[0.5+0.2im], 0.3
    prior = tau^2 * inv(mass)
    schur = rows * prior * adjoint(rows)
    expected_mean = prior * adjoint(rows) * (schur \ observations)
    expected_covariance = prior - prior * adjoint(rows) * (schur \ (rows * prior))
    conditioned = condition_error_observations(mass, rows, observations; tau)
    @test conditioned.rank == 1
    @test conditioned.mean_coefficients ≈ expected_mean rtol=1e-12
    @test rows * conditioned.mean_coefficients ≈ observations rtol=1e-12
    latent_projector = Matrix{ComplexF64}(I, 3, 3) -
                       conditioned.right_basis * adjoint(conditioned.right_basis)
    physical_factor = zeros(ComplexF64, 3, 3)
    physical_factor[conditioned.permutation, :] = tau * (adjoint(conditioned.lower) \ latent_projector)
    @test physical_factor * adjoint(physical_factor) ≈ expected_covariance rtol=1e-12
    @test norm(rows * physical_factor) < 1e-12
    @test_throws ArgumentError condition_error_observations(
        mass, vcat(rows, 2rows), vcat(observations, 2observations .+ 1); tau)

    context = RCSCalibrationContext(algorithm_hash=repeat("a", 64),
        training_revision="fixed", distribution_id="unit-test", reference_protocol="analytic",
        look_id="one-look", look_count=1, levels=(0,), field_floor=1e-8,
        absolute_field_tolerance=1e-12)
    ids = ["case-$i" for i in 1:19]
    profile = calibrate_rcs(Float64.(1:19), context;
        calibration_case_ids=ids, epsilon=1//20)
    @test (profile.quantile_index, profile.quantile) == (19, 19.0)
    insufficient = calibrate_rcs(Float64.(1:18), context;
        calibration_case_ids=ids[1:18], epsilon=1//20)
    @test isinf(insufficient.quantile)
    @test_throws ArgumentError calibrate_rcs([1.0], context;
        calibration_case_ids=["same"], training_case_ids=["same"])
    # Radial extrema give exact bounds without random perturbation trials.
    bounds = rcs_ball_bounds(ComplexF64[3, 4], 1.0)
    @test prevfloat(64pi, 4) <= bounds.lower <= 64BigFloat(pi)
    @test 144BigFloat(pi) <= bounds.upper <= nextfloat(144pi, 4)
end

end # module
