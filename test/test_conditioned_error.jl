using Test
using LinearAlgebra
using SparseArrays
using Random
using DiffMoM

@testset "Weighted RWG Gram: analytic triangle moments" begin
    mesh = make_rect_plate(0.1, 0.08, 2, 2)
    rwg = build_rwg(mesh)
    weights = collect(range(0.5, 1.5; length=ntriangles(mesh)))
    gram = assemble_rwg_gram(mesh, rwg; triangle_weights=weights)
    oracle = zeros(rwg.nedges, rwg.nedges)
    for triangle in 1:ntriangles(mesh)
        vertices = [Vec3(mesh.xyz[:, mesh.tri[i, triangle]]) for i in 1:3]
        area = norm(cross(vertices[2] - vertices[1], vertices[3] - vertices[1])) / 2
        centroid = sum(vertices) / 3
        # Uniform barycentric moments: E[lambda_i^2]=1/6 and
        # E[lambda_i*lambda_j]=1/12 for different indices.
        radius_second_moment = (sum(sum(abs2, v) for v in vertices) +
                                sum(abs2, sum(vertices))) / 12
        for i in 1:rwg.nedges, j in 1:rwg.nedges
            si = triangle == rwg.tplus[i] ? 1 : triangle == rwg.tminus[i] ? -1 : 0
            sj = triangle == rwg.tplus[j] ? 1 : triangle == rwg.tminus[j] ? -1 : 0
            iszero(si * sj) && continue
            oi = Vec3(mesh.xyz[:, si == 1 ? rwg.vplus_opp[i] : rwg.vminus_opp[i]])
            oj = Vec3(mesh.xyz[:, sj == 1 ? rwg.vplus_opp[j] : rwg.vminus_opp[j]])
            li = norm(mesh.xyz[:, rwg.evert[1, i]] - mesh.xyz[:, rwg.evert[2, i]])
            lj = norm(mesh.xyz[:, rwg.evert[1, j]] - mesh.xyz[:, rwg.evert[2, j]])
            oracle[i, j] += weights[triangle] * si * sj * li * lj / (4area) *
                (radius_second_moment - dot(centroid, oi + oj) + dot(oi, oj))
        end
    end
    @test isapprox(Matrix(gram), oracle; rtol=1e-12, atol=1e-17)
    @test ishermitian(gram)
    @test minimum(eigvals(Hermitian(Matrix(gram)))) > 0
    @test_throws ArgumentError assemble_rwg_gram(mesh, rwg; triangle_weights=zeros(length(weights)))
    @test_throws ArgumentError assemble_rwg_gram(mesh, rwg; triangle_weights=fill(Inf, length(weights)))
    @test_throws DimensionMismatch assemble_rwg_gram(mesh, rwg; triangle_weights=[1.0])
    @test_throws ArgumentError assemble_rwg_gram(mesh, rwg; max_work_bytes=1)
end

@testset "Inverse-mass proper-complex conditioning" begin
    rng = MersenneTwister(17120)
    m = 6
    b = randn(rng, ComplexF64, m, m)
    mass = Hermitian(adjoint(b) * b + m * I)
    v = randn(rng, ComplexF64, 2, m)
    t = randn(rng, ComplexF64, 2)
    tau = 0.3
    c0 = tau^2 * inv(Matrix(mass))
    j = v * c0 * adjoint(v)
    expected_mean = c0 * adjoint(v) * (j \ t)
    expected_cov = c0 - c0 * adjoint(v) * (j \ (v * c0))
    conditioned = condition_error_observations(mass, v, t; tau=tau)
    @test conditioned.rank == 2
    @test isapprox(conditioned.mean_coefficients, expected_mean; rtol=1e-12)
    @test isapprox(v * conditioned.mean_coefficients, t; rtol=1e-12)
    @test_throws ArgumentError condition_error_observations(mass, v, t; tau=0)
    @test_throws ArgumentError condition_error_observations(mass, v, t; max_work_bytes=1)
    @test_throws DimensionMismatch condition_error_observations(mass, v[:, 1:5], t)
    @test_throws ArgumentError condition_error_observations(mass, fill(NaN, 2, m), t)
    @test_throws ArgumentError condition_error_observations(mass, v, fill(Inf, 2))
    @test_throws PosDefException condition_error_observations(-Matrix{Float64}(I, m, m), v, t)

    no_info = condition_error_observations(mass, zeros(ComplexF64, 0, m), ComplexF64[]; tau=tau)
    @test iszero(norm(no_info.mean_coefficients))
    @test no_info.rank == 0
    truth = randn(rng, ComplexF64, m)
    full_v = randn(rng, ComplexF64, m, m) + 3m * I
    full = condition_error_observations(mass, full_v, full_v * truth; tau=tau)
    @test full.rank == m
    @test isapprox(full.mean_coefficients, truth; rtol=1e-12)
    @test isapprox(sample_conditioned_error(full, rng, 4), repeat(truth, 1, 4); rtol=1e-12)

    single_v = v[1:1, :]
    single_t = t[1:1]
    single = condition_error_observations(mass, single_v, single_t; tau=tau)
    dependent = condition_error_observations(
        mass, vcat(single_v, 2single_v), vcat(single_t, 2single_t); tau=tau)
    @test dependent.rank == 1
    @test isapprox(dependent.mean_coefficients, single.mean_coefficients; rtol=1e-12)
    @test_throws ArgumentError condition_error_observations(
        mass, vcat(single_v, 2single_v), vcat(single_t, 2single_t .+ 1); tau=tau)
    unstable_v = zeros(ComplexF64, 2, m)
    unstable_v[1, 1], unstable_v[2, 2] = 1, 1e-18
    @test_throws ArgumentError condition_error_observations(mass, unstable_v, ComplexF64[0, 1])

    transform = randn(rng, ComplexF64, m, m) + 3m * I
    changed = condition_error_observations(
        Hermitian(adjoint(transform) * mass * transform), v * transform, t; tau=tau)
    @test isapprox(transform * changed.mean_coefficients, expected_mean; rtol=1e-11)
    # The whitened factor gives an independent covariance route; the oracle
    # above uses the subtractive small dense conditional Gaussian formula.
    function coefficient_factor(c)
        latent_factor = Matrix{ComplexF64}(I, m, m) - c.right_basis * adjoint(c.right_basis)
        return DiffMoM._unwhiten_error_columns(c.lower, c.permutation, c.tau, latent_factor)
    end
    factor = coefficient_factor(conditioned)
    @test isapprox(factor * adjoint(factor), expected_cov; rtol=1e-12)
    changed_factor = transform * coefficient_factor(changed)
    @test isapprox(changed_factor * adjoint(changed_factor), expected_cov; rtol=1e-11)
    @test isapprox(coefficient_factor(no_info) * adjoint(coefficient_factor(no_info)), c0; rtol=1e-12)
    @test norm(v * factor) < 1e-12 * norm(v) * norm(factor)

    sample_count = 100_000
    samples = sample_conditioned_error(conditioned, rng, sample_count)
    @test norm(v * samples .- t) < 1e-10 * norm(t) * sqrt(sample_count)
    centered = samples .- expected_mean
    empirical_cov = centered * adjoint(centered) / sample_count
    # Proper-complex Wishart RMS Frobenius error is tr(C)/sqrt(n);
    # six RMS units cover the simultaneous matrix check at this fixed seed.
    @test norm(empirical_cov - expected_cov) < 6real(tr(expected_cov)) / sqrt(sample_count)
    @test norm(centered * transpose(centered) / sample_count) <
        6real(tr(expected_cov)) / sqrt(sample_count)
    @test size(sample_conditioned_error(conditioned, rng, 0)) == (m, 0)
    @test_throws ArgumentError sample_conditioned_error(conditioned, rng, -1)
    @test_throws ArgumentError sample_conditioned_error(conditioned, rng, typemax(Int))
    @test_throws ArgumentError sample_conditioned_error(conditioned, rng, 100; max_work_bytes=1)
end

@testset "EFIE restriction and complete conditioned fields" begin
    mesh = make_rect_plate(0.1, 0.08, 1, 1)
    frequency = 3e8
    k = 2pi * frequency / 299792458.0
    source = make_plane_wave(Vec3(0, 0, -k), 1.0, Vec3(1, 0, 0))
    prepared = solve_scattering(mesh, frequency, source; return_state=true, verbose=false)
    pair = build_nested_rwg_pair(mesh)
    fine_a = assemble_Z_efie(pair.fine_mesh, pair.fine_rwg, k)
    fine_b = assemble_excitation(pair.fine_mesh, pair.fine_rwg, source)
    system = prepare_galerkin_error(prepared.state, pair, fine_a, fine_b)
    p, q = pair.P, pair.Q
    a, b = adjoint(p) * fine_a * p, adjoint(p) * fine_a * q
    c, d = adjoint(q) * fine_a * p, adjoint(q) * fine_a * q
    schur = d - c * (a \ b)
    residual = adjoint(q) * fine_b - c * (a \ (adjoint(p) * fine_b))
    @test isapprox(system.coarse.operator, a; rtol=1e-12)
    @test isapprox(system.residual, residual; rtol=1e-12)
    @test system.restriction_report.exact_restriction
    @test system.restriction_report.coarse_rebuilt
    @test_throws ArgumentError prepare_galerkin_error(
        prepared.state, pair, fine_a, fine_b; rebuild_on_mismatch=false)
    @test_throws ArgumentError prepare_galerkin_error(
        prepared.state, pair, fine_a, fine_b; max_work_bytes=1)
    @test_throws DimensionMismatch prepare_galerkin_error(
        prepared.state, pair, fine_a[1:end-1, 1:end-1], fine_b[1:end-1])
    m = size(q, 2)
    grid = make_sph_grid(2, 2)
    g = radiation_vectors(pair.fine_mesh, pair.fine_rwg, grid, k)
    h = g * q - g * p * (a \ b)
    mass = Matrix(adjoint(q) * assemble_rwg_gram(pair.fine_mesh, pair.fine_rwg) * q)
    tau = 0.02
    prior_cov = tau^2 * inv(mass)
    base_field = g * p * (a \ (adjoint(p) * fine_b))
    no_data = condition_discretization_error(system, Int[]; tau=tau)
    unconditioned = evaluate_error_outputs(no_data, g; row_batch_size=3)
    @test isapprox(unconditioned.mean, base_field; rtol=1e-11)
    @test isapprox(unconditioned.covariance, h * prior_cov * adjoint(h); rtol=1e-10)

    indices = [1, 3]
    partial = condition_discretization_error(system, indices; tau=tau)
    v, t = schur[indices, :], residual[indices]
    j = v * prior_cov * adjoint(v)
    expected_w = prior_cov * adjoint(v) * (j \ t)
    expected_cov = prior_cov - prior_cov * adjoint(v) * (j \ (v * prior_cov))
    outputs = evaluate_error_outputs(partial, g; row_batch_size=2)
    @test isapprox(partial.conditioning.information, v; rtol=1e-11)
    @test isapprox(outputs.mean, base_field + h * expected_w; rtol=1e-10)
    @test isapprox(outputs.covariance, h * expected_cov * adjoint(h); rtol=1e-10)
    single_batch = evaluate_error_outputs(partial, g; row_batch_size=size(g, 1))
    @test isapprox(outputs.mean, single_batch.mean; rtol=1e-12)
    @test isapprox(outputs.covariance, single_batch.covariance; rtol=1e-12)
    prepared_outputs = prepare_error_outputs(system, g; row_batch_size=2)
    cached = evaluate_error_outputs(partial, prepared_outputs; row_batch_size=3)
    @test isapprox(cached.mean, outputs.mean; rtol=1e-12)
    @test isapprox(cached.covariance, outputs.covariance; rtol=1e-12)
    @test cached.diagnostics.coarse_adjoint_rhs == 0
    @test cached.diagnostics.prepared_output_rows
    output_probes = output_residual_probes(prepared_outputs; rows=[1, 3])
    @test size(output_probes, 2) == m
    @test isapprox(output_probes * adjoint(output_probes),
                   Matrix{ComplexF64}(I, size(output_probes, 1), size(output_probes, 1)); rtol=1e-12)
    informed = condition_discretization_error(system, output_probes; tau=tau)
    @test all(isfinite, evaluate_error_outputs(informed, prepared_outputs).mean)
    @test_throws ArgumentError output_residual_probes(prepared_outputs; rows=[0])
    @test_throws ArgumentError output_residual_probes(prepared_outputs; max_work_bytes=1)
    covariance_scale = opnorm(unconditioned.covariance)
    @test minimum(eigvals(Hermitian(unconditioned.covariance - outputs.covariance))) >=
        -128eps(Float64) * size(g, 1) * covariance_scale

    full = condition_discretization_error(system, collect(1:m); tau=tau)
    full_outputs = evaluate_error_outputs(full, g)
    @test isapprox(full_outputs.mean, g * (fine_a \ fine_b); rtol=1e-10)
    @test iszero(norm(full_outputs.covariance))
    @test iszero(norm(full_outputs.square_root))
    moments = quadratic_output_moments(outputs, Matrix{Float64}(I, size(g, 1), size(g, 1)))
    @test isapprox(moments.mean, sum(abs2, outputs.mean) + real(tr(outputs.covariance)); rtol=1e-12)
    @test isapprox(moments.variance, real(tr(outputs.covariance^2)) +
        2real(dot(outputs.mean, outputs.covariance * outputs.mean)); rtol=1e-12)
    samples = sample_error_outputs(outputs, MersenneTwister(17121), 100_000)
    powers = vec(sum(abs2, samples; dims=1))
    empirical_mean = sum(powers) / length(powers)
    @test abs(empirical_mean - moments.mean) <= 6sqrt(moments.variance / length(powers))
    @test size(sample_error_outputs(outputs, MersenneTwister(1), 0)) == (size(g, 1), 0)
    @test_throws ArgumentError sample_error_outputs(outputs, MersenneTwister(1), -1)
    @test_throws ArgumentError sample_error_outputs(outputs, MersenneTwister(1), 1; max_work_bytes=1)
    @test_throws ArgumentError evaluate_error_outputs(partial, g; max_work_bytes=1)
    @test_throws ArgumentError evaluate_error_outputs(partial, g; row_batch_size=0)
    @test_throws DimensionMismatch evaluate_error_outputs(partial, g[:, 1:end-1])
    @test_throws ArgumentError condition_discretization_error(system, [0]; tau=tau)
    @test_throws ArgumentError condition_discretization_error(system, [m+1]; tau=tau)
    @test_throws ArgumentError condition_discretization_error(system, [1]; tau=tau, max_work_bytes=1)
    changed_system = prepare_galerkin_error(prepared.state, pair, fine_a, fine_b)
    changed_model = condition_discretization_error(changed_system, Int[]; tau=tau)
    @test DiffMoM._validate_error_system(changed_system) === nothing
    solve_prepared!(changed_system.coarse; rhs=2changed_system.coarse.rhs)
    @test_throws ArgumentError evaluate_error_outputs(changed_model, g)
    prepared_outputs.rows[1] += 1
    @test_throws ArgumentError evaluate_error_outputs(partial, prepared_outputs)
    pair.P[1, 1] += 1
    @test_throws ArgumentError evaluate_error_outputs(partial, g)
end
