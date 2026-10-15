module FocusedOptimizationTests

using Test
using LinearAlgebra
using DiffMoM

@testset "Two bounded optimizer steps reduce the normalized scattering objective" begin
    mesh = make_rect_plate(0.1, 0.08, 2, 2)
    rwg = build_rwg(mesh)
    k = 2pi * 3e8 / 299792458.0
    base = assemble_Z_efie(mesh, rwg, k)
    mass = [Matrix(assemble_rwg_gram(mesh, rwg))]
    initial = [35.0]
    angles = [(theta_inc=0.0, phi_inc=0.0, pol=Vec3(1, 0, 0), weight=1.0),
              (theta_inc=pi/4, phi_inc=0.0, pol=Vec3(0, 1, 0), weight=1.0)]
    configs = build_multiangle_configs(mesh, rwg, k, angles;
        grid=make_sph_grid(3, 4), backscatter_cone=90.0)
    function angular_powers(theta)
        factor = lu(base - only(theta)*only(mass))
        return map(configs) do config
            current = factor \ config.v
            real(dot(current, config.Q*current))
        end
    end
    references = angular_powers(initial)
    theta, trace = optimize_multiangle_rcs(base, mass, configs, initial;
        maxiter=2, tol=0.0, alpha0=10.0, lb=[0.0], ub=[1000.0],
        objective=:smoothmax_log, reference_objectives=references,
        smooth_beta=6.0, verbose=false)
    @test length(trace) == 3
    @test all(state -> isfinite(state.J) && isfinite(state.gnorm), trace)
    @test 0.0 <= only(theta) <= 1000.0
    @test all(diff([state.J for state in trace]) .<= 1e-12)
    @test last(trace).J < first(trace).J - 1e-8
    # Recompute the completed objective from explicit linear solves and
    # log-sum-exp of the per-angle power ratios, independently of the helper.
    scaled = 6 .* log.(angular_powers(theta) ./ references)
    peak = maximum(scaled)
    expected = (peak + log(sum(exp.(scaled .- peak)))) / 6
    @test last(trace).J ≈ expected atol=1e-12 rtol=1e-10
    @test initial == [35.0]
end

end # module
