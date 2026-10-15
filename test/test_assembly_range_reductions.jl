using Test, DiffMoM, LinearAlgebra, SparseArrays

function _assembly_range_mul_allocation(y, operator, x)
    mul!(y, operator, x)
    return @allocated mul!(y, operator, x)
end

@testset "Composite impedance retains joint products" begin
    small = 2.0^-538
    exact = ComplexF64(-ldexp(1., -1073))
    for make_mass in (a -> fill(ComplexF64(a), 1, 1),
                      a -> sparse(fill(ComplexF64(a), 1, 1)),
                      a -> DiffMoM.LocalMassMatrix(1, [1], [1], [a]))
        masses = [make_mass(small) for _ in 1:8]
        operator = ImpedanceLoadedOperator(zeros(ComplexF64, 1, 1), masses, fill(small, 8))
        @test operator[1, 1] == exact
        @test operator * ComplexF64[1] == [exact]
        @test adjoint(operator) * ComplexF64[1] == [conj(exact)]
        @test assemble_Z_impedance(masses, fill(small, 8))[1, 1] == exact
    end

    # A child matvec itself may round to zero before another patch/alpha
    # restores its range. Matrices and parameters remain borrowed and mutable.
    minimum_value = nextfloat(0.)
    masses = [fill(ComplexF64(minimum_value), 1, 1) for _ in 1:8]
    operator = ImpedanceLoadedOperator(zeros(ComplexF64, 1, 1), masses, ones(8), true)
    @test operator * ComplexF64[.25] == ComplexF64[-2im * minimum_value]
    @test adjoint(operator) * ComplexF64[.25] == ComplexF64[2im * minimum_value]
    base = ones(ComplexF64, 1, 1)
    mutable_operator = ImpedanceLoadedOperator(base, Matrix{Float64}[], Float64[])
    x, y = ComplexF64[.25], ComplexF64[0]
    mul!(y, mutable_operator, x)
    base[1, 1] = minimum_value
    mul!(y, mutable_operator, x, 4., 0.)
    @test y == ComplexF64[minimum_value]
    mul!(y, mutable_operator, x, minimum_value, minimum_value)
    @test y == ComplexF64[0]
    @test DiffMoM._composite_scaled_output(ComplexF64(.25), ComplexF64(.75),
        minimum_value, minimum_value, false, 1) == minimum_value

    ordinary = ImpedanceLoadedOperator(ComplexF64[2 1; 1 3], [Matrix{Float64}(I, 2, 2)], [.1])
    @test _assembly_range_mul_allocation(zeros(ComplexF64, 2), ordinary, ones(ComplexF64, 2)) == 0
    previous = ComplexF64[7]
    overflowing = ImpedanceLoadedOperator(fill(ComplexF64(floatmax(Float64)), 1, 1),
                                         Matrix{Float64}[], Float64[])
    @test_throws OverflowError mul!(previous, overflowing, ComplexF64[2])
    @test previous == ComplexF64[7]
end

@testset "Static potential remains positive at far observations" begin
    vertices = (Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0))
    # For this unit right triangle and observation (0,0,d), area/d is the
    # leading potential, with a relative correction bounded by 1/(2d²).
    for distance in (1e20, 1e150)
        point = Vec3(0, 0, distance)
        value = analytical_integral_1overR(point, vertices...)
        gradient = grad_analytical_integral_1overR(point, vertices...)
        @test value > 0
        @test value ≈ .5 / distance rtol=4eps(Float64)
        @test gradient[3] ≈ -(.5 / distance) / distance rtol=4eps(Float64)
    end
    thin_triangle = (Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1e-20, 0))
    # In the thin limit, integrate y first: epsilon*integral_0^1 x/sqrt(1+x²) dx.
    @test analytical_integral_1overR(Vec3(0, 0, 1), thin_triangle...) ≈
        1e-20 * (sqrt(2.) - 1) rtol=4eps(Float64)
end

@testset "Periodic Green scaling avoids intermediate overflow" begin
    origin = Vec3(0, 0, 0)
    lattice = PeriodicLattice(1., 1., 0., 0., 1., 1., 0, 0)
    scale = 1e-308
    scaled_lattice = PeriodicLattice(scale, scale, 0., 0., 1 / scale, 1 / scale, 0, 0)
    for separation in (0., .3)
        expected = greens_periodic_correction(Vec3(0, 0, separation), origin, 1., lattice)
        point = Vec3(0, 0, separation * scale)
        actual = greens_periodic_correction(point, origin, 1 / scale, scaled_lattice)
        @test actual * scale ≈ expected rtol=8eps(Float64)
        spatial, spectral = DiffMoM._build_periodic_ewald_terms(scaled_lattice, 1)
        @test DiffMoM._greens_periodic_correction_cached(point, origin, 1 / scale,
            scaled_lattice, spatial, spectral) == actual
    end
    wave_number, distance = 1e308, 1e308
    saturated = PeriodicLattice(1., 1., 0., 0., wave_number, wave_number, 0, 0)
    expected = setprecision(BigFloat, 4352) do
        phase = exp(-im * BigFloat(wave_number) * BigFloat(distance))
        ComplexF64(-phase / (4big(pi) * distance) + phase / (2im * BigFloat(wave_number)))
    end
    @test greens_periodic_correction(Vec3(0, 0, distance), origin, wave_number, saturated) ≈
        expected rtol=32eps(Float64) atol=2nextfloat(0.)
end

@testset "Weighted multi-angle gradients round after weighting" begin
    for (mass, weight, rhs, tolerance) in ((1e308, .01, 1., 3e306),
                                         (nextfloat(0.), 8., .25, 1.))
        configuration = AngleConfig(Vec3(0, 0, 1), Vec3(1, 0, 0), ComplexF64[rhs],
                                    ones(ComplexF64, 1, 1), weight)
        _, trace = optimize_multiangle_rcs(ones(ComplexF64, 1, 1),
            [fill(ComplexF64(mass), 1, 1)], [configuration], [0.];
            maxiter=1, tol=tolerance, verbose=false)
        expected = setprecision(BigFloat, 8704) do
            Float64(2BigFloat(weight) * BigFloat(mass) * BigFloat(rhs)^2)
        end
        @test trace[1].gnorm == expected
    end
    configurations = [AngleConfig(Vec3(0, 0, 1), Vec3(1, 0, 0), ComplexF64[1],
        fill(ComplexF64(q), 1, 1), weight) for (q, weight) in ((1e30, 1.), (1., 1.), (1e30, -1.))]
    _, trace = optimize_multiangle_rcs(ones(ComplexF64, 1, 1), [ones(1, 1)],
        configurations, [0.]; maxiter=1, tol=3., verbose=false)
    @test trace[1].J == 1.
    @test trace[1].gnorm == 2.
    reactive_configuration = AngleConfig(Vec3(0, 0, 1), Vec3(1, 0, 0), ComplexF64[2],
                                         ones(ComplexF64, 1, 1), .01)
    _, trace = optimize_multiangle_rcs(fill(1. + 1im, 1, 1), [fill(1e308, 1, 1)],
        [reactive_configuration], [0.]; reactive=true, maxiter=1, tol=3e306, verbose=false)
    @test trace[1].gnorm ≈ 2e306 rtol=4eps(Float64)
end

@testset "Filter hashing retains close centroids with a large offset" begin
    coordinates = hcat([Vec3(x, y, z) for x in (-1e16, 1., nextfloat(1.))
        for (y, z) in ((0., 0.), (1., 0.), (0., 1.))]...)
    mesh = TriMesh(coordinates, reshape(collect(1:9), 3, 3))
    weights, sums = build_filter_weights(mesh, .01)
    @test weights[2, 3] == .01 - eps(Float64)
    @test weights[3, 2] == weights[2, 3]
    @test sums[2] == .01 + weights[2, 3]
    @test DiffMoM._filter_integer_cell_coordinate(1., 0., .01, 100.) == 99
end

@testset "Analytic dipole patterns preserve complex phase" begin
    direction = Vec3(1, 0, 0)
    for kind in (:electric, :magnetic), convention in (:exp_plus_iwt, :exp_minus_iwt)
        dipole = make_dipole(Vec3(0, 0, 0), CVec3(0, 0, 1 + 2im), Vec3(0, 0, 1),
                             kind, DiffMoM._C0 / (2pi))
        pattern = make_analytic_dipole_pattern_feed(dipole, [0., pi / 2, pi], [0., pi];
                                                   convention=convention)
        @test incident_farfield(pattern, direction, 1.) ≈ incident_farfield(dipole, direction, 1.) rtol=4eps(Float64)
    end
end
