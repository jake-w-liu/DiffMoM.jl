# test_periodic_topology.jl — Tests for periodic MoM + topology optimization modules
#
# Covers: PeriodicGreens (Helmholtz-Ewald), PeriodicEFIE, DensityInterpolation,
#         DensityFiltering, DensityAdjoint, PeriodicMetrics
#
# Test categories:
#   A: Analytical ground truth (independently derived expected values)
#   B: Mathematical properties (symmetry, PSD, reciprocity, adjoint)
#   C: Convergence & asymptotics
#   D: Edge cases & boundaries
#   E: Error handling
#   F: Cross-validation (FD vs analytical, alternative computations)

# ─────────────────────────────────────────────────────────────────
# Test 37: PeriodicGreens — Helmholtz-Ewald summation
# ─────────────────────────────────────────────────────────────────
println("\n── Test 37: PeriodicGreens (Helmholtz-Ewald) ──")

@testset "PeriodicGreens (Helmholtz-Ewald)" begin
    freq = 10e9; c0 = 3e8; lambda = c0 / freq; k = 2π / lambda
    dx = 0.5 * lambda; dy = 0.5 * lambda
    E_opt = sqrt(π / (dx * dy))

    # ── A: PeriodicLattice constructor ──
    @testset "A: PeriodicLattice constructor" begin
        lat = PeriodicLattice(dx, dy, 0.0, 0.0, k)
        # Normal incidence → kx = ky = 0
        @test lat.kx_bloch ≈ 0.0 atol=1e-15
        @test lat.ky_bloch ≈ 0.0 atol=1e-15
        # Optimal splitting: E = sqrt(π/A)
        @test lat.E ≈ E_opt rtol=1e-14
        @test lat.k ≈ k rtol=1e-14
        @test lat.N_spatial == 4
        @test lat.N_spectral == 4

        # Oblique incidence: θ=30°, φ=0° → kx = k sin(30°) = k/2
        lat_obl = PeriodicLattice(dx, dy, π/6, 0.0, k)
        @test lat_obl.kx_bloch ≈ k * 0.5 rtol=1e-14
        @test lat_obl.ky_bloch ≈ 0.0 atol=1e-15

        # Physical/numerical invariants are enforced at both constructors.
        for bad_period in (0.0, -dx, Inf, NaN)
            @test_throws ArgumentError PeriodicLattice(bad_period, dy, 0.0, 0.0, k)
            @test_throws ArgumentError PeriodicLattice(dx, bad_period, 0.0, 0.0, k)
        end
        for bad_angle in (Inf, -Inf, NaN)
            @test_throws ArgumentError PeriodicLattice(dx, dy, bad_angle, 0.0, k)
            @test_throws ArgumentError PeriodicLattice(dx, dy, 0.0, bad_angle, k)
        end
        for bad_k in (0.0, -k, Inf, NaN)
            @test_throws ArgumentError PeriodicLattice(dx, dy, 0.0, 0.0, bad_k)
        end
        @test_throws ArgumentError PeriodicLattice(dx, dy, 0.0, 0.0, k; N_spatial=-1)
        @test_throws ArgumentError PeriodicLattice(dx, dy, 0.0, 0.0, k; N_spectral=-1)
        maximum_periodic_order = DiffMoM._MAX_PERIODIC_TRUNCATION
        @test DiffMoM._periodic_term_count(maximum_periodic_order) <=
              DiffMoM._MAX_PERIODIC_TERM_COUNT
        @test DiffMoM._periodic_term_count(maximum_periodic_order + 1) >
              DiffMoM._MAX_PERIODIC_TERM_COUNT
        @test_throws ArgumentError PeriodicLattice(
            dx, dy, 0.0, 0.0, k;
            N_spatial=maximum_periodic_order + 1)
        @test_throws ArgumentError PeriodicLattice(
            dx, dy, 0.0, 0.0, k;
            N_spectral=maximum_periodic_order + 1)
        @test_throws ArgumentError PeriodicLattice(dx, dy, 0.0, 0.0, k, 0.0, 4, 4)
        @test_throws ArgumentError PeriodicLattice(dx, dy, NaN, 0.0, k, E_opt, 4, 4)
        @test_throws ArgumentError PeriodicLattice(dx, dy, 0.0, 0.0, k, E_opt, -1, 4)
    end

    # ── A: Ewald spatial kernel returns real value ──
    @testset "A: Spatial kernel real-valued" begin
        # K_sp(R) = Re[exp(-ikR) erfc(ER - ik/(2E))] / (4πR)
        # The two erfc terms are complex conjugates → sum is real
        for R in [0.001, 0.005, 0.01, 0.02]
            K_sp = DiffMoM._ewald_spatial_kernel(R, k, E_opt)
            @test isa(K_sp, Float64)  # real type from real() call
        end
    end

    # ── A: Self-correction at R=0 for k=0 (known Laplace result) ──
    @testset "A: Self-correction Laplace limit (k=0)" begin
        # For k=0: erfc(ik/(2E)) = erfc(0) = 1, exp(0) = 1
        # C_self = [2i×0×1 - 4E/√π × 1] / (8π) = -4E/(8π√π) = -E/(2π^{3/2})
        # Source: standard Ewald self-term for 2D Laplace lattice sum
        C_self_k0 = DiffMoM._ewald_self_correction(0.0, 0.0, E_opt)
        expected = -E_opt / (2 * π^(3/2))
        # Tolerance: exact formula, machine precision
        @test real(C_self_k0) ≈ expected rtol=1e-12
        @test abs(imag(C_self_k0)) < 1e-15  # purely real for k=0
    end

    # ── A: Self-correction continuity approaching R=0 ──
    @testset "A: Self-correction smooth at R→0" begin
        C0 = DiffMoM._ewald_self_correction(0.0, k, E_opt)
        # At R=1e-6: K_sp ~ 1/(4πR) ~ 8e4, difference ~ O(E) ~ 100
        # Cancellation: ~3 digits lost → ~12 digits remaining
        CR = DiffMoM._ewald_self_correction(1e-6, k, E_opt)
        @test abs(CR - C0) / abs(C0) < 1e-6
        # At R=1e-10: ~7 digits lost from cancellation → ~8 digits remaining
        CR2 = DiffMoM._ewald_self_correction(1e-10, k, E_opt)
        @test abs(CR2 - C0) / abs(C0) < 1e-3

        # The analytical self limit remains finite when the unfactored
        # intermediates k², 2k, and 4E overflow.  These reference values were
        # evaluated from the exact Float64 inputs at 200 decimal digits.
        large_k = 1.0e308
        large_E = large_k / 2
        large_reference = ComplexF64(
            0x1.52d72d91d8cecp+1016,
            0x1.6aa172e512d49p+1019)
        large_self = DiffMoM._ewald_self_correction(0.0, large_k, large_E)
        @test isfinite(large_self)
        @test large_self ≈ large_reference rtol=2e-15

        # The cold scaling path covers both a ratio that underflows and an
        # exponential factor that overflows even though the completed limits
        # are finite. References use the same 200-digit exact-input oracle.
        maximum_finite = floatmax(Float64)
        tiny_k = 1.0e-300
        tiny_ratio_reference = ComplexF64(
            -0x1.6fcb5f827b97ep+1020,
             0x1.b49266db89b9ep-1001)
        @test DiffMoM._ewald_self_correction(
            0.0, tiny_k, maximum_finite) == tiny_ratio_reference

        minimum_positive = nextfloat(0.0)
        large_ratio_k = 60minimum_positive
        exponential_reference = ComplexF64(
            0x1.197c7a56ca286p+210,
            0x0.0000000000005p-1022)
        @test DiffMoM._ewald_self_correction(
            0.0, large_ratio_k, minimum_positive) ==
              exponential_reference
        DiffMoM._ewald_self_correction(
            0.0, large_ratio_k, minimum_positive)
        @test @allocated(DiffMoM._ewald_self_correction(
            0.0, large_ratio_k, minimum_positive)) < 100_000

        # Near-origin selection is based on kR and ER, not an absolute length.
        # At this small physical separation the field is not in its R→0
        # regime because both dimensionless products are O(10).
        rapid_k = 1.0e16
        rapid_E = 2.0e16
        rapid_R = 1.0e-15
        rapid_reference = ComplexF64(
             0x1.e5d308d66b814p+45,
            -0x1.3afd4e8e040dcp+45)
        @test DiffMoM._ewald_self_correction(
            rapid_R, rapid_k, rapid_E) ≈ rapid_reference rtol=2e-15
    end

    # ── A: kz branch cut correctness ──
    @testset "A: Spectral kz branch cut" begin
        # Propagating: kt = 0 → kz = k (real positive)
        kz_prop = DiffMoM._spectral_kz(k, 0.0, 0.0)
        @test real(kz_prop) ≈ k rtol=1e-14
        @test abs(imag(kz_prop)) < 1e-14

        # A relative threshold formed as `1e-6*k` underflows at the bottom of
        # the Float64 range.  Exact grazing incidence must still omit the
        # singular Wood mode in both the public and cached Ewald paths.
        minimum_positive = nextfloat(0.0)
        tiny_wood_lattice = PeriodicLattice(
            1.0, 1.0, minimum_positive, 0.0,
            minimum_positive, 1.0, 0, 0)
        tiny_wood_kz = DiffMoM._spectral_kz(
            minimum_positive, minimum_positive, 0.0)
        @test iszero(tiny_wood_kz)
        @test DiffMoM._periodic_is_wood_anomaly(
            tiny_wood_kz, minimum_positive)
        tiny_wood_spatial, tiny_wood_spectral =
            DiffMoM._build_periodic_ewald_terms(tiny_wood_lattice, 1)
        @test isempty(tiny_wood_spatial)
        @test isempty(tiny_wood_spectral)
        tiny_wood_r = Vec3(0.125, -0.25, 0.0)
        tiny_wood_rp = Vec3(0.0, 0.0, 0.0)
        tiny_wood_value = greens_periodic_correction(
            tiny_wood_r, tiny_wood_rp,
            minimum_positive, tiny_wood_lattice)
        @test isfinite(tiny_wood_value)
        @test DiffMoM._greens_periodic_correction_cached(
            tiny_wood_r, tiny_wood_rp,
            minimum_positive, tiny_wood_lattice,
            tiny_wood_spatial, tiny_wood_spectral) == tiny_wood_value

        # Evanescent: kt = 2k → kz² = k²-4k² = -3k²
        # kz = sqrt(-3k²) = ik√3 → negate to get Im(kz) ≤ 0 → kz = -ik√3
        kz_evan = DiffMoM._spectral_kz(k, 2k, 0.0)
        @test abs(real(kz_evan)) < 1e-14
        @test imag(kz_evan) ≈ -sqrt(3) * k rtol=1e-14

        large_k = 1.0e200
        @test DiffMoM._spectral_kz(large_k, 0.0, 0.0) ==
              ComplexF64(large_k, 0.0)
        maximum_finite = floatmax(Float64)
        @test DiffMoM._spectral_kz(
            maximum_finite, maximum_finite, maximum_finite) ==
              ComplexF64(0.0, -maximum_finite)
        grazing_kappa = prevfloat(1.0)
        grazing_reference = setprecision(BigFloat, 256) do
            Float64(sqrt(BigFloat(1.0)^2 - BigFloat(grazing_kappa)^2))
        end
        @test DiffMoM._spectral_kz(1.0, grazing_kappa, 0.0) ==
              ComplexF64(grazing_reference, 0.0)

        large_k_lattice = PeriodicLattice(
            1.0, 1.0, 0.0, 0.0, large_k, 1.0, 0, 0)
        @test DiffMoM._kz_inc(large_k, large_k_lattice) == large_k
        invalid_incident_lattice = PeriodicLattice(
            1.0, 1.0,
            maximum_finite, maximum_finite,
            maximum_finite, 1.0, 0, 0)
        @test_throws ArgumentError DiffMoM._kz_inc(
            maximum_finite, invalid_incident_lattice)
    end

    # ── B: Reciprocity ΔG(r,rp) = ΔG(rp,r) for normal incidence ──
    @testset "B: Reciprocity (normal incidence)" begin
        lat = PeriodicLattice(dx, dy, 0.0, 0.0, k)
        r1 = SVector(0.002, 0.003, 0.0)
        r2 = SVector(0.007, 0.001, 0.0)
        dG_12 = greens_periodic_correction(r1, r2, k, lat)
        dG_21 = greens_periodic_correction(r2, r1, k, lat)
        # Periodic GF is symmetric for normal incidence (k_∥ = 0)
        # Tolerance: both evaluations use same Ewald code, small roundoff
        @test isapprox(dG_12, dG_21, rtol=1e-12)
    end

    @testset "B2: Exact high-argument Bloch phase" begin
        # A rounded finite kx*dx phase changes this bounded Ewald correction
        # by more than ten percent. The API is defined by its stored Float64
        # lattice values, so reduce their exact products before exponentiation.
        phase_lattice = PeriodicLattice(
            1.0e20, 1.0e20, 1.1, 0.7, 1.0e-20, 1.0e-20, 1, 0)
        phase_point = Vec3(0.0, 0.0, 0.0)
        phase_value = greens_periodic_correction(
            phase_point, phase_point, phase_lattice.k, phase_lattice)

        spatial_reference = setprecision(BigFloat, 2304) do
            total = zero(Complex{BigFloat})
            for m in -1:1, n in -1:1
                iszero(m) && iszero(n) && continue
                sx = m * phase_lattice.dx
                sy = n * phase_lattice.dy
                radius = hypot(sx, sy)
                argument = BigFloat(phase_lattice.kx_bloch) * BigFloat(sx) +
                           BigFloat(phase_lattice.ky_bloch) * BigFloat(sy)
                kernel = DiffMoM._ewald_spatial_kernel(
                    radius, phase_lattice.k, phase_lattice.E)
                total += exp(Complex{BigFloat}(0, -argument)) *
                         BigFloat(kernel)
            end
            ComplexF64(total)
        end
        self_reference = DiffMoM._ewald_self_correction(
            0.0, phase_lattice.k, phase_lattice.E)
        kz = DiffMoM._spectral_kz(
            phase_lattice.k,
            phase_lattice.kx_bloch,
            phase_lattice.ky_bloch,
        )
        zk = im * kz / (2phase_lattice.E)
        spectral_kernel = (erfc(zk) + erfc(zk)) /
                          (4im * kz)
        spectral_reference = DiffMoM._periodic_scale_by_cell_area(
            spectral_kernel, phase_lattice.dx, phase_lattice.dy)
        phase_reference = self_reference + spatial_reference +
                          spectral_reference
        @test phase_value ≈ phase_reference rtol=8eps(Float64) atol=0.0

        spatial_terms, spectral_terms = DiffMoM._build_periodic_ewald_terms(
            phase_lattice, 1)
        @test DiffMoM._greens_periodic_correction_cached(
            phase_point,
            phase_point,
            phase_lattice.k,
            phase_lattice,
            spatial_terms,
            spectral_terms,
        ) == phase_value
    end

    @testset "B3: Stable vertical Floquet phase" begin
        propagating_lattice = PeriodicLattice(
            1.0, 1.0, 0.0, 0.0, 1.1, 1.0, 0, 0)
        source = Vec3(0.0, 0.0, 0.0)
        observation = Vec3(0.0, 0.0, 1.0e20)
        propagating_value = greens_periodic_correction(
            observation, source, propagating_lattice.k,
            propagating_lattice)
        propagating_reference = setprecision(BigFloat, 2304) do
            phase = exp(Complex{BigFloat}(
                0, -BigFloat(propagating_lattice.k) *
                   BigFloat(observation[3])))
            self = Complex{BigFloat}(DiffMoM._ewald_self_correction(
                observation[3], propagating_lattice.k,
                propagating_lattice.E))
            ComplexF64(self + phase /
                (2 * Complex{BigFloat}(0, 1) *
                 BigFloat(propagating_lattice.k)))
        end
        @test isapprox(
            propagating_value, propagating_reference;
            rtol=8eps(Float64), atol=0.0)

        # An evanescent mode at extreme separation decays to zero. Forming its
        # growing exponential separately used to produce Inf*0 and reject this
        # finite correction.
        evanescent_lattice = PeriodicLattice(
            1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0, 0)
        extreme_observation = Vec3(0.0, 0.0, 1.0e308)
        evanescent_value = greens_periodic_correction(
            extreme_observation, source, evanescent_lattice.k,
            evanescent_lattice)
        @test evanescent_value == DiffMoM._ewald_self_correction(
            extreme_observation[3], evanescent_lattice.k,
            evanescent_lattice.E)
        @test isfinite(evanescent_value)
    end

    # ── C: Exponential convergence with truncation order ──
    @testset "C: Exponential Ewald convergence" begin
        r  = SVector(0.0, 0.0, 0.0)
        rp = SVector(0.002, 0.003, 0.0)
        # Reference: N=6 (fully converged)
        lat_ref = PeriodicLattice(dx, dy, 0.0, 0.0, k; N_spatial=6, N_spectral=6)
        dG_ref = greens_periodic_correction(r, rp, k, lat_ref)

        errors = Float64[]
        for N in [1, 2, 3, 4]
            lat_N = PeriodicLattice(dx, dy, 0.0, 0.0, k; N_spatial=N, N_spectral=N)
            dG_N = greens_periodic_correction(r, rp, k, lat_N)
            push!(errors, abs(dG_N - dG_ref))
        end
        # Exponential convergence: each refinement should reduce the error.
        @test errors[2] < errors[1] * 0.01  # N=1→2: 100× improvement
        @test errors[3] < 1e-10 * abs(dG_ref)  # N=3: machine precision
        @test errors[4] < 1e-13 * abs(dG_ref)  # N=4: essentially zero
    end

    # ── D: No NaN/Inf including self-point R=0 ──
    @testset "D: No NaN/Inf" begin
        lat = PeriodicLattice(dx, dy, 0.0, 0.0, k)
        r  = SVector(0.0, 0.0, 0.0)
        rp = SVector(0.002, 0.003, 0.0)
        dG = greens_periodic_correction(r, rp, k, lat)
        @test !isnan(abs(dG)) && !isinf(abs(dG))

        # Self-point (R=0): exercises analytical limit in self-correction
        dG_self = greens_periodic_correction(r, r, k, lat)
        @test !isnan(abs(dG_self)) && !isinf(abs(dG_self))
    end

    @testset "D: Wide finite distance and cell area" begin
        wide_k = 1.0e-200
        wide_E = 1.0e-200
        wide_period = 1.0e201
        wide_lattice = PeriodicLattice(
            wide_period, wide_period, 0.0, 0.0,
            wide_k, wide_E, 0, 0)
        wide_r = Vec3(1.0e200, 0.0, 0.0)
        wide_rp = Vec3(0.0, 0.0, 0.0)

        # Both the naive squared norm and dx*dy overflow, although the Ewald
        # self term and the cell-area-scaled spectral term are representable.
        wide_self = DiffMoM._ewald_self_correction(
            hypot(hypot(wide_r[1], wide_r[2]), wide_r[3]),
            wide_k, wide_E)
        wide_kz = ComplexF64(wide_k, 0.0)
        wide_zk = im * wide_kz / (2wide_E)
        wide_spectral =
            (erfc(wide_zk) + erfc(wide_zk)) /
            (4im * wide_kz)
        wide_scaled = setprecision(BigFloat, 256) do
            ComplexF64(
                Complex{BigFloat}(wide_spectral) /
                (BigFloat(wide_period) * BigFloat(wide_period)))
        end
        wide_reference = wide_self + wide_scaled
        @test isfinite(wide_reference)
        @test greens_periodic_correction(
            wide_r, wide_rp, wide_k, wide_lattice) == wide_reference

        # A rounded subnormal cell area is not an accurate divisor. Keep the
        # factors separate so the finite quotient retains the exact scale.
        subnormal_dx = ldexp(1.0, -600)
        subnormal_dy = ldexp(1.5, -474)
        subnormal_value = ComplexF64(ldexp(1.0, -500), 0.0)
        @test 0.0 < subnormal_dx * subnormal_dy < floatmin(Float64)
        subnormal_reference = setprecision(BigFloat, 256) do
            ComplexF64(
                Complex{BigFloat}(subnormal_value) /
                (BigFloat(subnormal_dx) * BigFloat(subnormal_dy)))
        end
        @test DiffMoM._periodic_scale_by_cell_area(
            subnormal_value, subnormal_dx, subnormal_dy) ==
              subnormal_reference
        DiffMoM._periodic_scale_by_cell_area(
            subnormal_value, subnormal_dx, subnormal_dy)
        @test @allocated(DiffMoM._periodic_scale_by_cell_area(
            subnormal_value, subnormal_dx, subnormal_dy)) == 0

        # Complete physical products stay representable even when the damped
        # kernel's phase product or the naive 4πR denominator overflows.
        @test DiffMoM._ewald_spatial_kernel(
            1.0e200, 1.0e200, 1.0e200) == 0.0
        damping_E = floatmax(Float64)
        damping_R = 30 / damping_E
        damping_reference = setprecision(BigFloat, 512) do
            R_big = BigFloat(damping_R)
            E_big = BigFloat(damping_E)
            Float64(erfc(E_big * R_big) /
                    (4big(π) * R_big))
        end
        @test DiffMoM._ewald_spatial_kernel(
            damping_R, nextfloat(0.0), damping_E) ≈
              damping_reference rtol=1e-12
        DiffMoM._ewald_spatial_kernel(
            damping_R, nextfloat(0.0), damping_E)
        @test @allocated(DiffMoM._ewald_spatial_kernel(
            damping_R, nextfloat(0.0), damping_E)) == 0
        maximum_radius = floatmax(Float64)
        overflow_phase_k = 2.0
        radial_reference = setprecision(BigFloat, 2304) do
            ComplexF64(
                -exp(-im * BigFloat(overflow_phase_k) *
                     BigFloat(maximum_radius)) /
                BigFloat(maximum_radius) / (4big(π)))
        end
        @test DiffMoM._ewald_self_correction(
            maximum_radius, overflow_phase_k, 1.0) == radial_reference

        overflow_image_lattice = PeriodicLattice(
            floatmax(Float64), 1.0, 0.0, 0.0,
            1.0, 1.0, 2, 0)
        overflow_image_point = Vec3(0.0, 0.0, 0.0)
        # Independent finite Ewald decomposition after zero-kernel images are
        # omitted; the m=±2 x shifts lie beyond the Float64 coordinate range.
        overflow_image_reference =
            -0.03648908496552651 + 0.07957747154594767im
        @test greens_periodic_correction(
            overflow_image_point,
            overflow_image_point,
            1.0,
            overflow_image_lattice,
        ) == overflow_image_reference
        overflow_spatial_terms, overflow_spectral_terms =
            DiffMoM._build_periodic_ewald_terms(
                overflow_image_lattice, 1)
        @test length(overflow_spatial_terms) <
              DiffMoM._periodic_term_count(2) - 1
        @test DiffMoM._greens_periodic_correction_cached(
            overflow_image_point,
            overflow_image_point,
            1.0,
            overflow_image_lattice,
            overflow_spatial_terms,
            overflow_spectral_terms,
        ) == overflow_image_reference
        @test DiffMoM._periodic_transverse_phase(
            0.0, 0.0, Inf, -Inf) == 1.0 + 0im
    end

    # ── E: Call-site wavenumber must agree with the lattice ──
    @testset "E: Wavenumber and coordinate validation" begin
        lat = PeriodicLattice(dx, dy, π/7, π/5, k)
        r = SVector(0.17dx, -0.11dy, 0.03dx)
        rp = SVector(-0.08dx, 0.05dy, -0.02dx)
        @test_throws ArgumentError greens_periodic_correction(r, rp, 1.01k, lat)
        @test_throws ArgumentError greens_periodic_correction(
            SVector(NaN, 0.0, 0.0), rp, k, lat
        )

        # A rounding-equivalent value is normalized to the stored lattice k.
        g_ref = greens_periodic_correction(r, rp, k, lat)
        g_roundoff = greens_periodic_correction(r, rp, k * (1 + 5e-13), lat)
        @test g_roundoff == g_ref
    end

    # ── D: Oblique incidence ──
    @testset "D: Oblique incidence (θ=30°)" begin
        lat_obl = PeriodicLattice(dx, dy, π/6, 0.0, k)
        r  = SVector(0.0, 0.0, 0.0)
        rp = SVector(0.002, 0.003, 0.0)
        dG = greens_periodic_correction(r, rp, k, lat_obl)
        @test !isnan(abs(dG)) && !isinf(abs(dG))
        @test abs(dG) > 0  # should be nonzero for typical parameters
    end

    # ── F: Vertical separation (z != z') supported and correct ──
    @testset "F: Vertical-separation Green's function" begin
        A = dx * dy
        Eopt = sqrt(π / A)
        latE(E) = PeriodicLattice(dx, dy, 0.0, 0.0, k, E, 8, 18)
        g0(R) = exp(-im * k * R) / (4π * R)
        # Pure Floquet spectral reference for the full G_per (converges for Δz != 0).
        function gper_spec(dxr, dyr, dz; P=60)
            v = 0.0 + 0.0im
            for p in -P:P, q in -P:P
                kxp = 2π * p / dx; kyq = 2π * q / dy
                kz = sqrt(complex(k^2 - kxp^2 - kyq^2)); imag(kz) > 0 && (kz = -kz)
                abs(kz) < 1e-9 * k && continue
                v += exp(-im * (kxp * dxr + kyq * dyr)) * exp(-im * kz * abs(dz)) / (2im * kz) / A
            end
            return v
        end
        for (dxr, dyr, dz) in [(0.0, 0.0, 0.10dx), (0.13dx, -0.07dy, 0.10dx),
                               (0.13dx, -0.07dy, 0.25dx), (-0.2dx, 0.1dy, 0.5dx)]
            r  = SVector(dxr, dyr, dz)
            rp = SVector(0.0, 0.0, 0.0)
            R  = sqrt(dxr^2 + dyr^2 + dz^2)
            g_lo = greens_periodic_correction(r, rp, k, latE(0.5Eopt))
            g_md = greens_periodic_correction(r, rp, k, latE(1.0Eopt))
            g_hi = greens_periodic_correction(r, rp, k, latE(2.0Eopt))
            # Ewald result is invariant to the splitting parameter E.
            @test abs(g_lo - g_md) ≤ 1e-9 * abs(g_md)
            @test abs(g_hi - g_md) ≤ 1e-9 * abs(g_md)
            # G_0 + ΔG matches the independent pure-Floquet-spectral representation.
            @test abs((g0(R) + g_md) - gper_spec(dxr, dyr, dz)) ≤ 1e-7 * abs(gper_spec(dxr, dyr, dz))
        end
        # Δz = 0 still recovers the coplanar correction (no regression).
        rc = SVector(0.13dx, -0.07dy, 0.0)
        @test isfinite(abs(greens_periodic_correction(rc, SVector(0.0, 0.0, 0.0), k, latE(Eopt))))
    end

    # ── C: Large-period Ewald convergence (d up to 100λ) ──
    @testset "C: Large-period Ewald convergence" begin
        # Use non-integer d/λ to avoid exact Wood anomaly (kz=0 singularity).
        # For non-Wood-anomaly periods, the Ewald sum converges and gives
        # finite results for arbitrarily large d/λ.
        # Note: |ΔG| is NOT monotonically decreasing because near-grazing
        # Floquet modes (near Wood anomalies) enhance the correction.
        r  = SVector(0.002, 0.003, 0.0)
        rp = SVector(0.0, 0.0, 0.0)
        for alpha in [2.5, 10.5, 50.5, 100.5]
            d = alpha * lambda
            lat = PeriodicLattice(d, d, 0.0, 0.0, k)
            dG = greens_periodic_correction(r, rp, k, lat)
            @test !isnan(abs(dG)) && !isinf(abs(dG))
            # Correction bounded: |ΔG| ≤ O(k) for any finite period
            # (each image contributes ~1/(4πd), summed over ~d/λ images per shell)
            @test abs(dG) < 100 * k
        end
    end

    # ── F: E-independence (non-Wood-anomaly periods) ──
    @testset "F: E-independence for non-Wood-anomaly periods" begin
        # Ewald splitting parameter E is mathematical; result must be
        # E-independent for non-Wood-anomaly periods. At integer d/λ with
        # normal incidence, Wood anomaly modes (kz=0) break the identity,
        # but non-integer d/λ should give machine-precision agreement.
        r  = SVector(0.002, 0.003, 0.0)
        rp = SVector(0.0, 0.0, 0.0)
        M_erfc = 5.0

        for alpha in [0.5, 2.5, 10.5]
            d = alpha * lambda
            E_min = k / (2 * sqrt(2.0))
            # Reference: E = E_min (α = 2)
            Nf1 = max(8, ceil(Int, d * sqrt(k^2 + 4*E_min^2*M_erfc^2) / (2π)))
            lat1 = PeriodicLattice(d, d, 0.0, 0.0, k, E_min, 8, Nf1)
            dG1 = greens_periodic_correction(r, rp, k, lat1)
            # Compare: E = 3 × E_min (α = 0.22)
            E2 = 3.0 * E_min
            Nf2 = max(8, ceil(Int, d * sqrt(k^2 + 4*E2^2*M_erfc^2) / (2π)))
            lat2 = PeriodicLattice(d, d, 0.0, 0.0, k, E2, 8, Nf2)
            dG2 = greens_periodic_correction(r, rp, k, lat2)
            # Machine-precision agreement for non-Wood-anomaly periods
            @test isapprox(dG1, dG2, rtol=1e-12)
        end
    end
end
println("  PASS ✓")

# ─────────────────────────────────────────────────────────────────
# Test 38: DensityInterpolation — SIMP material model
# ─────────────────────────────────────────────────────────────────
println("\n── Test 38: DensityInterpolation ──")

@testset "DensityInterpolation" begin
    mesh_di = make_rect_plate(0.01, 0.01, 3, 3)
    rwg_di = build_rwg(mesh_di; precheck=false)
    Nt_di = ntriangles(mesh_di)
    N_di = rwg_di.nedges
    Mt = precompute_triangle_mass(mesh_di, rwg_di)
    mass_triangle_nq = length(tri_quad_rule(3)[2])
    mass_triangle_profile = DiffMoM._mass_precompute_profile(
        rwg_di, Nt_di, mass_triangle_nq, Float64, nothing, Nt_di)
    @test_throws ArgumentError precompute_triangle_mass(
        mesh_di, rwg_di;
        max_work_bytes=mass_triangle_profile.work_bytes - 1,
        max_terms=mass_triangle_profile.term_count)
    @test_throws ArgumentError precompute_triangle_mass(
        mesh_di, rwg_di;
        max_work_bytes=mass_triangle_profile.work_bytes,
        max_terms=mass_triangle_profile.term_count - 1)
    Mt_at_resource_boundary = precompute_triangle_mass(
        mesh_di, rwg_di;
        max_work_bytes=mass_triangle_profile.work_bytes,
        max_terms=mass_triangle_profile.term_count)
    @test Matrix.(Mt_at_resource_boundary) == Matrix.(Mt)
    eta0 = 376.730313668
    config = DensityConfig(; p=3.0, Z_max_factor=1000.0, vf_target=0.5)

    # ── A: DensityConfig ──
    @testset "A: DensityConfig values" begin
        @test abs(config.Z_max) ≈ 1000.0 * eta0 rtol=1e-12
        @test config.p ≈ 3.0
        @test config.vf_target ≈ 0.5
        @test_throws ArgumentError DensityConfig(; p=0.0)
        @test_throws ArgumentError DensityConfig(; p=0.5)
        @test_throws ArgumentError DensityConfig(; Z_max_factor=0.0)
        @test_throws ArgumentError DensityConfig(; eta0=0.0)
        @test_throws ArgumentError DensityConfig(; vf_target=1.1)
        @test_throws ArgumentError DensityConfig(3.0, 0.0 + 0im, 0.5)
    end

    # ── A: Correct count of mass matrices ──
    @testset "A: Mass matrix count" begin
        @test length(Mt) == Nt_di
    end

    # ── B: Mass matrix symmetry ──
    @testset "B: Mass matrix symmetry" begin
        for t in 1:Nt_di
            M = Matrix(Mt[t])
            # M[m,n] = ∫ f_m·f_n dS is symmetric by definition
            # Tolerance: product quadrature, exact up to roundoff
            @test isapprox(M, M', atol=1e-14)
        end
    end

    # ── B: Mass matrix positive semi-definiteness ──
    @testset "B: Mass matrix PSD" begin
        for t in 1:Nt_di
            M = Matrix(Mt[t])
            eigs = eigvals(Symmetric(M))
            # f_m · f_n Gram matrix → all eigenvalues ≥ 0
            @test all(eigs .≥ -1e-14)
        end
    end

    # ── B: Mass matrix sparsity (only supported basis functions) ──
    @testset "B: Mass matrix sparsity" begin
        for t in 1:min(3, Nt_di)  # check a few triangles
            # Identify basis functions with support on triangle t
            supported = Set{Int}()
            for n in 1:N_di
                if rwg_di.tplus[n] == t || rwg_di.tminus[n] == t
                    push!(supported, n)
                end
            end
            for i in 1:N_di, j in 1:N_di
                if !(i ∈ supported) || !(j ∈ supported)
                    @test Mt[t][i, j] == 0.0
                end
            end
        end
    end

    # ── A: Z_penalty vanishes for all-metal (ρ̄ = 1) ──
    @testset "A: Z_penalty = 0 for all-metal" begin
        rho_bar_metal = ones(Nt_di)
        Z_pen = assemble_Z_penalty(Mt, rho_bar_metal, config)
        penalty_output_bytes = sizeof(ComplexF64) * N_di^2
        @test_throws ArgumentError assemble_Z_penalty(
            Mt, rho_bar_metal, config;
            max_output_bytes=penalty_output_bytes - 1)
        @test assemble_Z_penalty(
            Mt, rho_bar_metal, config;
            max_output_bytes=penalty_output_bytes) == Z_pen
        # (1 - 1^p) = 0 → zero penalty
        @test norm(Z_pen) < 1e-14
    end

    # ── A: Z_penalty for all-void (ρ̄ = 0) ──
    @testset "A: Z_penalty = Z_max × ΣM_t for all-void" begin
        rho_bar_void = zeros(Nt_di)
        Z_pen = assemble_Z_penalty(Mt, rho_bar_void, config)
        # (1 - 0^p) = 1 → Z_penalty = Z_max × Σ M_t
        M_total = sum(Matrix.(Mt))
        Z_expected = config.Z_max .* M_total
        # Tolerance: exact formula, machine precision
        @test isapprox(Z_pen, Z_expected, rtol=1e-12)
    end

    # ── B: Z_penalty linear in Z_max ──
    @testset "B: Z_penalty linear in Z_max" begin
        Random.seed!(77)
        rho_bar = 0.3 .+ 0.4 * rand(Nt_di)
        config_1x = DensityConfig(; p=3.0, Z_max_factor=500.0)
        config_2x = DensityConfig(; p=3.0, Z_max_factor=1000.0)
        Z_1x = assemble_Z_penalty(Mt, rho_bar, config_1x)
        Z_2x = assemble_Z_penalty(Mt, rho_bar, config_2x)
        # Doubling Z_max should double Z_penalty
        @test isapprox(Z_2x, 2.0 .* Z_1x, rtol=1e-12)
    end

    # ── F: dZ/dρ̄ vs finite difference ──
    @testset "F: dZ/dρ̄ vs finite difference" begin
        Random.seed!(78)
        rho_bar = 0.3 .+ 0.4 * rand(Nt_di)
        h = 1e-7
        for t in [1, div(Nt_di, 2), Nt_di]
            dZ_ana = Matrix(assemble_dZ_drhobar(Mt, rho_bar, config, t))
            rho_plus = copy(rho_bar); rho_plus[t] += h
            rho_minus = copy(rho_bar); rho_minus[t] -= h
            Z_plus = assemble_Z_penalty(Mt, rho_plus, config)
            Z_minus = assemble_Z_penalty(Mt, rho_minus, config)
            dZ_fd = (Z_plus - Z_minus) / (2h)
            # Tolerance: O(h²) central FD, h=1e-7 → error ~1e-14 absolute
            # Use rtol since values can span orders of magnitude
            @test isapprox(dZ_ana, dZ_fd, rtol=1e-5)
        end
    end

    @testset "Range-safe density matrices" begin
        extreme_mass = [fill(ComplexF64(floatmax(Float64)), 1, 1)]
        extreme_config = DensityConfig(1.0, 2.0, 0.5)
        @test_throws OverflowError assemble_Z_penalty(
            extreme_mass, [0.0], extreme_config)
        @test_throws OverflowError assemble_dZ_drhobar(
            extreme_mass, [0.0], extreme_config, 1)

        nonfinite_mass = [fill(ComplexF64(Inf), 1, 1)]
        @test_throws ArgumentError assemble_Z_penalty(
            nonfinite_mass, [1.0], extreme_config)
        @test_throws ArgumentError assemble_dZ_drhobar(
            nonfinite_mass, [1.0], extreme_config, 1)

        underflow_config = DensityConfig(
            500.0, floatmax(Float64), 0.5)
        recovered_derivative = assemble_dZ_drhobar(
            [ones(ComplexF64, 1, 1)], [0.1], underflow_config, 1)
        derivative_reference = setprecision(BigFloat, 4352) do
            ComplexF64(
                -BigFloat(underflow_config.p) *
                BigFloat(0.1)^(BigFloat(underflow_config.p) - 1) *
                Complex{BigFloat}(underflow_config.Z_max))
        end
        @test !iszero(derivative_reference)
        @test recovered_derivative[1, 1] == derivative_reference

        derivative_scale = sqrt(floatmax(Float64))
        derivative_config = DensityConfig(
            1.0,
            derivative_scale + 0.5 * derivative_scale * im,
            0.5,
        )
        derivative_mass_value =
            -1.2 * derivative_scale - 0.4 * derivative_scale * im
        derivative_coefficient = DiffMoM._density_derivative_coefficient(
            0.5, derivative_config)
        scaled_derivative_reference = setprecision(BigFloat, 4352) do
            ComplexF64(
                Complex{BigFloat}(derivative_coefficient) *
                Complex{BigFloat}(derivative_mass_value))
        end
        derivative_matrices = (
            reshape(ComplexF64[derivative_mass_value], 1, 1),
            sparse(
                [1], [1], ComplexF64[derivative_mass_value], 1, 1),
            LocalMassMatrix(
                1, [1], [1], ComplexF64[derivative_mass_value]),
        )
        for derivative_matrix in derivative_matrices
            scaled_derivative = assemble_dZ_drhobar(
                [derivative_matrix], [0.5], derivative_config, 1)
            @test scaled_derivative[1, 1] == scaled_derivative_reference
            @test isfinite(scaled_derivative[1, 1])
        end
    end

    # ── E: Dimension mismatch assertion ──
    @testset "E: Dimension mismatch" begin
        @test_throws DimensionMismatch assemble_Z_penalty(
            Mt, zeros(Nt_di + 1), config)
        @test_throws ArgumentError assemble_Z_penalty(
            Mt, fill(NaN, Nt_di), config)
        @test_throws ArgumentError assemble_Z_penalty(
            Mt, fill(1.1, Nt_di), config)
        @test_throws DimensionMismatch assemble_dZ_drhobar(
            Mt, zeros(Nt_di + 1), config, 1)
    end
end
println("  PASS ✓")

# ─────────────────────────────────────────────────────────────────
# Test 39: DensityFiltering — conic filter + Heaviside projection
# ─────────────────────────────────────────────────────────────────
println("\n── Test 39: DensityFiltering ──")

@testset "DensityFiltering" begin
    mesh_df = make_rect_plate(0.015, 0.015, 6, 6)
    Nt_df = ntriangles(mesh_df)
    edge_len_df = 0.015 / 6
    r_min = 2.5 * edge_len_df
    W, w_sum = build_filter_weights(mesh_df, r_min)

    # ── A: Filter weight structure ──
    @testset "A: Filter weight properties" begin
        @test_throws ArgumentError build_filter_weights(mesh_df, 0.0)
        @test_throws ArgumentError build_filter_weights(mesh_df, -r_min)
        @test_throws ArgumentError build_filter_weights(mesh_df, Inf)
        @test_throws ArgumentError build_filter_weights(
            mesh_df, r_min; max_triplet_bytes=0)

        filter_triplet_bytes = nnz(W) *
            (2 * sizeof(Int) + sizeof(Float64))
        W_capped, sums_capped = build_filter_weights(
            mesh_df, r_min; max_triplet_bytes=filter_triplet_bytes)
        @test W_capped == W
        @test sums_capped == w_sum
        @test_throws ArgumentError build_filter_weights(
            mesh_df, r_min;
            max_triplet_bytes=filter_triplet_bytes - 1)

        # Cell coordinates may exceed Int even though the mesh and filter
        # radius are finite. The wide-key path must retain the sparse hash and
        # produce the exact two isolated self-weights.
        wide_xyz = [
            0.0 1.0 0.0 1.0e200 1.0e200 + 1.0e190 1.0e200
            0.0 0.0 1.0 0.0     0.0                 1.0e190
            0.0 0.0 0.0 0.0     0.0                 0.0
        ]
        wide_tri = [1 4; 2 5; 3 6]
        wide_mesh = TriMesh(wide_xyz, wide_tri)
        wide_radius = 1.0e-200
        W_wide, sums_wide = build_filter_weights(wide_mesh, wide_radius)
        @test Matrix(W_wide) == [wide_radius 0.0; 0.0 wide_radius]
        @test sums_wide == [wide_radius, wide_radius]
        build_filter_weights(wide_mesh, wide_radius)
        let mesh = wide_mesh, radius = wide_radius
            @test (@allocated build_filter_weights(mesh, radius)) <= 100_000
        end

        # A rounded Euclidean distance can land exactly on the radius even
        # though the exact distance between the stored centroids is inside it.
        # The positive conic edge must not disappear from the sparse topology.
        boundary_x = prevfloat(1.0)
        boundary_y = prevfloat(ldexp(1.0, -26))
        boundary_vertices = Vec3[
            Vec3(-0.01, 0.0, 0.0),
            Vec3(0.01, -0.01, 0.0),
            Vec3(0.0, 0.01, 0.0),
            Vec3(prevfloat(boundary_x), boundary_y, 0.0),
            Vec3(nextfloat(boundary_x), prevfloat(boundary_y), 0.0),
            Vec3(boundary_x, nextfloat(boundary_y), 0.0),
        ]
        boundary_mesh = TriMesh(
            reduce(hcat, boundary_vertices), reshape(1:6, 3, 2))
        @test triangle_center(boundary_mesh, 2) ==
              Vec3(boundary_x, boundary_y, 0.0)
        boundary_weight = setprecision(BigFloat, 256) do
            Float64(1 - sqrt(BigFloat(boundary_x)^2 +
                             BigFloat(boundary_y)^2))
        end
        @test boundary_weight > 0.0
        W_boundary, sums_boundary =
            build_filter_weights(boundary_mesh, 1.0)
        @test nnz(W_boundary) == 4
        @test W_boundary[1, 2] == boundary_weight
        @test W_boundary[2, 1] == boundary_weight
        @test sums_boundary == vec(sum(W_boundary, dims=2))

        @test size(W) == (Nt_df, Nt_df)
        # All weights non-negative (conic: max(0, r_min - d))
        @test all(nonzeros(W) .≥ 0)
        # Self-weight positive (distance to self = 0 < r_min)
        for t in 1:Nt_df
            @test W[t, t] > 0
        end
        # Normalization is positive
        @test all(w_sum .> 0)
    end

    # ── A: Filter preserves constant fields (exact) ──
    @testset "A: Filter preserves constants" begin
        for c in [0.0, 0.42, 1.0]
            rho_const = fill(c, Nt_df)
            rho_tilde = apply_filter(W, w_sum, rho_const)
            # ρ̃_t = Σ w_ts c / Σ w_ts = c (exact identity)
            @test isapprox(rho_tilde, fill(c, Nt_df), atol=1e-14)
        end
        @test_throws DimensionMismatch apply_filter(
            W, w_sum, zeros(Nt_df + 1))
        @test_throws DimensionMismatch apply_filter(
            W, vcat(w_sum, 1.0), zeros(Nt_df))
        @test_throws ArgumentError apply_filter(
            W, zeros(Nt_df), zeros(Nt_df))
        @test_throws ArgumentError apply_filter(
            W, w_sum, fill(Inf, Nt_df))
        nonfinite_filter = sparse([1], [1], [Inf], 1, 1)
        @test_throws ArgumentError apply_filter(
            nonfinite_filter, [1.0], [1.0])
        @test_throws ArgumentError apply_filter_transpose(
            nonfinite_filter, [1.0], [1.0])
        overflowing_filter = sparse(
            [1], [1], [floatmax(Float64)], 1, 1)
        @test_throws OverflowError apply_filter(
            overflowing_filter, [1.0], [2.0])
        @test_throws OverflowError apply_filter_transpose(
            overflowing_filter, [1.0], [2.0])
        filter_probe = rand(MersenneTwister(3901), Nt_df)
        @test _filter_allocation(W, w_sum, filter_probe) <=
              _float_vector_output_allocation(Nt_df) + 128
        @test _filter_transpose_allocation(W, w_sum, filter_probe) <=
              _float_vector_output_allocation(Nt_df) + 128
    end

    # ── A: Heaviside boundary values (exact by construction) ──
    @testset "A: Heaviside H(0)=0, H(1)=1, H(η)=0.5" begin
        @test_throws ArgumentError heaviside_project([0.5], 0.0)
        @test_throws ArgumentError heaviside_project([0.5], Inf)
        @test_throws ArgumentError heaviside_project([0.5], 1.0, -0.1)
        @test_throws ArgumentError heaviside_derivative([0.5], 1.0, 1.1)
        @test_throws ArgumentError heaviside_project([1e300, NaN], 2.0)
        @test_throws ArgumentError heaviside_derivative([1e300, NaN], 2.0)
        @test_throws ArgumentError heaviside_project(
            BigFloat[Inf], BigFloat(2))
        @test_throws ArgumentError heaviside_derivative(
            BigFloat[NaN], BigFloat(2))
        for beta in [1.0, 4.0, 16.0, 64.0]
            # H(0) = [tanh(βη) + tanh(-βη)] / denom = 0 (odd function cancels)
            @test heaviside_project([0.0], beta)[1] ≈ 0.0 atol=1e-14
            # H(1) = denom / denom = 1
            @test heaviside_project([1.0], beta)[1] ≈ 1.0 atol=1e-14
            # H(η=0.5) = tanh(β/2) / (2tanh(β/2)) = 0.5 for any β
            @test heaviside_project([0.5], beta)[1] ≈ 0.5 atol=1e-14
        end
        @test heaviside_project([0.01], 100.0, 0.5)[1] ==
              2.3767774103081315e-43
        @test heaviside_derivative([0.01], 100.0, 0.5)[1] ==
              5.49757001582043e-41
        projection_minsub_beta = nextfloat(0.0)
        @test heaviside_project(
            [0.5], projection_minsub_beta, 0.0) == [0.5]
        @test heaviside_derivative(
            [0.5], projection_minsub_beta, 0.0) == [1.0]
        projection_max_beta = floatmax(Float64)
        @test heaviside_project(
            [0.5], projection_max_beta, 0.5) == [0.5]
        @test heaviside_derivative(
            [0.5], projection_max_beta, 0.5) ==
              [projection_max_beta / 2]
        projection_big_input = BigFloat[BigFloat(1) / 3]
        projection_big_result = heaviside_project(
            projection_big_input, BigFloat(4), BigFloat(0.5))
        projection_big_derivative = heaviside_derivative(
            projection_big_input, BigFloat(4), BigFloat(0.5))
        @test eltype(projection_big_result) == BigFloat
        @test eltype(projection_big_derivative) == BigFloat
        projection_float32 = heaviside_project(
            Float32[0.25], Float32(4), Float32(0.5))
        projection_float32_derivative = heaviside_derivative(
            Float32[0.25], Float32(4), Float32(0.5))
        @test eltype(projection_float32) == Float32
        @test eltype(projection_float32_derivative) == Float32
    end

    # ── B: Heaviside monotonicity ──
    @testset "B: Heaviside monotonicity" begin
        rho_sorted = collect(range(0.0, 1.0, length=20))
        for beta in [1.0, 4.0, 64.0]
            rho_bar = heaviside_project(rho_sorted, beta)
            @test all(diff(rho_bar) .≥ -1e-15)
        end
    end

    # ── B: Heaviside range [0, 1] ──
    @testset "B: Heaviside output range" begin
        Random.seed!(88)
        rho_rand = rand(100)
        for beta in [1.0, 4.0, 64.0]
            rho_bar = heaviside_project(rho_rand, beta)
            @test all(rho_bar .≥ -1e-14)
            @test all(rho_bar .≤ 1.0 + 1e-14)
        end
    end

    # ── B: Filter adjoint identity: ⟨u, F(v)⟩ = ⟨Fᵀ(u), v⟩ ──
    @testset "B: Filter adjoint identity" begin
        Random.seed!(89)
        for trial in 1:5
            u = randn(Nt_df)
            v = randn(Nt_df)
            Fv = apply_filter(W, w_sum, v)
            Ftu = apply_filter_transpose(W, w_sum, u)
            # ⟨u, W*v/w⟩ = ⟨Wᵀ(u/w), v⟩
            # Tolerance: machine eps × vector norms
            @test isapprox(dot(u, Fv), dot(Ftu, v), rtol=1e-12)
        end
    end

    # ── B: Filter approximately preserves mean density ──
    @testset "B: Filter preserves mean" begin
        Random.seed!(90)
        rho_rand = rand(Nt_df)
        rho_tilde = apply_filter(W, w_sum, rho_rand)
        # Exact for uniform, approximate for random (boundary effects)
        @test abs(mean(rho_rand) - mean(rho_tilde)) < 0.05
    end

    # ── C: Increasing β drives toward binary design ──
    @testset "C: β continuation increases binarization" begin
        Random.seed!(91)
        rho_rand = rand(Nt_df)
        rho_tilde = apply_filter(W, w_sum, rho_rand)
        prev_frac = 0.0
        for beta in [1.0, 4.0, 16.0, 64.0]
            rho_bar = heaviside_project(rho_tilde, beta)
            near_binary = count(x -> x < 0.05 || x > 0.95, rho_bar) / Nt_df
            @test near_binary ≥ prev_frac - 0.01
            prev_frac = near_binary
        end
    end

    # ── F: Heaviside derivative vs central finite difference ──
    @testset "F: Heaviside derivative vs FD" begin
        rho_test = [0.2, 0.4, 0.6, 0.8]
        beta = 8.0
        h = 1e-7
        dH_ana = heaviside_derivative(rho_test, beta)
        for i in eachindex(rho_test)
            rho_plus = copy(rho_test); rho_plus[i] += h
            rho_minus = copy(rho_test); rho_minus[i] -= h
            H_plus = heaviside_project(rho_plus, beta)[i]
            H_minus = heaviside_project(rho_minus, beta)[i]
            dH_fd = (H_plus - H_minus) / (2h)
            # Tolerance: O(h²) FD error ≈ h²/6 × |d³H/dρ̃³|
            # For β=8, derivatives are moderate → rtol=1e-6 conservative
            @test isapprox(dH_ana[i], dH_fd, rtol=1e-6)
        end
    end

    # ── F: Full gradient chain rule vs FD of pipeline ──
    @testset "F: Gradient chain rule vs FD" begin
        Random.seed!(92)
        rho_raw = rand(Nt_df)
        beta = 4.0
        rho_tilde, rho_bar = filter_and_project(W, w_sum, rho_raw, beta)

        # Arbitrary gradient w.r.t. projected density (simulate adjoint output)
        g_rho_bar = randn(Nt_df)
        g_rho = gradient_chain_rule(g_rho_bar, rho_tilde, W, w_sum, beta)

        # Verify via FD: perturb rho_raw[t], measure change in dot(g_rho_bar, rho_bar)
        h = 1e-7
        for t in [1, div(Nt_df, 2), Nt_df]
            rho_plus = copy(rho_raw); rho_plus[t] += h
            rho_minus = copy(rho_raw); rho_minus[t] -= h
            _, rho_bar_plus = filter_and_project(W, w_sum, rho_plus, beta)
            _, rho_bar_minus = filter_and_project(W, w_sum, rho_minus, beta)
            # Directional derivative of the linear functional g_rho_bar' * rho_bar
            fd_val = (dot(g_rho_bar, rho_bar_plus) - dot(g_rho_bar, rho_bar_minus)) / (2h)
            @test isapprox(g_rho[t], fd_val, rtol=1e-5)
        end
    end
end
println("  PASS ✓")

# ─────────────────────────────────────────────────────────────────
# Test 40: DensityAdjoint — gradient computation
# ─────────────────────────────────────────────────────────────────
println("\n── Test 40: DensityAdjoint ──")

@testset "DensityAdjoint" begin
    Random.seed!(42)
    lambda_da = 0.03; k_da = 2π / lambda_da
    mesh_da = make_rect_plate(0.015, 0.015, 5, 5)
    rwg_da = build_rwg(mesh_da; precheck=false)
    Nt_da = ntriangles(mesh_da)
    N_da = rwg_da.nedges

    Z_efie_da = assemble_Z_efie(mesh_da, rwg_da, k_da; mesh_precheck=false)
    pw_da = make_plane_wave(Vec3(0.0, 0.0, -k_da), 1.0, Vec3(1.0, 0.0, 0.0))
    v_da = assemble_excitation(mesh_da, rwg_da, pw_da)
    Mt_da = precompute_triangle_mass(mesh_da, rwg_da)
    grid_da = make_sph_grid(10, 20)
    G_mat_da = radiation_vectors(mesh_da, rwg_da, grid_da, k_da)
    pol_da = pol_linear_x(grid_da)
    Q_da = build_Q(G_mat_da, grid_da, pol_da)
    config_da = DensityConfig(; p=3.0, Z_max_factor=1000.0)

    edge_len_da = 0.015 / 5
    r_min_da = 2.5 * edge_len_da
    W_da, w_sum_da = build_filter_weights(mesh_da, r_min_da)

    rho_da = 0.3 .+ 0.4 * rand(Nt_da)
    beta_da = 4.0
    rho_tilde_da, rho_bar_da = filter_and_project(W_da, w_sum_da, rho_da, beta_da)

    Z_pen_da = assemble_Z_penalty(Mt_da, rho_bar_da, config_da)
    Z_total_da = Z_efie_da + Z_pen_da
    I_sol_da = Z_total_da \ v_da
    J0_da = compute_objective(I_sol_da, Q_da)
    lambda_adj_da = solve_adjoint(Z_total_da, Q_da, I_sol_da)

    # ── B: Gradient is real-valued ──
    @testset "B: Gradient is real-valued" begin
        g = gradient_density(Mt_da, I_sol_da, lambda_adj_da, rho_bar_da, config_da)
        @test eltype(g) == Float64
        @test_throws DimensionMismatch gradient_density(
            Mt_da, I_sol_da, lambda_adj_da, vcat(rho_bar_da, 0.5), config_da)
        @test_throws DimensionMismatch gradient_density(
            Mt_da, vcat(I_sol_da, 0.0 + 0im), lambda_adj_da,
            rho_bar_da, config_da)
        @test_throws DimensionMismatch gradient_density_full(
            Mt_da, I_sol_da, lambda_adj_da,
            vcat(rho_tilde_da, 0.5), rho_bar_da, config_da,
            W_da, w_sum_da, beta_da)

        density_extreme_scale = floatmax(Float64)
        density_extreme_I = ComplexF64[
            density_extreme_scale,
            density_extreme_scale,
            density_extreme_scale,
            density_extreme_scale,
            1.0,
        ]
        density_extreme_lambda = copy(density_extreme_I)
        density_extreme_matrix = zeros(ComplexF64, 5, 5)
        density_extreme_matrix[1, 1:4] .= ComplexF64[
            density_extreme_scale,
            density_extreme_scale,
            -density_extreme_scale,
            -density_extreme_scale,
        ]
        density_extreme_matrix[5, 5] = 3.0
        density_extreme_reference = setprecision(BigFloat, 8192) do
            dot(
                Complex{BigFloat}.(density_extreme_lambda),
                Matrix{Complex{BigFloat}}(density_extreme_matrix),
                Complex{BigFloat}.(density_extreme_I),
            )
        end
        @test density_extreme_reference == Complex{BigFloat}(3)
        density_extreme_local = LocalMassMatrix(
            5,
            [1, 1, 1, 1, 5],
            [1, 2, 3, 4, 5],
            ComplexF64[
                density_extreme_scale,
                density_extreme_scale,
                -density_extreme_scale,
                -density_extreme_scale,
                3.0,
            ],
        )
        density_extreme_config = DensityConfig(1.0, 1.0, 0.5)
        for density_matrix in (
            density_extreme_matrix,
            sparse(density_extreme_matrix),
            density_extreme_local,
        )
            @test gradient_density(
                [density_matrix],
                density_extreme_I,
                density_extreme_lambda,
                [1.0],
                density_extreme_config,
            ) == [6.0]
        end
        density_extreme_reactive_config = DensityConfig(
            1.0, 0.0 + 1.0im, 0.5)
        @test gradient_density(
            [-im .* density_extreme_matrix],
            density_extreme_I,
            density_extreme_lambda,
            [1.0],
            density_extreme_reactive_config,
        ) == [6.0]

        density_cancellation_values = ComplexF64[1e16, 1.0, -1e16]
        density_cancellation_matrix = Matrix(Diagonal(
            density_cancellation_values))
        density_cancellation_local = LocalMassMatrix(
            3,
            [1, 2, 3],
            [1, 2, 3],
            density_cancellation_values,
        )
        density_cancellation_vector = ones(ComplexF64, 3)
        for density_matrix in (
            density_cancellation_matrix,
            sparse(density_cancellation_matrix),
            density_cancellation_local,
        )
            @test gradient_density(
                [density_matrix],
                density_cancellation_vector,
                density_cancellation_vector,
                [1.0],
                density_extreme_config,
            ) == [2.0]
        end

        # Each rounded complex product is zero here, although the exact real
        # component retained by the stored Float64 operands is representable.
        density_weighted_impedance = ComplexF64(
            1e16, nextfloat(1e16))
        density_weighted_matrix = reshape(
            ComplexF64[ComplexF64(nextfloat(1.0), 1.0)], 1, 1)
        density_weighted_config = DensityConfig(
            1.0, density_weighted_impedance, 0.5)
        @test 2 * real(
            density_weighted_impedance * density_weighted_matrix[1]) == 0.0
        density_weighted_reference = setprecision(BigFloat, 11008) do
            Float64(2 * (
                BigFloat(real(density_weighted_impedance)) *
                BigFloat(real(density_weighted_matrix[1])) -
                BigFloat(imag(density_weighted_impedance)) *
                BigFloat(imag(density_weighted_matrix[1]))))
        end
        @test gradient_density(
            [density_weighted_matrix],
            ComplexF64[1.0],
            ComplexF64[1.0],
            [1.0],
            density_weighted_config,
        ) == [density_weighted_reference]
        @test_throws ArgumentError gradient_density(
            [reshape(ComplexF64[1.0], 1, 1)],
            ComplexF64[NaN], ComplexF64[1.0], [1.0],
            density_extreme_config)
        @test_throws ArgumentError gradient_density(
            [reshape(ComplexF64[1.0], 1, 1)],
            ComplexF64[1.0], ComplexF64[Inf], [1.0],
            density_extreme_config)
        @test_throws ArgumentError gradient_density(
            [reshape(ComplexF64[NaN], 1, 1)],
            ComplexF64[1.0], ComplexF64[1.0], [1.0],
            density_extreme_config)
        @test_throws OverflowError gradient_density(
            [reshape(ComplexF64[1.0], 1, 1)],
            ComplexF64[density_extreme_scale],
            ComplexF64[density_extreme_scale],
            [1.0],
            density_extreme_config)
        density_scale_config = DensityConfig(
            floatmax(Float64), 1.0, 0.5)
        @test 2 * density_scale_config.p * 0.25 == Inf
        @test gradient_density(
            [reshape(ComplexF64[0.25], 1, 1)],
            ComplexF64[1.0],
            ComplexF64[1.0],
            [1.0],
            density_scale_config,
        ) == [0.5 * floatmax(Float64)]

        density_allocation_matrices = [ComplexF64[2.0 0.0; 0.0 3.0]]
        density_allocation_vector = ComplexF64[1.0, 2.0]
        density_allocation_rho = [0.5]
        gradient_density(
            density_allocation_matrices,
            density_allocation_vector,
            density_allocation_vector,
            density_allocation_rho,
            config_da,
        )
        @test @allocated(gradient_density(
            density_allocation_matrices,
            density_allocation_vector,
            density_allocation_vector,
            density_allocation_rho,
            config_da,
        )) <= _float_vector_output_allocation(1) + 128
    end

    # ── A: Zero gradient for zero objective ──
    @testset "A: Zero gradient for zero Q" begin
        # If Q = 0, adjoint λ = 0 → gradient = 0
        Q_zero = zeros(ComplexF64, N_da, N_da)
        lambda_zero = solve_adjoint(Z_total_da, Q_zero, I_sol_da)
        g_zero = gradient_density(Mt_da, I_sol_da, lambda_zero, rho_bar_da, config_da)
        @test norm(g_zero) < 1e-14
    end

    # ── F: Full adjoint gradient vs central finite difference (CRITICAL) ──
    @testset "F: Full gradient vs finite difference" begin
        g_adj = gradient_density_full(Mt_da, I_sol_da, lambda_adj_da,
                                      rho_tilde_da, rho_bar_da, config_da,
                                      W_da, w_sum_da, beta_da)
        h = 1e-5
        n_check = min(5, Nt_da)
        check_indices = sort(randperm(Nt_da)[1:n_check])
        max_rel_err = 0.0

        for t in check_indices
            rho_plus = copy(rho_da); rho_plus[t] += h
            rho_minus = copy(rho_da); rho_minus[t] -= h

            _, rho_bar_plus = filter_and_project(W_da, w_sum_da, rho_plus, beta_da)
            Z_plus = Z_efie_da + assemble_Z_penalty(Mt_da, rho_bar_plus, config_da)
            J_plus = compute_objective(Z_plus \ v_da, Q_da)

            _, rho_bar_minus = filter_and_project(W_da, w_sum_da, rho_minus, beta_da)
            Z_minus = Z_efie_da + assemble_Z_penalty(Mt_da, rho_bar_minus, config_da)
            J_minus = compute_objective(Z_minus \ v_da, Q_da)

            g_fd = (J_plus - J_minus) / (2h)
            rel_err = abs(g_adj[t] - g_fd) / max(abs(g_fd), 1e-20)
            max_rel_err = max(max_rel_err, rel_err)

            # Tolerance: adjoint is exact; FD error = O(h²) ~ 1e-10
            # Divided by |g_fd| ~ 1e-13, gives rtol ~ 1e-3 to 1e-4
            @test rel_err < 1e-4
        end
        println("    Max adjoint vs FD relative error: $(round(max_rel_err, sigdigits=3))")
    end

    # ── G: Reactive Z_max gradient keeps complex coefficient inside Re{...} ──
    @testset "G: Reactive density gradient vs finite difference" begin
        config_rx = DensityConfig(; p=3.0, Z_max_factor=1000.0, reactive=true)
        Z_rx = Z_efie_da + assemble_Z_penalty(Mt_da, rho_bar_da, config_rx)
        I_rx = Z_rx \ v_da
        lambda_rx = solve_adjoint(Z_rx, Q_da, I_rx)
        g_rx = gradient_density_full(Mt_da, I_rx, lambda_rx,
                                     rho_tilde_da, rho_bar_da, config_rx,
                                     W_da, w_sum_da, beta_da)

        h = 1e-5
        check_indices = sort(randperm(Nt_da)[1:min(3, Nt_da)])
        max_rel_err = 0.0
        for t in check_indices
            rho_plus = copy(rho_da); rho_plus[t] += h
            rho_minus = copy(rho_da); rho_minus[t] -= h

            _, rho_bar_plus = filter_and_project(W_da, w_sum_da, rho_plus, beta_da)
            Z_plus = Z_efie_da + assemble_Z_penalty(Mt_da, rho_bar_plus, config_rx)
            J_plus = compute_objective(Z_plus \ v_da, Q_da)

            _, rho_bar_minus = filter_and_project(W_da, w_sum_da, rho_minus, beta_da)
            Z_minus = Z_efie_da + assemble_Z_penalty(Mt_da, rho_bar_minus, config_rx)
            J_minus = compute_objective(Z_minus \ v_da, Q_da)

            g_fd = (J_plus - J_minus) / (2h)
            rel_err = abs(g_rx[t] - g_fd) / max(abs(g_fd), 1e-20)
            max_rel_err = max(max_rel_err, rel_err)
            @test rel_err < 1e-4
        end
        println("    Reactive Z_max adjoint vs FD relative error: $(round(max_rel_err, sigdigits=3))")
    end
end
println("  PASS ✓")

# ─────────────────────────────────────────────────────────────────
# Test 41: PeriodicEFIE — periodic EFIE assembly
# ─────────────────────────────────────────────────────────────────
println("\n── Test 41: PeriodicEFIE ──")

@testset "PeriodicEFIE" begin
    lambda_pe = 0.03; k_pe = 2π / lambda_pe
    dx_pe = 0.5 * lambda_pe; dy_pe = 0.5 * lambda_pe
    mesh_pe = make_rect_plate(dx_pe, dy_pe, 3, 3)
    lat_pe = PeriodicLattice(dx_pe, dy_pe, 0.0, 0.0, k_pe; N_spatial=2, N_spectral=2)
    rwg_pe = build_rwg_periodic(mesh_pe, lat_pe; precheck=false)
    N_pe = rwg_pe.nedges

    # ── A: Output dimensions ──
    @testset "A: Output dimensions" begin
        Z_per = assemble_Z_efie_periodic(mesh_pe, rwg_pe, k_pe, lat_pe)
        @test size(Z_per) == (N_pe, N_pe)
        @test eltype(Z_per) == ComplexF64
        periodic_work_bytes = 3 * sizeof(ComplexF64) * N_pe^2
        @test_throws ArgumentError assemble_Z_efie_periodic(
            mesh_pe, rwg_pe, k_pe, lat_pe;
            max_work_bytes=periodic_work_bytes - 1,
        )
        @test assemble_Z_efie_periodic(
            mesh_pe, rwg_pe, k_pe, lat_pe;
            max_work_bytes=periodic_work_bytes,
        ) ≈ Z_per
        @test_throws ArgumentError assemble_Z_efie_periodic(
            mesh_pe, rwg_pe, 1.01k_pe, lat_pe
        )
        @test_throws ArgumentError assemble_Z_efie_periodic(
            mesh_pe, rwg_pe, k_pe, lat_pe; max_cache_bytes=0)
        @test_throws ArgumentError assemble_Z_efie_periodic(
            mesh_pe, rwg_pe, k_pe, lat_pe; max_adjacency_pairs=-1)
        @test_throws ArgumentError assemble_Z_efie_periodic(
            mesh_pe, rwg_pe, k_pe, lat_pe; max_green_terms=0)

        range_cell = 1.0
        range_k = (1 - 1e-12) * 2π / range_cell
        range_mesh = make_rect_plate(range_cell, range_cell, 1, 1)
        range_lattice = PeriodicLattice(
            range_cell, range_cell, 0.0, 0.0, range_k;
            N_spatial=1, N_spectral=1)
        range_rwg = build_rwg_periodic(
            range_mesh, range_lattice; precheck=false)
        range_eta = 0.5 * floatmax(Float64) / range_k
        @test_throws OverflowError assemble_Z_efie_periodic(
            range_mesh, range_rwg, range_k, range_lattice;
            quad_order=1, eta0=range_eta)
        @test_throws OverflowError DiffMoM._assemble_periodic_image_block(
            range_mesh, range_rwg, range_k, range_lattice, 0.2;
            quad_order=1, eta0=range_eta)

        _, periodic_weights = tri_quad_rule(3)
        incidence = DiffMoM._build_periodic_triangle_incidence(
            rwg_pe, ntriangles(mesh_pe))
        Tcoef = promote_type(
            eltype(rwg_pe.coeff_plus), eltype(rwg_pe.coeff_minus))
        TVec = SVector{3,Tcoef}
        spatial_term_count = DiffMoM._periodic_term_count(
            lat_pe.N_spatial) - 1
        spectral_term_count = DiffMoM._periodic_term_count(
            lat_pe.N_spectral)
        periodic_cache_bytes = DiffMoM._periodic_efie_cache_bytes(
            rwg_pe.nedges, ntriangles(mesh_pe), length(periodic_weights),
            max(1, min(Threads.nthreads(), ntriangles(mesh_pe))),
            incidence.max_incident, Tcoef, TVec;
            point_cache_count=1,
            spatial_term_count=spatial_term_count,
            spectral_term_count=spectral_term_count)
        periodic_green_terms = DiffMoM._periodic_efie_green_terms(
            incidence.active_triangles, length(periodic_weights), lat_pe,
            DiffMoM._periodic_correction_is_symmetric(rwg_pe, lat_pe))
        @test_throws ArgumentError DiffMoM._assemble_periodic_correction(
            mesh_pe, rwg_pe, k_pe, lat_pe;
            max_cache_bytes=periodic_cache_bytes - 1)
        @test_throws ArgumentError DiffMoM._assemble_periodic_correction(
            mesh_pe, rwg_pe, k_pe, lat_pe;
            max_cache_bytes=periodic_cache_bytes,
            max_green_terms=periodic_green_terms - 1)
        Z_corr_limited = DiffMoM._assemble_periodic_correction(
            mesh_pe, rwg_pe, k_pe, lat_pe;
            max_cache_bytes=periodic_cache_bytes,
            max_green_terms=periodic_green_terms)
        @test all(isfinite, Z_corr_limited)

        # Lattice-dependent Ewald data are cached once. At a Wood-mode lattice,
        # this avoids thousands of repeated BigFloat longitudinal-wavevector
        # fallbacks inside the triangle-pair loop.
        DiffMoM._assemble_periodic_correction(
            mesh_pe, rwg_pe, k_pe, lat_pe;
            max_cache_bytes=periodic_cache_bytes,
            max_green_terms=periodic_green_terms)
        periodic_alloc = @allocated DiffMoM._assemble_periodic_correction(
            mesh_pe, rwg_pe, k_pe, lat_pe;
            max_cache_bytes=periodic_cache_bytes,
            max_green_terms=periodic_green_terms)
        @test periodic_alloc <= 1_000_000

        spatial_terms, spectral_terms =
            DiffMoM._build_periodic_ewald_terms(lat_pe, 1)
        rp_cache = Vec3(0.11dx_pe, -0.07dy_pe, 0.0)
        r_cache = Vec3(-0.13dx_pe, 0.09dy_pe, 0.0)
        @test DiffMoM._greens_periodic_correction_cached(
            r_cache, rp_cache, k_pe, lat_pe,
            spatial_terms, spectral_terms) ==
            greens_periodic_correction(r_cache, rp_cache, k_pe, lat_pe)
    end

    @testset "A: Auxiliary cache rejects before mesh-sized allocation" begin
        empty_triangle_count = 1_000
        empty_xyz = Matrix{Float64}(undef, 3, 3empty_triangle_count)
        empty_tri = Matrix{Int}(undef, 3, empty_triangle_count)
        for t in 1:empty_triangle_count
            first_vertex = 3t - 2
            empty_xyz[:, first_vertex] .= (0.0, 0.0, 0.0)
            empty_xyz[:, first_vertex + 1] .= (0.01, 0.0, 0.0)
            empty_xyz[:, first_vertex + 2] .= (0.0, 0.01, 0.0)
            empty_tri[:, t] .=
                (first_vertex, first_vertex + 1, first_vertex + 2)
        end
        empty_mesh = TriMesh(empty_xyz, empty_tri)
        empty_rwg = build_rwg(empty_mesh; precheck=false)
        empty_lattice = PeriodicLattice(
            1.0, 1.0, 0.0, 0.0, k_pe;
            N_spatial=1, N_spectral=1)
        @test empty_rwg.nedges == 0

        empty_incidence = DiffMoM._build_periodic_triangle_incidence(
            empty_rwg, empty_triangle_count)
        empty_cache_bytes = DiffMoM._periodic_efie_cache_bytes(
            0, empty_triangle_count, 1,
            max(1, min(Threads.nthreads(), empty_triangle_count)),
            empty_incidence.max_incident, Float64, Vec3;
            point_cache_count=1)
        @test size(DiffMoM._assemble_periodic_correction(
            empty_mesh, empty_rwg, k_pe, empty_lattice;
            quad_order=1,
            max_cache_bytes=empty_cache_bytes,
            max_green_terms=1)) == (0, 0)

        try
            assemble_Z_efie_periodic(
                empty_mesh, empty_rwg, k_pe, empty_lattice;
                quad_order=1, max_cache_bytes=1)
        catch
        end
        rejected_alloc = @allocated try
            assemble_Z_efie_periodic(
                empty_mesh, empty_rwg, k_pe, empty_lattice;
                quad_order=1, max_cache_bytes=1)
        catch
        end
        @test rejected_alloc <= 10_000
    end

    # ── A: Bloch-paired RWG required for boundary-touching periodic meshes ──
    @testset "A: Bloch-paired RWG required + full-PEC sanity" begin
        dx_fp = 1.2 * lambda_pe
        dy_fp = 1.2 * lambda_pe
        mesh_fp = make_rect_plate(dx_fp, dy_fp, 3, 3)
        lat_fp = PeriodicLattice(dx_fp, dy_fp, 0.0, 0.0, k_pe; N_spatial=2, N_spectral=2)

        rwg_nonbloch = build_rwg(mesh_fp; precheck=false)
        rwg_bloch = build_rwg_periodic(mesh_fp, lat_fp; precheck=false)
        @test rwg_bloch.has_periodic_bloch
        @test rwg_bloch.nedges > rwg_nonbloch.nedges

        function solve_modes(mesh, rwg, lat)
            Z = assemble_Z_efie_periodic(mesh, rwg, k_pe, lat; quad_order=3)
            pw = make_plane_wave(Vec3(0.0, 0.0, -k_pe), 1.0, Vec3(1.0, 0.0, 0.0))
            v = assemble_excitation(mesh, rwg, pw; quad_order=3)
            I = Z \ v
            return reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I), k_pe, lat;
                                           N_orders=2, E0=1.0, pol=SVector(1.0, 0.0, 0.0))
        end

        @test_throws ArgumentError solve_modes(mesh_fp, rwg_nonbloch, lat_fp)
        modes_b, R_b = solve_modes(mesh_fp, rwg_bloch, lat_fp)
        idx00_b = findfirst(m -> (m.m == 0 && m.n == 0), modes_b)
        @test idx00_b !== nothing

        idxm10_b = findfirst(m -> (m.m == -1 && m.n == 0), modes_b)
        idxp10_b = findfirst(m -> (m.m == 1 && m.n == 0), modes_b)
        @test idxm10_b !== nothing && idxp10_b !== nothing

        pord_b = abs2(R_b[idxm10_b]) * real(modes_b[idxm10_b].kz) / k_pe +
                 abs2(R_b[idxp10_b]) * real(modes_b[idxp10_b].kz) / k_pe

        @test abs(R_b[idx00_b]) > 0.99
        @test pord_b < 5e-4
    end

    # ── A: Oblique incidence yields nontrivial Bloch phase in paired RWG coefficients ──
    @testset "A: Bloch coefficient phase (oblique)" begin
        dx_fp = 1.2 * lambda_pe
        dy_fp = 1.2 * lambda_pe
        mesh_fp = make_rect_plate(dx_fp, dy_fp, 3, 3)
        lat_obl = PeriodicLattice(dx_fp, dy_fp, π/6, 0.0, k_pe; N_spatial=2, N_spectral=2)
        rwg_bloch = build_rwg_periodic(mesh_fp, lat_obl; precheck=false)
        @test rwg_bloch.has_periodic_bloch
        @test any(abs.(imag.(rwg_bloch.coeff_minus)) .> 1e-8)
    end

    @testset "D: Periodic RWG range, tolerance, and resource safety" begin
        # Boundary tolerances are scale-aware per axis. No fixed metric floor
        # may erase a sub-picometer cell, and a long x period must not inflate
        # the y-boundary tolerance.
        for (dx_safe, dy_safe) in (
                (1e-15, 1e-15),
                (2e-12, 2e-12),
                (1e12, 1.0))
            mesh_safe = make_rect_plate(dx_safe, dy_safe, 1, 1)
            lat_safe = PeriodicLattice(
                dx_safe, dy_safe, 0.0, 0.0, 1.0, 1.0, 0, 0)
            rwg_safe = build_rwg_periodic(
                mesh_safe, lat_safe; precheck=false)
            @test rwg_safe.nedges == 3
            @test rwg_safe.has_periodic_bloch
        end
        # The assembly-side guard must use the same scale-aware tolerances.
        # A conductor strictly inside a sub-picometer cell needs no Bloch edge
        # pairing and must not be mistaken for a boundary-touching mesh.
        tiny_guard_period = 1e-15
        tiny_guard_mesh = make_rect_plate(
            0.2tiny_guard_period, 0.2tiny_guard_period, 1, 1)
        tiny_guard_lattice = PeriodicLattice(
            tiny_guard_period, tiny_guard_period, 0.0, 0.0, 1.0)
        tiny_guard_rwg = build_rwg(tiny_guard_mesh; precheck=false)
        @test !tiny_guard_rwg.has_periodic_bloch
        @test !DiffMoM._mesh_has_unitcell_boundary_edges(
            tiny_guard_mesh, tiny_guard_lattice)
        tiny_guard_matrix = assemble_Z_efie_periodic(
            tiny_guard_mesh, tiny_guard_rwg, 1.0, tiny_guard_lattice;
            quad_order=1)
        @test all(isfinite, tiny_guard_matrix)

        # Coplanarity is also a geometric boundary decision. A subtraction
        # rounded back to the tolerance must not hide an exact excess.
        coplanar_tolerance = 1e-12
        coplanar_delta = eps(coplanar_tolerance) / 4
        rounded_nonplanar_mesh = TriMesh(
            Float64[
                0 1 0
                0 0 1
                -coplanar_delta coplanar_tolerance coplanar_tolerance
            ],
            reshape(1:3, 3, 1),
        )
        @test abs(maximum(rounded_nonplanar_mesh.xyz[3, :]) -
                  minimum(rounded_nonplanar_mesh.xyz[3, :])) ==
              coplanar_tolerance
        @test_throws ArgumentError DiffMoM._assert_coplanar_periodic_mesh(
            rounded_nonplanar_mesh; atol=coplanar_tolerance)
        @test_throws ArgumentError DiffMoM._assert_coplanar_periodic_metrics_mesh(
            rounded_nonplanar_mesh; atol=coplanar_tolerance)

        # One relative tolerance can underflow to exact matching while the
        # other remains positive; zero-width hash dimensions stay valid.
        mixed_dx = 1e-323
        mixed_mesh = make_rect_plate(mixed_dx, 1.0, 1, 1)
        mixed_lattice = PeriodicLattice(
            mixed_dx, 1.0, 0.0, 0.0, 1.0, 1.0, 0, 0)
        mixed_rwg = build_rwg_periodic(
            mixed_mesh, mixed_lattice; precheck=false)
        @test mixed_rwg.nedges == 3
        mixed_basis = 1
        mixed_triangle = mixed_rwg.tplus[mixed_basis]
        mixed_point = Vec3(
            mixed_mesh.xyz[:, mixed_mesh.tri[1, mixed_triangle]])
        mixed_value_reference = setprecision(BigFloat, 2304) do
            opposite = mixed_rwg.vplus_opp[mixed_basis]
            scale = Complex{BigFloat}(mixed_rwg.coeff_plus[mixed_basis]) *
                    BigFloat(mixed_rwg.len[mixed_basis]) /
                    (BigFloat(2) * BigFloat(
                        mixed_rwg.area_plus[mixed_basis]))
            SVector{3,ComplexF64}(ntuple(component -> ComplexF64(
                scale * (BigFloat(mixed_point[component]) -
                         BigFloat(mixed_mesh.xyz[component, opposite]))), 3))
        end
        @test eval_rwg(
            mixed_rwg, mixed_basis, mixed_point,
            mixed_triangle) == mixed_value_reference
        @test all(isfinite, mixed_value_reference)
        mixed_centroid = triangle_center(mixed_mesh, mixed_triangle)
        @test_throws OverflowError eval_rwg(
            mixed_rwg, mixed_basis, mixed_centroid, mixed_triangle)
        @test_throws OverflowError div_rwg(
            mixed_rwg, mixed_basis, mixed_triangle)
        huge_cell_size = 1.4e154
        huge_cell_mesh = make_rect_plate(
            huge_cell_size, huge_cell_size, 1, 1)
        huge_cell_lattice = PeriodicLattice(
            huge_cell_size, huge_cell_size, 0.0, 0.0,
            1e-154, 1.0, 0, 0)
        huge_cell_rwg = build_rwg_periodic(
            huge_cell_mesh, huge_cell_lattice; precheck=false)
        huge_basis = 1
        huge_triangle = huge_cell_rwg.tplus[huge_basis]
        huge_point = Vec3(
            huge_cell_mesh.xyz[:, huge_cell_mesh.tri[1, huge_triangle]])
        huge_value = eval_rwg(
            huge_cell_rwg, huge_basis, huge_point, huge_triangle)
        huge_reference = setprecision(BigFloat, 2304) do
            opposite = huge_cell_rwg.vplus_opp[huge_basis]
            scale = Complex{BigFloat}(huge_cell_rwg.coeff_plus[huge_basis]) *
                    BigFloat(huge_cell_rwg.len[huge_basis]) /
                    (BigFloat(2) * BigFloat(
                        huge_cell_rwg.area_plus[huge_basis]))
            SVector{3,ComplexF64}(ntuple(component -> ComplexF64(
                scale * (BigFloat(huge_point[component]) -
                         BigFloat(huge_cell_mesh.xyz[component, opposite]))), 3))
        end
        @test huge_value == huge_reference
        @test all(isfinite, huge_value)
        phase_cell_dx = 1e308
        phase_cell_mesh = make_rect_plate(phase_cell_dx, 1.0, 1, 1)
        phase_cell_lattice = PeriodicLattice(
            phase_cell_dx, 1.0, (Float64(π) / 2) / phase_cell_dx, 0.0,
            1e-308, 1.0, 0, 0)
        phase_cell_rwg = build_rwg_periodic(
            phase_cell_mesh, phase_cell_lattice; precheck=false)
        phase_basis = findfirst(
            coefficient -> !iszero(imag(coefficient)),
            phase_cell_rwg.coeff_minus)
        @test phase_basis !== nothing
        phase_triangle = phase_cell_rwg.tminus[phase_basis]
        phase_opposite = phase_cell_rwg.vminus_opp[phase_basis]
        phase_vertex = argmax(local_vertex -> begin
            vertex = phase_cell_mesh.tri[local_vertex, phase_triangle]
            abs(phase_cell_mesh.xyz[1, vertex] -
                phase_cell_mesh.xyz[1, phase_opposite])
        end, 1:3)
        phase_point = Vec3(
            phase_cell_mesh.xyz[:, phase_cell_mesh.tri[phase_vertex, phase_triangle]])
        phase_value = eval_rwg(
            phase_cell_rwg, phase_basis, phase_point, phase_triangle)
        phase_reference = setprecision(BigFloat, 2304) do
            scale = Complex{BigFloat}(
                        phase_cell_rwg.coeff_minus[phase_basis]) *
                    BigFloat(phase_cell_rwg.len[phase_basis]) /
                    (BigFloat(2) * BigFloat(
                        phase_cell_rwg.area_minus[phase_basis]))
            SVector{3,ComplexF64}(ntuple(component -> ComplexF64(
                scale * (BigFloat(
                             phase_cell_mesh.xyz[component, phase_opposite]) -
                         BigFloat(phase_point[component]))), 3))
        end
        @test phase_value ≈ phase_reference rtol=0.0 atol=4eps(Float64)
        @test !iszero(real(phase_value[1]))
        ordinary_rwg_mesh = make_rect_plate(1.0, 1.0, 2, 2)
        ordinary_rwg = build_rwg(ordinary_rwg_mesh; precheck=false)
        ordinary_basis = 1
        ordinary_triangle = ordinary_rwg.tplus[ordinary_basis]
        ordinary_point = triangle_center(
            ordinary_rwg_mesh, ordinary_triangle)
        eval_rwg(
            ordinary_rwg, ordinary_basis, ordinary_point, ordinary_triangle)
        div_rwg(ordinary_rwg, ordinary_basis, ordinary_triangle)
        @test @allocated(eval_rwg(
            ordinary_rwg, ordinary_basis, ordinary_point,
            ordinary_triangle)) == 0
        @test @allocated(div_rwg(
            ordinary_rwg, ordinary_basis, ordinary_triangle)) == 0
        # Preserve the historical 1e-12 import-noise floor for ordinary cells.
        legacy_size = 1e-6
        legacy_mesh = make_rect_plate(legacy_size, legacy_size, 1, 1)
        legacy_mesh.xyz[1, legacy_mesh.xyz[1, :] .== -0.5legacy_size] .+= 5e-13
        legacy_lattice = PeriodicLattice(
            legacy_size, legacy_size, 0.0, 0.0, 1.0, 1.0, 0, 0)
        @test build_rwg_periodic(
            legacy_mesh, legacy_lattice; precheck=false).nedges == 3

        mesh_tol = make_rect_plate(2.0, 1.0, 1, 1)
        lat_tol = PeriodicLattice(2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0, 0)
        for bad_tolerance in (-1.0, Inf, -Inf, NaN)
            @test_throws ArgumentError build_rwg_periodic(
                mesh_tol, lat_tol; precheck=false,
                boundary_atol_abs=bad_tolerance)
            @test_throws ArgumentError build_rwg_periodic(
                mesh_tol, lat_tol; precheck=false,
                boundary_atol_rel=bad_tolerance)
        end
        @test_throws ArgumentError build_rwg_periodic(
            mesh_tol, lat_tol; precheck=false, boundary_atol_abs=0.5)
        @test_throws ArgumentError build_rwg_periodic(
            mesh_tol, lat_tol; precheck=false, boundary_atol_rel=0.5)

        # The exact Float64-input product defines the phase. These cases cover
        # a badly reduced finite product and products that overflow Float64.
        for (wavenumber, period) in (
            (1e100, 1e100),
            (1e200, 1e200),
            (floatmax(Float64), floatmax(Float64)),
        )
            phase_reference = setprecision(BigFloat, 2304) do
                ComplexF64(cis(-BigFloat(wavenumber) * BigFloat(period)))
            end
            @test DiffMoM._periodic_rwg_bloch_phase(
                wavenumber, period) ≈ phase_reference rtol=0.0 atol=8eps(Float64)
        end
        DiffMoM._periodic_rwg_bloch_phase(2.0, 3.0)
        @test @allocated(
            DiffMoM._periodic_rwg_bloch_phase(2.0, 3.0)) == 0

        mesh_phase = make_rect_plate(2.0, 1.0, 1, 1)
        lat_phase = PeriodicLattice(2.0, 1.0, 1e308, 0.0,
                                    1.0, 1.0, 0, 0)
        rwg_phase = build_rwg_periodic(
            mesh_phase, lat_phase; precheck=false)
        phase_reference = setprecision(BigFloat, 2304) do
            ComplexF64(cis(-BigFloat(1e308) * BigFloat(2.0)))
        end
        @test any(==(phase_reference), rwg_phase.coeff_minus)
        @test all(isfinite, rwg_phase.coeff_minus)

        # Pairing accepts reversed endpoint order and remains independent of
        # side-B ordering. Duplicate geometric candidates are rejected.
        PeriodicEdge = DiffMoM._PeriodicBoundaryEdge
        function synthetic_edge(y1, z1, y2, z2, x, id)
            return PeriodicEdge(
                (id, id + 1), id, 1,
                Vec3(x, y1, z1), Vec3(x, y2, z2))
        end
        side_a = [synthetic_edge(0.0, 10.0, 1.0, 0.0, -1.0, 1)]
        side_b = [synthetic_edge(0.9, 0.1, 0.1, 9.9, 1.0, 3)]
        @test DiffMoM._periodic_boundary_matches(
            side_a, side_b, :x, (0.2, 0.2)) == [1]
        # Independent endpoint-coordinate ordering can change under a legal
        # perturbation. The orientation-invariant index must still find it.
        crossing_a = [synthetic_edge(0.0, 0.0, 0.1, 100.0, -1.0, 1)]
        crossing_b = [synthetic_edge(0.15, 0.0, -0.05, 100.0, 1.0, 3)]
        @test DiffMoM._periodic_boundary_matches(
            crossing_a, crossing_b, :x, (0.2, 0.2)) == [1]
        unequal_tol_a = [synthetic_edge(0.0, 0.0, 100.0, 1.0, -1.0, 1)]
        unequal_tol_b = [synthetic_edge(0.5, 0.0, 100.5, 1.0, 1.0, 3)]
        @test DiffMoM._periodic_boundary_matches(
            unequal_tol_a, unequal_tol_b, :x, (1.0, 0.01)) == [1]
        signed_zero_a = [
            synthetic_edge(-0.0, 0.0, 1.0, -0.0, -1.0, 1),
        ]
        signed_zero_b = [
            synthetic_edge(1.0, 0.0, 0.0, -0.0, 1.0, 3),
        ]
        @test DiffMoM._periodic_boundary_matches(
            signed_zero_a, signed_zero_b, :x, (0.0, 0.0)) == [1]
        full_range_a = [synthetic_edge(
            -floatmax(Float64), nextfloat(0.0),
            floatmax(Float64), 1.0, -1.0, 1)]
        full_range_b = [synthetic_edge(
            floatmax(Float64), 1.0,
            -floatmax(Float64), nextfloat(0.0), 1.0, 3)]
        @test DiffMoM._periodic_boundary_matches(
            full_range_a, full_range_b, :x, (1e-300, 1e-300)) == [1]
        duplicate_b = [
            synthetic_edge(0.1, 0.0, 1.1, 0.0, 1.0, 3),
            synthetic_edge(-0.1, 0.0, 0.9, 0.0, 1.0, 5),
        ]
        ambiguous_a = [
            synthetic_edge(0.0, 0.0, 1.0, 0.0, -1.0, 1),
            synthetic_edge(100.0, 0.0, 101.0, 0.0, -1.0, 7),
        ]
        @test_throws ArgumentError DiffMoM._periodic_boundary_matches(
            ambiguous_a, duplicate_b, :x, (0.2, 0.2))
        @test_throws ArgumentError DiffMoM._periodic_boundary_matches(
            ambiguous_a, reverse(duplicate_b), :x, (0.2, 0.2))
        @test_throws ArgumentError DiffMoM._periodic_boundary_matches(
            ambiguous_a[1:1], PeriodicEdge[], :x, (0.2, 0.2))

        # Warm allocation must scale linearly with the boundary-edge count.
        function periodic_strip_allocation(n)
            mesh = make_rect_plate(1.0, 1.0, 1, n)
            lattice = PeriodicLattice(
                1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0, 0)
            rwg = build_rwg_periodic(mesh, lattice; precheck=false)
            rwg.nedges == 3n || error("periodic strip edge-count invariant failed")
            return nothing
        end
        periodic_strip_allocation(32)
        GC.gc()
        allocation_1000 = @allocated periodic_strip_allocation(1000)
        GC.gc()
        allocation_2000 = @allocated periodic_strip_allocation(2000)
        @test allocation_2000 <= 2.5 * allocation_1000
        @test allocation_2000 < 12_000_000
    end

    # ── F: ACA dense blocks match EFIE for complex Bloch RWG coefficients ──
    @testset "F: ACA dense blocks with Bloch RWG" begin
        dx_fp = 1.2 * lambda_pe
        dy_fp = 1.2 * lambda_pe
        mesh_fp = make_rect_plate(dx_fp, dy_fp, 4, 4)
        lat_obl = PeriodicLattice(dx_fp, dy_fp, π/6, 0.0, k_pe; N_spatial=1, N_spectral=1)
        rwg_bloch = build_rwg_periodic(mesh_fp, lat_obl; precheck=false)
        Z_ref = assemble_Z_efie(mesh_fp, rwg_bloch, k_pe; mesh_precheck=false, quad_order=3)
        A_aca = build_aca_operator(mesh_fp, rwg_bloch, k_pe;
                                   leaf_size=8, eta=0.1, aca_tol=1e-12,
                                   max_rank=20, quad_order=3)
        x = ComplexF64[sin(i) + 1im * cos(i) for i in 1:rwg_bloch.nedges]
        rel = norm(A_aca * x - Z_ref * x) / norm(Z_ref * x)
        @test rel < 1e-12
    end

    # ── A: Large period → Z_per approaches Z_free ──
    @testset "A: Large period → Z_per approaches Z_free" begin
        Z_free = assemble_Z_efie(mesh_pe, rwg_pe, k_pe; mesh_precheck=false)

        # Use non-integer d/λ to avoid Wood anomaly (kz=0 singularity).
        # At Wood anomaly (integer d/λ at normal incidence), the periodic
        # Green's function diverges — a physical singularity, not a bug.
        #
        # For non-Wood-anomaly periods, the Ewald sum is exact (E-independent
        # to machine precision) and works for arbitrarily large d/λ.
        #
        # Test: relative difference ‖Z_per - Z_free‖ / ‖Z_free‖ decreases
        # monotonically as d increases (images move farther away).
        prev_rel = Inf
        for alpha in [2.5, 10.5]
            d = alpha * lambda_pe
            lat = PeriodicLattice(d, d, 0.0, 0.0, k_pe; N_spatial=2, N_spectral=2)
            Z_per = assemble_Z_efie_periodic(mesh_pe, rwg_pe, k_pe, lat)
            @test !any(isnan, Z_per)
            @test !any(isinf, Z_per)
            rel_diff = norm(Z_per - Z_free) / norm(Z_free)
            # Correction shrinks as period grows (images recede)
            @test rel_diff < prev_rel
            prev_rel = rel_diff
        end
        # At d = 10.5λ, correction < 5% of free-space impedance
        @test prev_rel < 0.05
    end

    # ── D: No NaN/Inf in periodic EFIE ──
    @testset "D: No NaN/Inf" begin
        lat = PeriodicLattice(dx_pe, dy_pe, 0.0, 0.0, k_pe; N_spatial=2, N_spectral=2)
        Z_per = assemble_Z_efie_periodic(mesh_pe, rwg_pe, k_pe, lat)
        @test !any(isnan, Z_per)
        @test !any(isinf, Z_per)
    end

    # ── D: Boundary-touching mesh requires Bloch-paired RWG ──
    @testset "D: Boundary-touching non-Bloch rejected" begin
        mesh_w = make_rect_plate(dx_pe, dy_pe, 2, 2)
        rwg_w = build_rwg(mesh_w; precheck=false)
        lat_w = PeriodicLattice(dx_pe, dy_pe, 0.0, 0.0, k_pe; N_spatial=2, N_spectral=2)
        @test_throws ArgumentError assemble_Z_efie_periodic(mesh_w, rwg_w, k_pe, lat_w)
    end

    # ── E: Non-coplanar meshes are rejected ──
    @testset "E: Non-coplanar mesh rejected" begin
        xyz_bad = copy(mesh_pe.xyz)
        xyz_bad[3, 1] += 1e-4
        mesh_bad = TriMesh(xyz_bad, mesh_pe.tri)
        rwg_bad = build_rwg(mesh_bad; precheck=false)
        lat = PeriodicLattice(dx_pe, dy_pe, 0.0, 0.0, k_pe; N_spatial=2, N_spectral=2)
        @test_throws ArgumentError assemble_Z_efie_periodic(mesh_bad, rwg_bad, k_pe, lat)
    end

    # ── A: Z_corr symmetry exploit (normal incidence) and oblique fallback ──
    @testset "A: Z_corr symmetry / streaming assembly" begin
        # Normal incidence with real RWG coefficients: ΔG is reciprocal and the
        # entry kernel is symmetric, so the assembled correction is symmetric and
        # the streaming/upper-triangle path is used.
        @test DiffMoM._periodic_correction_is_symmetric(rwg_pe, lat_pe)
        @test @allocated(DiffMoM._periodic_correction_is_symmetric(rwg_pe, lat_pe)) == 0
        Zc_sym = DiffMoM._assemble_periodic_correction(mesh_pe, rwg_pe, k_pe, lat_pe;
                                                       quad_order=3)
        sym_err = maximum(abs.(Zc_sym .- transpose(Zc_sym))) / maximum(abs.(Zc_sym))
        @test sym_err < 1e-12

        # Symmetry completion is P + transpose(P), including doubled diagonal
        # entries, and operates in place without a transposed matrix allocation.
        P0 = ComplexF64[0.5 + 1im  2 - 3im;
                        4 + 5im    3 - 2im]
        P = copy(P0)
        @test DiffMoM._complete_periodic_triangle_symmetry!(P) === P
        @test P == P0 + transpose(P0)
        copyto!(P, P0)
        @test @allocated(DiffMoM._complete_periodic_triangle_symmetry!(P)) == 0

        # Oblique incidence with complex Bloch coefficients: the Bloch phase
        # breaks reciprocity, the correction is not symmetric, and the detector
        # must fall back to the full sweep.
        dx_o = 1.2 * lambda_pe; dy_o = 1.2 * lambda_pe
        mesh_o = make_rect_plate(dx_o, dy_o, 4, 4)
        lat_o = PeriodicLattice(dx_o, dy_o, π/6, 0.0, k_pe; N_spatial=1, N_spectral=1)
        rwg_o = build_rwg_periodic(mesh_o, lat_o; precheck=false)
        @test !DiffMoM._periodic_correction_is_symmetric(rwg_o, lat_o)
        Zc_obl = DiffMoM._assemble_periodic_correction(mesh_o, rwg_o, k_pe, lat_o;
                                                       quad_order=3)
        @test maximum(abs.(Zc_obl .- transpose(Zc_obl))) / maximum(abs.(Zc_obl)) > 1e-3
        @test !any(isnan, Zc_obl) && !any(isinf, Zc_obl)

        # Safety guard: a complex-coefficient (oblique-built) RWG combined with a
        # zero-phase lattice yields a reciprocal ΔG but a NON-symmetric Z_corr;
        # the detector must reject the symmetry fast path here.
        lat_zero = PeriodicLattice(dx_o, dy_o, 0.0, 0.0, k_pe; N_spatial=1, N_spectral=1)
        @test !DiffMoM._periodic_correction_is_symmetric(rwg_o, lat_zero)
        Zc_mix = DiffMoM._assemble_periodic_correction(mesh_o, rwg_o, k_pe, lat_zero;
                                                       quad_order=3)
        @test maximum(abs.(Zc_mix .- transpose(Zc_mix))) / maximum(abs.(Zc_mix)) > 1e-3
    end
end
println("  PASS ✓")

# ─────────────────────────────────────────────────────────────────
# Test 42: PeriodicMetrics — Floquet mode enumeration
# ─────────────────────────────────────────────────────────────────
println("\n── Test 42: PeriodicMetrics ──")

@testset "PeriodicMetrics" begin
    lambda_pm = 0.03; k_pm = 2π / lambda_pm
    dx_pm = 0.5 * lambda_pm; dy_pm = 0.5 * lambda_pm
    lat_pm = PeriodicLattice(dx_pm, dy_pm, 0.0, 0.0, k_pm)

    # ── A: Mode count = (2N_orders+1)² ──
    @testset "A: Mode count" begin
        for N_ord in [1, 2, 3]
            modes = floquet_modes(k_pm, lat_pm; N_orders=N_ord)
            @test length(modes) == (2 * N_ord + 1)^2
        end
        @test_throws ArgumentError floquet_modes(k_pm, lat_pm; N_orders=-1)
        @test_throws ArgumentError floquet_modes(
            k_pm,
            lat_pm;
            N_orders=DiffMoM._MAX_PERIODIC_TRUNCATION + 1,
        )
        @test_throws ArgumentError floquet_modes(1.01k_pm, lat_pm; N_orders=1)
    end

    @testset "A: Extreme-range Floquet modes" begin
        large_k = 1.0e200
        large_k_lattice = PeriodicLattice(
            1.0, 1.0, 0.0, 0.0, large_k, 1.0, 0, 0)
        large_k_modes = floquet_modes(
            large_k, large_k_lattice; N_orders=1)
        @test all(mode -> all(isfinite, (mode.kx, mode.ky, mode.kz)),
                  large_k_modes)
        @test all(mode -> mode.propagating, large_k_modes)
        @test all(mode -> real(mode.kz) == large_k, large_k_modes)

        maximum_finite = floatmax(Float64)
        balanced_lattice = PeriodicLattice(
            1.0, 1.0,
            maximum_finite, maximum_finite,
            maximum_finite, 1.0, 0, 0)
        balanced_mode = only(floquet_modes(
            maximum_finite, balanced_lattice; N_orders=0))
        @test !balanced_mode.propagating
        @test balanced_mode.kz == ComplexF64(0.0, maximum_finite)

        grazing_kx = prevfloat(1.0)
        grazing_lattice = PeriodicLattice(
            1.0, 1.0, grazing_kx, 0.0, 1.0, 1.0, 0, 0)
        grazing_mode = only(floquet_modes(
            1.0, grazing_lattice; N_orders=0))
        grazing_reference = setprecision(BigFloat, 256) do
            Float64(sqrt(BigFloat(1.0)^2 - BigFloat(grazing_kx)^2))
        end
        @test grazing_mode.propagating
        @test real(grazing_mode.kz) == grazing_reference

        tiny_period_lattice = PeriodicLattice(
            nextfloat(0.0), 1.0, 0.0, 0.0, 1.0, 1.0, 0, 0)
        @test_throws OverflowError floquet_modes(
            1.0, tiny_period_lattice; N_orders=1)

        floquet_modes(k_pm, lat_pm; N_orders=3)
        @test @allocated(floquet_modes(
            k_pm, lat_pm; N_orders=3)) <= 20_000
    end

    @testset "A: Extreme-range Floquet current coefficients" begin
        current_mesh = make_rect_plate(0.1, 0.1, 4, 4)
        current_rwg = build_rwg(current_mesh)
        current_lattice = PeriodicLattice(
            1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0, 0)
        unit_currents = fill(ComplexF64(1.0), current_rwg.nedges)
        extreme_scale = floatmax(Float64)
        extreme_currents = fill(
            ComplexF64(extreme_scale), current_rwg.nedges)

        _, unit_coefficients = DiffMoM._floquet_current_fourier_coefficients(
            current_mesh, current_rwg, unit_currents,
            1.0, current_lattice; N_orders=0)
        _, extreme_coefficients =
            DiffMoM._floquet_current_fourier_coefficients(
                current_mesh, current_rwg, extreme_currents,
                1.0, current_lattice; N_orders=0)
        expected_extreme = SVector{3,ComplexF64}(ntuple(component ->
            setprecision(BigFloat, 512) do
                ComplexF64(
                    Complex{BigFloat}(unit_coefficients[1][component]) *
                    BigFloat(extreme_scale))
            end,
            3,
        ))
        @test all(isfinite, extreme_coefficients[1])
        @test extreme_coefficients[1] ≈ expected_extreme rtol=8eps(Float64)

        DiffMoM._floquet_current_fourier_coefficients(
            current_mesh, current_rwg, extreme_currents,
            1.0, current_lattice; N_orders=0)
        @test @allocated(
            DiffMoM._floquet_current_fourier_coefficients(
                current_mesh, current_rwg, extreme_currents,
                1.0, current_lattice; N_orders=0)) <= 450_000

        phase_mesh = make_rect_plate(1.0e93, 1.0e93, 2, 2)
        phase_mesh.xyz[1, :] .+= 3.0e108
        phase_mesh.xyz[2, :] .-= 3.0e108
        phase_rwg = build_rwg(phase_mesh)
        phase_k = 1.0e200
        phase_lattice = PeriodicLattice(
            1.0e94, 1.0e94,
            6.0e199, 6.0e199,
            phase_k, 1.0, 0, 0)
        phase_modes, phase_coefficients =
            DiffMoM._floquet_current_fourier_coefficients(
                phase_mesh,
                phase_rwg,
                ones(ComplexF64, phase_rwg.nedges),
                phase_k,
                phase_lattice;
                N_orders=0,
            )
        phase_points = tri_quad_points(
            phase_mesh, 1, tri_quad_rule(3)[1])
        raw_phase_argument = phase_modes[1].kx * phase_points[1][1] +
                             phase_modes[1].ky * phase_points[1][2]
        @test !isfinite(raw_phase_argument)
        @test all(isfinite, phase_coefficients[1])

        cancellation_mesh = make_rect_plate(0.2, 0.2, 2, 2)
        cancellation_rwg = build_rwg(cancellation_mesh)
        cancellation_lattice = PeriodicLattice(
            1.0, 1.0, 0.0, 0.0, 2π)
        cancellation_currents = ComplexF64[
            3.3333333333333335e15 - 0.574859618608291im,
            -3.333333333333334e15 + 0.4952528582638097im,
            7.09246526965761e15 - 0.4934342258793853im,
            -2.9075347303423905e15 - 0.08016977075392688im,
            -1.027968762180252e15 + 0.37627715840682663im,
            -2.9075347303423905e15 + 0.8710733417957057im,
            -2.9075347303423895e15 - 0.2921997205621716im,
            1.0279687621802516e15 - 0.4919982633290702im,
        ]
        cancellation_modes, cancellation_coefficients =
            DiffMoM._floquet_current_fourier_coefficients(
                cancellation_mesh,
                cancellation_rwg,
                cancellation_currents,
                2π,
                cancellation_lattice;
                N_orders=0,
            )
        # Ground truth is the correctly rounded result of a 4352-bit direct
        # quadrature over the stored Float64 mesh, basis, and current values.
        cancellation_reference = CVec3(
            -0.008683327175737436 - 0.0006650854830341138im,
            -6.667282426256387e-5 - 0.0012881931696331354im,
            0.0 + 0.0im,
        )
        @test only(cancellation_coefficients) == cancellation_reference

        cancellation_exact_terms = DiffMoM._periodic_fourier_term_count(
            ntriangles(cancellation_mesh), cancellation_rwg.nedges,
            length(tri_quad_rule(3)[2]),
            count(mode -> mode.propagating, cancellation_modes);
            exact=true,
        )
        @test_throws ArgumentError DiffMoM._floquet_current_fourier_coefficients(
            cancellation_mesh,
            cancellation_rwg,
            cancellation_currents,
            2π,
            cancellation_lattice;
            N_orders=0,
            max_exact_fourier_terms=cancellation_exact_terms - 1,
        )
    end

    # ── A: Only specular mode propagates for λ/2 cell at normal incidence ──
    @testset "A: Only specular mode for λ/2 cell" begin
        modes = floquet_modes(k_pm, lat_pm; N_orders=3)
        prop_modes = filter(m -> m.propagating, modes)
        # dx=dy=λ/2: next mode (1,0) has kt = 2π/dx = 2k > k → evanescent
        @test length(prop_modes) == 1
        m00 = prop_modes[1]
        @test m00.m == 0 && m00.n == 0
        # kz = k for (0,0) at normal incidence
        @test real(m00.kz) ≈ k_pm rtol=1e-12
        # θ_r = 0 (broadside)
        @test m00.theta_r ≈ 0.0 atol=1e-12
    end

    # ── A: Multiple propagating modes for 2λ cell ──
    @testset "A: Multiple modes for 2λ cell" begin
        dx_big = 2.0 * lambda_pm; dy_big = 2.0 * lambda_pm
        lat_big = PeriodicLattice(dx_big, dy_big, 0.0, 0.0, k_pm)
        modes = floquet_modes(k_pm, lat_big; N_orders=3)
        prop_modes = filter(m -> m.propagating, modes)
        # (m,n) propagating when (m²+n²) < (2dx/λ)² = 4
        # Pairs: (0,0), (±1,0), (0,±1), (±1,±1) → 9 modes
        # (2,0) has kt = k → grazing/evanescent (kz² = 0, strict > check)
        @test length(prop_modes) == 9
    end

    # ── B: All (m,n) pairs present ──
    @testset "B: Complete mode enumeration" begin
        modes = floquet_modes(k_pm, lat_pm; N_orders=1)
        @test length(modes) == 9
        mn_pairs = Set([(m.m, m.n) for m in modes])
        for m in -1:1, n in -1:1
            @test (m, n) ∈ mn_pairs
        end
    end

    # ── B: Evanescent mode properties ──
    @testset "B: Evanescent mode fields" begin
        modes = floquet_modes(k_pm, lat_pm; N_orders=3)
        evan_modes = filter(m -> !m.propagating, modes)
        @test length(evan_modes) > 0
        for mode in evan_modes
            # kz purely imaginary (positive imag from the code: im * sqrt(-kz2))
            @test abs(real(mode.kz)) < 1e-12
            @test imag(mode.kz) > 0
            # Angles undefined for evanescent modes
            @test isnan(mode.theta_r)
            @test isnan(mode.phi_r)
        end
    end

    # ── A: Specular angle for oblique incidence ──
    @testset "A: Specular angle for oblique incidence" begin
        # θ_inc = 30° → specular θ_r = 30° for (0,0) mode
        # With large enough cell to ensure (0,0) propagates at oblique
        dx_obl = 1.0 * lambda_pm; dy_obl = 1.0 * lambda_pm
        lat_obl = PeriodicLattice(dx_obl, dy_obl, π/6, 0.0, k_pm)
        modes = floquet_modes(k_pm, lat_obl; N_orders=3)
        # Find (0,0) mode
        m00 = nothing
        for mode in modes
            if mode.m == 0 && mode.n == 0
                m00 = mode
                break
            end
        end
        @test m00 !== nothing
        @test m00.propagating
        # kx = k sin(30°) = k/2, ky = 0
        # kz = sqrt(k² - (k/2)²) = k√3/2
        @test real(m00.kz) ≈ k_pm * sqrt(3) / 2 rtol=1e-12
        # θ_r = acos(kz/k) = acos(√3/2) = 30° = π/6
        @test m00.theta_r ≈ π / 6 rtol=1e-10
    end

    # ── B: Power balance residual bookkeeping ──
    @testset "B: Power balance residual bookkeeping" begin
        modes = floquet_modes(k_pm, lat_pm; N_orders=1)
        R = zeros(ComplexF64, length(modes))
        # Specular reflection of a physical passive shunt sheet: R lies in the disk
        # |R + 1/2| ≤ 1/2, i.e. Re(R) ≤ 0. Use a lossless value (|R|² = -Re R).
        idx00 = findfirst(m -> (m.m == 0 && m.n == 0), modes)
        @test idx00 !== nothing
        R[idx00] = -0.5 + 0.5im  # |R|² = 0.5 = -Re(R): lossless shunt sheet

        I_test = ComplexF64[1.0 + 0im, 2.0 + 0im]
        Z_pen = ComplexF64[2.0 0.0; 0.0 3.0]  # positive real penalty
        pb = power_balance(I_test, Z_pen, dx_pm * dy_pm, k_pm, modes, R)

        @test pb.P_inc > 0
        @test pb.P_refl ≥ 0
        @test pb.P_abs ≥ 0
        @test pb.P_resid ≈ pb.P_inc - pb.P_refl - pb.P_abs atol=1e-15 rtol=1e-14
        @test pb.resid_frac ≈ 1 - pb.refl_frac - pb.abs_frac atol=1e-15 rtol=1e-14

        # Closure-based transmission estimate
        pb_closure = power_balance(I_test, Z_pen, dx_pm * dy_pm, k_pm, modes, R;
                                   transmission=:closure)
        @test pb_closure.P_trans ≥ 0
        @test pb_closure.P_resid ≈ pb_closure.P_inc - pb_closure.P_refl - pb_closure.P_abs - pb_closure.P_trans atol=1e-15 rtol=1e-14
        @test pb_closure.trans_frac ≥ 0
        @test pb_closure.resid_frac ≈ 1 - pb_closure.refl_frac - pb_closure.abs_frac - pb_closure.trans_frac atol=1e-15 rtol=1e-14

        # Floquet transmission mode: exact thin-sheet relation T₀₀ = 1 + R₀₀.
        T = transmission_coefficients(modes, R)
        @test length(T) == length(R)
        @test T[idx00] ≈ 1 + R[idx00] atol=1e-14

        pb_floquet = power_balance(I_test, Z_pen, dx_pm * dy_pm, k_pm, modes, R;
                                   transmission=:floquet)
        @test pb_floquet.P_trans ≥ 0
        @test pb_floquet.P_resid ≈ pb_floquet.P_inc - pb_floquet.P_refl - pb_floquet.P_abs - pb_floquet.P_trans atol=1e-15 rtol=1e-14

        @test_throws DimensionMismatch power_balance(
            I_test, Z_pen, dx_pm * dy_pm, k_pm, modes, R[1:end-1])
        @test_throws DimensionMismatch power_balance(
            I_test, zeros(ComplexF64, 1, 2), dx_pm * dy_pm, k_pm, modes, R)
        @test_throws ArgumentError power_balance(
            ComplexF64[NaN + 0im, 0.0 + 0im], Z_pen,
            dx_pm * dy_pm, k_pm, modes, R)
        Z_nonfinite = copy(Z_pen)
        Z_nonfinite[1, 1] = Inf + 0im
        @test_throws ArgumentError power_balance(
            I_test, Z_nonfinite, dx_pm * dy_pm, k_pm, modes, R)
        R_nonfinite = copy(R)
        R_nonfinite[idx00] = NaN + 0im
        @test_throws ArgumentError power_balance(
            I_test, Z_pen, dx_pm * dy_pm, k_pm, modes, R_nonfinite)
        @test_throws ArgumentError power_balance(
            I_test, Z_pen, 0.0, k_pm, modes, R)
        @test_throws ArgumentError power_balance(
            I_test, Z_pen, dx_pm * dy_pm, 0.0, modes, R)
        @test_throws ArgumentError power_balance(
            I_test, Z_pen, dx_pm * dy_pm, k_pm, modes, R; E0=0.0)
        @test_throws ArgumentError power_balance(
            I_test, Z_pen, dx_pm * dy_pm, k_pm, modes, R; eta0=0.0)
        @test_throws ArgumentError power_balance(
            I_test, Z_pen, dx_pm * dy_pm, k_pm, modes, R;
            incident_order=(99, 99))
        @test_throws ArgumentError power_balance(
            I_test, Z_pen, dx_pm * dy_pm, k_pm, modes, R;
            transmission=:unsupported)
        T_nonfinite = copy(T)
        T_nonfinite[idx00] = Inf + 0im
        @test_throws ArgumentError power_balance(
            I_test, Z_pen, dx_pm * dy_pm, k_pm, modes, R;
            transmission=:floquet, T_coeffs=T_nonfinite)

        @test_throws ArgumentError transmission_coefficients(
            modes, R; incident_order=(99, 99))
        @test_throws ArgumentError transmission_coefficients(modes, R_nonfinite)
    end

    # ── B: Lossless thin-sheet energy conservation (T₀₀ = 1 + R₀₀) ──
    @testset "B: Lossless sheet energy conservation" begin
        modes = floquet_modes(k_pm, lat_pm; N_orders=1)
        idx00 = findfirst(m -> (m.m == 0 && m.n == 0), modes)
        Z_zero = zeros(ComplexF64, 2, 2)            # no penalty ⇒ no absorption
        I_dummy = ComplexF64[1.0 + 0im, 0.0 + 0im]
        # Sweep the passive lossless circle R = -1/2 + 1/2 e^{iφ} (|R|² = -Re R).
        for φ in range(0, 2π; length=9)
            R = zeros(ComplexF64, length(modes))
            R[idx00] = -0.5 + 0.5 * cis(φ)
            @test real(R[idx00]) ≤ 1e-12                       # passivity: Re(R) ≤ 0
            T = transmission_coefficients(modes, R)
            @test T[idx00] ≈ 1 + R[idx00] atol=1e-14
            # |R|² + |T|² = 1 for a lossless free-standing sheet at normal incidence.
            @test abs2(R[idx00]) + abs2(T[idx00]) ≈ 1.0 atol=1e-12
            pb = power_balance(I_dummy, Z_zero, dx_pm * dy_pm, k_pm, modes, R;
                               transmission=:floquet)
            @test pb.refl_frac + pb.trans_frac ≈ 1.0 atol=1e-12     # closed power budget
            @test pb.resid_frac ≈ 0.0 atol=1e-12
        end
    end

    # ── B: Oblique-incidence power budget carries cos(θ_inc) (regression) ──
    @testset "B: Oblique power balance normalizes by cos(theta_inc)" begin
        # A lossless pure reflector (|R_00| = 1, all power in the specular order)
        # must give refl_frac = 1 at any incidence angle. P_inc must therefore be
        # the z-directed flux ∝ cos(θ_inc); otherwise refl_frac = cos(θ_inc) < 1.
        dx_obl = 1.0 * lambda_pm; dy_obl = 1.0 * lambda_pm
        lat_obl = PeriodicLattice(dx_obl, dy_obl, π/6, 0.0, k_pm)   # θ_inc = 30°
        modes = floquet_modes(k_pm, lat_obl; N_orders=1)
        idx00 = findfirst(m -> (m.m == 0 && m.n == 0), modes)
        @test idx00 !== nothing && modes[idx00].propagating
        R = zeros(ComplexF64, length(modes))
        R[idx00] = -1.0 + 0im                                       # |R|² = 1, PEC-like
        Z_zero = zeros(ComplexF64, 2, 2)
        I_dummy = ComplexF64[1.0 + 0im, 0.0 + 0im]
        pb = power_balance(I_dummy, Z_zero, dx_obl * dy_obl, k_pm, modes, R)
        @test pb.refl_frac ≈ 1.0 atol=1e-12                         # was cos(30°)≈0.866 before fix
    end

    # ── B: Vector Floquet power includes both transverse polarizations ──
    @testset "B: Vector Floquet power counts cross polarization" begin
        modes = floquet_modes(k_pm, lat_pm; N_orders=0)
        idx00 = findfirst(m -> (m.m == 0 && m.n == 0), modes)
        @test idx00 !== nothing

        zero_vec = SVector{3,ComplexF64}(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
        R_vecs = fill(zero_vec, length(modes))
        R_vecs[idx00] = SVector{3,ComplexF64}(0.6 + 0im, 0.8 + 0im, 0.0 + 0im)
        p = reflected_power_fractions(modes, R_vecs, k_pm)

        # A scalar co-polar budget would see only 0.6^2. The vector budget must
        # count the orthogonal 0.8 component too.
        @test p[idx00] ≈ 1.0 atol=1e-14
        @test abs2(0.6) < p[idx00]
        @test_throws DimensionMismatch reflected_power_fractions(modes, R_vecs[1:end-1], k_pm)
        @test_throws ArgumentError reflected_power_fractions(modes, R_vecs, 0.0)
        R_vecs_nonfinite = copy(R_vecs)
        R_vecs_nonfinite[idx00] = SVector{3,ComplexF64}(NaN + 0im, 0.0 + 0im, 0.0 + 0im)
        @test_throws ArgumentError reflected_power_fractions(
            modes, R_vecs_nonfinite, k_pm)
    end

    @testset "B: Extreme-range periodic power metrics" begin
        incident_mode = FloquetMode(
            0, 0, 0.0, 0.0, 1.0 + 0.0im,
            true, 0.0, 0.0)
        grazing_mode = FloquetMode(
            1, 0, 1.0, 0.0, 1.0e-200 + 0.0im,
            true, π / 2, 0.0)
        extreme_modes = [incident_mode, grazing_mode]
        zero_mode_vector = SVector{3,ComplexF64}(0.0, 0.0, 0.0)
        extreme_mode_vectors = [
            zero_mode_vector,
            SVector{3,ComplexF64}(1.0e200, 0.0, 0.0),
        ]
        grazing_power_reference = setprecision(BigFloat, 6656) do
            Float64(BigFloat(1.0e200)^2 * BigFloat(1.0e-200))
        end
        @test reflected_power_fractions(
            extreme_modes, extreme_mode_vectors, 1.0) ==
              [0.0, grazing_power_reference]

        cancellation_vector =
            SVector{3,ComplexF64}(1.0e8, 1.0, 1.0)
        cancellation_vector_reference = setprecision(BigFloat, 256) do
            Float64(BigFloat(1.0e8)^2 + 2)
        end
        @test only(reflected_power_fractions(
            [incident_mode], [cancellation_vector], 1.0)) ==
              cancellation_vector_reference

        zero_penalty = zeros(ComplexF64, 1, 1)
        dummy_current = ComplexF64[1.0]
        zero_reflection = ComplexF64[0.0]
        overflow_normalized = power_balance(
            dummy_current,
            zero_penalty,
            1.0e-200,
            1.0,
            [incident_mode],
            zero_reflection;
            eta0=1.0e200,
            E0=1.0e200,
        )
        underflow_normalized = power_balance(
            dummy_current,
            zero_penalty,
            1.0e200,
            1.0,
            [incident_mode],
            zero_reflection;
            eta0=1.0e-200,
            E0=1.0e-200,
        )
        @test overflow_normalized.P_inc == 0.5
        @test underflow_normalized.P_inc == 0.5

        extreme_reflection = ComplexF64[0.0, 1.0e200]
        reflected_balance = power_balance(
            dummy_current,
            zero_penalty,
            1.0,
            1.0,
            extreme_modes,
            extreme_reflection,
        )
        reflected_power_reference = setprecision(BigFloat, 6656) do
            Float64(
                BigFloat(grazing_power_reference) /
                (2 * BigFloat(376.730313668)))
        end
        @test reflected_balance.P_refl == reflected_power_reference

        cancellation_mode_count = 1001
        cancellation_modes = [
            FloquetMode(
                index - 1, 0, 0.0, 0.0, 1.0 + 0.0im,
                true, 0.0, 0.0)
            for index in 1:cancellation_mode_count
        ]
        cancellation_reflection = ones(
            ComplexF64, cancellation_mode_count)
        cancellation_reflection[1] = 1.0e8
        cancellation_mode_reference = setprecision(BigFloat, 256) do
            Float64(
                BigFloat(1.0e8)^2 + cancellation_mode_count - 1)
        end
        cancellation_balance = power_balance(
            dummy_current,
            zero_penalty,
            1.0,
            1.0,
            cancellation_modes,
            cancellation_reflection;
            eta0=0.5,
        )
        @test cancellation_balance.P_refl ==
              cancellation_mode_reference

        transmitted_balance = power_balance(
            dummy_current,
            zero_penalty,
            1.0,
            1.0,
            extreme_modes,
            zeros(ComplexF64, 2);
            transmission=:floquet,
            T_coeffs=extreme_reflection,
        )
        @test transmitted_balance.P_trans == reflected_power_reference

        minimum_subnormal = nextfloat(0.0)
        underflow_penalty = ComplexF64[
            0.0 minimum_subnormal
            minimum_subnormal 0.0
        ]
        underflow_current = ComplexF64[
            ldexp(1.0, 664), ldexp(1.0, -664)
        ]
        absorbed_underflow = power_balance(
            underflow_current,
            underflow_penalty,
            1.0,
            1.0,
            [incident_mode],
            zero_reflection,
        )
        @test absorbed_underflow.P_abs == minimum_subnormal

        maximum_finite = floatmax(Float64)
        absorbed_maximum = power_balance(
            ComplexF64[1.0, 1.0],
            ComplexF64[maximum_finite 0.0; 0.0 maximum_finite],
            maximum_finite,
            1.0,
            [incident_mode],
            zero_reflection;
            eta0=0.5,
        )
        @test absorbed_maximum.P_inc == maximum_finite
        @test absorbed_maximum.P_abs == maximum_finite
        @test absorbed_maximum.P_resid == 0.0
        @test absorbed_maximum.abs_frac == 1.0

        residual_reflection = sqrt(3.0e-16)
        residual_cancellation = power_balance(
            ComplexF64[1.0],
            ComplexF64[2.0e16;;],
            2.0e16,
            1.0,
            [incident_mode],
            ComplexF64[residual_reflection];
            eta0=1.0,
        )
        residual_cancellation_reference = setprecision(BigFloat, 256) do
            Float64(
                BigFloat(residual_cancellation.P_inc) -
                BigFloat(residual_cancellation.P_refl) -
                BigFloat(residual_cancellation.P_abs) -
                BigFloat(residual_cancellation.P_trans))
        end
        @test residual_cancellation.P_resid ==
              residual_cancellation_reference

        reflected_power_fractions(
            extreme_modes, extreme_mode_vectors, 1.0)
        @test @allocated(reflected_power_fractions(
            extreme_modes,
            extreme_mode_vectors,
            1.0,
        )) <= 250_000
    end

    # ── B: Specular objective kwargs/defaults ──
    @testset "B: Specular objective kwargs/defaults" begin
        mesh_q = make_rect_plate(dx_pm, dy_pm, 2, 2)
        rwg_q = build_rwg(mesh_q; precheck=false)
        grid_q = make_sph_grid(6, 12)

        Q_default = specular_rcs_objective(mesh_q, rwg_q, grid_q, k_pm, lat_pm)
        Q_explicit = specular_rcs_objective(mesh_q, rwg_q, grid_q, k_pm, lat_pm;
                                            half_angle=π/18, polarization=:x)
        @test size(Q_default) == (rwg_q.nedges, rwg_q.nedges)
        @test Q_default ≈ Q_explicit rtol=1e-13 atol=1e-13

        Q_narrow = specular_rcs_objective(mesh_q, rwg_q, grid_q, k_pm, lat_pm;
                                          half_angle=5π/180, polarization=:x)
        @test size(Q_narrow) == size(Q_default)

        pol_x = pol_linear_x(grid_q)
        pol_y = pol_linear_y(grid_q)
        @test size(pol_y) == size(pol_x)
        for q in eachindex(grid_q.w)
            rhat_q = Vec3(grid_q.rhat[:, q])
            @test abs(dot(rhat_q, pol_x[:, q])) < 1e-12
            @test abs(dot(rhat_q, pol_y[:, q])) < 1e-12
            @test abs(dot(pol_x[:, q], pol_y[:, q])) < 1e-12
            @test norm(pol_y[:, q]) ≈ 1.0 atol=1e-12
        end

        Q_y = specular_rcs_objective(mesh_q, rwg_q, grid_q, k_pm, lat_pm;
                                     polarization=:y)
        Q_phi = specular_rcs_objective(mesh_q, rwg_q, grid_q, k_pm, lat_pm;
                                       polarization=:phi)
        Q_te = specular_rcs_objective(mesh_q, rwg_q, grid_q, k_pm, lat_pm;
                                      polarization=:te)
        Q_theta = specular_rcs_objective(mesh_q, rwg_q, grid_q, k_pm, lat_pm;
                                         polarization=:theta)
        Q_tm = specular_rcs_objective(mesh_q, rwg_q, grid_q, k_pm, lat_pm;
                                      polarization=:tm)
        Q_custom = specular_rcs_objective(mesh_q, rwg_q, grid_q, k_pm, lat_pm;
                                          polarization=pol_y)
        DiffMoM._specular_objective_polarization(grid_q, pol_y)
        custom_pol = DiffMoM._specular_objective_polarization(grid_q, pol_y)
        custom_pol_bytes = @allocated DiffMoM._specular_objective_polarization(
            grid_q, pol_y)
        @test custom_pol_bytes <= Base.summarysize(custom_pol) + 1024

        @test size(Q_y) == size(Q_default)
        @test Q_phi ≈ Q_y rtol=1e-13 atol=1e-13
        @test Q_te ≈ Q_y rtol=1e-13 atol=1e-13
        @test Q_theta ≈ Q_default rtol=1e-13 atol=1e-13
        @test Q_tm ≈ Q_default rtol=1e-13 atol=1e-13
        @test Q_custom ≈ Q_y rtol=1e-13 atol=1e-13

        @test_throws ArgumentError specular_rcs_objective(mesh_q, rwg_q, grid_q, k_pm, lat_pm;
                                                          polarization=:circular)
        @test_throws DimensionMismatch specular_rcs_objective(mesh_q, rwg_q, grid_q, k_pm, lat_pm;
                                                              polarization=zeros(ComplexF64, 2, length(grid_q.w)))
        nonfinite_pol = copy(pol_y)
        nonfinite_pol[1, 1] = ComplexF64(NaN, 0.0)
        @test_throws ArgumentError specular_rcs_objective(
            mesh_q, rwg_q, grid_q, k_pm, lat_pm;
            polarization=nonfinite_pol)
        @test_throws ArgumentError specular_rcs_objective(
            mesh_q, rwg_q, grid_q, 1.01k_pm, lat_pm
        )
    end

    # ── E: Non-coplanar meshes are rejected for reflection coefficients ──
    @testset "E: Non-coplanar reflection mesh rejected" begin
        mesh_q = make_rect_plate(dx_pm, dy_pm, 2, 2)
        xyz_bad = copy(mesh_q.xyz)
        xyz_bad[3, 1] += 1e-4
        mesh_bad = TriMesh(xyz_bad, mesh_q.tri)
        rwg_bad = build_rwg(mesh_bad; precheck=false)
        I_zero = zeros(ComplexF64, rwg_bad.nedges)
        @test_throws ArgumentError reflection_coefficients(mesh_bad, rwg_bad, I_zero, k_pm, lat_pm)
    end

    # ── D: Reflection extraction requires Bloch-paired RWG on touching boundaries ──
    @testset "D: Reflection boundary-touching non-Bloch rejected" begin
        mesh_w = make_rect_plate(dx_pm, dy_pm, 2, 2)
        rwg_w = build_rwg(mesh_w; precheck=false)
        I_zero = zeros(ComplexF64, rwg_w.nedges)
        @test_throws ArgumentError reflection_coefficients(mesh_w, rwg_w, I_zero, k_pm, lat_pm)
        @test_throws ArgumentError reflection_coefficients(
            mesh_w, rwg_w, I_zero, 1.01k_pm, lat_pm
        )
        @test_throws ArgumentError reflection_coefficient_vectors(
            mesh_w, rwg_w, I_zero, 1.01k_pm, lat_pm
        )
    end

    @testset "D: Reflection metric inputs validated" begin
        mesh_i = make_rect_plate(0.5dx_pm, 0.5dy_pm, 2, 2)
        rwg_i = build_rwg(mesh_i; precheck=false)
        I_zero = zeros(ComplexF64, rwg_i.nedges)

        _, periodic_weights = tri_quad_rule(3)
        periodic_mode_count = DiffMoM._periodic_term_count(0)
        periodic_work_bytes = DiffMoM._periodic_reflection_work_bytes(
            ntriangles(mesh_i), rwg_i.nedges, length(periodic_weights),
            periodic_mode_count)
        periodic_terms = DiffMoM._periodic_fourier_term_count(
            ntriangles(mesh_i), rwg_i.nedges, length(periodic_weights), 1)
        @test_throws ArgumentError reflection_coefficients(
            mesh_i, rwg_i, I_zero, k_pm, lat_pm;
            N_orders=0, max_work_bytes=periodic_work_bytes - 1)
        @test_throws ArgumentError reflection_coefficient_vectors(
            mesh_i, rwg_i, I_zero, k_pm, lat_pm;
            N_orders=0, max_fourier_terms=periodic_terms - 1)
        maximum_mode_count = DiffMoM._periodic_term_count(
            DiffMoM._MAX_PERIODIC_TRUNCATION)
        maximum_base_bytes = DiffMoM._periodic_reflection_base_bytes(
            maximum_mode_count)
        try
            reflection_coefficients(
                mesh_i, rwg_i, I_zero, k_pm, lat_pm;
                N_orders=DiffMoM._MAX_PERIODIC_TRUNCATION,
                max_work_bytes=maximum_base_bytes - 1,
            )
        catch
        end
        @test @allocated(
            try
                reflection_coefficients(
                    mesh_i, rwg_i, I_zero, k_pm, lat_pm;
                    N_orders=DiffMoM._MAX_PERIODIC_TRUNCATION,
                    max_work_bytes=maximum_base_bytes - 1,
                )
            catch
            end) <= 4_000
        limited_modes, limited_coefficients = reflection_coefficients(
            mesh_i, rwg_i, I_zero, k_pm, lat_pm;
            N_orders=0,
            max_work_bytes=periodic_work_bytes,
            max_fourier_terms=periodic_terms)
        @test length(limited_modes) == periodic_mode_count
        @test all(iszero, limited_coefficients)

        extreme_periodic_terms = DiffMoM._periodic_fourier_term_count(
            ntriangles(mesh_i), rwg_i.nedges, length(periodic_weights), 1;
            exact=true)
        I_extreme = fill(ComplexF64(floatmax(Float64)), rwg_i.nedges)
        @test_throws ArgumentError reflection_coefficients(
            mesh_i, rwg_i, I_extreme, k_pm, lat_pm;
            N_orders=0,
            max_exact_fourier_terms=extreme_periodic_terms - 1)

        # Finite phase terms can lose an observable angle through dot-product
        # cancellation long before either primitive is exponent-extreme.
        phase_angle = 0.2
        phase_mode = FloquetMode(
            0, 0, cos(phase_angle), sin(phase_angle),
            1.0 + 0im, true, 0.0, 0.0)
        phase_point = Vec3(
            1.0e12,
            -Float64((phase_mode.kx / phase_mode.ky) * 1.0e12),
            0.0,
        )
        phase_reference = setprecision(BigFloat, 4352) do
            argument = BigFloat(phase_mode.kx) * BigFloat(phase_point[1]) +
                       BigFloat(phase_mode.ky) * BigFloat(phase_point[2])
            ComplexF64(exp(Complex{BigFloat}(0, argument)))
        end
        @test DiffMoM._periodic_phase_requires_fallback(
            phase_mode, phase_point)
        @test DiffMoM._periodic_phase(phase_mode, phase_point) ==
              phase_reference
        @test DiffMoM._periodic_fourier_requires_fallback(
            ComplexF64[1.0], 1.0, [1.0], [phase_mode],
            [[phase_point]], [1.0])

        cancellation_direction = Vec3(
            0.7831050509661788,
            0.43233094754335505,
            0.447030682333465)
        cancellation_current = CVec3(
            -5.516272612069405e15,
            1.5099132442827375e15,
            8.203102977780578e15)
        cancellation_mode = FloquetMode(
            0, 0, 0.0, 0.0, 1.0 + 0.0im,
            true, 0.0, 0.0)
        ordinary_projection = dot(
            cancellation_direction, cancellation_current)
        exact_scalar_reflection =
            DiffMoM._periodic_scalar_reflection_exact(
                cancellation_direction, cancellation_current,
                1.0, 1.0, 1.0, 1.0, cancellation_mode)
        @test -0.5 * ordinary_projection != exact_scalar_reflection
        @test DiffMoM._periodic_scalar_reflection_checked(
                  cancellation_direction, cancellation_current,
                  1.0, 1.0, 1.0, 1.0, cancellation_mode) ==
              exact_scalar_reflection
        @test DiffMoM._periodic_vector_reflection_checked(
                  cancellation_direction, cancellation_current,
                  1.0, 1.0, 1.0, 1.0, cancellation_mode) ==
              DiffMoM._periodic_vector_reflection_exact(
                  cancellation_direction, cancellation_current,
                  1.0, 1.0, 1.0, 1.0, cancellation_mode)

        @test_throws DimensionMismatch reflection_coefficients(
            mesh_i, rwg_i, I_zero[1:end-1], k_pm, lat_pm)
        @test_throws DimensionMismatch reflection_coefficients(
            mesh_i, rwg_i, [I_zero; 0.0 + 0im], k_pm, lat_pm)
        I_nonfinite = copy(I_zero)
        I_nonfinite[1] = NaN + 0im
        @test_throws ArgumentError reflection_coefficients(
            mesh_i, rwg_i, I_nonfinite, k_pm, lat_pm)
        @test_throws ArgumentError reflection_coefficients(
            mesh_i, rwg_i, I_zero, k_pm, lat_pm; E0=0.0)
        @test_throws ArgumentError reflection_coefficients(
            mesh_i, rwg_i, I_zero, k_pm, lat_pm; eta0=0.0)
        @test_throws ArgumentError reflection_coefficients(
            mesh_i, rwg_i, I_zero, k_pm, lat_pm;
            pol=SVector(0.0, 0.0, 0.0))
        @test_throws ArgumentError reflection_coefficient_vectors(
            mesh_i, rwg_i, I_zero, k_pm, lat_pm; E0=0.0)
        @test_throws ArgumentError reflection_coefficient_vectors(
            mesh_i, rwg_i, I_zero, k_pm, lat_pm; eta0=Inf)

        _, zero_scalar_reflection = reflection_coefficients(
            mesh_i,
            rwg_i,
            I_zero,
            k_pm,
            lat_pm;
            N_orders=0,
            E0=nextfloat(0.0),
            eta0=floatmax(Float64),
        )
        _, zero_vector_reflection = reflection_coefficient_vectors(
            mesh_i,
            rwg_i,
            I_zero,
            k_pm,
            lat_pm;
            N_orders=0,
            E0=nextfloat(0.0),
            eta0=floatmax(Float64),
        )
        @test all(iszero, zero_scalar_reflection)
        @test all(iszero, zero_vector_reflection)
    end

    # ── B/F: Grounded metasurface via image theory ──
    @testset "B: Grounded metasurface (image theory)" begin
        freq_g = 10e9; c0_g = 2.99792458e8; lam_g = c0_g / freq_g
        kg = 2π / lam_g; eta0g = 376.730313668
        dcg = 0.5 * lam_g; Nxg = 6
        mesh_g = make_rect_plate(dcg, dcg, Nxg, Nxg)
        lat_g = PeriodicLattice(dcg, dcg, 0.0, 0.0, kg)
        rwg_g = build_rwg_periodic(mesh_g, lat_g; precheck=true, allow_boundary=true, require_closed=false)
        excitation_range_cell = 10.0
        excitation_range_k = 2π / excitation_range_cell
        excitation_range_mesh = make_rect_plate(
            excitation_range_cell, excitation_range_cell, 1, 1)
        excitation_range_lattice = PeriodicLattice(
            excitation_range_cell, excitation_range_cell,
            0.0, 0.0, excitation_range_k;
            N_spatial=1, N_spectral=1)
        excitation_range_rwg = build_rwg_periodic(
            excitation_range_mesh, excitation_range_lattice;
            precheck=false)
        unit_range_wave = PlaneWaveExcitation(
            Vec3(0.0, 0.0, -excitation_range_k),
            1.0,
            Vec3(1.0, 0.0, 0.0),
        )
        unit_range_rhs = assemble_excitation(
            excitation_range_mesh, excitation_range_rwg, unit_range_wave;
            quad_order=1)
        range_amplitude = 0.75 * floatmax(Float64) /
                          maximum(abs, unit_range_rhs)
        overflowing_range_wave = PlaneWaveExcitation(
            unit_range_wave.k_vec,
            range_amplitude,
            unit_range_wave.pol,
        )
        @test_throws OverflowError assemble_excitation_grounded(
            excitation_range_mesh,
            excitation_range_rwg,
            overflowing_range_wave,
            excitation_range_k,
            excitation_range_lattice;
            height=excitation_range_cell / 4,
            quad_order=1,
        )
        unit_range_current = ones(
            ComplexF64, excitation_range_rwg.nedges)
        _, unit_range_reflection = reflection_coefficients(
            excitation_range_mesh,
            excitation_range_rwg,
            unit_range_current,
            excitation_range_k,
            excitation_range_lattice;
            quad_order=1,
            N_orders=0,
        )
        reflection_scale = 0.75 * floatmax(Float64) /
                           maximum(abs, unit_range_reflection)
        overflowing_range_current = reflection_scale .* unit_range_current
        _, finite_range_reflection = reflection_coefficients(
            excitation_range_mesh,
            excitation_range_rwg,
            overflowing_range_current,
            excitation_range_k,
            excitation_range_lattice;
            quad_order=1,
            N_orders=0,
        )
        @test all(isfinite, finite_range_reflection)
        @test_throws OverflowError reflection_coefficients_grounded(
            excitation_range_mesh,
            excitation_range_rwg,
            overflowing_range_current,
            excitation_range_k,
            excitation_range_lattice;
            height=excitation_range_cell / 4,
            quad_order=1,
            N_orders=0,
        )
        @test_throws OverflowError reflection_coefficient_vectors_grounded(
            excitation_range_mesh,
            excitation_range_rwg,
            overflowing_range_current,
            excitation_range_k,
            excitation_range_lattice;
            height=excitation_range_cell / 4,
            quad_order=1,
            N_orders=0,
        )
        grounded_work_bytes = 3 * sizeof(ComplexF64) * rwg_g.nedges^2
        @test_throws ArgumentError assemble_Z_efie_grounded(
            mesh_g, rwg_g, kg, lat_g;
            height=lam_g / 8,
            max_work_bytes=grounded_work_bytes - 1,
        )
        _, grounded_weights = tri_quad_rule(3)
        grounded_incidence = DiffMoM._build_periodic_triangle_incidence(
            rwg_g, ntriangles(mesh_g))
        grounded_symmetric = DiffMoM._periodic_correction_is_symmetric(
            rwg_g, lat_g)
        grounded_direct_terms = DiffMoM._periodic_efie_green_terms(
            grounded_incidence.active_triangles, length(grounded_weights),
            lat_g, grounded_symmetric)
        grounded_image_terms = DiffMoM._periodic_efie_green_terms(
            grounded_incidence.active_triangles, length(grounded_weights),
            lat_g, grounded_symmetric; image_block=true)
        grounded_green_terms =
            grounded_direct_terms + grounded_image_terms
        @test_throws ArgumentError assemble_Z_efie_grounded(
            mesh_g, rwg_g, kg, lat_g;
            height=lam_g / 8,
            max_green_terms=grounded_green_terms - 1)
        Zg_green_limited = assemble_Z_efie_grounded(
            mesh_g, rwg_g, kg, lat_g;
            height=lam_g / 8,
            max_green_terms=grounded_green_terms)
        @test all(isfinite, Zg_green_limited)
        pw_g = make_plane_wave(Vec3(0.0, 0.0, -kg), 1.0, Vec3(1.0, 0.0, 0.0))
        R00g(I, h) = begin
            modes, R = reflection_coefficients_grounded(mesh_g, rwg_g, I, kg, lat_g;
                          height=h, N_orders=1, E0=1.0, pol=SVector(1.0, 0.0, 0.0))
            R[findfirst(m -> m.m == 0 && m.n == 0, modes)]
        end

        # (1) A full PEC sheet reflects fully (R00 = -1) at any height.
        for h in (lam_g / 8, lam_g / 4)
            Zg = h == lam_g / 8 ? Zg_green_limited :
                 assemble_Z_efie_grounded(
                     mesh_g, rwg_g, kg, lat_g; height=h)
            vg = Vector{ComplexF64}(assemble_excitation_grounded(mesh_g, rwg_g, pw_g, kg, lat_g; height=h))
            @test abs(R00g(Zg \ vg, h)) ≈ 1.0 atol = 2e-3
        end

        # (2) A uniform reactive sheet matches the exact transmission-line solution
        #     R00 = (Z_in - η0)/(Z_in + η0),  Z_in = Z_s || (j η0 tan kh).
        Mt_g = precompute_triangle_mass(mesh_g, rwg_g)
        Zper_g = assemble_Z_efie_periodic(mesh_g, rwg_g, kg, lat_g)
        cfg_g = DensityConfig(; p=1.0, Z_max_factor=5.0, reactive=true)
        Zpen_g = assemble_Z_penalty(Mt_g, fill(0.5, ntriangles(mesh_g)), cfg_g)
        If = (Zper_g + Zpen_g) \ Vector{ComplexF64}(assemble_excitation(mesh_g, rwg_g, pw_g))
        modes_f, Rf = reflection_coefficients(mesh_g, rwg_g, If, kg, lat_g; N_orders=1, E0=1.0, pol=SVector(1.0, 0.0, 0.0))
        Rfree = Rf[findfirst(m -> m.m == 0 && m.n == 0, modes_f)]
        Zs = -eta0g * (1 + Rfree) / (2 * Rfree)
        for h in (lam_g / 8, lam_g / 4, 3lam_g / 8)
            Zg = assemble_Z_efie_grounded(mesh_g, rwg_g, kg, lat_g; height=h)
            vg = Vector{ComplexF64}(assemble_excitation_grounded(mesh_g, rwg_g, pw_g, kg, lat_g; height=h))
            Ig = (Zg + Zpen_g) \ vg
            Zin = Zs * (im * eta0g * tan(kg * h)) / (Zs + im * eta0g * tan(kg * h))
            Rtl = (Zin - eta0g) / (Zin + eta0g)
            @test R00g(Ig, h) ≈ Rtl atol = 2e-3
            # Lossless ⇒ all power reflected (no transmission past the ground, no absorption).
            modes_g, Rg = reflection_coefficients_grounded(mesh_g, rwg_g, Ig, kg, lat_g;
                              height=h, N_orders=1, E0=1.0, pol=SVector(1.0, 0.0, 0.0))
            refl = sum(m.propagating ? abs2(Rg[i]) * real(m.kz) / kg : 0.0 for (i, m) in enumerate(modes_g))
            @test refl ≈ 1.0 atol = 3e-3
            modes_v, Rv = reflection_coefficient_vectors_grounded(mesh_g, rwg_g, Ig, kg, lat_g;
                              height=h, N_orders=1, E0=1.0, pol=SVector(1.0, 0.0, 0.0))
            @test modes_v == modes_g
            @test sum(reflected_power_fractions(modes_v, Rv, kg)) ≈ 1.0 atol = 3e-3
        end

        # (3) A finite positive height and matching lattice wavenumber are required
        # across every grounded public entry point. The excitation must also be
        # the down-going plane wave represented by the lattice's Bloch vector.
        I_zero = zeros(ComplexF64, rwg_g.nedges)
        for bad_h in (-1.0, 0.0, Inf, NaN)
            @test_throws ArgumentError assemble_Z_efie_grounded(
                mesh_g, rwg_g, kg, lat_g; height=bad_h
            )
            @test_throws ArgumentError assemble_excitation_grounded(
                mesh_g, rwg_g, pw_g, kg, lat_g; height=bad_h
            )
            @test_throws ArgumentError reflection_coefficients_grounded(
                mesh_g, rwg_g, I_zero, kg, lat_g; height=bad_h
            )
            @test_throws ArgumentError reflection_coefficient_vectors_grounded(
                mesh_g, rwg_g, I_zero, kg, lat_g; height=bad_h
            )
        end
        @test_throws ArgumentError assemble_excitation_grounded(
            mesh_g, rwg_g,
            PortExcitation([1], 1.0 + 0im, 50.0 + 0im),
            kg, lat_g; height=lam_g / 8)
        for inconsistent_wave in (
            make_plane_wave(
                Vec3(0.0, 0.0, -0.5kg), 1.0,
                Vec3(1.0, 0.0, 0.0)),
            make_plane_wave(
                Vec3(0.5kg, 0.0, -sqrt(0.75) * kg), 1.0,
                Vec3(0.0, 1.0, 0.0)),
            make_plane_wave(
                Vec3(0.0, 0.0, kg), 1.0,
                Vec3(1.0, 0.0, 0.0)),
        )
            @test_throws ArgumentError assemble_excitation_grounded(
                mesh_g, rwg_g, inconsistent_wave, kg, lat_g;
                height=lam_g / 8)
        end
        oblique_lattice = PeriodicLattice(
            dcg, dcg, pi / 6, 0.0, kg;
            N_spatial=1, N_spectral=1)
        oblique_rwg = build_rwg_periodic(
            mesh_g, oblique_lattice;
            precheck=false)
        oblique_kz = DiffMoM._kz_inc(kg, oblique_lattice)
        oblique_wave = make_plane_wave(
            Vec3(oblique_lattice.kx_bloch,
                 oblique_lattice.ky_bloch,
                 -oblique_kz),
            1.0,
            Vec3(0.0, 1.0, 0.0),
        )
        oblique_incident = assemble_excitation(
            mesh_g, oblique_rwg, oblique_wave; quad_order=1)
        oblique_grounded = assemble_excitation_grounded(
            mesh_g, oblique_rwg, oblique_wave, kg, oblique_lattice;
            height=lam_g / 8, quad_order=1)
        @test oblique_grounded ==
              (1 - DiffMoM._grounded_round_trip_phase(
                  oblique_kz, lam_g / 8)) .* oblique_incident
        for bad_pol in (
            SVector(0.0, 0.0, 0.0),
            SVector(Inf, 0.0, 0.0),
            SVector(NaN, 0.0, 0.0),
        )
            @test_throws ArgumentError reflection_coefficient_vectors_grounded(
                mesh_g, rwg_g, I_zero, kg, lat_g;
                height=lam_g / 8, pol=bad_pol, N_orders=0, quad_order=1
            )
        end
        modes_unit, R_unit = reflection_coefficient_vectors_grounded(
            mesh_g, rwg_g, I_zero, kg, lat_g;
            height=lam_g / 8, pol=SVector(1.0, 0.0, 0.0),
            N_orders=0, quad_order=1
        )
        unit_idx = only(eachindex(modes_unit))
        @test norm(R_unit[unit_idx]) ≈ 1.0 atol=1e-14
        for extreme_pol in (
            SVector(floatmax(Float64), 0.0, 0.0),
            SVector(nextfloat(0.0), 0.0, 0.0),
        )
            modes_extreme, R_extreme = reflection_coefficient_vectors_grounded(
                mesh_g, rwg_g, I_zero, kg, lat_g;
                height=lam_g / 8, pol=extreme_pol,
                N_orders=0, quad_order=1
            )
            @test modes_extreme == modes_unit
            @test R_extreme[unit_idx] ≈ R_unit[unit_idx] atol=1e-14
            @test all(isfinite, R_extreme[unit_idx])
        end

        # The round-trip phase remains finite when 2*kz*h overflows even
        # though its unit-magnitude exponential is exactly representable.
        grounded_extreme_height = floatmax(Float64)
        grounded_extreme_phase = setprecision(BigFloat, 2304) do
            ComplexF64(cis(-2BigFloat(kg) *
                              BigFloat(grounded_extreme_height)))
        end
        @test DiffMoM._grounded_round_trip_phase(
            kg, grounded_extreme_height) == grounded_extreme_phase
        grounded_incident = assemble_excitation(
            mesh_g, rwg_g, pw_g; quad_order=1)
        grounded_extreme_incident = assemble_excitation_grounded(
            mesh_g, rwg_g, pw_g, kg, lat_g;
            height=grounded_extreme_height, quad_order=1)
        @test grounded_extreme_incident ==
              (1 - grounded_extreme_phase) .* grounded_incident
        grounded_extreme_modes, grounded_extreme_reflection =
            reflection_coefficients_grounded(
                mesh_g, rwg_g, I_zero, kg, lat_g;
                height=grounded_extreme_height,
                N_orders=0, quad_order=1)
        @test only(grounded_extreme_modes).m == 0
        @test only(grounded_extreme_reflection) == -grounded_extreme_phase
        @test_throws ArgumentError assemble_Z_efie_grounded(
            mesh_g, rwg_g, 1.01kg, lat_g; height=lam_g / 8
        )
        @test_throws ArgumentError assemble_excitation_grounded(
            mesh_g, rwg_g, pw_g, 1.01kg, lat_g; height=lam_g / 8
        )
        @test_throws ArgumentError reflection_coefficients_grounded(
            mesh_g, rwg_g, I_zero, 1.01kg, lat_g; height=lam_g / 8
        )
        @test_throws ArgumentError reflection_coefficient_vectors_grounded(
            mesh_g, rwg_g, I_zero, 1.01kg, lat_g; height=lam_g / 8
        )

        # The streamed image-block assembler must retain the full non-reciprocal
        # sweep at oblique incidence and when complex Bloch basis coefficients are
        # paired with a zero-phase lattice.
        mesh_go = make_rect_plate(dcg, dcg, 2, 2)
        lat_go = PeriodicLattice(dcg, dcg, π/7, π/9, kg;
                                 N_spatial=1, N_spectral=1)
        rwg_go = build_rwg_periodic(mesh_go, lat_go; precheck=false)
        Zimg_go = DiffMoM._assemble_periodic_image_block(
            mesh_go, rwg_go, kg, lat_go, lam_g / 4; quad_order=1)
        @test all(isfinite, Zimg_go)
        @test maximum(abs.(Zimg_go .- transpose(Zimg_go))) /
              maximum(abs.(Zimg_go)) > 1e-3

        lat_gz = PeriodicLattice(dcg, dcg, 0.0, 0.0, kg;
                                 N_spatial=1, N_spectral=1)
        @test !DiffMoM._periodic_correction_is_symmetric(rwg_go, lat_gz)
        Zimg_gz = DiffMoM._assemble_periodic_image_block(
            mesh_go, rwg_go, kg, lat_gz, lam_g / 4; quad_order=1)
        @test maximum(abs.(Zimg_gz .- transpose(Zimg_gz))) /
              maximum(abs.(Zimg_gz)) > 1e-3
    end
end
println("  PASS ✓")
