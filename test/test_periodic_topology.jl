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
    end

    # ── A: kz branch cut correctness ──
    @testset "A: Spectral kz branch cut" begin
        # Propagating: kt = 0 → kz = k (real positive)
        kz_prop = DiffMoM._spectral_kz(k, 0.0, 0.0)
        @test real(kz_prop) ≈ k rtol=1e-14
        @test abs(imag(kz_prop)) < 1e-14

        # Evanescent: kt = 2k → kz² = k²-4k² = -3k²
        # kz = sqrt(-3k²) = ik√3 → negate to get Im(kz) ≤ 0 → kz = -ik√3
        kz_evan = DiffMoM._spectral_kz(k, 2k, 0.0)
        @test abs(real(kz_evan)) < 1e-14
        @test imag(kz_evan) ≈ -sqrt(3) * k rtol=1e-14
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
        # Exponential convergence: each step should improve dramatically
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
    eta0 = 376.730313668
    config = DensityConfig(; p=3.0, Z_max_factor=1000.0, vf_target=0.5)

    # ── A: DensityConfig ──
    @testset "A: DensityConfig values" begin
        @test abs(config.Z_max) ≈ 1000.0 * eta0 rtol=1e-12
        @test config.p ≈ 3.0
        @test config.vf_target ≈ 0.5
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

    # ── E: Dimension mismatch assertion ──
    @testset "E: Dimension mismatch" begin
        @test_throws AssertionError assemble_Z_penalty(Mt, zeros(Nt_di + 1), config)
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
    end

    # ── A: Heaviside boundary values (exact by construction) ──
    @testset "A: Heaviside H(0)=0, H(1)=1, H(η)=0.5" begin
        for beta in [1.0, 4.0, 16.0, 64.0]
            # H(0) = [tanh(βη) + tanh(-βη)] / denom = 0 (odd function cancels)
            @test heaviside_project([0.0], beta)[1] ≈ 0.0 atol=1e-14
            # H(1) = denom / denom = 1
            @test heaviside_project([1.0], beta)[1] ≈ 1.0 atol=1e-14
            # H(η=0.5) = tanh(β/2) / (2tanh(β/2)) = 0.5 for any β
            @test heaviside_project([0.5], beta)[1] ≈ 0.5 atol=1e-14
        end
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
    end

    # ── B/F: Grounded metasurface via image theory ──
    @testset "B: Grounded metasurface (image theory)" begin
        freq_g = 10e9; c0_g = 2.99792458e8; lam_g = c0_g / freq_g
        kg = 2π / lam_g; eta0g = 376.730313668
        dcg = 0.5 * lam_g; Nxg = 6
        mesh_g = make_rect_plate(dcg, dcg, Nxg, Nxg)
        lat_g = PeriodicLattice(dcg, dcg, 0.0, 0.0, kg)
        rwg_g = build_rwg_periodic(mesh_g, lat_g; precheck=true, allow_boundary=true, require_closed=false)
        pw_g = make_plane_wave(Vec3(0.0, 0.0, -kg), 1.0, Vec3(1.0, 0.0, 0.0))
        R00g(I, h) = begin
            modes, R = reflection_coefficients_grounded(mesh_g, rwg_g, I, kg, lat_g;
                          height=h, N_orders=1, E0=1.0, pol=SVector(1.0, 0.0, 0.0))
            R[findfirst(m -> m.m == 0 && m.n == 0, modes)]
        end

        # (1) A full PEC sheet reflects fully (R00 = -1) at any height.
        for h in (lam_g / 8, lam_g / 4)
            Zg = assemble_Z_efie_grounded(mesh_g, rwg_g, kg, lat_g; height=h)
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

        # (3) Positive ground-plane height is required.
        @test_throws ArgumentError assemble_Z_efie_grounded(mesh_g, rwg_g, kg, lat_g; height=-1.0)
    end
end
println("  PASS ✓")
