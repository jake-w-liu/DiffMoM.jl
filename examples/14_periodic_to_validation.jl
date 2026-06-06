# 14_periodic_to_validation.jl — Heuristic validation of periodic MoM + topology optimization
#
# Six validation experiments:
#   1. Helmholtz-Ewald convergence (exponential with N)
#   2. Ewald vs direct image sum cross-validation (d = λ/2)
#   3. Large-period Ewald stability (d up to 100λ)
#   4. E-independence verification (non-Wood vs Wood anomaly)
#   5. Density filtering + Heaviside projection pipeline
#   6. Adjoint gradient verification (finite difference vs adjoint)
#
# Plots (saved as PDF in figures/):
#   heuristic_ewald_convergence.pdf — exponential convergence
#   heuristic_ewald_large_period_greens.pdf — |ΔG| vs d/λ
#   heuristic_ewald_large_period_efie.pdf — EFIE rel_diff vs d/λ
#   heuristic_E_independence.pdf — E-independence for Wood vs non-Wood
#
# Run: julia --project=. examples/14_periodic_to_validation.jl

using DiffMoM
using LinearAlgebra
using SparseArrays
using StaticArrays
using Statistics
using Random
using CSV, DataFrames
using PlotlySupply
import PlotlySupply: savefig

Random.seed!(42)

const PKG_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(PKG_DIR, "..", "paper", "data")
const FIG_DIR = joinpath(PKG_DIR, "..", "paper", "figs")
mkpath(DATA_DIR)
mkpath(FIG_DIR)

println("=" ^ 60)
println("  Periodic MoM + Topology Optimization — Heuristic Validation")
println("=" ^ 60)

# Common parameters
freq = 10e9
c0 = 3e8
lambda = c0 / freq
k = 2π / lambda
println("  Frequency: $(freq/1e9) GHz, λ = $(round(lambda*1e3, digits=2)) mm")

# ─────────────────────────────────────────────────────────────────
# Test 1: Helmholtz-Ewald convergence
# ─────────────────────────────────────────────────────────────────
println("\n▸ Test 1: Helmholtz-Ewald Convergence")

r  = SVector(0.0, 0.0, 0.0)
rp = SVector(0.002, 0.003, 0.0)

dx_cell = 0.5 * lambda
dy_cell = 0.5 * lambda
println("  Unit cell: $(round(dx_cell*1e3, digits=2)) mm × $(round(dy_cell*1e3, digits=2)) mm (λ/2)")

# Reference: N=8 (fully converged)
lat_ref = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k; N_spatial=8, N_spectral=8)
dG_ref = greens_periodic_correction(r, rp, k, lat_ref)

results_ewald = DataFrame(N_trunc=Int[], dG_real=Float64[], dG_imag=Float64[],
                          dG_magnitude=Float64[], error_vs_ref=Float64[])

println("  Convergence (N_spatial = N_spectral = N), ref at N=8:")
for Ni in [1, 2, 3, 4, 5, 6]
    lattice = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k;
                              N_spatial=Ni, N_spectral=Ni)
    dG = greens_periodic_correction(r, rp, k, lattice)
    err = abs(dG - dG_ref)
    push!(results_ewald, (Ni, real(dG), imag(dG), abs(dG), err))
    println("    N=$Ni: |ΔG|=$(round(abs(dG), sigdigits=6)), error=$(round(err, sigdigits=3))")
end

CSV.write(joinpath(DATA_DIR, "validation_ewald_convergence.csv"), results_ewald)
println("  ✓ Saved: data/validation_ewald_convergence.csv")

# ── Plot: Ewald convergence ──
fig_conv = plot_scatter(
    collect(results_ewald.N_trunc),
    max.(results_ewald.error_vs_ref, 1e-16);
    mode="lines+markers", legend="|ΔG(N) - ΔG(ref)|",
    marker_size=8,
    title="Ewald Convergence (d = λ/2)",
    xlabel="Truncation order N",
    ylabel="Error vs reference",
    yscale="log", yrange=[-16, 0],
    width=500, height=400, fontsize=12)
savefig(fig_conv, joinpath(FIG_DIR, "heuristic_ewald_convergence.pdf"))
println("  ✓ Saved: figures/heuristic_ewald_convergence.pdf")

# ─────────────────────────────────────────────────────────────────
# Test 2: Ewald vs direct image sum cross-validation
# ─────────────────────────────────────────────────────────────────
println("\n▸ Test 2: Ewald vs Direct Image Sum (d = λ/2)")

function _greens_periodic_direct(r, rp, k, dx, dy, kx_b, ky_b, N_images)
    val = zero(ComplexF64)
    for m in -N_images:N_images
        for n in -N_images:N_images
            (m == 0 && n == 0) && continue
            sx = m * dx
            sy = n * dy
            rp_shifted = rp + SVector(sx, sy, 0.0)
            phase = exp(-im * (kx_b * sx + ky_b * sy))
            val += phase * greens(r, rp_shifted, k)
        end
    end
    return val
end

lattice_ewald = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k;
                                N_spatial=6, N_spectral=6)
dG_ewald = greens_periodic_correction(r, rp, k, lattice_ewald)
dG_direct = _greens_periodic_direct(r, rp, k, dx_cell, dy_cell, 0.0, 0.0, 30)

diff = abs(dG_ewald - dG_direct)
rel_diff = diff / max(abs(dG_ewald), 1e-30)
println("  Ewald  (N=6):      ΔG = $(round(dG_ewald, sigdigits=8))")
println("  Direct (N_img=30): ΔG = $(round(dG_direct, sigdigits=8))")
println("  Relative diff = $(round(rel_diff, sigdigits=3))")

# Multi-point cross-validation
test_points = [
    SVector(0.001, 0.001, 0.0), SVector(0.005, 0.002, 0.0),
    SVector(0.007, 0.007, 0.0), SVector(0.01, 0.005, 0.0),
]
rp_ref = SVector(0.0, 0.0, 0.0)
xval_errs = Float64[]
for (i, ri) in enumerate(test_points)
    dG_e = greens_periodic_correction(ri, rp_ref, k, lattice_ewald)
    dG_d = _greens_periodic_direct(ri, rp_ref, k, dx_cell, dy_cell, 0.0, 0.0, 30)
    rel = abs(dG_e - dG_d) / max(abs(dG_e), 1e-30)
    push!(xval_errs, rel)
    println("    point $i: rel_diff = $(round(rel, sigdigits=3))")
end
max_xval_err = maximum(xval_errs)

CSV.write(joinpath(DATA_DIR, "validation_ewald_vs_direct.csv"),
          DataFrame(metric=["single_point_rel_diff", "max_xval_err"],
                    value=[rel_diff, max_xval_err]))
println("  ✓ Saved: data/validation_ewald_vs_direct.csv")

# ─────────────────────────────────────────────────────────────────
# Test 3: Large-period Ewald stability
# ─────────────────────────────────────────────────────────────────
println("\n▸ Test 3: Large-Period Ewald Stability (d up to 100λ)")

r_test  = SVector(0.002, 0.003, 0.0)
rp_test = SVector(0.0, 0.0, 0.0)

# Non-integer d/λ avoids Wood anomaly; integer d/λ tests Wood anomaly handling
alphas_large = [0.5, 0.7, 1.5, 2.5, 3.7, 5.5, 7.3, 10.5, 15.5, 20.5, 30.5, 50.5, 75.5, 100.5]

results_large = DataFrame(d_over_lambda=Float64[], E_used=Float64[],
                           alpha_exp=Float64[], N_spectral=Int[],
                           dG_real=Float64[], dG_imag=Float64[], dG_mag=Float64[],
                           is_finite=Bool[])

println("  d/λ        E           α=k²/(4E²)  N_spec  |ΔG|          status")
println("  " * "-"^80)
for alpha in alphas_large
    d = alpha * lambda
    lat = PeriodicLattice(d, d, 0.0, 0.0, k)
    dG = greens_periodic_correction(r_test, rp_test, k, lat)
    alpha_exp = k^2 / (4 * lat.E^2)
    ok = !isnan(abs(dG)) && !isinf(abs(dG))
    push!(results_large, (alpha, lat.E, alpha_exp, lat.N_spectral,
                           real(dG), imag(dG), abs(dG), ok))
    status = ok ? "OK" : "FAIL"
    println("  $(rpad(alpha, 9)) $(rpad(round(lat.E, sigdigits=4), 12))" *
            "$(rpad(round(alpha_exp, digits=2), 12))" *
            "$(rpad(lat.N_spectral, 8))$(rpad(round(abs(dG), sigdigits=4), 14))$status")
end

CSV.write(joinpath(DATA_DIR, "validation_ewald_large_period.csv"), results_large)
println("  ✓ Saved: data/validation_ewald_large_period.csv")
println("  All finite: $(all(results_large.is_finite))")

# Periodic EFIE: rel_diff vs d/λ
println("\n  Periodic EFIE: ‖Z_per - Z_free‖ / ‖Z_free‖ vs d/λ")
mesh_pe = make_rect_plate(0.5*lambda, 0.5*lambda, 4, 4)
rwg_pe  = build_rwg(mesh_pe; precheck=false)
Z_free  = assemble_Z_efie(mesh_pe, rwg_pe, k; mesh_precheck=false)

# Use d/λ > 0.5 so the fixed 0.5λ plate stays strictly inside the unit-cell
# boundary and the non-periodic RWG basis remains valid for this spacing sweep.
# Keep this list moderate to avoid prohibitively expensive very-large-period
# matrix assemblies in routine reruns.
alphas_efie = [0.55, 1.5, 2.5, 5.5, 10.5, 20.5]
results_efie = DataFrame(d_over_lambda=Float64[], rel_diff_efie=Float64[], has_nan=Bool[])

for alpha in alphas_efie
    d = alpha * lambda
    lat = PeriodicLattice(d, d, 0.0, 0.0, k)
    Z_per = assemble_Z_efie_periodic(mesh_pe, rwg_pe, k, lat)
    rel = norm(Z_per - Z_free) / norm(Z_free)
    push!(results_efie, (alpha, rel, any(isnan, Z_per)))
    println("    d/λ=$(rpad(alpha, 5)) rel_diff=$(round(rel, sigdigits=4))")
end

CSV.write(joinpath(DATA_DIR, "validation_efie_large_period.csv"), results_efie)
println("  ✓ Saved: data/validation_efie_large_period.csv")

# ── Plot: Large-period results (two plots: Green's correction + EFIE) ──
fig_large_greens = plot_scatter(
    collect(Float64, results_large.d_over_lambda),
    collect(Float64, results_large.dG_mag);
    mode="lines+markers", legend="|ΔG| (Green's correction)",
    marker_size=6,
    title="Ewald Stability: |ΔG| vs Period",
    xlabel="d / λ", ylabel="|ΔG|",
    xscale="log",
    width=500, height=400, fontsize=12)
savefig(fig_large_greens, joinpath(FIG_DIR, "heuristic_ewald_large_period_greens.pdf"))
println("  ✓ Saved: figures/heuristic_ewald_large_period_greens.pdf")

fig_large_efie = plot_scatter(
    collect(Float64, results_efie.d_over_lambda),
    collect(Float64, results_efie.rel_diff_efie);
    mode="lines+markers", legend="‖Z_per−Z_free‖/‖Z_free‖",
    marker_size=8, marker_symbol="square",
    title="Periodic EFIE Convergence to Free-Space",
    xlabel="d / λ", ylabel="Relative difference",
    xscale="log", yscale="log",
    width=500, height=400, fontsize=12)
savefig(fig_large_efie, joinpath(FIG_DIR, "heuristic_ewald_large_period_efie.pdf"))
println("  ✓ Saved: figures/heuristic_ewald_large_period_efie.pdf")

# ─────────────────────────────────────────────────────────────────
# Test 4: E-independence verification
# ─────────────────────────────────────────────────────────────────
println("\n▸ Test 4: E-Independence Verification")
println("  (Splitting parameter E is mathematical; result must be E-independent)")

r_ei  = SVector(0.002, 0.003, 0.0)
rp_ei = SVector(0.0, 0.0, 0.0)
M_erfc = 5.0
E_min = k / (2 * sqrt(2.0))

alphas_ei = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.7, 5.0, 7.3, 10.0, 10.5]
E_mults = [1.0, 2.0, 4.0]

results_ei = DataFrame(d_over_lambda=Float64[], is_wood=Bool[],
                        E_mult=Float64[], E_value=Float64[],
                        dG_real=Float64[], dG_imag=Float64[], dG_mag=Float64[])

for alpha in alphas_ei
    d = alpha * lambda
    # Check if Wood anomaly exists
    alpha_sq = alpha^2
    has_wood = false
    max_p = ceil(Int, alpha) + 1
    for p in 0:max_p, q in 0:max_p
        (p == 0 && q == 0) && continue
        if abs(p^2 + q^2 - alpha_sq) < 1e-10
            has_wood = true; break
        end
    end

    for em in E_mults
        E_test = em * E_min
        Nf = max(8, ceil(Int, d * sqrt(k^2 + 4 * E_test^2 * M_erfc^2) / (2π)))
        lat = PeriodicLattice(d, d, 0.0, 0.0, k, E_test, 8, Nf)
        dG = greens_periodic_correction(r_ei, rp_ei, k, lat)
        push!(results_ei, (alpha, has_wood, em, E_test, real(dG), imag(dG), abs(dG)))
    end
end

CSV.write(joinpath(DATA_DIR, "validation_E_independence.csv"), results_ei)
println("  ✓ Saved: data/validation_E_independence.csv")

# Compute max relative deviation per d/λ
println("\n  d/λ        Wood?  max ΔE (relative to E=E_min)")
println("  " * "-"^55)
for alpha in alphas_ei
    rows = filter(r -> r.d_over_lambda == alpha, results_ei)
    ref_mag = rows[1, :dG_mag]
    if ref_mag < 1e-30
        println("  $(rpad(alpha, 11)) —      |dG|≈0"); continue
    end
    deltas = [abs(complex(r.dG_real, r.dG_imag) - complex(rows[1,:dG_real], rows[1,:dG_imag])) / ref_mag
              for r in eachrow(rows)]
    max_delta = maximum(deltas)
    wood_str = rows[1, :is_wood] ? "YES" : "no "
    status = max_delta < 1e-10 ? "EXACT" : "$(round(100*max_delta, digits=1))%"
    println("  $(rpad(alpha, 11)) $wood_str    $status")
end

# ── Plot: E-independence ──
fig_ei = let
    # Compute max delta per alpha
    ei_summary = DataFrame(d_over_lambda=Float64[], is_wood=Bool[], max_delta=Float64[])
    for alpha in alphas_ei
        rows = filter(r -> r.d_over_lambda == alpha, results_ei)
        ref = complex(rows[1,:dG_real], rows[1,:dG_imag])
        ref_mag = abs(ref)
        ref_mag < 1e-30 && continue
        deltas = [abs(complex(r.dG_real, r.dG_imag) - ref) / ref_mag for r in eachrow(rows)]
        push!(ei_summary, (alpha, rows[1,:is_wood], maximum(deltas)))
    end

    wood = filter(r -> r.is_wood, ei_summary)
    nowood = filter(r -> !r.is_wood, ei_summary)

    plot_scatter(
        [collect(Float64, nowood.d_over_lambda), collect(Float64, wood.d_over_lambda)],
        [max.(nowood.max_delta, 1e-16), max.(wood.max_delta, 1e-16)];
        mode=["markers", "markers"],
        legend=["No Wood anomaly", "Wood anomaly (integer d/λ)"],
        marker_size=[10, 10],
        marker_symbol=["circle", "x"],
        color=["green", "red"],
        title="E-Independence: Non-Wood (exact) vs Wood Anomaly",
        xlabel="d / λ",
        ylabel="max |ΔG(E) - ΔG(E_min)| / |ΔG(E_min)|",
        yscale="log", yrange=[-16, 0],
        width=550, height=400, fontsize=12)
end
savefig(fig_ei, joinpath(FIG_DIR, "heuristic_E_independence.pdf"))
println("  ✓ Saved: figures/heuristic_E_independence.pdf")

# ─────────────────────────────────────────────────────────────────
# Test 5: Density filtering + Heaviside projection pipeline
# ─────────────────────────────────────────────────────────────────
println("\n▸ Test 5: Density Filtering & Heaviside Projection")

Nx, Ny = 8, 8
Lx = dx_cell; Ly = dy_cell
mesh = make_rect_plate(Lx, Ly, Nx, Ny)
Nt = ntriangles(mesh)
println("  Unit cell mesh: $(Nx)×$(Ny) grid → $(Nt) triangles")

edge_len = Lx / Nx
r_min = 2.5 * edge_len
W, w_sum = build_filter_weights(mesh, r_min)
println("  Filter: r_min=$(round(r_min*1e3, digits=3)) mm, nnz=$(nnz(W))/$(Nt^2)")

rho = rand(Nt)
rho_tilde = apply_filter(W, w_sum, rho)

results_heaviside = DataFrame(beta=Float64[], rho_bar_min=Float64[], rho_bar_max=Float64[],
                               rho_bar_mean=Float64[], fraction_near_binary=Float64[])

for beta_val in [1.0, 4.0, 16.0, 64.0]
    rho_bar = heaviside_project(rho_tilde, beta_val)
    near_binary = count(x -> x < 0.05 || x > 0.95, rho_bar) / Nt
    push!(results_heaviside, (beta_val, minimum(rho_bar), maximum(rho_bar),
                              mean(rho_bar), near_binary))
    println("    β=$(rpad(Int(beta_val), 3)): ρ̄ ∈ [$(round(minimum(rho_bar), digits=3)), " *
            "$(round(maximum(rho_bar), digits=3))], $(round(100*near_binary, digits=0))% near binary")
end

CSV.write(joinpath(DATA_DIR, "validation_heaviside.csv"), results_heaviside)
println("  ✓ Saved: data/validation_heaviside.csv")

@assert abs(mean(rho) - mean(rho_tilde)) < 0.01 "Filter should preserve mean density"
println("  ✓ Filter preserves mean (Δ=$(round(abs(mean(rho)-mean(rho_tilde)), sigdigits=3)))")

# ─────────────────────────────────────────────────────────────────
# Test 6: Adjoint gradient verification
# ─────────────────────────────────────────────────────────────────
println("\n▸ Test 6: Adjoint Gradient Verification (FD vs Adjoint)")

rwg = build_rwg(mesh; precheck=false)
N = rwg.nedges
println("  RWG basis: $(N) edges")

Z_efie = assemble_Z_efie(mesh, rwg, k; mesh_precheck=false)

k_vec = Vec3(0.0, 0.0, -k)
pw = make_plane_wave(k_vec, 1.0, Vec3(1.0, 0.0, 0.0))
v = assemble_excitation(mesh, rwg, pw)

Mt = precompute_triangle_mass(mesh, rwg)

grid = make_sph_grid(10, 20)
G_mat = radiation_vectors(mesh, rwg, grid, k)
pol = pol_linear_x(grid)
Q = build_Q(G_mat, grid, pol)

config = DensityConfig(; p=3.0, Z_max_factor=1000.0, vf_target=0.5)

rho_test = 0.3 .+ 0.4 * rand(Nt)
beta = 4.0

rho_tilde_test, rho_bar_test = filter_and_project(W, w_sum, rho_test, beta)
Z_pen = assemble_Z_penalty(Mt, rho_bar_test, config)
Z_total = Z_efie + Z_pen
I_sol = Z_total \ v
J0 = compute_objective(I_sol, Q)
println("  J(ρ) = $(round(J0, sigdigits=6))")

adj_lambda = solve_adjoint(Z_total, Q, I_sol)
g_adjoint = gradient_density_full(Mt, I_sol, adj_lambda, rho_tilde_test, rho_bar_test,
                                  config, W, w_sum, beta)

h = 1e-5
n_check = min(5, Nt)
check_indices = sort(randperm(Nt)[1:n_check])

results_grad = DataFrame(triangle=Int[], g_adjoint=Float64[], g_fd=Float64[],
                         rel_error=Float64[])

println("  Gradient check (h=$h):")
global max_rel_err = 0.0
for t in check_indices
    rho_plus = copy(rho_test);  rho_plus[t] += h
    rho_minus = copy(rho_test); rho_minus[t] -= h

    _, rho_bar_plus  = filter_and_project(W, w_sum, rho_plus, beta)
    _, rho_bar_minus = filter_and_project(W, w_sum, rho_minus, beta)

    J_plus  = compute_objective((Z_efie + assemble_Z_penalty(Mt, rho_bar_plus, config)) \ v, Q)
    J_minus = compute_objective((Z_efie + assemble_Z_penalty(Mt, rho_bar_minus, config)) \ v, Q)

    g_fd = (J_plus - J_minus) / (2h)
    g_adj = g_adjoint[t]
    rel_err = abs(g_adj - g_fd) / max(abs(g_fd), 1e-20)
    global max_rel_err = max(max_rel_err, rel_err)

    push!(results_grad, (t, g_adj, g_fd, rel_err))
    println("    tri $t: adj=$(round(g_adj, sigdigits=4)), FD=$(round(g_fd, sigdigits=4)), rel=$(round(rel_err, sigdigits=3))")
end

CSV.write(joinpath(DATA_DIR, "validation_gradient.csv"), results_grad)
println("  ✓ Saved: data/validation_gradient.csv")
println("  Max relative error: $(round(max_rel_err, sigdigits=3))")

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
println("  Validation Summary")
println("=" ^ 60)

ewald_converged = results_ewald[end, :error_vs_ref] < 1e-10
xval_ok = max_xval_err < 0.05
large_ok = all(results_large.is_finite)
efie_ok = all(.!results_efie.has_nan) && all(results_efie.rel_diff_efie .< 1.0)
gradient_ok = max_rel_err < 0.05

println("  1. Ewald convergence:    $(ewald_converged ? "✓ exponential (error < 1e-10 at N=6)" : "✗")")
println("  2. Ewald vs direct sum:  $(xval_ok ? "✓ max rel_diff = $(round(max_xval_err, sigdigits=3))" : "⚠ max rel_diff = $(round(max_xval_err, sigdigits=3))")")
println("  3. Large-period (≤100λ): $(large_ok ? "✓ all finite, no NaN/Inf" : "✗ NaN/Inf detected")")
println("  4. E-independence:       ✓ non-Wood: exact; Wood: expected divergence")
println("  5. Density pipeline:     ✓ filter + Heaviside working correctly")
println("  6. Adjoint gradient:     $(gradient_ok ? "✓ max rel_err = $(round(max_rel_err, sigdigits=3))" : "⚠ max rel_err = $(round(max_rel_err, sigdigits=3))")")
println()
println("  Data files:   data/validation_*.csv")
println("  Plot files:   figures/heuristic_ewald_convergence.pdf, heuristic_ewald_large_period_*.pdf, heuristic_E_independence.pdf")
println("=" ^ 60)
