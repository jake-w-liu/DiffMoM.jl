# 15_periodic_to_demo.jl — Topology optimization of an RCS-reducing periodic metasurface
#
# Optimizes PEC/void density distribution on a periodic unit cell to minimize
# specular scattered power via adjoint-based gradient + L-BFGS with beta continuation.
#
# Run: julia --project=. examples/15_periodic_to_demo.jl

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

const PKG_DIR  = dirname(@__DIR__)
const DATA_DIR = joinpath(PKG_DIR, "data")
const FIG_DIR  = joinpath(PKG_DIR, "figures")
mkpath(DATA_DIR)
mkpath(FIG_DIR)

println("=" ^ 70)
println("  Periodic Metasurface Topology Optimization — RCS Reduction Demo")
println("=" ^ 70)

# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Problem Setup
# ═══════════════════════════════════════════════════════════════════════════

freq   = 10e9
c0     = 3e8
lambda = c0 / freq
k      = 2π / lambda
eta0   = 376.730313668
println("  Frequency: $(freq/1e9) GHz, λ = $(round(lambda*1e3, digits=2)) mm")

dx_cell = 0.5 * lambda
dy_cell = 0.5 * lambda
println("  Unit cell: $(round(dx_cell*1e3, digits=2)) × $(round(dy_cell*1e3, digits=2)) mm (λ/2)")

Nx, Ny = 10, 10
mesh = make_rect_plate(dx_cell, dy_cell, Nx, Ny)
lattice = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k)
rwg  = build_rwg_periodic(mesh, lattice; precheck=false)
Nt = ntriangles(mesh)
N  = rwg.nedges
println("  Mesh: $(Nx)×$(Ny) → $(Nt) triangles, $(N) RWG edges")

println("  Computing Z_per (one-time)...")
Z_per = Matrix{ComplexF64}(assemble_Z_efie_periodic(mesh, rwg, k, lattice))
println("  Z_per: $(N)×$(N) dense")

Mt = precompute_triangle_mass(mesh, rwg)

edge_len = dx_cell / Nx
r_min_default = 2.5 * edge_len
W_default, w_sum_default = build_filter_weights(mesh, r_min_default)

pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
v  = Vector{ComplexF64}(assemble_excitation(mesh, rwg, pw))

Ntheta_ff, Nphi_ff = 20, 40
spec_half_angle = 10 * π / 180
grid_ff = make_sph_grid(Ntheta_ff, Nphi_ff)
Q_spec  = Matrix{ComplexF64}(specular_rcs_objective(
    mesh, rwg, grid_ff, k, lattice; half_angle=spec_half_angle, polarization=:x))

config  = DensityConfig(; p=3.0, Z_max_factor=10.0, vf_target=0.5, reactive=true)
alpha_vf = 0.1
println("  SIMP: p=$(config.p), Z_max=$(round(config.Z_max, sigdigits=4)), α_vf=$(alpha_vf)")
println("  Objective: Q_spec on $(Ntheta_ff)×$(Nphi_ff) spherical grid, x-pol, " *
        "specular cone half-angle=$(round(spec_half_angle * 180 / π, digits=1))°")

centroids = [triangle_center(mesh, t) for t in 1:Nt]

# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Reference PEC Solution
# ═══════════════════════════════════════════════════════════════════════════

println("\n▸ Reference: Full PEC plate (ρ̄ = 1)")
_, rho_bar_pec = filter_and_project(W_default, w_sum_default, ones(Nt), 64.0)
Z_pec = Z_per + assemble_Z_penalty(Mt, rho_bar_pec, config)
F_pec = lu(Z_pec)
I_pec = F_pec \ v
J_pec = real(dot(I_pec, Q_spec * I_pec))
println("  J_specular(PEC) = $(round(J_pec, sigdigits=6))")

pol_inc = SVector(1.0, 0.0, 0.0)  # x-polarized
modes_pec, R_pec = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_pec), k, lattice;
                                            pol=pol_inc, E0=1.0)
prop_idx_pec = findall(m -> m.propagating, modes_pec)
println("  Propagating Floquet modes: $(length(prop_idx_pec))")
for i in prop_idx_pec
    m = modes_pec[i]
    println("    ($(m.m),$(m.n)): R = $(round(R_pec[i], sigdigits=4)), |R| = $(round(abs(R_pec[i]), sigdigits=4))")
end

# Power balance for PEC with explicit Floquet transmission extraction
Z_pen_pec = assemble_Z_penalty(Mt, rho_bar_pec, config)
pb_pec = power_balance(Vector{ComplexF64}(I_pec), Z_pen_pec, dx_cell * dy_cell, k, modes_pec, R_pec;
                       transmission=:floquet)
println("  Power balance (PEC): P_refl/P_inc=$(round(100*pb_pec.refl_frac, digits=1))%, " *
        "P_abs/P_inc=$(round(100*pb_pec.abs_frac, digits=1))%, " *
        "P_trans/P_inc=$(round(100*pb_pec.trans_frac, digits=1))%, " *
        "P_resid/P_inc=$(round(100*pb_pec.resid_frac, digits=2))%")

# ═══════════════════════════════════════════════════════════════════════════
# Section 3: L-BFGS Optimization with Beta Continuation
# ═══════════════════════════════════════════════════════════════════════════

"""
    run_optimization(Z_per, Mt, v, Q, config, W, w_sum, rho0; kwargs...)

Density topology optimization with L-BFGS and Heaviside beta continuation.
Returns (rho_opt, trace::DataFrame, snapshots::Dict{Float64,Vector{Float64}}).
"""
function run_optimization(Z_per::Matrix{ComplexF64}, Mt, v, Q, config, W, w_sum, rho0;
                          betas=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
                          iters_per_beta=30, alpha_vf=1.0, m_lbfgs=10, verbose=true)
    Nt_loc = length(rho0)
    vf_target = config.vf_target
    rho = copy(rho0)
    project!(x) = (x .= clamp.(x, 0.0, 1.0); x)

    trace = DataFrame(iter=Int[], beta=Float64[], J_scatter=Float64[],
                      J_vf=Float64[], J_total=Float64[], gnorm=Float64[],
                      vf=Float64[], frac_binary=Float64[])
    snapshots = Dict{Float64, Vector{Float64}}()
    global_iter = 0

    for beta in betas
        verbose && println("\n  ── β = $(Int(beta)) ──")
        _, rb_snap = filter_and_project(W, w_sum, rho, beta)
        snapshots[beta] = copy(rb_snap)

        # Reset L-BFGS memory at beta transition
        s_list = Vector{Float64}[]
        y_list = Vector{Float64}[]
        rho_old = copy(rho)
        g_old   = zeros(Nt_loc)

        for it in 1:iters_per_beta
            global_iter += 1

            rho_tilde, rho_bar = filter_and_project(W, w_sum, rho, beta)
            Z_total = Z_per + assemble_Z_penalty(Mt, rho_bar, config)
            F = lu(Z_total)
            I_c = F \ v

            QI_c = Q * I_c
            J_scat  = real(dot(I_c, QI_c))
            vf_cur  = mean(rho_bar)
            J_vf    = alpha_vf * (vf_cur - vf_target)^2
            J_tot   = J_scat + J_vf

            # Forward and adjoint solves share the same Z_total; reuse one LU factorization.
            lam = F' \ QI_c

            g_rb = gradient_density(Mt, Vector{ComplexF64}(I_c),
                                    Vector{ComplexF64}(lam), rho_bar, config)
            g_rb .+= alpha_vf * 2.0 * (vf_cur - vf_target) / Nt_loc
            g = gradient_chain_rule(g_rb, rho_tilde, W, w_sum, beta)

            gnorm = norm(g)
            fb = count(x -> x < 0.05 || x > 0.95, rho_bar) / Nt_loc
            push!(trace, (global_iter, beta, J_scat, J_vf, J_tot, gnorm, vf_cur, fb))

            if verbose && (it <= 2 || it == iters_per_beta || it % 10 == 0)
                println("    it=$global_iter J=$(round(J_tot, sigdigits=5)) " *
                        "|g|=$(round(gnorm, sigdigits=3)) vf=$(round(vf_cur, digits=3)) " *
                        "bin=$(round(100fb, digits=0))%")
            end

            # L-BFGS curvature update
            if it > 1
                s_k = rho .- rho_old
                y_k = g .- g_old
                if dot(s_k, y_k) > 1e-30
                    push!(s_list, copy(s_k)); push!(y_list, copy(y_k))
                    length(s_list) > m_lbfgs && (popfirst!(s_list); popfirst!(y_list))
                end
            end

            # Two-loop recursion
            q_v = copy(g)
            mc = length(s_list)
            av = zeros(mc)
            for i in mc:-1:1
                ri = 1.0 / dot(y_list[i], s_list[i])
                av[i] = ri * dot(s_list[i], q_v)
                q_v .-= av[i] .* y_list[i]
            end
            gamma = mc > 0 ? dot(s_list[end], y_list[end]) / dot(y_list[end], y_list[end]) : 0.01
            r_v = gamma .* q_v
            for i in 1:mc
                ri = 1.0 / dot(y_list[i], s_list[i])
                bv = ri * dot(y_list[i], r_v)
                r_v .+= (av[i] - bv) .* s_list[i]
            end

            d = -r_v

            # Armijo backtracking line search
            alpha_ls = 1.0
            rho_old .= rho
            g_old .= g
            for ls in 1:20
                rho_trial = project!(rho_old .+ alpha_ls .* d)
                _, rb = filter_and_project(W, w_sum, rho_trial, beta)
                I_t = (Z_per + assemble_Z_penalty(Mt, rb, config)) \ v
                Jt = real(dot(I_t, Q * I_t)) + alpha_vf * (mean(rb) - vf_target)^2
                if Jt <= J_tot + 1e-4 * alpha_ls * dot(g, d)
                    rho .= rho_trial
                    break
                end
                alpha_ls *= 0.5
                ls == 20 && (rho .= project!(rho_old .+ alpha_ls .* d))
            end
        end
    end

    _, rb_final = filter_and_project(W, w_sum, rho, betas[end])
    snapshots[Inf] = copy(rb_final)
    return rho, trace, snapshots
end

# --- Run main optimization ---
println("\n▸ Running Topology Optimization")
println("  β schedule: [1,2,4,8,16,32,64] × 30 iters = 210 total")

rho0 = fill(0.5, Nt)

t0 = time()
rho_opt, trace, snapshots = run_optimization(
    Z_per, Mt, v, Q_spec, config, W_default, w_sum_default, rho0;
    betas=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], iters_per_beta=30, alpha_vf=alpha_vf)
t_opt = time() - t0
println("\n  Optimization completed in $(round(t_opt, digits=1)) s")

_, rho_bar_final = filter_and_project(W_default, w_sum_default, rho_opt, 64.0)
Z_opt = Z_per + assemble_Z_penalty(Mt, rho_bar_final, config)
F_opt = lu(Z_opt)
I_opt = F_opt \ v
J_opt = real(dot(I_opt, Q_spec * I_opt))
reduction_dB = 10 * log10(max(J_opt, 1e-30) / max(J_pec, 1e-30))
println("  J: PEC=$(round(J_pec, sigdigits=5)) → opt=$(round(J_opt, sigdigits=5)) " *
        "($(round(reduction_dB, digits=2)) dB)")
println("  VF=$(round(mean(rho_bar_final), digits=3)), " *
        "binary=$(round(100*count(x -> x < 0.05 || x > 0.95, rho_bar_final)/Nt, digits=1))%")

# Save optimization data
CSV.write(joinpath(DATA_DIR, "results_optimization_trace.csv"), trace)
CSV.write(joinpath(DATA_DIR, "results_rho_final.csv"), DataFrame(
    triangle=1:Nt, rho_bar=rho_bar_final,
    x=[c[1] for c in centroids], y=[c[2] for c in centroids]))

snap_df = DataFrame(triangle=Int[], beta=Float64[], rho_bar=Float64[])
for (b, rb) in sort(collect(snapshots), by=first)
    b == Inf && continue
    for t in 1:Nt; push!(snap_df, (t, b, rb[t])); end
end
CSV.write(joinpath(DATA_DIR, "results_rho_snapshots.csv"), snap_df)
println("  ✓ Saved: data/results_optimization_trace.csv, results_rho_final.csv, results_rho_snapshots.csv")

# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Floquet Mode Analysis
# ═══════════════════════════════════════════════════════════════════════════

println("\n▸ Floquet Mode Analysis (properly normalized)")
modes_opt, R_opt = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_opt), k, lattice;
                                            pol=pol_inc, E0=1.0)
prop_idx_opt = findall(m -> m.propagating, modes_opt)
floquet_df = DataFrame(
    m=[modes_opt[i].m for i in prop_idx_opt], n=[modes_opt[i].n for i in prop_idx_opt],
    R_pec_real=[real(R_pec[i]) for i in prop_idx_opt],
    R_pec_abs=[abs(R_pec[i]) for i in prop_idx_opt],
    R_opt_real=[real(R_opt[i]) for i in prop_idx_opt],
    R_opt_abs=[abs(R_opt[i]) for i in prop_idx_opt])
CSV.write(joinpath(DATA_DIR, "results_floquet_comparison.csv"), floquet_df)
for i in prop_idx_opt
    mo = modes_opt[i]
    println("  ($(mo.m),$(mo.n)): PEC R=$(round(R_pec[i], sigdigits=4)) (|R|=$(round(abs(R_pec[i]), sigdigits=4))) → " *
            "opt R=$(round(R_opt[i], sigdigits=4)) (|R|=$(round(abs(R_opt[i]), sigdigits=4)))")
end

# ═══════════════════════════════════════════════════════════════════════════
# Section 4b: Periodic Specular Angle Sweep at 10 GHz (single-mode regime)
# ═══════════════════════════════════════════════════════════════════════════

println("\n▸ Periodic |R₀₀| vs Incidence Angle at 10 GHz (TE, φ = 0°)")
# For d = λ/2 at 10 GHz, the normal-incidence single-mode regime persists until
# near grazing incidence. A θ sweep in the x-z plane therefore remains a
# periodic-correct specular metric visualization for this unit cell.
Z_pen_opt = assemble_Z_penalty(Mt, rho_bar_final, config)
phi_inc_deg = 0.0
phi_inc = phi_inc_deg * π / 180
theta_sweep_deg = collect(0.0:5.0:75.0)
pol_te = SVector(0.0, 1.0, 0.0)  # TE for φ=0 incidence plane (E || y)
r00_angle_df = DataFrame(
    theta_inc_deg = Float64[],
    phi_inc_deg = Float64[],
    propagating_modes = Int[],
    R00_pec_abs = Float64[],
    R00_opt_abs = Float64[],
    R00_pec_db = Float64[],
    R00_opt_db = Float64[],
    opt_vs_pec_dB = Float64[],
)

for (j, theta_deg) in enumerate(theta_sweep_deg)
    theta = theta_deg * π / 180
    kz_inc = k * cos(theta)
    kx_inc = k * sin(theta) * cos(phi_inc)
    ky_inc = k * sin(theta) * sin(phi_inc)
    lattice_a = PeriodicLattice(dx_cell, dy_cell, theta, phi_inc, k)

    println("  [$j/$(length(theta_sweep_deg))] θ=$(round(theta_deg, digits=1))° ...")

    Z_per_a = Matrix{ComplexF64}(assemble_Z_efie_periodic(mesh, rwg, k, lattice_a))
    pw_a = make_plane_wave(Vec3(kx_inc, ky_inc, -kz_inc), 1.0, Vec3(pol_te...))
    v_a = Vector{ComplexF64}(assemble_excitation(mesh, rwg, pw_a))

    I_pec_a = lu(Z_per_a + Z_pen_pec) \ v_a
    I_opt_a = lu(Z_per_a + Z_pen_opt) \ v_a

    modes_pec_a, R_pec_a = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_pec_a), k, lattice_a;
                                                   pol=pol_te, E0=1.0)
    modes_opt_a, R_opt_a = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_opt_a), k, lattice_a;
                                                   pol=pol_te, E0=1.0)

    prop_count_pec = count(m -> m.propagating, modes_pec_a)
    prop_count_opt = count(m -> m.propagating, modes_opt_a)
    prop_count_pec == prop_count_opt || error("PEC/optimized propagating-mode count mismatch at θ=$(theta_deg)°")
    prop_count_pec == 1 || error("Expected single-mode regime in angle sweep, got $(prop_count_pec) modes at θ=$(theta_deg)°")

    idx00_pec = findfirst(m -> m.m == 0 && m.n == 0, modes_pec_a)
    idx00_opt = findfirst(m -> m.m == 0 && m.n == 0, modes_opt_a)
    (idx00_pec === nothing || idx00_opt === nothing) && error("Missing (0,0) Floquet mode at θ=$(theta_deg)°")

    r00_pec_abs = abs(R_pec_a[idx00_pec])
    r00_opt_abs = abs(R_opt_a[idx00_opt])
    r00_pec_db = 20 * log10(max(r00_pec_abs, 1e-30))
    r00_opt_db = 20 * log10(max(r00_opt_abs, 1e-30))
    opt_vs_pec_dB = 20 * log10(max(r00_opt_abs, 1e-30) / max(r00_pec_abs, 1e-30))

    push!(r00_angle_df, (
        theta_deg,
        phi_inc_deg,
        prop_count_pec,
        r00_pec_abs,
        r00_opt_abs,
        r00_pec_db,
        r00_opt_db,
        opt_vs_pec_dB,
    ))
end

CSV.write(joinpath(DATA_DIR, "results_r00_angle_sweep.csv"), r00_angle_df)
idx_theta0 = findfirst(t -> isapprox(t, 0.0; atol=1e-12), r00_angle_df.theta_inc_deg)
idx_theta_max = nrow(r00_angle_df)
println("  θ=0°:  |R00| PEC=$(round(r00_angle_df.R00_pec_abs[idx_theta0], sigdigits=4)), " *
        "opt=$(round(r00_angle_df.R00_opt_abs[idx_theta0], sigdigits=4)), " *
        "Δ=$(round(r00_angle_df.opt_vs_pec_dB[idx_theta0], digits=2)) dB")
println("  θ=$(round(r00_angle_df.theta_inc_deg[idx_theta_max], digits=1))°: " *
        "|R00| PEC=$(round(r00_angle_df.R00_pec_abs[idx_theta_max], sigdigits=4)), " *
        "opt=$(round(r00_angle_df.R00_opt_abs[idx_theta_max], sigdigits=4)), " *
        "Δ=$(round(r00_angle_df.opt_vs_pec_dB[idx_theta_max], digits=2)) dB")
println("  ✓ Saved: data/results_r00_angle_sweep.csv")

# ═══════════════════════════════════════════════════════════════════════════
# Section 4c: Power Balance Analysis
# ═══════════════════════════════════════════════════════════════════════════

println("\n▸ Power Balance Analysis")
pb_opt = power_balance(Vector{ComplexF64}(I_opt), Z_pen_opt, dx_cell * dy_cell, k, modes_opt, R_opt;
                       transmission=:floquet)

println("  PEC:       P_refl/P_inc=$(round(100*pb_pec.refl_frac, digits=1))%, " *
        "P_abs/P_inc=$(round(100*pb_pec.abs_frac, digits=1))%, " *
        "P_trans/P_inc=$(round(100*pb_pec.trans_frac, digits=1))%, " *
        "P_resid/P_inc=$(round(100*pb_pec.resid_frac, digits=2))%")
println("  Optimized: P_refl/P_inc=$(round(100*pb_opt.refl_frac, digits=1))%, " *
        "P_abs/P_inc=$(round(100*pb_opt.abs_frac, digits=1))%, " *
        "P_trans/P_inc=$(round(100*pb_opt.trans_frac, digits=1))%, " *
        "P_resid/P_inc=$(round(100*pb_opt.resid_frac, digits=2))%")
println("  |R₀₀|² reduction: $(round(10*log10(abs(R_opt[prop_idx_opt[1]])^2 / abs(R_pec[prop_idx_opt[1]])^2), digits=2)) dB")

pb_df = DataFrame(
    case=["PEC", "Optimized"],
    P_inc=[pb_pec.P_inc, pb_opt.P_inc],
    P_refl=[pb_pec.P_refl, pb_opt.P_refl],
    P_abs=[pb_pec.P_abs, pb_opt.P_abs],
    P_trans=[pb_pec.P_trans, pb_opt.P_trans],
    P_resid=[pb_pec.P_resid, pb_opt.P_resid],
    refl_frac=[pb_pec.refl_frac, pb_opt.refl_frac],
    abs_frac=[pb_pec.abs_frac, pb_opt.abs_frac],
    trans_frac=[pb_pec.trans_frac, pb_opt.trans_frac],
    resid_frac=[pb_pec.resid_frac, pb_opt.resid_frac],
    R00_abs=[abs(R_pec[prop_idx_opt[1]]), abs(R_opt[prop_idx_opt[1]])])
CSV.write(joinpath(DATA_DIR, "results_power_balance.csv"), pb_df)
println("  ✓ Saved: data/results_power_balance.csv")

# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Full Gradient Verification (all triangles)
# ═══════════════════════════════════════════════════════════════════════════

println("\n▸ Gradient Verification (all $Nt triangles)")
rho_gv = 0.3 .+ 0.4 * rand(Nt)
beta_gv = 4.0
rt_gv, rb_gv = filter_and_project(W_default, w_sum_default, rho_gv, beta_gv)
Z_gv = Z_per + assemble_Z_penalty(Mt, rb_gv, config)
F_gv = lu(Z_gv)
I_gv = F_gv \ v
lam_gv = F_gv' \ (Q_spec * I_gv)
g_adj = gradient_density_full(Mt, Vector{ComplexF64}(I_gv), Vector{ComplexF64}(lam_gv),
                              rt_gv, rb_gv, config, W_default, w_sum_default, beta_gv)

h_fd = 1e-5
g_fd = zeros(Nt)
for t in 1:Nt
    rp = copy(rho_gv); rp[t] += h_fd
    rm = copy(rho_gv); rm[t] -= h_fd
    _, rbp = filter_and_project(W_default, w_sum_default, rp, beta_gv)
    _, rbm = filter_and_project(W_default, w_sum_default, rm, beta_gv)
    Ip = (Z_per + assemble_Z_penalty(Mt, rbp, config)) \ v
    Im = (Z_per + assemble_Z_penalty(Mt, rbm, config)) \ v
    g_fd[t] = (real(dot(Ip, Q_spec * Ip)) - real(dot(Im, Q_spec * Im))) / (2h_fd)
end

rel_err_gv = abs.(g_adj .- g_fd) ./ max.(abs.(g_fd), 1e-20)
println("  Max rel error:  $(round(maximum(rel_err_gv), sigdigits=3))")
println("  Mean rel error: $(round(mean(rel_err_gv), sigdigits=3))")
CSV.write(joinpath(DATA_DIR, "results_gradient_full.csv"),
          DataFrame(triangle=1:Nt, g_adjoint=g_adj, g_fd=g_fd, rel_error=rel_err_gv))
println("  ✓ Saved: data/results_gradient_full.csv")

# ═══════════════════════════════════════════════════════════════════════════
# Section 6: Parametric Sweeps
# ═══════════════════════════════════════════════════════════════════════════

println("\n▸ Parametric Sweep: Filter Radius")
rmin_factors = [1.5, 2.0, 2.5, 3.0, 3.5]
results_rmin = DataFrame(r_min_factor=Float64[], J_final=Float64[],
                         reduction_dB=Float64[], vf=Float64[], binary_pct=Float64[])
for factor in rmin_factors
    rm_s = factor * edge_len
    W_s, ws_s = build_filter_weights(mesh, rm_s)
    _, tr_s, _ = run_optimization(Z_per, Mt, v, Q_spec, config, W_s, ws_s, fill(0.5, Nt);
        betas=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], iters_per_beta=30,
        alpha_vf=alpha_vf, verbose=false)
    Jf = tr_s.J_scatter[end]
    dB = 10 * log10(max(Jf, 1e-30) / max(J_pec, 1e-30))
    push!(results_rmin, (factor, Jf, dB, tr_s.vf[end], 100*tr_s.frac_binary[end]))
    println("  r_min=$(factor)×h: J=$(round(Jf, sigdigits=4)) ($(round(dB, digits=2)) dB)")
end
CSV.write(joinpath(DATA_DIR, "results_parametric_rmin.csv"), results_rmin)

println("\n▸ Parametric Sweep: Maximum Beta")
beta_schedules = [
    ([1.0, 2.0, 4.0],                              4.0),
    ([1.0, 2.0, 4.0, 8.0, 16.0],                  16.0),
    ([1.0, 2.0, 4.0, 8.0, 16.0, 32.0],            32.0),
    ([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],      64.0),
]
results_beta = DataFrame(beta_max=Float64[], J_final=Float64[],
                         reduction_dB=Float64[], binary_pct=Float64[])
for (bs, bmax) in beta_schedules
    _, tr_s, _ = run_optimization(Z_per, Mt, v, Q_spec, config,
        W_default, w_sum_default, fill(0.5, Nt);
        betas=bs, iters_per_beta=30, alpha_vf=alpha_vf, verbose=false)
    Jf = tr_s.J_scatter[end]
    dB = 10 * log10(max(Jf, 1e-30) / max(J_pec, 1e-30))
    push!(results_beta, (bmax, Jf, dB, 100*tr_s.frac_binary[end]))
    println("  β_max=$(Int(bmax)): J=$(round(Jf, sigdigits=4)) ($(round(dB, digits=2)) dB), " *
            "binary=$(round(tr_s.frac_binary[end]*100, digits=0))%")
end
CSV.write(joinpath(DATA_DIR, "results_parametric_beta.csv"), results_beta)
println("  ✓ Saved parametric data")

# ═══════════════════════════════════════════════════════════════════════════
# Section 7: Publication Figures
# ═══════════════════════════════════════════════════════════════════════════

println("\n▸ Generating Publication Figures")

# --- Fig 1a,b: Validation (from existing data) ---
ewald_df = CSV.read(joinpath(DATA_DIR, "validation_ewald_convergence.csv"), DataFrame)
efie_df  = CSV.read(joinpath(DATA_DIR, "validation_efie_large_period.csv"), DataFrame)

fig1a = plot_scatter(
    collect(ewald_df.N_trunc), max.(ewald_df.error_vs_ref, 1e-16);
    mode="lines+markers", legend="|ΔG(N) − ΔG(ref)|",
    marker_size=8,
    xlabel="Truncation order N", ylabel="Error vs reference",
    yscale="log", yrange=[-16, 0],
    title="(a) Ewald convergence",
    width=500, height=380, fontsize=14)
savefig(fig1a, joinpath(FIG_DIR, "fig_results_ewald_convergence.pdf"))

fig1b = plot_scatter(
    collect(Float64, efie_df.d_over_lambda), collect(Float64, efie_df.rel_diff_efie);
    mode="lines+markers", legend="‖Z_per − Z_free‖ / ‖Z_free‖",
    marker_size=8,
    xlabel="d / λ", ylabel="Relative difference",
    xscale="log", yscale="log",
    title="(b) Periodic EFIE → free-space",
    width=500, height=380, fontsize=14)
savefig(fig1b, joinpath(FIG_DIR, "fig_results_efie_convergence.pdf"))
println("  ✓ Fig 1: Validation plots")

# --- Fig 2: Gradient verification scatter ---
grad_df = CSV.read(joinpath(DATA_DIR, "results_gradient_full.csv"), DataFrame)
gall = vcat(grad_df.g_adjoint, grad_df.g_fd)
glo, ghi = minimum(gall), maximum(gall)
gpad = 0.1 * (ghi - glo)

fig2 = plot_scatter(
    [collect(grad_df.g_fd), [glo - gpad, ghi + gpad]],
    [collect(grad_df.g_adjoint), [glo - gpad, ghi + gpad]];
    mode=["markers", "lines"],
    legend=["Per-triangle gradient", "Identity"],
    marker_size=[6, 0],
    dash=["", "dash"], color=["#1f77b4", "gray"],
    xlabel="Finite-difference gradient", ylabel="Adjoint gradient",
    title="Gradient verification ($Nt triangles)",
    width=480, height=440, fontsize=14)
set_legend!(fig2; position=:topleft)
savefig(fig2, joinpath(FIG_DIR, "fig_results_gradient_verification.pdf"))
println("  ✓ Fig 2: Gradient scatter plot")

# --- Fig 3: Optimization convergence ---
betas_uniq = unique(trace.beta)
colors7 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
x_segs = [collect(trace[trace.beta .== b, :iter]) for b in betas_uniq]
y_segs = [collect(trace[trace.beta .== b, :J_scatter]) for b in betas_uniq]

fig3 = plot_scatter(x_segs, y_segs;
    mode=fill("lines", length(betas_uniq)),
    legend=["β=$(Int(b))" for b in betas_uniq],
    color=colors7[1:length(betas_uniq)],
    xlabel="Iteration", ylabel="Scattered power J",
    title="Optimization convergence",
    width=550, height=400, fontsize=14)
set_legend!(fig3; position=:topleft)
savefig(fig3, joinpath(FIG_DIR, "fig_results_convergence.pdf"))
println("  ✓ Fig 3: Convergence")

# --- Fig 4: Topology snapshots ---
function density_to_grid(rho_bar_vec, cents, nx, ny, dxc, dyc)
    grid_rho = zeros(ny, nx)
    grid_cnt = zeros(Int, ny, nx)
    # Use physical cell bounds, not centroid minima, to avoid empty-bin artifacts.
    x0, y0 = -0.5 * dxc, -0.5 * dyc
    dxg, dyg = dxc / nx, dyc / ny
    for t in eachindex(rho_bar_vec)
        ix = clamp(floor(Int, (cents[t][1] - x0) / dxg) + 1, 1, nx)
        iy = clamp(floor(Int, (cents[t][2] - y0) / dyg) + 1, 1, ny)
        grid_rho[iy, ix] += rho_bar_vec[t]
        grid_cnt[iy, ix] += 1
    end
    grid_rho ./= max.(grid_cnt, 1)
    return grid_rho
end

snap_betas = sort(filter(b -> b != Inf, collect(keys(snapshots))))
snap_labels = [("initial", snap_betas[1]),
               ("mid",     snap_betas[length(snap_betas) ÷ 2]),
               ("final",   Inf)]

# Individual topology plots (used by LaTeX subfloat)
for (label, bkey) in snap_labels
    rb = snapshots[bkey]
    beta_disp = bkey == Inf ? Int(snap_betas[end]) : Int(bkey)
    G = density_to_grid(rb, centroids, Nx, Ny, dx_cell, dy_cell)
    fig = plot_heatmap(collect(1:Nx), collect(1:Ny), G;
        xlabel="Cell x", ylabel="Cell y",
        zrange=[0, 1], colorscale="Greys",
        width=400, height=400, fontsize=18, equalar=true)
    savefig(fig, joinpath(FIG_DIR, "fig_results_topology_$(label).pdf"))
end

println("  ✓ Fig 4: Topology snapshots (individual, equalar=true)")

# --- Fig 5: Parametric studies (individual) ---
fig5a = plot_scatter(
    collect(results_rmin.r_min_factor), collect(results_rmin.reduction_dB);
    mode="lines+markers", legend="Specular reduction",
    marker_size=8,
    xlabel="r_min / h", ylabel="Specular reduction (dB)",
    title="(a) Filter radius sensitivity",
    width=500, height=380, fontsize=14)
set_legend!(fig5a; position=:topleft)
savefig(fig5a, joinpath(FIG_DIR, "fig_results_parametric_rmin.pdf"))

fig5b = plot_scatter(
    collect(results_beta.beta_max), collect(results_beta.reduction_dB);
    mode="lines+markers", legend="Specular reduction",
    marker_size=8,
    xlabel="β_max", ylabel="Specular reduction (dB)",
    xscale="log",
    title="(b) Continuation schedule sensitivity",
    width=500, height=380, fontsize=14)
set_legend!(fig5b; position=:topleft)
savefig(fig5b, joinpath(FIG_DIR, "fig_results_parametric_beta.pdf"))

println("  ✓ Fig 5: Parametric studies (individual)")

# --- Fig 6: Periodic specular reflection vs incidence angle (10 GHz, TE) ---
r00a_data = CSV.read(joinpath(DATA_DIR, "results_r00_angle_sweep.csv"), DataFrame)
fig6 = plot_scatter(
    [collect(r00a_data.theta_inc_deg), collect(r00a_data.theta_inc_deg)],
    [collect(r00a_data.R00_pec_abs), collect(r00a_data.R00_opt_abs)];
    mode=["lines", "lines"],
    legend=["PEC plate", "Optimized"],
    color=["#1f77b4", "#d62728"],
    dash=["solid", "dashdot"],
    xlabel="Incidence angle θ_inc (deg), φ_inc = 0° (TE)",
    ylabel="Specular Floquet amplitude |R₀₀|",
    title="Periodic specular reflection vs incidence angle (10 GHz)",
    xrange=[0.0, 75.0], yrange=[0.0, 1.02],
    width=550, height=400, fontsize=14)
set_legend!(fig6; position=:topright)
savefig(fig6, joinpath(FIG_DIR, "fig_results_r00_angle_sweep.pdf"))
for stale in (
    joinpath(FIG_DIR, "fig_results_rcs_comparison.pdf"),
    joinpath(FIG_DIR, "fig_results_r00_frequency_sweep.pdf"),
    joinpath(DATA_DIR, "results_rcs_comparison.csv"),
    joinpath(DATA_DIR, "results_r00_frequency_sweep.csv"),
)
    isfile(stale) && rm(stale; force=true)
end
println("  ✓ Fig 6: Periodic |R00| angle sweep (10 GHz, TE)")

# --- Supplementary Fig: Power budget comparison (PEC vs Optimized) ---
pb_data = CSV.read(joinpath(DATA_DIR, "results_power_balance.csv"), DataFrame)
# Bar chart: reflected, absorbed, and transmitted power as % of P_inc
refl_pct = pb_data.refl_frac .* 100
abs_pct = pb_data.abs_frac .* 100
trans_pct = pb_data.trans_frac .* 100
fig7 = plot_scatter(
    [Float64[1, 2], Float64[1, 2], Float64[1, 2]],
    [refl_pct, abs_pct, trans_pct];
    mode=["markers", "markers", "markers"],
    marker_size=[16, 16, 16],
    legend=["Reflected (%)", "Absorbed (%)", "Transmitted (%)"],
    color=["#1f77b4", "#d62728", "#2ca02c"],
    xlabel="", ylabel="Fraction of incident power (%)",
    title="Power budget: PEC vs Optimized",
    xrange=[0.5, 2.5],
    width=500, height=400, fontsize=14)
set_legend!(fig7; position=:bottomleft)
savefig(fig7, joinpath(FIG_DIR, "fig_supp_power_balance.pdf"))
legacy_pb = joinpath(FIG_DIR, "fig_results_power_balance.pdf")
isfile(legacy_pb) && rm(legacy_pb; force=true)
println("  ✓ Supplementary Fig: Power balance (table is primary in paper)")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 70)
println("  Results Summary")
println("=" ^ 70)
println("  Specular RCS reduction: $(round(reduction_dB, digits=2)) dB")
R00_pec = abs(R_pec[prop_idx_pec[1]])
R00_opt = abs(R_opt[prop_idx_opt[1]])
R00_dB = 20 * log10(R00_opt / R00_pec)
println("  |R₀₀|: PEC=$(round(R00_pec, sigdigits=4)) → opt=$(round(R00_opt, sigdigits=4)) ($(round(R00_dB, digits=1)) dB)")
println("  Reflected:   PEC=$(round(100*pb_pec.refl_frac, digits=1))%, opt=$(round(100*pb_opt.refl_frac, digits=1))%")
println("  Absorbed:    PEC=$(round(100*pb_pec.abs_frac, digits=1))%, opt=$(round(100*pb_opt.abs_frac, digits=1))%")
println("  Transmitted: PEC=$(round(100*pb_pec.trans_frac, digits=1))%, opt=$(round(100*pb_opt.trans_frac, digits=1))%")
println("  Residual*:   PEC=$(round(100*pb_pec.resid_frac, digits=2))%, opt=$(round(100*pb_opt.resid_frac, digits=2))%")
println("    *Residual = 1 - reflected - absorbed - transmitted (power-balance closure residual)")
println("  Final volume fraction:  $(round(mean(rho_bar_final), digits=3))")
println("  Binary fraction:        $(round(100*count(x -> x < 0.05 || x > 0.95, rho_bar_final)/Nt, digits=1))%")
println("  Gradient max rel error: $(round(maximum(rel_err_gv), sigdigits=3))")
println()
results_csvs = filter(f -> startswith(f, "results_") && endswith(f, ".csv"), readdir(DATA_DIR))
result_figs = filter(f -> startswith(f, "fig_results_") && endswith(f, ".pdf"), readdir(FIG_DIR))
println("  Data:    data/results_*.csv ($(length(results_csvs)) files)")
println("  Figures: figures/fig_results_*.pdf ($(length(result_figs)) figures)")
println("=" ^ 70)
