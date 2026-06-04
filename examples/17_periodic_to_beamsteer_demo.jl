# 17_periodic_to_beamsteer_demo.jl — Periodic density-TO beam-steering demo with baseline comparison
#
# Adds a second application case for the paper: maximize reflected power toward
# a prescribed off-specular Floquet direction on a multi-mode unit cell.
#
# Run: julia --project=. examples/17_periodic_to_beamsteer_demo.jl

using DiffMoM
using LinearAlgebra
using StaticArrays
using Statistics
using Random
using CSV, DataFrames
using PlotlySupply
import PlotlySupply: savefig

Random.seed!(84)

const PKG_DIR  = dirname(@__DIR__)
const DATA_DIR = joinpath(PKG_DIR, "data")
const FIG_DIR  = joinpath(PKG_DIR, "figures")
mkpath(DATA_DIR)
mkpath(FIG_DIR)

println("=" ^ 70)
println("  Periodic Beam-Steering Demo (Density TO)")
println("=" ^ 70)

freq = 10e9
c0 = 3e8
lambda = c0 / freq
k = 2π / lambda

dx_cell = 1.2 * lambda
dy_cell = 1.2 * lambda

Nx = parse(Int, get(ENV, "DMOM_BS_NX", "14"))
Ny = parse(Int, get(ENV, "DMOM_BS_NY", string(Nx)))
iters_per_beta = parse(Int, get(ENV, "DMOM_BS_ITERS_PER_BETA", "40"))
betas = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

println("  Frequency: $(freq/1e9) GHz")
println("  Unit cell: $(round(dx_cell/lambda, digits=2))λ × $(round(dy_cell/lambda, digits=2))λ")
println("  Mesh: $(Nx)×$(Ny), β schedule=$(Int.(betas)) × $(iters_per_beta)")

mesh = make_rect_plate(dx_cell, dy_cell, Nx, Ny)
lattice = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k)
rwg = build_rwg_periodic(mesh, lattice; precheck=false)
Nt = ntriangles(mesh)
N = rwg.nedges
println("  Discretization: Nt=$Nt triangles, N=$N RWG")

println("  Assembling periodic EFIE matrix...")
Z_per = Matrix{ComplexF64}(assemble_Z_efie_periodic(mesh, rwg, k, lattice))
Mt = precompute_triangle_mass(mesh, rwg)

edge_len = dx_cell / Nx
r_min = 2.5 * edge_len
W, w_sum = build_filter_weights(mesh, r_min)

pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
v = Vector{ComplexF64}(assemble_excitation(mesh, rwg, pw))

config = DensityConfig(; p=3.0, Z_max_factor=10.0, vf_target=0.5, reactive=true)
alpha_vf = 0.1
alpha_bin = 0.01

# Target direction: propagating Floquet order (m,n) = (1,0)
all_modes = floquet_modes(k, lattice; N_orders=2)
idx_target_mode = findfirst(m -> m.m == 1 && m.n == 0, all_modes)
idx_target_mode === nothing && error("(1,0) Floquet mode not found")
mode_target = all_modes[idx_target_mode]
mode_target.propagating || error("Target (1,0) mode is not propagating at selected geometry/frequency")

println("  Beam target mode: (1,0), θ≈$(round(mode_target.theta_r * 180 / π, digits=2))°, φ≈$(round(mode_target.phi_r * 180 / π, digits=2))°")

# Build target Q objective (cone around target direction)
Ntheta_ff, Nphi_ff = 24, 48
grid_ff = make_sph_grid(Ntheta_ff, Nphi_ff)
G_mat = radiation_vectors(mesh, rwg, grid_ff, k)
pol = pol_linear_x(grid_ff)

target_dir = Vec3(
    sin(mode_target.theta_r) * cos(mode_target.phi_r),
    sin(mode_target.theta_r) * sin(mode_target.phi_r),
    cos(mode_target.theta_r),
)
mask_target = direction_mask(grid_ff, target_dir; half_angle=8 * π / 180)
Q_target = Matrix{ComplexF64}(build_Q(G_mat, grid_ff, pol; mask=mask_target))

function mode_power_fraction(mode, R, k)
    mode.propagating || return 0.0
    return abs2(R) * real(mode.kz) / k
end

function evaluate_design(rho_raw::Vector{Float64}, beta_eval::Float64,
                         Z_per::Matrix{ComplexF64}, Mt, v, Q_target,
                         W, w_sum, config, mesh, rwg, k, lattice)
    rho_tilde, rho_bar = filter_and_project(W, w_sum, rho_raw, beta_eval)
    Z = Z_per + assemble_Z_penalty(Mt, rho_bar, config)
    I = Z \ v
    J_target = real(dot(I, Q_target * I))

    modes, R = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I), k, lattice;
                                       N_orders=2, pol=SVector(1.0, 0.0, 0.0), E0=1.0)

    idx10 = findfirst(m -> m.m == 1 && m.n == 0, modes)
    idx00 = findfirst(m -> m.m == 0 && m.n == 0, modes)
    (idx10 === nothing || idx00 === nothing) && error("Missing (1,0) or (0,0) mode")

    target_amp = abs(R[idx10])
    spec_amp = abs(R[idx00])
    target_pfrac = mode_power_fraction(modes[idx10], R[idx10], k)
    spec_pfrac = mode_power_fraction(modes[idx00], R[idx00], k)
    vf = mean(rho_bar)
    bin_pct = 100 * count(x -> x < 0.05 || x > 0.95, rho_bar) / length(rho_bar)

    return (rho_tilde=rho_tilde, rho_bar=rho_bar, I=I, modes=modes, R=R,
            J_target=J_target, target_amp=target_amp, spec_amp=spec_amp,
            target_pfrac=target_pfrac, spec_pfrac=spec_pfrac,
            vf=vf, binary_pct=bin_pct)
end

function run_optimization_beamsteer(Z_per::Matrix{ComplexF64}, Mt, v, Q_target,
                                    config, W, w_sum, rho0;
                                    betas=[1.0, 2.0, 4.0, 8.0, 16.0],
                                    iters_per_beta=8, alpha_vf=0.1,
                                    alpha_bin=0.0,
                                    m_lbfgs=10, verbose=true)
    Nt_loc = length(rho0)
    rho = copy(rho0)
    vf_target = config.vf_target
    project!(x) = (x .= clamp.(x, 0.0, 1.0); x)

    trace = DataFrame(iter=Int[], beta=Float64[], J_target=Float64[],
                      J_vf=Float64[], J_total=Float64[], gnorm=Float64[],
                      vf=Float64[], frac_binary=Float64[])
    global_iter = 0

    for beta in betas
        verbose && println("\n  ── β = $(Int(beta)) ──")
        s_list = Vector{Float64}[]
        y_list = Vector{Float64}[]
        rho_old = copy(rho)
        g_old = zeros(Nt_loc)

        for it in 1:iters_per_beta
            global_iter += 1

            rho_tilde, rho_bar = filter_and_project(W, w_sum, rho, beta)
            Z_total = Z_per + assemble_Z_penalty(Mt, rho_bar, config)
            F = lu(Z_total)
            I_c = F \ v

            QI = Q_target * I_c
            J_target = real(dot(I_c, QI))
            vf_cur = mean(rho_bar)
            J_vf = alpha_vf * (vf_cur - vf_target)^2
            J_bin = alpha_bin * sum(rho_bar[t] * (1 - rho_bar[t]) for t in 1:Nt_loc) / Nt_loc
            # Minimize negative target power to maximize steering efficiency.
            J_total = -J_target + J_vf + J_bin

            lam = F' \ (-QI)
            g_rb = gradient_density(Mt, Vector{ComplexF64}(I_c), Vector{ComplexF64}(lam), rho_bar, config)
            g_rb .+= alpha_vf * 2.0 * (vf_cur - vf_target) / Nt_loc
            # Binarization gradient: ∂/∂ρ̄_t [ρ̄_t(1-ρ̄_t)] = 1 - 2ρ̄_t
            g_rb .+= alpha_bin .* (1.0 .- 2.0 .* rho_bar) ./ Nt_loc
            g = gradient_chain_rule(g_rb, rho_tilde, W, w_sum, beta)

            gnorm = norm(g)
            frac_binary = count(x -> x < 0.05 || x > 0.95, rho_bar) / Nt_loc
            push!(trace, (global_iter, beta, J_target, J_vf, J_total, gnorm, vf_cur, frac_binary))

            if verbose && (it <= 2 || it == iters_per_beta || it % 4 == 0)
                println("    it=$global_iter J_target=$(round(J_target, sigdigits=5)) J_total=$(round(J_total, sigdigits=5)) |g|=$(round(gnorm, sigdigits=3)) vf=$(round(vf_cur, digits=3))")
            end

            if it > 1
                s_k = rho .- rho_old
                y_k = g .- g_old
                if dot(s_k, y_k) > 1e-30
                    push!(s_list, copy(s_k)); push!(y_list, copy(y_k))
                    length(s_list) > m_lbfgs && (popfirst!(s_list); popfirst!(y_list))
                end
            end

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
            alpha_ls = 1.0
            rho_old .= rho
            g_old .= g

            accepted = false
            for ls in 1:20
                rho_trial = project!(rho_old .+ alpha_ls .* d)
                _, rb = filter_and_project(W, w_sum, rho_trial, beta)
                I_t = (Z_per + assemble_Z_penalty(Mt, rb, config)) \ v
                Jt = -real(dot(I_t, Q_target * I_t)) + alpha_vf * (mean(rb) - vf_target)^2 +
                     alpha_bin * sum(rb[tt] * (1 - rb[tt]) for tt in 1:Nt_loc) / Nt_loc
                if Jt <= J_total + 1e-4 * alpha_ls * dot(g, d)
                    rho .= rho_trial
                    accepted = true
                    break
                end
                alpha_ls *= 0.5
            end
            !accepted && (rho .= project!(rho_old .+ alpha_ls .* d))
        end
    end

    return rho, trace
end

# Baseline 1: uniform density
rho_uniform = fill(0.5, Nt)
res_uniform = evaluate_design(rho_uniform, 64.0, Z_per, Mt, v, Q_target,
                              W, w_sum, config, mesh, rwg, k, lattice)

# Baseline 2: phase-ramp-inspired deterministic pattern
centroids = [triangle_center(mesh, t) for t in 1:Nt]
xvals = [c[1] for c in centroids]
xspan = max(maximum(xvals) - minimum(xvals), eps())
rho_phase = 0.5 .+ 0.45 .* sin.(2π .* ((xvals .- minimum(xvals)) ./ xspan))
rho_phase .= clamp.(rho_phase .- mean(rho_phase) .+ 0.5, 0.0, 1.0)
res_phase = evaluate_design(rho_phase, 64.0, Z_per, Mt, v, Q_target,
                            W, w_sum, config, mesh, rwg, k, lattice)

# Initialize from phase-ramp baseline (uniform ρ=0.5 has zero target-mode
# power with reactive Z_max, producing zero gradients).
println("\n▸ Running beam-steering optimization (init from phase-ramp)")
rho_opt, trace = run_optimization_beamsteer(
    Z_per, Mt, v, Q_target, config, W, w_sum, copy(rho_phase);
    betas=betas, iters_per_beta=iters_per_beta, alpha_vf=alpha_vf,
    alpha_bin=alpha_bin, verbose=true)
res_opt = evaluate_design(rho_opt, 64.0, Z_per, Mt, v, Q_target,
                          W, w_sum, config, mesh, rwg, k, lattice)

println("\n  Target-cone objective: uniform=$(round(res_uniform.J_target, sigdigits=4)), phase-ramp=$(round(res_phase.J_target, sigdigits=4)), optimized=$(round(res_opt.J_target, sigdigits=4))")
println("  |R10| amplitude: uniform=$(round(res_uniform.target_amp, sigdigits=4)), phase-ramp=$(round(res_phase.target_amp, sigdigits=4)), optimized=$(round(res_opt.target_amp, sigdigits=4))")
println("  Target mode power fraction: uniform=$(round(100*res_uniform.target_pfrac, digits=2))%, phase-ramp=$(round(100*res_phase.target_pfrac, digits=2))%, optimized=$(round(100*res_opt.target_pfrac, digits=2))%")
println("  Specular |R00|: uniform=$(round(res_uniform.spec_amp, sigdigits=4)), phase-ramp=$(round(res_phase.spec_amp, sigdigits=4)), optimized=$(round(res_opt.spec_amp, sigdigits=4))")

summary_df = DataFrame(
    case=["uniform", "phase_ramp_baseline", "optimized"],
    J_target=[res_uniform.J_target, res_phase.J_target, res_opt.J_target],
    target_amp=[res_uniform.target_amp, res_phase.target_amp, res_opt.target_amp],
    target_pfrac=[res_uniform.target_pfrac, res_phase.target_pfrac, res_opt.target_pfrac],
    spec_amp=[res_uniform.spec_amp, res_phase.spec_amp, res_opt.spec_amp],
    spec_pfrac=[res_uniform.spec_pfrac, res_phase.spec_pfrac, res_opt.spec_pfrac],
    vf=[res_uniform.vf, res_phase.vf, res_opt.vf],
    binary_pct=[res_uniform.binary_pct, res_phase.binary_pct, res_opt.binary_pct],
)
summary_df.target_vs_uniform_db = 20 .* log10.(max.(summary_df.target_amp, 1e-30) ./ max(summary_df.target_amp[1], 1e-30))
CSV.write(joinpath(DATA_DIR, "results_beamsteer_summary.csv"), summary_df)
CSV.write(joinpath(DATA_DIR, "results_beamsteer_optimization_trace.csv"), trace)

# Save propagating Floquet comparison (phase-ramp baseline vs optimized)
prop_idx = findall(m -> m.propagating, res_opt.modes)
prop_sorted = sort(prop_idx; by=i -> (res_opt.modes[i].m == 0 && res_opt.modes[i].n == 0 ? -1 : 0,
                                      abs(res_opt.modes[i].m) + abs(res_opt.modes[i].n),
                                      res_opt.modes[i].m, res_opt.modes[i].n))
mode_labels = ["($(res_opt.modes[i].m),$(res_opt.modes[i].n))" for i in prop_sorted]
mode_df = DataFrame(
    mode_index=1:length(prop_sorted),
    mode_label=mode_labels,
    m=[res_opt.modes[i].m for i in prop_sorted],
    n=[res_opt.modes[i].n for i in prop_sorted],
    R_phase_abs=[abs(res_phase.R[i]) for i in prop_sorted],
    R_opt_abs=[abs(res_opt.R[i]) for i in prop_sorted],
)
mode_df.amp_change_db = 20 .* log10.(max.(mode_df.R_opt_abs, 1e-30) ./ max.(mode_df.R_phase_abs, 1e-30))
CSV.write(joinpath(DATA_DIR, "results_beamsteer_floquet.csv"), mode_df)

# Figure: propagating Floquet amplitudes (baseline vs optimized)
fig_modes = plot_scatter(
    [collect(mode_df.mode_index), collect(mode_df.mode_index)],
    [collect(mode_df.R_phase_abs), collect(mode_df.R_opt_abs)];
    mode=["lines+markers", "lines+markers"],
    legend=["Phase-ramp baseline", "Optimized"],
    color=["#1f77b4", "#d62728"],
    dash=["dash", "solid"],
    marker_size=[8, 8],
    xlabel="Propagating mode index",
    ylabel="Floquet reflection amplitude |R_mn|",
    title="Beam-steering case: propagating Floquet amplitudes",
    width=620, height=420, fontsize=14)
set_legend!(fig_modes; position=:topright)
savefig(fig_modes, joinpath(FIG_DIR, "fig_results_beamsteer_modes.pdf"))

# Figure: optimization trace
fig_trace = plot_scatter(
    collect(trace.iter), collect(trace.J_target);
    mode="lines+markers",
    legend="Target-cone objective",
    color="#0072B2",
    dash="solid",
    marker_size=6,
    xlabel="Iteration",
    ylabel="J_target",
    title="Beam-steering optimization convergence",
    width=560, height=400, fontsize=14)
set_legend!(fig_trace; position=:topright)
savefig(fig_trace, joinpath(FIG_DIR, "fig_results_beamsteer_convergence.pdf"))

# Save final density field
rho_tilde_opt, rho_bar_opt = filter_and_project(W, w_sum, rho_opt, 64.0)
CSV.write(joinpath(DATA_DIR, "results_beamsteer_rho_final.csv"),
          DataFrame(triangle=1:Nt, rho_raw=rho_opt, rho_bar=rho_bar_opt))

# Topology figure: heatmap of optimized rho_bar on unit-cell mesh
cx = [triangle_center(mesh, t)[1] for t in 1:Nt]
cy = [triangle_center(mesh, t)[2] for t in 1:Nt]
# Map to grid for heatmap
x_unique = sort(unique(round.(cx, digits=10)))
y_unique = sort(unique(round.(cy, digits=10)))
Ngx, Ngy = length(x_unique), length(y_unique)
Z_opt = fill(NaN, Ngy, Ngx)
Z_phase = fill(NaN, Ngy, Ngx)
_, rho_bar_phase = filter_and_project(W, w_sum, rho_phase, 64.0)
for t in 1:Nt
    ix = searchsortedfirst(x_unique, round(cx[t], digits=10))
    iy = searchsortedfirst(y_unique, round(cy[t], digits=10))
    if ix <= Ngx && iy <= Ngy
        Z_opt[iy, ix] = rho_bar_opt[t]
        Z_phase[iy, ix] = rho_bar_phase[t]
    end
end

fig_topo_opt = plot_heatmap(
    collect(x_unique) .* 1e3, collect(y_unique) .* 1e3, Z_opt;
    xlabel="x [mm]", ylabel="y [mm]",
    title="Optimized topology",
    colorscale="Greys", width=504, height=420)
savefig(fig_topo_opt, joinpath(FIG_DIR, "fig_results_beamsteer_topology_opt.pdf"))

fig_topo_phase = plot_heatmap(
    collect(x_unique) .* 1e3, collect(y_unique) .* 1e3, Z_phase;
    xlabel="x [mm]", ylabel="y [mm]",
    title="Phase-ramp baseline",
    colorscale="Greys", width=504, height=420)
savefig(fig_topo_phase, joinpath(FIG_DIR, "fig_results_beamsteer_topology_phase.pdf"))

println("\n  ✓ Saved: data/results_beamsteer_summary.csv")
println("  ✓ Saved: data/results_beamsteer_floquet.csv")
println("  ✓ Saved: data/results_beamsteer_optimization_trace.csv")
println("  ✓ Saved: data/results_beamsteer_rho_final.csv")
println("  ✓ Saved: figures/fig_results_beamsteer_modes.pdf")
println("  ✓ Saved: figures/fig_results_beamsteer_convergence.pdf")
println("  ✓ Saved: figures/fig_results_beamsteer_topology_opt.pdf")
println("  ✓ Saved: figures/fig_results_beamsteer_topology_phase.pdf")

println("\n" * "=" ^ 70)
println("  Beam-Steering Demo Summary (Single Target)")
println("=" ^ 70)
println("  Target mode: (1,0), θ=$(round(mode_target.theta_r * 180 / π, digits=2))°")
println("  Baseline -> optimized target power: $(round(100*res_phase.target_pfrac, digits=2))% -> $(round(100*res_opt.target_pfrac, digits=2))%")
println("  Baseline -> optimized |R10| change: $(round(20*log10(max(res_opt.target_amp,1e-30)/max(res_phase.target_amp,1e-30)), digits=2)) dB")
println("=" ^ 70)

# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Weighted Dual-Beam Steering [(1,0) + (0,1)]
# ═══════════════════════════════════════════════════════════════════════════
# A phase ramp along x targets (1,0) but not (0,1); along y the reverse.
# Neither 1D ramp can simultaneously steer to both diagonal modes.
# TO should discover a 2D pattern that splits power between both.

println("\n\n" * "=" ^ 70)
println("  Weighted Dual-Beam Steering Demo: (1,0) + (0,1)")
println("=" ^ 70)

idx_target_01 = findfirst(m -> m.m == 0 && m.n == 1, all_modes)
idx_target_01 === nothing && error("(0,1) Floquet mode not found")
mode_target_01 = all_modes[idx_target_01]
mode_target_01.propagating || error("Target (0,1) mode is not propagating")

println("  Second target: (0,1), θ≈$(round(mode_target_01.theta_r * 180 / π, digits=2))°, φ≈$(round(mode_target_01.phi_r * 180 / π, digits=2))°")

# Build Q for mode (0,1)
target_dir_01 = Vec3(
    sin(mode_target_01.theta_r) * cos(mode_target_01.phi_r),
    sin(mode_target_01.theta_r) * sin(mode_target_01.phi_r),
    cos(mode_target_01.theta_r),
)
mask_target_01 = direction_mask(grid_ff, target_dir_01; half_angle=8 * π / 180)
Q_target_01 = Matrix{ComplexF64}(build_Q(G_mat, grid_ff, pol; mask=mask_target_01))

# Combined objective: equal weight on both modes
Q_dual = Q_target + Q_target_01

function evaluate_dual(rho_raw::Vector{Float64}, beta_eval::Float64)
    rho_tilde, rho_bar = filter_and_project(W, w_sum, rho_raw, beta_eval)
    Z = Z_per + assemble_Z_penalty(Mt, rho_bar, config)
    I = Z \ v
    J_10 = real(dot(I, Q_target * I))
    J_01 = real(dot(I, Q_target_01 * I))
    J_dual = real(dot(I, Q_dual * I))

    modes_e, R_e = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I), k, lattice;
                                           N_orders=2, pol=SVector(1.0, 0.0, 0.0), E0=1.0)
    idx10 = findfirst(m -> m.m == 1 && m.n == 0, modes_e)
    idx01 = findfirst(m -> m.m == 0 && m.n == 1, modes_e)
    idx00 = findfirst(m -> m.m == 0 && m.n == 0, modes_e)
    (idx10 === nothing || idx01 === nothing || idx00 === nothing) && error("Missing mode")

    pfrac_10 = mode_power_fraction(modes_e[idx10], R_e[idx10], k)
    pfrac_01 = mode_power_fraction(modes_e[idx01], R_e[idx01], k)
    combined_pfrac = pfrac_10 + pfrac_01
    vf = mean(rho_bar)
    bin_pct = 100 * count(x -> x < 0.05 || x > 0.95, rho_bar) / length(rho_bar)

    return (J_10=J_10, J_01=J_01, J_dual=J_dual,
            amp_10=abs(R_e[idx10]), amp_01=abs(R_e[idx01]), amp_00=abs(R_e[idx00]),
            pfrac_10=pfrac_10, pfrac_01=pfrac_01, combined_pfrac=combined_pfrac,
            vf=vf, binary_pct=bin_pct, modes=modes_e, R=R_e)
end

# Baselines: phase ramp along x (targets (1,0) only) and along y (targets (0,1) only)
yvals = [c[2] for c in centroids]
yspan = max(maximum(yvals) - minimum(yvals), eps())
rho_phase_y = 0.5 .+ 0.45 .* sin.(2π .* ((yvals .- minimum(yvals)) ./ yspan))
rho_phase_y .= clamp.(rho_phase_y .- mean(rho_phase_y) .+ 0.5, 0.0, 1.0)

res_dual_phase_x = evaluate_dual(rho_phase, 64.0)
res_dual_phase_y = evaluate_dual(rho_phase_y, 64.0)
res_dual_uniform = evaluate_dual(rho_uniform, 64.0)

println("  Phase-ramp x: (1,0)=$(round(100*res_dual_phase_x.pfrac_10, digits=2))%, (0,1)=$(round(100*res_dual_phase_x.pfrac_01, digits=2))%, combined=$(round(100*res_dual_phase_x.combined_pfrac, digits=2))%")
println("  Phase-ramp y: (1,0)=$(round(100*res_dual_phase_y.pfrac_10, digits=2))%, (0,1)=$(round(100*res_dual_phase_y.pfrac_01, digits=2))%, combined=$(round(100*res_dual_phase_y.combined_pfrac, digits=2))%")

# Initialize dual-beam from a 2D phase-ramp that couples to both (1,0) and (0,1).
# Average of x- and y-ramps ensures both targets have non-zero initial power.
rho_dual_init = 0.5 .* (rho_phase .+ rho_phase_y)
rho_dual_init .= clamp.(rho_dual_init .- mean(rho_dual_init) .+ 0.5, 0.0, 1.0)
println("\n▸ Running dual-beam optimization (init from 2D ramp)")
rho_dual_opt, trace_dual = run_optimization_beamsteer(
    Z_per, Mt, v, Q_dual, config, W, w_sum, copy(rho_dual_init);
    betas=betas, iters_per_beta=iters_per_beta, alpha_vf=alpha_vf,
    alpha_bin=alpha_bin, verbose=true)
res_dual_opt = evaluate_dual(rho_dual_opt, 64.0)

println("\n  Dual-beam results:")
println("    Uniform:      combined=$(round(100*res_dual_uniform.combined_pfrac, digits=2))%  (1,0)=$(round(100*res_dual_uniform.pfrac_10, digits=2))%, (0,1)=$(round(100*res_dual_uniform.pfrac_01, digits=2))%")
println("    Phase-ramp x: combined=$(round(100*res_dual_phase_x.combined_pfrac, digits=2))%  (1,0)=$(round(100*res_dual_phase_x.pfrac_10, digits=2))%, (0,1)=$(round(100*res_dual_phase_x.pfrac_01, digits=2))%")
println("    Phase-ramp y: combined=$(round(100*res_dual_phase_y.combined_pfrac, digits=2))%  (1,0)=$(round(100*res_dual_phase_y.pfrac_10, digits=2))%, (0,1)=$(round(100*res_dual_phase_y.pfrac_01, digits=2))%")
println("    Optimized:    combined=$(round(100*res_dual_opt.combined_pfrac, digits=2))%  (1,0)=$(round(100*res_dual_opt.pfrac_10, digits=2))%, (0,1)=$(round(100*res_dual_opt.pfrac_01, digits=2))%")

# Best baseline = max combined of the two ramps
best_baseline_combined = max(res_dual_phase_x.combined_pfrac, res_dual_phase_y.combined_pfrac)
advantage_dB = 10 * log10(max(res_dual_opt.combined_pfrac, 1e-30) / max(best_baseline_combined, 1e-30))
println("    TO advantage over best ramp: $(round(advantage_dB, digits=2)) dB")

# Save weighted results
summary_w = DataFrame(
    case=["uniform", "phase_ramp_x", "phase_ramp_y", "optimized_dual"],
    J_dual=[res_dual_uniform.J_dual, res_dual_phase_x.J_dual, res_dual_phase_y.J_dual, res_dual_opt.J_dual],
    pfrac_10=[res_dual_uniform.pfrac_10, res_dual_phase_x.pfrac_10, res_dual_phase_y.pfrac_10, res_dual_opt.pfrac_10],
    pfrac_01=[res_dual_uniform.pfrac_01, res_dual_phase_x.pfrac_01, res_dual_phase_y.pfrac_01, res_dual_opt.pfrac_01],
    combined_pfrac=[res_dual_uniform.combined_pfrac, res_dual_phase_x.combined_pfrac, res_dual_phase_y.combined_pfrac, res_dual_opt.combined_pfrac],
    amp_10=[res_dual_uniform.amp_10, res_dual_phase_x.amp_10, res_dual_phase_y.amp_10, res_dual_opt.amp_10],
    amp_01=[res_dual_uniform.amp_01, res_dual_phase_x.amp_01, res_dual_phase_y.amp_01, res_dual_opt.amp_01],
    amp_00=[res_dual_uniform.amp_00, res_dual_phase_x.amp_00, res_dual_phase_y.amp_00, res_dual_opt.amp_00],
    binary_pct=[res_dual_uniform.binary_pct, res_dual_phase_x.binary_pct, res_dual_phase_y.binary_pct, res_dual_opt.binary_pct],
)
CSV.write(joinpath(DATA_DIR, "results_beamsteer_weighted_summary.csv"), summary_w)
CSV.write(joinpath(DATA_DIR, "results_beamsteer_weighted_trace.csv"), trace_dual)

# Floquet mode comparison figure for weighted case
prop_idx_w = findall(m -> m.propagating, res_dual_opt.modes)
prop_sorted_w = sort(prop_idx_w; by=i -> (res_dual_opt.modes[i].m == 0 && res_dual_opt.modes[i].n == 0 ? -1 : 0,
                                          abs(res_dual_opt.modes[i].m) + abs(res_dual_opt.modes[i].n),
                                          res_dual_opt.modes[i].m, res_dual_opt.modes[i].n))
mode_labels_w = ["($(res_dual_opt.modes[i].m),$(res_dual_opt.modes[i].n))" for i in prop_sorted_w]
mode_w_df = DataFrame(
    mode_index=1:length(prop_sorted_w),
    mode_label=mode_labels_w,
    R_phase_x_abs=[abs(res_dual_phase_x.R[i]) for i in prop_sorted_w],
    R_phase_y_abs=[abs(res_dual_phase_y.R[i]) for i in prop_sorted_w],
    R_opt_abs=[abs(res_dual_opt.R[i]) for i in prop_sorted_w],
)
CSV.write(joinpath(DATA_DIR, "results_beamsteer_weighted_floquet.csv"), mode_w_df)

fig_w = plot_scatter(
    [collect(mode_w_df.mode_index), collect(mode_w_df.mode_index), collect(mode_w_df.mode_index)],
    [collect(mode_w_df.R_phase_x_abs), collect(mode_w_df.R_phase_y_abs), collect(mode_w_df.R_opt_abs)];
    mode=["lines+markers", "lines+markers", "lines+markers"],
    legend=["Phase-ramp x", "Phase-ramp y", "Optimized (dual)"],
    color=["#0072B2", "#D55E00", "#009E73"],
    dash=["dash", "dashdot", "solid"],
    marker_size=[8, 8, 8],
    xlabel="Propagating mode index",
    ylabel="Floquet reflection amplitude |R_mn|",
    title="Weighted dual-beam: propagating Floquet amplitudes",
    width=620, height=420, fontsize=14)
set_legend!(fig_w; position=:topright)
savefig(fig_w, joinpath(FIG_DIR, "fig_results_beamsteer_weighted_modes.pdf"))

fig_w_trace = plot_scatter(
    collect(trace_dual.iter), collect(trace_dual.J_target);
    mode="lines+markers", legend="Dual-beam objective", color="#0072B2",
    dash="solid", marker_size=6,
    xlabel="Iteration", ylabel="J_dual",
    title="Weighted dual-beam optimization convergence",
    width=560, height=400, fontsize=14)
set_legend!(fig_w_trace; position=:topright)
savefig(fig_w_trace, joinpath(FIG_DIR, "fig_results_beamsteer_weighted_trace.pdf"))

println("\n  ✓ Saved: data/results_beamsteer_weighted_*.csv")
println("  ✓ Saved: figures/fig_results_beamsteer_weighted_*.pdf")

println("\n" * "=" ^ 70)
println("  Weighted Dual-Beam Summary")
println("=" ^ 70)
println("  Targets: (1,0) + (0,1)")
println("  Best ramp combined: $(round(100*best_baseline_combined, digits=2))%")
println("  Optimized combined: $(round(100*res_dual_opt.combined_pfrac, digits=2))%")
println("  TO advantage: $(round(advantage_dB, digits=2)) dB")
println("=" ^ 70)
