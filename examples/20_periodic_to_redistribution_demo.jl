# 20_periodic_to_redistribution_demo.jl — RCS reduction on a multi-mode cell
#
# Demonstrates TO-discovered binary topology on a 1.2λ cell that outperforms
# analytical baselines (checkerboard) for specular RCS reduction.
#
# Key: binarization penalty forces binary solutions; the optimizer must use
# spatial topology (diffraction into non-specular modes) instead of absorption.
#
# Run: julia --project=. examples/20_periodic_to_redistribution_demo.jl

using DiffMoM
using LinearAlgebra
using StaticArrays
using Statistics
using Random
using CSV, DataFrames
using PlotlySupply
import PlotlySupply: savefig

Random.seed!(2024)

const PKG_DIR  = dirname(@__DIR__)
const DATA_DIR = joinpath(PKG_DIR, "..", "paper", "data")
const FIG_DIR  = joinpath(PKG_DIR, "..", "paper", "figs")
mkpath(DATA_DIR)
mkpath(FIG_DIR)

println("=" ^ 70)
println("  RCS Reduction on Multi-Mode Cell (d = 1.2λ)")
println("=" ^ 70)

# ── Problem setup ──────────────────────────────────────────────────────────
freq   = 10e9
c0     = 3e8
lambda = c0 / freq
k      = 2π / lambda
eta0   = 376.730313668

dx_cell = 1.2 * lambda
dy_cell = 1.2 * lambda

Nx = parse(Int, get(ENV, "DMOM_RD_NX", "14"))
Ny = parse(Int, get(ENV, "DMOM_RD_NY", string(Nx)))
iters_per_beta = parse(Int, get(ENV, "DMOM_RD_ITERS", "30"))
betas = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

println("  Frequency: $(freq/1e9) GHz, λ = $(round(lambda*1e3, digits=2)) mm")
println("  Unit cell: $(round(dx_cell/lambda, digits=2))λ × $(round(dy_cell/lambda, digits=2))λ")
println("  Mesh: $(Nx)×$(Ny)")

mesh = make_rect_plate(dx_cell, dy_cell, Nx, Ny)
lattice = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k)
rwg = build_rwg_periodic(mesh, lattice; precheck=false)
Nt = ntriangles(mesh)
N  = rwg.nedges
println("  Discretization: $(Nt) triangles, $(N) RWG edges")

println("  Assembling periodic EFIE matrix...")
Z_per = Matrix{ComplexF64}(assemble_Z_efie_periodic(mesh, rwg, k, lattice))

Mt = precompute_triangle_mass(mesh, rwg)
edge_len = dx_cell / Nx
r_min = 2.5 * edge_len
W, w_sum = build_filter_weights(mesh, r_min)

pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
v  = Vector{ComplexF64}(assemble_excitation(mesh, rwg, pw))

# Specular cone objective
Ntheta_ff, Nphi_ff = 20, 40
spec_half_angle = 10 * π / 180
grid_ff = make_sph_grid(Ntheta_ff, Nphi_ff)
Q_spec = Matrix{ComplexF64}(specular_rcs_objective(
    mesh, rwg, grid_ff, k, lattice; half_angle=spec_half_angle, polarization=:x))

Z_max_f = parse(Float64, get(ENV, "DMOM_RD_ZMAX", "100.0"))
reactive = parse(Bool, get(ENV, "DMOM_RD_REACTIVE", "true"))
config = DensityConfig(; p=3.0, Z_max_factor=Z_max_f, vf_target=0.5, reactive=reactive)
alpha_vf = 0.5
centroids = [triangle_center(mesh, t) for t in 1:Nt]
pol_inc = SVector(1.0, 0.0, 0.0)

# ── Reference PEC ──────────────────────────────────────────────────────────
println("\n▸ Reference: full PEC plate")
_, rho_bar_pec = filter_and_project(W, w_sum, ones(Nt), betas[end])
Z_pen_pec = assemble_Z_penalty(Mt, rho_bar_pec, config)
I_pec = (Z_per + Z_pen_pec) \ v
J_spec_pec = real(dot(I_pec, Q_spec * I_pec))
modes_pec, R_pec = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_pec), k, lattice;
                                           pol=pol_inc, E0=1.0)
spec_pos = findfirst(i -> modes_pec[i].m == 0 && modes_pec[i].n == 0, eachindex(modes_pec))
println("  PEC: J_spec=$(round(J_spec_pec, sigdigits=4)), |R00|=$(round(abs(R_pec[spec_pos]), sigdigits=4))")

# ── Gradient verification ──────────────────────────────────────────────────
println("\n▸ Gradient verification (specular + binarization)")
rho_gv = 0.2 .+ 0.6 * rand(Nt)
beta_gv = 4.0
alpha_bin_gv = 1e-7  # binarization weight for gradient check
rt_gv, rb_gv = filter_and_project(W, w_sum, rho_gv, beta_gv)
Z_pen_gv = assemble_Z_penalty(Mt, rb_gv, config)
Z_gv = Z_per + Z_pen_gv
F_gv = lu(Z_gv)
I_gv = F_gv \ v

# Adjoint gradient of specular objective
lam_gv = F_gv' \ (Q_spec * I_gv)
g_adj = gradient_density_full(Mt, Vector{ComplexF64}(I_gv), Vector{ComplexF64}(lam_gv),
                              rt_gv, rb_gv, config, W, w_sum, beta_gv)
# Add binarization penalty gradient: d/dρ̄[α_bin * Σρ̄(1-ρ̄)] = α_bin*(1-2ρ̄)
g_bin_rb = alpha_bin_gv .* (1.0 .- 2.0 .* rb_gv)
g_adj .+= gradient_chain_rule(g_bin_rb, rt_gv, W, w_sum, beta_gv)
# Volume fraction gradient
vf_gv = mean(rb_gv)
g_adj .+= gradient_chain_rule(
    fill(alpha_vf * 2.0 * (vf_gv - config.vf_target) / Nt, Nt),
    rt_gv, W, w_sum, beta_gv)

subset = sort(randperm(Nt)[1:8])
h_fd = 1e-5
rel_errs = Float64[]
for t in subset
    rp = copy(rho_gv); rp[t] += h_fd
    rm = copy(rho_gv); rm[t] -= h_fd
    _, rbp = filter_and_project(W, w_sum, rp, beta_gv)
    _, rbm = filter_and_project(W, w_sum, rm, beta_gv)
    Ip = (Z_per + assemble_Z_penalty(Mt, rbp, config)) \ v
    Im = (Z_per + assemble_Z_penalty(Mt, rbm, config)) \ v
    Jp = real(dot(Ip, Q_spec * Ip)) + alpha_bin_gv * sum(rbp .* (1 .- rbp)) +
         alpha_vf * (mean(rbp) - config.vf_target)^2
    Jm = real(dot(Im, Q_spec * Im)) + alpha_bin_gv * sum(rbm .* (1 .- rbm)) +
         alpha_vf * (mean(rbm) - config.vf_target)^2
    g_fd_t = (Jp - Jm) / (2h_fd)
    push!(rel_errs, abs(g_adj[t] - g_fd_t) / max(abs(g_fd_t), 1e-20))
end
max_rel_gv = maximum(rel_errs)
println("  Max relative error: $(round(max_rel_gv, sigdigits=3))")
max_rel_gv < 1e-5 || error("Gradient check exceeds 1e-5 tolerance: $(max_rel_gv)")

# ── Optimization ───────────────────────────────────────────────────────────
# Binarization continuation: α_bin increases with β to gradually force binary
alpha_bin_max = 5e-7  # tuned so binarization penalty ~ specular at final stages

function run_rcs_binary_opt(Z_per, Mt, v, Q_spec, config, W, w_sum, rho0;
                            betas, iters_per_beta, alpha_vf, alpha_bin_max,
                            m_lbfgs=10, verbose=true)
    Nt_loc = length(rho0)
    beta_max = betas[end]
    vf_target = config.vf_target
    rho = copy(rho0)
    project!(x) = (x .= clamp.(x, 0.0, 1.0); x)

    trace = DataFrame(iter=Int[], beta=Float64[], J_spec=Float64[], J_bin=Float64[],
                      J_total=Float64[], gnorm=Float64[], vf=Float64[],
                      frac_binary=Float64[], R00=Float64[])
    rho_by_beta = Dict{Float64, Vector{Float64}}()
    global_iter = 0

    for beta in betas
        # Binarization ramp: zero for first 2 stages, then quadratic ramp
        alpha_bin = beta <= betas[min(2, end)] ? 0.0 :
                    alpha_bin_max * ((beta - betas[2]) / (beta_max - betas[2]))^2
        verbose && println("\n  ── β = $(Int(beta)), α_bin = $(round(alpha_bin, sigdigits=2)) ──")

        s_list = Vector{Float64}[]
        y_list = Vector{Float64}[]
        rho_old = copy(rho)
        g_old = zeros(Nt_loc)

        for it in 1:iters_per_beta
            global_iter += 1

            rho_tilde, rho_bar = filter_and_project(W, w_sum, rho, beta)
            Z_pen = assemble_Z_penalty(Mt, rho_bar, config)
            Z_total = Z_per + Z_pen
            F = lu(Z_total)
            I_c = F \ v

            J_spec = real(dot(I_c, Q_spec * I_c))
            J_bin = alpha_bin * sum(rho_bar .* (1.0 .- rho_bar))
            vf_cur = mean(rho_bar)
            J_vf = alpha_vf * (vf_cur - vf_target)^2
            J_total = J_spec + J_bin + J_vf

            # Adjoint for specular
            lam = F' \ (Q_spec * I_c)
            g_rb = gradient_density(Mt, Vector{ComplexF64}(I_c),
                                    Vector{ComplexF64}(lam), rho_bar, config)
            # Binarization gradient
            g_rb .+= alpha_bin .* (1.0 .- 2.0 .* rho_bar)
            # Volume fraction gradient
            g_rb .+= alpha_vf * 2.0 * (vf_cur - vf_target) / Nt_loc
            g = gradient_chain_rule(g_rb, rho_tilde, W, w_sum, beta)

            gnorm = norm(g)
            fb = count(x -> x < 0.05 || x > 0.95, rho_bar) / Nt_loc

            # Quick R00 check
            modes_it, R_it = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_c), k, lattice;
                                                     pol=pol_inc, E0=1.0)
            R00_it = abs(R_it[spec_pos])
            push!(trace, (global_iter, beta, J_spec, J_bin, J_total, gnorm, vf_cur, fb, R00_it))

            if verbose && (it <= 2 || it == iters_per_beta || it % 5 == 0)
                println("    it=$global_iter J=$(round(J_spec, sigdigits=4)) " *
                        "|R00|=$(round(R00_it, digits=4)) " *
                        "|g|=$(round(gnorm, sigdigits=3)) " *
                        "vf=$(round(vf_cur, digits=3)) bin=$(round(100fb, digits=0))%")
            end

            # L-BFGS direction
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
            gamma_bfgs = mc > 0 ? dot(s_list[end], y_list[end]) / dot(y_list[end], y_list[end]) : 0.01
            r_v = gamma_bfgs .* q_v
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
                Jt = real(dot(I_t, Q_spec * I_t)) +
                     alpha_bin * sum(rb .* (1.0 .- rb)) +
                     alpha_vf * (mean(rb) - vf_target)^2
                if Jt <= J_total + 1e-4 * alpha_ls * dot(g, d)
                    rho .= rho_trial
                    accepted = true
                    break
                end
                alpha_ls *= 0.5
            end
            !accepted && (rho .= project!(rho_old .+ alpha_ls .* d))
        end
        rho_by_beta[beta] = copy(rho)
    end

    return rho, trace, rho_by_beta
end

println("\n▸ Running RCS optimization with binarization (random init)")
rho0 = rand(Nt)
t0 = time()
rho_opt_last, trace, rho_by_beta = run_rcs_binary_opt(
    Z_per, Mt, v, Q_spec, config, W, w_sum, rho0;
    betas=betas, iters_per_beta=iters_per_beta, alpha_vf=alpha_vf,
    alpha_bin_max=alpha_bin_max, verbose=true)
println("  Runtime: $(round(time()-t0, digits=1)) s")

function evaluate_candidate(rho_raw)
    _, rho_bar = filter_and_project(W, w_sum, rho_raw, betas[end])
    Z_pen = assemble_Z_penalty(Mt, rho_bar, config)
    I_c = (Z_per + Z_pen) \ v
    modes_c, R_c = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_c), k, lattice;
                                           pol=pol_inc, E0=1.0)
    R00_c = abs(R_c[spec_pos])
    binary_c = count(x -> x < 0.05 || x > 0.95, rho_bar) / Nt
    return (; rho_raw=copy(rho_raw), rho_bar, R00=R00_c, binary=binary_c)
end

candidates = [evaluate_candidate(rho_opt_last)]
for beta in sort(collect(keys(rho_by_beta)))
    push!(candidates, evaluate_candidate(rho_by_beta[beta]))
end
eligible = filter(c -> c.binary >= 0.85, candidates)
isempty(eligible) && error("no redistribution candidate reached the binary threshold")
best_candidate = eligible[argmin([c.R00 for c in eligible])]
rho_opt = best_candidate.rho_raw

# ── Final analysis ─────────────────────────────────────────────────────────
rho_bar_final = best_candidate.rho_bar
Z_pen_opt = assemble_Z_penalty(Mt, rho_bar_final, config)
I_opt = (Z_per + Z_pen_opt) \ v
J_spec_opt = real(dot(I_opt, Q_spec * I_opt))

modes_opt, R_opt = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_opt), k, lattice;
                                           pol=pol_inc, E0=1.0)
pb_opt = power_balance(Vector{ComplexF64}(I_opt), Z_pen_opt, dx_cell*dy_cell, k, modes_opt, R_opt;
                       transmission=:floquet)

R00_pec = abs(R_pec[spec_pos])
R00_opt = abs(R_opt[spec_pos])
binary_frac = count(x -> x < 0.05 || x > 0.95, rho_bar_final) / Nt
vf_final = mean(rho_bar_final)

function mode_power_fraction(mode, R, k)
    mode.propagating || return 0.0
    return abs2(R) * real(mode.kz) / k
end

J_spec_red_dB = 10 * log10(max(J_spec_opt, 1e-30) / max(J_spec_pec, 1e-30))
R00_red_dB = 20 * log10(max(R00_opt, 1e-30) / max(R00_pec, 1e-30))

println("\n" * "=" ^ 70)
println("  RCS Optimization Results")
println("=" ^ 70)
println("  J_spec: PEC=$(round(J_spec_pec, sigdigits=4)) → opt=$(round(J_spec_opt, sigdigits=4)) ($(round(J_spec_red_dB, digits=2)) dB)")
println("  |R00|: $(round(R00_pec, sigdigits=4)) → $(round(R00_opt, sigdigits=4)) ($(round(R00_red_dB, digits=2)) dB)")
println("  Power: refl=$(round(100pb_opt.refl_frac, digits=1))%, abs=$(round(100pb_opt.abs_frac, digits=1))%, " *
        "trans=$(round(100pb_opt.trans_frac, digits=1))%, resid=$(round(100pb_opt.resid_frac, digits=1))%")
println("  Binary fraction: $(round(100binary_frac, digits=1))%")
println("  Volume fraction: $(round(vf_final, digits=3))")

# ── Baselines ──────────────────────────────────────────────────────────────
println("\n▸ Computing baselines")

function analyze_baseline(name, rho_bar_in)
    Z_pen = assemble_Z_penalty(Mt, rho_bar_in, config)
    I_b = (Z_per + Z_pen) \ v
    J_spec_b = real(dot(I_b, Q_spec * I_b))
    modes_b, R_b = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_b), k, lattice;
                                           pol=pol_inc, E0=1.0)
    pb_b = power_balance(Vector{ComplexF64}(I_b), Z_pen, dx_cell*dy_cell, k, modes_b, R_b;
                         transmission=:floquet)
    R00_b = abs(R_b[spec_pos])
    bin = count(x -> x < 0.05 || x > 0.95, rho_bar_in) / length(rho_bar_in)
    J_red = 10 * log10(max(J_spec_b, 1e-30) / max(J_spec_pec, 1e-30))
    R00_red = 20 * log10(max(R00_b, 1e-30) / max(R00_pec, 1e-30))

    println("  $(rpad(name, 20)) |R00|=$(round(R00_b, digits=4)) ($(round(R00_red, digits=1)) dB) " *
            "bin=$(round(100bin, digits=0))%")
    return (name=name, J_spec=J_spec_b, R00=R00_b, J_red_dB=J_red, R00_red_dB=R00_red,
            refl=pb_b.refl_frac, abs_f=pb_b.abs_frac, trans=pb_b.trans_frac,
            resid=pb_b.resid_frac, bin=bin, vf=mean(rho_bar_in),
            pb=pb_b, modes=modes_b, R=R_b)
end

# Checkerboard
rho_checker = zeros(Float64, Nt)
xs = [c[1] for c in centroids]; ys = [c[2] for c in centroids]
x0, y0 = minimum(xs), minimum(ys)
for t in 1:Nt
    ix = clamp(floor(Int, (centroids[t][1] - x0) / (dx_cell / Nx)) + 1, 1, Nx)
    iy = clamp(floor(Int, (centroids[t][2] - y0) / (dy_cell / Ny)) + 1, 1, Ny)
    rho_checker[t] = isodd(ix + iy) ? 1.0 : 0.0
end
rho_gray = fill(0.5, Nt)

bl_pec     = analyze_baseline("PEC plate", rho_bar_pec)
bl_checker = analyze_baseline("Checkerboard", rho_checker)
bl_gray    = analyze_baseline("Gray sheet (ρ=0.5)", rho_gray)
bl_opt     = analyze_baseline("Optimized (binary)", rho_bar_final)

# Advantage over checkerboard
adv_dB = bl_checker.R00_red_dB - bl_opt.R00_red_dB
println("\n  Advantage over checkerboard: $(round(adv_dB, digits=2)) dB")
bl_opt.R00 < bl_checker.R00 || error("optimized redistribution design does not beat checkerboard")

# ── Save data ──────────────────────────────────────────────────────────────
CSV.write(joinpath(DATA_DIR, "results_redistribution_trace.csv"), trace)
CSV.write(joinpath(DATA_DIR, "results_redistribution_topology.csv"), DataFrame(
    triangle=1:Nt, rho_raw=rho_opt, rho_bar=rho_bar_final,
    x=[c[1] for c in centroids], y=[c[2] for c in centroids]))

summary_df = DataFrame(
    case = [bl_pec.name, bl_checker.name, bl_gray.name, bl_opt.name],
    J_spec = [bl_pec.J_spec, bl_checker.J_spec, bl_gray.J_spec, bl_opt.J_spec],
    R00_abs = [bl_pec.R00, bl_checker.R00, bl_gray.R00, bl_opt.R00],
    R00_red_dB = [bl_pec.R00_red_dB, bl_checker.R00_red_dB, bl_gray.R00_red_dB, bl_opt.R00_red_dB],
    refl_pct = 100 .* [bl_pec.refl, bl_checker.refl, bl_gray.refl, bl_opt.refl],
    abs_pct  = 100 .* [bl_pec.abs_f, bl_checker.abs_f, bl_gray.abs_f, bl_opt.abs_f],
    trans_pct = 100 .* [bl_pec.trans, bl_checker.trans, bl_gray.trans, bl_opt.trans],
    binary_pct = 100 .* [bl_pec.bin, bl_checker.bin, bl_gray.bin, bl_opt.bin],
    vf = [bl_pec.vf, bl_checker.vf, bl_gray.vf, bl_opt.vf],
)
CSV.write(joinpath(DATA_DIR, "results_redistribution_summary.csv"), summary_df)

function find_mode_index(modes, m, n)
    idx = findfirst(x -> x.m == m && x.n == n, modes)
    idx === nothing && error("missing Floquet mode ($(m),$(n))")
    return idx
end

floquet_rows = DataFrame(mode_label=String[], m=Int[], n=Int[],
    R_pec_abs=Float64[], R_checker_abs=Float64[], R_gray_abs=Float64[], R_opt_abs=Float64[],
    pfrac_pec=Float64[], pfrac_checker=Float64[], pfrac_gray=Float64[], pfrac_opt=Float64[])
for mode in bl_opt.modes
    mode.propagating || continue
    i_pec = find_mode_index(bl_pec.modes, mode.m, mode.n)
    i_checker = find_mode_index(bl_checker.modes, mode.m, mode.n)
    i_gray = find_mode_index(bl_gray.modes, mode.m, mode.n)
    i_opt = find_mode_index(bl_opt.modes, mode.m, mode.n)
    push!(floquet_rows, (
        "($(mode.m),$(mode.n))", mode.m, mode.n,
        abs(bl_pec.R[i_pec]), abs(bl_checker.R[i_checker]), abs(bl_gray.R[i_gray]), abs(bl_opt.R[i_opt]),
        mode_power_fraction(bl_pec.modes[i_pec], bl_pec.R[i_pec], k),
        mode_power_fraction(bl_checker.modes[i_checker], bl_checker.R[i_checker], k),
        mode_power_fraction(bl_gray.modes[i_gray], bl_gray.R[i_gray], k),
        mode_power_fraction(bl_opt.modes[i_opt], bl_opt.R[i_opt], k),
    ))
end
CSV.write(joinpath(DATA_DIR, "results_redistribution_floquet.csv"), floquet_rows)
println("  ✓ Saved data/results_redistribution_*.csv")

# ── Figures ────────────────────────────────────────────────────────────────
println("\n▸ Generating figures")

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
const DASHES = ["solid", "dash", "dashdot", "dot"]
const IEEE_W = 504
const IEEE_H = 360

# Topology heatmap
function density_to_grid(rho_bar_vec, cents, nx, ny, dxc, dyc)
    grid_rho = zeros(ny, nx)
    grid_cnt = zeros(Int, ny, nx)
    xs_g = [c[1] for c in cents]; ys_g = [c[2] for c in cents]
    x0_g, y0_g = minimum(xs_g), minimum(ys_g)
    dxg, dyg = dxc / nx, dyc / ny
    for t in eachindex(rho_bar_vec)
        ix = clamp(floor(Int, (cents[t][1] - x0_g) / dxg) + 1, 1, nx)
        iy = clamp(floor(Int, (cents[t][2] - y0_g) / dyg) + 1, 1, ny)
        grid_rho[iy, ix] += rho_bar_vec[t]
        grid_cnt[iy, ix] += 1
    end
    grid_rho ./= max.(grid_cnt, 1)
    return grid_rho
end

G_opt = density_to_grid(rho_bar_final, centroids, Nx, Ny, dx_cell, dy_cell)
G_checker = density_to_grid(rho_checker, centroids, Nx, Ny, dx_cell, dy_cell)

x_mm = collect(range(-dx_cell/2, dx_cell/2, length=Nx)) .* 1e3
y_mm = collect(range(-dy_cell/2, dy_cell/2, length=Ny)) .* 1e3

fig_opt = plot_heatmap(x_mm, y_mm, G_opt;
    xlabel="x [mm]", ylabel="y [mm]", title="Optimized ($(round(100binary_frac, digits=0))% binary)",
    zrange=[0, 1], colorscale="Greys", width=480, height=420)
savefig(fig_opt, joinpath(FIG_DIR, "fig_results_redistribution_topology.pdf"))

fig_check = plot_heatmap(x_mm, y_mm, G_checker;
    xlabel="x [mm]", ylabel="y [mm]", title="Checkerboard baseline",
    zrange=[0, 1], colorscale="Greys", width=480, height=420)
savefig(fig_check, joinpath(FIG_DIR, "fig_results_redistribution_checker.pdf"))

# Convergence: |R00| vs iteration
fig_conv = plot_scatter(collect(trace.iter), collect(trace.R00);
    mode="lines", color=COLORS[1], dash=DASHES[1],
    xlabel="Iteration", ylabel="|R00|",
    title="Specular reflection convergence (d = 1.2lambda)",
    width=IEEE_W, height=IEEE_H, fontsize=12)
savefig(fig_conv, joinpath(FIG_DIR, "fig_results_redistribution_convergence.pdf"))

println("  ✓ Saved figures/fig_results_redistribution_*.pdf")

# ── Final summary ──────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("  Summary: RCS Reduction Demo (d = 1.2λ)")
println("=" ^ 70)
println("  Design               |R00|    dB vs PEC  Bin%")
for bl in [bl_pec, bl_checker, bl_gray, bl_opt]
    println("  $(rpad(bl.name, 22)) $(lpad(round(bl.R00, digits=4), 6))  " *
            "$(lpad(round(bl.R00_red_dB, digits=1), 8))  $(lpad(round(100bl.bin, digits=0), 4))")
end
println("  Gradient max rel error: $(round(max_rel_gv, sigdigits=3))")
println("  Advantage over checkerboard: $(round(adv_dB, digits=2)) dB")
println("=" ^ 70)
