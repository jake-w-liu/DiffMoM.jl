# 18_periodic_to_multistart_study.jl — Multi-start study for periodic TO nontriviality
#
# Purpose:
#   Stress-test initialization dependence for the lambda/2 periodic RCS objective.
#   This directly supports TAP-grade robustness claims by checking whether a
#   nontrivial topology branch exists beyond the near-uniform solution.
#
# Outputs:
#   data/results_multistart_summary.csv
#   data/results_multistart_best_rho.csv
#   figures/fig_results_multistart_distribution.pdf
#
# Run:
#   julia --project=. examples/18_periodic_to_multistart_study.jl
#
# Optional env vars:
#   DMOM_MS_NX=10
#   DMOM_MS_NY=10
#   DMOM_MS_ITERS_PER_BETA=20
#   DMOM_MS_BETAS=1,2,4,8,16,32,64
#   DMOM_MS_RANDOM_STARTS=4
#   DMOM_MS_SEED=2026

using DiffMoM
using LinearAlgebra
using SparseArrays
using StaticArrays
using Statistics
using Random
using CSV, DataFrames
using PlotlySupply
import PlotlySupply: savefig

const PKG_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(PKG_DIR, "data")
const FIG_DIR = joinpath(PKG_DIR, "figures")
mkpath(DATA_DIR)
mkpath(FIG_DIR)

function parse_betas(s::String)
    vals = Float64[]
    for part in split(s, ",")
        t = strip(part)
        isempty(t) && continue
        push!(vals, parse(Float64, t))
    end
    isempty(vals) && error("DMOM_MS_BETAS produced empty beta list")
    return vals
end

function smooth_raw_density(rho_raw::Vector{Float64}, W::SparseMatrixCSC{Float64, Int}, w_sum::Vector{Float64})
    return (W * rho_raw) ./ w_sum
end

"""
    run_optimization(Z_per, Mt, v, Q, config, W, w_sum, rho0; kwargs...)

Same optimization core as ex15, returned for multi-start analysis.
Returns `(rho_opt, trace)`.
"""
function run_optimization(Z_per::Matrix{ComplexF64}, Mt, v, Q, config, W, w_sum, rho0;
                          betas=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
                          iters_per_beta=20, alpha_vf=0.1, m_lbfgs=10, verbose=false)
    Nt_loc = length(rho0)
    vf_target = config.vf_target
    rho = copy(rho0)
    project!(x) = (x .= clamp.(x, 0.0, 1.0); x)

    trace = DataFrame(iter=Int[], beta=Float64[], J_scatter=Float64[],
                      J_vf=Float64[], J_total=Float64[], gnorm=Float64[],
                      vf=Float64[], frac_binary=Float64[])
    global_iter = 0

    for beta in betas
        s_list = Vector{Float64}[]
        y_list = Vector{Float64}[]
        rho_old = copy(rho)
        g_old = zeros(Nt_loc)

        for _ in 1:iters_per_beta
            global_iter += 1

            rho_tilde, rho_bar = filter_and_project(W, w_sum, rho, beta)
            Z_total = Z_per + assemble_Z_penalty(Mt, rho_bar, config)
            F = lu(Z_total)
            I_c = F \ v

            QI_c = Q * I_c
            J_scat = real(dot(I_c, QI_c))
            vf_cur = mean(rho_bar)
            J_vf = alpha_vf * (vf_cur - vf_target)^2
            J_tot = J_scat + J_vf

            lam = F' \ QI_c
            g_rb = gradient_density(Mt, Vector{ComplexF64}(I_c), Vector{ComplexF64}(lam), rho_bar, config)
            g_rb .+= alpha_vf * 2.0 * (vf_cur - vf_target) / Nt_loc
            g = gradient_chain_rule(g_rb, rho_tilde, W, w_sum, beta)

            gnorm = norm(g)
            fb = count(x -> x < 0.05 || x > 0.95, rho_bar) / Nt_loc
            push!(trace, (global_iter, beta, J_scat, J_vf, J_tot, gnorm, vf_cur, fb))

            verbose && global_iter % max(1, iters_per_beta) == 0 &&
                println("    beta=$(Int(beta)) it=$global_iter J=$(round(J_tot, sigdigits=5))")

            if global_iter > 1
                s_k = rho .- rho_old
                y_k = g .- g_old
                if dot(s_k, y_k) > 1e-30
                    push!(s_list, copy(s_k))
                    push!(y_list, copy(y_k))
                    if length(s_list) > m_lbfgs
                        popfirst!(s_list)
                        popfirst!(y_list)
                    end
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
            for _ in 1:20
                rho_trial = project!(rho_old .+ alpha_ls .* d)
                _, rb = filter_and_project(W, w_sum, rho_trial, beta)
                I_t = (Z_per + assemble_Z_penalty(Mt, rb, config)) \ v
                Jt = real(dot(I_t, Q * I_t)) + alpha_vf * (mean(rb) - vf_target)^2
                if Jt <= J_tot + 1e-4 * alpha_ls * dot(g, d)
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

println("="^72)
println("  Multi-Start Study: Periodic TO Nontriviality (lambda/2 unit cell)")
println("="^72)

seed = parse(Int, get(ENV, "DMOM_MS_SEED", "2026"))
Random.seed!(seed)

freq = 10e9
c0 = 3e8
lambda = c0 / freq
k = 2π / lambda

Nx = parse(Int, get(ENV, "DMOM_MS_NX", "10"))
Ny = parse(Int, get(ENV, "DMOM_MS_NY", string(Nx)))
iters_per_beta = parse(Int, get(ENV, "DMOM_MS_ITERS_PER_BETA", "20"))
betas = parse_betas(get(ENV, "DMOM_MS_BETAS", "1,2,4,8,16,32,64"))
n_rand = parse(Int, get(ENV, "DMOM_MS_RANDOM_STARTS", "4"))

dx_cell = 0.5 * lambda
dy_cell = 0.5 * lambda

mesh = make_rect_plate(dx_cell, dy_cell, Nx, Ny)
lattice = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k)
rwg = build_rwg_periodic(mesh, lattice; precheck=false)
Nt = ntriangles(mesh)
N = rwg.nedges

println("  Setup: Nx=$Nx, Ny=$Ny, Nt=$Nt, N=$N, random_starts=$n_rand")
println("  Betas: $(Int.(betas))  iters_per_beta=$iters_per_beta")

println("  Assembling Z_per ...")
Z_per = Matrix{ComplexF64}(assemble_Z_efie_periodic(mesh, rwg, k, lattice))
Mt = precompute_triangle_mass(mesh, rwg)

edge_len = dx_cell / Nx
r_min = 2.5 * edge_len
W, w_sum = build_filter_weights(mesh, r_min)

pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
v = Vector{ComplexF64}(assemble_excitation(mesh, rwg, pw))
grid_ff = make_sph_grid(20, 40)
Q_spec = Matrix{ComplexF64}(specular_rcs_objective(mesh, rwg, grid_ff, k, lattice;
                                                   half_angle=10 * π / 180,
                                                   polarization=:x))
config = DensityConfig(; p=3.0, Z_max_factor=10.0, vf_target=0.5, reactive=true)

# PEC baseline for absolute comparison
_, rho_bar_pec = filter_and_project(W, w_sum, ones(Nt), betas[end])
I_pec = (Z_per + assemble_Z_penalty(Mt, rho_bar_pec, config)) \ v
J_pec = real(dot(I_pec, Q_spec * I_pec))

pol_inc = SVector(1.0, 0.0, 0.0)
modes_pec, R_pec = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_pec), k, lattice;
                                           pol=pol_inc, E0=1.0)
idx00 = findfirst(m -> m.m == 0 && m.n == 0, modes_pec)
idx00 === nothing && error("Specular (0,0) mode missing")
R00_pec = abs(R_pec[idx00])

centroids = [triangle_center(mesh, t) for t in 1:Nt]
xs = [c[1] for c in centroids]
ys = [c[2] for c in centroids]
xmin, xmax = minimum(xs), maximum(xs)
ymin, ymax = minimum(ys), maximum(ys)

function make_checker_seed(mesh::TriMesh, Nx::Int, Ny::Int)
    Nt = ntriangles(mesh)
    cts = [triangle_center(mesh, t) for t in 1:Nt]
    xs = [c[1] for c in cts]
    ys = [c[2] for c in cts]
    x0, y0 = minimum(xs), minimum(ys)
    dx = (maximum(xs) - x0 + eps()) / Nx
    dy = (maximum(ys) - y0 + eps()) / Ny
    rho = zeros(Float64, Nt)
    for t in 1:Nt
        ix = clamp(floor(Int, (cts[t][1] - x0) / dx) + 1, 1, Nx)
        iy = clamp(floor(Int, (cts[t][2] - y0) / dy) + 1, 1, Ny)
        rho[t] = isodd(ix + iy) ? 0.8 : 0.2
    end
    return rho
end

starts = Vector{NamedTuple}()
push!(starts, (name="uniform", family="deterministic", rho=fill(0.5, Nt)))
push!(starts, (
    name="sinx",
    family="deterministic",
    rho=clamp.(0.5 .+ 0.2 .* sin.(2π .* ((xs .- xmin) ./ max(xmax - xmin, eps()))), 0.0, 1.0),
))
push!(starts, (
    name="sinxy",
    family="deterministic",
    rho=clamp.(0.5 .+ 0.25 .* sin.(2π .* ((xs .- xmin) ./ max(xmax - xmin, eps()))) .*
                sin.(2π .* ((ys .- ymin) ./ max(ymax - ymin, eps()))), 0.0, 1.0),
))
push!(starts, (name="checker_bias", family="deterministic", rho=make_checker_seed(mesh, Nx, Ny)))

for i in 1:n_rand
    rho_raw = clamp.(0.5 .+ 0.6 .* (rand(Nt) .- 0.5), 0.0, 1.0)
    rho_sm = clamp.(smooth_raw_density(rho_raw, W, w_sum), 0.0, 1.0)
    push!(starts, (name="random_$i", family="random", rho=rho_sm))
end

println("  Total starts: $(length(starts))")

rows = DataFrame(
    start_name=String[],
    family=String[],
    J_final=Float64[],
    J_vs_pec_dB=Float64[],
    R00_abs=Float64[],
    R00_vs_pec_dB=Float64[],
    vf=Float64[],
    binary_pct=Float64[],
    rho_min=Float64[],
    rho_max=Float64[],
    rho_std=Float64[],
    runtime_s=Float64[],
)

best_name = ""
best_rho = zeros(Float64, Nt)
best_J = Inf

for s in starts
    global best_J, best_name
    println("\n▸ Start: $(s.name)")
    t0 = time()
    rho_opt, trace = run_optimization(
        Z_per, Mt, v, Q_spec, config, W, w_sum, s.rho;
        betas=betas, iters_per_beta=iters_per_beta, alpha_vf=0.1, verbose=false,
    )
    runtime = time() - t0

    _, rho_bar = filter_and_project(W, w_sum, rho_opt, betas[end])
    I_opt = (Z_per + assemble_Z_penalty(Mt, rho_bar, config)) \ v
    J_opt = real(dot(I_opt, Q_spec * I_opt))

    modes, R = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_opt), k, lattice;
                                       pol=pol_inc, E0=1.0)
    idx = findfirst(m -> m.m == 0 && m.n == 0, modes)
    idx === nothing && error("Specular mode missing for start $(s.name)")
    R00 = abs(R[idx])

    dB_J = 10 * log10(max(J_opt, 1e-30) / max(J_pec, 1e-30))
    dB_R = 20 * log10(max(R00, 1e-30) / max(R00_pec, 1e-30))
    vf = mean(rho_bar)
    bin_pct = 100 * count(x -> x < 0.05 || x > 0.95, rho_bar) / Nt
    push!(rows, (s.name, s.family, J_opt, dB_J, R00, dB_R, vf, bin_pct,
                 minimum(rho_bar), maximum(rho_bar), std(rho_bar), runtime))

    println("  J=$(round(J_opt, sigdigits=6)), J_vs_PEC=$(round(dB_J, digits=3)) dB, " *
            "|R00|=$(round(R00, sigdigits=6)), binary=$(round(bin_pct, digits=1))%")
    println("  vf=$(round(vf, digits=6)), rho_std=$(round(std(rho_bar), sigdigits=4)), " *
            "runtime=$(round(runtime, digits=1)) s")

    if J_opt < best_J
        best_J = J_opt
        best_name = s.name
        best_rho .= rho_bar
    end
end

sort!(rows, :J_final)
rows.rank = 1:nrow(rows)
CSV.write(joinpath(DATA_DIR, "results_multistart_summary.csv"), rows)

CSV.write(joinpath(DATA_DIR, "results_multistart_best_rho.csv"),
          DataFrame(triangle=1:Nt, rho_bar=best_rho, x=xs, y=ys))

# Plot: J vs PEC (dB) and binary fraction per start
det = rows[rows.family .== "deterministic", :]
rnd = rows[rows.family .== "random", :]

fig = plot_scatter(
    [collect(det.binary_pct), collect(rnd.binary_pct)],
    [collect(det.J_vs_pec_dB), collect(rnd.J_vs_pec_dB)];
    mode=["markers", "markers"],
    legend=["Deterministic starts", "Random starts"],
    color=["#0072B2", "#D55E00"],
    marker_size=[10, 10],
    marker_symbol=["circle", "diamond"],
    xlabel="Final binary fraction [%]",
    ylabel="J vs PEC [dB]",
    title="Multi-start distribution: nontriviality vs objective",
    width=560, height=400, fontsize=14)
set_legend!(fig; position=:topright)
savefig(fig, joinpath(FIG_DIR, "fig_results_multistart_distribution.pdf"))

println("\n" * "="^72)
println("  Multi-Start Summary")
println("="^72)
println("  Baseline PEC: J=$(round(J_pec, sigdigits=6)), |R00|=$(round(R00_pec, sigdigits=6))")
println("  Best start: $(best_name), J=$(round(best_J, sigdigits=6))")
println("  Saved: data/results_multistart_summary.csv")
println("  Saved: data/results_multistart_best_rho.csv")
println("  Saved: figures/fig_results_multistart_distribution.pdf")
println("="^72)
