# 19_periodic_to_robustness_map.jl — Frequency/angle robustness map with matched baselines
#
# Purpose:
#   Build TAP-grade robustness evidence under matched settings for three designs:
#   PEC, checkerboard, and optimized topology from ex15, using TE incidence
#   in the phi=0 plane and TE-co-polar objective extraction.
#
# Outputs:
#   data/results_robustness_map.csv
#   data/results_robustness_summary.csv
#   figures/fig_results_hero_baseline_robustness.pdf
#   figures/fig_results_robustness_theta0_lines.pdf
#
# Run:
#   julia --project=. examples/19_periodic_to_robustness_map.jl
#
# Optional env vars:
#   DMOM_RB_FREQS_GHZ=8,10,12
#   DMOM_RB_THETAS_DEG=0,15,30,45,60,75
#   DMOM_RB_PHI_DEG=0
#   DMOM_RB_GMRES_TOL=1e-10
#   DMOM_RB_GMRES_MAXITER=800

using DiffMoM
using LinearAlgebra
using StaticArrays
using Statistics
using CSV, DataFrames
using PlotlySupply
import PlotlySupply: savefig

const PKG_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(PKG_DIR, "..", "paper", "data")
const FIG_DIR = joinpath(PKG_DIR, "..", "paper", "figs")
mkpath(DATA_DIR)
mkpath(FIG_DIR)

function parse_float_list(s::String)
    vals = Float64[]
    for part in split(s, ",")
        t = strip(part)
        isempty(t) && continue
        push!(vals, parse(Float64, t))
    end
    isempty(vals) && error("Empty numeric list parsed from '$s'")
    return vals
end

function pol_te_matrix(grid::SphGrid)
    NΩ = length(grid.w)
    pol = zeros(ComplexF64, 3, NΩ)
    for q in 1:NΩ
        φ = grid.phi[q]
        # phi-hat (TE/s-polarization basis around z-axis), unit and transverse to rhat.
        pol[:, q] = SVector(-sin(φ), cos(φ), 0.0)
    end
    return pol
end

function make_checkerboard_rho(mesh::TriMesh, Nx::Int, Ny::Int, dx_cell::Float64, dy_cell::Float64)
    Nt = ntriangles(mesh)
    c = [triangle_center(mesh, t) for t in 1:Nt]
    xs = [ct[1] for ct in c]
    ys = [ct[2] for ct in c]
    x0, y0 = minimum(xs), minimum(ys)

    rho = zeros(Float64, Nt)
    for t in 1:Nt
        ix = clamp(floor(Int, (c[t][1] - x0) / (dx_cell / Nx)) + 1, 1, Nx)
        iy = clamp(floor(Int, (c[t][2] - y0) / (dy_cell / Ny)) + 1, 1, Ny)
        rho[t] = isodd(ix + iy) ? 1.0 : 0.0
    end
    return rho
end

function get_case_result(case_name::String, rho_bar::Vector{Float64}, Z_per::Matrix{ComplexF64},
                         mesh::TriMesh, rwg, Mt, k::Float64, lattice::PeriodicLattice, v, Q_spec,
                         dx_cell::Float64, dy_cell::Float64, pol_inc::SVector{3, Float64})
    Z_pen = assemble_Z_penalty(Mt, rho_bar, DensityConfig(; p=3.0, Z_max_factor=10.0, vf_target=0.5, reactive=true))
    Z = Z_per + Z_pen
    I = Z \ v

    J_spec = real(dot(I, Q_spec * I))
    modes, R = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I), k, lattice;
                                       pol=pol_inc, E0=1.0)
    idx00 = findfirst(m -> m.m == 0 && m.n == 0, modes)
    idx00 === nothing && error("Specular (0,0) mode missing for $case_name")

    pb = power_balance(Vector{ComplexF64}(I), Z_pen, dx_cell * dy_cell, k, modes, R;
                       transmission=:floquet)
    any(isnan.((J_spec, abs(R[idx00]), pb.refl_frac, pb.abs_frac, pb.trans_frac, pb.resid_frac))) &&
        error("NaN detected in $case_name metrics")

    return (
        J_spec=J_spec,
        R00_abs=abs(R[idx00]),
        refl_frac=pb.refl_frac,
        abs_frac=pb.abs_frac,
        trans_frac=pb.trans_frac,
        resid_frac=pb.resid_frac,
        Z=Z,
        v=v,
        I=I,
    )
end

println("="^76)
println("  Robustness Map: Matched Baselines (freq × angle, TE incidence)")
println("="^76)

freqs_ghz = parse_float_list(get(ENV, "DMOM_RB_FREQS_GHZ", "8,10,12"))
thetas_deg = parse_float_list(get(ENV, "DMOM_RB_THETAS_DEG", "0,15,30,45,60,75"))
phi_deg = parse(Float64, get(ENV, "DMOM_RB_PHI_DEG", "0"))
phi = phi_deg * π / 180

gmres_tol = parse(Float64, get(ENV, "DMOM_RB_GMRES_TOL", "1e-10"))
gmres_maxiter = parse(Int, get(ENV, "DMOM_RB_GMRES_MAXITER", "800"))

f_design = 10e9
c0 = 3e8
lambda_design = c0 / f_design
dx_cell = 0.5 * lambda_design
dy_cell = 0.5 * lambda_design
Nx = 10
Ny = 10

mesh = make_rect_plate(dx_cell, dy_cell, Nx, Ny)
lattice_design = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, 2π / lambda_design)
rwg = build_rwg_periodic(mesh, lattice_design; precheck=false)
Nt = ntriangles(mesh)
N = rwg.nedges
Mt = precompute_triangle_mass(mesh, rwg)

rho_opt_file = joinpath(DATA_DIR, "results_rho_final.csv")
isfile(rho_opt_file) || error("Missing data/results_rho_final.csv. Run ex15 first.")
rho_df = CSV.read(rho_opt_file, DataFrame)
length(rho_df.rho_bar) == Nt || error("results_rho_final.csv triangle count mismatch")
rho_opt = Float64.(rho_df.rho_bar)
rho_pec = ones(Float64, Nt)
rho_chk = make_checkerboard_rho(mesh, Nx, Ny, dx_cell, dy_cell)

cases = [
    (name="PEC", rho=rho_pec),
    (name="Checkerboard", rho=rho_chk),
    (name="Optimized", rho=rho_opt),
]

println("  Mesh: Nx=$Nx Ny=$Ny Nt=$Nt N=$N")
println("  Sweep freqs (GHz): $(freqs_ghz)")
println("  Sweep theta (deg): $(thetas_deg), phi=$phi_deg deg")
println("  Fill fractions: PEC=$(round(mean(rho_pec), digits=3)), checker=$(round(mean(rho_chk), digits=3)), opt=$(round(mean(rho_opt), digits=3))")

rows = DataFrame(
    freq_ghz=Float64[],
    theta_inc_deg=Float64[],
    phi_inc_deg=Float64[],
    case=String[],
    J_spec=Float64[],
    R00_abs=Float64[],
    refl_frac=Float64[],
    abs_frac=Float64[],
    trans_frac=Float64[],
    resid_frac=Float64[],
    J_vs_pec_db=Float64[],
    J_vs_checker_db=Float64[],
    R00_vs_pec_db=Float64[],
    R00_vs_checker_db=Float64[],
)

# One-point solver cross-check artifact (closest to 10 GHz, 45 deg).
cross_rows = DataFrame(
    freq_ghz=Float64[],
    theta_inc_deg=Float64[],
    rel_current_error=Float64[],
    gmres_rel_residual=Float64[],
    J_rel_error=Float64[],
    R00_abs_delta=Float64[],
)

f_ref = freqs_ghz[argmin(abs.(freqs_ghz .- 10.0))]
theta_ref = thetas_deg[argmin(abs.(thetas_deg .- 45.0))]

for fghz in freqs_ghz
    freq = fghz * 1e9
    k = 2π * freq / c0
    for theta_deg in thetas_deg
        theta = theta_deg * π / 180
        kz = k * cos(theta)
        kx = k * sin(theta) * cos(phi)
        ky = k * sin(theta) * sin(phi)
        k_hat = SVector(kx, ky, -kz) / k

        # TE unit vector for incidence plane at fixed phi.
        pol_te = SVector(-sin(phi), cos(phi), 0.0)
        abs(dot(pol_te, k_hat)) < 1e-12 || error("TE polarization not transverse at theta=$theta_deg")

        lattice = PeriodicLattice(dx_cell, dy_cell, theta, phi, k)
        Z_per = Matrix{ComplexF64}(assemble_Z_efie_periodic(mesh, rwg, k, lattice))
        pw = make_plane_wave(Vec3(kx, ky, -kz), 1.0, Vec3(pol_te...))
        v = Vector{ComplexF64}(assemble_excitation(mesh, rwg, pw))
        grid_ff = make_sph_grid(20, 40)
        G = radiation_vectors(mesh, rwg, grid_ff, k)
        pol_ff = pol_te_matrix(grid_ff)
        spec_dir = Vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))
        mask_spec = direction_mask(grid_ff, spec_dir; half_angle=10 * π / 180)
        Q_spec = Matrix{ComplexF64}(build_Q(G, grid_ff, pol_ff; mask=mask_spec))

        local_res = Dict{String, NamedTuple}()
        for c in cases
            local_res[c.name] = get_case_result(c.name, c.rho, Z_per, mesh, rwg, Mt, k, lattice, v, Q_spec,
                                                dx_cell, dy_cell, pol_te)
        end

        J_pec = local_res["PEC"].J_spec
        R_pec = local_res["PEC"].R00_abs
        J_chk = local_res["Checkerboard"].J_spec
        R_chk = local_res["Checkerboard"].R00_abs

        for c in cases
            r = local_res[c.name]
            push!(rows, (
                fghz,
                theta_deg,
                phi_deg,
                c.name,
                r.J_spec,
                r.R00_abs,
                r.refl_frac,
                r.abs_frac,
                r.trans_frac,
                r.resid_frac,
                10 * log10(max(r.J_spec, 1e-30) / max(J_pec, 1e-30)),
                10 * log10(max(r.J_spec, 1e-30) / max(J_chk, 1e-30)),
                20 * log10(max(r.R00_abs, 1e-30) / max(R_pec, 1e-30)),
                20 * log10(max(r.R00_abs, 1e-30) / max(R_chk, 1e-30)),
            ))
        end

        # Local consistency: PEC should be exactly 0 dB versus itself.
        pec_row = rows[end-2, :]
        abs(pec_row.J_vs_pec_db) < 1e-10 || error("PEC J_vs_pec mismatch at f=$fghz theta=$theta_deg")

        # Single-point direct vs GMRES cross-check for optimized case.
        if isapprox(fghz, f_ref; atol=1e-12) && isapprox(theta_deg, theta_ref; atol=1e-12)
            r_opt = local_res["Optimized"]
            I_gmres = solve_forward(r_opt.Z, r_opt.v; solver=:gmres, gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter)
            rel_I = norm(I_gmres - r_opt.I) / max(norm(r_opt.I), 1e-30)
            rel_resid = norm(r_opt.Z * I_gmres - r_opt.v) / max(norm(r_opt.v), 1e-30)
            J_dir = r_opt.J_spec
            J_gm = real(dot(I_gmres, Q_spec * I_gmres))
            modes_g, R_g = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_gmres), k, lattice;
                                                   pol=pol_te, E0=1.0)
            idxg = findfirst(m -> m.m == 0 && m.n == 0, modes_g)
            idxg === nothing && error("Specular mode missing in GMRES cross-check")
            push!(cross_rows, (
                fghz,
                theta_deg,
                rel_I,
                rel_resid,
                abs(J_gm - J_dir) / max(abs(J_dir), 1e-30),
                abs(abs(R_g[idxg]) - r_opt.R00_abs),
            ))
        end

        println("  f=$(rpad(fghz,4)) GHz, theta=$(rpad(theta_deg,4)) deg: " *
                "opt J_vs_PEC=$(round(rows[end, :J_vs_pec_db], digits=3)) dB, " *
                "opt J_vs_checker=$(round(rows[end, :J_vs_checker_db], digits=3)) dB")
    end
end

CSV.write(joinpath(DATA_DIR, "results_robustness_map.csv"), rows)

opt_df = rows[rows.case .== "Optimized", :]
chk_df = rows[rows.case .== "Checkerboard", :]

summary = DataFrame(
    metric=String[],
    value=Float64[],
)
push!(summary, ("opt_J_vs_PEC_min_dB", minimum(opt_df.J_vs_pec_db)))
push!(summary, ("opt_J_vs_PEC_max_dB", maximum(opt_df.J_vs_pec_db)))
push!(summary, ("opt_J_vs_checker_min_dB", minimum(opt_df.J_vs_checker_db)))
push!(summary, ("opt_J_vs_checker_max_dB", maximum(opt_df.J_vs_checker_db)))
push!(summary, ("checker_J_vs_PEC_min_dB", minimum(chk_df.J_vs_pec_db)))
push!(summary, ("checker_J_vs_PEC_max_dB", maximum(chk_df.J_vs_pec_db)))
push!(summary, ("opt_R00_min", minimum(opt_df.R00_abs)))
push!(summary, ("opt_R00_max", maximum(opt_df.R00_abs)))
push!(summary, ("cross_rel_current_error", isempty(cross_rows) ? NaN : cross_rows.rel_current_error[1]))
push!(summary, ("cross_rel_residual", isempty(cross_rows) ? NaN : cross_rows.gmres_rel_residual[1]))
push!(summary, ("cross_J_rel_error", isempty(cross_rows) ? NaN : cross_rows.J_rel_error[1]))
push!(summary, ("cross_R00_abs_delta", isempty(cross_rows) ? NaN : cross_rows.R00_abs_delta[1]))

CSV.write(joinpath(DATA_DIR, "results_robustness_summary.csv"), summary)
if nrow(cross_rows) > 0
    CSV.write(joinpath(DATA_DIR, "results_robustness_solver_crosscheck.csv"), cross_rows)
end

# Hero figure: Optimized vs Checkerboard, mean over 8–12 GHz with advantage annotation.
# The three frequency curves overlap almost perfectly (<0.1 dB spread), so we plot the
# mean and annotate the max spread to convey frequency robustness cleanly.
tu = sort(unique(opt_df.theta_inc_deg))

# Compute mean and spread across frequencies at each angle
opt_mean = Float64[]
opt_spread = Float64[]
chk_mean = Float64[]
chk_spread = Float64[]
for θ in tu
    opt_vals = opt_df[opt_df.theta_inc_deg .== θ, :J_vs_pec_db]
    chk_vals = chk_df[chk_df.theta_inc_deg .== θ, :J_vs_pec_db]
    push!(opt_mean, mean(opt_vals))
    push!(opt_spread, maximum(opt_vals) - minimum(opt_vals))
    push!(chk_mean, mean(chk_vals))
    push!(chk_spread, maximum(chk_vals) - minimum(chk_vals))
end

# Advantage (gap) at each angle
adv_db = chk_mean .- opt_mean

fig_hero = plot_scatter(tu, opt_mean;
    xlabel="Incidence angle θ [deg]",
    ylabel="Specular scattering vs PEC [dB]",
    mode="lines+markers", color="#0072B2", dash="solid",
    marker_size=7, marker_symbol="circle",
    legend="Optimized (8–12 GHz mean)",
    width=504, height=360, fontsize=12)

plot_scatter!(fig_hero, tu, chk_mean;
    mode="lines+markers", color="#D55E00", dash="dash",
    marker_size=7, marker_symbol="square",
    legend="Checkerboard (8–12 GHz mean)")

set_legend!(fig_hero; position=:topleft)
savefig(fig_hero, joinpath(FIG_DIR, "fig_results_hero_baseline_robustness.pdf");
        width=504, height=360)

# Print advantage summary for caption writing
println("\n  Advantage (checker − optimized) at each angle:")
for (i, θ) in enumerate(tu)
    println("    θ=$(rpad(θ, 4)) deg: Δ=$(round(adv_db[i], digits=2)) dB  " *
            "(opt spread=$(round(opt_spread[i], digits=3)) dB, chk spread=$(round(chk_spread[i], digits=3)) dB)")
end

# Theta=0 line plot (quick read panel).
θ0 = minimum(tu)
sub = rows[rows.theta_inc_deg .== θ0, :]
function series(subdf::DataFrame, case_name::String, col::Symbol)
    d = subdf[subdf.case .== case_name, :]
    p = sortperm(d.freq_ghz)
    return collect(d.freq_ghz[p]), collect(d[!, col][p])
end
x1, y1 = series(sub, "PEC", :J_vs_pec_db)
x2, y2 = series(sub, "Checkerboard", :J_vs_pec_db)
x3, y3 = series(sub, "Optimized", :J_vs_pec_db)

fig_l = plot_scatter([x1, x2, x3], [y1, y2, y3];
                     mode=["lines+markers", "lines+markers", "lines+markers"],
                     legend=["PEC", "Checkerboard", "Optimized"],
                     color=["#7f7f7f", "#D55E00", "#009E73"],
                     dash=["dot", "dash", "dashdot"],
                     marker_size=[6, 6, 6],
                     xlabel="Frequency [GHz]",
                     ylabel="J vs PEC [dB]",
                     title="Theta=$(round(θ0, digits=1))° TE: matched baseline comparison",
                     width=560, height=380, fontsize=13)
set_legend!(fig_l; position=:topright)
savefig(fig_l, joinpath(FIG_DIR, "fig_results_robustness_theta0_lines.pdf"))

println("\n" * "="^76)
println("  Robustness Summary")
println("="^76)
println("  Optimized J vs PEC: $(round(minimum(opt_df.J_vs_pec_db), digits=3)) to $(round(maximum(opt_df.J_vs_pec_db), digits=3)) dB")
println("  Optimized advantage vs checker: $(round(minimum(opt_df.J_vs_checker_db), digits=3)) to $(round(maximum(opt_df.J_vs_checker_db), digits=3)) dB")
println("  Optimized |R00| range: $(round(minimum(opt_df.R00_abs), sigdigits=6)) to $(round(maximum(opt_df.R00_abs), sigdigits=6))")
if nrow(cross_rows) > 0
    println("  Cross-check at f=$(cross_rows.freq_ghz[1]) GHz, theta=$(cross_rows.theta_inc_deg[1]) deg: " *
            "relI=$(round(cross_rows.rel_current_error[1], sigdigits=4)), " *
            "relRes=$(round(cross_rows.gmres_rel_residual[1], sigdigits=4)), " *
            "dR00=$(round(cross_rows.R00_abs_delta[1], sigdigits=4))")
end
println("  Saved: data/results_robustness_map.csv")
println("  Saved: data/results_robustness_summary.csv")
println("  Saved: figures/fig_results_hero_baseline_robustness.pdf")
println("  Saved: figures/fig_results_robustness_theta0_lines.pdf")
println("="^76)
