# 19_periodic_to_robustness_map.jl — Frequency/angle robustness map with matched baselines
#
# Purpose:
#   Compare three reference designs under matched frequency, angle, mesh, and
#   polarization settings: PEC, a minimum-feature-compatible checkerboard, and
#   the relaxed density produced by ex15. This is a parameter sweep, not an
#   independent validation of the density model used to generate that design.
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
const DATA_DIR = joinpath(PKG_DIR, "data")
const FIG_DIR = joinpath(PKG_DIR, "figures")
const MIN_REDUCTION_VS_PEC_DB = 10.0
const MIN_ADVANTAGE_VS_CHECKER_DB = 1.0
const MAX_SOLVER_CROSSCHECK_ERROR = 1e-8
mkpath(DATA_DIR)
mkpath(FIG_DIR)

function parse_float_list(s::String, variable::String)
    vals = Float64[]
    for part in split(s, ",")
        t = strip(part)
        isempty(t) && error("$variable contains an empty list element")
        value = tryparse(Float64, t)
        value === nothing && error(
            "$variable must be a comma-separated numeric list; could not parse '$t'")
        isfinite(value) || error("$variable values must be finite; got '$t'")
        push!(vals, value)
    end
    isempty(vals) && error("$variable must contain at least one numeric value")
    allunique(vals) || error("$variable values must be unique; got '$s'")
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
    metrics = (J_spec, abs(R[idx00]), pb.refl_frac, pb.abs_frac,
               pb.trans_frac, pb.resid_frac)
    all(isfinite, metrics) || error(
        "Non-finite result for '$case_name'; inspect the assembled system and incident-wave settings")

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

freqs_ghz = parse_float_list(get(ENV, "DMOM_RB_FREQS_GHZ", "8,10,12"),
                             "DMOM_RB_FREQS_GHZ")
thetas_deg = parse_float_list(get(ENV, "DMOM_RB_THETAS_DEG", "0,15,30,45,60,75"),
                              "DMOM_RB_THETAS_DEG")
phi_text = get(ENV, "DMOM_RB_PHI_DEG", "0")
phi_deg = tryparse(Float64, phi_text)
phi_deg === nothing && error("DMOM_RB_PHI_DEG must be numeric; got '$phi_text'")
phi = phi_deg * π / 180

gmres_tol_text = get(ENV, "DMOM_RB_GMRES_TOL", "1e-10")
gmres_tol = tryparse(Float64, gmres_tol_text)
gmres_tol === nothing && error("DMOM_RB_GMRES_TOL must be numeric; got '$gmres_tol_text'")
gmres_maxiter_text = get(ENV, "DMOM_RB_GMRES_MAXITER", "800")
gmres_maxiter = tryparse(Int, gmres_maxiter_text)
gmres_maxiter === nothing && error(
    "DMOM_RB_GMRES_MAXITER must be an integer; got '$gmres_maxiter_text'")

all(>(0), freqs_ghz) || error("DMOM_RB_FREQS_GHZ values must be greater than zero")
all(theta -> 0 <= theta < 90, thetas_deg) || error(
    "DMOM_RB_THETAS_DEG values must be in [0, 90) degrees")
isfinite(phi_deg) || error("DMOM_RB_PHI_DEG must be finite")
(isfinite(gmres_tol) && 0 < gmres_tol < 1) || error(
    "DMOM_RB_GMRES_TOL must be finite and between zero and one")
gmres_maxiter > 0 || error("DMOM_RB_GMRES_MAXITER must be greater than zero")

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

rho_relaxed_file = joinpath(DATA_DIR, "results_rho_final.csv")
isfile(rho_relaxed_file) || error(
    "Missing '$rho_relaxed_file'. Generate it with " *
    "`julia --project=. examples/15_periodic_to_demo.jl`, then rerun this example")
rho_df = CSV.read(rho_relaxed_file, DataFrame)
hasproperty(rho_df, :rho_bar) || error(
    "'$rho_relaxed_file' has no 'rho_bar' column; regenerate it with example 15")
length(rho_df.rho_bar) == Nt || error(
    "'$rho_relaxed_file' has $(length(rho_df.rho_bar)) rows, but this $(Nx)×$(Ny) mesh " *
    "has $Nt triangles; regenerate it with the default example 15 mesh")
rho_relaxed = Float64.(rho_df.rho_bar)
all(x -> isfinite(x) && 0 <= x <= 1, rho_relaxed) || error(
    "'$rho_relaxed_file' contains a non-finite density or a value outside [0, 1]; " *
    "regenerate it with example 15")
rho_pec = ones(Float64, Nt)

# Match example 15's minimum-feature radius. A checkerboard with one square per
# mesh cell would have smaller features than the relaxed design is allowed to use.
minimum_feature = 2.5 * max(dx_cell / Nx, dy_cell / Ny)
checker_tiles_x = 2 * fld(floor(Int, dx_cell / minimum_feature), 2)
checker_tiles_y = 2 * fld(floor(Int, dy_cell / minimum_feature), 2)
(checker_tiles_x >= 2 && checker_tiles_y >= 2) || error(
    "The $(Nx)×$(Ny) mesh cannot represent an even checkerboard with feature " *
    "width at least $(round(minimum_feature / lambda_design, digits=3))λ")
checker_case_name = "Feasible $(checker_tiles_x)×$(checker_tiles_y) checker"
rho_chk = make_checkerboard_rho(mesh, checker_tiles_x, checker_tiles_y,
                                dx_cell, dy_cell)

cases = [
    (name="PEC", rho=rho_pec),
    (name=checker_case_name, rho=rho_chk),
    (name="Relaxed density", rho=rho_relaxed),
]

println("  Mesh: Nx=$Nx Ny=$Ny Nt=$Nt N=$N")
println("  Sweep freqs (GHz): $(freqs_ghz)")
println("  Sweep theta (deg): $(thetas_deg), phi=$phi_deg deg")
println("  Reference checker: $checker_case_name (feature width ≥ $(round(minimum_feature / lambda_design, digits=3))λ)")
println("  Fill fractions: PEC=$(round(mean(rho_pec), digits=3)), " *
        "checker=$(round(mean(rho_chk), digits=3)), relaxed=$(round(mean(rho_relaxed), digits=3))")

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
        J_chk = local_res[checker_case_name].J_spec
        R_chk = local_res[checker_case_name].R00_abs

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

        # Single-point direct vs GMRES cross-check for the relaxed-density case.
        if isapprox(fghz, f_ref; atol=1e-12) && isapprox(theta_deg, theta_ref; atol=1e-12)
            r_relaxed = local_res["Relaxed density"]
            I_gmres = solve_forward(r_relaxed.Z, r_relaxed.v; solver=:gmres,
                                    gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter)
            rel_I = norm(I_gmres - r_relaxed.I) / max(norm(r_relaxed.I), 1e-30)
            rel_resid = norm(r_relaxed.Z * I_gmres - r_relaxed.v) /
                        max(norm(r_relaxed.v), 1e-30)
            J_dir = r_relaxed.J_spec
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
                abs(abs(R_g[idxg]) - r_relaxed.R00_abs),
            ))
        end

        println("  f=$(rpad(fghz,4)) GHz, theta=$(rpad(theta_deg,4)) deg: " *
                "relaxed J_vs_PEC=$(round(rows[end, :J_vs_pec_db], digits=3)) dB, " *
                "relaxed J_vs_checker=$(round(rows[end, :J_vs_checker_db], digits=3)) dB")
    end
end

relaxed_df = rows[rows.case .== "Relaxed density", :]
chk_df = rows[rows.case .== checker_case_name, :]

expected_case_rows = length(freqs_ghz) * length(thetas_deg)
nrow(relaxed_df) == expected_case_rows || error(
    "Relaxed-density sweep produced $(nrow(relaxed_df)) rows; expected $expected_case_rows")
nrow(chk_df) == expected_case_rows || error(
    "Checker sweep produced $(nrow(chk_df)) rows; expected $expected_case_rows")
nrow(cross_rows) == 1 || error(
    "Expected one direct/GMRES cross-check row, but produced $(nrow(cross_rows))")

numeric_columns = (
    :freq_ghz, :theta_inc_deg, :phi_inc_deg, :J_spec, :R00_abs,
    :refl_frac, :abs_frac, :trans_frac, :resid_frac,
    :J_vs_pec_db, :J_vs_checker_db, :R00_vs_pec_db, :R00_vs_checker_db,
)
all(column -> all(isfinite, rows[!, column]), numeric_columns) || error(
    "The robustness sweep produced a non-finite metric; inspect the output point printed last")
all(>=(0), rows.J_spec) || error("The robustness sweep produced a negative J_spec value")
all(>=(0), rows.R00_abs) || error("The robustness sweep produced a negative |R00| value")

worst_reduction_vs_pec_db = -maximum(relaxed_df.J_vs_pec_db)
minimum_advantage_vs_checker_db = minimum(-relaxed_df.J_vs_checker_db)
crosscheck_errors = (
    cross_rows.rel_current_error[1],
    cross_rows.gmres_rel_residual[1],
    cross_rows.J_rel_error[1],
    cross_rows.R00_abs_delta[1],
)
all(isfinite, crosscheck_errors) || error(
    "The direct/GMRES cross-check produced a non-finite error")
maximum(crosscheck_errors) <= MAX_SOLVER_CROSSCHECK_ERROR || error(
    "Direct/GMRES cross-check error $(maximum(crosscheck_errors)) exceeds " *
    "$MAX_SOLVER_CROSSCHECK_ERROR; tighten DMOM_RB_GMRES_TOL or increase " *
    "DMOM_RB_GMRES_MAXITER")
worst_reduction_vs_pec_db >= MIN_REDUCTION_VS_PEC_DB || error(
    "Relaxed density reduces J by only $(round(worst_reduction_vs_pec_db, digits=3)) dB " *
    "at the weakest sweep point; expected at least $MIN_REDUCTION_VS_PEC_DB dB. " *
    "Regenerate data/results_rho_final.csv with example 15")
minimum_advantage_vs_checker_db >= MIN_ADVANTAGE_VS_CHECKER_DB || error(
    "Relaxed density beats the feasible checker by only " *
    "$(round(minimum_advantage_vs_checker_db, digits=3)) dB at the weakest sweep point; " *
    "expected at least $MIN_ADVANTAGE_VS_CHECKER_DB dB")

summary = DataFrame(
    metric=String[],
    value=Float64[],
)
push!(summary, ("relaxed_J_vs_PEC_min_dB", minimum(relaxed_df.J_vs_pec_db)))
push!(summary, ("relaxed_J_vs_PEC_max_dB", maximum(relaxed_df.J_vs_pec_db)))
push!(summary, ("relaxed_J_vs_checker_min_dB", minimum(relaxed_df.J_vs_checker_db)))
push!(summary, ("relaxed_J_vs_checker_max_dB", maximum(relaxed_df.J_vs_checker_db)))
push!(summary, ("checker_J_vs_PEC_min_dB", minimum(chk_df.J_vs_pec_db)))
push!(summary, ("checker_J_vs_PEC_max_dB", maximum(chk_df.J_vs_pec_db)))
push!(summary, ("relaxed_R00_min", minimum(relaxed_df.R00_abs)))
push!(summary, ("relaxed_R00_max", maximum(relaxed_df.R00_abs)))
push!(summary, ("cross_rel_current_error", cross_rows.rel_current_error[1]))
push!(summary, ("cross_rel_residual", cross_rows.gmres_rel_residual[1]))
push!(summary, ("cross_J_rel_error", cross_rows.J_rel_error[1]))
push!(summary, ("cross_R00_abs_delta", cross_rows.R00_abs_delta[1]))
push!(summary, ("worst_reduction_vs_PEC_dB", worst_reduction_vs_pec_db))
push!(summary, ("minimum_advantage_vs_checker_dB", minimum_advantage_vs_checker_db))
push!(summary, ("maximum_solver_crosscheck_error", maximum(crosscheck_errors)))

CSV.write(joinpath(DATA_DIR, "results_robustness_map.csv"), rows)
CSV.write(joinpath(DATA_DIR, "results_robustness_summary.csv"), summary)
CSV.write(joinpath(DATA_DIR, "results_robustness_solver_crosscheck.csv"), cross_rows)

# Hero figure: relaxed density vs feasible checker, averaged over the requested
# frequencies. The console reports the observed frequency spread at each angle.
tu = sort(unique(relaxed_df.theta_inc_deg))

# Compute mean and spread across frequencies at each angle
relaxed_mean = Float64[]
relaxed_spread = Float64[]
chk_mean = Float64[]
chk_spread = Float64[]
for θ in tu
    relaxed_vals = relaxed_df[relaxed_df.theta_inc_deg .== θ, :J_vs_pec_db]
    chk_vals = chk_df[chk_df.theta_inc_deg .== θ, :J_vs_pec_db]
    push!(relaxed_mean, mean(relaxed_vals))
    push!(relaxed_spread, maximum(relaxed_vals) - minimum(relaxed_vals))
    push!(chk_mean, mean(chk_vals))
    push!(chk_spread, maximum(chk_vals) - minimum(chk_vals))
end

# Advantage (gap) at each angle
adv_db = chk_mean .- relaxed_mean

fig_hero = plot_scatter(tu, relaxed_mean;
    xlabel="Incidence angle θ [deg]",
    ylabel="Specular scattering vs PEC [dB]",
    mode="lines+markers", color="#0072B2", dash="solid",
    marker_size=7, marker_symbol="circle",
    legend="Relaxed density (frequency mean)",
    width=504, height=360, fontsize=12)

plot_scatter!(fig_hero, tu, chk_mean;
    mode="lines+markers", color="#D55E00", dash="dash",
    marker_size=7, marker_symbol="square",
    legend="Feasible checker (frequency mean)")

set_legend!(fig_hero; position=:topleft)
savefig(fig_hero, joinpath(FIG_DIR, "fig_results_hero_baseline_robustness.pdf");
        width=504, height=360)

# Print advantage summary for caption writing
println("\n  Advantage (checker − relaxed density) at each angle:")
for (i, θ) in enumerate(tu)
    println("    θ=$(rpad(θ, 4)) deg: Δ=$(round(adv_db[i], digits=2)) dB  " *
            "(relaxed spread=$(round(relaxed_spread[i], digits=3)) dB, " *
            "checker spread=$(round(chk_spread[i], digits=3)) dB)")
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
x2, y2 = series(sub, checker_case_name, :J_vs_pec_db)
x3, y3 = series(sub, "Relaxed density", :J_vs_pec_db)

fig_l = plot_scatter([x1, x2, x3], [y1, y2, y3];
                     mode=["lines+markers", "lines+markers", "lines+markers"],
                     legend=["PEC", "Feasible checker", "Relaxed density"],
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
println("  Relaxed-density J relative to PEC: " *
        "$(round(minimum(relaxed_df.J_vs_pec_db), digits=3)) to " *
        "$(round(maximum(relaxed_df.J_vs_pec_db), digits=3)) dB")
println("  Weakest reduction relative to PEC: " *
        "$(round(worst_reduction_vs_pec_db, digits=3)) dB")
println("  Weakest advantage over feasible checker: " *
        "$(round(minimum_advantage_vs_checker_db, digits=3)) dB")
println("  Relaxed-density |R00| range: " *
        "$(round(minimum(relaxed_df.R00_abs), sigdigits=6)) to " *
        "$(round(maximum(relaxed_df.R00_abs), sigdigits=6))")
println("  Direct/GMRES cross-check at f=$(cross_rows.freq_ghz[1]) GHz, " *
        "theta=$(cross_rows.theta_inc_deg[1]) deg: " *
        "relI=$(round(cross_rows.rel_current_error[1], sigdigits=4)), " *
        "relRes=$(round(cross_rows.gmres_rel_residual[1], sigdigits=4)), " *
        "dR00=$(round(cross_rows.R00_abs_delta[1], sigdigits=4))")
println("  PASS: reduction, matched-baseline, finiteness, and solver cross-check gates")
println("  Saved: $(joinpath(DATA_DIR, "results_robustness_map.csv"))")
println("  Saved: $(joinpath(DATA_DIR, "results_robustness_summary.csv"))")
println("  Saved: $(joinpath(DATA_DIR, "results_robustness_solver_crosscheck.csv"))")
println("  Saved: $(joinpath(FIG_DIR, "fig_results_hero_baseline_robustness.pdf"))")
println("  Saved: $(joinpath(FIG_DIR, "fig_results_robustness_theta0_lines.pdf"))")
println("="^76)
