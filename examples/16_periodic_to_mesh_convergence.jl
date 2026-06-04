# 16_periodic_to_mesh_convergence.jl — Mesh-convergence study for final optimized periodic TO metrics
#
# Re-evaluates the final λ/2-unit-cell optimized density field from examples/15
# on multiple structured meshes covering the same physical unit cell. The mapped
# density is sampled using the exact piecewise-constant triangle field implied by
# make_rect_plate() triangulation, so the geometry and design remain fixed while
# the MoM discretization changes.
#
# Run: julia --project=. examples/16_periodic_to_mesh_convergence.jl

using DiffMoM
using LinearAlgebra
using SparseArrays
using StaticArrays
using Statistics
using CSV, DataFrames
using PlotlySupply
import PlotlySupply: savefig

const PKG_DIR  = dirname(@__DIR__)
const DATA_DIR = joinpath(PKG_DIR, "data")
const FIG_DIR  = joinpath(PKG_DIR, "figures")
mkpath(DATA_DIR)
mkpath(FIG_DIR)

println("=" ^ 70)
println("  Mesh Convergence for Final Periodic TO Metrics (TAP Upgrade)")
println("=" ^ 70)

# Base paper settings (must match examples/15)
freq   = 10e9
c0     = 3e8
lambda = c0 / freq
k      = 2π / lambda
dx_cell = 0.5 * lambda
dy_cell = 0.5 * lambda
eta0 = 376.730313668

Nx_ref = 10
Ny_ref = 10
Nt_ref_expected = 2 * Nx_ref * Ny_ref

rho_file = joinpath(DATA_DIR, "results_rho_final.csv")
isfile(rho_file) || error("Missing $rho_file. Run examples/15_periodic_to_demo.jl first.")
rho_df = CSV.read(rho_file, DataFrame)
length(rho_df.rho_bar) == Nt_ref_expected || error("Expected $(Nt_ref_expected) triangles in results_rho_final.csv")

# Sort defensively by triangle index and store the piecewise-constant final projected density.
perm_ref = sortperm(rho_df.triangle)
rho_ref = Vector{Float64}(rho_df.rho_bar[perm_ref])

# Structured-triangle sampler matching geometry/Mesh.jl:make_rect_plate triangulation.
# Cell-local coordinates (u,v) in [0,1]^2 split along diagonal u=v:
#   tri1=(v1,v2,v3) occupies u>=v ; tri2=(v1,v3,v4) occupies u<v.
function sample_structured_triangle_field(x::Float64, y::Float64,
                                          rho_tri::Vector{Float64},
                                          Lx::Float64, Ly::Float64,
                                          Nx::Int, Ny::Int)
    x_min = -Lx / 2
    y_min = -Ly / 2
    dx = Lx / Nx
    dy = Ly / Ny

    ξx = (x - x_min) / dx
    ξy = (y - y_min) / dy
    jx = clamp(floor(Int, ξx), 0, Nx - 1)
    jy = clamp(floor(Int, ξy), 0, Ny - 1)

    u = clamp(ξx - jx, 0.0, 1.0)
    v = clamp(ξy - jy, 0.0, 1.0)

    tri_local = (u >= v) ? 1 : 2
    t = 2 * (jy * Nx + jx) + tri_local
    return rho_tri[t]
end

function map_density_to_structured_mesh(mesh::TriMesh, rho_tri_ref::Vector{Float64},
                                        Lx::Float64, Ly::Float64,
                                        Nx0::Int, Ny0::Int)
    Nt = ntriangles(mesh)
    out = zeros(Float64, Nt)
    for t in 1:Nt
        c = triangle_center(mesh, t)
        out[t] = sample_structured_triangle_field(c[1], c[2], rho_tri_ref, Lx, Ly, Nx0, Ny0)
    end
    return out
end

# Self-consistency of the mapper on the original mesh: centroids should recover the exact vector.
mesh_ref = make_rect_plate(dx_cell, dy_cell, Nx_ref, Ny_ref)
rho_ref_mapped = map_density_to_structured_mesh(mesh_ref, rho_ref, dx_cell, dy_cell, Nx_ref, Ny_ref)
max_map_err = maximum(abs.(rho_ref_mapped .- rho_ref))
println("  Mapping self-check on reference mesh: max |Δρ̄| = $(max_map_err)")
max_map_err < 1e-14 || error("Structured density mapper failed self-consistency check")

function find_specular_index(modes)
    idx = findfirst(m -> m.m == 0 && m.n == 0, modes)
    idx === nothing && error("Specular Floquet mode (0,0) not found")
    return idx
end

function evaluate_mesh_case(Nx::Int, Ny::Int, rho_ref::Vector{Float64};
                            k::Float64, dx_cell::Float64, dy_cell::Float64)
    mesh = make_rect_plate(dx_cell, dy_cell, Nx, Ny)
    lattice = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k)
    rwg  = build_rwg_periodic(mesh, lattice; precheck=false)
    Nt = ntriangles(mesh)
    N = rwg.nedges

    Z_per = Matrix{ComplexF64}(assemble_Z_efie_periodic(mesh, rwg, k, lattice))
    Mt = precompute_triangle_mass(mesh, rwg)

    pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
    v = Vector{ComplexF64}(assemble_excitation(mesh, rwg, pw))

    grid_ff = make_sph_grid(20, 40)
    Q_spec = Matrix{ComplexF64}(specular_rcs_objective(mesh, rwg, grid_ff, k, lattice;
                                                       half_angle=10 * π / 180,
                                                       polarization=:x))

    config = DensityConfig(; p=3.0, Z_max_factor=10.0, vf_target=0.5, reactive=true)

    # Mapped final optimized density and PEC reference on the same discretization.
    rho_bar_opt = map_density_to_structured_mesh(mesh, rho_ref, dx_cell, dy_cell, Nx_ref, Ny_ref)
    rho_bar_pec = ones(Float64, Nt)

    # Optimized mapped design
    Z_pen_opt = assemble_Z_penalty(Mt, rho_bar_opt, config)
    Z_opt = Z_per + Z_pen_opt
    F_opt = lu(Z_opt)
    I_opt = F_opt \ v
    J_opt = real(dot(I_opt, Q_spec * I_opt))

    pol_inc = SVector(1.0, 0.0, 0.0)
    modes_opt, R_opt = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_opt), k, lattice;
                                               pol=pol_inc, E0=1.0)
    spec_idx = find_specular_index(modes_opt)
    pb_opt = power_balance(Vector{ComplexF64}(I_opt), Z_pen_opt, dx_cell * dy_cell, k, modes_opt, R_opt;
                           transmission=:floquet)

    # PEC reference on this mesh (for mesh-consistent reduction metrics)
    Z_pen_pec = assemble_Z_penalty(Mt, rho_bar_pec, config)
    Z_pec = Z_per + Z_pen_pec
    F_pec = lu(Z_pec)
    I_pec = F_pec \ v
    J_pec = real(dot(I_pec, Q_spec * I_pec))
    modes_pec, R_pec = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_pec), k, lattice;
                                               pol=pol_inc, E0=1.0)
    spec_idx_pec = find_specular_index(modes_pec)
    pb_pec = power_balance(Vector{ComplexF64}(I_pec), Z_pen_pec, dx_cell * dy_cell, k, modes_pec, R_pec;
                           transmission=:floquet)

    return (
        Nx=Nx, Ny=Ny, Nt=Nt, N_rwg=N,
        h_over_lambda=(dx_cell / Nx) / lambda,
        vf_opt=mean(rho_bar_opt),
        J_pec=J_pec, J_opt=J_opt,
        J_reduction_dB=10 * log10(max(J_opt, 1e-30) / max(J_pec, 1e-30)),
        R00_pec=abs(R_pec[spec_idx_pec]), R00_opt=abs(R_opt[spec_idx]),
        R00_amp_dB=20 * log10(max(abs(R_opt[spec_idx]), 1e-30) / max(abs(R_pec[spec_idx_pec]), 1e-30)),
        refl_frac_pec=pb_pec.refl_frac, abs_frac_pec=pb_pec.abs_frac,
        trans_frac_pec=pb_pec.trans_frac, resid_frac_pec=pb_pec.resid_frac,
        refl_frac_opt=pb_opt.refl_frac, abs_frac_opt=pb_opt.abs_frac,
        trans_frac_opt=pb_opt.trans_frac, resid_frac_opt=pb_opt.resid_frac,
        nprop=count(m -> m.propagating, modes_opt),
    )
end

# Chosen meshes around the paper baseline (runtime-conscious but informative).
mesh_sizes = [(8,8), (10,10), (12,12), (14,14), (16,16)]
println("  Evaluating meshes: $(mesh_sizes)")

rows = NamedTuple[]
for (Nx, Ny) in mesh_sizes
    println("\n▸ Mesh $(Nx)×$(Ny)")
    t0 = time()
    row = evaluate_mesh_case(Nx, Ny, rho_ref; k=k, dx_cell=dx_cell, dy_cell=dy_cell)
    push!(rows, row)
    println("  N=$(row.N_rwg), h/λ=$(round(row.h_over_lambda, digits=4)), nprop=$(row.nprop)")
    println("  |R00|=$(round(row.R00_opt, sigdigits=4)), J reduction=$(round(row.J_reduction_dB, digits=2)) dB")
    println("  opt power: refl=$(round(100row.refl_frac_opt,digits=1))%, abs=$(round(100row.abs_frac_opt,digits=1))%, trans=$(round(100row.trans_frac_opt,digits=1))%, resid=$(round(100row.resid_frac_opt,digits=3))%")
    println("  runtime=$(round(time()-t0, digits=1)) s")
end

conv_df = DataFrame(rows)
sort!(conv_df, :N_rwg)

# Convergence metrics relative to the finest mesh in this sweep.
ref_row = conv_df[end, :]
conv_df.J_delta_vs_finest_dB = 10 .* log10.(max.(conv_df.J_opt, 1e-30) ./ ref_row.J_opt)
conv_df.R00_delta_vs_finest_dB = 20 .* log10.(max.(conv_df.R00_opt, 1e-30) ./ ref_row.R00_opt)
conv_df.refl_delta_pctpt = 100 .* (conv_df.refl_frac_opt .- ref_row.refl_frac_opt)
conv_df.abs_delta_pctpt = 100 .* (conv_df.abs_frac_opt .- ref_row.abs_frac_opt)
conv_df.trans_delta_pctpt = 100 .* (conv_df.trans_frac_opt .- ref_row.trans_frac_opt)
conv_df.resid_delta_pctpt = 100 .* (conv_df.resid_frac_opt .- ref_row.resid_frac_opt)

# Reference-row consistency check against paper results from examples/15 (10×10 row).
row10 = conv_df[findfirst(conv_df.Nx .== 10 .&& conv_df.Ny .== 10), :]
floq_file = joinpath(DATA_DIR, "results_floquet_comparison.csv")
pb_file = joinpath(DATA_DIR, "results_power_balance.csv")
trace_file = joinpath(DATA_DIR, "results_optimization_trace.csv")
(isfile(floq_file) && isfile(pb_file) && isfile(trace_file)) || error("Missing one of results_floquet_comparison.csv / results_power_balance.csv / results_optimization_trace.csv; rerun ex15")

floq_df = CSV.read(floq_file, DataFrame)
pb_df = CSV.read(pb_file, DataFrame)
trace_df = CSV.read(trace_file, DataFrame)
idx00 = findfirst((floq_df.m .== 0) .&& (floq_df.n .== 0))
idx00 === nothing && error("(0,0) row missing from results_floquet_comparison.csv")
idx_opt_pb = findfirst(pb_df.case .== "Optimized")
idx_opt_pb === nothing && error("Optimized row missing from results_power_balance.csv")
J_opt_ex15 = Float64(trace_df.J_scatter[end])

checks = Dict(
    "R00_opt" => abs(row10.R00_opt - floq_df.R_opt_abs[idx00]),
    "refl_frac_opt" => abs(row10.refl_frac_opt - pb_df.refl_frac[idx_opt_pb]),
    "abs_frac_opt" => abs(row10.abs_frac_opt - pb_df.abs_frac[idx_opt_pb]),
    "trans_frac_opt" => abs(row10.trans_frac_opt - pb_df.trans_frac[idx_opt_pb]),
    "resid_frac_opt" => abs(row10.resid_frac_opt - pb_df.resid_frac[idx_opt_pb]),
    "J_opt" => abs(row10.J_opt - J_opt_ex15),
)
println("\n▸ Reference-row consistency check vs ex15 outputs")
for (kname, err) in checks
    println("  $kname: |Δ| = $(err)")
end
maximum(values(checks)) < 1e-10 || error("Mesh-convergence 10×10 row failed consistency check against ex15 outputs")

CSV.write(joinpath(DATA_DIR, "results_mesh_convergence.csv"), conv_df)
println("\n  ✓ Saved: data/results_mesh_convergence.csv")

# Plot: headline metric stability vs discretization (important, compact)
fig_conv = plot_scatter(
    [collect(conv_df.N_rwg), collect(conv_df.N_rwg)],
    [collect(conv_df.J_reduction_dB), collect(conv_df.R00_amp_dB)];
    mode=["lines+markers", "lines+markers"],
    legend=["J reduction (dB)", "|R00| amplitude change (dB)"],
    color=["#1f77b4", "#d62728"],
    dash=["solid", "dashdot"],
    marker_size=[8, 8],
    xlabel="RWG edges (N)",
    ylabel="Reduction relative to mesh-consistent PEC (dB)",
    title="Mesh convergence of final optimized headline metrics",
    width=620, height=420, fontsize=14)
set_legend!(fig_conv; position=:bottomleft)
savefig(fig_conv, joinpath(FIG_DIR, "fig_results_mesh_convergence.pdf"))
println("  ✓ Fig: figures/fig_results_mesh_convergence.pdf")

# Supplementary plot: power-fraction convergence (table is primary in paper)
fig_pb = plot_scatter(
    [collect(conv_df.N_rwg), collect(conv_df.N_rwg), collect(conv_df.N_rwg), collect(conv_df.N_rwg)],
    [100 .* collect(conv_df.refl_frac_opt), 100 .* collect(conv_df.abs_frac_opt),
     100 .* collect(conv_df.trans_frac_opt), 100 .* collect(conv_df.resid_frac_opt)];
    mode=["lines+markers", "lines+markers", "lines+markers", "lines+markers"],
    legend=["Reflected (%)", "Absorbed (%)", "Transmitted (%)", "Residual (%)"],
    color=["#1f77b4", "#d62728", "#2ca02c", "#9467bd"],
    dash=["solid", "dash", "dashdot", "dot"],
    marker_size=[8, 8, 8, 8],
    xlabel="RWG edges (N)",
    ylabel="Power fraction (% of incident power)",
    title="Mesh convergence of optimized power-balance fractions (supplementary)",
    width=620, height=420, fontsize=14)
set_legend!(fig_pb; position=:topright)
savefig(fig_pb, joinpath(FIG_DIR, "fig_supp_mesh_convergence_power.pdf"))
println("  ✓ Supp: figures/fig_supp_mesh_convergence_power.pdf")

println("\n" * "=" ^ 70)
println("  Mesh Convergence Summary")
println("=" ^ 70)
println("  Finest mesh in sweep: $(conv_df.Nx[end])×$(conv_df.Ny[end]) (N=$(conv_df.N_rwg[end]))")
println("  J reduction spread across sweep: $(round(maximum(conv_df.J_reduction_dB)-minimum(conv_df.J_reduction_dB), digits=3)) dB")
println("  |R00| amplitude-dB spread across sweep: $(round(maximum(conv_df.R00_amp_dB)-minimum(conv_df.R00_amp_dB), digits=3)) dB")
println("  Max |Δ(refl)| vs finest: $(round(maximum(abs.(conv_df.refl_delta_pctpt)), digits=3)) pct-pt")
println("  Max |Δ(abs)|  vs finest: $(round(maximum(abs.(conv_df.abs_delta_pctpt)), digits=3)) pct-pt")
println("  Max |Δ(trans)| vs finest: $(round(maximum(abs.(conv_df.trans_delta_pctpt)), digits=3)) pct-pt")
println("  Max |Δ(resid)| vs finest: $(round(maximum(abs.(conv_df.resid_delta_pctpt)), digits=3)) pct-pt")
println("=" ^ 70)
