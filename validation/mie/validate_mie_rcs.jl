# validate_mie_rcs.jl — PEC sphere RCS: MoM vs Mie analytical reference
#
# This validation script:
#   1. Builds an icosphere mesh and exports to STL
#   2. Reads the STL mesh back
#   3. Solves the MoM scattering problem (dense EFIE)
#   4. Computes bistatic RCS on phi=0 and phi=90 cuts
#   5. Compares against exact Mie series
#   6. Saves CSV data and PlotlySupply plots
#
# Run: julia --project=. validation/mie/validate_mie_rcs.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
include(joinpath(@__DIR__, "..", "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using PlotlySupply
import PlotlyKaleido
PlotlyKaleido.start()

# ── Icosphere generator ─────────────────────────────
function make_icosphere(radius::Float64; subdivisions::Int=2)
    phi_gold = (1 + sqrt(5.0)) / 2
    verts0 = [
        (-1.0,  phi_gold, 0.0), ( 1.0,  phi_gold, 0.0),
        (-1.0, -phi_gold, 0.0), ( 1.0, -phi_gold, 0.0),
        ( 0.0, -1.0, phi_gold), ( 0.0,  1.0, phi_gold),
        ( 0.0, -1.0,-phi_gold), ( 0.0,  1.0,-phi_gold),
        ( phi_gold, 0.0, -1.0), ( phi_gold, 0.0,  1.0),
        (-phi_gold, 0.0, -1.0), (-phi_gold, 0.0,  1.0),
    ]
    faces = [
        (1,12,6), (1,6,2), (1,2,8), (1,8,11), (1,11,12),
        (2,6,10), (6,12,5), (12,11,3), (11,8,7), (8,2,9),
        (4,10,5), (4,5,3), (4,3,7), (4,7,9), (4,9,10),
        (5,10,6), (3,5,12), (7,3,11), (9,7,8), (10,9,2),
    ]
    verts = [Vec3(v...) / norm(Vec3(v...)) for v in verts0]

    for _ in 1:subdivisions
        edge_mid = Dict{Tuple{Int,Int},Int}()
        new_faces = NTuple{3,Int}[]
        function midpoint_index(i::Int, j::Int)
            key = i < j ? (i, j) : (j, i)
            haskey(edge_mid, key) && return edge_mid[key]
            vmid = (verts[i] + verts[j]) / 2
            vmid /= norm(vmid)
            push!(verts, vmid)
            edge_mid[key] = length(verts)
            return length(verts)
        end
        for (i, j, k) in faces
            a = midpoint_index(i, j)
            b = midpoint_index(j, k)
            c = midpoint_index(k, i)
            push!(new_faces, (i, a, c))
            push!(new_faces, (j, b, a))
            push!(new_faces, (k, c, b))
            push!(new_faces, (a, b, c))
        end
        faces = new_faces
    end

    Nv = length(verts)
    Nt = length(faces)
    xyz = zeros(3, Nv)
    tri = zeros(Int, 3, Nt)
    for i in 1:Nv
        xyz[:, i] = radius .* verts[i]
    end
    for t in 1:Nt
        tri[1, t] = faces[t][1]
        tri[2, t] = faces[t][2]
        tri[3, t] = faces[t][3]
    end
    return TriMesh(xyz, tri)
end

println("="^70)
println("Mie Validation: PEC Sphere Bistatic RCS — MoM vs Mie Series")
println("="^70)

# ── 1. Problem parameters ─────────────────────────────
a     = 0.05                            # sphere radius = 5 cm
freq  = 2e9                             # 2 GHz
c0    = 299792458.0
lambda0 = c0 / freq
k     = 2π / lambda0
eta0  = 376.730313668
ka    = k * a

println("\nRadius:    $(a*100) cm")
println("Frequency: $(freq/1e9) GHz")
println("Lambda:    $(round(lambda0*100, digits=2)) cm")
println("ka:        $(round(ka, digits=3))")

# ── 2. Build icosphere mesh and export to STL ─────────
subdiv = 3
mesh_orig = make_icosphere(a; subdivisions=subdiv)
println("\nOriginal mesh: $(nvertices(mesh_orig)) vertices, $(ntriangles(mesh_orig)) triangles")

stl_path = joinpath(@__DIR__, "sphere_ka$(round(ka, digits=2)).stl")
write_stl_mesh(stl_path, mesh_orig; header="PEC sphere a=$(a)m ka=$(round(ka, digits=2))")
println("Exported to STL: $stl_path")

# ── 3. Import STL mesh back ──────────────────────────
mesh = read_stl_mesh(stl_path)
report = assert_mesh_quality(mesh; allow_boundary=false, require_closed=true)
println("Re-imported STL: $(nvertices(mesh)) vertices, $(ntriangles(mesh)) triangles")
println("Quality: OK (closed surface, 0 boundary edges)")

res = mesh_resolution_report(mesh, freq)
println("Edge max/lambda: $(round(res.edge_max_over_lambda, digits=3))  (target <= 0.1)")

# ── 4. Build RWG and assemble EFIE ───────────────────
rwg = build_rwg(mesh)
N = rwg.nedges
println("\nRWG basis functions: $N")
println("Estimated memory: $(round(estimate_dense_matrix_gib(N)*1024, digits=1)) MiB")

println("Assembling Z_efie ($N x $N)...")
t_asm = @elapsed Z = assemble_Z_efie(mesh, rwg, k)
println("  Done in $(round(t_asm, digits=2)) s")

# ── 5. Plane-wave excitation and solve ────────────────
# Normal incidence: wave propagating in -z, x-polarized
k_vec = Vec3(0.0, 0.0, -k)
khat  = Vec3(0.0, 0.0, -1.0)
pol   = Vec3(1.0, 0.0, 0.0)
E0    = 1.0
v_exc = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol)

println("Solving Z I = v...")
t_sol = @elapsed I_pec = Z \ v_exc
residual = norm(Z * I_pec - v_exc) / norm(v_exc)
println("  Relative residual: $(round(residual, sigdigits=3))")

# ── 6. Far-field computation (fine cuts) ──────────────
Ntheta_cut = 360
theta_cut = range(0, π, length=Ntheta_cut+1)[1:end-1] .+ π/(2*Ntheta_cut)
theta_deg = rad2deg.(theta_cut)

# Helper: build a single phi-cut grid
function build_phi_cut_grid(theta_vals, phi_val)
    Nth = length(theta_vals)
    rhat = zeros(3, Nth)
    theta_vec = zeros(Nth)
    phi_vec = zeros(Nth)
    w_vec = zeros(Nth)
    dtheta = π / Nth
    for i in 1:Nth
        theta_vec[i] = theta_vals[i]
        phi_vec[i] = phi_val
        rhat[1, i] = sin(theta_vals[i]) * cos(phi_val)
        rhat[2, i] = sin(theta_vals[i]) * sin(phi_val)
        rhat[3, i] = cos(theta_vals[i])
        w_vec[i] = sin(theta_vals[i]) * dtheta * 2π
    end
    return SphGrid(rhat, theta_vec, phi_vec, w_vec)
end

# ── 6a. Phi = 0 cut (E-plane for x-pol) ──────────────
println("\nComputing far-field phi=0 cut ($Ntheta_cut directions)...")
grid_phi0 = build_phi_cut_grid(theta_cut, 0.0)
G_phi0 = radiation_vectors(mesh, rwg, grid_phi0, k)
E_ff_phi0 = compute_farfield(G_phi0, Vector{ComplexF64}(I_pec), Ntheta_cut)
sigma_mom_phi0 = bistatic_rcs(E_ff_phi0; E0=E0)

# ── 6b. Phi = 90 cut (H-plane for x-pol) ─────────────
println("Computing far-field phi=90 cut ($Ntheta_cut directions)...")
grid_phi90 = build_phi_cut_grid(theta_cut, π/2)
G_phi90 = radiation_vectors(mesh, rwg, grid_phi90, k)
E_ff_phi90 = compute_farfield(G_phi90, Vector{ComplexF64}(I_pec), Ntheta_cut)
sigma_mom_phi90 = bistatic_rcs(E_ff_phi90; E0=E0)

# ── 7. Mie reference ─────────────────────────────────
println("Computing Mie series reference...")
sigma_mie_phi0  = zeros(Ntheta_cut)
sigma_mie_phi90 = zeros(Ntheta_cut)

for i in 1:Ntheta_cut
    rhat_phi0  = Vec3(sin(theta_cut[i]), 0.0, cos(theta_cut[i]))
    rhat_phi90 = Vec3(0.0, sin(theta_cut[i]), cos(theta_cut[i]))
    sigma_mie_phi0[i]  = mie_bistatic_rcs_pec(k, a, khat, pol, rhat_phi0)
    sigma_mie_phi90[i] = mie_bistatic_rcs_pec(k, a, khat, pol, rhat_phi90)
end

# ── 8. Comparison metrics ────────────────────────────
dB_mom_phi0  = 10 .* log10.(max.(sigma_mom_phi0, 1e-30))
dB_mie_phi0  = 10 .* log10.(max.(sigma_mie_phi0, 1e-30))
dB_mom_phi90 = 10 .* log10.(max.(sigma_mom_phi90, 1e-30))
dB_mie_phi90 = 10 .* log10.(max.(sigma_mie_phi90, 1e-30))

delta_phi0  = dB_mom_phi0  .- dB_mie_phi0
delta_phi90 = dB_mom_phi90 .- dB_mie_phi90

println("\n── Phi = 0 (E-plane) ──")
println("  MAE:     $(round(mean(abs.(delta_phi0)), digits=3)) dB")
println("  RMSE:    $(round(sqrt(mean(abs2, delta_phi0)), digits=3)) dB")
println("  Max |Δ|: $(round(maximum(abs.(delta_phi0)), digits=3)) dB")

println("\n── Phi = 90 (H-plane) ──")
println("  MAE:     $(round(mean(abs.(delta_phi90)), digits=3)) dB")
println("  RMSE:    $(round(sqrt(mean(abs2, delta_phi90)), digits=3)) dB")
println("  Max |Δ|: $(round(maximum(abs.(delta_phi90)), digits=3)) dB")

# Backscatter (theta = 180)
idx_back = argmin(abs.(theta_cut .- π))
println("\n── Backscatter (theta ≈ 180°) ──")
println("  MoM phi=0:  $(round(dB_mom_phi0[idx_back], digits=2)) dBsm")
println("  Mie phi=0:  $(round(dB_mie_phi0[idx_back], digits=2)) dBsm")
println("  Delta:      $(round(delta_phi0[idx_back], digits=2)) dB")

# Forward scatter (theta ≈ 0)
idx_fwd = argmin(abs.(theta_cut))
println("\n── Forward scatter (theta ≈ 0°) ──")
println("  MoM phi=0:  $(round(dB_mom_phi0[idx_fwd], digits=2)) dBsm")
println("  Mie phi=0:  $(round(dB_mie_phi0[idx_fwd], digits=2)) dBsm")
println("  Delta:      $(round(delta_phi0[idx_fwd], digits=2)) dB")

# Energy conservation (full sphere grid)
println("\nComputing energy conservation (full sphere)...")
grid_full = make_sph_grid(90, 72)
G_full = radiation_vectors(mesh, rwg, grid_full, k)
E_ff_full = compute_farfield(G_full, Vector{ComplexF64}(I_pec), length(grid_full.w))
P_in  = input_power(Vector{ComplexF64}(I_pec), Vector{ComplexF64}(v_exc))
P_rad = radiated_power(E_ff_full, grid_full)
println("  P_rad/P_in = $(round(P_rad / P_in, digits=4))  (should be ≈ 1 for PEC)")

# ── 9. Save CSV data ─────────────────────────────────
datadir = @__DIR__
df_phi0 = DataFrame(
    theta_deg = theta_deg,
    rcs_mom_dBsm = dB_mom_phi0,
    rcs_mie_dBsm = dB_mie_phi0,
    delta_dB = delta_phi0,
    rcs_mom_m2 = sigma_mom_phi0,
    rcs_mie_m2 = sigma_mie_phi0,
)
CSV.write(joinpath(datadir, "mie_rcs_phi0.csv"), df_phi0)

df_phi90 = DataFrame(
    theta_deg = theta_deg,
    rcs_mom_dBsm = dB_mom_phi90,
    rcs_mie_dBsm = dB_mie_phi90,
    delta_dB = delta_phi90,
    rcs_mom_m2 = sigma_mom_phi90,
    rcs_mie_m2 = sigma_mie_phi90,
)
CSV.write(joinpath(datadir, "mie_rcs_phi90.csv"), df_phi90)

df_summary = DataFrame(
    metric = [
        "radius_m", "freq_GHz", "lambda_m", "ka",
        "subdivisions", "N_rwg", "N_triangles",
        "phi0_mae_dB", "phi0_rmse_dB", "phi0_max_delta_dB",
        "phi90_mae_dB", "phi90_rmse_dB", "phi90_max_delta_dB",
        "backscatter_mom_dBsm", "backscatter_mie_dBsm", "backscatter_delta_dB",
        "P_rad_over_P_in",
    ],
    value = [
        a, freq/1e9, lambda0, ka,
        subdiv, N, ntriangles(mesh),
        mean(abs.(delta_phi0)), sqrt(mean(abs2, delta_phi0)), maximum(abs.(delta_phi0)),
        mean(abs.(delta_phi90)), sqrt(mean(abs2, delta_phi90)), maximum(abs.(delta_phi90)),
        dB_mom_phi0[idx_back], dB_mie_phi0[idx_back], delta_phi0[idx_back],
        P_rad / P_in,
    ],
)
CSV.write(joinpath(datadir, "mie_rcs_summary.csv"), df_summary)
println("\nCSV data saved to $datadir")

# ── 10. Plots ─────────────────────────────────────────
figdir = joinpath(@__DIR__, "figs")
mkpath(figdir)

ka_str = round(ka, digits=2)

# Plot 1: Phi = 0 (E-plane)
title_phi0 = "PEC Sphere ka=$(ka_str) — Bistatic RCS, E-plane (phi=0°)"
sf0 = subplots(1, 1; sync=false, width=900, height=550,
               subplot_titles=reshape([title_phi0], 1, 1))
addtraces!(sf0, scatter(x=theta_deg, y=dB_mie_phi0, mode="lines",
           name="Mie (exact)", line=attr(color="blue", width=2)); row=1, col=1)
addtraces!(sf0, scatter(x=theta_deg, y=dB_mom_phi0, mode="lines",
           name="MoM (N=$N)", line=attr(color="red", width=2, dash="dash")); row=1, col=1)
p0 = sf0.plot
relayout!(p0, xaxis=attr(title="theta (deg)", range=[0, 180], dtick=30),
          yaxis=attr(title="Bistatic RCS (dBsm)"),
          legend=attr(x=0.60, y=0.95),
          margin=attr(l=60, r=30, t=60, b=50))
fig0_path = joinpath(figdir, "mie_rcs_phi0.png")
PlotlyKaleido.savefig(p0, fig0_path; width=900, height=550)
println("Plot saved: $fig0_path")

# Plot 2: Phi = 90 (H-plane)
title_phi90 = "PEC Sphere ka=$(ka_str) — Bistatic RCS, H-plane (phi=90°)"
sf90 = subplots(1, 1; sync=false, width=900, height=550,
                subplot_titles=reshape([title_phi90], 1, 1))
addtraces!(sf90, scatter(x=theta_deg, y=dB_mie_phi90, mode="lines",
            name="Mie (exact)", line=attr(color="blue", width=2)); row=1, col=1)
addtraces!(sf90, scatter(x=theta_deg, y=dB_mom_phi90, mode="lines",
            name="MoM (N=$N)", line=attr(color="red", width=2, dash="dash")); row=1, col=1)
p90 = sf90.plot
relayout!(p90, xaxis=attr(title="theta (deg)", range=[0, 180], dtick=30),
          yaxis=attr(title="Bistatic RCS (dBsm)"),
          legend=attr(x=0.60, y=0.95),
          margin=attr(l=60, r=30, t=60, b=50))
fig90_path = joinpath(figdir, "mie_rcs_phi90.png")
PlotlyKaleido.savefig(p90, fig90_path; width=900, height=550)
println("Plot saved: $fig90_path")

# Plot 3: Both cuts overlaid
title_both = "PEC Sphere ka=$(ka_str) — Bistatic RCS ($(freq/1e9) GHz, N=$N)"
sf_both = subplots(1, 1; sync=false, width=900, height=550,
                   subplot_titles=reshape([title_both], 1, 1))
addtraces!(sf_both, scatter(x=theta_deg, y=dB_mie_phi0, mode="lines",
           name="Mie E-plane", line=attr(color="blue", width=2)); row=1, col=1)
addtraces!(sf_both, scatter(x=theta_deg, y=dB_mom_phi0, mode="lines",
           name="MoM E-plane", line=attr(color="red", width=2, dash="dash")); row=1, col=1)
addtraces!(sf_both, scatter(x=theta_deg, y=dB_mie_phi90, mode="lines",
           name="Mie H-plane", line=attr(color="green", width=2)); row=1, col=1)
addtraces!(sf_both, scatter(x=theta_deg, y=dB_mom_phi90, mode="lines",
           name="MoM H-plane", line=attr(color="orange", width=2, dash="dash")); row=1, col=1)
p_both = sf_both.plot
relayout!(p_both, xaxis=attr(title="theta (deg)", range=[0, 180], dtick=30),
          yaxis=attr(title="Bistatic RCS (dBsm)"),
          legend=attr(x=0.55, y=0.95),
          margin=attr(l=60, r=30, t=60, b=50))
fig_both_path = joinpath(figdir, "mie_rcs_both_cuts.png")
PlotlyKaleido.savefig(p_both, fig_both_path; width=900, height=550)
println("Plot saved: $fig_both_path")

# Plot 4: Error (delta dB) for both cuts
title_err = "PEC Sphere ka=$(ka_str) — MoM vs Mie Error"
sf_err = subplots(1, 1; sync=false, width=900, height=550,
                  subplot_titles=reshape([title_err], 1, 1))
addtraces!(sf_err, scatter(x=theta_deg, y=delta_phi0, mode="lines",
           name="E-plane (phi=0)", line=attr(color="red", width=2)); row=1, col=1)
addtraces!(sf_err, scatter(x=theta_deg, y=delta_phi90, mode="lines",
           name="H-plane (phi=90)", line=attr(color="blue", width=2)); row=1, col=1)
addtraces!(sf_err, scatter(x=[0, 180], y=[0, 0], mode="lines",
           name="zero", line=attr(color="gray", width=1, dash="dot"),
           showlegend=false); row=1, col=1)
p_err = sf_err.plot
relayout!(p_err, xaxis=attr(title="theta (deg)", range=[0, 180], dtick=30),
          yaxis=attr(title="MoM - Mie (dB)"),
          legend=attr(x=0.60, y=0.95),
          margin=attr(l=60, r=30, t=60, b=50))
fig_err_path = joinpath(figdir, "mie_rcs_error.png")
PlotlyKaleido.savefig(p_err, fig_err_path; width=900, height=550)
println("Plot saved: $fig_err_path")

println("\n" * "="^70)
println("Mie validation complete.")
println("  E-plane MAE:  $(round(mean(abs.(delta_phi0)), digits=3)) dB")
println("  H-plane MAE:  $(round(mean(abs.(delta_phi90)), digits=3)) dB")
println("  P_rad/P_in:   $(round(P_rad / P_in, digits=4))")
println("="^70)
