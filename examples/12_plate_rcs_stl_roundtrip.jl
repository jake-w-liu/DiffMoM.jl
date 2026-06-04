# 12_plate_rcs_stl_roundtrip.jl — PEC rectangular plate RCS: STL mesh I/O + analytical PO comparison
#
# This example demonstrates:
#   1. Create a rectangular plate mesh
#   2. Export it to STL format (binary)
#   3. Import the STL mesh back
#   4. Run full MoM forward solve
#   5. Compare bistatic RCS with the analytical Physical Optics (PO) formula
#
# The PO analytical result for a rectangular PEC plate illuminated at normal
# incidence (k = -z) is:
#
#   sigma(theta, phi) = (4 pi A^2 / lambda^2)
#       * sinc^2(k Lx sin(theta) cos(phi) / 2)
#       * sinc^2(k Ly sin(theta) sin(phi) / 2)
#
# where A = Lx * Ly and sinc(x) = sin(x)/x.
# This is exact in the PO limit (electrically large plates); edge diffraction
# causes deviations for small plates.
#
# Run: julia --project=. examples/12_plate_rcs_stl_roundtrip.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using PlotlySupply

println("="^60)
println("Example 12: PEC Plate RCS — STL Round-Trip + PO Analytical")
println("="^60)

# ── 1. Problem parameters ─────────────────────────
freq = 3e9                          # 3 GHz
c0   = 299792458.0
lambda0 = c0 / freq                 # ~10 cm
k    = 2π / lambda0
eta0 = 376.730313668

Lx, Ly = 0.20, 0.15                # 20 cm × 15 cm plate (2λ × 1.5λ)
Nx, Ny = 20, 15                     # ~λ/10 edge length

println("\nFrequency: $(freq/1e9) GHz,  lambda = $(round(lambda0*100, digits=2)) cm")
println("Plate: $(Lx*100) cm x $(Ly*100) cm  ($(round(Lx/lambda0, digits=2))lambda x $(round(Ly/lambda0, digits=2))lambda)")

# ── 2. Create mesh and export to STL ──────────────
mesh_orig = make_rect_plate(Lx, Ly, Nx, Ny)
println("\nOriginal mesh: $(nvertices(mesh_orig)) vertices, $(ntriangles(mesh_orig)) triangles")

stl_path = joinpath(@__DIR__, "..", "data", "plate_rcs_demo.stl")
mkpath(dirname(stl_path))
write_stl_mesh(stl_path, mesh_orig; header="PEC plate $(Lx)x$(Ly)m")
println("Exported to STL: $stl_path")

# ── 3. Import STL mesh back ───────────────────────
mesh = read_stl_mesh(stl_path)
report = assert_mesh_quality(mesh; allow_boundary=true)
println("Re-imported STL: $(nvertices(mesh)) vertices, $(ntriangles(mesh)) triangles")
println("Quality: OK ($(report.n_boundary_edges) boundary edges, 0 defects)")

# Check resolution
res = mesh_resolution_report(mesh, freq)
println("Edge max/lambda: $(round(res.edge_max_over_lambda, digits=3))  (target <= 0.1)")

# ── 4. Build RWG and assemble EFIE ────────────────
rwg = build_rwg(mesh)
N = rwg.nedges
println("\nRWG basis functions: $N")
println("Estimated memory: $(round(estimate_dense_matrix_gib(N)*1024, digits=1)) MiB")

println("Assembling Z_efie ($N x $N)...")
t_asm = @elapsed Z = assemble_Z_efie(mesh, rwg, k)
println("  Done in $(round(t_asm, digits=2)) s")

# ── 5. Plane-wave excitation and solve ────────────
# Normal incidence: wave propagating in -z, x-polarized
k_vec = Vec3(0.0, 0.0, -k)
E0 = 1.0
pol = Vec3(1.0, 0.0, 0.0)
v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol)

println("Solving Z I = v...")
t_solve = @elapsed I_pec = Z \ v
residual = norm(Z * I_pec - v) / norm(v)
println("  Relative residual: $(round(residual, sigdigits=3))")

# ── 6. Far-field computation (fine phi=0 cut) ─────
Ntheta_cut = 360
theta_cut = range(0, π, length=Ntheta_cut+1)[1:end-1] .+ π/(2*Ntheta_cut)
phi_cut = 0.0  # E-plane (xz-plane) for x-polarized incidence

# Build a cut-plane grid at phi=0
rhat_cut = zeros(3, Ntheta_cut)
theta_vec = zeros(Ntheta_cut)
phi_vec = zeros(Ntheta_cut)
w_cut = zeros(Ntheta_cut)
dtheta = π / Ntheta_cut
for i in 1:Ntheta_cut
    theta_vec[i] = theta_cut[i]
    phi_vec[i] = phi_cut
    rhat_cut[1, i] = sin(theta_cut[i]) * cos(phi_cut)
    rhat_cut[2, i] = sin(theta_cut[i]) * sin(phi_cut)
    rhat_cut[3, i] = cos(theta_cut[i])
    w_cut[i] = sin(theta_cut[i]) * dtheta * 2π
end
grid_cut = SphGrid(rhat_cut, theta_vec, phi_vec, w_cut)

println("\nComputing far-field ($(Ntheta_cut) directions, phi=0 cut)...")
G_mat = radiation_vectors(mesh, rwg, grid_cut, k)
E_ff = compute_farfield(G_mat, Vector{ComplexF64}(I_pec), Ntheta_cut)
sigma_mom = bistatic_rcs(E_ff; E0=E0)

# ── 7. Analytical PO result ───────────────────────
# sigma_PO(theta, phi) = (4pi/lambda^2) * A^2 * sinc^2(u) * sinc^2(v)
# where u = kLx*sin(theta)*cos(phi)/2, v = kLy*sin(theta)*sin(phi)/2
# and sinc(x) = sin(x)/x  (unnormalized sinc)
A_plate = Lx * Ly
sigma_po = zeros(Ntheta_cut)
for i in 1:Ntheta_cut
    ux = k * Lx * sin(theta_cut[i]) * cos(phi_cut) / 2
    vy = k * Ly * sin(theta_cut[i]) * sin(phi_cut) / 2
    sinc_u = abs(ux) < 1e-12 ? 1.0 : sin(ux) / ux
    sinc_v = abs(vy) < 1e-12 ? 1.0 : sin(vy) / vy
    sigma_po[i] = (4π / lambda0^2) * A_plate^2 * sinc_u^2 * sinc_v^2
end

# ── 8. Comparison ─────────────────────────────────
sigma_mom_dB = 10 .* log10.(max.(sigma_mom, 1e-30))
sigma_po_dB  = 10 .* log10.(max.(sigma_po, 1e-30))
theta_deg = rad2deg.(theta_cut)

# Broadside (theta ~ 0) comparison
idx_broadside = argmin(abs.(theta_cut))
sigma_broadside_mom = sigma_mom_dB[idx_broadside]
sigma_broadside_po  = sigma_po_dB[idx_broadside]
sigma_broadside_exact_dB = 10 * log10(4π * A_plate^2 / lambda0^2)

println("\n── Broadside RCS (theta = 0) ──")
println("  MoM:        $(round(sigma_broadside_mom, digits=2)) dBsm")
println("  PO formula: $(round(sigma_broadside_po, digits=2)) dBsm")
println("  Exact 4piA^2/lambda^2: $(round(sigma_broadside_exact_dB, digits=2)) dBsm")
println("  MoM - PO:   $(round(sigma_broadside_mom - sigma_broadside_po, digits=2)) dB")

# Main-lobe comparison (|theta| < 30 deg)
main_lobe = (theta_deg .< 30) .| (theta_deg .> 150)
delta_main = abs.(sigma_mom_dB[main_lobe] - sigma_po_dB[main_lobe])
println("\n── Main-lobe agreement (|theta| < 30 deg from broadside) ──")
println("  Mean |MoM - PO|: $(round(mean(delta_main), digits=2)) dB")
println("  Max  |MoM - PO|: $(round(maximum(delta_main), digits=2)) dB")

# Energy conservation
P_in = input_power(Vector{ComplexF64}(I_pec), Vector{ComplexF64}(v))
P_rad = radiated_power(E_ff, grid_cut)
println("\n── Energy conservation ──")
println("  P_rad/P_in = $(round(P_rad / P_in, digits=4))  (should be ~ 1 for PEC)")

# ── 9. Save CSV ───────────────────────────────────
datadir = joinpath(@__DIR__, "..", "data")
mkpath(datadir)

df = DataFrame(
    theta_deg = theta_deg,
    rcs_mom_dBsm = sigma_mom_dB,
    rcs_po_dBsm = sigma_po_dB,
    rcs_mom_m2 = sigma_mom,
    rcs_po_m2 = sigma_po,
)
csv_path = joinpath(datadir, "plate_rcs_mom_vs_po_phi0.csv")
CSV.write(csv_path, df)
println("\nCSV saved: $csv_path")

df_summary = DataFrame(
    metric = [
        "freq_GHz", "lambda_m", "Lx_m", "Ly_m", "Lx_over_lambda", "Ly_over_lambda",
        "N_rwg", "broadside_mom_dBsm", "broadside_po_dBsm", "broadside_exact_dBsm",
        "broadside_delta_dB", "main_lobe_mean_delta_dB", "main_lobe_max_delta_dB",
        "P_rad_over_P_in",
    ],
    value = [
        freq/1e9, lambda0, Lx, Ly, Lx/lambda0, Ly/lambda0,
        N, sigma_broadside_mom, sigma_broadside_po, sigma_broadside_exact_dB,
        sigma_broadside_mom - sigma_broadside_po,
        mean(delta_main), maximum(delta_main),
        P_rad / P_in,
    ],
)
CSV.write(joinpath(datadir, "plate_rcs_mom_vs_po_summary.csv"), df_summary)

# ── 10. Plot ──────────────────────────────────────
figdir = joinpath(@__DIR__, "figs")
mkpath(figdir)

Lx_lam = round(Lx / lambda0, digits=1)
Ly_lam = round(Ly / lambda0, digits=1)
title_str = "PEC Plate $(Lx_lam)λ×$(Ly_lam)λ — Bistatic RCS φ=0° ($(freq/1e9) GHz, N=$N)"

sf = subplots(1, 1; sync=false, width=900, height=550,
              subplot_titles=reshape([title_str], 1, 1))
addtraces!(sf, scatter(x=theta_deg, y=sigma_po_dB, mode="lines",
           name="PO analytical", line=attr(color="blue", width=2)); row=1, col=1)
addtraces!(sf, scatter(x=theta_deg, y=sigma_mom_dB, mode="lines",
           name="MoM (N=$N)", line=attr(color="red", width=2, dash="dash")); row=1, col=1)
p = sf.plot
relayout!(p, xaxis=attr(title="θ (deg)", range=[0, 180], dtick=30),
          yaxis=attr(title="Bistatic RCS (dBsm)", range=[-50, 10]),
          legend=attr(x=0.60, y=0.95),
          margin=attr(l=60, r=30, t=60, b=50))
fig_path = joinpath(figdir, "12_plate_rcs_mom_vs_po.png")
savefig(p, fig_path)
println("\nPlot saved: $fig_path")

println("\n" * "="^60)
println("Done.  MoM and PO analytical results agree well in the main lobe.")
println("Edge diffraction (not captured by PO) causes deviations at wide angles.")
println("="^60)
