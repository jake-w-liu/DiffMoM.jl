# 13_sphere_rcs_optimization.jl — Sphere RCS: Mie validation + impedance optimization
#
# End-to-end demonstration combining MoM validation and differentiable design:
#   Part A: Build an icosphere, compute PEC bistatic RCS, compare with Mie series
#   Part B: Apply spatial patch assignment and multi-angle RCS optimization
#           to reduce backscatter with resistive impedance coatings
#
# This showcases the full pipeline: geometry → MoM solve → Mie validation →
# spatial patches → adjoint gradient → L-BFGS optimization → RCS comparison.
#
# Run: julia --project=. examples/13_sphere_rcs_optimization.jl

using DiffMoM
using LinearAlgebra
using Statistics
using PlotlySupply

figdir = joinpath(@__DIR__, "figs")
mkpath(figdir)

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
println("Example 13: Sphere RCS — Mie Validation + Impedance Optimization")
println("="^70)

# ══════════════════════════════════════════════════════════════════════
# PART A: PEC Sphere — MoM vs Mie Validation
# ══════════════════════════════════════════════════════════════════════

println("\n" * "─"^70)
println("PART A: PEC Sphere — MoM vs Mie Validation")
println("─"^70)

# ── A1. Problem parameters ────────────────────────────
a     = 0.05                            # sphere radius = 5 cm
freq  = 2e9                             # 2 GHz
c0    = 299792458.0
lambda0 = c0 / freq
k     = 2π / lambda0
eta0  = 376.730313668
ka    = k * a

println("\nRadius:    $(a*100) cm")
println("Frequency: $(freq/1e9) GHz,  λ = $(round(lambda0*100, digits=2)) cm")
println("ka:        $(round(ka, digits=3))")

# ── A2. Build icosphere and export/reimport STL ───────
subdiv = 3
mesh = make_icosphere(a; subdivisions=subdiv)
println("\nIcosphere: $(nvertices(mesh)) vertices, $(ntriangles(mesh)) triangles (subdiv=$subdiv)")

# STL round-trip (demonstrate mesh I/O)
stl_path = joinpath(@__DIR__, "tmp_sphere_ex13.stl")
write_stl_mesh(stl_path, mesh; header="PEC sphere a=$(a)m ka=$(round(ka, digits=2))")
mesh = read_stl_mesh(stl_path)
println("STL round-trip: $(nvertices(mesh)) vertices, $(ntriangles(mesh)) triangles")

# ── A3. Build RWG and assemble EFIE ──────────────────
rwg = build_rwg(mesh)
N = rwg.nedges
println("RWG unknowns: $N")

println("\nAssembling Z_efie ($N × $N)...")
t_asm = @elapsed Z_efie = assemble_Z_efie(mesh, rwg, k)
println("  Done in $(round(t_asm, digits=2)) s")

# ── A4. Plane-wave excitation and PEC solve ──────────
khat   = Vec3(0.0, 0.0, -1.0)          # -z incidence
k_vec  = k * khat
pol    = Vec3(1.0, 0.0, 0.0)           # x-polarized
E0     = 1.0
v_exc  = assemble_excitation(mesh, rwg, make_plane_wave(k_vec, E0, pol))

println("Solving PEC system...")
t_sol = @elapsed I_pec = Z_efie \ v_exc
residual = norm(Z_efie * I_pec - v_exc) / norm(v_exc)
println("  Solve: $(round(t_sol, digits=3)) s,  residual = $(round(residual, sigdigits=3))")

# ── A5. Far-field and RCS on a full spherical grid ────
grid = make_sph_grid(90, 72)
NΩ = length(grid.w)
G_mat = radiation_vectors(mesh, rwg, grid, k)
E_ff_pec = compute_farfield(G_mat, Vector{ComplexF64}(I_pec), NΩ)
sigma_pec = bistatic_rcs(E_ff_pec; E0=E0)

# Energy conservation check
P_in  = input_power(Vector{ComplexF64}(I_pec), Vector{ComplexF64}(v_exc))
P_rad = radiated_power(E_ff_pec, grid)
println("\nEnergy: P_rad/P_in = $(round(P_rad/P_in, digits=4))  (should ≈ 1 for PEC)")

# ── A6. Backscatter: MoM vs Mie ──────────────────────
bs = backscatter_rcs(E_ff_pec, grid, khat; E0=E0)
sigma_bs_mie = mie_bistatic_rcs_pec(k, a, khat, pol, -khat)

bs_mom_dB = 10 * log10(max(bs.sigma, 1e-30))
bs_mie_dB = 10 * log10(max(sigma_bs_mie, 1e-30))

println("\n── Backscatter Comparison ──")
println("  MoM:  $(round(bs_mom_dB, digits=2)) dBsm")
println("  Mie:  $(round(bs_mie_dB, digits=2)) dBsm")
println("  Δ:    $(round(bs_mom_dB - bs_mie_dB, digits=2)) dB")

# ── A7. Phi=0 cut comparison (sample) ────────────────
phi_target = minimum(grid.phi)
cut_idx = [q for q in 1:NΩ if abs(grid.phi[q] - phi_target) < 1e-12]
sort!(cut_idx, by=q -> grid.theta[q])

sigma_mie_cut = [mie_bistatic_rcs_pec(k, a, khat, pol, Vec3(grid.rhat[:, q]))
                 for q in cut_idx]
sigma_mom_cut = sigma_pec[cut_idx]

dB_mom = 10 .* log10.(max.(sigma_mom_cut, 1e-30))
dB_mie = 10 .* log10.(max.(sigma_mie_cut, 1e-30))
delta_dB = dB_mom .- dB_mie

mae_dB = mean(abs.(delta_dB))
rmse_dB = sqrt(mean(abs2, delta_dB))

println("\n── Phi=0 Cut (E-plane) ──")
println("  MAE:     $(round(mae_dB, digits=3)) dB")
println("  RMSE:    $(round(rmse_dB, digits=3)) dB")
println("  Max |Δ|: $(round(maximum(abs.(delta_dB)), digits=3)) dB")

# Print a few sample points
println("\n  theta (deg)   σ_MoM (dBsm)   σ_Mie (dBsm)   Δ (dB)")
for i in 1:10:length(cut_idx)
    theta_i = rad2deg(grid.theta[cut_idx[i]])
    println("  $(lpad(round(theta_i, digits=1), 8))    " *
            "$(lpad(round(dB_mom[i], digits=2), 10))    " *
            "$(lpad(round(dB_mie[i], digits=2), 10))    " *
            "$(lpad(round(delta_dB[i], digits=2), 7))")
end


# ── A8. Fine phi=0 cut for plotting ─────────────────────
Ntheta_fine = 360
theta_fine = range(0, π, length=Ntheta_fine+1)[1:end-1] .+ π/(2*Ntheta_fine)
rhat_fine = zeros(3, Ntheta_fine)
theta_f_vec = zeros(Ntheta_fine)
phi_f_vec = zeros(Ntheta_fine)
w_fine = zeros(Ntheta_fine)
dtheta_fine = π / Ntheta_fine
for i in 1:Ntheta_fine
    theta_f_vec[i] = theta_fine[i]
    rhat_fine[1, i] = sin(theta_fine[i])
    rhat_fine[3, i] = cos(theta_fine[i])
    w_fine[i] = sin(theta_fine[i]) * dtheta_fine * 2π
end
grid_fine = SphGrid(rhat_fine, theta_f_vec, phi_f_vec, w_fine)

G_fine = radiation_vectors(mesh, rwg, grid_fine, k)
E_ff_fine_pec = compute_farfield(G_fine, Vector{ComplexF64}(I_pec), Ntheta_fine)
sigma_fine_pec = bistatic_rcs(E_ff_fine_pec; E0=E0)

sigma_mie_fine = [mie_bistatic_rcs_pec(k, a, khat, pol, Vec3(grid_fine.rhat[:, q]))
                  for q in 1:Ntheta_fine]

dB_fine_mom = 10 .* log10.(max.(sigma_fine_pec, 1e-30))
dB_fine_mie = 10 .* log10.(max.(sigma_mie_fine, 1e-30))
theta_fine_deg = rad2deg.(theta_fine)

# ── Plot 1: MoM vs Mie validation ──────────────────────
println("\nGenerating Plot 1: MoM vs Mie validation...")
sf1 = subplots(1, 1; sync=false, width=700, height=400)
addtraces!(sf1, scatter(x=theta_fine_deg, y=dB_fine_mie, mode="lines",
           name="Mie (analytical)", line=attr(color="blue", width=3)); row=1, col=1)
addtraces!(sf1, scatter(x=theta_fine_deg, y=dB_fine_mom, mode="lines",
           name="MoM (N=$N)", line=attr(color="red", width=2, dash="dash")); row=1, col=1)
p1 = sf1.plot
relayout!(p1,
    title="",
    annotations=[],
    xaxis=attr(title=attr(text="θ (deg)", font=attr(size=24)),
              tickfont=attr(size=20), range=[0, 180], dtick=30),
    yaxis=attr(title=attr(text="Bistatic RCS (dBsm)", font=attr(size=24)),
              tickfont=attr(size=20)),
    margin=attr(l=80, r=30, t=20, b=70))
set_legend!(p1; position=:bottomright)
fig1_path = joinpath(figdir, "13_sphere_mom_vs_mie.pdf")
savefig(p1, fig1_path)
println("  Saved: $fig1_path")


# ══════════════════════════════════════════════════════════════════════
# PART B: Impedance Optimization to Reduce Backscatter RCS
# ══════════════════════════════════════════════════════════════════════

println("\n\n" * "─"^70)
println("PART B: Impedance Optimization to Reduce Backscatter RCS")
println("─"^70)

# ── B1. Spatial patch assignment ──────────────────────
# Divide the sphere into patches with a 4×4×4 grid
partition = assign_patches_grid(mesh; nx=4, ny=4, nz=4)
P = partition.P
println("\nSpatial patches: $P (from 4×4×4 grid, empty cells skipped)")

# ── B2. Precompute patch mass matrices ────────────────
println("Precomputing patch mass matrices ($P patches)...")
Mp = precompute_patch_mass(mesh, rwg, partition)

# ── B3. Build Q matrix for backscatter ────────────────
# Target backscatter direction: -k̂ = +z
pol_grid = pol_linear_x(grid)
mask_bs = direction_mask(grid, -khat; half_angle=15*π/180)   # 15° cone
Q_bs = build_Q(G_mat, grid, pol_grid; mask=mask_bs)

n_bs = count(mask_bs)
println("Backscatter Q: 15° cone around -k̂, $n_bs of $NΩ directions selected")

# ── B4. Compute initial (PEC) backscatter objective ───
J_pec = real(dot(I_pec, Q_bs * I_pec))
println("\n── PEC baseline ──")
println("  J_backscatter(PEC) = $(round(J_pec, sigdigits=4))")
println("  Backscatter RCS:     $(round(bs_mom_dB, digits=2)) dBsm")

# ── B5. Single-angle optimization (dense, direct) ────
println("\n── Single-angle optimization (minimize backscatter) ──")
theta0 = zeros(P)
lb = zeros(P)                          # passive resistive: theta >= 0
ub = fill(1000.0, P)                   # upper bound: 1000 Ω/sq

theta_opt_single, trace_single = optimize_lbfgs(
    Z_efie, Mp, Vector{ComplexF64}(v_exc), Q_bs, theta0;
    maxiter=50,
    tol=1e-8,
    maximize=false,                     # minimize backscatter
    lb=lb, ub=ub,
    verbose=true,
)

J_opt_single = trace_single[end].J
reduction_dB_single = 10 * log10(J_pec / max(J_opt_single, 1e-30))
println("\n  J(PEC)       = $(round(J_pec, sigdigits=4))")
println("  J(optimized) = $(round(J_opt_single, sigdigits=4))")
println("  J reduction:   $(round(reduction_dB_single, digits=1)) dB")
println("  Iterations:    $(length(trace_single))")

# Recompute optimized RCS
Z_opt = assemble_full_Z(Z_efie, Mp, theta_opt_single)
I_opt_single = Z_opt \ Vector{ComplexF64}(v_exc)
E_ff_opt_single = compute_farfield(G_mat, Vector{ComplexF64}(I_opt_single), NΩ)
bs_opt_single = backscatter_rcs(E_ff_opt_single, grid, khat; E0=E0)
bs_opt_dB = 10 * log10(max(bs_opt_single.sigma, 1e-30))
println("  PEC backscatter:   $(round(bs_mom_dB, digits=2)) dBsm")
println("  Opt backscatter:   $(round(bs_opt_dB, digits=2)) dBsm")
println("  RCS reduction:     $(round(bs_mom_dB - bs_opt_dB, digits=2)) dB")

# ── B6. Multi-angle RCS optimization ─────────────────
println("\n── Multi-angle optimization (2 incidence angles) ──")

# Two incidence angles: broadside (-z) and 45° off broadside
# theta_inc is measured from +z, so broadside (-z incidence) = π
angles = [
    (theta_inc=π,      phi_inc=0.0, pol=Vec3(1.0, 0.0, 0.0), weight=1.0),   # broadside (-z)
    (theta_inc=3π/4,   phi_inc=0.0, pol=Vec3(1.0, 0.0, 0.0), weight=1.0),   # 45° off broadside
]

grid_opt = make_sph_grid(36, 36)
configs = build_multiangle_configs(mesh, rwg, k, angles;
                                    grid=grid_opt, backscatter_cone=15.0)

# Build preconditioner for GMRES
P_nf = build_nearfield_preconditioner(Z_efie, mesh, rwg, 1.0 * lambda0)

theta_opt_multi, trace_multi = optimize_multiangle_rcs(
    Z_efie, Mp, configs, zeros(P);
    lb=lb, ub=ub,
    preconditioner=P_nf,
    maxiter=30,
    tol=1e-10,
    gmres_tol=1e-6,
    gmres_maxiter=200,
    verbose=true,
)

J_multi_init = trace_multi[1].J
J_multi_opt  = trace_multi[end].J
reduction_dB_multi = 10 * log10(J_multi_init / max(J_multi_opt, 1e-30))
println("\n  J(initial) = $(round(J_multi_init, sigdigits=4))")
println("  J(optimal) = $(round(J_multi_opt, sigdigits=4))")
println("  J reduction:   $(round(reduction_dB_multi, digits=1)) dB")

# ── B7. Recompute and compare RCS for both angles ────
println("\n── RCS comparison (optimized vs PEC) ──")

Z_opt_multi = assemble_full_Z(Z_efie, Mp, theta_opt_multi)
for (a_idx, ang) in enumerate(angles)
    cfg = configs[a_idx]
    # PEC solve at this angle
    khat_a = Vec3(sin(ang.theta_inc)*cos(ang.phi_inc),
                  sin(ang.theta_inc)*sin(ang.phi_inc),
                  cos(ang.theta_inc))
    k_vec_a = k * khat_a
    v_a = assemble_excitation(mesh, rwg, make_plane_wave(k_vec_a, 1.0, cfg.pol))

    # PEC
    I_pec_a = Z_efie \ Vector{ComplexF64}(v_a)
    E_pec_a = compute_farfield(G_mat, Vector{ComplexF64}(I_pec_a), NΩ)
    bs_pec_a = backscatter_rcs(E_pec_a, grid, khat_a; E0=E0)

    # Optimized
    I_opt_a = Z_opt_multi \ Vector{ComplexF64}(v_a)
    E_opt_a = compute_farfield(G_mat, Vector{ComplexF64}(I_opt_a), NΩ)
    bs_opt_a = backscatter_rcs(E_opt_a, grid, khat_a; E0=E0)

    pec_dB = 10 * log10(max(bs_pec_a.sigma, 1e-30))
    opt_dB = 10 * log10(max(bs_opt_a.sigma, 1e-30))

    theta_deg = round(rad2deg(ang.theta_inc), digits=1)
    println("  Angle $a_idx (θ=$(theta_deg)°): PEC = $(round(pec_dB, digits=2)) dBsm, " *
            "Opt = $(round(opt_dB, digits=2)) dBsm, " *
            "Δ = $(round(pec_dB - opt_dB, digits=2)) dB reduction")
end

# ── B8. Compare with Mie at backscatter ───────────────
println("\n── Optimized vs Mie reference (broadside backscatter) ──")
I_opt_bs = Z_opt_multi \ Vector{ComplexF64}(v_exc)
E_opt_bs = compute_farfield(G_mat, Vector{ComplexF64}(I_opt_bs), NΩ)
bs_opt_final = backscatter_rcs(E_opt_bs, grid, khat; E0=E0)
opt_final_dB = 10 * log10(max(bs_opt_final.sigma, 1e-30))
println("  PEC MoM:       $(round(bs_mom_dB, digits=2)) dBsm")
println("  PEC Mie:       $(round(bs_mie_dB, digits=2)) dBsm")
println("  Optimized MoM: $(round(opt_final_dB, digits=2)) dBsm")

# ── B9. Impedance distribution summary ────────────────
println("\n── Impedance distribution (Ω/sq, resistive) ──")
active_vals = theta_opt_multi[theta_opt_multi .> 1e-3]
nonzero_patches = length(active_vals)
println("  Active patches: $nonzero_patches / $P")
if nonzero_patches > 0
    println("  Mean θ (active): $(round(mean(active_vals), digits=1)) Ω/sq")
    println("  Max θ:           $(round(maximum(active_vals), digits=1)) Ω/sq")
    println("  Min θ (active):  $(round(minimum(active_vals), digits=1)) Ω/sq")
else
    println("  All patches at PEC (θ ≈ 0)")
end

# Energy conservation check for optimized sphere
P_in_opt  = input_power(Vector{ComplexF64}(I_opt_bs), Vector{ComplexF64}(v_exc))
P_rad_opt = radiated_power(E_opt_bs, grid)
println("\n  P_rad/P_in (optimized): $(round(P_rad_opt/P_in_opt, digits=4))  (< 1 → energy absorbed by coating)")

# ── B10. Plot 2: Optimization comparison (broadside, phi=0 cut) ────
println("\nGenerating Plot 2: PEC vs optimized RCS (broadside incidence)...")

# Compute fine-cut far-fields for optimized coatings
E_ff_fine_single = compute_farfield(G_fine, Vector{ComplexF64}(I_opt_single), Ntheta_fine)
sigma_fine_single = bistatic_rcs(E_ff_fine_single; E0=E0)
dB_fine_single = 10 .* log10.(max.(sigma_fine_single, 1e-30))

E_ff_fine_multi = compute_farfield(G_fine, Vector{ComplexF64}(I_opt_bs), Ntheta_fine)
sigma_fine_multi = bistatic_rcs(E_ff_fine_multi; E0=E0)
dB_fine_multi = 10 .* log10.(max.(sigma_fine_multi, 1e-30))

sf2 = subplots(1, 1; sync=false, width=700, height=400)
addtraces!(sf2, scatter(x=theta_fine_deg, y=dB_fine_mom, mode="lines",
           name="PEC (baseline)", line=attr(color="black", width=3)); row=1, col=1)
addtraces!(sf2, scatter(x=theta_fine_deg, y=dB_fine_single, mode="lines",
           name="Single-angle opt", line=attr(color="#2ca02c", width=2, dash="dash")); row=1, col=1)
addtraces!(sf2, scatter(x=theta_fine_deg, y=dB_fine_multi, mode="lines",
           name="Multi-angle opt", line=attr(color="#d62728", width=2, dash="dot")); row=1, col=1)
addtraces!(sf2, scatter(x=theta_fine_deg, y=dB_fine_mie, mode="lines",
           name="Mie (PEC ref)", line=attr(color="blue", width=1, dash="dashdot")); row=1, col=1)
p2 = sf2.plot
relayout!(p2,
    title="",
    annotations=[],
    xaxis=attr(title=attr(text="θ (deg)", font=attr(size=24)),
              tickfont=attr(size=20), range=[0, 180], dtick=30),
    yaxis=attr(title=attr(text="Bistatic RCS (dBsm)", font=attr(size=24)),
              tickfont=attr(size=20)),
    margin=attr(l=80, r=30, t=20, b=70))
set_legend!(p2; position=:topright)
fig2_path = joinpath(figdir, "13_sphere_rcs_optimization.pdf")
savefig(p2, fig2_path)
println("  Saved: $fig2_path")

# ── B11. Plot 4: Multi-angle — signed-θ xz-plane cut (both angles, one plot) ──
println("Generating Plot 4: Multi-angle RCS comparison (signed-θ xz cut)...")

# Build a signed-θ great-circle cut in the xz-plane:
#   θ_signed > 0  →  φ=0 side  (+x half of xz-plane)
#   θ_signed < 0  →  φ=π side  (-x half of xz-plane)
#   θ_signed = 0  →  +z (broadside for -z incidence)
# This is the standard RCS paper convention.
Nsc = 720
theta_signed_deg = range(-180.0, 180.0, length=Nsc+1)[1:end-1] .+ 360.0/(2*Nsc)
rhat_sc = zeros(3, Nsc)
theta_sc = zeros(Nsc)
phi_sc = zeros(Nsc)
w_sc = zeros(Nsc)
for i in 1:Nsc
    ts_rad = deg2rad(theta_signed_deg[i])
    if ts_rad >= 0
        theta_sc[i] = ts_rad
        phi_sc[i] = 0.0
    else
        theta_sc[i] = -ts_rad
        phi_sc[i] = π
    end
    rhat_sc[1, i] = sin(theta_sc[i]) * cos(phi_sc[i])
    rhat_sc[2, i] = sin(theta_sc[i]) * sin(phi_sc[i])
    rhat_sc[3, i] = cos(theta_sc[i])
    w_sc[i] = sin(max(theta_sc[i], 1e-12)) * deg2rad(360.0/Nsc)  # dummy weights
end
grid_sc = SphGrid(rhat_sc, theta_sc, phi_sc, w_sc)
G_sc = radiation_vectors(mesh, rwg, grid_sc, k)

# Compute backscatter positions in signed-θ for annotation
bs_signed_deg = zeros(2)
angle_labels = ["Broadside (-z)", "45° off broadside"]
for (a_idx, ang) in enumerate(angles)
    khat_a = Vec3(sin(ang.theta_inc)*cos(ang.phi_inc),
                  sin(ang.theta_inc)*sin(ang.phi_inc),
                  cos(ang.theta_inc))
    bs_dir = -khat_a
    bs_theta = acos(clamp(bs_dir[3], -1.0, 1.0))
    if bs_theta < 1e-6
        bs_signed_deg[a_idx] = 0.0                              # +z pole
    elseif π - bs_theta < 1e-6
        bs_signed_deg[a_idx] = 180.0                            # -z pole
    else
        bs_phi = atan(bs_dir[2], bs_dir[1])
        if abs(bs_phi) < 0.1 || abs(bs_phi - 2π) < 0.1
            bs_signed_deg[a_idx] = rad2deg(bs_theta)            # φ≈0 → positive θ
        else
            bs_signed_deg[a_idx] = -rad2deg(bs_theta)           # φ≈π → negative θ
        end
    end
end
println("  Backscatter directions: $(round.(bs_signed_deg, digits=1))° (signed-θ)")

# Solve both angles on the signed-θ grid
pec_sc_dB = Vector{Vector{Float64}}(undef, 2)
opt_sc_dB = Vector{Vector{Float64}}(undef, 2)
for (a_idx, ang) in enumerate(angles)
    cfg = configs[a_idx]
    khat_a = Vec3(sin(ang.theta_inc)*cos(ang.phi_inc),
                  sin(ang.theta_inc)*sin(ang.phi_inc),
                  cos(ang.theta_inc))
    v_a = assemble_excitation(mesh, rwg, make_plane_wave(k * khat_a, 1.0, cfg.pol))

    I_pec_a = Z_efie \ Vector{ComplexF64}(v_a)
    E_pec_sc = compute_farfield(G_sc, Vector{ComplexF64}(I_pec_a), Nsc)
    pec_sc_dB[a_idx] = 10 .* log10.(max.(bistatic_rcs(E_pec_sc; E0=E0), 1e-30))

    I_opt_a = Z_opt_multi \ Vector{ComplexF64}(v_a)
    E_opt_sc = compute_farfield(G_sc, Vector{ComplexF64}(I_opt_a), Nsc)
    opt_sc_dB[a_idx] = 10 .* log10.(max.(bistatic_rcs(E_opt_sc; E0=E0), 1e-30))
end

sf4 = subplots(1, 1; sync=false, width=700, height=400)

# Angle 1: broadside (-z incidence)
addtraces!(sf4, scatter(x=theta_signed_deg, y=pec_sc_dB[1], mode="lines",
           name="PEC, $(angle_labels[1])", line=attr(color="black", width=2)); row=1, col=1)
addtraces!(sf4, scatter(x=theta_signed_deg, y=opt_sc_dB[1], mode="lines",
           name="Opt, $(angle_labels[1])", line=attr(color="#d62728", width=2, dash="dash")); row=1, col=1)

# Angle 2: 45° off broadside
addtraces!(sf4, scatter(x=theta_signed_deg, y=pec_sc_dB[2], mode="lines",
           name="PEC, $(angle_labels[2])", line=attr(color="gray", width=2)); row=1, col=1)
addtraces!(sf4, scatter(x=theta_signed_deg, y=opt_sc_dB[2], mode="lines",
           name="Opt, $(angle_labels[2])", line=attr(color="#ff7f0e", width=2, dash="dash")); row=1, col=1)

p4 = sf4.plot

# Add vertical lines at backscatter directions
relayout!(p4,
    title="",
    xaxis=attr(title=attr(text="θ (deg) — signed, xz-plane", font=attr(size=24)),
              tickfont=attr(size=20), range=[-180, 180], dtick=30),
    yaxis=attr(title=attr(text="Bistatic RCS (dBsm)", font=attr(size=24)),
              tickfont=attr(size=20)),
    margin=attr(l=80, r=30, t=20, b=70),
    shapes=[
        attr(type="line", x0=bs_signed_deg[1], x1=bs_signed_deg[1],
             y0=0, y1=1, yref="paper",
             line=attr(color="#d62728", width=1.5, dash="dot")),
        attr(type="line", x0=bs_signed_deg[2], x1=bs_signed_deg[2],
             y0=0, y1=1, yref="paper",
             line=attr(color="#ff7f0e", width=1.5, dash="dot")),
    ],
    annotations=[
        attr(x=bs_signed_deg[1]+3, y=1.02, yref="paper", text="BS₁",
             showarrow=false, font=attr(color="#d62728", size=18)),
        attr(x=bs_signed_deg[2]-3, y=1.02, yref="paper", text="BS₂",
             showarrow=false, font=attr(color="#ff7f0e", size=18)),
    ])
set_legend!(p4; position=:bottomright)
fig4_path = joinpath(figdir, "13_sphere_multiangle_rcs.pdf")
savefig(p4, fig4_path)
println("  Saved: $fig4_path")

# ── B12. Plot 3: Convergence history ──────────────────
println("Generating Plot 3: Optimization convergence...")

# Single-angle trace
iters_s = [t.iter for t in trace_single]
J_s = [t.J for t in trace_single]
J_s_dB = 10 .* log10.(max.(J_s, 1e-30))

# Multi-angle trace
iters_m = [t.iter for t in trace_multi]
J_m = [t.J for t in trace_multi]
J_m_dB = 10 .* log10.(max.(J_m, 1e-30))

sf3 = subplots(1, 1; sync=false, width=700, height=400)
addtraces!(sf3, scatter(x=iters_s, y=J_s_dB, mode="lines+markers",
           name="Single-angle", line=attr(color="#2ca02c", width=2),
           marker=attr(size=8)); row=1, col=1)
addtraces!(sf3, scatter(x=iters_m, y=J_m_dB, mode="lines+markers",
           name="Multi-angle (2 angles)", line=attr(color="#d62728", width=2),
           marker=attr(size=6)); row=1, col=1)
p3 = sf3.plot
relayout!(p3,
    title="",
    annotations=[],
    xaxis=attr(title=attr(text="L-BFGS Iteration", font=attr(size=24)),
              tickfont=attr(size=20), dtick=5),
    yaxis=attr(title=attr(text="10 log₁₀(J) (dB)", font=attr(size=24)),
              tickfont=attr(size=20)),
    margin=attr(l=80, r=30, t=20, b=70))
set_legend!(p3; position=:topright)
fig3_path = joinpath(figdir, "13_sphere_convergence.pdf")
savefig(p3, fig3_path)
println("  Saved: $fig3_path")

# ── B13. Plot 5: Mesh + patch partition (3D) ─────────────
println("\nGenerating Plot 5: Mesh + patch partition (3D)...")

face_patch_ids = Float64.(partition.tri_patch)

sf5 = subplots(1, 1; sync=false, width=900, height=800,
               specs=reshape([Spec(kind="scene")], 1, 1))
addtraces!(sf5, mesh3d(
    x = vec(mesh.xyz[1,:]),
    y = vec(mesh.xyz[2,:]),
    z = vec(mesh.xyz[3,:]),
    i = vec(mesh.tri[1,:] .- 1),
    j = vec(mesh.tri[2,:] .- 1),
    k = vec(mesh.tri[3,:] .- 1),
    intensity = face_patch_ids,
    intensitymode = "cell",
    colorscale = "Greys",
    colorbar = attr(title=attr(text="Patch ID", font=attr(size=28), side="right"),
                    tickfont=attr(size=22), len=0.65, thickness=25, x=0.92),
    showscale = true,
    flatshading = true,
    lighting = attr(ambient=0.7, diffuse=0.5, specular=0.1),
); row=1, col=1)
p5 = sf5.plot
relayout!(p5,
    title="",
    annotations=[],
    scene = attr(
        xaxis = attr(visible=false),
        yaxis = attr(visible=false),
        zaxis = attr(visible=false),
        aspectmode = "data",
        camera = attr(eye = attr(x=1.5, y=-1.5, z=1.2)),
        bgcolor = "white",
    ),
    margin = attr(l=0, r=80, t=10, b=10),
    paper_bgcolor = "white")
fig5_path = joinpath(figdir, "13_sphere_mesh_patches.pdf")
savefig(p5, fig5_path)
println("  Saved: $fig5_path")


# ── B14. Plot 6: Impedance distribution (3D) ────────────
println("Generating Plot 6: Impedance distribution (3D)...")

# Map per-patch impedance values to per-triangle values
face_impedance = [theta_opt_multi[partition.tri_patch[t]] for t in 1:ntriangles(mesh)]

sf6 = subplots(1, 1; sync=false, width=900, height=800,
               specs=reshape([Spec(kind="scene")], 1, 1))
addtraces!(sf6, mesh3d(
    x = vec(mesh.xyz[1,:]),
    y = vec(mesh.xyz[2,:]),
    z = vec(mesh.xyz[3,:]),
    i = vec(mesh.tri[1,:] .- 1),
    j = vec(mesh.tri[2,:] .- 1),
    k = vec(mesh.tri[3,:] .- 1),
    intensity = face_impedance,
    intensitymode = "cell",
    colorscale = "YlOrRd",
    colorbar = attr(title=attr(text="θ (Ω/sq)", font=attr(size=28), side="right"),
                    tickfont=attr(size=22), len=0.65, thickness=25, x=0.92),
    showscale = true,
    flatshading = true,
    lighting = attr(ambient=0.7, diffuse=0.5, specular=0.1),
); row=1, col=1)
p6 = sf6.plot
relayout!(p6,
    title="",
    annotations=[],
    scene = attr(
        xaxis = attr(visible=false),
        yaxis = attr(visible=false),
        zaxis = attr(visible=false),
        aspectmode = "data",
        camera = attr(eye = attr(x=1.5, y=-1.5, z=1.2)),
        bgcolor = "white",
    ),
    margin = attr(l=0, r=80, t=10, b=10),
    paper_bgcolor = "white")
fig6_path = joinpath(figdir, "13_sphere_impedance_dist.pdf")
savefig(p6, fig6_path)
println("  Saved: $fig6_path")


# ── B15. GMRES iteration count analysis ─────────────────
println("\n── GMRES Iteration Count Analysis ──")
println("  Forward solve (Z(θ)·I = v) convergence, rtol=1e-6, N=$N\n")

gmres_test_configs = [
    ("PEC (θ=0)",             zeros(P)),
    ("Optimized θ",           theta_opt_multi),
    ("Uniform 200 Ω/sq",     fill(200.0, P)),
    ("Uniform 500 Ω/sq",     fill(500.0, P)),
]

println("  " * rpad("Configuration", 24) * rpad("Unprecond.", 14) * "NF precond.")
println("  " * "─"^50)

gmres_results = []
for (label, theta_test) in gmres_test_configs
    Z_test = ImpedanceLoadedOperator(Z_efie, Mp, theta_test, false)

    # Without preconditioner
    _, stats_no = solve_gmres(Z_test, Vector{ComplexF64}(v_exc);
                               tol=1e-6, maxiter=500)

    # With NF preconditioner (right)
    _, stats_nf = solve_gmres(Z_test, Vector{ComplexF64}(v_exc);
                               preconditioner=P_nf,
                               tol=1e-6, maxiter=500)

    println("  " * rpad(label, 24) * rpad("$(stats_no.niter)", 14) * "$(stats_nf.niter)")
    push!(gmres_results, (label=label, no_precond=stats_no.niter, nf_precond=stats_nf.niter))
end


# ── Cleanup temp file ────────────────────────────────
rm(stl_path; force=true)

println("\n" * "="^70)
println("Example 13 complete.")
println("  Part A: PEC sphere RCS validated against Mie (MAE = $(round(mae_dB, digits=2)) dB)")
println("  Part B: Single-angle backscatter reduced by $(round(bs_mom_dB - bs_opt_dB, digits=1)) dB")
println("          Multi-angle (2 angles) J reduced by $(round(reduction_dB_multi, digits=1)) dB")
println("          via resistive impedance optimization ($P spatial patches)")
println("="^70)
