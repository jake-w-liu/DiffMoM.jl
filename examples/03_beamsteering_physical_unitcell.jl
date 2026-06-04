# 03_beamsteering_physical_unitcell.jl — Physically realizable beam steering
#
# Paper-aligned beam-steering demo. Instead of treating every RWG triangle as
# an independent impedance DOF (a numerical artefact of the Galerkin
# discretization), this example groups the two triangles sharing each
# rectangular mesh cell into ONE design variable:
# one rectangular cell ↔ one printable metasurface unit cell ↔ one Im(Z_s).
#
# Design grid: Np × Np rectangular unit cells on a 4λ × 4λ aperture.
# RWG mesh:    Nmesh × Nmesh rectangles → 2·Nmesh² right triangles.
# Constraint:  Np ≤ Nmesh and Nmesh must be a multiple of Np, so each
#              printable cell contains (Nmesh/Np)² mesh rectangles whose
#              triangles share one θ_p.
#
# The per-rectangle θ_p is the physically meaningful design parameter that
# can be mapped to a printable sub-wavelength resonator on a substrate
# (patches, loops, capacitor–inductor elements, etc.).
#
# Run: julia --project=. examples/03_beamsteering_physical_unitcell.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra

println("="^60)
println("Example 03: Beam Steering with Physical Unit-Cell Grouping")
println("="^60)

# ── 1. Geometry ─────────────────────────────────────
freq  = 3e9
c0    = 299792458.0
λ0    = c0 / freq
k     = 2π / λ0
Lx = Ly = 4λ0                              # 4λ × 4λ plate
Nmesh = 12                                 # RWG mesh resolution
Np    = 12                                 # printable unit-cell grid
@assert Nmesh % Np == 0 "Nmesh must be a multiple of Np"

mesh = make_rect_plate(Lx, Ly, Nmesh, Nmesh)
rwg  = build_rwg(mesh)
N    = rwg.nedges
Nt   = ntriangles(mesh)

cell_size_λ = (Lx / Np) / λ0
println("\nFrequency: $(freq/1e9) GHz,  λ = $(round(λ0*1e3,digits=1)) mm")
println("Aperture: $(round(Lx/λ0,digits=1))λ × $(round(Ly/λ0,digits=1))λ")
println("RWG unknowns: $N,  triangles: $Nt")
println("Unit-cell grid: $Np × $Np  (cell size = $(round(cell_size_λ,digits=3))λ ≈ $(round(Lx/Np*1e3,digits=1)) mm)")

# ── 2. Map each triangle to its printable unit cell ─
# Triangle centroid → rectangular unit-cell index in 1…Np².
tri_cell = zeros(Int, Nt)
for t in 1:Nt
    cx = sum(mesh.xyz[1, mesh.tri[:, t]]) / 3
    cy = sum(mesh.xyz[2, mesh.tri[:, t]]) / 3
    ix = clamp(floor(Int, (cx + Lx/2) / (Lx/Np)) + 1, 1, Np)
    iy = clamp(floor(Int, (cy + Ly/2) / (Ly/Np)) + 1, 1, Np)
    tri_cell[t] = (iy - 1) * Np + ix
end
partition = PatchPartition(tri_cell, Np^2)

# Sanity check: every printable cell contains (Nmesh/Np)² · 2 triangles.
counts = zeros(Int, Np^2)
for t in 1:Nt; counts[tri_cell[t]] += 1; end
@assert all(counts .== 2*(Nmesh ÷ Np)^2) "cell/triangle binning mismatch: $(extrema(counts))"
println("Triangles per unit cell: $(2*(Nmesh ÷ Np)^2)  ✓")

# ── 3. Precompute ───────────────────────────────────
Mp     = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
Z_efie = assemble_Z_efie(mesh, rwg, k)
k_vec  = Vec3(0.0, 0.0, -k)
v      = assemble_excitation(mesh, rwg, make_plane_wave(k_vec, 1.0, Vec3(1.0, 0.0, 0.0)))

# ── 4. Q matrices (same target as paper: θ_s = 30°, φ = 0) ─
grid  = make_sph_grid(36, 72)
G_mat = radiation_vectors(mesh, rwg, grid, k)
pol   = pol_linear_x(grid)
Q_total = build_Q(G_mat, grid, pol)
θs, φs = deg2rad(30), 0.0
dir_target = Vec3(sin(θs)*cos(φs), sin(θs)*sin(φs), cos(θs))
mask_target = direction_mask(grid, dir_target; half_angle=deg2rad(5))
Q_target = build_Q(G_mat, grid, pol; mask=mask_target)

# ── 5. Phase-gradient init along x (same convention as paper) ───
θ_init = zeros(Np^2)
for iy in 1:Np, ix in 1:Np
    p = (iy - 1) * Np + ix
    cx = -Lx/2 + (ix - 0.5) * (Lx/Np)
    θ_init[p] = 300.0 * cx / (Lx/2 - Lx/(2Np))
end
println("Phase-gradient init: θ ∈ [$(round(minimum(θ_init),digits=1)), $(round(maximum(θ_init),digits=1))] Ω")

# ── 6. Run optimization ─────────────────────────────
println("\n── Optimizing ($(Np^2) DOF, reactive, box = [-500, +500] Ω) ──")
θ_opt, trace = optimize_directivity(
    Z_efie, Mp, Vector{ComplexF64}(v),
    Q_target, Q_total, θ_init;
    maxiter=300, tol=1e-12, alpha0=1e8, reactive=true,
    lb=-500.0, ub=+500.0, verbose=true)

# ── 7. Report ───────────────────────────────────────
Z_pec = assemble_full_Z(Z_efie, Mp, zeros(Np^2); reactive=true)
I_pec = Z_pec \ v
J_pec = real(dot(I_pec, Q_target * I_pec)) / real(dot(I_pec, Q_total * I_pec))

Z_opt = assemble_full_Z(Z_efie, Mp, θ_opt; reactive=true)
I_opt = Z_opt \ v
J_opt = real(dot(I_opt, Q_target * I_opt)) / real(dot(I_opt, Q_total * I_opt))

println("\n── Result ──")
println("Directivity ratio at target:")
println("  PEC baseline: $(round(J_pec*100, digits=3)) %")
println("  Optimized:    $(round(J_opt*100, digits=3)) %")
println("  Gain:         ×$(round(J_opt/J_pec, digits=1))")
println("θ_opt range:    [$(round(minimum(θ_opt),digits=1)), $(round(maximum(θ_opt),digits=1))] Ω")
println("Saturated cells (|θ|=500): $(count(abs.(θ_opt) .≥ 499.9)) / $(Np^2)")
println("For paper CSV artifacts, run validation/paper/run_beam_steering_case.jl")
