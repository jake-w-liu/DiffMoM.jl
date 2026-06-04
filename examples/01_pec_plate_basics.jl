# 01_pec_plate_basics.jl — PEC plate scattering: the minimal MoM workflow
#
# This example walks through every step of a basic MoM simulation:
#   1. Create a mesh
#   2. Build RWG basis functions
#   3. Assemble the EFIE impedance matrix
#   4. Define a plane-wave excitation
#   5. Solve for currents (Z I = v)
#   6. Compute far-field and RCS
#   7. Check energy conservation
#
# Run: julia --project=. examples/01_pec_plate_basics.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra

println("="^60)
println("Example 01: PEC Plate Basics")
println("="^60)

# ── 1. Problem parameters ─────────────────────────
freq = 3e9                          # 3 GHz
c0   = 299792458.0
lambda0 = c0 / freq                 # ≈ 10 cm
k    = 2π / lambda0
eta0 = 376.730313668

println("\nFrequency: $(freq/1e9) GHz,  λ = $(round(lambda0*100, digits=2)) cm")

# ── 2. Create mesh and RWG basis ──────────────────
Lx, Ly = 0.1, 0.1                  # 10 cm × 10 cm plate (1λ × 1λ)
Nx, Ny = 5, 5                      # ~λ/5 elements
mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg  = build_rwg(mesh)
N    = rwg.nedges

println("Mesh:  $(nvertices(mesh)) vertices, $(ntriangles(mesh)) triangles")
println("RWG:   $N basis functions")

# Check mesh resolution
res = mesh_resolution_report(mesh, freq)
println("Edge max/λ: $(round(res.edge_max_over_lambda, digits=3))  (target ≤ 0.1)")

# ── 3. Assemble EFIE matrix ──────────────────────
println("\nAssembling Z_efie ($N × $N)...")
t_asm = @elapsed Z = assemble_Z_efie(mesh, rwg, k)
println("  Done in $(round(t_asm, digits=3)) s")
println("  Memory: $(round(estimate_dense_matrix_gib(N)*1024, digits=2)) MiB")

# ── 4. Plane-wave excitation ──────────────────────
k_vec = Vec3(0.0, 0.0, -k)         # propagating in -z
E0    = 1.0
pol   = Vec3(1.0, 0.0, 0.0)        # x-polarized
v     = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol)

# ── 5. Solve Z I = v ─────────────────────────────
println("\nSolving...")
I_pec = Z \ v
residual = norm(Z * I_pec - v) / norm(v)
println("  Relative residual: $residual")

# ── 6. Far-field and RCS ──────────────────────────
grid = make_sph_grid(18, 36)        # 18 θ × 36 φ directions
G_mat = radiation_vectors(mesh, rwg, grid, k)
NΩ = length(grid.w)
E_ff = compute_farfield(G_mat, Vector{ComplexF64}(I_pec), NΩ)

# Bistatic RCS
sigma = bistatic_rcs(E_ff; E0=1.0)
println("\nBistatic RCS:")
println("  min σ = $(round(minimum(sigma)*1e4, digits=3)) cm²")
println("  max σ = $(round(maximum(sigma)*1e4, digits=3)) cm²")

# Monostatic (backscatter) RCS
bs = backscatter_rcs(E_ff, grid, k_vec; E0=1.0)
println("  Monostatic σ = $(round(10*log10(bs.sigma), digits=2)) dBsm")

# ── 7. Energy conservation ────────────────────────
P_in  = input_power(Vector{ComplexF64}(I_pec), Vector{ComplexF64}(v))
P_rad = radiated_power(E_ff, grid)
ratio = P_rad / P_in
println("\nEnergy conservation:")
println("  P_in  = $(round(P_in, sigdigits=4)) W")
println("  P_rad = $(round(P_rad, sigdigits=4)) W")
println("  P_rad/P_in = $(round(ratio, digits=4))  (should be ≈ 1 for PEC)")

# ── 8. Condition number ───────────────────────────
diag = condition_diagnostics(Z)
println("\nMatrix conditioning:")
println("  cond(Z) = $(round(diag.cond, sigdigits=4))")

println("\n" * "="^60)
println("Done.")
