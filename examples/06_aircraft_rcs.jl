# 06_aircraft_rcs.jl — Aircraft OBJ mesh: import, repair, and RCS
#
# Demonstrates the full pipeline for imported meshes:
#   1. Read an OBJ mesh file
#   2. Repair (remove degenerate/non-manifold triangles, fix orientation)
#   3. Optionally coarsen to a target N
#   4. Check mesh resolution against frequency
#   5. Solve and compute bistatic + monostatic RCS
#
# Requires: examples/demo_aircraft.obj
#
# Run: julia --project=. examples/06_aircraft_rcs.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra

println("="^60)
println("Example 06: Aircraft RCS Pipeline")
println("="^60)

# ── 1. Load OBJ mesh ───────────────────────────────
obj_path = joinpath(@__DIR__, "demo_aircraft.obj")
if !isfile(obj_path)
    println("ERROR: $obj_path not found.")
    println("This example requires the demo_aircraft.obj file in the examples/ directory.")
    exit(1)
end

println("\nLoading: $obj_path")
mesh_raw = read_obj_mesh(obj_path)
println("  Raw mesh: $(nvertices(mesh_raw)) vertices, $(ntriangles(mesh_raw)) triangles")

# ── 2. Repair mesh ──────────────────────────────────
println("\nRepairing mesh...")
result = repair_mesh_for_simulation(mesh_raw;
    allow_boundary=true,
    auto_drop_nonmanifold=true,
)
mesh = result.mesh
println("  Removed invalid:      $(result.removed_invalid)")
println("  Removed degenerate:   $(result.removed_degenerate)")
println("  Removed non-manifold: $(result.removed_nonmanifold)")
println("  Flipped:              $(length(result.flipped_triangles))")
println("  Repaired mesh: $(nvertices(mesh)) vertices, $(ntriangles(mesh)) triangles")

# ── 3. Coarsen if needed ───────────────────────────
# For a quick demo, limit to ~200 RWG unknowns
target_rwg = 200
rwg_test = build_rwg(mesh)
if rwg_test.nedges > target_rwg * 1.5
    println("\nCoarsening to ~$target_rwg RWG unknowns...")
    coarsen_result = coarsen_mesh_to_target_rwg(mesh, target_rwg)
    mesh = coarsen_result.mesh
    println("  Coarsened: $(nvertices(mesh)) vertices, $(ntriangles(mesh)) triangles")
end

rwg = build_rwg(mesh)
N = rwg.nedges
println("\nFinal mesh: $N RWG unknowns")

# ── 4. Frequency and resolution check ──────────────
# Choose a frequency where the mesh is adequately resolved.
# Start with a low frequency and check resolution.
freq = 1e9                              # 1 GHz
c0 = 299792458.0
λ0 = c0 / freq
k  = 2π / λ0

res = mesh_resolution_report(mesh, freq)
println("\n── Resolution check at $(freq/1e9) GHz ──")
println("  Edge max/λ: $(round(res.edge_max_over_lambda, digits=3))  (target ≤ 0.1)")
println("  Meets target: $(res.meets_target)")

if !res.meets_target
    # Reduce frequency until mesh is adequate
    println("  Mesh is under-resolved. Reducing frequency...")
    # Target: edge_max / λ ≤ 0.1, so freq ≤ c0 * 0.1 / edge_max_m
    edge_max_m = res.edge_max_over_lambda * λ0
    freq = c0 * 0.08 / edge_max_m      # Target 0.08 for margin
    λ0 = c0 / freq
    k = 2π / λ0
    res = mesh_resolution_report(mesh, freq)
    println("  Adjusted to $(round(freq/1e9, digits=3)) GHz")
    println("  Edge max/λ: $(round(res.edge_max_over_lambda, digits=3))")
end

# ── 5. EFIE solve ──────────────────────────────────
println("\n── Solving at $(round(freq/1e9, digits=3)) GHz ($N unknowns) ──")

t_asm = @elapsed Z = assemble_Z_efie(mesh, rwg, k)
println("  Assembly: $(round(t_asm, digits=3)) s")

k_vec = Vec3(0.0, 0.0, -k)             # Nose-on incidence (-z)
v = assemble_excitation(mesh, rwg, make_plane_wave(k_vec, 1.0, Vec3(1.0, 0.0, 0.0)))

t_sol = @elapsed I_coeffs = Z \ v
residual = norm(Z * I_coeffs - v) / norm(v)
println("  Solve: $(round(t_sol, digits=3)) s, residual = $residual")

# ── 6. Far-field and RCS ───────────────────────────
grid = make_sph_grid(36, 72)
G_mat = radiation_vectors(mesh, rwg, grid, k)
NΩ = length(grid.w)
E_ff = compute_farfield(G_mat, Vector{ComplexF64}(I_coeffs), NΩ)

σ = bistatic_rcs(E_ff; E0=1.0)
σ_dB = 10 .* log10.(max.(σ, 1e-30))

println("\n── Bistatic RCS statistics ──")
println("  Min: $(round(minimum(σ_dB), digits=2)) dBsm")
println("  Max: $(round(maximum(σ_dB), digits=2)) dBsm")
println("  Mean: $(round(sum(σ_dB)/length(σ_dB), digits=2)) dBsm")

# Monostatic (backscatter) RCS
bs = backscatter_rcs(E_ff, grid, k_vec; E0=1.0)
println("  Backscatter (nose-on): $(round(10*log10(bs.sigma), digits=2)) dBsm")

# Energy conservation
P_in  = input_power(Vector{ComplexF64}(I_coeffs), Vector{ComplexF64}(v))
P_rad = radiated_power(E_ff, grid)
println("  P_rad/P_in = $(round(P_rad/P_in, digits=4))")

# ── 7. Condition diagnostics ───────────────────────
diag = condition_diagnostics(Z)
println("\n── Matrix conditioning ──")
println("  cond(Z) = $(round(diag.cond, sigdigits=4))")

println("\n" * "="^60)
println("Done.")
