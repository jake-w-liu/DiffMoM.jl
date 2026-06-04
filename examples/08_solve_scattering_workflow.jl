# 08_solve_scattering_workflow.jl — High-level solve_scattering API
#
# Demonstrates the `solve_scattering` workflow function which:
#   1. Validates mesh resolution against frequency
#   2. Auto-selects solver method (dense_direct / dense_gmres / aca_gmres)
#   3. Handles preconditioner setup automatically
#   4. Returns a ScatteringResult with all metadata
#
# This is the recommended entry point for production simulations.
#
# Run: julia --project=. examples/08_solve_scattering_workflow.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra

println("="^60)
println("Example 08: solve_scattering Workflow API")
println("="^60)

# ── 1. Setup ────────────────────────────────────────
freq = 3e9
c0   = 299792458.0
λ0   = c0 / freq
k    = 2π / λ0

Lx, Ly = 0.15, 0.15                    # 1.5λ × 1.5λ plate
Nx, Ny = 8, 8
mesh = make_rect_plate(Lx, Ly, Nx, Ny)

println("\nFrequency: $(freq/1e9) GHz, λ = $(round(λ0*100, digits=2)) cm")
println("Plate: $(Lx*100) × $(Ly*100) cm")

# Create excitation
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))

# ── 2. Auto method selection ───────────────────────
println("\n" * "─"^60)
println("Case 1: Auto method selection")
println("─"^60)

result = solve_scattering(mesh, freq, pw; verbose=true)

println("\n  Method selected: $(result.method)")
println("  N = $(result.N) unknowns")
println("  Assembly: $(round(result.assembly_time_s, digits=3)) s")
println("  Solve:    $(round(result.solve_time_s, digits=3)) s")
if result.gmres_iters >= 0
    println("  GMRES iters: $(result.gmres_iters)")
end
println("  Warnings: $(isempty(result.warnings) ? "none" : join(result.warnings, "; "))")

# ── 3. Force dense_gmres with NF preconditioner ────
println("\n" * "─"^60)
println("Case 2: Forced dense_gmres with NF preconditioner")
println("─"^60)

result_gmres = solve_scattering(mesh, freq, pw;
    method=:dense_gmres,
    nf_cutoff_lambda=1.0,
    preconditioner=:lu,
    gmres_tol=1e-8,
    verbose=true,
)

println("\n  Method: $(result_gmres.method)")
println("  Assembly: $(round(result_gmres.assembly_time_s, digits=3)) s")
println("  Precond:  $(round(result_gmres.preconditioner_time_s, digits=3)) s")
println("  Solve:    $(round(result_gmres.solve_time_s, digits=3)) s")
println("  GMRES iters: $(result_gmres.gmres_iters)")

# ── 4. Force ACA GMRES ─────────────────────────────
println("\n" * "─"^60)
println("Case 3: Forced aca_gmres")
println("─"^60)

result_aca = solve_scattering(mesh, freq, pw;
    method=:aca_gmres,
    nf_cutoff_lambda=1.0,
    aca_tol=1e-6,
    aca_leaf_size=32,
    verbose=true,
)

println("\n  Method: $(result_aca.method)")
println("  Assembly: $(round(result_aca.assembly_time_s, digits=3)) s")
println("  Precond:  $(round(result_aca.preconditioner_time_s, digits=3)) s")
println("  Solve:    $(round(result_aca.solve_time_s, digits=3)) s")
println("  GMRES iters: $(result_aca.gmres_iters)")

# ── 5. Compare solutions ───────────────────────────
err_gmres = norm(result_gmres.I_coeffs - result.I_coeffs) / norm(result.I_coeffs)
err_aca   = norm(result_aca.I_coeffs - result.I_coeffs) / norm(result.I_coeffs)

println("\n" * "─"^60)
println("Solution comparison (vs dense direct)")
println("─"^60)
println("  Dense GMRES error: $(round(err_gmres, sigdigits=2))")
println("  ACA GMRES error:   $(round(err_aca, sigdigits=2))")

# ── 6. Pre-assembled excitation vector ──────────────
println("\n" * "─"^60)
println("Case 4: Pre-assembled excitation vector")
println("─"^60)

# You can also pass a pre-built v vector directly
rwg = build_rwg(mesh)
v_pre = assemble_excitation(mesh, rwg, pw)

result_pre = solve_scattering(mesh, freq, v_pre; verbose=true)
err_pre = norm(result_pre.I_coeffs - result.I_coeffs) / norm(result.I_coeffs)
println("  Error vs auto: $(round(err_pre, sigdigits=2))")

# ── 7. Mesh resolution warning ─────────────────────
println("\n" * "─"^60)
println("Case 5: Mesh resolution warning")
println("─"^60)

# Use a high frequency where the mesh is under-resolved
freq_high = 30e9
println("Solving at $(freq_high/1e9) GHz (mesh will be under-resolved)...")
result_warn = solve_scattering(mesh, freq_high, pw;
    verbose=true,
    error_on_underresolved=false,
)
if !isempty(result_warn.warnings)
    println("  Warnings captured: $(length(result_warn.warnings))")
    for w in result_warn.warnings
        println("    - $w")
    end
end

# ── 8. Far-field from workflow result ───────────────
println("\n" * "─"^60)
println("Post-processing the workflow result")
println("─"^60)

# The result gives you I_coeffs; combine with standard far-field tools
rwg = build_rwg(mesh)
grid = make_sph_grid(18, 36)
G_mat = radiation_vectors(mesh, rwg, grid, k)
NΩ = length(grid.w)
E_ff = compute_farfield(G_mat, result.I_coeffs, NΩ)

σ = bistatic_rcs(E_ff; E0=1.0)
P_in  = input_power(result.I_coeffs, Vector{ComplexF64}(v_pre))
P_rad = radiated_power(E_ff, grid)

println("  Bistatic RCS max: $(round(maximum(10 .* log10.(max.(σ, 1e-30))), digits=2)) dBsm")
println("  P_rad/P_in = $(round(P_rad/P_in, digits=4))")

println("\n" * "="^60)
println("Done.")
