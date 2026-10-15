# 07_pattern_feed.jl — Pattern-feed excitation from analytical dipole
#
# Demonstrates the PatternFeedExcitation workflow:
#   1. Create an analytical dipole (electric or magnetic)
#   2. Generate its far-field pattern on a (θ, φ) grid
#   3. Build a PatternFeedExcitation from the pattern
#   4. Compare the pattern-feed field against the analytical field
#   5. Solve a scattering problem using the pattern feed
#
# This validates the pattern import → interpolation → excitation pipeline
# without needing external CSV files.
#
# Run: julia --project=. examples/07_pattern_feed.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra

println("="^60)
println("Example 07: Pattern Feed Excitation")
println("="^60)

# ── 1. Create an analytical electric dipole ─────────
freq = 3e9
c0   = 299792458.0
λ0   = c0 / freq
k    = 2π / λ0

# z-directed electric dipole at the origin
dipole_pos = Vec3(0.0, 0.0, 0.3)       # 3λ above the plate
moment     = CVec3(0.0 + 0im, 0.0 + 0im, 1e-3 + 0im)  # electric moment, C·m
dipole = make_dipole(dipole_pos, moment, Vec3(0.0, 0.0, 1.0), :electric, freq)

println("\nDipole: z-directed electric, position = (0, 0, $(dipole_pos[3])) m")
println("Frequency: $(freq/1e9) GHz, λ = $(round(λ0*100, digits=2)) cm")

# ── 2. Generate pattern on (θ, φ) grid ─────────────
Nθ = 37
Nφ = 72
θ_grid = collect(range(0.0, π, length=Nθ))
φ_grid = collect(range(0.0, 2π - 2π/Nφ, length=Nφ))

println("Pattern grid: $Nθ × $Nφ = $(Nθ * Nφ) samples")

# Build PatternFeedExcitation from the analytical dipole
pat = make_analytic_dipole_pattern_feed(dipole, θ_grid, φ_grid)

# ── 3. Validate pattern-feed field vs analytical ────
println("\n── Pattern feed vs analytical dipole field ──")
# The pattern feed retains the radiating 1/r term. Near the z-directed
# dipole's axial null, omitted 1/r² and 1/r³ terms can dominate even at 3λ.
println("  Far-field pattern approximation; near-axis errors include omitted near fields.")
# Evaluate at a few test points on the plate
test_pts = [
    Vec3(0.02, 0.0, 0.0),
    Vec3(-0.03, 0.01, 0.0),
    Vec3(0.0, -0.04, 0.0),
    Vec3(0.01, 0.03, 0.0),
]

println("  Point              |E_pat|        |E_exact|      rel_err")
for r in test_pts
    E_pat   = pattern_feed_field(r, pat)
    E_exact = DiffMoM.dipole_incident_field(r, dipole)
    mag_pat   = norm(E_pat)
    mag_exact = norm(E_exact)
    rel_err = norm(E_pat - E_exact) / max(mag_exact, 1e-30)
    println("  ($(round(r[1]*100, digits=1)), $(round(r[2]*100, digits=1)), $(round(r[3]*100, digits=1))) cm  " *
            "$(round(mag_pat, sigdigits=4))   $(round(mag_exact, sigdigits=4))   $(round(rel_err, sigdigits=2))")
end

# ── 4. Scattering from a plate with pattern-feed ───
Lx, Ly = 0.1, 0.1                      # 1λ × 1λ plate at z=0
Nx, Ny = 5, 5
mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg  = build_rwg(mesh)
N    = rwg.nedges

println("\n── Scattering solve ──")
println("Plate: $(Lx*100) × $(Ly*100) cm, $N RWG unknowns")

Z = assemble_Z_efie(mesh, rwg, k)

# Excitation using the pattern feed
v_pat = assemble_excitation(mesh, rwg, pat)

# Also compute excitation using the direct dipole (for comparison)
v_dip = assemble_excitation(mesh, rwg, dipole)

v_diff = norm(v_pat - v_dip) / norm(v_dip)
println("  RHS difference (pattern vs direct dipole): $(round(v_diff, sigdigits=2))")

# Solve with pattern feed excitation
I_pat = Z \ v_pat
I_dip = Z \ v_dip

I_diff = norm(I_pat - I_dip) / norm(I_dip)
println("  Solution difference: $(round(I_diff, sigdigits=2))")

# ── 5. Far-field from pattern-feed solve ────────────
grid  = make_sph_grid(18, 36)
G_mat = radiation_vectors(mesh, rwg, grid, k)
NΩ    = length(grid.w)
E_ff  = compute_farfield(G_mat, Vector{ComplexF64}(I_pat), NΩ)

# A localized dipole has no uniform incident-field amplitude for RCS
# normalization. Report the scattered far-field coefficient r*E_s instead.
farfield_amplitude = [norm(field) for field in eachcol(E_ff)]

println("\n── Far-field ──")
println("  Max |r E_s| = $(round(maximum(farfield_amplitude), sigdigits=4)) V")
println("  Min |r E_s| = $(round(minimum(farfield_amplitude), sigdigits=4)) V")

# Energy check
P_in  = input_power(Vector{ComplexF64}(I_pat), Vector{ComplexF64}(v_pat))
P_rad = radiated_power(E_ff, grid)
println("  P_rad = $(round(P_rad, sigdigits=4)) W")
println("  P_rad/P_in = $(round(P_rad/P_in, digits=4))")

# ── 6. Also demonstrate plane-wave for comparison ──
println("\n── Comparison: same plate with plane-wave excitation ──")
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
v_pw = assemble_excitation(mesh, rwg, pw)
I_pw = Z \ v_pw
E_ff_pw = compute_farfield(G_mat, Vector{ComplexF64}(I_pw), NΩ)
P_in_pw  = input_power(Vector{ComplexF64}(I_pw), Vector{ComplexF64}(v_pw))
P_rad_pw = radiated_power(E_ff_pw, grid)
println("  Plane-wave P_rad/P_in = $(round(P_rad_pw/P_in_pw, digits=4))")

println("\n" * "="^60)
println("Done.")
