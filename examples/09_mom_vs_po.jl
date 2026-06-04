# 09_mom_vs_po.jl — MoM vs Physical Optics RCS comparison (flat plate)
#
# Demonstrates:
#   Flat plate — PO vs MoM vs analytical PO formula
#
# See also: 09a_aircraft_po.jl for aircraft multi-mesh comparison
#
# Run: julia --project=. examples/09_mom_vs_po.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra
using PlotlySupply

println("="^60)
println("Example 09: MoM vs Physical Optics (Flat Plate)")
println("="^60)

c0 = 299792458.0
figdir = joinpath(@__DIR__, "figs")
mkpath(figdir)

freq = 3e9
λ0   = c0 / freq
k    = 2π / λ0

Lx, Ly = 2λ0, 2λ0
Nx, Ny = 10, 10
mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg  = build_rwg(mesh)
N    = rwg.nedges

println("\nFrequency: $(freq/1e9) GHz, λ = $(round(λ0*100, digits=2)) cm")
println("Plate: $(round(Lx/λ0, digits=1))λ × $(round(Ly/λ0, digits=1))λ, $N RWG unknowns")

pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))

# Far-field grid for plate
grid_plate = make_sph_grid(37, 72)
NΩ_plate = length(grid_plate.w)

# PO
t_po = @elapsed po = solve_po(mesh, freq, pw; grid=grid_plate)
println("\nPO: $(ntriangles(mesh)) tri, $(count(po.illuminated)) illum, $(round(t_po, digits=3)) s")

# MoM
t_asm = @elapsed Z = assemble_Z_efie(mesh, rwg, k)
v = assemble_excitation(mesh, rwg, pw)
t_sol = @elapsed I_mom = Z \ v
G_mat = radiation_vectors(mesh, rwg, grid_plate, k)
E_ff_mom = compute_farfield(G_mat, Vector{ComplexF64}(I_mom), NΩ_plate)
println("MoM: $N unknowns, asm $(round(t_asm, digits=3)) s, solve $(round(t_sol, digits=3)) s")

σ_po_plate  = 10 .* log10.(max.(bistatic_rcs(po.E_ff; E0=1.0), 1e-30))
σ_mom_plate = 10 .* log10.(max.(bistatic_rcs(E_ff_mom; E0=1.0), 1e-30))

A_plate = Lx * Ly
σ_analytical_dB = 10 * log10(4π * A_plate^2 / λ0^2)
best_idx = argmax([grid_plate.rhat[3, q] for q in 1:NΩ_plate])

println("\n── Specular (θ ≈ 0°) ──")
println("  Analytical: $(round(σ_analytical_dB, digits=2)) dBsm")
println("  PO:         $(round(σ_po_plate[best_idx], digits=2)) dBsm")
println("  MoM:        $(round(σ_mom_plate[best_idx], digits=2)) dBsm")

# Flat plate plot — all points are φ=0, sorted by θ
let
    idx = sortperm(grid_plate.theta)
    θ_deg = rad2deg.(grid_plate.theta[idx])
    sf = subplots(1, 1; sync=false, width=800, height=500,
                  subplot_titles=reshape(["Flat Plate 2λ×2λ — φ=0° Cut (3 GHz, 1° resolution)"], 1, 1))
    addtraces!(sf, scatter(x=θ_deg, y=σ_po_plate[idx], mode="lines",
               name="PO", line=attr(color="blue", width=2)); row=1, col=1)
    addtraces!(sf, scatter(x=θ_deg, y=σ_mom_plate[idx], mode="lines",
               name="MoM (N=$N)", line=attr(color="red", width=2, dash="dash")); row=1, col=1)
    p = sf.plot
    relayout!(p, xaxis=attr(title="θ (deg)", range=[0, 90]),
              yaxis=attr(title="Bistatic RCS (dBsm)", range=[-40, 10]),
              legend=attr(x=0.65, y=0.95), margin=attr(l=60, r=30, t=60, b=50))
    savefig(p, joinpath(figdir, "09_flat_plate_rcs.png"); width=800, height=500)
end

println("\n" * "="^60)
println("Done.")
