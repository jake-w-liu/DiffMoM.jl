# generate_julia_po_aircraft.jl — Generate Julia PO results for comparison
#
# Computes PO RCS for demo_aircraft.obj at 0.3 GHz using DiffMoM.jl
# for comparison with POFacets 4.5

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
include(joinpath(@__DIR__, "..", "..", "src", "DiffMoM.jl"))
using .DiffMoM
using CSV, DataFrames

println("="^60)
println("Julia PO: demo_aircraft.obj at 0.3 GHz")
println("="^60)

# Load aircraft mesh
obj_path = joinpath(@__DIR__, "..", "..", "examples", "demo_aircraft.obj")
mesh_raw = read_obj_mesh(obj_path)
rep = repair_mesh_for_simulation(mesh_raw; allow_boundary=true, auto_drop_nonmanifold=true)
mesh = rep.mesh

# Parameters matching MATLAB script
c0 = 299792458.0
freq = 0.3e9
λ0 = c0 / freq
k = 2π / λ0

println("\nMesh: $(nvertices(mesh)) verts, $(ntriangles(mesh)) tri")
println("Frequency: $(freq/1e9) GHz, λ = $(round(λ0, digits=2)) m")

# Plane wave: -z incidence, x-polarized
# In spherical coords: θ=180°, φ=0°, θ-polarized
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))

# Observation grid: 1° resolution at φ=0° and φ=90°
function make_cut_grid_1deg(phi_values::Vector{Float64})
    Ntheta = 180
    dtheta = π / Ntheta
    Nphi = length(phi_values)
    NΩ = Ntheta * Nphi
    rhat  = zeros(3, NΩ)
    theta = zeros(NΩ)
    phi   = zeros(NΩ)
    w     = zeros(NΩ)
    idx = 0
    for it in 1:Ntheta
        θ = (it - 0.5) * dtheta
        for φ in phi_values
            idx += 1
            theta[idx] = θ
            phi[idx]   = φ
            rhat[1, idx] = sin(θ) * cos(φ)
            rhat[2, idx] = sin(θ) * sin(φ)
            rhat[3, idx] = cos(θ)
            w[idx] = sin(θ) * dtheta * (2π / Nphi)
        end
    end
    return SphGrid(rhat, theta, phi, w)
end

grid = make_cut_grid_1deg([0.0, π/2])
NΩ = length(grid.w)

println("Observation grid: $NΩ points (1° θ × φ=0°,90°)")

# Compute PO
println("\nComputing PO...")
t_po = @elapsed po = solve_po(mesh, freq, pw; grid=grid)

σ = bistatic_rcs(po.E_ff; E0=1.0)
σ_dB = 10 .* log10.(max.(σ, 1e-30))

println("  Illuminated facets: $(count(po.illuminated)) / $(ntriangles(mesh)) ($(round(100*count(po.illuminated)/ntriangles(mesh), digits=1))%)")
println("  PO solve: $(round(t_po, digits=2))s")

# Extract cuts
phi0_idx = [q for q in 1:NΩ if abs(grid.phi[q]) < 0.01]
phi90_idx = [q for q in 1:NΩ if abs(grid.phi[q] - π/2) < 0.01]

sort!(phi0_idx; by=q -> grid.theta[q])
sort!(phi90_idx; by=q -> grid.theta[q])

# Backscatter RCS
bs = backscatter_rcs(po.E_ff, grid, Vec3(0.0, 0.0, -k); E0=1.0)
bs_dB = round(10*log10(max(bs.sigma, 1e-30)), digits=2)
println("  Backscatter: $bs_dB dBsm")

# Save results
out_dir = joinpath(@__DIR__, "data")
mkpath(out_dir)

# φ=0° cut
df0 = DataFrame(
    theta_deg = rad2deg.(grid.theta[phi0_idx]),
    sigma_m2 = σ[phi0_idx],
    sigma_dBsm = σ_dB[phi0_idx]
)
out_file_0 = joinpath(out_dir, "julia_po_aircraft_0p3_phi0.csv")
CSV.write(out_file_0, df0)
println("\nSaved: $out_file_0")

# φ=90° cut
df90 = DataFrame(
    theta_deg = rad2deg.(grid.theta[phi90_idx]),
    sigma_m2 = σ[phi90_idx],
    sigma_dBsm = σ_dB[phi90_idx]
)
out_file_90 = joinpath(out_dir, "julia_po_aircraft_0p3_phi90.csv")
CSV.write(out_file_90, df90)
println("Saved: $out_file_90")

println("\n" * "="^60)
println("Done. Run compare_po_aircraft.m in MATLAB for comparison.")
println("="^60)
