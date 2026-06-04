#!/usr/bin/env julia
# Export a checkerboard PEC/void mask on a 1.2λ cell for Meep cross-validation.
# This is the same checkerboard baseline used in the paper's multi-mode RCS-reduction
# demonstration (Section III-E).

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))

using LinearAlgebra
using StaticArrays
using Statistics
using JSON

include(joinpath(@__DIR__, "..", "..", "src", "DiffMoM.jl"))
using .DiffMoM

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATADIR = joinpath(PROJECT_ROOT, "data")
mkpath(DATADIR)

const C0 = 299792458.0

freq_ghz = 10.0
freq = freq_ghz * 1e9
lambda0 = C0 / freq
k = 2π / lambda0

dx_lambda = 1.2
dy_lambda = 1.2
Nx = 14
Ny = 14

dx_cell = dx_lambda * lambda0
dy_cell = dy_lambda * lambda0

# Build checkerboard mask: (ix+iy) odd → metal, even → void
# Same convention as 20_periodic_to_redistribution_demo.jl
mask = falses(Ny, Nx)
for jy in 1:Ny, jx in 1:Nx
    if isodd(jx + jy)
        mask[jy, jx] = true
    end
end

metal_fill = mean(Float64.(mask))
println("Checkerboard mask: $(Nx)×$(Ny), fill=$(metal_fill)")
println("Metal cells: $(sum(mask)) / $(Nx*Ny)")

# Build mesh with only metal triangles
tri_ids = Int[]
for jy in 1:Ny, jx in 1:Nx
    if mask[jy, jx]
        base = 2 * ((jy - 1) * Nx + (jx - 1)) + 1
        push!(tri_ids, base)
        push!(tri_ids, base + 1)
    end
end

mesh_full = make_rect_plate(dx_cell, dy_cell, Nx, Ny)
mesh = TriMesh(mesh_full.xyz, mesh_full.tri[:, tri_ids])
lattice = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k)
rwg = build_rwg_periodic(mesh, lattice; precheck=true, allow_boundary=true, require_closed=false)

println("  Triangles: $(ntriangles(mesh))")
println("  RWG edges: $(rwg.nedges)")
println("  Bloch-paired: $(rwg.has_periodic_bloch)")

# Solve periodic MoM
Z_per = assemble_Z_efie_periodic(mesh, rwg, k, lattice; quad_order=3)
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
v = Vector{ComplexF64}(assemble_excitation(mesh, rwg, pw; quad_order=3))
I = Z_per \ v

# Reflection coefficients
modes, R = reflection_coefficients(
    mesh, rwg, I, k, lattice;
    quad_order=3, N_orders=3, E0=1.0, pol=SVector(1.0, 0.0, 0.0))

idx00 = findfirst(m -> m.m == 0 && m.n == 0, modes)
R00 = R[idx00]

# Power balance (closure transmission)
N_rwg = rwg.nedges
Z_zero = zeros(ComplexF64, N_rwg, N_rwg)
pb = power_balance(
    Vector{ComplexF64}(I), Z_zero, dx_cell * dy_cell, k, modes, R;
    transmission=:closure, incident_order=(0, 0))

println("  |R₀₀| = $(round(abs(R00), sigdigits=4))")
println("  R_total = $(round(pb.refl_frac, sigdigits=4)) ($(round(100*pb.refl_frac, digits=2))%)")
println("  T_total = $(round(pb.trans_frac, sigdigits=4)) ($(round(100*pb.trans_frac, digits=2))%)")

# Export geometry JSON
output_prefix = "meep_checker_1p2lambda"
mask_row_major = [[mask[jy, jx] ? 1 : 0 for jx in 1:Nx] for jy in 1:Ny]

geometry_payload = Dict(
    "output_prefix" => output_prefix,
    "periodic_bc_model" => "bloch",
    "frequency_ghz" => freq_ghz,
    "lambda_m" => lambda0,
    "dx_cell_m" => dx_cell,
    "dy_cell_m" => dy_cell,
    "dx_lambda" => dx_lambda,
    "dy_lambda" => dy_lambda,
    "nx" => Nx,
    "ny" => Ny,
    "slot_wx_frac" => -1.0,  # not a slot — checkerboard
    "slot_wy_frac" => -1.0,
    "metal_fill_fraction" => metal_fill,
    "metal_mask_row_major" => mask_row_major,
)

# Mode-level data
mode_json = Dict{String,Any}[]
for i in eachindex(modes)
    mode = modes[i]
    kz_real = real(mode.kz)
    push!(mode_json, Dict(
        "m" => mode.m, "n" => mode.n,
        "propagating" => mode.propagating,
        "kz_real" => kz_real,
        "R_re" => real(R[i]), "R_im" => imag(R[i]),
        "R_abs2" => abs2(R[i]),
        "power_fraction" => mode.propagating ? abs2(R[i]) * kz_real / k : 0.0,
    ))
end

reference_payload = Dict(
    "output_prefix" => output_prefix,
    "periodic_bc_model" => "bloch",
    "has_periodic_bloch_rwg" => rwg.has_periodic_bloch,
    "frequency_ghz" => freq_ghz,
    "frequency_hz" => freq,
    "lambda_m" => lambda0,
    "wavenumber_rad_per_m" => k,
    "dx_cell_m" => dx_cell, "dy_cell_m" => dy_cell,
    "dx_lambda" => dx_lambda, "dy_lambda" => dy_lambda,
    "nx" => Nx, "ny" => Ny,
    "slot_wx_frac" => -1.0, "slot_wy_frac" => -1.0,
    "metal_fill_fraction" => metal_fill,
    "num_triangles" => ntriangles(mesh),
    "num_rwg" => rwg.nedges,
    "R00_re" => real(R00), "R00_im" => imag(R00), "R00_abs2" => abs2(R00),
    "refl_total_fraction" => pb.refl_frac,
    "trans_total_fraction" => pb.trans_frac,
    "abs_total_fraction" => pb.abs_frac,
    "modes" => mode_json,
)

geom_path = joinpath(DATADIR, "julia_$(output_prefix)_geometry.json")
ref_path = joinpath(DATADIR, "julia_$(output_prefix)_reference.json")

open(geom_path, "w") do io; write(io, JSON.json(geometry_payload, 2)); end
open(ref_path, "w") do io; write(io, JSON.json(reference_payload, 2)); end

println("Saved $(relpath(geom_path, PROJECT_ROOT))")
println("Saved $(relpath(ref_path, PROJECT_ROOT))")
