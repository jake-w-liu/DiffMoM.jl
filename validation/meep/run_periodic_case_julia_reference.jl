#!/usr/bin/env julia

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))

using LinearAlgebra
using StaticArrays
using Statistics
using CSV
using DataFrames
using JSON

include(joinpath(@__DIR__, "..", "..", "src", "DiffMoM.jl"))
using .DiffMoM

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATADIR = joinpath(PROJECT_ROOT, "data")
mkpath(DATADIR)

const C0 = 299792458.0

function parse_float_flag(args::Vector{String}, name::String, default::Float64)
    for i in eachindex(args)
        if args[i] == name
            i < length(args) || error("Missing value for $(name)")
            return parse(Float64, args[i + 1])
        end
    end
    return default
end

function parse_int_flag(args::Vector{String}, name::String, default::Int)
    for i in eachindex(args)
        if args[i] == name
            i < length(args) || error("Missing value for $(name)")
            return parse(Int, args[i + 1])
        end
    end
    return default
end

function parse_string_flag(args::Vector{String}, name::String, default::String)
    for i in eachindex(args)
        if args[i] == name
            i < length(args) || error("Missing value for $(name)")
            return args[i + 1]
        end
    end
    return default
end

function build_rect_slot_mask(Nx::Int, Ny::Int, dx_cell::Float64, dy_cell::Float64,
                              slot_wx_frac::Float64, slot_wy_frac::Float64)
    mask = trues(Ny, Nx)
    dx_pix = dx_cell / Nx
    dy_pix = dy_cell / Ny

    slot_wx = clamp(slot_wx_frac, 0.0, 1.0) * dx_cell
    slot_wy = clamp(slot_wy_frac, 0.0, 1.0) * dy_cell

    for jy in 1:Ny
        y = -0.5 * dy_cell + (jy - 0.5) * dy_pix
        for jx in 1:Nx
            x = -0.5 * dx_cell + (jx - 0.5) * dx_pix
            in_slot = (abs(x) <= 0.5 * slot_wx) && (abs(y) <= 0.5 * slot_wy)
            mask[jy, jx] = !in_slot
        end
    end
    return mask
end

function cell_triangle_indices(mask::BitMatrix, Nx::Int, Ny::Int)
    tri_ids = Int[]
    for jy in 1:Ny
        for jx in 1:Nx
            if mask[jy, jx]
                base = 2 * ((jy - 1) * Nx + (jx - 1)) + 1
                push!(tri_ids, base)
                push!(tri_ids, base + 1)
            end
        end
    end
    return tri_ids
end

freq_ghz = parse_float_flag(ARGS, "--freq-ghz", 10.0)
freq = freq_ghz * 1e9
lambda0 = C0 / freq
k = 2π / lambda0

dx_lambda = parse_float_flag(ARGS, "--dx-lambda", 1.2)
dy_lambda = parse_float_flag(ARGS, "--dy-lambda", 1.2)
Nx = parse_int_flag(ARGS, "--nx", 24)
Ny = parse_int_flag(ARGS, "--ny", 24)

slot_wx_frac = parse_float_flag(ARGS, "--slot-wx-frac", 0.40)
slot_wy_frac = parse_float_flag(ARGS, "--slot-wy-frac", 0.20)
output_prefix = parse_string_flag(ARGS, "--output-prefix", "meep_periodic")
periodic_bc = lowercase(parse_string_flag(ARGS, "--periodic-bc", "bloch"))
periodic_bc == "bloch" ||
    error("Unsupported --periodic-bc=$(periodic_bc). Legacy mode has been removed; use 'bloch'.")

dx_cell = dx_lambda * lambda0
dy_cell = dy_lambda * lambda0

mask = build_rect_slot_mask(Nx, Ny, dx_cell, dy_cell, slot_wx_frac, slot_wy_frac)
tri_ids = cell_triangle_indices(mask, Nx, Ny)
isempty(tri_ids) && error("No metal triangles selected. Increase metal fill in the mask.")

mesh_full = make_rect_plate(dx_cell, dy_cell, Nx, Ny)
mesh = TriMesh(mesh_full.xyz, mesh_full.tri[:, tri_ids])
lattice = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k)
rwg = build_rwg_periodic(mesh, lattice; precheck=true, allow_boundary=true, require_closed=false)

println("Julia periodic reference for Meep cross-validation")
println("  frequency = $(freq_ghz) GHz")
println("  unit cell = $(dx_lambda)λ × $(dy_lambda)λ")
println("  pixels = $(Nx) × $(Ny)")
println("  slot fractions = ($(slot_wx_frac), $(slot_wy_frac))")
println("  periodic BC model = $(periodic_bc)")
if Nx < 14 || Ny < 14
    println("  WARNING: coarse nx/ny can under-resolve periodic currents and bias reflectance low.")
    println("           Recommended for Meep cross-validation: nx,ny >= 14.")
end
println("  selected triangles = $(ntriangles(mesh))")
println("  RWG basis count = $(rwg.nedges)")
println("  Bloch-paired RWG active = $(rwg.has_periodic_bloch)")

Z_per = assemble_Z_efie_periodic(mesh, rwg, k, lattice; quad_order=3)
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
v = Vector{ComplexF64}(assemble_excitation(mesh, rwg, pw; quad_order=3))
I = Z_per \ v

res = Z_per * I - v
rhs_l2 = norm(v)
res_l2 = norm(res)
res_rel = rhs_l2 > 0 ? res_l2 / rhs_l2 : NaN

modes, R = reflection_coefficients(
    mesh, rwg, I, k, lattice;
    quad_order=3,
    N_orders=3,
    E0=1.0,
    pol=SVector(1.0, 0.0, 0.0),
)

idx00 = findfirst(m -> m.m == 0 && m.n == 0, modes)
idx00 === nothing && error("Could not locate Floquet order (0,0).")
R00 = R[idx00]

N = rwg.nedges
Z_zero = zeros(ComplexF64, N, N)
# Closure-based transmission is conservative and guaranteed power-bounded.
pb_closure = power_balance(
    Vector{ComplexF64}(I),
    Z_zero,
    dx_cell * dy_cell,
    k,
    modes,
    R;
    transmission=:closure,
    incident_order=(0, 0),
)

# Keep Floquet-derived transmission as a diagnostic channel.
pb_floquet = power_balance(
    Vector{ComplexF64}(I),
    Z_zero,
    dx_cell * dy_cell,
    k,
    modes,
    R;
    transmission=:floquet,
    incident_order=(0, 0),
)

rows = DataFrame(
    m=Int[],
    n=Int[],
    propagating=Bool[],
    kz_real=Float64[],
    theta_deg=Float64[],
    phi_deg=Float64[],
    R_re=Float64[],
    R_im=Float64[],
    R_abs2=Float64[],
    power_fraction=Float64[],
)

mode_json = Dict{String,Any}[]
for i in eachindex(modes)
    mode = modes[i]
    kz_real = real(mode.kz)
    theta_deg = mode.propagating ? rad2deg(mode.theta_r) : NaN
    phi_deg = mode.propagating ? rad2deg(mode.phi_r) : NaN
    theta_deg_json = mode.propagating ? rad2deg(mode.theta_r) : nothing
    phi_deg_json = mode.propagating ? rad2deg(mode.phi_r) : nothing
    power_fraction = mode.propagating ? abs2(R[i]) * kz_real / k : 0.0

    push!(rows, (
        mode.m, mode.n, mode.propagating, kz_real, theta_deg, phi_deg,
        real(R[i]), imag(R[i]), abs2(R[i]), power_fraction
    ))

    push!(mode_json, Dict(
        "m" => mode.m,
        "n" => mode.n,
        "propagating" => mode.propagating,
        "kz_real" => kz_real,
        "theta_deg" => theta_deg_json,
        "phi_deg" => phi_deg_json,
        "R_re" => real(R[i]),
        "R_im" => imag(R[i]),
        "R_abs2" => abs2(R[i]),
        "power_fraction" => power_fraction,
    ))
end

metal_fill_fraction = mean(Float64.(mask))
geometry_json_path = joinpath(DATADIR, "julia_$(output_prefix)_geometry.json")
reference_json_path = joinpath(DATADIR, "julia_$(output_prefix)_reference.json")
modes_csv_path = joinpath(DATADIR, "julia_$(output_prefix)_modes.csv")
mask_row_major = [[mask[jy, jx] ? 1 : 0 for jx in 1:Nx] for jy in 1:Ny]

geometry_payload = Dict(
    "output_prefix" => output_prefix,
    "periodic_bc_model" => periodic_bc,
    "frequency_ghz" => freq_ghz,
    "lambda_m" => lambda0,
    "dx_cell_m" => dx_cell,
    "dy_cell_m" => dy_cell,
    "dx_lambda" => dx_lambda,
    "dy_lambda" => dy_lambda,
    "nx" => Nx,
    "ny" => Ny,
    "slot_wx_frac" => slot_wx_frac,
    "slot_wy_frac" => slot_wy_frac,
    "metal_fill_fraction" => metal_fill_fraction,
    "metal_mask_row_major" => mask_row_major,
)

reference_payload = Dict(
    "output_prefix" => output_prefix,
    "periodic_bc_model" => periodic_bc,
    "has_periodic_bloch_rwg" => rwg.has_periodic_bloch,
    "frequency_ghz" => freq_ghz,
    "frequency_hz" => freq,
    "lambda_m" => lambda0,
    "wavenumber_rad_per_m" => k,
    "dx_cell_m" => dx_cell,
    "dy_cell_m" => dy_cell,
    "dx_lambda" => dx_lambda,
    "dy_lambda" => dy_lambda,
    "nx" => Nx,
    "ny" => Ny,
    "slot_wx_frac" => slot_wx_frac,
    "slot_wy_frac" => slot_wy_frac,
    "metal_fill_fraction" => metal_fill_fraction,
    "num_triangles" => ntriangles(mesh),
    "num_rwg" => rwg.nedges,
    "rhs_l2_norm" => rhs_l2,
    "solve_residual_l2_abs" => res_l2,
    "solve_residual_l2_rel" => res_rel,
    "R00_re" => real(R00),
    "R00_im" => imag(R00),
    "R00_abs2" => abs2(R00),
    "refl_total_fraction" => pb_closure.refl_frac,
    "trans_total_fraction" => pb_closure.trans_frac,
    "abs_total_fraction" => pb_closure.abs_frac,
    "resid_total_fraction" => pb_closure.resid_frac,
    "trans_total_fraction_closure" => pb_closure.trans_frac,
    "resid_total_fraction_closure" => pb_closure.resid_frac,
    "trans_total_fraction_floquet" => pb_floquet.trans_frac,
    "resid_total_fraction_floquet" => pb_floquet.resid_frac,
    "modes" => mode_json,
)

open(geometry_json_path, "w") do io
    write(io, JSON.json(geometry_payload, 2))
end

open(reference_json_path, "w") do io
    write(io, JSON.json(reference_payload, 2))
end

CSV.write(modes_csv_path, rows)

println("Saved $(relpath(geometry_json_path, PROJECT_ROOT))")
println("Saved $(relpath(reference_json_path, PROJECT_ROOT))")
println("Saved $(relpath(modes_csv_path, PROJECT_ROOT))")
