# regen_mom_reference_table.jl — regenerate the MoM-side total reflectance
# for the five Meep cross-validation cases using the current DiffMoM periodic
# solver. The external FDTD (Meep) reflectances are preserved from the existing
# results_meep_validation.csv; only R_MoM (and the |ΔR| derived from it) is
# recomputed with the newest code.
#
# Run: julia --project=DiffMoM.jl DiffMoM.jl/validation/meep/regen_mom_reference_table.jl

using DiffMoM
using LinearAlgebra
using StaticArrays
using CSV, DataFrames

const C0 = 299792458.0
const PKG_DIR  = normpath(joinpath(@__DIR__, "..", ".."))
const DATA_DIR = joinpath(PKG_DIR, "..", "paper", "data")

function build_rect_slot_mask(Nx, Ny, dx_cell, dy_cell, slot_wx_frac, slot_wy_frac)
    mask = trues(Ny, Nx)
    dx_pix = dx_cell / Nx; dy_pix = dy_cell / Ny
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

function cell_triangle_indices(mask, Nx, Ny)
    tri_ids = Int[]
    for jy in 1:Ny, jx in 1:Nx
        if mask[jy, jx]
            base = 2 * ((jy - 1) * Nx + (jx - 1)) + 1
            push!(tri_ids, base); push!(tri_ids, base + 1)
        end
    end
    return tri_ids
end

function mom_total_reflectance(; dx_lambda, dy_lambda, Nx, Ny, slot_wx_frac, slot_wy_frac, freq_ghz=10.0)
    freq = freq_ghz * 1e9
    lambda0 = C0 / freq
    k = 2π / lambda0
    dx_cell = dx_lambda * lambda0
    dy_cell = dy_lambda * lambda0
    mask = build_rect_slot_mask(Nx, Ny, dx_cell, dy_cell, slot_wx_frac, slot_wy_frac)
    tri_ids = cell_triangle_indices(mask, Nx, Ny)
    mesh_full = make_rect_plate(dx_cell, dy_cell, Nx, Ny)
    mesh = TriMesh(mesh_full.xyz, mesh_full.tri[:, tri_ids])
    lattice = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k)
    rwg = build_rwg_periodic(mesh, lattice; precheck=true, allow_boundary=true, require_closed=false)
    Z = assemble_Z_efie_periodic(mesh, rwg, k, lattice; quad_order=3)
    pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
    v = Vector{ComplexF64}(assemble_excitation(mesh, rwg, pw; quad_order=3))
    I = Z \ v
    modes, R = reflection_coefficients(mesh, rwg, I, k, lattice; quad_order=3, N_orders=3, E0=1.0, pol=SVector(1.0, 0.0, 0.0))
    N = rwg.nedges
    pb = power_balance(Vector{ComplexF64}(I), zeros(ComplexF64, N, N), dx_cell * dy_cell, k, modes, R;
                       transmission=:closure, incident_order=(0, 0))
    return pb.refl_frac
end

# Five cross-validation cases. PEC plate is a full half-wavelength cell; the four
# slots are 1.2λ, 14×14 pixelated PEC/void unit cells with centered rectangular slots.
cases = [
    (name="PEC_plate_half_lambda", fill=100.0, dx=0.5, Nx=10, swx=0.0,  swy=0.0),
    (name="slot_wx0p15_wy0p20",    fill=97.0,  dx=1.2, Nx=14, swx=0.15, swy=0.20),
    (name="slot_wx0p25_wy0p20",    fill=95.0,  dx=1.2, Nx=14, swx=0.25, swy=0.20),
    (name="slot_wx0p40_wy0p20",    fill=92.0,  dx=1.2, Nx=14, swx=0.40, swy=0.20),
    (name="slot_wx0p50_wy0p50",    fill=75.0,  dx=1.2, Nx=14, swx=0.50, swy=0.50),
]

# Preserve external FDTD (Meep) reflectances from the existing table.
oldpath = joinpath(DATA_DIR, "results_meep_validation.csv")
old = CSV.read(oldpath, DataFrame)
meep_R = Dict(string(old.case[i]) => old.R_MEEP[i] for i in eachindex(old.case))

rows = DataFrame(case=String[], metal_fill_pct=Float64[], R_MoM=Float64[], R_MEEP=Float64[],
                 abs_diff_R=Float64[], T_MoM=Float64[], T_MEEP=Float64[], abs_diff_T=Float64[])
for c in cases
    R_mom = mom_total_reflectance(dx_lambda=c.dx, dy_lambda=c.dx, Nx=c.Nx, Ny=c.Nx, slot_wx_frac=c.swx, slot_wy_frac=c.swy)
    R_meep = get(meep_R, c.name, NaN)
    T_mom = 1.0 - R_mom
    T_meep = 1.0 - R_meep
    push!(rows, (c.name, c.fill, R_mom, R_meep, abs(R_mom - R_meep), T_mom, T_meep, abs(T_mom - T_meep)))
    println("  $(c.name): R_MoM=$(round(R_mom, digits=5))  R_MEEP=$(round(R_meep, digits=5))  |ΔR|=$(round(abs(R_mom-R_meep), digits=5))")
end

CSV.write(oldpath, rows)
println("\n✓ Updated $(oldpath)")
