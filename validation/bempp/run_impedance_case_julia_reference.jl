#!/usr/bin/env julia

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))

using LinearAlgebra
using StaticArrays
using CSV
using DataFrames
using JSON

include(joinpath(@__DIR__, "..", "..", "src", "DiffMoM.jl"))
using .DiffMoM

const DATADIR = joinpath(@__DIR__, "..", "..", "data")
mkpath(DATADIR)

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

freq_ghz = parse_float_flag(ARGS, "--freq-ghz", 3.0)
freq = freq_ghz * 1e9
c0 = 299792458.0
lambda0 = c0 / freq
k = 2π / lambda0
eta0 = 376.730313668

Lx = 4 * lambda0
Ly = 4 * lambda0
Nx, Ny = 12, 12

theta_uniform = parse_float_flag(ARGS, "--theta-ohm", 200.0)  # Zs = i*theta_uniform [ohm]
n_theta = parse_int_flag(ARGS, "--n-theta", 180)
n_phi = parse_int_flag(ARGS, "--n-phi", 72)
theta_inc_deg = parse_float_flag(ARGS, "--theta-inc-deg", 0.0)
phi_inc_deg = parse_float_flag(ARGS, "--phi-inc-deg", 0.0)
output_prefix = parse_string_flag(ARGS, "--output-prefix", "impedance")

theta_inc = deg2rad(theta_inc_deg)
phi_inc = deg2rad(phi_inc_deg)

# Propagation direction points toward -z at normal incidence.
k_hat = Vec3(
    sin(theta_inc) * cos(phi_inc),
    sin(theta_inc) * sin(phi_inc),
    -cos(theta_inc),
)

# Choose theta-polarization unit vector so E ⟂ k_hat.
pol_inc = normalize(Vec3(
    cos(theta_inc) * cos(phi_inc),
    cos(theta_inc) * sin(phi_inc),
    sin(theta_inc),
))
k_vec = k * k_hat

mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg = build_rwg(mesh)
Nt = ntriangles(mesh)
N = rwg.nedges

println("Julia impedance-reference case")
println("  f = $(freq_ghz) GHz")
println("  theta_inc = $(theta_inc_deg) deg, phi_inc = $(phi_inc_deg) deg")
println("  Zs = i*$(theta_uniform) ohm")
println("  Mesh: N=$N RWG, Nt=$Nt triangles")

Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
partition = PatchPartition(collect(1:Nt), Nt)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)

theta_vec = fill(theta_uniform, Nt)
Z_imp = assemble_full_Z(Z_efie, Mp, theta_vec; reactive=true)

E0 = 1.0
v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol_inc; quad_order=3)
I_imp = Z_imp \ v

n_hat = Vec3(0.0, 0.0, 1.0)
pol_tan_raw = pol_inc - dot(pol_inc, n_hat) * n_hat
pol_tan = norm(pol_tan_raw) > 1e-12 ? normalize(pol_tan_raw) : Vec3(1.0, 0.0, 0.0)

tri_basis = [Int[] for _ in 1:Nt]
for n in 1:N
    push!(tri_basis[rwg.tplus[n]], n)
    push!(tri_basis[rwg.tminus[n]], n)
end

tri_id = Vector{Int}(undef, Nt)
x_m = Vector{Float64}(undef, Nt)
y_m = Vector{Float64}(undef, Nt)
z_m = Vector{Float64}(undef, Nt)
Jx_re = Vector{Float64}(undef, Nt)
Jx_im = Vector{Float64}(undef, Nt)
Jy_re = Vector{Float64}(undef, Nt)
Jy_im = Vector{Float64}(undef, Nt)
Jz_re = Vector{Float64}(undef, Nt)
Jz_im = Vector{Float64}(undef, Nt)
J_mag = Vector{Float64}(undef, Nt)
J_phase_deg = Vector{Float64}(undef, Nt)

for t in 1:Nt
    tri_id[t] = t
    rc = triangle_center(mesh, t)
    x_m[t] = rc[1]
    y_m[t] = rc[2]
    z_m[t] = rc[3]

    J = zeros(ComplexF64, 3)
    for n in tri_basis[t]
        fn = eval_rwg(rwg, n, rc, t)
        J[1] += I_imp[n] * fn[1]
        J[2] += I_imp[n] * fn[2]
        J[3] += I_imp[n] * fn[3]
    end

    Jx_re[t] = real(J[1]); Jx_im[t] = imag(J[1])
    Jy_re[t] = real(J[2]); Jy_im[t] = imag(J[2])
    Jz_re[t] = real(J[3]); Jz_im[t] = imag(J[3])

    Jmag = sqrt(abs2(J[1]) + abs2(J[2]) + abs2(J[3]))
    J_mag[t] = Jmag
    Jproj = pol_tan[1] * J[1] + pol_tan[2] * J[2] + pol_tan[3] * J[3]
    J_phase_deg[t] = Jmag > 1e-15 ? rad2deg(angle(Jproj)) : NaN
end

df_curr = DataFrame(
    tri_id = tri_id,
    x_m = x_m,
    y_m = y_m,
    z_m = z_m,
    Jx_re = Jx_re, Jx_im = Jx_im,
    Jy_re = Jy_re, Jy_im = Jy_im,
    Jz_re = Jz_re, Jz_im = Jz_im,
    J_mag = J_mag,
    J_phase_deg = J_phase_deg,
)

res = Z_imp * I_imp - v
rhs_l2 = norm(v)
res_l2 = norm(res)
res_rel = rhs_l2 > 0 ? res_l2 / rhs_l2 : NaN

operator_checks = Dict(
    "frequency_ghz" => freq_ghz,
    "theta_ohm" => theta_uniform,
    "theta_inc_deg" => theta_inc_deg,
    "phi_inc_deg" => phi_inc_deg,
    "num_rwg" => N,
    "num_triangles" => Nt,
    "rhs_l2_norm" => rhs_l2,
    "solve_residual_l2_abs" => res_l2,
    "solve_residual_l2_rel" => res_rel,
    "current_coeff_l2_norm" => norm(I_imp),
)

grid = make_sph_grid(n_theta, n_phi)
NΩ = length(grid.w)
G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
E_ff_imp = compute_farfield(G_mat, I_imp, NΩ)

ff_power_imp = [real(dot(E_ff_imp[:, q], E_ff_imp[:, q])) for q in 1:NΩ]
P_sphere_imp = sum(ff_power_imp[q] * grid.w[q] for q in 1:NΩ)
D_imp = [4π * ff_power_imp[q] / P_sphere_imp for q in 1:NΩ]
dir_imp_dBi = 10 .* log10.(max.(D_imp, 1e-30))

theta_steer = 30.0 * π / 180
phi_steer = 0.0
steer_rhat = Vec3(sin(theta_steer) * cos(phi_steer),
                  sin(theta_steer) * sin(phi_steer),
                  cos(theta_steer))
in_target = BitVector([begin
    rh = Vec3(grid.rhat[:, q])
    angle = acos(clamp(dot(rh, steer_rhat), -1.0, 1.0))
    angle <= 5.0 * π / 180
end for q in 1:NΩ])

df_ff = DataFrame(
    theta_deg = rad2deg.(grid.theta),
    phi_deg = rad2deg.(grid.phi),
    dir_julia_imp_dBi = dir_imp_dBi,
    in_target = in_target,
)
CSV.write(joinpath(DATADIR, "julia_$(output_prefix)_farfield.csv"), df_ff)

dphi = 2π / n_phi
phi0_idx = [q for q in 1:NΩ if min(grid.phi[q], 2π - grid.phi[q]) <= dphi / 2 + 1e-10]
df_cut = DataFrame(
    theta_deg = rad2deg.(grid.theta[phi0_idx]),
    dir_julia_imp_dBi = dir_imp_dBi[phi0_idx],
)
CSV.write(joinpath(DATADIR, "julia_$(output_prefix)_cut_phi0.csv"), df_cut)
CSV.write(joinpath(DATADIR, "julia_$(output_prefix)_element_currents.csv"), df_curr)

open(joinpath(DATADIR, "julia_$(output_prefix)_operator_checks.json"), "w") do io
    JSON.print(io, operator_checks, 4)
end

println("Saved data/julia_$(output_prefix)_farfield.csv")
println("Saved data/julia_$(output_prefix)_cut_phi0.csv")
println("Saved data/julia_$(output_prefix)_element_currents.csv")
println("Saved data/julia_$(output_prefix)_operator_checks.json")
