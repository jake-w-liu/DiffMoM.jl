# validate_dielectric_mie_dda.jl -- dielectric sphere: 3D DDA vs exact Mie
#
# This validation targets the current 3D material-scattering path with existing
# far-field postprocessing: voxel DDA/VIE-style scattering from a homogeneous
# dielectric sphere in vacuum.
#
# Run:
#   julia --project=. validation/mie/validate_dielectric_mie_dda.jl
#
# Optional environment overrides:
#   DDA_MIE_NSIDE=13       Cartesian voxels per side, default 11
#   DDA_MIE_KA=0.6         target exterior size parameter k*a, default 0.6
#   DDA_MIE_NTHETA=73      theta samples per phi cut, default 73
#   DDA_MIE_EFFECTIVE_A=1  compare to equal-volume sphere, default 1

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
include(joinpath(@__DIR__, "..", "..", "src", "DiffMoM.jl"))

using .DiffMoM
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using Printf

function _env_int(name::AbstractString, default::Int)
    return haskey(ENV, name) ? parse(Int, ENV[name]) : default
end

function _env_float(name::AbstractString, default::Float64)
    return haskey(ENV, name) ? parse(Float64, ENV[name]) : default
end

function _env_bool(name::AbstractString, default::Bool)
    return haskey(ENV, name) ? ENV[name] in ("1", "true", "TRUE", "yes", "YES") : default
end

function _phi_cut_dirs(theta_vals, phi::Float64)
    return [Vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))
            for theta in theta_vals]
end

@inline _rcs_from_farfield(F) = 4π * real(dot(F, F))

function main()
println("="^72)
println("Dielectric Sphere Mie Validation: 3D DDA vs exact Mie")
println("="^72)

a_target = 0.05
ka_target = _env_float("DDA_MIE_KA", 0.6)
k0 = ka_target / a_target
lambda0 = 2π / k0
epsr = 2.5 - 0.02im
nside = _env_int("DDA_MIE_NSIDE", 11)
ntheta = _env_int("DDA_MIE_NTHETA", 73)
use_effective_radius = _env_bool("DDA_MIE_EFFECTIVE_A", true)

nside >= 3 || error("DDA_MIE_NSIDE must be >= 3")
ntheta >= 3 || error("DDA_MIE_NTHETA must be >= 3")

grid = VoxelGrid3D((-a_target, a_target),
                   (-a_target, a_target),
                   (-a_target, a_target),
                   nside, nside, nside)
epsv = ones(ComplexF64, grid.nvoxels)
inside = 0
voxel_volume = 0.0
for j in 1:grid.nvoxels
    if norm(grid.centers[j]) <= a_target
        epsv[j] = epsr
        inside += 1
        voxel_volume += grid.volumes[j]
    end
end
inside > 0 || error("No voxels selected inside dielectric sphere.")

a_effective = (3 * voxel_volume / (4π))^(1 / 3)
a_ref = use_effective_radius ? a_effective : a_target

println()
@printf("target radius:       %.6f m\n", a_target)
@printf("Mie reference radius: %.6f m%s\n", a_ref,
        use_effective_radius ? " (equal voxel volume)" : "")
@printf("ka target/ref:       %.4f / %.4f\n", ka_target, k0 * a_ref)
@printf("lambda0:             %.6f m\n", lambda0)
println("eps_r:               $epsr")
println("grid:                $nside x $nside x $nside voxels")
println("inside voxels:       $inside / $(grid.nvoxels)")
@printf("voxel volume ratio:  %.5f\n", voxel_volume / (4π * a_target^3 / 3))

khat = Vec3(0.0, 0.0, 1.0)
pol = Vec3(1.0, 0.0, 0.0)
E_inc = planewave_dda_3d(grid, k0 * khat, 1.0 + 0im, pol)

println("\nSolving DDA system with matrix-free GMRES...")
t_solve = @elapsed res = solve_dda_3d(grid, k0, epsv, E_inc;
                                      solver=:gmres,
                                      tol=1e-9,
                                      maxiter=250,
                                      memory=50)
println("solve time:          $(round(t_solve, digits=3)) s")
println("GMRES stats:         $(res.stats)")

theta = collect(range(0.0, π, length=ntheta))
theta_deg = rad2deg.(theta)

function _cut_metrics(phi::Float64)
    dirs = _phi_cut_dirs(theta, phi)
    sigma_dda = zeros(Float64, length(dirs))
    sigma_mie = zeros(Float64, length(dirs))
    for (i, rhat) in pairs(dirs)
        sigma_dda[i] = _rcs_from_farfield(farfield_dda_3d(res, rhat))
        sigma_mie[i] = mie_bistatic_rcs_dielectric(k0, a_ref, khat, pol, rhat, epsr)
    end
    dB_dda = 10 .* log10.(max.(sigma_dda, 1e-30))
    dB_mie = 10 .* log10.(max.(sigma_mie, 1e-30))
    delta = dB_dda .- dB_mie
    return sigma_dda, sigma_mie, dB_dda, dB_mie, delta
end

sigma0, mie0, dB0, dBm0, delta0 = _cut_metrics(0.0)
sigma90, mie90, dB90, dBm90, delta90 = _cut_metrics(π / 2)

idx_back = argmin(abs.(theta .- π))
idx_fwd = argmin(abs.(theta .- 0.0))

mae0 = mean(abs.(delta0))
rmse0 = sqrt(mean(abs2, delta0))
max0 = maximum(abs.(delta0))
mae90 = mean(abs.(delta90))
rmse90 = sqrt(mean(abs2, delta90))
max90 = maximum(abs.(delta90))
back_delta = delta0[idx_back]
fwd_delta = delta0[idx_fwd]

println("\n-- Phi = 0 deg cut --")
@printf("MAE/RMSE/max |delta|: %.3f / %.3f / %.3f dB\n", mae0, rmse0, max0)
@printf("forward/back delta:   %.3f / %.3f dB\n", fwd_delta, back_delta)

println("\n-- Phi = 90 deg cut --")
@printf("MAE/RMSE/max |delta|: %.3f / %.3f / %.3f dB\n", mae90, rmse90, max90)

outdir = @__DIR__
CSV.write(joinpath(outdir, "dielectric_mie_dda_phi0.csv"), DataFrame(
    theta_deg=theta_deg,
    rcs_dda_dBsm=dB0,
    rcs_mie_dBsm=dBm0,
    delta_dB=delta0,
    rcs_dda_m2=sigma0,
    rcs_mie_m2=mie0,
))
CSV.write(joinpath(outdir, "dielectric_mie_dda_phi90.csv"), DataFrame(
    theta_deg=theta_deg,
    rcs_dda_dBsm=dB90,
    rcs_mie_dBsm=dBm90,
    delta_dB=delta90,
    rcs_dda_m2=sigma90,
    rcs_mie_m2=mie90,
))
CSV.write(joinpath(outdir, "dielectric_mie_dda_summary.csv"), DataFrame(
    metric=[
        "target_radius_m", "reference_radius_m", "ka_target", "ka_reference",
        "epsr_real", "epsr_imag", "nside", "inside_voxels",
        "voxel_volume_ratio", "phi0_mae_dB", "phi0_rmse_dB",
        "phi0_max_delta_dB", "phi90_mae_dB", "phi90_rmse_dB",
        "phi90_max_delta_dB", "forward_delta_dB", "backscatter_delta_dB",
    ],
    value=[
        a_target, a_ref, ka_target, k0 * a_ref,
        real(epsr), imag(epsr), nside, inside,
        voxel_volume / (4π * a_target^3 / 3), mae0, rmse0,
        max0, mae90, rmse90, max90, fwd_delta, back_delta,
    ],
))
println("\nCSV data saved to $outdir")

pass = mae0 < 0.4 && rmse0 < 0.7 && max0 < 3.0 &&
       mae90 < 0.4 && rmse90 < 0.7 && max90 < 3.0 &&
       abs(back_delta) < 0.5

pass || error("Dielectric DDA-vs-Mie validation failed benchmark thresholds.")

println("\nPASS: dielectric DDA far field is consistent with exact Mie benchmark.")
println("="^72)
end

main()
