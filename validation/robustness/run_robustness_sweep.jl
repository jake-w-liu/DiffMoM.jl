#!/usr/bin/env julia

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))

using LinearAlgebra
using StaticArrays
using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "..", "src", "DiffMoM.jl"))
using .DiffMoM

const DATADIR = joinpath(@__DIR__, "..", "..", "data")
mkpath(DATADIR)

const C0 = 299792458.0
const ETA0 = 376.730313668
const FREF = 3e9

function build_target_mask(grid, theta_target_deg::Float64, cone_deg::Float64, phi_target_deg::Float64)
    θ0 = deg2rad(theta_target_deg)
    ϕ0 = deg2rad(phi_target_deg)
    r0 = Vec3(sin(θ0) * cos(ϕ0), sin(θ0) * sin(ϕ0), cos(θ0))
    return BitVector([begin
        rh = Vec3(grid.rhat[:, q])
        angle = acos(clamp(dot(rh, r0), -1.0, 1.0))
        angle <= deg2rad(cone_deg)
    end for q in 1:length(grid.w)])
end

function mean_dir_at_theta(theta_vals::Vector{Float64}, dir_vals::Vector{Float64}, theta_target::Float64)
    θuniq = unique(theta_vals)
    θnear = θuniq[argmin(abs.(θuniq .- theta_target))]
    idx = findall(t -> abs(t - θnear) < 1e-12, theta_vals)
    return θnear, sum(dir_vals[idx]) / length(idx)
end

function main()
    println("Robustness sweep (frequency/angle perturbations)")

    # Geometry fixed to the designed physical aperture at the 3 GHz reference.
    λref = C0 / FREF
    Lx = 4 * λref
    Ly = 4 * λref
    Nx = Ny = 12

    mesh = make_rect_plate(Lx, Ly, Nx, Ny)
    rwg = build_rwg(mesh)
    Nt = ntriangles(mesh)
    partition = PatchPartition(collect(1:Nt), Nt)
    Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)

    imp_path = joinpath(DATADIR, "beam_steer_impedance.csv")
    isfile(imp_path) || error(
        "Missing $(basename(imp_path)). Run " *
        "`julia --project=. validation/paper/run_beam_steering_case.jl` first."
    )
    df_imp = CSV.read(imp_path, DataFrame)
    theta_opt = Vector{Float64}(df_imp.theta_opt)
    @assert length(theta_opt) == Nt

    grid = make_sph_grid(180, 72)
    mask = build_target_mask(grid, 30.0, 5.0, 0.0)

    cases = DataFrame(
        case = ["f_-2pct", "f_nom", "f_+2pct", "ang_-2deg", "ang_+2deg"],
        freq_GHz = [2.94, 3.00, 3.06, 3.00, 3.00],
        theta_inc_deg = [0.0, 0.0, 0.0, -2.0, 2.0],
    )

    J_opt = Float64[]
    J_pec = Float64[]
    gain_target_dB = Float64[]
    target_theta_deg = Float64[]
    peak_theta_opt_deg = Float64[]
    peak_opt_dBi = Float64[]

    for row in eachrow(cases)
        f = row.freq_GHz * 1e9
        θinc = deg2rad(row.theta_inc_deg)
        λ = C0 / f
        k = 2π / λ

        println("  Case $(row.case): f=$(row.freq_GHz) GHz, theta_inc=$(row.theta_inc_deg) deg")

        k_dir = Vec3(sin(θinc), 0.0, -cos(θinc))
        k_vec = k * k_dir
        pol = normalize(Vec3(cos(θinc), 0.0, sin(θinc))) # transverse to k_dir

        Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=ETA0)
        v = assemble_v_plane_wave(mesh, rwg, k_vec, 1.0, pol; quad_order=3)

        G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=ETA0)
        pol_mat = pol_linear_x(grid)
        Q_target = build_Q(G_mat, grid, pol_mat; mask=mask)
        Q_total = build_Q(G_mat, grid, pol_mat)

        I_pec = Z_efie \ v
        f_pec = real(dot(I_pec, Q_target * I_pec))
        g_pec = real(dot(I_pec, Q_total * I_pec))
        J_pec_case = f_pec / g_pec

        Z_opt = assemble_full_Z(Z_efie, Mp, theta_opt; reactive=true)
        I_opt = Z_opt \ v
        f_opt = real(dot(I_opt, Q_target * I_opt))
        g_opt = real(dot(I_opt, Q_total * I_opt))
        J_opt_case = f_opt / g_opt

        E_ff_pec = compute_farfield(G_mat, I_pec, length(grid.w))
        E_ff_opt = compute_farfield(G_mat, I_opt, length(grid.w))
        p_pec = [real(dot(E_ff_pec[:, q], E_ff_pec[:, q])) for q in 1:length(grid.w)]
        p_opt = [real(dot(E_ff_opt[:, q], E_ff_opt[:, q])) for q in 1:length(grid.w)]
        Ppec = sum(p_pec[q] * grid.w[q] for q in 1:length(grid.w))
        Popt = sum(p_opt[q] * grid.w[q] for q in 1:length(grid.w))
        Dpec = [4π * p_pec[q] / Ppec for q in 1:length(grid.w)]
        Dopt = [4π * p_opt[q] / Popt for q in 1:length(grid.w)]
        dir_pec = 10 .* log10.(max.(Dpec, 1e-30))
        dir_opt = 10 .* log10.(max.(Dopt, 1e-30))

        dphi = 2π / 72
        phi0_idx = [q for q in 1:length(grid.w) if min(grid.phi[q], 2π - grid.phi[q]) <= dphi / 2 + 1e-10]
        theta_cut = rad2deg.(grid.theta[phi0_idx])
        dir_pec_cut = dir_pec[phi0_idx]
        dir_opt_cut = dir_opt[phi0_idx]

        θtarget, pec_at_target = mean_dir_at_theta(theta_cut, dir_pec_cut, 30.0)
        _, opt_at_target = mean_dir_at_theta(theta_cut, dir_opt_cut, 30.0)
        gain = opt_at_target - pec_at_target

        # Peak of optimized pattern in phi~0 cut
        θuniq = unique(theta_cut)
        mean_opt_per_theta = [sum(dir_opt_cut[findall(t -> abs(t - θ) < 1e-12, theta_cut)]) /
                              length(findall(t -> abs(t - θ) < 1e-12, theta_cut)) for θ in θuniq]
        idx_peak = argmax(mean_opt_per_theta)

        push!(J_opt, J_opt_case * 100)
        push!(J_pec, J_pec_case * 100)
        push!(gain_target_dB, gain)
        push!(target_theta_deg, θtarget)
        push!(peak_theta_opt_deg, θuniq[idx_peak])
        push!(peak_opt_dBi, mean_opt_per_theta[idx_peak])
    end

    out = DataFrame(
        case = cases.case,
        freq_GHz = cases.freq_GHz,
        theta_inc_deg = cases.theta_inc_deg,
        J_opt_pct = J_opt,
        J_pec_pct = J_pec,
        gain_target_dB = gain_target_dB,
        target_theta_deg = target_theta_deg,
        peak_theta_opt_deg = peak_theta_opt_deg,
        peak_opt_dBi = peak_opt_dBi,
    )

    CSV.write(joinpath(DATADIR, "robustness_sweep.csv"), out)
    println("Saved data/robustness_sweep.csv")
end

main()
