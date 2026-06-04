#!/usr/bin/env julia

using LinearAlgebra
using StaticArrays
using CSV
using DataFrames
using DiffMoM

const DATADIR = joinpath(@__DIR__, "..", "..", "data")
mkpath(DATADIR)

println("="^60)
println("Mesh Refinement Convergence and Conditioning Study")
println("="^60)

freq = 3e9
c0 = 299792458.0
lambda0 = c0 / freq
k = 2π / lambda0
eta0 = 376.730313668

Lx, Ly = 0.1, 0.1
mesh_sizes = [2, 3, 4, 5, 6, 8]

k_vec = Vec3(0.0, 0.0, -k)
E0 = 1.0
pol_inc = Vec3(1.0, 0.0, 0.0)

grid = make_sph_grid(16, 32)
NΩ = length(grid.w)
pol_mat = pol_linear_x(grid)

results = DataFrame(
    Nx = Int[],
    Nv = Int[],
    Nt = Int[],
    N_rwg = Int[],
    cond_Z = Float64[],
    P_rad = Float64[],
    P_in = Float64[],
    energy_ratio = Float64[],
    J_pec = Float64[],
    max_grad_err = Float64[],
)

println("\n── Running refinement sweep ──\n")

for Nx in mesh_sizes
    Ny = Nx

    mesh = make_rect_plate(Lx, Ly, Nx, Ny)
    rwg = build_rwg(mesh)
    N = rwg.nedges
    Nt = ntriangles(mesh)

    Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
    cond_info = condition_diagnostics(Z_efie)

    v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol_inc; quad_order=3)
    I_pec = Z_efie \ v

    G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
    E_ff = compute_farfield(G_mat, I_pec, NΩ)

    P_rad = radiated_power(E_ff, grid)
    P_in = input_power(I_pec, v)
    e_ratio = P_rad / P_in

    mask = cap_mask(grid; theta_max=30 * π / 180)
    Q = build_Q(G_mat, grid, pol_mat; mask=mask)
    J_pec = real(dot(I_pec, Q * I_pec))

    partition = PatchPartition(collect(1:Nt), Nt)
    Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
    theta_test = fill(200.0, Nt)
    Z_full = assemble_full_Z(Z_efie, Mp, theta_test)
    I_imp = Z_full \ v
    lambda = Z_full' \ (Q * I_imp)
    g_adj = gradient_impedance(Mp, I_imp, lambda)

    function J_of_theta(theta_vec)
        Z_t = copy(Z_efie)
        for p in eachindex(theta_vec)
            Z_t .-= theta_vec[p] .* Mp[p]
        end
        I_t = Z_t \ v
        return real(dot(I_t, Q * I_t))
    end

    n_check = min(5, Nt)
    max_gerr = 0.0
    for p in 1:n_check
        g_fd = fd_grad(J_of_theta, theta_test, p; h=1e-5)
        rel_err = abs(g_adj[p] - g_fd) / max(abs(g_adj[p]), 1e-30)
        max_gerr = max(max_gerr, rel_err)
    end

    push!(results, (
        Nx, nvertices(mesh), Nt, N, cond_info.cond, P_rad, P_in, e_ratio, J_pec, max_gerr
    ))

    println(
        "  Nx=$Nx: Nv=$(nvertices(mesh)), Nt=$Nt, N=$N, " *
        "cond=$(round(cond_info.cond, sigdigits=3)), " *
        "P_rad/P_in=$(round(e_ratio, sigdigits=4)), " *
        "grad_err=$(round(max_gerr, sigdigits=3))"
    )
end

CSV.write(joinpath(DATADIR, "convergence_study.csv"), results)

println("\n── Per-parameter gradient verification (Nx=3) ──\n")

let
    Nx_gv, Ny_gv = 3, 3
    mesh_gv = make_rect_plate(Lx, Ly, Nx_gv, Ny_gv)
    rwg_gv = build_rwg(mesh_gv)
    Nt_gv = ntriangles(mesh_gv)

    Z_efie_gv = assemble_Z_efie(mesh_gv, rwg_gv, k; quad_order=3, eta0=eta0)
    v_gv = assemble_v_plane_wave(mesh_gv, rwg_gv, k_vec, E0, pol_inc; quad_order=3)

    grid_gv = make_sph_grid(16, 32)
    pol_mat_gv = pol_linear_x(grid_gv)
    G_mat_gv = radiation_vectors(mesh_gv, rwg_gv, grid_gv, k; quad_order=3, eta0=eta0)
    mask_gv = cap_mask(grid_gv; theta_max=30 * π / 180)
    Q_gv = build_Q(G_mat_gv, grid_gv, pol_mat_gv; mask=mask_gv)

    partition_gv = PatchPartition(collect(1:Nt_gv), Nt_gv)
    Mp_gv = precompute_patch_mass(mesh_gv, rwg_gv, partition_gv; quad_order=3)
    theta_gv = fill(200.0, Nt_gv)

    Z_full_gv = assemble_full_Z(Z_efie_gv, Mp_gv, theta_gv)
    I_gv = Z_full_gv \ v_gv
    lambda_gv = Z_full_gv' \ (Q_gv * I_gv)
    g_adj_gv = gradient_impedance(Mp_gv, I_gv, lambda_gv)

    function J_of_theta_gv(theta_vec)
        Z_t = copy(Z_efie_gv)
        for p in eachindex(theta_vec)
            Z_t .-= theta_vec[p] .* Mp_gv[p]
        end
        I_t = Z_t \ v_gv
        return real(dot(I_t, Q_gv * I_t))
    end

    gv_results = DataFrame(
        param_idx = Int[],
        adjoint = Float64[],
        fd_central = Float64[],
        rel_error = Float64[],
    )

    for p in 1:min(10, Nt_gv)
        g_fd = fd_grad(J_of_theta_gv, theta_gv, p; h=1e-5)
        rel_err = abs(g_adj_gv[p] - g_fd) / max(abs(g_adj_gv[p]), 1e-30)
        push!(gv_results, (p, g_adj_gv[p], g_fd, rel_err))
        println(
            "  param $p: adjoint=$(round(g_adj_gv[p], sigdigits=6)), " *
            "fd=$(round(g_fd, sigdigits=6)), rel_err=$(round(rel_err, sigdigits=3))"
        )
    end

    CSV.write(joinpath(DATADIR, "gradient_verification.csv"), gv_results)
    println("\nGradient verification data saved to: $(joinpath(DATADIR, "gradient_verification.csv"))")
    println("Max rel error: $(round(maximum(gv_results.rel_error), sigdigits=3))")
end

println("\n── Summary ──")
println("  Condition numbers: $(round.(results.cond_Z, sigdigits=3))")
println("  Energy ratios:     $(round.(results.energy_ratio, sigdigits=4))")
println("  Max gradient errors: $(round.(results.max_grad_err, sigdigits=3))")

if length(results.cond_Z) >= 3
    h_vals = Lx ./ results.Nx
    log_h = log.(h_vals)
    log_kappa = log.(results.cond_Z)
    x_mean = sum(log_h) / length(log_h)
    y_mean = sum(log_kappa) / length(log_kappa)
    slope = sum((log_h .- x_mean) .* (log_kappa .- y_mean)) /
            sum((log_h .- x_mean) .^ 2)
    println("  Condition number growth rate: κ ~ h^($(round(slope, sigdigits=2)))")
end

println("\n" * "="^60)
println("CONVERGENCE STUDY COMPLETE")
println("="^60)
println("Data saved to: $(joinpath(DATADIR, "convergence_study.csv"))")
