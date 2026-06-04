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

function timed_min(f::Function; repeats::Int=2)
    vals = Float64[]
    for _ in 1:repeats
        push!(vals, @elapsed f())
    end
    return minimum(vals)
end

function main()
    println("Cost/scaling sweep")

    freq = 3e9
    λ = C0 / freq
    k = 2π / λ
    Lx = λ
    Ly = λ
    nx_list = [3, 4, 5, 6, 8]

    rows = DataFrame(
        Nx = Int[],
        N_rwg = Int[],
        Nt = Int[],
        assembly_s = Float64[],
        solve_s = Float64[],
        opt_iter_s = Float64[],
    )

    for Nx in nx_list
        Ny = Nx
        mesh = make_rect_plate(Lx, Ly, Nx, Ny)
        rwg = build_rwg(mesh)
        Nt = ntriangles(mesh)
        N = rwg.nedges

        println("  Nx=$Nx  N=$N")

        partition = PatchPartition(collect(1:Nt), Nt)
        Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
        tri_centers = [triangle_center(mesh, t) for t in 1:Nt]
        cx = [tc[1] for tc in tri_centers]
        x_center = (minimum(cx) + maximum(cx)) / 2
        x_halfspan = max((maximum(cx) - minimum(cx)) / 2, 1e-12)
        theta0 = [200.0 * (cx_t - x_center) / x_halfspan for cx_t in cx]

        k_vec = Vec3(0.0, 0.0, -k)
        pol_inc = Vec3(1.0, 0.0, 0.0)
        v = assemble_v_plane_wave(mesh, rwg, k_vec, 1.0, pol_inc; quad_order=3)

        grid = make_sph_grid(60, 24)
        G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=ETA0)
        pol_mat = pol_linear_x(grid)
        mask = cap_mask(grid; theta_max=deg2rad(35.0))
        Q_target = build_Q(G_mat, grid, pol_mat; mask=mask)
        Q_total = build_Q(G_mat, grid, pol_mat)

        Z_efie_ref = Ref{Matrix{ComplexF64}}()
        t_assembly = timed_min(repeats=2) do
            Z_efie_ref[] = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=ETA0)
        end
        Z_efie = Z_efie_ref[]

        t_solve = timed_min(repeats=3) do
            Z = assemble_full_Z(Z_efie, Mp, theta0; reactive=true)
            _ = Z \ v
        end

        t_opt_iter = timed_min(repeats=2) do
            optimize_directivity(
                Z_efie,
                Mp,
                v,
                Q_target,
                Q_total,
                theta0;
                maxiter=1,
                tol=0.0,
                alpha0=1e6,
                verbose=false,
                reactive=true,
                lb=-500.0,
                ub=500.0,
            )
        end

        push!(rows, (Nx, N, Nt, t_assembly, t_solve, t_opt_iter))
    end

    CSV.write(joinpath(DATADIR, "cost_scaling.csv"), rows)
    println("Saved data/cost_scaling.csv")
end

main()

