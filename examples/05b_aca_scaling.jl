# 05b_aca_scaling.jl — ACA vs Dense scaling comparison
#
# Demonstrates the N-dependent crossover where ACA becomes faster than
# dense methods. Sweeps over increasing mesh sizes and reports timing.
#
# Run: julia --project=. examples/05b_aca_scaling.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra

println("="^60)
println("Example 05b: ACA vs Dense Scaling")
println("="^60)

freq = 3e9
c0   = 299792458.0
λ0   = c0 / freq
k    = 2π / λ0

# Sweep over increasing mesh sizes
# Plate size grows to keep resolution constant (~λ/5 edges)
configs = [
    (Nx=5,  Ny=5),    # small
    (Nx=10, Ny=10),   # medium
    (Nx=15, Ny=15),   # larger
    (Nx=20, Ny=20),   # large
    (Nx=25, Ny=25),   # bigger
    (Nx=30, Ny=30),   # even bigger
]

println("\n  N       Dense asm  Dense solve  ACA build  ACA precond  ACA solve  ACA total  Speedup")
println("  " * "─"^95)

for cfg in configs
    Lx = cfg.Nx * λ0 / 5               # Keep ~λ/5 element size
    Ly = cfg.Ny * λ0 / 5
    mesh = make_rect_plate(Lx, Ly, cfg.Nx, cfg.Ny)
    rwg = build_rwg(mesh)
    N = rwg.nedges

    k_vec = Vec3(0.0, 0.0, -k)
    v = assemble_excitation(mesh, rwg, make_plane_wave(k_vec, 1.0, Vec3(1.0, 0.0, 0.0)))

    # Dense direct
    t_dense_asm = @elapsed Z = assemble_Z_efie(mesh, rwg, k)
    t_dense_sol = @elapsed I_dense = Z \ v
    t_dense_total = t_dense_asm + t_dense_sol

    # ACA + NF-preconditioned GMRES (extract preconditioner from ACA dense blocks)
    t_aca_build = @elapsed A_aca = build_aca_operator(mesh, rwg, k;
        leaf_size=32, eta=1.5, aca_tol=1e-6, max_rank=50)
    t_aca_pre = @elapsed P_nf = build_nearfield_preconditioner(A_aca)
    t_aca_sol = @elapsed begin
        I_aca, stats = solve_gmres(A_aca, v; preconditioner=P_nf, tol=1e-6, maxiter=300)
    end
    t_aca_total = t_aca_build + t_aca_pre + t_aca_sol

    err = norm(I_aca - I_dense) / norm(I_dense)
    speedup = t_dense_total / t_aca_total
    iters = stats.niter
    n_dense_blk = length(A_aca.dense_blocks)
    n_lr_blk = length(A_aca.lowrank_blocks)

    println("  $(lpad(N, 5))   " *
            "$(lpad(round(t_dense_asm, digits=3), 8))s  " *
            "$(lpad(round(t_dense_sol, digits=3), 9))s  " *
            "$(lpad(round(t_aca_build, digits=3), 8))s  " *
            "$(lpad(round(t_aca_pre, digits=3), 9))s  " *
            "$(lpad(round(t_aca_sol, digits=3), 8))s  " *
            "$(lpad(round(t_aca_total, digits=3), 8))s  " *
            "$(lpad(round(speedup, digits=2), 6))×  " *
            "($(iters) iters, err=$(round(err, sigdigits=1)), $(n_dense_blk)d+$(n_lr_blk)lr)")
end

println("\n  Note: Speedup > 1× means ACA is faster than dense direct.")
println("  ACA overhead dominates at small N; crossover depends on hardware.")

println("\n" * "="^60)
println("Done.")
