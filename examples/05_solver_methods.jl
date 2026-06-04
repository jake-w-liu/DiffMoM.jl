# 05_solver_methods.jl — Solver comparison: Direct LU vs GMRES vs ACA
#
# Compares the three solver strategies on the same PEC plate problem:
#   1. Dense direct (LU factorization)
#   2. Dense GMRES with near-field preconditioner
#   3. ACA H-matrix + NF-preconditioned GMRES
#
# Also demonstrates:
#   - Near-field preconditioner construction and sparsity
#   - GMRES iteration count scaling
#   - ACA compression ratio
#   - Solution accuracy comparison
#
# Run: julia --project=. examples/05_solver_methods.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra

println("="^60)
println("Example 05: Solver Methods Comparison")
println("="^60)

# ── 1. Problem setup ────────────────────────────────
freq  = 3e9
c0    = 299792458.0
λ0    = c0 / freq
k     = 2π / λ0

# Use a larger plate to make iterative methods worthwhile
Lx, Ly = 0.3, 0.3                      # 3λ × 3λ
Nx, Ny = 15, 15
mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg  = build_rwg(mesh)
N    = rwg.nedges

println("\nFrequency: $(freq/1e9) GHz, λ = $(round(λ0*100, digits=2)) cm")
println("Plate: $(round(Lx/λ0, digits=1))λ × $(round(Ly/λ0, digits=1))λ")
println("N = $N RWG unknowns")
println("Dense matrix: $(round(estimate_dense_matrix_gib(N)*1024, digits=2)) MiB")

k_vec = Vec3(0.0, 0.0, -k)
v = assemble_excitation(mesh, rwg, make_plane_wave(k_vec, 1.0, Vec3(1.0, 0.0, 0.0)))

# ── 2. Method 1: Dense Direct (LU) ─────────────────
println("\n── Method 1: Dense Direct (LU) ──")
t_asm1 = @elapsed Z = assemble_Z_efie(mesh, rwg, k)
println("  Assembly: $(round(t_asm1, digits=3)) s")

t_sol1 = @elapsed I_direct = Z \ v
res1 = norm(Z * I_direct - v) / norm(v)
println("  Solve:    $(round(t_sol1, digits=3)) s")
println("  Residual: $res1")

# ── 3. Method 2: Dense GMRES + NF Preconditioner ───
println("\n── Method 2: Dense GMRES + NF Preconditioner ──")

# Build near-field preconditioner from the dense matrix
cutoff = 1.0 * λ0                      # 1.0λ cutoff
t_pre = @elapsed P_nf = build_nearfield_preconditioner(Z, mesh, rwg, cutoff)
println("  NF preconditioner: $(round(t_pre, digits=3)) s")
println("    cutoff = $(round(cutoff/λ0, digits=1))λ")
println("    nnz ratio: $(round(P_nf.nnz_ratio*100, digits=1))%")

t_sol2 = @elapsed begin
    I_gmres, stats2 = solve_gmres(Z, v; preconditioner=P_nf, tol=1e-6, maxiter=300)
end
res2 = norm(Z * I_gmres - v) / norm(v)
err2 = norm(I_gmres - I_direct) / norm(I_direct)
println("  Solve:    $(round(t_sol2, digits=3)) s, $(stats2.niter) iters")
println("  Residual: $res2")
println("  Error vs direct: $err2")

# ── 4. Method 3: ACA H-matrix + GMRES ──────────────
println("\n── Method 3: ACA H-matrix + GMRES ──")

t_aca = @elapsed A_aca = build_aca_operator(mesh, rwg, k;
    leaf_size=32, eta=1.5, aca_tol=1e-6, max_rank=50)
n_dense = length(A_aca.dense_blocks)
n_lr = length(A_aca.lowrank_blocks)
println("  ACA build: $(round(t_aca, digits=3)) s")
println("    Dense blocks: $n_dense, Low-rank blocks: $n_lr")

# Build NF preconditioner directly from ACA dense blocks (no recomputation)
t_pre3 = @elapsed P_nf_aca = build_nearfield_preconditioner(A_aca)
println("  NF preconditioner (from ACA blocks): $(round(t_pre3, digits=3)) s")

t_sol3 = @elapsed begin
    I_aca, stats3 = solve_gmres(A_aca, v; preconditioner=P_nf_aca, tol=1e-6, maxiter=300)
end
err3 = norm(I_aca - I_direct) / norm(I_direct)
println("  Solve:    $(round(t_sol3, digits=3)) s, $(stats3.niter) iters")
println("  Error vs direct: $err3")

# ── 5. Method 2b: Unpreconditioned GMRES (for comparison) ──
println("\n── Method 2b: Dense GMRES (NO preconditioner) ──")
t_sol2b = @elapsed begin
    I_nopre, stats2b = solve_gmres(Z, v; tol=1e-6, maxiter=500)
end
res2b = norm(Z * I_nopre - v) / norm(v)
println("  Solve: $(round(t_sol2b, digits=3)) s, $(stats2b.niter) iters")
println("  Residual: $res2b")

# ── 6. Summary ──────────────────────────────────────
println("\n" * "─"^60)
println("Summary:")
println("─"^60)
println("  Method              Assembly   Solve      Iters  Error vs LU")
println("  Dense direct (LU)   $(lpad(round(t_asm1, digits=3), 6))s   $(lpad(round(t_sol1, digits=3), 6))s   -      -")
println("  Dense GMRES+NF      $(lpad("-", 6))     $(lpad(round(t_sol2, digits=3), 6))s   $(lpad(stats2.niter, 4))   $(round(err2, sigdigits=2))")
println("  ACA GMRES+NF        $(lpad(round(t_aca, digits=3), 6))s   $(lpad(round(t_sol3, digits=3), 6))s   $(lpad(stats3.niter, 4))   $(round(err3, sigdigits=2))")
println("  Dense GMRES (none)  $(lpad("-", 6))     $(lpad(round(t_sol2b, digits=3), 6))s   $(lpad(stats2b.niter, 4))   -")

# ── 7. Far-field verification ───────────────────────
# Verify all three methods give the same RCS
grid  = make_sph_grid(18, 36)
G_mat = radiation_vectors(mesh, rwg, grid, k)
NΩ    = length(grid.w)

E1 = compute_farfield(G_mat, Vector{ComplexF64}(I_direct), NΩ)
E2 = compute_farfield(G_mat, Vector{ComplexF64}(I_gmres), NΩ)
E3 = compute_farfield(G_mat, Vector{ComplexF64}(I_aca), NΩ)

σ1 = bistatic_rcs(E1; E0=1.0)
σ2 = bistatic_rcs(E2; E0=1.0)
σ3 = bistatic_rcs(E3; E0=1.0)

rcs_err_gmres = maximum(abs.(10 .* log10.(max.(σ2, 1e-30)) .- 10 .* log10.(max.(σ1, 1e-30))))
rcs_err_aca   = maximum(abs.(10 .* log10.(max.(σ3, 1e-30)) .- 10 .* log10.(max.(σ1, 1e-30))))

println("\nRCS agreement (max |Δ| dB vs direct):")
println("  GMRES+NF: $(round(rcs_err_gmres, digits=4)) dB")
println("  ACA+NF:   $(round(rcs_err_aca, digits=4)) dB")

println("\n" * "="^60)
println("Done.")
