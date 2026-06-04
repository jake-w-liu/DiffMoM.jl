# 02_impedance_optimization.jl — Impedance-loaded plate: adjoint gradient + L-BFGS
#
# Demonstrates the differentiable optimization pipeline:
#   1. Define patch impedance distribution on a metasurface
#   2. Forward solve with impedance loading
#   3. Compute far-field objective I†QI
#   4. Adjoint gradient ∂J/∂θ
#   5. Verify gradient via finite differences
#   6. Run L-BFGS optimization to maximize broadside radiation
#
# Run: julia --project=. examples/02_impedance_optimization.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra

println("="^60)
println("Example 02: Impedance Optimization (Adjoint + L-BFGS)")
println("="^60)

# ── 1. Problem setup ────────────────────────────────
freq  = 3e9
c0    = 299792458.0
λ0    = c0 / freq
k     = 2π / λ0
eta0  = 376.730313668

Lx, Ly = 0.2, 0.2                      # 2λ × 2λ plate
Nx, Ny = 10, 10
mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg  = build_rwg(mesh)
N    = rwg.nedges

println("\nFrequency: $(freq/1e9) GHz,  λ = $(round(λ0*100, digits=2)) cm")
println("Plate: $(Lx*100) × $(Ly*100) cm  ($(round(Lx/λ0, digits=1))λ × $(round(Ly/λ0, digits=1))λ)")
println("Mesh:  $N RWG unknowns")

# ── 2. Define patch partition ───────────────────────
# One patch per column strip (Nx patches along x)
Nt = ntriangles(mesh)
P  = Nx
tri_patch = zeros(Int, Nt)
for t in 1:Nt
    # Compute triangle center x-coordinate
    cx = sum(mesh.xyz[1, mesh.tri[:, t]]) / 3
    # Map to patch index
    px = clamp(floor(Int, (cx + Lx/2) / (Lx/P)) + 1, 1, P)
    tri_patch[t] = px
end
partition = PatchPartition(tri_patch, P)

println("Patches: $P column strips")

# ── 3. Precompute patch mass matrices ───────────────
println("\nPrecomputing patch mass matrices...")
Mp = precompute_patch_mass(mesh, rwg, partition)

# ── 4. EFIE assembly and excitation ─────────────────
println("Assembling Z_efie ($N × $N)...")
Z_efie = assemble_Z_efie(mesh, rwg, k)

k_vec = Vec3(0.0, 0.0, -k)
v = assemble_excitation(mesh, rwg, make_plane_wave(k_vec, 1.0, Vec3(1.0, 0.0, 0.0)))

# ── 5. Build Q matrix for broadside objective ──────
grid  = make_sph_grid(18, 36)
G_mat = radiation_vectors(mesh, rwg, grid, k)
NΩ    = length(grid.w)
pol   = pol_linear_x(grid)
mask  = cap_mask(grid; theta_max=π/6)   # 30° cone around z
Q     = build_Q(G_mat, grid, pol; mask=mask)

println("Q matrix: $N × $N, target cone: 30° half-angle")

# ── 6. Forward solve with initial impedance ─────────
theta0 = 100.0 * ones(P)               # Initial: 100 Ω/sq per patch

Z_full = assemble_full_Z(Z_efie, Mp, theta0)
I_init = Z_full \ v
J_init = real(dot(I_init, Q * I_init))

println("\n── Initial state (θ = $(theta0[1]) Ω/sq uniform) ──")
println("  J(θ₀) = $(round(J_init, sigdigits=4))")

# ── 7. Adjoint gradient ────────────────────────────
lambda = solve_adjoint(Z_full, Q, Vector{ComplexF64}(I_init))
g_adj  = gradient_impedance(Mp, Vector{ComplexF64}(I_init), lambda)

println("\n── Adjoint gradient (first 5 patches) ──")
for p in 1:min(5, P)
    println("  ∂J/∂θ_$p = $(round(g_adj[p], sigdigits=4))")
end

# ── 8. Verify gradient with finite differences ─────
println("\n── Gradient verification (FD, h=1e-5) ──")
function J_of_theta(θ)
    Z_t = assemble_full_Z(Z_efie, Mp, θ)
    I_t = Z_t \ Vector{ComplexF64}(v)
    return real(dot(I_t, Q * I_t))
end

println("  patch   adjoint       FD           rel_err")
for p in 1:min(5, P)
    g_fd = fd_grad(J_of_theta, theta0, p; h=1e-5)
    rel = abs(g_adj[p] - g_fd) / max(abs(g_fd), 1e-30)
    println("  $p      $(lpad(round(g_adj[p], sigdigits=4), 12))  " *
            "$(lpad(round(g_fd, sigdigits=4), 12))  $(round(rel, sigdigits=2))")
end

# ── 9. L-BFGS optimization ─────────────────────────
println("\n── L-BFGS optimization (maximize broadside power) ──")
theta_opt, trace = optimize_lbfgs(
    Z_efie, Mp, Vector{ComplexF64}(v), Q, theta0;
    maxiter=30,
    tol=1e-8,
    maximize=true,
    lb=fill(10.0, P),                   # Lower bound: 10 Ω/sq
    ub=fill(500.0, P),                  # Upper bound: 500 Ω/sq
    verbose=true,
)

J_opt = trace[end].J
println("\n── Result ──")
println("  J(θ₀) = $(round(J_init, sigdigits=4))")
println("  J(θ*) = $(round(J_opt, sigdigits=4))")
println("  Improvement: $(round(J_opt/J_init, digits=2))×")
println("  Optimal θ: ", [round(t, digits=1) for t in theta_opt])

println("\n" * "="^60)
println("Done.")
