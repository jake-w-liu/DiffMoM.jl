# 04_pec_sphere_mie.jl — PEC sphere: MoM vs Mie analytical benchmark
#
# Validates MoM accuracy against the exact Mie series solution for a
# PEC sphere. Steps:
#   1. Generate an icosphere mesh (no external mesh file needed)
#   2. Solve the MoM scattering problem
#   3. Compute bistatic RCS on a φ-cut
#   4. Compare against Mie theory point-by-point
#   5. Report error metrics (MAE, RMSE in dB)
#
# Run: julia --project=. examples/04_pec_sphere_mie.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra
using Statistics

println("="^60)
println("Example 04: PEC Sphere — MoM vs Mie Benchmark")
println("="^60)

# ── 1. Generate icosphere mesh ──────────────────────
# Build a triangulated sphere via recursive icosahedron subdivision.
# This avoids needing an external OBJ file.
function make_icosphere(radius::Float64; subdivisions::Int=2)
    ϕ = (1 + sqrt(5.0)) / 2
    verts0 = [
        (-1.0,  ϕ, 0.0), ( 1.0,  ϕ, 0.0), (-1.0, -ϕ, 0.0), ( 1.0, -ϕ, 0.0),
        ( 0.0, -1.0, ϕ), ( 0.0,  1.0, ϕ), ( 0.0, -1.0,-ϕ), ( 0.0,  1.0,-ϕ),
        ( ϕ, 0.0, -1.0), ( ϕ, 0.0,  1.0), (-ϕ, 0.0, -1.0), (-ϕ, 0.0,  1.0),
    ]
    faces = [
        (1,12,6), (1,6,2), (1,2,8), (1,8,11), (1,11,12),
        (2,6,10), (6,12,5), (12,11,3), (11,8,7), (8,2,9),
        (4,10,5), (4,5,3), (4,3,7), (4,7,9), (4,9,10),
        (5,10,6), (3,5,12), (7,3,11), (9,7,8), (10,9,2),
    ]
    verts = [Vec3(v...) / norm(Vec3(v...)) for v in verts0]

    for _ in 1:subdivisions
        edge_mid = Dict{Tuple{Int,Int},Int}()
        new_faces = NTuple{3,Int}[]
        function midpoint_index(i::Int, j::Int)
            key = i < j ? (i, j) : (j, i)
            haskey(edge_mid, key) && return edge_mid[key]
            vmid = (verts[i] + verts[j]) / 2
            vmid /= norm(vmid)
            push!(verts, vmid)
            edge_mid[key] = length(verts)
            return length(verts)
        end
        for (i, j, k) in faces
            a = midpoint_index(i, j)
            b = midpoint_index(j, k)
            c = midpoint_index(k, i)
            push!(new_faces, (i, a, c))
            push!(new_faces, (j, b, a))
            push!(new_faces, (k, c, b))
            push!(new_faces, (a, b, c))
        end
        faces = new_faces
    end

    Nv = length(verts)
    Nt = length(faces)
    xyz = zeros(3, Nv)
    tri = zeros(Int, 3, Nt)
    for i in 1:Nv
        xyz[:, i] = radius .* verts[i]
    end
    for t in 1:Nt
        tri[1, t] = faces[t][1]
        tri[2, t] = faces[t][2]
        tri[3, t] = faces[t][3]
    end
    return TriMesh(xyz, tri)
end

# ── 2. Problem setup ────────────────────────────────
a     = 0.05                            # sphere radius = 5 cm
freq  = 2e9                             # 2 GHz
c0    = 299792458.0
λ0    = c0 / freq
k     = 2π / λ0
eta0  = 376.730313668

println("\nRadius: $(a*100) cm,  Frequency: $(freq/1e9) GHz")
println("Size parameter ka = $(round(k*a, digits=3))")

mesh = make_icosphere(a; subdivisions=3)
rwg  = build_rwg(mesh)
N    = rwg.nedges
println("Mesh:  $(nvertices(mesh)) verts, $(ntriangles(mesh)) tris, $N RWG unknowns")

# ── 3. MoM solve ────────────────────────────────────
k_vec   = Vec3(0.0, 0.0, -k)           # -z incidence
khat    = k_vec / norm(k_vec)
pol     = Vec3(1.0, 0.0, 0.0)          # x-polarized

println("\nAssembling EFIE ($N × $N)...")
t_asm = @elapsed Z = assemble_Z_efie(mesh, rwg, k)
println("  Assembly: $(round(t_asm, digits=3)) s")

v = assemble_excitation(mesh, rwg, make_plane_wave(k_vec, 1.0, pol))

println("Solving...")
t_sol = @elapsed I_coeffs = Z \ v
residual = norm(Z * I_coeffs - v) / norm(v)
println("  Solve: $(round(t_sol, digits=3)) s,  residual = $residual")

# ── 4. Far-field and bistatic RCS ───────────────────
grid  = make_sph_grid(90, 72)
G_mat = radiation_vectors(mesh, rwg, grid, k)
NΩ    = length(grid.w)
E_ff  = compute_farfield(G_mat, Vector{ComplexF64}(I_coeffs), NΩ)
σ_mom = bistatic_rcs(E_ff; E0=1.0)

# Energy conservation check
P_in  = input_power(Vector{ComplexF64}(I_coeffs), Vector{ComplexF64}(v))
P_rad = radiated_power(E_ff, grid)
println("\nEnergy: P_rad/P_in = $(round(P_rad/P_in, digits=4))  (should ≈ 1)")

# ── 5. Extract a φ-cut for comparison ──────────────
# Take the φ ≈ 0 cut
phi_target = minimum(grid.phi)
cut_idx = [q for q in 1:NΩ if abs(grid.phi[q] - phi_target) < 1e-12]
sort!(cut_idx, by=q -> grid.theta[q])

# Compute scattering angle γ = angle between k̂_inc and r̂
γ = [acos(clamp(dot(khat, Vec3(grid.rhat[:, q])), -1.0, 1.0)) for q in cut_idx]

# ── 6. Mie reference ───────────────────────────────
σ_mie = [mie_bistatic_rcs_pec(k, a, khat, pol, Vec3(grid.rhat[:, q]))
         for q in cut_idx]

# ── 7. Compare ─────────────────────────────────────
σ_mom_cut = σ_mom[cut_idx]
dB_mom = 10 .* log10.(max.(σ_mom_cut, 1e-30))
dB_mie = 10 .* log10.(max.(σ_mie, 1e-30))
ΔdB    = dB_mom .- dB_mie

mae_db  = mean(abs.(ΔdB))
rmse_db = sqrt(mean(abs2, ΔdB))
max_db  = maximum(abs.(ΔdB))

# Backscatter comparison
σ_bs_mom = backscatter_rcs(E_ff, grid, khat; E0=1.0).sigma
σ_bs_mie = mie_bistatic_rcs_pec(k, a, khat, pol, -khat)
Δbs_db   = 10log10(max(σ_bs_mom, 1e-30)) - 10log10(max(σ_bs_mie, 1e-30))

println("\n── MoM vs Mie (φ = $(round(rad2deg(phi_target), digits=1))° cut) ──")
println("  MAE:     $(round(mae_db, digits=3)) dB")
println("  RMSE:    $(round(rmse_db, digits=3)) dB")
println("  Max |Δ|: $(round(max_db, digits=3)) dB")
println("  Backscatter Δ: $(round(Δbs_db, digits=3)) dB")

# Print a few sample points
println("\n  γ (deg)    σ_MoM (dBsm)   σ_Mie (dBsm)   Δ (dB)")
for i in 1:10:length(cut_idx)
    println("  $(lpad(round(rad2deg(γ[i]), digits=1), 7))    " *
            "$(lpad(round(dB_mom[i], digits=2), 10))    " *
            "$(lpad(round(dB_mie[i], digits=2), 10))    " *
            "$(lpad(round(ΔdB[i], digits=2), 7))")
end

println("\n" * "="^60)
println("Done.")
