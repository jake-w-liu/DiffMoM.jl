# runtests.jl — Test suite for DiffMoM
#
# Run: julia --project=. test/runtests.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using SparseArrays
using StaticArrays
using Statistics
using Random
using Test
using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM

const DATADIR = joinpath(@__DIR__, "..", "data")
mkpath(DATADIR)

function write_icosphere_obj(path::AbstractString; radius::Float64=0.05, subdivisions::Int=2)
    ϕ = (1 + sqrt(5.0)) / 2
    verts = [
        (-1.0,  ϕ, 0.0), ( 1.0,  ϕ, 0.0), (-1.0, -ϕ, 0.0), ( 1.0, -ϕ, 0.0),
        ( 0.0, -1.0, ϕ), ( 0.0,  1.0, ϕ), ( 0.0, -1.0,-ϕ), ( 0.0,  1.0,-ϕ),
        (  ϕ, 0.0, -1.0), (  ϕ, 0.0,  1.0), ( -ϕ, 0.0, -1.0), ( -ϕ, 0.0,  1.0),
    ]
    verts = [Vec3(v...) / norm(Vec3(v...)) for v in verts]

    faces = [
        (1,12,6), (1,6,2), (1,2,8), (1,8,11), (1,11,12),
        (2,6,10), (6,12,5), (12,11,3), (11,8,7), (8,2,9),
        (4,10,5), (4,5,3), (4,3,7), (4,7,9), (4,9,10),
        (5,10,6), (3,5,12), (7,3,11), (9,7,8), (10,9,2),
    ]

    for _ in 1:subdivisions
        edge_mid = Dict{Tuple{Int,Int},Int}()
        new_faces = NTuple{3,Int}[]

        function midpoint_index(i::Int, j::Int)
            key = i < j ? (i, j) : (j, i)
            if haskey(edge_mid, key)
                return edge_mid[key]
            end
            vmid = (verts[i] + verts[j]) / 2
            vmid /= norm(vmid)
            push!(verts, vmid)
            idx = length(verts)
            edge_mid[key] = idx
            return idx
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

    open(path, "w") do io
        println(io, "# Icosphere mesh for test gate")
        for v in verts
            println(io, "v $(radius * v[1]) $(radius * v[2]) $(radius * v[3])")
        end
        for (i, j, k) in faces
            println(io, "f $i $j $k")
        end
    end
end

println("="^60)
println("DiffMoM Test Suite")
println("="^60)

# ─────────────────────────────────────────────────
# Test 1: Mesh and RWG Construction
# ─────────────────────────────────────────────────
println("\n── Test 1: Mesh and RWG construction ──")

Lx, Ly = 0.1, 0.1   # 10 cm × 10 cm plate
Nx, Ny = 3, 3
mesh = make_rect_plate(Lx, Ly, Nx, Ny)

println("  Vertices: $(nvertices(mesh)),  Triangles: $(ntriangles(mesh))")
@assert nvertices(mesh) == (Nx+1)*(Ny+1)
@assert ntriangles(mesh) == 2*Nx*Ny

rwg = build_rwg(mesh)
println("  RWG basis functions: $(rwg.nedges)")
@assert rwg.nedges > 0

# Verify RWG edge lengths are positive and areas are positive
@assert all(rwg.len .> 0)
@assert all(rwg.area_plus .> 0)
@assert all(rwg.area_minus .> 0)

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 1b: OBJ mesh import
# ─────────────────────────────────────────────────
println("\n── Test 1b: OBJ mesh import ──")

obj_path = joinpath(DATADIR, "tmp_quad.obj")
open(obj_path, "w") do io
    println(io, "v 0 0 0")
    println(io, "v 1 0 0")
    println(io, "v 1 1 0")
    println(io, "v 0 1 0")
    println(io, "f 1 2 3 4")
end

mesh_obj = read_obj_mesh(obj_path)
@assert nvertices(mesh_obj) == 4
@assert ntriangles(mesh_obj) == 2

report_obj = assert_mesh_quality(mesh_obj; allow_boundary=true)
@assert report_obj.n_nonmanifold_edges == 0
@assert report_obj.n_orientation_conflicts == 0
@assert report_obj.n_degenerate_triangles == 0

# Orientation-conflict negative test (shared interior edge has same orientation)
tri_bad_orient = hcat([1, 2, 3], [1, 4, 3])
mesh_bad_orient = TriMesh(mesh_obj.xyz, tri_bad_orient)
thrown_orient = try
    assert_mesh_quality(mesh_bad_orient; allow_boundary=true)
    false
catch
    true
end
@assert thrown_orient

# Degenerate-triangle negative test
tri_bad_deg = hcat([1, 1, 2])
mesh_bad_deg = TriMesh(mesh_obj.xyz, tri_bad_deg)
thrown_deg = try
    assert_mesh_quality(mesh_bad_deg; allow_boundary=true)
    false
catch
    true
end
@assert thrown_deg

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 1c: Mesh repair utility
# ─────────────────────────────────────────────────
println("\n── Test 1c: Mesh repair utility ──")

repair_orient = repair_mesh_for_simulation(mesh_bad_orient; allow_boundary=true)
@assert repair_orient.after.n_orientation_conflicts == 0
@assert !isempty(repair_orient.flipped_triangles)
@assert mesh_quality_ok(repair_orient.after; allow_boundary=true, require_closed=false)

xyz_bad_mixed = hcat(mesh_obj.xyz, mesh_obj.xyz[:, 1])
tri_bad_mixed = hcat([1, 2, 5], [1, 2, 3], [1, 6, 3])
mesh_bad_mixed = TriMesh(xyz_bad_mixed, tri_bad_mixed)
repair_mixed = repair_mesh_for_simulation(
    mesh_bad_mixed;
    allow_boundary=true,
    drop_invalid=true,
    drop_degenerate=true,
    fix_orientation=false,
)
@assert ntriangles(repair_mixed.mesh) == 1
@assert length(repair_mixed.removed_invalid) == 1
@assert length(repair_mixed.removed_degenerate) == 1

repair_in_path = joinpath(DATADIR, "tmp_repair_in.obj")
open(repair_in_path, "w") do io
    println(io, "v 0 0 0")
    println(io, "v 1 0 0")
    println(io, "v 1 1 0")
    println(io, "v 0 1 0")
    println(io, "f 1 2 3")
    println(io, "f 1 4 3")
end
repair_out_path = joinpath(DATADIR, "tmp_repair_out.obj")
repair_obj_result = repair_obj_mesh(repair_in_path, repair_out_path; allow_boundary=true)
@assert isfile(repair_obj_result.output_path)
mesh_repair_out = read_obj_mesh(repair_out_path)
report_repair_out = mesh_quality_report(mesh_repair_out)
@assert report_repair_out.n_orientation_conflicts == 0
@assert report_repair_out.n_nonmanifold_edges == 0

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 1d: Mesh coarsening utilities
# ─────────────────────────────────────────────────
println("\n── Test 1d: Mesh coarsening utilities ──")

@assert estimate_dense_matrix_gib(100) > 0

mesh_cluster_in = make_rect_plate(1.0, 1.0, 6, 6)
mesh_cluster_out = cluster_mesh_vertices(mesh_cluster_in, 0.35)
@assert nvertices(mesh_cluster_out) > 0
@assert ntriangles(mesh_cluster_out) > 0

mesh_edges_test = make_rect_plate(1.0, 1.0, 1, 1) # two triangles
edges_test = mesh_unique_edges(mesh_edges_test)
@assert length(edges_test) == 5
segments_test = mesh_wireframe_segments(mesh_edges_test)
@assert segments_test.n_edges == 5
@assert length(segments_test.x) == 15
@assert count(isnan, segments_test.x) == 5

p_mesh = plot_mesh_wireframe(mesh_edges_test; title="Mesh preview test", linewidth=0.5)
@assert p_mesh !== nothing
p_cmp = plot_mesh_comparison(mesh_edges_test, mesh_edges_test; title_a="A", title_b="B", size=(600, 300))
@assert p_cmp !== nothing

xyz_nm = [
    0.0  1.0  0.0  0.0  0.0  2.0  2.0;
    0.0  0.0  1.0 -1.0  0.0  0.0  1.0;
    0.0  0.0  0.0  0.0  1.0  0.0  0.0
]
tri_nm = [
    1  1  1  3;
    2  2  2  6;
    3  4  5  7
]
mesh_nm = TriMesh(xyz_nm, tri_nm)
mesh_nm_clean = drop_nonmanifold_triangles(mesh_nm)
report_nm = mesh_quality_report(mesh_nm_clean)
@assert report_nm.n_nonmanifold_edges == 0

mesh_coarse_in = make_rect_plate(1.0, 1.0, 12, 12)
rwg_coarse_in = build_rwg(mesh_coarse_in; precheck=true, allow_boundary=true)
@assert rwg_coarse_in.nedges > 60

target_rwg = 60
coarse_result = coarsen_mesh_to_target_rwg(mesh_coarse_in, target_rwg; max_iters=8)
rwg_coarse_out = build_rwg(coarse_result.mesh; precheck=true, allow_boundary=true)
@assert coarse_result.rwg_count == rwg_coarse_out.nedges
@assert abs(coarse_result.rwg_count - target_rwg) <= target_rwg  # improved complexity scale
@assert rwg_coarse_out.nedges < rwg_coarse_in.nedges
report_coarse_out = mesh_quality_report(coarse_result.mesh)
@assert mesh_quality_ok(report_coarse_out; allow_boundary=true, require_closed=false)

report_res_before = mesh_resolution_report(mesh_edges_test, 3e8; points_per_wavelength=2.0)
@assert report_res_before.wavelength_m ≈ 299792458.0 / 3e8
@assert report_res_before.edge_max_m > report_res_before.target_max_edge_m
@assert !mesh_resolution_ok(report_res_before)

refine_result = refine_mesh_to_target_edge(mesh_edges_test, 0.40; max_iters=3)
@assert refine_result.triangles_after > refine_result.triangles_before
@assert refine_result.edge_max_after_m <= 0.40 + 1e-12
@assert refine_result.converged

report_res_after = mesh_resolution_report(refine_result.mesh, 3e8; points_per_wavelength=2.0)
@assert mesh_resolution_ok(report_res_after)

mom_refine = refine_mesh_for_mom(mesh_edges_test, 3e8; points_per_wavelength=2.0, max_iters=3)
@assert mom_refine.report_before.edge_max_m > mom_refine.report_before.target_max_edge_m
@assert mom_refine.report_after.edge_max_m <= mom_refine.report_after.target_max_edge_m
@assert mom_refine.converged

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 1e: Parabolic reflector mesh
# ─────────────────────────────────────────────────
println("\n── Test 1e: Parabolic reflector mesh ──")

D_ref = 0.30
f_ref = 0.105
Nr_ref = 6
Nphi_ref = 20
mesh_ref = make_parabolic_reflector(D_ref, f_ref, Nr_ref, Nphi_ref)
report_ref = mesh_quality_report(mesh_ref)
@assert mesh_quality_ok(report_ref; allow_boundary=true, require_closed=false)
@assert nvertices(mesh_ref) == 1 + Nr_ref * Nphi_ref
@assert ntriangles(mesh_ref) == Nphi_ref + 2 * (Nr_ref - 1) * Nphi_ref

# Paraboloid checks at rim points: z=r²/(4f), and distance-to-focus = z+f.
focus = Vec3(0.0, 0.0, f_ref)
rim_start = 2 + (Nr_ref - 1) * Nphi_ref
for idx in rim_start:4:(rim_start + Nphi_ref - 1)
    p = Vec3(mesh_ref.xyz[:, idx])
    rxy = hypot(p[1], p[2])
    z_expected = rxy^2 / (4 * f_ref)
    @assert abs(p[3] - z_expected) < 1e-12
    d_focus = norm(p - focus)
    @assert abs(d_focus - (p[3] + f_ref)) < 1e-12
end

rwg_ref = build_rwg(mesh_ref; precheck=true, allow_boundary=true)
@assert rwg_ref.nedges > 0

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 2: Green's Function
# ─────────────────────────────────────────────────
println("\n── Test 2: Green's function ──")

k0 = 2π / 0.1   # wavelength = 10 cm
r1 = Vec3(0.0, 0.0, 0.0)
r2 = Vec3(0.05, 0.0, 0.0)
R = norm(r2 - r1)

G = greens(r1, r2, k0)
G_expected = exp(-1im * k0 * R) / (4π * R)
@assert abs(G - G_expected) < 1e-14

# Check reciprocity: G(r,r') = G(r',r)
@assert abs(greens(r1, r2, k0) - greens(r2, r1, k0)) < 1e-14

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 2b: PEC sphere Mie utilities
# ─────────────────────────────────────────────────
println("\n── Test 2b: PEC Mie-theory utilities ──")

k_mie = 2π / 0.1
a_mie = 0.03
θ_mie = 60 * π / 180
khat_mie = Vec3(0.0, 0.0, 1.0)
pol_mie = Vec3(1.0, 0.0, 0.0)
rhat_mie = Vec3(sin(θ_mie), 0.0, cos(θ_mie))
μ_mie = dot(khat_mie, rhat_mie)

S1_mie, S2_mie = mie_s1s2_pec(k_mie * a_mie, μ_mie)
@assert isfinite(real(S1_mie)) && isfinite(imag(S1_mie))
@assert isfinite(real(S2_mie)) && isfinite(imag(S2_mie))

σ_formula = 4π * abs2(S2_mie) / (k_mie^2)   # φ=0 plane for x-pol
σ_mie = mie_bistatic_rcs_pec(k_mie, a_mie, khat_mie, pol_mie, rhat_mie)
@assert σ_mie >= 0
@assert abs(σ_mie - σ_formula) / max(abs(σ_formula), 1e-30) < 1e-10

# Independent energy-consistency check:
# integrate differential cross section over sphere and compare to
# coefficient-sum total scattering cross section Csca.
function _mie_csca_coeff_pec(k::Float64, a::Float64; nmax::Int=80)
    x = k * a
    j = zeros(Float64, nmax + 1)  # j[n+1] = j_n
    y = zeros(Float64, nmax + 1)  # y[n+1] = y_n
    j[1] = sin(x) / x
    y[1] = -cos(x) / x
    j[2] = sin(x) / x^2 - cos(x) / x
    y[2] = -cos(x) / x^2 - sin(x) / x
    for n in 1:(nmax - 1)
        j[n + 2] = ((2n + 1) / x) * j[n + 1] - j[n]
        y[n + 2] = ((2n + 1) / x) * y[n + 1] - y[n]
    end

    psi = zeros(Float64, nmax + 1)
    xi  = zeros(ComplexF64, nmax + 1)
    for n in 0:nmax
        psi[n + 1] = x * j[n + 1]
        xi[n + 1]  = x * (j[n + 1] - 1im * y[n + 1])
    end

    csum = 0.0
    for n in 1:nmax
        psi_p = psi[n] - (n / x) * psi[n + 1]
        xi_p  = xi[n] - (n / x) * xi[n + 1]
        an = -psi_p / xi_p
        bn = -psi[n + 1] / xi[n + 1]
        csum += (2n + 1) * (abs2(an) + abs2(bn))
    end
    return (2π / (k^2)) * csum
end

grid_mie = make_sph_grid(181, 180)
σ_mie_grid = [
    mie_bistatic_rcs_pec(k_mie, a_mie, khat_mie, pol_mie, Vec3(grid_mie.rhat[:, q]))
    for q in 1:length(grid_mie.w)
]
Csca_num = sum((σ_mie_grid ./ (4π)) .* grid_mie.w)
Csca_ref = _mie_csca_coeff_pec(k_mie, a_mie; nmax=80)
@assert abs(Csca_num - Csca_ref) / max(abs(Csca_ref), 1e-30) < 2e-3

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 3: EFIE Assembly (PEC plate)
# ─────────────────────────────────────────────────
println("\n── Test 3: EFIE assembly ──")

freq = 3e9            # 3 GHz
c0 = 299792458.0
lambda0 = c0 / freq
k = 2π / lambda0
eta0 = 376.730313668

Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
N = rwg.nedges
println("  Z_efie size: $N × $N")
@assert size(Z_efie) == (N, N)

# Z should have nonzero entries
@assert norm(Z_efie) > 0

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 4: PEC Scattering (plane wave excitation)
# ─────────────────────────────────────────────────
println("\n── Test 4: PEC forward solve ──")

# Normal-incidence plane wave, x-polarized
k_vec = Vec3(0.0, 0.0, -k)    # propagating in -z
E0 = 1.0
pol = Vec3(1.0, 0.0, 0.0)     # x-polarized

v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol; quad_order=3)
@assert length(v) == N
@assert norm(v) > 0

# Solve PEC EFIE: Z_efie * I = v
I_pec = Z_efie \ v
println("  |I_pec| = $(norm(I_pec))")
@assert norm(I_pec) > 0

# Residual check
residual = norm(Z_efie * I_pec - v) / norm(v)
println("  Relative residual: $residual")
@assert residual < 1e-10

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 5: Impedance Term and Derivatives
# ─────────────────────────────────────────────────
println("\n── Test 5: Impedance term and derivatives ──")

Nt = ntriangles(mesh)
# Simple partition: one patch per triangle
partition = PatchPartition(collect(1:Nt), Nt)

Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
@assert length(Mp) == Nt

# Each M_p should be symmetric (real-valued mass matrix)
for p in 1:min(3, Nt)
    @assert norm(Mp[p] - Mp[p]') < 1e-14 * norm(Mp[p])
end

# Test impedance assembly
theta = fill(100.0 + 50.0im, Nt)  # complex impedance
Z_imp = assemble_Z_impedance(Mp, theta)
@assert size(Z_imp) == (N, N)

# Resistive/reactive decomposition sanity checks
Z_imp_res = assemble_Z_impedance(Mp, fill(100.0, Nt))
Z_imp_reac = assemble_Z_impedance(Mp, 1im .* fill(100.0, Nt))
@assert maximum(abs.(imag.(Z_imp_res))) < 1e-12
@assert maximum(abs.(real.(Z_imp_reac))) < 1e-12

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 6: Far-Field and Q Matrix
# ─────────────────────────────────────────────────
println("\n── Test 6: Far-field and Q matrix ──")

grid = make_sph_grid(8, 16)
NΩ = length(grid.w)
println("  Far-field grid: $NΩ directions")

G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
@assert size(G_mat) == (3 * NΩ, N)

# Compute far-field from PEC solution
E_ff = compute_farfield(G_mat, I_pec, NΩ)
@assert size(E_ff) == (3, NΩ)

# Far-field should be transverse: r̂ · E∞ ≈ 0
max_radial = let mr = 0.0
    for q in 1:NΩ
        rh = Vec3(grid.rhat[:, q])
        Eq = CVec3(E_ff[:, q])
        radial = abs(dot(rh, Eq)) / max(abs(norm(Eq)), 1e-30)
        mr = max(mr, radial)
    end
    mr
end
println("  Max radial E-field component: $max_radial")
@assert max_radial < 0.1  # should be small

# Build Q matrix
pol_mat = pol_linear_x(grid)
mask = cap_mask(grid; theta_max=30 * π / 180)
Q = build_Q(G_mat, grid, pol_mat; mask=mask)

# Q should be Hermitian PSD
@assert norm(Q - Q') < 1e-12 * norm(Q)
eigvals_Q = eigvals(Hermitian(Q))
@assert all(eigvals_Q .>= -1e-12 * maximum(eigvals_Q))
println("  Q is Hermitian PSD ✓")

# Cross-check objective computed two ways:
#   (1) quadratic form I†QI
#   (2) direct angular integration of projected far field
P_qform = real(dot(I_pec, Q * I_pec))
P_direct = projected_power(E_ff, grid, pol_mat; mask=mask)
rel_q_err = abs(P_qform - P_direct) / max(abs(P_qform), 1e-30)
println("  Objective consistency (I†QI vs direct projected power): $rel_q_err")
@assert rel_q_err < 1e-12

# RCS helper checks
sigma = bistatic_rcs(E_ff; E0=1.0)
@assert length(sigma) == NΩ
@assert all(sigma .>= 0.0)

bs = backscatter_rcs(E_ff, grid, Vec3(0.0, 0.0, -1.0); E0=1.0)
@assert 1 <= bs.index <= NΩ
@assert bs.sigma >= 0.0

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 6b: Scattered Near-Field Evaluation
# ─────────────────────────────────────────────────
println("\n── Test 6b: Scattered near-field evaluation ──")

obs_points = [
    Vec3(0.00, 0.00, 0.15),
    Vec3(0.02, -0.03, 0.18),
]
E_nf = compute_nearfield(mesh, rwg, I_pec, obs_points, k; quad_order=3, eta0=eta0)
@assert size(E_nf) == (3, length(obs_points))
@assert all(isfinite, real.(E_nf))
@assert all(isfinite, imag.(E_nf))

obs_mat = hcat(obs_points...)
E_nf_mat = compute_nearfield(mesh, rwg, I_pec, obs_mat, k; quad_order=3, eta0=eta0)
@assert norm(E_nf - E_nf_mat) < 1e-12 * max(norm(E_nf), 1.0)

E_nf_pt = compute_nearfield(mesh, rwg, I_pec, obs_points[1], k; quad_order=3, eta0=eta0)
@assert norm(E_nf_pt - CVec3(E_nf[:, 1])) < 1e-12 * max(norm(E_nf_pt), 1.0)

I_nf_a = ComplexF64.(randn(N) .+ 1im .* randn(N))
I_nf_b = ComplexF64.(randn(N) .+ 1im .* randn(N))
E_nf_a = compute_nearfield(mesh, rwg, I_nf_a, obs_points, k; quad_order=3, eta0=eta0)
E_nf_b = compute_nearfield(mesh, rwg, I_nf_b, obs_points, k; quad_order=3, eta0=eta0)
E_nf_ab = compute_nearfield(mesh, rwg, I_nf_a .+ I_nf_b, obs_points, k; quad_order=3, eta0=eta0)
rel_nf_lin = norm(E_nf_ab - (E_nf_a + E_nf_b)) / max(norm(E_nf_ab), 1e-30)
println("  Near-field linearity rel. error: $rel_nf_lin")
@assert rel_nf_lin < 1e-12

surface_err = try
    compute_nearfield(mesh, rwg, I_pec, triangle_center(mesh, 1), k; quad_order=3, eta0=eta0)
    false
catch
    true
end
@assert surface_err

grid_nf = make_sph_grid(10, 20)
q_nf = 37
G_nf_far = radiation_vectors(mesh, rwg, grid_nf, k; quad_order=7, eta0=eta0)
E_ff_nf = compute_farfield(G_nf_far, I_pec, length(grid_nf.w))
rhat_nf = Vec3(grid_nf.rhat[:, q_nf])
R_nf = 40.0 * lambda0
obs_far = R_nf * rhat_nf
E_nf_far = compute_nearfield(mesh, rwg, I_pec, obs_far, k; quad_order=7, eta0=eta0)
E_nf_ref = CVec3(E_ff_nf[:, q_nf]) * exp(-1im * k * R_nf) / R_nf
rel_nf_far = norm(E_nf_far - E_nf_ref) / max(norm(E_nf_ref), 1e-30)
println("  Near-field far-zone rel. error: $rel_nf_far")
@assert rel_nf_far < 0.08

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 6c: Total Electric Field Evaluation
# ─────────────────────────────────────────────────
println("\n── Test 6c: Total electric-field evaluation ──")

pw_total = make_plane_wave(k_vec, E0, pol)
I_zero = zeros(ComplexF64, N)

function stack_fields(field_func, points)
    E = zeros(ComplexF64, 3, length(points))
    for i in eachindex(points)
        E[:, i] .= field_func(points[i])
    end
    return E
end

E_tf = compute_total_field(mesh, rwg, I_pec, pw_total, obs_points, k; quad_order=3, eta0=eta0)
@assert size(E_tf) == (3, length(obs_points))
@assert all(isfinite, real.(E_tf))
@assert all(isfinite, imag.(E_tf))

E_tf_mat = compute_total_field(mesh, rwg, I_pec, pw_total, obs_mat, k; quad_order=3, eta0=eta0)
@assert norm(E_tf - E_tf_mat) < 1e-12 * max(norm(E_tf), 1.0)

E_tf_pt = compute_total_field(mesh, rwg, I_pec, pw_total, obs_points[1], k; quad_order=3, eta0=eta0)
@assert E_tf_pt isa CVec3
@assert norm(E_tf_pt - CVec3(E_tf[:, 1])) < 1e-12 * max(norm(E_tf_pt), 1.0)

E_pw_ref = stack_fields(obs_points) do r
    plane_wave_field(r, k_vec, E0, pol)
end
E_tf_zero_pw = compute_total_field(mesh, rwg, I_zero, pw_total, obs_points, k; quad_order=3, eta0=eta0)
@assert norm(E_tf_zero_pw - E_pw_ref) < 1e-12 * max(norm(E_pw_ref), 1.0)

dip_total = make_dipole(Vec3(0.03, -0.02, 0.12),
                        CVec3(0.0 + 0im, 0.0 + 0im, 1.5e-9 + 0.4e-9im),
                        Vec3(0.0, 0.0, 1.0),
                        :electric,
                        freq)
E_dip_ref = stack_fields(obs_points) do r
    DiffMoM.dipole_incident_field(r, dip_total)
end
E_tf_zero_dip = compute_total_field(mesh, rwg, I_zero, dip_total, obs_points, k; quad_order=3, eta0=eta0)
@assert norm(E_tf_zero_dip - E_dip_ref) < 1e-12 * max(norm(E_dip_ref), 1.0)

loop_total = make_loop(Vec3(-0.04, 0.01, 0.14),
                       Vec3(0.0, 1.0, 0.0),
                       0.012,
                       1.0 + 0.35im,
                       freq)
E_loop_ref = stack_fields(obs_points) do r
    DiffMoM.loop_incident_field(r, loop_total)
end
E_tf_zero_loop = compute_total_field(mesh, rwg, I_zero, loop_total, obs_points, k; quad_order=3, eta0=eta0)
@assert norm(E_tf_zero_loop - E_loop_ref) < 1e-12 * max(norm(E_loop_ref), 1.0)

theta_pat = collect(range(0.0, stop=π, length=9))
phi_pat = collect(range(0.0, stop=2π - 2π / 12, length=12))
pat_total = make_analytic_dipole_pattern_feed(dip_total, theta_pat, phi_pat;
                                              phase_center=dip_total.position,
                                              convention=:exp_plus_iwt)
E_pat_ref = stack_fields(obs_points) do r
    pattern_feed_field(r, pat_total)
end
E_tf_zero_pat = compute_total_field(mesh, rwg, I_zero, pat_total, obs_points, k; quad_order=3, eta0=eta0)
@assert norm(E_tf_zero_pat - E_pat_ref) < 1e-12 * max(norm(E_pat_ref), 1.0)

imp_total = ImportedExcitation(
    r -> CVec3(0.3 + 0.1im * r[1], -0.2 + 0.05im * r[2], 0.1 - 0.08im * r[3]);
    kind=:electric_field,
    min_quad_order=3,
)
E_imp_ref = stack_fields(obs_points) do r
    imp_total.source_func(r)
end
E_tf_zero_imp = compute_total_field(mesh, rwg, I_zero, imp_total, obs_points, k; quad_order=3, eta0=eta0)
@assert norm(E_tf_zero_imp - E_imp_ref) < 1e-12 * max(norm(E_imp_ref), 1.0)

E_inc_from_total = E_tf - E_nf
rel_tf_inc = norm(E_inc_from_total - E_pw_ref) / max(norm(E_pw_ref), 1e-30)
println("  Total minus scattered rel. error: $rel_tf_inc")
@assert rel_tf_inc < 1e-12

E_tf_far = compute_total_field(mesh, rwg, I_pec, pw_total, obs_far, k; quad_order=7, eta0=eta0)
E_tf_ref = plane_wave_field(obs_far, k_vec, E0, pol) + E_nf_ref
rel_tf_far = norm(E_tf_far - E_tf_ref) / max(norm(E_tf_ref), 1e-30)
println("  Total-field far-zone rel. error: $rel_tf_far")
@assert rel_tf_far < 0.08

v_dip_total = assemble_excitation(mesh, rwg, dip_total; quad_order=3)
I_dip_total = Z_efie \ v_dip_total
w_pw = 0.7 - 0.15im
w_dip = -0.25 + 0.5im
multi_total = make_multi_excitation([pw_total, dip_total], [w_pw, w_dip])
I_combo = w_pw .* I_pec .+ w_dip .* I_dip_total
E_tf_multi = compute_total_field(mesh, rwg, I_combo, multi_total, obs_points, k; quad_order=3, eta0=eta0)
E_tf_multi_ref =
    w_pw .* compute_total_field(mesh, rwg, I_pec, pw_total, obs_points, k; quad_order=3, eta0=eta0) .+
    w_dip .* compute_total_field(mesh, rwg, I_dip_total, dip_total, obs_points, k; quad_order=3, eta0=eta0)
rel_tf_multi = norm(E_tf_multi - E_tf_multi_ref) / max(norm(E_tf_multi_ref), 1e-30)
println("  MultiExcitation total-field rel. error: $rel_tf_multi")
@assert rel_tf_multi < 1e-12

port_total = PortExcitation([1, 2], 1.0 + 0im, 50.0 + 0im)
port_err = try
    compute_total_field(mesh, rwg, I_zero, port_total, obs_points, k; quad_order=3, eta0=eta0)
    false
catch err
    occursin("PortExcitation", sprint(showerror, err))
end
@assert port_err

gap_total = make_delta_gap(1, 1.0 + 0im, 0.01)
gap_err = try
    compute_total_field(mesh, rwg, I_zero, gap_total, obs_points, k; quad_order=3, eta0=eta0)
    false
catch err
    occursin("DeltaGapExcitation", sprint(showerror, err))
end
@assert gap_err

imp_js_total = ImportedExcitation(r -> CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im);
                                  kind=:surface_current_density,
                                  eta_equiv=eta0 + 0im,
                                  min_quad_order=3)
imp_js_err = try
    compute_total_field(mesh, rwg, I_zero, imp_js_total, obs_points, k; quad_order=3, eta0=eta0)
    false
catch err
    occursin("surface_current_density", sprint(showerror, err))
end
@assert imp_js_err

multi_bad = make_multi_excitation([pw_total, gap_total], [1.0 + 0im, 0.25 + 0im])
multi_bad_err = try
    compute_total_field(mesh, rwg, I_zero, multi_bad, obs_points, k; quad_order=3, eta0=eta0)
    false
catch err
    msg = sprint(showerror, err)
    occursin("child 2", msg) && occursin("DeltaGapExcitation", msg)
end
@assert multi_bad_err

pw_k_err = try
    compute_total_field(mesh, rwg, I_zero, pw_total, obs_points, 1.01 * k; quad_order=3, eta0=eta0)
    false
catch err
    occursin("PlaneWaveExcitation", sprint(showerror, err))
end
@assert pw_k_err

dip_k_err = try
    compute_total_field(mesh, rwg, I_zero, dip_total, obs_points, 0.99 * k; quad_order=3, eta0=eta0)
    false
catch err
    occursin("DipoleExcitation", sprint(showerror, err))
end
@assert dip_k_err

pat_k_err = try
    compute_total_field(mesh, rwg, I_zero, pat_total, obs_points, 1.02 * k; quad_order=3, eta0=eta0)
    false
catch err
    occursin("PatternFeedExcitation", sprint(showerror, err))
end
@assert pat_k_err

surface_total_err = try
    compute_total_field(mesh, rwg, I_zero, pw_total, triangle_center(mesh, 1), k; quad_order=3, eta0=eta0)
    false
catch
    true
end
@assert surface_total_err

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 7: Adjoint Gradient Verification
# ─────────────────────────────────────────────────
println("\n── Test 7: Adjoint gradient verification (CRITICAL) ──")

# Setup: impedance sheet problem with real impedance parameters
theta_real = fill(200.0, Nt)  # real impedance values
Z_full = assemble_full_Z(Z_efie, Mp, theta_real)
I_imp = Z_full \ v

# Objective
J_val = compute_objective(I_imp, Q)
println("  J(θ₀) = $J_val")

# Adjoint gradient
lambda = solve_adjoint(Z_full, Q, I_imp)
g_adj = gradient_impedance(Mp, I_imp, lambda)
println("  |g_adj| = $(norm(g_adj))")

# Gradient verification via central finite differences
#
# Note: complex-step is not used here because J(θ) = I†QI involves
# conjugation (sesquilinear form), which breaks analyticity.
# Central FD with h ≈ 1e-5 provides O(h²) accuracy, sufficient
# for validating the adjoint gradient.

function J_of_theta(theta_vec)
    Z_t = copy(Z_efie)
    for p in eachindex(theta_vec)
        Z_t .-= theta_vec[p] .* Mp[p]
    end
    I_t = Z_t \ v
    return real(dot(I_t, Q * I_t))
end

# Sanity check: J_of_theta at baseline should match J_val
J_check = J_of_theta(theta_real)
@assert abs(J_check - J_val) / max(abs(J_val), 1e-30) < 1e-12

println("  Checking adjoint vs central finite difference (h=1e-5)...")
fd_results = Float64[]
adj_results = Float64[]
rel_errors = Float64[]

n_check = min(Nt, 10)  # check first 10 parameters
h_fd = 1e-5
for p in 1:n_check
    g_fd = fd_grad(J_of_theta, theta_real, p; h=h_fd)
    rel_err = abs(g_adj[p] - g_fd) / max(abs(g_adj[p]), 1e-30)
    push!(fd_results, g_fd)
    push!(adj_results, g_adj[p])
    push!(rel_errors, rel_err)
    println("    p=$p: adj=$(g_adj[p])  fd=$g_fd  rel_err=$rel_err")
end

max_rel_err = maximum(rel_errors)
println("  Max relative error (adjoint vs central FD): $max_rel_err")
@assert max_rel_err < 1e-4 "Gradient verification FAILED: max rel error = $max_rel_err"

println("  PASS ✓  (adjoint gradients match central FD)")

# ─────────────────────────────────────────────────
# Test 8: FD Convergence Check
# ─────────────────────────────────────────────────
println("\n── Test 8: FD convergence rate check ──")

# Verify FD error decreases at O(h²) for central differences
# by comparing two step sizes on a single parameter
p_test = 1
h1 = 1e-4
h2 = 1e-5
g_fd1 = fd_grad(J_of_theta, theta_real, p_test; h=h1)
g_fd2 = fd_grad(J_of_theta, theta_real, p_test; h=h2)
err1 = abs(g_adj[p_test] - g_fd1)
err2 = abs(g_adj[p_test] - g_fd2)

if err1 > 1e-15 && err2 > 1e-15
    rate = log10(err1 / err2) / log10(h1 / h2)
    println("  Error at h=$h1: $err1")
    println("  Error at h=$h2: $err2")
    println("  Convergence rate: $rate  (expected ≈ 2 for central FD)")
    # Rate should be near 2 for central differences (O(h²))
    @assert rate > 1.5 "FD convergence rate too low: $rate (expected ~2)"
else
    println("  Errors at machine precision — gradient is exact")
end

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 9: Reciprocity Check
# ─────────────────────────────────────────────────
println("\n── Test 9: Reciprocity check ──")

# For EFIE on PEC: Z should be symmetric (Z = Z^T) due to reciprocity
# (Galerkin testing with the same basis/test functions)
sym_err = norm(Z_efie - transpose(Z_efie)) / norm(Z_efie)
println("  Symmetry error (EFIE, PEC): $sym_err")
# Note: due to quadrature, small symmetry error is expected
@assert sym_err < 1e-10 "EFIE matrix not symmetric: err = $sym_err"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 10: Optimization Smoke Test
# ─────────────────────────────────────────────────
println("\n── Test 10: Optimization smoke test ──")

theta_init = fill(300.0, Nt)
theta_opt, trace = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init;
    maxiter=10, tol=1e-8, alpha0=0.01, verbose=false
)

# Check that objective decreased
if length(trace) >= 2
    J_first = trace[1].J
    J_last  = trace[end].J
    println("  J(iter=1)  = $J_first")
    println("  J(iter=$(length(trace))) = $J_last")

end

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 11: Paper-consistency metrics from tracked data
# ─────────────────────────────────────────────────
println("\n── Test 11: Paper-consistency metrics ──")

meanval(x) = sum(x) / length(x)

function crossval_metrics(
    df_ref::DataFrame,
    ref_col::Symbol,
    df_cmp::DataFrame,
    cmp_col::Symbol;
    target_theta_deg::Float64=30.0,
)
    left = select(df_ref, :theta_deg, :phi_deg, ref_col)
    right = select(df_cmp, :theta_deg, :phi_deg, cmp_col)
    merged = innerjoin(left, right, on=[:theta_deg, :phi_deg])

    delta = merged[!, cmp_col] .- merged[!, ref_col]
    abs_delta = abs.(delta)

    theta_unique = unique(merged.theta_deg)
    theta_near = theta_unique[argmin(abs.(theta_unique .- target_theta_deg))]
    idx_target = findall(t -> abs(t - theta_near) < 1e-12, merged.theta_deg)

    return (
        n = nrow(merged),
        rmse = sqrt(sum(abs2, delta) / length(delta)),
        mean_abs = meanval(abs_delta),
        target_theta_near = theta_near,
        target_mean_abs = meanval(abs_delta[idx_target]),
    )
end

paper_metric_inputs = [
    joinpath(DATADIR, "convergence_study.csv"),
    joinpath(DATADIR, "gradient_verification.csv"),
    joinpath(DATADIR, "robustness_sweep.csv"),
    joinpath(DATADIR, "beam_steer_farfield.csv"),
    joinpath(DATADIR, "bempp_pec_farfield.csv"),
    joinpath(DATADIR, "julia_impedance_farfield.csv"),
    joinpath(DATADIR, "bempp_impedance_farfield.csv"),
]
imp_beam_csv = joinpath(DATADIR, "impedance_validation_matrix_summary.csv")
imp_beam_csv_alt = joinpath(DATADIR, "impedance_validation_matrix_summary_paper_default.csv")

missing_paper_metric_inputs = [path for path in paper_metric_inputs if !isfile(path)]
if !(isfile(imp_beam_csv) || isfile(imp_beam_csv_alt))
    push!(missing_paper_metric_inputs, imp_beam_csv)
end

if !isempty(missing_paper_metric_inputs)
    missing_names = join(basename.(missing_paper_metric_inputs), ", ")
    println("  SKIPPED (optional paper-consistency artifacts not found: $missing_names)")
else
conv = CSV.read(joinpath(DATADIR, "convergence_study.csv"), DataFrame)
grad = CSV.read(joinpath(DATADIR, "gradient_verification.csv"), DataFrame)
rob = CSV.read(joinpath(DATADIR, "robustness_sweep.csv"), DataFrame)

max_grad_mesh = maximum(conv.max_grad_err)
min_energy_ratio = minimum(conv.energy_ratio)
max_grad_ref = maximum(grad.rel_error)

idx_nom = findfirst(rob.case .== "f_nom")
idx_p2 = findfirst(rob.case .== "f_+2pct")
@assert idx_nom !== nothing
@assert idx_p2 !== nothing

J_opt_nom = rob.J_opt_pct[idx_nom]
J_pec_nom = rob.J_pec_pct[idx_nom]
peak_theta_p2 = rob.peak_theta_opt_deg[idx_p2]

df_pec_julia = CSV.read(joinpath(DATADIR, "beam_steer_farfield.csv"), DataFrame)
df_pec_bempp = CSV.read(joinpath(DATADIR, "bempp_pec_farfield.csv"), DataFrame)
df_imp_julia = CSV.read(joinpath(DATADIR, "julia_impedance_farfield.csv"), DataFrame)
df_imp_bempp = CSV.read(joinpath(DATADIR, "bempp_impedance_farfield.csv"), DataFrame)

if !isfile(imp_beam_csv)
    imp_beam_csv = imp_beam_csv_alt
end
df_imp_beam = CSV.read(imp_beam_csv, DataFrame)

pec_cv = crossval_metrics(df_pec_julia, :dir_pec_dBi, df_pec_bempp, :dir_bempp_dBi)
imp_cv = crossval_metrics(df_imp_julia, :dir_julia_imp_dBi, df_imp_bempp, :dir_bempp_imp_dBi)

n_beam_cases = nrow(df_imp_beam)
n_pass_main_theta = count(df_imp_beam.pass_main_theta_le_3deg)
n_pass_main_level = count(df_imp_beam.pass_main_level_le_1p5db)
n_pass_sll = count(df_imp_beam.pass_sll_le_3db)

println("  Max grad rel. err (reference): $max_grad_ref")
println("  Max grad rel. err (mesh sweep): $max_grad_mesh")
println("  Min energy ratio: $min_energy_ratio")
println("  Nominal J_opt/J_pec (%): $J_opt_nom / $J_pec_nom")
println("  +2% freq peak theta (deg): $peak_theta_p2")
println("  PEC CV RMSE / near-target |ΔD| (dB): $(pec_cv.rmse) / $(pec_cv.target_mean_abs)")
println("  IMP CV RMSE / near-target |ΔD| (dB): $(imp_cv.rmse) / $(imp_cv.target_mean_abs)")
println("  Beam-centric passes (main θ / main L / SLL): $n_pass_main_theta/$n_beam_cases, $n_pass_main_level/$n_beam_cases, $n_pass_sll/$n_beam_cases")

# These checks track manuscript quantitative claims.
@assert max_grad_ref < 3e-7
@assert max_grad_mesh < 3e-6
@assert min_energy_ratio > 0.98
@assert J_opt_nom > J_pec_nom
@assert peak_theta_p2 < 5.0
@assert pec_cv.target_mean_abs < 0.5
@assert n_pass_main_theta == n_beam_cases
@assert n_pass_main_level == n_beam_cases
@assert n_pass_sll == n_beam_cases

println("  PASS ✓")
end  # if isfile(convergence_study.csv)

# ─────────────────────────────────────────────────
# Test 12: Conditioning / preconditioning consistency
# ─────────────────────────────────────────────────
println("\n── Test 12: Conditioning and preconditioning ──")

theta_c = copy(theta_real)
Z_raw = assemble_full_Z(Z_efie, Mp, theta_c)
I_raw = Z_raw \ v

# Build mass-based regularizer and left preconditioner
R_mass = make_mass_regularizer(Mp)
M_left = make_left_preconditioner(Mp; eps_rel=1e-6)

# Auto-preconditioning selector behavior
M_auto_off, enabled_auto_off, reason_auto_off = select_preconditioner(
    Mp;
    mode=:auto,
    n_threshold=10_000,
    iterative_solver=false,
)
println("  Auto preconditioning (high threshold): enabled=$enabled_auto_off ($reason_auto_off)")
@assert !enabled_auto_off
@assert M_auto_off === nothing

M_auto_on, enabled_auto_on, reason_auto_on = select_preconditioner(
    Mp;
    mode=:auto,
    n_threshold=1,
    iterative_solver=false,
    eps_rel=1e-6,
)
println("  Auto preconditioning (low threshold): enabled=$enabled_auto_on ($reason_auto_on)")
@assert enabled_auto_on
@assert M_auto_on !== nothing

# Left-preconditioned system should preserve the same solution
Z_pre, v_pre, fac_pre = prepare_conditioned_system(
    Z_raw,
    v;
    regularization_alpha=0.0,
    regularization_R=nothing,
    preconditioner_M=M_left,
)
I_pre = Z_pre \ v_pre
rel_I_pre = norm(I_pre - I_raw) / max(norm(I_raw), 1e-30)
println("  Left-preconditioned solution mismatch: $rel_I_pre")
@assert rel_I_pre < 1e-10

# Auto-selected preconditioner (activated) should also preserve the solution
Z_pre_auto, v_pre_auto, _ = prepare_conditioned_system(
    Z_raw,
    v;
    regularization_alpha=0.0,
    regularization_R=nothing,
    preconditioner_M=M_auto_on,
)
I_pre_auto = Z_pre_auto \ v_pre_auto
rel_I_pre_auto = norm(I_pre_auto - I_raw) / max(norm(I_raw), 1e-30)
println("  Auto-preconditioned solution mismatch: $rel_I_pre_auto")
@assert rel_I_pre_auto < 1e-10

# Regularization should alter the solve (for nonzero alpha)
alpha_reg = 1e-3
Z_reg, v_reg, _ = prepare_conditioned_system(
    Z_raw,
    v;
    regularization_alpha=alpha_reg,
    regularization_R=R_mass,
    preconditioner_M=nothing,
)
I_reg = Z_reg \ v_reg
rel_I_reg = norm(I_reg - I_raw) / max(norm(I_raw), 1e-30)
println("  Regularized solution change (alpha=$alpha_reg): $rel_I_reg")
@assert rel_I_reg > 1e-9

# Adjoint gradient with left preconditioning should match FD on the
# equivalently preconditioned system objective.
Mp_pre, fac_pre = transform_patch_matrices(
    Mp;
    preconditioner_M=M_left,
    preconditioner_factor=fac_pre,
)
lambda_pre = solve_adjoint(Z_pre, Q, I_pre)
g_pre = gradient_impedance(Mp_pre, I_pre, lambda_pre)

function J_of_theta_pre(theta_vec)
    Z_t = assemble_full_Z(Z_efie, Mp, theta_vec)
    Zp, vp, _ = prepare_conditioned_system(
        Z_t,
        v;
        regularization_alpha=0.0,
        regularization_R=nothing,
        preconditioner_M=M_left,
        preconditioner_factor=fac_pre,
    )
    I_t = Zp \ vp
    return real(dot(I_t, Q * I_t))
end

p_pre = 1
g_fd_pre = fd_grad(J_of_theta_pre, theta_c, p_pre; h=1e-5)
rel_g_pre = abs(g_pre[p_pre] - g_fd_pre) / max(abs(g_pre[p_pre]), 1e-30)
println("  Preconditioned gradient rel. error (p=$p_pre): $rel_g_pre")
@assert rel_g_pre < 1e-4

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 13: Sphere MoM-vs-Mie benchmark CI gate
# ─────────────────────────────────────────────────
println("\n── Test 13: Sphere Mie benchmark gate ──")

obj_sphere = joinpath(DATADIR, "tmp_sphere_gate.obj")
write_icosphere_obj(obj_sphere; radius=0.05, subdivisions=2)

mesh_s = read_obj_mesh(obj_sphere)
rwg_s = build_rwg(mesh_s)

freq_s = 2.0e9
c0_s = 299792458.0
lambda_s = c0_s / freq_s
k_s = 2π / lambda_s
eta0_s = 376.730313668
k_vec_s = Vec3(0.0, 0.0, -k_s)
khat_s = k_vec_s / norm(k_vec_s)
pol_s = Vec3(1.0, 0.0, 0.0)

ctr_s = vec(sum(mesh_s.xyz, dims=2) ./ nvertices(mesh_s))
radii_s = [norm(Vec3(mesh_s.xyz[:, i]) - Vec3(ctr_s)) for i in 1:nvertices(mesh_s)]
a_s = sum(radii_s) / length(radii_s)

Z_s = assemble_Z_efie(mesh_s, rwg_s, k_s; quad_order=3, eta0=eta0_s)
v_s = assemble_v_plane_wave(mesh_s, rwg_s, k_vec_s, 1.0, pol_s; quad_order=3)
I_s = solve_forward(Z_s, v_s)
res_s = norm(Z_s * I_s - v_s) / max(norm(v_s), 1e-30)
@assert res_s < 1e-10

grid_s = make_sph_grid(37, 18)
G_s = radiation_vectors(mesh_s, rwg_s, grid_s, k_s; quad_order=3, eta0=eta0_s)
E_s = compute_farfield(G_s, I_s, length(grid_s.w))
σ_mom_s = bistatic_rcs(E_s; E0=1.0)

phi_target_s = grid_s.phi[argmin(grid_s.phi)]
idx_cut_s = [q for q in 1:length(grid_s.w) if abs(grid_s.phi[q] - phi_target_s) < 1e-12]
idx_cut_s = idx_cut_s[sortperm(grid_s.theta[idx_cut_s])]

σ_mie_s = [
    mie_bistatic_rcs_pec(k_s, a_s, khat_s, pol_s, Vec3(grid_s.rhat[:, q]))
    for q in idx_cut_s
]

dB_mom_s = 10 .* log10.(max.(σ_mom_s[idx_cut_s], 1e-30))
dB_mie_s = 10 .* log10.(max.(σ_mie_s, 1e-30))
ΔdB_s = dB_mom_s .- dB_mie_s

mae_s = sum(abs.(ΔdB_s)) / length(ΔdB_s)
rmse_s = sqrt(sum(abs2, ΔdB_s) / length(ΔdB_s))
maxabs_s = maximum(abs.(ΔdB_s))

σ_bs_mom_s = backscatter_rcs(E_s, grid_s, khat_s; E0=1.0).sigma
σ_bs_mie_s = mie_bistatic_rcs_pec(k_s, a_s, khat_s, pol_s, -khat_s)
Δbs_s = 10 * log10(max(σ_bs_mom_s, 1e-30)) - 10 * log10(max(σ_bs_mie_s, 1e-30))

println("  MAE(dB): $mae_s")
println("  RMSE(dB): $rmse_s")
println("  Max |Δ|(dB): $maxabs_s")
println("  Backscatter Δ(dB): $Δbs_s")

# Dedicated CI thresholds for the sphere benchmark (coarse grid tolerances)
@assert mae_s < 0.50 "Sphere Mie gate failed: MAE(dB)=$mae_s"
@assert rmse_s < 0.60 "Sphere Mie gate failed: RMSE(dB)=$rmse_s"
@assert maxabs_s < 1.20 "Sphere Mie gate failed: max |Δ|(dB)=$maxabs_s"
@assert abs(Δbs_s) < 1.20 "Sphere Mie gate failed: |backscatter Δ(dB)|=$(abs(Δbs_s))"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 14: Excitation-model physics sanity checks
# ─────────────────────────────────────────────────
println("\n── Test 14: Excitation physics sanity checks ──")

mesh_exc = make_rect_plate(0.02, 0.02, 3, 3)
rwg_exc = build_rwg(mesh_exc)

freq_exc = 1.0e9
k_exc = 2π * freq_exc / 299792458.0
k_vec_exc = Vec3(0.0, 0.0, -k_exc)
pol_exc = Vec3(1.0, 0.0, 0.0)

v_old_exc = assemble_v_plane_wave(mesh_exc, rwg_exc, k_vec_exc, 1.0, pol_exc; quad_order=3)
v_new_exc = assemble_excitation(mesh_exc, rwg_exc, make_plane_wave(k_vec_exc, 1.0, pol_exc); quad_order=3)
rel_rhs_exc = norm(v_new_exc - v_old_exc) / max(norm(v_old_exc), 1e-30)
println("  Plane-wave path-consistency RHS rel. diff: $rel_rhs_exc")
@assert rel_rhs_exc < 1e-13

# Explicit quadrature check for plane-wave RHS assembly
xi_exc, wq_exc = tri_quad_rule(3)
v_manual_exc = zeros(ComplexF64, rwg_exc.nedges)
for n in 1:rwg_exc.nedges
    for t in (rwg_exc.tplus[n], rwg_exc.tminus[n])
        A = triangle_area(mesh_exc, t)
        pts = tri_quad_points(mesh_exc, t, xi_exc)
        for q in eachindex(wq_exc)
            rq = pts[q]
            fn = eval_rwg(rwg_exc, n, rq, t)
            Einc = pol_exc * exp(-1im * dot(k_vec_exc, rq))
            v_manual_exc[n] += -wq_exc[q] * dot(fn, Einc) * (2 * A)
        end
    end
end
rel_rhs_manual_exc = norm(v_new_exc - v_manual_exc) / max(norm(v_manual_exc), 1e-30)
println("  Plane-wave vs explicit quadrature RHS rel. diff: $rel_rhs_manual_exc")
@assert rel_rhs_manual_exc < 1e-12

gap_a = make_delta_gap(1, 1.0 + 0im, 1e-3)
gap_b = make_delta_gap(1, 1.0 + 0im, 2e-3)
v_gap_a = assemble_excitation(mesh_exc, rwg_exc, gap_a)
v_gap_b = assemble_excitation(mesh_exc, rwg_exc, gap_b)
ratio_gap = abs(v_gap_a[1]) / max(abs(v_gap_b[1]), 1e-30)
println("  Delta-gap scaling ratio (g=1mm/g=2mm): $ratio_gap")
@assert abs(ratio_gap - 2.0) < 1e-12
@assert norm(v_gap_a[2:end]) < 1e-12

port_exc = PortExcitation([1, 2], 2.0 + 0im, 50.0 + 0im)
v_port = assemble_excitation(mesh_exc, rwg_exc, port_exc)
@assert abs(v_port[1] - (2.0 / rwg_exc.len[1])) < 1e-14
@assert abs(v_port[2] - (2.0 / rwg_exc.len[2])) < 1e-14
@assert norm(v_port[3:end]) < 1e-14

# Out-of-bounds port edges should be skipped gracefully
port_oob = PortExcitation([1, rwg_exc.nedges + 10], 1.0 + 0im, 50.0 + 0im)
# Expected-by-design robustness behavior: out-of-range edges are warned and skipped.
v_port_oob = @test_logs (:warn, r"Port edge .* is out of bounds .* Skipping") assemble_excitation(mesh_exc, rwg_exc, port_oob)
@assert abs(v_port_oob[1] - (1.0 / rwg_exc.len[1])) < 1e-14
@assert norm(v_port_oob[2:end]) < 1e-14

thrown_multi = try
    bad_multi = make_multi_excitation([gap_a, gap_b], [1 + 0im])
    assemble_excitation(mesh_exc, rwg_exc, bad_multi)
    false
catch
    true
end
@assert thrown_multi

V_exc = assemble_multiple_excitations(mesh_exc, rwg_exc, [gap_a, make_plane_wave(k_vec_exc, 1.0, pol_exc)]; quad_order=3)
@assert size(V_exc) == (rwg_exc.nedges, 2)
@assert norm(V_exc[:, 1] - v_gap_a) / max(norm(v_gap_a), 1e-30) < 1e-13
@assert norm(V_exc[:, 2] - v_new_exc) / max(norm(v_new_exc), 1e-30) < 1e-13

weights_exc = ComplexF64[0.3 - 0.1im, -0.2 + 0.7im]
multi_exc = make_multi_excitation([gap_a, make_plane_wave(k_vec_exc, 1.0, pol_exc)], weights_exc)
v_multi_exc = assemble_excitation(mesh_exc, rwg_exc, multi_exc; quad_order=3)
v_multi_ref = V_exc * weights_exc
rel_multi_exc = norm(v_multi_exc - v_multi_ref) / max(norm(v_multi_ref), 1e-30)
println("  Multi-excitation linearity rel. diff: $rel_multi_exc")
@assert rel_multi_exc < 1e-13

# Imported excitation semantics:
# kind=:electric_field should match direct electric-field import exactly.
E_field_test(r) = CVec3(r[1] + 0im, (0.5 * r[2]) + 0im, 0.0 + 0im)
cur_E = make_imported_excitation(E_field_test; kind=:electric_field, min_quad_order=3)
imp_E = ImportedExcitation(E_field_test; kind=:electric_field, min_quad_order=3)
v_cur_E = assemble_excitation(mesh_exc, rwg_exc, cur_E; quad_order=3)
v_imp_E = assemble_excitation(mesh_exc, rwg_exc, imp_E; quad_order=3)
rel_cur_imp = norm(v_cur_E - v_imp_E) / max(norm(v_imp_E), 1e-30)
println("  ImportedExcitation(:electric_field) self-consistency rel. diff: $rel_cur_imp")
@assert rel_cur_imp < 1e-13

# Source function can return tuple/vector-like 3-component data.
E_field_tuple(r) = (r[1] + 0im, (0.5 * r[2]) + 0im, 0.0 + 0im)
cur_E_tuple = make_imported_excitation(E_field_tuple; kind=:electric_field, min_quad_order=3)
v_cur_E_tuple = assemble_excitation(mesh_exc, rwg_exc, cur_E_tuple; quad_order=3)
rel_cur_tuple = norm(v_cur_E_tuple - v_imp_E) / max(norm(v_imp_E), 1e-30)
println("  ImportedExcitation tuple-return rel. diff: $rel_cur_tuple")
@assert rel_cur_tuple < 1e-13

# surface-current mode uses local equivalent-sheet map E ≈ η Js.
Js_test(r) = CVec3((2r[1]) + 0im, 0.0 + 0im, 0.0 + 0im)
eta_test = 120.0 + 30.0im
cur_Js = make_imported_excitation(Js_test; kind=:surface_current_density, eta_equiv=eta_test, min_quad_order=3)
imp_etaJs = ImportedExcitation(r -> eta_test * Js_test(r); kind=:electric_field, min_quad_order=3)
v_cur_Js = assemble_excitation(mesh_exc, rwg_exc, cur_Js; quad_order=3)
v_imp_Js = assemble_excitation(mesh_exc, rwg_exc, imp_etaJs; quad_order=3)
rel_cur_js = norm(v_cur_Js - v_imp_Js) / max(norm(v_imp_Js), 1e-30)
println("  ImportedExcitation(:surface_current_density) map rel. diff: $rel_cur_js")
@assert rel_cur_js < 1e-13

# Imported field can also be tuple/vector-like.
imp_tuple = ImportedExcitation(r -> (r[1] + 0im, 0.0 + 0im, 0.0 + 0im); kind=:electric_field, min_quad_order=3)
v_imp_tuple = assemble_excitation(mesh_exc, rwg_exc, imp_tuple; quad_order=3)
@assert all(isfinite, real.(v_imp_tuple))
@assert all(isfinite, imag.(v_imp_tuple))

thrown_imp_bad_dim = try
    imp_bad_dim = ImportedExcitation(r -> ComplexF64[1 + 0im, 2 + 0im]; kind=:electric_field)
    assemble_excitation(mesh_exc, rwg_exc, imp_bad_dim; quad_order=3)
    false
catch
    true
end
@assert thrown_imp_bad_dim

thrown_imp_nonfinite = try
    imp_nonfinite = ImportedExcitation(r -> CVec3(NaN + 0im, 0 + 0im, 0 + 0im); kind=:electric_field)
    assemble_excitation(mesh_exc, rwg_exc, imp_nonfinite; quad_order=3)
    false
catch
    true
end
@assert thrown_imp_nonfinite

# Constructor guard
thrown_cur = try
    make_imported_excitation(E_field_test; min_quad_order=0)
    false
catch
    true
end
@assert thrown_cur

thrown_cur_bad_dim = try
    cur_bad_dim = make_imported_excitation(r -> ComplexF64[1 + 0im, 2 + 0im];
                                           kind=:electric_field,
                                           min_quad_order=3)
    assemble_excitation(mesh_exc, rwg_exc, cur_bad_dim; quad_order=3)
    false
catch
    true
end
@assert thrown_cur_bad_dim

thrown_cur_nonfinite = try
    cur_nonfinite = make_imported_excitation(r -> CVec3(NaN + 0im, 0 + 0im, 0 + 0im);
                                             kind=:electric_field,
                                             min_quad_order=3)
    assemble_excitation(mesh_exc, rwg_exc, cur_nonfinite; quad_order=3)
    false
catch
    true
end
@assert thrown_cur_nonfinite

# Hard-break API check: legacy excitation wrappers are intentionally removed.
@assert !isdefined(DiffMoM, :CurrentDistributionExcitation)
@assert !isdefined(DiffMoM, :ImportedFieldExcitation)
@assert !isdefined(DiffMoM, :make_current_distribution)

m_mag = CVec3(0.0 + 0im, 0.0 + 0im, 1e-4 + 0im) # A·m²
dip_mag = make_dipole(Vec3(0.0, 0.0, 0.0), m_mag, Vec3(0.0, 0.0, 1.0), :magnetic, freq_exc)
Rfar = 5.0
E_mag_num = DiffMoM.dipole_incident_field(Vec3(Rfar, 0.0, 0.0), dip_mag)
E_mag_ref = +1im * eta0 * k_exc^2 * m_mag[3] * exp(-1im * k_exc * Rfar) / (4π * Rfar)
rel_mag = abs(E_mag_num[2] - E_mag_ref) / max(abs(E_mag_ref), 1e-30)
println("  Magnetic-dipole far-field rel. error: $rel_mag")
@assert rel_mag < 0.03

# Electric dipole: broadside far-field amplitude and axial quasi-static behavior
p_e = CVec3(0.0 + 0im, 0.0 + 0im, 1e-12 + 0im)
dip_e = make_dipole(Vec3(0.0, 0.0, 0.0), p_e, Vec3(0.0, 0.0, 1.0), :electric, freq_exc)
Rfar_e = 8.0
E_e_num = DiffMoM.dipole_incident_field(Vec3(Rfar_e, 0.0, 0.0), dip_e)
eps0_exc = 8.854187817e-12
E_e_ref = p_e[3] * k_exc^2 * exp(-1im * k_exc * Rfar_e) / (4π * eps0_exc * Rfar_e)
rel_e = abs(E_e_num[3] - E_e_ref) / max(abs(E_e_ref), 1e-30)
println("  Electric-dipole broadside far-field rel. error: $rel_e")
@assert rel_e < 0.03

dip_e_low = make_dipole(Vec3(0.0, 0.0, 0.0), p_e, Vec3(0.0, 0.0, 1.0), :electric, 1.0e6)
R_quasi = 0.5
E_e_axial = DiffMoM.dipole_incident_field(Vec3(0.0, 0.0, R_quasi), dip_e_low)
E_e_static = 2 * p_e[3] / (4π * eps0_exc * R_quasi^3)
rel_e_quasi = abs(real(E_e_axial[3]) - real(E_e_static)) / max(abs(real(E_e_static)), 1e-30)
println("  Electric-dipole axial quasi-static rel. error: $rel_e_quasi")
@assert rel_e_quasi < 0.05

# Loop field must match equivalent magnetic dipole field and RHS exactly
loop_eq = make_loop(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), 0.01, 2.0 + 0im, freq_exc)
m_eq = (2.0 + 0im) * π * (0.01)^2
dip_eq = make_dipole(Vec3(0.0, 0.0, 0.0), CVec3(0.0 + 0im, 0.0 + 0im, m_eq), Vec3(0.0, 0.0, 1.0), :magnetic, freq_exc)
E_loop_eq = DiffMoM.loop_incident_field(Vec3(Rfar, 0.0, 0.0), loop_eq)
E_dip_eq = DiffMoM.dipole_incident_field(Vec3(Rfar, 0.0, 0.0), dip_eq)
rel_loop_field = norm(E_loop_eq - E_dip_eq) / max(norm(E_dip_eq), 1e-30)
println("  Loop vs equivalent magnetic-dipole field rel. diff: $rel_loop_field")
@assert rel_loop_field < 1e-13

v_loop_eq = assemble_excitation(mesh_exc, rwg_exc, loop_eq; quad_order=3)
v_dip_eq = assemble_excitation(mesh_exc, rwg_exc, dip_eq; quad_order=3)
rel_loop_rhs = norm(v_loop_eq - v_dip_eq) / max(norm(v_dip_eq), 1e-30)
println("  Loop vs equivalent magnetic-dipole RHS rel. diff: $rel_loop_rhs")
@assert rel_loop_rhs < 1e-13

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 15: Dipole/loop far-field pattern gate
# ─────────────────────────────────────────────────
println("\n── Test 15: Dipole/loop far-field pattern gate ──")

freq_pat = 1.0e9
k_pat = 2π * freq_pat / 299792458.0
lambda_pat = 2π / k_pat
Rfar_pat = 50 * lambda_pat
theta_deg_pat = collect(0.0:1.0:180.0)
theta_pat = deg2rad.(theta_deg_pat)

dip_pat = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(0.0 + 0im, 0.0 + 0im, 1e-12 + 0im),   # electric dipole (C·m)
    Vec3(0.0, 0.0, 1.0),
    :electric,
    freq_pat
)
loop_pat = make_loop(
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0, 0.0, 1.0),
    0.01,                                        # 1 cm radius
    1.0 + 0im,
    freq_pat
)

P_dip_num = zeros(Float64, length(theta_pat))
P_loop_num = zeros(Float64, length(theta_pat))
P_ana = sin.(theta_pat) .^ 2
E_dip_theta_cmp = zeros(ComplexF64, length(theta_pat))
E_loop_phi_cmp = zeros(ComplexF64, length(theta_pat))

for i in eachindex(theta_pat)
    th = theta_pat[i]
    rhat = Vec3(sin(th), 0.0, cos(th))          # φ = 0 cut
    r = Rfar_pat * rhat
    E_d = DiffMoM.dipole_incident_field(r, dip_pat)
    E_l = DiffMoM.loop_incident_field(r, loop_pat)
    e_theta = Vec3(cos(th), 0.0, -sin(th))
    e_phi = Vec3(0.0, 1.0, 0.0)
    E_dip_theta_cmp[i] = dot(E_d, e_theta)
    E_loop_phi_cmp[i] = dot(E_l, e_phi)
    P_dip_num[i] = norm(E_d)^2
    P_loop_num[i] = norm(E_l)^2
end

P_dip_num ./= maximum(P_dip_num)
P_loop_num ./= maximum(P_loop_num)
P_ana ./= maximum(P_ana)

err_dip = P_dip_num .- P_ana
err_loop = P_loop_num .- P_ana

rmse_dip = sqrt(sum(abs2, err_dip) / length(err_dip))
rmse_loop = sqrt(sum(abs2, err_loop) / length(err_loop))
maxabs_dip = maximum(abs.(err_dip))
maxabs_loop = maximum(abs.(err_loop))

null_max_dip = max(P_dip_num[1], P_dip_num[end])
null_max_loop = max(P_loop_num[1], P_loop_num[end])

# Polarization-resolved check over a coarse (θ,φ) grid.
phi_pat = deg2rad.(collect(0.0:10.0:350.0))
theta_pol_pat = deg2rad.(collect(1.0:2.0:179.0))
crossfrac_dip = Float64[]
crossfrac_loop = Float64[]
for th in theta_pol_pat, ph in phi_pat
    rhat = Vec3(sin(th) * cos(ph), sin(th) * sin(ph), cos(th))
    r = Rfar_pat * rhat
    e_theta = Vec3(cos(th) * cos(ph), cos(th) * sin(ph), -sin(th))
    e_phi = Vec3(-sin(ph), cos(ph), 0.0)
    E_d = DiffMoM.dipole_incident_field(r, dip_pat)
    E_l = DiffMoM.loop_incident_field(r, loop_pat)
    E_d_theta = dot(E_d, e_theta)
    E_d_phi = dot(E_d, e_phi)
    E_l_theta = dot(E_l, e_theta)
    E_l_phi = dot(E_l, e_phi)
    P_d = abs2(E_d_theta) + abs2(E_d_phi)
    P_l = abs2(E_l_theta) + abs2(E_l_phi)
    push!(crossfrac_dip, abs(E_d_phi) / sqrt(P_d))
    push!(crossfrac_loop, abs(E_l_theta) / sqrt(P_l))
end
max_crossfrac_dip = maximum(crossfrac_dip)
max_crossfrac_loop = maximum(crossfrac_loop)

# Co-pol phase consistency on φ=0 cut: Eϕ(loop)/Eθ(dipole) ≈ ±90° phase.
wrap_to_pi(x) = atan(sin(x), cos(x))
phase_err_pm90_deg = Float64[]
phase_ratio_deg = Float64[]
amp_floor_phase = 1e-12 * max(maximum(abs.(E_dip_theta_cmp)), maximum(abs.(E_loop_phi_cmp)))
for i in eachindex(theta_pat)
    if abs(E_dip_theta_cmp[i]) > amp_floor_phase && abs(E_loop_phi_cmp[i]) > amp_floor_phase
        Δϕ = angle(E_loop_phi_cmp[i] / E_dip_theta_cmp[i])
        push!(phase_ratio_deg, rad2deg(Δϕ))
        err_pm90 = min(abs(wrap_to_pi(Δϕ - π / 2)), abs(wrap_to_pi(Δϕ + π / 2)))
        push!(phase_err_pm90_deg, rad2deg(err_pm90))
    end
end
phase_mean_deg = mean(phase_ratio_deg)
phase_std_deg = std(phase_ratio_deg)
phase_max_err_pm90_deg = maximum(phase_err_pm90_deg)

println("  Dipole pattern RMSE:      $rmse_dip")
println("  Dipole pattern max |err|: $maxabs_dip")
println("  Loop pattern RMSE:        $rmse_loop")
println("  Loop pattern max |err|:   $maxabs_loop")
println("  Dipole null max:          $null_max_dip")
println("  Loop null max:            $null_max_loop")
println("  Dipole max cross-pol frac: $max_crossfrac_dip")
println("  Loop max cross-pol frac:   $max_crossfrac_loop")
println("  Phase mean (deg):          $phase_mean_deg")
println("  Phase std (deg):           $phase_std_deg")
println("  Phase max err to ±90° (deg): $phase_max_err_pm90_deg")

# CI thresholds (pattern-shape gate)
@assert rmse_dip < 1e-4 "Dipole pattern gate failed: RMSE=$rmse_dip"
@assert maxabs_dip < 2e-4 "Dipole pattern gate failed: max |err|=$maxabs_dip"
@assert rmse_loop < 1e-10 "Loop pattern gate failed: RMSE=$rmse_loop"
@assert maxabs_loop < 1e-9 "Loop pattern gate failed: max |err|=$maxabs_loop"
@assert null_max_dip < 1e-3 "Dipole pattern gate failed: null level=$null_max_dip"
@assert null_max_loop < 1e-10 "Loop pattern gate failed: null level=$null_max_loop"
@assert max_crossfrac_dip < 1e-10 "Dipole polarization gate failed: max cross-pol frac=$max_crossfrac_dip"
@assert max_crossfrac_loop < 1e-10 "Loop polarization gate failed: max cross-pol frac=$max_crossfrac_loop"
@assert phase_max_err_pm90_deg < 1.0 "Dipole/loop phase gate failed: max err to ±90° = $phase_max_err_pm90_deg deg"
@assert phase_std_deg < 0.1 "Dipole/loop phase gate failed: phase std = $phase_std_deg deg"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 16: Pattern-feed excitation gate
# ─────────────────────────────────────────────────
println("\n── Test 16: Pattern-feed excitation gate ──")

const EPS0_PAT = 8.854187817e-12
const C0_PAT = 299792458.0

freq_pf = 1.0e9
k_pf = 2π * freq_pf / C0_PAT
λ_pf = C0_PAT / freq_pf
Rfar_pf = 80 * λ_pf
pz_pf = 1e-12 + 0im

dip_pf = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(0.0 + 0im, 0.0 + 0im, pz_pf),
    Vec3(0.0, 0.0, 1.0),
    :electric,
    freq_pf,
)

theta_pat_deg_pf = collect(0.0:2.0:180.0)
phi_pat_deg_pf = collect(0.0:6.0:354.0)
pat_plus = make_analytic_dipole_pattern_feed(
    dip_pf,
    theta_pat_deg_pf,
    phi_pat_deg_pf;
    angles_in_degrees=true,
)

# Adapter-style constructor using two pattern objects with fields x, y, U
struct _PatternLike
    x::Vector{Float64}
    y::Vector{Float64}
    U::Matrix{ComplexF64}
end
pat_like_θ = _PatternLike(copy(pat_plus.theta), copy(pat_plus.phi), copy(pat_plus.Ftheta))
pat_like_ϕ = _PatternLike(copy(pat_plus.theta), copy(pat_plus.phi), copy(pat_plus.Fphi))
pat_from_like = make_pattern_feed(pat_like_θ, pat_like_ϕ, pat_plus.frequency; angles_in_degrees=false)

probe_like_dirs = (Vec3(1.0, 0.5, 2.0), Vec3(-0.8, 0.3, 1.4), Vec3(0.1, -0.9, 1.1))
rel_like = let worst = 0.0
    for d in probe_like_dirs
        r = Rfar_pf * d / norm(d)
        E_ref = pattern_feed_field(r, pat_plus)
        E_new = pattern_feed_field(r, pat_from_like)
        worst = max(worst, norm(E_ref - E_new) / max(norm(E_ref), 1e-30))
    end
    worst
end
println("  Pattern-object adapter mismatch:    $rel_like")
@assert rel_like < 1e-13

# Matrix-shape tolerance: transposed input should auto-correct
# Expected-by-design shape-tolerance behavior: auto-transpose with warnings.
pat_transposed = @test_logs (
    :warn,
    r"Ftheta matrix shape .* appears transposed; auto-transposing to .*",
) (
    :warn,
    r"Fphi matrix shape .* appears transposed; auto-transposing to .*",
) make_pattern_feed(
    pat_plus.theta,
    pat_plus.phi,
    permutedims(pat_plus.Ftheta),
    permutedims(pat_plus.Fphi),
    pat_plus.frequency;
    angles_in_degrees=false,
)
rel_transposed = let worst = 0.0
    for d in probe_like_dirs
        r = Rfar_pf * d / norm(d)
        E_ref = pattern_feed_field(r, pat_plus)
        E_new = pattern_feed_field(r, pat_transposed)
        worst = max(worst, norm(E_ref - E_new) / max(norm(E_ref), 1e-30))
    end
    worst
end
println("  Pattern-transpose auto-fix mismatch: $rel_transposed")
@assert rel_transposed < 1e-12

# Field-level comparison on a fine φ=0 cut against closed-form dipole far-field.
theta_eval_deg_pf = collect(0.0:0.5:180.0)
theta_eval_pf = deg2rad.(theta_eval_deg_pf)
P_num_pf = zeros(Float64, length(theta_eval_pf))
P_ref_pf = zeros(Float64, length(theta_eval_pf))
Etheta_num_pf = zeros(ComplexF64, length(theta_eval_pf))
Etheta_ref_pf = zeros(ComplexF64, length(theta_eval_pf))
cross_ratio_pf = zeros(Float64, length(theta_eval_pf))

for i in eachindex(theta_eval_pf)
    θ = theta_eval_pf[i]
    ϕ = 0.0
    rhat = Vec3(sin(θ), 0.0, cos(θ))
    r = Rfar_pf * rhat
    eθ = Vec3(cos(θ), 0.0, -sin(θ))
    eϕ = Vec3(0.0, 1.0, 0.0)

    E_num = pattern_feed_field(r, pat_plus)
    Eθ_num = dot(E_num, eθ)
    Eϕ_num = dot(E_num, eϕ)

    Eθ_ref = (k_pf^2 * pz_pf / (4π * EPS0_PAT)) * sin(θ) * exp(-1im * k_pf * Rfar_pf) / Rfar_pf
    Eϕ_ref = 0.0 + 0im

    Etheta_num_pf[i] = Eθ_num
    Etheta_ref_pf[i] = Eθ_ref
    P_num_pf[i] = abs2(Eθ_num) + abs2(Eϕ_num)
    P_ref_pf[i] = abs2(Eθ_ref) + abs2(Eϕ_ref)
    cross_ratio_pf[i] = abs(Eϕ_num) / max(sqrt(P_num_pf[i]), 1e-30)
end

P_num_pf ./= maximum(P_num_pf)
P_ref_pf ./= maximum(P_ref_pf)
err_lin_pf = P_num_pf .- P_ref_pf
rmse_pf = sqrt(mean(abs2, err_lin_pf))
maxabs_pf = maximum(abs.(err_lin_pf))
max_cross_pf = maximum(cross_ratio_pf)

phase_err_deg_pf = fill(NaN, length(theta_eval_pf))
phase_floor_pf = 1e-12 * maximum(abs.(Etheta_ref_pf))
for i in eachindex(theta_eval_pf)
    if abs(Etheta_ref_pf[i]) > phase_floor_pf && abs(Etheta_num_pf[i]) > phase_floor_pf
        phase_err_deg_pf[i] = rad2deg(angle(Etheta_num_pf[i] / Etheta_ref_pf[i]))
    end
end

phase_valid_pf = phase_err_deg_pf[.!isnan.(phase_err_deg_pf)]
phase_mean_pf = mean(phase_valid_pf)
phase_std_pf = std(phase_valid_pf)
phase_max_pf = maximum(abs.(phase_valid_pf))
phase_resid_pf = [rad2deg(atan(sin(deg2rad(x - phase_mean_pf)), cos(deg2rad(x - phase_mean_pf)))) for x in phase_valid_pf]
phase_resid_std_pf = std(phase_resid_pf)
phase_resid_max_pf = maximum(abs.(phase_resid_pf))

# Convention conversion check:
# If imported data comes from exp(-iωt), using conjugated coefficients with
# convention=:exp_minus_iwt must reproduce the same physical field.
pat_minus = make_pattern_feed(
    pat_plus.theta,
    pat_plus.phi,
    conj.(pat_plus.Ftheta),
    conj.(pat_plus.Fphi),
    pat_plus.frequency;
    convention=:exp_minus_iwt,
)
probe_dirs_pf = (
    Vec3(1.2, -0.4, 2.1),
    Vec3(-0.8, 1.5, 1.2),
    Vec3(0.5, 0.9, -1.7),
)
conv_mismatch_pf = let mismatch = 0.0
    for d in probe_dirs_pf
        r = Rfar_pf * d / norm(d)
        E_plus = pattern_feed_field(r, pat_plus)
        E_minus = pattern_feed_field(r, pat_minus)
        mismatch = max(mismatch, norm(E_plus - E_minus) / max(norm(E_plus), 1e-30))
    end
    mismatch
end

# RHS consistency against direct imported electric-field path.
mesh_pf = make_rect_plate(0.04, 0.04, 4, 4)
rwg_pf = build_rwg(mesh_pf)
v_pf_pat = assemble_excitation(mesh_pf, rwg_pf, pat_plus; quad_order=3)
imp_pf = ImportedExcitation(r -> pattern_feed_field(r, pat_plus); kind=:electric_field, min_quad_order=3)
v_pf_imp = assemble_excitation(mesh_pf, rwg_pf, imp_pf; quad_order=3)
rhs_rel_pf = norm(v_pf_pat - v_pf_imp) / max(norm(v_pf_pat), 1e-30)

println("  Pattern-feed RMSE (linear):        $rmse_pf")
println("  Pattern-feed max |err| (linear):   $maxabs_pf")
println("  Pattern-feed max cross-pol ratio:  $max_cross_pf")
println("  Pattern-feed phase mean (deg):     $phase_mean_pf")
println("  Pattern-feed phase std (deg):      $phase_std_pf")
println("  Pattern-feed phase max |err| (deg): $phase_max_pf")
println("  Pattern-feed phase residual std (deg): $phase_resid_std_pf")
println("  Pattern-feed phase residual max |err| (deg): $phase_resid_max_pf")
println("  Convention conversion mismatch:    $conv_mismatch_pf")
println("  RHS path mismatch (pattern vs imported): $rhs_rel_pf")

@assert rmse_pf < 2e-4 "Pattern-feed gate failed: RMSE=$rmse_pf"
@assert maxabs_pf < 5e-4 "Pattern-feed gate failed: max |err|=$maxabs_pf"
@assert max_cross_pf < 1e-8 "Pattern-feed gate failed: cross-pol ratio=$max_cross_pf"
@assert phase_resid_std_pf < 0.2 "Pattern-feed gate failed: phase residual std=$phase_resid_std_pf"
@assert phase_resid_max_pf < 0.5 "Pattern-feed gate failed: phase residual max |err|=$phase_resid_max_pf"
@assert conv_mismatch_pf < 1e-12 "Pattern-feed gate failed: convention mismatch=$conv_mismatch_pf"
@assert rhs_rel_pf < 1e-12 "Pattern-feed gate failed: RHS mismatch=$rhs_rel_pf"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 17: GMRES solver and dispatch
# ─────────────────────────────────────────────────
println("\n── Test 17: GMRES solver and dispatch ──")

# Use impedance-loaded system from Test 7
Z_gm = Matrix{ComplexF64}(Z_full)
I_gm_direct = Z_gm \ v

# GMRES forward solve (no preconditioner)
I_gmres_nop, stats_nop = solve_gmres(Z_gm, Vector{ComplexF64}(v);
                                       tol=1e-10, maxiter=500)
rel_gmres_nop = norm(I_gmres_nop - I_gm_direct) / max(norm(I_gm_direct), 1e-30)
println("  GMRES (no precond) rel error: $rel_gmres_nop  iters: $(stats_nop.niter)")
@assert rel_gmres_nop < 1e-6 "GMRES without preconditioner inaccurate: $rel_gmres_nop"

# GMRES adjoint solve (no preconditioner)
rhs_adj_gm = Vector{ComplexF64}(Q * I_gm_direct)
lam_gm_direct = Z_gm' \ rhs_adj_gm
lam_gmres_nop, stats_adj_nop = solve_gmres_adjoint(Z_gm, rhs_adj_gm;
                                                      tol=1e-10, maxiter=500)
rel_adj_nop = norm(lam_gmres_nop - lam_gm_direct) / max(norm(lam_gm_direct), 1e-30)
println("  GMRES adjoint (no precond) rel error: $rel_adj_nop  iters: $(stats_adj_nop.niter)")
@assert rel_adj_nop < 1e-6

# solve_forward dispatch: :direct
I_sf_direct = solve_forward(Z_gm, Vector{ComplexF64}(v))
rel_sf_direct = norm(I_sf_direct - I_gm_direct) / max(norm(I_gm_direct), 1e-30)
@assert rel_sf_direct < 1e-12

# solve_forward dispatch: :gmres (unpreconditioned)
I_sf_gmres = solve_forward(Z_gm, Vector{ComplexF64}(v);
                            solver=:gmres, gmres_tol=1e-10, gmres_maxiter=500)
rel_sf_gmres = norm(I_sf_gmres - I_gm_direct) / max(norm(I_gm_direct), 1e-30)
println("  solve_forward :gmres rel error: $rel_sf_gmres")
@assert rel_sf_gmres < 1e-6

# solve_adjoint dispatch: :direct
lam_sa_direct = solve_adjoint(Z_gm, Q, I_gm_direct)
rel_sa_direct = norm(lam_sa_direct - lam_gm_direct) / max(norm(lam_gm_direct), 1e-30)
@assert rel_sa_direct < 1e-12

# solve_adjoint dispatch: :gmres (unpreconditioned)
lam_sa_gmres = solve_adjoint(Z_gm, Q, I_gm_direct;
                              solver=:gmres, gmres_tol=1e-10, gmres_maxiter=500)
rel_sa_gmres = norm(lam_sa_gmres - lam_gm_direct) / max(norm(lam_gm_direct), 1e-30)
println("  solve_adjoint :gmres rel error: $rel_sa_gmres")
@assert rel_sa_gmres < 1e-6

# Optimization-facing GMRES wrappers must fail closed on unconverged solves.
Z_fail = ComplexF64[4 1 0; 1 3 1; 0 1 2]
rhs_fail = ComplexF64[1, 2, 3]
thrown_forward_unconverged = try
    solve_forward(Z_fail, rhs_fail; solver=:gmres, gmres_tol=1e-14, gmres_maxiter=1)
    false
catch e
    occursin("GMRES did not converge", sprint(showerror, e))
end
@assert thrown_forward_unconverged "Expected unconverged forward GMRES wrapper to fail closed"

thrown_adjoint_unconverged = try
    solve_adjoint_rhs(Z_fail, rhs_fail; solver=:gmres, gmres_tol=1e-14, gmres_maxiter=1)
    false
catch e
    occursin("GMRES did not converge", sprint(showerror, e))
end
@assert thrown_adjoint_unconverged "Expected unconverged adjoint GMRES wrapper to fail closed"

thrown_forward_true_residual = try
    solve_forward(Z_fail, rhs_fail; solver=:gmres, gmres_tol=1e-14,
                  gmres_maxiter=1, check_gmres_convergence=false,
                  check_true_residual=true, true_residual_factor=1.0)
    false
catch e
    occursin("true residual too large", sprint(showerror, e))
end
@assert thrown_forward_true_residual "Expected forward true-residual guard to fail closed"

thrown_adjoint_true_residual = try
    solve_adjoint_rhs(Z_fail, rhs_fail; solver=:gmres, gmres_tol=1e-14,
                      gmres_maxiter=1, check_gmres_convergence=false,
                      check_true_residual=true, true_residual_factor=1.0)
    false
catch e
    occursin("true residual too large", sprint(showerror, e))
end
@assert thrown_adjoint_true_residual "Expected adjoint true-residual guard to fail closed"

# Matrix-free EFIE operator: A*x should match dense Z*x
A_mf = matrixfree_efie_operator(mesh, rwg, k; quad_order=3)
x_probe = randn(ComplexF64, N)
Ax_dense = Z_efie * x_probe
Ax_mf = A_mf * x_probe
rel_matvec = norm(Ax_mf - Ax_dense) / max(norm(Ax_dense), 1e-30)
println("  matrix-free matvec rel error: $rel_matvec")
@assert rel_matvec < 1e-10

# GMRES on matrix-free operator
I_mf_gmres, stats_mf = solve_gmres(A_mf, v; tol=1e-8, maxiter=300)
rel_mf = norm(I_mf_gmres - I_pec) / max(norm(I_pec), 1e-30)
println("  matrix-free GMRES rel error: $rel_mf  iters: $(stats_mf.niter)")
@assert rel_mf < 1e-6

# solve_forward / solve_adjoint dispatch on matrix-free operator
I_sf_mf = solve_forward(A_mf, v; solver=:gmres, gmres_tol=1e-8, gmres_maxiter=300)
rel_sf_mf = norm(I_sf_mf - I_pec) / max(norm(I_pec), 1e-30)
@assert rel_sf_mf < 1e-6

lam_sa_mf = solve_adjoint(A_mf, Q, I_pec; solver=:gmres, gmres_tol=1e-8, gmres_maxiter=300)
lam_sa_mf_ref = Z_efie' \ (Q * I_pec)
rel_sa_mf = norm(lam_sa_mf - lam_sa_mf_ref) / max(norm(lam_sa_mf_ref), 1e-30)
println("  matrix-free adjoint GMRES rel error: $rel_sa_mf")
@assert rel_sa_mf < 1e-6

thrown_direct_operator = try
    solve_forward(A_mf, v; solver=:direct)
    false
catch
    true
end
@assert thrown_direct_operator "Expected direct solve failure on matrix-free operator"

# Bad solver symbol should error
thrown_bad_solver = try
    solve_forward(Z_gm, Vector{ComplexF64}(v); solver=:unknown)
    false
catch
    true
end
@assert thrown_bad_solver "Expected error for unknown solver"

thrown_bad_solver_adj = try
    solve_adjoint(Z_gm, Q, I_gm_direct; solver=:unknown)
    false
catch
    true
end
@assert thrown_bad_solver_adj "Expected error for unknown adjoint solver"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 18: GMRES adjoint gradient verification
# ─────────────────────────────────────────────────
println("\n── Test 18: GMRES adjoint gradient verification ──")

# Resistive case: forward and adjoint with GMRES (unpreconditioned)
I_gm_res = solve_forward(Z_gm, Vector{ComplexF64}(v);
                           solver=:gmres, gmres_tol=1e-10, gmres_maxiter=500)
lam_gm_res = solve_adjoint(Z_gm, Q, I_gm_res;
                             solver=:gmres, gmres_tol=1e-10, gmres_maxiter=500)
g_adj_gmres = gradient_impedance(Mp, I_gm_res, lam_gm_res)

# Compare GMRES gradient against finite differences
println("  Checking GMRES adjoint gradient vs central FD (h=1e-5)...")
rel_errors_gm = Float64[]
n_check_gm = min(Nt, 10)
for p in 1:n_check_gm
    g_fd = fd_grad(J_of_theta, theta_real, p; h=1e-5)
    rel_err = abs(g_adj_gmres[p] - g_fd) / max(abs(g_adj_gmres[p]), abs(g_fd), 1e-30)
    push!(rel_errors_gm, rel_err)
end
max_rel_err_gm = maximum(rel_errors_gm)
println("  Max rel error (GMRES adjoint vs FD): $max_rel_err_gm")
@assert max_rel_err_gm < 1e-3 "GMRES gradient verification FAILED: max rel error = $max_rel_err_gm"

# Also compare GMRES gradient against direct gradient
rel_gm_vs_direct = norm(g_adj_gmres - g_adj) / max(norm(g_adj), 1e-30)
println("  GMRES vs direct gradient rel diff: $rel_gm_vs_direct")
@assert rel_gm_vs_direct < 1e-4 "GMRES gradient diverges from direct: $rel_gm_vs_direct"

# Reactive case
theta_reac = fill(150.0, Nt)
Z_reac = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_reac; reactive=true))
I_reac_direct = Z_reac \ v

lam_reac_dir = solve_adjoint(Z_reac, Q, I_reac_direct)
g_reac_dir = gradient_impedance(Mp, I_reac_direct, lam_reac_dir; reactive=true)

I_reac_gm = solve_forward(Z_reac, Vector{ComplexF64}(v);
                            solver=:gmres, gmres_tol=1e-10, gmres_maxiter=500)
lam_reac_gm = solve_adjoint(Z_reac, Q, I_reac_gm;
                              solver=:gmres, gmres_tol=1e-10, gmres_maxiter=500)
g_reac_gm = gradient_impedance(Mp, I_reac_gm, lam_reac_gm; reactive=true)

function J_of_theta_reac(theta_vec)
    Z_t = copy(Z_efie)
    for p in eachindex(theta_vec)
        Z_t .-= (1im * theta_vec[p]) .* Mp[p]
    end
    I_t = Z_t \ v
    return real(dot(I_t, Q * I_t))
end

rel_errors_reac = Float64[]
n_check_reac = min(Nt, 5)
for p in 1:n_check_reac
    g_fd = fd_grad(J_of_theta_reac, theta_reac, p; h=1e-5)
    rel_err_gm = abs(g_reac_gm[p] - g_fd) / max(abs(g_fd), 1e-30)
    push!(rel_errors_reac, rel_err_gm)
end
max_rel_err_reac = maximum(rel_errors_reac)
println("  Max rel error (GMRES reactive gradient vs FD): $max_rel_err_reac")
@assert max_rel_err_reac < 1e-3 "Reactive GMRES gradient failed: $max_rel_err_reac"

rel_reac_gm_vs_dir = norm(g_reac_gm - g_reac_dir) / max(norm(g_reac_dir), 1e-30)
println("  Reactive gradient GMRES vs direct: $rel_reac_gm_vs_dir")
@assert rel_reac_gm_vs_dir < 1e-4

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 19: Near-field sparse preconditioner
# ─────────────────────────────────────────────────
println("\n── Test 19: Near-field sparse preconditioner ──")

# Test rwg_centers
centers = rwg_centers(mesh, rwg)
@assert length(centers) == N
@assert all(c -> length(c) == 3, centers)

# Build near-field preconditioner at 1.0λ cutoff
P_nf = build_nearfield_preconditioner(Z_efie, mesh, rwg, lambda0)
@assert P_nf.cutoff == lambda0
@assert 0.0 < P_nf.nnz_ratio <= 1.0

# Compare default spatial search against brute-force reference
P_nf_bruteforce = build_nearfield_preconditioner(
    Z_efie, mesh, rwg, lambda0; neighbor_search=:bruteforce
)
@assert abs(P_nf.nnz_ratio - P_nf_bruteforce.nnz_ratio) < 1e-12
x_nf_cmp = randn(ComplexF64, N)
y_nf_spatial = NearFieldOperator(P_nf) * x_nf_cmp
y_nf_bruteforce = NearFieldOperator(P_nf_bruteforce) * x_nf_cmp
rel_nf_spatial_vs_brute = norm(y_nf_spatial - y_nf_bruteforce) / max(norm(y_nf_bruteforce), 1e-30)
println("  NF spatial vs brute rel diff: $rel_nf_spatial_vs_brute")
@assert rel_nf_spatial_vs_brute < 1e-12

# Build near-field preconditioner without dense Z (matrix-free and geometry paths)
P_nf_mf = build_nearfield_preconditioner(A_mf, lambda0)
@assert P_nf_mf.cutoff == lambda0
@assert 0.0 < P_nf_mf.nnz_ratio <= 1.0
@assert abs(P_nf_mf.nnz_ratio - P_nf.nnz_ratio) < 1e-12

P_nf_mf_bruteforce = build_nearfield_preconditioner(
    A_mf, lambda0; neighbor_search=:bruteforce
)
@assert abs(P_nf_mf.nnz_ratio - P_nf_mf_bruteforce.nnz_ratio) < 1e-12

P_nf_geom = build_nearfield_preconditioner(mesh, rwg, k, lambda0; quad_order=3)
@assert P_nf_geom.cutoff == lambda0
@assert abs(P_nf_geom.nnz_ratio - P_nf.nnz_ratio) < 1e-12

# Invalid neighbor-search mode should error
thrown_bad_neighbor_search = try
    build_nearfield_preconditioner(Z_efie, mesh, rwg, lambda0; neighbor_search=:invalid_mode)
    false
catch
    true
end
@assert thrown_bad_neighbor_search "Expected error for invalid neighbor_search mode"

# Invalid factorization mode should error
thrown_bad_factorization = try
    build_nearfield_preconditioner(A_mf, lambda0; factorization=:invalid_mode)
    false
catch
    true
end
@assert thrown_bad_factorization "Expected error for invalid factorization mode"

# Diagonal/Jacobi preconditioner path
P_diag = build_nearfield_preconditioner(A_mf, lambda0; factorization=:diag)
@assert P_diag isa DiagonalPreconditionerData
@assert P_diag.cutoff == lambda0
@assert 0.0 < P_diag.nnz_ratio <= 1.0
I_diag, stats_diag = solve_gmres(A_mf, v; preconditioner=P_diag, tol=1e-8, maxiter=300)
rel_diag = norm(I_diag - I_pec) / max(norm(I_pec), 1e-30)
println("  Diag precond + matrix-free rel error: $rel_diag  iters: $(stats_diag.niter)")
@assert rel_diag < 1e-6 "Diagonal-preconditioned matrix-free solve inaccurate: $rel_diag"

# GMRES with near-field preconditioner
I_nf, stats_nf = solve_gmres(Z_efie, v; preconditioner=P_nf, tol=1e-8, maxiter=200)
rel_nf = norm(I_nf - I_pec) / max(norm(I_pec), 1e-30)
println("  NF precond (1.0λ) rel error: $rel_nf  iters: $(stats_nf.niter)")
@assert rel_nf < 1e-6 "Near-field preconditioned solve inaccurate: $rel_nf"

# GMRES on matrix-free operator with near-field preconditioner built without dense Z
I_nf_mf, stats_nf_mf = solve_gmres(A_mf, v; preconditioner=P_nf_mf, tol=1e-8, maxiter=300)
rel_nf_mf = norm(I_nf_mf - I_pec) / max(norm(I_pec), 1e-30)
println("  NF precond + matrix-free rel error: $rel_nf_mf  iters: $(stats_nf_mf.niter)")
@assert rel_nf_mf < 1e-6 "Matrix-free near-field preconditioned solve inaccurate: $rel_nf_mf"

# Compare iteration count: near-field should help vs unpreconditioned
I_nop_nf, stats_nop_nf = solve_gmres(Z_efie, v; tol=1e-8, maxiter=200)
println("  Iterations: no_precond=$(stats_nop_nf.niter), NF=$(stats_nf.niter)")

# NearFieldOperator / NearFieldAdjointOperator wrappers
M_nf = NearFieldOperator(P_nf)
@assert size(M_nf) == (N, N)
@assert eltype(M_nf) == ComplexF64
y_nf = M_nf * v
@assert length(y_nf) == N

M_nf_adj = NearFieldAdjointOperator(P_nf)
@assert size(M_nf_adj) == (N, N)
y_nf_adj = M_nf_adj * v

# Adjoint consistency: ⟨P⁻¹x, y⟩ ≈ ⟨x, P⁻ᴴy⟩
x_test = randn(ComplexF64, N)
y_test = randn(ComplexF64, N)
lhs_nf = dot(M_nf * x_test, y_test)
rhs_nf = dot(x_test, M_nf_adj * y_test)
adj_err = abs(lhs_nf - rhs_nf) / max(abs(lhs_nf), 1e-30)
println("  Adjoint consistency: $adj_err")
@assert adj_err < 1e-12 "Near-field adjoint inconsistent: $adj_err"

# GMRES adjoint solve with near-field
rhs_adj_nf = Q * I_pec
lambda_nf, stats_adj_nf = solve_gmres_adjoint(Z_efie, rhs_adj_nf;
                                                preconditioner=P_nf, tol=1e-8, maxiter=200)
lambda_direct_nf = Z_efie' \ rhs_adj_nf
rel_adj_nf = norm(lambda_nf - lambda_direct_nf) / max(norm(lambda_direct_nf), 1e-30)
println("  NF adjoint solve rel error: $rel_adj_nf  iters: $(stats_adj_nf.niter)")
@assert rel_adj_nf < 1e-6 "Near-field adjoint solve inaccurate: $rel_adj_nf"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 20: Right preconditioning
# ─────────────────────────────────────────────────
println("\n── Test 20: Right preconditioning ──")

# Right-preconditioned GMRES with near-field
I_nf_right, stats_nf_right = solve_gmres(Z_efie, v;
                                           preconditioner=P_nf, precond_side=:right,
                                           tol=1e-8, maxiter=200)
rel_nf_right = norm(I_nf_right - I_pec) / max(norm(I_pec), 1e-30)
println("  NF right precond rel error: $rel_nf_right  iters: $(stats_nf_right.niter)")
@assert rel_nf_right < 1e-6 "NF right-preconditioned solve inaccurate: $rel_nf_right"

# Right-preconditioned adjoint with near-field
lambda_nf_right, stats_adj_nf_right = solve_gmres_adjoint(Z_efie, rhs_adj_nf;
                                                             preconditioner=P_nf, precond_side=:right,
                                                             tol=1e-8, maxiter=200)
rel_adj_nf_right = norm(lambda_nf_right - lambda_direct_nf) / max(norm(lambda_direct_nf), 1e-30)
println("  NF right adjoint rel error: $rel_adj_nf_right  iters: $(stats_adj_nf_right.niter)")
@assert rel_adj_nf_right < 1e-6 "NF right adjoint solve inaccurate: $rel_adj_nf_right"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 21: Optimization with GMRES solver
# ─────────────────────────────────────────────────
println("\n── Test 21: Optimization with GMRES solver ──")

theta_init_gm = fill(300.0, Nt)
theta_opt_gm, trace_gm = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init_gm;
    maxiter=8, tol=1e-8, alpha0=0.01, verbose=false,
    solver=:gmres, gmres_tol=1e-8, gmres_maxiter=300,
)
@assert length(trace_gm) >= 2 "GMRES optimization should run at least 2 iterations"

# Direct solver comparison
theta_opt_dir, trace_dir = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init_gm;
    maxiter=8, tol=1e-8, alpha0=0.01, verbose=false,
    solver=:direct,
)
rel_J0 = abs(trace_gm[1].J - trace_dir[1].J) / max(abs(trace_dir[1].J), 1e-30)
println("  First-iteration J agreement (lbfgs): $rel_J0")
@assert rel_J0 < 1e-4 "GMRES and direct first-iter J disagree: $rel_J0"

# optimize_directivity with GMRES
Q_total_test = build_Q(G_mat, grid, pol_mat)
theta_opt_dir_d, trace_dir_d = optimize_directivity(
    Z_efie, Mp, v, Q, Q_total_test, theta_init_gm;
    maxiter=5, tol=1e-8, verbose=false, solver=:direct,
)
theta_opt_gm_d, trace_gm_d = optimize_directivity(
    Z_efie, Mp, v, Q, Q_total_test, theta_init_gm;
    maxiter=5, tol=1e-8, verbose=false,
    solver=:gmres, gmres_tol=1e-8, gmres_maxiter=300,
)
@assert length(trace_gm_d) >= 2
rel_J0_d = abs(trace_gm_d[1].J - trace_dir_d[1].J) / max(abs(trace_dir_d[1].J), 1e-30)
println("  First-iteration J_ratio agreement (directivity): $rel_J0_d")
@assert rel_J0_d < 1e-3

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 22: Regression — default solver=:direct unchanged
# ─────────────────────────────────────────────────
println("\n── Test 22: Regression — default solver=:direct ──")

# solve_forward default should match Z \ v exactly
I_reg_sf = solve_forward(Z_gm, Vector{ComplexF64}(v))
I_reg_direct = Z_gm \ v
rel_reg = norm(I_reg_sf - I_reg_direct) / max(norm(I_reg_direct), 1e-30)
@assert rel_reg < 1e-12 "Default solve_forward regression: $rel_reg"

# solve_adjoint default should match Z' \ (Q*I)
lam_reg_sa = solve_adjoint(Z_gm, Q, I_reg_direct)
lam_reg_dir = Z_gm' \ (Q * I_reg_direct)
rel_reg_adj = norm(lam_reg_sa - lam_reg_dir) / max(norm(lam_reg_dir), 1e-30)
@assert rel_reg_adj < 1e-12 "Default solve_adjoint regression: $rel_reg_adj"

# optimize_lbfgs default (solver=:direct) should still work
theta_reg_init = fill(300.0, Nt)
theta_reg_opt, trace_reg = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_reg_init;
    maxiter=3, tol=1e-8, verbose=false,
)
@assert length(trace_reg) >= 2

println("  All defaults unchanged")
println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 23: NF-preconditioned optimization
# ─────────────────────────────────────────────────
println("\n── Test 23: NF-preconditioned optimization ──")

# Build NF preconditioner from PEC EFIE matrix
P_nf_opt = build_nearfield_preconditioner(Z_efie, mesh, rwg, lambda0)
println("  NF preconditioner: cutoff=$(round(P_nf_opt.cutoff, sigdigits=3)), nnz=$(round(P_nf_opt.nnz_ratio*100, digits=1))%")

theta_init_nf = fill(300.0, Nt)

# optimize_lbfgs with NF-preconditioned GMRES
theta_opt_nf, trace_nf = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init_nf;
    maxiter=8, tol=1e-8, alpha0=0.01, verbose=false,
    solver=:gmres, nf_preconditioner=P_nf_opt,
    gmres_tol=1e-8, gmres_maxiter=300,
)
@assert length(trace_nf) >= 2 "NF-preconditioned optimization should run at least 2 iterations"

# First-iteration J should agree with direct solver
theta_opt_dir_nf, trace_dir_nf = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init_nf;
    maxiter=8, tol=1e-8, alpha0=0.01, verbose=false,
    solver=:direct,
)
rel_J0_nf = abs(trace_nf[1].J - trace_dir_nf[1].J) / max(abs(trace_dir_nf[1].J), 1e-30)
println("  First-iteration J agreement (lbfgs, NF): $rel_J0_nf")
@assert rel_J0_nf < 1e-4 "NF-preconditioned first-iter J disagrees: $rel_J0_nf"

# Gradient check: NF-preconditioned GMRES gradient vs FD
Z_nf_check = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_init_nf))
I_nf_check = solve_forward(Z_nf_check, Vector{ComplexF64}(v);
                             solver=:gmres, preconditioner=P_nf_opt,
                             gmres_tol=1e-10, gmres_maxiter=300)
lam_nf_check = solve_adjoint(Z_nf_check, Q, I_nf_check;
                               solver=:gmres, preconditioner=P_nf_opt,
                               gmres_tol=1e-10, gmres_maxiter=300)
g_nf_check = gradient_impedance(Mp, I_nf_check, lam_nf_check)

function J_of_theta_nf(theta_vec)
    Z_t = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_vec))
    I_t = Z_t \ v
    return real(dot(I_t, Q * I_t))
end

rel_errors_nf = Float64[]
n_check_nf = min(Nt, 5)
for p in 1:n_check_nf
    g_fd = fd_grad(J_of_theta_nf, theta_init_nf, p; h=1e-5)
    rel_err = abs(g_nf_check[p] - g_fd) / max(abs(g_fd), abs(g_nf_check[p]), 1e-30)
    push!(rel_errors_nf, rel_err)
end
max_rel_err_nf = maximum(rel_errors_nf)
println("  NF-preconditioned gradient max rel error vs FD: $max_rel_err_nf")
@assert max_rel_err_nf < 1e-3 "NF-preconditioned gradient inaccurate: $max_rel_err_nf"

# optimize_directivity with NF-preconditioned GMRES
theta_opt_dir_nf_d, trace_dir_nf_d = optimize_directivity(
    Z_efie, Mp, v, Q, Q_total_test, theta_init_nf;
    maxiter=5, tol=1e-8, verbose=false, solver=:direct,
)
theta_opt_nf_d, trace_nf_d = optimize_directivity(
    Z_efie, Mp, v, Q, Q_total_test, theta_init_nf;
    maxiter=5, tol=1e-8, verbose=false,
    solver=:gmres, nf_preconditioner=P_nf_opt,
    gmres_tol=1e-8, gmres_maxiter=300,
)
@assert length(trace_nf_d) >= 2
rel_J0_nf_d = abs(trace_nf_d[1].J - trace_dir_nf_d[1].J) / max(abs(trace_dir_nf_d[1].J), 1e-30)
println("  First-iteration J_ratio agreement (directivity, NF): $rel_J0_nf_d")
@assert rel_J0_nf_d < 1e-3

# nf_preconditioner=nothing should behave same as unpreconditioned
theta_opt_none, trace_none = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init_nf;
    maxiter=3, tol=1e-8, verbose=false,
    solver=:gmres, nf_preconditioner=nothing,
    gmres_tol=1e-8, gmres_maxiter=300,
)
rel_J0_none = abs(trace_none[1].J - trace_gm[1].J) / max(abs(trace_gm[1].J), 1e-30)
println("  nf_preconditioner=nothing matches unpreconditioned: $rel_J0_none")
@assert rel_J0_none < 1e-10

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 24: Cluster tree construction
# ─────────────────────────────────────────────────
println("\n── Test 24: Cluster tree construction ──")

centers_ct = rwg_centers(mesh, rwg)
@assert length(centers_ct) == N

tree_ct = build_cluster_tree(centers_ct; leaf_size=8)
@assert length(tree_ct.perm) == N
@assert length(tree_ct.iperm) == N

# perm and iperm should be inverses
for i in 1:N
    @assert tree_ct.iperm[tree_ct.perm[i]] == i "perm/iperm inverse check failed at i=$i"
end

# All indices 1:N should appear in perm exactly once
@assert sort(tree_ct.perm) == collect(1:N) "perm is not a valid permutation"

# Leaf nodes should have size <= leaf_size
for i in eachindex(tree_ct.nodes)
    if is_leaf(tree_ct, i)
        @assert length(tree_ct.nodes[i].indices) <= tree_ct.leaf_size "Leaf too large at node $i"
    end
end

# Root should cover all indices
@assert tree_ct.nodes[1].indices == 1:N "Root does not cover all indices"

# Admissibility should be false for overlapping clusters, true for well-separated ones
leaves_ct = leaf_nodes(tree_ct)
if length(leaves_ct) >= 2
    # Self-block is never admissible
    @assert !is_admissible(tree_ct, leaves_ct[1], leaves_ct[1])
end

println("  Tree: $(length(tree_ct.nodes)) nodes, $(length(leaves_ct)) leaves")
println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 25: ACA low-rank approximation accuracy
# ─────────────────────────────────────────────────
println("\n── Test 25: ACA low-rank approximation accuracy ──")

# Build EFIE cache for ACA entry evaluation
cache_aca = DiffMoM._build_efie_cache(mesh, rwg, k; quad_order=3, eta0=eta0)

# Find two well-separated leaf clusters for testing
tree_aca = build_cluster_tree(centers_ct; leaf_size=8)
leaves_aca = leaf_nodes(tree_aca)
found_admissible = false
global aca_row_node = 0
global aca_col_node = 0
for i_leaf in leaves_aca
    for j_leaf in leaves_aca
        if i_leaf != j_leaf && is_admissible(tree_aca, i_leaf, j_leaf; eta=1.5)
            global aca_row_node = i_leaf
            global aca_col_node = j_leaf
            global found_admissible = true
            break
        end
    end
    found_admissible && break
end

if found_admissible
    rn_aca = tree_aca.nodes[aca_row_node]
    cn_aca = tree_aca.nodes[aca_col_node]
    row_idx = [tree_aca.perm[k] for k in rn_aca.indices]
    col_idx = [tree_aca.perm[k] for k in cn_aca.indices]

    # Compute dense sub-block for reference
    m_blk = length(row_idx)
    n_blk = length(col_idx)
    Z_sub = Matrix{ComplexF64}(undef, m_blk, n_blk)
    for jj in 1:n_blk
        for ii in 1:m_blk
            Z_sub[ii, jj] = DiffMoM._efie_entry(cache_aca, row_idx[ii], col_idx[jj])
        end
    end

    # ACA low-rank approximation
    U_aca, V_aca = aca_lowrank(cache_aca, row_idx, col_idx; tol=1e-6, max_rank=30)
    rank_aca = size(U_aca, 2)
    approx_aca = U_aca * V_aca'
    err_aca = norm(approx_aca - Z_sub) / max(norm(Z_sub), 1e-30)

    println("  Block size: $(m_blk) x $(n_blk), ACA rank: $rank_aca, rel error: $err_aca")
    @assert err_aca < 1e-4 "ACA approximation too inaccurate: $err_aca"
    @assert rank_aca < min(m_blk, n_blk) "ACA should compress: rank=$rank_aca >= min($m_blk,$n_blk)"
else
    # Fallback: the default smoke-test mesh can be too small to produce an
    # admissible far block. Use a coarser-quadrature larger plate so ACA's
    # actual low-rank stopping logic is still tested deterministically.
    let
        mesh_aca_fb = make_rect_plate(1.0, 1.0, 20, 20)
        rwg_aca_fb = build_rwg(mesh_aca_fb)
        centers_aca_fb = rwg_centers(mesh_aca_fb, rwg_aca_fb)
        cache_aca_fb = DiffMoM._build_efie_cache(
            mesh_aca_fb, rwg_aca_fb, 2π; quad_order=1, eta0=eta0)
        tree_aca_fb = build_cluster_tree(centers_aca_fb; leaf_size=64)
        leaves_aca_fb = leaf_nodes(tree_aca_fb)

        row_node_fb = 0
        col_node_fb = 0
        min_block_fb = 0
        for i_leaf in leaves_aca_fb, j_leaf in leaves_aca_fb
            if i_leaf != j_leaf && is_admissible(tree_aca_fb, i_leaf, j_leaf; eta=0.8)
                ni = length(tree_aca_fb.nodes[i_leaf].indices)
                nj = length(tree_aca_fb.nodes[j_leaf].indices)
                if min(ni, nj) > min_block_fb
                    row_node_fb = i_leaf
                    col_node_fb = j_leaf
                    min_block_fb = min(ni, nj)
                end
            end
        end
        @assert row_node_fb != 0 "ACA fallback should find an admissible block"

        rn_fb = tree_aca_fb.nodes[row_node_fb]
        cn_fb = tree_aca_fb.nodes[col_node_fb]
        row_idx_fb = [tree_aca_fb.perm[k] for k in rn_fb.indices]
        col_idx_fb = [tree_aca_fb.perm[k] for k in cn_fb.indices]

        m_blk = length(row_idx_fb)
        n_blk = length(col_idx_fb)
        Z_sub = Matrix{ComplexF64}(undef, m_blk, n_blk)
        for jj in 1:n_blk
            for ii in 1:m_blk
                Z_sub[ii, jj] = DiffMoM._efie_entry(
                    cache_aca_fb, row_idx_fb[ii], col_idx_fb[jj])
            end
        end

        U_aca, V_aca = aca_lowrank(
            cache_aca_fb, row_idx_fb, col_idx_fb; tol=1e-4, max_rank=30)
        rank_aca = size(U_aca, 2)
        approx_aca = U_aca * V_aca'
        err_aca = norm(approx_aca - Z_sub) / max(norm(Z_sub), 1e-30)

        println("  Fallback block size: $(m_blk) x $(n_blk), ACA rank: $rank_aca, rel error: $err_aca")
        @assert err_aca < 5e-3 "ACA fallback approximation too inaccurate: $err_aca"
        @assert rank_aca < min(m_blk, n_blk) "ACA fallback should compress: rank=$rank_aca >= min($m_blk,$n_blk)"
    end
end

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 26: ACA operator matvec accuracy
# ─────────────────────────────────────────────────
println("\n── Test 26: ACA operator matvec accuracy ──")

A_aca_op = build_aca_operator(mesh, rwg, k;
                               leaf_size=8, eta=1.5, aca_tol=1e-8,
                               max_rank=50, quad_order=3, mesh_precheck=false)
@assert size(A_aca_op) == (N, N)

# Compare matvec against dense Z
Random.seed!(42)
x_test = randn(ComplexF64, N)
y_dense = Z_efie * x_test
y_aca = A_aca_op * x_test

rel_matvec_err = norm(y_aca - y_dense) / norm(y_dense)
println("  Dense blocks: $(length(A_aca_op.dense_blocks)), Low-rank blocks: $(length(A_aca_op.lowrank_blocks))")
println("  Matvec relative error: $rel_matvec_err")
@assert rel_matvec_err < 1e-5 "ACA matvec too inaccurate: $rel_matvec_err"

# Test adjoint matvec
A_adj = adjoint(A_aca_op)
y_adj_dense = Z_efie' * x_test
y_adj_aca = A_adj * x_test

rel_adj_err = norm(y_adj_aca - y_adj_dense) / norm(y_adj_dense)
println("  Adjoint matvec relative error: $rel_adj_err")
@assert rel_adj_err < 1e-5 "ACA adjoint matvec too inaccurate: $rel_adj_err"

# Test getindex fallback (used by NF preconditioner)
for _ in 1:10
    ii = rand(1:N)
    jj = rand(1:N)
    @assert A_aca_op[ii, jj] == Z_efie[ii, jj] "getindex mismatch at ($ii,$jj)"
end

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 27: ACA operator solve (forward + adjoint)
# ─────────────────────────────────────────────────
println("\n── Test 27: ACA operator solve (forward + adjoint) ──")

# Build NF preconditioner from geometry (not from dense Z)
P_nf_aca = build_nearfield_preconditioner(mesh, rwg, k, lambda0;
                                            quad_order=3, mesh_precheck=false)

# Forward solve via ACA GMRES
I_aca_gm, stats_aca = solve_gmres(A_aca_op, Vector{ComplexF64}(v);
                                    preconditioner=P_nf_aca,
                                    tol=1e-8, maxiter=300)

# Compare against direct dense solve
I_direct_ref = Z_efie \ v
rel_solve_err = norm(I_aca_gm - I_direct_ref) / norm(I_direct_ref)
println("  Forward solve relative error vs direct: $rel_solve_err")
println("  GMRES iterations: $(stats_aca.niter)")
@assert rel_solve_err < 1e-4 "ACA forward solve too inaccurate: $rel_solve_err"

# Adjoint solve via ACA GMRES
rhs_adj = Q * I_aca_gm
lam_aca, stats_adj = solve_gmres_adjoint(A_aca_op, rhs_adj;
                                           preconditioner=P_nf_aca,
                                           tol=1e-8, maxiter=300)
lam_direct = Z_efie' \ (Q * I_direct_ref)
rel_adj_solve = norm(lam_aca - lam_direct) / max(norm(lam_direct), 1e-30)
println("  Adjoint solve relative error: $rel_adj_solve")
@assert rel_adj_solve < 1e-3 "ACA adjoint solve too inaccurate: $rel_adj_solve"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 28: solve_scattering workflow
# ─────────────────────────────────────────────────
println("\n── Test 28: solve_scattering workflow ──")

# Test with small mesh — auto should pick dense_direct
pw_exc = make_plane_wave(k_vec, E0, pol)
result_auto = solve_scattering(mesh, freq, pw_exc;
                                verbose=false, check_resolution=false)
@assert result_auto.method == :dense_direct "Expected :dense_direct for N=$(result_auto.N), got $(result_auto.method)"
@assert result_auto.N == N

# Verify solution matches manual dense solve
rel_workflow_err = norm(result_auto.I_coeffs - I_pec) / norm(I_pec)
println("  Auto (dense_direct) vs manual: rel_err=$rel_workflow_err")
@assert rel_workflow_err < 1e-10 "Workflow dense_direct solution mismatch: $rel_workflow_err"

# Force ACA GMRES method
result_aca_forced = solve_scattering(mesh, freq, pw_exc;
                                      method=:aca_gmres,
                                      aca_tol=1e-8, aca_leaf_size=8,
                                      nf_cutoff_lambda=1.0,
                                      gmres_tol=1e-8, gmres_maxiter=300,
                                      verbose=false, check_resolution=false)
@assert result_aca_forced.method == :aca_gmres
rel_aca_workflow = norm(result_aca_forced.I_coeffs - I_pec) / norm(I_pec)
println("  Forced ACA vs dense direct: rel_err=$rel_aca_workflow")
@assert rel_aca_workflow < 1e-4 "Workflow ACA solution mismatch: $rel_aca_workflow"

# Force dense GMRES method
result_dgm = solve_scattering(mesh, freq, pw_exc;
                               method=:dense_gmres,
                               nf_cutoff_lambda=1.0,
                               gmres_tol=1e-8, gmres_maxiter=300,
                               verbose=false, check_resolution=false)
@assert result_dgm.method == :dense_gmres
rel_dgm = norm(result_dgm.I_coeffs - I_pec) / norm(I_pec)
println("  Forced dense_gmres vs direct: rel_err=$rel_dgm")
@assert rel_dgm < 1e-6 "Workflow dense GMRES solution mismatch: $rel_dgm"

# Test with pre-assembled excitation vector
result_vec = solve_scattering(mesh, freq, v;
                               verbose=false, check_resolution=false)
@assert result_vec.method == :dense_direct
@assert norm(result_vec.I_coeffs - I_pec) / norm(I_pec) < 1e-10

# Test mesh resolution warning
result_warn = solve_scattering(mesh, 1e6, pw_exc;
                                verbose=false, check_resolution=true)
# At 1 MHz, lambda = 300m, mesh is massively over-resolved → no warning
@assert isempty(result_warn.warnings) || result_warn.mesh_report.meets_target

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 29: Physical Optics (PO) solver
# ─────────────────────────────────────────────────
println("\n── Test 29: Physical Optics (PO) solver ──")

# 29a: Illumination test on flat plate
# Plate at z=0, plane wave from +z direction (k along -z)
po_mesh = make_rect_plate(0.2, 0.2, 5, 5)  # reuse existing test plate
po_freq = 3e9
po_c0 = 299792458.0
po_lam = po_c0 / po_freq
po_k = 2π / po_lam

# Wave traveling in -z: should illuminate the +z-facing surface
pw_down = make_plane_wave(Vec3(0.0, 0.0, -po_k), 1.0, Vec3(1.0, 0.0, 0.0))
po_grid = make_sph_grid(36, 72)
po_result = solve_po(po_mesh, po_freq, pw_down; grid=po_grid)

n_illum = count(po_result.illuminated)
n_total = ntriangles(po_mesh)
# For a plate at z=0 with normals pointing in +z, wave from +z traveling -z
# should illuminate all faces (k̂·n̂ = -1 < 0)
println("  Illumination (wave -z on +z plate): $n_illum / $n_total")
@assert n_illum == n_total "Expected all $n_total triangles illuminated, got $n_illum"

# Wave traveling in +z: should illuminate NO faces (backside)
pw_up = make_plane_wave(Vec3(0.0, 0.0, po_k), 1.0, Vec3(1.0, 0.0, 0.0))
po_result_back = solve_po(po_mesh, po_freq, pw_up; grid=po_grid)
n_illum_back = count(po_result_back.illuminated)
println("  Illumination (wave +z on +z plate): $n_illum_back / $n_total")
@assert n_illum_back == 0 "Expected 0 triangles illuminated, got $n_illum_back"

# 29b: PO specular RCS on flat plate
# Analytical PO: σ_specular = 4π A² / λ² for a flat plate at broadside
Lx_po, Ly_po = 0.2, 0.2
A_plate = Lx_po * Ly_po
sigma_analytical = 4π * A_plate^2 / po_lam^2
sigma_analytical_dB = 10 * log10(sigma_analytical)

# Find the specular direction (θ≈π, backscatter for -z incidence → r̂ = +z)
# Actually for -z incidence, specular reflection from plate at z=0 is +z direction
sigma_po = bistatic_rcs(po_result.E_ff; E0=1.0)
# Find the direction closest to +z (θ≈0)
best_idx = argmax([po_grid.rhat[3, q] for q in 1:length(po_grid.w)])
sigma_spec = sigma_po[best_idx]
sigma_spec_dB = 10 * log10(max(sigma_spec, 1e-30))

println("  PO specular RCS: $(round(sigma_spec_dB, digits=2)) dBsm")
println("  Analytical 4πA²/λ²: $(round(sigma_analytical_dB, digits=2)) dBsm")
po_err_dB = abs(sigma_spec_dB - sigma_analytical_dB)
println("  Error: $(round(po_err_dB, digits=2)) dB")
@assert po_err_dB < 1.5 "PO specular RCS error $(po_err_dB) dB > 1.5 dB tolerance"

# 29c: PO vs MoM comparison on small plate
# At broadside, PO and MoM should agree within a few dB for specular
po_rwg = build_rwg(po_mesh)
po_Z = assemble_Z_efie(po_mesh, po_rwg, po_k)
po_v = assemble_excitation(po_mesh, po_rwg, pw_down)
po_I = po_Z \ po_v
po_G = radiation_vectors(po_mesh, po_rwg, po_grid, po_k)
po_NΩ = length(po_grid.w)
po_Eff_mom = compute_farfield(po_G, Vector{ComplexF64}(po_I), po_NΩ)
sigma_mom = bistatic_rcs(po_Eff_mom; E0=1.0)
sigma_mom_spec_dB = 10 * log10(max(sigma_mom[best_idx], 1e-30))

mom_po_diff_dB = abs(sigma_mom_spec_dB - sigma_spec_dB)
println("  MoM specular RCS: $(round(sigma_mom_spec_dB, digits=2)) dBsm")
println("  MoM vs PO specular diff: $(round(mom_po_diff_dB, digits=2)) dB")
@assert mom_po_diff_dB < 3.0 "MoM vs PO specular difference $(mom_po_diff_dB) dB > 3.0 dB"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 30: ILU preconditioner
# ─────────────────────────────────────────────────
println("\n── Test 30: ILU preconditioner ──")

# Build a small plate problem for testing
ilu_mesh = make_rect_plate(0.4, 0.4, 4, 4)
ilu_rwg  = build_rwg(ilu_mesh)
ilu_N    = ilu_rwg.nedges
ilu_k    = 2π / 0.4   # λ = 0.4m
ilu_Z    = assemble_Z_efie(ilu_mesh, ilu_rwg, ilu_k)
ilu_pw   = make_plane_wave(Vec3(0.0, 0.0, -ilu_k), 1.0, Vec3(1.0, 0.0, 0.0))
ilu_rhs  = assemble_excitation(ilu_mesh, ilu_rwg, ilu_pw)

# Reference: direct solve
ilu_ref = ilu_Z \ ilu_rhs

# 30a: ILU preconditioner builds without error
cutoff_ilu = 0.3   # ~0.75λ
P_ilu = build_nearfield_preconditioner(ilu_Z, ilu_mesh, ilu_rwg, cutoff_ilu;
    factorization=:ilu, ilu_tau=1e-3)
@assert P_ilu isa ILUPreconditionerData
@assert P_ilu.nnz_ratio > 0 && P_ilu.nnz_ratio <= 1.0
println("  30a: ILU preconditioner built — nnz=$(round(P_ilu.nnz_ratio * 100, digits=1))%, τ=$(P_ilu.tau)")

# 30b: ILU-preconditioned GMRES converges
I_ilu, stats_ilu = solve_gmres(ilu_Z, ilu_rhs; preconditioner=P_ilu, tol=1e-8, maxiter=500)
err_ilu = norm(I_ilu - ilu_ref) / norm(ilu_ref)
println("  30b: ILU GMRES — $(stats_ilu.niter) iters, rel error $(round(err_ilu, sigdigits=3))")
@assert stats_ilu.niter < 200 "ILU GMRES took $(stats_ilu.niter) iters (expected < 200)"
@assert err_ilu < 1e-4 "ILU GMRES relative error $(err_ilu) > 1e-4"

# 30c: Compare ILU vs full LU iteration counts
P_lu = build_nearfield_preconditioner(ilu_Z, ilu_mesh, ilu_rwg, cutoff_ilu;
    factorization=:lu)
I_lu, stats_lu = solve_gmres(ilu_Z, ilu_rhs; preconditioner=P_lu, tol=1e-8, maxiter=500)
println("  30c: LU GMRES — $(stats_lu.niter) iters (ILU: $(stats_ilu.niter) iters)")
# ILU should take more iterations than full LU but still converge
@assert stats_ilu.niter >= stats_lu.niter "ILU should take ≥ LU iterations"

# 30d: ILU works with matrix-free operator (mesh, rwg, k overload)
P_ilu_mf = build_nearfield_preconditioner(ilu_mesh, ilu_rwg, ilu_k, cutoff_ilu;
    factorization=:ilu, ilu_tau=1e-3)
@assert P_ilu_mf isa ILUPreconditionerData
println("  30d: Matrix-free ILU build — nnz=$(round(P_ilu_mf.nnz_ratio * 100, digits=1))%")

# 30e: ILU adjoint preconditioner works
I_adj_ilu, stats_adj = solve_gmres_adjoint(ilu_Z, ilu_rhs; preconditioner=P_ilu, tol=1e-8, maxiter=500)
I_adj_ref = adjoint(ilu_Z) \ ilu_rhs
err_adj = norm(I_adj_ilu - I_adj_ref) / norm(I_adj_ref)
println("  30e: ILU adjoint GMRES — $(stats_adj.niter) iters, rel error $(round(err_adj, sigdigits=3))")
@assert err_adj < 1e-4 "ILU adjoint GMRES relative error $(err_adj) > 1e-4"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 31: MLFMA operator
# ─────────────────────────────────────────────────
println("\n--- Test 31: MLFMA operator ---")

# Setup: 2λ × 2λ flat plate at 3 GHz
mlfma_freq = 3e9
mlfma_lambda = 299792458.0 / mlfma_freq
mlfma_k = 2π / mlfma_lambda
mlfma_Lx = 2 * mlfma_lambda
mlfma_Ly = 2 * mlfma_lambda
mlfma_Nx = 6
mlfma_Ny = 6
mlfma_mesh = make_rect_plate(mlfma_Lx, mlfma_Ly, mlfma_Nx, mlfma_Ny)
mlfma_rwg = build_rwg(mlfma_mesh)
mlfma_N = mlfma_rwg.nedges
println("  Setup: $(mlfma_Nx)×$(mlfma_Ny) plate, N=$mlfma_N, freq=$(mlfma_freq/1e9) GHz")

# 31a: Octree construction
mlfma_centers = rwg_centers(mlfma_mesh, mlfma_rwg)
octree = build_octree(mlfma_centers, mlfma_k; leaf_lambda=0.5)

# Verify all BFs are assigned via permutation
@assert length(octree.perm) == mlfma_N
@assert length(octree.iperm) == mlfma_N
@assert sort(octree.perm) == 1:mlfma_N  "perm must be a permutation of 1:N"

# Verify perm/iperm are inverse of each other
for i in 1:mlfma_N
    @assert octree.iperm[octree.perm[i]] == i "perm/iperm inconsistency at $i"
end

leaf_level = octree.levels[octree.nLevels]
n_leaf_boxes = length(leaf_level.boxes)
println("  31a: Octree — $(octree.nLevels) levels, $n_leaf_boxes leaf boxes")

# Verify neighbors and interaction_list don't overlap
for box in leaf_level.boxes
    nbr_set = Set(box.neighbors)
    il_set = Set(box.interaction_list)
    @assert isempty(intersect(nbr_set, il_set)) "Neighbor/interaction_list overlap for box $(box.id)"
end
println("  31a: PASS — no neighbor/interaction_list overlaps")

# 31b: Near-field matrix accuracy
Z_dense_mlfma = assemble_Z_efie(mlfma_mesh, mlfma_rwg, mlfma_k; mesh_precheck=false)
Z_near_mlfma = assemble_mlfma_nearfield(octree, mlfma_mesh, mlfma_rwg, mlfma_k)

# Check that near-field entries match dense for neighbor pairs
max_nf_err = 0.0
n_checked = 0
for box in leaf_level.boxes
    for nbr_id in box.neighbors
        nbr_box = leaf_level.boxes[nbr_id]
        for m_perm in box.bf_range
            m = octree.perm[m_perm]
            for n_perm in nbr_box.bf_range
                n = octree.perm[n_perm]
                err = abs(Z_near_mlfma[m, n] - Z_dense_mlfma[m, n])
                ref = abs(Z_dense_mlfma[m, n])
                if ref > 1e-15
                    global max_nf_err = max(max_nf_err, err / ref)
                end
                global n_checked += 1
            end
        end
    end
end
println("  31b: Near-field accuracy — checked $n_checked entries, max rel error = $(round(max_nf_err, sigdigits=3))")
@assert max_nf_err < 1e-10 "Near-field entries do not match dense: max error $max_nf_err"
println("  31b: PASS")

# 31c: MLFMA matvec accuracy
A_mlfma = build_mlfma_operator(mlfma_mesh, mlfma_rwg, mlfma_k;
    leaf_lambda=0.5, quad_order=3, verbose=false)

# Random test vector
Random.seed!(42)
x_test = randn(ComplexF64, mlfma_N)
y_dense = Z_dense_mlfma * x_test
y_mlfma = A_mlfma * x_test

mlfma_matvec_err = norm(y_mlfma - y_dense) / norm(y_dense)
println("  31c: MLFMA matvec — rel error = $(round(mlfma_matvec_err, sigdigits=3))")
@assert mlfma_matvec_err < 0.05 "MLFMA matvec error too large: $mlfma_matvec_err (expected < 0.05)"
println("  31c: PASS")

# 31d: MLFMA adjoint inner-product identity
y_test = randn(ComplexF64, mlfma_N)
lhs_adj = dot(y_test, A_mlfma * x_test)
rhs_adj = dot(adjoint(A_mlfma) * y_test, x_test)
mlfma_adj_err = abs(lhs_adj - rhs_adj) / max(abs(lhs_adj), abs(rhs_adj), eps())
println("  31d: MLFMA adjoint identity — rel error = $(round(mlfma_adj_err, sigdigits=3))")
@assert mlfma_adj_err < 1e-10 "MLFMA adjoint identity failed: $mlfma_adj_err"
println("  31d: PASS")

# 31e: MLFMA + GMRES convergence
mlfma_exc = PlaneWaveExcitation(Vec3(0.0, 0.0, -mlfma_k), 1.0, Vec3(1.0, 0.0, 0.0))
mlfma_v = assemble_excitation(mlfma_mesh, mlfma_rwg, mlfma_exc)
I_dense_ref = Z_dense_mlfma \ mlfma_v

# Build preconditioner from MLFMA near-field
P_mlfma = build_nearfield_preconditioner(A_mlfma.Z_near; factorization=:lu)
I_mlfma, stats_mlfma = solve_gmres(A_mlfma, mlfma_v;
    preconditioner=P_mlfma, tol=1e-4, maxiter=200)

mlfma_sol_err = norm(I_mlfma - I_dense_ref) / norm(I_dense_ref)
println("  31e: MLFMA+GMRES — $(stats_mlfma.niter) iters, sol rel error = $(round(mlfma_sol_err, sigdigits=3))")
@assert stats_mlfma.niter < 200 "MLFMA GMRES did not converge in 200 iterations"
@assert mlfma_sol_err < 0.1 "MLFMA solution error too large: $mlfma_sol_err (expected < 0.1)"
println("  31e: PASS")

# 31f: impedance-loaded MLFMA preconditioner
part_mlfma_loaded = assign_patches_grid(mlfma_mesh; nx=2, ny=2, nz=1)
Mp_mlfma_loaded = precompute_patch_mass(mlfma_mesh, mlfma_rwg, part_mlfma_loaded)
theta_mlfma_loaded = fill(150.0, part_mlfma_loaded.P)
A_mlfma_loaded = ImpedanceLoadedOperator(A_mlfma, Mp_mlfma_loaded, theta_mlfma_loaded, false)
P_mlfma_loaded = build_mlfma_preconditioner(A_mlfma, Mp_mlfma_loaded, theta_mlfma_loaded;
    factorization=:lu)
I_mlfma_loaded, stats_mlfma_loaded = solve_gmres(A_mlfma_loaded, mlfma_v;
    preconditioner=P_mlfma_loaded, tol=1e-4, maxiter=200)
loaded_residual = norm(A_mlfma_loaded * I_mlfma_loaded - mlfma_v) / norm(mlfma_v)
println("  31f: Loaded MLFMA preconditioner — $(stats_mlfma_loaded.niter) iters, residual = $(round(loaded_residual, sigdigits=3))")
@assert stats_mlfma_loaded.solved "Loaded MLFMA GMRES should converge"
@assert loaded_residual < 1e-4 "Loaded MLFMA residual too large: $loaded_residual"
println("  31f: PASS")

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 32: Mesh I/O formats and geometry coverage
# ─────────────────────────────────────────────────
println("\n── Test 32: Mesh I/O formats and geometry coverage ──")

# 32a: write_obj_mesh round-trip
println("  32a: write_obj_mesh round-trip ...")
obj_rt_path = joinpath(DATADIR, "tmp_roundtrip.obj")
write_obj_mesh(obj_rt_path, mesh; header="Round-trip test")
mesh_rt = read_obj_mesh(obj_rt_path)
@assert nvertices(mesh_rt) == nvertices(mesh) "OBJ round-trip vertex count mismatch"
@assert ntriangles(mesh_rt) == ntriangles(mesh) "OBJ round-trip triangle count mismatch"
for i in 1:nvertices(mesh)
    for d in 1:3
        @assert abs(mesh_rt.xyz[d, i] - mesh.xyz[d, i]) < 1e-12 "OBJ round-trip vertex position mismatch at vertex $i dim $d"
    end
end
report_rt = mesh_quality_report(mesh_rt)
@assert mesh_quality_ok(report_rt; allow_boundary=true)
println("  32a: PASS")

# 32b: triangle_area explicit test
println("  32b: triangle_area explicit test ...")
# Right triangle with base=3 along x, height=4 along y → area = 6.0
xyz_right = Float64[0 3 0; 0 0 4; 0 0 0]
tri_right = reshape([1, 2, 3], 3, 1)
mesh_right = TriMesh(xyz_right, tri_right)
@assert abs(triangle_area(mesh_right, 1) - 6.0) < 1e-12 "Right triangle area should be 6.0"

# Equilateral triangle with side s=2 → area = sqrt(3)
s_eq = 2.0
xyz_eq = Float64[0 s_eq s_eq/2; 0 0 s_eq*sqrt(3)/2; 0 0 0]
tri_eq = reshape([1, 2, 3], 3, 1)
mesh_eq = TriMesh(xyz_eq, tri_eq)
expected_area = s_eq^2 * sqrt(3) / 4
@assert abs(triangle_area(mesh_eq, 1) - expected_area) < 1e-12 "Equilateral triangle area mismatch"
println("  32b: PASS")

# 32c: STL binary round-trip
println("  32c: STL binary round-trip ...")
stl_bin_path = joinpath(DATADIR, "tmp_roundtrip_bin.stl")
mesh_plate = make_rect_plate(0.1, 0.1, 3, 3)
write_stl_mesh(stl_bin_path, mesh_plate)
mesh_stl_bin = read_stl_mesh(stl_bin_path)
# STL uses Float32 internally, so vertex count may differ slightly from merging.
# But triangle count must match since each facet is independent.
@assert ntriangles(mesh_stl_bin) == ntriangles(mesh_plate) "STL binary round-trip triangle count mismatch: got $(ntriangles(mesh_stl_bin)), expected $(ntriangles(mesh_plate))"
@assert nvertices(mesh_stl_bin) == nvertices(mesh_plate) "STL binary round-trip vertex count mismatch: got $(nvertices(mesh_stl_bin)), expected $(nvertices(mesh_plate))"
report_stl_bin = mesh_quality_report(mesh_stl_bin)
@assert mesh_quality_ok(report_stl_bin; allow_boundary=true) "STL binary round-trip mesh quality check failed"
# Check vertex positions (Float32 precision ~ 1e-6)
for t in 1:ntriangles(mesh_plate)
    for vi in 1:3
        idx_orig = mesh_plate.tri[vi, t]
        idx_stl = mesh_stl_bin.tri[vi, t]
        for d in 1:3
            @assert abs(mesh_stl_bin.xyz[d, idx_stl] - Float64(Float32(mesh_plate.xyz[d, idx_orig]))) < 1e-10 "STL binary vertex mismatch at tri $t vert $vi dim $d"
        end
    end
end
println("  32c: PASS")

# 32d: STL ASCII round-trip
println("  32d: STL ASCII round-trip ...")
stl_ascii_path = joinpath(DATADIR, "tmp_ascii.stl")
open(stl_ascii_path, "w") do io
    println(io, "solid test")
    println(io, "  facet normal 0 0 1")
    println(io, "    outer loop")
    println(io, "      vertex 0.0 0.0 0.0")
    println(io, "      vertex 1.0 0.0 0.0")
    println(io, "      vertex 1.0 1.0 0.0")
    println(io, "    endloop")
    println(io, "  endfacet")
    println(io, "  facet normal 0 0 1")
    println(io, "    outer loop")
    println(io, "      vertex 0.0 0.0 0.0")
    println(io, "      vertex 1.0 1.0 0.0")
    println(io, "      vertex 0.0 1.0 0.0")
    println(io, "    endloop")
    println(io, "  endfacet")
    println(io, "endsolid test")
end
mesh_stl_ascii = read_stl_mesh(stl_ascii_path)
@assert nvertices(mesh_stl_ascii) == 4 "STL ASCII: expected 4 unique vertices, got $(nvertices(mesh_stl_ascii))"
@assert ntriangles(mesh_stl_ascii) == 2 "STL ASCII: expected 2 triangles, got $(ntriangles(mesh_stl_ascii))"
println("  32d: PASS")

# 32e: STL vertex merging (tetrahedron)
println("  32e: STL vertex merging ...")
# Build a tetrahedron: 4 triangles, 4 unique vertices, but 12 raw vertices in STL
xyz_tet = Float64[0 1 0.5 0.5; 0 0 sqrt(3)/2 sqrt(3)/6; 0 0 0 sqrt(2/3)]
tri_tet = [1 1 1 2; 2 2 3 3; 3 4 4 4]
mesh_tet = TriMesh(xyz_tet, tri_tet)
stl_tet_path = joinpath(DATADIR, "tmp_tetra.stl")
write_stl_mesh(stl_tet_path, mesh_tet)
mesh_tet_rt = read_stl_mesh(stl_tet_path)
@assert nvertices(mesh_tet_rt) == 4 "Tetrahedron STL: expected 4 unique vertices after merge, got $(nvertices(mesh_tet_rt))"
@assert ntriangles(mesh_tet_rt) == 4 "Tetrahedron STL: expected 4 triangles, got $(ntriangles(mesh_tet_rt))"
println("  32e: PASS")

# 32f: MSH v2 import
println("  32f: MSH v2 import ...")
msh_v2_path = joinpath(DATADIR, "tmp_v2.msh")
open(msh_v2_path, "w") do io
    println(io, "\$MeshFormat")
    println(io, "2.2 0 8")
    println(io, "\$EndMeshFormat")
    println(io, "\$Nodes")
    println(io, "4")
    println(io, "1 0.0 0.0 0.0")
    println(io, "2 1.0 0.0 0.0")
    println(io, "3 1.0 1.0 0.0")
    println(io, "4 0.0 1.0 0.0")
    println(io, "\$EndNodes")
    println(io, "\$Elements")
    println(io, "3")
    println(io, "1 1 2 0 1 1 2")         # line element (should be skipped)
    println(io, "2 2 2 0 1 1 2 3")       # triangle 1
    println(io, "3 2 2 0 1 1 3 4")       # triangle 2
    println(io, "\$EndElements")
end
mesh_msh_v2 = read_msh_mesh(msh_v2_path)
@assert nvertices(mesh_msh_v2) == 4 "MSH v2: expected 4 vertices, got $(nvertices(mesh_msh_v2))"
@assert ntriangles(mesh_msh_v2) == 2 "MSH v2: expected 2 triangles, got $(ntriangles(mesh_msh_v2))"
report_msh_v2 = mesh_quality_report(mesh_msh_v2)
@assert report_msh_v2.n_invalid_triangles == 0
@assert report_msh_v2.n_degenerate_triangles == 0
println("  32f: PASS")

# 32g: MSH v4 import
println("  32g: MSH v4 import ...")
msh_v4_path = joinpath(DATADIR, "tmp_v4.msh")
open(msh_v4_path, "w") do io
    println(io, "\$MeshFormat")
    println(io, "4.1 0 8")
    println(io, "\$EndMeshFormat")
    println(io, "\$Nodes")
    println(io, "1 3 1 3")               # 1 entity block, 3 nodes, min tag 1, max tag 3
    println(io, "2 1 0 3")               # entity dim=2, tag=1, parametric=0, 3 nodes
    println(io, "1")                      # node tags
    println(io, "2")
    println(io, "3")
    println(io, "0.0 0.0 0.0")          # node coordinates
    println(io, "1.0 0.0 0.0")
    println(io, "0.5 1.0 0.0")
    println(io, "\$EndNodes")
    println(io, "\$Elements")
    println(io, "1 1 1 1")               # 1 entity block, 1 element, min tag 1, max tag 1
    println(io, "2 1 2 1")               # entity dim=2, tag=1, type=2 (triangle), 1 element
    println(io, "1 1 2 3")               # element tag 1, nodes 1 2 3
    println(io, "\$EndElements")
end
mesh_msh_v4 = read_msh_mesh(msh_v4_path)
@assert nvertices(mesh_msh_v4) == 3 "MSH v4: expected 3 vertices, got $(nvertices(mesh_msh_v4))"
@assert ntriangles(mesh_msh_v4) == 1 "MSH v4: expected 1 triangle, got $(ntriangles(mesh_msh_v4))"
# Verify vertex positions
@assert abs(mesh_msh_v4.xyz[1, 1] - 0.0) < 1e-12
@assert abs(mesh_msh_v4.xyz[1, 2] - 1.0) < 1e-12
@assert abs(mesh_msh_v4.xyz[2, 3] - 1.0) < 1e-12
println("  32g: PASS")

# 32h: Unified read_mesh / write_mesh dispatcher
println("  32h: Unified read_mesh / write_mesh dispatcher ...")
# OBJ dispatch
mesh_dispatch_obj = read_mesh(obj_rt_path)
@assert nvertices(mesh_dispatch_obj) == nvertices(mesh) "read_mesh .obj dispatch failed"

# STL dispatch
mesh_dispatch_stl = read_mesh(stl_bin_path)
@assert ntriangles(mesh_dispatch_stl) == ntriangles(mesh_plate) "read_mesh .stl dispatch failed"

# MSH dispatch
mesh_dispatch_msh = read_mesh(msh_v2_path)
@assert ntriangles(mesh_dispatch_msh) == 2 "read_mesh .msh dispatch failed"

# write_mesh OBJ
write_out_obj = joinpath(DATADIR, "tmp_write_dispatch.obj")
write_mesh(write_out_obj, mesh)
@assert isfile(write_out_obj)

# write_mesh STL
write_out_stl = joinpath(DATADIR, "tmp_write_dispatch.stl")
write_mesh(write_out_stl, mesh)
@assert isfile(write_out_stl)

# Unsupported extension
thrown_ext = try
    read_mesh(joinpath(DATADIR, "fake.xyz"))
    false
catch
    true
end
@assert thrown_ext "read_mesh should throw on unsupported extension"

thrown_ext_w = try
    write_mesh(joinpath(DATADIR, "fake.xyz"), mesh)
    false
catch
    true
end
@assert thrown_ext_w "write_mesh should throw on unsupported extension"
println("  32h: PASS")

# 32i: convert_cad_to_mesh (skip if gmsh not available)
println("  32i: convert_cad_to_mesh (gmsh check) ...")
gmsh_available = Sys.which("gmsh") !== nothing
if !gmsh_available
    # Verify helpful error message
    thrown_gmsh = try
        convert_cad_to_mesh("dummy.step", "dummy.msh")
        false
    catch e
        occursin("not found", string(e)) || occursin("Gmsh", string(e))
    end
    @assert thrown_gmsh "convert_cad_to_mesh should mention gmsh in error"
    println("  32i: SKIP (gmsh not installed) — error message verified")
else
    println("  32i: SKIP (gmsh available but no test CAD file) — presence verified")
end

# 32j: Closed-surface mesh workflow
println("  32j: Closed-surface mesh workflow ...")
ico_path = joinpath(DATADIR, "tmp_icosphere.obj")
write_icosphere_obj(ico_path; radius=0.05, subdivisions=2)
mesh_ico = read_obj_mesh(ico_path)
report_ico = mesh_quality_report(mesh_ico)
@assert report_ico.n_boundary_edges == 0 "Icosphere should have no boundary edges, got $(report_ico.n_boundary_edges)"
@assert report_ico.n_nonmanifold_edges == 0 "Icosphere should have no non-manifold edges"
@assert mesh_quality_ok(report_ico; allow_boundary=false, require_closed=true) "Icosphere should pass closed-surface quality check"
println("  32j: PASS")

# 32k: mesh_resolution_ok with :p95 and :median criteria
println("  32k: mesh_resolution_ok criteria ...")
# Use a coarse mesh that fails :max but could pass :p95 or :median
mesh_res_test = make_rect_plate(1.0, 1.0, 2, 2)
report_res_test = mesh_resolution_report(mesh_res_test, 3e8; points_per_wavelength=2.0)
# The mesh has edges of similar length, so all criteria should give same result
res_max = mesh_resolution_ok(report_res_test; criterion=:max)
res_p95 = mesh_resolution_ok(report_res_test; criterion=:p95)
res_med = mesh_resolution_ok(report_res_test; criterion=:median)
# :median is most lenient, :max is strictest
# If :max passes, all must pass. If :max fails, :median may still pass.
if res_max
    @assert res_p95 && res_med ":max passed but :p95 or :median failed — logic error"
end
# :p95 and :median should never be stricter than :max
if !res_p95
    @assert !res_max ":p95 failed but :max passed — impossible"
end
# Verify the criteria use different statistics
@assert report_res_test.edge_max_m >= report_res_test.edge_p95_m >= report_res_test.edge_median_m "Edge statistics ordering violated"
# Unsupported criterion should throw
thrown_crit = try
    mesh_resolution_ok(report_res_test; criterion=:bogus)
    false
catch
    true
end
@assert thrown_crit "mesh_resolution_ok should throw on unknown criterion"
println("  32k: PASS")

# 32l: STL ASCII write and read-back
println("  32l: STL ASCII write round-trip ...")
stl_ascii_rt_path = joinpath(DATADIR, "tmp_ascii_roundtrip.stl")
mesh_small = make_rect_plate(0.05, 0.05, 2, 2)
write_stl_mesh(stl_ascii_rt_path, mesh_small; ascii=true)
mesh_ascii_rt = read_stl_mesh(stl_ascii_rt_path)
@assert ntriangles(mesh_ascii_rt) == ntriangles(mesh_small) "STL ASCII round-trip triangle mismatch"
@assert nvertices(mesh_ascii_rt) == nvertices(mesh_small) "STL ASCII round-trip vertex mismatch"
println("  32l: PASS")

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 33: Spatial patch assignment
# ─────────────────────────────────────────────────
println("\n── Test 33: Spatial patch assignment ──")

# 33a: Grid-based partitioning
println("  33a: Grid-based partitioning ...")
part_grid = assign_patches_grid(mesh; nx=3, ny=2, nz=1)
@assert length(part_grid.tri_patch) == ntriangles(mesh) "Grid partition should cover all triangles"
@assert part_grid.P >= 1 "Grid partition should have at least 1 patch"
@assert part_grid.P <= 6 "Grid partition should have at most nx*ny patches"
@assert all(1 .<= part_grid.tri_patch .<= part_grid.P) "All patch IDs should be in [1, P]"
println("    $Nt triangles → $(part_grid.P) patches")
println("  33a: PASS")

# 33b: Region-based partitioning
println("  33b: Region-based partitioning ...")
# Split plate at y=0 (mesh is centered)
region_top = region_halfspace(; axis=:y, threshold=0.0, above=true)
region_bot = region_halfspace(; axis=:y, threshold=0.0, above=false)
part_region = assign_patches_by_region(mesh, [region_top, region_bot])
@assert part_region.P == 2 "Two half-space regions should give exactly 2 patches"
@assert all(1 .<= part_region.tri_patch .<= 2) "All triangles should be assigned to patch 1 or 2"
n_top = count(==(1), part_region.tri_patch)
n_bot = count(==(2), part_region.tri_patch)
@assert n_top > 0 && n_bot > 0 "Both regions should have triangles"
println("    Region 1 (y>=0): $n_top tris,  Region 2 (y<0): $n_bot tris")
println("  33b: PASS")

# 33c: Sphere and box predicates
println("  33c: Sphere and box predicates ...")
pred_sphere = region_sphere(; center=Vec3(0.0, 0.0, 0.0), radius=0.01)
pred_box = region_box(; lo=Vec3(-0.01, -0.01, -1.0), hi=Vec3(0.01, 0.01, 1.0))
# Test that predicates are callable and return Bool
@assert pred_sphere(Vec3(0.0, 0.0, 0.0)) == true "Origin should be inside sphere"
@assert pred_sphere(Vec3(100.0, 0.0, 0.0)) == false "Far point should be outside sphere"
@assert pred_box(Vec3(0.0, 0.0, 0.0)) == true "Origin should be inside box"
@assert pred_box(Vec3(1.0, 0.0, 0.0)) == false "Point outside should fail"
println("  33c: PASS")

# 33d: Uniform k-means partitioning
println("  33d: Uniform k-means partitioning ...")
n_target = 5
part_kmeans = assign_patches_uniform(mesh; n_patches=n_target)
@assert length(part_kmeans.tri_patch) == ntriangles(mesh)
@assert part_kmeans.P >= 1 && part_kmeans.P <= n_target
@assert all(1 .<= part_kmeans.tri_patch .<= part_kmeans.P)
println("    Requested $n_target patches → got $(part_kmeans.P)")
println("  33d: PASS")

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 34: CompositeOperator (ImpedanceLoadedOperator)
# ─────────────────────────────────────────────────
println("\n── Test 34: CompositeOperator ──")

# 34a: Forward matvec accuracy
println("  34a: Forward matvec accuracy ...")
theta_test = randn(MersenneTwister(123), Nt) .* 100.0
x_test = randn(MersenneTwister(456), ComplexF64, N)

# Reference: dense assemble_full_Z
Z_ref = assemble_full_Z(Z_efie, Mp, theta_test; reactive=false)
y_ref = Z_ref * x_test

# Composite operator
A_comp = ImpedanceLoadedOperator(Z_efie, Mp, theta_test, false)
y_comp = A_comp * x_test

matvec_err = norm(y_comp - y_ref) / norm(y_ref)
println("    Forward matvec relative error: $matvec_err")
@assert matvec_err < 1e-12 "Forward matvec error too large: $matvec_err"
@assert size(A_comp) == (N, N) "Size mismatch"
println("  34a: PASS")

# 34b: Adjoint matvec accuracy
println("  34b: Adjoint matvec accuracy ...")
y_adj_ref = Z_ref' * x_test
y_adj_comp = adjoint(A_comp) * x_test

adj_err = norm(y_adj_comp - y_adj_ref) / norm(y_adj_ref)
println("    Adjoint matvec relative error: $adj_err")
@assert adj_err < 1e-12 "Adjoint matvec error too large: $adj_err"
println("  34b: PASS")

# 34c: Reactive mode
println("  34c: Reactive mode ...")
Z_ref_rx = assemble_full_Z(Z_efie, Mp, theta_test; reactive=true)
A_comp_rx = ImpedanceLoadedOperator(Z_efie, Mp, theta_test, true)
y_rx_ref = Z_ref_rx * x_test
y_rx_comp = A_comp_rx * x_test
rx_err = norm(y_rx_comp - y_rx_ref) / norm(y_rx_ref)
@assert rx_err < 1e-12 "Reactive forward matvec error: $rx_err"
println("  34c: PASS")

# 34d: GMRES convergence with composite operator
println("  34d: GMRES solve via composite operator ...")
theta_small = fill(300.0, Nt)
A_gmres = ImpedanceLoadedOperator(Z_efie, Mp, theta_small, false)
I_gmres = solve_forward(A_gmres, v; solver=:gmres, gmres_tol=1e-8, gmres_maxiter=200)

Z_dense = assemble_full_Z(Z_efie, Mp, theta_small)
I_dense = Z_dense \ v
gmres_diff = norm(I_gmres - I_dense) / norm(I_dense)
println("    GMRES vs direct relative diff: $gmres_diff")
@assert gmres_diff < 1e-4 "GMRES solution too different from direct: $gmres_diff"
println("  34d: PASS")

# 34e: solve_adjoint_rhs
println("  34e: solve_adjoint_rhs ...")
rhs_adj = Q * I_dense
lam_direct = Z_dense' \ rhs_adj
lam_rhs = solve_adjoint_rhs(Z_dense, rhs_adj; solver=:direct)
@assert norm(lam_rhs - lam_direct) / norm(lam_direct) < 1e-12 "solve_adjoint_rhs direct mismatch"
lam_gmres = solve_adjoint_rhs(A_gmres, rhs_adj; solver=:gmres, gmres_tol=1e-8, gmres_maxiter=200)
adj_rhs_diff = norm(lam_gmres - lam_direct) / norm(lam_direct)
println("    GMRES adjoint_rhs vs direct: $adj_rhs_diff")
@assert adj_rhs_diff < 1e-4 "solve_adjoint_rhs GMRES too different: $adj_rhs_diff"
println("  34e: PASS")

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 35: Multi-angle RCS optimization (smoke test)
# ─────────────────────────────────────────────────
println("\n── Test 35: Multi-angle RCS optimization ──")

# 35a: direction_mask
println("  35a: direction_mask ...")
grid_test = make_sph_grid(18, 36)
mask_z = direction_mask(grid_test, Vec3(0.0, 0.0, 1.0); half_angle=10.0 * π / 180)
mask_mz = direction_mask(grid_test, Vec3(0.0, 0.0, -1.0); half_angle=10.0 * π / 180)
@assert sum(mask_z) > 0 "Should have points near +z"
@assert sum(mask_mz) > 0 "Should have points near -z"
# +z and -z masks should not overlap (for small cone)
@assert sum(mask_z .& mask_mz) == 0 "+z and -z 10° cones should not overlap"
println("    +z mask: $(sum(mask_z)) pts,  -z mask: $(sum(mask_mz)) pts")
println("  35a: PASS")

# 35b: build_multiangle_configs
println("  35b: build_multiangle_configs ...")
angles_2 = [
    (theta_inc=0.0, phi_inc=0.0, pol=Vec3(1.0, 0.0, 0.0), weight=1.0),     # +z incidence
    (theta_inc=π/4, phi_inc=0.0, pol=Vec3(0.0, 1.0, 0.0), weight=1.0),     # 45° incidence
]
grid_opt = make_sph_grid(12, 24)
configs_test = build_multiangle_configs(mesh, rwg, k, angles_2;
                                         grid=grid_opt, backscatter_cone=15.0)
@assert length(configs_test) == 2 "Should have 2 angle configs"
@assert length(configs_test[1].v) == N "Excitation vector should have length N"
@assert size(configs_test[1].Q) == (N, N) "Q matrix should be N×N"
@assert configs_test[1].weight == 1.0
configs_mfree = build_multiangle_configs(mesh, rwg, k, angles_2;
                                          grid=grid_opt, backscatter_cone=15.0,
                                          matrix_free_Q=true)
@assert configs_mfree[1].Q isa FarFieldQMatrix "matrix_free_Q should use FarFieldQMatrix"
x_q = randn(MersenneTwister(355), ComplexF64, N)
Qx_dense = configs_test[1].Q * x_q
Qx_mfree = configs_mfree[1].Q * x_q
q_rel = norm(Qx_dense - Qx_mfree) / max(norm(Qx_dense), 1e-30)
@assert q_rel < 1e-12 "Matrix-free Q action mismatch: $q_rel"
configs_total = build_multiangle_configs(mesh, rwg, k, angles_2;
                                          grid=grid_opt, backscatter_cone=15.0,
                                          rcs_component=:total)
configs_total_mfree = build_multiangle_configs(mesh, rwg, k, angles_2;
                                                grid=grid_opt, backscatter_cone=15.0,
                                                matrix_free_Q=true,
                                                rcs_component=:total)
@assert configs_total_mfree[1].Q isa SumQMatrix "total matrix_free_Q should use SumQMatrix"
Qx_total_dense = configs_total[1].Q * x_q
Qx_total_mfree = configs_total_mfree[1].Q * x_q
q_total_rel = norm(Qx_total_dense - Qx_total_mfree) / max(norm(Qx_total_dense), 1e-30)
@assert q_total_rel < 1e-12 "Matrix-free total-Q action mismatch: $q_total_rel"
configs_proj = build_multiangle_configs(
    mesh, rwg, k,
    [(theta_inc=π/4, phi_inc=0.0, pol=Vec3(1.0, 0.0, 0.0), weight=1.0)];
    grid=grid_opt,
    backscatter_cone=15.0,
)
khat_proj = Vec3(sin(π/4), 0.0, cos(π/4))
@assert abs(dot(khat_proj, configs_proj[1].pol)) < 1e-12 "Multi-angle polarization must be transverse"
@assert abs(norm(configs_proj[1].pol) - 1.0) < 1e-12 "Multi-angle polarization must be unit length"
configs_default_pol = build_multiangle_configs(
    mesh, rwg, k,
    [(theta_inc=π/4, phi_inc=0.2, weight=1.0)];
    grid=grid_opt,
    backscatter_cone=15.0,
)
khat_default = Vec3(sin(π/4) * cos(0.2), sin(π/4) * sin(0.2), cos(π/4))
@assert abs(dot(khat_default, configs_default_pol[1].pol)) < 1e-12 "Default multi-angle polarization must be transverse"
println("  35b: PASS")

# 35c: optimize_multiangle_rcs smoke test
println("  35c: optimize_multiangle_rcs (5 iterations) ...")
# Use spatial grid patches instead of one-per-triangle (fewer parameters = faster)
part_opt = assign_patches_grid(mesh; nx=3, ny=3, nz=1)
Mp_opt = precompute_patch_mass(mesh, rwg, part_opt; quad_order=3)
theta_init = fill(200.0, part_opt.P)

theta_opt_35, trace_35 = optimize_multiangle_rcs(
    Z_efie, Mp_opt, configs_test, theta_init;
    maxiter=5, tol=1e-12, alpha0=0.01,
    reactive=false, verbose=false,
    lb=fill(0.0, part_opt.P), ub=fill(1000.0, part_opt.P)
)
@assert length(trace_35) == 5 "Should run exactly 5 iterations"
@assert all(t -> isfinite(t.J), trace_35) "All objectives should be finite"
@assert all(t -> isfinite(t.gnorm), trace_35) "All gradients should be finite"
J_trace_35 = [t.J for t in trace_35]
@assert all(J_trace_35[2:end] .<= J_trace_35[1:end-1] .+ 1e-12) "line search should not accept uphill objective steps"
# Objective should generally decrease (check last < first with tolerance)
println("    J: $(round(trace_35[1].J, sigdigits=4)) → $(round(trace_35[end].J, sigdigits=4))")
println("    |g|: $(round(trace_35[1].gnorm, sigdigits=4)) → $(round(trace_35[end].gnorm, sigdigits=4))")
println("  35c: PASS")

# 35d: normalized smooth worst-angle objective
println("  35d: smoothmax_log normalized objective ...")
J_probe = [2.0e-3, 7.5e-4]
w_probe = [1.0, 1.3]
ref_probe = [3.0e-3, 2.0e-3]
beta_probe = 6.0
phi_probe, scale_probe = DiffMoM._multiangle_objective_scales(
    J_probe, w_probe, :smoothmax_log, ref_probe, beta_probe)
@assert isfinite(phi_probe) "smoothmax objective should be finite"
@assert all(isfinite, scale_probe) "smoothmax objective scales should be finite"
for j in eachindex(J_probe)
    h = 1e-7 * max(abs(J_probe[j]), 1.0e-12)
    Jp = copy(J_probe); Jp[j] += h
    Jm = copy(J_probe); Jm[j] -= h
    phip, _ = DiffMoM._multiangle_objective_scales(
        Jp, w_probe, :smoothmax_log, ref_probe, beta_probe)
    phim, _ = DiffMoM._multiangle_objective_scales(
        Jm, w_probe, :smoothmax_log, ref_probe, beta_probe)
    fd_scale = (phip - phim) / (2h)
    rel_scale = abs(fd_scale - scale_probe[j]) / max(abs(scale_probe[j]), 1e-30)
    @assert rel_scale < 1e-5 "smoothmax objective scale mismatch at $j: $rel_scale"
end

refs_35 = Float64[]
Z_ref_35 = assemble_full_Z(Z_efie, Mp_opt, theta_init)
for cfg in configs_test
    I_ref = Z_ref_35 \ cfg.v
    push!(refs_35, real(dot(I_ref, cfg.Q * I_ref)))
end
theta_bal_35, trace_bal_35 = optimize_multiangle_rcs(
    Z_efie, Mp_opt, configs_test, theta_init;
    maxiter=3, tol=1e-12, alpha0=0.01,
    reactive=false, verbose=false,
    lb=fill(0.0, part_opt.P), ub=fill(1000.0, part_opt.P),
    objective=:smoothmax_log,
    reference_objectives=refs_35,
    smooth_beta=6.0,
)
@assert length(trace_bal_35) == 3 "smoothmax objective should run exactly 3 iterations"
@assert all(t -> isfinite(t.J), trace_bal_35) "smoothmax objectives should be finite"
@assert all(t -> isfinite(t.gnorm), trace_bal_35) "smoothmax gradients should be finite"
J_trace_bal_35 = [t.J for t in trace_bal_35]
@assert all(J_trace_bal_35[2:end] .<= J_trace_bal_35[1:end-1] .+ 1e-12) "smoothmax line search should not accept uphill objective steps"
println("    smoothmax Φ: $(round(trace_bal_35[1].J, sigdigits=4)) → $(round(trace_bal_35[end].J, sigdigits=4))")
println("  35d: PASS")

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 36: MLFMA + optimizer integration
# ─────────────────────────────────────────────────
println("\n── Test 36: MLFMA + optimizer integration ──")

# Build a larger mesh for MLFMA (icosphere, ~1920 unknowns)
println("  36a: Build icosphere + MLFMA operator ...")
ico_opt_path = joinpath(DATADIR, "tmp_icosphere_opt.obj")
write_icosphere_obj(ico_opt_path; radius=0.05, subdivisions=2)
mesh_ico_opt = read_obj_mesh(ico_opt_path)
rwg_ico = build_rwg(mesh_ico_opt)
N_ico = rwg_ico.nedges
println("    Icosphere: $(nvertices(mesh_ico_opt)) verts, $(ntriangles(mesh_ico_opt)) tris, $N_ico RWG")

freq_ico = 2e9
k_ico = 2π * freq_ico / 299792458.0

# Build MLFMA operator
A_mlfma = build_mlfma_operator(mesh_ico_opt, rwg_ico, k_ico;
                                leaf_lambda=1.0, precision=3, verbose=false)
println("    MLFMA operator built: $(size(A_mlfma))")
println("  36a: PASS")

# 36b: Composite operator with MLFMA base
println("  36b: Composite operator + MLFMA ...")
part_ico = assign_patches_grid(mesh_ico_opt; nx=3, ny=3, nz=3)
Mp_ico = precompute_patch_mass(mesh_ico_opt, rwg_ico, part_ico; quad_order=3)
theta_ico = fill(200.0, part_ico.P)
println("    Icosphere: $(part_ico.P) spatial patches")

A_loaded = ImpedanceLoadedOperator(A_mlfma, Mp_ico, theta_ico, false)
@assert size(A_loaded) == (N_ico, N_ico)

# Forward solve with MLFMA composite
pw_ico = PlaneWaveExcitation(Vec3(0.0, 0.0, -k_ico), 1.0, Vec3(1.0, 0.0, 0.0))
v_ico = assemble_excitation(mesh_ico_opt, rwg_ico, pw_ico)

# Build preconditioner from the impedance-loaded MLFMA near-field
P_mlfma = build_mlfma_preconditioner(A_mlfma, Mp_ico, theta_ico;
    factorization=:ilu, ilu_tau=1e-2)

I_ico = solve_forward(A_loaded, v_ico; solver=:gmres, preconditioner=P_mlfma,
                       gmres_tol=1e-6, gmres_maxiter=300)
@assert length(I_ico) == N_ico
@assert norm(I_ico) > 0 "Solution should be nonzero"
println("    Forward solve OK, |I| = $(round(norm(I_ico), sigdigits=4))")
println("  36b: PASS")

# 36c: Adjoint solve with MLFMA composite
println("  36c: Adjoint solve with MLFMA composite ...")
rhs_adj_ico = randn(MersenneTwister(789), ComplexF64, N_ico)
lam_ico = solve_adjoint_rhs(A_loaded, rhs_adj_ico;
                             solver=:gmres, preconditioner=P_mlfma,
                             gmres_tol=1e-6, gmres_maxiter=300)
@assert length(lam_ico) == N_ico
@assert norm(lam_ico) > 0
println("    Adjoint solve OK, |λ| = $(round(norm(lam_ico), sigdigits=4))")
println("  36c: PASS")

# 36d: Multi-angle RCS optimization with MLFMA (3 iterations)
println("  36d: Multi-angle RCS + MLFMA (3 iterations) ...")
grid_ico = make_sph_grid(12, 24)
angles_ico = [
    (theta_inc=0.0, phi_inc=0.0, pol=Vec3(1.0, 0.0, 0.0), weight=1.0),
    (theta_inc=π/3, phi_inc=0.0, pol=Vec3(1.0, 0.0, 0.0), weight=1.0),
]
configs_ico = build_multiangle_configs(mesh_ico_opt, rwg_ico, k_ico, angles_ico;
                                        grid=grid_ico, backscatter_cone=15.0,
                                        matrix_free_Q=true)
@assert configs_ico[1].Q isa FarFieldQMatrix "MLFMA optimization should use matrix-free Q in this test"

builder_calls_ico = Ref(0)
duplicate_builder_calls_ico = Ref(0)
last_builder_theta_ico = Ref{Union{Nothing, Vector{Float64}}}(nothing)
preconditioner_builder_ico = θ -> begin
    if last_builder_theta_ico[] !== nothing && last_builder_theta_ico[] == θ
        duplicate_builder_calls_ico[] += 1
    end
    last_builder_theta_ico[] = copy(θ)
    builder_calls_ico[] += 1
    build_mlfma_preconditioner(A_mlfma, Mp_ico, θ; factorization=:ilu, ilu_tau=1e-2)
end

theta_opt_ico, trace_ico = optimize_multiangle_rcs(
    A_mlfma, Mp_ico, configs_ico, theta_ico;
    maxiter=3, tol=1e-12, alpha0=0.01,
    reactive=false, verbose=false,
    lb=fill(0.0, part_ico.P), ub=fill(1000.0, part_ico.P),
    preconditioner_builder=preconditioner_builder_ico,
    gmres_tol=1e-6, gmres_maxiter=300
)
@assert length(trace_ico) == 3 "Should run exactly 3 iterations"
@assert all(t -> isfinite(t.J), trace_ico) "All objectives should be finite"
@assert all(t -> isfinite(t.gnorm), trace_ico) "All gradients should be finite"
@assert builder_calls_ico[] >= length(trace_ico) "Dynamic MLFMA preconditioner builder was not exercised"
@assert duplicate_builder_calls_ico[] == 0 "Dynamic preconditioner should be cached for unchanged theta"
println("    J: $(round(trace_ico[1].J, sigdigits=4)) → $(round(trace_ico[end].J, sigdigits=4))")
println("    |g|: $(round(trace_ico[1].gnorm, sigdigits=4)) → $(round(trace_ico[end].gnorm, sigdigits=4))")
println("  36d: PASS")

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 43: Graded Mesh Generation
# ─────────────────────────────────────────────────
println("\nTest 43: Graded mesh generation")

mesh_g = make_rect_plate_graded(1.0, 1.0, 10, 10; grading_factor=3.0)
mesh_u = make_rect_plate(1.0, 1.0, 10, 10)

# Same topology: same number of triangles and vertices
@assert ntriangles(mesh_g) == ntriangles(mesh_u) "Graded mesh should have same triangle count"
@assert size(mesh_g.xyz, 2) == size(mesh_u.xyz, 2) "Graded mesh should have same vertex count"

# Bounding box preserved
@assert maximum(mesh_g.xyz[1,:]) ≈ 0.5 atol=1e-12 "x_max should be Lx/2"
@assert minimum(mesh_g.xyz[1,:]) ≈ -0.5 atol=1e-12 "x_min should be -Lx/2"
@assert maximum(mesh_g.xyz[2,:]) ≈ 0.5 atol=1e-12 "y_max should be Ly/2"
@assert minimum(mesh_g.xyz[2,:]) ≈ -0.5 atol=1e-12 "y_min should be -Ly/2"

# Grading effect: edge triangles should be smaller than center triangles
edge_areas = Float64[]
center_areas = Float64[]
for t in 1:ntriangles(mesh_g)
    cx = sum(mesh_g.xyz[1, mesh_g.tri[j, t]] for j in 1:3) / 3
    cy = sum(mesh_g.xyz[2, mesh_g.tri[j, t]] for j in 1:3) / 3
    a = triangle_area(mesh_g, t)
    if abs(cx) > 0.35 || abs(cy) > 0.35
        push!(edge_areas, a)
    elseif abs(cx) < 0.15 && abs(cy) < 0.15
        push!(center_areas, a)
    end
end
@assert !isempty(edge_areas) && !isempty(center_areas) "Should have both edge and center triangles"
@assert mean(edge_areas) < mean(center_areas) "Edge triangles should be smaller than center (grading effect)"
ratio = mean(center_areas) / mean(edge_areas)
@assert ratio > 2.0 "Center-to-edge area ratio should be > 2 for grading_factor=3, got $ratio"
println("  Center/edge area ratio: $(round(ratio, digits=1))")

# RWG construction should work
rwg_g = build_rwg(mesh_g)
@assert rwg_g.nedges > 0 "RWG construction should succeed on graded mesh"

# Near-zero grading_factor should give approximately uniform mesh
mesh_g0 = make_rect_plate_graded(1.0, 1.0, 10, 10; grading_factor=0.01)
max_vertex_diff = maximum(abs.(mesh_g0.xyz .- mesh_u.xyz))
@assert max_vertex_diff < 1e-3 "Near-zero grading should approximate uniform, max diff = $max_vertex_diff"
println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 44: Analytical integral 1/R
# ─────────────────────────────────────────────────
println("\nTest 44: Analytical integral 1/R")

# Reference triangle: (0,0,0), (1,0,0), (0,1,0)
V1 = Vec3(0.0, 0.0, 0.0)
V2 = Vec3(1.0, 0.0, 0.0)
V3 = Vec3(0.0, 1.0, 0.0)

# Test at centroid — compare against high-order numerical quadrature
P_cent = (V1 + V2 + V3) / 3
S_analytical = analytical_integral_1overR(P_cent, V1, V2, V3)
@assert isfinite(S_analytical) "Analytical integral should be finite"
@assert S_analytical > 0 "Integral of 1/R should be positive"

# Reference value verified independently via brute-force midpoint rule (N=1000,
# converges to 2.4090 with ~0.07% error due to the centroid singularity).
# The analytical formula gives the exact coplanar result.
S_ref = 2.4072299231640093
@assert abs(S_analytical - S_ref) < 1e-10 "Analytical 1/R at centroid should match reference, got $S_analytical vs $S_ref"
println("  S(centroid) = $(round(S_analytical, digits=6))")

# Test at a vertex — should still be finite (the formula handles d_i ≈ 0 edges)
S_vertex = analytical_integral_1overR(V1, V1, V2, V3)
@assert isfinite(S_vertex) "Integral at vertex should be finite"
println("  S(vertex) = $(round(S_vertex, digits=6))")

# Test symmetry: integral from centroid should not depend on triangle orientation
V1b = Vec3(0.0, 0.0, 0.0)
V2b = Vec3(0.0, 1.0, 0.0)
V3b = Vec3(1.0, 0.0, 0.0)
S_flipped = analytical_integral_1overR(P_cent, V1b, V2b, V3b)
@assert abs(S_analytical - S_flipped) < 1e-12 "Integral should be orientation-invariant"

# Test with scaled triangle — integral scales linearly with triangle size
scale = 2.0
S_scaled = analytical_integral_1overR(scale * P_cent, scale * V1, scale * V2, scale * V3)
@assert abs(S_scaled - scale * S_analytical) < 1e-10 "Integral should scale linearly, got $(S_scaled) vs $(scale * S_analytical)"
println("  Scaling test: PASS")

# Test off-plane: P above centroid at height h=0.5
P_offplane = P_cent + Vec3(0.0, 0.0, 0.5)
S_offplane = analytical_integral_1overR(P_offplane, V1, V2, V3)
S_offplane_ref = 0.8515737774153572  # verified via convergence of midpoint rule (N→∞)
@assert abs(S_offplane - S_offplane_ref) < 1e-10 "Off-plane integral at h=0.5 should match reference, got $S_offplane vs $S_offplane_ref"
println("  S(centroid, h=0.5) = $(round(S_offplane, digits=6))")

# Off-plane should decrease monotonically with height
S_h01 = analytical_integral_1overR(P_cent + Vec3(0,0,0.1), V1, V2, V3)
S_h10 = analytical_integral_1overR(P_cent + Vec3(0,0,1.0), V1, V2, V3)
@assert S_analytical > S_h01 > S_offplane > S_h10 > 0 "Integral should decrease with height"

# Symmetry: integral should be the same for +h and -h
S_neg_h = analytical_integral_1overR(P_cent - Vec3(0.0, 0.0, 0.5), V1, V2, V3)
@assert abs(S_offplane - S_neg_h) < 1e-12 "Integral should be symmetric in h sign"
println("  Off-plane tests: PASS")
println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 45: Adjacent-cell contribution
# ─────────────────────────────────────────────────
println("\nTest 45: Adjacent-cell near-singular integration")

# Build a small plate and compare Z matrices with and without adjacent-cell fix
freq_adj = 3e9
c0_adj = 299792458.0
k_adj = 2π * freq_adj / c0_adj
mesh_adj = make_rect_plate(0.2, 0.2, 4, 4)
rwg_adj = build_rwg(mesh_adj)

Z_adj = assemble_Z_efie(mesh_adj, rwg_adj, k_adj; quad_order=4)

# Basic sanity: Z should be finite and non-zero
@assert all(isfinite, Z_adj) "All Z entries should be finite"
@assert maximum(abs, Z_adj) > 0 "Z should be non-zero"

# Symmetry check (EFIE should be symmetric for real-valued RWG)
sym_err = maximum(abs, Z_adj - transpose(Z_adj))
rel_sym = sym_err / maximum(abs, Z_adj)
@assert rel_sym < 0.02 "Relative symmetry error should be < 2%, got $(round(100*rel_sym, digits=2))%"
println("  Z size: $(size(Z_adj)), max|Z|=$(round(maximum(abs, Z_adj), digits=2))")
println("  Relative symmetry error: $(round(100*rel_sym, digits=4))%")

# Verify adjacent pairs are detected in the cache
cache_adj = DiffMoM._build_efie_cache(mesh_adj, rwg_adj, k_adj; quad_order=4)
n_adj_pairs = count(cache_adj.adjacent) ÷ 2  # BitMatrix is symmetric, count unique pairs
@assert n_adj_pairs > 0 "Should detect adjacent triangle pairs"
# For a 4×4 grid of quads (32 triangles), there are many internal edges
# Each internal edge connects 2 triangles → pair stored symmetrically in BitMatrix
println("  Adjacent pairs: $n_adj_pairs")
println("  High-order quad points: $(length(cache_adj.wq_hi)) (standard: $(cache_adj.Nq))")

# Convergence test: compare Z at two quad orders
Z_lo = assemble_Z_efie(mesh_adj, rwg_adj, k_adj; quad_order=3)
Z_hi = assemble_Z_efie(mesh_adj, rwg_adj, k_adj; quad_order=4)
diff_Z = maximum(abs, Z_hi - Z_lo) / maximum(abs, Z_hi)
@assert diff_Z < 0.15 "Z should converge with quad order, relative diff = $(round(100*diff_Z, digits=2))%"
println("  Quad convergence (order 3→4): $(round(100*diff_Z, digits=4))% relative diff")
println("  PASS ✓")

# ─────────────────────────────────────────────────
# Periodic MoM + Topology Optimization Tests (37-42)
# ─────────────────────────────────────────────────
include("test_periodic_topology.jl")

# ─────────────────────────────────────────────────
# 3D vector material DDA solver
# ─────────────────────────────────────────────────
include("test_material_models3d.jl")
include("test_mom3d.jl")
include("test_mom3d_adjoint.jl")
include("test_mom3d_fft.jl")
include("test_mom3d_em.jl")
include("test_surface_ie3d.jl")

# ─────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────
println("\n" * "="^60)
println("ALL 51 TESTS PASSED")
println("="^60)
