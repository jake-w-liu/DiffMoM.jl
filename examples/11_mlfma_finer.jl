# 11_mlfma_finer.jl — MLFMA + PO on aircraft at finer meshes
#
# Scales MLFMA beyond dense-feasible sizes on the 14λ aircraft at 0.3 GHz.
# PO is computed on the SAME mesh at each level for fair comparison.
#
# Mesh levels:
#   Level 0: 4λ edge (N≈7.5k)  — Dense (ground truth) + MLFMA + PO
#   Level 1: 2λ edge (N≈15k)   — MLFMA + PO  (demonstrates mesh convergence)
#
# Run: julia -t 4 --project=. examples/11_mlfma_finer.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra
using SparseArrays
using PlotlySupply

println("="^72)
println("Example 11: MLFMA + PO — Aircraft at 0.3 GHz, Finer Meshes")
println("="^72)

c0 = 299792458.0
figdir = joinpath(@__DIR__, "figs")
mkpath(figdir)

function make_cut_grid(Ntheta::Int, phi_values::Vector{Float64})
    dtheta = π / Ntheta
    Nphi = length(phi_values)
    NΩ = Ntheta * Nphi
    rhat  = zeros(3, NΩ)
    theta = zeros(NΩ)
    phi   = zeros(NΩ)
    w     = zeros(NΩ)
    idx = 0
    for it in 1:Ntheta
        θ = (it - 0.5) * dtheta
        for φ in phi_values
            idx += 1
            theta[idx] = θ
            phi[idx]   = φ
            rhat[1, idx] = sin(θ) * cos(φ)
            rhat[2, idx] = sin(θ) * sin(φ)
            rhat[3, idx] = cos(θ)
            w[idx] = sin(θ) * dtheta * (2π / Nphi)
        end
    end
    return SphGrid(rhat, theta, phi, w)
end

# ════════════════════════════════════════════════════
# Setup
# ════════════════════════════════════════════════════
obj_path = joinpath(@__DIR__, "demo_aircraft.obj")
isfile(obj_path) || error("demo_aircraft.obj not found at $obj_path")

mesh_raw = read_obj_mesh(obj_path)
rep = repair_mesh_for_simulation(mesh_raw; allow_boundary=true, auto_drop_nonmanifold=true)
mesh_air = rep.mesh

freq = 0.3e9
λ0 = c0 / freq
k  = 2π / λ0

println("\nAircraft: $(nvertices(mesh_air)) verts, $(ntriangles(mesh_air)) tri")
println("Frequency: $(freq/1e9) GHz, λ = $(round(λ0, digits=2)) m")
println("Wingspan ~14m ≈ $(round(14.0/λ0, digits=1))λ")

pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
grid_cuts = make_cut_grid(180, [0.0, π/2])
NΩ_cuts = length(grid_cuts.w)

# ── Mesh refinement ──
println("\n── Mesh refinement ──")
ref0 = refine_mesh_to_target_edge(mesh_air, 4.0 * λ0; max_iters=2, max_triangles=50_000)
println("  Level 0 (4λ): $(ntriangles(ref0.mesh)) tri, max edge $(round(ref0.edge_max_after_m, digits=2))m ($(round(ref0.edge_max_after_m/λ0, digits=2))λ)")

ref1 = refine_mesh_to_target_edge(mesh_air, 2.0 * λ0; max_iters=3, max_triangles=50_000)
println("  Level 1 (2λ): $(ntriangles(ref1.mesh)) tri, max edge $(round(ref1.edge_max_after_m, digits=3))m ($(round(ref1.edge_max_after_m/λ0, digits=3))λ)")

ILU_TAU = 1e-3
GMRES_TOL = 1e-6

# Storage for all curves across levels
all_curves = Dict{String, Vector{Float64}}()
curve_order = String[]

# ════════════════════════════════════════════════════════════════════════
# Level 0: 4λ mesh — Dense (ground truth) + MLFMA + PO
# ════════════════════════════════════════════════════════════════════════
println("\n" * "="^72)
println("Level 0: 4λ mesh — Dense + MLFMA + PO")
println("="^72)

mesh0 = ref0.mesh
rwg0 = build_rwg(mesh0)
N0 = rwg0.nedges
v0 = assemble_excitation(mesh0, rwg0, pw)
println("  N = $N0 RWG unknowns")

# ── PO (on same 4λ mesh) ──
println("\n  [PO]")
t = @elapsed po0 = solve_po(mesh0, freq, pw; grid=grid_cuts)
σ_po0 = 10 .* log10.(max.(bistatic_rcs(po0.E_ff; E0=1.0), 1e-30))
bs_po0 = backscatter_rcs(po0.E_ff, grid_cuts, Vec3(0.0, 0.0, -k); E0=1.0)
println("    $(ntriangles(mesh0)) tri, $(count(po0.illuminated)) illum, $(round(t, digits=2))s")
println("    Backscatter: $(round(10*log10(max(bs_po0.sigma, 1e-30)), digits=2)) dBsm")
all_curves["PO (4λ)"] = σ_po0
push!(curve_order, "PO (4λ)")

# ── Dense direct ──
println("\n  [Dense direct]")
t_asm = @elapsed Z0 = assemble_Z_efie(mesh0, rwg0, k; mesh_precheck=false)
t_sol = @elapsed I_dense0 = Z0 \ v0
G0 = radiation_vectors(mesh0, rwg0, grid_cuts, k)
E_ff_d0 = compute_farfield(G0, Vector{ComplexF64}(I_dense0), NΩ_cuts)
σ_d0 = 10 .* log10.(max.(bistatic_rcs(E_ff_d0; E0=1.0), 1e-30))
bs_d0 = backscatter_rcs(E_ff_d0, grid_cuts, Vec3(0.0, 0.0, -k); E0=1.0)
println("    Assembly: $(round(t_asm, digits=1))s, Solve: $(round(t_sol, digits=1))s")
println("    Backscatter: $(round(10*log10(max(bs_d0.sigma, 1e-30)), digits=2)) dBsm")
all_curves["Dense (4λ, N=$N0)"] = σ_d0
push!(curve_order, "Dense (4λ, N=$N0)")

# ── MLFMA ──
println("\n  [MLFMA + GMRES]")
t_build = @elapsed A0 = build_mlfma_operator(mesh0, rwg0, k; leaf_lambda=3.0, verbose=true)
t_pre = @elapsed P0 = build_nearfield_preconditioner(A0.Z_near; factorization=:ilu, ilu_tau=ILU_TAU)
t_sol_m = @elapsed begin
    global I_mlfma0, stats0
    I_mlfma0, stats0 = solve_gmres(A0, v0; preconditioner=P0, tol=GMRES_TOL, maxiter=2000)
end
E_ff_m0 = compute_farfield(G0, Vector{ComplexF64}(I_mlfma0), NΩ_cuts)
σ_m0 = 10 .* log10.(max.(bistatic_rcs(E_ff_m0; E0=1.0), 1e-30))
bs_m0 = backscatter_rcs(E_ff_m0, grid_cuts, Vec3(0.0, 0.0, -k); E0=1.0)
err0 = norm(I_mlfma0 - I_dense0) / norm(I_dense0)
nL0 = A0.octree.nLevels
n_leaf0 = length(A0.octree.levels[end].boxes)
nnz0 = round(nnz(A0.Z_near) / N0^2 * 100, digits=1)
println("    Build: $(round(t_build, digits=1))s, Pre: $(round(t_pre, digits=1))s, Solve: $(round(t_sol_m, digits=1))s ($(stats0.niter) iters)")
println("    $nL0 levels, $n_leaf0 leaf boxes, NF nnz=$(nnz0)%")
println("    Backscatter: $(round(10*log10(max(bs_m0.sigma, 1e-30)), digits=2)) dBsm, coeff err vs dense: $(round(err0, sigdigits=2))")
all_curves["MLFMA (4λ, N=$N0)"] = σ_m0
push!(curve_order, "MLFMA (4λ, N=$N0)")

# Free Level 0 data to reclaim memory for Level 1
Z0 = nothing; I_dense0 = nothing; A0 = nothing; P0 = nothing; G0 = nothing
GC.gc()

# ════════════════════════════════════════════════════════════════════════
# Level 1: 2λ mesh — MLFMA + PO
# ════════════════════════════════════════════════════════════════════════
println("\n" * "="^72)
println("Level 1: 2λ mesh — MLFMA + PO")
println("="^72)

mesh1 = ref1.mesh
rwg1 = build_rwg(mesh1)
N1 = rwg1.nedges
v1 = assemble_excitation(mesh1, rwg1, pw)
println("  N = $N1 RWG unknowns (dense Z would be $(round(N1^2*16/1e9, digits=1)) GB)")

# ── PO (on same 1λ mesh) ──
println("\n  [PO]")
t = @elapsed po1 = solve_po(mesh1, freq, pw; grid=grid_cuts)
σ_po1 = 10 .* log10.(max.(bistatic_rcs(po1.E_ff; E0=1.0), 1e-30))
bs_po1 = backscatter_rcs(po1.E_ff, grid_cuts, Vec3(0.0, 0.0, -k); E0=1.0)
println("    $(ntriangles(mesh1)) tri, $(count(po1.illuminated)) illum, $(round(t, digits=2))s")
println("    Backscatter: $(round(10*log10(max(bs_po1.sigma, 1e-30)), digits=2)) dBsm")
all_curves["PO (2λ)"] = σ_po1
push!(curve_order, "PO (2λ)")

# ── MLFMA + GMRES ──
# Reorder Z_near to MLFMA ordering (block-banded) before ILU → fast factorization.
println("\n  [MLFMA + reordered-ILU GMRES]")
t_build = @elapsed A1 = build_mlfma_operator(mesh1, rwg1, k; leaf_lambda=1.0, verbose=true)
nL1 = A1.octree.nLevels
n_leaf1 = length(A1.octree.levels[end].boxes)
nnz1 = round(nnz(A1.Z_near) / N1^2 * 100, digits=1)
nf_gb = round(nnz(A1.Z_near) * 16 / 1e9, digits=1)
println("    NF nnz=$(nnz1)% → $(nf_gb) GB sparse values")

t_pre = @elapsed P1 = build_mlfma_preconditioner(A1; factorization=:ilu, ilu_tau=1e-3)
t_sol1 = @elapsed begin
    global I_mlfma1, stats1
    I_mlfma1, stats1 = solve_gmres(A1, v1; preconditioner=P1, tol=GMRES_TOL, maxiter=2000, memory=100)
end

# True (unpreconditioned) residual check
y_check = similar(v1)
mul!(y_check, A1, I_mlfma1)
true_res = norm(v1 - y_check) / norm(v1)
println("    True residual: $(round(true_res, sigdigits=3))")

G1 = radiation_vectors(mesh1, rwg1, grid_cuts, k)
E_ff_m1 = compute_farfield(G1, Vector{ComplexF64}(I_mlfma1), NΩ_cuts)
σ_m1 = 10 .* log10.(max.(bistatic_rcs(E_ff_m1; E0=1.0), 1e-30))
bs_m1 = backscatter_rcs(E_ff_m1, grid_cuts, Vec3(0.0, 0.0, -k); E0=1.0)
println("    Build: $(round(t_build, digits=1))s, Pre: $(round(t_pre, digits=1))s, Solve: $(round(t_sol1, digits=1))s ($(stats1.niter) iters)")
println("    $nL1 levels, $n_leaf1 leaf boxes, NF nnz=$(nnz1)%")
println("    Backscatter: $(round(10*log10(max(bs_m1.sigma, 1e-30)), digits=2)) dBsm")
all_curves["MLFMA (2λ, N=$N1)"] = σ_m1
push!(curve_order, "MLFMA (2λ, N=$N1)")

# ════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════
println("\n" * "="^72)
println("Summary")
println("="^72)
println("  Level  Mesh    N        Method     Backscatter  Iters  Notes")
println("  " * "─"^68)
println("  L0     4λ   $(lpad(N0, 6))    PO         $(lpad(round(10*log10(max(bs_po0.sigma, 1e-30)), digits=2), 8)) dBsm      —")
println("  L0     4λ   $(lpad(N0, 6))    Dense      $(lpad(round(10*log10(max(bs_d0.sigma, 1e-30)), digits=2), 8)) dBsm      —     ground truth")
println("  L0     4λ   $(lpad(N0, 6))    MLFMA      $(lpad(round(10*log10(max(bs_m0.sigma, 1e-30)), digits=2), 8)) dBsm  $(lpad(stats0.niter, 4))     err=$(round(err0, sigdigits=2))")
println("  L1     2λ   $(lpad(N1, 6))    PO         $(lpad(round(10*log10(max(bs_po1.sigma, 1e-30)), digits=2), 8)) dBsm      —")
println("  L1     2λ   $(lpad(N1, 6))    MLFMA      $(lpad(round(10*log10(max(bs_m1.sigma, 1e-30)), digits=2), 8)) dBsm  $(lpad(stats1.niter, 4))")

# ════════════════════════════════════════════════════════════════════════
# RCS plots
# ════════════════════════════════════════════════════════════════════════
println("\n" * "="^72)
println("RCS plots")
println("="^72)

phi0_idx = [q for q in 1:NΩ_cuts if abs(grid_cuts.phi[q]) < 0.01]
phi90_idx = [q for q in 1:NΩ_cuts if abs(grid_cuts.phi[q] - π/2) < 0.01]
sort!(phi0_idx; by=q -> grid_cuts.theta[q])
sort!(phi90_idx; by=q -> grid_cuts.theta[q])
θ_deg_0  = [rad2deg(grid_cuts.theta[q]) for q in phi0_idx]
θ_deg_90 = [rad2deg(grid_cuts.theta[q]) for q in phi90_idx]

style_map = Dict{String, NamedTuple{(:color, :width, :dash), Tuple{String, Int, String}}}()
style_map["PO (4λ)"]                = (color="gray",    width=2, dash="dot")
style_map["PO (2λ)"]                = (color="gray",    width=2, dash="solid")
style_map["Dense (4λ, N=$N0)"]      = (color="black",   width=3, dash="solid")
style_map["MLFMA (4λ, N=$N0)"]      = (color="#ff7f0e", width=2, dash="dash")
style_map["MLFMA (2λ, N=$N1)"]      = (color="#d62728", width=3, dash="solid")

function make_rcs_plot(cut_idx, θ_deg, title_str)
    sf = subplots(1, 1; sync=false, width=950, height=600,
                  subplot_titles=reshape([title_str], 1, 1))
    for clabel in curve_order
        haskey(all_curves, clabel) || continue
        dB = all_curves[clabel]
        s = get(style_map, clabel, (color="gray", width=1, dash="solid"))
        addtraces!(sf,
            scatter(x=θ_deg, y=[dB[q] for q in cut_idx],
                    mode="lines", name=clabel,
                    line=attr(color=s.color, width=s.width, dash=s.dash));
            row=1, col=1)
    end
    p = sf.plot
    relayout!(p,
        xaxis=attr(title="θ (deg)", range=[0, 180]),
        yaxis=attr(title="Bistatic RCS (dBsm)"),
        legend=attr(x=0.01, y=0.01, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.8)", font=attr(size=10)),
        margin=attr(l=60, r=30, t=60, b=50))
    return p
end

p0 = make_rcs_plot(phi0_idx, θ_deg_0,
    "Aircraft RCS φ=0° — MLFMA + PO ($(freq/1e9) GHz, mesh convergence)")
savefig(p0, joinpath(figdir, "11_aircraft_rcs_phi0.png"); width=950, height=600)
println("Plot saved: 11_aircraft_rcs_phi0.png")

p90 = make_rcs_plot(phi90_idx, θ_deg_90,
    "Aircraft RCS φ=90° — MLFMA + PO ($(freq/1e9) GHz, mesh convergence)")
savefig(p90, joinpath(figdir, "11_aircraft_rcs_phi90.png"); width=950, height=600)
println("Plot saved: 11_aircraft_rcs_phi90.png")

println("\n" * "="^72)
println("Done.")
