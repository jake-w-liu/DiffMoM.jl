# 10_mlfma_scaling.jl — Aircraft MLFMA Comparison at 0.3 GHz
#
# Demonstrates Dense vs ACA vs MLFMA solvers on the aircraft at 0.3 GHz
# across two mesh refinement levels (4λ and 1λ), matching the setup from ex09 Part B.
#
# Key results:
#   - MLFMA gives same RCS patterns as Dense/ACA (within ~0.5 dB)
#   - MLFMA scales to N>10k where dense methods become impractical
#   - All methods produce consistent backscatter RCS values
#   - MIN_N_MLFMA threshold (2000) prevents inefficient use on small problems
#
# Run: julia -t 4 --project=. examples/10_mlfma_scaling.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra
using SparseArrays
using PlotlySupply

println("="^72)
println("Example 10: Dense vs ACA vs MLFMA — Aircraft at 0.3 GHz")
println("="^72)

c0 = 299792458.0
figdir = joinpath(@__DIR__, "figs")
mkpath(figdir)

"""
    make_cut_grid(Ntheta, phi_values)

Create a SphGrid with 1° θ resolution at specific φ planes only.
"""
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
println("Far-field grid: $NΩ_cuts points (1° θ × φ=0°,90°)")

# ── Mesh refinement levels ──
println("\n── Mesh refinement ──")
ref1 = refine_mesh_to_target_edge(mesh_air, 4.0 * λ0; max_iters=2, max_triangles=50_000)
println("  Level 1: target 4λ, $(ntriangles(ref1.mesh)) tri, $(ref1.iterations) pass, max edge $(round(ref1.edge_max_after_m/λ0, digits=2))λ")
ref2 = refine_mesh_to_target_edge(mesh_air, 1.0 * λ0; max_iters=5, max_triangles=80_000)
println("  Level 2: target 1λ, $(ntriangles(ref2.mesh)) tri, $(ref2.iterations) passes, max edge $(round(ref2.edge_max_after_m/λ0, digits=2))λ")

mesh_levels = [
    (ref1.mesh, "4λ mesh"),
    (ref2.mesh, "1λ mesh"),
]

# Parameters
MAX_N_DENSE = 10_000
MIN_N_MLFMA = 2_000  # MLFMA not suitable for small problems
LEAF_LAMBDA = 2.0     # Octree leaf size in wavelengths (larger = fewer levels, faster build)
ILU_TAU = 1e-3
GMRES_TOL = 1e-6

# Cutoff and factorization strategy per N
nf_cutoff_for(N_m) = N_m <= 10_000 ? 5.0 * λ0 : 2.0 * λ0
nf_fac_for(N_m) = N_m <= 10_000 ? :lu : :ilu
gmres_maxiter_for(N_m) = N_m <= 10_000 ? 1000 : 2000

# Storage for RCS curves
all_curves = Dict{String, Vector{Float64}}()
curve_order = String[]

# ════════════════════════════════════════════════════
# Solver comparison across mesh levels
# ════════════════════════════════════════════════════
println("\n" * "="^72)
println("Solver comparison (NF: 5λ+LU for N≤10k, 2λ+ILU for N>10k)")
println("="^72)
println("  Level        N    Method           Asm(s)  Sol(s)  Iters  Backscatter   Notes")
println("  " * "─"^72)

for (m, label) in mesh_levels
    rwg_m = build_rwg(m)
    N_m = rwg_m.nedges
    v_m = assemble_excitation(m, rwg_m, pw)

    # ────────────────────────────────────────
    # 1. Dense direct (if N ≤ MAX_N_DENSE)
    # ────────────────────────────────────────
    Z_dense = nothing
    I_dense = nothing

    if N_m <= MAX_N_DENSE
        t_asm_d = @elapsed Z_dense = assemble_Z_efie(m, rwg_m, k; mesh_precheck=false)
        t_sol_d = @elapsed I_dense = Z_dense \ v_m

        G_d = radiation_vectors(m, rwg_m, grid_cuts, k)
        E_ff_d = compute_farfield(G_d, Vector{ComplexF64}(I_dense), NΩ_cuts)
        σ_d = 10 .* log10.(max.(bistatic_rcs(E_ff_d; E0=1.0), 1e-30))
        bs_d = backscatter_rcs(E_ff_d, grid_cuts, Vec3(0.0, 0.0, -k); E0=1.0)
        bs_d_dB = round(10*log10(max(bs_d.sigma, 1e-30)), digits=2)

        println("  $(rpad(label, 12))  $(lpad(N_m, 5))  $(rpad("Dense (LU)", 16))  " *
                "$(lpad(round(t_asm_d, digits=1), 5))  $(lpad(round(t_sol_d, digits=1), 6))      —  " *
                "$(lpad(bs_d_dB, 10)) dBsm  ground truth")

        dlabel = "$label Dense"
        all_curves[dlabel] = σ_d
        push!(curve_order, dlabel)
    else
        println("  $(rpad(label, 12))  $(lpad(N_m, 5))  $(rpad("Dense (LU)", 16))    SKIP (N > $MAX_N_DENSE)")
    end

    # ────────────────────────────────────────
    # 2. ACA + GMRES
    # ────────────────────────────────────────
    t_aca = @elapsed A_aca = build_aca_operator(m, rwg_m, k;
        leaf_size=32, eta=1.5, aca_tol=1e-6, max_rank=80, mesh_precheck=false)

    cutoff_a = nf_cutoff_for(N_m)
    fac_a = nf_fac_for(N_m)
    t_pre_a = @elapsed P_nf_a = build_nearfield_preconditioner(A_aca;
        factorization=fac_a, ilu_tau=ILU_TAU)

    t_sol_a = @elapsed begin
        I_aca, stats_a = solve_gmres(A_aca, v_m; preconditioner=P_nf_a,
            tol=GMRES_TOL, maxiter=gmres_maxiter_for(N_m))
    end

    G_a = radiation_vectors(m, rwg_m, grid_cuts, k)
    E_ff_a = compute_farfield(G_a, Vector{ComplexF64}(I_aca), NΩ_cuts)
    σ_a = 10 .* log10.(max.(bistatic_rcs(E_ff_a; E0=1.0), 1e-30))
    bs_a = backscatter_rcs(E_ff_a, grid_cuts, Vec3(0.0, 0.0, -k); E0=1.0)
    bs_a_dB = round(10*log10(max(bs_a.sigma, 1e-30)), digits=2)

    err_note_a = ""
    if !isnothing(I_dense)
        err_a = norm(I_aca - I_dense) / norm(I_dense)
        err_note_a = "err=$(round(err_a, sigdigits=2))"
    else
        err_note_a = "—"
    end

    println("  $(rpad(label, 12))  $(lpad(N_m, 5))  $(rpad("ACA+GMRES", 16))  " *
            "$(lpad(round(t_aca+t_pre_a, digits=1), 5))  $(lpad(round(t_sol_a, digits=1), 6))  " *
            "$(lpad(stats_a.niter, 5))  $(lpad(bs_a_dB, 10)) dBsm  $err_note_a")

    alabel = "$label ACA"
    all_curves[alabel] = σ_a
    push!(curve_order, alabel)

    # ────────────────────────────────────────
    # 3. MLFMA + GMRES (if N ≥ MIN_N_MLFMA)
    # ────────────────────────────────────────
    if N_m >= MIN_N_MLFMA
        t_mlfma = @elapsed A_mlfma = build_mlfma_operator(m, rwg_m, k;
            leaf_lambda=LEAF_LAMBDA, verbose=true)

        t_pre_m = @elapsed P_nf_m = build_nearfield_preconditioner(A_mlfma.Z_near;
            factorization=:ilu, ilu_tau=ILU_TAU)

        t_sol_m = @elapsed begin
            I_mlfma, stats_m = solve_gmres(A_mlfma, v_m;
                preconditioner=P_nf_m, tol=GMRES_TOL, maxiter=gmres_maxiter_for(N_m))
        end

        G_m = radiation_vectors(m, rwg_m, grid_cuts, k)
        E_ff_m = compute_farfield(G_m, Vector{ComplexF64}(I_mlfma), NΩ_cuts)
        σ_m = 10 .* log10.(max.(bistatic_rcs(E_ff_m; E0=1.0), 1e-30))
        bs_m = backscatter_rcs(E_ff_m, grid_cuts, Vec3(0.0, 0.0, -k); E0=1.0)
        bs_m_dB = round(10*log10(max(bs_m.sigma, 1e-30)), digits=2)

        nL = A_mlfma.octree.nLevels
        n_leaf = length(A_mlfma.octree.levels[end].boxes)
        nnz_pct = round(nnz(A_mlfma.Z_near) / N_m^2 * 100, digits=1)

        err_note_m = ""
        if !isnothing(I_dense)
            err_m = norm(I_mlfma - I_dense) / norm(I_dense)
            err_note_m = "err=$(round(err_m, sigdigits=2))"
        else
            err_note_m = "—"
        end

        println("  $(rpad(label, 12))  $(lpad(N_m, 5))  $(rpad("MLFMA+GMRES", 16))  " *
                "$(lpad(round(t_mlfma+t_pre_m, digits=1), 5))  $(lpad(round(t_sol_m, digits=1), 6))  " *
                "$(lpad(stats_m.niter, 5))  $(lpad(bs_m_dB, 10)) dBsm  $err_note_m")
        println("      └─ $nL levels, $n_leaf leaf boxes, NF nnz=$(nnz_pct)%")

        mlabel = "$label MLFMA"
        all_curves[mlabel] = σ_m
        push!(curve_order, mlabel)
    else
        println("  $(rpad(label, 12))  $(lpad(N_m, 5))  $(rpad("MLFMA+GMRES", 16))    SKIP (N < $MIN_N_MLFMA)")
    end
end

# ════════════════════════════════════════════════════
# RCS plots
# ════════════════════════════════════════════════════
println("\n" * "="^72)
println("RCS plots")
println("="^72)

phi0_idx = [q for q in 1:NΩ_cuts if abs(grid_cuts.phi[q]) < 0.01]
phi90_idx = [q for q in 1:NΩ_cuts if abs(grid_cuts.phi[q] - π/2) < 0.01]
sort!(phi0_idx; by=q -> grid_cuts.theta[q])
sort!(phi90_idx; by=q -> grid_cuts.theta[q])
θ_deg_0  = [rad2deg(grid_cuts.theta[q]) for q in phi0_idx]
θ_deg_90 = [rad2deg(grid_cuts.theta[q]) for q in phi90_idx]

# Color/dash scheme: color per mesh level, solid=dense, dashed=ACA, dotted=MLFMA
mesh_colors = ["#ff7f0e", "#d62728"]  # orange, red (for 4λ and 1λ meshes)
style_map = Dict{String, NamedTuple{(:color, :width, :dash), Tuple{String, Int, String}}}()

for (i, (_, label)) in enumerate(mesh_levels)
    style_map["$label Dense"]  = (color=mesh_colors[i], width=3, dash="solid")
    style_map["$label ACA"]    = (color=mesh_colors[i], width=2, dash="dash")
    style_map["$label MLFMA"]  = (color=mesh_colors[i], width=2, dash="dot")
end

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
    "Aircraft RCS φ=0° — Dense vs ACA vs MLFMA ($(freq/1e9) GHz)")
savefig(p0, joinpath(figdir, "10_aircraft_rcs_phi0.png"); width=950, height=600)
println("Plot saved: 10_aircraft_rcs_phi0.png")

p90 = make_rcs_plot(phi90_idx, θ_deg_90,
    "Aircraft RCS φ=90° — Dense vs ACA vs MLFMA ($(freq/1e9) GHz)")
savefig(p90, joinpath(figdir, "10_aircraft_rcs_phi90.png"); width=950, height=600)
println("Plot saved: 10_aircraft_rcs_phi90.png")

println("\n" * "="^72)
println("Done.")
