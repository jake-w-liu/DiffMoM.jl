# 09a_aircraft_po.jl — Aircraft RCS: PO vs MoM at multiple mesh refinements
#
# Demonstrates:
#   Aircraft at 0.3 GHz (14λ wingspan) — PO, direct, iterative solvers
#   at three mesh refinement levels, with 1° RCS pattern plots
#
# Requires: examples/demo_aircraft.obj
#
# Run: julia -t 4 --project=. examples/09a_aircraft_po.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra
using SparseArrays
using CSV
using DataFrames
using PlotlySupply

println("="^60)
println("Example 09a: Aircraft PO vs MoM")
println("="^60)

c0 = 299792458.0
figdir = joinpath(@__DIR__, "figs")
mkpath(figdir)
datadir = joinpath(@__DIR__, "..", "data")
mkpath(datadir)

"""
    make_cut_grid(Ntheta, phi_values)

Create a SphGrid with 1° θ resolution at specific φ planes only.
Much smaller than a full sphere grid → feasible for large N far-field.
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
# Aircraft at 0.3 GHz — direct vs iterative
# ════════════════════════════════════════════════════
println("\n" * "─"^60)
println("Aircraft at 0.3 GHz — PO / Direct / Iterative")
println("─"^60)

obj_path = joinpath(@__DIR__, "demo_aircraft.obj")
if !isfile(obj_path)
    println("SKIP: demo_aircraft.obj not found")
else
    mesh_raw = read_obj_mesh(obj_path)
    rep = repair_mesh_for_simulation(mesh_raw; allow_boundary=true, auto_drop_nonmanifold=true)
    mesh_air = rep.mesh

    freq_air = 0.3e9
    λ_air = c0 / freq_air
    k_air = 2π / λ_air

    res = mesh_resolution_report(mesh_air, freq_air)
    println("\nAircraft: $(nvertices(mesh_air)) verts, $(ntriangles(mesh_air)) tri")
    println("Frequency: $(freq_air/1e9) GHz, λ = $(round(λ_air, digits=2)) m")
    println("Wingspan ~14m ≈ $(round(14.0/λ_air, digits=1))λ")
    println("Original max edge/λ: $(round(res.edge_max_over_lambda, digits=2))")

    pw_air = make_plane_wave(Vec3(0.0, 0.0, -k_air), 1.0, Vec3(1.0, 0.0, 0.0))

    # 1° θ resolution at φ=0° and φ=90° (360 observation points)
    grid_cuts = make_cut_grid(180, [0.0, π/2])
    NΩ_cuts = length(grid_cuts.w)
    println("Far-field grid: $NΩ_cuts points (1° θ × φ=0°,90°)")

    # Also keep a coarse sphere for RMSE (small N only)
    grid_coarse = make_sph_grid(36, 72)
    NΩ_coarse = length(grid_coarse.w)

    # ── 1. Refine mesh to 1λ edge ──
    target_1lam = 1.0 * λ_air

    println("\n── Mesh refinement ──")
    ref1 = refine_mesh_to_target_edge(mesh_air, target_1lam; max_iters=5, max_triangles=80_000)
    println("  1λ mesh: $(ntriangles(ref1.mesh)) tri, $(ref1.iterations) passes, max edge $(round(ref1.edge_max_after_m, digits=2))m")

    mesh_levels = [
        (ref1.mesh, "1λ mesh"),
    ]

    # ── 2. PO reference on 1λ mesh ──
    println("\n── PO reference (on 1λ mesh) ──")
    po_air = solve_po(ref1.mesh, freq_air, pw_air; grid=grid_cuts)
    σ_po_cuts = bistatic_rcs(po_air.E_ff; E0=1.0)
    σ_po_cuts_dB = 10 .* log10.(max.(σ_po_cuts, 1e-30))
    bs_po = backscatter_rcs(po_air.E_ff, grid_cuts, Vec3(0.0, 0.0, -k_air); E0=1.0)
    println("  $(ntriangles(ref1.mesh)) tri, $(count(po_air.illuminated)) illum")
    println("  Backscatter: $(round(10*log10(bs_po.sigma), digits=2)) dBsm")

    # PO on coarse grid for RMSE reference
    po_air_coarse = solve_po(ref1.mesh, freq_air, pw_air; grid=grid_coarse)
    σ_po_coarse_dB = 10 .* log10.(max.(bistatic_rcs(po_air_coarse.E_ff; E0=1.0), 1e-30))

    # ── 3. Solve with Direct / ACA / MLFMA ──
    all_curves = Dict{String, Vector{Float64}}()
    all_curves["PO"] = σ_po_cuts_dB
    curve_order = ["PO"]

    LEAF_LAMBDA = 2.0
    ILU_TAU = 1e-3
    GMRES_TOL = 1e-6

    println("\n── MLFMA solver ──")
    println("  Level          N     Solver        Asm(s)  Sol(s)  Iters  Backscatter  RMSE vs PO")
    println("  " * "─"^88)

    for (m, label) in mesh_levels
        rwg_m = build_rwg(m)
        N_m = rwg_m.nedges
        v_m = assemble_excitation(m, rwg_m, pw_air)

        # ── MLFMA + GMRES ──
        begin
            println("    Building MLFMA operator...")
            t_mlfma = @elapsed A_mlfma = build_mlfma_operator(m, rwg_m, k_air;
                leaf_lambda=LEAF_LAMBDA, verbose=true)

            t_pre_m = @elapsed P_nf_m = build_nearfield_preconditioner(A_mlfma.Z_near;
                factorization=:ilu, ilu_tau=ILU_TAU)

            nL = A_mlfma.octree.nLevels
            n_leaf = length(A_mlfma.octree.levels[end].boxes)
            nnz_pct = round(nnz(A_mlfma.Z_near) / N_m^2 * 100, digits=1)
            println("    MLFMA: $nL levels, $n_leaf leaf boxes, NF nnz=$(nnz_pct)%, build $(round(t_mlfma, digits=1))s, precond $(round(t_pre_m, digits=1))s")

            t_sol_m = @elapsed begin
                I_mlfma, stats_m = solve_gmres(A_mlfma, v_m;
                    preconditioner=P_nf_m, tol=GMRES_TOL, maxiter=gmres_maxiter_for(N_m))
            end

            G_m = radiation_vectors(m, rwg_m, grid_cuts, k_air)
            E_ff_m = compute_farfield(G_m, Vector{ComplexF64}(I_mlfma), NΩ_cuts)
            σ_m_dB = 10 .* log10.(max.(bistatic_rcs(E_ff_m; E0=1.0), 1e-30))
            bs_m = backscatter_rcs(E_ff_m, grid_cuts, Vec3(0.0, 0.0, -k_air); E0=1.0)

            rmse_str_m = ""
            if N_m <= 15_000
                G_m_c = radiation_vectors(m, rwg_m, grid_coarse, k_air)
                E_ff_m_c = compute_farfield(G_m_c, Vector{ComplexF64}(I_mlfma), NΩ_coarse)
                σ_m_c_dB = 10 .* log10.(max.(bistatic_rcs(E_ff_m_c; E0=1.0), 1e-30))
                rmse_m = sqrt(sum((σ_m_c_dB .- σ_po_coarse_dB).^2) / NΩ_coarse)
                rmse_str_m = "$(round(rmse_m, digits=2)) dB"
            else
                rmse_str_m = "—"
            end

            bs_m_dB = round(10 * log10(max(bs_m.sigma, 1e-30)), digits=2)
            println("  $(rpad(label, 14))  $(lpad(N_m, 5))  $(lpad("mlfma+gmres", 12))  " *
                    "$(lpad(round(t_mlfma+t_pre_m, digits=1), 6))  $(lpad(round(t_sol_m, digits=1), 6))  " *
                    "$(lpad(stats_m.niter, 5))  $(lpad(bs_m_dB, 10)) dBsm  $rmse_str_m")

            mlabel = "$label MLFMA (N=$N_m)"
            all_curves[mlabel] = σ_m_dB
            push!(curve_order, mlabel)
        end
    end

    # ════════════════════════════════════════════════════
    # Save RCS data to CSV
    # ════════════════════════════════════════════════════
    println("\n── Saving RCS data ──")
    df_rcs = DataFrame(
        theta_deg = rad2deg.(grid_cuts.theta),
        phi_deg   = rad2deg.(grid_cuts.phi),
    )
    for label in curve_order
        haskey(all_curves, label) || continue
        col_name = replace(label, " " => "_", "(" => "", ")" => "", "=" => "")
        df_rcs[!, Symbol(col_name)] = all_curves[label]
    end
    csv_path = joinpath(datadir, "09a_aircraft_rcs.csv")
    CSV.write(csv_path, df_rcs)
    println("  Saved: $csv_path ($(nrow(df_rcs)) rows, $(ncol(df_rcs)) cols)")

    # ════════════════════════════════════════════════════
    # RCS pattern plots (1° resolution)
    # ════════════════════════════════════════════════════
    println("\n" * "─"^60)
    println("RCS pattern plots (1° resolution)")
    println("─"^60)

    # Extract φ=0 and φ=90 cut indices from the cut grid
    phi0_idx = [q for q in 1:NΩ_cuts if abs(grid_cuts.phi[q]) < 0.01]
    phi90_idx = [q for q in 1:NΩ_cuts if abs(grid_cuts.phi[q] - π/2) < 0.01]
    sort!(phi0_idx; by=q -> grid_cuts.theta[q])
    sort!(phi90_idx; by=q -> grid_cuts.theta[q])
    θ_deg_0  = [rad2deg(grid_cuts.theta[q]) for q in phi0_idx]
    θ_deg_90 = [rad2deg(grid_cuts.theta[q]) for q in phi90_idx]

    println("  φ=0° cut: $(length(phi0_idx)) points")
    println("  φ=90° cut: $(length(phi90_idx)) points")

    # Color/dash scheme: PO=black solid, MLFMA=red solid
    mesh_colors = ["#d62728"]  # red for 1λ mesh
    style_map = Dict{String, NamedTuple{(:color, :width, :dash), Tuple{String, Int, String}}}()
    style_map["PO"] = (color="black", width=3, dash="solid")
    for (i, (_, label)) in enumerate(mesh_levels)
        N_m = build_rwg(mesh_levels[i][1]).nedges
        mlabel = "$label MLFMA (N=$N_m)"
        style_map[mlabel] = (color=mesh_colors[min(i, length(mesh_colors))], width=2, dash="solid")
    end

    function make_rcs_plot(cut_idx, θ_deg, title_str)
        sf = subplots(1, 1; sync=false, width=950, height=600,
                      subplot_titles=reshape([title_str], 1, 1))
        for label in curve_order
            haskey(all_curves, label) || continue
            dB = all_curves[label]
            s = get(style_map, label, (color="gray", width=1, dash="solid"))
            addtraces!(sf,
                scatter(x=θ_deg, y=[dB[q] for q in cut_idx],
                        mode="lines", name=label,
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
        "Aircraft RCS — φ=0° ($(freq_air/1e9) GHz, λ=$(round(λ_air, digits=2))m, 1° resolution)")
    savefig(p0, joinpath(figdir, "09a_aircraft_rcs_phi0.png"))
    println("Plot saved: 09a_aircraft_rcs_phi0.png")

    p90 = make_rcs_plot(phi90_idx, θ_deg_90,
        "Aircraft RCS — φ=90° ($(freq_air/1e9) GHz, λ=$(round(λ_air, digits=2))m, 1° resolution)")
    savefig(p90, joinpath(figdir, "09a_aircraft_rcs_phi90.png"))
    println("Plot saved: 09a_aircraft_rcs_phi90.png")
end

println("\n" * "="^60)
println("Done.")
