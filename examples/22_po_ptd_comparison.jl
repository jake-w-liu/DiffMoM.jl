# 22_po_ptd_comparison.jl — MoM vs PO vs PO+PTD RCS comparison
#
# Validates the PTD (Physical Theory of Diffraction) implementation by
# comparing MoM, PO-only, and PO+PTD bistatic RCS for flat plates at
# three sizes (2λ, 5λ, 10λ). PTD should improve side-lobe prediction
# over PO-only.
#
# Run: julia --project=. examples/22_po_ptd_comparison.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra
using PlotlySupply

println("="^60)
println("Example 22: MoM vs PO vs PO+PTD")
println("="^60)

c0 = 299792458.0
figdir = joinpath(@__DIR__, "figs")
mkpath(figdir)

"""
    make_cut_grid(Ntheta, phi_values)

Create a SphGrid with fine θ resolution at specific φ planes only.
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

"""
Run MoM, PO, PO+PTD for a plate and return all RCS data.
"""
function run_plate_case(nλ::Int, freq, λ0, k, pw, grid, NΩ)
    L = nλ * λ0
    Ns = 5 * nλ
    mesh = make_rect_plate(L, L, Ns, Ns)
    rwg  = build_rwg(mesh)
    N    = rwg.nedges

    println("\n  Plate: $(nλ)λ × $(nλ)λ, $(Ns)×$(Ns) mesh, $N RWG unknowns")

    # PO
    t_po = @elapsed po = solve_po(mesh, freq, pw; grid=grid)
    println("  PO: $(count(po.illuminated)) illum tri, $(round(t_po, digits=3)) s")

    # PO+PTD
    t_ptd = @elapsed ptd = solve_ptd(mesh, freq, pw; grid=grid)
    println("  PTD: $(length(ptd.edges)) edges, $(round(t_ptd, digits=3)) s")

    # MoM
    t_asm = @elapsed Z = assemble_Z_efie(mesh, rwg, k)
    v = assemble_excitation(mesh, rwg, pw)
    t_sol = @elapsed I_mom = Z \ v
    G_mat = radiation_vectors(mesh, rwg, grid, k)
    E_ff_mom = compute_farfield(G_mat, Vector{ComplexF64}(I_mom), NΩ)
    println("  MoM: asm $(round(t_asm, digits=2)) s, solve $(round(t_sol, digits=2)) s")

    σ_po_dB   = 10 .* log10.(max.(bistatic_rcs(po.E_ff; E0=1.0), 1e-30))
    σ_ptd_dB  = 10 .* log10.(max.(bistatic_rcs(ptd.E_ff; E0=1.0), 1e-30))
    σ_edge_dB = 10 .* log10.(max.(bistatic_rcs(ptd.E_ff_ptd; E0=1.0), 1e-30))
    σ_mom_dB  = 10 .* log10.(max.(bistatic_rcs(E_ff_mom; E0=1.0), 1e-30))

    # RMS error (θ < 90°, σ > -40 dBsm)
    mask = (grid.theta .< π/2) .& (σ_mom_dB .> -40.0)
    rms_po  = count(mask) > 0 ? sqrt(sum((σ_po_dB[mask] .- σ_mom_dB[mask]).^2) / count(mask)) : NaN
    rms_ptd = count(mask) > 0 ? sqrt(sum((σ_ptd_dB[mask] .- σ_mom_dB[mask]).^2) / count(mask)) : NaN
    println("  RMS vs MoM: PO=$(round(rms_po, digits=1)) dB, PO+PTD=$(round(rms_ptd, digits=1)) dB")

    return (; nλ, N, σ_po_dB, σ_ptd_dB, σ_edge_dB, σ_mom_dB, rms_po, rms_ptd)
end

freq = 3e9
λ0   = c0 / freq
k    = 2π / λ0

println("\nFrequency: $(freq/1e9) GHz, λ = $(round(λ0*100, digits=2)) cm")

# Normal incidence from -z, x-polarized
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))

# 1° cut grid (φ=0 only)
grid = make_cut_grid(180, [0.0])
NΩ = length(grid.w)

# ═══════════════════════════════════════════════════════
# Run all three plate sizes
# ═══════════════════════════════════════════════════════
cases = [2, 5, 10]
results = []
for nλ in cases
    println("\n" * "─"^60)
    println("Plate $(nλ)λ × $(nλ)λ")
    println("─"^60)
    push!(results, run_plate_case(nλ, freq, λ0, k, pw, grid, NΩ))
end

# ═══════════════════════════════════════════════════════
# Plot: 3-panel comparison
# ═══════════════════════════════════════════════════════
let
    idx = sortperm(grid.theta)
    θ_deg = rad2deg.(grid.theta[idx])

    titles = ["$(r.nλ)λ × $(r.nλ)λ Plate — φ=0° Cut" for r in results]
    sf = subplots(1, 3; sync=false, width=1800, height=500,
                  subplot_titles=reshape(titles, 1, 3),
                  horizontal_spacing=0.05)

    yranges = [[-50, 15], [-50, 25], [-50, 35]]

    for (col, r) in enumerate(results)
        show_legend = col == 1
        addtraces!(sf, scatter(x=θ_deg, y=r.σ_mom_dB[idx], mode="lines",
                   name="MoM (N=$(r.N))", legendgroup="mom", showlegend=show_legend,
                   line=attr(color="black", width=2.5)); row=1, col=col)
        addtraces!(sf, scatter(x=θ_deg, y=r.σ_po_dB[idx], mode="lines",
                   name="PO", legendgroup="po", showlegend=show_legend,
                   line=attr(color="blue", width=2, dash="dash")); row=1, col=col)
        addtraces!(sf, scatter(x=θ_deg, y=r.σ_ptd_dB[idx], mode="lines",
                   name="PO+PTD", legendgroup="ptd", showlegend=show_legend,
                   line=attr(color="red", width=2)); row=1, col=col)
        addtraces!(sf, scatter(x=θ_deg, y=r.σ_edge_dB[idx], mode="lines",
                   name="PTD edge only", legendgroup="edge", showlegend=show_legend,
                   line=attr(color="orange", width=1.5, dash="dot")); row=1, col=col)
    end

    p = sf.plot
    for (col, yr) in enumerate(yranges)
        xkey = col == 1 ? :xaxis : Symbol("xaxis$(col)")
        ykey = col == 1 ? :yaxis : Symbol("yaxis$(col)")
        relayout!(p; Dict(
            xkey => attr(title="θ (deg)", range=[0, 180]),
            ykey => attr(title= col==1 ? "Bistatic RCS (dBsm)" : "", range=yr),
        )...)
    end
    relayout!(p, legend=attr(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
              margin=attr(l=60, r=20, t=50, b=50))
    savefig(p, joinpath(figdir, "22_ptd_3panel.png"); width=1800, height=500)
    println("\nPlot saved: 22_ptd_3panel.png")
end

# Individual plots per plate size
let
    idx = sortperm(grid.theta)
    θ_deg = rad2deg.(grid.theta[idx])
    yranges = [[-50, 15], [-50, 25], [-50, 35]]

    for (i, r) in enumerate(results)
        sf = subplots(1, 1; sync=false, width=900, height=550,
                      subplot_titles=reshape(["$(r.nλ)λ × $(r.nλ)λ Plate — φ=0° Cut (3 GHz, RMS: PO=$(round(r.rms_po,digits=1))dB, PTD=$(round(r.rms_ptd,digits=1))dB)"], 1, 1))
        addtraces!(sf, scatter(x=θ_deg, y=r.σ_mom_dB[idx], mode="lines",
                   name="MoM (N=$(r.N))", line=attr(color="black", width=2.5)); row=1, col=1)
        addtraces!(sf, scatter(x=θ_deg, y=r.σ_po_dB[idx], mode="lines",
                   name="PO", line=attr(color="blue", width=2, dash="dash")); row=1, col=1)
        addtraces!(sf, scatter(x=θ_deg, y=r.σ_ptd_dB[idx], mode="lines",
                   name="PO+PTD", line=attr(color="red", width=2)); row=1, col=1)
        addtraces!(sf, scatter(x=θ_deg, y=r.σ_edge_dB[idx], mode="lines",
                   name="PTD edge only", line=attr(color="orange", width=1.5, dash="dot")); row=1, col=1)
        p = sf.plot
        relayout!(p, xaxis=attr(title="θ (deg)", range=[0, 180]),
                  yaxis=attr(title="Bistatic RCS (dBsm)", range=yranges[i]),
                  legend=attr(x=0.55, y=0.95), margin=attr(l=60, r=30, t=60, b=50))
        savefig(p, joinpath(figdir, "22_plate_$(r.nλ)lam_rcs.png"); width=900, height=550)
        println("Plot saved: 22_plate_$(r.nλ)lam_rcs.png")
    end
end

# ═══════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════
println("\n" * "="^60)
println("Summary: RMS Error vs MoM (dB)")
println("="^60)
println("  Plate size    PO       PO+PTD   Improvement")
println("  ──────────    ──────   ──────   ───────────")
for r in results
    imp = r.rms_po - r.rms_ptd
    println("  $(rpad("$(r.nλ)λ × $(r.nλ)λ", 12))$(lpad(round(r.rms_po, digits=1), 5))    $(lpad(round(r.rms_ptd, digits=1), 5))    $(lpad(round(imp, digits=1), 5)) dB")
end

println("\nDone.")
