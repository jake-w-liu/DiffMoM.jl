# validate_po_vs_pofacets.jl — Self-contained PO validation
#
# Implements POFacets 4.5 bistatic RCS algorithm (facetRCS.m) directly in Julia,
# then compares against DiffMoM.jl's solve_po on the SAME mesh.
#
# No MATLAB required — the POFacets algorithm is faithfully transliterated.
#
# Run: julia --project=. validation/po/validate_po_vs_pofacets.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
include(joinpath(@__DIR__, "..", "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra

# ═══════════════════════════════════════════════════════════
# POFacets 4.5 Reference Implementation (transliterated)
# ═══════════════════════════════════════════════════════════

"""
POFacets G(n,w) recursive helper (G.m).
"""
function pof_G(n::Int, w::ComplexF64)
    jw = 1im * w
    g = (exp(jw) - 1) / jw
    for m in 1:n
        go = g
        g = (exp(jw) - m * go) / jw
    end
    return g
end

"""
    pofacets_bistatic_rcs(mesh, freq_hz, itheta_deg, iphi_deg, i_pol,
                          theta_obs, phi_obs; Lt=1e-5, Nt=5)

Compute bistatic RCS using POFacets 4.5 algorithm (facetRCS.m).

Convention: (itheta, iphi) is the SOURCE direction in standard spherical coords.
The wave propagates FROM (itheta,iphi) TOWARD the origin, i.e. k̂_prop = -D0i.

Returns (sigma_theta, sigma_phi) in m² at each observation angle.
"""
function pofacets_bistatic_rcs(mesh::TriMesh, freq_hz::Float64,
                               itheta_deg::Float64, iphi_deg::Float64, i_pol::Int,
                               theta_obs::Vector{Float64}, phi_obs::Vector{Float64};
                               Lt::Float64=1e-5, Nt_terms::Int=5)
    c0 = 299792458.0
    wave = c0 / freq_hz
    bk = 2π / wave

    rad = π / 180
    ithetar = itheta_deg * rad
    iphir = iphi_deg * rad

    # Incidence direction cosines
    cpi = cos(iphir); spi = sin(iphir)
    sti = sin(ithetar); cti = cos(ithetar)
    ui = sti * cpi; vi = sti * spi; wi = cti
    uui = cti * cpi; vvi = cti * spi; wwi = -sti

    # Incident field polarization (global Cartesian)
    if i_pol == 1  # θ-polarized
        Et = 1.0 + 0im; Ep = 0.0 + 0im
    else           # φ-polarized
        Et = 0.0 + 0im; Ep = 1.0 + 0im
    end
    e0 = [uui * Et - spi * Ep, vvi * Et + cpi * Ep, wwi * Et]

    ntria = ntriangles(mesh)

    # Build per-triangle data: normals, areas, rotation angles
    N_vec = Vector{Vector{Float64}}(undef, ntria)
    Area = zeros(ntria)
    alpha = zeros(ntria)
    beta = zeros(ntria)

    for i in 1:ntria
        v1 = _get_vertex(mesh, i, 1)
        v2 = _get_vertex(mesh, i, 2)
        v3 = _get_vertex(mesh, i, 3)
        A_edge = v2 - v1
        B_edge = v3 - v2
        nvec = -cross(B_edge, A_edge)  # same as cross(v2-v1, v3-v1)
        Nn = norm(nvec)
        if Nn == 0
            N_vec[i] = [0.0, 0.0, 1.0]
        else
            N_vec[i] = nvec / Nn
        end
        d1 = norm(A_edge)
        d2 = norm(B_edge)
        d3 = norm(v1 - v3)
        ss = 0.5 * (d1 + d2 + d3)
        Area[i] = sqrt(max(ss * (ss - d1) * (ss - d2) * (ss - d3), 0.0))
        beta[i] = acos(clamp(N_vec[i][3], -1.0, 1.0))
        alpha[i] = atan(N_vec[i][2], N_vec[i][1])
    end

    # PEC: Rs=0, ilum=1 for all, rsmethod=1
    Rs = 0.0
    Co = 1.0  # wave amplitude at vertices
    Ri = [ui, vi, wi]  # incident direction for illumination test

    Nobs = length(theta_obs)
    sigma_t = zeros(Nobs)
    sigma_p = zeros(Nobs)
    n_illum = 0

    for iobs in 1:Nobs
        thr = theta_obs[iobs]
        phr = phi_obs[iobs]

        st = sin(thr); ct = cos(thr)
        cp = cos(phr); sp = sin(phr)
        u = st * cp; v = st * sp; w = ct
        uu = ct * cp; vv = ct * sp; ww = -st

        sumt = 0.0 + 0im
        sump = 0.0 + 0im

        for m in 1:ntria
            # Illumination test: n̂ · Ri ≥ 0
            nidotk = dot(N_vec[m], Ri)
            if nidotk < 0
                continue
            end
            if iobs == 1
                n_illum += 1
            end

            # Local coordinate transformation matrices
            ca = cos(alpha[m]); sa = sin(alpha[m])
            cb = cos(beta[m]); sb = sin(beta[m])
            T1 = [ca sa 0; -sa ca 0; 0 0 1]
            T2 = [cb 0 -sb; 0 1 0; sb 0 cb]

            # Transform incidence to local
            D0i = [ui, vi, wi]
            D2i = T2 * (T1 * D0i)
            ui2 = D2i[1]; vi2 = D2i[2]; wi2 = D2i[3]
            sti2 = sqrt(ui2^2 + vi2^2) * sign(wi2)
            cti2 = sqrt(1 - sti2^2)
            iphi2 = atan(vi2, ui2 + 1e-10)
            cpi2 = cos(iphi2); spi2 = sin(iphi2)

            # Transform observation to local
            D0 = [u, v, w]
            D2 = T2 * (T1 * D0)
            u2 = D2[1]; v2 = D2[2]; w2 = D2[3]
            st2 = sqrt(u2^2 + v2^2) * sign(w2)
            ct2 = sqrt(1 - st2^2)
            phi2 = atan(v2, u2 + 1e-10)
            cp2 = cos(phi2); sp2 = sin(phi2)

            # Incident field in local coordinates
            e2 = T2 * (T1 * e0)
            Et2 = e2[1] * cti2 * cpi2 + e2[2] * cti2 * spi2 - e2[3] * sti2
            Ep2 = -e2[1] * spi2 + e2[2] * cpi2

            # PEC reflection coefficients (Rs=0)
            # Note: the original POFacets formula para = -cti2/(2Rs+cti2) has a 0/0
            # singularity at grazing incidence (cti2=0) for PEC (Rs=0). The correct
            # limit is -1, so we handle that case explicitly.
            perp = -1.0 / (2 * Rs * cti2 + 1)  # = -1 for PEC
            denom_para = 2 * Rs + cti2
            if denom_para != 0
                para = -cti2 / denom_para
            elseif Rs == 0  # PEC grazing: lim_{cti2→0} -cti2/cti2 = -1
                para = -1.0
            else
                para = 0.0
            end

            # Surface current in local Cartesian
            Jx2 = (-Et2 * cpi2 * para + Ep2 * spi2 * perp * cti2)
            Jy2 = (-Et2 * spi2 * para - Ep2 * cpi2 * perp * cti2)

            # Phase terms at triangle vertices
            v1 = _get_vertex(mesh, m, 1)
            v2v = _get_vertex(mesh, m, 2)
            v3 = _get_vertex(mesh, m, 3)
            x_v = [v1[1], v2v[1], v3[1]]
            y_v = [v1[2], v2v[2], v3[2]]
            z_v = [v1[3], v2v[3], v3[3]]

            Dp = bk * ((x_v[1] - x_v[3]) * (u + ui) +
                       (y_v[1] - y_v[3]) * (v + vi) +
                       (z_v[1] - z_v[3]) * (w + wi))
            Dq = bk * ((x_v[2] - x_v[3]) * (u + ui) +
                       (y_v[2] - y_v[3]) * (v + vi) +
                       (z_v[2] - z_v[3]) * (w + wi))
            Do = bk * (x_v[3] * (u + ui) + y_v[3] * (v + vi) + z_v[3] * (w + wi))

            # Analytical phase integral with Taylor-series special cases
            DD = Dq - Dp
            expDo = exp(1im * Do)
            expDp = exp(1im * Dp)
            expDq = exp(1im * Dq)

            Ic::ComplexF64 = 0.0 + 0im
            if abs(Dp) < Lt && abs(Dq) >= Lt
                sic = 0.0 + 0im
                for n in 0:Nt_terms
                    sic += (1im * Dp)^n / factorial(n) *
                           (-Co / (n + 1) + expDq * (Co * pof_G(n, ComplexF64(-Dq))))
                end
                Ic = sic * 2 * Area[m] * expDo / 1im / Dq
            elseif abs(Dp) < Lt && abs(Dq) < Lt
                sic = 0.0 + 0im
                for n in 0:Nt_terms
                    for nn in 0:Nt_terms
                        sic += (1im * Dp)^n * (1im * Dq)^nn /
                               factorial(nn + n + 2) * Co
                    end
                end
                Ic = sic * 2 * Area[m] * expDo
            elseif abs(Dp) >= Lt && abs(Dq) < Lt
                sic = 0.0 + 0im
                for n in 0:Nt_terms
                    sic += (1im * Dq)^n / factorial(n) *
                           Co * pof_G(n + 1, ComplexF64(-Dp)) / (n + 1)
                end
                Ic = sic * 2 * Area[m] * expDo * expDp
            elseif abs(Dp) >= Lt && abs(Dq) >= Lt && abs(DD) < Lt
                sic = 0.0 + 0im
                for n in 0:Nt_terms
                    sic += (1im * DD)^n / factorial(n) *
                           (-Co * pof_G(n, ComplexF64(Dq)) + expDq * Co / (n + 1))
                end
                Ic = sic * 2 * Area[m] * expDo / 1im / Dq
            else
                Ic = 2 * Area[m] * expDo *
                     (expDp * Co / Dp / DD - expDq * Co / Dq / DD - Co / Dp / Dq)
            end

            # Scattered field in local coordinates
            Es2 = [Jx2 * Ic, Jy2 * Ic, 0.0 + 0im]

            # Transform back to global
            Es1 = T2' * Es2
            Es0 = T1' * Es1

            # Project to spherical θ̂, φ̂ components
            Ets = uu * Es0[1] + vv * Es0[2] + ww * Es0[3]
            Eps = -sp * Es0[1] + cp * Es0[2]

            sumt += Ets
            sump += Eps
        end  # triangle loop

        sigma_t[iobs] = 4π * abs2(sumt) / wave^2
        sigma_p[iobs] = 4π * abs2(sump) / wave^2
    end  # observation loop

    return sigma_t, sigma_p, n_illum
end

# Helper to get mesh vertices matching POFacets indexing
function _get_vertex(mesh::TriMesh, tri_idx::Int, local_idx::Int)
    vi = mesh.tri[local_idx, tri_idx]
    return [mesh.xyz[1, vi], mesh.xyz[2, vi], mesh.xyz[3, vi]]
end

# ═══════════════════════════════════════════════════════════
# Far-field grid matching DiffMoM.jl convention
# ═══════════════════════════════════════════════════════════
function make_cut_grid_1deg(phi_values::Vector{Float64})
    Ntheta = 180
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

# ═══════════════════════════════════════════════════════════
# Main validation
# ═══════════════════════════════════════════════════════════
println("="^65)
println("PO Validation: DiffMoM.jl vs POFacets 4.5 algorithm")
println("="^65)

c0 = 299792458.0

# ─── Load mesh ───
obj_path = joinpath(@__DIR__, "..", "..", "examples", "demo_aircraft.obj")
if !isfile(obj_path)
    error("demo_aircraft.obj not found at $obj_path")
end
mesh_raw = read_obj_mesh(obj_path)
rep = repair_mesh_for_simulation(mesh_raw; allow_boundary=true, auto_drop_nonmanifold=true)
mesh = rep.mesh

freq = 0.3e9
λ0 = c0 / freq
k = 2π / λ0

println("\nMesh: $(nvertices(mesh)) verts, $(ntriangles(mesh)) tri")
println("Frequency: $(freq/1e9) GHz, λ = $(round(λ0, digits=2)) m")

# ─── Observation grid: 1° at φ=0° and φ=90° ───
grid = make_cut_grid_1deg([0.0, π/2])
NΩ = length(grid.w)

phi0_idx  = [q for q in 1:NΩ if abs(grid.phi[q]) < 0.01]
phi90_idx = [q for q in 1:NΩ if abs(grid.phi[q] - π/2) < 0.01]
sort!(phi0_idx;  by=q -> grid.theta[q])
sort!(phi90_idx; by=q -> grid.theta[q])

# ═══════════════════════════════════════════════════════════
# Wave direction: propagation in -z (from +z toward -z)
#
# DiffMoM.jl: k_vec = [0,0,-k], pol = [1,0,0]
# POFacets equiv:       θ_i=0°, φ_i=0°, θ-pol → e0=[1,0,0], D0i=[0,0,1]
#                       k̂_prop = -D0i = [0,0,-1]
# ═══════════════════════════════════════════════════════════

println("\n" * "─"^65)
println("Test case: wave propagating -z, x-polarized")
println("  Julia:    k_vec=[0,0,-k], pol=[1,0,0]")
println("  POFacets: θ_i=0°, φ_i=0°, θ-pol → e0=[1,0,0]")
println("─"^65)

# ─── 1. Julia PO (DiffMoM.jl) ───
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
t_julia = @elapsed po = solve_po(mesh, freq, pw; grid=grid)
σ_julia = bistatic_rcs(po.E_ff; E0=1.0)
σ_julia_dB = 10 .* log10.(max.(σ_julia, 1e-30))

println("\nJulia PO:")
println("  Illuminated: $(count(po.illuminated)) / $(ntriangles(mesh))")
println("  Time: $(round(t_julia, digits=3))s")

# ─── 2. POFacets reference (Julia transliteration) ───
t_pof = @elapsed begin
    σ_t_pof, σ_p_pof, n_illum_pof = pofacets_bistatic_rcs(
        mesh, Float64(freq), 0.0, 0.0, 1,    # θ_i=0°, φ_i=0°, θ-pol
        collect(grid.theta), collect(grid.phi))
end
σ_pof_total = σ_t_pof .+ σ_p_pof
σ_pof_dB = 10 .* log10.(max.(σ_pof_total, 1e-30))

println("\nPOFacets 4.5 (Julia transliteration):")
println("  Illuminated: $n_illum_pof / $(ntriangles(mesh))")
println("  Time: $(round(t_pof, digits=3))s")

# ─── 3. Compare ───
println("\n" * "─"^65)
println("Comparison")
println("─"^65)

# Illumination match
# Allow ±1 tolerance for grazing facets (n̂·k̂ ≈ 0) that contribute negligible power
illum_diff = abs(count(po.illuminated) - n_illum_pof)
illum_match = illum_diff <= 1
println("\nIlluminated facets: Julia=$(count(po.illuminated)), POFacets=$n_illum_pof → $(illum_match ? "MATCH ✓ (±$illum_diff)" : "MISMATCH ✗ (Δ=$illum_diff)")")

# Backscatter (θ ≈ 0°, propagation is -z so backscatter toward +z)
bs_julia = backscatter_rcs(po.E_ff, grid, Vec3(0.0, 0.0, -k); E0=1.0)
bs_julia_dB = 10 * log10(max(bs_julia.sigma, 1e-30))

# POFacets backscatter: find closest to θ=0°
bs_pof_idx = argmin(abs.(grid.theta))
bs_pof_dB = 10 * log10(max(σ_pof_total[bs_pof_idx], 1e-30))

println("\nBackscatter RCS:")
println("  Julia:    $(round(bs_julia_dB, digits=2)) dBsm")
println("  POFacets: $(round(bs_pof_dB, digits=2)) dBsm")
println("  Diff:     $(round(abs(bs_julia_dB - bs_pof_dB), digits=3)) dB")

# RMSE per cut
function rmse_cut(idx, label)
    dB_j = σ_julia_dB[idx]
    dB_p = σ_pof_dB[idx]
    # Clamp both to same dynamic range before RMSE
    floor_dB = -60.0
    dB_j = max.(dB_j, floor_dB)
    dB_p = max.(dB_p, floor_dB)
    err = dB_j .- dB_p
    rmse = sqrt(sum(err.^2) / length(idx))
    max_err = maximum(abs.(err))
    println("  $label: RMSE = $(round(rmse, digits=3)) dB, max|err| = $(round(max_err, digits=3)) dB")
    return rmse
end

println("\nPattern RMSE (dB, clamped to -60 dBsm floor):")
rmse_0 = rmse_cut(phi0_idx, "φ=0° ")
rmse_90 = rmse_cut(phi90_idx, "φ=90°")

# Overall assessment
overall_pass = rmse_0 < 1.0 && rmse_90 < 1.0 && illum_match
println("\n" * "═"^65)
if overall_pass
    println("PASS: Julia PO matches POFacets 4.5 (RMSE < 1 dB)")
else
    println("FAIL: Significant mismatch detected")
    if !illum_match
        println("  → Illumination count mismatch")
    end
    if rmse_0 >= 1.0
        println("  → φ=0° RMSE = $(round(rmse_0, digits=2)) dB (threshold: 1 dB)")
    end
    if rmse_90 >= 1.0
        println("  → φ=90° RMSE = $(round(rmse_90, digits=2)) dB (threshold: 1 dB)")
    end
end
println("═"^65)

# ─── Save comparison CSV for plotting ───
using DelimitedFiles

out_dir = joinpath(@__DIR__, "data")
mkpath(out_dir)

θ_deg_0 = rad2deg.(grid.theta[phi0_idx])
data_0 = hcat(θ_deg_0, σ_julia_dB[phi0_idx], σ_pof_dB[phi0_idx])
open(joinpath(out_dir, "po_validation_phi0.csv"), "w") do f
    println(f, "theta_deg,julia_dBsm,pofacets_dBsm")
    for row in eachrow(data_0)
        println(f, "$(row[1]),$(row[2]),$(row[3])")
    end
end

θ_deg_90 = rad2deg.(grid.theta[phi90_idx])
data_90 = hcat(θ_deg_90, σ_julia_dB[phi90_idx], σ_pof_dB[phi90_idx])
open(joinpath(out_dir, "po_validation_phi90.csv"), "w") do f
    println(f, "theta_deg,julia_dBsm,pofacets_dBsm")
    for row in eachrow(data_90)
        println(f, "$(row[1]),$(row[2]),$(row[3])")
    end
end

println("\nSaved: validation/data/po_validation_phi0.csv")
println("Saved: validation/data/po_validation_phi90.csv")
