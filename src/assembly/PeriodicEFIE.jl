# PeriodicEFIE.jl — Periodic EFIE assembly via image correction
#
# Assembles Z_per = Z_free + Z_correction, where:
#   Z_free      = standard free-space EFIE (existing code, handles singularity)
#   Z_correction = contribution from periodic images (m,n) ≠ (0,0)
#
# The correction uses greens_periodic_correction() which is smooth (no 1/R
# singularity), so standard product quadrature suffices for all entries.

export assemble_Z_efie_periodic

function _assert_coplanar_periodic_mesh(mesh::TriMesh; atol::Float64=1e-12)
    zvals = @view mesh.xyz[3, :]
    zmin = minimum(zvals)
    zmax = maximum(zvals)
    if abs(zmax - zmin) > atol
        throw(ArgumentError(
            "PeriodicEFIE currently supports coplanar unit-cell meshes only " *
            "(max z spread <= $(atol)). Got z spread=$(abs(zmax - zmin))."
        ))
    end
end

function _mesh_has_unitcell_boundary_edges(mesh::TriMesh, lattice::PeriodicLattice;
                                           atol_abs::Float64=1e-12,
                                           atol_rel::Float64=1e-9)
    tol = max(atol_abs, atol_rel * max(lattice.dx, lattice.dy))
    xmin = -0.5 * lattice.dx
    xmax =  0.5 * lattice.dx
    ymin = -0.5 * lattice.dy
    ymax =  0.5 * lattice.dy

    edge_counts = Dict{Tuple{Int,Int}, Int}()
    Nt = ntriangles(mesh)
    for t in 1:Nt
        for le in 1:3
            v1 = mesh.tri[le, t]
            v2 = mesh.tri[mod1(le + 1, 3), t]
            key = v1 < v2 ? (v1, v2) : (v2, v1)
            edge_counts[key] = get(edge_counts, key, 0) + 1
        end
    end

    for ((va, vb), count) in edge_counts
        count == 1 || continue  # boundary edge in the provided unit-cell mesh

        xa = mesh.xyz[1, va]; xb = mesh.xyz[1, vb]
        ya = mesh.xyz[2, va]; yb = mesh.xyz[2, vb]

        on_xmin = abs(xa - xmin) <= tol && abs(xb - xmin) <= tol
        on_xmax = abs(xa - xmax) <= tol && abs(xb - xmax) <= tol
        on_ymin = abs(ya - ymin) <= tol && abs(yb - ymin) <= tol
        on_ymax = abs(ya - ymax) <= tol && abs(yb - ymax) <= tol

        if on_xmin || on_xmax || on_ymin || on_ymax
            return true
        end
    end

    return false
end

function _assert_boundary_touching_periodic_mesh_requires_bloch(mesh::TriMesh,
                                                                lattice::PeriodicLattice,
                                                                rwg::Union{Nothing,RWGData}=nothing)
    isnothing(rwg) && return
    rwg.has_periodic_bloch && return
    _mesh_has_unitcell_boundary_edges(mesh, lattice) || return
    throw(ArgumentError(
        "Mesh has conductor boundary edges on the unit-cell boundary, but RWG basis " *
        "does not carry Bloch-periodic boundary pairing. Build RWG with " *
        "`build_rwg_periodic(mesh, lattice; ...)` for boundary-touching periodic cells."
    ))
end

"""
    assemble_Z_efie_periodic(mesh, rwg, k, lattice; quad_order=3, eta0=376.730313668)

Assemble the dense periodic EFIE matrix `Z_per ∈ C^{N×N}` for a unit cell
with 2D periodicity defined by `lattice::PeriodicLattice`.

Strategy:
  Z_per = Z_free + Z_correction

- Z_free: standard free-space EFIE (with singularity extraction for self-cells)
- Z_correction: image sum using ΔG = G_per - G_0 (smooth, no singularity)

Both use the mixed-potential form:
  Z_mn = -iωμ₀ [ ∫∫ f_m·f_n G dS dS' - (1/k²) ∫∫ (∇·f_m)(∇'·f_n) G dS dS' ]
"""
function assemble_Z_efie_periodic(mesh::TriMesh, rwg::RWGData, k,
                                  lattice::PeriodicLattice;
                                  quad_order::Int=3,
                                  eta0::Float64=376.730313668)
    _assert_coplanar_periodic_mesh(mesh)
    _assert_boundary_touching_periodic_mesh_requires_bloch(mesh, lattice, rwg)

    # Step 1: Free-space EFIE (handles self-cell singularity)
    Z_free = assemble_Z_efie(mesh, rwg, k;
                             quad_order=quad_order, eta0=eta0,
                             mesh_precheck=false)

    # Step 2: Periodic image correction (smooth, no singularity)
    Z_corr = _assemble_periodic_correction(mesh, rwg, k, lattice;
                                            quad_order=quad_order, eta0=eta0)

    return Z_free + Z_corr
end

"""
Assemble the periodic correction matrix using ΔG = G_per - G_0.
Since ΔG is smooth everywhere, standard product quadrature is used for all entries.
"""
function _assemble_periodic_correction(mesh::TriMesh, rwg::RWGData, k,
                                       lattice::PeriodicLattice;
                                       quad_order::Int=3,
                                       eta0::Float64=376.730313668)
    N = rwg.nedges
    Nt = ntriangles(mesh)
    Tcoef = promote_type(eltype(rwg.coeff_plus), eltype(rwg.coeff_minus))
    TVec = SVector{3,Tcoef}
    omega_mu0 = k * eta0

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    # Precompute quadrature points and areas
    quad_pts = [tri_quad_points(mesh, t, xi) for t in 1:Nt]
    areas = [triangle_area(mesh, t) for t in 1:Nt]

    # Precompute RWG values and divergences
    tri_ids = zeros(Int, 2, N)
    div_vals = zeros(Tcoef, 2, N)
    rwg_vals = Vector{NTuple{2,Vector{TVec}}}(undef, N)

    for n in 1:N
        tp = rwg.tplus[n]
        tm = rwg.tminus[n]
        tri_ids[1, n] = tp
        tri_ids[2, n] = tm
        div_vals[1, n] = div_rwg(rwg, n, tp)
        div_vals[2, n] = div_rwg(rwg, n, tm)
        vals_p = [eval_rwg(rwg, n, quad_pts[tp][q], tp) for q in 1:Nq]
        vals_m = [eval_rwg(rwg, n, quad_pts[tm][q], tm) for q in 1:Nq]
        rwg_vals[n] = (vals_p, vals_m)
    end

    CT = ComplexF64

    # Precompute ΔG for all unique triangle pairs. ΔG is smooth everywhere
    # (no singularity), so every pair is handled by standard quadrature.
    # Shared triangle pairs across RWG functions are evaluated only once.
    dG_cache = Array{CT, 4}(undef, Nq, Nq, Nt, Nt)
    Threads.@threads for tm in 1:Nt
        @inbounds for tn in 1:Nt
            for qm in 1:Nq, qn in 1:Nq
                dG_cache[qm, qn, tm, tn] =
                    greens_periodic_correction(quad_pts[tm][qm], quad_pts[tn][qn], k, lattice)
            end
        end
    end

    inv_k2 = 1 / (k^2)
    Z_corr = zeros(CT, N, N)

    Threads.@threads for m_idx in 1:N
        @inbounds for n_idx in 1:N
            val = zero(CT)

            for itm in 1:2
                tm = tri_ids[itm, m_idx]
                Am = areas[tm]
                dvm = div_vals[itm, m_idx]
                fm_vals = itm == 1 ? rwg_vals[m_idx][1] : rwg_vals[m_idx][2]

                for itn in 1:2
                    tn = tri_ids[itn, n_idx]
                    An = areas[tn]
                    dvn = div_vals[itn, n_idx]
                    fn_vals = itn == 1 ? rwg_vals[n_idx][1] : rwg_vals[n_idx][2]

                    dvmn_inv_k2 = conj(dvm) * dvn * inv_k2
                    for qm in 1:Nq
                        fm = fm_vals[qm]
                        for qn in 1:Nq
                            fn = fn_vals[qn]
                            dG = dG_cache[qm, qn, tm, tn]
                            vec_part = dot(fm, fn) * dG
                            scl_part = dvmn_inv_k2 * dG
                            weight = wq[qm] * wq[qn] * (2 * Am) * (2 * An)
                            val += (vec_part - scl_part) * weight
                        end
                    end
                end
            end

            Z_corr[m_idx, n_idx] = -1im * omega_mu0 * val
        end
    end

    return Z_corr
end
