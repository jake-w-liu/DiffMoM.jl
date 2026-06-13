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
Return `true` when the assembled periodic correction `Z_corr` is provably
symmetric (`Z_corr[m,n] == Z_corr[n,m]`), so the assembly may compute only the
upper triangle and mirror it (matching `assemble_Z_efie`).

Symmetry requires BOTH:

1. Zero Bloch phase (`kx_bloch == ky_bloch == 0`): the quasi-periodic correction
   is then reciprocal, `ΔG(a,b) == ΔG(b,a)`, so the kernel is symmetric.
2. Real RWG coefficients: with real `f_m`/`∇·f_m` the entry kernel
   `dot(f_m,f_n) - conj(∇·f_m)(∇·f_n)/k²` is symmetric under `m↔n`.

Both are needed independently: a Bloch-paired RWG carrying complex coefficients
combined with a zero-phase lattice (a mismatched build) yields a reciprocal `ΔG`
but a *non*-symmetric `Z_corr`. Checking the coefficients directly guards that
case rather than trusting `kx_bloch/ky_bloch` alone. Boundary-paired RWGs built
at normal incidence have phase `exp(0)=1`, so their coefficients are real and the
fast path still applies.
"""
function _periodic_correction_is_symmetric(rwg::RWGData, lattice::PeriodicLattice)
    (iszero(lattice.kx_bloch) && iszero(lattice.ky_bloch)) || return false
    all(iszero, imag.(rwg.coeff_plus)) || return false
    all(iszero, imag.(rwg.coeff_minus)) || return false
    return true
end

"""
Assemble the periodic correction matrix using ΔG = G_per - G_0.
Since ΔG is smooth everywhere, standard product quadrature is used for all entries.

Memory: each source/observation triangle pair `(ts, tn)` is streamed through a
single reused `Nq×Nq` ΔG block (`O(Nq²)`, independent of `Nt`) and scattered into
per-task `N×N` accumulators that are reduced at the end. This avoids the dense
`O(Nq²·Nt²)` ΔG cache the entry-wise assembly previously held resident, while
evaluating each ΔG triangle-pair block exactly once (Ewald sums are unchanged).

Symmetry: when `_periodic_correction_is_symmetric` holds, only target triangles
`tn ≥ ts` are evaluated and each block contribution is mirrored into the
transposed `(n,m)` entry, halving both the Ewald work and the scatter — matching
the symmetry exploit in `assemble_Z_efie`. Otherwise the full `tn` sweep is used.
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

    # Triangle → incident RWG map: for each triangle `t`, the list of
    # (rwg index, slot) pairs that use `t` (slot 1 = T⁺, slot 2 = T⁻). At most
    # three RWG functions touch a triangle on each side, so this is tiny.
    tri_to_rwg = [Vector{Tuple{Int,Int}}() for _ in 1:Nt]
    for n in 1:N
        push!(tri_to_rwg[tri_ids[1, n]], (n, 1))
        push!(tri_to_rwg[tri_ids[2, n]], (n, 2))
    end

    CT = ComplexF64
    inv_k2 = 1 / (k^2)
    symmetric = _periodic_correction_is_symmetric(rwg, lattice)

    # Partition the source triangles across tasks; each task owns a private N×N
    # accumulator so the scatter sweep is lock-free without an O(Nq²·Nt²)
    # resident ΔG cache. `@spawn` tasks may migrate between threads, so each
    # buffer is bound to its chunk index `c` (not to `threadid()`, which is
    # unsafe under migration). Chunks are interleaved (strided) to balance the
    # triangular workload in the symmetric short sweep.
    ntasks = max(1, min(Threads.nthreads(), Nt))
    Z_bufs = [zeros(CT, N, N) for _ in 1:ntasks]

    @sync for c in 1:ntasks
        Threads.@spawn begin
            Zb = Z_bufs[c]
            # Streaming ΔG slab for the current source triangle: slab[qm, qn].
            slab = Matrix{CT}(undef, Nq, Nq)
            for ts in c:ntasks:Nt
                incident_s = tri_to_rwg[ts]
                isempty(incident_s) && continue
                Am = areas[ts]

                tn_start = symmetric ? ts : 1
                @inbounds for tn in tn_start:Nt
                    incident_t = tri_to_rwg[tn]
                    isempty(incident_t) && continue
                    An = areas[tn]
                    for qn in 1:Nq, qm in 1:Nq
                        slab[qm, qn] =
                            greens_periodic_correction(quad_pts[ts][qm], quad_pts[tn][qn], k, lattice)
                    end

                    wAA = (2 * Am) * (2 * An)
                    for (m_idx, itm) in incident_s
                        dvm = div_vals[itm, m_idx]
                        fm_vals = itm == 1 ? rwg_vals[m_idx][1] : rwg_vals[m_idx][2]
                        conj_dvm_ik2 = conj(dvm) * inv_k2

                        for (n_idx, itn) in incident_t
                            dvn = div_vals[itn, n_idx]
                            fn_vals = itn == 1 ? rwg_vals[n_idx][1] : rwg_vals[n_idx][2]
                            dvmn_inv_k2 = conj_dvm_ik2 * dvn

                            val = zero(CT)
                            for qm in 1:Nq
                                fm = fm_vals[qm]
                                wqm = wq[qm]
                                for qn in 1:Nq
                                    fn = fn_vals[qn]
                                    dG = slab[qm, qn]
                                    vec_part = dot(fm, fn) * dG
                                    scl_part = dvmn_inv_k2 * dG
                                    val += (vec_part - scl_part) * (wqm * wq[qn])
                                end
                            end
                            val *= wAA

                            Zb[m_idx, n_idx] += val
                            if symmetric && tn != ts
                                # Reciprocity + real coefficients: the (tn,ts)
                                # block contribution to Z_corr[n,m] equals this
                                # block's. Mirror it so only tn ≥ ts source pairs
                                # are evaluated.
                                Zb[n_idx, m_idx] += val
                            end
                        end
                    end
                end
            end
        end
    end

    Z_corr = Z_bufs[1]
    @inbounds for b in 2:ntasks
        Z_corr .+= Z_bufs[b]
    end
    Z_corr .*= -1im * omega_mu0

    return Z_corr
end
