# GroundedEFIE.jl — periodic EFIE for a metasurface at height h above a PEC ground plane.
#
# Image theory for horizontal (coplanar) electric currents above an infinite PEC
# ground: the image of a horizontal current J at height h is -J at depth -h, and its
# associated charge images with the same -1 factor. Hence both the vector- and
# scalar-potential kernels acquire the image with a single -1, so the grounded EFIE
# is obtained from the free-standing one by replacing the scalar Green's function
#
#     G_per(Δρ, 0)  ->  G_grounded(Δρ) = G_per(Δρ, 0) - G_per(Δρ, 2h)
#
# in both the f·f and (∇·f)(∇·f) integrals. The image block uses the full periodic
# Green's function at vertical separation 2h (smooth, no singularity).
#
#   Z_grounded = Z_direct - Z_image
#
# Z_direct is the existing coplanar periodic EFIE; Z_image is assembled below.

export assemble_Z_efie_grounded, assemble_excitation_grounded
export reflection_coefficients_grounded, reflection_coefficient_vectors_grounded

# Full periodic Green's function G_per = G_0 + ΔG between two points (no singularity
# extraction; valid only for non-coincident points, which holds for the image block).
@inline function _gper_full(r::SVector{3}, rp::SVector{3}, k, lattice::PeriodicLattice)
    R = norm(r - rp)
    g0 = exp(-im * k * R) / (4π * R)
    return g0 + greens_periodic_correction(r, rp, k, lattice)
end

# Mixed-potential EFIE block between the real layer (mesh, z = z0) and its mirror
# image at z = z0 - two_h, using the full periodic Green's function at Δz = two_h.
function _assemble_periodic_image_block(mesh::TriMesh, rwg::RWGData, k,
                                        lattice::PeriodicLattice, two_h::Float64;
                                        quad_order::Int=3,
                                        eta0::Float64=376.730313668)
    N = rwg.nedges
    Nt = ntriangles(mesh)
    Tcoef = promote_type(eltype(rwg.coeff_plus), eltype(rwg.coeff_minus))
    TVec = SVector{3,Tcoef}
    omega_mu0 = k * eta0

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    quad_pts = [tri_quad_points(mesh, t, xi) for t in 1:Nt]
    # Image source quadrature points: same in-plane geometry, shifted down by 2h.
    shift = SVector(0.0, 0.0, two_h)
    quad_pts_img = [[quad_pts[t][q] - shift for q in 1:Nq] for t in 1:Nt]
    areas = [triangle_area(mesh, t) for t in 1:Nt]

    tri_ids = zeros(Int, 2, N)
    div_vals = zeros(Tcoef, 2, N)
    rwg_vals = Vector{NTuple{2,Vector{TVec}}}(undef, N)
    for n in 1:N
        tp = rwg.tplus[n]; tm = rwg.tminus[n]
        tri_ids[1, n] = tp; tri_ids[2, n] = tm
        div_vals[1, n] = div_rwg(rwg, n, tp)
        div_vals[2, n] = div_rwg(rwg, n, tm)
        vals_p = [eval_rwg(rwg, n, quad_pts[tp][q], tp) for q in 1:Nq]
        vals_m = [eval_rwg(rwg, n, quad_pts[tm][q], tm) for q in 1:Nq]
        rwg_vals[n] = (vals_p, vals_m)
    end

    CT = ComplexF64
    # G_per_full between observation (real, z0) and image source (z0 - 2h).
    #
    # Reciprocity at normal incidence: _gper_full(a@z0, b@z0-2h) =
    # _gper_full(b@z0, a@z0-2h) (verified to machine precision), so the upper
    # triangle determines the cache; the mirrored (tn,tm) write is disjoint from
    # other threads. Oblique incidence (Bloch phase) uses the full sweep.
    G_cache = Array{CT,4}(undef, Nq, Nq, Nt, Nt)
    if iszero(lattice.kx_bloch) && iszero(lattice.ky_bloch)
        # :dynamic scheduling balances the triangular (uneven) workload.
        Threads.@threads :dynamic for tm in 1:Nt
            @inbounds for tn in tm:Nt
                for qm in 1:Nq, qn in 1:Nq
                    g = _gper_full(quad_pts[tm][qm], quad_pts_img[tn][qn], k, lattice)
                    G_cache[qm, qn, tm, tn] = g
                    G_cache[qn, qm, tn, tm] = g
                end
            end
        end
    else
        Threads.@threads for tm in 1:Nt
            @inbounds for tn in 1:Nt
                for qm in 1:Nq, qn in 1:Nq
                    G_cache[qm, qn, tm, tn] =
                        _gper_full(quad_pts[tm][qm], quad_pts_img[tn][qn], k, lattice)
                end
            end
        end
    end

    inv_k2 = 1 / (k^2)
    Z_img = zeros(CT, N, N)
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
                            G = G_cache[qm, qn, tm, tn]
                            weight = wq[qm] * wq[qn] * (2 * Am) * (2 * An)
                            val += (dot(fm, fn) * G - dvmn_inv_k2 * G) * weight
                        end
                    end
                end
            end
            Z_img[m_idx, n_idx] = -1im * omega_mu0 * val
        end
    end
    return Z_img
end

"""
    assemble_Z_efie_grounded(mesh, rwg, k, lattice; height, quad_order=3)

Periodic EFIE impedance matrix for a coplanar metasurface a distance `height` (h) above
an infinite PEC ground plane, via image theory:

    Z_grounded = Z_direct - Z_image,

with `Z_direct` the free-standing coplanar periodic EFIE and `Z_image` the interaction
with the mirror currents at depth 2h (full periodic Green's function, no singularity).
"""
function assemble_Z_efie_grounded(mesh::TriMesh, rwg::RWGData, k,
                                  lattice::PeriodicLattice; height::Real,
                                  quad_order::Int=3, eta0::Float64=376.730313668)
    height > 0 || throw(ArgumentError("ground-plane height must be positive (got $height)"))
    Z_direct = assemble_Z_efie_periodic(mesh, rwg, k, lattice; quad_order=quad_order)
    Z_image = _assemble_periodic_image_block(mesh, rwg, k, lattice, 2 * Float64(height);
                                             quad_order=quad_order, eta0=eta0)
    return Z_direct - Z_image
end

# Incident vertical wavenumber of the specular order (= k cosθ_inc).
@inline _kz_inc(k, lattice) = sqrt(max(k^2 - lattice.kx_bloch^2 - lattice.ky_bloch^2, 0.0))

"""
    assemble_excitation_grounded(mesh, rwg, pw, k, lattice; height, quad_order=3)

Excitation vector for the grounded problem: the metasurface is illuminated by the
incident plane wave plus its bare-ground reflection. For a TE/normal-incidence plane
wave referenced at the metasurface plane (z=0), the total tangential drive is scaled by
`(1 - exp(-2i kz_inc h))`, where `kz_inc` is the incident vertical wavenumber.
"""
function assemble_excitation_grounded(mesh::TriMesh, rwg::RWGData, pw, k,
                                      lattice::PeriodicLattice; height::Real, quad_order::Int=3)
    v_inc = assemble_excitation(mesh, rwg, pw; quad_order=quad_order)
    factor = 1 - exp(-2im * _kz_inc(k, lattice) * height)
    return factor .* v_inc
end

"""
    reflection_coefficients_grounded(mesh, rwg, I, k, lattice; height, kwargs...)

Floquet reflection coefficients for a metasurface a height `h` above a PEC ground.
Adds the image-current contribution and the bare-ground specular background to the
free-standing per-mode coefficients:

    R_mn^grounded = R_mn^cur (1 - e^{-2i kz_mn h}) - δ_{mn,(0,0)} e^{-2i kz_inc h}.

Limits: an empty cell gives the bare-ground R_00 = -e^{-2i kz_inc h} (|R|=1); a full PEC
sheet at z=0 gives R_00 = -1 for any h.
"""
function reflection_coefficients_grounded(mesh::TriMesh, rwg::RWGData, I, k,
                                          lattice::PeriodicLattice; height::Real, kwargs...)
    modes, R_cur = reflection_coefficients(mesh, rwg, I, k, lattice; kwargs...)
    kzi = _kz_inc(k, lattice)
    h = Float64(height)
    R_g = similar(R_cur)
    for (i, m) in enumerate(modes)
        R_g[i] = R_cur[i] * (1 - exp(-2im * m.kz * h))
        if m.m == 0 && m.n == 0
            R_g[i] -= exp(-2im * kzi * h)
        end
    end
    return modes, R_g
end

"""
    reflection_coefficient_vectors_grounded(mesh, rwg, I, k, lattice; height, kwargs...)

Full vector Floquet reflection amplitudes for a grounded metasurface. This is
the energy-budget counterpart to `reflection_coefficients_grounded`: it retains
both transverse polarizations in every propagating order before applying the
image-current phase factor and bare-ground background.
"""
function reflection_coefficient_vectors_grounded(mesh::TriMesh, rwg::RWGData, I, k,
                                                 lattice::PeriodicLattice; height::Real,
                                                 pol::SVector{3,Float64}=SVector(1.0, 0.0, 0.0),
                                                 kwargs...)
    modes, R_cur = reflection_coefficient_vectors(mesh, rwg, I, k, lattice; kwargs...)
    kzi = _kz_inc(k, lattice)
    h = Float64(height)
    R_g = copy(R_cur)
    for (i, m) in enumerate(modes)
        R_g[i] = R_cur[i] * (1 - exp(-2im * real(m.kz) * h))
        if m.m == 0 && m.n == 0
            pol_mode = _mode_transverse_projection(pol, m, k)
            if !isnothing(pol_mode)
                R_g[i] -= exp(-2im * kzi * h) .* ComplexF64.(pol_mode)
            end
        end
    end
    return modes, R_g
end
