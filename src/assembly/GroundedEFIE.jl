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

@inline _validated_ground_height(height::Real) =
    _positive_periodic_parameter("ground-plane height", height)

# Full periodic Green's function G_per = G_0 + ΔG between two points (no singularity
# extraction; valid only for non-coincident points, which holds for the image block).
@inline function _gper_full(r::SVector{3}, rp::SVector{3}, k, lattice::PeriodicLattice)
    kw = _validated_lattice_wavenumber(k, lattice)
    displacement = r - rp
    R = hypot(hypot(displacement[1], displacement[2]), displacement[3])
    g0 = (_periodic_rwg_bloch_phase(kw, R) / R) / (4π)
    return g0 + greens_periodic_correction(r, rp, kw, lattice)
end

@inline function _gper_full_cached(
        r::SVector{3},
        rp::SVector{3},
        k::Float64,
        lattice::PeriodicLattice,
        spatial_terms::Vector{_PeriodicSpatialEwaldTerm},
        spectral_terms::Vector{_PeriodicSpectralEwaldTerm})
    displacement = r - rp
    R = hypot(hypot(displacement[1], displacement[2]), displacement[3])
    g0 = (_periodic_rwg_bloch_phase(k, R) / R) / (4π)
    return g0 + _greens_periodic_correction_cached(
        r, rp, k, lattice, spatial_terms, spectral_terms)
end

# Mixed-potential EFIE block between the real layer (mesh, z = z0) and its mirror
# image at z = z0 - two_h, using the full periodic Green's function at Δz = two_h.
function _assemble_periodic_image_block(mesh::TriMesh, rwg::RWGData, k,
                                        lattice::PeriodicLattice, two_h::Float64;
                                        quad_order::Int=3,
                                        eta0::Float64=376.730313668,
                                        max_cache_bytes::Integer=
                                            _DEFAULT_MAX_PERIODIC_EFIE_CACHE_BYTES,
                                        max_green_terms::Integer=
                                            _DEFAULT_MAX_PERIODIC_GREEN_TERMS)
    kw = _validated_lattice_wavenumber(k, lattice)
    _positive_periodic_parameter("twice the ground-plane height", two_h)
    N = rwg.nedges
    Nt = ntriangles(mesh)
    _, inv_k2, omega_mu0 = _validated_efie_prefactors(kw, eta0)
    cache, _, _ = _build_periodic_efie_geometry_cache(
        mesh, rwg, lattice;
        quad_order=quad_order,
        point_cache_count=2,
        image_block=true,
        max_cache_bytes=max_cache_bytes,
        max_green_terms=max_green_terms)
    wq = cache.wq
    Nq = cache.Nq
    quad_pts = cache.quad_pts
    areas = cache.areas
    div_vals = cache.div_vals
    rwg_vals = cache.rwg_vals
    incidence = cache.incidence

    # Image source quadrature points: same in-plane geometry, shifted down by 2h.
    shift = SVector(0.0, 0.0, two_h)
    quad_pts_img = [[quad_pts[t][q] - shift for q in 1:Nq] for t in 1:Nt]

    CT = ComplexF64
    symmetric = _periodic_correction_is_symmetric(rwg, lattice)

    # Stream one source/image triangle-pair block at a time. The global result is
    # protected row-wise, while each task retains only one Nq×Nq Green-function
    # slab and the rows incident on its current source triangle. This removes the
    # former O(Nq²·Nt²) resident Green-function cache.
    ntasks = cache.ntasks
    max_incident = incidence.max_incident
    Z_img = zeros(CT, N, N)
    row_locks = [Threads.SpinLock() for _ in 1:N]

    @sync for c in 1:ntasks
        Threads.@spawn begin
            slab = Matrix{CT}(undef, Nq, Nq)
            row_buffer = zeros(CT, max_incident, N)
            for ts in c:ntasks:Nt
                incident_s = _periodic_incidence_range(incidence, ts)
                isempty(incident_s) && continue
                fill!(row_buffer, zero(CT))
                Am = areas[ts]

                tn_start = symmetric ? ts : 1
                @inbounds for tn in tn_start:Nt
                    incident_t = _periodic_incidence_range(incidence, tn)
                    isempty(incident_t) && continue
                    An = areas[tn]

                    for qn in 1:Nq, qm in 1:Nq
                        slab[qm, qn] =
                            _gper_full_cached(
                                quad_pts[ts][qm], quad_pts_img[tn][qn], kw,
                                lattice, cache.spatial_terms,
                                cache.spectral_terms)
                    end

                    wAA = (2 * Am) * (2 * An)
                    for (source_slot, source_idx) in enumerate(incident_s)
                        m_idx, itm = incidence.entries[source_idx]
                        dvm = div_vals[itm, m_idx]
                        fm_vals = itm == 1 ? rwg_vals[m_idx][1] : rwg_vals[m_idx][2]
                        conj_dvm_ik2 = conj(dvm) * inv_k2

                        for target_idx in incident_t
                            n_idx, itn = incidence.entries[target_idx]
                            dvn = div_vals[itn, n_idx]
                            fn_vals = itn == 1 ? rwg_vals[n_idx][1] : rwg_vals[n_idx][2]
                            dvmn_inv_k2 = conj_dvm_ik2 * dvn

                            val = zero(CT)
                            for qm in 1:Nq
                                fm = fm_vals[qm]
                                wqm = wq[qm]
                                for qn in 1:Nq
                                    fn = fn_vals[qn]
                                    G = slab[qm, qn]
                                    val += (dot(fm, fn) * G - dvmn_inv_k2 * G) *
                                           (wqm * wq[qn])
                                end
                            end
                            val *= wAA

                            if symmetric && tn == ts
                                val *= 0.5
                            end
                            row_buffer[source_slot, n_idx] += val
                        end
                    end
                end
                _merge_periodic_triangle_rows!(
                    Z_img, row_locks, row_buffer, incidence, incident_s)
            end
        end
    end

    symmetric && _complete_periodic_triangle_symmetry!(Z_img)
    Z_img .*= -1im * omega_mu0
    all(isfinite, Z_img) ||
        throw(OverflowError(
            "periodic EFIE image block contains entries outside the " *
            "representable ComplexF64 range"))
    return Z_img
end

"""
    assemble_Z_efie_grounded(mesh, rwg, k, lattice; height, quad_order=3,
                             max_work_bytes=2_000_000_000,
                             max_cache_bytes=2_000_000_000,
                             max_adjacency_pairs=20_000_000,
                             max_green_terms=500_000_000)

Periodic EFIE impedance matrix for a coplanar metasurface a distance `height` (h) above
an infinite PEC ground plane, via image theory:

    Z_grounded = Z_direct - Z_image,

with `Z_direct` the free-standing coplanar periodic EFIE and `Z_image` the interaction
with the mirror currents at depth 2h (full periodic Green's function, no singularity).

`max_work_bytes` bounds the three simultaneously resident dense matrices.
`max_cache_bytes` bounds each sequential auxiliary cache,
`max_adjacency_pairs` bounds free-space adjacency construction, and
`max_green_terms` bounds the aggregate direct-correction plus image Ewald work.
"""
function assemble_Z_efie_grounded(mesh::TriMesh, rwg::RWGData, k,
                                  lattice::PeriodicLattice; height::Real,
                                  quad_order::Int=3, eta0::Float64=376.730313668,
                                  max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                                  max_cache_bytes::Integer=
                                      _DEFAULT_MAX_PERIODIC_EFIE_CACHE_BYTES,
                                  max_adjacency_pairs::Integer=
                                      _DEFAULT_MAX_EFIE_ADJACENCY_PAIRS,
                                  max_green_terms::Integer=
                                      _DEFAULT_MAX_PERIODIC_GREEN_TERMS)
    _validate_mesh_rwg_pair(mesh, rwg)
    work_bytes = _checked_array_payload_bytes(
        ComplexF64, 3, rwg.nedges, rwg.nedges;
        label="grounded EFIE dense work matrices")
    work_limit = _validated_resource_limit("max_work_bytes", max_work_bytes)
    _enforce_payload_limit(
        work_bytes, work_limit,
        "grounded EFIE dense work matrices", "max_work_bytes")
    cache_limit = _validated_resource_limit(
        "max_cache_bytes", max_cache_bytes)
    adjacency_limit = _validated_nonnegative_resource_limit(
        "max_adjacency_pairs", max_adjacency_pairs)
    green_limit = _validated_resource_limit(
        "max_green_terms", max_green_terms)
    tri_quad_rule(quad_order)

    kw = _validated_lattice_wavenumber(k, lattice)
    _validated_efie_prefactors(kw, eta0)
    h = _validated_ground_height(height)
    two_h = _positive_periodic_parameter("twice the ground-plane height", 2 * h)

    # Both the direct correction and image block traverse the periodic Green
    # series. Charge their aggregate work before either dense matrix is built.
    _, periodic_weights = tri_quad_rule(quad_order)
    Nq = length(periodic_weights)
    Nt = ntriangles(mesh)
    Tcoef = promote_type(eltype(rwg.coeff_plus), eltype(rwg.coeff_minus))
    TVec = SVector{3,Tcoef}
    ntasks = max(1, min(Threads.nthreads(), Nt))
    minimum_cache_bytes = _periodic_efie_cache_bytes(
        rwg.nedges, Nt, Nq, ntasks, 0, Tcoef, TVec;
        point_cache_count=2)
    _enforce_periodic_efie_cache_limit(minimum_cache_bytes, cache_limit)
    incidence = _build_periodic_triangle_incidence(rwg, Nt)
    spatial_term_count = iszero(incidence.active_triangles) ?
        0 : _periodic_term_count(lattice.N_spatial) - 1
    spectral_term_count = iszero(incidence.active_triangles) ?
        0 : _periodic_term_count(lattice.N_spectral)
    cache_bytes = _periodic_efie_cache_bytes(
        rwg.nedges, Nt, Nq, ntasks, incidence.max_incident, Tcoef, TVec;
        point_cache_count=2,
        spatial_term_count=spatial_term_count,
        spectral_term_count=spectral_term_count)
    _enforce_periodic_efie_cache_limit(cache_bytes, cache_limit)
    symmetric = _periodic_correction_is_symmetric(rwg, lattice)
    direct_terms = _periodic_efie_green_terms(
        incidence.active_triangles, Nq, lattice, symmetric)
    image_terms = _periodic_efie_green_terms(
        incidence.active_triangles, Nq, lattice, symmetric;
        image_block=true)
    total_green_terms_big = BigInt(direct_terms) + image_terms
    total_green_terms_big <= typemax(Int) ||
        throw(ArgumentError(
            "grounded EFIE Green-series work estimate overflows Int"))
    total_green_terms = Int(total_green_terms_big)
    total_green_terms <= green_limit ||
        throw(ArgumentError(
            "grounded EFIE assembly requires $total_green_terms Green-series " *
            "terms, exceeding max_green_terms=$green_limit"))

    Z_direct = assemble_Z_efie_periodic(
        mesh, rwg, kw, lattice;
        quad_order=quad_order, eta0=eta0,
        max_work_bytes=work_limit,
        max_cache_bytes=cache_limit,
        max_adjacency_pairs=adjacency_limit,
        max_green_terms=green_limit)
    Z_image = _assemble_periodic_image_block(mesh, rwg, kw, lattice, two_h;
                                             quad_order=quad_order, eta0=eta0,
                                             max_cache_bytes=cache_limit,
                                             max_green_terms=green_limit)
    Z_grounded = Z_direct - Z_image
    all(isfinite, Z_grounded) ||
        throw(OverflowError(
            "grounded EFIE matrix contains entries outside the " *
            "representable ComplexF64 range"))
    return Z_grounded
end

# Incident vertical wavenumber of the specular order (= k cosθ_inc).
@inline function _kz_inc(k, lattice)
    magnitude, propagating = _periodic_longitudinal_magnitude(
        Float64(k), lattice.kx_bloch, lattice.ky_bloch,
        "incident longitudinal wavevector")
    (propagating || iszero(magnitude)) ||
        throw(ArgumentError(
            "incident Bloch wavevector ($(lattice.kx_bloch), " *
            "$(lattice.ky_bloch)) exceeds k=$k"))
    return magnitude
end

@noinline function _grounded_round_trip_phase_exact(
    vertical_wavenumber::Float64,
    height::Float64,
)
    return setprecision(BigFloat, _PERIODIC_RWG_PHASE_FALLBACK_PRECISION) do
        ComplexF64(cis(
            -2BigFloat(vertical_wavenumber) * BigFloat(height)))
    end
end

@inline function _grounded_round_trip_phase(
    vertical_wavenumber::Float64,
    height::Float64,
)
    product_hi = vertical_wavenumber * height
    if isfinite(product_hi)
        product_lo = fma(vertical_wavenumber, height, -product_hi)
        if isfinite(product_lo)
            reduced = rem2pi(
                2rem2pi(product_hi, RoundNearest) +
                2rem2pi(product_lo, RoundNearest),
                RoundNearest,
            )
            phase = cis(-reduced)
            isfinite(real(phase)) && isfinite(imag(phase)) &&
                return ComplexF64(phase)
        end
    end
    return _grounded_round_trip_phase_exact(vertical_wavenumber, height)
end

@noinline function _grounded_interference_factor_exact(
    vertical_wavenumber::Float64,
    height::Float64,
)
    return setprecision(BigFloat, _PERIODIC_RWG_PHASE_FALLBACK_PRECISION) do
        ComplexF64(-expm1(Complex{BigFloat}(
            0,
            -2BigFloat(vertical_wavenumber) * BigFloat(height),
        )))
    end
end

@inline function _grounded_interference_factor(
    vertical_wavenumber::Float64,
    height::Float64,
)
    iszero(vertical_wavenumber) && return zero(ComplexF64)
    phase = _grounded_round_trip_phase(vertical_wavenumber, height)
    factor = one(ComplexF64) - phase
    return _scaled_sum_requires_exact(one(ComplexF64), -phase, factor) ?
        _grounded_interference_factor_exact(vertical_wavenumber, height) :
        factor
end

@noinline function _grounded_reflection_component_exact(
    current::ComplexF64,
    factor::ComplexF64,
    background_phase::ComplexF64,
    background_projection::ComplexF64,
    index,
)
    return setprecision(BigFloat, _PERIODIC_RWG_PHASE_FALLBACK_PRECISION) do
        value = Complex{BigFloat}(current) * Complex{BigFloat}(factor) -
                Complex{BigFloat}(background_phase) *
                Complex{BigFloat}(background_projection)
        _local_mass_convert_bigfloat(
            ComplexF64, value, "grounded reflection coefficient", index)
    end
end

@inline function _grounded_reflection_component(
    current::ComplexF64,
    factor::ComplexF64,
    background_phase::ComplexF64,
    background_projection::ComplexF64,
    index,
)
    image = current * factor
    background = background_phase * background_projection
    value = image - background
    requires_exact =
        !isfinite(value) ||
        _source_product_requires_exact(current, factor, image) ||
        _source_product_requires_exact(
            background_phase, background_projection, background) ||
        _scaled_sum_requires_exact(image, -background, value)
    return requires_exact ?
        _grounded_reflection_component_exact(
            current,
            factor,
            background_phase,
            background_projection,
            index,
        ) : value
end

function _validated_grounded_plane_wave(
        pw,
        k::Float64,
        lattice::PeriodicLattice)
    pw isa PlaneWaveExcitation ||
        throw(ArgumentError(
            "assemble_excitation_grounded requires a PlaneWaveExcitation; " *
            "got $(typeof(pw)). Use assemble_excitation for other source models."))
    _validate_plane_wave_wavenumber(
        pw, k, "assemble_excitation_grounded")

    vertical_wavenumber = _kz_inc(k, lattice)
    expected = Vec3(
        lattice.kx_bloch,
        lattice.ky_bloch,
        -vertical_wavenumber,
    )
    normalized_actual = pw.k_vec / k
    normalized_expected = expected / k
    @inbounds for component in 1:3
        isapprox(
            normalized_actual[component],
            normalized_expected[component];
            rtol=0.0,
            atol=1e-8,
        ) || throw(ArgumentError(
            "assemble_excitation_grounded plane-wave k_vec=$(pw.k_vec) " *
            "does not match the lattice's down-going incident wavevector " *
            "$expected. Rebuild the plane wave from the lattice incidence " *
            "angles and wavenumber."))
    end
    return vertical_wavenumber
end

"""
    assemble_excitation_grounded(mesh, rwg, pw, k, lattice; height, quad_order=3)

Excitation vector for the grounded problem: the metasurface is illuminated by the
incident plane wave plus its bare-ground reflection. For a TE/normal-incidence plane
wave referenced at the metasurface plane (z=0), the total tangential drive is scaled by
`(1 - exp(-2i kz_inc h))`, where `kz_inc` is the incident vertical wavenumber. `pw`
must be a down-going `PlaneWaveExcitation` whose wavevector matches the wavenumber and
Bloch components stored in `lattice`.
"""
function assemble_excitation_grounded(mesh::TriMesh, rwg::RWGData, pw, k,
                                      lattice::PeriodicLattice; height::Real, quad_order::Int=3)
    _validate_mesh_rwg_pair(mesh, rwg)
    kw = _validated_lattice_wavenumber(k, lattice)
    h = _validated_ground_height(height)
    vertical_wavenumber = _validated_grounded_plane_wave(pw, kw, lattice)
    v_inc = assemble_excitation(mesh, rwg, pw; quad_order=quad_order)
    factor = _grounded_interference_factor(vertical_wavenumber, h)
    v_grounded = similar(v_inc)
    @inbounds for index in eachindex(v_inc)
        v_grounded[index] = _checked_number_product(
            ComplexF64,
            factor,
            v_inc[index],
            "grounded excitation",
            index,
        )
    end
    all(isfinite, v_grounded) ||
        throw(OverflowError(
            "grounded excitation contains entries outside the " *
            "representable ComplexF64 range"))
    return v_grounded
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
    _validate_mesh_rwg_pair(mesh, rwg)
    kw = _validated_lattice_wavenumber(k, lattice)
    h = _validated_ground_height(height)
    modes, R_cur = reflection_coefficients(mesh, rwg, I, kw, lattice; kwargs...)
    kzi = _kz_inc(kw, lattice)
    R_g = similar(R_cur)
    background_phase = _grounded_round_trip_phase(kzi, h)
    for (i, m) in enumerate(modes)
        # Use real(m.kz): evanescent orders store kz = i·β (positive imaginary), so
        # exp(-2im·kz·h) = exp(2βh) overflows and 0·Inf = NaN (R_cur is 0 there).
        # The image phase delay is governed by the real vertical wavenumber; this
        # matches reflection_coefficient_vectors_grounded.
        factor = _grounded_interference_factor(real(m.kz), h)
        background_projection = m.m == 0 && m.n == 0 ?
            one(ComplexF64) : zero(ComplexF64)
        R_g[i] = _grounded_reflection_component(
            R_cur[i],
            factor,
            background_phase,
            background_projection,
            i,
        )
    end
    all(isfinite, R_g) ||
        throw(OverflowError(
            "grounded reflection coefficients contain values outside the " *
            "representable ComplexF64 range"))
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
    _validate_mesh_rwg_pair(mesh, rwg)
    kw = _validated_lattice_wavenumber(k, lattice)
    h = _validated_ground_height(height)
    _validate_periodic_polarization(pol)
    modes, R_cur = reflection_coefficient_vectors(mesh, rwg, I, kw, lattice; kwargs...)
    kzi = _kz_inc(kw, lattice)
    R_g = similar(R_cur)
    background_phase = _grounded_round_trip_phase(kzi, h)
    for (i, m) in enumerate(modes)
        factor = _grounded_interference_factor(real(m.kz), h)
        pol_mode = m.m == 0 && m.n == 0 ?
            _mode_transverse_projection(pol, m, kw) : nothing
        R_g[i] = CVec3(ntuple(component ->
            _grounded_reflection_component(
                R_cur[i][component],
                factor,
                background_phase,
                isnothing(pol_mode) ? zero(ComplexF64) :
                    ComplexF64(pol_mode[component]),
                (i, component),
            ), 3))
    end
    all(vector -> all(isfinite, vector), R_g) ||
        throw(OverflowError(
            "grounded reflection vectors contain values outside the " *
            "representable ComplexF64 range"))
    return modes, R_g
end
