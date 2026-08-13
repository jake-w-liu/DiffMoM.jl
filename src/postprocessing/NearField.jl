# NearField.jl — Scattered electric near-field evaluation
#
# Evaluates the scattered electric field E_sca(r) from solved RWG current
# coefficients at arbitrary observation points using the mixed-potential form:
#
#   E_sca(r) = -i k eta0 ∫_Γ J(r') G(r,r') dS'
#              -i (eta0 / k) ∫_Γ (∇'·J(r')) ∇G(r,r') dS'
#
# This is consistent with the package's exp(+iωt) convention and the EFIE
# assembly sign convention after Galerkin testing.

export compute_nearfield, compute_total_field

const _DEFAULT_MAX_NEARFIELD_WORK_BYTES = 512 * 1024 * 1024
const _DEFAULT_MAX_NEARFIELD_INTERACTION_TERMS = 200_000_000

@inline function _nearfield_quadrature_count(quad_order::Int)
    quad_order == 1 && return 1
    quad_order == 3 && return 3
    quad_order == 4 && return 4
    quad_order == 7 && return 7
    throw(ArgumentError(
        "compute_nearfield: unsupported quadrature order $quad_order; " *
        "use 1, 3, 4, or 7"))
end

function _nearfield_work_bytes(
        triangle_count::Int,
        basis_count::Int,
        observation_count::Int,
        quadrature_count::Int)
    total = BigInt(0)
    # Output field and the owned observation snapshot.
    total += BigInt(3sizeof(ComplexF64) + sizeof(Vec3)) * observation_count
    # Per-triangle quadrature points, current samples, affine current values,
    # scalar samples, geometry caches, and the top-level vector references.
    total += BigInt(
        quadrature_count * (sizeof(Vec3) + sizeof(CVec3)) +
        3sizeof(CVec3) + sizeof(Float64) + sizeof(ComplexF64) +
        3sizeof(Vec3) + sizeof(Float64) + sizeof(Vector{Vec3}) +
        sizeof(Vector{Int}),
    ) * triangle_count
    # Every RWG belongs to two triangle adjacency lists.  Include their stored
    # integer payload; object headers and allocator bookkeeping are excluded.
    total += BigInt(2sizeof(Int)) * basis_count
    total <= typemax(Int) ||
        throw(ArgumentError("near-field raw-workspace estimate overflows Int"))
    return Int(total)
end

function _preflight_nearfield_work(
        triangle_count::Int,
        basis_count::Int,
        observation_count::Int,
        quadrature_count::Int,
        max_work_bytes::Integer,
        max_interaction_terms::Integer)
    work_bytes = _nearfield_work_bytes(
        triangle_count, basis_count, observation_count, quadrature_count)
    _enforce_payload_limit(
        work_bytes, max_work_bytes,
        "near-field output and workspace", "max_work_bytes")
    term_limit = _validated_resource_limit(
        "max_interaction_terms", max_interaction_terms)
    terms = BigInt(triangle_count) * observation_count * quadrature_count
    terms <= term_limit ||
        throw(ArgumentError(
            "near-field evaluation requires $terms triangle-quadrature " *
            "interaction terms, exceeding max_interaction_terms=$term_limit"))
    return work_bytes, Int(terms)
end

@inline function _point_triangle_distance(p::Vec3, a::Vec3, b::Vec3, c::Vec3)
    ab_raw = b - a
    ac_raw = c - a
    ap_raw = p - a
    bp_raw = p - b
    cp_raw = p - c
    scale = max(
        maximum(abs, ab_raw), maximum(abs, ac_raw),
        maximum(abs, ap_raw), maximum(abs, bp_raw),
        maximum(abs, cp_raw))
    isfinite(scale) && scale > 0.0 ||
        throw(ArgumentError(
            "point-triangle distance geometry has no finite nonzero scale"))
    ab = ab_raw / scale
    ac = ac_raw / scale
    ap = ap_raw / scale
    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    if d1 <= 0.0 && d2 <= 0.0
        return scale * norm(ap)
    end

    bp = bp_raw / scale
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    if d3 >= 0.0 && d4 <= d3
        return scale * norm(bp)
    end

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0
        v = d1 / (d1 - d3)
        return scale * norm(ap - v * ab)
    end

    cp = cp_raw / scale
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)
    if d6 >= 0.0 && d5 <= d6
        return scale * norm(cp)
    end

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0
        w = d2 / (d2 - d6)
        return scale * norm(ap - w * ac)
    end

    va = d3 * d6 - d5 * d4
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return scale * norm(bp - w * (ac - ab))
    end

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    distance = scale * norm(ap - v * ab - w * ac)
    isfinite(distance) ||
        throw(OverflowError(
            "point-triangle distance is outside the Float64 range"))
    return distance
end

function _surface_distance(mesh::TriMesh, p::Vec3)
    dmin = Inf
    Nt = ntriangles(mesh)
    @inbounds for t in 1:Nt
        a = _mesh_vertex(mesh, mesh.tri[1, t])
        b = _mesh_vertex(mesh, mesh.tri[2, t])
        c = _mesh_vertex(mesh, mesh.tri[3, t])
        d = _point_triangle_distance(p, a, b, c)
        dmin = min(dmin, d)
    end
    return dmin
end

@inline function _default_nearfield_surface_tol(mesh::TriMesh)
    diagonal = _bbox_diagonal(mesh)
    tolerance = 1e-10 * diagonal
    if iszero(tolerance) && diagonal > 0.0
        return nextfloat(0.0)
    end
    return tolerance
end

function _collect_observation_points(points::Vector{Vec3})
    return points
end

function _collect_observation_points(points::AbstractVector{<:Vec3})
    return collect(points)
end

function _preflight_observation_snapshot(
        observation_count::Int, max_work_bytes::Integer)
    snapshot_bytes = _checked_array_payload_bytes(
        Vec3, observation_count; label="near-field observation snapshot")
    _enforce_payload_limit(
        snapshot_bytes, max_work_bytes,
        "near-field observation snapshot", "max_work_bytes")
    return nothing
end

function _collect_observation_points(points::AbstractMatrix{<:Real})
    size(points, 1) == 3 ||
        throw(DimensionMismatch("Observation-point matrix must have size (3, Nobs), got $(size(points))."))
    Nobs = size(points, 2)
    obs = Vector{Vec3}(undef, Nobs)
    @inbounds for i in 1:Nobs
        obs[i] = Vec3(points[1, i], points[2, i], points[3, i])
    end
    return obs
end


function _prepare_nearfield_observations(
        observation_points::AbstractVector{<:Vec3},
        max_work_bytes::Integer)
    _preflight_observation_snapshot(
        length(observation_points), max_work_bytes)
    return _collect_observation_points(observation_points)
end

function _prepare_nearfield_observations(
        observation_points::AbstractMatrix{<:Real},
        max_work_bytes::Integer)
    size(observation_points, 1) == 3 ||
        throw(DimensionMismatch(
            "Observation-point matrix must have size (3, Nobs), got " *
            "$(size(observation_points))."))
    _preflight_observation_snapshot(
        size(observation_points, 2), max_work_bytes)
    return _collect_observation_points(observation_points)
end

function _precompute_nearfield_triangle_data(mesh::TriMesh, rwg::RWGData,
                                             I_coeffs::AbstractVector{<:Number},
                                             xi::Vector{<:SVector{2}})
    Nt = ntriangles(mesh)
    Nq = length(xi)
    N = rwg.nedges

    quad_pts = Vector{Vector{Vec3}}(undef, Nt)
    areas = Vector{Float64}(undef, Nt)
    tri_to_basis = [Int[] for _ in 1:Nt]

    @inbounds for n in 1:N
        push!(tri_to_basis[rwg.tplus[n]], n)
        push!(tri_to_basis[rwg.tminus[n]], n)
    end

    # Flat matrix avoids per-triangle Vector{CVec3} heap allocations
    J_samples = zeros(CVec3, Nq, Nt)
    div_samples = Vector{ComplexF64}(undef, Nt)

    # Vertex values of the surface current J(r') on each triangle.  Because the
    # RWG current is affine on a flat triangle, storing the three vertex values
    # lets us reconstruct J at ANY point (e.g. the near-singular projection
    # point) exactly via barycentric interpolation — needed by the
    # singularity-subtracted near-field branch.
    J_verts = Matrix{CVec3}(undef, 3, Nt)

    @inbounds for t in 1:Nt
        quad_pts[t] = tri_quad_points(mesh, t, xi)
        areas[t] = triangle_area(mesh, t)
        v1 = _mesh_vertex(mesh, mesh.tri[1, t])
        v2 = _mesh_vertex(mesh, mesh.tri[2, t])
        v3 = _mesh_vertex(mesh, mesh.tri[3, t])
        Jv1 = zero(CVec3); Jv2 = zero(CVec3); Jv3 = zero(CVec3)
        divt = 0.0 + 0im

        for n in tri_to_basis[t]
            In = ComplexF64(I_coeffs[n])
            divt += In * div_rwg(rwg, n, t)
            for q in 1:Nq
                J_samples[q, t] += In * eval_rwg(rwg, n, quad_pts[t][q], t)
            end
            Jv1 += In * eval_rwg(rwg, n, v1, t)
            Jv2 += In * eval_rwg(rwg, n, v2, t)
            Jv3 += In * eval_rwg(rwg, n, v3, t)
        end

        J_verts[1, t] = Jv1
        J_verts[2, t] = Jv2
        J_verts[3, t] = Jv3
        div_samples[t] = divt
    end

    return quad_pts, areas, J_samples, div_samples, J_verts
end

# Reconstruct the (affine) RWG current J at point `r` on triangle `t` from its
# three precomputed vertex values, using barycentric interpolation.  Exact for
# the linear RWG basis.  Used for the near-singular leading-term evaluation.
@inline function _eval_J_affine(J_verts::AbstractMatrix{CVec3}, t::Int,
                                r::Vec3, V1::Vec3, V2::Vec3, V3::Vec3)
    e1 = V2 - V1
    e2 = V3 - V1
    rp = r - V1
    scale = max(maximum(abs, e1), maximum(abs, e2), maximum(abs, rp))
    isfinite(scale) && scale > 0.0 || return J_verts[1, t]
    e1 /= scale
    e2 /= scale
    rp /= scale
    cross_edges = cross(e1, e2)
    drop_axis = argmax(abs.(cross_edges))
    first_axis, second_axis = drop_axis == 1 ? (2, 3) :
                              drop_axis == 2 ? (1, 3) : (1, 2)
    denominator = e1[first_axis] * e2[second_axis] -
                  e1[second_axis] * e2[first_axis]
    if !(isfinite(denominator) && !iszero(denominator))
        return setprecision(BigFloat, 4352) do
            e1a = BigFloat(V2[first_axis]) - BigFloat(V1[first_axis])
            e1b = BigFloat(V2[second_axis]) - BigFloat(V1[second_axis])
            e2a = BigFloat(V3[first_axis]) - BigFloat(V1[first_axis])
            e2b = BigFloat(V3[second_axis]) - BigFloat(V1[second_axis])
            rpa = BigFloat(r[first_axis]) - BigFloat(V1[first_axis])
            rpb = BigFloat(r[second_axis]) - BigFloat(V1[second_axis])
            determinant = e1a * e2b - e1b * e2a
            iszero(determinant) && return J_verts[1, t]
            lam2_big = (rpa * e2b - rpb * e2a) / determinant
            lam3_big = (e1a * rpb - e1b * rpa) / determinant
            lam1_big = 1 - lam2_big - lam3_big
            ComplexF64(lam1_big) * J_verts[1, t] +
            ComplexF64(lam2_big) * J_verts[2, t] +
            ComplexF64(lam3_big) * J_verts[3, t]
        end
    end
    lam2 = (rp[first_axis] * e2[second_axis] -
            rp[second_axis] * e2[first_axis]) / denominator
    lam3 = (e1[first_axis] * rp[second_axis] -
            e1[second_axis] * rp[first_axis]) / denominator
    lam1 = 1.0 - lam2 - lam3                     # weight of V1
    return lam1 * J_verts[1, t] + lam2 * J_verts[2, t] + lam3 * J_verts[3, t]
end

function _compute_nearfield_matrix(mesh::TriMesh, rwg::RWGData,
                                   I_coeffs::AbstractVector{<:Number},
                                   observation_points::Vector{Vec3}, k;
                                   quad_order::Int=3,
                                   eta0::Float64=376.730313668,
                                   check_surface::Bool=true,
                                   surface_tol::Union{Nothing,Float64}=nothing,
                                   max_work_bytes::Integer=
                                       _DEFAULT_MAX_NEARFIELD_WORK_BYTES,
                                   max_interaction_terms::Integer=
                                       _DEFAULT_MAX_NEARFIELD_INTERACTION_TERMS)
    length(I_coeffs) == rwg.nedges ||
        throw(DimensionMismatch("I_coeffs length $(length(I_coeffs)) != rwg.nedges=$(rwg.nedges)."))
    (isfinite(real(k)) && isfinite(imag(k)) && abs(k) > 0.0) ||
        throw(ArgumentError("compute_nearfield: k must be finite and nonzero, got $k."))
    (isfinite(eta0) && eta0 > 0.0) ||
        throw(ArgumentError(
            "compute_nearfield: eta0 must be finite and positive, got $eta0."))

    @inbounds for i in eachindex(I_coeffs)
        isfinite(I_coeffs[i]) ||
            throw(ArgumentError(
                "compute_nearfield: current coefficient $i must be finite, got $(I_coeffs[i])."))
    end
    @inbounds for i in eachindex(observation_points)
        p = observation_points[i]
        all(isfinite, p) ||
            throw(ArgumentError(
                "compute_nearfield: observation point $i must be finite, got $p."))
    end

    tol = isnothing(surface_tol) ? _default_nearfield_surface_tol(mesh) : surface_tol
    (isfinite(tol) && tol >= 0.0) ||
        throw(ArgumentError(
            "compute_nearfield: surface_tol must be finite and nonnegative, got $tol."))

    Nobs = length(observation_points)
    Nq = _nearfield_quadrature_count(quad_order)
    _preflight_nearfield_work(
        ntriangles(mesh), rwg.nedges, Nobs, Nq,
        max_work_bytes, max_interaction_terms)
    Nobs == 0 && return zeros(ComplexF64, 3, 0)

    if check_surface
        for (i, p) in enumerate(observation_points)
            d = _surface_distance(mesh, p)
            if d <= tol
                error(
                    "compute_nearfield does not support observation points on the surface " *
                    "or within surface_tol=$tol of it. Point $i has minimum distance $d."
                )
            end
        end
    end

    # Preserve the exact zero field without forming irrelevant mixed-scale
    # prefactors. Surface and input validation above still runs in full.
    all(iszero, I_coeffs) && return zeros(ComplexF64, 3, Nobs)

    effective_currents = I_coeffs
    effective_eta0 = eta0
    pref_vec = -1im * k * effective_eta0
    pref_scl = -1im * effective_eta0 / k
    if !(isfinite(pref_vec) && isfinite(pref_scl))
        # Transfer an exact power of two from eta0 into every current.  Both
        # potential terms depend on eta0*I, so this leaves the physical field
        # unchanged while keeping the standalone prefactors representable.
        # The transfer is accepted only when every converted current survives
        # it exactly; otherwise a joint high-precision field kernel is needed
        # and this routine fails closed.
        shift = 0
        while shift < 2_150
            shift += 1
            candidate_eta0 = ldexp(eta0, -shift)
            iszero(candidate_eta0) && break
            candidate_vec = -1im * k * candidate_eta0
            candidate_scl = -1im * candidate_eta0 / k
            if isfinite(candidate_vec) && isfinite(candidate_scl)
                # This is the smallest scalar-feasible shift.  A larger
                # positive shift only increases every current magnitude, so
                # it cannot repair an overflow or an inexact round trip.
                representable = true
                @inbounds for index in eachindex(I_coeffs)
                    value = ComplexF64(I_coeffs[index])
                    real_scaled = ldexp(real(value), shift)
                    imag_scaled = ldexp(imag(value), shift)
                    if !isfinite(real_scaled) || !isfinite(imag_scaled) ||
                       ldexp(real_scaled, -shift) != real(value) ||
                       ldexp(imag_scaled, -shift) != imag(value)
                        representable = false
                        break
                    end
                end
                representable || break
                scaled = Vector{ComplexF64}(undef, length(I_coeffs))
                @inbounds for index in eachindex(I_coeffs)
                    value = ComplexF64(I_coeffs[index])
                    scaled[index] = ComplexF64(
                        ldexp(real(value), shift),
                        ldexp(imag(value), shift),
                    )
                end
                effective_currents = scaled
                effective_eta0 = candidate_eta0
                pref_vec = candidate_vec
                pref_scl = candidate_scl
                break
            end
        end
        (isfinite(pref_vec) && isfinite(pref_scl)) ||
            throw(OverflowError(
                "compute_nearfield cannot represent the coupled k, eta0, " *
                "and current scaling in ComplexF64"))
    end

    xi, wq = tri_quad_rule(quad_order)
    @assert length(wq) == Nq
    quad_pts, areas, J_samples, div_samples, J_verts =
        _precompute_nearfield_triangle_data(
            mesh, rwg, effective_currents, xi)

    Nt = ntriangles(mesh)
    E = zeros(ComplexF64, 3, Nobs)

    # Near-singular quadrature is activated per-triangle when the observation
    # point is closer than the triangle's characteristic edge length.
    # This is a physics-based criterion: standard Gaussian quadrature of
    # order Nq on a triangle of edge h resolves integrands varying on scale
    # ~h. When the 1/R singularity is at distance d < h, the integrand
    # varies faster than the quadrature can resolve → singularity subtraction
    # is needed. No global threshold — each triangle uses its own size.

    inv4pi = 1.0 / (4π)

    # Precompute triangle vertices to avoid repeated mesh lookups
    V1_all = Vector{Vec3}(undef, Nt)
    V2_all = Vector{Vec3}(undef, Nt)
    V3_all = Vector{Vec3}(undef, Nt)
    h_t_all = Vector{Float64}(undef, Nt)
    @inbounds for t in 1:Nt
        V1_all[t] = _mesh_vertex(mesh, mesh.tri[1, t])
        V2_all[t] = _mesh_vertex(mesh, mesh.tri[2, t])
        V3_all[t] = _mesh_vertex(mesh, mesh.tri[3, t])
        h_t_all[t] = sqrt(2 * areas[t])
    end

    Threads.@threads for i in 1:Nobs
        @inbounds begin
        robs = observation_points[i]
        Ex = 0.0 + 0im
        Ey = 0.0 + 0im
        Ez = 0.0 + 0im

        for t in 1:Nt
            At = areas[t]
            divt = div_samples[t]

            V1 = V1_all[t]
            V2 = V2_all[t]
            V3 = V3_all[t]
            dist = _point_triangle_distance(robs, V1, V2, V3)

            h_t = h_t_all[t]
            if dist < h_t / Nq
                # ── Near-singular branch: singularity subtraction on BOTH the
                #    vector (1/R) and scalar-gradient (1/R²) potential terms,
                #    mirroring the EFIE self-cell treatment in
                #    SingularIntegrals.jl. ──
                #
                # Vector term  ∫_T J(r')/(4πR) dS' splits as
                #   J(r'_*)·S/(4π)  +  ∫_T [J(r') − J(r'_*)]/(4πR) dS'
                # where r'_* is the in-plane projection of the observation point.
                # The remainder is bounded because the RWG current is affine.
                #
                # Scalar term  ∫_T ∇_r G dS' splits as
                #   ∫_T ∇_r G_smooth dS'  +  (1/4π) ∇_r S
                # where ∇_r G_smooth = ∇_r G − ∇_r(1/4πR), ∇_r(1/4πR) =
                # −(r−r')/(4πR³), and ∇_r S is the analytical gradient of the
                # 1/R potential integral.  This subtracts the 1/R² singularity
                # that the old code integrated directly.
                S = analytical_integral_1overR(
                    robs, V1, V2, V3)

                # In-plane projection r'_* of robs onto the triangle plane.
                nhatT = triangle_normal(mesh, t)
                h_proj = dot(robs - V1, nhatT)
                r_star = robs - h_proj * nhatT
                J_star = _eval_J_affine(J_verts, t, r_star, V1, V2, V3)

                for q in 1:Nq
                    rq = quad_pts[t][q]
                    wt = wq[q] * (2 * At)
                    Gs = _greens_smooth_unchecked(robs, rq, k)
                    Jq = J_samples[q, t]

                    # Vector smooth part
                    Ex += pref_vec * Jq[1] * (wt * Gs)
                    Ey += pref_vec * Jq[2] * (wt * Gs)
                    Ez += pref_vec * Jq[3] * (wt * Gs)

                    # Vector singular remainder: [J(rq) − J(r'_*)]/(4πR)
                    Rv = robs - rq
                    R = sqrt(dot(Rv, Rv))
                    if !iszero(R)
                        dJ = Jq - J_star
                        crem = (wt * inv4pi) / R
                        Ex += pref_vec * dJ[1] * crem
                        Ey += pref_vec * dJ[2] * crem
                        Ez += pref_vec * dJ[3] * crem
                    end

                    if abs(divt) > 0.0
                        # Scalar smooth part: ∇G_smooth = ∇G − ∇(1/4πR)
                        gradG = _grad_greens_unchecked(robs, rq, k)
                        if !iszero(R)
                            direction = Rv / R
                            inv4piR2 = (inv4pi / R) / R
                            gradG = gradG + direction * inv4piR2
                        end
                        Ex += pref_scl * divt * (wt * gradG[1])
                        Ey += pref_scl * divt * (wt * gradG[2])
                        Ez += pref_scl * divt * (wt * gradG[3])
                    end
                end

                # Vector singular leading term: J(r'_*) · S/(4π)
                Ex += pref_vec * J_star[1] * (inv4pi * S)
                Ey += pref_vec * J_star[2] * (inv4pi * S)
                Ez += pref_vec * J_star[3] * (inv4pi * S)

                # Scalar singular term: (1/4π) ∇_r S (analytical)
                if abs(divt) > 0.0
                    gradS = grad_analytical_integral_1overR(
                        robs, V1, V2, V3)
                    cscl = pref_scl * divt * inv4pi
                    Ex += cscl * gradS[1]
                    Ey += cscl * gradS[2]
                    Ez += cscl * gradS[3]
                end
            else
                # ── Standard quadrature (far from surface) ──
                for q in 1:Nq
                    rq = quad_pts[t][q]
                    wt = wq[q] * (2 * At)
                    G = _greens_unchecked(robs, rq, k)
                    Jq = J_samples[q, t]

                    Ex += pref_vec * Jq[1] * (wt * G)
                    Ey += pref_vec * Jq[2] * (wt * G)
                    Ez += pref_vec * Jq[3] * (wt * G)

                    if abs(divt) > 0.0
                        gradG = _grad_greens_unchecked(robs, rq, k)
                        Ex += pref_scl * divt * (wt * gradG[1])
                        Ey += pref_scl * divt * (wt * gradG[2])
                        Ez += pref_scl * divt * (wt * gradG[3])
                    end
                end
            end
        end

        E[1, i] = Ex
        E[2, i] = Ey
        E[3, i] = Ez
        end  # @inbounds
    end

    all(isfinite, E) ||
        error(
            "compute_nearfield produced non-finite field values despite finite inputs; " *
            "check the observation distance and numerical scales.")
    return E
end

function _compute_total_field_matrix(mesh::TriMesh, rwg::RWGData,
                                     I_coeffs::AbstractVector{<:Number},
                                     excitation::AbstractExcitation,
                                     observation_points::Vector{Vec3}, k;
                                     quad_order::Int=3,
                                     eta0::Float64=376.730313668,
                                     check_surface::Bool=true,
                                     surface_tol::Union{Nothing,Float64}=nothing,
                                     max_work_bytes::Integer=
                                         _DEFAULT_MAX_NEARFIELD_WORK_BYTES,
                                     max_interaction_terms::Integer=
                                         _DEFAULT_MAX_NEARFIELD_INTERACTION_TERMS)
    _validate_incident_electric_field_wavenumber(excitation, k)
    E_total = _compute_nearfield_matrix(mesh, rwg, I_coeffs, observation_points, k;
                                        quad_order=quad_order,
                                        eta0=eta0,
                                        check_surface=check_surface,
                                        surface_tol=surface_tol,
                                        max_work_bytes=max_work_bytes,
                                        max_interaction_terms=max_interaction_terms)
    @inbounds for i in eachindex(observation_points)
        E_inc = _check_finite_cvec3(
            _incident_electric_field(excitation, observation_points[i], k),
            "incident electric field",
        )
        E_total[1, i] += E_inc[1]
        E_total[2, i] += E_inc[2]
        E_total[3, i] += E_inc[3]
    end
    all(isfinite, E_total) ||
        error("compute_total_field produced non-finite field values.")
    return E_total
end

"""
    compute_nearfield(mesh, rwg, I_coeffs, observation_points, k; kwargs...)

Compute the scattered electric near field `E_sca(r)` at arbitrary observation
points from solved RWG current coefficients.

The implementation uses the same `exp(+iωt)` convention and mixed-potential EFIE
sign convention as the rest of the package:

`E_sca(r) = -i k eta0 ∫ J(r') G(r,r') dS' - i (eta0/k) ∫ (∇'·J(r')) ∇G(r,r') dS'`

# Arguments
- `mesh::TriMesh`: surface mesh
- `rwg::RWGData`: RWG basis data
- `I_coeffs`: current coefficients, length `rwg.nedges`
- `observation_points`: either a single `Vec3`, a `Vector{Vec3}`, or a `3 x Nobs`
  real matrix of points
- `k`: wavenumber

# Keyword arguments
- `quad_order=3`: triangle quadrature order
- `eta0=376.730313668`: free-space impedance
- `check_surface=true`: reject points on the surface
- `surface_tol=nothing`: optional minimum point-to-surface tolerance; defaults to
  `1e-10 * bbox_diagonal(mesh)` (rounded up to the least positive Float64
  when that product underflows)
- `max_work_bytes=536_870_912`: maximum raw payload of the output and retained
  construction workspaces, checked before geometry or field arrays are built
- `max_interaction_terms=200_000_000`: maximum number of direct
  triangle-quadrature interactions across all observation points

# Returns
- Single-point input: `CVec3`
- Multi-point input: `Matrix{ComplexF64}` of size `(3, Nobs)`

# Limitations
- This is a direct quadrature evaluator. For observation points close to the
  surface it automatically switches to a singularity-subtracted near-field
  scheme (the vector `1/R` and scalar-gradient `1/R²` singularities are removed
  analytically/semi-analytically, mirroring the EFIE self-cell treatment), so
  near-surface accuracy no longer degrades with the singularity.
- On-surface evaluation is not supported.
"""
function compute_nearfield(mesh::TriMesh, rwg::RWGData,
                           I_coeffs::AbstractVector{<:Number},
                           observation_point::Vec3, k;
                           quad_order::Int=3,
                           eta0::Float64=376.730313668,
                           check_surface::Bool=true,
                           surface_tol::Union{Nothing,Float64}=nothing,
                           max_work_bytes::Integer=
                               _DEFAULT_MAX_NEARFIELD_WORK_BYTES,
                           max_interaction_terms::Integer=
                               _DEFAULT_MAX_NEARFIELD_INTERACTION_TERMS)
    E = _compute_nearfield_matrix(mesh, rwg, I_coeffs, [observation_point], k;
                                  quad_order=quad_order,
                                  eta0=eta0,
                                  check_surface=check_surface,
                                  surface_tol=surface_tol,
                                  max_work_bytes=max_work_bytes,
                                  max_interaction_terms=max_interaction_terms)
    return CVec3(E[:, 1])
end

function compute_nearfield(mesh::TriMesh, rwg::RWGData,
                           I_coeffs::AbstractVector{<:Number},
                           observation_points::AbstractVector{<:Vec3}, k;
                           quad_order::Int=3,
                           eta0::Float64=376.730313668,
                           check_surface::Bool=true,
                           surface_tol::Union{Nothing,Float64}=nothing,
                           max_work_bytes::Integer=
                               _DEFAULT_MAX_NEARFIELD_WORK_BYTES,
                           max_interaction_terms::Integer=
                               _DEFAULT_MAX_NEARFIELD_INTERACTION_TERMS)
    obs = _prepare_nearfield_observations(
        observation_points, max_work_bytes)
    return _compute_nearfield_matrix(mesh, rwg, I_coeffs, obs, k;
                                     quad_order=quad_order,
                                     eta0=eta0,
                                     check_surface=check_surface,
                                     surface_tol=surface_tol,
                                     max_work_bytes=max_work_bytes,
                                     max_interaction_terms=max_interaction_terms)
end

function compute_nearfield(mesh::TriMesh, rwg::RWGData,
                           I_coeffs::AbstractVector{<:Number},
                           observation_points::AbstractMatrix{<:Real}, k;
                           quad_order::Int=3,
                           eta0::Float64=376.730313668,
                           check_surface::Bool=true,
                           surface_tol::Union{Nothing,Float64}=nothing,
                           max_work_bytes::Integer=
                               _DEFAULT_MAX_NEARFIELD_WORK_BYTES,
                           max_interaction_terms::Integer=
                               _DEFAULT_MAX_NEARFIELD_INTERACTION_TERMS)
    obs = _prepare_nearfield_observations(
        observation_points, max_work_bytes)
    return _compute_nearfield_matrix(mesh, rwg, I_coeffs, obs, k;
                                     quad_order=quad_order,
                                     eta0=eta0,
                                     check_surface=check_surface,
                                     surface_tol=surface_tol,
                                     max_work_bytes=max_work_bytes,
                                     max_interaction_terms=max_interaction_terms)
end

"""
    compute_total_field(mesh, rwg, I_coeffs, excitation, observation_points, k; kwargs...)

Compute the total electric field `E_total(r) = E_inc(r) + E_sca(r)` at arbitrary
observation points from a solved RWG current distribution and its associated
incident excitation.

The scattered component `E_sca` uses the same mixed-potential EFIE
representation and `exp(+iωt)` sign convention as `compute_nearfield`.

# Arguments
- `mesh::TriMesh`: surface mesh
- `rwg::RWGData`: RWG basis data
- `I_coeffs`: current coefficients, length `rwg.nedges`
- `excitation::AbstractExcitation`: excitation used to define the incident field
- `observation_points`: either a single `Vec3`, a `Vector{Vec3}`, or a `3 x Nobs`
  real matrix of points
- `k`: wavenumber used in the forward solve

# Keyword arguments
- `quad_order=3`: triangle quadrature order for the scattered field
- `eta0=376.730313668`: free-space impedance
- `check_surface=true`: reject points on the surface
- `surface_tol=nothing`: optional minimum point-to-surface tolerance
- `max_work_bytes=536_870_912`: maximum raw payload of the scattered-field
  output and retained construction workspaces
- `max_interaction_terms=200_000_000`: maximum number of direct
  triangle-quadrature interactions across all observation points

# Returns
- Single-point input: `CVec3`
- Multi-point input: `Matrix{ComplexF64}` of size `(3, Nobs)`

# Supported excitations
- `PlaneWaveExcitation`
- `DipoleExcitation`
- `LoopExcitation`
- `PatternFeedExcitation`
- `ImportedExcitation(kind=:electric_field)`
- `MultiExcitation` composed only of supported pointwise incident-field models

# Limitations
- `PortExcitation`, `DeltaGapExcitation`, and
  `ImportedExcitation(kind=:surface_current_density)` are not supported because
  they do not define rigorous observation-point incident electric fields in the
  current formulation.
- On-surface evaluation is not supported.
"""
function compute_total_field(mesh::TriMesh, rwg::RWGData,
                             I_coeffs::AbstractVector{<:Number},
                             excitation::AbstractExcitation,
                             observation_point::Vec3, k;
                             quad_order::Int=3,
                             eta0::Float64=376.730313668,
                             check_surface::Bool=true,
                             surface_tol::Union{Nothing,Float64}=nothing,
                             max_work_bytes::Integer=
                                 _DEFAULT_MAX_NEARFIELD_WORK_BYTES,
                             max_interaction_terms::Integer=
                                 _DEFAULT_MAX_NEARFIELD_INTERACTION_TERMS)
    E = _compute_total_field_matrix(mesh, rwg, I_coeffs, excitation, [observation_point], k;
                                    quad_order=quad_order,
                                    eta0=eta0,
                                    check_surface=check_surface,
                                    surface_tol=surface_tol,
                                    max_work_bytes=max_work_bytes,
                                    max_interaction_terms=max_interaction_terms)
    return CVec3(E[:, 1])
end

function compute_total_field(mesh::TriMesh, rwg::RWGData,
                             I_coeffs::AbstractVector{<:Number},
                             excitation::AbstractExcitation,
                             observation_points::AbstractVector{<:Vec3}, k;
                             quad_order::Int=3,
                             eta0::Float64=376.730313668,
                             check_surface::Bool=true,
                             surface_tol::Union{Nothing,Float64}=nothing,
                             max_work_bytes::Integer=
                                 _DEFAULT_MAX_NEARFIELD_WORK_BYTES,
                             max_interaction_terms::Integer=
                                 _DEFAULT_MAX_NEARFIELD_INTERACTION_TERMS)
    obs = _prepare_nearfield_observations(
        observation_points, max_work_bytes)
    return _compute_total_field_matrix(mesh, rwg, I_coeffs, excitation, obs, k;
                                       quad_order=quad_order,
                                       eta0=eta0,
                                       check_surface=check_surface,
                                       surface_tol=surface_tol,
                                       max_work_bytes=max_work_bytes,
                                       max_interaction_terms=max_interaction_terms)
end

function compute_total_field(mesh::TriMesh, rwg::RWGData,
                             I_coeffs::AbstractVector{<:Number},
                             excitation::AbstractExcitation,
                             observation_points::AbstractMatrix{<:Real}, k;
                             quad_order::Int=3,
                             eta0::Float64=376.730313668,
                             check_surface::Bool=true,
                             surface_tol::Union{Nothing,Float64}=nothing,
                             max_work_bytes::Integer=
                                 _DEFAULT_MAX_NEARFIELD_WORK_BYTES,
                             max_interaction_terms::Integer=
                                 _DEFAULT_MAX_NEARFIELD_INTERACTION_TERMS)
    obs = _prepare_nearfield_observations(
        observation_points, max_work_bytes)
    return _compute_total_field_matrix(mesh, rwg, I_coeffs, excitation, obs, k;
                                       quad_order=quad_order,
                                       eta0=eta0,
                                       check_surface=check_surface,
                                       surface_tol=surface_tol,
                                       max_work_bytes=max_work_bytes,
                                       max_interaction_terms=max_interaction_terms)
end
