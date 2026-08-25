# SingularIntegrals.jl — Singularity extraction for EFIE self-cell integrals
#
# When source and observation triangles coincide (self-cell), the Green's
# function G = exp(-ikR)/(4πR) has a 1/R singularity that standard Gaussian
# quadrature cannot resolve.  We split G = G_smooth + 1/(4πR) and compute:
#   - Smooth part: standard product quadrature with G_smooth (bounded)
#   - Singular part: semi-analytical (outer quadrature, analytical inner)

export analytical_integral_1overR, grad_analytical_integral_1overR,
       self_cell_contribution, adjacent_cell_contribution

@inline function _static_triangle_integral_fast_geometry(
    P::Vec3,
    V1::Vec3,
    V2::Vec3,
    V3::Vec3,
)
    all(isfinite, P) ||
        throw(ArgumentError(
            "static-integral observation point must be finite"))
    all(isfinite, V1) && all(isfinite, V2) && all(isfinite, V3) ||
        throw(ArgumentError(
            "static-integral triangle vertices must be finite"))

    edge1 = V2 - V1
    edge2 = V3 - V1
    scale1 = max(abs(edge1[1]), abs(edge1[2]), abs(edge1[3]))
    scale2 = max(abs(edge2[1]), abs(edge2[2]), abs(edge2[3]))
    (isfinite(scale1) && isfinite(scale2) &&
     scale1 > 0.0 && scale2 > 0.0) || return nothing
    scaled_normal = cross(edge1 / scale1, edge2 / scale2)
    normal_norm = norm(scaled_normal)
    (isfinite(normal_norm) && normal_norm > 0.0) || return nothing
    scale = maximum((
        scale1, scale2, maximum(abs, P - V1)))
    isfinite(scale) && scale > 0.0 || return nothing
    # A single normalization scale is useful only if it preserves every
    # nonzero stored difference.  Otherwise an off-plane observation or a thin
    # edge can become exactly coplanar/degenerate and change the closed form.
    point_offset = P - V1
    @inbounds for vector in (edge1, edge2, point_offset)
        for component in vector
            if !iszero(component) && iszero(component / scale)
                return nothing
            end
        end
    end
    # Branching on the signed plane height also has to be certified.  Even
    # when every normalized coordinate survives, cancellation followed by a
    # subnormal product can round a genuinely off-plane point onto the plane.
    scaled_edge1 = edge1 / scale
    scaled_edge2 = edge2 / scale
    scaled_offset = point_offset / scale
    scaled_normal_for_height = cross(scaled_edge1, scaled_edge2)
    fast_height = dot(scaled_offset, scaled_normal_for_height)
    if iszero(fast_height)
        exactly_coplanar = setprecision(
                BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
            first_big = SVector{3,BigFloat}(BigFloat.(V1))
            edge1_big = SVector{3,BigFloat}(BigFloat.(V2)) - first_big
            edge2_big = SVector{3,BigFloat}(BigFloat.(V3)) - first_big
            offset_big = SVector{3,BigFloat}(BigFloat.(P)) - first_big
            iszero(dot(offset_big, cross(edge1_big, edge2_big)))
        end
        exactly_coplanar || return nothing
    end
    return (origin=V1, scale=scale)
end

@noinline function _static_triangle_integral_big_geometry(
        P::Vec3, V1::Vec3, V2::Vec3, V3::Vec3)
    return setprecision(
            BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        point = SVector{3,BigFloat}(BigFloat.(P))
        first = SVector{3,BigFloat}(BigFloat.(V1))
        second = SVector{3,BigFloat}(BigFloat.(V2))
        third = SVector{3,BigFloat}(BigFloat.(V3))
        edge1 = second - first
        edge2 = third - first
        normal = cross(edge1, edge2)
        normal_norm = norm(normal)
        normal_norm > 0 ||
            throw(ArgumentError(
                "static-integral triangle must be nondegenerate"))
        scale = max(
            maximum(abs, edge1), maximum(abs, edge2),
            maximum(abs, point - first))
        scale > 0 ||
            throw(ArgumentError("static-integral geometry has zero extent"))
        return point, first, second, third, scale
    end
end

"""
    analytical_integral_1overR(P, V1, V2, V3)

Compute ∫_T 1/|P - r'| dS' analytically for observation point P and
flat triangle T = (V1, V2, V3).  Works for both coplanar and off-plane P.

Uses the Graglia (1993) / Wilton et al. (1984) formula:

  ∫_T 1/R dS' = Σ_edges d_i log[(s⁺+R⁺)/(s⁻+R⁻)]
                − |h| Σ_edges [atan(d_i s⁺/(R₀²+|h|R⁺)) − atan(d_i s⁻/(R₀²+|h|R⁻))]

where h = signed height of P above the triangle plane, R₀² = d_i² + h²,
and all other quantities are in-plane projections relative to the projection
of P onto the triangle plane.
"""
function analytical_integral_1overR(P::Vec3, V1::Vec3, V2::Vec3, V3::Vec3)
    geometry = _static_triangle_integral_fast_geometry(P, V1, V2, V3)
    geometry === nothing &&
        return _analytical_integral_1overR_big(P, V1, V2, V3)
    value = geometry.scale * _analytical_integral_1overR_unchecked(
        (P - geometry.origin) / geometry.scale,
        Vec3(0.0, 0.0, 0.0),
        (V2 - geometry.origin) / geometry.scale,
        (V3 - geometry.origin) / geometry.scale,
    )
    isfinite(value) ||
        throw(OverflowError(
            "analytical static triangle integral is non-finite"))
    return value
end


@noinline function _analytical_integral_1overR_big(
        P::Vec3, V1::Vec3, V2::Vec3, V3::Vec3)
    return setprecision(
            BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        point, first, second, third, scale =
            _static_triangle_integral_big_geometry(P, V1, V2, V3)
        value = scale * _analytical_integral_1overR_unchecked(
            (point - first) / scale,
            SVector{3,BigFloat}(0, 0, 0),
            (second - first) / scale,
            (third - first) / scale,
        )
        converted = Float64(value)
        isfinite(converted) ||
            throw(OverflowError(
                "analytical static triangle integral is non-finite"))
        converted
    end
end

function _analytical_integral_1overR_unchecked(
    P::SVector{3,T},
    V1::SVector{3,T},
    V2::SVector{3,T},
    V3::SVector{3,T},
) where {T<:AbstractFloat}
    n_T = cross(V2 - V1, V3 - V1)
    n_norm = norm(n_T)
    if iszero(n_norm)
        return 0.0
    end
    n_T = n_T / n_norm

    # Signed height and in-plane projection
    h = dot(P - V1, n_T)
    abs_h = abs(h)
    xi = P - h * n_T   # projection of P onto triangle plane

    edges = ((V1, V2), (V2, V3), (V3, V1))
    log_sum = zero(T)
    atan_sum = zero(T)

    for (A, B) in edges
        edge_vec = B - A
        edge_len = norm(edge_vec)
        if iszero(edge_len)
            continue
        end
        lhat = edge_vec / edge_len
        nhat = cross(lhat, n_T)

        d_i = dot(A - xi, nhat)

        s_minus = dot(A - xi, lhat)
        s_plus  = dot(B - xi, lhat)
        R_minus = norm(P - A)
        R_plus  = norm(P - B)
        R0_sq = d_i^2 + h^2

        # Log term: d_i * log[(s⁺ + R⁺)/(s⁻ + R⁻)]
        if !iszero(d_i)
            denom = s_minus + R_minus
            numer = s_plus + R_plus
            if denom > 0.0 && numer > 0.0
                log_sum += d_i * log(numer / denom)
            end
        end

        # Arctan term (only contributes when P is off-plane)
        if !iszero(abs_h) && !iszero(d_i)
            atan_plus  = atan(d_i * s_plus  / (R0_sq + abs_h * R_plus))
            atan_minus = atan(d_i * s_minus / (R0_sq + abs_h * R_minus))
            atan_sum += atan_plus - atan_minus
        end
    end

    return log_sum - abs_h * atan_sum
end

"""
    grad_analytical_integral_1overR(P, V1, V2, V3)

Analytical gradient (with respect to the observation point `P`) of the static
potential integral `S(P) = ∫_T 1/|P - r'| dS'` over the flat triangle
`T = (V1, V2, V3)`:

  ∇_P S(P) = -∫_T (P - r') / |P - r'|³ dS'

Closed form (Graglia 1993; Wilton et al. 1984), split into an in-plane part
and a part along the triangle normal `n̂`:

  ∇_P S = -Σ_edges û_i · log[(R⁺ + s⁺)/(R⁻ + s⁻)]
          - n̂ · sign(h) · Σ_edges [atan(d_i s⁺/(R₀²+|h|R⁺)) − atan(d_i s⁻/(R₀²+|h|R⁻))]

where `û_i = l̂_i × n̂` is the in-plane unit vector normal to edge `i`, `h` is the
signed height of `P` above the triangle plane, and the remaining quantities are
the same in-plane projections used by [`analytical_integral_1overR`](@ref).

Returns an `SVector{3,Float64}`. This is the gradient counterpart of the scalar
`analytical_integral_1overR`, used to subtract the `1/R²` singularity of the
mixed-potential scalar term `∫_T ∇_r G dS'` near the surface.
"""
function grad_analytical_integral_1overR(P::Vec3, V1::Vec3, V2::Vec3, V3::Vec3)
    geometry = _static_triangle_integral_fast_geometry(P, V1, V2, V3)
    geometry === nothing &&
        return _grad_analytical_integral_1overR_big(P, V1, V2, V3)
    # The gradient of this degree-one homogeneous potential is degree zero.
    value = _grad_analytical_integral_1overR_unchecked(
        (P - geometry.origin) / geometry.scale,
        Vec3(0.0, 0.0, 0.0),
        (V2 - geometry.origin) / geometry.scale,
        (V3 - geometry.origin) / geometry.scale,
    )
    all(isfinite, value) ||
        throw(OverflowError(
            "analytical static triangle-integral gradient is non-finite"))
    return value
end


@noinline function _grad_analytical_integral_1overR_big(
        P::Vec3, V1::Vec3, V2::Vec3, V3::Vec3)
    return setprecision(
            BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        point, first, second, third, scale =
            _static_triangle_integral_big_geometry(P, V1, V2, V3)
        value = _grad_analytical_integral_1overR_unchecked(
            (point - first) / scale,
            SVector{3,BigFloat}(0, 0, 0),
            (second - first) / scale,
            (third - first) / scale,
        )
        converted = Vec3(Float64.(value))
        all(isfinite, converted) ||
            throw(OverflowError(
                "analytical static triangle-integral gradient is non-finite"))
        converted
    end
end

function _grad_analytical_integral_1overR_unchecked(
    P::SVector{3,T},
    V1::SVector{3,T},
    V2::SVector{3,T},
    V3::SVector{3,T},
) where {T<:AbstractFloat}
    n_T = cross(V2 - V1, V3 - V1)
    n_norm = norm(n_T)
    if iszero(n_norm)
        return SVector{3,T}(zero(T), zero(T), zero(T))
    end
    n_T = n_T / n_norm

    h = dot(P - V1, n_T)
    abs_h = abs(h)
    sgn_h = h >= 0.0 ? 1.0 : -1.0
    xi = P - h * n_T   # projection of P onto triangle plane

    edges = ((V1, V2), (V2, V3), (V3, V1))
    tang = SVector{3,T}(zero(T), zero(T), zero(T))
    atan_sum = zero(T)

    for (A, B) in edges
        edge_vec = B - A
        edge_len = norm(edge_vec)
        if iszero(edge_len)
            continue
        end
        lhat = edge_vec / edge_len
        nhat = cross(lhat, n_T)

        d_i = dot(A - xi, nhat)

        s_minus = dot(A - xi, lhat)
        s_plus  = dot(B - xi, lhat)
        R_minus = norm(P - A)
        R_plus  = norm(P - B)
        R0_sq = d_i^2 + h^2

        # In-plane part: û_i log[(R⁺ + s⁺)/(R⁻ + s⁻)]
        denom = s_minus + R_minus
        numer = s_plus + R_plus
        if denom > 0.0 && numer > 0.0
            tang = tang + nhat * log(numer / denom)
        end

        # Normal part: arctan terms (vanish for P in-plane, where h = 0)
        if !iszero(abs_h) && !iszero(d_i)
            atan_plus  = atan(d_i * s_plus  / (R0_sq + abs_h * R_plus))
            atan_minus = atan(d_i * s_minus / (R0_sq + abs_h * R_minus))
            atan_sum += atan_plus - atan_minus
        end
    end

    return -tang - n_T * (sgn_h * atan_sum)
end

"""
    self_cell_contribution(mesh, rwg, m, n, tm,
                           quad_pts_tm, rwg_vals_m, rwg_vals_n,
                           div_m, div_n, Am, wq, k,
                           [wq_hi, quad_pts_tm_hi])

Compute the EFIE self-cell integral for basis functions m, n on the same
triangle tm using singularity extraction.

Returns `(vec_part - scl_part)` before multiplication by `-iωμ₀`.

The integral splits as:
  I = I_smooth  (product quadrature with G_smooth)
    + I_singular (outer quadrature, analytical inner ∫ 1/R dS')

When high-order quadrature data (wq_hi, quad_pts_tm_hi) is provided, the
self-cell uses it for both smooth and singular parts.  This matches the
adjacent-cell treatment and improves impedance matrix accuracy.
"""
function self_cell_contribution(
    mesh::TriMesh, rwg::RWGData,
    m_test::Int, n_src::Int, tm::Int,
    quad_pts_tm::Vector{Vec3},
    rwg_vals_m::Vector{<:SVector{3,<:Number}},
    rwg_vals_n::Vector{<:SVector{3,<:Number}},
    div_m::Number, div_n::Number,
    Am::Float64, wq, k,
    wq_hi, quad_pts_tm_hi::Vector{Vec3})

    _validate_mesh_rwg_pair(mesh, rwg)
    Nq = length(wq)
    Nq_hi = length(wq_hi)
    CT = complex(typeof(real(k)))

    V1 = _mesh_vertex(mesh, mesh.tri[1, tm])
    V2 = _mesh_vertex(mesh, mesh.tri[2, tm])
    V3 = _mesh_vertex(mesh, mesh.tri[3, tm])

    # ── Smooth part: high-order product quadrature with G_smooth ──
    # Self-cell G_smooth has a cusp at R=0; use high-order rule for accuracy
    val_smooth = zero(CT)
    for qm in 1:Nq_hi
        rm = quad_pts_tm_hi[qm]
        fm = eval_rwg(rwg, m_test, rm, tm)
        for qn in 1:Nq_hi
            rn = quad_pts_tm_hi[qn]
            fn = eval_rwg(rwg, n_src, rn, tm)

            Gs = _greens_smooth_unchecked(rm, rn, k)
            vec_part = dot(fm, fn) * Gs
            scl_part = conj(div_m) * div_n * Gs / (k^2)
            weight = wq_hi[qm] * wq_hi[qn] * (2 * Am) * (2 * Am)
            val_smooth += (vec_part - scl_part) * weight
        end
    end

    # ── Singular part: semi-analytical with high-order outer quadrature ──
    # For each outer quad point r_qm, compute:
    #   S = ∫_T 1/|r_qm - r'| dS'  (analytical)
    #
    # Vector part: ∫_T f_m·f_n/(4πR) dS'
    #   = f_m·f_n(r_qm) × S/(4π)
    #     + ∫_T f_m·[f_n(r') - f_n(r_qm)]/(4πR) dS'  (regular, high-order quad)
    #
    # Scalar part: div_m × div_n × S / (4πk²)
    inv4pi = 1.0 / (4π)

    val_singular = zero(CT)
    for qm in 1:Nq_hi
        rm = quad_pts_tm_hi[qm]
        fm = eval_rwg(rwg, m_test, rm, tm)

        S = analytical_integral_1overR(rm, V1, V2, V3)
        inner_scalar = inv4pi * S

        # Scalar potential singular part
        scl_sing = conj(div_m) * div_n * inner_scalar / (k^2)

        # Vector potential singular part: leading term
        fn_at_rm = eval_rwg(rwg, n_src, rm, tm)
        vec_lead = dot(fm, fn_at_rm) * inner_scalar

        # Vector potential singular part: remainder (bounded integrand)
        vec_rem = zero(CT)
        for qn in 1:Nq_hi
            rn = quad_pts_tm_hi[qn]
            fn = eval_rwg(rwg, n_src, rn, tm)

            R_vec = rm - rn
            R = hypot(hypot(R_vec[1], R_vec[2]), R_vec[3])
            if iszero(R)
                continue  # [f_n(rn) - f_n(rm)] = 0 when rn = rm
            end

            delta_fn = fn - fn_at_rm
            vec_rem += dot(fm, delta_fn) * (inv4pi / R) * wq_hi[qn] * (2 * Am)
        end

        outer_weight = wq_hi[qm] * (2 * Am)
        val_singular += ((vec_lead + vec_rem) - scl_sing) * outer_weight
    end

    return val_smooth + val_singular
end

# Backward-compatible method without high-order data (uses wq for both)
function self_cell_contribution(
    mesh::TriMesh, rwg::RWGData,
    m_test::Int, n_src::Int, tm::Int,
    quad_pts_tm::Vector{Vec3},
    rwg_vals_m::Vector{<:SVector{3,<:Number}},
    rwg_vals_n::Vector{<:SVector{3,<:Number}},
    div_m::Number, div_n::Number,
    Am::Float64, wq, k)

    return self_cell_contribution(mesh, rwg, m_test, n_src, tm,
        quad_pts_tm, rwg_vals_m, rwg_vals_n,
        div_m, div_n, Am, wq, k,
        wq, quad_pts_tm)
end

"""
    adjacent_cell_contribution(mesh, rwg, m_test, n_src, tm, tn,
                                quad_pts_tm, quad_pts_tn,
                                rwg_vals_m, rwg_vals_n,
                                div_m, div_n, Am, An,
                                wq, k,
                                wq_hi, quad_pts_tm_hi, quad_pts_tn_hi)

Compute the EFIE integral for adjacent triangle pairs (sharing an edge)
using singularity subtraction.

For coplanar triangles, splits G = G_smooth + 1/(4πR):
- Smooth part: standard Nq×Nq product quadrature with G_smooth (bounded)
- Singular 1/(4πR) part (uses high-order quadrature on BOTH triangles for symmetry):
  * Scalar potential: high-order outer quad × analytical inner `∫ 1/R dS'`
  * Vector potential: high-order outer quad × high-order inner quad with `f_n/R`

Returns `(vec_part - scl_part)` before multiplication by `-iωμ₀`.
"""
function adjacent_cell_contribution(
    mesh::TriMesh, rwg::RWGData,
    m_test::Int, n_src::Int, tm::Int, tn::Int,
    quad_pts_tm::Vector{Vec3},
    quad_pts_tn::Vector{Vec3},
    rwg_vals_m::Vector{<:SVector{3,<:Number}},
    rwg_vals_n::Vector{<:SVector{3,<:Number}},
    div_m::Number, div_n::Number,
    Am::Float64, An::Float64,
    wq, k,
    wq_hi, quad_pts_tm_hi::Vector{Vec3}, quad_pts_tn_hi::Vector{Vec3})

    _validate_mesh_rwg_pair(mesh, rwg)
    Nq = length(wq)
    Nq_hi = length(wq_hi)
    CT = complex(typeof(real(k)))

    V1n = _mesh_vertex(mesh, mesh.tri[1, tn])
    V2n = _mesh_vertex(mesh, mesh.tri[2, tn])
    V3n = _mesh_vertex(mesh, mesh.tri[3, tn])

    inv4pi = 1.0 / (4π)

    # ── Smooth part: standard product quadrature with G_smooth ──
    val_smooth = zero(CT)
    for qm in 1:Nq
        rm = quad_pts_tm[qm]
        fm = rwg_vals_m[qm]
        for qn in 1:Nq
            rn = quad_pts_tn[qn]
            fn = rwg_vals_n[qn]

            Gs = _greens_smooth_unchecked(rm, rn, k)
            vec_part = dot(fm, fn) * Gs
            scl_part = conj(div_m) * div_n * Gs / (k^2)
            weight = wq[qm] * wq[qn] * (2 * Am) * (2 * An)
            val_smooth += (vec_part - scl_part) * weight
        end
    end

    # ── Singular 1/(4πR) part: semi-analytical ──
    # Both outer and inner use high-order quadrature to preserve Z symmetry.
    # Scalar potential: analytical inner ∫1/R, high-order outer.
    # Vector potential: high-order outer × high-order inner with f_n/R.
    val_singular = zero(CT)
    for qm in 1:Nq_hi
        rm = quad_pts_tm_hi[qm]
        fm = eval_rwg(rwg, m_test, rm, tm)

        # Analytical inner integral: S = ∫_{T_n} 1/|rm - r'| dS'
        S = analytical_integral_1overR(rm, V1n, V2n, V3n)
        inner_scalar = inv4pi * S

        # Scalar potential singular part (exact via analytical integral)
        scl_sing = conj(div_m) * div_n * inner_scalar / (k^2)

        # Vector potential singular part: ∫_{T_n} f_n(r') / (4πR) dS'
        # Use high-order quadrature — the integrand is near-singular but bounded
        # (observation point rm is on the adjacent triangle, not on T_n itself)
        vec_sing = zero(CT)
        for qn in 1:Nq_hi
            rn = quad_pts_tn_hi[qn]
            fn_hi = eval_rwg(rwg, n_src, rn, tn)

            R_vec = rm - rn
            R = hypot(hypot(R_vec[1], R_vec[2]), R_vec[3])
            if iszero(R)
                continue
            end

            vec_sing += dot(fm, fn_hi) * (inv4pi / R) * wq_hi[qn] * (2 * An)
        end

        outer_weight = wq_hi[qm] * (2 * Am)
        val_singular += (vec_sing - scl_sing) * outer_weight
    end

    return val_smooth + val_singular
end
