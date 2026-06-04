# SingularIntegrals.jl — Singularity extraction for EFIE self-cell integrals
#
# When source and observation triangles coincide (self-cell), the Green's
# function G = exp(-ikR)/(4πR) has a 1/R singularity that standard Gaussian
# quadrature cannot resolve.  We split G = G_smooth + 1/(4πR) and compute:
#   - Smooth part: standard product quadrature with G_smooth (bounded)
#   - Singular part: semi-analytical (outer quadrature, analytical inner)

export analytical_integral_1overR, self_cell_contribution, adjacent_cell_contribution

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
    n_T = cross(V2 - V1, V3 - V1)
    n_norm = norm(n_T)
    if n_norm < 1e-30
        return 0.0
    end
    n_T = n_T / n_norm

    # Signed height and in-plane projection
    h = dot(P - V1, n_T)
    abs_h = abs(h)
    xi = P - h * n_T   # projection of P onto triangle plane

    edges = ((V1, V2), (V2, V3), (V3, V1))
    log_sum = 0.0
    atan_sum = 0.0

    for (A, B) in edges
        edge_vec = B - A
        edge_len = norm(edge_vec)
        if edge_len < 1e-30
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
        if abs(d_i) > 1e-15
            denom = s_minus + R_minus
            numer = s_plus + R_plus
            if abs(denom) > 1e-30 && abs(numer) > 1e-30
                log_sum += d_i * log(numer / denom)
            end
        end

        # Arctan term (only contributes when P is off-plane)
        if abs_h > 1e-15 && abs(d_i) > 1e-15
            atan_plus  = atan(d_i * s_plus  / (R0_sq + abs_h * R_plus))
            atan_minus = atan(d_i * s_minus / (R0_sq + abs_h * R_minus))
            atan_sum += atan_plus - atan_minus
        end
    end

    return log_sum - abs_h * atan_sum
end

"""
    self_cell_contribution(mesh, rwg, m, n, tm,
                           quad_pts_tm, rwg_vals_m, rwg_vals_n,
                           div_m, div_n, Am, wq, k,
                           [wq_hi, quad_pts_tm_hi])

Compute the EFIE self-cell integral for basis functions m, n on the same
triangle tm using singularity extraction.

Returns the value (vec_part - scl_part), not yet multiplied by -iωμ₀.

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

            Gs = greens_smooth(rm, rn, k)
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
            R = sqrt(dot(R_vec, R_vec))
            if R < 1e-14
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

Returns the value (vec_part - scl_part), not yet multiplied by -iωμ₀.
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

            Gs = greens_smooth(rm, rn, k)
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
            R = sqrt(dot(R_vec, R_vec))
            if R < 1e-14
                continue
            end

            vec_sing += dot(fm, fn_hi) * (inv4pi / R) * wq_hi[qn] * (2 * An)
        end

        outer_weight = wq_hi[qm] * (2 * Am)
        val_singular += (vec_sing - scl_sing) * outer_weight
    end

    return val_smooth + val_singular
end
