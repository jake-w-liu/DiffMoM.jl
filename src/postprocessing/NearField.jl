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
const _DEFAULT_MAX_NEARFIELD_EXACT_WORK = 20_000_000
const _NEARFIELD_EXACT_PRECISION = 8704
const _NEARFIELD_TOTAL_CANCELLATION_THRESHOLD = sqrt(eps(Float64))

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

@inline function _nearfield_exact_point_work(
        nonzero_current_count::Int, quadrature_count::Int)
    # Each current has two triangle sides.  Charge every surface quadrature
    # value plus the two analytical terms used by the near-surface branch,
    # even when the observation takes the cheaper far-field branch.
    units = BigInt(2) * nonzero_current_count * (quadrature_count + 2)
    return units * _NEARFIELD_EXACT_PRECISION
end

@inline function _nearfield_total_reduction_requires_exact(
        total::CVec3, scattered::CVec3, incident::CVec3)
    @inbounds for component in 1:3
        combined = total[component]
        scattered_component = scattered[component]
        incident_component = incident[component]
        if !(isfinite(combined) && isfinite(scattered_component) &&
             isfinite(incident_component))
            return true
        end
        magnitude = abs(scattered_component) + abs(incident_component)
        isfinite(magnitude) || return true
        if !iszero(magnitude) &&
           abs(combined) <=
               _NEARFIELD_TOTAL_CANCELLATION_THRESHOLD * magnitude
            return true
        end
    end
    return false
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

@inline _nearfield_exact_subtract(first, second) =
    ntuple(component -> first[component] - second[component], 3)

@inline _nearfield_exact_dot(first, second) =
    first[1] * second[1] + first[2] * second[2] +
    first[3] * second[3]

@inline _nearfield_exact_cross(first, second) = (
    first[2] * second[3] - first[3] * second[2],
    first[3] * second[1] - first[1] * second[3],
    first[1] * second[2] - first[2] * second[1],
)

@inline function _nearfield_point_segment_within_exact(
        point, first, second, tolerance_squared)
    edge = _nearfield_exact_subtract(second, first)
    offset = _nearfield_exact_subtract(point, first)
    edge_squared = _nearfield_exact_dot(edge, edge)
    iszero(edge_squared) &&
        return _nearfield_exact_dot(offset, offset) <= tolerance_squared

    projection = _nearfield_exact_dot(offset, edge)
    projection <= 0 &&
        return _nearfield_exact_dot(offset, offset) <= tolerance_squared
    if projection >= edge_squared
        endpoint_offset = _nearfield_exact_subtract(point, second)
        return _nearfield_exact_dot(endpoint_offset, endpoint_offset) <=
               tolerance_squared
    end

    # Compare the squared point-to-line distance without dividing by the
    # rational segment parameter.
    residual_numerator = ntuple(component ->
        offset[component] * edge_squared -
        edge[component] * projection, 3)
    return _nearfield_exact_dot(
               residual_numerator, residual_numerator) <=
           tolerance_squared * edge_squared * edge_squared
end

function _nearfield_point_triangle_within_exact(
        point, first, second, third, tolerance_squared)
    edge_ab = _nearfield_exact_subtract(second, first)
    edge_ac = _nearfield_exact_subtract(third, first)
    point_offset = _nearfield_exact_subtract(point, first)
    normal = _nearfield_exact_cross(edge_ab, edge_ac)
    normal_squared = _nearfield_exact_dot(normal, normal)

    # When the orthogonal plane projection lies inside a nondegenerate
    # triangle, its plane distance is the global minimum. All remaining
    # closest points lie on one of the three closed edges.
    if normal_squared > 0
        d00 = _nearfield_exact_dot(edge_ab, edge_ab)
        d01 = _nearfield_exact_dot(edge_ab, edge_ac)
        d11 = _nearfield_exact_dot(edge_ac, edge_ac)
        d20 = _nearfield_exact_dot(point_offset, edge_ab)
        d21 = _nearfield_exact_dot(point_offset, edge_ac)
        denominator = d00 * d11 - d01 * d01
        coordinate_b = d11 * d20 - d01 * d21
        coordinate_c = d00 * d21 - d01 * d20
        if coordinate_b >= 0 && coordinate_c >= 0 &&
           coordinate_b + coordinate_c <= denominator
            plane_numerator = _nearfield_exact_dot(point_offset, normal)
            plane_numerator * plane_numerator <=
                tolerance_squared * normal_squared && return true
        end
    end

    return _nearfield_point_segment_within_exact(
               point, first, second, tolerance_squared) ||
           _nearfield_point_segment_within_exact(
               point, second, third, tolerance_squared) ||
           _nearfield_point_segment_within_exact(
               point, third, first, tolerance_squared)
end

@noinline function _surface_within_tolerance_exact(
        mesh::TriMesh, point::Vec3, tolerance::Float64)
    point_exact = ntuple(
        component -> Rational{BigInt}(point[component]), 3)
    tolerance_exact = Rational{BigInt}(tolerance)
    tolerance_squared = tolerance_exact * tolerance_exact
    @inbounds for triangle in 1:ntriangles(mesh)
        first = ntuple(component -> Rational{BigInt}(
            mesh.xyz[component, mesh.tri[1, triangle]]), 3)
        second = ntuple(component -> Rational{BigInt}(
            mesh.xyz[component, mesh.tri[2, triangle]]), 3)
        third = ntuple(component -> Rational{BigInt}(
            mesh.xyz[component, mesh.tri[3, triangle]]), 3)
        _nearfield_point_triangle_within_exact(
            point_exact, first, second, third, tolerance_squared) &&
            return true
    end
    return false
end

@inline function _surface_within_tolerance(
        mesh::TriMesh, point::Vec3, tolerance::Float64)
    distance = _surface_distance(mesh, point)
    isfinite(distance) || return false, distance
    scale = max(distance, tolerance)
    uncertainty = 64 * eps(scale)
    separation = abs(distance - tolerance)
    if isfinite(uncertainty) && separation > uncertainty
        return distance <= tolerance, distance
    end
    return _surface_within_tolerance_exact(mesh, point, tolerance), distance
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

@noinline function _nearfield_weighted_green_exact(
    observation::Vec3,
    source::Vec3,
    k,
    area::Float64,
    quadrature_weight::Float64,
)
    return setprecision(BigFloat, _GREEN_FALLBACK_PRECISION) do
        dx = BigFloat(observation[1]) - BigFloat(source[1])
        dy = BigFloat(observation[2]) - BigFloat(source[2])
        dz = BigFloat(observation[3]) - BigFloat(source[3])
        distance = sqrt(dx * dx + dy * dy + dz * dz)
        iszero(distance) && return 0.0 + 0.0im
        phase_rate = Complex{BigFloat}(
            BigFloat(imag(k)), -BigFloat(real(k)))
        surface_weight =
            2 * BigFloat(area) * BigFloat(quadrature_weight)
        value = ComplexF64(
            surface_weight * exp(phase_rate * distance) /
            (4 * BigFloat(π) * distance))
        isfinite(value) ||
            throw(OverflowError(
                "weighted near-field Green function is outside the " *
                "ComplexF64 range"))
        return value
    end
end

@inline function _nearfield_weighted_green(
    observation::Vec3,
    source::Vec3,
    k,
    area::Float64,
    quadrature_weight::Float64,
)
    surface_weight = area * (2 * quadrature_weight)
    green = _greens_unchecked(observation, source, k)
    value = surface_weight * green
    if isfinite(value) &&
       !(iszero(value) && !iszero(area) &&
         !iszero(quadrature_weight) && !iszero(green))
        return value
    end
    return _nearfield_weighted_green_exact(
        observation, source, k, area, quadrature_weight)
end

@noinline function _nearfield_weighted_smooth_green_exact(
    observation::Vec3,
    source::Vec3,
    k,
    area::Float64,
    quadrature_weight::Float64,
)
    return setprecision(BigFloat, _GREEN_FALLBACK_PRECISION) do
        dx = BigFloat(observation[1]) - BigFloat(source[1])
        dy = BigFloat(observation[2]) - BigFloat(source[2])
        dz = BigFloat(observation[3]) - BigFloat(source[3])
        distance = sqrt(dx * dx + dy * dy + dz * dz)
        phase_rate = Complex{BigFloat}(
            BigFloat(imag(k)), -BigFloat(real(k)))
        surface_weight =
            2 * BigFloat(area) * BigFloat(quadrature_weight)
        value = if iszero(distance)
            ComplexF64(surface_weight * phase_rate / (4 * BigFloat(π)))
        else
            ComplexF64(
                surface_weight * expm1(phase_rate * distance) /
                (4 * BigFloat(π) * distance))
        end
        isfinite(value) ||
            throw(OverflowError(
                "weighted smooth near-field Green function is outside " *
                "the ComplexF64 range"))
        return value
    end
end

@inline function _nearfield_weighted_smooth_green(
    observation::Vec3,
    source::Vec3,
    k,
    area::Float64,
    quadrature_weight::Float64,
)
    surface_weight = area * (2 * quadrature_weight)
    green = _greens_smooth_unchecked(observation, source, k)
    value = surface_weight * green
    if isfinite(value) &&
       !(iszero(value) && !iszero(area) &&
         !iszero(quadrature_weight) && !iszero(green))
        return value
    end
    return _nearfield_weighted_smooth_green_exact(
        observation, source, k, area, quadrature_weight)
end

@noinline function _nearfield_weighted_grad_green_exact(
    observation::Vec3,
    source::Vec3,
    k,
    area::Float64,
    quadrature_weight::Float64,
)
    return setprecision(BigFloat, _GREEN_FALLBACK_PRECISION) do
        dx = BigFloat(observation[1]) - BigFloat(source[1])
        dy = BigFloat(observation[2]) - BigFloat(source[2])
        dz = BigFloat(observation[3]) - BigFloat(source[3])
        distance = sqrt(dx * dx + dy * dy + dz * dz)
        if iszero(distance)
            return CVec3(0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im)
        end
        phase_rate = Complex{BigFloat}(
            BigFloat(imag(k)), -BigFloat(real(k)))
        phase_argument = phase_rate * distance
        surface_weight =
            2 * BigFloat(area) * BigFloat(quadrature_weight)
        radial_factor = surface_weight * exp(phase_argument) *
                        (phase_argument - one(phase_argument)) /
                        (4 * BigFloat(π) * distance * distance)
        zero_value = zero(phase_argument)
        value = CVec3(
            ComplexF64(iszero(dx) ? zero_value :
                       radial_factor * (dx / distance)),
            ComplexF64(iszero(dy) ? zero_value :
                       radial_factor * (dy / distance)),
            ComplexF64(iszero(dz) ? zero_value :
                       radial_factor * (dz / distance)),
        )
        all(isfinite, value) ||
            throw(OverflowError(
                "weighted near-field Green-function gradient is outside " *
                "the ComplexF64 range"))
        return value
    end
end

@inline function _nearfield_weighted_grad_green(
    observation::Vec3,
    source::Vec3,
    k,
    area::Float64,
    quadrature_weight::Float64,
)
    surface_weight = area * (2 * quadrature_weight)
    gradient = _grad_greens_unchecked(observation, source, k)
    value = surface_weight * gradient
    input_weight_is_nonzero =
        !iszero(area) && !iszero(quadrature_weight)
    preserved_components =
        (!iszero(value[1]) || !input_weight_is_nonzero ||
         iszero(gradient[1])) &&
        (!iszero(value[2]) || !input_weight_is_nonzero ||
         iszero(gradient[2])) &&
        (!iszero(value[3]) || !input_weight_is_nonzero ||
         iszero(gradient[3]))
    if all(isfinite, value) && preserved_components
        return value
    end
    return _nearfield_weighted_grad_green_exact(
        observation, source, k, area, quadrature_weight)
end

@noinline function _nearfield_weighted_smooth_grad_green_exact(
    observation::Vec3,
    source::Vec3,
    k,
    area::Float64,
    quadrature_weight::Float64,
)
    return setprecision(BigFloat, _GREEN_FALLBACK_PRECISION) do
        dx = BigFloat(observation[1]) - BigFloat(source[1])
        dy = BigFloat(observation[2]) - BigFloat(source[2])
        dz = BigFloat(observation[3]) - BigFloat(source[3])
        distance = sqrt(dx * dx + dy * dy + dz * dz)
        if iszero(distance)
            return CVec3(0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im)
        end
        phase_rate = Complex{BigFloat}(
            BigFloat(imag(k)), -BigFloat(real(k)))
        phase_argument = phase_rate * distance
        surface_weight =
            2 * BigFloat(area) * BigFloat(quadrature_weight)
        radial_factor = surface_weight * phase_rate * phase_rate /
                        (4 * BigFloat(π)) *
                        _green_smooth_radial_ratio(phase_argument)
        zero_value = zero(phase_argument)
        value = CVec3(
            ComplexF64(iszero(dx) ? zero_value :
                       radial_factor * (dx / distance)),
            ComplexF64(iszero(dy) ? zero_value :
                       radial_factor * (dy / distance)),
            ComplexF64(iszero(dz) ? zero_value :
                       radial_factor * (dz / distance)),
        )
        all(isfinite, value) ||
            throw(OverflowError(
                "weighted smooth near-field Green-function gradient is " *
                "outside the ComplexF64 range"))
        return value
    end
end

@inline function _nearfield_weighted_smooth_grad_green(
    observation::Vec3,
    source::Vec3,
    k,
    area::Float64,
    quadrature_weight::Float64,
)
    surface_weight = area * (2 * quadrature_weight)
    gradient = _grad_greens_smooth_unchecked(observation, source, k)
    value = surface_weight * gradient
    input_weight_is_nonzero =
        !iszero(area) && !iszero(quadrature_weight)
    preserved_components =
        (!iszero(value[1]) || !input_weight_is_nonzero ||
         iszero(gradient[1])) &&
        (!iszero(value[2]) || !input_weight_is_nonzero ||
         iszero(gradient[2])) &&
        (!iszero(value[3]) || !input_weight_is_nonzero ||
         iszero(gradient[3]))
    if all(isfinite, value) && preserved_components
        return value
    end
    return _nearfield_weighted_smooth_grad_green_exact(
        observation, source, k, area, quadrature_weight)
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
    _validate_mesh_rwg_pair(mesh, rwg)
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
            within_tolerance, d = _surface_within_tolerance(mesh, p, tol)
            if within_tolerance
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
        h_t_all[t] = sqrt(areas[t]) * sqrt(2.0)
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
                    weighted_smooth_green =
                        _nearfield_weighted_smooth_green(
                            robs, rq, k, At, wq[q])
                    Jq = J_samples[q, t]

                    # Vector smooth part
                    Ex += pref_vec * Jq[1] * weighted_smooth_green
                    Ey += pref_vec * Jq[2] * weighted_smooth_green
                    Ez += pref_vec * Jq[3] * weighted_smooth_green

                    # Vector singular remainder: [J(rq) − J(r'_*)]/(4πR)
                    Rv = robs - rq
                    R = hypot(hypot(Rv[1], Rv[2]), Rv[3])
                    if !iszero(R)
                        dJ = Jq - J_star
                        crem = (At / R) * (2 * wq[q] * inv4pi)
                        Ex += pref_vec * dJ[1] * crem
                        Ey += pref_vec * dJ[2] * crem
                        Ez += pref_vec * dJ[3] * crem
                    end

                    if abs(divt) > 0.0
                        # Evaluate ∇G_smooth directly.  Subtracting its two
                        # O(1/R²) constituents can overflow independently or
                        # cancel all significant bits even though their smooth
                        # remainder is finite.
                        weighted_smooth_gradient =
                            _nearfield_weighted_smooth_grad_green(
                                robs, rq, k, At, wq[q])
                        Ex += pref_scl * divt * weighted_smooth_gradient[1]
                        Ey += pref_scl * divt * weighted_smooth_gradient[2]
                        Ez += pref_scl * divt * weighted_smooth_gradient[3]
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
                    weighted_green = _nearfield_weighted_green(
                        robs, rq, k, At, wq[q])
                    Jq = J_samples[q, t]

                    Ex += pref_vec * Jq[1] * weighted_green
                    Ey += pref_vec * Jq[2] * weighted_green
                    Ez += pref_vec * Jq[3] * weighted_green

                    if abs(divt) > 0.0
                        weighted_gradient = _nearfield_weighted_grad_green(
                            robs, rq, k, At, wq[q])
                        Ex += pref_scl * divt * weighted_gradient[1]
                        Ey += pref_scl * divt * weighted_gradient[2]
                        Ez += pref_scl * divt * weighted_gradient[3]
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

@inline function _nearfield_incident_big(
        ::AbstractExcitation, ::Vec3, cached::CVec3)
    return SVector{3,Complex{BigFloat}}(
        Complex{BigFloat}(cached[1]),
        Complex{BigFloat}(cached[2]),
        Complex{BigFloat}(cached[3]),
    )
end

@inline function _nearfield_incident_big(
        excitation::PlaneWaveExcitation, observation::Vec3, ::CVec3)
    phase_argument = zero(BigFloat)
    @inbounds for component in 1:3
        phase_argument += BigFloat(excitation.k_vec[component]) *
                          BigFloat(observation[component])
    end
    phase = exp(Complex{BigFloat}(zero(BigFloat), -phase_argument))
    amplitude = BigFloat(excitation.E0)
    return SVector{3,Complex{BigFloat}}(ntuple(component ->
        amplitude * BigFloat(excitation.pol[component]) * phase, 3))
end

@inline function _nearfield_big_vertex(mesh::TriMesh, vertex::Int)
    value = _mesh_vertex(mesh, vertex)
    return SVector{3,BigFloat}(
        BigFloat(value[1]), BigFloat(value[2]), BigFloat(value[3]))
end

@inline function _nearfield_big_quadrature_point(
        first::SVector{3,BigFloat},
        second::SVector{3,BigFloat},
        third::SVector{3,BigFloat},
        reference::SVector{2,Float64})
    xi_first = BigFloat(reference[1])
    xi_second = BigFloat(reference[2])
    return first * (1 - xi_first - xi_second) +
           second * xi_first + third * xi_second
end

@inline function _nearfield_big_green_values(
        observation::SVector{3,BigFloat},
        source::SVector{3,BigFloat},
        phase_rate::Complex{BigFloat})
    offset = observation - source
    distance = norm(offset)
    if iszero(distance)
        zero_complex = zero(phase_rate)
        smooth_green = phase_rate / (4 * BigFloat(pi))
        return zero_complex, smooth_green,
               SVector{3,Complex{BigFloat}}(
                   zero_complex, zero_complex, zero_complex),
               SVector{3,Complex{BigFloat}}(
                   zero_complex, zero_complex, zero_complex),
               offset, distance
    end

    inv_four_pi = inv(4 * BigFloat(pi))
    phase_argument = phase_rate * distance
    inverse_distance = inv(distance)
    green = exp(phase_argument) * inv_four_pi * inverse_distance
    smooth_green = expm1(phase_argument) * inv_four_pi * inverse_distance
    direction = offset * inverse_distance
    gradient =
        ((phase_rate - inverse_distance) * green) * direction
    smooth_radial =
        (exp(phase_argument) * (phase_argument - 1) + 1) *
        inv_four_pi * inverse_distance * inverse_distance
    smooth_gradient = smooth_radial * direction
    return green, smooth_green, gradient, smooth_gradient,
           offset, distance
end

@noinline function _compute_total_field_point_exact(
        mesh::TriMesh,
        rwg::RWGData,
        I_coeffs::AbstractVector{<:Number},
        excitation::AbstractExcitation,
        observation::Vec3,
        cached_incident::CVec3,
        k,
        eta0::Float64,
        quad_order::Int,
        observation_index::Int)
    return setprecision(BigFloat, _NEARFIELD_EXACT_PRECISION) do
        complex_big = Complex{BigFloat}
        observation_big = SVector{3,BigFloat}(
            BigFloat(observation[1]),
            BigFloat(observation[2]),
            BigFloat(observation[3]))
        k_big = complex_big(k)
        eta_big = BigFloat(eta0)
        phase_rate = -complex_big(0, 1) * k_big
        vector_prefactor = phase_rate * eta_big
        scalar_prefactor = -complex_big(0, 1) * eta_big / k_big
        inverse_four_pi = inv(4 * BigFloat(pi))
        xi, quadrature_weights = tri_quad_rule(quad_order)
        quadrature_count = length(quadrature_weights)
        total = zeros(complex_big, 3)

        @inbounds for basis in 1:rwg.nedges
            current = complex_big(ComplexF64(I_coeffs[basis]))
            iszero(current) && continue

            for is_plus in (true, false)
                triangle = is_plus ?
                    rwg.tplus[basis] : rwg.tminus[basis]
                first = _nearfield_big_vertex(
                    mesh, mesh.tri[1, triangle])
                second = _nearfield_big_vertex(
                    mesh, mesh.tri[2, triangle])
                third = _nearfield_big_vertex(
                    mesh, mesh.tri[3, triangle])
                opposite_vertex = is_plus ?
                    rwg.vplus_opp[basis] : rwg.vminus_opp[basis]
                opposite = _nearfield_big_vertex(mesh, opposite_vertex)
                coefficient = complex_big(is_plus ?
                    rwg.coeff_plus[basis] : rwg.coeff_minus[basis])
                area_float = triangle_area(mesh, triangle)
                area = BigFloat(area_float)
                edge_length = BigFloat(rwg.len[basis])
                basis_scale = coefficient * edge_length / (2 * area)
                divergence = current * coefficient * edge_length / area
                is_plus || (divergence = -divergence)

                distance_float = _point_triangle_distance(
                    observation,
                    _mesh_vertex(mesh, mesh.tri[1, triangle]),
                    _mesh_vertex(mesh, mesh.tri[2, triangle]),
                    _mesh_vertex(mesh, mesh.tri[3, triangle]))
                characteristic_length = sqrt(area_float) * sqrt(2.0)
                near_surface =
                    distance_float < characteristic_length / quadrature_count

                if near_surface
                    static_integral =
                        _analytical_integral_1overR_unchecked(
                            observation_big, first, second, third)
                    normal = cross(second - first, third - first)
                    normal /= norm(normal)
                    projection_height =
                        dot(observation_big - first, normal)
                    projected_observation =
                        observation_big - projection_height * normal
                    star_delta = is_plus ?
                        projected_observation - opposite :
                        opposite - projected_observation
                    current_at_projection =
                        current * basis_scale * star_delta

                    for quadrature in eachindex(quadrature_weights)
                        source = _nearfield_big_quadrature_point(
                            first, second, third, xi[quadrature])
                        _, smooth_green, _, smooth_gradient,
                            offset, distance = _nearfield_big_green_values(
                                observation_big, source, phase_rate)
                        source_delta = is_plus ?
                            source - opposite : opposite - source
                        current_at_source =
                            current * basis_scale * source_delta
                        surface_weight =
                            2 * area * BigFloat(
                                quadrature_weights[quadrature])

                        for component in 1:3
                            total[component] +=
                                vector_prefactor *
                                current_at_source[component] *
                                (surface_weight * smooth_green)
                        end

                        if !iszero(distance)
                            remainder_weight = surface_weight *
                                inverse_four_pi / distance
                            current_difference =
                                current_at_source - current_at_projection
                            for component in 1:3
                                total[component] +=
                                    vector_prefactor *
                                    current_difference[component] *
                                    remainder_weight
                            end
                        end

                        if !iszero(divergence)
                            for component in 1:3
                                total[component] +=
                                    scalar_prefactor * divergence *
                                    (surface_weight *
                                     smooth_gradient[component])
                            end
                        end
                    end

                    singular_weight = inverse_four_pi * static_integral
                    for component in 1:3
                        total[component] +=
                            vector_prefactor *
                            current_at_projection[component] *
                            singular_weight
                    end
                    if !iszero(divergence)
                        static_gradient =
                            _grad_analytical_integral_1overR_unchecked(
                                observation_big, first, second, third)
                        scalar_scale = scalar_prefactor * divergence *
                                       inverse_four_pi
                        for component in 1:3
                            total[component] +=
                                scalar_scale * static_gradient[component]
                        end
                    end
                else
                    for quadrature in eachindex(quadrature_weights)
                        source = _nearfield_big_quadrature_point(
                            first, second, third, xi[quadrature])
                        green, _, gradient, _, _, _ =
                            _nearfield_big_green_values(
                                observation_big, source, phase_rate)
                        source_delta = is_plus ?
                            source - opposite : opposite - source
                        current_at_source =
                            current * basis_scale * source_delta
                        surface_weight =
                            2 * area * BigFloat(
                                quadrature_weights[quadrature])
                        for component in 1:3
                            total[component] +=
                                vector_prefactor *
                                current_at_source[component] *
                                (surface_weight * green)
                            if !iszero(divergence)
                                total[component] +=
                                    scalar_prefactor * divergence *
                                    (surface_weight * gradient[component])
                            end
                        end
                    end
                end
            end
        end

        incident = _nearfield_incident_big(
            excitation, observation, cached_incident)
        return CVec3(ntuple(component -> begin
            converted = ComplexF64(total[component] + incident[component])
            isfinite(converted) ||
                throw(OverflowError(
                    "compute_total_field result is outside the " *
                    "ComplexF64 range at observation $observation_index, " *
                    "component $component"))
            converted
        end, 3))
    end
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
                                         _DEFAULT_MAX_NEARFIELD_INTERACTION_TERMS,
                                     max_exact_work::Integer=
                                         _DEFAULT_MAX_NEARFIELD_EXACT_WORK)
    _validate_incident_electric_field_wavenumber(excitation, k)
    exact_limit = _validated_resource_limit(
        "max_exact_work", max_exact_work)
    E_total = _compute_nearfield_matrix(mesh, rwg, I_coeffs, observation_points, k;
                                        quad_order=quad_order,
                                        eta0=eta0,
                                        check_surface=check_surface,
                                        surface_tol=surface_tol,
                                        max_work_bytes=max_work_bytes,
                                        max_interaction_terms=max_interaction_terms)
    nonzero_current_count = count(current ->
        !iszero(ComplexF64(current)), I_coeffs)
    quadrature_count = _nearfield_quadrature_count(quad_order)
    point_exact_work = _nearfield_exact_point_work(
        nonzero_current_count, quadrature_count)
    exact_work = BigInt(0)
    @inbounds for i in eachindex(observation_points)
        E_inc = _check_finite_cvec3(
            _incident_electric_field(excitation, observation_points[i], k),
            "incident electric field",
        )
        scattered = CVec3(
            E_total[1, i], E_total[2, i], E_total[3, i])
        combined = CVec3(
            scattered[1] + E_inc[1],
            scattered[2] + E_inc[2],
            scattered[3] + E_inc[3])
        if _nearfield_total_reduction_requires_exact(
                combined, scattered, E_inc)
            next_exact_work = exact_work + point_exact_work
            next_exact_work <= exact_limit ||
                throw(ArgumentError(
                    "compute_total_field exact retries require " *
                    "$next_exact_work precision-weighted terms, " *
                    "exceeding max_exact_work=$exact_limit"))
            exact_work = next_exact_work
            combined = _compute_total_field_point_exact(
                mesh, rwg, I_coeffs, excitation,
                observation_points[i], E_inc, k, eta0,
                quad_order, i)
        end
        E_total[1, i] = combined[1]
        E_total[2, i] = combined[2]
        E_total[3, i] = combined[3]
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
    _validate_mesh_rwg_pair(mesh, rwg)
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
    _validate_mesh_rwg_pair(mesh, rwg)
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
- `max_exact_work=20_000_000`: maximum precision-weighted basis-side terms
  used by exceptional coupled incident/scattered cancellation retries

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
                                 _DEFAULT_MAX_NEARFIELD_INTERACTION_TERMS,
                             max_exact_work::Integer=
                                 _DEFAULT_MAX_NEARFIELD_EXACT_WORK)
    E = _compute_total_field_matrix(mesh, rwg, I_coeffs, excitation, [observation_point], k;
                                    quad_order=quad_order,
                                    eta0=eta0,
                                    check_surface=check_surface,
                                    surface_tol=surface_tol,
                                    max_work_bytes=max_work_bytes,
                                    max_interaction_terms=max_interaction_terms,
                                    max_exact_work=max_exact_work)
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
                                 _DEFAULT_MAX_NEARFIELD_INTERACTION_TERMS,
                             max_exact_work::Integer=
                                 _DEFAULT_MAX_NEARFIELD_EXACT_WORK)
    _validate_mesh_rwg_pair(mesh, rwg)
    obs = _prepare_nearfield_observations(
        observation_points, max_work_bytes)
    return _compute_total_field_matrix(mesh, rwg, I_coeffs, excitation, obs, k;
                                       quad_order=quad_order,
                                       eta0=eta0,
                                       check_surface=check_surface,
                                       surface_tol=surface_tol,
                                       max_work_bytes=max_work_bytes,
                                       max_interaction_terms=max_interaction_terms,
                                       max_exact_work=max_exact_work)
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
                                 _DEFAULT_MAX_NEARFIELD_INTERACTION_TERMS,
                             max_exact_work::Integer=
                                 _DEFAULT_MAX_NEARFIELD_EXACT_WORK)
    _validate_mesh_rwg_pair(mesh, rwg)
    obs = _prepare_nearfield_observations(
        observation_points, max_work_bytes)
    return _compute_total_field_matrix(mesh, rwg, I_coeffs, excitation, obs, k;
                                       quad_order=quad_order,
                                       eta0=eta0,
                                       check_surface=check_surface,
                                       surface_tol=surface_tol,
                                       max_work_bytes=max_work_bytes,
                                       max_interaction_terms=max_interaction_terms,
                                       max_exact_work=max_exact_work)
end
