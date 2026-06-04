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

@inline function _point_triangle_distance(p::Vec3, a::Vec3, b::Vec3, c::Vec3)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    if d1 <= 0.0 && d2 <= 0.0
        return norm(ap)
    end

    bp = p - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    if d3 >= 0.0 && d4 <= d3
        return norm(bp)
    end

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0
        v = d1 / (d1 - d3)
        proj = a + v * ab
        return norm(p - proj)
    end

    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)
    if d6 >= 0.0 && d5 <= d6
        return norm(cp)
    end

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0
        w = d2 / (d2 - d6)
        proj = a + w * ac
        return norm(p - proj)
    end

    va = d3 * d6 - d5 * d4
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        proj = b + w * (c - b)
        return norm(p - proj)
    end

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    proj = a + v * ab + w * ac
    return norm(p - proj)
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
    return max(1e-12, 1e-10 * _bbox_diagonal(mesh))
end

function _collect_observation_points(points::AbstractVector{<:Vec3})
    return collect(points)
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

    @inbounds for t in 1:Nt
        quad_pts[t] = tri_quad_points(mesh, t, xi)
        areas[t] = triangle_area(mesh, t)
        divt = 0.0 + 0im

        for n in tri_to_basis[t]
            In = ComplexF64(I_coeffs[n])
            divt += In * div_rwg(rwg, n, t)
            for q in 1:Nq
                J_samples[q, t] += In * eval_rwg(rwg, n, quad_pts[t][q], t)
            end
        end

        div_samples[t] = divt
    end

    return quad_pts, areas, J_samples, div_samples
end

function _compute_nearfield_matrix(mesh::TriMesh, rwg::RWGData,
                                   I_coeffs::AbstractVector{<:Number},
                                   observation_points::Vector{Vec3}, k;
                                   quad_order::Int=3,
                                   eta0::Float64=376.730313668,
                                   check_surface::Bool=true,
                                   surface_tol::Union{Nothing,Float64}=nothing)
    length(I_coeffs) == rwg.nedges ||
        throw(DimensionMismatch("I_coeffs length $(length(I_coeffs)) != rwg.nedges=$(rwg.nedges)."))
    abs(k) > 0 || error("compute_nearfield: k must be nonzero.")

    tol = isnothing(surface_tol) ? _default_nearfield_surface_tol(mesh) : surface_tol
    tol >= 0 || error("compute_nearfield: surface_tol must be nonnegative.")

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

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)
    quad_pts, areas, J_samples, div_samples =
        _precompute_nearfield_triangle_data(mesh, rwg, I_coeffs, xi)

    Nobs = length(observation_points)
    Nt = ntriangles(mesh)
    E = zeros(ComplexF64, 3, Nobs)
    pref_vec = -1im * k * eta0
    pref_scl = -1im * eta0 / k

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

    wq_sum_inv = 1.0 / sum(wq[qq] for qq in 1:Nq)

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
                # ── Vector potential: singularity subtraction ──
                S = analytical_integral_1overR(robs, V1, V2, V3)

                for q in 1:Nq
                    rq = quad_pts[t][q]
                    wt = wq[q] * (2 * At)
                    Gs = greens_smooth(robs, rq, k)
                    Jq = J_samples[q, t]

                    Ex += pref_vec * Jq[1] * (wt * Gs)
                    Ey += pref_vec * Jq[2] * (wt * Gs)
                    Ez += pref_vec * Jq[3] * (wt * Gs)

                    if abs(divt) > 0.0
                        gradG = grad_greens(robs, rq, k)
                        Ex += pref_scl * divt * (wt * gradG[1])
                        Ey += pref_scl * divt * (wt * gradG[2])
                        Ez += pref_scl * divt * (wt * gradG[3])
                    end
                end

                # Vector potential singular part: J_avg · ∫ 1/(4πR) dS'
                J_avg = SVector{3,ComplexF64}(0.0, 0.0, 0.0)
                for q in 1:Nq
                    J_avg = J_avg + J_samples[q, t] * wq[q]
                end
                J_avg = J_avg * wq_sum_inv
                Ex += pref_vec * J_avg[1] * (inv4pi * S)
                Ey += pref_vec * J_avg[2] * (inv4pi * S)
                Ez += pref_vec * J_avg[3] * (inv4pi * S)
            else
                # ── Standard quadrature (far from surface) ──
                for q in 1:Nq
                    rq = quad_pts[t][q]
                    wt = wq[q] * (2 * At)
                    G = greens(robs, rq, k)
                    Jq = J_samples[q, t]

                    Ex += pref_vec * Jq[1] * (wt * G)
                    Ey += pref_vec * Jq[2] * (wt * G)
                    Ez += pref_vec * Jq[3] * (wt * G)

                    if abs(divt) > 0.0
                        gradG = grad_greens(robs, rq, k)
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

    return E
end

function _compute_incident_field_matrix(excitation::AbstractExcitation,
                                        observation_points::Vector{Vec3}, k)
    _validate_incident_electric_field_wavenumber(excitation, k)
    Nobs = length(observation_points)
    E = zeros(ComplexF64, 3, Nobs)
    @inbounds for i in 1:Nobs
        E[:, i] .= _incident_electric_field(excitation, observation_points[i], k)
    end
    return E
end

function _compute_total_field_matrix(mesh::TriMesh, rwg::RWGData,
                                     I_coeffs::AbstractVector{<:Number},
                                     excitation::AbstractExcitation,
                                     observation_points::Vector{Vec3}, k;
                                     quad_order::Int=3,
                                     eta0::Float64=376.730313668,
                                     check_surface::Bool=true,
                                     surface_tol::Union{Nothing,Float64}=nothing)
    E_inc = _compute_incident_field_matrix(excitation, observation_points, k)
    E_sca = _compute_nearfield_matrix(mesh, rwg, I_coeffs, observation_points, k;
                                      quad_order=quad_order,
                                      eta0=eta0,
                                      check_surface=check_surface,
                                      surface_tol=surface_tol)
    return E_inc + E_sca
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
  `max(1e-12, 1e-10 * bbox_diagonal(mesh))`

# Returns
- Single-point input: `CVec3`
- Multi-point input: `Matrix{ComplexF64}` of size `(3, Nobs)`

# Limitations
- This is a direct quadrature evaluator for observation points away from the
  surface.
- On-surface evaluation is not supported.
- Very near-surface points can require higher `quad_order`; dedicated
  near-singular quadrature is not implemented yet.
"""
function compute_nearfield(mesh::TriMesh, rwg::RWGData,
                           I_coeffs::AbstractVector{<:Number},
                           observation_point::Vec3, k;
                           quad_order::Int=3,
                           eta0::Float64=376.730313668,
                           check_surface::Bool=true,
                           surface_tol::Union{Nothing,Float64}=nothing)
    E = _compute_nearfield_matrix(mesh, rwg, I_coeffs, [observation_point], k;
                                  quad_order=quad_order,
                                  eta0=eta0,
                                  check_surface=check_surface,
                                  surface_tol=surface_tol)
    return CVec3(E[:, 1])
end

function compute_nearfield(mesh::TriMesh, rwg::RWGData,
                           I_coeffs::AbstractVector{<:Number},
                           observation_points::AbstractVector{<:Vec3}, k;
                           quad_order::Int=3,
                           eta0::Float64=376.730313668,
                           check_surface::Bool=true,
                           surface_tol::Union{Nothing,Float64}=nothing)
    obs = _collect_observation_points(observation_points)
    return _compute_nearfield_matrix(mesh, rwg, I_coeffs, obs, k;
                                     quad_order=quad_order,
                                     eta0=eta0,
                                     check_surface=check_surface,
                                     surface_tol=surface_tol)
end

function compute_nearfield(mesh::TriMesh, rwg::RWGData,
                           I_coeffs::AbstractVector{<:Number},
                           observation_points::AbstractMatrix{<:Real}, k;
                           quad_order::Int=3,
                           eta0::Float64=376.730313668,
                           check_surface::Bool=true,
                           surface_tol::Union{Nothing,Float64}=nothing)
    obs = _collect_observation_points(observation_points)
    return _compute_nearfield_matrix(mesh, rwg, I_coeffs, obs, k;
                                     quad_order=quad_order,
                                     eta0=eta0,
                                     check_surface=check_surface,
                                     surface_tol=surface_tol)
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
                             surface_tol::Union{Nothing,Float64}=nothing)
    E = _compute_total_field_matrix(mesh, rwg, I_coeffs, excitation, [observation_point], k;
                                    quad_order=quad_order,
                                    eta0=eta0,
                                    check_surface=check_surface,
                                    surface_tol=surface_tol)
    return CVec3(E[:, 1])
end

function compute_total_field(mesh::TriMesh, rwg::RWGData,
                             I_coeffs::AbstractVector{<:Number},
                             excitation::AbstractExcitation,
                             observation_points::AbstractVector{<:Vec3}, k;
                             quad_order::Int=3,
                             eta0::Float64=376.730313668,
                             check_surface::Bool=true,
                             surface_tol::Union{Nothing,Float64}=nothing)
    obs = _collect_observation_points(observation_points)
    return _compute_total_field_matrix(mesh, rwg, I_coeffs, excitation, obs, k;
                                       quad_order=quad_order,
                                       eta0=eta0,
                                       check_surface=check_surface,
                                       surface_tol=surface_tol)
end

function compute_total_field(mesh::TriMesh, rwg::RWGData,
                             I_coeffs::AbstractVector{<:Number},
                             excitation::AbstractExcitation,
                             observation_points::AbstractMatrix{<:Real}, k;
                             quad_order::Int=3,
                             eta0::Float64=376.730313668,
                             check_surface::Bool=true,
                             surface_tol::Union{Nothing,Float64}=nothing)
    obs = _collect_observation_points(observation_points)
    return _compute_total_field_matrix(mesh, rwg, I_coeffs, excitation, obs, k;
                                       quad_order=quad_order,
                                       eta0=eta0,
                                       check_surface=check_surface,
                                       surface_tol=surface_tol)
end
