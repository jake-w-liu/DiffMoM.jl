# RWG.jl — Rao-Wilton-Glisson basis function construction and evaluation

export build_rwg, build_rwg_periodic, eval_rwg, div_rwg, basis_triangles

function _build_edge_triangle_map(mesh::TriMesh)
    Nt = ntriangles(mesh)
    edge_tris = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()

    for t in 1:Nt
        for le in 1:3
            v1 = mesh.tri[le, t]
            v2 = mesh.tri[mod1(le + 1, 3), t]
            key = v1 < v2 ? (v1, v2) : (v2, v1)
            if !haskey(edge_tris, key)
                edge_tris[key] = Tuple{Int,Int}[]
            end
            push!(edge_tris[key], (t, le))
        end
    end

    return edge_tris
end

@inline function _opposite_vertex(mesh::TriMesh, t::Int, le::Int)
    # edge le connects tri[le,t] and tri[mod1(le+1,3),t]
    # opposite is tri[mod1(le+2,3),t]
    return mesh.tri[mod1(le + 2, 3), t]
end

function _finalize_rwg(mesh::TriMesh,
                       tplus_vec::Vector{Int},
                       tminus_vec::Vector{Int},
                       evert_arr::Vector{Tuple{Int,Int}},
                       vplus_opp_vec::Vector{Int},
                       vminus_opp_vec::Vector{Int},
                       len_vec::Vector{Float64},
                       area_p_vec::Vector{Float64},
                       area_m_vec::Vector{Float64},
                       coeff_plus_vec::Vector{T},
                       coeff_minus_vec::Vector{T};
                       has_periodic_bloch::Bool=false) where {T<:Number}
    nedges = length(tplus_vec)
    evert = zeros(Int, 2, nedges)
    for n in 1:nedges
        evert[1, n] = evert_arr[n][1]
        evert[2, n] = evert_arr[n][2]
    end

    return RWGData{T}(mesh, nedges, tplus_vec, tminus_vec, evert,
                      vplus_opp_vec, vminus_opp_vec, len_vec,
                      area_p_vec, area_m_vec, coeff_plus_vec, coeff_minus_vec,
                      has_periodic_bloch)
end

function _append_rwg_entry!(mesh::TriMesh,
                            tplus_vec::Vector{Int},
                            tminus_vec::Vector{Int},
                            evert_arr::Vector{Tuple{Int,Int}},
                            vplus_opp_vec::Vector{Int},
                            vminus_opp_vec::Vector{Int},
                            len_vec::Vector{Float64},
                            area_p_vec::Vector{Float64},
                            area_m_vec::Vector{Float64},
                            coeff_plus_vec::Vector{T},
                            coeff_minus_vec::Vector{T},
                            tplus::Int, le_plus::Int,
                            tminus::Int, le_minus::Int,
                            edge_key::Tuple{Int,Int},
                            cplus::T, cminus::T) where {T<:Number}
    push!(tplus_vec, tplus)
    push!(tminus_vec, tminus)
    push!(evert_arr, edge_key)
    push!(vplus_opp_vec, _opposite_vertex(mesh, tplus, le_plus))
    push!(vminus_opp_vec, _opposite_vertex(mesh, tminus, le_minus))

    r1 = _mesh_vertex(mesh, edge_key[1])
    r2 = _mesh_vertex(mesh, edge_key[2])
    push!(len_vec, norm(r2 - r1))
    push!(area_p_vec, triangle_area(mesh, tplus))
    push!(area_m_vec, triangle_area(mesh, tminus))

    push!(coeff_plus_vec, cplus)
    push!(coeff_minus_vec, cminus)
    return nothing
end

"""
    build_rwg(mesh::TriMesh; precheck=true, allow_boundary=true, require_closed=false, area_tol_rel=1e-12)

Construct standard RWG basis functions from interior edges of the triangle mesh.
Each interior edge shared by two triangles defines one RWG basis function.

When `precheck=true` (default), run mesh-quality checks before basis
construction and error out on invalid/degenerate/non-manifold or
orientation-inconsistent meshes.
"""
function build_rwg(mesh::TriMesh;
                   precheck::Bool=true,
                   allow_boundary::Bool=true,
                   require_closed::Bool=false,
                   area_tol_rel::Float64=1e-12)
    if precheck
        assert_mesh_quality(mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end

    edge_tris = _build_edge_triangle_map(mesh)

    tplus_vec = Int[]
    tminus_vec = Int[]
    evert_arr = Tuple{Int,Int}[]
    vplus_opp_vec = Int[]
    vminus_opp_vec = Int[]
    len_vec = Float64[]
    area_p_vec = Float64[]
    area_m_vec = Float64[]
    coeff_plus_vec = Float64[]
    coeff_minus_vec = Float64[]

    for (edge_key, tlist) in edge_tris
        length(tlist) == 2 || continue  # skip boundary edges
        (t1, le1) = tlist[1]
        (t2, le2) = tlist[2]
        _append_rwg_entry!(mesh, tplus_vec, tminus_vec, evert_arr,
                           vplus_opp_vec, vminus_opp_vec, len_vec,
                           area_p_vec, area_m_vec,
                           coeff_plus_vec, coeff_minus_vec,
                           t1, le1, t2, le2, edge_key, 1.0, 1.0)
    end

    return _finalize_rwg(mesh, tplus_vec, tminus_vec, evert_arr,
                         vplus_opp_vec, vminus_opp_vec, len_vec,
                         area_p_vec, area_m_vec, coeff_plus_vec, coeff_minus_vec;
                         has_periodic_bloch=false)
end

const _PERIODIC_RWG_PHASE_FALLBACK_PRECISION = 2304
const _PERIODIC_PAIR_SMALL_CELL_LIMIT = 1_073_741_824.0 # 2^30
const _PERIODIC_PAIR_CHECKS_PER_EDGE = 64
const _PERIODIC_PAIR_MIN_CHECK_BUDGET = 4096
const _RWG_EVALUATION_FALLBACK_PRECISION = 2304

struct _PeriodicBoundaryEdge
    edge_key::Tuple{Int,Int}
    t::Int
    le::Int
    r1::Vec3
    r2::Vec3
end

@noinline function _periodic_rwg_bloch_phase_exact(wavenumber::Float64,
                                                   period::Float64)
    return setprecision(BigFloat, _PERIODIC_RWG_PHASE_FALLBACK_PRECISION) do
        phase = cis(-BigFloat(wavenumber) * BigFloat(period))
        ComplexF64(phase)
    end
end

@inline function _periodic_rwg_bloch_phase(wavenumber::Float64,
                                           period::Float64)
    product_hi = wavenumber * period
    if isfinite(product_hi)
        product_lo = fma(wavenumber, period, -product_hi)
        if isfinite(product_lo)
            reduced = rem2pi(
                rem2pi(product_hi, RoundNearest) +
                rem2pi(product_lo, RoundNearest),
                RoundNearest,
            )
            phase = cis(-reduced)
            isfinite(real(phase)) && isfinite(imag(phase)) &&
                return ComplexF64(phase)
        end
    end
    return _periodic_rwg_bloch_phase_exact(wavenumber, period)
end

@inline function _periodic_coordinate_within(first::Float64,
                                             second::Float64,
                                             tolerance::Float64)
    iszero(tolerance) && return first == second
    distance = abs(first - second)
    distance == 0.0 && return true
    if isfinite(distance)
        uncertainty = 32 * eps(Float64) * max(distance, tolerance)
        separation = abs(distance - tolerance)
        separation > uncertainty && return distance <= tolerance
    end
    return abs(Rational{BigInt}(first) - Rational{BigInt}(second)) <=
           Rational{BigInt}(tolerance)
end

@inline function _periodic_boundary_coordinate_within(value::Float64,
                                                      boundary::Float64,
                                                      tolerance::Float64)
    iszero(tolerance) && return value == boundary
    distance = abs(value - boundary)
    distance == 0.0 && return true
    if isfinite(distance) && distance > tolerance
        uncertainty = 32 * eps(Float64) * max(distance, tolerance)
        distance - tolerance > uncertainty && return false
    end
    return abs(Rational{BigInt}(value) - Rational{BigInt}(boundary)) <=
           Rational{BigInt}(tolerance)
end

@inline function _boundary_side_flags(mesh::TriMesh, v1::Int, v2::Int,
                                      xmin::Float64, xmax::Float64,
                                      ymin::Float64, ymax::Float64,
                                      tol_x::Float64, tol_y::Float64)
    x1 = mesh.xyz[1, v1]; x2 = mesh.xyz[1, v2]
    y1 = mesh.xyz[2, v1]; y2 = mesh.xyz[2, v2]
    return (
        on_xmin = _periodic_boundary_coordinate_within(x1, xmin, tol_x) &&
                  _periodic_boundary_coordinate_within(x2, xmin, tol_x),
        on_xmax = _periodic_boundary_coordinate_within(x1, xmax, tol_x) &&
                  _periodic_boundary_coordinate_within(x2, xmax, tol_x),
        on_ymin = _periodic_boundary_coordinate_within(y1, ymin, tol_y) &&
                  _periodic_boundary_coordinate_within(y2, ymin, tol_y),
        on_ymax = _periodic_boundary_coordinate_within(y1, ymax, tol_y) &&
                  _periodic_boundary_coordinate_within(y2, ymax, tol_y),
    )
end

@inline _periodic_pair_zero(value::Float64) = iszero(value) ? 0.0 : value

@inline function _periodic_edge_signature(edge, axis::Symbol)
    if axis === :x
        first = (_periodic_pair_zero(edge.r1[2]),
                 _periodic_pair_zero(edge.r1[3]))
        second = (_periodic_pair_zero(edge.r2[2]),
                  _periodic_pair_zero(edge.r2[3]))
    else
        first = (_periodic_pair_zero(edge.r1[1]),
                 _periodic_pair_zero(edge.r1[3]))
        second = (_periodic_pair_zero(edge.r2[1]),
                  _periodic_pair_zero(edge.r2[3]))
    end
    return (first[1], first[2], second[1], second[2])
end

@inline function _periodic_bucket_signature(signature::NTuple{4,Float64})
    return (min(signature[1], signature[3]),
            max(signature[1], signature[3]),
            min(signature[2], signature[4]),
            max(signature[2], signature[4]))
end

@inline function _periodic_signatures_match(first::NTuple{4,Float64},
                                            second::NTuple{4,Float64},
                                            tolerances::NTuple{4,Float64})
    same = _periodic_coordinate_within(
               first[1], second[1], tolerances[1]) &&
           _periodic_coordinate_within(
               first[2], second[2], tolerances[2]) &&
           _periodic_coordinate_within(
               first[3], second[3], tolerances[3]) &&
           _periodic_coordinate_within(
               first[4], second[4], tolerances[4])
    same && return true
    return _periodic_coordinate_within(
               first[1], second[3], tolerances[1]) &&
           _periodic_coordinate_within(
               first[2], second[4], tolerances[2]) &&
           _periodic_coordinate_within(
               first[3], second[1], tolerances[3]) &&
           _periodic_coordinate_within(
               first[4], second[2], tolerances[4])
end

@inline function _periodic_exact_signature_key(signature::NTuple{4,Float64})
    return (reinterpret(UInt64, signature[1]),
            reinterpret(UInt64, signature[2]),
            reinterpret(UInt64, signature[3]),
            reinterpret(UInt64, signature[4]))
end

@inline function _periodic_small_cell_coordinate(value::Float64,
                                                 origin::Float64,
                                                 width::Float64)
    iszero(width) && return true, reinterpret(Int64, value)
    scaled = (value - origin) / width
    if isfinite(scaled) && abs(scaled) <= _PERIODIC_PAIR_SMALL_CELL_LIMIT
        return true, floor(Int64, scaled)
    end
    return false, Int64(0)
end

@inline function _periodic_small_signature_cell(
        signature::NTuple{4,Float64}, origin::NTuple{4,Float64},
        widths::NTuple{4,Float64})
    fit1, cell1 = _periodic_small_cell_coordinate(
        signature[1], origin[1], widths[1])
    fit2, cell2 = _periodic_small_cell_coordinate(
        signature[2], origin[2], widths[2])
    fit3, cell3 = _periodic_small_cell_coordinate(
        signature[3], origin[3], widths[3])
    fit4, cell4 = _periodic_small_cell_coordinate(
        signature[4], origin[4], widths[4])
    return fit1 && fit2 && fit3 && fit4, (cell1, cell2, cell3, cell4)
end

@noinline function _periodic_exact_cell_coordinate(value::Float64,
                                                   origin::Float64,
                                                   width::Float64)
    iszero(width) && return BigInt(reinterpret(Int64, value))
    quotient = (Rational{BigInt}(value) - Rational{BigInt}(origin)) /
               Rational{BigInt}(width)
    return fld(numerator(quotient), denominator(quotient))
end

@noinline function _periodic_exact_signature_cell(
        signature::NTuple{4,Float64}, origin::NTuple{4,Float64},
        widths::NTuple{4,Float64})
    return (_periodic_exact_cell_coordinate(signature[1], origin[1], widths[1]),
            _periodic_exact_cell_coordinate(signature[2], origin[2], widths[2]),
            _periodic_exact_cell_coordinate(signature[3], origin[3], widths[3]),
            _periodic_exact_cell_coordinate(signature[4], origin[4], widths[4]))
end

function _periodic_exact_boundary_matches(signatures_a, signatures_b,
                                          axis::Symbol)
    heads = Dict{NTuple{4,UInt64},Int}()
    next_in_bucket = zeros(Int, length(signatures_b))
    @inbounds for j in eachindex(signatures_b)
        key = _periodic_exact_signature_key(
            _periodic_bucket_signature(signatures_b[j]))
        next_in_bucket[j] = get(heads, key, 0)
        heads[key] = j
    end

    matches = zeros(Int, length(signatures_a))
    matched_b = falses(length(signatures_b))
    check_budget = _periodic_pair_check_budget(
        Base.checked_add(length(signatures_a), length(signatures_b)))
    candidate_checks = 0
    @inbounds for i in eachindex(signatures_a)
        key = _periodic_exact_signature_key(
            _periodic_bucket_signature(signatures_a[i]))
        j = get(heads, key, 0)
        found = 0
        nfound = 0
        while !iszero(j)
            candidate_checks += 1
            candidate_checks <= check_budget ||
                throw(ArgumentError(
                    "Periodic boundary pairing on axis $axis is too dense."))
            if _periodic_signatures_match(
                    signatures_a[i], signatures_b[j], (0.0, 0.0, 0.0, 0.0))
                nfound += 1
                found = j
                nfound > 1 && break
            end
            j = next_in_bucket[j]
        end
        nfound == 1 ||
            throw(ArgumentError(
                "Could not uniquely pair boundary edge on axis $axis."))
        !matched_b[found] ||
            throw(ArgumentError(
                "Boundary edge on axis $axis was paired more than once."))
        matches[i] = found
        matched_b[found] = true
    end
    all(matched_b) ||
        throw(ArgumentError(
            "Unmatched boundary edges remain on paired side for axis $axis."))
    return matches
end

@inline function _periodic_pair_check_budget(nedges::Int)
    product = Base.checked_mul(_PERIODIC_PAIR_CHECKS_PER_EDGE, nedges)
    return max(_PERIODIC_PAIR_MIN_CHECK_BUDGET, product)
end

function _periodic_hashed_boundary_matches(signatures_a, signatures_b,
                                           tolerances::NTuple{4,Float64},
                                           axis::Symbol)
    bucket_signatures_a = _periodic_bucket_signature.(signatures_a)
    bucket_signatures_b = _periodic_bucket_signature.(signatures_b)
    origin = bucket_signatures_b[1]
    # Bucket coordinates are `(min t1, max t1, min t2, max t2)`.
    widths = (2 * tolerances[1], 2 * tolerances[3],
              2 * tolerances[2], 2 * tolerances[4])
    use_small_cells = true
    @inbounds for signature in bucket_signatures_b
        fits, _ = _periodic_small_signature_cell(signature, origin, widths)
        if !fits
            use_small_cells = false
            break
        end
    end
    if use_small_cells
        @inbounds for signature in bucket_signatures_a
            fits, _ = _periodic_small_signature_cell(signature, origin, widths)
            if !fits
                use_small_cells = false
                break
            end
        end
    end

    matches = zeros(Int, length(signatures_a))
    matched_b = falses(length(signatures_b))
    next_in_bucket = zeros(Int, length(signatures_b))
    check_budget = _periodic_pair_check_budget(
        Base.checked_add(length(signatures_a), length(signatures_b)))
    candidate_checks = 0

    if use_small_cells
        heads = Dict{NTuple{4,Int64},Int}()
        @inbounds for j in eachindex(bucket_signatures_b)
            _, cell = _periodic_small_signature_cell(
                bucket_signatures_b[j], origin, widths)
            next_in_bucket[j] = get(heads, cell, 0)
            heads[cell] = j
        end

        @inbounds for i in eachindex(bucket_signatures_a)
            _, cell = _periodic_small_signature_cell(
                bucket_signatures_a[i], origin, widths)
            found = 0
            nfound = 0
            for d4 in -1:1, d3 in -1:1, d2 in -1:1, d1 in -1:1
                neighbor = (cell[1] + d1, cell[2] + d2,
                            cell[3] + d3, cell[4] + d4)
                j = get(heads, neighbor, 0)
                while !iszero(j)
                    candidate_checks += 1
                    candidate_checks <= check_budget ||
                        throw(ArgumentError(
                            "Periodic boundary pairing on axis $axis is " *
                            "too dense for the requested tolerance."))
                    if _periodic_signatures_match(
                            signatures_a[i], signatures_b[j], tolerances)
                        nfound += 1
                        found = j
                        nfound > 1 && break
                    end
                    j = next_in_bucket[j]
                end
                nfound > 1 && break
            end
            nfound == 1 ||
                throw(ArgumentError(
                    "Could not uniquely pair boundary edge on axis $axis."))
            !matched_b[found] ||
                throw(ArgumentError(
                    "Boundary edge on axis $axis was paired more than once."))
            matches[i] = found
            matched_b[found] = true
        end
    else
        heads = Dict{NTuple{4,BigInt},Int}()
        @inbounds for j in eachindex(bucket_signatures_b)
            cell = _periodic_exact_signature_cell(
                bucket_signatures_b[j], origin, widths)
            next_in_bucket[j] = get(heads, cell, 0)
            heads[cell] = j
        end

        @inbounds for i in eachindex(bucket_signatures_a)
            cell = _periodic_exact_signature_cell(
                bucket_signatures_a[i], origin, widths)
            found = 0
            nfound = 0
            for d4 in -1:1, d3 in -1:1, d2 in -1:1, d1 in -1:1
                neighbor = (cell[1] + d1, cell[2] + d2,
                            cell[3] + d3, cell[4] + d4)
                j = get(heads, neighbor, 0)
                while !iszero(j)
                    candidate_checks += 1
                    candidate_checks <= check_budget ||
                        throw(ArgumentError(
                            "Periodic boundary pairing on axis $axis is " *
                            "too dense for the requested tolerance."))
                    if _periodic_signatures_match(
                            signatures_a[i], signatures_b[j], tolerances)
                        nfound += 1
                        found = j
                        nfound > 1 && break
                    end
                    j = next_in_bucket[j]
                end
                nfound > 1 && break
            end
            nfound == 1 ||
                throw(ArgumentError(
                    "Could not uniquely pair boundary edge on axis $axis."))
            !matched_b[found] ||
                throw(ArgumentError(
                    "Boundary edge on axis $axis was paired more than once."))
            matches[i] = found
            matched_b[found] = true
        end
    end

    all(matched_b) ||
        throw(ArgumentError(
            "Unmatched boundary edges remain on paired side for axis $axis."))
    return matches
end

function _periodic_boundary_matches(side_a, side_b, axis::Symbol,
                                    tangential_tolerances::NTuple{2,Float64})
    length(side_a) == length(side_b) ||
        throw(ArgumentError(
            "Periodic boundary mismatch on $axis: side counts " *
            "$(length(side_a)) != $(length(side_b))."))
    isempty(side_a) && return Int[]
    signatures_a = [_periodic_edge_signature(edge, axis) for edge in side_a]
    signatures_b = [_periodic_edge_signature(edge, axis) for edge in side_b]
    tolerances = (tangential_tolerances[1], tangential_tolerances[2],
                  tangential_tolerances[1], tangential_tolerances[2])
    if iszero(tolerances[1]) && iszero(tolerances[2])
        return _periodic_exact_boundary_matches(
            signatures_a, signatures_b, axis)
    end
    return _periodic_hashed_boundary_matches(
        signatures_a, signatures_b, tolerances, axis)
end

function _pair_boundary_edges!(mesh::TriMesh,
                               side_a, side_b,
                               axis::Symbol, phase::ComplexF64,
                               tangential_tolerances::NTuple{2,Float64},
                               tplus_vec::Vector{Int},
                               tminus_vec::Vector{Int},
                               evert_arr::Vector{Tuple{Int,Int}},
                               vplus_opp_vec::Vector{Int},
                               vminus_opp_vec::Vector{Int},
                               len_vec::Vector{Float64},
                               area_p_vec::Vector{Float64},
                               area_m_vec::Vector{Float64},
                               coeff_plus_vec::Vector{ComplexF64},
                               coeff_minus_vec::Vector{ComplexF64})
    isempty(side_a) && isempty(side_b) && return
    length(side_a) == length(side_b) ||
        throw(ArgumentError(
            "Periodic boundary mismatch on $axis: side counts " *
            "$(length(side_a)) != $(length(side_b))."))

    matches = _periodic_boundary_matches(
        side_a, side_b, axis, tangential_tolerances)
    for (i, ea) in enumerate(side_a)
        found_j = matches[i]
        eb = side_b[found_j]

        cplus = 1.0 + 0im
        cminus = phase
        _append_rwg_entry!(mesh, tplus_vec, tminus_vec, evert_arr,
                           vplus_opp_vec, vminus_opp_vec, len_vec,
                           area_p_vec, area_m_vec,
                           coeff_plus_vec, coeff_minus_vec,
                           ea.t, ea.le, eb.t, eb.le, ea.edge_key, cplus, cminus)
    end
    return nothing
end

"""
    build_rwg_periodic(mesh, lattice;
        precheck=true, allow_boundary=true, require_closed=false,
        area_tol_rel=1e-12, boundary_atol_abs=nothing,
        boundary_atol_rel=1e-9)

Construct RWG data with Bloch-periodic pairing of opposite unit-cell boundary edges.
Interior edges are included as standard RWG functions. Boundary edges lying on
`xmin/xmax` and `ymin/ymax` are paired into additional RWG functions whose
`T-` side is multiplied by the Bloch phase.

`lattice` must provide finite fields `dx`, `dy`, `kx_bloch`, and `ky_bloch`,
with positive periods. Numeric `boundary_atol_abs` values and
`boundary_atol_rel` are finite, nonnegative coordinate tolerances. The relative tolerance is applied separately
to each period. With the default `boundary_atol_abs=nothing`, a `1e-12` absolute
floor is retained only when both periods are greater than `2e-12`; smaller cells use
no absolute floor. Pass a numeric value to select an explicit absolute tolerance.
Each resulting x/y tolerance must be less than half of its period.

Opposite boundary edges must pair one-to-one. Pairing uses a bounded spatial
index, so malformed or overly dense tolerance neighborhoods fail rather than
falling back to quadratic work. Bloch phases use the exact product of the stored
Float64 wavenumber and period, with high-precision range reduction only when the
product overflows.
"""
function build_rwg_periodic(mesh::TriMesh, lattice;
                            precheck::Bool=true,
                            allow_boundary::Bool=true,
                            require_closed::Bool=false,
                            area_tol_rel::Float64=1e-12,
                            boundary_atol_abs::Union{Nothing,Float64}=nothing,
                            boundary_atol_rel::Float64=1e-9)
    dx = Float64(getproperty(lattice, :dx))
    dy = Float64(getproperty(lattice, :dy))
    kx = Float64(getproperty(lattice, :kx_bloch))
    ky = Float64(getproperty(lattice, :ky_bloch))

    isfinite(dx) && dx > 0.0 ||
        throw(ArgumentError("lattice.dx must be finite and positive, got $dx."))
    isfinite(dy) && dy > 0.0 ||
        throw(ArgumentError("lattice.dy must be finite and positive, got $dy."))
    isfinite(kx) ||
        throw(ArgumentError("lattice.kx_bloch must be finite, got $kx."))
    isfinite(ky) ||
        throw(ArgumentError("lattice.ky_bloch must be finite, got $ky."))
    if boundary_atol_abs !== nothing
        isfinite(boundary_atol_abs) && boundary_atol_abs >= 0.0 ||
            throw(ArgumentError(
                "boundary_atol_abs must be finite and nonnegative, got " *
                "$boundary_atol_abs."))
    end
    isfinite(boundary_atol_rel) && 0.0 <= boundary_atol_rel < 0.5 ||
        throw(ArgumentError(
            "boundary_atol_rel must be finite and in [0, 0.5), got " *
            "$boundary_atol_rel."))

    if precheck
        assert_mesh_quality(mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end

    xmin = -0.5 * dx
    xmax = 0.5 * dx
    ymin = -0.5 * dy
    ymax = 0.5 * dy
    xmin < xmax ||
        throw(ArgumentError(
            "lattice.dx=$dx is too small to form distinct Float64 boundaries."))
    ymin < ymax ||
        throw(ArgumentError(
            "lattice.dy=$dy is too small to form distinct Float64 boundaries."))

    absolute_tolerance = boundary_atol_abs === nothing ?
        (min(dx, dy) > 2e-12 ? 1e-12 : 0.0) : boundary_atol_abs
    tol_x = max(absolute_tolerance, boundary_atol_rel * dx)
    tol_y = max(absolute_tolerance, boundary_atol_rel * dy)
    tol_z = max(absolute_tolerance, boundary_atol_rel * min(dx, dy))
    tol_x < 0.5 * dx ||
        throw(ArgumentError(
            "x-boundary tolerance $tol_x must be less than half dx=$dx."))
    tol_y < 0.5 * dy ||
        throw(ArgumentError(
            "y-boundary tolerance $tol_y must be less than half dy=$dy."))

    edge_tris = _build_edge_triangle_map(mesh)

    tplus_vec = Int[]
    tminus_vec = Int[]
    evert_arr = Tuple{Int,Int}[]
    vplus_opp_vec = Int[]
    vminus_opp_vec = Int[]
    len_vec = Float64[]
    area_p_vec = Float64[]
    area_m_vec = Float64[]
    coeff_plus_vec = ComplexF64[]
    coeff_minus_vec = ComplexF64[]

    xmin_edges = _PeriodicBoundaryEdge[]
    xmax_edges = _PeriodicBoundaryEdge[]
    ymin_edges = _PeriodicBoundaryEdge[]
    ymax_edges = _PeriodicBoundaryEdge[]

    for (edge_key, tlist) in edge_tris
        if length(tlist) == 2
            (t1, le1) = tlist[1]
            (t2, le2) = tlist[2]
            _append_rwg_entry!(mesh, tplus_vec, tminus_vec, evert_arr,
                               vplus_opp_vec, vminus_opp_vec, len_vec,
                               area_p_vec, area_m_vec,
                               coeff_plus_vec, coeff_minus_vec,
                               t1, le1, t2, le2, edge_key, 1.0 + 0im, 1.0 + 0im)
            continue
        end

        length(tlist) == 1 || continue
        (t, le) = tlist[1]
        v1 = mesh.tri[le, t]
        v2 = mesh.tri[mod1(le + 1, 3), t]
        flags = _boundary_side_flags(
            mesh, v1, v2, xmin, xmax, ymin, ymax, tol_x, tol_y)
        side_count = Int(flags.on_xmin) + Int(flags.on_xmax) + Int(flags.on_ymin) + Int(flags.on_ymax)
        side_count == 0 && continue
        side_count == 1 ||
            throw(ArgumentError(
                "Ambiguous boundary-edge classification " *
                "(edge lies on multiple periodic sides)."))

        entry = _PeriodicBoundaryEdge(
            edge_key, t, le, _mesh_vertex(mesh, v1), _mesh_vertex(mesh, v2))

        if flags.on_xmin
            push!(xmin_edges, entry)
        elseif flags.on_xmax
            push!(xmax_edges, entry)
        elseif flags.on_ymin
            push!(ymin_edges, entry)
        elseif flags.on_ymax
            push!(ymax_edges, entry)
        end
    end

    phase_x = _periodic_rwg_bloch_phase(kx, dx)
    phase_y = _periodic_rwg_bloch_phase(ky, dy)
    _pair_boundary_edges!(mesh, xmin_edges, xmax_edges,
                          :x, phase_x, (tol_y, tol_z),
                          tplus_vec, tminus_vec, evert_arr,
                          vplus_opp_vec, vminus_opp_vec, len_vec,
                          area_p_vec, area_m_vec, coeff_plus_vec, coeff_minus_vec)
    _pair_boundary_edges!(mesh, ymin_edges, ymax_edges,
                          :y, phase_y, (tol_x, tol_z),
                          tplus_vec, tminus_vec, evert_arr,
                          vplus_opp_vec, vminus_opp_vec, len_vec,
                          area_p_vec, area_m_vec, coeff_plus_vec, coeff_minus_vec)

    return _finalize_rwg(mesh, tplus_vec, tminus_vec, evert_arr,
                         vplus_opp_vec, vminus_opp_vec, len_vec,
                         area_p_vec, area_m_vec, coeff_plus_vec, coeff_minus_vec;
                         has_periodic_bloch=true)
end

@inline _rwg_evaluation_finite(value::Number) =
    isfinite(real(value)) && isfinite(imag(value))

@inline function _rwg_evaluation_convert(
        ::Type{T}, value::Complex{BigFloat}, label::AbstractString) where {T}
    converted = if T <: Real
        iszero(imag(value)) || throw(InexactError(:eval_rwg, T, value))
        convert(T, real(value))
    else
        convert(T, value)
    end
    _rwg_evaluation_finite(converted) ||
        throw(OverflowError("$label is outside the representable $T range"))
    return converted
end

@noinline function _eval_rwg_bigfloat(
        ::Type{T}, coefficient::Number, edge_length::Float64,
        area::Float64, delta::Vec3) where {T<:Number}
    return setprecision(BigFloat, _RWG_EVALUATION_FALLBACK_PRECISION) do
        scale = Complex{BigFloat}(coefficient) * BigFloat(edge_length) /
                (BigFloat(2) * BigFloat(area))
        first = _rwg_evaluation_convert(
            T, scale * BigFloat(delta[1]), "RWG value component 1")
        second = _rwg_evaluation_convert(
            T, scale * BigFloat(delta[2]), "RWG value component 2")
        third = _rwg_evaluation_convert(
            T, scale * BigFloat(delta[3]), "RWG value component 3")
        SVector{3,T}(first, second, third)
    end
end

@inline function _eval_rwg_scaled(
        ::Type{T}, coefficient::Number, edge_length::Float64,
        area::Float64, delta::Vec3) where {T<:Number}
    twice_area = 2.0 * area
    scale_ratio = edge_length / twice_area
    if !isfinite(twice_area) ||
       (iszero(scale_ratio) && !iszero(edge_length))
        return _eval_rwg_bigfloat(
            T, coefficient, edge_length, area, delta)
    end
    first = convert(T, coefficient * (scale_ratio * delta[1]))
    second = convert(T, coefficient * (scale_ratio * delta[2]))
    third = convert(T, coefficient * (scale_ratio * delta[3]))
    if _rwg_evaluation_finite(first) &&
       _rwg_evaluation_finite(second) &&
       _rwg_evaluation_finite(third)
        return SVector{3,T}(first, second, third)
    end
    return _eval_rwg_bigfloat(
        T, coefficient, edge_length, area, delta)
end

@noinline function _div_rwg_bigfloat(
        ::Type{T}, coefficient::Number, edge_length::Float64,
        area::Float64) where {T<:Number}
    return setprecision(BigFloat, _RWG_EVALUATION_FALLBACK_PRECISION) do
        value = Complex{BigFloat}(coefficient) * BigFloat(edge_length) /
                BigFloat(area)
        _rwg_evaluation_convert(T, value, "RWG divergence")
    end
end

@inline function _div_rwg_scaled(
        ::Type{T}, coefficient::Number, edge_length::Float64,
        area::Float64) where {T<:Number}
    scale_ratio = edge_length / area
    if iszero(scale_ratio) && !iszero(edge_length)
        return _div_rwg_bigfloat(T, coefficient, edge_length, area)
    end
    value = coefficient * scale_ratio
    if _rwg_evaluation_finite(value)
        return convert(T, value)
    end
    return _div_rwg_bigfloat(T, coefficient, edge_length, area)
end

"""
    eval_rwg(rwg, n, r, t)

Evaluate RWG basis function `n` at point `r` on triangle `t`.
Returns zero if `t` is not part of basis function `n`.

For T+: f_n(r) = c⁺ (l_n / 2A+) * (r - r_opp+)
For T-: f_n(r) = c⁻ (l_n / 2A-) * (r_opp- - r)
where `(c⁺, c⁻)` are side coefficients (unit for standard RWG).
"""
function eval_rwg(rwg::RWGData, n::Int, r::Vec3, t::Int)
    Tcoef = promote_type(eltype(rwg.coeff_plus), eltype(rwg.coeff_minus))
    z3 = SVector{3,Tcoef}(zero(Tcoef), zero(Tcoef), zero(Tcoef))

    if t == rwg.tplus[n]
        r_opp = _mesh_vertex(rwg.mesh, rwg.vplus_opp[n])
        return _eval_rwg_scaled(
            Tcoef, rwg.coeff_plus[n], rwg.len[n], rwg.area_plus[n],
            r - r_opp)
    elseif t == rwg.tminus[n]
        r_opp = _mesh_vertex(rwg.mesh, rwg.vminus_opp[n])
        return _eval_rwg_scaled(
            Tcoef, rwg.coeff_minus[n], rwg.len[n], rwg.area_minus[n],
            r_opp - r)
    else
        return z3
    end
end

"""
    div_rwg(rwg, n, t)

Compute the surface divergence of RWG basis function `n` on triangle `t`.
For T+: div f_n = c⁺ l_n / A+
For T-: div f_n = -c⁻ l_n / A-
"""
function div_rwg(rwg::RWGData, n::Int, t::Int)
    Tcoef = promote_type(eltype(rwg.coeff_plus), eltype(rwg.coeff_minus))
    if t == rwg.tplus[n]
        return _div_rwg_scaled(
            Tcoef, rwg.coeff_plus[n], rwg.len[n], rwg.area_plus[n])
    elseif t == rwg.tminus[n]
        return _div_rwg_scaled(
            Tcoef, -rwg.coeff_minus[n], rwg.len[n], rwg.area_minus[n])
    else
        return zero(Tcoef)
    end
end

"""
    basis_triangles(rwg, n)

Return the two triangle indices supporting basis function `n`.
"""
basis_triangles(rwg::RWGData, n::Int) = (rwg.tplus[n], rwg.tminus[n])
