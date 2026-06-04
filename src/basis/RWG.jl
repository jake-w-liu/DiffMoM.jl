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

@inline function _boundary_side_flags(mesh::TriMesh, v1::Int, v2::Int,
                                      xmin::Float64, xmax::Float64,
                                      ymin::Float64, ymax::Float64,
                                      tol::Float64)
    x1 = mesh.xyz[1, v1]; x2 = mesh.xyz[1, v2]
    y1 = mesh.xyz[2, v1]; y2 = mesh.xyz[2, v2]
    return (
        on_xmin = abs(x1 - xmin) <= tol && abs(x2 - xmin) <= tol,
        on_xmax = abs(x1 - xmax) <= tol && abs(x2 - xmax) <= tol,
        on_ymin = abs(y1 - ymin) <= tol && abs(y2 - ymin) <= tol,
        on_ymax = abs(y1 - ymax) <= tol && abs(y2 - ymax) <= tol,
    )
end

@inline function _pair_translated_edge_match(ea, eb, shift::Vec3, tol::Float64)
    a1 = ea.r1 + shift
    a2 = ea.r2 + shift

    same = norm(a1 - eb.r1) <= tol && norm(a2 - eb.r2) <= tol
    opp = norm(a1 - eb.r2) <= tol && norm(a2 - eb.r1) <= tol

    return same || opp
end

function _pair_boundary_edges!(mesh::TriMesh,
                               side_a, side_b,
                               shift::Vec3, axis::Symbol, phase::ComplexF64,
                               tol::Float64,
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
        error("Periodic boundary mismatch on $axis: side counts $(length(side_a)) != $(length(side_b)).")

    matched_b = falses(length(side_b))
    for ea in side_a
        found_j = 0
        nfound = 0

        for (j, eb) in enumerate(side_b)
            matched_b[j] && continue
            _pair_translated_edge_match(ea, eb, shift, tol) || continue
            nfound += 1
            found_j = j
        end

        nfound == 1 || error("Could not uniquely pair boundary edge on axis $axis.")
        eb = side_b[found_j]
        matched_b[found_j] = true

        cplus = 1.0 + 0im
        cminus = phase
        _append_rwg_entry!(mesh, tplus_vec, tminus_vec, evert_arr,
                           vplus_opp_vec, vminus_opp_vec, len_vec,
                           area_p_vec, area_m_vec,
                           coeff_plus_vec, coeff_minus_vec,
                           ea.t, ea.le, eb.t, eb.le, ea.edge_key, cplus, cminus)
    end

    all(matched_b) || error("Unmatched boundary edges remain on paired side for axis $axis.")
    return nothing
end

"""
    build_rwg_periodic(mesh, lattice; kwargs...)

Construct RWG data with Bloch-periodic pairing of opposite unit-cell boundary edges.
Interior edges are included as standard RWG functions. Boundary edges lying on
`xmin/xmax` and `ymin/ymax` are paired into additional RWG functions whose
`T-` side is multiplied by the Bloch phase.

`lattice` must provide fields `dx`, `dy`, `kx_bloch`, and `ky_bloch`.
"""
function build_rwg_periodic(mesh::TriMesh, lattice;
                            precheck::Bool=true,
                            allow_boundary::Bool=true,
                            require_closed::Bool=false,
                            area_tol_rel::Float64=1e-12,
                            boundary_atol_abs::Float64=1e-12,
                            boundary_atol_rel::Float64=1e-9)
    if precheck
        assert_mesh_quality(mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end

    dx = Float64(getproperty(lattice, :dx))
    dy = Float64(getproperty(lattice, :dy))
    kx = Float64(getproperty(lattice, :kx_bloch))
    ky = Float64(getproperty(lattice, :ky_bloch))

    tol = max(boundary_atol_abs, boundary_atol_rel * max(dx, dy))
    xmin = -0.5 * dx
    xmax = 0.5 * dx
    ymin = -0.5 * dy
    ymax = 0.5 * dy

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

    xmin_edges = NamedTuple[]
    xmax_edges = NamedTuple[]
    ymin_edges = NamedTuple[]
    ymax_edges = NamedTuple[]

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
        flags = _boundary_side_flags(mesh, v1, v2, xmin, xmax, ymin, ymax, tol)
        side_count = Int(flags.on_xmin) + Int(flags.on_xmax) + Int(flags.on_ymin) + Int(flags.on_ymax)
        side_count == 0 && continue
        side_count == 1 || error("Ambiguous boundary-edge classification (edge lies on multiple periodic sides).")

        entry = (edge_key=edge_key, t=t, le=le, v1=v1, v2=v2,
                 r1=_mesh_vertex(mesh, v1), r2=_mesh_vertex(mesh, v2))

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

    phase_x = exp(-im * kx * dx)
    phase_y = exp(-im * ky * dy)
    _pair_boundary_edges!(mesh, xmin_edges, xmax_edges,
                          Vec3(dx, 0.0, 0.0), :x, phase_x, tol,
                          tplus_vec, tminus_vec, evert_arr,
                          vplus_opp_vec, vminus_opp_vec, len_vec,
                          area_p_vec, area_m_vec, coeff_plus_vec, coeff_minus_vec)
    _pair_boundary_edges!(mesh, ymin_edges, ymax_edges,
                          Vec3(0.0, dy, 0.0), :y, phase_y, tol,
                          tplus_vec, tminus_vec, evert_arr,
                          vplus_opp_vec, vminus_opp_vec, len_vec,
                          area_p_vec, area_m_vec, coeff_plus_vec, coeff_minus_vec)

    return _finalize_rwg(mesh, tplus_vec, tminus_vec, evert_arr,
                         vplus_opp_vec, vminus_opp_vec, len_vec,
                         area_p_vec, area_m_vec, coeff_plus_vec, coeff_minus_vec;
                         has_periodic_bloch=true)
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
        scale = rwg.coeff_plus[n] * (rwg.len[n] / (2.0 * rwg.area_plus[n]))
        return scale * (r - r_opp)
    elseif t == rwg.tminus[n]
        r_opp = _mesh_vertex(rwg.mesh, rwg.vminus_opp[n])
        scale = rwg.coeff_minus[n] * (rwg.len[n] / (2.0 * rwg.area_minus[n]))
        return scale * (r_opp - r)
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
        return rwg.coeff_plus[n] * (rwg.len[n] / rwg.area_plus[n])
    elseif t == rwg.tminus[n]
        return -rwg.coeff_minus[n] * (rwg.len[n] / rwg.area_minus[n])
    else
        return zero(Tcoef)
    end
end

"""
    basis_triangles(rwg, n)

Return the two triangle indices supporting basis function `n`.
"""
basis_triangles(rwg::RWGData, n::Int) = (rwg.tplus[n], rwg.tminus[n])
