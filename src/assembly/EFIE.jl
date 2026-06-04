# EFIE.jl — Dense and matrix-free EFIE operators
#
# Z_mn = <f_m, T[f_n]>  (EFIE operator only, without impedance sheet)
#
# Uses mixed-potential form:
#   Z_mn = -iωμ₀ [ ∫∫ f_m(r)·f_n(r') G(r,r') dS dS'
#                   - (1/k²) ∫∫ (∇·f_m)(∇'·f_n) G(r,r') dS dS' ]

export assemble_Z_efie
export MatrixFreeEFIEOperator, MatrixFreeEFIEAdjointOperator
export matrixfree_efie_operator, matrixfree_efie_adjoint_operator
export efie_entry

struct EFIEApplyCache{TK, Tω, TD, TV}
    mesh::TriMesh
    rwg::RWGData
    k::TK
    inv_k2::TK                           # precomputed 1/k²
    omega_mu0::Tω
    wq::Vector{Float64}
    Nq::Int
    quad_pts::Vector{Vector{Vec3}}
    areas::Vector{Float64}
    tri_ids::Matrix{Int}                 # (2, N)
    div_vals::Matrix{TD}                 # (2, N)
    rwg_vals::Vector{NTuple{2,Vector{TV}}}
    # Adjacent-cell near-singular integration data
    adjacent::BitMatrix                  # (Nt, Nt) adjacency matrix
    wq_hi::Vector{Float64}              # high-order quadrature weights
    quad_pts_hi::Vector{Vector{Vec3}}    # high-order quadrature points per triangle
    rwg_vals_hi::Vector{NTuple{2,Vector{TV}}}  # RWG values at high-order quad pts
end

function _build_efie_cache(mesh::TriMesh, rwg::RWGData, k;
                           quad_order::Int=3, eta0=376.730313668)
    N = rwg.nedges
    Nt = ntriangles(mesh)
    Tcoef = promote_type(eltype(rwg.coeff_plus), eltype(rwg.coeff_minus))
    TVec = SVector{3,Tcoef}

    omega_mu0 = k * eta0   # ωμ₀ = k η₀
    inv_k2 = 1 / k^2

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    # Precompute quadrature points and areas for all triangles
    quad_pts = Vector{Vector{Vec3}}(undef, Nt)
    areas = Vector{Float64}(undef, Nt)
    for t in 1:Nt
        quad_pts[t] = tri_quad_points(mesh, t, xi)
        areas[t] = triangle_area(mesh, t)
    end

    # High-order quadrature for inner integration of self/adjacent cells
    xi_hi, wq_hi = tri_quad_rule(7)
    Nq_hi = length(wq_hi)
    quad_pts_hi = Vector{Vector{Vec3}}(undef, Nt)
    for t in 1:Nt
        quad_pts_hi[t] = tri_quad_points(mesh, t, xi_hi)
    end

    tri_ids = zeros(Int, 2, N)
    div_vals = zeros(Tcoef, 2, N)
    rwg_vals = Vector{NTuple{2,Vector{TVec}}}(undef, N)
    rwg_vals_hi = Vector{NTuple{2,Vector{TVec}}}(undef, N)

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

        vals_p_hi = [eval_rwg(rwg, n, quad_pts_hi[tp][q], tp) for q in 1:Nq_hi]
        vals_m_hi = [eval_rwg(rwg, n, quad_pts_hi[tm][q], tm) for q in 1:Nq_hi]
        rwg_vals_hi[n] = (vals_p_hi, vals_m_hi)
    end

    # Build triangle adjacency BitMatrix from mesh connectivity
    # Two triangles are adjacent if they share a mesh edge (pair of vertex indices)
    adjacent = falses(Nt, Nt)
    edge_to_tri = Dict{NTuple{2,Int}, Vector{Int}}()
    for t in 1:Nt
        v1 = mesh.tri[1, t]
        v2 = mesh.tri[2, t]
        v3 = mesh.tri[3, t]
        for (va, vb) in ((v1, v2), (v2, v3), (v3, v1))
            ekey = va < vb ? (va, vb) : (vb, va)
            if haskey(edge_to_tri, ekey)
                push!(edge_to_tri[ekey], t)
            else
                edge_to_tri[ekey] = [t]
            end
        end
    end
    for (_, tris) in edge_to_tri
        for i in 1:length(tris)
            for j in (i+1):length(tris)
                t1, t2 = tris[i], tris[j]
                adjacent[t1, t2] = true
                adjacent[t2, t1] = true
            end
        end
    end

    return EFIEApplyCache(mesh, rwg, k, inv_k2, omega_mu0, wq, Nq, quad_pts, areas,
                          tri_ids, div_vals, rwg_vals,
                          adjacent, wq_hi, quad_pts_hi, rwg_vals_hi)
end

@inline function _is_adjacent(cache::EFIEApplyCache, t1::Int, t2::Int)
    return @inbounds cache.adjacent[t1, t2]
end

@inline function _efie_entry(cache::EFIEApplyCache, m::Int, n::Int)
    # For non-Bloch RWG, normalize to canonical order (m ≤ n) so that
    # _efie_entry(c, i, j) == _efie_entry(c, j, i) bitwise.
    # This keeps dense assembly (symmetry exploit) and on-the-fly getindex consistent.
    if !cache.rwg.has_periodic_bloch && m > n
        return _efie_entry(cache, n, m)
    end
    CT = ComplexF64
    val = zero(CT)
    inv_k2 = cache.inv_k2

    @inbounds for itm in 1:2
        tm = cache.tri_ids[itm, m]
        Am = cache.areas[tm]
        dvm = cache.div_vals[itm, m]
        fm_vals = _rwg_vals(cache, m, itm)
        fm_vals_hi = _rwg_vals_hi(cache, m, itm)

        for itn in 1:2
            tn = cache.tri_ids[itn, n]
            An = cache.areas[tn]
            dvn = cache.div_vals[itn, n]
            fn_vals = _rwg_vals(cache, n, itn)
            fn_vals_hi = _rwg_vals_hi(cache, n, itn)

            if tm == tn
                # Self-cell singularity extraction (high-order quadrature)
                val += _self_cell_cached(
                    cache.mesh, tm,
                    fm_vals_hi,
                    fn_vals_hi,
                    dvm, dvn,
                    Am, cache.k, inv_k2,
                    cache.wq_hi,
                    cache.quad_pts_hi[tm],
                )
            elseif _is_adjacent(cache, tm, tn)
                # Adjacent-cell near-singular integration
                val += _adjacent_cell_cached(
                    cache.mesh, tn,
                    cache.quad_pts[tm], cache.quad_pts[tn],
                    fm_vals, fn_vals,
                    fm_vals_hi, fn_vals_hi,
                    dvm, dvn,
                    Am, An,
                    cache.wq, cache.k, inv_k2,
                    cache.wq_hi, cache.quad_pts_hi[tm], cache.quad_pts_hi[tn],
                )
            else
                # Non-adjacent: standard product quadrature
                dvmn_inv_k2 = conj(dvm) * dvn * inv_k2
                for qm in 1:cache.Nq
                    rm = cache.quad_pts[tm][qm]
                    fm = fm_vals[qm]

                    for qn in 1:cache.Nq
                        rn = cache.quad_pts[tn][qn]
                        fn = fn_vals[qn]

                        G = greens(rm, rn, cache.k)

                        vec_part = dot(fm, fn) * G
                        scl_part = dvmn_inv_k2 * G
                        weight = cache.wq[qm] * cache.wq[qn] * (2 * Am) * (2 * An)

                        val += (vec_part - scl_part) * weight
                    end
                end
            end
        end
    end

    return -1im * cache.omega_mu0 * val
end

@inline function _rwg_vals(cache::EFIEApplyCache, n::Int, it::Int)
    if it == 1
        return cache.rwg_vals[n][1]
    end
    return cache.rwg_vals[n][2]
end

@inline function _rwg_vals_hi(cache::EFIEApplyCache, n::Int, it::Int)
    if it == 1
        return cache.rwg_vals_hi[n][1]
    end
    return cache.rwg_vals_hi[n][2]
end

# ── Internal cached versions of self/adjacent cell (no eval_rwg calls) ──

"""
Self-cell contribution using precomputed high-order RWG values.
Avoids calling eval_rwg in the hot inner loops.
"""
@inline function _self_cell_cached(
    mesh::TriMesh, tm::Int,
    fm_hi::Vector{<:SVector{3,<:Number}},
    fn_hi::Vector{<:SVector{3,<:Number}},
    div_m::Number, div_n::Number,
    Am::Float64, k, inv_k2,
    wq_hi, quad_pts_tm_hi::Vector{Vec3})

    Nq_hi = length(wq_hi)
    CT = complex(typeof(real(k)))

    V1 = _mesh_vertex(mesh, mesh.tri[1, tm])
    V2 = _mesh_vertex(mesh, mesh.tri[2, tm])
    V3 = _mesh_vertex(mesh, mesh.tri[3, tm])

    inv4pi = 1.0 / (4π)
    dvmn_inv_k2 = conj(div_m) * div_n * inv_k2

    # ── Smooth part: high-order product quadrature with G_smooth ──
    val_smooth = zero(CT)
    @inbounds for qm in 1:Nq_hi
        rm = quad_pts_tm_hi[qm]
        fm = fm_hi[qm]
        for qn in 1:Nq_hi
            rn = quad_pts_tm_hi[qn]
            fn = fn_hi[qn]

            Gs = greens_smooth(rm, rn, k)
            vec_part = dot(fm, fn) * Gs
            scl_part = dvmn_inv_k2 * Gs
            weight = wq_hi[qm] * wq_hi[qn] * (2 * Am) * (2 * Am)
            val_smooth += (vec_part - scl_part) * weight
        end
    end

    # ── Singular part: semi-analytical with high-order outer quadrature ──
    val_singular = zero(CT)
    @inbounds for qm in 1:Nq_hi
        rm = quad_pts_tm_hi[qm]
        fm = fm_hi[qm]

        S = analytical_integral_1overR(rm, V1, V2, V3)
        inner_scalar = inv4pi * S

        # Scalar potential singular part
        scl_sing = dvmn_inv_k2 * inner_scalar

        # Vector potential singular part: leading term
        fn_at_rm = fn_hi[qm]
        vec_lead = dot(fm, fn_at_rm) * inner_scalar

        # Vector potential singular part: remainder (bounded integrand)
        vec_rem = zero(CT)
        for qn in 1:Nq_hi
            rn = quad_pts_tm_hi[qn]
            fn = fn_hi[qn]

            R_vec = rm - rn
            R = sqrt(dot(R_vec, R_vec))
            if R < 1e-14
                continue
            end

            delta_fn = fn - fn_at_rm
            vec_rem += dot(fm, delta_fn) * (inv4pi / R) * wq_hi[qn] * (2 * Am)
        end

        outer_weight = wq_hi[qm] * (2 * Am)
        val_singular += ((vec_lead + vec_rem) - scl_sing) * outer_weight
    end

    return val_smooth + val_singular
end

"""
Adjacent-cell contribution using precomputed high-order RWG values.
"""
@inline function _adjacent_cell_cached(
    mesh::TriMesh, tn::Int,
    quad_pts_tm::Vector{Vec3}, quad_pts_tn::Vector{Vec3},
    fm_vals::Vector{<:SVector{3,<:Number}},
    fn_vals::Vector{<:SVector{3,<:Number}},
    fm_hi::Vector{<:SVector{3,<:Number}},
    fn_hi::Vector{<:SVector{3,<:Number}},
    div_m::Number, div_n::Number,
    Am::Float64, An::Float64,
    wq, k, inv_k2,
    wq_hi, quad_pts_tm_hi::Vector{Vec3}, quad_pts_tn_hi::Vector{Vec3})

    Nq = length(wq)
    Nq_hi = length(wq_hi)
    CT = complex(typeof(real(k)))

    V1n = _mesh_vertex(mesh, mesh.tri[1, tn])
    V2n = _mesh_vertex(mesh, mesh.tri[2, tn])
    V3n = _mesh_vertex(mesh, mesh.tri[3, tn])

    inv4pi = 1.0 / (4π)
    dvmn_inv_k2 = conj(div_m) * div_n * inv_k2

    # ── Smooth part: standard product quadrature with G_smooth ──
    val_smooth = zero(CT)
    @inbounds for qm in 1:Nq
        rm = quad_pts_tm[qm]
        fm = fm_vals[qm]
        for qn in 1:Nq
            rn = quad_pts_tn[qn]
            fn = fn_vals[qn]

            Gs = greens_smooth(rm, rn, k)
            vec_part = dot(fm, fn) * Gs
            scl_part = dvmn_inv_k2 * Gs
            weight = wq[qm] * wq[qn] * (2 * Am) * (2 * An)
            val_smooth += (vec_part - scl_part) * weight
        end
    end

    # ── Singular 1/(4πR) part: semi-analytical ──
    val_singular = zero(CT)
    @inbounds for qm in 1:Nq_hi
        rm = quad_pts_tm_hi[qm]
        fm = fm_hi[qm]

        # Analytical inner integral: S = ∫_{T_n} 1/|rm - r'| dS'
        S = analytical_integral_1overR(rm, V1n, V2n, V3n)
        inner_scalar = inv4pi * S

        # Scalar potential singular part
        scl_sing = dvmn_inv_k2 * inner_scalar

        # Vector potential singular part: ∫_{T_n} f_n(r') / (4πR) dS'
        vec_sing = zero(CT)
        for qn in 1:Nq_hi
            rn = quad_pts_tn_hi[qn]
            fn_qn = fn_hi[qn]

            R_vec = rm - rn
            R = sqrt(dot(R_vec, R_vec))
            if R < 1e-14
                continue
            end

            vec_sing += dot(fm, fn_qn) * (inv4pi / R) * wq_hi[qn] * (2 * An)
        end

        outer_weight = wq_hi[qm] * (2 * Am)
        val_singular += (vec_sing - scl_sing) * outer_weight
    end

    return val_smooth + val_singular
end

"""
    assemble_Z_efie(mesh, rwg, k; quad_order=3, mesh_precheck=true, allow_boundary=true, require_closed=false)

Assemble the dense EFIE matrix `Z_efie ∈ C^{N×N}`.
`k` is the wavenumber (can be complex for complex-step).
"""
function assemble_Z_efie(mesh::TriMesh, rwg::RWGData, k;
                         quad_order::Int=3, eta0=376.730313668,
                         mesh_precheck::Bool=true,
                         allow_boundary::Bool=true,
                         require_closed::Bool=false,
                         area_tol_rel::Float64=1e-12)
    if mesh_precheck
        assert_mesh_quality(mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end

    cache = _build_efie_cache(mesh, rwg, k; quad_order=quad_order, eta0=eta0)
    N = rwg.nedges
    CT = ComplexF64
    Z = zeros(CT, N, N)

    # Exploit Z symmetry for non-Bloch RWG (real coefficients → Z_mn = Z_nm)
    if !rwg.has_periodic_bloch
        Threads.@threads for m in 1:N
            @inbounds for n in m:N
                z = _efie_entry(cache, m, n)
                Z[m, n] = z
                Z[n, m] = z
            end
        end
    else
        Threads.@threads for m in 1:N
            @inbounds for n in 1:N
                Z[m, n] = _efie_entry(cache, m, n)
            end
        end
    end

    return Z
end

"""
Matrix-free EFIE operator `A` such that `A * x` computes the EFIE matvec
without allocating a dense `N×N` matrix.
"""
struct MatrixFreeEFIEOperator{T, TC<:EFIEApplyCache} <: AbstractMatrix{T}
    cache::TC
end

"""
Adjoint matrix-free EFIE operator `Aᴴ` used by GMRES adjoint solves.
"""
struct MatrixFreeEFIEAdjointOperator{T, TO<:MatrixFreeEFIEOperator{T}} <: AbstractMatrix{T}
    op::TO
end

Base.size(A::MatrixFreeEFIEOperator) = (A.cache.rwg.nedges, A.cache.rwg.nedges)
Base.eltype(::MatrixFreeEFIEOperator{T}) where {T} = T

Base.size(A::MatrixFreeEFIEAdjointOperator) = size(A.op)
Base.eltype(::MatrixFreeEFIEAdjointOperator{T}) where {T} = T

function efie_entry(A::MatrixFreeEFIEOperator, m::Int, n::Int)
    return _efie_entry(A.cache, m, n)
end

function Base.getindex(A::MatrixFreeEFIEOperator, i::Int, j::Int)
    return efie_entry(A, i, j)
end

function Base.getindex(A::MatrixFreeEFIEAdjointOperator, i::Int, j::Int)
    return conj(efie_entry(A.op, j, i))
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::MatrixFreeEFIEOperator{T}, x::AbstractVector) where {T<:Number}
    N = size(A, 1)
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))

    fill!(y, zero(T))
    @inbounds for m in 1:N
        acc = zero(T)
        for n in 1:N
            acc += efie_entry(A, m, n) * x[n]
        end
        y[m] = acc
    end
    return y
end

function Base.:*(A::MatrixFreeEFIEOperator{T}, x::AbstractVector) where {T<:Number}
    y = zeros(T, size(A, 1))
    mul!(y, A, x)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::MatrixFreeEFIEAdjointOperator{T}, x::AbstractVector) where {T<:Number}
    N = size(A, 1)
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))

    fill!(y, zero(T))
    @inbounds for n in 1:N
        acc = zero(T)
        for m in 1:N
            acc += conj(efie_entry(A.op, m, n)) * x[m]
        end
        y[n] = acc
    end
    return y
end

function Base.:*(A::MatrixFreeEFIEAdjointOperator{T}, x::AbstractVector) where {T<:Number}
    y = zeros(T, size(A, 1))
    mul!(y, A, x)
    return y
end

LinearAlgebra.adjoint(A::MatrixFreeEFIEOperator{T}) where {T<:Number} =
    MatrixFreeEFIEAdjointOperator{T,typeof(A)}(A)

"""
    matrixfree_efie_operator(mesh, rwg, k; kwargs...)

Create a matrix-free EFIE operator for GMRES matvecs without dense `Z` allocation.
"""
function matrixfree_efie_operator(mesh::TriMesh, rwg::RWGData, k;
                                  quad_order::Int=3, eta0=376.730313668,
                                  mesh_precheck::Bool=true,
                                  allow_boundary::Bool=true,
                                  require_closed::Bool=false,
                                  area_tol_rel::Float64=1e-12)
    if mesh_precheck
        assert_mesh_quality(mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end
    cache = _build_efie_cache(mesh, rwg, k; quad_order=quad_order, eta0=eta0)
    return MatrixFreeEFIEOperator{ComplexF64,typeof(cache)}(cache)
end

"""
    matrixfree_efie_adjoint_operator(A)

Return matrix-free adjoint operator `Aᴴ` for Krylov adjoint solves.
"""
matrixfree_efie_adjoint_operator(A::MatrixFreeEFIEOperator) = adjoint(A)
