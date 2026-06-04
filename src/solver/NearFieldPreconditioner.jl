# NearFieldPreconditioner.jl — Near-field sparse preconditioner for EFIE systems
#
# Builds a sparse approximation of Z by retaining only entries corresponding
# to RWG basis functions within a near-field cutoff distance. The LU
# factorization of this sparse matrix serves as a preconditioner for GMRES.
#
# This is the standard and effective approach for preconditioning dense MoM
# systems: near-field interactions dominate the matrix structure, while
# far-field entries decay as 1/r and contribute less to conditioning.
#
# Key advantage: the near-field preconditioner explicitly captures the spatial
# coupling structure of the Green's function, giving N-independent iteration
# counts for GMRES.

using SparseArrays
using IncompleteLU

export NearFieldPreconditionerData,
       ILUPreconditionerData,
       DiagonalPreconditionerData,
       BlockDiagPrecondData,
       PermutedPrecondData,
       AbstractPreconditionerData,
       build_nearfield_preconditioner,
       build_block_diag_preconditioner,
       build_mlfma_preconditioner,
       NearFieldOperator,
       NearFieldAdjointOperator,
       rwg_centers

abstract type AbstractPreconditionerData end

"""
    NearFieldPreconditionerData <: AbstractPreconditionerData

Stores the sparse-LU near-field preconditioner.
"""
struct NearFieldPreconditionerData <: AbstractPreconditionerData
    Z_nf_fac::SparseArrays.UMFPACK.UmfpackLU{ComplexF64, Int64}
    cutoff::Float64
    nnz_ratio::Float64
end

"""
    DiagonalPreconditionerData <: AbstractPreconditionerData

Stores a Jacobi/diagonal preconditioner as the inverse diagonal entries.
"""
struct DiagonalPreconditionerData <: AbstractPreconditionerData
    dinv::Vector{ComplexF64}
    cutoff::Float64
    nnz_ratio::Float64
end

"""
    ILUPreconditionerData <: AbstractPreconditionerData

Stores an incomplete LU (ILU) factorization of the near-field sparse matrix.
Uses IncompleteLU.jl's Crout ILU with drop tolerance τ.

Compared to full sparse LU (`NearFieldPreconditionerData`):
- Much less fill-in → feasible for large N with moderate nnz%
- Slightly weaker preconditioner → more GMRES iterations
- Memory: controlled by τ (smaller τ = more fill = better preconditioner)
"""
struct ILUPreconditionerData <: AbstractPreconditionerData
    ilu_fac::IncompleteLU.ILUFactorization{ComplexF64, Int64}
    cutoff::Float64
    nnz_ratio::Float64
    tau::Float64
end

"""
    BlockDiagPrecondData <: AbstractPreconditionerData

Block-diagonal (block-Jacobi) preconditioner from MLFMA leaf boxes.
Each leaf box contributes a small dense diagonal block from Z_near,
which is LU-factorized independently. Very fast to build and memory-efficient.
"""
struct BlockDiagPrecondData <: AbstractPreconditionerData
    lu_blocks::Vector{LinearAlgebra.LU{ComplexF64, Matrix{ComplexF64}, Vector{Int64}}}
    box_bf_indices::Vector{Vector{Int}}   # original BF indices per leaf box
    N::Int
    nnz_ratio::Float64
end

"""
    PermutedPrecondData <: AbstractPreconditionerData

Wraps an inner preconditioner (ILU or LU) that operates in a permuted ordering.
On apply: permute input → apply inner preconditioner → unpermute output.
Used for MLFMA where reordering Z_near to block-banded form before ILU
dramatically speeds up factorization and improves quality.
"""
struct PermutedPrecondData{T<:AbstractPreconditionerData} <: AbstractPreconditionerData
    inner::T
    perm::Vector{Int}    # perm[new] = old (original → permuted)
    iperm::Vector{Int}   # iperm[old] = new (permuted → original)
    N::Int
    nnz_ratio::Float64
end

"""
    rwg_centers(mesh, rwg)

Compute the center point of each RWG basis function, defined as the average
of the centroids of its two supporting triangles.

Returns a Vector{Vec3} of length N.
"""
function rwg_centers(mesh::TriMesh, rwg::RWGData)
    N = rwg.nedges
    centers = Vector{Vec3}(undef, N)
    for n in 1:N
        c_plus  = triangle_center(mesh, rwg.tplus[n])
        c_minus = triangle_center(mesh, rwg.tminus[n])
        centers[n] = 0.5 * (c_plus + c_minus)
    end
    return centers
end

@inline function _cell_key(r::Vec3, inv_cell::Float64)
    return (
        floor(Int, r[1] * inv_cell),
        floor(Int, r[2] * inv_cell),
        floor(Int, r[3] * inv_cell),
    )
end

function _nearfield_triplets_bruteforce(centers::Vector{Vec3}, cutoff::Float64, getvalue)
    N = length(centers)
    I_idx = Int[]
    J_idx = Int[]
    V_val = ComplexF64[]

    if N == 0
        return I_idx, J_idx, V_val
    end

    if cutoff <= 0
        sizehint!(I_idx, N)
        sizehint!(J_idx, N)
        sizehint!(V_val, N)
        @inbounds for m in 1:N
            push!(I_idx, m)
            push!(J_idx, m)
            push!(V_val, ComplexF64(getvalue(m, m)))
        end
        return I_idx, J_idx, V_val
    end

    if !isfinite(cutoff)
        sizehint!(I_idx, N * N)
        sizehint!(J_idx, N * N)
        sizehint!(V_val, N * N)
        @inbounds for m in 1:N
            for n in 1:N
                push!(I_idx, m)
                push!(J_idx, n)
                push!(V_val, ComplexF64(getvalue(m, n)))
            end
        end
        return I_idx, J_idx, V_val
    end

    cutoff2 = cutoff * cutoff
    @inbounds for m in 1:N
        cm = centers[m]
        for n in 1:N
            δ = cm - centers[n]
            d2 = dot(δ, δ)
            if m == n || d2 <= cutoff2
                push!(I_idx, m)
                push!(J_idx, n)
                push!(V_val, ComplexF64(getvalue(m, n)))
            end
        end
    end
    return I_idx, J_idx, V_val
end

function _nearfield_triplets_spatial(centers::Vector{Vec3}, cutoff::Float64, getvalue)
    N = length(centers)
    I_idx = Int[]
    J_idx = Int[]
    V_val = ComplexF64[]

    if N == 0
        return I_idx, J_idx, V_val
    end

    if cutoff <= 0 || !isfinite(cutoff)
        return _nearfield_triplets_bruteforce(centers, cutoff, getvalue)
    end

    inv_cell = 1.0 / cutoff
    buckets = Dict{NTuple{3,Int}, Vector{Int}}()
    @inbounds for i in 1:N
        key = _cell_key(centers[i], inv_cell)
        push!(get!(() -> Int[], buckets, key), i)
    end

    cutoff2 = cutoff * cutoff
    est_pairs = max(N, min(N * N, 27 * N))
    sizehint!(I_idx, est_pairs)
    sizehint!(J_idx, est_pairs)
    sizehint!(V_val, est_pairs)

    @inbounds for m in 1:N
        cm = centers[m]
        key = _cell_key(cm, inv_cell)
        for dz in -1:1
            for dy in -1:1
                for dx in -1:1
                    key_n = (key[1] + dx, key[2] + dy, key[3] + dz)
                    n_list = get(buckets, key_n, nothing)
                    n_list === nothing && continue
                    for n in n_list
                        if m == n
                            push!(I_idx, m)
                            push!(J_idx, n)
                            push!(V_val, ComplexF64(getvalue(m, n)))
                        else
                            δ = cm - centers[n]
                            d2 = dot(δ, δ)
                            if d2 <= cutoff2
                                push!(I_idx, m)
                                push!(J_idx, n)
                                push!(V_val, ComplexF64(getvalue(m, n)))
                            end
                        end
                    end
                end
            end
        end
    end
    return I_idx, J_idx, V_val
end

"""
Single-pass near-field triplet assembly with lazy Green's caching for EFIE.

Iterates the spatial-hash neighbor structure and computes EFIE entries on
the fly, but caches Green's quadrature matrices G[qm,qn] per unique
(tm, tn) triangle pair so that shared pairs are evaluated only once.
Self-cell and adjacent-cell terms bypass the cache entirely.
"""
function _nearfield_triplets_batched(cache::EFIEApplyCache, centers::Vector{Vec3},
                                      cutoff::Float64)
    N = length(centers)
    N == 0 && return Int[], Int[], ComplexF64[]

    if cutoff <= 0 || !isfinite(cutoff)
        return _nearfield_triplets_spatial(centers, cutoff,
                                            (m, n) -> _efie_entry(cache, m, n))
    end

    # Spatial hash
    inv_cell = 1.0 / cutoff
    buckets = Dict{NTuple{3,Int}, Vector{Int}}()
    @inbounds for i in 1:N
        key = _cell_key(centers[i], inv_cell)
        push!(get!(() -> Int[], buckets, key), i)
    end

    cutoff2 = cutoff * cutoff
    est_pairs = max(N, min(N * N, 27 * N))
    I_idx = Int[];       sizehint!(I_idx, est_pairs)
    J_idx = Int[];       sizehint!(J_idx, est_pairs)
    V_val = ComplexF64[]; sizehint!(V_val, est_pairs)

    Nq     = cache.Nq
    inv_k2 = cache.inv_k2

    # Lazy Green's cache: computed once per unique (tm, tn) pair
    green_cache = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()

    @inline function _get_greens(tm::Int, tn::Int)
        val = get(green_cache, (tm, tn), nothing)
        val !== nothing && return val
        G_mat = Matrix{ComplexF64}(undef, Nq, Nq)
        @inbounds for qm in 1:Nq, qn in 1:Nq
            G_mat[qm, qn] = greens(cache.quad_pts[tm][qm], cache.quad_pts[tn][qn], cache.k)
        end
        green_cache[(tm, tn)] = G_mat
        return G_mat
    end

    @inbounds for m in 1:N
        cm = centers[m]
        key = _cell_key(cm, inv_cell)
        for dz in -1:1, dy in -1:1, dx in -1:1
            key_n = (key[1] + dx, key[2] + dy, key[3] + dz)
            n_list = get(buckets, key_n, nothing)
            n_list === nothing && continue
            for n in n_list
                is_near = (m == n) || (let δ = cm - centers[n]; dot(δ, δ) end) <= cutoff2
                is_near || continue

                # Compute EFIE entry with cached Green's
                val = zero(ComplexF64)
                for itm in 1:2
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
                            val += _self_cell_cached(
                                cache.mesh, tm,
                                fm_vals_hi, fn_vals_hi,
                                dvm, dvn,
                                Am, cache.k, inv_k2,
                                cache.wq_hi, cache.quad_pts_hi[tm])
                        elseif _is_adjacent(cache, tm, tn)
                            val += _adjacent_cell_cached(
                                cache.mesh, tn,
                                cache.quad_pts[tm], cache.quad_pts[tn],
                                fm_vals, fn_vals,
                                fm_vals_hi, fn_vals_hi,
                                dvm, dvn, Am, An,
                                cache.wq, cache.k, inv_k2,
                                cache.wq_hi, cache.quad_pts_hi[tm], cache.quad_pts_hi[tn])
                        else
                            G_mat = _get_greens(tm, tn)
                            dvmn_inv_k2 = conj(dvm) * dvn * inv_k2
                            for qm in 1:Nq
                                fm = fm_vals[qm]
                                for qn in 1:Nq
                                    G = G_mat[qm, qn]
                                    vec_part = dot(fm, fn_vals[qn]) * G
                                    scl_part = dvmn_inv_k2 * G
                                    weight = cache.wq[qm] * cache.wq[qn] * (2*Am) * (2*An)
                                    val += (vec_part - scl_part) * weight
                                end
                            end
                        end
                    end
                end

                push!(I_idx, m)
                push!(J_idx, n)
                push!(V_val, -1im * cache.omega_mu0 * val)
            end
        end
    end

    return I_idx, J_idx, V_val
end

function _build_diagonal_preconditioner_data(getvalue, N::Int, cutoff::Float64)
    diag_entries = Vector{ComplexF64}(undef, N)
    maxabs = 0.0
    @inbounds for i in 1:N
        zii = ComplexF64(getvalue(i, i))
        diag_entries[i] = zii
        ai = abs(zii)
        if ai > maxabs
            maxabs = ai
        end
    end

    floor_abs = 1e-10 * max(maxabs, 1.0)
    @inbounds for i in 1:N
        if abs(diag_entries[i]) < floor_abs
            diag_entries[i] = floor_abs + 0im
        end
    end

    dinv = similar(diag_entries)
    @inbounds for i in 1:N
        dinv[i] = inv(diag_entries[i])
    end

    nnz_ratio = N == 0 ? 0.0 : (N / (N * N))
    return DiagonalPreconditionerData(dinv, cutoff, nnz_ratio)
end

function _build_nearfield_preconditioner_from_entries(mesh::TriMesh, rwg::RWGData, cutoff::Float64,
                                                       getvalue;
                                                       neighbor_search::Symbol=:spatial,
                                                       factorization::Symbol=:lu,
                                                       ilu_tau::Float64=1e-3)
    N = rwg.nedges

    if factorization == :diag
        return _build_diagonal_preconditioner_data(getvalue, N, cutoff)
    elseif factorization ∉ (:lu, :ilu)
        error("Invalid factorization: $factorization (expected :lu, :ilu, or :diag)")
    end

    centers = rwg_centers(mesh, rwg)

    I_idx, J_idx, V_val = if neighbor_search == :spatial
        _nearfield_triplets_spatial(centers, cutoff, getvalue)
    elseif neighbor_search == :bruteforce
        _nearfield_triplets_bruteforce(centers, cutoff, getvalue)
    else
        error("Invalid neighbor_search: $neighbor_search (expected :spatial or :bruteforce)")
    end

    Z_nf = sparse(I_idx, J_idx, V_val, N, N)
    nnz_ratio = nnz(Z_nf) / max(N * N, 1)

    if factorization == :ilu
        ilu_fac = IncompleteLU.ilu(Z_nf, τ = ilu_tau)
        return ILUPreconditionerData(ilu_fac, cutoff, nnz_ratio, ilu_tau)
    else
        Z_nf_fac = lu(Z_nf)
        return NearFieldPreconditionerData(Z_nf_fac, cutoff, nnz_ratio)
    end
end

"""
    build_nearfield_preconditioner(Z, mesh, rwg, cutoff)

Build a near-field sparse preconditioner for the MoM matrix Z.

Retains entries Z[m,n] where the distance between the centers of RWG basis
functions m and n is ≤ `cutoff`. The sparse matrix is then LU-factorized.

# Arguments
- `Z::Matrix{ComplexF64}`: the full N×N MoM matrix
- `mesh::TriMesh`: the triangle mesh
- `rwg::RWGData`: RWG basis function data
- `cutoff::Float64`: distance cutoff (e.g., 0.5λ to 2λ)
- `neighbor_search`: `:spatial` (default) or `:bruteforce` reference mode

# Returns
A `NearFieldPreconditionerData` containing the factorized near-field matrix.
"""
function build_nearfield_preconditioner(Z::Matrix{<:Number}, mesh::TriMesh,
                                         rwg::RWGData, cutoff::Float64;
                                         neighbor_search::Symbol=:spatial,
                                         factorization::Symbol=:lu,
                                         ilu_tau::Float64=1e-3)
    N = rwg.nedges
    size(Z, 1) == N && size(Z, 2) == N ||
        throw(DimensionMismatch("Z has size $(size(Z)), expected ($N, $N) for RWG basis"))
    return _build_nearfield_preconditioner_from_entries(mesh, rwg, cutoff,
        (m, n) -> Z[m, n];
        neighbor_search=neighbor_search,
        factorization=factorization,
        ilu_tau=ilu_tau,
    )
end

"""
    build_nearfield_preconditioner(A, mesh, rwg, cutoff)

Build a near-field sparse preconditioner directly from an operator `A`
without allocating a full dense matrix.
"""
function build_nearfield_preconditioner(A::AbstractMatrix{<:Number}, mesh::TriMesh,
                                         rwg::RWGData, cutoff::Float64;
                                         neighbor_search::Symbol=:spatial,
                                         factorization::Symbol=:lu,
                                         ilu_tau::Float64=1e-3)
    N = rwg.nedges
    size(A, 1) == N && size(A, 2) == N ||
        throw(DimensionMismatch("A has size $(size(A)), expected ($N, $N) for RWG basis"))
    return _build_nearfield_preconditioner_from_entries(mesh, rwg, cutoff,
        (m, n) -> A[m, n];
        neighbor_search=neighbor_search,
        factorization=factorization,
        ilu_tau=ilu_tau,
    )
end

"""
    build_nearfield_preconditioner(A::MatrixFreeEFIEOperator, cutoff)

Build a near-field preconditioner from a matrix-free EFIE operator using
batched Green's function evaluation.  Triangle-pair Green's matrices are
precomputed once and reused across RWG pairs that share triangles.
"""
function build_nearfield_preconditioner(A::MatrixFreeEFIEOperator, cutoff::Float64;
                                         neighbor_search::Symbol=:spatial,
                                         factorization::Symbol=:lu,
                                         ilu_tau::Float64=1e-3)
    cache = A.cache
    N = cache.rwg.nedges

    if factorization == :diag
        return _build_diagonal_preconditioner_data(
            (m, n) -> _efie_entry(cache, m, n), N, cutoff)
    elseif factorization ∉ (:lu, :ilu)
        error("Invalid factorization: $factorization (expected :lu, :ilu, or :diag)")
    end

    if neighbor_search ∉ (:spatial, :bruteforce)
        error("Invalid neighbor_search: $neighbor_search (expected :spatial or :bruteforce)")
    end

    centers = rwg_centers(cache.mesh, cache.rwg)
    I_idx, J_idx, V_val = if neighbor_search == :bruteforce
        _nearfield_triplets_bruteforce(centers, cutoff,
                                        (m, n) -> _efie_entry(cache, m, n))
    else
        _nearfield_triplets_batched(cache, centers, cutoff)
    end

    Z_nf = sparse(I_idx, J_idx, V_val, N, N)
    nnz_ratio = nnz(Z_nf) / max(N * N, 1)

    if factorization == :ilu
        ilu_fac = IncompleteLU.ilu(Z_nf, τ = ilu_tau)
        return ILUPreconditionerData(ilu_fac, cutoff, nnz_ratio, ilu_tau)
    else
        Z_nf_fac = lu(Z_nf)
        return NearFieldPreconditionerData(Z_nf_fac, cutoff, nnz_ratio)
    end
end

"""
    build_nearfield_preconditioner(mesh, rwg, k, cutoff; kwargs...)

Build a near-field sparse preconditioner directly from EFIE geometry/physics
inputs, without dense EFIE assembly.
"""
function build_nearfield_preconditioner(mesh::TriMesh, rwg::RWGData, k, cutoff::Float64;
                                         quad_order::Int=3,
                                         eta0::Float64=376.730313668,
                                         mesh_precheck::Bool=true,
                                         allow_boundary::Bool=true,
                                         require_closed::Bool=false,
                                         area_tol_rel::Float64=1e-12,
                                         factorization::Symbol=:lu,
                                         ilu_tau::Float64=1e-3)
    A = matrixfree_efie_operator(mesh, rwg, k;
        quad_order=quad_order,
        eta0=eta0,
        mesh_precheck=mesh_precheck,
        allow_boundary=allow_boundary,
        require_closed=require_closed,
        area_tol_rel=area_tol_rel,
    )
    return build_nearfield_preconditioner(A, cutoff;
        factorization=factorization,
        ilu_tau=ilu_tau,
    )
end

"""
    build_nearfield_preconditioner(A_aca::ACAOperator; factorization=:lu, ilu_tau=1e-3)

Build a preconditioner by extracting the dense (inadmissible) blocks that are
already computed inside the ACA H-matrix operator.  This avoids recomputing
any EFIE entries — the near-field matrix is assembled directly from the block
data using the cluster-tree permutation.
"""
function build_nearfield_preconditioner(A_aca::ACAOperator;
                                         factorization::Symbol=:lu,
                                         ilu_tau::Float64=1e-3)
    N = A_aca.N
    perm = A_aca.tree.perm   # tree-order → original index

    # Pre-allocate triplets: dense blocks tile the near field without overlap
    total_entries = sum(length(db.row_range) * length(db.col_range)
                        for db in A_aca.dense_blocks; init=0)
    I_idx = Vector{Int}(undef, total_entries)
    J_idx = Vector{Int}(undef, total_entries)
    V_val = Vector{ComplexF64}(undef, total_entries)

    pos = 0
    @inbounds for db in A_aca.dense_blocks
        nr = length(db.row_range)
        nc = length(db.col_range)
        for jj in 1:nc
            j_orig = perm[db.col_range[jj]]
            for ii in 1:nr
                i_orig = perm[db.row_range[ii]]
                pos += 1
                I_idx[pos] = i_orig
                J_idx[pos] = j_orig
                V_val[pos] = db.data[ii, jj]
            end
        end
    end

    Z_nf = sparse(I_idx, J_idx, V_val, N, N)
    return build_nearfield_preconditioner(Z_nf; factorization=factorization,
                                           ilu_tau=ilu_tau)
end

"""
    build_nearfield_preconditioner(Z_nf_sparse; factorization=:lu, ilu_tau=1e-3)

Build a preconditioner directly from a pre-assembled sparse near-field matrix.
Skips the distance-based neighbor search (useful for MLFMA where Z_near is
already assembled with the correct sparsity pattern from the octree).
"""
function build_nearfield_preconditioner(Z_nf::SparseMatrixCSC{ComplexF64,Int};
                                         factorization::Symbol=:lu,
                                         ilu_tau::Float64=1e-3)
    N = size(Z_nf, 1)
    nnz_ratio = nnz(Z_nf) / max(N * N, 1)

    if factorization == :ilu
        ilu_fac = IncompleteLU.ilu(Z_nf, τ = ilu_tau)
        return ILUPreconditionerData(ilu_fac, Inf, nnz_ratio, ilu_tau)
    elseif factorization == :lu
        Z_nf_fac = lu(Z_nf)
        return NearFieldPreconditionerData(Z_nf_fac, Inf, nnz_ratio)
    elseif factorization == :diag
        dinv = Vector{ComplexF64}(undef, N)
        maxabs = 0.0
        @inbounds for i in 1:N
            zii = Z_nf[i, i]
            dinv[i] = zii
            ai = abs(zii)
            if ai > maxabs; maxabs = ai; end
        end
        floor_abs = 1e-10 * max(maxabs, 1.0)
        @inbounds for i in 1:N
            if abs(dinv[i]) < floor_abs
                dinv[i] = floor_abs + 0im
            end
            dinv[i] = inv(dinv[i])
        end
        return DiagonalPreconditionerData(dinv, Inf, nnz_ratio)
    else
        error("Invalid factorization: $factorization (expected :lu, :ilu, or :diag)")
    end
end

"""
    NearFieldOperator

Callable wrapper for use with Krylov.jl as a preconditioner.
Applies Z_nf⁻¹ v.
"""
struct NearFieldOperator{PType<:AbstractPreconditionerData}
    P::PType
end

@inline _preconditioner_size(P::NearFieldPreconditionerData) = size(P.Z_nf_fac, 1)
@inline _preconditioner_size(P::DiagonalPreconditionerData) = length(P.dinv)
@inline _preconditioner_size(P::ILUPreconditionerData) = size(P.ilu_fac.L, 1)
@inline _preconditioner_size(P::BlockDiagPrecondData) = P.N
@inline _preconditioner_size(P::PermutedPrecondData) = P.N

@inline function _apply_preconditioner!(y::StridedVector{ComplexF64}, P::NearFieldPreconditionerData)
    ldiv!(P.Z_nf_fac, y)
    return y
end

@inline function _apply_preconditioner!(y::StridedVector{ComplexF64}, P::DiagonalPreconditionerData)
    @inbounds @simd for i in eachindex(y)
        y[i] *= P.dinv[i]
    end
    return y
end

@inline function _apply_preconditioner!(y::StridedVector{ComplexF64}, P::ILUPreconditionerData)
    ldiv!(P.ilu_fac, y)
    return y
end

@inline function _apply_preconditioner!(y::StridedVector{ComplexF64}, P::BlockDiagPrecondData)
    # Apply block solves using original BF indices (Z_near is in original ordering)
    for (i, idx) in enumerate(P.box_bf_indices)
        y[idx] = P.lu_blocks[i] \ y[idx]
    end
    return y
end

@inline function _apply_preconditioner_adjoint!(y::StridedVector{ComplexF64}, P::NearFieldPreconditionerData)
    ldiv!(adjoint(P.Z_nf_fac), y)
    return y
end

@inline function _apply_preconditioner_adjoint!(y::StridedVector{ComplexF64}, P::DiagonalPreconditionerData)
    @inbounds @simd for i in eachindex(y)
        y[i] *= conj(P.dinv[i])
    end
    return y
end

@inline function _apply_preconditioner_adjoint!(y::StridedVector{ComplexF64}, P::ILUPreconditionerData)
    # (LU)⁻ᴴ y = U⁻ᴴ L⁻ᴴ y
    # Step 1: solve Uᴴ z = y  (Uᴴ is lower triangular)
    ldiv!(adjoint(UpperTriangular(P.ilu_fac.U)), y)
    # Step 2: solve Lᴴ x = z  (Lᴴ is unit upper triangular)
    ldiv!(adjoint(UnitLowerTriangular(P.ilu_fac.L)), y)
    return y
end

@inline function _apply_preconditioner_adjoint!(y::StridedVector{ComplexF64}, P::BlockDiagPrecondData)
    for (i, idx) in enumerate(P.box_bf_indices)
        y[idx] = adjoint(P.lu_blocks[i]) \ y[idx]
    end
    return y
end

@inline function _apply_preconditioner!(y::StridedVector{ComplexF64}, P::PermutedPrecondData)
    # Permute to reordered space, apply inner preconditioner, unpermute
    y .= y[P.perm]                         # original → permuted ordering
    _apply_preconditioner!(y, P.inner)      # solve in permuted ordering
    y .= y[P.iperm]                         # permuted → original ordering
    return y
end

@inline function _apply_preconditioner_adjoint!(y::StridedVector{ComplexF64}, P::PermutedPrecondData)
    y .= y[P.perm]
    _apply_preconditioner_adjoint!(y, P.inner)
    y .= y[P.iperm]
    return y
end

Base.size(op::NearFieldOperator) = (_preconditioner_size(op.P), _preconditioner_size(op.P))
Base.eltype(::NearFieldOperator) = ComplexF64

function Base.:*(op::NearFieldOperator, v::AbstractVector)
    y = Vector{ComplexF64}(v)
    _apply_preconditioner!(y, op.P)
    return y
end

function LinearAlgebra.mul!(y::StridedVector{ComplexF64}, op::NearFieldOperator, x::StridedVector{ComplexF64})
    length(y) == length(x) || throw(DimensionMismatch("x length $(length(x)) != $(length(y))"))
    if y !== x
        copyto!(y, x)
    end
    _apply_preconditioner!(y, op.P)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, op::NearFieldOperator, x::AbstractVector)
    length(y) == length(x) || throw(DimensionMismatch("x length $(length(x)) != $(length(y))"))
    y .= op * x
    return y
end

"""
    NearFieldAdjointOperator

Callable wrapper for the adjoint preconditioner Z_nf⁻ᴴ v.
"""
struct NearFieldAdjointOperator{PType<:AbstractPreconditionerData}
    P::PType
end

Base.size(op::NearFieldAdjointOperator) = (_preconditioner_size(op.P), _preconditioner_size(op.P))
Base.eltype(::NearFieldAdjointOperator) = ComplexF64

function Base.:*(op::NearFieldAdjointOperator, v::AbstractVector)
    y = Vector{ComplexF64}(v)
    _apply_preconditioner_adjoint!(y, op.P)
    return y
end

function LinearAlgebra.mul!(y::StridedVector{ComplexF64}, op::NearFieldAdjointOperator, x::StridedVector{ComplexF64})
    length(y) == length(x) || throw(DimensionMismatch("x length $(length(x)) != $(length(y))"))
    if y !== x
        copyto!(y, x)
    end
    _apply_preconditioner_adjoint!(y, op.P)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, op::NearFieldAdjointOperator, x::AbstractVector)
    length(y) == length(x) || throw(DimensionMismatch("x length $(length(x)) != $(length(y))"))
    y .= op * x
    return y
end

"""
    build_block_diag_preconditioner(A_mlfma)

Build a block-diagonal (block-Jacobi) preconditioner from MLFMA leaf boxes.
Each leaf box's diagonal block in Z_near is LU-factorized independently.

Much faster than ILU for large N: O(n_boxes × n_bf³) where n_bf is the
average BFs per box (typically 100-500), versus O(nnz × fill) for ILU.
Memory: n_boxes × n_bf² × 16 bytes (typically < 100 MB).
"""
function build_block_diag_preconditioner(A_mlfma)
    octree = A_mlfma.octree
    leaf_level = octree.levels[end]
    N = size(A_mlfma, 1)

    lu_blocks = LinearAlgebra.LU{ComplexF64, Matrix{ComplexF64}, Vector{Int64}}[]
    box_bf_indices = Vector{Int}[]
    total_nnz = 0
    for box in leaf_level.boxes
        # Map from MLFMA ordering to original BF indices (Z_near is in original ordering)
        orig_idx = [octree.perm[i] for i in box.bf_range]
        block = Matrix(A_mlfma.Z_near[orig_idx, orig_idx])
        push!(lu_blocks, lu(block))
        push!(box_bf_indices, orig_idx)
        total_nnz += length(orig_idx)^2
    end

    nnz_ratio = total_nnz / N^2
    mem_mb = round(total_nnz * 16 / 1e6, digits=1)
    n_boxes = length(lu_blocks)
    avg_bf = round(N / n_boxes, digits=0)
    println("  Block-diag preconditioner: $n_boxes blocks, avg $(Int(avg_bf)) BFs/block, $(mem_mb) MB")

    return BlockDiagPrecondData(lu_blocks, box_bf_indices, N, nnz_ratio)
end

"""
    build_mlfma_preconditioner(A_mlfma; factorization=:ilu, ilu_tau=1e-2)

Build a preconditioner for MLFMA by reordering Z_near to MLFMA ordering
(block-banded structure) before factorization. This makes ILU dramatically
faster and more effective than operating on the original scattered ordering.

The resulting preconditioner handles the permutation automatically.
"""
function build_mlfma_preconditioner(A_mlfma;
                                     factorization::Symbol=:ilu,
                                     ilu_tau::Float64=1e-2)
    octree = A_mlfma.octree
    perm = copy(octree.perm)
    N = size(A_mlfma, 1)

    # Build inverse permutation
    iperm = zeros(Int, N)
    for i in 1:N
        iperm[perm[i]] = i
    end

    # Reorder Z_near: Z_perm[i,j] = Z_near[perm[i], perm[j]]
    # In MLFMA ordering, the matrix has block-banded structure
    println("  Reordering Z_near to MLFMA ordering...")
    Z_perm = A_mlfma.Z_near[perm, perm]

    nnz_ratio = nnz(Z_perm) / max(N * N, 1)

    if factorization == :ilu
        println("  Running ILU(τ=$(ilu_tau)) on reordered matrix ($(nnz(Z_perm)) nnz)...")
        t = @elapsed ilu_fac = IncompleteLU.ilu(Z_perm, τ = ilu_tau)
        Z_perm = nothing; GC.gc()  # free the reordered copy to save memory
        inner = ILUPreconditionerData(ilu_fac, Inf, nnz_ratio, ilu_tau)
        println("  ILU done in $(round(t, digits=1))s")
    elseif factorization == :lu
        println("  Running sparse LU on reordered matrix...")
        t = @elapsed Z_fac = lu(Z_perm)
        Z_perm = nothing; GC.gc()
        inner = NearFieldPreconditionerData(Z_fac, Inf, nnz_ratio)
        println("  LU done in $(round(t, digits=1))s")
    else
        error("Invalid factorization: $factorization (expected :ilu or :lu)")
    end

    return PermutedPrecondData(inner, perm, iperm, N, nnz_ratio)
end
