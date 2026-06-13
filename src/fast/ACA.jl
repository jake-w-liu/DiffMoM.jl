# ACA.jl — Adaptive Cross Approximation and H-matrix operator
#
# Compresses the EFIE matrix into near-field (dense) and far-field (low-rank)
# blocks using a cluster tree and the standard ACA admissibility condition.
# The resulting ACAOperator <: AbstractMatrix supports O(N log² N) matvec
# and plugs directly into the existing GMRES/preconditioner infrastructure.

export ACAOperator, ACAAdjointOperator, build_aca_operator
export aca_lowrank

"""
    DenseBlock

A near-field (inadmissible) block stored as a full dense sub-matrix.
Row/column ranges are in tree-permuted order.
"""
struct DenseBlock
    row_range::UnitRange{Int}
    col_range::UnitRange{Int}
    data::Matrix{ComplexF64}
end

"""
    LowRankBlock

A far-field (admissible) block stored in factored form U * V'.
`U` is (m, k) and `V` is (n, k), so the block approximation is U * V'.
Row/column ranges are in tree-permuted order.
"""
struct LowRankBlock
    row_range::UnitRange{Int}
    col_range::UnitRange{Int}
    U::Matrix{ComplexF64}
    V::Matrix{ComplexF64}
end

"""
Pre-allocated workspace for ACA `mul!`, eliminating per-call allocations.
"""
mutable struct ACAWorkspace
    x_perm::Vector{ComplexF64}
    y_perm::Vector{ComplexF64}
    tmp::Vector{ComplexF64}   # sized to max rank across all low-rank blocks
end

"""
    ACAOperator{TC} <: AbstractMatrix{ComplexF64}

H-matrix operator assembled via ACA. Supports `mul!` for GMRES
and `getindex` for preconditioner construction.
"""
struct ACAOperator{TC<:EFIEApplyCache} <: AbstractMatrix{ComplexF64}
    cache::TC
    tree::ClusterTree
    dense_blocks::Vector{DenseBlock}
    lowrank_blocks::Vector{LowRankBlock}
    N::Int
    workspace::ACAWorkspace
end

"""
    ACAAdjointOperator{TA} <: AbstractMatrix{ComplexF64}

Adjoint of an ACA operator for adjoint GMRES solves.
"""
struct ACAAdjointOperator{TA<:ACAOperator} <: AbstractMatrix{ComplexF64}
    op::TA
end

Base.size(A::ACAOperator) = (A.N, A.N)
Base.eltype(::ACAOperator) = ComplexF64
Base.size(A::ACAAdjointOperator) = size(A.op)
Base.eltype(::ACAAdjointOperator) = ComplexF64

# Fallback getindex via entry evaluation (for NF preconditioner construction)
function Base.getindex(A::ACAOperator, i::Int, j::Int)
    return _efie_entry(A.cache, i, j)
end

function Base.getindex(A::ACAAdjointOperator, i::Int, j::Int)
    return conj(_efie_entry(A.op.cache, j, i))
end

LinearAlgebra.adjoint(A::ACAOperator) = ACAAdjointOperator{typeof(A)}(A)

# ─── ACA low-rank approximation ──────────────────────────────────

"""
    aca_lowrank(cache, row_indices, col_indices; tol=1e-6, max_rank=50)

Compute a low-rank approximation of the sub-block Z[row_indices, col_indices]
using partially-pivoted Adaptive Cross Approximation.

Returns `(U, V)` where the approximation is `U * V'`, with
`U` of size `(m, k)` and `V` of size `(n, k)`.
"""
function aca_lowrank(cache::EFIEApplyCache,
                     row_indices::AbstractVector{Int},
                     col_indices::AbstractVector{Int};
                     tol::Float64=1e-6,
                     max_rank::Int=50)
    m = length(row_indices)
    n = length(col_indices)
    max_rank = min(max_rank, m, n)

    U_cols = Vector{Vector{ComplexF64}}()
    V_cols = Vector{Vector{ComplexF64}}()

    used_rows = falses(m)
    used_cols = falses(n)
    frob_sq = 0.0  # running Frobenius norm squared of approximation

    # Start with the first row
    pivot_row = 1
    converged = false

    for k in 1:max_rank
        # Compute residual row at pivot_row
        ri = row_indices[pivot_row]
        row_vec = Vector{ComplexF64}(undef, n)
        @inbounds for jj in 1:n
            row_vec[jj] = _efie_entry(cache, ri, col_indices[jj])
        end

        # Subtract contributions from previous rank-1 terms
        for prev in eachindex(U_cols)
            u_val = U_cols[prev][pivot_row]
            @inbounds for jj in 1:n
                row_vec[jj] -= u_val * conj(V_cols[prev][jj])
            end
        end

        # Find column pivot: max magnitude in residual row
        pivot_col = 0
        best_val = 0.0
        @inbounds for jj in 1:n
            av = abs(row_vec[jj])
            if av > best_val && !used_cols[jj]
                best_val = av
                pivot_col = jj
            end
        end

        # If no valid pivot, try any column
        if pivot_col == 0 || best_val < 1e-30
            for jj in 1:n
                if !used_cols[jj]
                    pivot_col = jj
                    break
                end
            end
            pivot_col == 0 && break
            best_val = abs(row_vec[pivot_col])
            best_val < 1e-30 && break
        end

        pivot_val = row_vec[pivot_col]

        # Compute residual column at pivot_col
        cj = col_indices[pivot_col]
        col_vec = Vector{ComplexF64}(undef, m)
        @inbounds for ii in 1:m
            col_vec[ii] = _efie_entry(cache, row_indices[ii], cj)
        end

        for prev in eachindex(U_cols)
            v_val = V_cols[prev][pivot_col]
            @inbounds for ii in 1:m
                col_vec[ii] -= U_cols[prev][ii] * conj(v_val)
            end
        end

        # Form rank-1 update: u = col / pivot_val, v = row (conjugated for V')
        u_k = col_vec / pivot_val
        v_k = conj.(row_vec)  # store conjugate so block = U * V'

        # Robustness: a non-finite pivot/update (e.g. NaN/Inf from a degenerate
        # entry or near-zero pivot) would corrupt the low-rank factors. Stop and
        # return the rank accumulated so far rather than propagating NaN/Inf.
        (all(isfinite, u_k) && all(isfinite, v_k)) || break

        push!(U_cols, u_k)
        push!(V_cols, v_k)
        used_rows[pivot_row] = true
        used_cols[pivot_col] = true

        # Update Frobenius norm estimate
        # ||A_k||_F^2 = ||A_{k-1}||_F^2 + 2 Re{Σ_{j<k} (u_j'u_k)(v_k'v_j)} + ||u_k||^2 ||v_k||^2
        norm_u = sqrt(real(dot(u_k, u_k)))
        norm_v = sqrt(real(dot(v_k, v_k)))
        cross_term = 0.0
        for prev in 1:(length(U_cols)-1)
            cross_term += 2.0 * real(dot(U_cols[prev], u_k) * conj(dot(V_cols[prev], v_k)))
        end
        frob_sq += norm_u^2 * norm_v^2 + cross_term
        frob_sq = max(frob_sq, 1e-30)  # guard

        # Convergence check
        if norm_u * norm_v < tol * sqrt(frob_sq)
            converged = true
            break
        end

        # Choose next pivot row: row with max |u_k| among unused rows
        next_row = 0
        best_u = 0.0
        @inbounds for ii in 1:m
            if !used_rows[ii] && abs(u_k[ii]) > best_u
                best_u = abs(u_k[ii])
                next_row = ii
            end
        end

        if next_row == 0
            # All rows used
            break
        end
        pivot_row = next_row
    end

    if !converged && length(U_cols) == max_rank
        @warn "ACA reached max_rank=$max_rank without meeting tolerance tol=$tol; " *
              "the low-rank block may be inaccurate — increase max_rank or loosen tol." maxlog=1
    end

    rank = length(U_cols)
    if rank == 0
        return zeros(ComplexF64, m, 0), zeros(ComplexF64, n, 0)
    end

    U = Matrix{ComplexF64}(undef, m, rank)
    V = Matrix{ComplexF64}(undef, n, rank)
    for kk in 1:rank
        U[:, kk] = U_cols[kk]
        V[:, kk] = V_cols[kk]
    end

    return U, V
end

# ─── Batched dense block fill ────────────────────────────────────
#
# Precomputes Green's function for unique triangle pairs in a block,
# avoiding redundant evaluations when multiple RWG functions share triangles.

function _fill_dense_block_batched!(data::Matrix{ComplexF64}, cache::EFIEApplyCache,
                                     row_indices::Vector{Int}, col_indices::Vector{Int})
    mr, nc = length(row_indices), length(col_indices)
    Nq = cache.Nq

    # 1. Collect unique triangles referenced by row/col RWG indices
    row_tris = Set{Int}()
    col_tris = Set{Int}()
    @inbounds for m_idx in row_indices
        push!(row_tris, cache.tri_ids[1, m_idx])
        push!(row_tris, cache.tri_ids[2, m_idx])
    end
    @inbounds for n_idx in col_indices
        push!(col_tris, cache.tri_ids[1, n_idx])
        push!(col_tris, cache.tri_ids[2, n_idx])
    end

    # 2. Precompute Green's function for all (row_tri, col_tri) pairs
    #    Skip self-cell pairs (tm == tn) — handled by singularity extraction
    green_cache = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    for tm in row_tris, tn in col_tris
        tm == tn && continue
        G_mat = Matrix{ComplexF64}(undef, Nq, Nq)
        @inbounds for qm in 1:Nq, qn in 1:Nq
            G_mat[qm, qn] = greens(cache.quad_pts[tm][qm], cache.quad_pts[tn][qn], cache.k)
        end
        green_cache[(tm, tn)] = G_mat
    end

    # 3. Fill block entries using cached Green's values
    @inbounds for jj in 1:nc
        n_idx = col_indices[jj]
        for ii in 1:mr
            m_idx = row_indices[ii]
            val = zero(ComplexF64)

            for itm in 1:2
                tm = cache.tri_ids[itm, m_idx]
                Am = cache.areas[tm]
                dvm = cache.div_vals[itm, m_idx]
                fm_vals = _rwg_vals(cache, m_idx, itm)

                for itn in 1:2
                    tn = cache.tri_ids[itn, n_idx]
                    An = cache.areas[tn]
                    dvn = cache.div_vals[itn, n_idx]
                    fn_vals = _rwg_vals(cache, n_idx, itn)

                    if tm == tn
                        val += self_cell_contribution(
                            cache.mesh, cache.rwg, m_idx, n_idx, tm,
                            cache.quad_pts[tm], fm_vals, fn_vals,
                            dvm, dvn, Am, cache.wq, cache.k,
                            cache.wq_hi, cache.quad_pts_hi[tm])
                    elseif _is_adjacent(cache, tm, tn)
                        val += adjacent_cell_contribution(
                            cache.mesh, cache.rwg, m_idx, n_idx, tm, tn,
                            cache.quad_pts[tm], cache.quad_pts[tn],
                            fm_vals, fn_vals,
                            dvm, dvn, Am, An,
                            cache.wq, cache.k,
                            cache.wq_hi, cache.quad_pts_hi[tm], cache.quad_pts_hi[tn])
                    else
                        G_mat = green_cache[(tm, tn)]
                        for qm in 1:Nq
                            fm = fm_vals[qm]
                            for qn in 1:Nq
                                G = G_mat[qm, qn]
                                vec_part = dot(fm, fn_vals[qn]) * G
                                scl_part = conj(dvm) * dvn * G / (cache.k^2)
                                weight = cache.wq[qm] * cache.wq[qn] * (2 * Am) * (2 * An)
                                val += (vec_part - scl_part) * weight
                            end
                        end
                    end
                end
            end
            data[ii, jj] = -1im * cache.omega_mu0 * val
        end
    end
end

# ─── Two-phase parallel block assembly ───────────────────────────
#
# Phase 1 (serial): Enumerate all block tasks into a flat array.
# Phase 2 (parallel): Process blocks with Threads.@threads.

struct BlockTask
    row_node::Int
    col_node::Int
    kind::Symbol   # :dense or :lowrank
end

function _enum_block_tasks!(tasks::Vector{BlockTask}, tree::ClusterTree,
                             row_node::Int, col_node::Int; eta::Float64)
    # If admissible → low-rank task
    if is_admissible(tree, row_node, col_node; eta=eta)
        push!(tasks, BlockTask(row_node, col_node, :lowrank))
        return
    end

    # If both leaves and inadmissible → dense task
    if is_leaf(tree, row_node) && is_leaf(tree, col_node)
        push!(tasks, BlockTask(row_node, col_node, :dense))
        return
    end

    # Recurse
    rn = tree.nodes[row_node]
    cn = tree.nodes[col_node]
    if is_leaf(tree, row_node)
        _enum_block_tasks!(tasks, tree, row_node, cn.left; eta=eta)
        _enum_block_tasks!(tasks, tree, row_node, cn.right; eta=eta)
    elseif is_leaf(tree, col_node)
        _enum_block_tasks!(tasks, tree, rn.left, col_node; eta=eta)
        _enum_block_tasks!(tasks, tree, rn.right, col_node; eta=eta)
    else
        _enum_block_tasks!(tasks, tree, rn.left, cn.left; eta=eta)
        _enum_block_tasks!(tasks, tree, rn.left, cn.right; eta=eta)
        _enum_block_tasks!(tasks, tree, rn.right, cn.left; eta=eta)
        _enum_block_tasks!(tasks, tree, rn.right, cn.right; eta=eta)
    end
end

"""
    build_aca_operator(mesh, rwg, k; kwargs...)

Build an H-matrix EFIE operator using Adaptive Cross Approximation.

Near-field blocks (inadmissible) are stored dense; far-field blocks
(admissible) are compressed to low-rank form via partially-pivoted ACA.

Dense blocks use triangle-pair batched Green's function evaluation for
~8× fewer Green's calls. Block construction is parallelized with @threads.

# Arguments
- `mesh::TriMesh`: triangle mesh
- `rwg::RWGData`: RWG basis function data
- `k`: wavenumber
- `leaf_size=64`: cluster tree leaf size
- `eta=1.5`: admissibility parameter
- `aca_tol=1e-6`: ACA convergence tolerance
- `max_rank=50`: maximum rank for low-rank blocks
- `quad_order=3`: quadrature order for EFIE entries
- `eta0=376.730313668`: free-space impedance
"""
function build_aca_operator(mesh::TriMesh, rwg::RWGData, k;
                            leaf_size::Int=64,
                            eta::Float64=1.5,
                            aca_tol::Float64=1e-6,
                            max_rank::Int=50,
                            quad_order::Int=3,
                            eta0::Float64=376.730313668,
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

    N = rwg.nedges
    cache = _build_efie_cache(mesh, rwg, k; quad_order=quad_order, eta0=eta0)

    centers = rwg_centers(mesh, rwg)
    tree = build_cluster_tree(centers; leaf_size=leaf_size)

    # Phase 1: enumerate all block tasks (serial, fast)
    tasks = BlockTask[]
    _enum_block_tasks!(tasks, tree, 1, 1; eta=eta)

    # Phase 2: process blocks in parallel
    results = Vector{Union{DenseBlock, LowRankBlock}}(undef, length(tasks))
    Threads.@threads for i in eachindex(tasks)
        task = tasks[i]
        rn = tree.nodes[task.row_node]
        cn = tree.nodes[task.col_node]
        row_indices = [tree.perm[k] for k in rn.indices]
        col_indices = [tree.perm[k] for k in cn.indices]

        if task.kind == :dense
            data = Matrix{ComplexF64}(undef, length(row_indices), length(col_indices))
            _fill_dense_block_batched!(data, cache, row_indices, col_indices)
            results[i] = DenseBlock(rn.indices, cn.indices, data)
        else
            U, V = aca_lowrank(cache, row_indices, col_indices;
                               tol=aca_tol, max_rank=max_rank)
            results[i] = LowRankBlock(rn.indices, cn.indices, U, V)
        end
    end

    # Separate into dense and low-rank block vectors
    dense_blocks = DenseBlock[]
    lowrank_blocks = LowRankBlock[]
    for r in results
        if r isa DenseBlock
            push!(dense_blocks, r)
        else
            push!(lowrank_blocks, r)
        end
    end

    max_rank = isempty(lowrank_blocks) ? 0 : maximum(size(blk.U, 2) for blk in lowrank_blocks)
    ws = ACAWorkspace(Vector{ComplexF64}(undef, N),
                       zeros(ComplexF64, N),
                       Vector{ComplexF64}(undef, max(max_rank, 1)))
    return ACAOperator{typeof(cache)}(cache, tree, dense_blocks, lowrank_blocks, N, ws)
end

# ─── Matvec ───────────────────────────────────────────────────────

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64}, A::ACAOperator, x::AbstractVector)
    N = A.N
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))

    ws = A.workspace
    x_perm = ws.x_perm
    y_perm = ws.y_perm

    # Permute x to tree order
    @inbounds for k in 1:N
        x_perm[k] = x[A.tree.perm[k]]
    end
    fill!(y_perm, zero(ComplexF64))

    # Dense blocks — BLAS gemv
    for blk in A.dense_blocks
        rows = blk.row_range
        cols = blk.col_range
        mul!(@view(y_perm[rows]), blk.data, @view(x_perm[cols]), one(ComplexF64), one(ComplexF64))
    end

    # Low-rank blocks: y += U * (V' * x)
    for blk in A.lowrank_blocks
        k_rank = size(blk.U, 2)
        k_rank == 0 && continue
        rows = blk.row_range
        cols = blk.col_range
        tmp = @view(ws.tmp[1:k_rank])
        mul!(tmp, blk.V', @view(x_perm[cols]))
        mul!(@view(y_perm[rows]), blk.U, tmp, one(ComplexF64), one(ComplexF64))
    end

    # Un-permute y back to original order
    @inbounds for k in 1:N
        y[A.tree.perm[k]] = y_perm[k]
    end

    return y
end

function Base.:*(A::ACAOperator, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, Vector{ComplexF64}(x))
    return y
end

# ─── Adjoint matvec ───────────────────────────────────────────────

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64}, A::ACAAdjointOperator, x::AbstractVector)
    N = A.op.N
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))

    tree = A.op.tree
    ws = A.op.workspace
    x_perm = ws.x_perm
    y_perm = ws.y_perm

    # Permute x to tree order
    @inbounds for k in 1:N
        x_perm[k] = x[tree.perm[k]]
    end
    fill!(y_perm, zero(ComplexF64))

    # Dense blocks: adjoint means transpose-conjugate
    for blk in A.op.dense_blocks
        rows = blk.row_range
        cols = blk.col_range
        mul!(@view(y_perm[cols]), blk.data', @view(x_perm[rows]), one(ComplexF64), one(ComplexF64))
    end

    # Low-rank blocks: adjoint of U * V' is V * U'
    for blk in A.op.lowrank_blocks
        k_rank = size(blk.U, 2)
        k_rank == 0 && continue
        rows = blk.row_range
        cols = blk.col_range
        tmp = @view(ws.tmp[1:k_rank])
        mul!(tmp, blk.U', @view(x_perm[rows]))
        mul!(@view(y_perm[cols]), blk.V, tmp, one(ComplexF64), one(ComplexF64))
    end

    # Un-permute
    @inbounds for k in 1:N
        y[tree.perm[k]] = y_perm[k]
    end

    return y
end

function Base.:*(A::ACAAdjointOperator, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, Vector{ComplexF64}(x))
    return y
end
