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

const _DEFAULT_MAX_NEARFIELD_TRIPLET_BYTES = 512 * 1024 * 1024
const _DEFAULT_MAX_NEARFIELD_GREEN_WORKSPACE_BYTES = 256 * 1024 * 1024
const _DEFAULT_MAX_NEARFIELD_GREEN_CACHE_ENTRIES = 250_000
const _DEFAULT_MAX_BLOCK_DIAG_STORAGE_BYTES = 512 * 1024 * 1024
const _DEFAULT_MAX_BLOCK_DIAG_EXACT_WORK = 20_000_000
const _NEARFIELD_TRIPLET_ENTRY_BYTES =
    2 * sizeof(Int) + sizeof(ComplexF64)

mutable struct _NearFieldTripletBudget
    count::Int
    entry_limit::Int
    byte_limit::Int
end

function _nearfield_triplet_budget(max_triplet_bytes::Integer)
    byte_limit = _validated_resource_limit(
        "max_triplet_bytes", max_triplet_bytes)
    return _NearFieldTripletBudget(
        0, div(byte_limit, _NEARFIELD_TRIPLET_ENTRY_BYTES), byte_limit)
end

@inline function _register_nearfield_triplet!(
        budget::_NearFieldTripletBudget)
    budget.count < budget.entry_limit ||
        throw(ArgumentError(
            "near-field triplet payload exceeds " *
            "max_triplet_bytes=$(budget.byte_limit) after " *
            "$(budget.count) retained entries"))
    budget.count += 1
    return nothing
end

function _preflight_nearfield_triplets!(
        budget::_NearFieldTripletBudget,
        required_entries::Int)
    required_entries <= budget.entry_limit ||
        throw(ArgumentError(
            "near-field triplet payload requires at least " *
            "$required_entries entries, " *
            "exceeding max_triplet_bytes=$(budget.byte_limit)"))
    return required_entries
end

@inline function _checked_nearfield_product(a::Int, b::Int,
                                             label::AbstractString)
    try
        return Base.Checked.checked_mul(a, b)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("$label exceeds the supported Int range"))
    end
end

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
One reusable, lock-protected work vector holds the largest block so repeated
forward and adjoint applications do not allocate per-block RHS vectors.
"""
struct BlockDiagPrecondData <: AbstractPreconditionerData
    lu_blocks::Vector{LinearAlgebra.LU{ComplexF64, Matrix{ComplexF64}, Vector{Int64}}}
    box_bf_indices::Vector{Vector{Int}}   # original BF indices per leaf box
    N::Int
    nnz_ratio::Float64
    work::Vector{ComplexF64}
    work_lock::ReentrantLock
end

function BlockDiagPrecondData(
        lu_blocks::Vector{LinearAlgebra.LU{
            ComplexF64, Matrix{ComplexF64}, Vector{Int64}}},
        box_bf_indices::Vector{Vector{Int}},
        N::Int,
        nnz_ratio::Float64)
    max_block_size = isempty(box_bf_indices) ?
        0 : maximum(length, box_bf_indices)
    return BlockDiagPrecondData(
        lu_blocks, box_bf_indices, N, nnz_ratio,
        zeros(ComplexF64, max_block_size), ReentrantLock())
end

"""
    PermutedPrecondData <: AbstractPreconditionerData

Wraps an inner preconditioner (ILU or LU) that operates in a permuted ordering.
On apply: permute input → apply inner preconditioner → unpermute output.
Used for MLFMA where reordering Z_near to block-banded form before ILU
dramatically speeds up factorization and improves quality.
A reusable, lock-protected vector performs both permutations without
allocating indexed copies on every application.
"""
struct PermutedPrecondData{T<:AbstractPreconditionerData} <: AbstractPreconditionerData
    inner::T
    perm::Vector{Int}    # perm[new] = old (original → permuted)
    iperm::Vector{Int}   # iperm[old] = new (permuted → original)
    N::Int
    nnz_ratio::Float64
    work::Vector{ComplexF64}
    work_lock::ReentrantLock
end

function PermutedPrecondData(inner::T,
                             perm::Vector{Int},
                             iperm::Vector{Int},
                             N::Int,
                             nnz_ratio::Float64) where {
        T<:AbstractPreconditionerData}
    return PermutedPrecondData(
        inner, perm, iperm, N, nnz_ratio,
        zeros(ComplexF64, N), ReentrantLock())
end

function _validate_preconditioner_factorization(factorization::Symbol,
                                                ilu_tau::Float64;
                                                allow_diag::Bool=true)
    allowed = allow_diag ? (:lu, :ilu, :diag) : (:lu, :ilu)
    factorization in allowed ||
        throw(ArgumentError(
            "invalid factorization=$factorization; expected one of $allowed"
        ))
    if factorization == :ilu
        isfinite(ilu_tau) && ilu_tau >= 0.0 ||
            throw(ArgumentError(
                "ilu_tau must be finite and nonnegative, got $ilu_tau"
            ))
    end
    return nothing
end

function _validate_nearfield_build_controls(cutoff::Float64,
                                            neighbor_search::Symbol,
                                            factorization::Symbol,
                                            ilu_tau::Float64)
    !isnan(cutoff) && cutoff >= 0.0 ||
        throw(ArgumentError(
            "near-field cutoff must be nonnegative and not NaN, got $cutoff"
        ))
    neighbor_search in (:spatial, :bruteforce) ||
        throw(ArgumentError(
            "invalid neighbor_search=$neighbor_search; expected :spatial or :bruteforce"
        ))
    _validate_preconditioner_factorization(factorization, ilu_tau)
    return nothing
end

function _validate_nearfield_triplet_values(values::Vector{ComplexF64})
    all(isfinite, values) ||
        throw(ArgumentError(
            "near-field matrix entries must contain only finite values"))
    return nothing
end

function _validate_sparse_preconditioner_matrix(Z::SparseMatrixCSC,
                                                label::AbstractString)
    size(Z, 1) == size(Z, 2) ||
        throw(DimensionMismatch("$label must be square, got size $(size(Z))"))
    size(Z, 1) >= 1 ||
        throw(ArgumentError("$label must contain at least one row and column"))
    all(isfinite, nonzeros(Z)) ||
        throw(ArgumentError("$label must contain only finite stored values"))
    return size(Z, 1)
end

"""
    rwg_centers(mesh, rwg)

Compute the center point of each RWG basis function, defined as the average
of the centroids of its two supporting triangles.

Returns a Vector{Vec3} of length N.
"""
function rwg_centers(mesh::TriMesh, rwg::RWGData)
    _validate_mesh_rwg_pair(mesh, rwg)
    N = rwg.nedges
    centers = Vector{Vec3}(undef, N)
    for n in 1:N
        c_plus  = triangle_center(mesh, rwg.tplus[n])
        c_minus = triangle_center(mesh, rwg.tminus[n])
        centers[n] = _safe_edge_midpoint(c_plus, c_minus)
    end
    return centers
end

const _MAX_SPATIAL_HASH_AXIS_INDEX = Float64(typemax(Int) ÷ 4)

function _spatial_hash_parameters(centers::Vector{Vec3}, cutoff::Float64)
    first_center = first(centers)
    all(isfinite, first_center) ||
        throw(ArgumentError(
            "RWG center 1 must be finite, got $first_center"))

    xmin = xmax = first_center[1]
    ymin = ymax = first_center[2]
    zmin = zmax = first_center[3]
    @inbounds for i in 2:length(centers)
        center = centers[i]
        all(isfinite, center) ||
            throw(ArgumentError(
                "RWG center $i must be finite, got $center"))
        xmin = min(xmin, center[1]); xmax = max(xmax, center[1])
        ymin = min(ymin, center[2]); ymax = max(ymax, center[2])
        zmin = min(zmin, center[3]); zmax = max(zmax, center[3])
    end

    xspan = xmax - xmin
    yspan = ymax - ymin
    zspan = zmax - zmin
    all(isfinite, (xspan, yspan, zspan)) ||
        throw(ArgumentError(
            "RWG-center extent is outside the supported Float64 range"))
    max_span = max(xspan, yspan, zspan)

    # A hash cell may be wider than the physical cutoff: any pair within the
    # cutoff still lies in the same or an adjacent cell. Widen only when needed
    # to keep origin-relative cell indices safely representable as Int.
    cell_size = max(cutoff, max_span / _MAX_SPATIAL_HASH_AXIS_INDEX)
    isfinite(cell_size) && cell_size > 0.0 ||
        throw(ArgumentError(
            "spatial-hash cell size is not representable for cutoff=$cutoff"))
    return Vec3(xmin, ymin, zmin), cell_size
end

@inline function _cell_key(r::Vec3, origin::Vec3, cell_size::Float64)
    return (
        floor(Int, (r[1] - origin[1]) / cell_size),
        floor(Int, (r[2] - origin[2]) / cell_size),
        floor(Int, (r[3] - origin[3]) / cell_size),
    )
end

@noinline function _within_nearfield_cutoff_exact(
        a::Vec3, b::Vec3, cutoff::Float64)
    dx = Rational{BigInt}(a[1]) - Rational{BigInt}(b[1])
    dy = Rational{BigInt}(a[2]) - Rational{BigInt}(b[2])
    dz = Rational{BigInt}(a[3]) - Rational{BigInt}(b[3])
    cutoff_exact = Rational{BigInt}(cutoff)
    return dx * dx + dy * dy + dz * dz <=
           cutoff_exact * cutoff_exact
end

@inline function _within_nearfield_cutoff(a::Vec3, b::Vec3, cutoff::Float64)
    isfinite(cutoff) || return true
    distance = hypot(a[1] - b[1], a[2] - b[2], a[3] - b[3])
    distance == cutoff &&
        return _within_nearfield_cutoff_exact(a, b, cutoff)
    return distance <= cutoff
end

@inline function _foreach_nearfield_pair_bruteforce(
        centers::Vector{Vec3}, cutoff::Float64, visit)
    N = length(centers)
    if cutoff <= 0.0
        @inbounds for m in 1:N
            visit(m, m)
        end
    elseif !isfinite(cutoff)
        @inbounds for m in 1:N, n in 1:N
            visit(m, n)
        end
    else
        @inbounds for m in 1:N
            cm = centers[m]
            for n in 1:N
                if m == n ||
                   _within_nearfield_cutoff(cm, centers[n], cutoff)
                    visit(m, n)
                end
            end
        end
    end
    return nothing
end

@inline function _foreach_nearfield_pair_spatial(
        centers::Vector{Vec3}, cutoff::Float64,
        origin::Vec3, cell_size::Float64,
        buckets::Dict{NTuple{3,Int},Vector{Int}}, visit)
    @inbounds for m in eachindex(centers)
        cm = centers[m]
        key = _cell_key(cm, origin, cell_size)
        for dz in -1:1, dy in -1:1, dx in -1:1
            key_n = (key[1] + dx, key[2] + dy, key[3] + dz)
            n_list = get(buckets, key_n, nothing)
            n_list === nothing && continue
            for n in n_list
                if m == n ||
                   _within_nearfield_cutoff(cm, centers[n], cutoff)
                    visit(m, n)
                end
            end
        end
    end
    return nothing
end

function _count_nearfield_pairs!(
        foreach_pair, budget::_NearFieldTripletBudget)
    foreach_pair() do _, _
        _register_nearfield_triplet!(budget)
    end
    return budget.count
end

function _materialize_nearfield_triplets(
        count::Int, foreach_pair, getvalue)
    I_idx = Vector{Int}(undef, count)
    J_idx = Vector{Int}(undef, count)
    V_val = Vector{ComplexF64}(undef, count)
    position = Ref(0)
    foreach_pair() do m, n
        position[] += 1
        I_idx[position[]] = m
        J_idx[position[]] = n
        V_val[position[]] = ComplexF64(getvalue(m, n))
    end
    position[] == count ||
        error("near-field neighbor enumeration changed between count and fill passes")
    return I_idx, J_idx, V_val
end

function _nearfield_triplets_bruteforce(
        centers::Vector{Vec3}, cutoff::Float64, getvalue;
        max_triplet_bytes::Integer=_DEFAULT_MAX_NEARFIELD_TRIPLET_BYTES,
        _budget::Union{Nothing,_NearFieldTripletBudget}=nothing)
    N = length(centers)
    budget = _budget === nothing ?
        _nearfield_triplet_budget(max_triplet_bytes) : _budget
    N == 0 && return Int[], Int[], ComplexF64[]

    if cutoff <= 0
        _preflight_nearfield_triplets!(budget, N)
        return _materialize_nearfield_triplets(
            N,
            visit -> _foreach_nearfield_pair_bruteforce(
                centers, cutoff, visit),
            getvalue)
    elseif !isfinite(cutoff)
        required_entries = _checked_nearfield_product(
            N, N, "near-field all-pairs entry count")
        _preflight_nearfield_triplets!(budget, required_entries)
        return _materialize_nearfield_triplets(
            required_entries,
            visit -> _foreach_nearfield_pair_bruteforce(
                centers, cutoff, visit),
            getvalue)
    end

    foreach_pair = visit -> _foreach_nearfield_pair_bruteforce(
        centers, cutoff, visit)
    count = _count_nearfield_pairs!(foreach_pair, budget)
    return _materialize_nearfield_triplets(count, foreach_pair, getvalue)
end

function _nearfield_triplets_spatial(
        centers::Vector{Vec3}, cutoff::Float64, getvalue;
        max_triplet_bytes::Integer=_DEFAULT_MAX_NEARFIELD_TRIPLET_BYTES,
        _budget::Union{Nothing,_NearFieldTripletBudget}=nothing)
    N = length(centers)
    budget = _budget === nothing ?
        _nearfield_triplet_budget(max_triplet_bytes) : _budget
    N == 0 && return Int[], Int[], ComplexF64[]

    if cutoff <= 0 || !isfinite(cutoff)
        return _nearfield_triplets_bruteforce(
            centers, cutoff, getvalue; _budget=budget)
    end

    _preflight_nearfield_triplets!(budget, N)

    origin, cell_size = _spatial_hash_parameters(centers, cutoff)
    buckets = Dict{NTuple{3,Int}, Vector{Int}}()
    @inbounds for i in 1:N
        key = _cell_key(centers[i], origin, cell_size)
        push!(get!(() -> Int[], buckets, key), i)
    end

    foreach_pair = visit -> _foreach_nearfield_pair_spatial(
        centers, cutoff, origin, cell_size, buckets, visit)
    count = _count_nearfield_pairs!(foreach_pair, budget)
    return _materialize_nearfield_triplets(count, foreach_pair, getvalue)
end

"""
Single-pass near-field triplet assembly with lazy Green's caching for EFIE.

Iterates the spatial-hash neighbor structure and computes EFIE entries on
the fly, but caches Green's quadrature matrices G[qm,qn] per unique
(tm, tn) triangle pair so that shared pairs are evaluated only once.
Self-cell and adjacent-cell terms bypass the cache entirely.
"""
function _nearfield_triplets_batched(
        cache::EFIEApplyCache, centers::Vector{Vec3}, cutoff::Float64;
        max_triplet_bytes::Integer=_DEFAULT_MAX_NEARFIELD_TRIPLET_BYTES,
        max_green_cache_bytes::Integer=
            _DEFAULT_MAX_NEARFIELD_GREEN_WORKSPACE_BYTES,
        max_green_cache_entries::Integer=
            _DEFAULT_MAX_NEARFIELD_GREEN_CACHE_ENTRIES)
    N = length(centers)
    N == 0 && return Int[], Int[], ComplexF64[]
    budget = _nearfield_triplet_budget(max_triplet_bytes)

    if cutoff <= 0 || !isfinite(cutoff)
        return _nearfield_triplets_spatial(
            centers, cutoff, (m, n) -> _efie_entry(cache, m, n);
            _budget=budget)
    end


    _preflight_nearfield_triplets!(budget, N)

    # Spatial hash
    origin, cell_size = _spatial_hash_parameters(centers, cutoff)
    buckets = Dict{NTuple{3,Int}, Vector{Int}}()
    @inbounds for i in 1:N
        key = _cell_key(centers[i], origin, cell_size)
        push!(get!(() -> Int[], buckets, key), i)
    end

    foreach_pair = visit -> _foreach_nearfield_pair_spatial(
        centers, cutoff, origin, cell_size, buckets, visit)
    pair_count = _count_nearfield_pairs!(foreach_pair, budget)
    I_idx = Vector{Int}(undef, pair_count)
    J_idx = Vector{Int}(undef, pair_count)
    V_val = Vector{ComplexF64}(undef, pair_count)
    position = 0

    Nq     = cache.Nq
    inv_k2 = cache.inv_k2

    green_workspace_limit = _validated_resource_limit(
        "max_green_cache_bytes", max_green_cache_bytes)
    green_entry_limit = _validated_resource_limit(
        "max_green_cache_entries", max_green_cache_entries)
    green_matrix_bytes = _checked_array_payload_bytes(
        ComplexF64, Nq, Nq; label="near-field Green matrix")
    matrix_slots = div(green_workspace_limit, green_matrix_bytes)

    # Reserve one matrix slot for a reusable scratch matrix after the cache is
    # full. This keeps uncached triangle pairs bounded instead of growing the
    # dictionary without limit.
    max_cached_greens = min(green_entry_limit, max(0, matrix_slots - 1))
    green_cache = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    green_scratch = Ref{Union{Nothing,Matrix{ComplexF64}}}(nothing)

    @inline function _fill_greens!(G_mat::Matrix{ComplexF64},
                                    tm::Int, tn::Int)
        @inbounds for qm in 1:Nq, qn in 1:Nq
            G_mat[qm, qn] = _greens_unchecked(
                cache.quad_pts[tm][qm], cache.quad_pts[tn][qn], cache.k)
        end
        return G_mat
    end

    @inline function _get_greens(tm::Int, tn::Int)
        val = get(green_cache, (tm, tn), nothing)
        val !== nothing && return val
        matrix_slots >= 1 ||
            throw(ArgumentError(
                "one near-field Green matrix requires $green_matrix_bytes " *
                "raw bytes, exceeding max_green_cache_bytes=" *
                "$green_workspace_limit"))
        if length(green_cache) < max_cached_greens
            G_mat = _fill_greens!(
                Matrix{ComplexF64}(undef, Nq, Nq), tm, tn)
            green_cache[(tm, tn)] = G_mat
            return G_mat
        end
        if green_scratch[] === nothing
            green_scratch[] = Matrix{ComplexF64}(undef, Nq, Nq)
        end
        return _fill_greens!(something(green_scratch[]), tm, tn)
    end

    @inbounds for m in 1:N
        cm = centers[m]
        key = _cell_key(cm, origin, cell_size)
        for dz in -1:1, dy in -1:1, dx in -1:1
            key_n = (key[1] + dx, key[2] + dy, key[3] + dz)
            n_list = get(buckets, key_n, nothing)
            n_list === nothing && continue
            for n in n_list
                is_near = (m == n) ||
                    _within_nearfield_cutoff(cm, centers[n], cutoff)
                is_near || continue
                position += 1

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

                I_idx[position] = m
                J_idx[position] = n
                V_val[position] = -1im * cache.omega_mu0 * val
            end
        end
    end

    position == pair_count ||
        error("near-field neighbor enumeration changed between count and fill passes")
    return I_idx, J_idx, V_val
end

function _build_diagonal_preconditioner_data(getvalue, N::Int, cutoff::Float64)
    N >= 1 ||
        throw(ArgumentError(
            "cannot build a preconditioner for a zero-dimensional system"))
    diag_entries = Vector{ComplexF64}(undef, N)
    maxabs = 0.0
    @inbounds for i in 1:N
        zii = ComplexF64(getvalue(i, i))
        isfinite(zii) ||
            throw(ArgumentError(
                "near-field diagonal entry $i must be finite, got $zii"))
        diag_entries[i] = zii
        ai = abs(zii)
        isfinite(ai) ||
            throw(OverflowError(
                "magnitude of near-field diagonal entry $i is not representable"))
        if ai > maxabs
            maxabs = ai
        end
    end

    maxabs > 0.0 ||
        throw(ArgumentError(
            "cannot build a diagonal preconditioner from an all-zero diagonal"))
    # Regularize relative to the matrix's own diagonal scale.  An absolute
    # unit floor destroys Jacobi scaling for globally small but otherwise
    # well-conditioned systems.
    floor_abs = 1e-10 * maxabs
    iszero(floor_abs) && (floor_abs = nextfloat(0.0))
    @inbounds for i in 1:N
        if abs(diag_entries[i]) < floor_abs
            diag_entries[i] = floor_abs + 0im
        end
    end

    dinv = similar(diag_entries)
    @inbounds for i in 1:N
        dinv[i] = inv(diag_entries[i])
        isfinite(dinv[i]) ||
            throw(OverflowError(
                "inverse of near-field diagonal entry $i is non-finite"))
    end

    nnz_ratio = inv(Float64(N))
    return DiagonalPreconditionerData(dinv, cutoff, nnz_ratio)
end

function _build_nearfield_preconditioner_from_entries(mesh::TriMesh, rwg::RWGData, cutoff::Float64,
                                                       getvalue;
                                                       neighbor_search::Symbol=:spatial,
                                                       factorization::Symbol=:lu,
                                                       ilu_tau::Float64=1e-3,
                                                       max_triplet_bytes::Integer=
                                                           _DEFAULT_MAX_NEARFIELD_TRIPLET_BYTES)
    _validate_mesh_rwg_pair(mesh, rwg)
    N = rwg.nedges
    _validate_nearfield_build_controls(
        cutoff, neighbor_search, factorization, ilu_tau)
    N >= 1 ||
        throw(ArgumentError(
            "cannot build a preconditioner for a zero-dimensional RWG system"))

    if factorization == :diag
        return _build_diagonal_preconditioner_data(getvalue, N, cutoff)
    end

    centers = rwg_centers(mesh, rwg)

    I_idx, J_idx, V_val = if neighbor_search == :spatial
        _nearfield_triplets_spatial(
            centers, cutoff, getvalue; max_triplet_bytes)
    elseif neighbor_search == :bruteforce
        _nearfield_triplets_bruteforce(
            centers, cutoff, getvalue; max_triplet_bytes)
    end

    _validate_nearfield_triplet_values(V_val)

    Z_nf = sparse(I_idx, J_idx, V_val, N, N)
    nnz_ratio = (Float64(nnz(Z_nf)) / Float64(N)) / Float64(N)

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
- `cutoff::Float64`: nonnegative distance cutoff (e.g., 0.5λ to 2λ);
  `Inf` retains every entry
- `neighbor_search`: `:spatial` (default) or `:bruteforce` reference mode

# Returns
A `NearFieldPreconditionerData` containing the factorized near-field matrix.
"""
function build_nearfield_preconditioner(Z::Matrix{<:Number}, mesh::TriMesh,
                                         rwg::RWGData, cutoff::Float64;
                                         neighbor_search::Symbol=:spatial,
                                         factorization::Symbol=:lu,
                                         ilu_tau::Float64=1e-3,
                                         max_triplet_bytes::Integer=
                                             _DEFAULT_MAX_NEARFIELD_TRIPLET_BYTES)
    N = rwg.nedges
    _validate_nearfield_build_controls(
        cutoff, neighbor_search, factorization, ilu_tau)
    size(Z, 1) == N && size(Z, 2) == N ||
        throw(DimensionMismatch("Z has size $(size(Z)), expected ($N, $N) for RWG basis"))
    all(isfinite, Z) ||
        throw(ArgumentError("Z must contain only finite values"))
    return _build_nearfield_preconditioner_from_entries(mesh, rwg, cutoff,
        (m, n) -> Z[m, n];
        neighbor_search=neighbor_search,
        factorization=factorization,
        ilu_tau=ilu_tau,
        max_triplet_bytes=max_triplet_bytes,
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
                                         ilu_tau::Float64=1e-3,
                                         max_triplet_bytes::Integer=
                                             _DEFAULT_MAX_NEARFIELD_TRIPLET_BYTES)
    N = rwg.nedges
    _validate_nearfield_build_controls(
        cutoff, neighbor_search, factorization, ilu_tau)
    size(A, 1) == N && size(A, 2) == N ||
        throw(DimensionMismatch("A has size $(size(A)), expected ($N, $N) for RWG basis"))
    return _build_nearfield_preconditioner_from_entries(mesh, rwg, cutoff,
        (m, n) -> A[m, n];
        neighbor_search=neighbor_search,
        factorization=factorization,
        ilu_tau=ilu_tau,
        max_triplet_bytes=max_triplet_bytes,
    )
end

"""
    build_nearfield_preconditioner(A::MLFMAOperator, mesh, rwg, cutoff)

Build a distance-truncated preconditioner from the near-field matrix already
stored by an MLFMA operator. Scalar indexing of `MLFMAOperator` represents the
complete operator and requires a full matvec, so this specialization preserves
the intended near-field construction cost.
"""
function build_nearfield_preconditioner(
        A::MLFMAOperator,
        mesh::TriMesh,
        rwg::RWGData,
        cutoff::Float64;
        neighbor_search::Symbol=:spatial,
        factorization::Symbol=:lu,
        ilu_tau::Float64=1e-3,
        max_triplet_bytes::Integer=
            _DEFAULT_MAX_NEARFIELD_TRIPLET_BYTES)
    N = rwg.nedges
    _validate_nearfield_build_controls(
        cutoff, neighbor_search, factorization, ilu_tau)
    size(A) == (N, N) ||
        throw(DimensionMismatch(
            "A has size $(size(A)), expected ($N, $N) for RWG basis"))
    return _build_nearfield_preconditioner_from_entries(
        mesh, rwg, cutoff, (m, n) -> A.Z_near[m, n];
        neighbor_search=neighbor_search,
        factorization=factorization,
        ilu_tau=ilu_tau,
        max_triplet_bytes=max_triplet_bytes,
    )
end

function build_nearfield_preconditioner(
        A::MLFMAAdjointOperator,
        mesh::TriMesh,
        rwg::RWGData,
        cutoff::Float64;
        neighbor_search::Symbol=:spatial,
        factorization::Symbol=:lu,
        ilu_tau::Float64=1e-3,
        max_triplet_bytes::Integer=
            _DEFAULT_MAX_NEARFIELD_TRIPLET_BYTES)
    N = rwg.nedges
    _validate_nearfield_build_controls(
        cutoff, neighbor_search, factorization, ilu_tau)
    size(A) == (N, N) ||
        throw(DimensionMismatch(
            "A has size $(size(A)), expected ($N, $N) for RWG basis"))
    return _build_nearfield_preconditioner_from_entries(
        mesh, rwg, cutoff, (m, n) -> conj(A.op.Z_near[n, m]);
        neighbor_search=neighbor_search,
        factorization=factorization,
        ilu_tau=ilu_tau,
        max_triplet_bytes=max_triplet_bytes,
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
                                         ilu_tau::Float64=1e-3,
                                         max_triplet_bytes::Integer=
                                             _DEFAULT_MAX_NEARFIELD_TRIPLET_BYTES,
                                         max_green_cache_bytes::Integer=
                                             _DEFAULT_MAX_NEARFIELD_GREEN_WORKSPACE_BYTES,
                                         max_green_cache_entries::Integer=
                                             _DEFAULT_MAX_NEARFIELD_GREEN_CACHE_ENTRIES)
    cache = A.cache
    N = cache.rwg.nedges
    _validate_nearfield_build_controls(
        cutoff, neighbor_search, factorization, ilu_tau)
    N >= 1 ||
        throw(ArgumentError(
            "cannot build a preconditioner for a zero-dimensional RWG system"))

    if factorization == :diag
        return _build_diagonal_preconditioner_data(
            (m, n) -> _efie_entry(cache, m, n), N, cutoff)
    end

    centers = rwg_centers(cache.mesh, cache.rwg)
    I_idx, J_idx, V_val = if neighbor_search == :bruteforce
        _nearfield_triplets_bruteforce(centers, cutoff,
                                        (m, n) -> _efie_entry(cache, m, n);
                                        max_triplet_bytes)
    else
        _nearfield_triplets_batched(
            cache, centers, cutoff;
            max_triplet_bytes,
            max_green_cache_bytes,
            max_green_cache_entries)
    end

    _validate_nearfield_triplet_values(V_val)

    Z_nf = sparse(I_idx, J_idx, V_val, N, N)
    nnz_ratio = (Float64(nnz(Z_nf)) / Float64(N)) / Float64(N)

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
                                         ilu_tau::Float64=1e-3,
                                         max_triplet_bytes::Integer=
                                             _DEFAULT_MAX_NEARFIELD_TRIPLET_BYTES,
                                         max_cache_bytes::Integer=
                                             _DEFAULT_MAX_EFIE_CACHE_BYTES,
                                         max_adjacency_pairs::Integer=
                                             _DEFAULT_MAX_EFIE_ADJACENCY_PAIRS,
                                         max_green_cache_bytes::Integer=
                                             _DEFAULT_MAX_NEARFIELD_GREEN_WORKSPACE_BYTES,
                                         max_green_cache_entries::Integer=
                                             _DEFAULT_MAX_NEARFIELD_GREEN_CACHE_ENTRIES)
    _validate_mesh_rwg_pair(mesh, rwg)
    _validate_nearfield_build_controls(
        cutoff, :spatial, factorization, ilu_tau)
    A = matrixfree_efie_operator(mesh, rwg, k;
        quad_order=quad_order,
        eta0=eta0,
        mesh_precheck=mesh_precheck,
        allow_boundary=allow_boundary,
        require_closed=require_closed,
        area_tol_rel=area_tol_rel,
        max_cache_bytes=max_cache_bytes,
        max_adjacency_pairs=max_adjacency_pairs,
    )
    return build_nearfield_preconditioner(A, cutoff;
        factorization=factorization,
        ilu_tau=ilu_tau,
        max_triplet_bytes=max_triplet_bytes,
        max_green_cache_bytes=max_green_cache_bytes,
        max_green_cache_entries=max_green_cache_entries,
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
                                         ilu_tau::Float64=1e-3,
                                         max_triplet_bytes::Integer=
                                             _DEFAULT_MAX_NEARFIELD_TRIPLET_BYTES)
    _validate_preconditioner_factorization(factorization, ilu_tau)
    N = A_aca.N
    perm = A_aca.tree.perm   # tree-order → original index

    # Pre-allocate triplets: dense blocks tile the near field without overlap
    budget = _nearfield_triplet_budget(max_triplet_bytes)
    total_entries = 0
    for db in A_aca.dense_blocks
        block_entries = _checked_nearfield_product(
            length(db.row_range), length(db.col_range),
            "ACA near-field dense-block entry count")
        total_entries = try
            Base.Checked.checked_add(total_entries, block_entries)
        catch err
            err isa OverflowError || rethrow()
            throw(ArgumentError(
                "ACA near-field triplet count exceeds the supported Int range"))
        end
        _preflight_nearfield_triplets!(budget, total_entries)
    end
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
    _validate_preconditioner_factorization(factorization, ilu_tau)
    N = _validate_sparse_preconditioner_matrix(Z_nf, "Z_nf")
    nnz_ratio = (Float64(nnz(Z_nf)) / Float64(N)) / Float64(N)

    if factorization == :ilu
        ilu_fac = IncompleteLU.ilu(Z_nf, τ = ilu_tau)
        return ILUPreconditionerData(ilu_fac, Inf, nnz_ratio, ilu_tau)
    elseif factorization == :lu
        Z_nf_fac = lu(Z_nf)
        return NearFieldPreconditionerData(Z_nf_fac, Inf, nnz_ratio)
    elseif factorization == :diag
        return _build_diagonal_preconditioner_data(
            (i, _) -> Z_nf[i, i], N, Inf)
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
    # Gather each block into reusable contiguous storage so dense `ldiv!` can
    # operate in place without allocating indexed RHS and result vectors.
    lock(P.work_lock)
    try
        for (i, idx) in enumerate(P.box_bf_indices)
            n = length(idx)
            length(P.work) >= n ||
                error("Block-diagonal preconditioner workspace is too small.")
            @inbounds for j in 1:n
                P.work[j] = y[idx[j]]
            end
            block_work = @view P.work[1:n]
            ldiv!(P.lu_blocks[i], block_work)
            @inbounds for j in 1:n
                y[idx[j]] = P.work[j]
            end
        end
    finally
        unlock(P.work_lock)
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

@inline function _ilu_u_adjoint_ldiv!(y::StridedVector{ComplexF64}, F::IncompleteLU.ILUFactorization{ComplexF64,Int})
    U = F.U
    @inbounds for col in 1:U.n
        diag = conj(U.nzval[U.colptr[col]])
        y[col] /= diag
        y_col = y[col]
        for idx in (U.colptr[col] + 1):(U.colptr[col + 1] - 1)
            y[U.rowval[idx]] -= conj(U.nzval[idx]) * y_col
        end
    end
    return y
end

@inline function _ilu_l_adjoint_ldiv!(y::StridedVector{ComplexF64}, F::IncompleteLU.ILUFactorization{ComplexF64,Int})
    L = F.L
    @inbounds for col in (L.n - 1):-1:1
        for idx in L.colptr[col]:(L.colptr[col + 1] - 1)
            y[col] -= conj(L.nzval[idx]) * y[L.rowval[idx]]
        end
    end
    return y
end

@inline function _apply_preconditioner_adjoint!(y::StridedVector{ComplexF64}, P::ILUPreconditionerData)
    # IncompleteLU.jl stores U row-wise (effectively transposed).  The
    # adjoint preconditioner must mirror ldiv!(F, y) without interpreting
    # F.U as an ordinary CSC upper-triangular matrix.
    _ilu_u_adjoint_ldiv!(y, P.ilu_fac)
    _ilu_l_adjoint_ldiv!(y, P.ilu_fac)
    return y
end

@inline function _apply_preconditioner_adjoint!(y::StridedVector{ComplexF64}, P::BlockDiagPrecondData)
    lock(P.work_lock)
    try
        for (i, idx) in enumerate(P.box_bf_indices)
            n = length(idx)
            length(P.work) >= n ||
                error("Block-diagonal preconditioner workspace is too small.")
            @inbounds for j in 1:n
                P.work[j] = y[idx[j]]
            end
            block_work = @view P.work[1:n]
            ldiv!(adjoint(P.lu_blocks[i]), block_work)
            @inbounds for j in 1:n
                y[idx[j]] = P.work[j]
            end
        end
    finally
        unlock(P.work_lock)
    end
    return y
end

@inline function _apply_preconditioner!(y::StridedVector{ComplexF64}, P::PermutedPrecondData)
    lock(P.work_lock)
    try
        length(y) == P.N && length(P.work) == P.N ||
            error("Permuted preconditioner workspace size does not match N=$(P.N).")
        copyto!(P.work, y)
        @inbounds for new_index in 1:P.N
            y[new_index] = P.work[P.perm[new_index]]
        end
        _apply_preconditioner!(y, P.inner)
        copyto!(P.work, y)
        @inbounds for old_index in 1:P.N
            y[old_index] = P.work[P.iperm[old_index]]
        end
    finally
        unlock(P.work_lock)
    end
    return y
end

@inline function _apply_preconditioner_adjoint!(y::StridedVector{ComplexF64}, P::PermutedPrecondData)
    lock(P.work_lock)
    try
        length(y) == P.N && length(P.work) == P.N ||
            error("Permuted preconditioner workspace size does not match N=$(P.N).")
        copyto!(P.work, y)
        @inbounds for new_index in 1:P.N
            y[new_index] = P.work[P.perm[new_index]]
        end
        _apply_preconditioner_adjoint!(y, P.inner)
        copyto!(P.work, y)
        @inbounds for old_index in 1:P.N
            y[old_index] = P.work[P.iperm[old_index]]
        end
    finally
        unlock(P.work_lock)
    end
    return y
end

Base.size(op::NearFieldOperator) = (_preconditioner_size(op.P), _preconditioner_size(op.P))
Base.eltype(::NearFieldOperator) = ComplexF64

@inline function _validate_preconditioner_application_lengths(
        P::AbstractPreconditionerData,
        y::Union{Nothing,AbstractVector},
        x::AbstractVector)
    N = _preconditioner_size(P)
    length(x) == N ||
        throw(DimensionMismatch(
            "preconditioner input length $(length(x)) != system size $N"))
    if y !== nothing
        length(y) == N ||
            throw(DimensionMismatch(
                "preconditioner output length $(length(y)) != system size $N"))
    end
    return nothing
end

function Base.:*(op::NearFieldOperator, v::AbstractVector)
    _validate_preconditioner_application_lengths(op.P, nothing, v)
    y = Vector{ComplexF64}(v)
    _apply_preconditioner!(y, op.P)
    return y
end

function LinearAlgebra.mul!(y::StridedVector{ComplexF64}, op::NearFieldOperator, x::StridedVector{ComplexF64})
    _validate_preconditioner_application_lengths(op.P, y, x)
    if y !== x
        copyto!(y, x)
    end
    _apply_preconditioner!(y, op.P)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, op::NearFieldOperator, x::AbstractVector)
    _validate_preconditioner_application_lengths(op.P, y, x)
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
    _validate_preconditioner_application_lengths(op.P, nothing, v)
    y = Vector{ComplexF64}(v)
    _apply_preconditioner_adjoint!(y, op.P)
    return y
end

function LinearAlgebra.mul!(y::StridedVector{ComplexF64}, op::NearFieldAdjointOperator, x::StridedVector{ComplexF64})
    _validate_preconditioner_application_lengths(op.P, y, x)
    if y !== x
        copyto!(y, x)
    end
    _apply_preconditioner_adjoint!(y, op.P)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, op::NearFieldAdjointOperator, x::AbstractVector)
    _validate_preconditioner_application_lengths(op.P, y, x)
    y .= op * x
    return y
end

function _block_diag_storage_bytes(A_mlfma)
    octree = A_mlfma.octree
    N = size(A_mlfma, 1)
    size(A_mlfma, 2) == N ||
        throw(DimensionMismatch(
            "MLFMA operator must be square, got size $(size(A_mlfma))"))
    length(octree.perm) == N ||
        throw(DimensionMismatch(
            "octree permutation length $(length(octree.perm)) != operator size $N"))
    N >= 1 ||
        throw(ArgumentError(
            "block-diagonal preconditioner requires a nonempty operator"))

    dense_entries = 0
    indexed_entries = 0
    max_block_size = 0
    try
        for box in octree.levels[end].boxes
            block_size = length(box.bf_range)
            dense_entries = Base.Checked.checked_add(
                dense_entries,
                Base.Checked.checked_mul(block_size, block_size))
            indexed_entries = Base.Checked.checked_add(
                indexed_entries, block_size)
            max_block_size = max(max_block_size, block_size)
        end
        indexed_entries == N ||
            throw(DimensionMismatch(
                "MLFMA leaf boxes cover $indexed_entries basis functions, expected $N"))

        factor_bytes = Base.Checked.checked_mul(
            dense_entries, sizeof(ComplexF64))
        # Each leaf retains its original-index vector and the LU pivot vector.
        index_bytes = Base.Checked.checked_mul(
            Base.Checked.checked_mul(2, indexed_entries), sizeof(Int))
        work_bytes = Base.Checked.checked_mul(
            max_block_size, sizeof(ComplexF64))
        return Base.Checked.checked_add(
            Base.Checked.checked_add(factor_bytes, index_bytes), work_bytes)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "block-diagonal preconditioner storage estimate overflows Int"))
    end
end

function _preflight_block_diag_storage(
        A_mlfma,
        max_storage_bytes::Integer)
    storage_bytes = _block_diag_storage_bytes(A_mlfma)
    return _enforce_payload_limit(
        storage_bytes, max_storage_bytes,
        "block-diagonal preconditioner", "max_storage_bytes")
end

function _block_diag_leaf_indices(octree, box, N::Int)
    indices = Vector{Int}(undef, length(box.bf_range))
    @inbounds for (offset, tree_index) in enumerate(box.bf_range)
        original_index = octree.perm[tree_index]
        1 <= original_index <= N ||
            throw(ArgumentError(
                "octree permutation entry $tree_index=$original_index is outside 1:$N"))
        indices[offset] = original_index
    end
    return indices
end

function _block_diag_base_matrix(A_mlfma, indices::Vector{Int})
    block_size = length(indices)
    block = Matrix{ComplexF64}(undef, block_size, block_size)
    @inbounds for column in 1:block_size, row in 1:block_size
        value = ComplexF64(
            A_mlfma.Z_near[indices[row], indices[column]])
        isfinite(value) ||
            throw(ArgumentError(
                "MLFMA near-field block contains a non-finite entry at " *
                "($(indices[row]), $(indices[column]))"))
        block[row, column] = value
    end
    return block
end

function _validate_block_diag_mass_entries(
        matrix::AbstractMatrix,
        indices::Vector{Int},
        patch::Int)
    @inbounds for original_column in indices, original_row in indices
        isfinite(matrix[original_row, original_column]) ||
            throw(ArgumentError(
                "Mp[$patch] contains a non-finite entry at " *
                "($original_row, $original_column)"))
    end
    return nothing
end

function _validate_block_diag_mass_entries(
        matrix::Union{StridedMatrix,SparseMatrixCSC,LocalMassMatrix},
        ::Vector{Int},
        patch::Int)
    _validate_known_matrix_entries(matrix, "Mp[$patch]")
    return nothing
end


function _validate_block_diag_mass_entries(
        matrix::LocalMassMatrix,
        ::Vector{Int},
        patch::Int)
    _validate_known_matrix_entries(matrix, "Mp[$patch]")
    return nothing
end

@inline function _block_diag_number_requires_exact(value::Number)
    converted = ComplexF64(value)
    isfinite(converted) || return true
    for (original_component, converted_component) in (
            (real(value), real(converted)),
            (imag(value), imag(converted)))
        if !iszero(original_component) &&
           (iszero(converted_component) ||
            abs(converted_component) < floatmin(Float64))
            return true
        end
    end
    return _ieee_dense_extreme_factor(value, Float64)
end

@inline _block_diag_impedance_coefficient(
    theta::AbstractVector,
    patch::Int,
    reactive::Bool,
) = reactive ? im * theta[patch] : theta[patch]

@inline function _block_diag_entry_requires_exact(
        base::ComplexF64,
        matrices::Vector{<:AbstractMatrix},
        theta::AbstractVector,
        reactive::Bool,
        original_row::Int,
        original_column::Int)
    _block_diag_number_requires_exact(base) && return true
    @inbounds for patch in eachindex(theta)
        coefficient = _block_diag_impedance_coefficient(
            theta, patch, reactive)
        iszero(coefficient) && continue
        value = matrices[patch][original_row, original_column]
        isfinite(value) ||
            throw(ArgumentError(
                "Mp[$patch] contains a non-finite entry at " *
                "($original_row, $original_column)"))
        if _block_diag_number_requires_exact(coefficient) ||
           _block_diag_number_requires_exact(value)
            return true
        end
    end
    return false
end

@noinline function _block_diag_loaded_entry_bigfloat(
        base::ComplexF64,
        matrices::Vector{<:AbstractMatrix},
        theta::AbstractVector,
        reactive::Bool,
        original_row::Int,
        original_column::Int)
    return setprecision(BigFloat, _LOCAL_MASS_FALLBACK_PRECISION) do
        total = Complex{BigFloat}(base)
        @inbounds for patch in eachindex(theta)
            coefficient = _block_diag_impedance_coefficient(
                theta, patch, reactive)
            iszero(coefficient) && continue
            total -= Complex{BigFloat}(coefficient) *
                     Complex{BigFloat}(
                         matrices[patch][original_row, original_column])
        end
        value = ComplexF64(total)
        isfinite(value) ||
            throw(OverflowError(
                "loaded block-diagonal entry ($original_row, " *
                "$original_column) is outside the ComplexF64 range"))
        return value
    end
end

@inline function _block_diag_loaded_entry_float(
        base::ComplexF64,
        matrices::Vector{<:AbstractMatrix},
        theta::AbstractVector,
        reactive::Bool,
        original_row::Int,
        original_column::Int)
    value = base
    real_magnitude = abs(real(base))
    imag_magnitude = abs(imag(base))
    @inbounds for patch in eachindex(theta)
        raw_coefficient = _block_diag_impedance_coefficient(
            theta, patch, reactive)
        iszero(raw_coefficient) && continue
        coefficient = ComplexF64(raw_coefficient)
        matrix_value = ComplexF64(
            matrices[patch][original_row, original_column])
        cr = real(coefficient); ci = imag(coefficient)
        vr = real(matrix_value); vi = imag(matrix_value)
        real_magnitude += abs(cr * vr) + abs(ci * vi)
        imag_magnitude += abs(cr * vi) + abs(ci * vr)
        value -= coefficient * matrix_value
    end

    # A rounded product can cancel the accumulator even when the exact stored-
    # input result is representable. Bound ordinary-path roundoff from the
    # primitive real products and additions; ambiguous components are redone
    # with exact Float-input BigFloat arithmetic.
    operation_factor = min(
        1.0,
        16 * eps(Float64) * (4 * Float64(length(theta)) + 1),
    )
    ambiguous =
        (!iszero(real_magnitude) &&
         abs(real(value)) <= operation_factor * real_magnitude) ||
        (!iszero(imag_magnitude) &&
         abs(imag(value)) <= operation_factor * imag_magnitude)
    return value, ambiguous
end

@inline function _register_block_diag_exact_work!(
        used_work::Base.RefValue{Int},
        term_count::Int,
        work_limit::Int)
    try
        entry_work = Base.Checked.checked_mul(
            term_count, _LOCAL_MASS_FALLBACK_PRECISION)
        next_work = Base.Checked.checked_add(used_work[], entry_work)
        next_work <= work_limit ||
            throw(ArgumentError(
                "loaded block-diagonal exact accumulation exceeds " *
                "max_exact_work=$work_limit"))
        used_work[] = next_work
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "loaded block-diagonal exact-work estimate overflows Int"))
    end
    return nothing
end

function _load_block_diag_matrix!(
        block::Matrix{ComplexF64},
        matrices::Vector{<:AbstractMatrix},
        theta::AbstractVector,
        reactive::Bool,
        indices::Vector{Int},
        exact_work::Base.RefValue{Int},
        exact_work_limit::Int)
    @inbounds for column in eachindex(indices), row in eachindex(indices)
        original_row = indices[row]
        original_column = indices[column]
        base = block[row, column]
        needs_exact = _block_diag_entry_requires_exact(
            base, matrices, theta, reactive,
            original_row, original_column)
        if !needs_exact
            value, needs_exact = _block_diag_loaded_entry_float(
                base, matrices, theta, reactive,
                original_row, original_column)
            if !needs_exact
                isfinite(value) ||
                    throw(OverflowError(
                        "loaded block-diagonal entry ($original_row, " *
                        "$original_column) is outside the ComplexF64 range"))
                block[row, column] = value
                continue
            end
        end
        if needs_exact
            _register_block_diag_exact_work!(
                exact_work, length(theta) + 1, exact_work_limit)
            block[row, column] = _block_diag_loaded_entry_bigfloat(
                base, matrices, theta, reactive,
                original_row, original_column)
        end
    end
    return block
end

"""
    build_block_diag_preconditioner(A_mlfma;
                                    max_storage_bytes=536_870_912)

Build a block-diagonal (block-Jacobi) preconditioner from MLFMA leaf boxes.
Each leaf box's diagonal block in Z_near is LU-factorized independently.

Much faster than ILU for large N: O(n_boxes × n_bf³) where n_bf is the
average BFs per box (typically 100-500), versus O(nnz × fill) for ILU.
`max_storage_bytes` bounds the raw payload of dense LU factors, leaf and pivot
indices, and the reusable largest-block work vector. The complete bound is
checked before the first block is allocated or factorized.
"""
function build_block_diag_preconditioner(
        A_mlfma;
        max_storage_bytes::Integer=_DEFAULT_MAX_BLOCK_DIAG_STORAGE_BYTES)
    _preflight_block_diag_storage(A_mlfma, max_storage_bytes)
    octree = A_mlfma.octree
    leaf_level = octree.levels[end]
    N = size(A_mlfma, 1)

    lu_blocks = LinearAlgebra.LU{ComplexF64, Matrix{ComplexF64}, Vector{Int64}}[]
    box_bf_indices = Vector{Int}[]
    total_nnz = 0
    for box in leaf_level.boxes
        # Map from MLFMA ordering to original BF indices (Z_near is in original ordering)
        orig_idx = _block_diag_leaf_indices(octree, box, N)
        block = _block_diag_base_matrix(A_mlfma, orig_idx)
        push!(lu_blocks, lu!(block))
        push!(box_bf_indices, orig_idx)
        total_nnz = Base.Checked.checked_add(
            total_nnz,
            Base.Checked.checked_mul(length(orig_idx), length(orig_idx)))
    end

    nnz_ratio = total_nnz / N^2
    mem_mb = round(total_nnz * 16 / 1e6, digits=1)
    n_boxes = length(lu_blocks)
    avg_bf = round(N / n_boxes, digits=0)
    println("  Block-diag preconditioner: $n_boxes blocks, avg $(Int(avg_bf)) BFs/block, $(mem_mb) MB")

    return BlockDiagPrecondData(lu_blocks, box_bf_indices, N, nnz_ratio)
end

function build_block_diag_preconditioner(A_mlfma,
                                         Mp::Vector{<:AbstractMatrix},
                                         theta::AbstractVector;
                                         reactive::Bool=false,
                                         max_storage_bytes::Integer=
                                             _DEFAULT_MAX_BLOCK_DIAG_STORAGE_BYTES,
                                         max_exact_work::Integer=
                                             _DEFAULT_MAX_BLOCK_DIAG_EXACT_WORK)
    _validate_impedance_inputs(Mp, theta, size(A_mlfma))
    _preflight_block_diag_storage(A_mlfma, max_storage_bytes)
    exact_work_limit = _validated_resource_limit(
        "max_exact_work", max_exact_work)
    octree = A_mlfma.octree
    leaf_level = octree.levels[end]
    N = size(A_mlfma, 1)

    lu_blocks = LinearAlgebra.LU{ComplexF64, Matrix{ComplexF64}, Vector{Int64}}[]
    box_bf_indices = Vector{Int}[]
    total_nnz = 0
    for patch in eachindex(theta), box in leaf_level.boxes
        indices = _block_diag_leaf_indices(octree, box, N)
        _validate_block_diag_mass_entries(Mp[patch], indices, patch)
    end
    exact_work = Ref(0)
    for box in leaf_level.boxes
        orig_idx = _block_diag_leaf_indices(octree, box, N)
        block = _block_diag_base_matrix(A_mlfma, orig_idx)
        _load_block_diag_matrix!(
            block, Mp, theta, reactive, orig_idx,
            exact_work, exact_work_limit)
        push!(lu_blocks, lu!(block))
        push!(box_bf_indices, orig_idx)
        total_nnz = Base.Checked.checked_add(
            total_nnz,
            Base.Checked.checked_mul(length(orig_idx), length(orig_idx)))
    end

    nnz_ratio = total_nnz / N^2
    mem_mb = round(total_nnz * 16 / 1e6, digits=1)
    n_boxes = length(lu_blocks)
    avg_bf = round(N / n_boxes, digits=0)
    println("  Loaded block-diag preconditioner: $n_boxes blocks, avg $(Int(avg_bf)) BFs/block, $(mem_mb) MB")

    return BlockDiagPrecondData(lu_blocks, box_bf_indices, N, nnz_ratio)
end

function _sparse_complex(A::SparseMatrixCSC)
    return SparseMatrixCSC{ComplexF64, Int}(
        size(A, 1), size(A, 2),
        Vector{Int}(A.colptr),
        Vector{Int}(A.rowval),
        ComplexF64.(A.nzval),
    )
end

function _sparse_complex(A::AbstractMatrix)
    return _sparse_complex(sparse(A))
end

function _loaded_nearfield_matrix(Z_near::SparseMatrixCSC,
                                  Mp::Vector{<:AbstractMatrix},
                                  theta::AbstractVector;
                                  reactive::Bool=false)
    _validate_impedance_inputs(Mp, theta, size(Z_near))
    Z_loaded = _sparse_complex(Z_near)
    for p in eachindex(theta)
        iszero(theta[p]) && continue
        coeff = reactive ? (1im * theta[p]) : ComplexF64(theta[p])
        Z_loaded = Z_loaded - coeff * _sparse_complex(Mp[p])
    end
    dropzeros!(Z_loaded)
    return Z_loaded
end

function _build_mlfma_preconditioner_from_nearfield(A_mlfma,
                                                    Z_near::SparseMatrixCSC;
                                                    factorization::Symbol=:ilu,
                                                    ilu_tau::Float64=1e-2)
    _validate_preconditioner_factorization(
        factorization, ilu_tau; allow_diag=false)
    octree = A_mlfma.octree
    perm = copy(octree.perm)
    N = size(A_mlfma, 1)
    _validate_sparse_preconditioner_matrix(Z_near, "MLFMA near-field matrix")
    size(Z_near) == (N, N) ||
        throw(DimensionMismatch(
            "MLFMA near-field matrix has size $(size(Z_near)), expected ($N, $N)"
        ))

    # Build inverse permutation
    iperm = zeros(Int, N)
    for i in 1:N
        iperm[perm[i]] = i
    end

    # Reorder Z_near: Z_perm[i,j] = Z_near[perm[i], perm[j]]
    # In MLFMA ordering, the matrix has block-banded structure
    println("  Reordering Z_near to MLFMA ordering...")
    Z_perm = Z_near[perm, perm]

    nnz_ratio = (Float64(nnz(Z_perm)) / Float64(N)) / Float64(N)

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
    end

    return PermutedPrecondData(inner, perm, iperm, N, nnz_ratio)
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
    _validate_preconditioner_factorization(
        factorization, ilu_tau; allow_diag=false)
    return _build_mlfma_preconditioner_from_nearfield(A_mlfma, A_mlfma.Z_near;
        factorization=factorization,
        ilu_tau=ilu_tau)
end

"""
    build_mlfma_preconditioner(A_mlfma, Mp, theta; reactive=false, factorization=:ilu, ilu_tau=1e-2)

Build an MLFMA near-field preconditioner for the current impedance-loaded
operator `Z(theta) = Z_mlfma - sum(theta[p] * Mp[p])`. This is intended for
matrix-free optimization, where a preconditioner built only from the PEC
near field may become stale as the sheet impedance changes.
"""
function build_mlfma_preconditioner(A_mlfma,
                                    Mp::Vector{<:AbstractMatrix},
                                    theta::AbstractVector;
                                    reactive::Bool=false,
                                    factorization::Symbol=:ilu,
                                    ilu_tau::Float64=1e-2)
    _validate_preconditioner_factorization(
        factorization, ilu_tau; allow_diag=false)
    Z_loaded = _loaded_nearfield_matrix(A_mlfma.Z_near, Mp, theta;
        reactive=reactive)
    return _build_mlfma_preconditioner_from_nearfield(A_mlfma, Z_loaded;
        factorization=factorization,
        ilu_tau=ilu_tau)
end
