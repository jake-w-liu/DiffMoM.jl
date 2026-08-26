# ACA.jl — Adaptive Cross Approximation and H-matrix operator
#
# Compresses the EFIE matrix into near-field (dense) and far-field (low-rank)
# blocks using a cluster tree and the standard ACA admissibility condition.
# The resulting ACAOperator <: AbstractMatrix supports O(N log² N) matvec
# and plugs directly into the existing GMRES/preconditioner infrastructure.

export ACAOperator, ACAAdjointOperator, build_aca_operator
export aca_lowrank

# A component of αv + βy contains four products of Float64 components. Their
# exact binary terms can span bit positions -2148 through 2049, requiring at
# most 4198 significant bits after cancellation. Keep a guard margin while
# confining the allocation-heavy path to exceptional exponent ranges.
const _ACA_SCALED_OUTPUT_FALLBACK_PRECISION = 4352

# A real component of alpha*U*conj(V)*x can contain four finite Float64
# factors. Written as integer multiples of 2^-1074, the exact product needs
# at most 8,392 coefficient bits; any addressable reduction adds fewer than
# 64 bits. The guard margin makes the exceptional block product exact while
# the fixed output chunk keeps its working memory independent of N.
const _ACA_PRODUCT_FALLBACK_PRECISION = 8704
const _ACA_BIGFLOAT_OUTPUT_CHUNK = 512
const _ACA_SAFE_FACTOR_EXPONENT = 128
const _DEFAULT_MAX_ACA_BLOCK_TASKS = 2_000_000
const _DEFAULT_MAX_ACA_STORAGE_BYTES = 2_000_000_000
const _DEFAULT_MAX_ACA_GREEN_WORKSPACE_BYTES = 256 * 1024 * 1024
const _DEFAULT_MAX_ACA_GREEN_CACHE_ENTRIES = 250_000

@noinline function _aca_scaled_output_bigfloat(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    return setprecision(
            BigFloat, _ACA_SCALED_OUTPUT_FALLBACK_PRECISION) do
        total = Complex{BigFloat}(alpha_scale) *
                Complex{BigFloat}(value)
        if !overwrite
            total += Complex{BigFloat}(beta_scale) *
                     Complex{BigFloat}(previous)
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "ACA scaled output is outside the representable " *
                "ComplexF64 range at row $row."))
        return converted
    end
end

@inline function _aca_scaled_output(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    needs_fallback =
        _aca_extreme_factor(value) ||
        _aca_extreme_factor(alpha_scale) ||
        (!overwrite &&
         (_aca_extreme_factor(previous) ||
          _aca_extreme_factor(beta_scale)))
    needs_fallback && return _aca_scaled_output_bigfloat(
        value, previous, alpha_scale, beta_scale, overwrite, row)
    alpha_term = alpha_scale * value
    if overwrite
        converted = ComplexF64(alpha_term)
        return isfinite(converted) ? converted :
               _aca_scaled_output_bigfloat(
                   value, previous, alpha_scale, beta_scale, true, row)
    end

    beta_term = beta_scale * previous
    combined = alpha_term + beta_term
    magnitude_sum =
        max(abs(real(alpha_term)), abs(imag(alpha_term))) +
        max(abs(real(beta_term)), abs(imag(beta_term)))
    converted = ComplexF64(combined)
    if isfinite(converted) && isfinite(magnitude_sum) &&
       !_scaled_sum_requires_exact(alpha_term, beta_term, combined)
        return converted
    end
    return _aca_scaled_output_bigfloat(
        value, previous, alpha_scale, beta_scale, false, row)
end

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

@inline function _aca_extreme_component(component::Real)
    scale = abs(Float64(component))
    isfinite(scale) || return true
    iszero(scale) && return false
    value_exponent = exponent(scale)
    return value_exponent < -_ACA_SAFE_FACTOR_EXPONENT ||
           value_exponent > _ACA_SAFE_FACTOR_EXPONENT
end

@inline function _aca_extreme_factor(value::Number)
    return _aca_extreme_component(real(value)) ||
           _aca_extreme_component(imag(value))
end

function _aca_blocks_have_extreme_factor(
        dense_blocks::Vector{DenseBlock},
        lowrank_blocks::Vector{LowRankBlock})
    @inbounds for block in dense_blocks
        any(_aca_extreme_factor, block.data) && return true
    end
    @inbounds for block in lowrank_blocks
        any(_aca_extreme_factor, block.U) && return true
        any(_aca_extreme_factor, block.V) && return true
    end
    return false
end

"""
Pre-allocated workspace for ACA `mul!`, eliminating per-call allocations.
"""
mutable struct ACAWorkspace
    x_perm::Vector{ComplexF64}
    y_perm::Vector{ComplexF64}
    tmp::Vector{ComplexF64}   # sized to max rank across all low-rank blocks
    error_bounds::Vector{ComplexF64}
    work_lock::ReentrantLock
end

ACAWorkspace(x_perm::Vector{ComplexF64},
             y_perm::Vector{ComplexF64},
             tmp::Vector{ComplexF64}) =
    ACAWorkspace(x_perm, y_perm, tmp, ReentrantLock())

ACAWorkspace(x_perm::Vector{ComplexF64},
             y_perm::Vector{ComplexF64},
             tmp::Vector{ComplexF64},
             work_lock::ReentrantLock) =
    ACAWorkspace(
        x_perm, y_perm, tmp,
        zeros(ComplexF64, length(y_perm)), work_lock)

@inline function _aca_product_component_magnitudes(
        left::ComplexF64,
        right::ComplexF64)
    left_real = real(left)
    left_imag = imag(left)
    right_real = real(right)
    right_imag = imag(right)
    return (
        abs(left_real * right_real) + abs(left_imag * right_imag),
        abs(left_real * right_imag) + abs(left_imag * right_real),
    )
end

@inline function _aca_add_error_bound(
        bound::ComplexF64,
        real_magnitude::Float64,
        imag_magnitude::Float64)
    return ComplexF64(
        real(bound) + real_magnitude,
        imag(bound) + imag_magnitude)
end

function _aca_accumulate_dense_error_bounds!(
        error_bounds::Vector{ComplexF64},
        block::DenseBlock,
        x_perm::Vector{ComplexF64},
        ::Val{ADJOINT}) where {ADJOINT}
    output_range = ADJOINT ? block.col_range : block.row_range
    input_range = ADJOINT ? block.row_range : block.col_range
    @inbounds for (local_input, input_index) in enumerate(input_range)
        input_value = x_perm[input_index]
        for (local_output, output_index) in enumerate(output_range)
            matrix_value = ADJOINT ?
                conj(block.data[local_input, local_output]) :
                block.data[local_output, local_input]
            real_magnitude, imag_magnitude =
                _aca_product_component_magnitudes(
                    matrix_value, input_value)
            error_bounds[output_index] = _aca_add_error_bound(
                error_bounds[output_index],
                real_magnitude, imag_magnitude)
        end
    end
    return nothing
end

function _aca_lowrank_inner_requires_exact(
        tmp::AbstractVector{ComplexF64},
        factor::Matrix{ComplexF64},
        input_range::UnitRange{Int},
        x_perm::Vector{ComplexF64})
    needs_fallback = false
    @inbounds for rank in axes(factor, 2)
        real_magnitude = 0.0
        imag_magnitude = 0.0
        for (local_input, input_index) in enumerate(input_range)
            term_real, term_imag =
                _aca_product_component_magnitudes(
                    conj(factor[local_input, rank]),
                    x_perm[input_index])
            real_magnitude += term_real
            imag_magnitude += term_imag
        end
        needs_fallback |=
            !isfinite(real_magnitude) || !isfinite(imag_magnitude) ||
            _matrixfree_complex_reduction_requires_exact(
                tmp[rank], real_magnitude, imag_magnitude,
                length(input_range))
    end
    return needs_fallback
end

function _aca_accumulate_lowrank_error_bounds!(
        error_bounds::Vector{ComplexF64},
        factor::Matrix{ComplexF64},
        output_range::UnitRange{Int},
        tmp::AbstractVector{ComplexF64})
    @inbounds for rank in axes(factor, 2)
        inner = tmp[rank]
        for (local_output, output_index) in enumerate(output_range)
            real_magnitude, imag_magnitude =
                _aca_product_component_magnitudes(
                    factor[local_output, rank], inner)
            error_bounds[output_index] = _aca_add_error_bound(
                error_bounds[output_index],
                real_magnitude, imag_magnitude)
        end
    end
    return nothing
end

function _aca_output_reduction_requires_exact(
        output::Vector{ComplexF64},
        error_bounds::Vector{ComplexF64},
        term_count::Int)
    @inbounds for output_index in eachindex(output, error_bounds)
        bound = error_bounds[output_index]
        if !isfinite(real(bound)) || !isfinite(imag(bound)) ||
           _matrixfree_complex_reduction_requires_exact(
               output[output_index], real(bound), imag(bound), term_count)
            return true
        end
    end
    return false
end

"""
    ACAOperator{TC} <: AbstractMatrix{ComplexF64}

H-matrix operator assembled via ACA. `mul!` and `getindex` represent the same
compressed matrix; `efie_entry` evaluates an uncompressed EFIE entry on demand.
"""
struct ACAOperator{TC<:EFIEApplyCache} <: AbstractMatrix{ComplexF64}
    cache::TC
    tree::ClusterTree
    dense_blocks::Vector{DenseBlock}
    lowrank_blocks::Vector{LowRankBlock}
    N::Int
    workspace::ACAWorkspace
    extreme_operator_factor::Bool
end


ACAOperator{TC}(
        cache::TC,
        tree::ClusterTree,
        dense_blocks::Vector{DenseBlock},
        lowrank_blocks::Vector{LowRankBlock},
        N::Int,
        workspace::ACAWorkspace) where {TC<:EFIEApplyCache} =
    ACAOperator{TC}(
        cache, tree, dense_blocks, lowrank_blocks, N, workspace,
        _aca_blocks_have_extreme_factor(dense_blocks, lowrank_blocks))

ACAOperator(
        cache::TC,
        tree::ClusterTree,
        dense_blocks::Vector{DenseBlock},
        lowrank_blocks::Vector{LowRankBlock},
        N::Int,
        workspace::ACAWorkspace) where {TC<:EFIEApplyCache} =
    ACAOperator{TC}(
        cache, tree, dense_blocks, lowrank_blocks, N, workspace)

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

@noinline function _aca_lowrank_entry_bigfloat(
        block::LowRankBlock,
        local_row::Int,
        local_column::Int)
    return setprecision(BigFloat, _ACA_SCALED_OUTPUT_FALLBACK_PRECISION) do
        total = zero(Complex{BigFloat})
        @inbounds for rank in axes(block.U, 2)
            total += Complex{BigFloat}(block.U[local_row, rank]) *
                     conj(Complex{BigFloat}(block.V[local_column, rank]))
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "ACA low-rank entry is outside the representable " *
                "ComplexF64 range"))
        return converted
    end
end

function _aca_lowrank_entry(
        block::LowRankBlock,
        local_row::Int,
        local_column::Int)
    value = zero(ComplexF64)
    real_magnitude = 0.0
    imag_magnitude = 0.0
    needs_fallback = false
    @inbounds for rank in axes(block.U, 2)
        left = block.U[local_row, rank]
        right = conj(block.V[local_column, rank])
        if !iszero(left) && !iszero(right) &&
           (_aca_extreme_factor(left) || _aca_extreme_factor(right))
            needs_fallback = true
            break
        end
        term = left * right
        value += term
        real_magnitude += abs(real(term))
        imag_magnitude += abs(imag(term))
        if !isfinite(value) || !isfinite(real_magnitude) ||
           !isfinite(imag_magnitude)
            needs_fallback = true
            break
        end
    end
    needs_fallback |= _matrixfree_complex_reduction_requires_exact(
        value, real_magnitude, imag_magnitude, size(block.U, 2))
    return needs_fallback ?
           _aca_lowrank_entry_bigfloat(block, local_row, local_column) : value
end

function Base.getindex(A::ACAOperator, i::Int, j::Int)
    checkbounds(A, i, j)
    tree_row = A.tree.iperm[i]
    tree_column = A.tree.iperm[j]
    @inbounds for block in A.dense_blocks
        if tree_row in block.row_range && tree_column in block.col_range
            return block.data[
                tree_row - first(block.row_range) + 1,
                tree_column - first(block.col_range) + 1,
            ]
        end
    end
    @inbounds for block in A.lowrank_blocks
        if tree_row in block.row_range && tree_column in block.col_range
            return _aca_lowrank_entry(
                block,
                tree_row - first(block.row_range) + 1,
                tree_column - first(block.col_range) + 1,
            )
        end
    end
    error("ACA block partition does not cover entry ($i, $j)")
end

function Base.getindex(A::ACAAdjointOperator, i::Int, j::Int)
    checkbounds(A, i, j)
    return conj(A.op[j, i])
end

LinearAlgebra.adjoint(A::ACAOperator) = ACAAdjointOperator{typeof(A)}(A)
LinearAlgebra.adjoint(A::ACAAdjointOperator) = A.op

function efie_entry(A::ACAOperator, i::Int, j::Int)
    checkbounds(A, i, j)
    return _efie_entry(A.cache, i, j)
end

function efie_entry(A::ACAAdjointOperator, i::Int, j::Int)
    checkbounds(A, i, j)
    return conj(_efie_entry(A.op.cache, j, i))
end

# ─── ACA low-rank approximation ──────────────────────────────────

struct _ACAScaledMagnitude
    fraction::Float64
    exponent::Int
end

struct _ACAVectorNorm
    component_scale::Float64
    scaled_norm::Float64
    magnitude::_ACAScaledMagnitude
end

function _aca_vector_norm_geometry(values::Vector{ComplexF64})
    component_scale = 0.0
    @inbounds for value in values
        component_scale = max(
            component_scale,
            abs(real(value)),
            abs(imag(value)),
        )
    end
    component_scale > 0.0 ||
        throw(OverflowError("ACA low-rank update has zero norm"))

    scaled_norm_sq = 0.0
    @inbounds for value in values
        real_scaled = real(value) / component_scale
        imag_scaled = imag(value) / component_scale
        scaled_norm_sq = muladd(
            real_scaled, real_scaled, scaled_norm_sq)
        scaled_norm_sq = muladd(
            imag_scaled, imag_scaled, scaled_norm_sq)
    end
    scaled_norm = sqrt(scaled_norm_sq)

    scale_fraction, scale_exponent = frexp(component_scale)
    norm_fraction, norm_exponent = frexp(
        scale_fraction * scaled_norm)
    magnitude_exponent = Base.Checked.checked_add(
        scale_exponent, norm_exponent)
    return _ACAVectorNorm(
        component_scale,
        scaled_norm,
        _ACAScaledMagnitude(norm_fraction, magnitude_exponent),
    )
end

@inline function _aca_multiply_magnitudes(
        left::_ACAScaledMagnitude,
        right::_ACAScaledMagnitude)
    product_fraction, product_exponent = frexp(
        left.fraction * right.fraction)
    exponent_sum = Base.Checked.checked_add(
        left.exponent, right.exponent)
    return _ACAScaledMagnitude(
        product_fraction,
        Base.Checked.checked_add(exponent_sum, product_exponent),
    )
end

@inline function _aca_magnitude_isless(
        left::_ACAScaledMagnitude,
        right::_ACAScaledMagnitude)
    left.exponent == right.exponent &&
        return left.fraction < right.fraction
    return left.exponent < right.exponent
end

@inline function _aca_magnitude_ratio(
        numerator::_ACAScaledMagnitude,
        denominator::_ACAScaledMagnitude)
    exponent_delta = numerator.exponent - denominator.exponent
    return ldexp(
        numerator.fraction / denominator.fraction,
        exponent_delta,
    )
end

function _aca_normalized_dot(
        left::Vector{ComplexF64},
        right::Vector{ComplexF64},
        left_norm::_ACAVectorNorm,
        right_norm::_ACAVectorNorm)
    length(left) == length(right) ||
        throw(DimensionMismatch("ACA factor columns must have equal lengths"))
    real_sum = 0.0
    imag_sum = 0.0
    @inbounds for index in eachindex(left, right)
        left_real = real(left[index]) / left_norm.component_scale
        left_imag = imag(left[index]) / left_norm.component_scale
        right_real = real(right[index]) / right_norm.component_scale
        right_imag = imag(right[index]) / right_norm.component_scale
        real_sum = muladd(left_real, right_real, real_sum)
        real_sum = muladd(left_imag, right_imag, real_sum)
        imag_sum = muladd(left_real, right_imag, imag_sum)
        imag_sum = muladd(-left_imag, right_real, imag_sum)
    end
    norm_product = left_norm.scaled_norm * right_norm.scaled_norm
    return ComplexF64(real_sum / norm_product, imag_sum / norm_product)
end

@inline function _validate_aca_options(tol::Float64, max_rank::Int)
    (isfinite(tol) && tol > 0.0) ||
        throw(ArgumentError(
            "ACA tolerance must be finite and positive, got $tol"))
    max_rank >= 1 ||
        throw(ArgumentError(
            "ACA max_rank must be at least 1, got $max_rank"))
    return nothing
end

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
                     max_rank::Int=50,
                     max_output_bytes::Integer=
                         _DEFAULT_MAX_ACA_STORAGE_BYTES)
    _validate_aca_options(tol, max_rank)
    m = length(row_indices)
    n = length(col_indices)
    full_rank = min(m, n)
    max_rank = min(max_rank, full_rank)
    planned_entries = try
        Base.Checked.checked_mul(max_rank, Base.Checked.checked_add(m, n))
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("ACA low-rank output estimate overflows Int"))
    end
    planned_bytes = try
        Base.Checked.checked_mul(planned_entries, sizeof(ComplexF64))
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("ACA low-rank output payload estimate overflows Int"))
    end
    _enforce_payload_limit(
        planned_bytes, max_output_bytes,
        "ACA low-rank factors", "max_output_bytes")

    U_cols = Vector{Vector{ComplexF64}}()
    V_cols = Vector{Vector{ComplexF64}}()
    U_norms = _ACAVectorNorm[]
    V_norms = _ACAVectorNorm[]
    term_norms = _ACAScaledMagnitude[]
    sizehint!(U_cols, max_rank)
    sizehint!(V_cols, max_rank)
    sizehint!(U_norms, max_rank)
    sizehint!(V_norms, max_rank)
    sizehint!(term_norms, max_rank)

    used_rows = falses(m)
    used_cols = falses(n)
    frob_scale = _ACAScaledMagnitude(0.0, 0)
    frob_sq_scaled = 0.0
    have_frob_scale = false

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
        if pivot_col == 0 || iszero(best_val)
            for jj in 1:n
                if !used_cols[jj]
                    pivot_col = jj
                    break
                end
            end
            pivot_col == 0 && break
            best_val = abs(row_vec[pivot_col])
            iszero(best_val) && break
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

        # A partial factorization is not a valid approximation after a
        # non-finite pivot/update.  Signal the block builder so it can fall back
        # to a complete dense block instead of silently returning stale rank.
        (all(isfinite, u_k) && all(isfinite, v_k)) ||
            throw(OverflowError(
                "ACA low-rank update became non-finite"))

        u_norm = _aca_vector_norm_geometry(u_k)
        v_norm = _aca_vector_norm_geometry(v_k)
        term_norm = _aca_multiply_magnitudes(
            u_norm.magnitude, v_norm.magnitude)
        push!(U_cols, u_k)
        push!(V_cols, v_k)
        push!(U_norms, u_norm)
        push!(V_norms, v_norm)
        push!(term_norms, term_norm)
        used_rows[pivot_row] = true
        used_cols[pivot_col] = true

        # Update the Frobenius estimate in units of the largest rank-one
        # term. Keeping every magnitude as a fraction/exponent pair makes the
        # relative stopping rule invariant to a common matrix scale, including
        # scales whose squared norms underflow or overflow Float64.
        if !have_frob_scale
            frob_scale = term_norm
            have_frob_scale = true
        elseif _aca_magnitude_isless(frob_scale, term_norm)
            old_scale_ratio = _aca_magnitude_ratio(
                frob_scale, term_norm)
            frob_sq_scaled *= old_scale_ratio^2
            frob_scale = term_norm
        end

        term_ratio = _aca_magnitude_ratio(term_norm, frob_scale)
        cross_term_scaled = 0.0
        for prev in 1:(length(U_cols)-1)
            previous_ratio = _aca_magnitude_ratio(
                term_norms[prev], frob_scale)
            u_correlation = _aca_normalized_dot(
                U_cols[prev], u_k, U_norms[prev], u_norm)
            v_correlation = _aca_normalized_dot(
                V_cols[prev], v_k, V_norms[prev], v_norm)
            cross_term_scaled +=
                2.0 * previous_ratio * term_ratio *
                real(u_correlation * conj(v_correlation))
        end
        frob_sq_scaled += term_ratio^2 + cross_term_scaled
        frob_sq_scaled = max(
            frob_sq_scaled, 0.0)  # guard against roundoff below zero

        # Convergence check
        if term_ratio < tol * sqrt(frob_sq_scaled)
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

    if !converged && length(U_cols) == max_rank && max_rank < full_rank
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

function _fill_dense_block_batched!(
        data::Matrix{ComplexF64}, cache::EFIEApplyCache,
        row_indices::Vector{Int}, col_indices::Vector{Int};
        max_green_cache_bytes::Integer=
            _DEFAULT_MAX_ACA_GREEN_WORKSPACE_BYTES,
        max_green_cache_entries::Int=
            _DEFAULT_MAX_ACA_GREEN_CACHE_ENTRIES)
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

    max_green_cache_entries >= 0 ||
        throw(ArgumentError(
            "max_green_cache_entries must be nonnegative, got " *
            "$max_green_cache_entries"))
    green_limit = _validated_resource_limit(
        "max_green_cache_bytes", max_green_cache_bytes)
    green_matrix_bytes = _checked_array_payload_bytes(
        ComplexF64, Nq, Nq; label="ACA Green matrix")
    matrix_slots = div(green_limit, green_matrix_bytes)
    max_cached_greens = min(
        max_green_cache_entries, max(0, matrix_slots - 1))

    # 2. Cache as many regular triangle-pair Green matrices as the workspace
    # budget permits. Once full, one matrix is reused as scratch.
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
        cached = get(green_cache, (tm, tn), nothing)
        cached !== nothing && return cached
        matrix_slots >= 1 ||
            throw(ArgumentError(
                "one ACA Green matrix requires $green_matrix_bytes raw " *
                "bytes, exceeding max_green_cache_bytes=$green_limit"))
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

    for tm in row_tris, tn in col_tris
        (tm == tn || _is_adjacent(cache, tm, tn)) && continue
        length(green_cache) >= max_cached_greens && break
        _get_greens(tm, tn)
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
                        G_mat = _get_greens(tm, tn)
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
            data[ii, jj] = _finalize_efie_entry(
                cache, val, m_idx, n_idx)
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
                             row_node::Int, col_node::Int;
                             eta::Float64,
                             max_block_tasks::Int=typemax(Int))
    length(tasks) < max_block_tasks ||
        throw(ArgumentError(
            "ACA block enumeration exceeds max_block_tasks=$max_block_tasks"))
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
        _enum_block_tasks!(tasks, tree, row_node, cn.left;
                           eta=eta, max_block_tasks=max_block_tasks)
        _enum_block_tasks!(tasks, tree, row_node, cn.right;
                           eta=eta, max_block_tasks=max_block_tasks)
    elseif is_leaf(tree, col_node)
        _enum_block_tasks!(tasks, tree, rn.left, col_node;
                           eta=eta, max_block_tasks=max_block_tasks)
        _enum_block_tasks!(tasks, tree, rn.right, col_node;
                           eta=eta, max_block_tasks=max_block_tasks)
    else
        _enum_block_tasks!(tasks, tree, rn.left, cn.left;
                           eta=eta, max_block_tasks=max_block_tasks)
        _enum_block_tasks!(tasks, tree, rn.left, cn.right;
                           eta=eta, max_block_tasks=max_block_tasks)
        _enum_block_tasks!(tasks, tree, rn.right, cn.left;
                           eta=eta, max_block_tasks=max_block_tasks)
        _enum_block_tasks!(tasks, tree, rn.right, cn.right;
                           eta=eta, max_block_tasks=max_block_tasks)
    end
end

function _validate_aca_task_storage(
    tasks::Vector{BlockTask},
    tree::ClusterTree,
    max_rank::Int,
    max_storage_bytes::Int,
)
    bytes = try
        entries = 0
        @inbounds for task in tasks
            rows = length(tree.nodes[task.row_node].indices)
            columns = length(tree.nodes[task.col_node].indices)
            block_entries = if task.kind === :dense
                Base.Checked.checked_mul(rows, columns)
            else
                rank = min(max_rank, rows, columns)
                Base.Checked.checked_mul(
                    rank, Base.Checked.checked_add(rows, columns))
            end
            entries = Base.Checked.checked_add(entries, block_entries)
        end
        Base.Checked.checked_mul(entries, sizeof(ComplexF64))
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("ACA block storage estimate overflows Int"))
    end
    bytes <= max_storage_bytes ||
        throw(ArgumentError(
            "ACA block storage requires at most $bytes raw bytes, exceeding " *
            "max_storage_bytes=$max_storage_bytes"))
    return bytes
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
- `max_block_tasks=2_000_000`: block-enumeration limit
- `max_storage_bytes=2_000_000_000`: raw persistent block-payload limit,
  including any dense replacement of a failed low-rank block
- `max_cache_bytes=2_000_000_000`: EFIE quadrature/cache workspace limit
- `max_adjacency_pairs=20_000_000`: triangle-adjacency pair-record limit
- `max_green_cache_bytes=268_435_456`: per-worker cached/scratch Green-matrix
  raw-payload limit
- `max_green_cache_entries=250_000`: per-worker retained triangle-pair count
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
                            area_tol_rel::Float64=1e-12,
                            max_block_tasks::Int=_DEFAULT_MAX_ACA_BLOCK_TASKS,
                            max_storage_bytes::Int=_DEFAULT_MAX_ACA_STORAGE_BYTES,
                            max_cache_bytes::Integer=
                                _DEFAULT_MAX_EFIE_CACHE_BYTES,
                            max_adjacency_pairs::Integer=
                                _DEFAULT_MAX_EFIE_ADJACENCY_PAIRS,
                            max_green_cache_bytes::Integer=
                                _DEFAULT_MAX_ACA_GREEN_WORKSPACE_BYTES,
                            max_green_cache_entries::Int=
                                _DEFAULT_MAX_ACA_GREEN_CACHE_ENTRIES)
    _validate_mesh_rwg_pair(mesh, rwg)
    _validate_aca_options(aca_tol, max_rank)
    leaf_size >= 1 ||
        throw(ArgumentError(
            "ACA leaf_size must be at least 1, got $leaf_size"))
    isfinite(eta) && eta > 0.0 ||
        throw(ArgumentError(
            "ACA eta must be finite and positive, got $eta"))
    max_block_tasks >= 1 ||
        throw(ArgumentError(
            "max_block_tasks must be positive, got $max_block_tasks"))
    max_storage_bytes >= 1 ||
        throw(ArgumentError(
            "max_storage_bytes must be positive, got $max_storage_bytes"))
    green_cache_limit = _validated_resource_limit(
        "max_green_cache_bytes", max_green_cache_bytes)
    max_green_cache_entries >= 0 ||
        throw(ArgumentError(
            "max_green_cache_entries must be nonnegative, got " *
            "$max_green_cache_entries"))

    # Validate the smallest possible Green workspace before mesh auditing,
    # EFIE-cache construction, or threaded block allocation.  Worker-local
    # checks remain in _fill_dense_block_batched! as defense in depth.
    _, aca_quad_weights = tri_quad_rule(quad_order)
    green_matrix_bytes = _checked_array_payload_bytes(
        ComplexF64, length(aca_quad_weights), length(aca_quad_weights);
        label="ACA Green matrix")
    green_matrix_bytes <= green_cache_limit ||
        throw(ArgumentError(
            "one ACA Green matrix requires $green_matrix_bytes raw bytes, " *
            "exceeding max_green_cache_bytes=$green_cache_limit"))
    if mesh_precheck
        assert_mesh_quality(mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end

    N = rwg.nedges
    cache = _build_efie_cache(
        mesh, rwg, k;
        quad_order=quad_order,
        eta0=eta0,
        max_cache_bytes=max_cache_bytes,
        max_adjacency_pairs=max_adjacency_pairs)

    centers = rwg_centers(mesh, rwg)
    tree = build_cluster_tree(centers; leaf_size=leaf_size)

    # Phase 1: enumerate all block tasks (serial, fast)
    tasks = BlockTask[]
    _enum_block_tasks!(tasks, tree, 1, 1;
                       eta=eta, max_block_tasks=max_block_tasks)
    claimed_storage = Ref(_validate_aca_task_storage(
        tasks, tree, max_rank, max_storage_bytes))
    storage_lock = ReentrantLock()

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
            _fill_dense_block_batched!(
                data, cache, row_indices, col_indices;
                max_green_cache_bytes=max_green_cache_bytes,
                max_green_cache_entries=max_green_cache_entries)
            results[i] = DenseBlock(rn.indices, cn.indices, data)
        else
            try
                U, V = aca_lowrank(cache, row_indices, col_indices;
                                   tol=aca_tol, max_rank=max_rank,
                                   max_output_bytes=max_storage_bytes)
                results[i] = LowRankBlock(rn.indices, cn.indices, U, V)
            catch error
                if error isa OverflowError &&
                   occursin("ACA low-rank update became non-finite",
                            sprint(showerror, error))
                    dense_entries = try
                        Base.Checked.checked_mul(
                            length(row_indices), length(col_indices))
                    catch overflow
                        overflow isa OverflowError || rethrow()
                        throw(ArgumentError(
                            "ACA dense fallback storage estimate overflows Int"))
                    end
                    planned_entries = try
                        planned_rank = min(
                            max_rank,
                            length(row_indices),
                            length(col_indices),
                        )
                        Base.Checked.checked_mul(
                            planned_rank,
                            Base.Checked.checked_add(
                                length(row_indices), length(col_indices)))
                    catch overflow
                        overflow isa OverflowError || rethrow()
                        throw(ArgumentError(
                            "ACA low-rank storage estimate overflows Int"))
                    end
                    extra_entries = max(0, dense_entries - planned_entries)
                    extra_bytes = try
                        Base.Checked.checked_mul(
                            extra_entries, sizeof(ComplexF64))
                    catch overflow
                        overflow isa OverflowError || rethrow()
                        throw(ArgumentError(
                            "ACA dense fallback storage estimate overflows Int"))
                    end
                    lock(storage_lock)
                    try
                        extra_bytes <= max_storage_bytes - claimed_storage[] ||
                            throw(ArgumentError(
                                "ACA dense fallback would exceed " *
                                "max_storage_bytes=$max_storage_bytes"))
                        claimed_storage[] += extra_bytes
                    finally
                        unlock(storage_lock)
                    end
                    data = Matrix{ComplexF64}(
                        undef, length(row_indices), length(col_indices))
                    _fill_dense_block_batched!(
                        data, cache, row_indices, col_indices;
                        max_green_cache_bytes=max_green_cache_bytes,
                        max_green_cache_entries=max_green_cache_entries)
                    results[i] = DenseBlock(rn.indices, cn.indices, data)
                else
                    rethrow()
                end
            end
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

    workspace_rank = isempty(lowrank_blocks) ?
        0 : maximum(size(block.U, 2) for block in lowrank_blocks)
    ws = ACAWorkspace(Vector{ComplexF64}(undef, N),
                       zeros(ComplexF64, N),
                       Vector{ComplexF64}(undef, max(workspace_rank, 1)))
    return ACAOperator{typeof(cache)}(cache, tree, dense_blocks, lowrank_blocks, N, ws)
end

# ─── Matvec ───────────────────────────────────────────────────────

@noinline function _aca_product_bigfloat!(
        y::AbstractVector{ComplexF64},
        A::ACAOperator,
        x_perm::Vector{ComplexF64},
        normal_product::Vector{ComplexF64},
        alpha_scale::Number,
        beta_scale::Number,
        adjoint_mode::Val{ADJOINT}) where {ADJOINT}
    N = A.N
    return setprecision(BigFloat, _ACA_PRODUCT_FALLBACK_PRECISION) do
        chunk_capacity = min(N, _ACA_BIGFLOAT_OUTPUT_CHUNK)
        totals = Vector{Complex{BigFloat}}(undef, chunk_capacity)
        alpha_big = Complex{BigFloat}(alpha_scale)
        include_previous = !iszero(beta_scale)
        beta_big = include_previous ?
                   Complex{BigFloat}(beta_scale) :
                   zero(Complex{BigFloat})

        for chunk_first in 1:_ACA_BIGFLOAT_OUTPUT_CHUNK:N
            chunk_last = min(N, chunk_first + _ACA_BIGFLOAT_OUTPUT_CHUNK - 1)
            chunk_length = chunk_last - chunk_first + 1
            # This routine is entered only after an extreme stored factor or
            # input was detected.  Recompute every row in the chunk: a tiny
            # inner product may have rounded to zero before multiplication by
            # a large low-rank factor even when the final Float64 row is
            # ordinary and finite.
            chunk_needs_fallback = true

            if chunk_needs_fallback
                @inbounds for offset in 1:chunk_length
                    totals[offset] = zero(Complex{BigFloat})
                end
            end

            if chunk_needs_fallback && ADJOINT
                @inbounds for block in A.dense_blocks
                    output_range = block.col_range
                    output_first = max(first(output_range), chunk_first)
                    output_last = min(last(output_range), chunk_last)
                    output_first > output_last && continue
                    input_range = block.row_range
                    for output_index in output_first:output_last
                        total = totals[output_index - chunk_first + 1]
                        local_column = output_index - first(output_range) + 1
                        for input_index in input_range
                            local_row = input_index - first(input_range) + 1
                            total +=
                                conj(Complex{BigFloat}(
                                    block.data[local_row, local_column])) *
                                Complex{BigFloat}(x_perm[input_index])
                        end
                        totals[output_index - chunk_first + 1] = total
                    end
                end

                @inbounds for block in A.lowrank_blocks
                    output_range = block.col_range
                    output_first = max(first(output_range), chunk_first)
                    output_last = min(last(output_range), chunk_last)
                    output_first > output_last && continue
                    input_range = block.row_range
                    for rank in axes(block.U, 2)
                        inner = zero(Complex{BigFloat})
                        for input_index in input_range
                            local_row = input_index - first(input_range) + 1
                            inner +=
                                conj(Complex{BigFloat}(
                                    block.U[local_row, rank])) *
                                Complex{BigFloat}(x_perm[input_index])
                        end
                        for output_index in output_first:output_last
                            local_column =
                                output_index - first(output_range) + 1
                            totals[output_index - chunk_first + 1] +=
                                Complex{BigFloat}(
                                    block.V[local_column, rank]) * inner
                        end
                    end
                end
            elseif chunk_needs_fallback
                @inbounds for block in A.dense_blocks
                    output_range = block.row_range
                    output_first = max(first(output_range), chunk_first)
                    output_last = min(last(output_range), chunk_last)
                    output_first > output_last && continue
                    input_range = block.col_range
                    for output_index in output_first:output_last
                        total = totals[output_index - chunk_first + 1]
                        local_row = output_index - first(output_range) + 1
                        for input_index in input_range
                            local_column = input_index - first(input_range) + 1
                            total +=
                                Complex{BigFloat}(
                                    block.data[local_row, local_column]) *
                                Complex{BigFloat}(x_perm[input_index])
                        end
                        totals[output_index - chunk_first + 1] = total
                    end
                end

                @inbounds for block in A.lowrank_blocks
                    output_range = block.row_range
                    output_first = max(first(output_range), chunk_first)
                    output_last = min(last(output_range), chunk_last)
                    output_first > output_last && continue
                    input_range = block.col_range
                    for rank in axes(block.V, 2)
                        inner = zero(Complex{BigFloat})
                        for input_index in input_range
                            local_column =
                                input_index - first(input_range) + 1
                            inner +=
                                conj(Complex{BigFloat}(
                                    block.V[local_column, rank])) *
                                Complex{BigFloat}(x_perm[input_index])
                        end
                        for output_index in output_first:output_last
                            local_row = output_index - first(output_range) + 1
                            totals[output_index - chunk_first + 1] +=
                                Complex{BigFloat}(
                                    block.U[local_row, rank]) * inner
                        end
                    end
                end
            end

            @inbounds for output_index in chunk_first:chunk_last
                original_index = A.tree.perm[output_index]
                normal_value = normal_product[output_index]
                total = alpha_big * totals[output_index - chunk_first + 1]
                if include_previous
                    total += beta_big * Complex{BigFloat}(y[original_index])
                end
                converted = ComplexF64(total)
                isfinite(converted) ||
                    throw(OverflowError(
                        "ACA $(ADJOINT ? "adjoint " : "")product is " *
                        "outside the representable ComplexF64 range at " *
                        "row $original_index."))
                y[original_index] = converted
            end
        end
        return y
    end
end

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64}, A::ACAOperator,
                            x::AbstractVector, alpha_scale::Number,
                            beta_scale::Number)
    N = A.N
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))
    if iszero(alpha_scale)
        if iszero(beta_scale)
            fill!(y, zero(ComplexF64))
        elseif beta_scale != one(beta_scale)
            @inbounds for row in eachindex(y)
                y[row] = _aca_scaled_output(
                    zero(ComplexF64), y[row], alpha_scale,
                    beta_scale, false, row)
            end
        end
        return y
    end

    ws = A.workspace
    lock(ws.work_lock)
    try
        x_perm = ws.x_perm
        y_perm = ws.y_perm
        error_bounds = ws.error_bounds

        # Permute x to tree order and decide whether Float64 block products
        # can preserve their full exponent range.
        needs_fallback = A.extreme_operator_factor
        @inbounds for k in 1:N
            x_perm[k] = x[A.tree.perm[k]]
            needs_fallback |= _aca_extreme_factor(x_perm[k])
        end
        fill!(y_perm, zero(ComplexF64))
        fill!(error_bounds, zero(ComplexF64))

        # Dense blocks — BLAS gemv
        for blk in A.dense_blocks
            rows = blk.row_range
            cols = blk.col_range
            mul!(@view(y_perm[rows]), blk.data, @view(x_perm[cols]),
                 one(ComplexF64), one(ComplexF64))
            _aca_accumulate_dense_error_bounds!(
                error_bounds, blk, x_perm, Val(false))
        end

        # Low-rank blocks: y += U * (V' * x)
        for blk in A.lowrank_blocks
            k_rank = size(blk.U, 2)
            k_rank == 0 && continue
            rows = blk.row_range
            cols = blk.col_range
            tmp = @view(ws.tmp[1:k_rank])
            mul!(tmp, blk.V', @view(x_perm[cols]))
            needs_fallback |= _aca_lowrank_inner_requires_exact(
                tmp, blk.V, cols, x_perm)
            mul!(@view(y_perm[rows]), blk.U, tmp,
                 one(ComplexF64), one(ComplexF64))
            _aca_accumulate_lowrank_error_bounds!(
                error_bounds, blk.U, rows, tmp)
        end

        reduction_terms = N > typemax(Int) - length(ws.tmp) ?
            typemax(Int) : N + length(ws.tmp)
        needs_fallback |= _aca_output_reduction_requires_exact(
            y_perm, error_bounds, reduction_terms)

        if needs_fallback
            return _aca_product_bigfloat!(
                y, A, x_perm, y_perm,
                alpha_scale, beta_scale, Val(false))
        end

        # Un-permute y back to original order
        if iszero(beta_scale)
            if alpha_scale == one(alpha_scale)
                @inbounds for k in 1:N
                    y[A.tree.perm[k]] = y_perm[k]
                end
            else
                @inbounds for k in 1:N
                    original_idx = A.tree.perm[k]
                    y[original_idx] = _aca_scaled_output(
                        y_perm[k], y[original_idx], alpha_scale,
                        beta_scale, true, original_idx)
                end
            end
        else
            @inbounds for k in 1:N
                original_idx = A.tree.perm[k]
                y[original_idx] = _aca_scaled_output(
                    y_perm[k], y[original_idx], alpha_scale,
                    beta_scale, false, original_idx)
            end
        end
    finally
        unlock(ws.work_lock)
    end

    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64}, A::ACAOperator,
                   x::AbstractVector) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

function Base.:*(A::ACAOperator, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, _complex_vector_input(x))
    return y
end

# ─── Adjoint matvec ───────────────────────────────────────────────

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64}, A::ACAAdjointOperator,
                            x::AbstractVector, alpha_scale::Number,
                            beta_scale::Number)
    N = A.op.N
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))
    if iszero(alpha_scale)
        if iszero(beta_scale)
            fill!(y, zero(ComplexF64))
        elseif beta_scale != one(beta_scale)
            @inbounds for row in eachindex(y)
                y[row] = _aca_scaled_output(
                    zero(ComplexF64), y[row], alpha_scale,
                    beta_scale, false, row)
            end
        end
        return y
    end

    tree = A.op.tree
    ws = A.op.workspace
    lock(ws.work_lock)
    try
        x_perm = ws.x_perm
        y_perm = ws.y_perm
        error_bounds = ws.error_bounds

        # Permute x to tree order and decide whether Float64 block products
        # can preserve their full exponent range.
        needs_fallback = A.op.extreme_operator_factor
        @inbounds for k in 1:N
            x_perm[k] = x[tree.perm[k]]
            needs_fallback |= _aca_extreme_factor(x_perm[k])
        end
        fill!(y_perm, zero(ComplexF64))
        fill!(error_bounds, zero(ComplexF64))

        # Dense blocks: adjoint means transpose-conjugate
        for blk in A.op.dense_blocks
            rows = blk.row_range
            cols = blk.col_range
            mul!(@view(y_perm[cols]), blk.data', @view(x_perm[rows]),
                 one(ComplexF64), one(ComplexF64))
            _aca_accumulate_dense_error_bounds!(
                error_bounds, blk, x_perm, Val(true))
        end

        # Low-rank blocks: adjoint of U * V' is V * U'
        for blk in A.op.lowrank_blocks
            k_rank = size(blk.U, 2)
            k_rank == 0 && continue
            rows = blk.row_range
            cols = blk.col_range
            tmp = @view(ws.tmp[1:k_rank])
            mul!(tmp, blk.U', @view(x_perm[rows]))
            needs_fallback |= _aca_lowrank_inner_requires_exact(
                tmp, blk.U, rows, x_perm)
            mul!(@view(y_perm[cols]), blk.V, tmp,
                 one(ComplexF64), one(ComplexF64))
            _aca_accumulate_lowrank_error_bounds!(
                error_bounds, blk.V, cols, tmp)
        end

        reduction_terms = N > typemax(Int) - length(ws.tmp) ?
            typemax(Int) : N + length(ws.tmp)
        needs_fallback |= _aca_output_reduction_requires_exact(
            y_perm, error_bounds, reduction_terms)

        if needs_fallback
            return _aca_product_bigfloat!(
                y, A.op, x_perm, y_perm,
                alpha_scale, beta_scale, Val(true))
        end

        # Un-permute
        if iszero(beta_scale)
            if alpha_scale == one(alpha_scale)
                @inbounds for k in 1:N
                    y[tree.perm[k]] = y_perm[k]
                end
            else
                @inbounds for k in 1:N
                    original_idx = tree.perm[k]
                    y[original_idx] = _aca_scaled_output(
                        y_perm[k], y[original_idx], alpha_scale,
                        beta_scale, true, original_idx)
                end
            end
        else
            @inbounds for k in 1:N
                original_idx = tree.perm[k]
                y[original_idx] = _aca_scaled_output(
                    y_perm[k], y[original_idx], alpha_scale,
                    beta_scale, false, original_idx)
            end
        end
    finally
        unlock(ws.work_lock)
    end

    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64}, A::ACAAdjointOperator,
                   x::AbstractVector) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

function Base.:*(A::ACAAdjointOperator, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, _complex_vector_input(x))
    return y
end
