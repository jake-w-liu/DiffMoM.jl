# QMatrix.jl — Build the Q matrix for the quadratic far-field objective
#
# Q_mn = Σ_q w_q (p†·g_m)* (p†·g_n)
# J(θ) = I† Q I  (radiated power in selected direction/polarization)

export FarFieldQMatrix, SumQMatrix, build_Q, build_Q_operator, apply_Q, pol_linear_x, pol_linear_y, cap_mask, direction_mask

"""
    FarFieldQMatrix

Matrix-free representation of the Hermitian far-field objective matrix
`Q = G' W G`. It stores the radiation-vector matrix and applies `Q*x`
without forming the dense `N x N` matrix.
"""
struct FarFieldQMatrix <: AbstractMatrix{ComplexF64}
    G_mat::Matrix{ComplexF64}
    weights::Vector{Float64}
    pol::Matrix{ComplexF64}
    mask::Union{Nothing,BitVector}
    N::Int
    work::Vector{ComplexF64}
    work_lock::ReentrantLock
end

FarFieldQMatrix(G_mat::Matrix{ComplexF64},
                weights::Vector{Float64},
                pol::Matrix{ComplexF64},
                mask::Union{Nothing,BitVector},
                N::Int) =
    FarFieldQMatrix(
        G_mat, weights, pol, mask, N,
        zeros(ComplexF64, N), ReentrantLock())

# A primitive real term in G'WGx contains up to six finite Float64 factors.
# Their exact product needs at most 12588 bits; the number of addressable
# quadrature/input terms adds fewer than 70 bits. Keep the remainder as guard
# precision in the exceptional path.
const _FARFIELD_Q_FALLBACK_PRECISION = 12800
const _FARFIELD_Q_SCALED_OUTPUT_FALLBACK_PRECISION = 4352

@inline function _farfield_q_projection_bigfloat(
        Q::FarFieldQMatrix,
        q::Int,
        column::Int)
    offset = 3 * (q - 1)
    total = zero(Complex{BigFloat})
    @inbounds for component in 1:3
        total += conj(Complex{BigFloat}(Q.pol[component, q])) *
                 Complex{BigFloat}(Q.G_mat[offset + component, column])
    end
    return total
end

@noinline function _farfield_q_product_bigfloat!(
        result::Vector{ComplexF64},
        Q::FarFieldQMatrix,
        input::AbstractVector{ComplexF64})
    return setprecision(BigFloat, _FARFIELD_Q_FALLBACK_PRECISION) do
        quadrature_count = length(Q.weights)
        if Q.N <= quadrature_count
            totals = fill(zero(Complex{BigFloat}), Q.N)
            @inbounds for q in 1:quadrature_count
                Q.mask !== nothing && !Q.mask[q] && continue
                projected_input = zero(Complex{BigFloat})
                for n in 1:Q.N
                    projected_input +=
                        _farfield_q_projection_bigfloat(Q, q, n) *
                        Complex{BigFloat}(input[n])
                end
                weight = BigFloat(Q.weights[q])
                for m in 1:Q.N
                    totals[m] += weight *
                        conj(_farfield_q_projection_bigfloat(Q, q, m)) *
                        projected_input
                end
            end
            @inbounds for m in 1:Q.N
                converted = ComplexF64(totals[m])
                isfinite(converted) ||
                    throw(OverflowError(
                        "FarFieldQMatrix product is outside the " *
                        "representable ComplexF64 range at row $m."))
                result[m] = converted
            end
        else
            projected_inputs = fill(
                zero(Complex{BigFloat}), quadrature_count)
            @inbounds for q in 1:quadrature_count
                Q.mask !== nothing && !Q.mask[q] && continue
                for n in 1:Q.N
                    projected_inputs[q] +=
                        _farfield_q_projection_bigfloat(Q, q, n) *
                        Complex{BigFloat}(input[n])
                end
            end
            @inbounds for m in 1:Q.N
                total = zero(Complex{BigFloat})
                for q in 1:quadrature_count
                    Q.mask !== nothing && !Q.mask[q] && continue
                    total += BigFloat(Q.weights[q]) *
                        conj(_farfield_q_projection_bigfloat(Q, q, m)) *
                        projected_inputs[q]
                end
                converted = ComplexF64(total)
                isfinite(converted) ||
                    throw(OverflowError(
                        "FarFieldQMatrix product is outside the " *
                        "representable ComplexF64 range at row $m."))
                result[m] = converted
            end
        end
        return result
    end
end

function _farfield_q_product!(
        result::Vector{ComplexF64},
        Q::FarFieldQMatrix,
        input::AbstractVector{ComplexF64})
    fill!(result, zero(ComplexF64))
    quadrature_count = length(Q.weights)
    @inbounds for q in 1:quadrature_count
        Q.mask !== nothing && !Q.mask[q] && continue
        offset = 3 * (q - 1)
        p = SVector{3,ComplexF64}(
            Q.pol[1, q], Q.pol[2, q], Q.pol[3, q])
        projected_input = zero(ComplexF64)
        for n in 1:Q.N
            gn = SVector{3,ComplexF64}(
                Q.G_mat[offset + 1, n],
                Q.G_mat[offset + 2, n],
                Q.G_mat[offset + 3, n])
            projected_input += dot(p, gn) * input[n]
        end
        isfinite(projected_input) ||
            return _farfield_q_product_bigfloat!(result, Q, input)

        weight = Q.weights[q]
        for m in 1:Q.N
            gm = SVector{3,ComplexF64}(
                Q.G_mat[offset + 1, m],
                Q.G_mat[offset + 2, m],
                Q.G_mat[offset + 3, m])
            result[m] += weight * conj(dot(p, gm)) * projected_input
        end
    end
    all(isfinite, result) ||
        return _farfield_q_product_bigfloat!(result, Q, input)
    return result
end

@noinline function _farfield_q_scaled_output_bigfloat(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    return setprecision(
            BigFloat, _FARFIELD_Q_SCALED_OUTPUT_FALLBACK_PRECISION) do
        total = Complex{BigFloat}(alpha_scale) *
                Complex{BigFloat}(value)
        if !overwrite
            total += Complex{BigFloat}(beta_scale) *
                     Complex{BigFloat}(previous)
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "FarFieldQMatrix scaled output is outside the " *
                "representable ComplexF64 range at row $row."))
        return converted
    end
end

@inline function _farfield_q_scaled_output(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    alpha_term = alpha_scale * value
    if overwrite
        converted = ComplexF64(alpha_term)
        return isfinite(converted) ? converted :
               _farfield_q_scaled_output_bigfloat(
                   value, previous, alpha_scale, beta_scale, true, row)
    end
    beta_term = beta_scale * previous
    combined = alpha_term + beta_term
    magnitude_sum =
        max(abs(real(alpha_term)), abs(imag(alpha_term))) +
        max(abs(real(beta_term)), abs(imag(beta_term)))
    converted = ComplexF64(combined)
    if isfinite(converted) && isfinite(magnitude_sum)
        return converted
    end
    return _farfield_q_scaled_output_bigfloat(
        value, previous, alpha_scale, beta_scale, false, row)
end

struct SumQMatrix{A<:AbstractMatrix{ComplexF64},B<:AbstractMatrix{ComplexF64}} <: AbstractMatrix{ComplexF64}
    A::A
    B::B
end

function sum_q_matrix(A::AbstractMatrix{ComplexF64}, B::AbstractMatrix{ComplexF64})
    size(A) == size(B) || throw(DimensionMismatch("summed Q matrices must have the same size"))
    return SumQMatrix{typeof(A),typeof(B)}(A, B)
end

Base.size(Q::SumQMatrix) = size(Q.A)
Base.eltype(::SumQMatrix) = ComplexF64
Base.getindex(Q::SumQMatrix, i::Int, j::Int) = Q.A[i, j] + Q.B[i, j]

function LinearAlgebra.mul!(result::AbstractVector{ComplexF64},
                            Q::SumQMatrix,
                            x::AbstractVector{ComplexF64})
    return LinearAlgebra.mul!(
        result, Q, x, one(ComplexF64), zero(ComplexF64))
end

function LinearAlgebra.mul!(result::AbstractVector{ComplexF64},
                            Q::SumQMatrix,
                            x::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    length(result) == size(Q, 1) || throw(DimensionMismatch("result length $(length(result)) != $(size(Q, 1))"))
    length(x) == size(Q, 2) || throw(DimensionMismatch("input length $(length(x)) != $(size(Q, 2))"))
    if iszero(alpha_scale)
        if iszero(beta_scale)
            fill!(result, zero(ComplexF64))
        elseif beta_scale != one(beta_scale)
            result .*= beta_scale
        end
        return result
    end
    xread = Base.mightalias(result, x) ? copy(x) : x
    mul!(result, Q.A, xread, alpha_scale, beta_scale)
    mul!(result, Q.B, xread, alpha_scale, one(ComplexF64))
    return result
end

Base.size(Q::FarFieldQMatrix) = (Q.N, Q.N)
Base.eltype(::FarFieldQMatrix) = ComplexF64

@noinline function _farfield_q_entry_bigfloat(
        Q::FarFieldQMatrix,
        m::Int,
        n::Int)
    return setprecision(BigFloat, _FARFIELD_Q_FALLBACK_PRECISION) do
        total = zero(Complex{BigFloat})
        @inbounds for q in eachindex(Q.weights)
            Q.mask !== nothing && !Q.mask[q] && continue
            total += BigFloat(Q.weights[q]) *
                conj(_farfield_q_projection_bigfloat(Q, q, m)) *
                _farfield_q_projection_bigfloat(Q, q, n)
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "FarFieldQMatrix entry is outside the representable " *
                "ComplexF64 range at index ($m, $n)."))
        return converted
    end
end

function _validate_q_inputs(G_mat::Matrix{ComplexF64}, grid::SphGrid,
                            pol::Matrix{ComplexF64}, mask)
    NΩ = _validate_sph_grid(grid)
    size(G_mat, 1) == 3 * NΩ ||
        throw(DimensionMismatch(
            "G_mat has $(size(G_mat, 1)) rows, expected $(3 * NΩ)"))
    size(pol) == (3, NΩ) ||
        throw(DimensionMismatch(
            "pol has size $(size(pol)), expected (3, $NΩ)"))
    all(isfinite, G_mat) ||
        throw(ArgumentError("G_mat must contain only finite values"))
    all(isfinite, pol) ||
        throw(ArgumentError("pol must contain only finite values"))
    if mask !== nothing
        mask isa AbstractVector{Bool} ||
            throw(ArgumentError("mask must be a boolean vector"))
        length(mask) == NΩ ||
            throw(DimensionMismatch(
                "mask length $(length(mask)) != $NΩ"))
    end
    return NΩ, size(G_mat, 2)
end

function Base.getindex(Q::FarFieldQMatrix, m::Int, n::Int)
    1 <= m <= Q.N || throw(BoundsError(Q, (m, n)))
    1 <= n <= Q.N || throw(BoundsError(Q, (m, n)))
    NΩ = length(Q.weights)
    val = zero(ComplexF64)
    @inbounds for q in 1:NΩ
        if Q.mask !== nothing && !Q.mask[q]
            continue
        end
        idx = 3 * (q - 1)
        p = SVector{3,ComplexF64}(
            Q.pol[1, q], Q.pol[2, q], Q.pol[3, q])
        gm = SVector{3,ComplexF64}(Q.G_mat[idx+1, m], Q.G_mat[idx+2, m], Q.G_mat[idx+3, m])
        gn = SVector{3,ComplexF64}(Q.G_mat[idx+1, n], Q.G_mat[idx+2, n], Q.G_mat[idx+3, n])
        val += Q.weights[q] * conj(dot(p, gm)) * dot(p, gn)
    end
    return isfinite(val) ? val : _farfield_q_entry_bigfloat(Q, m, n)
end

function LinearAlgebra.mul!(result::AbstractVector{ComplexF64},
                            Q::FarFieldQMatrix,
                            I_coeffs::AbstractVector{ComplexF64})
    return LinearAlgebra.mul!(
        result, Q, I_coeffs, one(ComplexF64), zero(ComplexF64))
end

function LinearAlgebra.mul!(result::AbstractVector{ComplexF64},
                            Q::FarFieldQMatrix,
                            I_coeffs::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    length(result) == Q.N || throw(DimensionMismatch("result length $(length(result)) != $(Q.N)"))
    length(I_coeffs) == Q.N || throw(DimensionMismatch("input length $(length(I_coeffs)) != $(Q.N)"))
    if iszero(alpha_scale)
        if iszero(beta_scale)
            fill!(result, zero(ComplexF64))
        elseif beta_scale != one(beta_scale)
            @inbounds for m in eachindex(result)
                result[m] *= beta_scale
            end
        end
        return result
    end
    input_read = Base.mightalias(result, I_coeffs) ? copy(I_coeffs) : I_coeffs
    overwrite = iszero(beta_scale)
    lock(Q.work_lock)
    try
        _farfield_q_product!(Q.work, Q, input_read)
        @inbounds for m in eachindex(result)
            result[m] = _farfield_q_scaled_output(
                Q.work[m], result[m], alpha_scale, beta_scale,
                overwrite, m)
        end
    finally
        unlock(Q.work_lock)
    end
    return result
end

"""
    build_Q(G_mat, grid, pol; mask=nothing)

Build the Hermitian PSD matrix Q from radiation vectors and polarization.

  G_mat: (3*NΩ, N) radiation vector matrix
  grid:  SphGrid with quadrature weights
  pol:   (3, NΩ) complex polarization vectors (unit, transverse to r̂)
  mask:  optional BitVector of length NΩ selecting target directions

Returns Q ∈ C^{N×N}, Hermitian positive semidefinite.
"""
function build_Q(G_mat::Matrix{ComplexF64}, grid::SphGrid,
                 pol::Matrix{ComplexF64}; mask=nothing)
    NΩ, N = _validate_q_inputs(G_mat, grid, pol, mask)

    # Compute scalar projections: y_q_n = p†(r̂_q) · g_n(r̂_q)
    # y is (NΩ, N)
    y = zeros(ComplexF64, NΩ, N)
    for q in 1:NΩ
        if mask !== nothing && !mask[q]
            continue
        end
        p = SVector{3,ComplexF64}(pol[1, q], pol[2, q], pol[3, q])  # avoid slice alloc
        idx = 3 * (q - 1)
        for n in 1:N
            gn = SVector{3,ComplexF64}(G_mat[idx+1, n], G_mat[idx+2, n], G_mat[idx+3, n])
            y[q, n] = dot(p, gn)
        end
    end

    # Q_mn = Σ_q w_q conj(y_qm) y_qn
    Q = zeros(ComplexF64, N, N)
    for q in 1:NΩ
        if mask !== nothing && !mask[q]
            continue
        end
        wq = grid.w[q]
        for m in 1:N
            ym = conj(y[q, m])
            for n in 1:N
                Q[m, n] += wq * ym * y[q, n]
            end
        end
    end

    return Q
end

"""
    build_Q_operator(G_mat, grid, pol; mask=nothing)

Build a matrix-free far-field objective operator with the same action as
`build_Q(G_mat, grid, pol; mask)`, but without forming the dense matrix.
"""
function build_Q_operator(G_mat::Matrix{ComplexF64}, grid::SphGrid,
                          pol::Matrix{ComplexF64}; mask=nothing)
    _, N = _validate_q_inputs(G_mat, grid, pol, mask)
    mask_copy = mask === nothing ? nothing : BitVector(mask)
    return FarFieldQMatrix(G_mat, copy(grid.w), pol, mask_copy, N)
end

"""
    apply_Q(G_mat, grid, pol, I_coeffs; mask=nothing)

Apply Q*I without forming Q explicitly.
Returns Q*I ∈ C^N.
"""
function apply_Q(G_mat::Matrix{ComplexF64}, grid::SphGrid,
                 pol::Matrix{ComplexF64}, I_coeffs::Vector{ComplexF64};
                 mask=nothing)
    NΩ, N = _validate_q_inputs(G_mat, grid, pol, mask)
    length(I_coeffs) == N ||
        throw(DimensionMismatch(
            "I_coeffs length $(length(I_coeffs)) != $N"))

    result = zeros(ComplexF64, N)
    for q in 1:NΩ
        if mask !== nothing && !mask[q]
            continue
        end
        p = SVector{3,ComplexF64}(
            pol[1, q], pol[2, q], pol[3, q])
        wq = grid.w[q]

        # Compute y_q = p† · E∞(r̂_q) = Σ_n I_n (p† · g_n)
        yq = zero(ComplexF64)
        for n in 1:N
            idx = 3 * (q - 1)
            gn = SVector{3,ComplexF64}(G_mat[idx+1, n], G_mat[idx+2, n], G_mat[idx+3, n])
            yq += dot(p, gn) * I_coeffs[n]
        end

        # Accumulate: (Q*I)_m += w_q conj(p†·g_m) y_q
        for m in 1:N
            idx = 3 * (q - 1)
            gm = SVector{3,ComplexF64}(G_mat[idx+1, m], G_mat[idx+2, m], G_mat[idx+3, m])
            result[m] += wq * conj(dot(p, gm)) * yq
        end
    end

    return result
end

"""
    pol_linear_x(grid)

Generate x-polarized far-field polarization vectors (θ̂ component for
broadside radiation along z).
Returns (3, NΩ) complex matrix.
"""
function pol_linear_x(grid::SphGrid)
    NΩ = _validate_sph_grid(grid)
    pol = zeros(ComplexF64, 3, NΩ)
    for q in 1:NΩ
        θ = grid.theta[q]
        φ = grid.phi[q]
        # θ̂ unit vector
        theta_hat = Vec3(cos(θ) * cos(φ), cos(θ) * sin(φ), -sin(θ))
        pol[:, q] = theta_hat
    end
    return pol
end

"""
    pol_linear_y(grid)

Generate the orthogonal far-field polarization vectors (`φ̂` component), which
correspond to y-polarized broadside radiation and the TE/s-polarized basis for
the common `φ = 0` incidence plane in periodic workflows.
Returns `(3, NΩ)` complex matrix.
"""
function pol_linear_y(grid::SphGrid)
    NΩ = _validate_sph_grid(grid)
    pol = zeros(ComplexF64, 3, NΩ)
    for q in 1:NΩ
        φ = grid.phi[q]
        phi_hat = Vec3(-sin(φ), cos(φ), 0.0)
        pol[:, q] = phi_hat
    end
    return pol
end

"""
    cap_mask(grid; theta_max=π/18)

Create a mask selecting directions within a cone of half-angle θ_max
around the z-axis (broadside).
"""
function cap_mask(grid::SphGrid; theta_max=π/18)
    NΩ = _validate_sph_grid(grid)
    isfinite(theta_max) && 0 <= theta_max <= π ||
        throw(ArgumentError(
            "theta_max must be finite and lie in [0, π], got $theta_max"))
    mask = BitVector(undef, NΩ)
    @inbounds for q in 1:NΩ
        mask[q] = grid.theta[q] <= theta_max
    end
    return mask
end

"""
    direction_mask(grid, direction; half_angle=π/18)

Create a mask selecting directions within a cone of `half_angle` (radians)
around an arbitrary `direction` vector. Generalizes `cap_mask` to any direction.

# Example: backscatter mask for incidence from +z
```julia
mask = direction_mask(grid, Vec3(0,0,-1); half_angle=10*π/180)
```
"""
function direction_mask(grid::SphGrid, direction::Vec3; half_angle::Real=π/18)
    NΩ = _validate_sph_grid(grid)
    isfinite(half_angle) && 0 <= half_angle <= π ||
        throw(ArgumentError(
            "half_angle must be finite and lie in [0, π], got $half_angle"))
    all(isfinite, direction) ||
        throw(ArgumentError("direction components must be finite"))
    direction_norm = norm(direction)
    isfinite(direction_norm) && direction_norm > 0 ||
        throw(ArgumentError("direction must have a finite, nonzero norm"))

    d = direction / direction_norm
    threshold = cos(half_angle)
    mask = BitVector(undef, NΩ)
    @inbounds for q in 1:NΩ
        mask[q] = grid.rhat[1, q] * d[1] +
                  grid.rhat[2, q] * d[2] +
                  grid.rhat[3, q] * d[3] >= threshold
    end
    return mask
end
