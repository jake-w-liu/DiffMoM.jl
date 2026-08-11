# CompositeOperator.jl — Impedance-loaded operator for matrix-free optimization
#
# Wraps any AbstractMatrix base operator (MLFMA, ACA, dense) with impedance
# perturbation:  Z(θ) = Z_base - Σ_p θ_p M_p   (resistive)
#            or  Z(θ) = Z_base - Σ_p (iθ_p) M_p (reactive)
#
# This enables GMRES-based optimization with fast operators without forming
# the full dense matrix.

export ImpedanceLoadedOperator, ImpedanceLoadedAdjointOperator

# A primitive real term in α(Z_base - Σ θₚMₚ)x can contain four finite
# Float64 factors. Their exact binary terms can span more than 8,300 bit
# positions. The allocation-heavy path is reserved for exceptional exponent
# ranges; ordinary products use the reusable ComplexF64 workspaces below.
const _COMPOSITE_PRODUCT_FALLBACK_PRECISION = 8704
const _COMPOSITE_SCALED_OUTPUT_FALLBACK_PRECISION = 4352

"""
    ImpedanceLoadedOperator(Z_base, Mp, theta)
    ImpedanceLoadedOperator(Z_base, Mp, theta, reactive)

Matrix-free operator representing Z(θ) = Z_base + Z_imp(θ) where
Z_imp = -Σ_p coeff_p * M_p.

Supports `mul!`, `adjoint`, and `size` for use with Krylov.jl GMRES.

# Arguments
- `Z_base`: any `AbstractMatrix{ComplexF64}` (MLFMAOperator, ACAOperator, dense Matrix)
- `Mp`: vector of sparse patch mass matrices
- `theta`: current impedance parameter vector
- `reactive`: positional `Bool` flag (defaults to `false` in the 3-argument form);
  if `true`, Z_imp = -Σ (iθ_p) M_p; otherwise Z_imp = -Σ θ_p M_p
"""
struct ImpedanceLoadedOperator{T<:AbstractMatrix{ComplexF64},
                                S<:AbstractMatrix} <: AbstractMatrix{ComplexF64}
    Z_base::T
    Mp::Vector{S}
    theta::Vector{Float64}
    reactive::Bool
    work_total::Vector{ComplexF64}
    work_term::Vector{ComplexF64}
    work_lock::ReentrantLock
    function ImpedanceLoadedOperator{T,S}(
        Z_base::T,
        Mp::Vector{S},
        theta::Vector{Float64},
        reactive::Bool,
    ) where {T<:AbstractMatrix{ComplexF64},S<:AbstractMatrix}
        _validate_impedance_inputs(Mp, theta, size(Z_base))
        row_count = size(Z_base, 1)
        return new{T,S}(
            Z_base, Mp, theta, reactive,
            zeros(ComplexF64, row_count),
            zeros(ComplexF64, row_count),
            ReentrantLock())
    end
end

function ImpedanceLoadedOperator(
    Z_base::T,
    Mp::Vector{S},
    theta::Vector{Float64},
    reactive::Bool,
) where {T<:AbstractMatrix{ComplexF64},S<:AbstractMatrix}
    return ImpedanceLoadedOperator{T,S}(Z_base, Mp, theta, reactive)
end

function ImpedanceLoadedOperator(Z_base::AbstractMatrix{ComplexF64},
                                  Mp::Vector{<:AbstractMatrix},
                                  theta::Vector{Float64})
    return ImpedanceLoadedOperator(Z_base, Mp, theta, false)
end

struct ImpedanceLoadedAdjointOperator{T<:AbstractMatrix{ComplexF64},
                                       S<:AbstractMatrix} <: AbstractMatrix{ComplexF64}
    parent::ImpedanceLoadedOperator{T,S}
end

# ─── AbstractMatrix interface ──────────────────────────────────────

Base.size(A::ImpedanceLoadedOperator) = size(A.Z_base)
Base.size(A::ImpedanceLoadedOperator, d::Int) = size(A.Z_base, d)
Base.eltype(::ImpedanceLoadedOperator) = ComplexF64

Base.size(A::ImpedanceLoadedAdjointOperator) = size(A.parent)
Base.size(A::ImpedanceLoadedAdjointOperator, d::Int) = size(A.parent, d)
Base.eltype(::ImpedanceLoadedAdjointOperator) = ComplexF64

LinearAlgebra.adjoint(A::ImpedanceLoadedOperator) = ImpedanceLoadedAdjointOperator(A)
LinearAlgebra.adjoint(A::ImpedanceLoadedAdjointOperator) = A.parent

@inline function _composite_patch_factor(
        A::ImpedanceLoadedOperator,
        patch::Int,
        ::Val{ADJOINT}) where {ADJOINT}
    theta = A.theta[patch]
    if A.reactive
        return ComplexF64(0.0, ADJOINT ? theta : -theta)
    end
    return ComplexF64(-theta, 0.0)
end

@inline _composite_base_entry(
    A::ImpedanceLoadedOperator, row::Int, column::Int, ::Val{false}) =
    A.Z_base[row, column]

@inline _composite_base_entry(
    A::ImpedanceLoadedOperator, row::Int, column::Int, ::Val{true}) =
    conj(A.Z_base[column, row])

@inline _composite_mass_entry(
    A::ImpedanceLoadedOperator,
    patch::Int,
    row::Int,
    column::Int,
    ::Val{false}) = A.Mp[patch][row, column]

@inline _composite_mass_entry(
    A::ImpedanceLoadedOperator,
    patch::Int,
    row::Int,
    column::Int,
    ::Val{true}) = conj(A.Mp[patch][column, row])

@noinline function _composite_entry_bigfloat(
        A::ImpedanceLoadedOperator,
        row::Int,
        column::Int,
        adjoint_mode::Val{ADJOINT}) where {ADJOINT}
    return setprecision(
            BigFloat, _COMPOSITE_PRODUCT_FALLBACK_PRECISION) do
        total = Complex{BigFloat}(
            _composite_base_entry(A, row, column, adjoint_mode))
        @inbounds for patch in eachindex(A.theta)
            iszero(A.theta[patch]) && continue
            total +=
                Complex{BigFloat}(
                    _composite_patch_factor(A, patch, adjoint_mode)) *
                Complex{BigFloat}(
                    _composite_mass_entry(
                        A, patch, row, column, adjoint_mode))
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "ImpedanceLoadedOperator entry is outside the " *
                "representable ComplexF64 range at index " *
                "($row, $column)."))
        return converted
    end
end

function _composite_entry(
        A::ImpedanceLoadedOperator,
        row::Int,
        column::Int,
        adjoint_mode::Val{ADJOINT}) where {ADJOINT}
    length(A.Mp) == length(A.theta) ||
        throw(DimensionMismatch(
            "Mp length $(length(A.Mp)) must match " *
            "theta length $(length(A.theta))"))
    _validate_impedance_coefficients(A.theta)

    total = ComplexF64(
        _composite_base_entry(A, row, column, adjoint_mode))
    @inbounds for patch in eachindex(A.theta)
        iszero(A.theta[patch]) && continue
        contribution =
            _composite_patch_factor(A, patch, adjoint_mode) *
            _composite_mass_entry(
                A, patch, row, column, adjoint_mode)
        combined = total + contribution
        real_magnitude = abs(real(total)) + abs(real(contribution))
        imag_magnitude = abs(imag(total)) + abs(imag(contribution))
        if !isfinite(combined) ||
           !isfinite(real_magnitude) ||
           !isfinite(imag_magnitude)
            return _composite_entry_bigfloat(
                A, row, column, adjoint_mode)
        end
        total = combined
    end
    return total
end

Base.getindex(A::ImpedanceLoadedOperator, row::Int, column::Int) =
    _composite_entry(A, row, column, Val(false))

Base.getindex(A::ImpedanceLoadedAdjointOperator, row::Int, column::Int) =
    _composite_entry(A.parent, row, column, Val(true))

@noinline function _composite_scaled_output_bigfloat(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    return setprecision(
            BigFloat, _COMPOSITE_SCALED_OUTPUT_FALLBACK_PRECISION) do
        total = Complex{BigFloat}(alpha_scale) *
                Complex{BigFloat}(value)
        if !overwrite
            total += Complex{BigFloat}(beta_scale) *
                     Complex{BigFloat}(previous)
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "ImpedanceLoadedOperator scaled output is outside the " *
                "representable ComplexF64 range at row $row."))
        return converted
    end
end

@inline function _composite_scaled_output(
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
               _composite_scaled_output_bigfloat(
                   value, previous, alpha_scale, beta_scale, true, row)
    end

    beta_term = beta_scale * previous
    combined = alpha_term + beta_term
    real_magnitude = abs(real(alpha_term)) + abs(real(beta_term))
    imag_magnitude = abs(imag(alpha_term)) + abs(imag(beta_term))
    converted = ComplexF64(combined)
    if isfinite(converted) &&
       isfinite(real_magnitude) &&
       isfinite(imag_magnitude)
        return converted
    end
    return _composite_scaled_output_bigfloat(
        value, previous, alpha_scale, beta_scale, false, row)
end

function _composite_unscaled_product!(
        A::ImpedanceLoadedOperator,
        input::AbstractVector{ComplexF64},
        adjoint_mode::Val{ADJOINT}) where {ADJOINT}
    try
        if ADJOINT
            mul!(A.work_total, adjoint(A.Z_base), input)
        else
            mul!(A.work_total, A.Z_base, input)
        end
        all(isfinite, A.work_total) || return false

        @inbounds for patch in eachindex(A.theta)
            iszero(A.theta[patch]) && continue
            if ADJOINT
                mul!(A.work_term, adjoint(A.Mp[patch]), input)
            else
                mul!(A.work_term, A.Mp[patch], input)
            end
            factor = _composite_patch_factor(A, patch, adjoint_mode)
            for row in eachindex(A.work_total)
                contribution = factor * A.work_term[row]
                combined = A.work_total[row] + contribution
                real_magnitude =
                    abs(real(A.work_total[row])) + abs(real(contribution))
                imag_magnitude =
                    abs(imag(A.work_total[row])) + abs(imag(contribution))
                if !isfinite(combined) ||
                   !isfinite(real_magnitude) ||
                   !isfinite(imag_magnitude)
                    return false
                end
                A.work_total[row] = combined
            end
        end
    catch error
        error isa OverflowError || rethrow()
        return false
    end
    return true
end

@noinline function _composite_scaled_product_bigfloat!(
        output::Vector{ComplexF64},
        A::ImpedanceLoadedOperator,
        input::AbstractVector{ComplexF64},
        previous::AbstractVector{ComplexF64},
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        adjoint_mode::Val{ADJOINT}) where {ADJOINT}
    return setprecision(
            BigFloat, _COMPOSITE_PRODUCT_FALLBACK_PRECISION) do
        alpha_big = Complex{BigFloat}(alpha_scale)
        beta_big = overwrite ? zero(Complex{BigFloat}) :
                   Complex{BigFloat}(beta_scale)
        @inbounds for row in eachindex(output)
            product = zero(Complex{BigFloat})
            for column in axes(A, 2)
                product +=
                    Complex{BigFloat}(
                        _composite_base_entry(
                            A, row, column, adjoint_mode)) *
                    Complex{BigFloat}(input[column])
            end
            for patch in eachindex(A.theta)
                iszero(A.theta[patch]) && continue
                patch_product = zero(Complex{BigFloat})
                for column in axes(A, 2)
                    patch_product +=
                        Complex{BigFloat}(
                            _composite_mass_entry(
                                A, patch, row, column,
                                adjoint_mode)) *
                        Complex{BigFloat}(input[column])
                end
                product +=
                    Complex{BigFloat}(
                        _composite_patch_factor(
                            A, patch, adjoint_mode)) *
                    patch_product
            end

            total = alpha_big * product
            if !overwrite
                total += beta_big * Complex{BigFloat}(previous[row])
            end
            converted = ComplexF64(total)
            isfinite(converted) ||
                throw(OverflowError(
                    "ImpedanceLoadedOperator scaled product is outside " *
                    "the representable ComplexF64 range at row $row."))
            output[row] = converted
        end
        return output
    end
end

function _composite_mul!(
        y::AbstractVector{ComplexF64},
        A::ImpedanceLoadedOperator,
        x::AbstractVector{ComplexF64},
        alpha_scale::Number,
        beta_scale::Number,
        adjoint_mode::Val{ADJOINT}) where {ADJOINT}
    N = size(A, 1)
    length(x) == N ||
        throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N ||
        throw(DimensionMismatch("y length $(length(y)) != $N"))
    length(A.Mp) == length(A.theta) ||
        throw(DimensionMismatch(
            "Mp length $(length(A.Mp)) must match " *
            "theta length $(length(A.theta))"))
    if iszero(alpha_scale)
        if iszero(beta_scale)
            fill!(y, zero(ComplexF64))
        elseif beta_scale != one(beta_scale)
            lock(A.work_lock)
            try
                @inbounds for row in eachindex(y)
                    A.work_term[row] = _composite_scaled_output(
                        zero(ComplexF64), y[row], alpha_scale,
                        beta_scale, false, row)
                end
                copyto!(y, A.work_term)
            finally
                unlock(A.work_lock)
            end
        end
        return y
    end
    _validate_impedance_coefficients(A.theta)

    input_read = Base.mightalias(y, x) ? copy(x) : x
    overwrite = iszero(beta_scale)
    lock(A.work_lock)
    try
        if _composite_unscaled_product!(A, input_read, adjoint_mode)
            @inbounds for row in eachindex(A.work_total)
                A.work_term[row] = _composite_scaled_output(
                    A.work_total[row], y[row],
                    alpha_scale, beta_scale, overwrite, row)
            end
        else
            _composite_scaled_product_bigfloat!(
                A.work_term, A, input_read, y,
                alpha_scale, beta_scale, overwrite, adjoint_mode)
        end
        copyto!(y, A.work_term)
    finally
        unlock(A.work_lock)
    end
    return y
end

# ─── Forward matvec: y = Z(θ) * x ─────────────────────────────────

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            A::ImpedanceLoadedOperator,
                            x::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    return _composite_mul!(
        y, A, x, alpha_scale, beta_scale, Val(false))
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::ImpedanceLoadedOperator,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

function Base.:*(A::ImpedanceLoadedOperator, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, _complex_vector_input(x))
    return y
end

# ─── Adjoint matvec: y = Z(θ)' * x ────────────────────────────────

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            A::ImpedanceLoadedAdjointOperator,
                            x::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    return _composite_mul!(
        y, A.parent, x, alpha_scale, beta_scale, Val(true))
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::ImpedanceLoadedAdjointOperator,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

function Base.:*(A::ImpedanceLoadedAdjointOperator, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, _complex_vector_input(x))
    return y
end
