# CompositeOperator.jl — Impedance-loaded operator for matrix-free optimization
#
# Wraps any AbstractMatrix base operator (MLFMA, ACA, dense) with impedance
# perturbation:  Z(θ) = Z_base - Σ_p θ_p M_p   (resistive)
#            or  Z(θ) = Z_base - Σ_p (iθ_p) M_p (reactive)
#
# This enables GMRES-based optimization with fast operators without forming
# the full dense matrix.

export ImpedanceLoadedOperator, ImpedanceLoadedAdjointOperator

"""
    ImpedanceLoadedOperator(Z_base, Mp, theta, reactive=false)

Matrix-free operator representing Z(θ) = Z_base + Z_imp(θ) where
Z_imp = -Σ_p coeff_p * M_p.

Supports `mul!`, `adjoint`, and `size` for use with Krylov.jl GMRES.

# Arguments
- `Z_base`: any `AbstractMatrix{ComplexF64}` (MLFMAOperator, ACAOperator, dense Matrix)
- `Mp`: vector of sparse patch mass matrices
- `theta`: current impedance parameter vector
- `reactive`: if true, Z_imp = -Σ (iθ_p) M_p; otherwise Z_imp = -Σ θ_p M_p
"""
struct ImpedanceLoadedOperator{T<:AbstractMatrix{ComplexF64},
                                S<:AbstractMatrix} <: AbstractMatrix{ComplexF64}
    Z_base::T
    Mp::Vector{S}
    theta::Vector{Float64}
    reactive::Bool
end

function ImpedanceLoadedOperator(Z_base::AbstractMatrix{ComplexF64},
                                  Mp::Vector{<:AbstractMatrix},
                                  theta::Vector{Float64})
    return ImpedanceLoadedOperator{typeof(Z_base), eltype(Mp)}(Z_base, Mp, theta, false)
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

# ─── Forward matvec: y = Z(θ) * x ─────────────────────────────────

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            A::ImpedanceLoadedOperator,
                            x::AbstractVector{ComplexF64})
    mul!(y, A.Z_base, x)
    @inbounds for p in eachindex(A.theta)
        coeff = A.reactive ? (1im * A.theta[p]) : ComplexF64(A.theta[p])
        mul!(y, A.Mp[p], x, -coeff, one(ComplexF64))
    end
    return y
end

function Base.:*(A::ImpedanceLoadedOperator, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, Vector{ComplexF64}(x))
    return y
end

# ─── Adjoint matvec: y = Z(θ)' * x ────────────────────────────────

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            A::ImpedanceLoadedAdjointOperator,
                            x::AbstractVector{ComplexF64})
    mul!(y, adjoint(A.parent.Z_base), x)
    @inbounds for p in eachindex(A.parent.theta)
        # Conjugate of the coefficient for adjoint
        coeff = A.parent.reactive ? (-1im * A.parent.theta[p]) : ComplexF64(A.parent.theta[p])
        mul!(y, A.parent.Mp[p]', x, -coeff, one(ComplexF64))
    end
    return y
end

function Base.:*(A::ImpedanceLoadedAdjointOperator, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, Vector{ComplexF64}(x))
    return y
end
