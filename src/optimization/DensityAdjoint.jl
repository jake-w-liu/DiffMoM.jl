# DensityAdjoint.jl — Adjoint sensitivity for density-based topology optimization
#
# Extends the existing adjoint framework (Adjoint.jl) for per-triangle density variables.
#
# Given:
#   Z(ρ̄) I = V,  J = f(I) = Re(I† Q I)
#
# Adjoint equation:
#   Z† λ = Q I
#
# Gradient w.r.t. projected densities:
#   ∂J/∂ρ̄_t = -2 Re{ λ† (∂Z/∂ρ̄_t) I }
#            = -2 Re{ λ† (-p ρ̄_t^(p-1) Z_max M_t) I }
#            = 2p ρ̄_t^(p-1) Re{ Z_max λ† M_t I }
#
# Then chain rule through filtering/projection gives ∂J/∂ρ.

export gradient_density, gradient_density_full

# The exceptional path combines the exact Float64 bilinear accumulator with
# Z_max and the SIMP derivative scale. Five finite Float64 factors plus the
# full stored-entry sum need fewer than 10560 bits; retain a guard margin for
# the non-integer density power.
const _IEEE_DENSITY_GRADIENT_FALLBACK_PRECISION = 11008

@inline function _density_weighted_bilinear_requires_fallback(
    impedance::Number,
    bilinear::Number,
    weighted_value::Real,
)
    impedance_real = Float64(real(impedance))
    impedance_imag = Float64(imag(impedance))
    bilinear_real = Float64(real(bilinear))
    bilinear_imag = Float64(imag(bilinear))
    magnitude = abs(impedance_real * bilinear_real) +
                abs(impedance_imag * bilinear_imag)
    error_factor = _ieee_product_error_factor(Float64, 2, 1)
    return _ieee_product_component_is_suspicious(
        weighted_value, magnitude, error_factor)
end

@noinline function _density_gradient_bigfloat(
    matrix::AbstractMatrix,
    I::Vector{<:Number},
    lambda::Vector{<:Number},
    rho_bar::Real,
    config::DensityConfig,
)
    return setprecision(
            BigFloat, _IEEE_DENSITY_GRADIENT_FALLBACK_PRECISION) do
        bilinear_real = _accumulate_bilinear_component_bigfloat(
            lambda, matrix, I, Val(:real))
        bilinear_imag = _accumulate_bilinear_component_bigfloat(
            lambda, matrix, I, Val(:imag))
        impedance_real = BigFloat(real(config.Z_max))
        impedance_imag = BigFloat(imag(config.Z_max))
        weighted_bilinear = impedance_real * bilinear_real -
                            impedance_imag * bilinear_imag
        p = BigFloat(config.p)
        density_power = isone(config.p) ?
                        one(BigFloat) :
                        BigFloat(rho_bar)^(p - 1)
        value = 2 * p * density_power * weighted_bilinear
        converted = Float64(value)
        isfinite(converted) ||
            throw(OverflowError(
                "density gradient is outside the representable Float64 range"))
        return converted
    end
end

"""
    gradient_density(Mt, I, lambda, rho_bar, config)

Compute the adjoint gradient w.r.t. projected densities ρ̄:

  g[t] = ∂J/∂ρ̄_t = -2 Re{ λ† (∂Z/∂ρ̄_t) I }

where ∂Z/∂ρ̄_t = -p ρ̄_t^(p-1) Z_max M_t.

Returns g ∈ R^{Nt}.
"""
function gradient_density(Mt::Vector{<:AbstractMatrix},
                          I::Vector{<:Number},
                          lambda::Vector{<:Number},
                          rho_bar::AbstractVector{<:Real},
                          config::DensityConfig)
    Nt = length(Mt)
    matrix_size = _validate_density_mass_inputs(Mt, rho_bar)
    length(I) == matrix_size[2] ||
        throw(DimensionMismatch(
            "I length $(length(I)) != $(matrix_size[2])"))
    length(lambda) == matrix_size[1] ||
        throw(DimensionMismatch(
            "lambda length $(length(lambda)) != $(matrix_size[1])"))
    all(isfinite, I) ||
        throw(ArgumentError("I must contain only finite values"))
    all(isfinite, lambda) ||
        throw(ArgumentError("lambda must contain only finite values"))
    @inbounds for t in eachindex(Mt)
        _validate_known_matrix_entries(Mt[t], "triangle mass matrix")
    end
    g = zeros(Float64, Nt)

    for t in 1:Nt
        # ∂Z/∂ρ̄_t = -p * ρ̄_t^(p-1) * Z_max * M_t
        # g[t] = -2 Re{ λ† (∂Z/∂ρ̄_t) I }
        #      = 2p ρ̄_t^(p-1) Re{ Z_max λ† M_t I }
        density_power = rho_bar[t]^(config.p - 1)
        lMI = _dot_left_matrix_right(lambda, Mt[t], I)
        weighted_bilinear = real(config.Z_max * lMI)
        needs_fallback =
            _ieee_dense_extreme_factor(config.Z_max, Float64) ||
            _ieee_dense_extreme_factor(config.p, Float64) ||
            _ieee_dense_extreme_factor(density_power, Float64) ||
            (!iszero(real(config.Z_max)) &&
             _ieee_bilinear_component_requires_fallback(
                 lambda, Mt[t], I, Val(:real), real(lMI), Float64)) ||
            (!iszero(imag(config.Z_max)) &&
             _ieee_bilinear_component_requires_fallback(
                 lambda, Mt[t], I, Val(:imag), imag(lMI), Float64)) ||
            _density_weighted_bilinear_requires_fallback(
                config.Z_max, lMI, weighted_bilinear)
        if needs_fallback
            g[t] = _density_gradient_bigfloat(
                Mt[t], I, lambda, rho_bar[t], config)
            continue
        end
        value = 2 * config.p * density_power * weighted_bilinear
        g[t] = isfinite(value) ? value : _density_gradient_bigfloat(
            Mt[t], I, lambda, rho_bar[t], config)
    end

    all(isfinite, g) ||
        error("gradient_density produced non-finite values")
    return g
end

"""
    gradient_density_full(Mt, I, lambda, rho_tilde, rho_bar, config,
                          W, w_sum, beta; eta=0.5)

Full gradient computation with chain rule through filtering and projection:

  1. g_ρ̄ = gradient_density(Mt, I, lambda, rho_bar, config)
  2. g_ρ  = chain_rule(g_ρ̄, rho_tilde, W, w_sum, beta, eta)

Returns the gradient w.r.t. the raw design variables ρ.
"""
function gradient_density_full(Mt::Vector{<:AbstractMatrix},
                               I::Vector{<:Number},
                               lambda::Vector{<:Number},
                               rho_tilde::AbstractVector{<:Real},
                               rho_bar::AbstractVector{<:Real},
                               config::DensityConfig,
                               W::AbstractSparseMatrix,
                               w_sum::AbstractVector,
                               beta::Real; eta::Real=0.5)
    _validate_density_values(rho_tilde, length(Mt), "rho_tilde")

    # Step 1: gradient w.r.t. projected densities
    g_rho_bar = gradient_density(Mt, I, lambda, rho_bar, config)

    # Step 2: chain rule through Heaviside + filter
    g_rho = gradient_chain_rule(g_rho_bar, rho_tilde, W, w_sum, beta, eta)

    return g_rho
end
