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
    g = zeros(Float64, Nt)

    for t in 1:Nt
        # ∂Z/∂ρ̄_t = -p * ρ̄_t^(p-1) * Z_max * M_t
        # g[t] = -2 Re{ λ† (∂Z/∂ρ̄_t) I }
        #      = 2p ρ̄_t^(p-1) Re{ Z_max λ† M_t I }
        lMI = _dot_left_matrix_right(lambda, Mt[t], I)
        g[t] = 2 * config.p * rho_bar[t]^(config.p - 1) * real(config.Z_max * lMI)
    end

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
    # Step 1: gradient w.r.t. projected densities
    g_rho_bar = gradient_density(Mt, I, lambda, rho_bar, config)

    # Step 2: chain rule through Heaviside + filter
    g_rho = gradient_chain_rule(g_rho_bar, rho_tilde, W, w_sum, beta, eta)

    return g_rho
end
