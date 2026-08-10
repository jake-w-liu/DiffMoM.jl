# Adjoint.jl — Adjoint solve and gradient evaluation
#
# Adjoint eq:  Z† λ = ∂Φ/∂I* = Q I
# Gradient:    ∂J/∂θ_p = -2 Re{ λ† (∂Z/∂θ_p) I }
#            = -2 Re{ λ† (-M_p) I }
#            = +2 Re{ λ† M_p I }

export solve_adjoint, solve_adjoint_rhs, gradient_impedance, compute_objective

"""
    compute_objective(I, Q)

Compute the quadratic objective J = Re(I† Q I).
"""
function compute_objective(I::Vector{<:Number}, Q::Matrix{<:Number})
    _validate_linear_system_inputs(Q, I, "quadratic objective")
    value = real(_dot_left_matrix_right(I, Q, I))
    isfinite(value) ||
        error("quadratic objective produced a non-finite value")
    return value
end

"""
    solve_adjoint(Z, Q, I; solver=:direct, preconditioner=nothing, gmres_precond_side=:left, gmres_tol=1e-8, gmres_maxiter=200, gmres_memory=20, check_gmres_convergence=true, check_true_residual=false, true_residual_factor=100.0)

Solve the adjoint system: Z† λ = Q I
Returns λ ∈ C^N.

When `solver=:gmres`, uses GMRES with the adjoint preconditioner P⁻ᴴ.
By default, an unconverged GMRES solve throws instead of returning an
unverified adjoint vector.
"""
function solve_adjoint(Z::AbstractMatrix{<:Number}, Q::Matrix{<:Number},
                       I::AbstractVector{<:Number};
                       solver::Symbol=:direct,
                       preconditioner=nothing,
                       gmres_precond_side::Symbol=:left,
                       gmres_tol::Float64=1e-8,
                       gmres_maxiter::Int=200,
                       gmres_memory::Int=20,
                       check_gmres_convergence::Bool=true,
                       check_true_residual::Bool=false,
                       true_residual_factor::Float64=100.0)
    solver in (:direct, :gmres) ||
        throw(ArgumentError(
            "Unknown solver: $solver (expected :direct or :gmres)"))
    N = _validate_linear_system_inputs(Z, I, "adjoint solve")
    size(Q) == (N, N) ||
        throw(DimensionMismatch(
            "adjoint objective matrix has size $(size(Q)), expected ($N, $N)"))
    _validate_known_matrix_entries(Q, "adjoint objective matrix")
    rhs = _finite_matrix_vector_product(Q, I, "adjoint RHS")
    if solver == :direct
        Z isa Matrix ||
            throw(ArgumentError(
                "Direct adjoint solver requires a dense Matrix; use solver=:gmres for operator-based systems."))
        return _solve_factored_linear_system(
            lu(adjoint(Z)), rhs, "direct adjoint solution")
    else
        x, stats = solve_gmres_adjoint(Z, rhs;
                                        preconditioner=preconditioner,
                                        precond_side=gmres_precond_side,
                                        tol=gmres_tol, maxiter=gmres_maxiter,
                                        memory=gmres_memory,
                                        check_gmres_convergence=check_gmres_convergence)
        check_true_residual &&
            _assert_true_residual(adjoint(Z), x, rhs, "adjoint";
                                  tol=gmres_tol,
                                  factor=true_residual_factor)
        return _assert_finite_linear_vector(x, "GMRES adjoint solution")
    end
end

"""
    solve_adjoint_rhs(Z, rhs; solver=:direct, preconditioner=nothing, gmres_precond_side=:left, gmres_tol=1e-8, gmres_maxiter=200, gmres_memory=20, check_gmres_convergence=true, check_true_residual=false, true_residual_factor=100.0)

Solve the adjoint system Z† λ = rhs where rhs is pre-computed.
Unlike `solve_adjoint(Z, Q, I)` which internally computes rhs = Q*I,
this accepts the RHS directly — useful when using `apply_Q` for matrix-free
Q application or for multi-angle objectives.
"""
function solve_adjoint_rhs(Z::AbstractMatrix{<:Number}, rhs::AbstractVector{<:Number};
                           solver::Symbol=:direct,
                           preconditioner=nothing,
                           gmres_precond_side::Symbol=:left,
                           gmres_tol::Float64=1e-8,
                           gmres_maxiter::Int=200,
                           gmres_memory::Int=20,
                           check_gmres_convergence::Bool=true,
                           check_true_residual::Bool=false,
                           true_residual_factor::Float64=100.0)
    solver in (:direct, :gmres) ||
        throw(ArgumentError(
            "Unknown solver: $solver (expected :direct or :gmres)"))
    if solver == :direct
        Z isa Matrix ||
            throw(ArgumentError(
                "Direct adjoint solver requires a dense Matrix; use solver=:gmres for operator-based systems."))
        _validate_linear_system_inputs(Z, rhs, "adjoint solve")
        complex_rhs = _as_complex_rhs(rhs)
        return _solve_factored_linear_system(
            lu(adjoint(Z)), complex_rhs, "direct adjoint solution")
    else
        x, stats = solve_gmres_adjoint(Z, _as_complex_rhs(rhs);
                                        preconditioner=preconditioner,
                                        precond_side=gmres_precond_side,
                                        tol=gmres_tol, maxiter=gmres_maxiter,
                                        memory=gmres_memory,
                                        check_gmres_convergence=check_gmres_convergence)
        check_true_residual &&
            _assert_true_residual(adjoint(Z), x, rhs, "adjoint";
                                  tol=gmres_tol,
                                  factor=true_residual_factor)
        return _assert_finite_linear_vector(x, "GMRES adjoint solution")
    end
end

"""
    gradient_impedance(Mp, I, lambda; reactive=false)

Compute the adjoint gradient for impedance parameters:
  g[p] = -2 Re{ λ† (∂Z/∂θ_p) I }

For resistive impedance (Z_s = θ_p, default):
  ∂Z/∂θ_p = -M_p  →  g[p] = +2 Re{ λ† M_p I }

For reactive impedance (Z_s = iθ_p, reactive=true):
  ∂Z/∂θ_p = -iM_p  →  g[p] = +2 Re{ i λ† M_p I } = -2 Im{ λ† M_p I }

Returns g ∈ R^P.
"""
function gradient_impedance(Mp::Vector{<:AbstractMatrix},
                            I::Vector{<:Number},
                            lambda::Vector{<:Number};
                            reactive::Bool=false)
    matrix_size = _validate_mass_matrix_sizes(Mp)
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
    @inbounds for p in eachindex(Mp)
        _validate_known_matrix_entries(Mp[p], "patch mass matrix")
    end
    P = length(Mp)
    g = zeros(Float64, P)
    for p in 1:P
        lMI = _dot_left_matrix_right(lambda, Mp[p], I)
        if reactive
            # ∂Z/∂θ_p = -iM_p
            # g[p] = -2 Re{ λ† (-iM_p) I } = 2 Re{ i λ† M_p I } = -2 Im{ λ† M_p I }
            g[p] = -2 * imag(lMI)
        else
            # ∂Z/∂θ_p = -M_p
            # g[p] = -2 Re{ λ† (-M_p) I } = 2 Re{ λ† M_p I }
            g[p] = 2 * real(lMI)
        end
    end
    all(isfinite, g) ||
        error("gradient_impedance produced non-finite values")
    return g
end
