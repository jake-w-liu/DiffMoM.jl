# Adjoint.jl — Adjoint solve and gradient evaluation
#
# Adjoint eq:  Z† λ = ∂Φ/∂I* = Q I
# Gradient:    ∂J/∂θ_p = -2 Re{ λ† (∂Z/∂θ_p) I }
#            = -2 Re{ λ† (-M_p) I }
#            = +2 Re{ λ† M_p I }

export solve_adjoint, solve_adjoint_rhs, gradient_impedance, compute_objective

# Expanding one component of left' * A * right produces at most four real
# triple products per stored matrix entry. Three finite Float64 coefficients
# need at most 6295 bits, and an addressable matrix has at most typemax(Int)
# entries. The extra 65 bits cover the full sum, with the remaining precision
# as guard margin.
const _IEEE_BILINEAR_FALLBACK_PRECISION = 6656

@inline function _bilinear_component_bigfloat(
    left_real::BigFloat,
    left_imag::BigFloat,
    matrix_real::BigFloat,
    matrix_imag::BigFloat,
    right_real::BigFloat,
    right_imag::BigFloat,
    ::Val{:real},
)
    product_real = matrix_real * right_real - matrix_imag * right_imag
    product_imag = matrix_real * right_imag + matrix_imag * right_real
    return left_real * product_real + left_imag * product_imag
end

@inline function _bilinear_component_bigfloat(
    left_real::BigFloat,
    left_imag::BigFloat,
    matrix_real::BigFloat,
    matrix_imag::BigFloat,
    right_real::BigFloat,
    right_imag::BigFloat,
    ::Val{:imag},
)
    product_real = matrix_real * right_real - matrix_imag * right_imag
    product_imag = matrix_real * right_imag + matrix_imag * right_real
    return left_real * product_imag - left_imag * product_real
end

@noinline function _quadratic_objective_bigfloat(
    I::Vector{<:Number},
    Q::Matrix{<:Number},
    ::Type{T},
) where {T<:AbstractFloat}
    return setprecision(BigFloat, _IEEE_BILINEAR_FALLBACK_PRECISION) do
        total = zero(BigFloat)
        @inbounds for row in axes(Q, 1)
            left_value = I[row]
            left_real = BigFloat(real(left_value))
            left_imag = BigFloat(imag(left_value))
            for column in axes(Q, 2)
                matrix_value = Q[row, column]
                right_value = I[column]
                matrix_real = BigFloat(real(matrix_value))
                matrix_imag = BigFloat(imag(matrix_value))
                right_real = BigFloat(real(right_value))
                right_imag = BigFloat(imag(right_value))
                total += _bilinear_component_bigfloat(
                    left_real, left_imag,
                    matrix_real, matrix_imag,
                    right_real, right_imag,
                    Val(:real),
                )
            end
        end
        converted = convert(T, total)
        isfinite(converted) ||
            throw(OverflowError(
                "quadratic objective is outside the representable $T range"))
        return converted
    end
end

@inline _bilinear_component(value::Number, ::Val{:real}) = real(value)
@inline _bilinear_component(value::Number, ::Val{:imag}) = imag(value)
function _accumulate_bilinear_component_bigfloat(
    left::AbstractVector,
    matrix::AbstractMatrix,
    right::AbstractVector,
    component,
)
    total = zero(BigFloat)
    @inbounds for row in axes(matrix, 1)
        left_value = left[row]
        left_real = BigFloat(real(left_value))
        left_imag = BigFloat(imag(left_value))
        for column in axes(matrix, 2)
            matrix_value = matrix[row, column]
            right_value = right[column]
            total += _bilinear_component_bigfloat(
                left_real, left_imag,
                BigFloat(real(matrix_value)),
                BigFloat(imag(matrix_value)),
                BigFloat(real(right_value)),
                BigFloat(imag(right_value)),
                component,
            )
        end
    end
    return total
end

function _accumulate_bilinear_component_bigfloat(
    left::AbstractVector,
    matrix::SparseArrays.AbstractSparseMatrixCSC,
    right::AbstractVector,
    component,
)
    total = zero(BigFloat)
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    @inbounds for column in axes(matrix, 2)
        right_value = right[column]
        right_real = BigFloat(real(right_value))
        right_imag = BigFloat(imag(right_value))
        for position in nzrange(matrix, column)
            left_value = left[rows[position]]
            matrix_value = values[position]
            total += _bilinear_component_bigfloat(
                BigFloat(real(left_value)),
                BigFloat(imag(left_value)),
                BigFloat(real(matrix_value)),
                BigFloat(imag(matrix_value)),
                right_real, right_imag,
                component,
            )
        end
    end
    return total
end


function _accumulate_bilinear_component_bigfloat(
    left::AbstractVector,
    matrix::LocalMassMatrix,
    right::AbstractVector,
    component,
)
    total = zero(BigFloat)
    @inbounds for position in eachindex(matrix.vals)
        left_value = left[matrix.rows[position]]
        matrix_value = matrix.vals[position]
        right_value = right[matrix.cols[position]]
        total += _bilinear_component_bigfloat(
            BigFloat(real(left_value)),
            BigFloat(imag(left_value)),
            BigFloat(real(matrix_value)),
            BigFloat(imag(matrix_value)),
            BigFloat(real(right_value)),
            BigFloat(imag(right_value)),
            component,
        )
    end
    return total
end

@noinline function _bilinear_component_bigfloat(
    left::AbstractVector,
    matrix::AbstractMatrix,
    right::AbstractVector,
    component,
    ::Type{T},
    label::AbstractString,
) where {T<:AbstractFloat}
    return setprecision(BigFloat, _IEEE_BILINEAR_FALLBACK_PRECISION) do
        total = _accumulate_bilinear_component_bigfloat(
            left, matrix, right, component)
        converted = convert(T, total)
        isfinite(converted) ||
            throw(OverflowError(
                "$label is outside the representable $T range"))
        return converted
    end
end

function _finite_bilinear_component(
    left::AbstractVector,
    matrix::AbstractMatrix,
    right::AbstractVector,
    component,
    label::AbstractString,
)
    value = _bilinear_component(
        _dot_left_matrix_right(left, matrix, right), component)
    isfinite(value) && return value
    value_type = typeof(value)
    value_type <: Union{Float32,Float64} ||
        error("$label produced a non-finite value")
    return _bilinear_component_bigfloat(
        left, matrix, right, component, value_type, label)
end

"""
    compute_objective(I, Q)

Compute the quadratic objective J = Re(I† Q I).
"""
function compute_objective(I::Vector{<:Number}, Q::Matrix{<:Number})
    _validate_linear_system_inputs(Q, I, "quadratic objective")
    value = real(_dot_left_matrix_right(I, Q, I))
    isfinite(value) && return value
    value_type = typeof(value)
    value_type <: Union{Float32,Float64} ||
        error("quadratic objective produced a non-finite value")
    return _quadratic_objective_bigfloat(I, Q, value_type)
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
        adjoint_Z = adjoint(Z)
        return _solve_factored_linear_system(
            lu(adjoint_Z), adjoint_Z, rhs, "direct adjoint solution")
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
        adjoint_Z = adjoint(Z)
        return _solve_factored_linear_system(
            lu(adjoint_Z), adjoint_Z, complex_rhs,
            "direct adjoint solution")
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
    component = reactive ? Val(:imag) : Val(:real)
    for p in 1:P
        lMI_component = _finite_bilinear_component(
            lambda, Mp[p], I, component,
            "gradient_impedance bilinear product")
        if reactive
            # ∂Z/∂θ_p = -iM_p
            # g[p] = -2 Re{ λ† (-iM_p) I } = 2 Re{ i λ† M_p I } = -2 Im{ λ† M_p I }
            g[p] = -2 * lMI_component
        else
            # ∂Z/∂θ_p = -M_p
            # g[p] = -2 Re{ λ† (-M_p) I } = 2 Re{ λ† M_p I }
            g[p] = 2 * lMI_component
        end
    end
    all(isfinite, g) ||
        error("gradient_impedance produced non-finite values")
    return g
end
