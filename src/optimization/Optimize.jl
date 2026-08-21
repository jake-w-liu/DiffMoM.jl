# Optimize.jl — L-BFGS optimization loop with optional box constraints

export optimize_lbfgs, optimize_directivity

@inline function _optimizer_bound_value(bound, p::Int)
    return bound isa Real ? bound : bound[p]
end

function _validate_optimizer_bound(bound, name::AbstractString, P::Int;
                                   lower::Bool)
    bound === nothing && return nothing
    if !(bound isa Real || bound isa AbstractVector)
        throw(ArgumentError(
            "$name must be a real scalar, a real vector, or nothing, got $(typeof(bound))"))
    end
    if bound isa AbstractVector
        length(bound) == P ||
            throw(DimensionMismatch(
                "$name length $(length(bound)) must match theta length $P"))
    end
    @inbounds for p in 1:P
        value = _optimizer_bound_value(bound, p)
        value isa Real ||
            throw(ArgumentError("$name[$p] must be real, got $(typeof(value))"))
        valid = isfinite(value) || (lower ? value == -Inf : value == Inf)
        if !valid
            allowed_infinity = lower ? "-Inf" : "Inf"
            throw(ArgumentError(
                "$name[$p] must be finite or $allowed_infinity, got $value"))
        end
    end
    return nothing
end

function _validate_optimizer_bounds(lb, ub, P::Int)
    _validate_optimizer_bound(lb, "lb", P; lower=true)
    _validate_optimizer_bound(ub, "ub", P; lower=false)
    if lb !== nothing && ub !== nothing
        @inbounds for p in 1:P
            lower = _optimizer_bound_value(lb, p)
            upper = _optimizer_bound_value(ub, p)
            lower <= upper ||
                throw(ArgumentError(
                    "lb[$p] must not exceed ub[$p], got $lower > $upper"))
        end
    end
    return nothing
end

function _validate_optimizer_controls(; maxiter::Int, tol::Float64,
                                      m_lbfgs::Int, alpha0::Float64,
                                      regularization_alpha::Float64,
                                      preconditioning::Symbol,
                                      auto_precondition_n_threshold::Int,
                                      auto_precondition_eps_rel::Float64,
                                      solver::Symbol,
                                      gmres_tol::Float64,
                                      gmres_maxiter::Int,
                                      gmres_memory::Int)
    maxiter >= 0 ||
        throw(ArgumentError("maxiter must be nonnegative, got $maxiter"))
    (isfinite(tol) && tol >= 0.0) ||
        throw(ArgumentError("tol must be finite and nonnegative, got $tol"))
    m_lbfgs >= 0 ||
        throw(ArgumentError("m_lbfgs must be nonnegative, got $m_lbfgs"))
    (isfinite(alpha0) && alpha0 > 0.0) ||
        throw(ArgumentError("alpha0 must be finite and positive, got $alpha0"))
    (isfinite(regularization_alpha) && regularization_alpha >= 0.0) ||
        throw(ArgumentError(
            "regularization_alpha must be finite and nonnegative, got $regularization_alpha"))
    preconditioning in (:off, :on, :auto) ||
        throw(ArgumentError(
            "preconditioning must be :off, :on, or :auto, got $preconditioning"))
    auto_precondition_n_threshold >= 0 ||
        throw(ArgumentError(
            "auto_precondition_n_threshold must be nonnegative, got $auto_precondition_n_threshold"))
    (isfinite(auto_precondition_eps_rel) && auto_precondition_eps_rel > 0.0) ||
        throw(ArgumentError(
            "auto_precondition_eps_rel must be finite and positive, got $auto_precondition_eps_rel"))
    solver in (:direct, :gmres) ||
        throw(ArgumentError("solver must be :direct or :gmres, got $solver"))
    if solver == :gmres
        _validate_gmres_options(gmres_tol, gmres_maxiter, gmres_memory, :left)
    end
    return nothing
end

function _validate_optimizer_problem(
    Z_efie::Matrix{ComplexF64},
    Mp::Vector{<:AbstractMatrix},
    v::Vector{ComplexF64},
    theta0::Vector{Float64},
    objective_matrices,
)
    P = length(theta0)
    P >= 1 ||
        throw(ArgumentError("optimizer requires at least one design parameter"))
    N = size(Z_efie, 1)
    size(Z_efie, 2) == N ||
        throw(DimensionMismatch(
            "Z_efie must be square, got size $(size(Z_efie))"))
    length(v) == N ||
        throw(DimensionMismatch("v length $(length(v)) must equal system size $N"))
    _validate_impedance_inputs(Mp, theta0, size(Z_efie))
    all(isfinite, Z_efie) ||
        throw(ArgumentError("Z_efie entries must be finite"))
    all(isfinite, v) ||
        throw(ArgumentError("v entries must be finite"))
    for (name, Q) in objective_matrices
        size(Q) == size(Z_efie) ||
            throw(DimensionMismatch(
                "$name has size $(size(Q)), expected $(size(Z_efie))"))
        all(isfinite, Q) ||
            throw(ArgumentError("$name entries must be finite"))
    end
    return (N, P)
end

function _validate_optimizer_auxiliary_matrices(
    system_size::Tuple{Int,Int},
    regularization_alpha::Float64,
    regularization_R,
    preconditioner_M,
)
    if regularization_alpha > 0.0 && regularization_R !== nothing
        size(regularization_R) == system_size ||
            throw(DimensionMismatch(
                "regularization_R has size $(size(regularization_R)), expected $system_size"))
        all(isfinite, regularization_R) ||
            throw(ArgumentError("regularization_R entries must be finite"))
    end
    if preconditioner_M !== nothing
        size(preconditioner_M) == system_size ||
            throw(DimensionMismatch(
                "preconditioner_M has size $(size(preconditioner_M)), expected $system_size"))
        all(isfinite, preconditioner_M) ||
            throw(ArgumentError("preconditioner_M entries must be finite"))
    end
    return nothing
end

@inline function _assert_finite_optimizer_vector(x::AbstractVector,
                                                 label::AbstractString)
    @inbounds for i in eachindex(x)
        isfinite(x[i]) ||
            error("$label contains a non-finite value at index $i: $(x[i])")
    end
    return x
end

# Curvature is geometric: rescaling the parameter or gradient units must not
# decide whether an L-BFGS pair is retained.  A small cosine floor rejects only
# nearly orthogonal pairs whose inverse-Hessian update would be ill-conditioned.
const _OPTIMIZER_MIN_CURVATURE_COSINE = sqrt(eps(Float64))

@inline function _optimizer_dot_requires_exact(
        left::AbstractVector{Float64},
        right::AbstractVector{Float64},
        value::Float64)
    (!isfinite(value) ||
     _ieee_bilinear_values_require_fallback(left, Float64) ||
     _ieee_bilinear_values_require_fallback(right, Float64)) && return true
    absolute_sum = 0.0
    @inbounds for index in eachindex(left, right)
        term = left[index] * right[index]
        isfinite(term) || return true
        absolute_sum += abs(term)
        isfinite(absolute_sum) || return true
    end
    error_factor = _ieee_product_error_factor(
        Float64, 1, length(left))
    return !iszero(absolute_sum) &&
           abs(value) <= error_factor * absolute_sum
end

@inline function _optimizer_dot_bigfloat(
        left::AbstractVector{Float64},
        right::AbstractVector{Float64})
    total = zero(BigFloat)
    @inbounds for index in eachindex(left, right)
        total += BigFloat(left[index]) * BigFloat(right[index])
    end
    return total
end

function _optimizer_curvature_pair_acceptable(
        step::Vector{Float64}, gradient_change::Vector{Float64})
    length(step) == length(gradient_change) ||
        throw(DimensionMismatch(
            "L-BFGS step and gradient-change lengths must match"))
    all(isfinite, step) && all(isfinite, gradient_change) || return false
    sy = dot(step, gradient_change)
    ss = dot(step, step)
    yy = dot(gradient_change, gradient_change)
    needs_exact =
        _optimizer_dot_requires_exact(step, gradient_change, sy) ||
        _optimizer_dot_requires_exact(step, step, ss) ||
        _optimizer_dot_requires_exact(
            gradient_change, gradient_change, yy)
    if !needs_exact
        norm_product = sqrt(ss) * sqrt(yy)
        if isfinite(norm_product)
            return sy > _OPTIMIZER_MIN_CURVATURE_COSINE * norm_product
        end
    end
    return setprecision(BigFloat, _IEEE_BILINEAR_FALLBACK_PRECISION) do
        sy_big = _optimizer_dot_bigfloat(step, gradient_change)
        ss_big = _optimizer_dot_bigfloat(step, step)
        yy_big = _optimizer_dot_bigfloat(
            gradient_change, gradient_change)
        ss_big > 0 && yy_big > 0 || return false
        return sy_big > BigFloat(_OPTIMIZER_MIN_CURVATURE_COSINE) *
                        sqrt(ss_big) * sqrt(yy_big)
    end
end

function _optimizer_dot_ratio(
        numerator_left::AbstractVector{Float64},
        numerator_right::AbstractVector{Float64},
        denominator_left::AbstractVector{Float64},
        denominator_right::AbstractVector{Float64})
    length(numerator_left) == length(numerator_right) ||
        throw(DimensionMismatch("L-BFGS numerator vector lengths must match"))
    length(denominator_left) == length(denominator_right) ||
        throw(DimensionMismatch("L-BFGS denominator vector lengths must match"))
    (all(isfinite, numerator_left) && all(isfinite, numerator_right) &&
     all(isfinite, denominator_left) && all(isfinite, denominator_right)) ||
        return nothing

    numerator = dot(numerator_left, numerator_right)
    denominator = dot(denominator_left, denominator_right)
    needs_exact = denominator <= 0.0 ||
        _optimizer_dot_requires_exact(
            numerator_left, numerator_right, numerator) ||
        _optimizer_dot_requires_exact(
            denominator_left, denominator_right, denominator)
    if !needs_exact
        ratio = numerator / denominator
        isfinite(ratio) && return ratio
    end

    return setprecision(BigFloat, _IEEE_BILINEAR_FALLBACK_PRECISION) do
        numerator_big = _optimizer_dot_bigfloat(
            numerator_left, numerator_right)
        denominator_big = _optimizer_dot_bigfloat(
            denominator_left, denominator_right)
        denominator_big > 0 || return nothing
        ratio_big = numerator_big / denominator_big
        ratio = Float64(ratio_big)
        return isfinite(ratio) &&
               (!iszero(ratio) || iszero(ratio_big)) ? ratio : nothing
    end
end

function _optimizer_dot_is_negative(
        left::AbstractVector{Float64},
        right::AbstractVector{Float64})
    length(left) == length(right) ||
        throw(DimensionMismatch("L-BFGS direction vector lengths must match"))
    all(isfinite, left) && all(isfinite, right) || return false
    value = dot(left, right)
    _optimizer_dot_requires_exact(left, right, value) || return value < 0.0
    return setprecision(BigFloat, _IEEE_BILINEAR_FALLBACK_PRECISION) do
        _optimizer_dot_bigfloat(left, right) < 0
    end
end

function _optimizer_scaled_gradient(
        gradient::Vector{Float64}, scale::Float64)
    result = copy(gradient)
    rmul!(result, scale)
    if !_linear_array_is_finite(result)
        copyto!(result, gradient)
    end
    return result
end

function _optimizer_lbfgs_inverse_product(
        gradient::Vector{Float64},
        step_history::Vector{Vector{Float64}},
        gradient_history::Vector{Vector{Float64}},
        initial_scale::Float64)
    length(step_history) == length(gradient_history) ||
        throw(DimensionMismatch("L-BFGS history lengths must match"))
    q = copy(gradient)
    history_count = length(step_history)
    alpha = zeros(Float64, history_count)

    @inbounds for history_index in history_count:-1:1
        coefficient = _optimizer_dot_ratio(
            step_history[history_index], q,
            gradient_history[history_index],
            step_history[history_index])
        coefficient === nothing &&
            return _optimizer_scaled_gradient(
                gradient, initial_scale), false
        alpha[history_index] = coefficient
        axpy!(-coefficient, gradient_history[history_index], q)
        _linear_array_is_finite(q) ||
            return _optimizer_scaled_gradient(
                gradient, initial_scale), false
    end

    gamma = history_count == 0 ? initial_scale :
        _optimizer_dot_ratio(
            step_history[end], gradient_history[end],
            gradient_history[end], gradient_history[end])
    (gamma !== nothing && gamma > 0.0) ||
        return _optimizer_scaled_gradient(gradient, initial_scale), false
    rmul!(q, gamma)
    _linear_array_is_finite(q) ||
        return _optimizer_scaled_gradient(gradient, initial_scale), false

    @inbounds for history_index in 1:history_count
        beta = _optimizer_dot_ratio(
            gradient_history[history_index], q,
            gradient_history[history_index],
            step_history[history_index])
        beta === nothing &&
            return _optimizer_scaled_gradient(
                gradient, initial_scale), false
        axpy!(alpha[history_index] - beta,
              step_history[history_index], q)
        _linear_array_is_finite(q) ||
            return _optimizer_scaled_gradient(
                gradient, initial_scale), false
    end
    return q, true
end

@inline function _directivity_ratio(f_value::Float64, denominator::Float64,
                                    label::AbstractString)
    isfinite(f_value) ||
        error("$label directivity numerator is non-finite: $f_value")
    (isfinite(denominator) && denominator > 0.0) ||
        throw(DomainError(
            denominator,
            "$label directivity denominator must be finite and positive"))
    ratio = f_value / denominator
    isfinite(ratio) ||
        error("$label directivity ratio is non-finite: $ratio")
    return ratio
end

function _directivity_bigfloat_product(
    matrix::Matrix{ComplexF64},
    current::Vector{ComplexF64},
)
    result = Vector{Complex{BigFloat}}(undef, length(current))
    @inbounds for row in axes(matrix, 1)
        total_real = zero(BigFloat)
        total_imag = zero(BigFloat)
        for column in axes(matrix, 2)
            matrix_value = matrix[row, column]
            current_value = current[column]
            matrix_real = BigFloat(real(matrix_value))
            matrix_imag = BigFloat(imag(matrix_value))
            current_real = BigFloat(real(current_value))
            current_imag = BigFloat(imag(current_value))
            total_real += matrix_real * current_real -
                          matrix_imag * current_imag
            total_imag += matrix_real * current_imag +
                          matrix_imag * current_real
        end
        result[row] = Complex{BigFloat}(total_real, total_imag)
    end
    return result
end

function _directivity_bigfloat_objective(
    current::Vector{ComplexF64},
    product::Vector{Complex{BigFloat}},
)
    total = zero(BigFloat)
    @inbounds for index in eachindex(current, product)
        current_value = current[index]
        product_value = product[index]
        total += BigFloat(real(current_value)) * real(product_value) +
                 BigFloat(imag(current_value)) * imag(product_value)
    end
    return total
end

@inline function _directivity_ratio_from_bigfloat(
    target_objective::BigFloat,
    total_objective::BigFloat,
    label::AbstractString,
)
    total_objective > 0 ||
        throw(DomainError(
            Float64(total_objective),
            "$label directivity denominator must be positive",
        ))
    ratio_big = target_objective / total_objective
    ratio = Float64(ratio_big)
    isfinite(ratio) ||
        throw(OverflowError(
            "$label directivity ratio is outside the representable Float64 range"))
    return ratio_big, ratio
end

@noinline function _directivity_ratio_bigfloat(
    Q_target::Matrix{ComplexF64},
    Q_total::Matrix{ComplexF64},
    current::Vector{ComplexF64},
    label::AbstractString,
)
    return setprecision(BigFloat, _IEEE_BILINEAR_FALLBACK_PRECISION) do
        target_big = _directivity_bigfloat_product(Q_target, current)
        total_big = _directivity_bigfloat_product(Q_total, current)
        target_objective = _directivity_bigfloat_objective(
            current, target_big)
        total_objective = _directivity_bigfloat_objective(
            current, total_big)
        _, ratio = _directivity_ratio_from_bigfloat(
            target_objective, total_objective, label)
        return ratio
    end
end

@noinline function _directivity_ratio_gradient_bigfloat(
    system::Matrix{ComplexF64},
    Mp::Vector{<:AbstractMatrix},
    Q_target::Matrix{ComplexF64},
    Q_total::Matrix{ComplexF64},
    current::Vector{ComplexF64},
    label::AbstractString;
    reactive::Bool,
)
    return setprecision(BigFloat, _IEEE_BILINEAR_FALLBACK_PRECISION) do
        target_big = _directivity_bigfloat_product(Q_target, current)
        total_big = _directivity_bigfloat_product(Q_total, current)
        target_objective = _directivity_bigfloat_objective(
            current, target_big)
        total_objective = _directivity_bigfloat_objective(
            current, total_big)
        ratio_big, ratio = _directivity_ratio_from_bigfloat(
            target_objective, total_objective, label)

        # Solve once for the normalized ratio derivative. Forming the
        # difference before conversion preserves cancellation between the two
        # objective products even when neither RHS fits in one Float64 scale.
        @inbounds for index in eachindex(target_big, total_big)
            target_big[index] = (
                target_big[index] - ratio_big * total_big[index]
            ) / total_objective
        end
        adjoint_system = Matrix{Complex{BigFloat}}(adjoint(system))
        ldiv!(lu!(adjoint_system), target_big)
        all(isfinite, target_big) ||
            error("$label high-precision directivity adjoint solve produced non-finite values")

        gradient = Vector{Float64}(undef, length(Mp))
        component = reactive ? Val(:imag) : Val(:real)
        @inbounds for parameter in eachindex(Mp)
            bilinear_component = _accumulate_bilinear_component_bigfloat(
                target_big, Mp[parameter], current, component)
            gradient_big = reactive ?
                           -2 * bilinear_component :
                           2 * bilinear_component
            gradient[parameter] = Float64(gradient_big)
            isfinite(gradient[parameter]) ||
                throw(OverflowError(
                    "$label directivity gradient at parameter $parameter is " *
                    "outside the representable Float64 range"))
        end
        return ratio, gradient
    end
end

@inline function _update_directivity_exponent_bounds(
    value::BigFloat,
    minimum_exponent::Int,
    maximum_exponent::Int,
)
    iszero(value) && return minimum_exponent, maximum_exponent
    value_exponent = exponent(abs(value))
    return min(minimum_exponent, value_exponent),
           max(maximum_exponent, value_exponent)
end

@noinline function _scaled_directivity_products_bigfloat!(
    target_product::Vector{ComplexF64},
    total_product::Vector{ComplexF64},
    Q_target::Matrix{ComplexF64},
    Q_total::Matrix{ComplexF64},
    current::Vector{ComplexF64},
    label::AbstractString,
)
    return setprecision(BigFloat, _IEEE_BILINEAR_FALLBACK_PRECISION) do
        target_big = _directivity_bigfloat_product(Q_target, current)
        total_big = _directivity_bigfloat_product(Q_total, current)
        target_objective = _directivity_bigfloat_objective(
            current, target_big)
        total_objective = _directivity_bigfloat_objective(
            current, total_big)
        total_objective > 0 ||
            throw(DomainError(
                Float64(total_objective),
                "$label directivity denominator must be positive",
            ))

        minimum_exponent = typemax(Int)
        maximum_exponent = typemin(Int)
        @inbounds for product in (target_big, total_big), value in product
            minimum_exponent, maximum_exponent =
                _update_directivity_exponent_bounds(
                    real(value), minimum_exponent, maximum_exponent)
            minimum_exponent, maximum_exponent =
                _update_directivity_exponent_bounds(
                    imag(value), minimum_exponent, maximum_exponent)
        end
        minimum_exponent, maximum_exponent =
            _update_directivity_exponent_bounds(
                target_objective, minimum_exponent, maximum_exponent)
        minimum_exponent, maximum_exponent =
            _update_directivity_exponent_bounds(
                total_objective, minimum_exponent, maximum_exponent)

        lower_shift = maximum_exponent - 1022
        upper_shift = minimum_exponent + 1074
        lower_shift <= upper_shift ||
            throw(OverflowError(
                "$label directivity products span more exponent range than " *
                "one common Float64 scale can represent"))
        scale_shift = clamp(
            maximum_exponent, lower_shift, upper_shift)

        @inbounds for index in eachindex(target_product, target_big)
            target_product[index] = ComplexF64(
                Float64(ldexp(real(target_big[index]), -scale_shift)),
                Float64(ldexp(imag(target_big[index]), -scale_shift)),
            )
            total_product[index] = ComplexF64(
                Float64(ldexp(real(total_big[index]), -scale_shift)),
                Float64(ldexp(imag(total_big[index]), -scale_shift)),
            )
        end
        _assert_finite_optimizer_vector(
            target_product, "$label scaled directivity numerator product")
        _assert_finite_optimizer_vector(
            total_product, "$label scaled directivity denominator product")
        target_scaled = Float64(ldexp(target_objective, -scale_shift))
        total_scaled = Float64(ldexp(total_objective, -scale_shift))
        isfinite(target_scaled) ||
            throw(OverflowError(
                "$label scaled directivity numerator is outside Float64 range"))
        isfinite(total_scaled) && total_scaled > 0 ||
            throw(OverflowError(
                "$label scaled directivity denominator is outside Float64 range"))
        return target_scaled, total_scaled
    end
end

function _directivity_products_and_objectives!(
    target_product::Vector{ComplexF64},
    total_product::Vector{ComplexF64},
    Q_target::Matrix{ComplexF64},
    Q_total::Matrix{ComplexF64},
    current::Vector{ComplexF64},
    label::AbstractString,
)
    try
        target_product_used_fallback = _finite_matrix_vector_product_status!(
            target_product, Q_target, current,
            "directivity numerator product")
        total_product_used_fallback = _finite_matrix_vector_product_status!(
            total_product, Q_total, current,
            "directivity denominator product")
        target_value = _quadratic_objective_from_product(
            current, Q_target, target_product,
            target_product_used_fallback)
        total_value = _quadratic_objective_from_product(
            current, Q_total, total_product,
            total_product_used_fallback)
        return target_value, total_value
    catch err
        err isa OverflowError || rethrow()
    end
    return _scaled_directivity_products_bigfloat!(
        target_product,
        total_product,
        Q_target,
        Q_total,
        current,
        label,
    )
end

@noinline function _directivity_gradient_component_bigfloat(
    target_gradient::Float64,
    total_gradient::Float64,
    ratio::Float64,
    denominator::Float64,
    parameter::Int,
)
    return setprecision(BigFloat, _IEEE_BILINEAR_FALLBACK_PRECISION) do
        value = (
            BigFloat(target_gradient) -
            BigFloat(ratio) * BigFloat(total_gradient)
        ) / BigFloat(denominator)
        converted = Float64(value)
        isfinite(converted) ||
            throw(OverflowError(
                "directivity gradient at parameter $parameter is outside " *
                "the representable Float64 range"))
        return converted
    end
end

function _combine_directivity_gradient!(
    target_gradient::Vector{Float64},
    total_gradient::Vector{Float64},
    ratio::Float64,
    denominator::Float64,
)
    @inbounds for parameter in eachindex(target_gradient, total_gradient)
        target_value = target_gradient[parameter]
        total_value = total_gradient[parameter]
        value = muladd(
            -ratio,
            total_value,
            target_value,
        ) / denominator
        if !isfinite(value) ||
           _ieee_dense_extreme_factor(target_value, Float64) ||
           _ieee_dense_extreme_factor(total_value, Float64) ||
           _ieee_dense_extreme_factor(ratio, Float64) ||
           _ieee_dense_extreme_factor(denominator, Float64)
            value = _directivity_gradient_component_bigfloat(
                target_value,
                total_value,
                ratio,
                denominator,
                parameter,
            )
        end
        target_gradient[parameter] = value
    end
    return target_gradient
end

@inline function _recoverable_optimizer_trial_error(err)
    return err isa LinearAlgebra.SingularException ||
           err isa LinearAlgebra.LAPACKException ||
           err isa LinearAlgebra.PosDefException ||
           err isa LinearAlgebra.ZeroPivotException ||
           err isa OverflowError
end

@inline function _projected_directional_derivative(
    gradient::AbstractVector{Float64},
    trial::AbstractVector{Float64},
    current::AbstractVector{Float64},
)
    needs_fallback = false
    @inbounds for p in eachindex(gradient, trial, current)
        if _ieee_dense_extreme_factor(gradient[p], Float64) ||
           _ieee_dense_extreme_factor(trial[p], Float64) ||
           _ieee_dense_extreme_factor(current[p], Float64)
            needs_fallback = true
            break
        end
    end
    derivative = 0.0
    absolute_sum = 0.0
    if !needs_fallback
        @inbounds for p in eachindex(gradient, trial, current)
            term = gradient[p] * (trial[p] - current[p])
            derivative += term
            absolute_sum += abs(term)
            if !isfinite(derivative) || !isfinite(absolute_sum)
                needs_fallback = true
                break
            end
        end
        error_factor = _ieee_product_error_factor(
            Float64, 1, length(gradient))
        needs_fallback |= !iszero(absolute_sum) &&
            abs(derivative) <= error_factor * absolute_sum
    end
    if needs_fallback
        return setprecision(
                BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
            derivative_big = zero(BigFloat)
            @inbounds for p in eachindex(gradient, trial, current)
                derivative_big += BigFloat(gradient[p]) *
                    (BigFloat(trial[p]) - BigFloat(current[p]))
            end
            converted = Float64(derivative_big)
            if iszero(converted) && !iszero(derivative_big)
                # The line search first needs the exact sign. Preserve it when
                # the magnitude lies below the Float64 range; returning ±0
                # would incorrectly classify every such direction as flat.
                return derivative_big < 0 ?
                    -nextfloat(0.0) : nextfloat(0.0)
            end
            # Signed infinities make the caller shrink the trial step. Throwing
            # here aborts the whole optimization before backtracking can act.
            return converted
        end
    end
    return derivative
end

"""
    optimize_lbfgs(Z_efie, Mp, v, Q, theta0; kwargs...)

L-BFGS optimization with precomputed EFIE matrix, patch mass matrices,
excitation vector, and Q matrix.

Options:
  reactive:  if true, impedance is Z_s = iθ (reactive/lossless)
  maximize:  if true, maximize J = I†QI instead of minimizing
  lb, ub:    box constraints on θ (projected L-BFGS-B)
  solver:    `:direct` (default) for LU, `:gmres` for GMRES
  nf_preconditioner: an `AbstractPreconditionerData` for GMRES preconditioning, or `nothing`
  gmres_tol: GMRES relative tolerance (default 1e-8)
  gmres_maxiter: maximum GMRES iterations (default 200)
  gmres_memory: Krylov restart/memory parameter (default 20)

Returns (theta_opt, trace) where trace records (iter, J, |g|) per iteration.
"""
function optimize_lbfgs(Z_efie::Matrix{ComplexF64},
                        Mp::Vector{<:AbstractMatrix},
                        v::Vector{ComplexF64},
                        Q::Matrix{ComplexF64},
                        theta0::Vector{Float64};
                        maxiter::Int=100,
                        tol::Float64=1e-10,
                        m_lbfgs::Int=10,
                        alpha0::Float64=0.01,
                        verbose::Bool=true,
                        reactive::Bool=false,
                        maximize::Bool=false,
                        lb=nothing,
                        ub=nothing,
                        regularization_alpha::Float64=0.0,
                        regularization_R=nothing,
                        preconditioner_M=nothing,
                        preconditioning::Symbol=:off,
                        auto_precondition_n_threshold::Int=256,
                        iterative_solver::Bool=false,
                        auto_precondition_eps_rel::Float64=1e-6,
                        solver::Symbol=:direct,
                        nf_preconditioner::Union{Nothing, AbstractPreconditionerData}=nothing,
                        gmres_tol::Float64=1e-8,
                        gmres_maxiter::Int=200,
                        gmres_memory::Int=20,
                        max_workspace_bytes::Integer=
                            _DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    _validate_optimizer_controls(
        maxiter=maxiter,
        tol=tol,
        m_lbfgs=m_lbfgs,
        alpha0=alpha0,
        regularization_alpha=regularization_alpha,
        preconditioning=preconditioning,
        auto_precondition_n_threshold=auto_precondition_n_threshold,
        auto_precondition_eps_rel=auto_precondition_eps_rel,
        solver=solver,
        gmres_tol=gmres_tol,
        gmres_maxiter=gmres_maxiter,
        gmres_memory=gmres_memory,
    )
    N_sys, P = _validate_optimizer_problem(
        Z_efie, Mp, v, theta0, (("Q", Q),))
    _validate_optimizer_bounds(lb, ub, P)
    _validate_optimizer_auxiliary_matrices(
        size(Z_efie), regularization_alpha, regularization_R, preconditioner_M)
    workspace_bytes = _checked_array_payload_bytes(
        ComplexF64, N_sys, N_sys;
        label="L-BFGS dense system workspace")
    _enforce_payload_limit(
        workspace_bytes, max_workspace_bytes,
        "L-BFGS dense system workspace", "max_workspace_bytes")

    theta = copy(theta0)
    sense = maximize ? -1.0 : 1.0

    # Projection for box constraints
    function project!(x)
        if lb !== nothing
            x .= max.(x, lb)
        end
        if ub !== nothing
            x .= min.(x, ub)
        end
        return x
    end
    project!(theta)

    # Conditioning setup (constant across iterations)
    R_mat = if regularization_alpha == 0.0
        nothing
    else
        regularization_R === nothing ?
            make_mass_regularizer(Mp; max_output_bytes=max_workspace_bytes) :
            Matrix{ComplexF64}(regularization_R)
    end

    precond_M_eff, precond_enabled, precond_reason = select_preconditioner(
        Mp;
        mode=preconditioning,
        preconditioner_M=preconditioner_M,
        n_threshold=auto_precondition_n_threshold,
        iterative_solver=iterative_solver,
        eps_rel=auto_precondition_eps_rel,
        max_output_bytes=max_workspace_bytes,
    )
    verbose && (preconditioning == :auto || preconditioner_M !== nothing || precond_enabled) &&
        println("  preconditioning: ", precond_enabled ? "ON ($precond_reason)" : "OFF ($precond_reason)")

    Mp_eff, precond_fac = transform_patch_matrices(
        Mp;
        preconditioner_M=precond_M_eff,
        max_output_bytes=max_workspace_bytes,
    )
    rhs_eff_base = precond_fac === nothing ?
                   Vector{ComplexF64}(v) :
                   _solve_factored_linear_system(
                       precond_fac,
                       precond_M_eff,
                       v,
                       "optimizer conditioned RHS",
                   )

    if solver == :gmres && verbose
        if nf_preconditioner !== nothing
            println("  GMRES solver: NF preconditioned (cutoff=$(round(nf_preconditioner.cutoff, sigdigits=3)), nnz=$(round(nf_preconditioner.nnz_ratio*100, digits=1))%)")
        else
            println("  GMRES solver: unpreconditioned")
        end
    end

    # L-BFGS history
    s_list = Vector{Vector{Float64}}()
    y_list = Vector{Vector{Float64}}()

    trace = Vector{NamedTuple{(:iter, :J, :gnorm, :n_fwd, :n_adj), Tuple{Int,Float64,Float64,Int,Int}}}()

    # Solve counters — track every forward and adjoint solve including line search
    n_fwd_solves = 0
    n_adj_solves = 0

    # Reused iteration and objective workspaces.
    theta_old = similar(theta)
    g_old = similar(theta)
    theta_trial = similar(theta)
    Z_raw = Matrix{ComplexF64}(undef, N_sys, N_sys)
    QI = Vector{ComplexF64}(undef, N_sys)

    for iter in 1:maxiter
        # Assemble and solve
        assemble_full_Z!(Z_raw, Z_efie, Mp, theta; reactive=reactive)
        Z, _, _ = prepare_conditioned_system(
            Z_raw,
            v;
            regularization_alpha=regularization_alpha,
            regularization_R=R_mat,
            preconditioner_M=precond_M_eff,
            preconditioner_factor=precond_fac,
        )

        I_coeffs = solve_forward(Z, rhs_eff_base;
                                  solver=solver, preconditioner=nf_preconditioner,
                                  gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter,
                                  gmres_memory=gmres_memory)
        n_fwd_solves += 1
        _assert_finite_optimizer_vector(I_coeffs, "forward solution")

        # Objective (always report the true J)
        objective_product_used_fallback =
            _finite_matrix_vector_product_status!(
                QI, Q, I_coeffs, "accepted objective product")
        J_val = _quadratic_objective_from_product(
            I_coeffs, Q, QI, objective_product_used_fallback)
        isfinite(J_val) ||
            error("objective is non-finite at iteration $iter: $J_val")

        # Adjoint solve
        lambda = solve_adjoint_rhs(Z, QI;
                                    solver=solver, preconditioner=nf_preconditioner,
                                    gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter,
                                    gmres_memory=gmres_memory)
        n_adj_solves += 1
        _assert_finite_optimizer_vector(lambda, "adjoint solution")

        # Gradient of J (true objective)
        g = gradient_impedance(Mp_eff, I_coeffs, lambda; reactive=reactive)
        _assert_finite_optimizer_vector(g, "objective gradient")
        gnorm = norm(g)
        isfinite(gnorm) ||
            error("objective gradient norm is non-finite at iteration $iter: $gnorm")

        # For minimization: g = g_true; for maximization: g = -g_true
        rmul!(g, sense)

        push!(trace, (iter=iter, J=J_val, gnorm=gnorm, n_fwd=n_fwd_solves, n_adj=n_adj_solves))
        if verbose
            println("  iter=$iter  J=$J_val  |g|=$gnorm  solves(fwd=$n_fwd_solves, adj=$n_adj_solves)")
        end

        if gnorm <= tol
            if verbose
                println("Converged at iteration $iter")
            end
            break
        end

        # L-BFGS curvature pair update
        if iter > 1
            s_k = theta - theta_old
            y_k = g - g_old
            if m_lbfgs > 0 &&
               _optimizer_curvature_pair_acceptable(s_k, y_k)
                push!(s_list, s_k)
                push!(y_list, y_k)
                if length(s_list) > m_lbfgs
                    popfirst!(s_list)
                    popfirst!(y_list)
                end
            end
        end

        # Scale-safe two-loop recursion. Ratios are evaluated directly so a
        # representable quotient is not lost through an overflowing reciprocal
        # or an underflowed/overflowed dot product.
        r, history_valid = _optimizer_lbfgs_inverse_product(
            g, s_list, y_list, alpha0)
        if !history_valid
            empty!(s_list)
            empty!(y_list)
        end

        # Line search: backtracking Armijo with projection
        d = r
        rmul!(d, -1.0)

        # Check if L-BFGS direction is descent; if not, reset to steepest descent
        if !_optimizer_dot_is_negative(g, d)
            copyto!(d, g)
            rmul!(d, -1.0)
            empty!(s_list)
            empty!(y_list)
        end

        alpha = 1.0
        copyto!(theta_old, theta)
        copyto!(g_old, g)
        J_internal = sense * J_val
        ls_success = false

        for ls in 1:20
            @. theta_trial = theta_old + alpha * d
            project!(theta_trial)
            theta_trial == theta_old && break
            if !all(isfinite, theta_trial)
                alpha *= 0.5
                continue
            end
            projected_gd =
                _projected_directional_derivative(g, theta_trial, theta_old)
            if !(isfinite(projected_gd) && projected_gd < 0.0)
                alpha *= 0.5
                continue
            end
            trial_finite = false
            J_trial_internal = Inf
            try
                assemble_full_Z!(Z_raw, Z_efie, Mp, theta_trial; reactive=reactive)
                Z_trial, _, _ = prepare_conditioned_system(
                    Z_raw,
                    v;
                    regularization_alpha=regularization_alpha,
                    regularization_R=R_mat,
                    preconditioner_M=precond_M_eff,
                    preconditioner_factor=precond_fac,
                )
                I_trial = solve_forward(Z_trial, rhs_eff_base;
                                         solver=solver, preconditioner=nf_preconditioner,
                                         gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter,
                                         gmres_memory=gmres_memory)
                n_fwd_solves += 1
                trial_finite = all(isfinite, I_trial)
                if trial_finite
                    trial_product_used_fallback =
                        _finite_matrix_vector_product_status!(
                            QI, Q, I_trial, "trial objective product")
                    J_trial_internal = sense *
                                       _quadratic_objective_from_product(
                                           I_trial, Q, QI,
                                           trial_product_used_fallback)
                    trial_finite = isfinite(J_trial_internal)
                end
            catch err
                _recoverable_optimizer_trial_error(err) || rethrow()
                verbose &&
                    println("Line-search trial $ls failed at iteration $iter; reducing step ($err)")
            end

            if trial_finite &&
               J_trial_internal <= J_internal + 1e-4 * projected_gd
                copyto!(theta, theta_trial)
                ls_success = true
                break
            end
            alpha *= 0.5
        end
        if !ls_success
            copyto!(theta, theta_old)
            empty!(s_list)
            empty!(y_list)
            verbose &&
                println("Line search failed at iteration $iter; stopping without accepting trial step")
            break
        end
    end

    return theta, trace
end

"""
    optimize_directivity(Z_efie, Mp, v, Q_target, Q_total, theta0; kwargs...)

Maximize the directivity ratio J = (I†Q_target I) / (I†Q_total I) using
projected L-BFGS on the minimization of -J.

At each iteration, evaluates the ratio directly and computes the gradient via
the quotient rule using two adjoint solves (one for Q_target and one for
Q_total). The ratio J_ratio is evaluated directly in the line search.  This
naturally steers the beam rather than just broadening it.

Options:
  solver:    `:direct` (default) for LU, `:gmres` for GMRES
  nf_preconditioner: an `AbstractPreconditionerData` for GMRES preconditioning, or `nothing`
  gmres_tol: GMRES relative tolerance (default 1e-8)
  gmres_maxiter: maximum GMRES iterations (default 200)
  gmres_memory: Krylov restart/memory parameter (default 20)

Returns (theta_opt, trace) where trace records (iter, J, |g|) per iteration.
"""
function optimize_directivity(Z_efie::Matrix{ComplexF64},
                              Mp::Vector{<:AbstractMatrix},
                              v::Vector{ComplexF64},
                              Q_target::Matrix{ComplexF64},
                              Q_total::Matrix{ComplexF64},
                              theta0::Vector{Float64};
                              maxiter::Int=100,
                              tol::Float64=1e-6,
                              m_lbfgs::Int=10,
                              alpha0::Float64=0.01,
                              verbose::Bool=true,
                              reactive::Bool=false,
                              lb=nothing,
                              ub=nothing,
                              regularization_alpha::Float64=0.0,
                              regularization_R=nothing,
                              preconditioner_M=nothing,
                              preconditioning::Symbol=:off,
                              auto_precondition_n_threshold::Int=256,
                              iterative_solver::Bool=false,
                              auto_precondition_eps_rel::Float64=1e-6,
                              solver::Symbol=:direct,
                              nf_preconditioner::Union{Nothing, AbstractPreconditionerData}=nothing,
                              gmres_tol::Float64=1e-8,
                              gmres_maxiter::Int=200,
                              gmres_memory::Int=20,
                              max_workspace_bytes::Integer=
                                  _DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    _validate_optimizer_controls(
        maxiter=maxiter,
        tol=tol,
        m_lbfgs=m_lbfgs,
        alpha0=alpha0,
        regularization_alpha=regularization_alpha,
        preconditioning=preconditioning,
        auto_precondition_n_threshold=auto_precondition_n_threshold,
        auto_precondition_eps_rel=auto_precondition_eps_rel,
        solver=solver,
        gmres_tol=gmres_tol,
        gmres_maxiter=gmres_maxiter,
        gmres_memory=gmres_memory,
    )
    N_sys, P = _validate_optimizer_problem(
        Z_efie,
        Mp,
        v,
        theta0,
        (("Q_target", Q_target), ("Q_total", Q_total)),
    )
    _validate_optimizer_bounds(lb, ub, P)
    _validate_optimizer_auxiliary_matrices(
        size(Z_efie), regularization_alpha, regularization_R, preconditioner_M)
    workspace_bytes = _checked_array_payload_bytes(
        ComplexF64, N_sys, N_sys;
        label="directivity dense system workspace")
    _enforce_payload_limit(
        workspace_bytes, max_workspace_bytes,
        "directivity dense system workspace", "max_workspace_bytes")

    theta = copy(theta0)

    function project!(x)
        lb !== nothing && (x .= max.(x, lb))
        ub !== nothing && (x .= min.(x, ub))
        return x
    end
    project!(theta)

    # Conditioning setup (constant across iterations)
    R_mat = if regularization_alpha == 0.0
        nothing
    else
        regularization_R === nothing ?
            make_mass_regularizer(Mp; max_output_bytes=max_workspace_bytes) :
            Matrix{ComplexF64}(regularization_R)
    end

    precond_M_eff, precond_enabled, precond_reason = select_preconditioner(
        Mp;
        mode=preconditioning,
        preconditioner_M=preconditioner_M,
        n_threshold=auto_precondition_n_threshold,
        iterative_solver=iterative_solver,
        eps_rel=auto_precondition_eps_rel,
        max_output_bytes=max_workspace_bytes,
    )
    verbose && (preconditioning == :auto || preconditioner_M !== nothing || precond_enabled) &&
        println("  preconditioning: ", precond_enabled ? "ON ($precond_reason)" : "OFF ($precond_reason)")

    Mp_eff, precond_fac = transform_patch_matrices(
        Mp;
        preconditioner_M=precond_M_eff,
        max_output_bytes=max_workspace_bytes,
    )
    rhs_eff_base = precond_fac === nothing ?
                   Vector{ComplexF64}(v) :
                   _solve_factored_linear_system(
                       precond_fac,
                       precond_M_eff,
                       v,
                       "directivity optimizer conditioned RHS",
                   )

    if solver == :gmres && verbose
        if nf_preconditioner !== nothing
            println("  GMRES solver: NF preconditioned (cutoff=$(round(nf_preconditioner.cutoff, sigdigits=3)), nnz=$(round(nf_preconditioner.nnz_ratio*100, digits=1))%)")
        else
            println("  GMRES solver: unpreconditioned")
        end
    end

    s_list = Vector{Vector{Float64}}()
    y_list = Vector{Vector{Float64}}()
    trace = Vector{NamedTuple{(:iter, :J, :gnorm), Tuple{Int,Float64,Float64}}}()

    # Reused iteration and objective workspaces.
    theta_old = similar(theta)
    g_old = similar(theta)
    theta_trial = similar(theta)
    Z_raw = Matrix{ComplexF64}(undef, N_sys, N_sys)
    QI_target = Vector{ComplexF64}(undef, N_sys)
    QI_total = Vector{ComplexF64}(undef, N_sys)

    for iter in 1:maxiter
        assemble_full_Z!(Z_raw, Z_efie, Mp, theta; reactive=reactive)
        Z, _, _ = prepare_conditioned_system(
            Z_raw,
            v;
            regularization_alpha=regularization_alpha,
            regularization_R=R_mat,
            preconditioner_M=precond_M_eff,
            preconditioner_factor=precond_fac,
        )

        I_c = solve_forward(Z, rhs_eff_base;
                             solver=solver, preconditioner=nf_preconditioner,
                             gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter,
                             gmres_memory=gmres_memory)
        _assert_finite_optimizer_vector(I_c, "forward solution")

        local J_ratio::Float64
        local g_f::Vector{Float64}
        try
            f_val, g_val = _directivity_products_and_objectives!(
                QI_target,
                QI_total,
                Q_target,
                Q_total,
                I_c,
                "accepted iterate",
            )
            J_ratio = _directivity_ratio(f_val, g_val, "accepted iterate")

            # Two separate adjoint solves for numerically stable ratio gradient
            # ∂(f/g)/∂θ = (g·∂f/∂θ - f·∂g/∂θ) / g²
            lam_t = solve_adjoint_rhs(Z, QI_target;
                                       solver=solver, preconditioner=nf_preconditioner,
                                       gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter,
                                       gmres_memory=gmres_memory)
            lam_a = solve_adjoint_rhs(Z, QI_total;
                                       solver=solver, preconditioner=nf_preconditioner,
                                       gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter,
                                       gmres_memory=gmres_memory)
            _assert_finite_optimizer_vector(lam_t, "target adjoint solution")
            _assert_finite_optimizer_vector(lam_a, "total-power adjoint solution")
            g_f = gradient_impedance(Mp_eff, I_c, lam_t; reactive=reactive)
            g_g = gradient_impedance(Mp_eff, I_c, lam_a; reactive=reactive)
            _combine_directivity_gradient!(g_f, g_g, J_ratio, g_val)
        catch err
            err isa OverflowError || rethrow()
            J_ratio, g_f = _directivity_ratio_gradient_bigfloat(
                Z,
                Mp_eff,
                Q_target,
                Q_total,
                I_c,
                "accepted iterate";
                reactive=reactive,
            )
        end
        _assert_finite_optimizer_vector(g_f, "directivity gradient")
        gnorm = norm(g_f)
        isfinite(gnorm) ||
            error("directivity gradient norm is non-finite at iteration $iter: $gnorm")

        # For L-BFGS we minimize -J_ratio, so g_lbfgs = -g_true
        g_lbfgs = g_f
        rmul!(g_lbfgs, -1.0)

        push!(trace, (iter=iter, J=J_ratio, gnorm=gnorm))
        verbose && println("  iter=$iter  J_ratio=$(round(J_ratio, sigdigits=6))  |g|=$gnorm")

        if gnorm <= tol
            verbose && println("Converged at iteration $iter")
            break
        end

        # L-BFGS curvature pair update
        if iter > 1
            s_k = theta - theta_old
            y_k = g_lbfgs - g_old
            if m_lbfgs > 0 &&
               _optimizer_curvature_pair_acceptable(s_k, y_k)
                push!(s_list, s_k)
                push!(y_list, y_k)
                if length(s_list) > m_lbfgs
                    popfirst!(s_list)
                    popfirst!(y_list)
                end
            end
        end

        r, history_valid = _optimizer_lbfgs_inverse_product(
            g_lbfgs, s_list, y_list, alpha0)
        if !history_valid
            empty!(s_list)
            empty!(y_list)
        end

        d = r
        rmul!(d, -1.0)

        # Guard: if the L-BFGS two-loop direction is not a descent direction
        # (dot(g, d) ≥ 0, which can happen with stale/indefinite curvature pairs),
        # reset to steepest descent and clear the memory — matches optimize_lbfgs.
        if !_optimizer_dot_is_negative(g_lbfgs, d)
            copyto!(d, g_lbfgs)
            rmul!(d, -1.0)
            empty!(s_list)
            empty!(y_list)
        end

        alpha_ls = 1.0
        copyto!(theta_old, theta)
        copyto!(g_old, g_lbfgs)
        ls_success = false

        # Line search on J_ratio directly
        for ls in 1:20
            @. theta_trial = theta_old + alpha_ls * d
            project!(theta_trial)
            theta_trial == theta_old && break
            if !all(isfinite, theta_trial)
                alpha_ls *= 0.5
                continue
            end
            projected_gd = _projected_directional_derivative(
                g_lbfgs, theta_trial, theta_old)
            if !(isfinite(projected_gd) && projected_gd < 0.0)
                alpha_ls *= 0.5
                continue
            end
            trial_valid = false
            J_trial = -Inf
            try
                assemble_full_Z!(Z_raw, Z_efie, Mp, theta_trial; reactive=reactive)
                Z_trial, _, _ = prepare_conditioned_system(
                    Z_raw,
                    v;
                    regularization_alpha=regularization_alpha,
                    regularization_R=R_mat,
                    preconditioner_M=precond_M_eff,
                    preconditioner_factor=precond_fac,
                )
                I_trial = solve_forward(Z_trial, rhs_eff_base;
                                         solver=solver, preconditioner=nf_preconditioner,
                                         gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter,
                                         gmres_memory=gmres_memory)
                trial_valid = all(isfinite, I_trial)
                if trial_valid
                    try
                        f_trial, g_trial =
                            _directivity_products_and_objectives!(
                                QI_target,
                                QI_total,
                                Q_target,
                                Q_total,
                                I_trial,
                                "trial iterate",
                            )
                        J_trial = _directivity_ratio(
                            f_trial, g_trial, "trial iterate")
                    catch err
                        err isa OverflowError || rethrow()
                        J_trial = _directivity_ratio_bigfloat(
                            Q_target,
                            Q_total,
                            I_trial,
                            "trial iterate",
                        )
                    end
                    trial_valid = true
                end
            catch err
                _recoverable_optimizer_trial_error(err) || rethrow()
                verbose &&
                    println("Line-search trial $ls failed at iteration $iter; reducing step ($err)")
            end

            # Armijo for minimizing -J_ratio
            if trial_valid &&
               -J_trial <= -J_ratio + 1e-4 * projected_gd
                copyto!(theta, theta_trial)
                ls_success = true
                break
            end
            alpha_ls *= 0.5
        end
        if !ls_success
            copyto!(theta, theta_old)
            empty!(s_list)
            empty!(y_list)
            verbose &&
                println("Line search failed at iteration $iter; stopping without accepting trial step")
            break
        end
    end

    return theta, trace
end
