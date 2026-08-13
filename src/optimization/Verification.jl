# Verification.jl — Gradient verification via complex-step and finite differences

export complex_step_grad, fd_grad, verify_gradient

const _GradientVerificationRow =
    NamedTuple{(:p, :adj, :cs, :fd, :rel_err_cs, :rel_err_fd),
               Tuple{Int,Float64,Float64,Float64,Float64,Float64}}

@inline function _validate_gradient_parameters(theta::Vector{Float64},
                                               p::Int)
    all(isfinite, theta) ||
        throw(ArgumentError("theta must contain only finite values"))
    checkbounds(Bool, theta, p) ||
        throw(ArgumentError(
            "parameter index p=$p is outside 1:$(length(theta))"))
    return nothing
end

@inline function _validate_gradient_step(step::Float64,
                                         label::AbstractString)
    (isfinite(step) && step > 0.0) ||
        throw(ArgumentError("$label must be finite and positive, got $step"))
    return nothing
end

@inline function _validated_gradient_value(value,
                                           label::AbstractString)
    value isa Number ||
        throw(ArgumentError(
            "$label must return a scalar Number, got $(typeof(value))"))
    isfinite(value) ||
        throw(ArgumentError("$label returned a non-finite value: $value"))
    return value
end

function _complex_step_grad_validated(f::Function,
                                      theta::Vector{Float64},
                                      p::Int,
                                      step::Float64)
    theta_c = complex.(theta)
    baseline = _validated_gradient_value(
        f(theta_c), "complex-step objective")
    iszero(imag(baseline)) ||
        throw(ArgumentError(
            "complex-step requires an objective that is real at real theta; " *
            "the baseline value was $baseline"))

    trial_step = step
    # A valid analytic derivative may be representable even when f's imaginary
    # response at the requested step is not.  Grow the perturbation by exact
    # powers of two until a signal appears; this preserves the complex-step
    # formula for linear/local analytic behavior and avoids certifying a false
    # zero caused solely by Float64 underflow.
    previous_derivative = nothing
    for attempt in 1:64
        theta_c[p] = ComplexF64(theta[p], trial_step)
        perturbed = _validated_gradient_value(
            f(theta_c), "complex-step objective")
        signal = imag(perturbed)
        if !iszero(signal)
            derivative = signal / trial_step
            isfinite(derivative) ||
                throw(OverflowError(
                    "complex-step derivative is non-finite at parameter $p"))
            # A first nonzero subnormal signal can still carry very few
            # significant bits. Increase the exact power-of-two step until
            # two successive quotients agree; this preserves a linear
            # complex-step derivative while avoiding a noisy early return.
            previous_derivative !== nothing &&
                derivative == previous_derivative && return derivative
            previous_derivative = derivative
        end
        attempt == 64 && break
        next_step = ldexp(trial_step, 16)
        isfinite(next_step) || break
        trial_step = next_step
    end
    previous_derivative !== nothing && return previous_derivative
    throw(ArgumentError(
        "complex-step response remained exactly zero after adaptive " *
        "representable perturbations at parameter $p; the derivative is " *
        "inconclusive"))
end

function _fd_grad_validated(f::Function,
                            theta::Vector{Float64},
                            p::Int,
                            step::Float64,
                            scheme::Symbol)
    base = theta[p]
    plus = base + step
    isfinite(plus) ||
        throw(ArgumentError(
            "theta[$p] + h is non-finite; reduce h or rescale theta"))
    plus != base ||
        throw(ArgumentError(
            "h=$step is too small to perturb theta[$p]=$base"))

    derivative = if scheme == :central
        minus = base - step
        isfinite(minus) ||
            throw(ArgumentError(
                "theta[$p] - h is non-finite; reduce h or rescale theta"))
        minus != base ||
            throw(ArgumentError(
                "h=$step is too small to perturb theta[$p]=$base"))
        denominator = plus - minus
        isfinite(denominator) && denominator > 0.0 ||
            throw(ArgumentError(
                "the representable central-difference perturbation is invalid"))

        tp = copy(theta)
        tm = copy(theta)
        tp[p] = plus
        tm[p] = minus
        fp = _validated_gradient_value(
            f(tp), "finite-difference objective")
        fm = _validated_gradient_value(
            f(tm), "finite-difference objective")
        (fp - fm) / denominator
    elseif scheme == :forward
        tp = copy(theta)
        tp[p] = plus
        fp = _validated_gradient_value(
            f(tp), "finite-difference objective")
        f0 = _validated_gradient_value(
            f(theta), "finite-difference objective")
        (fp - f0) / (plus - base)
    else
        throw(ArgumentError(
            "Unknown FD scheme: $scheme (expected :central or :forward)"))
    end

    isfinite(derivative) ||
        throw(OverflowError(
            "finite-difference derivative is non-finite at parameter $p"))
    return derivative
end

"""
    complex_step_grad(f, theta, p; eps=1e-30)

Complex-step derivative of a real scalar function `f(θ)` with respect to
`θ_p`:

  ∂f/∂θ_p ≈ Im[f(θ + i ε e_p)] / ε

Requires that `f` can handle complex-valued `θ`, is holomorphic in the
perturbed parameter, and is real when `θ` is real. For real-valued
objectives involving complex conjugation, e.g. `I' * Q * I`, complex-step
is not directly applicable.
"""
function complex_step_grad(f::Function, theta::Vector{Float64}, p::Int;
                           eps::Float64=1e-30)
    _validate_gradient_parameters(theta, p)
    _validate_gradient_step(eps, "eps")
    return _complex_step_grad_validated(f, theta, p, eps)
end

"""
    fd_grad(f, theta, p; h=1e-6, scheme=:central)

Finite-difference derivative of scalar function `f(θ)` with respect to
`θ_p`. `h` must be finite, positive, and large enough to change `θ_p` in
`Float64` arithmetic.
"""
function fd_grad(f::Function, theta::Vector{Float64}, p::Int;
                 h::Float64=1e-6, scheme::Symbol=:central)
    _validate_gradient_parameters(theta, p)
    _validate_gradient_step(h, "h")
    return _fd_grad_validated(f, theta, p, h, scheme)
end

function _validated_gradient_indices(indices, P::Int)
    indices === nothing && return Base.OneTo(P)
    applicable(iterate, indices) ||
        throw(ArgumentError("indices must be an iterable of integer indices"))

    checked = Int[]
    for raw_index in indices
        (raw_index isa Integer && !(raw_index isa Bool)) ||
            throw(ArgumentError(
                "indices must contain integers, got $(repr(raw_index))"))
        1 <= raw_index <= P ||
            throw(ArgumentError(
                "gradient index $raw_index is outside 1:$P"))
        push!(checked, Int(raw_index))
    end
    return checked
end

"""
    verify_gradient(f_objective, adjoint_grad, theta;
                    indices=nothing, eps_cs=1e-30, h_fd=1e-6)

Verify an adjoint gradient against complex-step and central finite
differences. Returns a vector of named tuples with fields `p`, `adj`, `cs`,
`fd`, `rel_err_cs`, and `rel_err_fd`.

  f_objective: `θ -> J(θ)` (must support `ComplexF64` input for complex-step)
  adjoint_grad: the adjoint gradient vector `g ∈ R^P`
  theta: the parameter vector `θ ∈ R^P`
  indices: which parameters to check (default: all)
"""
function verify_gradient(f_objective::Function,
                         adjoint_grad::Vector{Float64},
                         theta::Vector{Float64};
                         indices=nothing,
                         eps_cs::Float64=1e-30,
                         h_fd::Float64=1e-6)
    P = length(theta)
    length(adjoint_grad) == P ||
        throw(DimensionMismatch(
            "adjoint_grad length $(length(adjoint_grad)) != theta length $P"))
    all(isfinite, theta) ||
        throw(ArgumentError("theta must contain only finite values"))
    all(isfinite, adjoint_grad) ||
        throw(ArgumentError(
            "adjoint_grad must contain only finite values"))
    _validate_gradient_step(eps_cs, "eps_cs")
    _validate_gradient_step(h_fd, "h_fd")
    checked_indices = _validated_gradient_indices(indices, P)

    results = Vector{_GradientVerificationRow}(
        undef, length(checked_indices))
    f_real = θ -> begin
        value = _validated_gradient_value(
            f_objective(θ), "gradient-verification objective")
        iszero(imag(value)) ||
            throw(ArgumentError(
                "finite differences require a real objective at real theta; " *
                "the objective returned $value"))
        return real(value)
    end

    @inbounds for (result_index, p) in enumerate(checked_indices)
        g_adj = adjoint_grad[p]
        g_cs = _complex_step_grad_validated(
            f_objective, theta, p, eps_cs)
        g_fd = _fd_grad_validated(
            f_real, theta, p, h_fd, :central)

        # Complex-step can be invalid for non-holomorphic objectives, so its
        # value is not used to normalize the independent adjoint-vs-FD check.
        rel_cs = abs(g_adj - g_cs) / max(abs(g_cs), 1e-30)
        fd_scale = max(abs(g_fd), abs(g_adj), 1e-30)
        rel_fd = abs(g_adj / fd_scale - g_fd / fd_scale)

        results[result_index] = (
            p=p,
            adj=g_adj,
            cs=g_cs,
            fd=g_fd,
            rel_err_cs=rel_cs,
            rel_err_fd=rel_fd,
        )
    end

    return results
end
