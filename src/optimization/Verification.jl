# Verification.jl — Gradient verification via complex-step and finite differences

export complex_step_grad, fd_grad, verify_gradient

"""
    complex_step_grad(f, theta, p; eps=1e-30)

Complex-step derivative of scalar function f(θ) with respect to θ_p.
  ∂f/∂θ_p ≈ Im[f(θ + i ε e_p)] / ε

Requires that f can handle complex-valued θ and is holomorphic in the
perturbed parameter. (For real-valued objectives involving complex
conjugation, e.g. `I' * Q * I`, complex-step is not directly applicable.)
"""
function complex_step_grad(f::Function, theta::Vector{Float64}, p::Int;
                           eps::Float64=1e-30)
    theta_c = complex.(theta)
    theta_c[p] += 1im * eps
    return imag(f(theta_c)) / eps
end

"""
    fd_grad(f, theta, p; h=1e-6, scheme=:central)

Finite-difference derivative of f(θ) with respect to θ_p.
"""
function fd_grad(f::Function, theta::Vector{Float64}, p::Int;
                 h::Float64=1e-6, scheme::Symbol=:central)
    if scheme == :central
        tp = copy(theta); tp[p] += h
        tm = copy(theta); tm[p] -= h
        return (f(tp) - f(tm)) / (2h)
    elseif scheme == :forward
        tp = copy(theta); tp[p] += h
        return (f(tp) - f(theta)) / h
    else
        error("Unknown FD scheme: $scheme")
    end
end

"""
    verify_gradient(f_objective, adjoint_grad, theta;
                    indices=nothing, eps_cs=1e-30, h_fd=1e-6)

Verify adjoint gradient against complex-step and finite-difference.
Returns a DataFrame-like structure with columns: p, adj, cs, fd, rel_err_cs, rel_err_fd.

  f_objective: θ -> J(θ) (must support ComplexF64 input for complex-step)
  adjoint_grad: the adjoint gradient vector g ∈ R^P
  theta: the parameter vector ∈ R^P
  indices: which parameters to check (default: all)
"""
function verify_gradient(f_objective::Function,
                         adjoint_grad::Vector{Float64},
                         theta::Vector{Float64};
                         indices=nothing,
                         eps_cs::Float64=1e-30,
                         h_fd::Float64=1e-6)
    P = length(theta)
    if indices === nothing
        indices = 1:P
    end

    results = Vector{NamedTuple{(:p, :adj, :cs, :fd, :rel_err_cs, :rel_err_fd),
                                Tuple{Int,Float64,Float64,Float64,Float64,Float64}}}()

    for p in indices
        g_adj = adjoint_grad[p]

        # Complex-step
        g_cs = complex_step_grad(f_objective, theta, p; eps=eps_cs)

        # Finite difference
        f_real = θ -> real(f_objective(θ))
        g_fd = fd_grad(f_real, theta, p; h=h_fd)

        denom = max(abs(g_cs), 1e-30)
        rel_cs = abs(g_adj - g_cs) / denom
        rel_fd = abs(g_adj - g_fd) / denom

        push!(results, (p=p, adj=g_adj, cs=g_cs, fd=g_fd,
                        rel_err_cs=rel_cs, rel_err_fd=rel_fd))
    end

    return results
end
