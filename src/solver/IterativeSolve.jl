# IterativeSolve.jl — GMRES iterative solver via Krylov.jl
#
# Provides iterative solve alternatives to the dense direct factorization
# in Solve.jl, with support for near-field sparse preconditioning.

using Krylov

export solve_gmres, solve_gmres_adjoint

@inline function _as_complex_rhs(rhs::AbstractVector{<:Number})
    if rhs isa Vector{ComplexF64}
        return rhs
    end
    return Vector{ComplexF64}(rhs)
end

@inline function _gmres_final_residual(stats)
    return hasproperty(stats, :residuals) && !isempty(stats.residuals) ?
           stats.residuals[end] : NaN
end

function _assert_gmres_converged(stats, label::AbstractString; tol::Float64, maxiter::Int)
    solved = hasproperty(stats, :solved) ? Bool(stats.solved) : false
    solved && return stats
    niter = hasproperty(stats, :niter) ? stats.niter : missing
    status = hasproperty(stats, :status) ? stats.status : "unknown"
    resid = _gmres_final_residual(stats)
    error("$label GMRES did not converge: niter=$niter, status=$status, " *
          "final_residual=$resid, tol=$tol, maxiter=$maxiter")
end

function _assert_gmres_result(x::AbstractVector, stats, label::AbstractString;
                              tol::Float64, maxiter::Int)
    _assert_gmres_converged(stats, label; tol=tol, maxiter=maxiter)
    @inbounds for i in eachindex(x)
        isfinite(x[i]) ||
            error("$label GMRES returned a non-finite solution component at index $i: $(x[i])")
    end
    return stats
end

@inline function _validate_gmres_options(tol::Float64, maxiter::Int,
                                         memory::Int, precond_side::Symbol)
    (isfinite(tol) && tol > 0.0) ||
        throw(ArgumentError("GMRES tol must be finite and positive, got $tol"))
    maxiter >= 1 ||
        throw(ArgumentError("GMRES maxiter must be at least 1, got $maxiter"))
    memory >= 1 ||
        throw(ArgumentError("GMRES memory must be at least 1, got $memory"))
    precond_side in (:left, :right) ||
        throw(ArgumentError("Invalid precond_side: $precond_side (expected :left or :right)"))
    return nothing
end

function _assert_true_residual(A::AbstractMatrix, x::AbstractVector, rhs::AbstractVector,
                               label::AbstractString;
                               tol::Float64,
                               factor::Float64=100.0)
    factor > 0 || error("true residual factor must be positive")
    rhs_c = _as_complex_rhs(rhs)
    relres = norm(A * x - rhs_c) / max(norm(rhs_c), eps(Float64))
    limit = max(factor * tol, sqrt(eps(Float64)))
    relres <= limit && return relres
    error("$label GMRES true residual too large: relative_residual=$relres, " *
          "limit=$limit, tol=$tol, factor=$factor")
end

"""
    solve_gmres(Z, rhs; preconditioner=nothing, precond_side=:left, tol=1e-8,
                maxiter=200, memory=20, verbose=false,
                check_gmres_convergence=true)

Solve Z x = rhs using GMRES from Krylov.jl.

If `preconditioner` is an `AbstractPreconditionerData`, it is applied via:
- `precond_side=:left` (default): left preconditioner M in Krylov.gmres
- `precond_side=:right`: right preconditioner N in Krylov.gmres

Returns `(x, stats)` where `stats` is the Krylov.jl convergence info.
By default, an unconverged or non-finite result throws. Set
`check_gmres_convergence=false` only when intentionally inspecting a partial
iterate and its `stats`.
"""
function solve_gmres(Z::AbstractMatrix{<:Number}, rhs::AbstractVector{<:Number};
                     preconditioner::Union{Nothing, AbstractPreconditionerData}=nothing,
                     precond_side::Symbol=:left,
                     tol::Float64=1e-8,
                     maxiter::Int=200,
                     memory::Int=20,
                     verbose::Bool=false,
                     check_gmres_convergence::Bool=true)
    _validate_gmres_options(tol, maxiter, memory, precond_side)
    rhs_c = _as_complex_rhs(rhs)

    if preconditioner === nothing
        x, stats = Krylov.gmres(Z, rhs_c;
                                 memory=memory,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif precond_side == :right
        N_op = NearFieldOperator(preconditioner)
        x, stats = Krylov.gmres(Z, rhs_c;
                                 N=N_op,
                                 memory=memory,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif precond_side == :left
        M = NearFieldOperator(preconditioner)
        x, stats = Krylov.gmres(Z, rhs_c;
                                 M=M,
                                 memory=memory,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    end
    check_gmres_convergence &&
        _assert_gmres_result(x, stats, "forward";
                             tol=tol, maxiter=maxiter)
    return x, stats
end

"""
    solve_gmres_adjoint(Z, rhs; preconditioner=nothing, precond_side=:left,
                        tol=1e-8, maxiter=200, memory=20, verbose=false,
                        check_gmres_convergence=true)

Solve Z† x = rhs using GMRES, with the adjoint preconditioner Z_nf⁻ᴴ.

This is used for the adjoint linear system in sensitivity analysis:
  Z†(θ) λ = ∂Φ/∂I*

Returns `(x, stats)`. By default, an unconverged or non-finite result throws.
Set `check_gmres_convergence=false` only when intentionally inspecting a
partial iterate and its `stats`.
"""
function solve_gmres_adjoint(Z::AbstractMatrix{<:Number}, rhs::AbstractVector{<:Number};
                              preconditioner::Union{Nothing, AbstractPreconditionerData}=nothing,
                              precond_side::Symbol=:left,
                              tol::Float64=1e-8,
                              maxiter::Int=200,
                              memory::Int=20,
                              verbose::Bool=false,
                              check_gmres_convergence::Bool=true)
    _validate_gmres_options(tol, maxiter, memory, precond_side)
    rhs_c = _as_complex_rhs(rhs)

    if preconditioner === nothing
        x, stats = Krylov.gmres(adjoint(Z), rhs_c;
                                 memory=memory,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif precond_side == :right
        N_adj = NearFieldAdjointOperator(preconditioner)
        x, stats = Krylov.gmres(adjoint(Z), rhs_c;
                                 N=N_adj,
                                 memory=memory,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif precond_side == :left
        M_adj = NearFieldAdjointOperator(preconditioner)
        x, stats = Krylov.gmres(adjoint(Z), rhs_c;
                                 M=M_adj,
                                 memory=memory,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    end
    check_gmres_convergence &&
        _assert_gmres_result(x, stats, "adjoint";
                             tol=tol, maxiter=maxiter)
    return x, stats
end
