# Solve.jl — Forward and adjoint linear system solves

export solve_forward, solve_system, assemble_full_Z, assemble_full_Z!,
       make_mass_regularizer, make_left_preconditioner,
       select_preconditioner, transform_patch_matrices, prepare_conditioned_system

"""
    solve_forward(Z, v; solver=:direct, preconditioner=nothing, gmres_tol=1e-8, gmres_maxiter=200, verbose_gmres=false)

Solve Z I = v. Uses direct factorization by default, or GMRES when `solver=:gmres`.

# Arguments
- `solver`: `:direct` for LU factorization, `:gmres` for preconditioned GMRES
- `preconditioner`: a preconditioner object (e.g., `AbstractPreconditionerData`), or `nothing`
- `gmres_tol`: relative tolerance for GMRES convergence
- `gmres_maxiter`: maximum GMRES iterations
"""
function solve_forward(Z::AbstractMatrix{<:Number}, v::AbstractVector{<:Number};
                       solver::Symbol=:direct,
                       preconditioner=nothing,
                       gmres_tol::Float64=1e-8,
                       gmres_maxiter::Int=200,
                       verbose_gmres::Bool=false)
    if solver == :direct
        Z isa Matrix || error("Direct solver requires a dense Matrix; use solver=:gmres for operator-based systems.")
        return Z \ v
    elseif solver == :gmres
        x, stats = solve_gmres(Z, v;
                                preconditioner=preconditioner,
                                tol=gmres_tol, maxiter=gmres_maxiter,
                                verbose=verbose_gmres)
        return x
    else
        error("Unknown solver: $solver (expected :direct or :gmres)")
    end
end

"""
    solve_system(Z, rhs; solver=:direct, preconditioner=nothing, gmres_tol=1e-8, gmres_maxiter=200)

General linear solve Z x = rhs with solver dispatch.
"""
function solve_system(Z::AbstractMatrix{<:Number}, rhs::AbstractVector{<:Number};
                      solver::Symbol=:direct,
                      preconditioner=nothing,
                      gmres_tol::Float64=1e-8,
                      gmres_maxiter::Int=200)
    return solve_forward(Z, rhs; solver=solver, preconditioner=preconditioner,
                          gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter)
end

"""
    assemble_full_Z(Z_efie, Mp, theta; reactive=false)

Assemble the full MoM matrix: Z(θ) = Z_efie + Z_imp(θ)

For resistive impedance (default):  Z_imp = -Σ_p θ_p M_p
For reactive impedance:             Z_imp = -Σ_p (iθ_p) M_p
"""
function assemble_full_Z(Z_efie::Matrix{<:Number},
                         Mp::Vector{<:AbstractMatrix},
                         theta::AbstractVector;
                         reactive::Bool=false)
    Z = copy(Z_efie)
    assemble_full_Z!(Z, Z_efie, Mp, theta; reactive=reactive)
    return Z
end

"""
    assemble_full_Z!(Z, Z_efie, Mp, theta; reactive=false)

In-place variant: writes Z(θ) = Z_efie + Z_imp(θ) into pre-allocated `Z`.
"""
function assemble_full_Z!(Z::Matrix{<:Number},
                          Z_efie::Matrix{<:Number},
                          Mp::Vector{<:AbstractMatrix},
                          theta::AbstractVector;
                          reactive::Bool=false)
    copyto!(Z, Z_efie)
    for p in eachindex(theta)
        coeff = reactive ? (1im * theta[p]) : theta[p]
        _add_scaled_matrix!(Z, -coeff, Mp[p])
    end
    return Z
end

"""
    make_mass_regularizer(Mp)

Build a Hermitian positive-semidefinite mass-based regularizer from patch
mass matrices:
  R = Σ_p M_p

Returns a dense `ComplexF64` matrix so it can be used directly in
regularized solves.
"""
function make_mass_regularizer(Mp::Vector{<:AbstractMatrix})
    N = size(Mp[1], 1)
    R = zeros(ComplexF64, N, N)
    for p in eachindex(Mp)
        _add_scaled_matrix!(R, one(ComplexF64), Mp[p])
    end
    # Enforce Hermitian symmetry up to numerical tolerance
    return 0.5 .* (R + R')
end

"""
    make_left_preconditioner(Mp; eps_rel=1e-8)

Build a simple mass-based left preconditioner matrix:
  M = R + ϵ I,  R = Σ_p M_p

`eps_rel` scales the diagonal shift as
  ϵ = eps_rel * max(tr(R)/N, 1).
"""
function make_left_preconditioner(Mp::Vector{<:AbstractMatrix};
                                  eps_rel::Float64=1e-8)
    R = make_mass_regularizer(Mp)
    N = size(R, 1)
    scale = max(real(tr(R)) / N, 1.0)
    ϵ = eps_rel * scale
    return R + (ϵ .* Matrix{ComplexF64}(I, N, N))
end

"""
    select_preconditioner(Mp;
                          mode=:off,
                          preconditioner_M=nothing,
                          n_threshold=256,
                          iterative_solver=false,
                          eps_rel=1e-6)

Select the effective left preconditioner matrix used by the solver.

Modes:
- `:off`: disable preconditioning (unless `preconditioner_M` is provided).
- `:on`: always build/use a mass-based preconditioner.
- `:auto`: enable only when `iterative_solver=true` or `N >= n_threshold`.

If `preconditioner_M` is provided, it takes precedence over `mode`.

Returns `(M_eff, enabled, reason)` where:
- `M_eff` is either a dense `ComplexF64` matrix or `nothing`,
- `enabled` indicates whether preconditioning is active,
- `reason` is a short status string for logging/debugging.
"""
function select_preconditioner(Mp::Vector{<:AbstractMatrix};
                               mode::Symbol=:off,
                               preconditioner_M=nothing,
                               n_threshold::Int=256,
                               iterative_solver::Bool=false,
                               eps_rel::Float64=1e-6)
    mode ∈ (:off, :on, :auto) || error("Invalid preconditioner mode: $mode (expected :off, :on, or :auto)")

    if preconditioner_M !== nothing
        return Matrix{ComplexF64}(preconditioner_M), true, "user-provided preconditioner"
    end

    N = size(Mp[1], 1)
    if mode == :off
        return nothing, false, "mode=:off"
    elseif mode == :on
        return make_left_preconditioner(Mp; eps_rel=eps_rel), true, "mode=:on"
    else
        if iterative_solver
            return make_left_preconditioner(Mp; eps_rel=eps_rel), true, "mode=:auto (iterative_solver=true)"
        elseif N >= n_threshold
            return make_left_preconditioner(Mp; eps_rel=eps_rel), true, "mode=:auto (N=$N >= $n_threshold)"
        else
            return nothing, false, "mode=:auto (N=$N < $n_threshold)"
        end
    end
end

"""
    transform_patch_matrices(Mp; preconditioner_M=nothing, preconditioner_factor=nothing)

Transform derivative blocks under left preconditioning:
  M_p_tilde = M^{-1} M_p

When `preconditioner_M === nothing`, returns `Mp` unchanged.
If `preconditioner_factor` is provided, it is reused instead of factorizing
`preconditioner_M`.

Returns `(Mp_tilde, factor)` where `factor` is `nothing` for the unpreconditioned
case.
"""
function transform_patch_matrices(Mp::Vector{<:AbstractMatrix};
                                  preconditioner_M=nothing,
                                  preconditioner_factor=nothing)
    if preconditioner_M === nothing && preconditioner_factor === nothing
        return Mp, nothing
    end

    fac = preconditioner_factor === nothing ? lu(Matrix{ComplexF64}(preconditioner_M)) : preconditioner_factor
    Mp_tilde = [fac \ Matrix{ComplexF64}(Mp[p]) for p in eachindex(Mp)]
    return Mp_tilde, fac
end

"""
    prepare_conditioned_system(Z_raw, rhs;
                               regularization_alpha=0.0,
                               regularization_R=nothing,
                               preconditioner_M=nothing,
                               preconditioner_factor=nothing)

Build the linear system used by forward/adjoint solves:
  Z_reg = Z_raw + αR
  Z_eff = M^{-1} Z_reg
  rhs_eff = M^{-1} rhs

If no regularization/preconditioning is requested, returns `(Z_raw, rhs, nothing)`.
Returns `(Z_eff, rhs_eff, factor)` where `factor` is the LU factorization used
for preconditioning (or `nothing`).
"""
function prepare_conditioned_system(Z_raw::Matrix{<:Number},
                                    rhs::Vector{<:Number};
                                    regularization_alpha::Float64=0.0,
                                    regularization_R=nothing,
                                    preconditioner_M=nothing,
                                    preconditioner_factor=nothing)
    Z_eff = Matrix{ComplexF64}(Z_raw)
    rhs_eff = Vector{ComplexF64}(rhs)

    if regularization_alpha != 0.0
        regularization_R === nothing &&
            error("regularization_alpha is nonzero but regularization_R is nothing")
        Z_eff .+= regularization_alpha .* Matrix{ComplexF64}(regularization_R)
    end

    if preconditioner_M === nothing && preconditioner_factor === nothing
        return Z_eff, rhs_eff, nothing
    end

    fac = preconditioner_factor === nothing ? lu(Matrix{ComplexF64}(preconditioner_M)) : preconditioner_factor
    Z_eff = fac \ Z_eff
    rhs_eff = fac \ rhs_eff
    return Z_eff, rhs_eff, fac
end
