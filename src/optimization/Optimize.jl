# Optimize.jl — L-BFGS optimization loop with optional box constraints

export optimize_lbfgs, optimize_directivity

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
                        gmres_maxiter::Int=200)
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
        regularization_R === nothing ? make_mass_regularizer(Mp) : Matrix{ComplexF64}(regularization_R)
    end

    precond_M_eff, precond_enabled, precond_reason = select_preconditioner(
        Mp;
        mode=preconditioning,
        preconditioner_M=preconditioner_M,
        n_threshold=auto_precondition_n_threshold,
        iterative_solver=iterative_solver,
        eps_rel=auto_precondition_eps_rel,
    )
    verbose && (preconditioning == :auto || preconditioner_M !== nothing || precond_enabled) &&
        println("  preconditioning: ", precond_enabled ? "ON ($precond_reason)" : "OFF ($precond_reason)")

    Mp_eff, precond_fac = transform_patch_matrices(
        Mp;
        preconditioner_M=precond_M_eff,
    )
    rhs_eff_base = precond_fac === nothing ? Vector{ComplexF64}(v) : (precond_fac \ Vector{ComplexF64}(v))

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

    theta_old = copy(theta)
    g_old = zeros(length(theta))

    # Pre-allocate Z_raw buffer for in-place assembly
    N_sys = size(Z_efie, 1)
    Z_raw = Matrix{ComplexF64}(undef, N_sys, N_sys)

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
                                  gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter)
        n_fwd_solves += 1

        # Objective (always report the true J)
        J_val = real(dot(I_coeffs, Q * I_coeffs))

        # Adjoint solve
        lambda = solve_adjoint(Z, Q, I_coeffs;
                                solver=solver, preconditioner=nf_preconditioner,
                                gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter)
        n_adj_solves += 1

        # Gradient of J (true objective)
        g_true = gradient_impedance(Mp_eff, I_coeffs, lambda; reactive=reactive)

        # For minimization: g = g_true; for maximization: g = -g_true
        g = sense .* g_true
        gnorm = norm(g_true)

        push!(trace, (iter=iter, J=J_val, gnorm=gnorm, n_fwd=n_fwd_solves, n_adj=n_adj_solves))
        if verbose
            println("  iter=$iter  J=$J_val  |g|=$gnorm  solves(fwd=$n_fwd_solves, adj=$n_adj_solves)")
        end

        if gnorm < tol
            if verbose
                println("Converged at iteration $iter")
            end
            break
        end

        # L-BFGS curvature pair update
        if iter > 1
            s_k = theta - theta_old
            y_k = g - g_old
            sy = dot(s_k, y_k)
            if sy > 1e-30
                push!(s_list, s_k)
                push!(y_list, y_k)
                if length(s_list) > m_lbfgs
                    popfirst!(s_list)
                    popfirst!(y_list)
                end
            end
        end

        # Two-loop recursion
        q = copy(g)
        m_cur = length(s_list)
        alpha_vec = zeros(m_cur)

        for i in m_cur:-1:1
            rho_i = 1.0 / dot(y_list[i], s_list[i])
            alpha_vec[i] = rho_i * dot(s_list[i], q)
            q .-= alpha_vec[i] .* y_list[i]
        end

        if m_cur > 0
            gamma = dot(s_list[end], y_list[end]) / dot(y_list[end], y_list[end])
        else
            gamma = alpha0
        end
        r = gamma .* q

        for i in 1:m_cur
            rho_i = 1.0 / dot(y_list[i], s_list[i])
            beta = rho_i * dot(y_list[i], r)
            r .+= (alpha_vec[i] - beta) .* s_list[i]
        end

        # Line search: backtracking Armijo with projection
        d = -r

        # Check if L-BFGS direction is descent; if not, reset to steepest descent
        gd = dot(g, d)
        if gd >= 0
            d = -g
            gd = -gnorm^2
            empty!(s_list)
            empty!(y_list)
        end

        alpha = 1.0
        theta_old = copy(theta)
        g_old = copy(g)
        J_internal = sense * J_val

        for ls in 1:20
            theta_trial = project!(theta_old + alpha * d)
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
                                     gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter)
            n_fwd_solves += 1
            J_trial_internal = sense * real(dot(I_trial, Q * I_trial))

            if J_trial_internal <= J_internal + 1e-4 * alpha * dot(g, d)
                theta = theta_trial
                break
            end
            alpha *= 0.5
            if ls == 20
                theta = project!(theta_old + alpha * d)
            end
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
                              gmres_maxiter::Int=200)
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
        regularization_R === nothing ? make_mass_regularizer(Mp) : Matrix{ComplexF64}(regularization_R)
    end

    precond_M_eff, precond_enabled, precond_reason = select_preconditioner(
        Mp;
        mode=preconditioning,
        preconditioner_M=preconditioner_M,
        n_threshold=auto_precondition_n_threshold,
        iterative_solver=iterative_solver,
        eps_rel=auto_precondition_eps_rel,
    )
    verbose && (preconditioning == :auto || preconditioner_M !== nothing || precond_enabled) &&
        println("  preconditioning: ", precond_enabled ? "ON ($precond_reason)" : "OFF ($precond_reason)")

    Mp_eff, precond_fac = transform_patch_matrices(
        Mp;
        preconditioner_M=precond_M_eff,
    )
    rhs_eff_base = precond_fac === nothing ? Vector{ComplexF64}(v) : (precond_fac \ Vector{ComplexF64}(v))

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

    theta_old = copy(theta)
    g_old = zeros(length(theta))

    # Pre-allocate Z_raw buffer for in-place assembly
    N_sys = size(Z_efie, 1)
    Z_raw = Matrix{ComplexF64}(undef, N_sys, N_sys)

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
                             gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter)

        # Directivity ratio
        f_val = real(dot(I_c, Q_target * I_c))
        g_val = real(dot(I_c, Q_total * I_c))
        J_ratio = f_val / g_val

        # Two separate adjoint solves for numerically stable ratio gradient
        # ∂(f/g)/∂θ = (g·∂f/∂θ - f·∂g/∂θ) / g²
        lam_t = solve_adjoint(Z, Q_target, I_c;
                               solver=solver, preconditioner=nf_preconditioner,
                               gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter)
        lam_a = solve_adjoint(Z, Q_total, I_c;
                               solver=solver, preconditioner=nf_preconditioner,
                               gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter)
        g_f = gradient_impedance(Mp_eff, I_c, lam_t; reactive=reactive)
        g_g = gradient_impedance(Mp_eff, I_c, lam_a; reactive=reactive)
        g_true = (g_val .* g_f .- f_val .* g_g) ./ (g_val^2)

        # For L-BFGS we minimize -J_ratio, so g_lbfgs = -g_true
        g_lbfgs = -g_true
        gnorm = norm(g_true)

        push!(trace, (iter=iter, J=J_ratio, gnorm=gnorm))
        verbose && println("  iter=$iter  J_ratio=$(round(J_ratio, sigdigits=6))  |g|=$gnorm")

        if gnorm < tol
            verbose && println("Converged at iteration $iter")
            break
        end

        # L-BFGS curvature pair update
        if iter > 1
            s_k = theta - theta_old
            y_k = g_lbfgs - g_old
            sy = dot(s_k, y_k)
            if sy > 1e-30
                push!(s_list, s_k)
                push!(y_list, y_k)
                if length(s_list) > m_lbfgs
                    popfirst!(s_list)
                    popfirst!(y_list)
                end
            end
        end

        # Two-loop recursion
        q = copy(g_lbfgs)
        m_cur = length(s_list)
        alpha_vec = zeros(m_cur)

        for i in m_cur:-1:1
            rho_i = 1.0 / dot(y_list[i], s_list[i])
            alpha_vec[i] = rho_i * dot(s_list[i], q)
            q .-= alpha_vec[i] .* y_list[i]
        end

        gamma = m_cur > 0 ? dot(s_list[end], y_list[end]) / dot(y_list[end], y_list[end]) : alpha0
        r = gamma .* q

        for i in 1:m_cur
            rho_i = 1.0 / dot(y_list[i], s_list[i])
            beta = rho_i * dot(y_list[i], r)
            r .+= (alpha_vec[i] - beta) .* s_list[i]
        end

        d = -r
        alpha_ls = 1.0
        theta_old = copy(theta)
        g_old = copy(g_lbfgs)

        # Line search on J_ratio directly
        for ls in 1:20
            theta_trial = project!(theta_old + alpha_ls * d)
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
                                     gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter)
            f_trial = real(dot(I_trial, Q_target * I_trial))
            g_trial = real(dot(I_trial, Q_total * I_trial))
            J_trial = f_trial / g_trial

            # Armijo for minimizing -J_ratio
            if -J_trial <= -J_ratio + 1e-4 * alpha_ls * dot(g_lbfgs, d)
                theta = theta_trial
                break
            end
            alpha_ls *= 0.5
            if ls == 20
                theta = project!(theta_old + alpha_ls * d)
            end
        end
    end

    return theta, trace
end
