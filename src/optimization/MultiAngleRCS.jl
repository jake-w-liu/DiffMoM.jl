# MultiAngleRCS.jl — Multi-angle monostatic RCS minimization
#
# Minimizes total weighted backscatter RCS over multiple incidence angles
# using adjoint-based gradients and L-BFGS with box constraints.
#
# Supports any AbstractMatrix base operator (MLFMA, ACA, dense) via
# ImpedanceLoadedOperator for the composite system Z(θ) = Z_base + Z_imp(θ).
#
# Objective:  J(θ) = Σ_a w_a (I_a† Q_a I_a)
# Gradient:   g[p] = Σ_a w_a · 2 Re{ λ_a† M_p I_a }
#             where Z(θ)† λ_a = Q_a I_a

export AngleConfig, build_multiangle_configs, optimize_multiangle_rcs

"""
    AngleConfig

Configuration for one incidence angle in a multi-angle RCS optimization.
"""
struct AngleConfig
    k_vec::Vec3                     # Incidence wave vector (rad/m)
    pol::Vec3                       # Polarization (unit vector)
    v::Vector{ComplexF64}           # Pre-assembled excitation vector
    Q::AbstractMatrix{ComplexF64}    # Backscatter Q operator for this angle
    weight::Float64                 # Weight in composite objective
end

function _multiangle_objective_scales(J_angles::Vector{Float64},
                                      weights::Vector{Float64},
                                      objective::Symbol,
                                      reference_objectives::Vector{Float64},
                                      smooth_beta::Float64)
    M = length(J_angles)
    length(weights) == M ||
        error("weights length $(length(weights)) does not match objective length $M")
    length(reference_objectives) == M ||
        error("reference_objectives length $(length(reference_objectives)) does not match objective length $M")
    all(isfinite, J_angles) || error("per-angle objective contains non-finite values")
    all(isfinite, weights) || error("objective weights must be finite")
    all(isfinite, reference_objectives) || error("reference objectives must be finite")
    all(reference_objectives .> 0) || error("reference objectives must be positive")

    tiny = 1e-300
    J_safe = max.(J_angles, tiny)

    if objective == :linear
        return sum(weights .* J_angles), copy(weights)
    elseif objective == :sum_log
        all(weights .>= 0) || error("sum_log objective weights must be nonnegative")
        return sum(weights .* log.(J_safe ./ reference_objectives)), weights ./ J_safe
    elseif objective == :smoothmax_log
        smooth_beta > 0 || error("smooth_beta must be positive")
        all(weights .> 0) || error("smoothmax_log objective weights must be positive")
        z = log.(J_safe ./ reference_objectives)
        u = smooth_beta .* z .+ log.(weights)
        umax = maximum(u)
        expu = exp.(u .- umax)
        denom = sum(expu)
        probs = expu ./ denom
        value = (umax + log(denom)) / smooth_beta
        return value, probs ./ J_safe
    else
        error("Unknown multi-angle objective: $objective (expected :linear, :sum_log, or :smoothmax_log)")
    end
end

function _default_transverse_pol(khat::Vec3)
    ref = abs(khat[1]) < 0.9 ? Vec3(1.0, 0.0, 0.0) : Vec3(0.0, 1.0, 0.0)
    p = ref - dot(ref, khat) * khat
    return p / norm(p)
end

function _transverse_unit_pol(khat::Vec3, pol::Vec3)
    all(isfinite, pol) || error("Incident polarization must be finite.")
    p = pol - dot(pol, khat) * khat
    pn = norm(p)
    pn > 1e-12 ||
        error("Incident polarization must have a nonzero transverse component.")
    return p / pn
end

"""
    build_multiangle_configs(mesh, rwg, k, angles; grid, backscatter_cone=10.0, matrix_free_Q=false, rcs_component=:copol)

Build `AngleConfig` entries for multi-angle monostatic RCS optimization.

# Arguments
- `mesh`, `rwg`: mesh and RWG basis data
- `k`: wavenumber (rad/m)
- `angles`: vector of named tuples, each with fields:
  - `theta_inc`: polar angle of incidence (radians, from +z)
  - `phi_inc`: azimuthal angle of incidence (radians, from +x)
  - `pol`: polarization unit vector (Vec3)
  - `weight`: weight in objective (default 1.0)
- `grid`: `SphGrid` for far-field evaluation (shared across all angles)
- `backscatter_cone`: half-angle in degrees for backscatter mask (default 15°)
- `matrix_free_Q`: if true, store a matrix-free far-field objective operator
- `rcs_component`: `:copol` for θ-polarized RCS, `:crosspol` for φ-polarized
  RCS, or `:total` for the sum of both tangential components

# Returns
Vector of `AngleConfig`, one per incidence angle.
"""
function build_multiangle_configs(mesh::TriMesh, rwg::RWGData, k::Float64,
                                   angles::Vector{<:NamedTuple};
                                   grid::SphGrid,
                                   backscatter_cone::Float64=15.0,
                                   matrix_free_Q::Bool=false,
                                   rcs_component::Symbol=:copol)
    eta0 = 376.730313668
    G_mat = radiation_vectors(mesh, rwg, grid, k; eta0=eta0)
    pol_theta = pol_linear_x(grid)  # θ̂ polarization
    pol_phi = pol_linear_y(grid)    # φ̂ polarization
    rcs_component in (:copol, :crosspol, :total) ||
        error("rcs_component must be :copol, :crosspol, or :total")

    configs = AngleConfig[]
    for ang in angles
        θ_i = ang.theta_inc
        φ_i = ang.phi_inc

        # Incidence direction: k̂ = (sinθ cosφ, sinθ sinφ, cosθ)
        khat = Vec3(sin(θ_i) * cos(φ_i), sin(θ_i) * sin(φ_i), cos(θ_i))
        k_vec = k * khat

        # Backscatter direction = -k̂
        bs_dir = -khat

        # Excitation
        pw_pol_raw = hasfield(typeof(ang), :pol) ? Vec3(ang.pol) : _default_transverse_pol(khat)
        pw_pol = _transverse_unit_pol(khat, pw_pol_raw)
        E0 = 1.0
        v = assemble_excitation(mesh, rwg, PlaneWaveExcitation(k_vec, E0, pw_pol))

        # Q matrix targeting backscatter direction
        mask = direction_mask(grid, bs_dir; half_angle=backscatter_cone * π / 180)

        # Build per-angle polarization objective. The default θ̂ component keeps
        # historical co-pol behavior. The total option matches bistatic_rcs,
        # which sums both tangential far-field components.
        Q = if rcs_component == :copol
            matrix_free_Q ?
                build_Q_operator(G_mat, grid, pol_theta; mask=mask) :
                build_Q(G_mat, grid, pol_theta; mask=mask)
        elseif rcs_component == :crosspol
            matrix_free_Q ?
                build_Q_operator(G_mat, grid, pol_phi; mask=mask) :
                build_Q(G_mat, grid, pol_phi; mask=mask)
        else
            if matrix_free_Q
                sum_q_matrix(
                    build_Q_operator(G_mat, grid, pol_theta; mask=mask),
                    build_Q_operator(G_mat, grid, pol_phi; mask=mask),
                )
            else
                build_Q(G_mat, grid, pol_theta; mask=mask) +
                build_Q(G_mat, grid, pol_phi; mask=mask)
            end
        end

        w = hasfield(typeof(ang), :weight) ? ang.weight : 1.0

        push!(configs, AngleConfig(k_vec, pw_pol, v, Q, w))
    end

    return configs
end

"""
    optimize_multiangle_rcs(Z_base, Mp, configs, theta0; kwargs...)

Minimize total weighted backscatter RCS over multiple incidence angles
using projected L-BFGS.

Supports any `AbstractMatrix{ComplexF64}` as base operator (MLFMA, ACA, dense).
Uses `ImpedanceLoadedOperator` internally to build Z(θ) = Z_base + Z_imp(θ).

# Arguments
- `Z_base`: base EFIE operator (MLFMAOperator, ACAOperator, or dense Matrix)
- `Mp`: vector of sparse patch mass matrices
- `configs`: vector of `AngleConfig` from `build_multiangle_configs`
- `theta0`: initial impedance parameter vector

# Keyword arguments
- `maxiter`, `tol`, `m_lbfgs`, `alpha0`: L-BFGS parameters
- `reactive`: impedance mode (false=resistive, true=reactive)
- `lb`, `ub`: box constraints on θ
- `preconditioner`: `AbstractPreconditionerData` for GMRES
- `gmres_tol`, `gmres_maxiter`, `gmres_memory`: GMRES parameters
- `objective`: `:linear` for Σw_aJ_a, `:sum_log` for Σw_a log(J_a/J_ref,a),
  or `:smoothmax_log` for a smooth worst-angle normalized log objective
- `reference_objectives`: positive per-angle reference values for normalized
  objectives, typically the PEC objective values
- `smooth_beta`: sharpness parameter for `:smoothmax_log`
- `verbose`: print progress

# Returns
`(theta_opt, trace)` where trace records `(iter, J, gnorm, n_fwd, n_adj)` per iteration.
"""
function optimize_multiangle_rcs(Z_base::AbstractMatrix{ComplexF64},
                                  Mp::Vector{<:AbstractMatrix},
                                  configs::Vector{AngleConfig},
                                  theta0::Vector{Float64};
                                  maxiter::Int=100,
                                  tol::Float64=1e-10,
                                  m_lbfgs::Int=10,
                                  alpha0::Float64=0.01,
                                  verbose::Bool=true,
                                  reactive::Bool=false,
                                  lb=nothing,
                                  ub=nothing,
                                  preconditioner::Union{Nothing, AbstractPreconditionerData}=nothing,
                                  gmres_tol::Float64=1e-6,
                                  gmres_maxiter::Int=300,
                                  gmres_memory::Int=20,
                                  objective::Symbol=:linear,
                                  reference_objectives::Union{Nothing, Vector{Float64}}=nothing,
                                  smooth_beta::Float64=8.0)
    M = length(configs)    # number of angles
    P = length(theta0)     # number of design parameters
    theta = copy(theta0)

    # Use dense LU when Z_base is a dense Matrix for exact solves;
    # fall back to GMRES for matrix-free operators (MLFMA, ACA)
    use_dense_lu = Z_base isa Matrix{ComplexF64}
    solver = use_dense_lu ? :direct : :gmres

    function project!(x)
        lb !== nothing && (x .= max.(x, lb))
        ub !== nothing && (x .= min.(x, ub))
        return x
    end
    project!(theta)
    objective in (:linear, :sum_log, :smoothmax_log) ||
        error("Unknown multi-angle objective: $objective (expected :linear, :sum_log, or :smoothmax_log)")
    weights = [cfg.weight for cfg in configs]
    refs = reference_objectives === nothing ? ones(Float64, M) : copy(reference_objectives)
    objective == :linear || all(refs .> 0) ||
        error("reference_objectives must be positive for normalized objectives")

    if verbose
        println("Multi-angle RCS optimization: $M angles, $P parameters, solver=$solver, objective=$objective")
        if !use_dense_lu && preconditioner !== nothing
            println("  GMRES preconditioned")
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
    g_old = zeros(P)

    # Pre-allocate Z buffer for in-place assembly (dense LU path)
    Z_buf = use_dense_lu ? Matrix{ComplexF64}(undef, size(Z_base)...) : Matrix{ComplexF64}(undef, 0, 0)

    for iter in 1:maxiter
        # ── 1. Build system matrix Z(θ) ──────────────────────────
        if use_dense_lu
            assemble_full_Z!(Z_buf, Z_base, Mp, theta; reactive=reactive)
            Z_full = Z_buf
            Z_factor = lu(Z_full)
        else
            Z_full = ImpedanceLoadedOperator(Z_base, Mp, theta, reactive)
            Z_factor = nothing
        end

        # ── 2. Forward solves: I_a = Z(θ)⁻¹ v_a ────────────────
        I_all = Vector{Vector{ComplexF64}}(undef, M)
        for a in 1:M
            I_all[a] = if use_dense_lu
                Z_factor \ configs[a].v
            else
                solve_forward(Z_full, configs[a].v;
                              solver=solver,
                              preconditioner=preconditioner,
                              gmres_tol=gmres_tol,
                              gmres_maxiter=gmres_maxiter,
                              gmres_memory=gmres_memory)
            end
            n_fwd_solves += 1
        end

        # ── 3. Objective and scalarization weights ───────────────
        QI_all = Vector{Vector{ComplexF64}}(undef, M)
        J_angles = zeros(Float64, M)
        for a in 1:M
            QI_all[a] = Vector{ComplexF64}(configs[a].Q * I_all[a])
            J_angles[a] = real(dot(I_all[a], QI_all[a]))
        end
        J_val, objective_scales = _multiangle_objective_scales(
            J_angles, weights, objective, refs, smooth_beta)

        # ── 4. Adjoint solves: Z(θ)† λ_a = Q_a I_a ────────────
        lambda_all = Vector{Vector{ComplexF64}}(undef, M)
        for a in 1:M
            rhs_a = QI_all[a]
            lambda_all[a] = if use_dense_lu
                Z_factor' \ Vector{ComplexF64}(rhs_a)
            else
                solve_adjoint_rhs(Z_full, rhs_a;
                                  solver=solver,
                                  preconditioner=preconditioner,
                                  gmres_tol=gmres_tol,
                                  gmres_maxiter=gmres_maxiter,
                                  gmres_memory=gmres_memory)
            end
            n_adj_solves += 1
        end

        # ── 5. Gradient: g[p] = Σ_a scalar_a · ∂J_a/∂θ_p ───────
        g = zeros(P)
        for a in 1:M
            g_a = gradient_impedance(Mp, I_all[a], lambda_all[a]; reactive=reactive)
            g .+= objective_scales[a] .* g_a
        end
        gnorm = norm(g)

        push!(trace, (iter=iter, J=J_val, gnorm=gnorm, n_fwd=n_fwd_solves, n_adj=n_adj_solves))
        if verbose
            println("  iter=$iter  J=$(round(J_val, sigdigits=6))  |g|=$(round(gnorm, sigdigits=4))  solves(fwd=$n_fwd_solves, adj=$n_adj_solves)")
        end

        if gnorm < tol
            verbose && println("Converged at iteration $iter (gradient < tol)")
            break
        end

        # Stagnation detection: stop if J hasn't improved by >0.1% in 10 iterations
        if length(trace) >= 11
            J_10_ago = trace[end-10].J
            if abs(J_val - J_10_ago) / max(abs(J_10_ago), 1e-30) < 1e-3
                verbose && println("Stagnated at iteration $iter (J unchanged for 10 iters)")
                break
            end
        end

        # ── 6. L-BFGS curvature pair update ─────────────────────
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

        # ── 7. Two-loop recursion ────────────────────────────────
        q = copy(g)
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

        # ── 8. Projected line search ─────────────────────────────
        d = -r
        alpha_ls = 1.0
        theta_old = copy(theta)
        g_old = copy(g)

        ls_success = false

        # Check if L-BFGS direction is descent; if not, use steepest descent
        gd = dot(g, d)
        if gd >= 0
            d = -g
            gd = -gnorm^2
            empty!(s_list)
            empty!(y_list)
        end

        for ls in 1:20
            theta_trial = project!(theta_old + alpha_ls * d)
            if use_dense_lu
                assemble_full_Z!(Z_buf, Z_base, Mp, theta_trial; reactive=reactive)
                Z_trial = Z_buf
                Z_trial_factor = lu(Z_trial)
            else
                Z_trial = ImpedanceLoadedOperator(Z_base, Mp, theta_trial, reactive)
                Z_trial_factor = nothing
            end

            # Evaluate trial objective
            J_trial_angles = zeros(Float64, M)
            for a in 1:M
                I_trial = if use_dense_lu
                    Z_trial_factor \ configs[a].v
                else
                    solve_forward(Z_trial, configs[a].v;
                                  solver=solver,
                                  preconditioner=preconditioner,
                                  gmres_tol=gmres_tol,
                                  gmres_maxiter=gmres_maxiter,
                                  gmres_memory=gmres_memory)
                end
                n_fwd_solves += 1
                QI_trial = configs[a].Q * I_trial
                J_trial_angles[a] = real(dot(I_trial, QI_trial))
            end
            J_trial, _ = _multiangle_objective_scales(
                J_trial_angles, weights, objective, refs, smooth_beta)

            # Armijo condition
            if J_trial <= J_val + 1e-4 * alpha_ls * gd
                theta = theta_trial
                ls_success = true
                break
            end
            alpha_ls *= 0.5
        end

        if !ls_success
            theta .= theta_old
            empty!(s_list)
            empty!(y_list)
            verbose && println("Line search failed at iteration $iter; stopping without accepting trial step")
            break
        end
    end

    return theta, trace
end
