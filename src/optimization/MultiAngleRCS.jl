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

function _validate_multiangle_q(Q::AbstractMatrix{ComplexF64},
                                label::AbstractString)
    if Q isa FarFieldQMatrix
        size(Q.G_mat) == (3 * length(Q.weights), Q.N) ||
            throw(DimensionMismatch(
                "$label FarFieldQMatrix storage is inconsistent"))
        size(Q.pol) == (3, length(Q.weights)) ||
            throw(DimensionMismatch(
                "$label FarFieldQMatrix polarization storage is inconsistent"))
        Q.mask === nothing || length(Q.mask) == length(Q.weights) ||
            throw(DimensionMismatch(
                "$label FarFieldQMatrix mask storage is inconsistent"))
        all(isfinite, Q.G_mat) ||
            throw(ArgumentError("$label radiation vectors must be finite"))
        all(isfinite, Q.weights) ||
            throw(ArgumentError("$label quadrature weights must be finite"))
        all(isfinite, Q.pol) ||
            throw(ArgumentError("$label polarization vectors must be finite"))
    elseif Q isa SumQMatrix
        _validate_multiangle_q(Q.A, "$label first summand")
        _validate_multiangle_q(Q.B, "$label second summand")
    else
        _validate_known_matrix_entries(Q, label)
    end
    return nothing
end

function _validate_angle_config(config::AngleConfig, index::Int, N::Int)
    all(isfinite, config.k_vec) ||
        throw(ArgumentError(
            "angle configuration $index k_vec components must be finite"))
    k_norm = norm(config.k_vec)
    isfinite(k_norm) && k_norm > 0.0 ||
        throw(ArgumentError(
            "angle configuration $index k_vec must have a finite, nonzero norm"))
    all(isfinite, config.pol) ||
        throw(ArgumentError(
            "angle configuration $index polarization components must be finite"))
    pol_norm = norm(config.pol)
    isfinite(pol_norm) &&
    isapprox(pol_norm, 1.0; rtol=1e-10, atol=1e-12) ||
        throw(ArgumentError(
            "angle configuration $index polarization must be a unit vector"))
    transverse_error =
        abs(dot(config.k_vec / k_norm, config.pol / pol_norm))
    transverse_error <= 1e-10 ||
        throw(ArgumentError(
            "angle configuration $index polarization must be transverse to k_vec"))
    length(config.v) == N ||
        throw(DimensionMismatch(
            "angle configuration $index excitation length $(length(config.v)) != $N"))
    all(isfinite, config.v) ||
        throw(ArgumentError(
            "angle configuration $index excitation must contain only finite values"))
    size(config.Q) == (N, N) ||
        throw(DimensionMismatch(
            "angle configuration $index Q has size $(size(config.Q)), expected ($N, $N)"))
    _validate_multiangle_q(config.Q, "angle configuration $index Q")
    isfinite(config.weight) ||
        throw(ArgumentError(
            "angle configuration $index weight must be finite, got $(config.weight)"))
    return k_norm
end

function _validate_multiangle_problem(
    Z_base::AbstractMatrix{ComplexF64},
    Mp::Vector{<:AbstractMatrix},
    configs::Vector{AngleConfig},
    theta0::Vector{Float64},
)
    N = size(Z_base, 1)
    N >= 1 ||
        throw(ArgumentError("multi-angle system must contain at least one unknown"))
    size(Z_base, 2) == N ||
        throw(DimensionMismatch(
            "Z_base must be square, got size $(size(Z_base))"))
    _validate_impedance_inputs(Mp, theta0, size(Z_base))
    _validate_known_matrix_entries(Z_base, "Z_base")
    @inbounds for p in eachindex(Mp)
        _validate_known_matrix_entries(Mp[p], "patch mass matrix")
    end

    first_k = _validate_angle_config(configs[1], 1, N)
    @inbounds for i in 2:length(configs)
        k_norm = _validate_angle_config(configs[i], i, N)
        isapprox(k_norm, first_k; rtol=1e-10, atol=0.0) ||
            throw(ArgumentError(
                "angle configuration $i has |k_vec|=$k_norm, expected $first_k"))
    end
    return N
end

function _multiangle_q_product_and_objective(
    Q::AbstractMatrix{ComplexF64},
    current::Vector{ComplexF64},
    label::AbstractString,
)
    product = if Q isa Matrix{ComplexF64}
        _finite_matrix_vector_product(Q, current, label)
    else
        converted = Vector{ComplexF64}(Q * current)
        _assert_finite_linear_vector(converted, label)
    end

    objective = real(dot(current, product))
    if !isfinite(objective)
        Q isa Matrix{ComplexF64} ||
            error("$label produced a non-finite objective")
        objective = compute_objective(current, Q)
    end
    return product, objective
end

@noinline function _multiangle_linear_objective_bigfloat(
    J_angles::Vector{Float64},
    weights::Vector{Float64},
)
    return setprecision(
            BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        total = zero(BigFloat)
        @inbounds for index in eachindex(J_angles, weights)
            total += BigFloat(weights[index]) * BigFloat(J_angles[index])
        end
        converted = Float64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "linear multi-angle objective is outside the representable Float64 range"))
        return converted
    end
end

@noinline function _multiangle_sum_log_objective_bigfloat(
    J_angles::Vector{Float64},
    weights::Vector{Float64},
    reference_objectives::Vector{Float64},
    tiny::Float64,
)
    return setprecision(
            BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        total = zero(BigFloat)
        @inbounds for index in eachindex(
            J_angles, weights, reference_objectives)
            objective_value = BigFloat(max(J_angles[index], tiny))
            reference_value = BigFloat(reference_objectives[index])
            total += BigFloat(weights[index]) *
                     (log(objective_value) - log(reference_value))
        end
        converted = Float64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "sum_log multi-angle objective is outside the representable Float64 range"))
        return converted
    end
end

@noinline function _multiangle_gradient_bigfloat(
    Mp::Vector{<:AbstractMatrix},
    I_all::Vector{Vector{ComplexF64}},
    lambda_all::Vector{Vector{ComplexF64}},
    objective_scales::Vector{Float64},
    reactive::Bool,
)
    parameter_count = length(Mp)
    return setprecision(
            BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        totals = fill(zero(BigFloat), parameter_count)
        @inbounds for angle in eachindex(
            I_all, lambda_all, objective_scales)
            angle_gradient = gradient_impedance(
                Mp, I_all[angle], lambda_all[angle]; reactive=reactive)
            scale = BigFloat(objective_scales[angle])
            for parameter in eachindex(totals, angle_gradient)
                totals[parameter] +=
                    scale * BigFloat(angle_gradient[parameter])
            end
        end

        gradient = Vector{Float64}(undef, parameter_count)
        @inbounds for parameter in eachindex(gradient, totals)
            gradient[parameter] = Float64(totals[parameter])
            isfinite(gradient[parameter]) ||
                throw(OverflowError(
                    "multi-angle objective gradient at parameter $parameter " *
                    "is outside the representable Float64 range"))
        end
        return gradient
    end
end

function _multiangle_objective_scales(J_angles::Vector{Float64},
                                      weights::Vector{Float64},
                                      objective::Symbol,
                                      reference_objectives::Vector{Float64},
                                      smooth_beta::Float64)
    M = length(J_angles)
    M >= 1 ||
        throw(ArgumentError("multi-angle objective requires at least one angle"))
    length(weights) == M ||
        error("weights length $(length(weights)) does not match objective length $M")
    length(reference_objectives) == M ||
        error("reference_objectives length $(length(reference_objectives)) does not match objective length $M")
    @inbounds for i in 1:M
        isfinite(J_angles[i]) ||
            error("per-angle objective contains a non-finite value at index $i")
        isfinite(weights[i]) ||
            error("objective weight $i must be finite, got $(weights[i])")
        (isfinite(reference_objectives[i]) && reference_objectives[i] > 0.0) ||
            error(
                "reference objective $i must be finite and positive, got $(reference_objectives[i])")
    end

    # A positive floor keeps log objectives and their derivatives defined when
    # numerical roundoff produces a zero or slightly negative quadratic form.
    tiny = 1e-300
    scales = Vector{Float64}(undef, M)

    if objective == :linear
        copyto!(scales, weights)
        value = dot(weights, J_angles)
        isfinite(value) ||
            (value = _multiangle_linear_objective_bigfloat(
                J_angles, weights))
        return value, scales
    elseif objective == :sum_log
        value = 0.0
        @inbounds for i in 1:M
            wi = weights[i]
            wi >= 0.0 ||
                error("sum_log objective weight $i must be nonnegative, got $wi")
            Ji = max(J_angles[i], tiny)
            log_ratio = log(Ji) - log(reference_objectives[i])
            value += wi * log_ratio
            scales[i] = wi / Ji
        end
        isfinite(value) ||
            (value = _multiangle_sum_log_objective_bigfloat(
                J_angles, weights, reference_objectives, tiny))
        all(isfinite, scales) ||
            error("sum_log derivative scales overflowed")
        return value, scales
    elseif objective == :smoothmax_log
        (isfinite(smooth_beta) && smooth_beta > 0.0) ||
            throw(ArgumentError(
                "smooth_beta must be finite and positive, got $smooth_beta"))

        # Stabilize around max(z), before multiplying by beta. This avoids the
        # `Inf - Inf` failure of the conventional beta*z log-sum-exp formula
        # for large but finite beta, while preserving weight ratios for ties.
        zmax = -Inf
        @inbounds for i in 1:M
            weights[i] > 0.0 ||
                error(
                    "smoothmax_log objective weight $i must be positive, got $(weights[i])")
            Ji = max(J_angles[i], tiny)
            zi = log(Ji) - log(reference_objectives[i])
            zmax = max(zmax, zi)
        end

        shiftmax = -Inf
        @inbounds for i in 1:M
            Ji = max(J_angles[i], tiny)
            zi = log(Ji) - log(reference_objectives[i])
            shift = smooth_beta * (zi - zmax) + log(weights[i])
            shiftmax = max(shiftmax, shift)
        end

        denom = 0.0
        @inbounds for i in 1:M
            Ji = max(J_angles[i], tiny)
            zi = log(Ji) - log(reference_objectives[i])
            shift = smooth_beta * (zi - zmax) + log(weights[i])
            denom += exp(shift - shiftmax)
        end

        value = zmax + (shiftmax + log(denom)) / smooth_beta
        @inbounds for i in 1:M
            Ji = max(J_angles[i], tiny)
            zi = log(Ji) - log(reference_objectives[i])
            shift = smooth_beta * (zi - zmax) + log(weights[i])
            scales[i] = exp(shift - shiftmax) / (denom * Ji)
        end
        (isfinite(value) && all(isfinite, scales)) ||
            error("smoothmax_log objective or derivative scales overflowed")
        return value, scales
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
    build_multiangle_configs(mesh, rwg, k, angles; grid, backscatter_cone=15.0, matrix_free_Q=false, rcs_component=:copol)

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
    (isfinite(k) && k > 0.0) ||
        throw(ArgumentError(
            "multi-angle wavenumber must be finite and positive, got $k"))
    isempty(angles) &&
        throw(ArgumentError(
            "build_multiangle_configs requires at least one incidence angle"))
    (isfinite(backscatter_cone) && 0.0 <= backscatter_cone <= 180.0) ||
        throw(ArgumentError(
            "backscatter_cone must be finite and lie in [0, 180] degrees, got $backscatter_cone"))
    rcs_component in (:copol, :crosspol, :total) ||
        throw(ArgumentError(
            "rcs_component must be :copol, :crosspol, or :total"))

    for (i, ang) in enumerate(angles)
        hasproperty(ang, :theta_inc) ||
            throw(ArgumentError("incidence angle $i is missing theta_inc"))
        hasproperty(ang, :phi_inc) ||
            throw(ArgumentError("incidence angle $i is missing phi_inc"))
        θ_i = ang.theta_inc
        φ_i = ang.phi_inc
        θ_i isa Real && isfinite(θ_i) && 0.0 <= θ_i <= π ||
            throw(ArgumentError(
                "incidence angle $i theta_inc must be finite and lie in [0, π], got $θ_i"))
        φ_i isa Real && isfinite(φ_i) ||
            throw(ArgumentError(
                "incidence angle $i phi_inc must be finite, got $φ_i"))
        if hasproperty(ang, :pol)
            pol = try
                Vec3(ang.pol)
            catch err
                throw(ArgumentError(
                    "incidence angle $i pol must be convertible to Vec3: $(sprint(showerror, err))"))
            end
            all(isfinite, pol) ||
                throw(ArgumentError(
                    "incidence angle $i pol components must be finite"))
        end
        if hasproperty(ang, :weight)
            weight = ang.weight
            weight isa Real && isfinite(weight) ||
                throw(ArgumentError(
                    "incidence angle $i weight must be a finite real number, got $weight"))
        end
    end

    eta0 = 376.730313668
    G_mat = radiation_vectors(mesh, rwg, grid, k; eta0=eta0)
    pol_theta = pol_linear_x(grid)  # θ̂ polarization
    pol_phi = pol_linear_y(grid)    # φ̂ polarization

    configs = AngleConfig[]
    for ang in angles
        θ_i = Float64(ang.theta_inc)
        φ_i = Float64(ang.phi_inc)

        # Incidence direction: k̂ = (sinθ cosφ, sinθ sinφ, cosθ)
        khat = Vec3(sin(θ_i) * cos(φ_i), sin(θ_i) * sin(φ_i), cos(θ_i))
        k_vec = k * khat

        # Backscatter direction = -k̂
        bs_dir = -khat

        # Excitation
        pw_pol_raw =
            hasproperty(ang, :pol) ? Vec3(ang.pol) :
            _default_transverse_pol(khat)
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

        w = hasproperty(ang, :weight) ? Float64(ang.weight) : 1.0

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
- `preconditioner`: fixed `AbstractPreconditionerData` for GMRES
- `preconditioner_builder`: optional function `theta -> preconditioner` for
  matrix-free optimization with design-dependent preconditioners
  (cached only for exactly unchanged `theta`)
- `trial_preconditioner_mode`: `:rebuild` rebuilds the design-dependent
  preconditioner for every line-search trial; `:current` reuses the current
  accepted-iterate preconditioner for exploratory trial solves; and
  `:current_then_rebuild` retries a failed current-preconditioner trial with a
  rebuilt trial preconditioner before rejecting the step
- `gmres_precond_side`: `:left` or `:right` preconditioner application side
- `fallback_to_steepest`: retry projected steepest descent after a failed
  L-BFGS line search before stopping
- `lbfgs_line_search_maxiter`, `steepest_line_search_maxiter`: maximum Armijo
  backtracking trials for L-BFGS and fallback steepest directions
- `gmres_tol`, `gmres_maxiter`, `gmres_memory`: GMRES parameters
- `check_gmres_true_residual`: verify true physical residuals for GMRES solves
- `gmres_true_residual_factor`: allowed true-residual multiple of `gmres_tol`
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
                                  preconditioner_builder=nothing,
                                  trial_preconditioner_mode::Symbol=:rebuild,
                                  gmres_precond_side::Symbol=:left,
                                  fallback_to_steepest::Bool=true,
                                  lbfgs_line_search_maxiter::Int=20,
                                  steepest_line_search_maxiter::Int=20,
                                  gmres_tol::Float64=1e-6,
                                  gmres_maxiter::Int=300,
                                  gmres_memory::Int=20,
                                  check_gmres_true_residual::Bool=true,
                                  gmres_true_residual_factor::Float64=100.0,
                                  objective::Symbol=:linear,
                                  reference_objectives::Union{Nothing, Vector{Float64}}=nothing,
                                  smooth_beta::Float64=8.0)
    M = length(configs)    # number of angles
    P = length(theta0)     # number of design parameters
    M >= 1 ||
        throw(ArgumentError("optimize_multiangle_rcs requires at least one angle configuration"))
    P >= 1 ||
        throw(ArgumentError("optimize_multiangle_rcs requires at least one design parameter"))
    maxiter >= 0 ||
        throw(ArgumentError("maxiter must be nonnegative, got $maxiter"))
    (isfinite(tol) && tol >= 0.0) ||
        throw(ArgumentError("tol must be finite and nonnegative, got $tol"))
    m_lbfgs >= 0 ||
        throw(ArgumentError("m_lbfgs must be nonnegative, got $m_lbfgs"))
    (isfinite(alpha0) && alpha0 > 0.0) ||
        throw(ArgumentError("alpha0 must be finite and positive, got $alpha0"))
    objective in (:linear, :sum_log, :smoothmax_log) ||
        throw(ArgumentError(
            "Unknown multi-angle objective: $objective (expected :linear, :sum_log, or :smoothmax_log)"))
    trial_preconditioner_mode in (:rebuild, :current, :current_then_rebuild) ||
        throw(ArgumentError(
            "trial_preconditioner_mode must be :rebuild, :current, or :current_then_rebuild"))
    lbfgs_line_search_maxiter > 0 ||
        throw(ArgumentError("lbfgs_line_search_maxiter must be positive"))
    steepest_line_search_maxiter > 0 ||
        throw(ArgumentError("steepest_line_search_maxiter must be positive"))
    _validate_optimizer_bounds(lb, ub, P)
    _validate_multiangle_problem(Z_base, Mp, configs, theta0)

    # Use dense LU when Z_base is a dense Matrix for exact solves;
    # fall back to GMRES for matrix-free operators (MLFMA, ACA)
    use_dense_lu = Z_base isa Matrix{ComplexF64}
    solver = use_dense_lu ? :direct : :gmres
    if !use_dense_lu
        _validate_gmres_options(
            gmres_tol, gmres_maxiter, gmres_memory, gmres_precond_side)
        if check_gmres_true_residual
            (isfinite(gmres_true_residual_factor) &&
             gmres_true_residual_factor > 0.0) ||
                throw(ArgumentError(
                    "gmres_true_residual_factor must be finite and positive, got $gmres_true_residual_factor"))
        end
    end

    theta = copy(theta0)
    function project!(x)
        @inbounds for p in eachindex(x)
            lb === nothing ||
                (x[p] = max(x[p], _optimizer_bound_value(lb, p)))
            ub === nothing ||
                (x[p] = min(x[p], _optimizer_bound_value(ub, p)))
        end
        return x
    end
    project!(theta)

    weights = [cfg.weight for cfg in configs]
    refs = reference_objectives === nothing ? ones(Float64, M) : copy(reference_objectives)
    length(refs) == M ||
        throw(DimensionMismatch(
            "reference_objectives length $(length(refs)) must match angle count $M"))
    @inbounds for i in 1:M
        isfinite(refs[i]) && refs[i] > 0.0 ||
            throw(ArgumentError(
                "reference objective $i must be finite and positive, got $(refs[i])"))
        if objective == :sum_log
            weights[i] >= 0.0 ||
                throw(ArgumentError(
                    "sum_log objective weight $i must be nonnegative, got $(weights[i])"))
        elseif objective == :smoothmax_log
            weights[i] > 0.0 ||
                throw(ArgumentError(
                    "smoothmax_log objective weight $i must be positive, got $(weights[i])"))
        end
    end
    if objective == :smoothmax_log
        (isfinite(smooth_beta) && smooth_beta > 0.0) ||
            throw(ArgumentError(
                "smooth_beta must be finite and positive, got $smooth_beta"))
    end

    if verbose
        println("Multi-angle RCS optimization: $M angles, $P parameters, solver=$solver, objective=$objective")
        if !use_dense_lu && preconditioner_builder !== nothing
            println("  GMRES preconditioner rebuilt when theta changes")
            if trial_preconditioner_mode == :current
                println("  line-search trials reuse the current accepted-iterate preconditioner")
            elseif trial_preconditioner_mode == :current_then_rebuild
                println("  line-search trials try the current preconditioner, then rebuild on solve failure")
            end
        elseif !use_dense_lu && preconditioner !== nothing
            println("  GMRES preconditioned")
        end
    end

    cached_preconditioner_theta = Vector{Float64}(undef, 0)
    cached_preconditioner = Ref{Union{Nothing, AbstractPreconditionerData}}(nothing)

    function active_preconditioner(theta_current::Vector{Float64})
        if use_dense_lu || preconditioner_builder === nothing
            return preconditioner
        end
        if cached_preconditioner[] !== nothing &&
           length(cached_preconditioner_theta) == length(theta_current) &&
           cached_preconditioner_theta == theta_current
            return cached_preconditioner[]
        end
        P_current = preconditioner_builder(theta_current)
        P_current isa AbstractPreconditionerData ||
            error("preconditioner_builder must return an AbstractPreconditionerData")
        resize!(cached_preconditioner_theta, length(theta_current))
        copyto!(cached_preconditioner_theta, theta_current)
        cached_preconditioner[] = P_current
        return P_current
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
        P_iter = active_preconditioner(theta)

        # ── 2. Forward solves: I_a = Z(θ)⁻¹ v_a ────────────────
        I_all = Vector{Vector{ComplexF64}}(undef, M)
        for a in 1:M
            I_all[a] = if use_dense_lu
                _solve_factored_linear_system(
                    Z_factor, Z_full, configs[a].v,
                    "multi-angle direct forward solution")
            else
                solve_forward(Z_full, configs[a].v;
                              solver=solver,
                              preconditioner=P_iter,
                              gmres_precond_side=gmres_precond_side,
                              gmres_tol=gmres_tol,
                              gmres_maxiter=gmres_maxiter,
                              gmres_memory=gmres_memory,
                              check_true_residual=check_gmres_true_residual,
                              true_residual_factor=gmres_true_residual_factor)
            end
            n_fwd_solves += 1
        end

        # ── 3. Objective and scalarization weights ───────────────
        QI_all = Vector{Vector{ComplexF64}}(undef, M)
        J_angles = zeros(Float64, M)
        for a in 1:M
            QI_all[a], J_angles[a] = _multiangle_q_product_and_objective(
                configs[a].Q, I_all[a], "multi-angle Q product")
        end
        J_val, objective_scales = _multiangle_objective_scales(
            J_angles, weights, objective, refs, smooth_beta)

        # ── 4. Adjoint solves: Z(θ)† λ_a = Q_a I_a ────────────
        lambda_all = Vector{Vector{ComplexF64}}(undef, M)
        for a in 1:M
            rhs_a = QI_all[a]
            lambda_all[a] = if use_dense_lu
                adjoint_Z = adjoint(Z_full)
                _solve_factored_linear_system(
                    adjoint(Z_factor), adjoint_Z,
                    Vector{ComplexF64}(rhs_a),
                    "multi-angle direct adjoint solution")
            else
                solve_adjoint_rhs(Z_full, rhs_a;
                                  solver=solver,
                                  preconditioner=P_iter,
                                  gmres_precond_side=gmres_precond_side,
                                  gmres_tol=gmres_tol,
                                  gmres_maxiter=gmres_maxiter,
                                  gmres_memory=gmres_memory,
                                  check_true_residual=check_gmres_true_residual,
                                  true_residual_factor=gmres_true_residual_factor)
            end
            n_adj_solves += 1
        end

        # ── 5. Gradient: g[p] = Σ_a scalar_a · ∂J_a/∂θ_p ───────
        g = zeros(P)
        gradient_finite = true
        for a in 1:M
            g_a = gradient_impedance(Mp, I_all[a], lambda_all[a]; reactive=reactive)
            axpy!(objective_scales[a], g_a, g)
            if !_linear_array_is_finite(g)
                gradient_finite = false
                break
            end
        end
        if !gradient_finite
            g = _multiangle_gradient_bigfloat(
                Mp, I_all, lambda_all, objective_scales, reactive)
        end
        _assert_finite_optimizer_vector(g, "multi-angle objective gradient")
        gnorm = norm(g)
        isfinite(gnorm) ||
            error(
                "multi-angle objective gradient norm is non-finite at iteration $iter")

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
            if isfinite(sy) && sy > 1e-30 && m_lbfgs > 0
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

        gamma =
            m_cur > 0 ?
            dot(s_list[end], y_list[end]) /
            dot(y_list[end], y_list[end]) :
            alpha0
        if !(isfinite(gamma) && gamma > 0.0)
            gamma = alpha0
            copyto!(q, g)
            empty!(s_list)
            empty!(y_list)
        end
        r = q
        rmul!(r, gamma)

        for i in 1:m_cur
            rho_i = 1.0 / dot(y_list[i], s_list[i])
            beta = rho_i * dot(y_list[i], r)
            r .+= (alpha_vec[i] - beta) .* s_list[i]
        end

        # ── 8. Projected line search ─────────────────────────────
        d = r
        rmul!(d, -1.0)
        alpha_ls = 1.0
        copyto!(theta_old, theta)
        copyto!(g_old, g)

        ls_success = false

        # Check if L-BFGS direction is descent; if not, use steepest descent
        gd = dot(g, d)
        if !isfinite(gd) || gd >= 0
            copyto!(d, g)
            rmul!(d, -alpha0)
            gd = -alpha0 * gnorm^2
            empty!(s_list)
            empty!(y_list)
        end
        isfinite(gd) ||
            error(
                "multi-angle search directional derivative is non-finite at iteration $iter")

        theta_trial = similar(theta)
        J_trial_angles = Vector{Float64}(undef, M)
        use_current_trial_preconditioner =
            trial_preconditioner_mode in (:current, :current_then_rebuild)
        function try_projected_line_search!(direction::Vector{Float64},
                                            label::String,
                                            max_line_search::Int)
            alpha_try = 1.0
            for ls in 1:max_line_search
                @. theta_trial = theta_old + alpha_try * direction
                project!(theta_trial)
                theta_trial == theta_old && break
                if !all(isfinite, theta_trial)
                    alpha_try *= 0.5
                    continue
                end
                projected_gd = _projected_directional_derivative(
                    g, theta_trial, theta_old)
                if !(isfinite(projected_gd) && projected_gd < 0.0)
                    alpha_try *= 0.5
                    continue
                end
                trial_valid = true
                J_trial = Inf
                trial_modes = if trial_preconditioner_mode == :current_then_rebuild
                    use_current_trial_preconditioner ? (:current, :rebuild) : (:rebuild,)
                else
                    (trial_preconditioner_mode,)
                end
                trial_error = nothing
                for trial_mode in trial_modes
                    try
                        if use_dense_lu
                            assemble_full_Z!(Z_buf, Z_base, Mp, theta_trial; reactive=reactive)
                            Z_trial = Z_buf
                            Z_trial_factor = lu(Z_trial)
                        else
                            Z_trial = ImpedanceLoadedOperator(Z_base, Mp, theta_trial, reactive)
                            Z_trial_factor = nothing
                        end
                        P_trial = trial_mode == :current ? P_iter :
                            active_preconditioner(theta_trial)

                        # Evaluate trial objective. A failed exploratory trial is
                        # not an accepted iterate. In adaptive mode, first
                        # distinguish stale-preconditioner failure from a bad step.
                        for a in 1:M
                            I_trial = if use_dense_lu
                                _solve_factored_linear_system(
                                    Z_trial_factor, Z_trial, configs[a].v,
                                    "multi-angle trial direct solution")
                            else
                                solve_forward(Z_trial, configs[a].v;
                                              solver=solver,
                                              preconditioner=P_trial,
                                              gmres_precond_side=gmres_precond_side,
                                              gmres_tol=gmres_tol,
                                              gmres_maxiter=gmres_maxiter,
                                              gmres_memory=gmres_memory,
                                              check_true_residual=check_gmres_true_residual,
                                              true_residual_factor=gmres_true_residual_factor)
                            end
                            n_fwd_solves += 1
                            _, J_trial_angles[a] =
                                _multiangle_q_product_and_objective(
                                    configs[a].Q, I_trial,
                                    "multi-angle trial Q product")
                        end
                        J_trial, _ = _multiangle_objective_scales(
                            J_trial_angles, weights, objective, refs, smooth_beta)
                        trial_valid = true
                        break
                    catch err
                        _recoverable_optimizer_trial_error(err) || rethrow()
                        trial_valid = false
                        trial_error = err
                        if trial_preconditioner_mode == :current_then_rebuild &&
                           trial_mode == :current
                            use_current_trial_preconditioner = false
                            verbose && println("$label trial $ls current-preconditioner solve failed at iteration $iter; retrying with rebuilt preconditioner ($err)")
                            continue
                        end
                    end
                end
                if !trial_valid
                    verbose && println("$label trial $ls failed at iteration $iter; reducing step ($trial_error)")
                end

                # Armijo condition
                if trial_valid &&
                   J_trial <= J_val + 1e-4 * projected_gd
                    theta .= theta_trial
                    return true
                end
                alpha_try *= 0.5
            end
            return false
        end

        ls_success = try_projected_line_search!(
            d, "Line-search", lbfgs_line_search_maxiter)
        if !ls_success && fallback_to_steepest && !isempty(s_list)
            empty!(s_list)
            empty!(y_list)
            copyto!(d, g)
            rmul!(d, -alpha0)
            gd = -alpha0 * gnorm^2
            verbose && println("L-BFGS line search failed at iteration $iter; retrying projected steepest descent")
            ls_success = try_projected_line_search!(
                d, "Steepest-descent line-search",
                steepest_line_search_maxiter)
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
