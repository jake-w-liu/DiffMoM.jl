export RetainedScatteringState, solve_prepared!, solve_prepared_adjoint!

"""
    RetainedScatteringState

Operator, checked factorization, preconditioner, RWG data, and physical RHS
retained by `solve_scattering(...; return_state=true)`. Treat its physical
configuration and operator data as read-only. Geometry, operator, frequency,
quadrature, or solver-configuration changes invalidate subsequent reuse.

`solve_prepared!` replaces `rhs` and `I_coeffs` only after every RHS succeeds,
and increments `rhs_revision`. An adjoint solve leaves the forward state
unchanged. `last_solve` is `nothing` until a retained solve succeeds; it then
records per-RHS iterations and true residuals against the selected operator.
Calls through this state are serialized by its lock.
"""
mutable struct RetainedScatteringState{O,F,P}
    mesh::TriMesh
    rwg::RWGData
    operator::O
    factorization::F
    preconditioner::P
    frequency_hz::Float64
    c0::Float64
    quad_order::Int
    method::Symbol
    solver_options::NamedTuple
    excitation::Union{Nothing,AbstractExcitation}
    max_work_bytes::Int
    signature::UInt
    rhs::Union{Vector{ComplexF64},Matrix{ComplexF64}}
    I_coeffs::Union{Vector{ComplexF64},Matrix{ComplexF64}}
    rhs_revision::UInt
    last_solve::Union{Nothing,NamedTuple}
    work_lock::ReentrantLock
end

function _content_fingerprint(
        operator::Union{ACAOperator,MLFMAOperator}, seed::UInt)
    result = hash(typeof(operator), seed)
    for name in fieldnames(typeof(operator))
        name === :workspace && continue
        result = _content_fingerprint(getfield(operator, name), result)
    end
    return result
end

_content_fingerprint(operator::MatrixFreeEFIEOperator, seed::UInt) =
    _content_fingerprint(operator.cache, seed)

function _retained_configuration_signature(state::RetainedScatteringState)
    return _content_fingerprint(
        (state.mesh, state.rwg, state.operator, state.frequency_hz, state.c0,
         state.quad_order, state.method, state.solver_options,
         state.excitation, state.rhs, state.I_coeffs), UInt(0))
end

function _validate_retained_state(state::RetainedScatteringState)
    state.signature == _retained_configuration_signature(state) ||
        throw(ArgumentError(
            "retained scattering configuration changed; call solve_scattering again"))
    return nothing
end

function _retained_storage_bytes(state::RetainedScatteringState)
    # summarysize includes object metadata and deduplicates shared buffers, so
    # it conservatively charges more than the retained raw array payload.
    # Exclude the state lock, whose current owner is the caller's live task.
    return Base.summarysize(
        (state.mesh, state.rwg, state.operator, state.factorization,
         state.preconditioner, state.excitation, state.rhs, state.I_coeffs))
end

function _retained_solve_work_bytes(state::RetainedScatteringState, columns::Int)
    n = state.rwg.nedges
    rhs_bytes = _checked_array_payload_bytes(
        ComplexF64, n, columns; label="retained solve RHS")
    vector_bytes = _checked_array_payload_bytes(
        ComplexF64, n, 8; label="retained solve column workspace")
    residual_exact_bytes = _checked_payload_sum(
        "retained exact residual workspace",
        _checked_array_payload_bytes(
            UInt8, 2 * _DIRECT_BIGFLOAT_BYTES_PER_REAL + sizeof(Complex{BigFloat}), n),
        24 * _DIRECT_BIGFLOAT_BYTES_PER_REAL)
    # Input conversion, returned solution, column solves/residuals, and reports
    # coexist with the previous physical RHS/current and all retained operators.
    total = BigInt(_retained_storage_bytes(state)) + 2BigInt(rhs_bytes) +
            vector_bytes + residual_exact_bytes + BigInt(2sizeof(Float64)) * columns
    total <= typemax(Int) ||
        throw(ArgumentError("retained solve workspace estimate overflows Int"))
    return Int(total)
end

function _retain_scattering_state(
        mesh, rwg, operator, factorization, preconditioner, rhs, coefficients,
        frequency_hz, c0, quad_order, method, solver_options, max_work_bytes;
        excitation::Union{Nothing,AbstractExcitation}=nothing)
    limit = _validated_resource_limit("max_work_bytes", max_work_bytes)
    exact_terms = _validated_nonnegative_resource_limit(
        "max_true_residual_exact_terms",
        get(solver_options, :max_true_residual_exact_terms,
            _DEFAULT_MAX_TRUE_RESIDUAL_EXACT_TERMS))
    options = merge(solver_options, (max_true_residual_exact_terms=exact_terms,))
    state = RetainedScatteringState(
        mesh, rwg, operator, factorization, preconditioner,
        frequency_hz, c0, quad_order, method, options, excitation,
        limit, UInt(0),
        rhs, coefficients, UInt(0), nothing, ReentrantLock())
    state.signature = _retained_configuration_signature(state)
    _enforce_payload_limit(
        _retained_solve_work_bytes(state, 1), limit,
        "retained scattering state and solve workspace", "max_work_bytes")
    return state
end

function _solve_retained!(
        state::RetainedScatteringState,
        rhs::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}},
        adjoint_solve::Bool, max_work_bytes::Integer,
        max_true_residual_exact_terms::Integer)
    return lock(state.work_lock) do
        _validate_retained_state(state)
        exact_term_limit = _validated_nonnegative_resource_limit(
            "max_true_residual_exact_terms", max_true_residual_exact_terms)
        options = merge(state.solver_options,
            (max_true_residual_exact_terms=exact_term_limit,))
        n = state.rwg.nedges
        size(rhs, 1) == n ||
            throw(DimensionMismatch("retained solve RHS must have $n rows"))
        columns = size(rhs, 2)
        columns >= 1 ||
            throw(ArgumentError("retained solve requires at least one RHS"))
        all(isfinite, rhs) ||
            throw(ArgumentError("retained solve RHS must contain finite values"))
        limit = _validated_resource_limit("max_work_bytes", max_work_bytes)
        base_bytes = _retained_solve_work_bytes(state, columns)
        _enforce_payload_limit(
            base_bytes, limit, "retained solve workspace", "max_work_bytes")
        available = limit - base_bytes
        if state.method !== :dense_direct
            _preflight_gmres_workspace(n, get(options, :memory, 20), available)
        end
        rhs_copy = rhs isa AbstractVector ?
                   Vector{ComplexF64}(rhs) : Matrix{ComplexF64}(rhs)
        all(isfinite, rhs_copy) ||
            throw(ArgumentError("retained solve RHS is not representable as ComplexF64"))
        solution = similar(rhs_copy)
        rhs_columns = reshape(rhs_copy, n, columns)
        solution_columns = reshape(solution, n, columns)
        iterations = zeros(Int, columns)
        residuals = zeros(Float64, columns)
        operator = adjoint_solve ? adjoint(state.operator) : state.operator
        factorization = adjoint_solve && state.factorization !== nothing ?
                        adjoint(state.factorization) : state.factorization
        exact_check = () -> _enforce_payload_limit(
            _checked_exact_dense_solve_work_bytes(
                ComplexF64, n, 3, 3, 3, base_bytes;
                label="retained exact dense solve"),
            limit, "retained exact dense solve", "max_work_bytes")
        elapsed = @elapsed for column in 1:columns
            physical_rhs = _as_complex_rhs(view(rhs_columns, :, column))
            if state.method === :dense_direct
                values = _solve_factored_linear_system(
                    factorization, operator, physical_rhs,
                    "retained direct solve"; exact_fallback_check=exact_check)
                iterations[column] = -1
            else
                solver = adjoint_solve ? solve_gmres_adjoint : solve_gmres
                values, stats = solver(
                    state.operator, physical_rhs;
                    preconditioner=state.preconditioner,
                    max_workspace_bytes=available, options...)
                iterations[column] = stats.niter
            end
            solution_columns[:, column] .= values
            residuals[column] = _true_residual_ratio(
                operator, values, physical_rhs, "retained solve";
                max_true_residual_exact_terms=exact_term_limit)
        end
        _validate_retained_state(state)
        if !adjoint_solve
            # A different numeric RHS does not specify an incident field on
            # a refined mesh. Discard the old source association in that case.
            rhs === state.rhs || (state.excitation = nothing)
            state.rhs = rhs_copy
            state.I_coeffs = solution
            state.rhs_revision += UInt(1)
            state.signature = _retained_configuration_signature(state)
        end
        state.last_solve = (
            adjoint=adjoint_solve, iterations=iterations,
            true_residuals=residuals, elapsed_s=elapsed,
            max_true_residual_exact_terms=exact_term_limit)
        return solution
    end
end

"""
    solve_prepared!(state; rhs=state.rhs, max_work_bytes=state.max_work_bytes)

Solve one vector RHS or a matrix of RHS columns using retained scattering
operators and the canonical checked solver. Return coefficients with the same
shape as `rhs`. A successful forward call updates the state's physical RHS,
current, revision, and solve report; a failed call leaves them unchanged.
Supplying a different numeric RHS clears the retained excitation descriptor,
because its continuation to a finer mesh is then unspecified.

The workspace ceiling includes retained data, the old forward state, new
input/output buffers, and column-solver work. Direct exceptional-precision
work is checked separately against the same ceiling. Iterative solves retain
the configured convergence and true-residual checks. The selected operator's
residual does not measure mesh or compression error.
The per-evaluation `max_true_residual_exact_terms` budget is inherited from
the state and can be overridden for this call without changing solve tolerances.
"""
function solve_prepared!(state::RetainedScatteringState;
                         rhs=state.rhs,
                         max_work_bytes::Integer=state.max_work_bytes,
                         max_true_residual_exact_terms::Integer=
                             state.solver_options.max_true_residual_exact_terms)
    return _solve_retained!(state, rhs, false, max_work_bytes,
        max_true_residual_exact_terms)
end

"""
    solve_prepared_adjoint!(state, rhs; max_work_bytes=state.max_work_bytes)

Solve `state.operator' * x = rhs` with the retained checked factorization or
adjoint GMRES/preconditioner path. Accept a vector or matrix of RHS columns.
Update `last_solve` while preserving the forward RHS, current, and revision.
The state lock serializes this call with `solve_prepared!`.
`max_true_residual_exact_terms` has the same per-evaluation meaning as in
`solve_prepared!`; the effective budget is recorded in `last_solve`.
"""
function solve_prepared_adjoint!(
        state::RetainedScatteringState,
        rhs::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}};
        max_work_bytes::Integer=state.max_work_bytes,
        max_true_residual_exact_terms::Integer=
            state.solver_options.max_true_residual_exact_terms)
    return _solve_retained!(state, rhs, true, max_work_bytes,
        max_true_residual_exact_terms)
end
