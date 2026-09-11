export GalerkinErrorSystem, prepare_galerkin_error
export PreparedErrorOutputs, prepare_error_outputs, output_residual_probes

"""
    GalerkinErrorSystem

Finite enriched system and its restricted coarse solve. The restriction
report records the difference from the original coarse operator/RHS and
whether its factor was reused. A nonzero permitted mismatch is an operator
approximation. Treat the stored arrays and operators as read-only.
"""
struct GalerkinErrorSystem{A}
    origin::RetainedScatteringState
    coarse::RetainedScatteringState
    pair::NestedRWGPair
    fine_operator::A
    fine_rhs::Vector{ComplexF64}
    residual::Vector{ComplexF64}
    origin_revision::UInt
    coarse_revision::UInt
    restriction_report::NamedTuple
    preparation_s::Float64
    signature::UInt
end

function _error_action(operator, vector)
    if operator isa Union{StridedMatrix,SparseMatrixCSC,
                          Adjoint{<:Any,<:StridedMatrix}}
        return _finite_matrix_vector_product(
            operator, vector, "enriched operator action")
    end
    return _assert_finite_linear_vector(
        operator * vector, "enriched operator action")
end

function _error_action_columns(operator, matrix::AbstractMatrix)
    if operator isa Union{StridedMatrix{ComplexF64},Adjoint{ComplexF64,<:StridedMatrix}} &&
       matrix isa StridedMatrix{ComplexF64}
        return _finite_matrix_columns(operator, matrix, "enriched operator columns")
    end
    output = Matrix{ComplexF64}(undef, size(operator, 1), size(matrix, 2))
    for column in axes(matrix, 2)
        output[:, column] .= _error_column_action(operator, matrix, column)
    end
    return output
end

_error_column_action(operator, matrix, column) =
    _error_action(operator, Vector{ComplexF64}(view(matrix, :, column)))

function _error_column_action(
        operator::Union{StridedMatrix,Adjoint{<:Any,<:StridedMatrix}},
        matrix::SparseMatrixCSC, column)
    positions = nzrange(matrix, column)
    isempty(positions) && return zeros(ComplexF64, size(operator, 1))
    support = rowvals(matrix)[positions]
    values = ComplexF64.(nonzeros(matrix)[positions])
    # Sparse injection has local support. Restrict the checked dense product
    # to its nonzero columns instead of revisiting every zero coefficient.
    return _finite_matrix_vector_product(
        view(operator, :, support), values, "sparse enriched column action")
end

function _error_system_signature(operator, rhs, residual, pair)
    return _content_fingerprint((operator, rhs, residual, pair.signature), UInt(0))
end

function _validate_error_system(system::GalerkinErrorSystem)
    _validate_nested_pair(system.pair)
    _validate_retained_state(system.origin)
    _validate_retained_state(system.coarse)
    system.origin.rhs_revision == system.origin_revision ||
        throw(ArgumentError("the source RHS changed; rebuild the enriched error system"))
    system.coarse.rhs_revision == system.coarse_revision ||
        throw(ArgumentError("the restricted coarse RHS changed; rebuild the enriched error system"))
    system.signature == _error_system_signature(
        system.fine_operator, system.fine_rhs, system.residual, system.pair) ||
        throw(ArgumentError("the enriched operator or RHS changed; rebuild the error system"))
    return nothing
end

function _error_system_storage_bytes(system::GalerkinErrorSystem)
    return _checked_payload_sum(
        "enriched system",
        _retained_storage_bytes(system.origin),
        _retained_storage_bytes(system.coarse),
        Base.summarysize((system.pair, system.fine_operator,
                          system.fine_rhs, system.residual)))
end

function _error_coarse_solve_budget(system, columns, limit, additional)
    coarse_storage = _retained_storage_bytes(system.coarse)
    outside = _checked_payload_sum(
        "enriched coarse solve",
        _error_system_storage_bytes(system) - coarse_storage, additional)
    solver_work = _retained_solve_work_bytes(system.coarse, columns)
    if system.coarse.method !== :dense_direct
        solver_work = _checked_payload_sum(
            "enriched Krylov solve", solver_work,
            _gmres_workspace_bytes(system.coarse.rwg.nedges, 20))
    end
    _enforce_payload_limit(
        _checked_payload_sum("enriched coarse solve", outside, solver_work),
        limit, "enriched coarse solve", "max_work_bytes")
    return limit - outside
end

"""
    prepare_galerkin_error(state, pair, fine_operator, fine_rhs;
        restriction_rtol=0.0, restriction_atol=0.0,
        rebuild_on_mismatch=true, max_work_bytes=2_000_000_000)

Form A=P'Z_fP, b_c=P'b_f and r=Q'(b_f-Z_f*P*x_h) using canonical operator
actions and checked solves. The supplied fine operator/RHS must use the
fine RWG ordering of the pair.

The defaults require exact agreement before reusing the original coarse
operator/factor. Otherwise build and charge a factor of the consistent
restriction. Positive tolerances explicitly permit a measured approximation;
its effect on outputs requires a separate numerical budget. The original
state is preserved. Preparation time excludes the caller's fine assembly.
"""
function prepare_galerkin_error(
        state::RetainedScatteringState, pair::NestedRWGPair,
        fine_operator::AbstractMatrix{<:Number}, fine_rhs::AbstractVector{<:Number};
        restriction_rtol::Real=0.0, restriction_atol::Real=0.0,
        rebuild_on_mismatch::Bool=true,
        max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    return lock(state.work_lock) do
        started = time_ns()
        _validate_retained_state(state)
        _validate_nested_pair(pair)
        pair.coarse_mesh === state.mesh ||
            throw(ArgumentError("nested pair must use the retained coarse mesh"))
        _content_fingerprint(pair.coarse_rwg, UInt(0)) ==
            _content_fingerprint(state.rwg, UInt(0)) ||
            throw(ArgumentError("coarse RWG ordering does not match the nested pair"))
        state.rhs isa Vector && state.I_coeffs isa Vector ||
            throw(ArgumentError("an enriched error system requires one physical RHS"))
        nf, nc = size(pair.P)
        m = size(pair.Q, 2)
        _validate_linear_system_inputs(fine_operator, fine_rhs, "enriched system")
        size(fine_operator, 1) == nf ||
            throw(DimensionMismatch("fine operator must have $nf rows"))
        rtol, atol = Float64(restriction_rtol), Float64(restriction_atol)
        all(x -> isfinite(x) && x >= 0, (rtol, atol)) ||
            throw(ArgumentError("restriction tolerances must be finite and nonnegative"))
        limit = _validated_resource_limit("max_work_bytes", max_work_bytes)
        retained = _checked_payload_sum(
            "enriched preparation", _retained_storage_bytes(state),
            Base.summarysize((pair, fine_operator, fine_rhs)))
        buffers = _checked_payload_sum(
            "enriched preparation",
            _checked_array_payload_bytes(ComplexF64, 6, nc, nc),
            _checked_array_payload_bytes(ComplexF64, 12, nf),
            _checked_array_payload_bytes(ComplexF64, 8, nc),
            _checked_array_payload_bytes(ComplexF64, 3, m))
        _enforce_payload_limit(
            _checked_payload_sum("enriched preparation", retained, buffers),
            limit, "enriched preparation", "max_work_bytes")
        pt, qt = sparse(adjoint(pair.P)), sparse(adjoint(pair.Q))
        restricted = Matrix{ComplexF64}(undef, nc, nc)
        previous = state.operator isa StridedMatrix ?
                   Matrix{ComplexF64}(state.operator) : similar(restricted)
        unit = zeros(ComplexF64, nc)
        for column in 1:nc
            restricted[:, column] .= _error_action(
                pt, _error_column_action(fine_operator, pair.P, column))
            if !(state.operator isa StridedMatrix)
                fill!(unit, 0)
                unit[column] = 1
                previous[:, column] .= _error_action(state.operator, unit)
            end
        end
        fine_b = Vector{ComplexF64}(fine_rhs)
        bc = _error_action(pt, fine_b)
        matrix_difference = opnorm(restricted - previous, Inf)
        rhs_difference = norm(bc - state.rhs)
        matrix_scale = opnorm(restricted, Inf)
        rhs_scale = norm(bc)
        matrix_matches = matrix_difference <= atol + rtol * matrix_scale
        rhs_matches = rhs_difference <= atol + rtol * rhs_scale
        reuse = matrix_matches && rhs_matches
        reuse || rebuild_on_mismatch ||
            throw(ArgumentError(
                "coarse/fine Galerkin mismatch exceeds the declared reuse budget"))
        operator = reuse ? state.operator : restricted
        method = reuse ? state.method : :dense_direct
        available = limit - retained - buffers
        available > 0 ||
            throw(ArgumentError("no workspace remains for the consistent coarse solve"))
        exact_check = () -> _enforce_payload_limit(
            _checked_exact_dense_solve_work_bytes(
                ComplexF64, nc, 3, 3, 3, retained, buffers;
                label="consistent coarse exact factor"),
            limit, "consistent coarse exact factor", "max_work_bytes")
        factor = reuse ? state.factorization : _factor_dense_linear_system(
            restricted, ComplexF64, "consistent coarse factor";
            exact_fallback_check=exact_check)
        coarse = _retain_scattering_state(
            pair.coarse_mesh, pair.coarse_rwg, operator, factor,
            reuse ? state.preconditioner : nothing, bc, copy(state.I_coeffs),
            state.frequency_hz, state.c0, state.quad_order, method,
            state.solver_options, available; excitation=state.excitation)
        solve_prepared!(coarse)
        residual = _error_action(
            qt, fine_b - _error_action(fine_operator, pair.P * coarse.I_coeffs))
        _validate_retained_state(state)
        report = (
            matrix_absolute=matrix_difference,
            matrix_relative=iszero(matrix_scale) ? matrix_difference : matrix_difference / matrix_scale,
            rhs_absolute=rhs_difference,
            rhs_relative=iszero(rhs_scale) ? rhs_difference : rhs_difference / rhs_scale,
            factor_reused=reuse, coarse_rebuilt=!reuse,
            exact_restriction=(iszero(matrix_difference) && iszero(rhs_difference)) || !reuse,
            restriction_rtol=rtol, restriction_atol=atol,
            fine_forward_actions=nc + 1)
        return GalerkinErrorSystem(
            state, coarse, pair, fine_operator, fine_b, residual,
            state.rhs_revision, coarse.rhs_revision, report, (time_ns() - started) / 1e9,
            _error_system_signature(fine_operator, fine_b, residual, pair))
    end
end

function _enriched_information_rows(system::GalerkinErrorSystem, probes, solve_budget)
    p, q = system.pair.P, system.pair.Q
    pt, qt = sparse(adjoint(p)), sparse(adjoint(q))
    tests = q * adjoint(probes)
    tested_adjoint = _error_action_columns(adjoint(system.fine_operator), tests)
    coarse_rhs = pt * tested_adjoint
    psi = solve_prepared_adjoint!(
        system.coarse, coarse_rhs; max_work_bytes=solve_budget)
    lifted = tests - p * psi
    return Matrix(adjoint(qt * _error_action_columns(
        adjoint(system.fine_operator), lifted)))
end

function _enriched_output_rows(system::GalerkinErrorSystem, fine_rows, solve_budget)
    p, q = system.pair.P, system.pair.Q
    gc = fine_rows * p
    lambda = solve_prepared_adjoint!(
        system.coarse, Matrix(adjoint(gc)); max_work_bytes=solve_budget)
    correction = _error_action_columns(
        adjoint(system.fine_operator), p * lambda)
    h = fine_rows * q - adjoint(correction) * q
    _assert_finite_linear_array(h, "complete enriched output rows")
    return h, _error_action(gc, system.coarse.I_coeffs)
end

"""
    PreparedErrorOutputs

Complete enriched output rows H and the restricted coarse mean for one
unchanged error system. Preparation retains the fine map and records its
coarse-adjoint cost. The same rows can serve several probe/prior choices.
"""
struct PreparedErrorOutputs{S}
    system::S
    fine_output_map::Matrix{ComplexF64}
    rows::Matrix{ComplexF64}
    coarse_mean::Vector{ComplexF64}
    preparation_s::Float64
    signature::UInt
end

function _prepared_output_signature(system, fine_map, rows, base)
    return _content_fingerprint(
        (system.signature, system.origin_revision, system.coarse_revision,
         fine_map, rows, base), UInt(0))
end

function _validate_prepared_outputs(prepared::PreparedErrorOutputs)
    _validate_error_system(prepared.system)
    prepared.signature == _prepared_output_signature(
        prepared.system, prepared.fine_output_map, prepared.rows, prepared.coarse_mean) ||
        throw(ArgumentError("prepared output rows changed; rebuild the output map"))
    return nothing
end

"""
    prepare_error_outputs(system, fine_output_map;
        row_batch_size=24, max_work_bytes=2_000_000_000)

Compute H=G_f*Q-G_f*P*A^(-1)*B using coarse adjoints and fine actions.
Retain common rows for conditioning comparisons or an operator-based probe
rule. Geometry, RHS, operator or map changes invalidate the result.
"""
function prepare_error_outputs(
        system::GalerkinErrorSystem, fine_output_map::AbstractMatrix{<:Number};
        row_batch_size::Integer=24,
        max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    started = time_ns()
    _validate_error_system(system)
    nf, nc = size(system.pair.P)
    p, m = size(fine_output_map, 1), size(system.pair.Q, 2)
    p > 0 && size(fine_output_map, 2) == nf ||
        throw(DimensionMismatch("output map must have nonzero rows and $nf columns"))
    batch = min(p, _validated_resource_limit("row_batch_size", row_batch_size))
    limit = _validated_resource_limit("max_work_bytes", max_work_bytes)
    work = _checked_payload_sum(
        "prepared output rows", Base.summarysize(fine_output_map),
        _checked_array_payload_bytes(ComplexF64, p, nf),
        _checked_array_payload_bytes(ComplexF64, p, m),
        _checked_array_payload_bytes(ComplexF64, 8, batch, nf),
        _checked_array_payload_bytes(ComplexF64, 6, batch, m),
        _checked_array_payload_bytes(ComplexF64, 6, batch, nc),
        _checked_array_payload_bytes(ComplexF64, 8, nc),
        _checked_array_payload_bytes(ComplexF64, 4, p))
    solve_budget = _error_coarse_solve_budget(system, batch, limit, work)
    all(isfinite, fine_output_map) ||
        throw(ArgumentError("output map entries must be finite"))
    fine_map = Matrix{ComplexF64}(fine_output_map)
    rows = Matrix{ComplexF64}(undef, p, m)
    base = Vector{ComplexF64}(undef, p)
    for first in 1:batch:p
        indices = first:min(p, first + batch - 1)
        h, coarse = _enriched_output_rows(system, view(fine_map, indices, :), solve_budget)
        rows[indices, :] .= h
        base[indices] .= coarse
    end
    _validate_error_system(system)
    return PreparedErrorOutputs(
        system, fine_map, rows, base, (time_ns() - started) / 1e9,
        _prepared_output_signature(system, fine_map, rows, base))
end

"""
    output_residual_probes(prepared; rows=nothing,
        rank_rtol=64eps(Float64), max_work_bytes=2_000_000_000)

Build fixed output-informed residual tests from H*inv(diag(D)), where D is
the unresolved fine block. An SVD returns an orthonormal basis for the
selected row span. The rule uses operators and prescribed outputs, never
reference errors or the observed residual. Diagonal-entry access and output
preparation are part of its computational cost.
"""
function output_residual_probes(prepared::PreparedErrorOutputs;
        rows::Union{Nothing,AbstractVector{<:Integer}}=nothing,
        rank_rtol::Real=64eps(Float64),
        max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    _validate_prepared_outputs(prepared)
    p, m = size(prepared.rows)
    selected = rows === nothing ? collect(1:p) : Int.(rows)
    all(row -> 1 <= row <= p, selected) ||
        throw(ArgumentError("selected output row is outside 1:$p"))
    tolerance = Float64(rank_rtol)
    isfinite(tolerance) && 0 < tolerance < 1 ||
        throw(ArgumentError("probe rank tolerance must lie strictly between zero and one"))
    work = _checked_payload_sum(
        "output-informed probes", _error_system_storage_bytes(prepared.system),
        Base.summarysize((prepared.fine_output_map, prepared.rows, prepared.coarse_mean)),
        _checked_array_payload_bytes(ComplexF64, 8, length(selected), m),
        _checked_array_payload_bytes(ComplexF64, 4, length(selected), length(selected)),
        _checked_array_payload_bytes(ComplexF64, 3, m))
    _enforce_payload_limit(work, max_work_bytes, "output-informed probes", "max_work_bytes")
    isempty(selected) && return zeros(ComplexF64, 0, m)
    q = prepared.system.pair.Q
    diagonal = Vector{ComplexF64}(undef, m)
    for column in 1:m
        position = only(nzrange(q, column))
        edge = rowvals(q)[position]
        coefficient = nonzeros(q)[position]
        diagonal[column] = abs2(coefficient) * prepared.system.fine_operator[edge, edge]
    end
    all(value -> isfinite(value) && !iszero(value), diagonal) ||
        throw(ArgumentError("the unresolved fine diagonal cannot define these probes"))
    candidates = prepared.rows[selected, :] ./ transpose(diagonal)
    for row in axes(candidates, 1)
        scale = norm(view(candidates, row, :))
        isfinite(scale) || throw(ArgumentError("output-informed probes exceed the numerical range"))
        iszero(scale) || (candidates[row, :] ./= scale)
    end
    decomposition = LinearAlgebra.svd(candidates; full=false)
    threshold = tolerance * max(size(candidates)...) * maximum(decomposition.S)
    rank_value = count(value -> value > threshold, decomposition.S)
    return Matrix(decomposition.Vt[1:rank_value, :])
end
