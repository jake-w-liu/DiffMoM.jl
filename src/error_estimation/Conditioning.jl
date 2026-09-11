export ResidualConditioning, ConditionedErrorModel, ErrorOutputs,
       condition_error_observations, condition_discretization_error,
       evaluate_error_outputs, sample_conditioned_error, sample_error_outputs,
       quadratic_output_moments

"""
    ResidualConditioning

Proper-complex inverse-mass prior conditioned on numerical equations V*w=t.
Stores sparse Cholesky data and a thin whitened observation basis, not a dense
coefficient covariance. Diagnostics record numerical rank decisions and
observation consistency. Stored arrays and factors are read-only.
"""
struct ResidualConditioning{L}
    mass::SparseMatrixCSC{ComplexF64,Int}
    lower::L
    permutation::Vector{Int}
    information::Matrix{ComplexF64}
    observations::Vector{ComplexF64}
    tau::Float64
    right_basis::Matrix{ComplexF64}
    latent_mean::Vector{ComplexF64}
    mean_coefficients::Vector{ComplexF64}
    rank::Int
    diagnostics::NamedTuple
    signature::UInt
end

function _conditioning_signature(mass, lower, permutation, information,
                                 observations, tau, right, latent, mean)
    return _content_fingerprint(
        (mass, lower, permutation, information, observations, tau, right, latent, mean),
        UInt(0))
end

function _validate_conditioning(conditioning::ResidualConditioning)
    conditioning.signature == _conditioning_signature(
        conditioning.mass, conditioning.lower, conditioning.permutation,
        conditioning.information, conditioning.observations,
        conditioning.tau, conditioning.right_basis, conditioning.latent_mean,
        conditioning.mean_coefficients) ||
        throw(ArgumentError("conditioning data changed; rebuild the conditional model"))
    return nothing
end

function _conditioning_work_bytes(m::Int, q::Int)
    # Reserve worst-case sparse Cholesky fill, native/extracted factors,
    # sparse indices, SVD work and conversion copies before factorization.
    return _checked_payload_sum(
        "residual conditioning",
        _checked_array_payload_bytes(ComplexF64, 8, m, m),
        _checked_array_payload_bytes(Int, 4, m, m),
        _checked_array_payload_bytes(ComplexF64, 10, m, q),
        _checked_array_payload_bytes(ComplexF64, 4, q, q),
        _checked_array_payload_bytes(ComplexF64, 12, m),
        _checked_array_payload_bytes(Int, 8, m))
end

function _whiten_error_rows(lower, permutation, tau, rows)
    rhs = Matrix(adjoint(rows))[permutation, :]
    values = tau * Matrix(adjoint(lower \ rhs))
    return _assert_finite_linear_array(values, "whitened error rows")
end

function _unwhiten_error_columns(lower, permutation, tau, columns)
    values = tau * (adjoint(lower) \ columns)
    output = similar(values)
    output[permutation, :] .= values
    return _assert_finite_linear_array(output, "conditional coefficients")
end

"""
    condition_error_observations(mass, V, t; tau=1.0,
        rank_rtol=64eps(Float64), consistency_rtol=1e-10,
        max_work_bytes=2_000_000_000)

Condition w ~ CN(0, tau^2 * inv(mass)) on V*w=t. Sparse Cholesky whitens
the coefficient prior; a thin SVD identifies independent information.
Numerically dependent but consistent rows do not add information.
Inconsistent observations and non-positive-definite mass matrices fail.
No diagonal covariance substitution is used.
"""
function condition_error_observations(
        mass::AbstractMatrix{<:Number}, information::AbstractMatrix{<:Number},
        observations::AbstractVector{<:Number};
        tau::Real=1.0, rank_rtol::Real=64eps(Float64),
        consistency_rtol::Real=1e-10,
        max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    m, q = size(mass, 1), size(information, 1)
    m > 0 && size(mass, 2) == m ||
        throw(DimensionMismatch("prior mass must be nonempty and square"))
    size(information, 2) == m && length(observations) == q ||
        throw(DimensionMismatch("residual information dimensions do not match"))
    q <= m || throw(ArgumentError("probe count must not exceed the unresolved dimension"))
    limit = _validated_resource_limit("max_work_bytes", max_work_bytes)
    _enforce_payload_limit(
        _conditioning_work_bytes(m, q), limit, "residual conditioning", "max_work_bytes")
    _validate_known_matrix_entries(mass, "prior mass")
    LinearAlgebra.ishermitian(mass) ||
        throw(ArgumentError("prior mass must be Hermitian"))
    all(isfinite, information) && all(isfinite, observations) ||
        throw(ArgumentError("residual information and observations must be finite"))
    scale = Float64(tau)
    rank_tol, consistency_tol = Float64(rank_rtol), Float64(consistency_rtol)
    isfinite(scale) && scale > 0 ||
        throw(ArgumentError("tau must be finite and positive"))
    all(tol -> isfinite(tol) && 0 < tol < 1, (rank_tol, consistency_tol)) ||
        throw(ArgumentError("conditioning tolerances must lie strictly between zero and one"))
    _enforce_payload_limit(
        _checked_payload_sum("conditioning", Base.summarysize((mass, information, observations)),
                             _conditioning_work_bytes(m, q)),
        limit, "residual conditioning", "max_work_bytes")
    stored_mass = SparseMatrixCSC{ComplexF64,Int}(sparse(mass))
    v, t = Matrix{ComplexF64}(information), Vector{ComplexF64}(observations)
    _validate_known_matrix_entries(stored_mass, "converted prior mass")
    _assert_finite_linear_array(v, "converted residual information")
    _assert_finite_linear_array(t, "converted observations")
    factor = LinearAlgebra.cholesky(Hermitian(stored_mass))
    permutation = copy(factor.p)
    lower = LinearAlgebra.LowerTriangular(sparse(factor.L))
    whitened = _whiten_error_rows(lower, permutation, scale, v)
    right = zeros(ComplexF64, m, 0)
    latent = zeros(ComplexF64, m)
    values = Float64[]
    threshold = 0.0
    rank_value = 0
    support_error = 0.0
    if q > 0
        decomposition = LinearAlgebra.svd(whitened; full=false)
        values = decomposition.S
        threshold = rank_tol * max(m, q) * maximum(values)
        rank_value = count(value -> value > threshold, values)
        left = decomposition.U[:, 1:rank_value]
        right = Matrix(adjoint(decomposition.Vt[1:rank_value, :]))
        data = adjoint(left) * t
        support_error = norm(t - left * data)
        support_error <= consistency_tol * norm(t) ||
            throw(ArgumentError("observations are inconsistent with the numerical information rank"))
        if rank_value > 0
            latent .= right * (data ./ values[1:rank_value])
        end
    end
    mean = vec(_unwhiten_error_columns(
        lower, permutation, scale, reshape(latent, m, 1)))
    residual = norm(v * mean - t)
    residual <= consistency_tol * max(norm(t), norm(v) * norm(mean)) ||
        throw(ArgumentError("conditional mean fails the observation equations"))
    diagnostics = (
        supplied_rows=q, effective_rank=rank_value,
        rank_threshold=threshold, singular_values=values,
        support_residual=support_error, observation_residual=residual,
        rank_rtol=rank_tol, consistency_rtol=consistency_tol)
    return ResidualConditioning(
        stored_mass, lower, permutation, v, t, scale, right, latent, mean,
        rank_value, diagnostics,
        _conditioning_signature(stored_mass, lower, permutation, v, t, scale, right, latent, mean))
end

"""
    ErrorOutputs

Conditional complex-field mean, covariance, and a shared latent square-root
factor. The coarse mean uses the restricted coarse system. Square-root
columns are shared across all rows. Independent samples from separate output
batches would not preserve their joint distribution.
"""
struct ErrorOutputs
    mean::Vector{ComplexF64}
    covariance::Matrix{ComplexF64}
    square_root::Matrix{ComplexF64}
    coarse_mean::Vector{ComplexF64}
    diagnostics::NamedTuple
end

"""
    sample_error_outputs(outputs, rng, count; max_work_bytes=2_000_000_000)

Draw joint proper-complex output samples. A thin SVD of the existing output
factor reduces the sampling dimension to at most the number of output rows.
All singular directions are retained; there is no variance truncation.
"""
function sample_error_outputs(
        outputs::ErrorOutputs, rng::Random.AbstractRNG, count::Integer;
        max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    p, m = size(outputs.square_root)
    length(outputs.mean) == p ||
        throw(DimensionMismatch("output factor and mean dimensions do not match"))
    all(isfinite, outputs.mean) && all(isfinite, outputs.square_root) ||
        throw(ArgumentError("output mean and square-root factor must be finite"))
    n = _validated_nonnegative_resource_limit("sample count", count)
    rank_bound = min(p, m)
    work = _checked_payload_sum(
        "output samples", Base.summarysize(outputs),
        _checked_array_payload_bytes(ComplexF64, 4, p, m),
        _checked_array_payload_bytes(ComplexF64, 4, p, p),
        _checked_array_payload_bytes(ComplexF64, 3, p, n),
        _checked_array_payload_bytes(ComplexF64, 3, rank_bound, n))
    _enforce_payload_limit(work, max_work_bytes, "output samples", "max_work_bytes")
    if n == 0 || iszero(norm(outputs.square_root))
        return repeat(outputs.mean, 1, n)
    end
    decomposition = LinearAlgebra.svd(outputs.square_root; full=false)
    samples = decomposition.U * (decomposition.S .* Base.randn(rng, ComplexF64, rank_bound, n))
    samples .+= outputs.mean
    return _assert_finite_linear_array(samples, "sampled complex outputs")
end

"""
    ConditionedErrorModel

Selected enriched-residual observations and an inverse-mass conditional
model, bound to one Galerkin system and forward-RHS revision. This is a
working finite-space Gaussian model, not a calibrated prediction set.
"""
struct ConditionedErrorModel{S,C}
    system::S
    conditioning::C
    probes::Matrix{ComplexF64}
    conditioning_s::Float64
    probe_signature::UInt
end

function _validate_error_model(model::ConditionedErrorModel)
    _validate_error_system(model.system)
    _validate_conditioning(model.conditioning)
    model.probe_signature == _content_fingerprint(model.probes, UInt(0)) ||
        throw(ArgumentError("probe selection changed; rebuild the error model"))
    return nothing
end

"""
    condition_discretization_error(system, probes; tau,
        triangle_weights=nothing, max_work_bytes=2_000_000_000, kwargs...)

Compute selected Schur rows through coarse adjoints and fine operator
actions, assemble the weighted fine-space mass, restrict it with Q, and
condition unresolved coefficients. Probes are a matrix R or a vector of
unresolved row indices. No reference errors guide row selection.
Omitted triangle weights mean unit precision on every fine triangle.
"""
function condition_discretization_error(
        system::GalerkinErrorSystem, probes::AbstractMatrix{<:Number};
        tau::Real,
        triangle_weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
        max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
        rank_rtol::Real=64eps(Float64), consistency_rtol::Real=1e-10)
    started = time_ns()
    _validate_error_system(system)
    nf, nc = size(system.pair.P)
    m = size(system.pair.Q, 2)
    q = size(probes, 1)
    size(probes, 2) == m && q <= m ||
        throw(DimensionMismatch("probe matrix must have $m columns and at most $m rows"))
    all(isfinite, probes) || throw(ArgumentError("probe entries must be finite"))
    limit = _validated_resource_limit("max_work_bytes", max_work_bytes)
    retained = _error_system_storage_bytes(system)
    rows_bytes = _checked_payload_sum(
        "Schur rows", _checked_array_payload_bytes(ComplexF64, 8, nf, q),
        _checked_array_payload_bytes(ComplexF64, 6, nc, q),
        _checked_array_payload_bytes(ComplexF64, 8, nc),
        _checked_array_payload_bytes(ComplexF64, 4, m, q))
    _enforce_payload_limit(
        _checked_payload_sum("Schur conditioning", retained, rows_bytes,
                             _conditioning_work_bytes(m, q)),
        limit, "Schur conditioning", "max_work_bytes")
    solve_budget = q == 0 ? 0 : _error_coarse_solve_budget(system, q, limit, rows_bytes)
    r = Matrix{ComplexF64}(probes)
    v = q == 0 ? zeros(ComplexF64, 0, m) :
        _enriched_information_rows(system, r, solve_budget)
    t = r * system.residual
    fine_mass = assemble_rwg_gram(
        system.pair.fine_mesh, system.pair.fine_rwg;
        triangle_weights=triangle_weights, max_work_bytes=limit - retained - rows_bytes)
    mass = sparse(adjoint(system.pair.Q) * fine_mass * system.pair.Q)
    conditioning = condition_error_observations(
        mass, v, t; tau=tau, rank_rtol=rank_rtol,
        consistency_rtol=consistency_rtol,
        max_work_bytes=limit - retained -
            Base.summarysize((fine_mass, r, v, t)))
    _validate_error_system(system)
    return ConditionedErrorModel(
        system, conditioning, r, (time_ns() - started) / 1e9,
        _content_fingerprint(r, UInt(0)))
end

function condition_discretization_error(
        system::GalerkinErrorSystem, rows::AbstractVector{<:Integer}; kwargs...)
    m = size(system.pair.Q, 2)
    length(rows) <= m || throw(ArgumentError("too many residual probes"))
    all(row -> 1 <= row <= m, rows) ||
        throw(ArgumentError("residual probe indices must lie in 1:$m"))
    probes = sparse(collect(1:length(rows)), Int.(rows), ones(length(rows)), length(rows), m)
    return condition_discretization_error(system, probes; kwargs...)
end

"""
    evaluate_error_outputs(model, fine_output_map;
        row_batch_size=24, max_work_bytes=2_000_000_000)

Propagate H=G_f*Q-G_f*P*A^(-1)*B through selected coarse adjoints. Project
whitened output rows and form their Gram covariance. Batches retain common
latent coordinates and cross-row covariance. No current covariance is formed.
"""
function evaluate_error_outputs(
        model::ConditionedErrorModel, fine_output_map::AbstractMatrix{<:Number};
        row_batch_size::Integer=24,
        max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    started = time_ns()
    prepared = prepare_error_outputs(
        model.system, fine_output_map; row_batch_size=row_batch_size,
        max_work_bytes=max_work_bytes)
    outputs = evaluate_error_outputs(
        model, prepared; row_batch_size=row_batch_size, max_work_bytes=max_work_bytes)
    diagnostics = merge(outputs.diagnostics, (
        elapsed_s=(time_ns() - started) / 1e9,
        coarse_adjoint_rhs=size(fine_output_map, 1),
        fine_adjoint_actions=size(fine_output_map, 1),
        prepared_output_rows=false))
    return ErrorOutputs(
        outputs.mean, outputs.covariance, outputs.square_root,
        outputs.coarse_mean, diagnostics)
end

function evaluate_error_outputs(
        model::ConditionedErrorModel, prepared::PreparedErrorOutputs;
        row_batch_size::Integer=24,
        max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    started = time_ns()
    system, conditioning = model.system, model.conditioning
    _validate_error_model(model)
    prepared.system === system ||
        throw(ArgumentError("prepared outputs belong to a different error system"))
    _validate_prepared_outputs(prepared)
    p, m = size(prepared.rows)
    batch = min(p, _validated_resource_limit("row_batch_size", row_batch_size))
    limit = _validated_resource_limit("max_work_bytes", max_work_bytes)
    retained = _checked_payload_sum(
        "output moments", _error_system_storage_bytes(system),
        Base.summarysize((conditioning, prepared.fine_output_map, prepared.rows,
                          prepared.coarse_mean, model.probes)))
    work = _checked_payload_sum(
        "output moments",
        _checked_array_payload_bytes(ComplexF64, 8, batch, m),
        _checked_array_payload_bytes(ComplexF64, p, m),
        _checked_array_payload_bytes(ComplexF64, 2, p, p),
        _checked_array_payload_bytes(ComplexF64, 4, p))
    _enforce_payload_limit(
        _checked_payload_sum("output moments", retained, work),
        limit, "output moments", "max_work_bytes")
    mean = zeros(ComplexF64, p)
    coarse_mean = copy(prepared.coarse_mean)
    square_root = Matrix{ComplexF64}(undef, p, m)
    for first in 1:batch:p
        indices = first:min(p, first + batch - 1)
        h, base = view(prepared.rows, indices, :), view(coarse_mean, indices)
        mean[indices] .= base + _error_action(h, conditioning.mean_coefficients)
        whitened = _whiten_error_rows(
            conditioning.lower, conditioning.permutation, conditioning.tau, h)
        if conditioning.rank == m
            square_root[indices, :] .= 0
        else
            projected = whitened * conditioning.right_basis
            square_root[indices, :] .= whitened - projected * adjoint(conditioning.right_basis)
        end
    end
    covariance = square_root * adjoint(square_root)
    _assert_finite_linear_array(mean, "conditional output mean")
    _assert_finite_linear_array(covariance, "conditional output covariance")
    _validate_error_model(model)
    return ErrorOutputs(
        mean, covariance, square_root, coarse_mean,
        (output_rows=p, coarse_adjoint_rhs=0, fine_adjoint_actions=0,
         elapsed_s=(time_ns() - started) / 1e9,
         output_preparation_s=prepared.preparation_s, prepared_output_rows=true,
         exact_restriction=system.restriction_report.exact_restriction,
         observation_rank=conditioning.rank))
end

"""
    sample_conditioned_error(conditioning, rng, count;
        max_work_bytes=2_000_000_000)

Draw proper-complex coefficient samples with shared latent coordinates and
an explicit RNG. A zero count returns an empty matrix. Full information
returns conditional means without random perturbations.
"""
function sample_conditioned_error(
        conditioning::ResidualConditioning, rng::Random.AbstractRNG, count::Integer;
        max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    _validate_conditioning(conditioning)
    n = _validated_nonnegative_resource_limit("sample count", count)
    m = size(conditioning.mass, 1)
    bytes = _checked_payload_sum(
        "conditional samples", Base.summarysize(conditioning),
        _checked_array_payload_bytes(ComplexF64, 5, m, n),
        _checked_array_payload_bytes(ComplexF64, 3, conditioning.rank, n))
    _enforce_payload_limit(bytes, max_work_bytes, "conditional samples", "max_work_bytes")
    latent = conditioning.rank == m ? zeros(ComplexF64, m, n) :
             Base.randn(rng, ComplexF64, m, n)
    if 0 < conditioning.rank < m
        latent .-= conditioning.right_basis * (adjoint(conditioning.right_basis) * latent)
    end
    latent .+= conditioning.latent_mean
    return _unwhiten_error_columns(
        conditioning.lower, conditioning.permutation, conditioning.tau, latent)
end

"""
    quadratic_output_moments(outputs, W)

Return mean and variance of y'W*y for the working proper-complex Gaussian
output and Hermitian W. These are model moments, not calibrated probabilities
or the RCS of the mean current alone.
"""
function quadratic_output_moments(outputs::ErrorOutputs, weight::AbstractMatrix{<:Number})
    p = length(outputs.mean)
    size(weight) == (p, p) || throw(DimensionMismatch("quadratic weight must be $p by $p"))
    all(isfinite, weight) && LinearAlgebra.ishermitian(weight) ||
        throw(ArgumentError("quadratic weight must be finite and Hermitian"))
    weighted_mean = weight * outputs.mean
    product = weight * outputs.covariance
    expected = real(dot(outputs.mean, weighted_mean) + LinearAlgebra.tr(product))
    variance = real(LinearAlgebra.tr(product * product) +
                    2dot(weighted_mean, outputs.covariance * weighted_mean))
    isfinite(expected) && isfinite(variance) && variance >= 0 ||
        throw(ArgumentError("quadratic moments exceed the supported numerical range"))
    return (mean=expected, variance=variance)
end
