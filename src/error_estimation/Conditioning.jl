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

@inline function _conditioning_can_defer_collection(
        released_bytes::Int, work_bytes::Int, input_bytes::Int,
        remaining_budget::Int, available_bytes::Integer)
    # Subtract only after each comparison so neither a tight budget nor a
    # large unsigned OS estimate can overflow the decision arithmetic.
    work_bytes <= remaining_budget || return false
    input_bytes <= remaining_budget - work_bytes || return false
    released_bytes <= remaining_budget - work_bytes - input_bytes || return false
    available_bytes > 0 && work_bytes <= available_bytes || return false
    return released_bytes <= available_bytes - work_bytes
end

function _whiten_error_rows(lower, permutation, tau, rows)
    # rhs[i, j] = conj(rows[j, permutation[i]]): fuse the adjoint and the
    # row permutation into a single fill instead of materializing both.
    m, q = size(rows, 2), size(rows, 1)
    rhs = Matrix{ComplexF64}(undef, m, q)
    @inbounds for column in 1:q, row in 1:m
        rhs[row, column] = conj(rows[column, permutation[row]])
    end
    ldiv!(lower, rhs)
    values = Matrix{ComplexF64}(undef, q, m)
    @inbounds for column in 1:m, row in 1:q
        values[row, column] = tau * conj(rhs[column, row])
    end
    return _assert_finite_linear_array(values, "whitened error rows")
end

function _unwhiten_error_columns(lower, permutation, tau, columns)
    solved = adjoint(lower) \ columns
    output = similar(solved)
    @inbounds for column in axes(output, 2), row in axes(output, 1)
        output[permutation[row], column] = tau * solved[row, column]
    end
    return _assert_finite_linear_array(output, "conditional coefficients")
end

function _preflight_conditioning_range_work(m, q, input_bytes, limit)
    scalar_bytes = cld(_LOCAL_MASS_FALLBACK_PRECISION, 8sizeof(UInt)) *
                   sizeof(UInt) + 8sizeof(UInt)
    wide = _checked_array_payload_bytes(UInt8, 2scalar_bytes,
        3BigInt(m) * q + 2BigInt(q)^2 + 8BigInt(m) + 12BigInt(q) + 64)
    _enforce_payload_limit(
        _checked_payload_sum("wide residual conditioning", input_bytes,
                             _conditioning_work_bytes(m, q), wide),
        limit, "wide residual conditioning", "max_work_bytes")
end

function _conditioning_range_projection(
        left, right, values, t, tolerance, m, q, input_bytes, limit)
    _preflight_conditioning_range_work(m, q, input_bytes, limit)
    return setprecision(BigFloat, _LOCAL_MASS_FALLBACK_PRECISION) do
        setrounding(BigFloat, RoundNearest) do
            basis = Complex{BigFloat}.(left)
            observations = Complex{BigFloat}.(t)
            data = adjoint(basis) * observations
            support = norm(observations - basis * data)
            support <= BigFloat(tolerance) * norm(observations) ||
                throw(ArgumentError("observations are inconsistent with the numerical information rank"))
            latent = ComplexF64.(Complex{BigFloat}.(right) *
                                 (data ./ BigFloat.(values)))
            _assert_finite_linear_array(latent, "conditional latent mean")
            return latent, Float64(support)
        end
    end
end

function _conditioning_range_residual(
        v, mean, t, tolerance, m, q, input_bytes, limit)
    _preflight_conditioning_range_work(m, q, input_bytes, limit)
    return setprecision(BigFloat, _LOCAL_MASS_FALLBACK_PRECISION) do
        setrounding(BigFloat, RoundNearest) do
            information = Complex{BigFloat}.(v)
            coefficients = Complex{BigFloat}.(mean)
            observations = Complex{BigFloat}.(t)
            residual = norm(information * coefficients - observations)
            bound = BigFloat(tolerance) *
                    max(norm(observations), norm(information) * norm(coefficients))
            residual <= bound ||
                throw(ArgumentError("conditional mean fails the observation equations"))
            return Float64(residual)
        end
    end
end

"""
    condition_error_observations(mass, V, t; tau=1.0,
        rank_rtol=64eps(Float64), consistency_rtol=1e-10,
        max_work_bytes=Sys.free_memory())

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
        max_work_bytes::Integer=_default_max_dense_payload_bytes())
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
    input_bytes = Base.summarysize((mass, information, observations))
    _enforce_payload_limit(
        _checked_payload_sum("conditioning", input_bytes,
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
        observation_norm = norm(t)
        if isfinite(support_error) && isfinite(observation_norm)
            support_error <= consistency_tol * observation_norm ||
                throw(ArgumentError("observations are inconsistent with the numerical information rank"))
            if rank_value > 0
                latent .= right * (data ./ values[1:rank_value])
            end
        end
        # Infinity is not a consistency certificate. Recompute the coupled
        # projection/division before narrowing when finite inputs overflow
        # an intermediate norm, SVD coordinate, or latent coefficient.
        if !isfinite(support_error) || !isfinite(observation_norm) || !all(isfinite, latent)
            latent, support_error = _conditioning_range_projection(left, right,
                view(values, 1:rank_value), t, consistency_tol, m, q, input_bytes, limit)
        end
    end
    mean = vec(_unwhiten_error_columns(
        lower, permutation, scale, reshape(latent, m, 1)))
    residual = norm(v * mean - t)
    residual_bound = consistency_tol * max(norm(t), norm(v) * norm(mean))
    if isfinite(residual) && isfinite(residual_bound)
        residual <= residual_bound ||
            throw(ArgumentError("conditional mean fails the observation equations"))
    else
        residual = _conditioning_range_residual(
            v, mean, t, consistency_tol, m, q, input_bytes, limit)
    end
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
    sample_error_outputs(outputs, rng, count; max_work_bytes=Sys.free_memory())

Draw joint proper-complex output samples. A thin SVD of the existing output
factor reduces the sampling dimension to at most the number of output rows.
All singular directions are retained; there is no variance truncation.
"""
function sample_error_outputs(
        outputs::ErrorOutputs, rng::Random.AbstractRNG, count::Integer;
        max_work_bytes::Integer=_default_max_dense_payload_bytes())
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
        triangle_weights=nothing, max_work_bytes=Sys.free_memory(), kwargs...)

Compute selected Schur rows through coarse adjoints and fine operator
actions, assemble the weighted fine-space mass, restrict it with Q, and
condition unresolved coefficients. Probes are a matrix R or a vector of
unresolved row indices. No reference errors guide row selection.
Omitted triangle weights mean unit precision on every fine triangle.
When budget and available-memory estimates allow it, the discarded fine mass
remains charged against the conditioning workspace until automatic collection;
tighter or memory-constrained calls collect it before sparse factorization.
"""
function condition_discretization_error(
        system::GalerkinErrorSystem, probes::AbstractMatrix{<:Number};
        tau::Real,
        triangle_weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
        max_work_bytes::Integer=_default_max_dense_payload_bytes(),
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
    released_mass_bytes = Base.summarysize(fine_mass)
    conditioning_budget = limit - retained - Base.summarysize((r, v, t))
    condition_work = _conditioning_work_bytes(m, q)
    condition_inputs = Base.summarysize((mass, v, t))
    defer_collection = _conditioning_can_defer_collection(
        released_mass_bytes, condition_work, condition_inputs,
        conditioning_budget, _planar_os_available_memory())
    fine_mass = nothing
    if defer_collection
        # Charge the whole discarded object even if automatic GC reclaims it
        # sooner. Together with the child's input/work preflight, this keeps
        # the same total ceiling without forcing a process-wide collection.
        conditioning_budget -= released_mass_bytes
    else
        GC.gc()
    end
    conditioning = condition_error_observations(
        mass, v, t; tau=tau, rank_rtol=rank_rtol,
        consistency_rtol=consistency_rtol,
        max_work_bytes=conditioning_budget)
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
        row_batch_size=24, max_work_bytes=Sys.free_memory())

Propagate H=G_f*Q-G_f*P*A^(-1)*B through selected coarse adjoints. Project
whitened output rows and form their Gram covariance. Batches retain common
latent coordinates and cross-row covariance. No current covariance is formed.
"""
function evaluate_error_outputs(
        model::ConditionedErrorModel, fine_output_map::AbstractMatrix{<:Number};
        row_batch_size::Integer=24,
        max_work_bytes::Integer=_default_max_dense_payload_bytes())
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
        max_work_bytes::Integer=_default_max_dense_payload_bytes())
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
            target = view(square_root, indices, :)
            copyto!(target, whitened)
            mul!(target, projected, adjoint(conditioning.right_basis),
                 -one(ComplexF64), true)
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
        max_work_bytes=Sys.free_memory())

Draw proper-complex coefficient samples with shared latent coordinates and
an explicit RNG. A zero count returns an empty matrix. Full information
returns conditional means without random perturbations.
"""
function sample_conditioned_error(
        conditioning::ResidualConditioning, rng::Random.AbstractRNG, count::Integer;
        max_work_bytes::Integer=_default_max_dense_payload_bytes())
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
        coefficients = adjoint(conditioning.right_basis) * latent
        mul!(latent, conditioning.right_basis, coefficients,
             -one(ComplexF64), true)
    end
    latent .+= conditioning.latent_mean
    return _unwhiten_error_columns(
        conditioning.lower, conditioning.permutation, conditioning.tau, latent)
end

function _quadratic_output_bilinear(left, weight, right, component, ::Type{T}) where {T}
    if T <: Union{Float32,Float64}
        return _bilinear_component_bigfloat(left, weight, right, component, T,
            "quadratic output moment")
    end
    span=_local_mass_dyadic_span((value for array in (left,weight,right) for value in array))
    bits=span===nothing ? max(2*_LOCAL_MASS_FALLBACK_PRECISION,precision(BigFloat)) :
        _local_mass_dyadic_precision(3*(BigInt(span[2])+1)+1,3*BigInt(span[1]),
            8*BigInt(max(length(left),1))*max(length(right),1))
    return setprecision(BigFloat,bits) do
        setrounding(BigFloat,RoundNearest) do
            _accumulate_bilinear_component_bigfloat(left, weight, right, component)
        end
    end
end

function _quadratic_output_moment_precision(outputs,weight)
    values=(value for array in (outputs.square_root,outputs.mean,weight) for value in array)
    span=_local_mass_dyadic_span(values)
    # Variance expands into W^2*F^4 and W^2*mu^2*F^2. Six original
    # dyadic factors and the reduction carry suffice for every intermediate,
    # so cancellation is retained before the one final output conversion.
    # Nondyadic Number inputs retain a finite approximation budget.
    span===nothing && return max(2*_LOCAL_MASS_FALLBACK_PRECISION,precision(BigFloat))
    p,m=size(outputs.square_root)
    terms=64*BigInt(max(p,1))^4*(BigInt(m)+1)^2
    return _local_mass_dyadic_precision(
        6*(BigInt(span[2])+1)+2,6*BigInt(span[1]),terms)
end

function _quadratic_output_wide_factor_moments(outputs,weight,::Type{T}) where {T}
    p,m=size(outputs.square_root)
    bits=_quadratic_output_moment_precision(outputs,weight)
    scalar_bytes=cld(BigInt(bits),8sizeof(UInt))*sizeof(UInt)+8sizeof(UInt)
    # Three p-by-p wide matrices, linear column/vector work, and scalar
    # temporaries. The latent dimension never produces an m-by-m array.
    work=_checked_array_payload_bytes(UInt8,2scalar_bytes,3BigInt(p)^2+8BigInt(p)+64)
    _enforce_payload_limit(work,_default_max_dense_payload_bytes(),
        "wide quadratic moment workspace","available memory")
    return setprecision(BigFloat,bits) do
        setrounding(BigFloat,RoundNearest) do
            covariance=zeros(Complex{BigFloat},p,p)
            column=Vector{Complex{BigFloat}}(undef,p)
            for latent in 1:m
                for row in 1:p
                    column[row]=Complex{BigFloat}(outputs.square_root[row,latent])
                end
                for j in 1:p,i in 1:j
                    covariance[i,j]+=column[i]*conj(column[j])
                end
            end
            for j in 1:p,i in j+1:p
                covariance[i,j]=conj(covariance[j,i])
            end
            w=Complex{BigFloat}.(weight)
            mean=Complex{BigFloat}.(outputs.mean)
            product=w*covariance
            weighted_mean=w*mean
            expected=real(dot(mean,weighted_mean)+LinearAlgebra.tr(product))
            # Contract the trace directly instead of retaining product^2.
            trace_square=sum(product[i,j]*product[j,i] for i in 1:p,j in 1:p)
            shifted=sum(conj(weighted_mean[i])*covariance[i,j]*weighted_mean[j]
                for i in 1:p,j in 1:p)
            variance=real(trace_square)+2real(shifted)
            return (mean=convert(T,expected),variance=convert(T,variance))
        end
    end
end

function _quadratic_output_factor_moments(outputs, weight)
    factor = outputs.square_root
    all(isfinite, factor) ||
        throw(ArgumentError("output square-root factor must be finite"))
    real_type = isempty(weight) ? Float64 :
        typeof(real(weight[1, 1] * zero(ComplexF64)))
    if all(iszero, factor)
        expected = all(iszero, outputs.mean) ? zero(real_type) :
            _quadratic_output_bilinear(
                outputs.mean, weight, outputs.mean, Val(:real), real_type)
        return (mean=expected, variance=zero(expected))
    end
    p, m = size(factor)
    m>p && return _quadratic_output_wide_factor_moments(outputs,weight,real_type)
    rank = size(factor, 2)
    latent_weight = Matrix{Complex{real_type}}(undef, rank, rank)
    latent_mean = Vector{Complex{real_type}}(undef, rank)
    for column in 1:rank
        right = view(factor, :, column)
        for row in 1:rank
            left = view(factor, :, row)
            latent_weight[row, column] = complex(
                _quadratic_output_bilinear(left, weight, right, Val(:real), real_type),
                _quadratic_output_bilinear(left, weight, right, Val(:imag), real_type))
        end
        latent_mean[column] = complex(
            _quadratic_output_bilinear(right, weight, outputs.mean, Val(:real), real_type),
            _quadratic_output_bilinear(right, weight, outputs.mean, Val(:imag), real_type))
    end
    expected = _quadratic_output_bilinear(
        outputs.mean, weight, outputs.mean, Val(:real), real_type) +
        real(LinearAlgebra.tr(latent_weight))
    variance = sum(abs2, latent_weight) + 2sum(abs2, latent_mean)
    return (mean=expected, variance=variance)
end

function _quadratic_output_factor_mean(outputs, weight, ::Type{T}) where {T}
    factor = outputs.square_root
    all(isfinite, factor) ||
        throw(ArgumentError("output square-root factor must be finite"))
    if all(iszero, factor)
        return all(iszero, outputs.mean) ? zero(T) : _quadratic_output_bilinear(
            outputs.mean, weight, outputs.mean, Val(:real), T)
    end
    if T <: Union{Float32,Float64} &&
       _ieee_bilinear_superaccumulator_supported(outputs.mean, weight, outputs.mean, T) &&
       _ieee_bilinear_supported_eltype(eltype(factor), T)
        accumulator = _IEEEBilinearAccumulator(T)
        _accumulate_bilinear_component_ieee!(
            accumulator, outputs.mean, weight, outputs.mean, Val(:real))
        for column in axes(factor, 2)
            values = view(factor, :, column)
            _accumulate_bilinear_component_ieee!(
                accumulator, values, weight, values, Val(:real))
        end
        return _ieee_bilinear_finish(accumulator, "quadratic output mean")
    end
    return setprecision(BigFloat,_quadratic_output_moment_precision(outputs,weight)) do
        setrounding(BigFloat,RoundNearest) do
            expected = _accumulate_bilinear_component_bigfloat(
                outputs.mean, weight, outputs.mean, Val(:real))
            for column in axes(factor, 2)
                values = view(factor, :, column)
                expected += _accumulate_bilinear_component_bigfloat(
                    values, weight, values, Val(:real))
            end
            convert(T, expected)
        end
    end
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
    size(outputs.covariance) == (p, p) && size(outputs.square_root, 1) == p ||
        throw(DimensionMismatch("output moments and square-root dimensions do not match"))
    all(isfinite, outputs.mean) && all(isfinite, outputs.covariance) ||
        throw(ArgumentError("output means and covariance must be finite"))
    all(isfinite, weight) && LinearAlgebra.ishermitian(weight) ||
        throw(ArgumentError("quadratic weight must be finite and Hermitian"))
    if all(iszero, weight)
        # An exactly zero weight proves both moments vanish, regardless of
        # covariance scale. Avoid a redundant SVD and latent rank-squared work.
        all(isfinite, outputs.square_root) ||
            throw(ArgumentError("output square-root factor must be finite"))
        value = isempty(weight) ? 0.0 : zero(real(weight[1, 1] * zero(ComplexF64)))
        return (mean=value, variance=value)
    end
    weighted_mean = weight * outputs.mean
    product = weight * outputs.covariance
    expected = real(dot(outputs.mean, weighted_mean) + LinearAlgebra.tr(product))
    variance = real(LinearAlgebra.tr(product * product) +
                    2dot(weighted_mean, outputs.covariance * weighted_mean))
    # Indefinite Hermitian weights can make the trace cancel between large
    # terms even for a perfectly valid Gaussian covariance. In that case
    # use the equivalent sum of squared latent weights, which cannot erase
    # a positive variance by subtracting those squared terms.
    weight_norm, mean_norm = norm(weight), norm(outputs.mean)
    covariance_norm = norm(outputs.covariance)
    expected_scale = (weight_norm * mean_norm) * mean_norm + weight_norm * covariance_norm
    mean_requires_exact = !isfinite(expected) || !isfinite(expected_scale) ||
                          abs(expected) <= 64eps(Float64) * expected_scale
    variance_scale = norm(product)^2 +
                     2covariance_norm * norm(weighted_mean)^2
    if !isfinite(variance) || !isfinite(variance_scale) ||
       variance < 0 || variance <= 64eps(Float64) * variance_scale
        stable = _quadratic_output_factor_moments(outputs, weight)
        expected, variance = stable.mean, stable.variance
    end
    # Preserve cancellation within each quadratic form and between its mean
    # and covariance contributions until one final conversion. Accumulate the
    # original factor columns directly: SVD compression or rounded latent
    # diagonal entries can erase a small, representable expectation.
    mean_requires_exact &&
        (expected = _quadratic_output_factor_mean(outputs, weight, typeof(expected)))
    isfinite(expected) && isfinite(variance) && variance >= 0 ||
        throw(ArgumentError("quadratic moments exceed the supported numerical range"))
    return (mean=expected, variance=variance)
end
