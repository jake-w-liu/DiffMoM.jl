# DensityFiltering.jl — Density filter and Heaviside projection for topology optimization
#
# Two-stage pipeline:  ρ (raw) → ρ̃ (filtered) → ρ̄ (projected)
#
# Stage 1: Conic density filter (ensures minimum feature size)
#   ρ̃_t = Σ_s w_ts ρ_s / Σ_s w_ts
#   where w_ts = max(0, r_min - d_ts)
#
# Stage 2: Heaviside projection (drives toward 0/1 binary design)
#   ρ̄_t = [tanh(β η) + tanh(β (ρ̃_t - η))] / [tanh(β η) + tanh(β (1 - η))]
#
# Both stages have exact derivatives for adjoint chain rule.
#
# Reference: Lazarov & Sigmund (2011), Wang et al. (2011)

export build_filter_weights, apply_filter, apply_filter_transpose
export heaviside_project, heaviside_derivative
export filter_and_project, gradient_chain_rule

const _DEFAULT_MAX_FILTER_TRIPLET_BYTES = 512 * 1024 * 1024
const _FILTER_TRIPLET_ENTRY_BYTES =
    2 * sizeof(Int) + sizeof(Float64)

@inline function _filter_cell_indices_fit_int(
    centroids::Vector{Vec3},
    origin::Vec3,
    r_min::Float64,
)
    inv_h = 1.0 / r_min
    isfinite(inv_h) || return false
    int_upper_bound = Float64(typemax(Int))
    @inbounds for centroid in centroids
        for component in 1:3
            delta = centroid[component] - origin[component]
            scaled = delta * inv_h
            (isfinite(scaled) && scaled >= 0.0 &&
             scaled < int_upper_bound) || return false
        end
    end
    return true
end

function _filter_triplet_count(
    centroids::Vector{Vec3},
    r_min::Float64,
    cells::Vector{NTuple{3,T}},
    buckets::Dict{NTuple{3,T},Vector{Int}},
    byte_limit::Int,
) where {T<:Integer}
    entry_limit = div(byte_limit, _FILTER_TRIPLET_ENTRY_BYTES)
    entry_count = 0

    @inbounds for t in eachindex(centroids)
        ct = centroids[t]
        (cx, cy, cz) = cells[t]
        for dz in -1:1, dy in -1:1, dx in -1:1
            neigh = get(buckets, (cx + dx, cy + dy, cz + dz), nothing)
            neigh === nothing && continue
            for s in neigh
                s < t && continue
                d = norm(ct - centroids[s])
                w = max(0.0, r_min - d)
                w > 0 || continue
                required = s == t ? 1 : 2
                required <= entry_limit - entry_count ||
                    throw(ArgumentError(
                        "filter triplet payload exceeds " *
                        "max_triplet_bytes=$byte_limit after " *
                        "$entry_count retained entries"))
                entry_count += required
            end
        end
    end
    return entry_count
end

function _filter_triplets_from_cells(
    centroids::Vector{Vec3},
    r_min::Float64,
    cells::Vector{NTuple{3,T}},
    buckets::Dict{NTuple{3,T},Vector{Int}},
    byte_limit::Int,
) where {T<:Integer}
    entry_count = _filter_triplet_count(
        centroids, r_min, cells, buckets, byte_limit)
    rows = Vector{Int}(undef, entry_count)
    cols = Vector{Int}(undef, entry_count)
    vals = Vector{Float64}(undef, entry_count)
    cursor = 0

    @inbounds for t in eachindex(centroids)
        ct = centroids[t]
        (cx, cy, cz) = cells[t]
        for dz in -1:1, dy in -1:1, dx in -1:1
            neigh = get(buckets, (cx + dx, cy + dy, cz + dz), nothing)
            neigh === nothing && continue
            for s in neigh
                s < t && continue
                d = norm(ct - centroids[s])
                w = max(0.0, r_min - d)
                w > 0 || continue
                cursor += 1
                rows[cursor] = t
                cols[cursor] = s
                vals[cursor] = w
                if s != t
                    cursor += 1
                    rows[cursor] = s
                    cols[cursor] = t
                    vals[cursor] = w
                end
            end
        end
    end
    cursor == entry_count ||
        error("internal filter triplet count changed between passes")
    return rows, cols, vals
end

function _filter_weight_triplets_int(
    centroids::Vector{Vec3},
    origin::Vec3,
    r_min::Float64,
    byte_limit::Int,
)
    Nt = length(centroids)
    inv_h = 1.0 / r_min
    buckets = Dict{NTuple{3,Int},Vector{Int}}()
    cells = Vector{NTuple{3,Int}}(undef, Nt)
    @inbounds for t in 1:Nt
        centroid = centroids[t]
        key = ntuple(component ->
            floor(Int, (centroid[component] - origin[component]) * inv_h), 3)
        cells[t] = key
        push!(get!(() -> Int[], buckets, key), t)
    end
    return _filter_triplets_from_cells(
        centroids, r_min, cells, buckets, byte_limit)
end

@noinline function _filter_weight_triplets_bigint(
    centroids::Vector{Vec3},
    origin::Vec3,
    r_min::Float64,
    byte_limit::Int,
)
    Nt = length(centroids)
    exact_origin = ntuple(
        component -> Rational{BigInt}(origin[component]), 3)
    exact_cell_size = Rational{BigInt}(r_min)
    buckets = Dict{NTuple{3,BigInt},Vector{Int}}()
    cells = Vector{NTuple{3,BigInt}}(undef, Nt)
    @inbounds for t in 1:Nt
        centroid = centroids[t]
        key = ntuple(component -> begin
            exact_delta = Rational{BigInt}(centroid[component]) -
                          exact_origin[component]
            floor(BigInt, exact_delta / exact_cell_size)
        end, 3)
        cells[t] = key
        push!(get!(() -> Int[], buckets, key), t)
    end
    return _filter_triplets_from_cells(
        centroids, r_min, cells, buckets, byte_limit)
end

"""
    build_filter_weights(mesh, r_min;
                         max_triplet_bytes=536_870_912)

Build the sparse conic filter weight matrix W and normalization vector w_sum.

For each triangle pair (t, s):
  w_ts = max(0, r_min - dist(centroid_t, centroid_s))

Returns (W::SparseMatrix, w_sum::Vector) where w_sum[t] = Σ_s W[t,s].
`r_min` must be finite and positive.
`max_triplet_bytes` is a positive raw-payload limit for the temporary row,
column, and value arrays used to construct `W`. The required number of
triplets is counted and checked before those arrays are allocated.

A filter edge exists only when `dist < r_min`. Rather than testing all
`O(Nt^2)` centroid pairs, centroids are bucketed into a uniform spatial grid
with cell size `r_min`; any pair within `r_min` must then lie in the same or an
adjacent cell, so only the 3×3×3 neighbour stencil of each cell is searched
(~`O(Nt)` average for quasi-uniform meshes). The conic weight is symmetric
(`w_ts = w_st`), so each unordered pair is computed once and stored on both
sides, producing the identical sparsity pattern and values as the brute-force
double loop.
"""
function build_filter_weights(
        mesh::TriMesh,
        r_min::Float64;
        max_triplet_bytes::Integer=_DEFAULT_MAX_FILTER_TRIPLET_BYTES)
    isfinite(r_min) && r_min > 0 ||
        throw(ArgumentError(
            "r_min must be finite and positive, got $r_min"))
    byte_limit = _validated_resource_limit(
        "max_triplet_bytes", max_triplet_bytes)
    Nt = ntriangles(mesh)

    # Compute triangle centroids
    centroids = Vector{Vec3}(undef, Nt)
    for t in 1:Nt
        centroids[t] = triangle_center(mesh, t)
        all(isfinite, centroids[t]) ||
            throw(ArgumentError(
                "triangle $t centroid is outside the finite Float64 range"))
    end

    if Nt == 0
        rows = Int[]
        cols = Int[]
        vals = Float64[]
        W = sparse(rows, cols, vals, Nt, Nt)
        w_sum = vec(sum(W, dims=2))
        return W, w_sum
    end

    # Spatial hash grid: cell size = r_min so neighbours within r_min lie in the
    # 3×3×3 stencil around a centroid's cell. Cell indices are integers derived
    # from a common origin (the minimum corner of the centroid bounding box).
    xmin = centroids[1][1]; ymin = centroids[1][2]; zmin = centroids[1][3]
    @inbounds for t in 2:Nt
        c = centroids[t]
        xmin = min(xmin, c[1]); ymin = min(ymin, c[2]); zmin = min(zmin, c[3])
    end
    origin = Vec3(xmin, ymin, zmin)

    rows, cols, vals = if _filter_cell_indices_fit_int(
            centroids, origin, r_min)
        _filter_weight_triplets_int(
            centroids, origin, r_min, byte_limit)
    else
        _filter_weight_triplets_bigint(
            centroids, origin, r_min, byte_limit)
    end

    # Global scaling leaves W*rho ./ row_sums(W) and its adjoint unchanged,
    # while preventing the finite conic weights from overflowing their row
    # reductions when r_min is close to floatmax.
    maximum_weight = isempty(vals) ? 0.0 : maximum(vals)
    if maximum_weight > 1.0
        inv_scale = inv(maximum_weight)
        @inbounds for index in eachindex(vals)
            vals[index] *= inv_scale
        end
    end

    W = sparse(rows, cols, vals, Nt, Nt)
    w_sum = vec(sum(W, dims=2))

    return W, w_sum
end

"""
    apply_filter(W, w_sum, rho)

Apply the conic density filter:
  ρ̃_t = Σ_s W[t,s] ρ_s / w_sum[t]

The input dimensions must match and every normalization weight must be finite
and positive.
"""
function _validate_filter_inputs(W::AbstractSparseMatrix,
                                 w_sum::AbstractVector{<:Real},
                                 input::AbstractVector, transpose::Bool)
    expected_input = transpose ? size(W, 1) : size(W, 2)
    expected_sum = size(W, 1)
    length(input) == expected_input ||
        throw(DimensionMismatch(
            "input length $(length(input)) != $expected_input"))
    length(w_sum) == expected_sum ||
        throw(DimensionMismatch(
            "w_sum length $(length(w_sum)) != $expected_sum"))
    all(value -> isfinite(value) && value > 0, w_sum) ||
        throw(ArgumentError(
            "w_sum entries must all be finite and positive"))
    all(isfinite, input) ||
        throw(ArgumentError(
            "filter input entries must all be finite"))
    matrix_values = W isa SparseMatrixCSC ? nonzeros(W) : W
    all(isfinite, matrix_values) ||
        throw(ArgumentError(
            "filter weights must all be finite"))
    return nothing
end

function apply_filter(W::AbstractSparseMatrix,
                      w_sum::AbstractVector{<:Real},
                      rho::AbstractVector)
    _validate_filter_inputs(W, w_sum, rho, false)
    result = W * rho
    result ./= w_sum
    all(isfinite, result) ||
        throw(OverflowError(
            "density filter output is outside the representable range"))
    return result
end

"""
    apply_filter_transpose(W, w_sum, g_rho_tilde)

Apply the transpose of the filter for gradient backpropagation:
  g_ρ = Wᵀ (g_ρ̃ ./ w_sum)

This is the adjoint of apply_filter with respect to ρ.
The input dimensions must match and every normalization weight must be finite
and positive.
"""
function apply_filter_transpose(W::AbstractSparseMatrix,
                                w_sum::AbstractVector{<:Real},
                                g_rho_tilde::AbstractVector)
    _validate_filter_inputs(W, w_sum, g_rho_tilde, true)
    result = W' * (g_rho_tilde ./ w_sum)
    all(isfinite, result) ||
        throw(OverflowError(
            "transpose density filter output is outside the " *
            "representable range"))
    return result
end

function apply_filter_transpose(W::SparseMatrixCSC,
                                w_sum::AbstractVector{<:Real},
                                g_rho_tilde::AbstractVector)
    _validate_filter_inputs(W, w_sum, g_rho_tilde, true)
    scaled_type = Base.promote_op(
        /, eltype(g_rho_tilde), eltype(w_sum))
    result_type = Base.promote_op(*, eltype(W), scaled_type)
    result = zeros(result_type, size(W, 2))
    rows = rowvals(W)
    values = nonzeros(W)
    @inbounds for col in 1:size(W, 2)
        value = zero(result_type)
        for index in nzrange(W, col)
            row = rows[index]
            value += conj(values[index]) *
                     (g_rho_tilde[row] / w_sum[row])
        end
        result[col] = value
    end
    all(isfinite, result) ||
        throw(OverflowError(
            "transpose density filter output is outside the " *
            "representable range"))
    return result
end

function _projection_constants(beta::Real, eta::Real)
    isfinite(beta) && beta > 0 ||
        throw(ArgumentError(
            "beta must be finite and positive, got $beta"))
    isfinite(eta) && 0 <= eta <= 1 ||
        throw(ArgumentError(
            "eta must be finite and lie in [0, 1], got $eta"))
    offset = tanh(beta * eta)
    denominator = offset + tanh(beta * (1 - eta))
    if !(isfinite(denominator) && denominator > 0)
        return setprecision(BigFloat, 4352) do
            beta_big = BigFloat(beta)
            eta_big = BigFloat(eta)
            offset_big = tanh(beta_big * eta_big)
            denominator_big = offset_big +
                              tanh(beta_big * (1 - eta_big))
            offset_big, denominator_big
        end
    end
    return offset, denominator
end


@noinline function _heaviside_project_bigfloat(
        rho_tilde::AbstractVector, beta::Real, eta::Real)
    return setprecision(BigFloat, 4352) do
        beta_big = BigFloat(beta)
        eta_big = BigFloat(eta)
        offset = tanh(beta_big * eta_big)
        denominator = offset + tanh(beta_big * (1 - eta_big))
        [Float64((offset + tanh(beta_big *
                  (BigFloat(rt) - eta_big))) / denominator)
         for rt in rho_tilde]
    end
end

@noinline function _heaviside_derivative_bigfloat(
        rho_tilde::AbstractVector, beta::Real, eta::Real)
    return setprecision(BigFloat, 4352) do
        beta_big = BigFloat(beta)
        eta_big = BigFloat(eta)
        denominator = tanh(beta_big * eta_big) +
                      tanh(beta_big * (1 - eta_big))
        [Float64(beta_big *
                 (1 - tanh(beta_big * (BigFloat(rt) - eta_big))^2) /
                 denominator)
         for rt in rho_tilde]
    end
end

@inline function _projection_result_type(
    rho_tilde::AbstractVector,
    beta::Real,
    eta::Real,
)
    promoted = promote_type(eltype(rho_tilde), typeof(beta), typeof(eta))
    return promoted <: AbstractFloat ? promoted : Float64
end

@inline function _projection_convert_result(
    values::Vector{Float64},
    ::Type{Float64},
)
    return values
end

@inline function _projection_convert_result(
    values::Vector{Float64},
    ::Type{T},
) where {T<:AbstractFloat}
    return T.(values)
end

@inline function _projection_logcosh(value::Float64)
    magnitude = abs(value)
    return magnitude + log1p(exp(-2magnitude)) - log(2.0)
end

@inline function _projection_logsinh_positive(value::Float64)
    value > 0.0 || return -Inf
    if value < 20.0
        return log(sinh(value))
    end
    return value + log1p(-exp(-2value)) - log(2.0)
end

"""
    heaviside_project(rho_tilde, beta, eta=0.5)

Smooth Heaviside projection:
  ρ̄ = [tanh(β η) + tanh(β (ρ̃ - η))] / [tanh(β η) + tanh(β (1 - η))]

- β controls sharpness (β=1 nearly linear, β=64 nearly binary)
- η is the threshold (default 0.5)
- β must be finite and positive; η must be finite and lie in [0, 1]
"""
function heaviside_project(rho_tilde::AbstractVector, beta::Real, eta::Real=0.5)
    if beta isa BigFloat || eta isa BigFloat || eltype(rho_tilde) <: BigFloat
        isfinite(beta) && beta > 0 ||
            throw(ArgumentError(
                "beta must be finite and positive, got $beta"))
        isfinite(eta) && 0 <= eta <= 1 ||
            throw(ArgumentError(
                "eta must be finite and lie in [0, 1], got $eta"))
        T = promote_type(eltype(rho_tilde), typeof(beta), typeof(eta))
        beta_value = T(beta)
        eta_value = T(eta)
        offset = tanh(beta_value * eta_value)
        denominator = offset + tanh(beta_value * (one(T) - eta_value))
        return T[(offset + tanh(beta_value * (T(rt) - eta_value))) /
                 denominator for rt in rho_tilde]
    end
    result_type = _projection_result_type(rho_tilde, beta, eta)
    offset, denominator = _projection_constants(beta, eta)
    if offset isa BigFloat
        return _projection_convert_result(
            _heaviside_project_bigfloat(rho_tilde, beta, eta), result_type)
    end
    result = Vector{Float64}(undef, length(rho_tilde))
    beta_float = Float64(beta)
    eta_float = Float64(eta)
    beta_float > 0x1.0p52 &&
        return _projection_convert_result(
            _heaviside_project_bigfloat(rho_tilde, beta, eta), result_type)
    log_denominator_sinh = _projection_logsinh_positive(beta_float)
    log_cosh_right = _projection_logcosh(beta_float * (1 - eta_float))
    previous_rt = -Inf
    previous_value = -Inf
    @inbounds for (output_index, index) in enumerate(eachindex(rho_tilde))
        rt = Float64(rho_tilde[index])
        isfinite(rt) ||
            throw(ArgumentError(
                "rho_tilde entries must be finite, got $rt at index $index"))
        if iszero(rt)
            result[output_index] = copysign(0.0, rt)
            continue
        end
        beta_rho = beta_float * rt
        (isfinite(beta_rho) &&
         !(iszero(beta_rho) && !iszero(beta_float) && !iszero(rt))) ||
            return _projection_convert_result(
                _heaviside_project_bigfloat(rho_tilde, beta, eta), result_type)
        abs(beta_float * (rt - eta_float)) > 32.0 &&
            return _projection_convert_result(
                _heaviside_project_bigfloat(rho_tilde, beta, eta), result_type)
        log_magnitude = _projection_logsinh_positive(abs(beta_rho)) +
                        log_cosh_right - log_denominator_sinh -
                        _projection_logcosh(beta_float * (rt - eta_float))
        value = copysign(exp(log_magnitude), rt)
        isfinite(value) ||
            return _projection_convert_result(
                _heaviside_project_bigfloat(rho_tilde, beta, eta), result_type)
        # The analytical projection maps [0,1] into [0,1]. Log-domain
        # reconstruction can overshoot an endpoint by a few ulps after the
        # exact value has saturated there; clamp that rounding artifact.
        bounded = clamp(value, 0.0, 1.0)
        # Preserve the mathematical monotonicity for already sorted inputs
        # when saturated log-domain values differ only through rounding.
        if rt >= previous_rt && bounded < previous_value
            bounded = previous_value
        end
        result[output_index] = bounded
        previous_rt = rt
        previous_value = bounded
    end
    return _projection_convert_result(result, result_type)
end

"""
    heaviside_derivative(rho_tilde, beta, eta=0.5)

Derivative of the Heaviside projection:
  dρ̄/dρ̃ = β (1 - tanh²(β(ρ̃ - η))) / [tanh(βη) + tanh(β(1-η))]

Returns a vector of per-element derivatives.
"""
function heaviside_derivative(rho_tilde::AbstractVector, beta::Real, eta::Real=0.5)
    if beta isa BigFloat || eta isa BigFloat || eltype(rho_tilde) <: BigFloat
        isfinite(beta) && beta > 0 ||
            throw(ArgumentError(
                "beta must be finite and positive, got $beta"))
        isfinite(eta) && 0 <= eta <= 1 ||
            throw(ArgumentError(
                "eta must be finite and lie in [0, 1], got $eta"))
        T = promote_type(eltype(rho_tilde), typeof(beta), typeof(eta))
        beta_value = T(beta)
        eta_value = T(eta)
        denominator = tanh(beta_value * eta_value) +
                      tanh(beta_value * (one(T) - eta_value))
        return T[beta_value *
                 (one(T) - tanh(beta_value * (T(rt) - eta_value))^2) /
                 denominator for rt in rho_tilde]
    end
    result_type = _projection_result_type(rho_tilde, beta, eta)
    _, denominator = _projection_constants(beta, eta)
    if denominator isa BigFloat
        return _projection_convert_result(
            _heaviside_derivative_bigfloat(rho_tilde, beta, eta), result_type)
    end
    result = Vector{Float64}(undef, length(rho_tilde))
    beta_float = Float64(beta)
    eta_float = Float64(eta)
    beta_float > 0x1.0p52 &&
        return _projection_convert_result(
            _heaviside_derivative_bigfloat(rho_tilde, beta, eta), result_type)
    common_log = log(beta_float) +
                 _projection_logcosh(beta_float * eta_float) +
                 _projection_logcosh(beta_float * (1 - eta_float)) -
                 _projection_logsinh_positive(beta_float)
    @inbounds for (output_index, index) in enumerate(eachindex(rho_tilde))
        rt = Float64(rho_tilde[index])
        isfinite(rt) ||
            throw(ArgumentError(
                "rho_tilde entries must be finite, got $rt at index $index"))
        argument = beta_float * (rt - eta_float)
        isfinite(argument) ||
            return _projection_convert_result(
                _heaviside_derivative_bigfloat(rho_tilde, beta, eta), result_type)
        abs(argument) > 32.0 &&
            return _projection_convert_result(
                _heaviside_derivative_bigfloat(rho_tilde, beta, eta), result_type)
        value = exp(common_log - 2_projection_logcosh(argument))
        isfinite(value) ||
            return _projection_convert_result(
                _heaviside_derivative_bigfloat(rho_tilde, beta, eta), result_type)
        result[output_index] = value
    end
    return _projection_convert_result(result, result_type)
end

"""
    filter_and_project(W, w_sum, rho, beta, eta=0.5)

Full density pipeline: ρ → ρ̃ → ρ̄

Returns (rho_tilde, rho_bar).
"""
function filter_and_project(W::AbstractSparseMatrix, w_sum::AbstractVector,
                            rho::AbstractVector, beta::Real, eta::Real=0.5)
    rho_tilde = apply_filter(W, w_sum, rho)
    rho_bar = heaviside_project(rho_tilde, beta, eta)
    return rho_tilde, rho_bar
end

"""
    gradient_chain_rule(g_rho_bar, rho_tilde, W, w_sum, beta, eta=0.5)

Apply the full chain rule for density gradient:
  g_ρ̃ = g_ρ̄ .* dH/dρ̃          (Heaviside derivative)
  g_ρ  = Wᵀ (g_ρ̃ ./ w_sum)    (filter transpose)

Input:  g_rho_bar = ∂J/∂ρ̄ (gradient w.r.t. projected densities)
Output: g_rho     = ∂J/∂ρ  (gradient w.r.t. raw design variables)
"""
function gradient_chain_rule(g_rho_bar::AbstractVector, rho_tilde::AbstractVector,
                             W::AbstractSparseMatrix, w_sum::AbstractVector,
                             beta::Real, eta::Real=0.5)
    # Chain through Heaviside
    dH = heaviside_derivative(rho_tilde, beta, eta)
    g_rho_tilde = g_rho_bar .* dH

    # Chain through filter
    g_rho = apply_filter_transpose(W, w_sum, g_rho_tilde)

    return g_rho
end
