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

function _filter_triplets_from_cells(
    centroids::Vector{Vec3},
    r_min::Float64,
    cells::Vector{NTuple{3,T}},
    buckets::Dict{NTuple{3,T},Vector{Int}},
) where {T<:Integer}
    Nt = length(centroids)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    sizehint!(rows, Nt)
    sizehint!(cols, Nt)
    sizehint!(vals, Nt)

    @inbounds for t in 1:Nt
        ct = centroids[t]
        (cx, cy, cz) = cells[t]
        for dz in -1:1, dy in -1:1, dx in -1:1
            neigh = get(buckets, (cx + dx, cy + dy, cz + dz), nothing)
            neigh === nothing && continue
            for s in neigh
                # Visit each unordered pair once (s > t) plus the diagonal once.
                s < t && continue
                # Use exactly the brute-force distance/weight expression so the
                # stored values are bit-identical: norm(SVector) == sqrt(Σδ²).
                d = norm(ct - centroids[s])
                w = max(0.0, r_min - d)
                w > 0 || continue
                if s == t
                    push!(rows, t); push!(cols, t); push!(vals, w)
                else
                    # Symmetric conic weight: store (t,s) and (s,t).
                    push!(rows, t); push!(cols, s); push!(vals, w)
                    push!(rows, s); push!(cols, t); push!(vals, w)
                end
            end
        end
    end
    return rows, cols, vals
end

function _filter_weight_triplets_int(
    centroids::Vector{Vec3},
    origin::Vec3,
    r_min::Float64,
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
        centroids, r_min, cells, buckets)
end

@noinline function _filter_weight_triplets_bigint(
    centroids::Vector{Vec3},
    origin::Vec3,
    r_min::Float64,
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
        centroids, r_min, cells, buckets)
end

"""
    build_filter_weights(mesh, r_min)

Build the sparse conic filter weight matrix W and normalization vector w_sum.

For each triangle pair (t, s):
  w_ts = max(0, r_min - dist(centroid_t, centroid_s))

Returns (W::SparseMatrix, w_sum::Vector) where w_sum[t] = Σ_s W[t,s].
`r_min` must be finite and positive.

A filter edge exists only when `dist < r_min`. Rather than testing all
`O(Nt^2)` centroid pairs, centroids are bucketed into a uniform spatial grid
with cell size `r_min`; any pair within `r_min` must then lie in the same or an
adjacent cell, so only the 3×3×3 neighbour stencil of each cell is searched
(~`O(Nt)` average for quasi-uniform meshes). The conic weight is symmetric
(`w_ts = w_st`), so each unordered pair is computed once and stored on both
sides, producing the identical sparsity pattern and values as the brute-force
double loop.
"""
function build_filter_weights(mesh::TriMesh, r_min::Float64)
    isfinite(r_min) && r_min > 0 ||
        throw(ArgumentError(
            "r_min must be finite and positive, got $r_min"))
    Nt = ntriangles(mesh)

    # Compute triangle centroids
    centroids = Vector{Vec3}(undef, Nt)
    for t in 1:Nt
        v1 = _mesh_vertex(mesh, mesh.tri[1, t])
        v2 = _mesh_vertex(mesh, mesh.tri[2, t])
        v3 = _mesh_vertex(mesh, mesh.tri[3, t])
        centroids[t] = (v1 + v2 + v3) / 3
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
        _filter_weight_triplets_int(centroids, origin, r_min)
    else
        _filter_weight_triplets_bigint(centroids, origin, r_min)
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
    return nothing
end

function apply_filter(W::AbstractSparseMatrix,
                      w_sum::AbstractVector{<:Real},
                      rho::AbstractVector)
    _validate_filter_inputs(W, w_sum, rho, false)
    result = W * rho
    result ./= w_sum
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
    return W' * (g_rho_tilde ./ w_sum)
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
    return offset, denominator
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
    offset, denominator = _projection_constants(beta, eta)
    return [(offset + tanh(beta * (rt - eta))) / denominator
            for rt in rho_tilde]
end

"""
    heaviside_derivative(rho_tilde, beta, eta=0.5)

Derivative of the Heaviside projection:
  dρ̄/dρ̃ = β (1 - tanh²(β(ρ̃ - η))) / [tanh(βη) + tanh(β(1-η))]

Returns a vector of per-element derivatives.
"""
function heaviside_derivative(rho_tilde::AbstractVector, beta::Real, eta::Real=0.5)
    _, denominator = _projection_constants(beta, eta)
    return [beta * (1 - tanh(beta * (rt - eta))^2) / denominator
            for rt in rho_tilde]
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
