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

"""
    build_filter_weights(mesh, r_min)

Build the sparse conic filter weight matrix W and normalization vector w_sum.

For each triangle pair (t, s):
  w_ts = max(0, r_min - dist(centroid_t, centroid_s))

Returns (W::SparseMatrix, w_sum::Vector) where w_sum[t] = Σ_s W[t,s].

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
    Nt = ntriangles(mesh)

    # Compute triangle centroids
    centroids = Vector{Vec3}(undef, Nt)
    for t in 1:Nt
        v1 = _mesh_vertex(mesh, mesh.tri[1, t])
        v2 = _mesh_vertex(mesh, mesh.tri[2, t])
        v3 = _mesh_vertex(mesh, mesh.tri[3, t])
        centroids[t] = (v1 + v2 + v3) / 3
    end

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    # Degenerate guards: with r_min <= 0 no off-diagonal weight is positive and
    # even the self weight max(0, r_min - 0) = max(0, r_min) is zero, matching
    # the brute-force loop (which would push nothing). Also avoid building a
    # grid with a non-positive cell size.
    if Nt == 0 || r_min <= 0
        W = sparse(rows, cols, vals, Nt, Nt)
        w_sum = vec(sum(W, dims=2))
        return W, w_sum
    end

    # Spatial hash grid: cell size = r_min so neighbours within r_min lie in the
    # 3×3×3 stencil around a centroid's cell. Cell indices are integers derived
    # from a common origin (the minimum corner of the centroid bounding box).
    inv_h = 1.0 / r_min
    xmin = centroids[1][1]; ymin = centroids[1][2]; zmin = centroids[1][3]
    @inbounds for t in 2:Nt
        c = centroids[t]
        xmin = min(xmin, c[1]); ymin = min(ymin, c[2]); zmin = min(zmin, c[3])
    end
    origin = Vec3(xmin, ymin, zmin)

    cell_of(c) = (floor(Int, (c[1] - origin[1]) * inv_h),
                  floor(Int, (c[2] - origin[2]) * inv_h),
                  floor(Int, (c[3] - origin[3]) * inv_h))

    buckets = Dict{NTuple{3,Int},Vector{Int}}()
    cells = Vector{NTuple{3,Int}}(undef, Nt)
    @inbounds for t in 1:Nt
        key = cell_of(centroids[t])
        cells[t] = key
        push!(get!(() -> Int[], buckets, key), t)
    end

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

    W = sparse(rows, cols, vals, Nt, Nt)
    w_sum = vec(sum(W, dims=2))

    return W, w_sum
end

"""
    apply_filter(W, w_sum, rho)

Apply the conic density filter:
  ρ̃_t = Σ_s W[t,s] ρ_s / w_sum[t]
"""
function apply_filter(W::AbstractSparseMatrix, w_sum::AbstractVector, rho::AbstractVector)
    return (W * rho) ./ w_sum
end

"""
    apply_filter_transpose(W, w_sum, g_rho_tilde)

Apply the transpose of the filter for gradient backpropagation:
  g_ρ = Wᵀ (g_ρ̃ ./ w_sum)

This is the adjoint of apply_filter with respect to ρ.
"""
function apply_filter_transpose(W::AbstractSparseMatrix, w_sum::AbstractVector,
                                g_rho_tilde::AbstractVector)
    return W' * (g_rho_tilde ./ w_sum)
end

"""
    heaviside_project(rho_tilde, beta, eta=0.5)

Smooth Heaviside projection:
  ρ̄ = [tanh(β η) + tanh(β (ρ̃ - η))] / [tanh(β η) + tanh(β (1 - η))]

- β controls sharpness (β=1 nearly linear, β=64 nearly binary)
- η is the threshold (default 0.5)
"""
function heaviside_project(rho_tilde::AbstractVector, beta::Real, eta::Real=0.5)
    denom = tanh(beta * eta) + tanh(beta * (1 - eta))
    return [(tanh(beta * eta) + tanh(beta * (rt - eta))) / denom for rt in rho_tilde]
end

"""
    heaviside_derivative(rho_tilde, beta, eta=0.5)

Derivative of the Heaviside projection:
  dρ̄/dρ̃ = β (1 - tanh²(β(ρ̃ - η))) / [tanh(βη) + tanh(β(1-η))]

Returns a vector of per-element derivatives.
"""
function heaviside_derivative(rho_tilde::AbstractVector, beta::Real, eta::Real=0.5)
    denom = tanh(beta * eta) + tanh(beta * (1 - eta))
    return [beta * (1 - tanh(beta * (rt - eta))^2) / denom for rt in rho_tilde]
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
