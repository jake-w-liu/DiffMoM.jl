# SpatialPatches.jl — Automatic spatial patch assignment for impedance optimization
#
# Partitions mesh triangles into impedance design patches based on spatial
# location, enabling region-selective coating optimization.

export assign_patches_grid, assign_patches_by_region, assign_patches_uniform
export region_halfspace, region_sphere, region_box

@inline function _patch_grid_axis_index(value::Float64, lo::Float64,
                                        hi::Float64, ncells::Int)
    span = hi - lo
    isfinite(span) && span >= 0.0 ||
        throw(ArgumentError(
            "mesh centroid span is outside the supported Float64 range: [$lo, $hi]"
        ))
    iszero(span) && return 0
    value <= lo && return 0
    value >= hi && return ncells - 1

    scaled = ((value - lo) / span) * Float64(ncells)
    isfinite(scaled) ||
        throw(ArgumentError(
            "mesh centroid $value cannot be assigned within bounds [$lo, $hi]"
        ))
    scaled >= Float64(ncells) && return ncells - 1
    return clamp(floor(Int, scaled), 0, ncells - 1)
end

@inline function _patch_grid_strides(nx::Int, ny::Int, nz::Int)
    nx <= typemax(Int) ÷ ny ||
        throw(ArgumentError("nx * ny exceeds the supported Int range"))
    nxy = nx * ny
    nxy <= typemax(Int) ÷ nz ||
        throw(ArgumentError("nx * ny * nz exceeds the supported Int range"))
    return nxy
end

"""
    assign_patches_grid(mesh; nx=4, ny=4, nz=1)

Partition mesh triangles into patches by dividing the bounding box into
an nx × ny × nz grid. Each occupied cell becomes a patch.

Returns a `PatchPartition` with consecutive patch IDs (empty cells skipped).
"""
function assign_patches_grid(mesh::TriMesh; nx::Int=4, ny::Int=4, nz::Int=1)
    Nt = ntriangles(mesh)
    Nt >= 1 ||
        throw(ArgumentError(
            "assign_patches_grid requires a non-empty mesh (got 0 triangles)"))
    nx >= 1 && ny >= 1 && nz >= 1 ||
        throw(ArgumentError("grid dimensions must be at least 1"))
    nxy = _patch_grid_strides(nx, ny, nz)

    # Compute triangle centroids
    centroids = [triangle_center(mesh, t) for t in 1:Nt]
    for (t, centroid) in enumerate(centroids)
        all(isfinite, centroid) ||
            throw(ArgumentError(
                "triangle $t has a non-finite centroid $centroid"))
    end

    # Bounding box
    xmin = xmax = centroids[1][1]
    ymin = ymax = centroids[1][2]
    zmin = zmax = centroids[1][3]
    @inbounds for t in 2:Nt
        c = centroids[t]
        xmin = min(xmin, c[1]); xmax = max(xmax, c[1])
        ymin = min(ymin, c[2]); ymax = max(ymax, c[2])
        zmin = min(zmin, c[3]); zmax = max(zmax, c[3])
    end

    # Assign each triangle to a grid cell
    raw_ids = zeros(Int, Nt)
    @inbounds for t in 1:Nt
        c = centroids[t]
        ix = _patch_grid_axis_index(c[1], xmin, xmax, nx)
        iy = _patch_grid_axis_index(c[2], ymin, ymax, ny)
        iz = _patch_grid_axis_index(c[3], zmin, zmax, nz)
        raw_ids[t] = ix + iy * nx + iz * nxy + 1  # 1-based
    end

    # Renumber to consecutive IDs (skip empty cells)
    unique_ids = sort(unique(raw_ids))
    id_map = Dict(old => new for (new, old) in enumerate(unique_ids))
    tri_patch = [id_map[raw_ids[t]] for t in 1:Nt]
    P = length(unique_ids)

    return PatchPartition(tri_patch, P)
end

"""
    assign_patches_by_region(mesh, regions)

Assign triangles to patches based on spatial predicate functions.

Each element of `regions` is a function `f(centroid::Vec3) -> Bool`.
Triangle t is assigned to the first region whose predicate returns `true`.
Unmatched triangles are collected into an extra "background" patch (last patch).

Returns a `PatchPartition`.
"""
function assign_patches_by_region(mesh::TriMesh, regions::Vector{<:Function})
    Nt = ntriangles(mesh)
    R = length(regions)
    Nt >= 1 ||
        throw(ArgumentError(
            "assign_patches_by_region requires a non-empty mesh (got 0 triangles)"))
    R >= 1 || throw(ArgumentError("at least one region predicate is required"))

    tri_patch = zeros(Int, Nt)
    for t in 1:Nt
        c = triangle_center(mesh, t)
        all(isfinite, c) ||
            throw(ArgumentError("triangle $t has a non-finite centroid $c"))
        assigned = false
        for r in 1:R
            if regions[r](c)
                tri_patch[t] = r
                assigned = true
                break
            end
        end
        if !assigned
            tri_patch[t] = R + 1  # background patch
        end
    end

    P = maximum(tri_patch)
    return PatchPartition(tri_patch, P)
end

"""
    region_halfspace(; axis, threshold, above=true)

Create a predicate selecting triangles whose centroid satisfies
`centroid[axis] >= threshold` (if `above=true`) or `centroid[axis] < threshold`.

`axis` must be `:x`, `:y`, or `:z`.
"""
function region_halfspace(; axis::Symbol, threshold::Float64, above::Bool=true)
    idx = axis == :x ? 1 : axis == :y ? 2 : axis == :z ? 3 :
          throw(ArgumentError("axis must be :x, :y, or :z; got $axis"))
    isfinite(threshold) ||
        throw(ArgumentError("threshold must be finite, got $threshold"))
    if above
        return c::Vec3 -> c[idx] >= threshold
    else
        return c::Vec3 -> c[idx] < threshold
    end
end

"""
    region_sphere(; center, radius)

Create a predicate selecting triangles whose centroid is within `radius`
of `center`.
"""
function region_sphere(; center::Vec3, radius::Float64)
    all(isfinite, center) ||
        throw(ArgumentError("sphere center must be finite, got $center"))
    isfinite(radius) && radius >= 0.0 ||
        throw(ArgumentError(
            "sphere radius must be finite and nonnegative, got $radius"))
    return c::Vec3 -> norm(c - center) <= radius
end

"""
    region_box(; lo, hi)

Create a predicate selecting triangles whose centroid is inside the
axis-aligned box [lo, hi].
"""
function region_box(; lo::Vec3, hi::Vec3)
    all(isfinite, lo) && all(isfinite, hi) ||
        throw(ArgumentError("box bounds must be finite, got lo=$lo and hi=$hi"))
    all(lo .<= hi) ||
        throw(ArgumentError(
            "box lower bounds must not exceed upper bounds, got lo=$lo and hi=$hi"
        ))
    return c::Vec3 -> (c[1] >= lo[1] && c[1] <= hi[1] &&
                        c[2] >= lo[2] && c[2] <= hi[2] &&
                        c[3] >= lo[3] && c[3] <= hi[3])
end

"""
    assign_patches_uniform(mesh; n_patches)

Partition all triangles into `n_patches` spatial groups using k-means
clustering of triangle centroids.

Returns a `PatchPartition`.
"""
function assign_patches_uniform(mesh::TriMesh; n_patches::Int)
    Nt = ntriangles(mesh)
    Nt >= 1 ||
        throw(ArgumentError(
            "assign_patches_uniform requires a non-empty mesh (got 0 triangles)"))
    n_patches >= 1 || throw(ArgumentError("n_patches must be at least 1"))
    n_patches <= Nt ||
        throw(ArgumentError(
            "n_patches ($n_patches) exceeds triangle count ($Nt)"))

    centroids = [triangle_center(mesh, t) for t in 1:Nt]
    for (t, centroid) in enumerate(centroids)
        all(isfinite, centroid) ||
            throw(ArgumentError(
                "triangle $t has a non-finite centroid $centroid"))
    end

    # Initialize cluster centers via uniform sampling
    rng = Random.MersenneTwister(42)  # deterministic
    indices = randperm(rng, Nt)[1:min(n_patches, Nt)]
    centers = [centroids[i] for i in indices]

    tri_patch = zeros(Int, Nt)
    max_kmeans_iter = 100
    center_sums = zeros(Float64, 3, n_patches)
    member_counts = zeros(Int, n_patches)

    for _ in 1:max_kmeans_iter
        # Assign each triangle to nearest center
        changed = false
        for t in 1:Nt
            best_k = 1
            best_dist = Inf
            for k in 1:n_patches
                d = norm(centroids[t] - centers[k])
                if d < best_dist
                    best_dist = d
                    best_k = k
                end
            end
            if tri_patch[t] != best_k
                tri_patch[t] = best_k
                changed = true
            end
        end

        !changed && break

        # Update centers
        fill!(center_sums, 0.0)
        fill!(member_counts, 0)
        @inbounds for t in 1:Nt
            k = tri_patch[t]
            center_sums[1, k] += centroids[t][1]
            center_sums[2, k] += centroids[t][2]
            center_sums[3, k] += centroids[t][3]
            member_counts[k] += 1
        end
        for k in 1:n_patches
            if !iszero(member_counts[k])
                centers[k] = Vec3(
                    center_sums[1, k] / member_counts[k],
                    center_sums[2, k] / member_counts[k],
                    center_sums[3, k] / member_counts[k],
                )
            end
        end
    end

    # Renumber to ensure consecutive IDs (in case some clusters are empty)
    used = sort(unique(tri_patch))
    if length(used) < n_patches
        id_map = Dict(old => new for (new, old) in enumerate(used))
        @inbounds for t in 1:Nt
            tri_patch[t] = id_map[tri_patch[t]]
        end
        return PatchPartition(tri_patch, length(used))
    end

    return PatchPartition(tri_patch, n_patches)
end
