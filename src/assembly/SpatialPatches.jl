# SpatialPatches.jl — Automatic spatial patch assignment for impedance optimization
#
# Partitions mesh triangles into impedance design patches based on spatial
# location, enabling region-selective coating optimization.

export assign_patches_grid, assign_patches_by_region, assign_patches_uniform
export region_halfspace, region_sphere, region_box

"""
    assign_patches_grid(mesh; nx=4, ny=4, nz=1)

Partition mesh triangles into patches by dividing the bounding box into
an nx × ny × nz grid. Each occupied cell becomes a patch.

Returns a `PatchPartition` with consecutive patch IDs (empty cells skipped).
"""
function assign_patches_grid(mesh::TriMesh; nx::Int=4, ny::Int=4, nz::Int=1)
    Nt = ntriangles(mesh)
    nx >= 1 && ny >= 1 && nz >= 1 || error("Grid dimensions must be >= 1")

    # Compute triangle centroids
    centroids = [triangle_center(mesh, t) for t in 1:Nt]

    # Bounding box
    xs = [c[1] for c in centroids]
    ys = [c[2] for c in centroids]
    zs = [c[3] for c in centroids]
    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)
    zmin, zmax = minimum(zs), maximum(zs)

    # Add small padding to avoid edge cases
    eps_pad = 1e-12 * max(xmax - xmin, ymax - ymin, zmax - zmin, 1e-10)
    xmin -= eps_pad; xmax += eps_pad
    ymin -= eps_pad; ymax += eps_pad
    zmin -= eps_pad; zmax += eps_pad

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz

    # Assign each triangle to a grid cell
    raw_ids = zeros(Int, Nt)
    for t in 1:Nt
        c = centroids[t]
        ix = clamp(floor(Int, (c[1] - xmin) / dx), 0, nx - 1)
        iy = clamp(floor(Int, (c[2] - ymin) / dy), 0, ny - 1)
        iz = clamp(floor(Int, (c[3] - zmin) / dz), 0, nz - 1)
        raw_ids[t] = ix + iy * nx + iz * nx * ny + 1  # 1-based
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
    R >= 1 || error("At least one region predicate required")

    tri_patch = zeros(Int, Nt)
    for t in 1:Nt
        c = triangle_center(mesh, t)
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
          error("axis must be :x, :y, or :z")
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
    return c::Vec3 -> norm(c - center) <= radius
end

"""
    region_box(; lo, hi)

Create a predicate selecting triangles whose centroid is inside the
axis-aligned box [lo, hi].
"""
function region_box(; lo::Vec3, hi::Vec3)
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
    n_patches >= 1 || error("n_patches must be >= 1")
    n_patches <= Nt || error("n_patches ($n_patches) exceeds triangle count ($Nt)")

    centroids = [triangle_center(mesh, t) for t in 1:Nt]

    # Initialize cluster centers via uniform sampling
    rng = Random.MersenneTwister(42)  # deterministic
    indices = randperm(rng, Nt)[1:min(n_patches, Nt)]
    centers = [centroids[i] for i in indices]

    tri_patch = zeros(Int, Nt)
    max_kmeans_iter = 100

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
        for k in 1:n_patches
            members = [centroids[t] for t in 1:Nt if tri_patch[t] == k]
            if !isempty(members)
                centers[k] = sum(members) / length(members)
            end
        end
    end

    # Renumber to ensure consecutive IDs (in case some clusters are empty)
    used = sort(unique(tri_patch))
    if length(used) < n_patches
        id_map = Dict(old => new for (new, old) in enumerate(used))
        tri_patch = [id_map[tri_patch[t]] for t in 1:Nt]
        return PatchPartition(tri_patch, length(used))
    end

    return PatchPartition(tri_patch, n_patches)
end
