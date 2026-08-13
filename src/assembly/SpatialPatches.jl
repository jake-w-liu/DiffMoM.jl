# SpatialPatches.jl — Automatic spatial patch assignment for impedance optimization
#
# Partitions mesh triangles into impedance design patches based on spatial
# location, enabling region-selective coating optimization.

export assign_patches_grid, assign_patches_by_region, assign_patches_uniform
export region_halfspace, region_sphere, region_box

const _DEFAULT_MAX_PATCH_DISTANCE_EVALUATIONS = 50_000_000
const _DEFAULT_MAX_PATCH_EXACT_COMPARISONS = 1_000
const _PATCH_FLOAT_EXPONENT_MIN = -1074
const _PATCH_FLOAT_EXPONENT_MAX = 971
const _PATCH_FLOAT_EXPONENT_COUNT =
    _PATCH_FLOAT_EXPONENT_MAX - _PATCH_FLOAT_EXPONENT_MIN + 1

@noinline function _patch_cluster_mean_exact(
        centroids::Vector{Vec3}, tri_patch::Vector{Int},
        cluster::Int, component::Int, count::Int,
        exponent_bins::Vector{Int128})
    fill!(exponent_bins, Int128(0))
    @inbounds for triangle in eachindex(centroids)
        tri_patch[triangle] == cluster || continue
        value = centroids[triangle][component]
        iszero(value) && continue
        significand, exponent, sign = Base.decompose(value)
        index = exponent - _PATCH_FLOAT_EXPONENT_MIN + 1
        exponent_bins[index] += Int128(sign) * Int128(significand)
    end
    first_nonzero = 0
    last_nonzero = 0
    @inbounds for index in eachindex(exponent_bins)
        if !iszero(exponent_bins[index])
            iszero(first_nonzero) && (first_nonzero = index)
            last_nonzero = index
        end
    end
    iszero(first_nonzero) && return 0.0

    exact_significand = BigInt(exponent_bins[last_nonzero])
    @inbounds for index in (last_nonzero - 1):-1:first_nonzero
        exact_significand <<= 1
        exact_significand += exponent_bins[index]
    end
    exact_exponent = _PATCH_FLOAT_EXPONENT_MIN + first_nonzero - 1
    return setprecision(BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        mean = Float64(ldexp(
            BigFloat(exact_significand) / BigFloat(count), exact_exponent))
        isfinite(mean) ||
            throw(OverflowError(
                "patch-center mean is outside the representable Float64 range"))
        return mean
    end
end

@noinline function _patch_grid_axis_index_big(
        value::Float64, lo::Float64, hi::Float64, ncells::Int)
    return setprecision(BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        lo_big = BigFloat(lo)
        hi_big = BigFloat(hi)
        span_big = hi_big - lo_big
        span_big > 0 || return 0
        scaled_big =
            ((BigFloat(value) - lo_big) / span_big) * BigFloat(ncells)
        scaled_big <= 0 && return 0
        scaled_big >= ncells && return ncells - 1
        return clamp(floor(Int, scaled_big), 0, ncells - 1)
    end
end

@inline function _patch_grid_axis_index_scaled_fast(
        value::Float64, lo::Float64, hi::Float64, ncells::Int)
    maximum_magnitude = max(abs(value), abs(lo), abs(hi))
    _, exponent = frexp(maximum_magnitude)
    value_scaled = ldexp(value, -exponent)
    lo_scaled = ldexp(lo, -exponent)
    hi_scaled = ldexp(hi, -exponent)
    span_scaled = hi_scaled - lo_scaled
    numerator_scaled = value_scaled - lo_scaled
    scaled = (numerator_scaled / span_scaled) * Float64(ncells)
    isfinite(scaled) || return -1

    nearest_boundary = round(scaled)
    boundary_uncertainty = 32 * eps(Float64) * max(1.0, abs(scaled))
    if value != 0.0 && value_scaled == 0.0
        return -1
    end
    if abs(scaled - nearest_boundary) <= boundary_uncertainty
        return -1
    end
    scaled >= Float64(ncells) && return ncells - 1
    return clamp(floor(Int, scaled), 0, ncells - 1)
end

@inline function _patch_scaled_sum_add!(
        scales::Matrix{Float64}, sums::Matrix{Float64},
        corrections::Matrix{Float64}, absolute_sums::Matrix{Float64},
        exceptional::BitMatrix, component::Int, cluster::Int,
        value::Float64)
    magnitude = abs(value)
    scale = scales[component, cluster]
    if magnitude > scale
        ratio = iszero(scale) ? 0.0 : scale / magnitude
        !iszero(scale) && iszero(ratio) &&
            (exceptional[component, cluster] = true)
        sums[component, cluster] *= ratio
        corrections[component, cluster] *= ratio
        absolute_sums[component, cluster] *= ratio
        scales[component, cluster] = magnitude
        scale = magnitude
    end
    iszero(magnitude) && return nothing

    normalized = value / scale
    iszero(normalized) &&
        (exceptional[component, cluster] = true)
    previous = sums[component, cluster]
    updated = previous + normalized
    correction = if abs(previous) >= abs(normalized)
        (previous - updated) + normalized
    else
        (normalized - updated) + previous
    end
    sums[component, cluster] = updated
    corrections[component, cluster] += correction
    absolute_sums[component, cluster] += abs(normalized)
    return nothing
end

function _update_patch_centers!(
        centers::Vector{Vec3}, centroids::Vector{Vec3},
        tri_patch::Vector{Int}, member_counts::Vector{Int},
        scales::Matrix{Float64}, sums::Matrix{Float64},
        corrections::Matrix{Float64}, absolute_sums::Matrix{Float64},
        exceptional::BitMatrix, exponent_bins::Vector{Int128})
    fill!(member_counts, 0)
    fill!(scales, 0.0)
    fill!(sums, 0.0)
    fill!(corrections, 0.0)
    fill!(absolute_sums, 0.0)
    fill!(exceptional, false)

    @inbounds for triangle in eachindex(centroids)
        cluster = tri_patch[triangle]
        member_counts[cluster] += 1
        for component in 1:3
            _patch_scaled_sum_add!(
                scales, sums, corrections, absolute_sums, exceptional,
                component, cluster, centroids[triangle][component])
        end
    end

    @inbounds for cluster in eachindex(centers)
        count = member_counts[cluster]
        iszero(count) && continue
        means = ntuple(component -> begin
            scale = scales[component, cluster]
            absolute_sum = absolute_sums[component, cluster]
            if iszero(scale) || iszero(absolute_sum)
                return 0.0
            end
            total = sums[component, cluster] +
                    corrections[component, cluster]
            ill_conditioned = exceptional[component, cluster] ||
                abs(total) <= 64 * eps(Float64) * absolute_sum
            normalized_mean = total / count
            mean = scales[component, cluster] * normalized_mean
            if ill_conditioned ||
               (total != 0.0 && normalized_mean == 0.0) ||
               (normalized_mean != 0.0 && mean == 0.0) ||
               !isfinite(mean)
                _patch_cluster_mean_exact(
                    centroids, tri_patch, cluster, component, count,
                    exponent_bins)
            else
                mean
            end
        end, 3)
        centers[cluster] = Vec3(means)
    end
    return nothing
end

@inline function _validated_patch_evaluation_limit(value::Integer)
    value isa Bool &&
        throw(ArgumentError(
            "max_distance_evaluations must be a positive integer, got $value"))
    1 <= value <= typemax(Int) ||
        throw(ArgumentError(
            "max_distance_evaluations must be in 1:$(typemax(Int)), got $value"))
    return Int(value)
end

@inline function _patch_distance(point::Vec3, center::Vec3)
    dx = point[1] - center[1]
    dy = point[2] - center[2]
    dz = point[3] - center[3]
    return hypot(hypot(dx, dy), dz)
end

@inline function _patch_exact_distance_precision(
        point::Vec3, candidate::Vec3, incumbent::Vec3)
    lowest_exponent = typemax(Int)
    highest_bit = typemin(Int)
    @inbounds for vector in (point, candidate, incumbent)
        for component in 1:3
            value = vector[component]
            iszero(value) && continue
            significand, exponent, _ = Base.decompose(value)
            lowest_exponent = min(lowest_exponent, exponent)
            significand_high_bit = 63 - leading_zeros(UInt64(significand))
            highest_bit = max(
                highest_bit, exponent + significand_high_bit)
        end
    end
    lowest_exponent == typemax(Int) && return 128

    # Relative to the least input bit, an endpoint difference needs at most
    # `highest_bit - lowest_exponent + 2` bits (the extra bit covers
    # opposite-sign subtraction). Squaring and summing three components needs
    # at most twice that width plus two carry bits. At full Float64 range this
    # is 4,200 bits; ordinary same-scale geometry typically needs only 128.
    difference_bits = highest_bit - lowest_exponent + 2
    return max(128, 2 * difference_bits + 2)
end

@noinline function _patch_distance_less_big(
        point::Vec3, candidate::Vec3, incumbent::Vec3)
    precision_bits = _patch_exact_distance_precision(
        point, candidate, incumbent)
    return setprecision(BigFloat, precision_bits) do
        candidate_squared = BigFloat(0)
        incumbent_squared = BigFloat(0)
        @inbounds for component in 1:3
            candidate_delta =
                BigFloat(point[component]) - BigFloat(candidate[component])
            incumbent_delta =
                BigFloat(point[component]) - BigFloat(incumbent[component])
            candidate_squared += candidate_delta * candidate_delta
            incumbent_squared += incumbent_delta * incumbent_delta
        end
        candidate_squared < incumbent_squared
    end
end

@inline function _patch_distance_squared_scaled(
        point::Vec3, center::Vec3, exponent::Int)
    dx = ldexp(point[1], -exponent) - ldexp(center[1], -exponent)
    dy = ldexp(point[2], -exponent) - ldexp(center[2], -exponent)
    dz = ldexp(point[3], -exponent) - ldexp(center[3], -exponent)
    return muladd(dx, dx, muladd(dy, dy, dz * dz))
end

@inline function _patch_distance_tie_is_nearer(
        point::Vec3, candidate::Vec3, incumbent::Vec3,
        exact_comparisons::Base.RefValue{Int}, exact_comparison_limit::Int)
    _patch_centroid_key(candidate) == _patch_centroid_key(incumbent) &&
        return false
    reflected = ntuple(component -> begin
        candidate_delta = candidate[component] - point[component]
        incumbent_delta = incumbent[component] - point[component]
        isfinite(candidate_delta) && isfinite(incumbent_delta) &&
        _two_difference_error(
            candidate[component], point[component], candidate_delta) == 0.0 &&
        _two_difference_error(
            incumbent[component], point[component], incumbent_delta) == 0.0 &&
        candidate_delta == -incumbent_delta
    end, 3)
    all(reflected) && return false
    maximum_magnitude = max(
        maximum(abs, point), maximum(abs, candidate),
        maximum(abs, incumbent))
    iszero(maximum_magnitude) && return false
    _, exponent = frexp(maximum_magnitude)
    candidate_squared = _patch_distance_squared_scaled(
        point, candidate, exponent)
    incumbent_squared = _patch_distance_squared_scaled(
        point, incumbent, exponent)
    difference = candidate_squared - incumbent_squared
    uncertainty = 64 * eps(Float64) * max(
        candidate_squared, incumbent_squared, floatmin(Float64))
    abs(difference) > uncertainty && return difference < 0.0
    exact_comparisons[] < exact_comparison_limit ||
        throw(ArgumentError(
            "assign_patches_uniform exceeded the bounded exact-distance " *
            "comparison budget ($exact_comparison_limit); the centroid " *
            "geometry contains too many numerically indistinguishable " *
            "center distances"))
    exact_comparisons[] += 1
    return _patch_distance_less_big(point, candidate, incumbent)
end

@inline function _patch_candidate_is_nearer(
        point::Vec3, candidate::Vec3, incumbent::Vec3,
        candidate_distance::Float64, incumbent_distance::Float64,
        exact_comparisons::Base.RefValue{Int}=Ref(0),
        exact_comparison_limit::Int=_DEFAULT_MAX_PATCH_EXACT_COMPARISONS)
    if isfinite(candidate_distance)
        !isfinite(incumbent_distance) && return true
        distance_gap = abs(candidate_distance - incumbent_distance)
        distance_uncertainty = 64 * eps(Float64) * max(
            candidate_distance, incumbent_distance, floatmin(Float64))
        if distance_gap > distance_uncertainty
            return candidate_distance < incumbent_distance
        end
        return _patch_distance_tie_is_nearer(
            point, candidate, incumbent,
            exact_comparisons, exact_comparison_limit)
    end
    isfinite(incumbent_distance) && return false
    return _patch_distance_tie_is_nearer(
        point, candidate, incumbent,
        exact_comparisons, exact_comparison_limit)
end

@inline function _patch_centroid_key(centroid::Vec3)
    return ntuple(component -> begin
        value = centroid[component]
        reinterpret(UInt64, iszero(value) ? 0.0 : value)
    end, 3)
end

function _uniform_patch_per_distinct_centroid(
        centroids::Vector{Vec3}, indices::Vector{Int})
    center_for_centroid = Dict{NTuple{3,UInt64},Int}()
    @inbounds for center in eachindex(indices)
        key = _patch_centroid_key(centroids[indices[center]])
        haskey(center_for_centroid, key) ||
            (center_for_centroid[key] = center)
    end
    tri_patch = Vector{Int}(undef, length(centroids))
    @inbounds for triangle in eachindex(centroids)
        tri_patch[triangle] = center_for_centroid[
            _patch_centroid_key(centroids[triangle])]
    end
    used = sort!(collect(values(center_for_centroid)))
    id_map = Dict(old => new for (new, old) in enumerate(used))
    @inbounds for triangle in eachindex(tri_patch)
        tri_patch[triangle] = id_map[tri_patch[triangle]]
    end
    return PatchPartition(tri_patch, length(used))
end

@inline function _patch_grid_axis_index_fast(value::Float64, lo::Float64,
                                             hi::Float64, ncells::Int)
    span = hi - lo
    span >= 0.0 ||
        throw(ArgumentError("mesh centroid bounds are not ordered: [$lo, $hi]"))
    iszero(span) && return 0
    value <= lo && return 0
    value >= hi && return ncells - 1
    value == 0.0 && lo == -hi && return ncells ÷ 2

    if !isfinite(span)
        return _patch_grid_axis_index_scaled_fast(value, lo, hi, ncells)
    end

    scaled = ((value - lo) / span) * Float64(ncells)
    isfinite(scaled) ||
        return _patch_grid_axis_index_scaled_fast(value, lo, hi, ncells)
    nearest_boundary = round(scaled)
    boundary_uncertainty = 32 * eps(Float64) * max(1.0, abs(scaled))
    if abs(scaled - nearest_boundary) <= boundary_uncertainty
        exact_midpoint = value == 0.0 && lo == -hi
        exact_midpoint || return -1
    end
    scaled >= Float64(ncells) && return ncells - 1
    return clamp(floor(Int, scaled), 0, ncells - 1)
end


@inline function _patch_grid_axis_index(value::Float64, lo::Float64,
                                        hi::Float64, ncells::Int)
    fast_index = _patch_grid_axis_index_fast(value, lo, hi, ncells)
    return fast_index >= 0 ? fast_index :
           _patch_grid_axis_index_big(value, lo, hi, ncells)
end

@inline function _patch_grid_axis_index_cached(
        value::Float64, lo::Float64, hi::Float64, ncells::Int,
        cache::Dict{UInt64,Int})
    fast_index = _patch_grid_axis_index_fast(value, lo, hi, ncells)
    fast_index >= 0 && return fast_index
    key = reinterpret(UInt64, iszero(value) ? 0.0 : value)
    haskey(cache, key) && return cache[key]
    length(cache) < _DEFAULT_MAX_PATCH_EXACT_COMPARISONS ||
        throw(ArgumentError(
            "assign_patches_grid exceeded the bounded exact-boundary " *
            "cache ($_DEFAULT_MAX_PATCH_EXACT_COMPARISONS distinct " *
            "coordinates on one axis)"))
    index = _patch_grid_axis_index_big(value, lo, hi, ncells)
    cache[key] = index
    return index
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
    max(nx, ny, nz) <= 10_000_000 ||
        throw(ArgumentError(
            "grid dimensions above 10,000,000 cells per axis are not " *
            "supported by the bounded Float64 classifier"))
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
    x_index_cache = Dict{UInt64,Int}()
    y_index_cache = Dict{UInt64,Int}()
    z_index_cache = Dict{UInt64,Int}()
    @inbounds for t in 1:Nt
        c = centroids[t]
        ix = _patch_grid_axis_index_cached(
            c[1], xmin, xmax, nx, x_index_cache)
        iy = _patch_grid_axis_index_cached(
            c[2], ymin, ymax, ny, y_index_cache)
        iz = _patch_grid_axis_index_cached(
            c[3], zmin, zmax, nz, z_index_cache)
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
    exact_cache = Dict{NTuple{3,UInt64},Bool}()
    cache_lock = ReentrantLock()
    return c::Vec3 -> _patch_point_within_sphere(
        c, center, radius, exact_cache, cache_lock)
end

@noinline function _patch_point_within_sphere_big(
        point::Vec3, center::Vec3, radius::Float64)
    return setprecision(BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        squared = BigFloat(0)
        @inbounds for component in 1:3
            delta = BigFloat(point[component]) - BigFloat(center[component])
            squared += delta * delta
        end
        squared <= BigFloat(radius) * BigFloat(radius)
    end
end

@inline function _patch_point_within_sphere(
        point::Vec3, center::Vec3, radius::Float64,
        exact_cache::Union{Nothing,Dict{NTuple{3,UInt64},Bool}}=nothing,
        cache_lock::Union{Nothing,ReentrantLock}=nothing)
    distance = _patch_distance(point, center)
    isfinite(distance) || return false
    margin = 32 * eps(Float64) * max(distance, radius, floatmin(Float64))
    distance < radius - margin && return true
    distance > radius + margin && return false
    exact_cache === nothing &&
        return _patch_point_within_sphere_big(point, center, radius)
    key = _patch_centroid_key(point)
    lock(cache_lock::ReentrantLock)
    try
        haskey(exact_cache, key) && return exact_cache[key]
        length(exact_cache) < _DEFAULT_MAX_PATCH_EXACT_COMPARISONS ||
            throw(ArgumentError(
                "region_sphere exceeded the bounded exact-boundary cache " *
                "($_DEFAULT_MAX_PATCH_EXACT_COMPARISONS distinct points)"))
        result = _patch_point_within_sphere_big(point, center, radius)
        exact_cache[key] = result
        return result
    finally
        unlock(cache_lock)
    end
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
    assign_patches_uniform(mesh; n_patches,
                           max_distance_evaluations=50_000_000)

Partition all triangles into `n_patches` spatial groups using k-means
clustering of triangle centroids.

`max_distance_evaluations` is a positive total work cap. The function fails
closed if Lloyd's algorithm cannot converge within the cap.

Returns a `PatchPartition`.
"""
function assign_patches_uniform(
        mesh::TriMesh; n_patches::Int,
        max_distance_evaluations::Integer=
            _DEFAULT_MAX_PATCH_DISTANCE_EVALUATIONS)
    Nt = ntriangles(mesh)
    Nt >= 1 ||
        throw(ArgumentError(
            "assign_patches_uniform requires a non-empty mesh (got 0 triangles)"))
    n_patches >= 1 || throw(ArgumentError("n_patches must be at least 1"))
    n_patches <= Nt ||
        throw(ArgumentError(
            "n_patches ($n_patches) exceeds triangle count ($Nt)"))
    evaluation_limit = _validated_patch_evaluation_limit(
        max_distance_evaluations)

    evaluations_per_iteration = if n_patches == Nt || n_patches == 1
        0
    else
        try
            Base.Checked.checked_mul(Nt, n_patches)
        catch err
            err isa OverflowError || rethrow()
            throw(ArgumentError(
                "Nt*n_patches overflows the supported Int work count"))
        end
    end
    evaluations_per_iteration <= evaluation_limit ||
        throw(ArgumentError(
            "assign_patches_uniform requires $evaluations_per_iteration " *
            "distance evaluations per Lloyd iteration, exceeding " *
            "max_distance_evaluations=$evaluation_limit"))

    centroids = [triangle_center(mesh, t) for t in 1:Nt]
    for (t, centroid) in enumerate(centroids)
        all(isfinite, centroid) ||
            throw(ArgumentError(
                "triangle $t has a non-finite centroid $centroid"))
    end
    n_patches == 1 && return PatchPartition(fill(1, Nt), 1)

    # Initialize cluster centers via uniform sampling
    rng = Random.MersenneTwister(42)  # deterministic
    indices = randperm(rng, Nt)[1:min(n_patches, Nt)]
    n_patches == Nt &&
        return _uniform_patch_per_distinct_centroid(centroids, indices)
    centers = [centroids[i] for i in indices]

    tri_patch = zeros(Int, Nt)
    max_kmeans_iter = 100
    member_counts = zeros(Int, n_patches)
    scales = zeros(Float64, 3, n_patches)
    sums = zeros(Float64, 3, n_patches)
    corrections = zeros(Float64, 3, n_patches)
    absolute_sums = zeros(Float64, 3, n_patches)
    exceptional = falses(3, n_patches)
    exponent_bins = zeros(Int128, _PATCH_FLOAT_EXPONENT_COUNT)
    evaluations_used = 0
    exact_comparisons = Ref(0)
    assignment_cache = Dict{NTuple{3,UInt64},Int}()

    converged = false
    for iteration in 1:max_kmeans_iter
        evaluations_used <= evaluation_limit - evaluations_per_iteration ||
            throw(ArgumentError(
                "assign_patches_uniform did not converge within " *
                "max_distance_evaluations=$evaluation_limit; increase the " *
                "work limit or reduce n_patches"))
        evaluations_used += evaluations_per_iteration
        # Assign each triangle to nearest center
        changed = false
        empty!(assignment_cache)
        for t in 1:Nt
            centroid_key = _patch_centroid_key(centroids[t])
            best_k = get(assignment_cache, centroid_key, 0)
            if iszero(best_k)
                best_k = 1
                best_dist = _patch_distance(centroids[t], centers[1])
                for k in 2:n_patches
                    d = _patch_distance(centroids[t], centers[k])
                    if _patch_candidate_is_nearer(
                            centroids[t], centers[k], centers[best_k],
                            d, best_dist, exact_comparisons,
                            _DEFAULT_MAX_PATCH_EXACT_COMPARISONS)
                        best_dist = d
                        best_k = k
                    end
                end
                assignment_cache[centroid_key] = best_k
            end
            if tri_patch[t] != best_k
                tri_patch[t] = best_k
                changed = true
            end
        end

        if !changed
            converged = true
            break
        end

        # Update centers
        _update_patch_centers!(
            centers, centroids, tri_patch, member_counts,
            scales, sums, corrections, absolute_sums, exceptional,
            exponent_bins)
    end
    converged ||
        throw(ArgumentError(
            "assign_patches_uniform did not converge within " *
            "$max_kmeans_iter Lloyd iterations"))

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
