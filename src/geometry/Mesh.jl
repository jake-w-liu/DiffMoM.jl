# Mesh.jl — Simple mesh generation and geometry utilities

export make_rect_plate, make_rect_plate_graded, make_circular_plate, make_parabolic_reflector, read_obj_mesh, triangle_area, triangle_center, triangle_normal
export mesh_quality_report, mesh_quality_ok, assert_mesh_quality
export write_obj_mesh, repair_mesh_for_simulation, repair_obj_mesh
export estimate_dense_matrix_gib, cluster_mesh_vertices, drop_nonmanifold_triangles
export coarsen_mesh_to_target_rwg
export mesh_unique_edges, mesh_wireframe_segments
export mesh_resolution_report, mesh_resolution_ok
export refine_mesh_to_target_edge, refine_mesh_for_mom

"""
    make_rect_plate(Lx, Ly, Nx, Ny)

Generate a triangulated rectangular plate in the xy-plane, centered at the
origin. Returns a `TriMesh` with `(Nx+1)*(Ny+1)` vertices and `2*Nx*Ny`
triangles.
"""
function make_rect_plate(Lx::Real, Ly::Real, Nx::Int, Ny::Int)
    Nv = (Nx + 1) * (Ny + 1)
    Nt = 2 * Nx * Ny

    xyz = zeros(3, Nv)
    tri = zeros(Int, 3, Nt)

    # Vertex grid
    dx = Lx / Nx
    dy = Ly / Ny
    idx = 0
    for jy in 0:Ny
        for jx in 0:Nx
            idx += 1
            xyz[1, idx] = -Lx / 2 + jx * dx
            xyz[2, idx] = -Ly / 2 + jy * dy
            xyz[3, idx] = 0.0
        end
    end

    # Linear index helper: (ix, iy) -> vertex id (0-based ix, iy)
    vidx(ix, iy) = iy * (Nx + 1) + ix + 1

    # Triangulation: two triangles per grid cell
    tidx = 0
    for jy in 0:Ny-1
        for jx in 0:Nx-1
            v1 = vidx(jx,   jy)
            v2 = vidx(jx+1, jy)
            v3 = vidx(jx+1, jy+1)
            v4 = vidx(jx,   jy+1)

            tidx += 1
            tri[1, tidx] = v1
            tri[2, tidx] = v2
            tri[3, tidx] = v3

            tidx += 1
            tri[1, tidx] = v1
            tri[2, tidx] = v3
            tri[3, tidx] = v4
        end
    end

    return TriMesh(xyz, tri)
end

"""
    make_circular_plate(radius, Nr, Nphi)

Generate a triangulated circular plate (disk) in the xy-plane, centered at
the origin. Uses radial rings with azimuthal subdivision.

Returns a `TriMesh` with approximately `Nr*Nphi + 1` vertices.
"""
function make_circular_plate(radius::Real, Nr::Int, Nphi::Int)
    # Vertices: center + Nr rings × Nphi points each
    Nv = 1 + Nr * Nphi
    verts = zeros(3, Nv)

    # Center vertex
    verts[:, 1] = [0.0, 0.0, 0.0]

    # Ring vertices
    idx = 1
    for ir in 1:Nr
        r = radius * ir / Nr
        for ip in 1:Nphi
            phi = 2π * (ip - 1) / Nphi
            idx += 1
            verts[1, idx] = r * cos(phi)
            verts[2, idx] = r * sin(phi)
        end
    end

    # Triangles
    tris = Int[]

    # Inner ring: triangles from center to first ring
    for ip in 1:Nphi
        v1 = 1   # center
        v2 = 1 + ip
        v3 = 1 + mod(ip, Nphi) + 1  # wraps: ip=Nphi -> next is 1+1=2
        # Fix: the next vertex in the ring
        v3 = ip < Nphi ? 1 + ip + 1 : 1 + 1
        push!(tris, v1, v2, v3)
    end

    # Outer rings: quads split into two triangles
    for ir in 1:Nr-1
        off_inner = 1 + (ir - 1) * Nphi
        off_outer = 1 + ir * Nphi
        for ip in 1:Nphi
            ip_next = ip < Nphi ? ip + 1 : 1
            v1 = off_inner + ip
            v2 = off_outer + ip
            v3 = off_outer + ip_next
            v4 = off_inner + ip_next
            push!(tris, v1, v2, v3)
            push!(tris, v1, v3, v4)
        end
    end

    Nt = length(tris) ÷ 3
    tri = reshape(tris, 3, Nt)

    return TriMesh(verts, tri)
end

"""
    _grade_1d(N, L, grading_factor)

Map `N+1` uniform grid indices to graded physical coordinates on `[-L/2, L/2]`.
Uses tanh-based grading that clusters points near the edges.

`grading_factor > 0`: larger values → more clustering near edges.
When `grading_factor → 0`, the mapping degenerates; use `grading_factor ≥ 0.1`.
"""
function _grade_1d(N::Int, L::Real, grading_factor::Real)
    coords = Vector{Float64}(undef, N + 1)
    inv_tanh_g = 1.0 / tanh(grading_factor)
    half_L = L / 2
    for j in 0:N
        u = j / N                           # uniform parameter [0, 1]
        s = 2u - 1                          # map to [-1, 1]
        g = tanh(grading_factor * s) * inv_tanh_g  # graded [-1, 1]
        coords[j + 1] = half_L * g          # physical coordinate
    end
    return coords
end

"""
    make_rect_plate_graded(Lx, Ly, Nx, Ny; grading_factor=3.0)

Generate a triangulated rectangular plate in the xy-plane with graded mesh
density near the edges.  Same topology as `make_rect_plate` but vertex
positions are redistributed using a tanh grading function.

`grading_factor` controls edge clustering:
- `1.0`: nearly uniform
- `3.0` (default): ~5:1 edge-to-center density ratio
- `5.0`: ~10:1 ratio

Practical range: 1.0–5.0.  Values above 6 may create highly skewed
center elements.
"""
function make_rect_plate_graded(Lx::Real, Ly::Real, Nx::Int, Ny::Int;
                                 grading_factor::Real=3.0)
    grading_factor > 0 || error("grading_factor must be positive, got $grading_factor")

    xs = _grade_1d(Nx, Lx, grading_factor)
    ys = _grade_1d(Ny, Ly, grading_factor)

    Nv = (Nx + 1) * (Ny + 1)
    Nt = 2 * Nx * Ny

    xyz = zeros(3, Nv)
    tri = zeros(Int, 3, Nt)

    idx = 0
    for jy in 0:Ny
        for jx in 0:Nx
            idx += 1
            xyz[1, idx] = xs[jx + 1]
            xyz[2, idx] = ys[jy + 1]
            xyz[3, idx] = 0.0
        end
    end

    vidx(ix, iy) = iy * (Nx + 1) + ix + 1

    tidx = 0
    for jy in 0:Ny-1
        for jx in 0:Nx-1
            v1 = vidx(jx,   jy)
            v2 = vidx(jx+1, jy)
            v3 = vidx(jx+1, jy+1)
            v4 = vidx(jx,   jy+1)

            tidx += 1
            tri[1, tidx] = v1
            tri[2, tidx] = v2
            tri[3, tidx] = v3

            tidx += 1
            tri[1, tidx] = v1
            tri[2, tidx] = v3
            tri[3, tidx] = v4
        end
    end

    return TriMesh(xyz, tri)
end

"""
    make_parabolic_reflector(D, f, Nr, Nphi; center=Vec3(0,0,0))

Generate a triangulated open parabolic reflector with aperture diameter `D`
and focal length `f`, aligned with +z:

`z = (x² + y²)/(4f)`, for `x² + y² ≤ (D/2)²`.

The mesh uses `Nr` radial rings and `Nphi` azimuth samples per ring.
Returns a `TriMesh` suitable for open-surface EFIE runs (`allow_boundary=true`).
"""
function make_parabolic_reflector(D::Real, f::Real, Nr::Int, Nphi::Int;
                                  center::Vec3=Vec3(0.0, 0.0, 0.0))
    D > 0 || error("Reflector diameter D must be positive.")
    f > 0 || error("Reflector focal length f must be positive.")
    Nr >= 2 || error("Nr must be at least 2.")
    Nphi >= 3 || error("Nphi must be at least 3.")

    R = D / 2
    Nv = 1 + Nr * Nphi
    Nt = Nphi + 2 * (Nr - 1) * Nphi

    xyz = zeros(3, Nv)
    tri = zeros(Int, 3, Nt)

    # Vertex 1: apex
    xyz[:, 1] = center

    @inline vid(ir, j) = 2 + (ir - 1) * Nphi + (j - 1)  # ir=1:Nr, j=1:Nphi
    @inline jnext(j) = (j == Nphi) ? 1 : (j + 1)

    # Ring vertices
    for ir in 1:Nr
        r = R * ir / Nr
        z = r^2 / (4f)
        for j in 1:Nphi
            ϕ = 2π * (j - 1) / Nphi
            idx = vid(ir, j)
            xyz[1, idx] = center[1] + r * cos(ϕ)
            xyz[2, idx] = center[2] + r * sin(ϕ)
            xyz[3, idx] = center[3] + z
        end
    end

    # Center fan
    tid = 0
    for j in 1:Nphi
        tid += 1
        tri[:, tid] = [1, vid(1, j), vid(1, jnext(j))]
    end

    # Ring-to-ring quads split into 2 triangles
    for ir in 1:(Nr - 1)
        for j in 1:Nphi
            v00 = vid(ir, j)
            v01 = vid(ir, jnext(j))
            v10 = vid(ir + 1, j)
            v11 = vid(ir + 1, jnext(j))

            tid += 1
            tri[:, tid] = [v00, v10, v11]
            tid += 1
            tri[:, tid] = [v00, v11, v01]
        end
    end

    return TriMesh(xyz, tri)
end

function _bbox_diagonal(mesh::TriMesh)
    mins = map(i -> minimum(@view mesh.xyz[i, :]), 1:3)
    maxs = map(i -> maximum(@view mesh.xyz[i, :]), 1:3)
    return norm(Vec3(maxs...) - Vec3(mins...))
end

"""
    mesh_quality_report(mesh; area_tol_rel=1e-12, check_orientation=true)

Compute mesh-quality diagnostics for a triangle surface mesh.
The report includes:
- invalid triangles (index out of bounds or repeated vertices),
- degenerate triangles (area below tolerance),
- boundary-edge count,
- non-manifold-edge count (>2 incident triangles),
- orientation-conflict count on interior edges.
"""
function mesh_quality_report(mesh::TriMesh; area_tol_rel::Float64=1e-12, check_orientation::Bool=true)
    Nv = nvertices(mesh)
    Nt = ntriangles(mesh)

    size(mesh.xyz, 1) == 3 || error("Mesh xyz must have size (3, Nv)")
    size(mesh.tri, 1) == 3 || error("Mesh tri must have size (3, Nt)")

    scale = max(_bbox_diagonal(mesh), 1.0)
    area_tol_abs = area_tol_rel * scale^2

    invalid_triangles = Int[]
    degenerate_triangles = Int[]

    # edge_map[(i,j)] = directions of edge traversal in each incident triangle
    # direction +1 means (i->j) where i<j, -1 means (j->i)
    edge_map = Dict{Tuple{Int,Int}, Vector{Int8}}()

    for t in 1:Nt
        i1 = mesh.tri[1, t]
        i2 = mesh.tri[2, t]
        i3 = mesh.tri[3, t]

        valid_idx = (1 <= i1 <= Nv) && (1 <= i2 <= Nv) && (1 <= i3 <= Nv)
        distinct = (i1 != i2) && (i2 != i3) && (i3 != i1)
        if !(valid_idx && distinct)
            push!(invalid_triangles, t)
            continue
        end

        if triangle_area(mesh, t) <= area_tol_abs
            push!(degenerate_triangles, t)
        end

        for (a, b) in ((i1, i2), (i2, i3), (i3, i1))
            key = a < b ? (a, b) : (b, a)
            dir = a < b ? Int8(1) : Int8(-1)
            push!(get!(edge_map, key, Int8[]), dir)
        end
    end

    n_boundary_edges = 0
    n_nonmanifold_edges = 0
    n_orientation_conflicts = 0

    for dirs in values(edge_map)
        nd = length(dirs)
        if nd == 1
            n_boundary_edges += 1
        elseif nd == 2
            if check_orientation && dirs[1] == dirs[2]
                n_orientation_conflicts += 1
            end
        elseif nd > 2
            n_nonmanifold_edges += 1
        end
    end

    n_edges_total = length(edge_map)
    n_interior_edges = n_edges_total - n_boundary_edges - n_nonmanifold_edges

    return (
        n_vertices = Nv,
        n_triangles = Nt,
        n_edges_total = n_edges_total,
        n_interior_edges = n_interior_edges,
        n_boundary_edges = n_boundary_edges,
        n_nonmanifold_edges = n_nonmanifold_edges,
        n_orientation_conflicts = n_orientation_conflicts,
        n_invalid_triangles = length(invalid_triangles),
        n_degenerate_triangles = length(degenerate_triangles),
        invalid_triangles = invalid_triangles,
        degenerate_triangles = degenerate_triangles,
        area_tol_abs = area_tol_abs,
    )
end

"""
    mesh_quality_ok(report; allow_boundary=true, require_closed=false)

Return `true` if a mesh-quality report passes hard checks:
- no invalid triangles,
- no degenerate triangles,
- no non-manifold edges,
- no orientation conflicts,
- boundary edges allowed unless `allow_boundary=false` or `require_closed=true`.
"""
function mesh_quality_ok(report; allow_boundary::Bool=true, require_closed::Bool=false)
    if report.n_invalid_triangles > 0
        return false
    end
    if report.n_degenerate_triangles > 0
        return false
    end
    if report.n_nonmanifold_edges > 0
        return false
    end
    if report.n_orientation_conflicts > 0
        return false
    end
    if require_closed && report.n_boundary_edges > 0
        return false
    end
    if !allow_boundary && report.n_boundary_edges > 0
        return false
    end
    return true
end

"""
    assert_mesh_quality(mesh; allow_boundary=true, require_closed=false, area_tol_rel=1e-12)

Run mesh-quality checks and throw a detailed error if the mesh is unsuitable.
Returns the computed quality report on success.
"""
function assert_mesh_quality(mesh::TriMesh;
                             allow_boundary::Bool=true,
                             require_closed::Bool=false,
                             area_tol_rel::Float64=1e-12)
    report = mesh_quality_report(mesh; area_tol_rel=area_tol_rel, check_orientation=true)
    problems = String[]

    if report.n_invalid_triangles > 0
        sample = join(report.invalid_triangles[1:min(end, 5)], ", ")
        push!(problems, "invalid triangles: $(report.n_invalid_triangles) (sample: $sample)")
    end
    if report.n_degenerate_triangles > 0
        sample = join(report.degenerate_triangles[1:min(end, 5)], ", ")
        push!(problems, "degenerate triangles: $(report.n_degenerate_triangles) (sample: $sample), area_tol_abs=$(report.area_tol_abs)")
    end
    if report.n_nonmanifold_edges > 0
        push!(problems, "non-manifold edges: $(report.n_nonmanifold_edges)")
    end
    if report.n_orientation_conflicts > 0
        push!(problems, "orientation conflicts on interior edges: $(report.n_orientation_conflicts)")
    end
    if require_closed && report.n_boundary_edges > 0
        push!(problems, "boundary edges present but closed surface required: $(report.n_boundary_edges)")
    elseif !allow_boundary && report.n_boundary_edges > 0
        push!(problems, "boundary edges not allowed: $(report.n_boundary_edges)")
    end

    if !isempty(problems)
        msg = "Mesh quality precheck failed:\n  - " * join(problems, "\n  - ")
        error(msg)
    end

    return report
end

"""
    read_obj_mesh(path)

Read a triangle mesh from a Wavefront OBJ file and return a `TriMesh`.

Supported records:
- `v x y z`
- `f i j k ...` (triangles or polygons; polygons are fan-triangulated)

Texture/normal indices (`f v/t/n`) are ignored. Positive and negative OBJ
vertex indices are supported.
"""
function read_obj_mesh(path::AbstractString)
    vertices = Vector{NTuple{3,Float64}}()
    faces = Vector{NTuple{3,Int}}()

    open(path, "r") do io
        for raw_line in eachline(io)
            line = strip(raw_line)
            isempty(line) && continue
            startswith(line, "#") && continue

            if startswith(line, "v ")
                fields = split(line)
                length(fields) < 4 && error("Invalid OBJ vertex line: $line")
                x = parse(Float64, fields[2])
                y = parse(Float64, fields[3])
                z = parse(Float64, fields[4])
                push!(vertices, (x, y, z))

            elseif startswith(line, "f ")
                fields = split(line)[2:end]
                length(fields) < 3 && error("Invalid OBJ face line: $line")

                face_idx = Int[]
                for token in fields
                    vtoken = split(token, "/")[1]
                    isempty(vtoken) && error("Invalid OBJ face token: $token")
                    idx_raw = parse(Int, vtoken)
                    idx = idx_raw > 0 ? idx_raw : (length(vertices) + idx_raw + 1)
                    (1 <= idx <= length(vertices)) || error("OBJ face index out of range in line: $line")
                    push!(face_idx, idx)
                end

                v1 = face_idx[1]
                for j in 2:(length(face_idx) - 1)
                    push!(faces, (v1, face_idx[j], face_idx[j + 1]))
                end
            end
        end
    end

    isempty(vertices) && error("OBJ mesh has no vertices: $path")
    isempty(faces) && error("OBJ mesh has no faces: $path")

    xyz = zeros(Float64, 3, length(vertices))
    for i in eachindex(vertices)
        x, y, z = vertices[i]
        xyz[1, i] = x
        xyz[2, i] = y
        xyz[3, i] = z
    end

    tri = zeros(Int, 3, length(faces))
    for t in eachindex(faces)
        i1, i2, i3 = faces[t]
        tri[1, t] = i1
        tri[2, t] = i2
        tri[3, t] = i3
    end

    return TriMesh(xyz, tri)
end

"""
    write_obj_mesh(path, mesh; header="...")

Write a `TriMesh` to a Wavefront OBJ file using triangle faces.
"""
function write_obj_mesh(path::AbstractString, mesh::TriMesh; header::AbstractString="Exported by DiffMoM")
    open(path, "w") do io
        println(io, "# $header")
        for i in 1:nvertices(mesh)
            println(io, "v $(mesh.xyz[1, i]) $(mesh.xyz[2, i]) $(mesh.xyz[3, i])")
        end
        for t in 1:ntriangles(mesh)
            println(io, "f $(mesh.tri[1, t]) $(mesh.tri[2, t]) $(mesh.tri[3, t])")
        end
    end
    return path
end

"""
    estimate_dense_matrix_gib(N)

Estimate memory (GiB) for a dense complex `N × N` matrix with `ComplexF64`
entries (16 bytes per entry).
"""
estimate_dense_matrix_gib(N::Integer) = 16.0 * float(N) * float(N) / 1024.0^3

"""
    cluster_mesh_vertices(mesh, h)

Voxel-cluster a mesh using cubic cell size `h`, replacing all vertices in each
cell by their centroid and remapping triangles. Degenerate and duplicate
triangles created by remapping are removed.
"""
function cluster_mesh_vertices(mesh::TriMesh, h::Float64)
    h > 0 || error("cluster_mesh_vertices: h must be > 0, got $h")

    nv = nvertices(mesh)
    mins = (
        minimum(@view mesh.xyz[1, :]),
        minimum(@view mesh.xyz[2, :]),
        minimum(@view mesh.xyz[3, :]),
    )

    key_to_id = Dict{NTuple{3,Int},Int}()
    vmap = Vector{Int}(undef, nv)
    sx = Float64[]
    sy = Float64[]
    sz = Float64[]
    sc = Int[]

    for i in 1:nv
        x = mesh.xyz[1, i]
        y = mesh.xyz[2, i]
        z = mesh.xyz[3, i]
        key = (
            floor(Int, (x - mins[1]) / h),
            floor(Int, (y - mins[2]) / h),
            floor(Int, (z - mins[3]) / h),
        )
        id = get!(key_to_id, key) do
            push!(sx, 0.0)
            push!(sy, 0.0)
            push!(sz, 0.0)
            push!(sc, 0)
            length(sx)
        end
        vmap[i] = id
        sx[id] += x
        sy[id] += y
        sz[id] += z
        sc[id] += 1
    end

    nnew = length(sx)
    xyz_new = zeros(Float64, 3, nnew)
    for i in 1:nnew
        invc = 1.0 / sc[i]
        xyz_new[1, i] = sx[i] * invc
        xyz_new[2, i] = sy[i] * invc
        xyz_new[3, i] = sz[i] * invc
    end

    tri_vec = Int[]
    seen = Set{NTuple{3,Int}}()
    for t in 1:ntriangles(mesh)
        a = vmap[mesh.tri[1, t]]
        b = vmap[mesh.tri[2, t]]
        c = vmap[mesh.tri[3, t]]
        if a == b || b == c || c == a
            continue
        end
        ss = sort([a, b, c])
        key = (ss[1], ss[2], ss[3])
        if key in seen
            continue
        end
        push!(seen, key)
        push!(tri_vec, a, b, c)
    end

    isempty(tri_vec) && error("cluster_mesh_vertices: clustering removed all triangles.")
    tri_new = reshape(tri_vec, 3, :)
    return TriMesh(xyz_new, tri_new)
end

"""
    drop_nonmanifold_triangles(mesh; max_passes=8)

Iteratively remove triangles attached to non-manifold edges (edges with more
than two incident triangles). Returns a mesh with only manifold/boundary edges.
"""
function drop_nonmanifold_triangles(mesh::TriMesh; max_passes::Int=8)
    nt = ntriangles(mesh)
    keep = trues(nt)

    for _ in 1:max_passes
        edge_to_tris = Dict{Tuple{Int,Int}, Vector{Int}}()
        for t in 1:nt
            keep[t] || continue
            i1 = mesh.tri[1, t]
            i2 = mesh.tri[2, t]
            i3 = mesh.tri[3, t]
            for (a, b) in ((i1, i2), (i2, i3), (i3, i1))
                key = a < b ? (a, b) : (b, a)
                push!(get!(edge_to_tris, key, Int[]), t)
            end
        end

        bad = falses(nt)
        nbad_edges = 0
        for tris in values(edge_to_tris)
            if length(tris) > 2
                nbad_edges += 1
                for t in tris
                    bad[t] = true
                end
            end
        end

        nbad_edges == 0 && break
        keep .&= .!bad
    end

    tri_new = copy(mesh.tri[:, keep])
    isempty(tri_new) && error("drop_nonmanifold_triangles: empty mesh after cleanup.")
    return TriMesh(copy(mesh.xyz), tri_new)
end

"""
    coarsen_mesh_to_target_rwg(mesh, target_rwg; kwargs...)

Auto-coarsen a mesh by voxel clustering to approach a target RWG count.
Each candidate mesh is non-manifold cleaned and repaired before RWG counting.

Returns a named tuple:
`(mesh, rwg_count, target_rwg, best_gap, iterations)`.
"""
function coarsen_mesh_to_target_rwg(mesh::TriMesh, target_rwg::Int;
                                    max_iters::Int=10,
                                    allow_boundary::Bool=true,
                                    require_closed::Bool=false,
                                    area_tol_rel::Float64=1e-12,
                                    strict_nonmanifold::Bool=true)
    target_rwg > 0 || error("coarsen_mesh_to_target_rwg: target_rwg must be > 0")

    mins = [minimum(@view mesh.xyz[i, :]) for i in 1:3]
    maxs = [maximum(@view mesh.xyz[i, :]) for i in 1:3]
    span = maxs .- mins
    bbox_vol_raw = prod(span)
    if bbox_vol_raw <= 1e-18
        max_span = max(maximum(span), 1e-6)
        bbox_vol = max_span^3
    else
        bbox_vol = bbox_vol_raw
    end
    target_vertices = max(80, Int(round(target_rwg / 3)))
    h = cbrt(bbox_vol / target_vertices)

    best_mesh = mesh
    best_gap = typemax(Int)
    best_rwg = build_rwg(mesh; precheck=true, allow_boundary=allow_boundary,
                         require_closed=require_closed, area_tol_rel=area_tol_rel).nedges
    niter = 0

    for iter in 1:max_iters
        cand = cluster_mesh_vertices(mesh, h)
        cand = drop_nonmanifold_triangles(cand)
        cand_rep = repair_mesh_for_simulation(
            cand;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
            drop_invalid=true,
            drop_degenerate=true,
            fix_orientation=true,
            strict_nonmanifold=strict_nonmanifold,
        )
        cand_mesh = cand_rep.mesh
        nrwg = build_rwg(
            cand_mesh;
            precheck=true,
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        ).nedges

        gap = abs(nrwg - target_rwg)
        if gap < best_gap
            best_gap = gap
            best_mesh = cand_mesh
            best_rwg = nrwg
        end
        niter = iter

        ratio = nrwg / max(target_rwg, 1)
        if 0.85 <= ratio <= 1.15
            return (mesh=cand_mesh, rwg_count=nrwg, target_rwg=target_rwg, best_gap=gap, iterations=iter)
        end

        h *= ratio^(1 / 3)
    end

    return (mesh=best_mesh, rwg_count=best_rwg, target_rwg=target_rwg, best_gap=best_gap, iterations=niter)
end

function _clean_mesh_triangles(mesh::TriMesh;
                               drop_invalid::Bool=true,
                               drop_degenerate::Bool=true,
                               area_tol_rel::Float64=1e-12)
    nv = nvertices(mesh)
    nt = ntriangles(mesh)
    tri = mesh.tri
    xyz = mesh.xyz

    scale = max(_bbox_diagonal(mesh), 1.0)
    area_tol_abs = area_tol_rel * scale^2

    keep_triangle = trues(nt)
    removed_invalid = Int[]
    removed_degenerate = Int[]

    for t in 1:nt
        i1 = tri[1, t]
        i2 = tri[2, t]
        i3 = tri[3, t]

        valid_idx = (1 <= i1 <= nv) && (1 <= i2 <= nv) && (1 <= i3 <= nv)
        distinct = (i1 != i2) && (i2 != i3) && (i3 != i1)

        if !(valid_idx && distinct)
            if drop_invalid
                keep_triangle[t] = false
                push!(removed_invalid, t)
                continue
            else
                error("Triangle $t is invalid and `drop_invalid=false`.")
            end
        end

        v1 = Vec3(xyz[:, i1])
        v2 = Vec3(xyz[:, i2])
        v3 = Vec3(xyz[:, i3])
        area = 0.5 * norm(cross(v2 - v1, v3 - v1))

        if area <= area_tol_abs
            if drop_degenerate
                keep_triangle[t] = false
                push!(removed_degenerate, t)
            else
                error("Triangle $t is degenerate (area=$area <= $area_tol_abs) and `drop_degenerate=false`.")
            end
        end
    end

    tri_clean = copy(tri[:, keep_triangle])
    cleaned_mesh = TriMesh(copy(xyz), tri_clean)
    return cleaned_mesh, removed_invalid, removed_degenerate, area_tol_abs
end

function _edge_orientation_adjacency(mesh::TriMesh)
    nt = ntriangles(mesh)
    edge_map = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int8}}}()

    for t in 1:nt
        i1 = mesh.tri[1, t]
        i2 = mesh.tri[2, t]
        i3 = mesh.tri[3, t]
        for (a, b) in ((i1, i2), (i2, i3), (i3, i1))
            key = a < b ? (a, b) : (b, a)
            dir = a < b ? Int8(1) : Int8(-1)
            push!(get!(edge_map, key, Tuple{Int,Int8}[]), (t, dir))
        end
    end

    adjacency = [Tuple{Int,Int8}[] for _ in 1:nt]
    for refs in values(edge_map)
        if length(refs) == 2
            (t1, d1) = refs[1]
            (t2, d2) = refs[2]
            parity = d1 == d2 ? Int8(1) : Int8(0)
            push!(adjacency[t1], (t2, parity))
            push!(adjacency[t2], (t1, parity))
        end
    end

    return adjacency
end

function _compute_orientation_flips(mesh::TriMesh)
    nt = ntriangles(mesh)
    adjacency = _edge_orientation_adjacency(mesh)

    flip_flag = fill(Int8(-1), nt)
    queue = Int[]

    for start in 1:nt
        if flip_flag[start] != -1
            continue
        end

        flip_flag[start] = 0
        empty!(queue)
        push!(queue, start)
        queue_index = 1

        while queue_index <= length(queue)
            t = queue[queue_index]
            queue_index += 1

            for (nbr, parity) in adjacency[t]
                expected = Int8(mod(Int(flip_flag[t]) + Int(parity), 2))
                if flip_flag[nbr] == -1
                    flip_flag[nbr] = expected
                    push!(queue, nbr)
                elseif flip_flag[nbr] != expected
                    error("Orientation repair failed: inconsistent winding constraints in triangle graph.")
                end
            end
        end
    end

    return flip_flag
end

function _apply_orientation_flips(mesh::TriMesh, flip_flag::Vector{Int8})
    tri = copy(mesh.tri)
    flipped_triangles = Int[]
    for t in 1:ntriangles(mesh)
        if flip_flag[t] == 1
            tri[2, t], tri[3, t] = tri[3, t], tri[2, t]
            push!(flipped_triangles, t)
        end
    end
    return TriMesh(copy(mesh.xyz), tri), flipped_triangles
end

"""
    repair_mesh_for_simulation(mesh;
        allow_boundary=true, require_closed=false, area_tol_rel=1e-12,
        drop_invalid=true, drop_degenerate=true,
        fix_orientation=true, strict_nonmanifold=true,
        auto_drop_nonmanifold=true)

Repair a triangle mesh so it can pass solver prechecks:
- optionally remove invalid/degenerate triangles,
- optionally drop triangles causing non-manifold edges (enabled by default),
- orient triangles consistently across manifold interior edges.

Set `auto_drop_nonmanifold=false` when you want strict fail-fast behavior on
non-manifold edges.

Returns a named tuple containing the repaired mesh and before/after reports.
"""
function repair_mesh_for_simulation(mesh::TriMesh;
                                    allow_boundary::Bool=true,
                                    require_closed::Bool=false,
                                    area_tol_rel::Float64=1e-12,
                                    drop_invalid::Bool=true,
                                    drop_degenerate::Bool=true,
                                    fix_orientation::Bool=true,
                                    strict_nonmanifold::Bool=true,
                                    auto_drop_nonmanifold::Bool=true)
    report_before = mesh_quality_report(mesh; area_tol_rel=area_tol_rel, check_orientation=true)

    cleaned_mesh, removed_invalid, removed_degenerate, area_tol_abs = _clean_mesh_triangles(
        mesh;
        drop_invalid=drop_invalid,
        drop_degenerate=drop_degenerate,
        area_tol_rel=area_tol_rel,
    )
    report_cleaned = mesh_quality_report(cleaned_mesh; area_tol_rel=area_tol_rel, check_orientation=true)
    removed_nonmanifold = 0

    if auto_drop_nonmanifold && report_cleaned.n_nonmanifold_edges > 0
        mesh_nm = drop_nonmanifold_triangles(cleaned_mesh)
        removed_nonmanifold = ntriangles(cleaned_mesh) - ntriangles(mesh_nm)
        cleaned_mesh = mesh_nm
        report_cleaned = mesh_quality_report(cleaned_mesh; area_tol_rel=area_tol_rel, check_orientation=true)
    end

    if strict_nonmanifold && report_cleaned.n_nonmanifold_edges > 0
        error("Mesh repair cannot continue with non-manifold edges ($(report_cleaned.n_nonmanifold_edges)).")
    end

    repaired_mesh = cleaned_mesh
    flipped_triangles = Int[]
    if fix_orientation && report_cleaned.n_orientation_conflicts > 0
        flip_flag = _compute_orientation_flips(cleaned_mesh)
        repaired_mesh, flipped_triangles = _apply_orientation_flips(cleaned_mesh, flip_flag)
    end

    report_after = mesh_quality_report(repaired_mesh; area_tol_rel=area_tol_rel, check_orientation=true)
    assert_mesh_quality(
        repaired_mesh;
        allow_boundary=allow_boundary,
        require_closed=require_closed,
        area_tol_rel=area_tol_rel,
    )

    return (
        mesh = repaired_mesh,
        before = report_before,
        cleaned = report_cleaned,
        after = report_after,
        removed_invalid = removed_invalid,
        removed_degenerate = removed_degenerate,
        removed_nonmanifold = removed_nonmanifold,
        flipped_triangles = flipped_triangles,
        area_tol_abs = area_tol_abs,
    )
end

"""
    repair_obj_mesh(input_path, output_path; kwargs...)

Read an OBJ mesh, repair it for solver prechecks, and write a repaired OBJ.
Returns the same metadata as `repair_mesh_for_simulation`, plus `output_path`.
"""
function repair_obj_mesh(input_path::AbstractString, output_path::AbstractString; kwargs...)
    mesh = read_obj_mesh(input_path)
    result = repair_mesh_for_simulation(mesh; kwargs...)
    write_obj_mesh(output_path, result.mesh; header="Repaired from $input_path by DiffMoM")
    return (; result..., output_path=output_path)
end

@inline _mesh_vertex(mesh::TriMesh, i::Int) = Vec3(mesh.xyz[1, i], mesh.xyz[2, i], mesh.xyz[3, i])

"""
    triangle_area(mesh, t)

Compute the area of triangle `t` in the mesh.
"""
function triangle_area(mesh::TriMesh, t::Int)
    v1 = _mesh_vertex(mesh, mesh.tri[1, t])
    v2 = _mesh_vertex(mesh, mesh.tri[2, t])
    v3 = _mesh_vertex(mesh, mesh.tri[3, t])
    return 0.5 * norm(cross(v2 - v1, v3 - v1))
end

"""
    triangle_center(mesh, t)

Compute the centroid of triangle `t`.
"""
function triangle_center(mesh::TriMesh, t::Int)
    v1 = _mesh_vertex(mesh, mesh.tri[1, t])
    v2 = _mesh_vertex(mesh, mesh.tri[2, t])
    v3 = _mesh_vertex(mesh, mesh.tri[3, t])
    return (v1 + v2 + v3) / 3
end

"""
    triangle_normal(mesh, t)

Compute the outward unit normal of triangle `t`.
"""
function triangle_normal(mesh::TriMesh, t::Int)
    v1 = _mesh_vertex(mesh, mesh.tri[1, t])
    v2 = _mesh_vertex(mesh, mesh.tri[2, t])
    v3 = _mesh_vertex(mesh, mesh.tri[3, t])
    n = cross(v2 - v1, v3 - v1)
    return n / norm(n)
end

"""
    mesh_unique_edges(mesh)

Return the unique undirected edges of a triangle mesh as a vector of
`(i, j)` vertex-index pairs with `i < j`.
"""
function mesh_unique_edges(mesh::TriMesh)
    edges = Set{Tuple{Int,Int}}()
    for t in 1:ntriangles(mesh)
        i1 = mesh.tri[1, t]
        i2 = mesh.tri[2, t]
        i3 = mesh.tri[3, t]
        for (a, b) in ((i1, i2), (i2, i3), (i3, i1))
            key = a < b ? (a, b) : (b, a)
            push!(edges, key)
        end
    end
    return collect(edges)
end

"""
    mesh_wireframe_segments(mesh)

Build line-segment arrays for lightweight 3D wireframe visualization.
Returns a named tuple `(x, y, z, n_edges)` where each edge contributes
`(p1, p2, NaN)` to each coordinate vector, suitable for `Plots.path3d`.
"""
function mesh_wireframe_segments(mesh::TriMesh)
    edges = mesh_unique_edges(mesh)
    n_edges = length(edges)
    x = Vector{Float64}(undef, 3 * n_edges)
    y = Vector{Float64}(undef, 3 * n_edges)
    z = Vector{Float64}(undef, 3 * n_edges)

    k = 1
    for (i, j) in edges
        x[k] = mesh.xyz[1, i]
        y[k] = mesh.xyz[2, i]
        z[k] = mesh.xyz[3, i]
        k += 1
        x[k] = mesh.xyz[1, j]
        y[k] = mesh.xyz[2, j]
        z[k] = mesh.xyz[3, j]
        k += 1
        x[k] = NaN
        y[k] = NaN
        z[k] = NaN
        k += 1
    end

    return (x=x, y=y, z=z, n_edges=n_edges)
end

function _mesh_edge_lengths(mesh::TriMesh)
    edges = mesh_unique_edges(mesh)
    lens = Vector{Float64}(undef, length(edges))
    for (k, (i, j)) in enumerate(edges)
        lens[k] = norm(_mesh_vertex(mesh, i) - _mesh_vertex(mesh, j))
    end
    return lens
end

function _percentile_from_sorted(sorted_vals::Vector{Float64}, p::Float64)
    n = length(sorted_vals)
    n == 0 && return 0.0
    idx = clamp(ceil(Int, p * n), 1, n)
    return sorted_vals[idx]
end

"""
    mesh_resolution_report(mesh, freq_hz; points_per_wavelength=10.0, c0=299792458.0)

Compute electrical mesh-resolution diagnostics for MoM at frequency `freq_hz`.

The core criterion is `h_max <= λ / points_per_wavelength`, where `h_max` is
the maximum unique edge length.
"""
function mesh_resolution_report(mesh::TriMesh, freq_hz::Real;
                                points_per_wavelength::Real=10.0,
                                c0::Real=299792458.0)
    freq_hz > 0 || error("mesh_resolution_report: freq_hz must be > 0")
    points_per_wavelength > 0 || error("mesh_resolution_report: points_per_wavelength must be > 0")
    c0 > 0 || error("mesh_resolution_report: c0 must be > 0")

    λ = Float64(c0) / Float64(freq_hz)
    target_h = λ / Float64(points_per_wavelength)

    lens = _mesh_edge_lengths(mesh)
    isempty(lens) && error("mesh_resolution_report: mesh has no edges")
    lens_sorted = sort(lens)

    h_min = lens_sorted[1]
    h_med = _percentile_from_sorted(lens_sorted, 0.50)
    h_p95 = _percentile_from_sorted(lens_sorted, 0.95)
    h_mean = sum(lens_sorted) / length(lens_sorted)
    h_max = lens_sorted[end]

    meets = h_max <= target_h

    return (
        freq_hz = Float64(freq_hz),
        wavelength_m = λ,
        points_per_wavelength = Float64(points_per_wavelength),
        target_max_edge_m = target_h,
        n_vertices = nvertices(mesh),
        n_triangles = ntriangles(mesh),
        n_edges = length(lens_sorted),
        edge_min_m = h_min,
        edge_median_m = h_med,
        edge_p95_m = h_p95,
        edge_mean_m = h_mean,
        edge_max_m = h_max,
        edge_median_over_lambda = h_med / λ,
        edge_p95_over_lambda = h_p95 / λ,
        edge_max_over_lambda = h_max / λ,
        meets_target = meets,
    )
end

"""
    mesh_resolution_ok(report; criterion=:max)

Evaluate a `mesh_resolution_report` against a selected criterion:
- `:max` (default): uses `edge_max_m`
- `:p95`: uses `edge_p95_m`
- `:median`: uses `edge_median_m`
"""
function mesh_resolution_ok(report; criterion::Symbol=:max)
    if criterion == :max
        return report.edge_max_m <= report.target_max_edge_m
    elseif criterion == :p95
        return report.edge_p95_m <= report.target_max_edge_m
    elseif criterion == :median
        return report.edge_median_m <= report.target_max_edge_m
    end
    error("mesh_resolution_ok: unknown criterion=$(criterion). Use :max, :p95, or :median.")
end

function _midpoint_refine_once(mesh::TriMesh)
    Nv = nvertices(mesh)
    Nt = ntriangles(mesh)

    xyz_list = [_mesh_vertex(mesh, i) for i in 1:Nv]
    edge_mid = Dict{Tuple{Int,Int}, Int}()

    function midpoint_index(i::Int, j::Int)
        key = i < j ? (i, j) : (j, i)
        if haskey(edge_mid, key)
            return edge_mid[key]
        end
        vm = 0.5 * (xyz_list[i] + xyz_list[j])
        push!(xyz_list, vm)
        idx = length(xyz_list)
        edge_mid[key] = idx
        return idx
    end

    tri_new = Matrix{Int}(undef, 3, 4 * Nt)
    tid = 0
    for t in 1:Nt
        a = mesh.tri[1, t]
        b = mesh.tri[2, t]
        c = mesh.tri[3, t]

        mab = midpoint_index(a, b)
        mbc = midpoint_index(b, c)
        mca = midpoint_index(c, a)

        tid += 1
        tri_new[1, tid] = a
        tri_new[2, tid] = mab
        tri_new[3, tid] = mca
        tid += 1
        tri_new[1, tid] = mab
        tri_new[2, tid] = b
        tri_new[3, tid] = mbc
        tid += 1
        tri_new[1, tid] = mca
        tri_new[2, tid] = mbc
        tri_new[3, tid] = c
        tid += 1
        tri_new[1, tid] = mab
        tri_new[2, tid] = mbc
        tri_new[3, tid] = mca
    end

    xyz_new = zeros(3, length(xyz_list))
    for i in 1:length(xyz_list)
        xyz_new[:, i] = xyz_list[i]
    end
    return TriMesh(xyz_new, tri_new)
end

"""
    refine_mesh_to_target_edge(mesh, target_max_edge_m; max_iters=8, max_triangles=2_000_000)

Uniformly refine a triangle mesh via midpoint subdivision until
`edge_max_m <= target_max_edge_m` or limits are reached.
"""
function refine_mesh_to_target_edge(mesh::TriMesh, target_max_edge_m::Real;
                                    max_iters::Int=8,
                                    max_triangles::Int=2_000_000)
    target_max_edge_m > 0 || error("refine_mesh_to_target_edge: target_max_edge_m must be > 0")
    max_iters >= 0 || error("refine_mesh_to_target_edge: max_iters must be >= 0")
    max_triangles > 0 || error("refine_mesh_to_target_edge: max_triangles must be > 0")

    mesh_cur = mesh
    before_lens = _mesh_edge_lengths(mesh_cur)
    isempty(before_lens) && error("refine_mesh_to_target_edge: mesh has no edges")
    edge_max_before = maximum(before_lens)

    hist_edge_max = Float64[edge_max_before]
    hist_triangles = Int[ntriangles(mesh_cur)]

    converged = edge_max_before <= target_max_edge_m
    iters = 0

    while !converged && iters < max_iters
        ntriangles(mesh_cur) * 4 <= max_triangles || break
        mesh_cur = _midpoint_refine_once(mesh_cur)
        iters += 1

        lens = _mesh_edge_lengths(mesh_cur)
        edge_max = maximum(lens)
        push!(hist_edge_max, edge_max)
        push!(hist_triangles, ntriangles(mesh_cur))
        converged = edge_max <= target_max_edge_m
    end

    return (
        mesh = mesh_cur,
        iterations = iters,
        converged = converged,
        target_max_edge_m = Float64(target_max_edge_m),
        edge_max_before_m = edge_max_before,
        edge_max_after_m = hist_edge_max[end],
        triangles_before = hist_triangles[1],
        triangles_after = hist_triangles[end],
        history_edge_max_m = hist_edge_max,
        history_triangles = hist_triangles,
    )
end

"""
    refine_mesh_for_mom(mesh, freq_hz; points_per_wavelength=10.0, max_iters=8, max_triangles=2_000_000)

Refine a mesh to satisfy a frequency-based MoM edge-length target:
`target_max_edge_m = λ / points_per_wavelength`.
"""
function refine_mesh_for_mom(mesh::TriMesh, freq_hz::Real;
                             points_per_wavelength::Real=10.0,
                             max_iters::Int=8,
                             max_triangles::Int=2_000_000,
                             c0::Real=299792458.0)
    report_before = mesh_resolution_report(mesh, freq_hz;
                                           points_per_wavelength=points_per_wavelength,
                                           c0=c0)
    result = refine_mesh_to_target_edge(mesh, report_before.target_max_edge_m;
                                        max_iters=max_iters,
                                        max_triangles=max_triangles)
    report_after = mesh_resolution_report(result.mesh, freq_hz;
                                          points_per_wavelength=points_per_wavelength,
                                          c0=c0)
    return (; result..., report_before=report_before, report_after=report_after)
end
