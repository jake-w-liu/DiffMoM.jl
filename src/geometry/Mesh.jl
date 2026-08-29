# Mesh.jl — Simple mesh generation and geometry utilities

export make_rect_plate, make_rect_plate_graded, make_circular_plate, make_parabolic_reflector, read_obj_mesh, triangle_area, triangle_center, triangle_normal
export mesh_quality_report, mesh_quality_ok, assert_mesh_quality
export write_obj_mesh, repair_mesh_for_simulation, repair_obj_mesh
export estimate_dense_matrix_gib, cluster_mesh_vertices, drop_nonmanifold_triangles
export coarsen_mesh_to_target_rwg
export mesh_unique_edges, mesh_wireframe_segments
export mesh_resolution_report, mesh_resolution_ok
export refine_mesh_to_target_edge, refine_mesh_for_mom

const _DEFAULT_MESH_MAX_VERTICES = 5_000_000
const _DEFAULT_MESH_MAX_TRIANGLES = 10_000_000
const _DEFAULT_MESH_MAX_RAW_BYTES = 512 * 1024 * 1024
const _DEFAULT_MESH_MAX_INPUT_BYTES = 1024 * 1024 * 1024
const _DEFAULT_MESH_MAX_LINE_BYTES = 1024 * 1024
const _MESH_INPUT_SCAN_BUFFER_BYTES = 64 * 1024
const _DEFAULT_MAX_MESH_EDGE_RECORDS = 30_000_000
const _DEFAULT_MAX_MESH_EDGE_OUTPUT_BYTES = 512 * 1024 * 1024

@inline function _validated_mesh_resource_limit(
        name::AbstractString, value::Integer)
    value isa Bool &&
        throw(ArgumentError("$name must be a positive integer byte/count limit, got $value"))
    value > 0 ||
        throw(ArgumentError("$name must be positive, got $value"))
    value <= typemax(Int) ||
        throw(ArgumentError("$name is outside the supported Int range: $value"))
    return Int(value)
end

function _mesh_input_size(io::IO, path::AbstractString;
                          max_input_bytes::Integer=_DEFAULT_MESH_MAX_INPUT_BYTES)
    input_limit = _validated_mesh_resource_limit(
        "max_input_bytes", max_input_bytes)
    input_bytes = try
        seekend(io)
        position(io)
    catch err
        throw(ArgumentError(
            "Could not determine the byte size of mesh input $path: " *
            sprint(showerror, err)))
    finally
        seekstart(io)
    end
    input_bytes <= input_limit ||
        throw(ArgumentError(
            "Mesh input $path contains $input_bytes bytes, exceeding " *
            "max_input_bytes=$input_limit"))
    return input_bytes
end

mutable struct _BoundedMeshTextIO{I<:IO} <: IO
    io::I
    path::String
    input_limit::Int
    line_limit::Int
    buffer::Vector{UInt8}
    buffer_position::Int
    buffer_length::Int
    bytes_read::Int
    line_number::Int
end

function _BoundedMeshTextIO(
        io::I, path::AbstractString;
        max_input_bytes::Integer=_DEFAULT_MESH_MAX_INPUT_BYTES,
        max_line_bytes::Integer=_DEFAULT_MESH_MAX_LINE_BYTES) where {I<:IO}
    input_limit = _validated_mesh_resource_limit(
        "max_input_bytes", max_input_bytes)
    line_limit = _validated_mesh_resource_limit(
        "max_line_bytes", max_line_bytes)
    buffer = Vector{UInt8}(undef, min(
        _MESH_INPUT_SCAN_BUFFER_BYTES, input_limit))
    return _BoundedMeshTextIO(
        io, String(path), input_limit, line_limit, buffer,
        1, 0, 0, 1)
end

function Base.seekstart(reader::_BoundedMeshTextIO)
    seekstart(reader.io)
    reader.buffer_position = 1
    reader.buffer_length = 0
    reader.bytes_read = 0
    reader.line_number = 1
    return reader
end

@inline function _refill_mesh_text_buffer!(reader::_BoundedMeshTextIO)
    remaining = reader.input_limit - reader.bytes_read
    if remaining == 0
        eof(reader.io) ||
            throw(ArgumentError(
                "Mesh input $(reader.path) exceeds " *
                "max_input_bytes=$(reader.input_limit) while parsing"))
        reader.buffer_position = 1
        reader.buffer_length = 0
        return false
    end
    requested = min(length(reader.buffer), remaining)
    bytes_read = readbytes!(reader.io, reader.buffer, requested)
    reader.buffer_position = 1
    reader.buffer_length = bytes_read
    reader.bytes_read = Base.Checked.checked_add(reader.bytes_read, bytes_read)
    return bytes_read > 0
end

function Base.eof(reader::_BoundedMeshTextIO)
    reader.buffer_position <= reader.buffer_length && return false
    _refill_mesh_text_buffer!(reader) && return false
    return true
end

function Base.readline(reader::_BoundedMeshTextIO; keep::Bool=false)
    line = UInt8[]
    while true
        if reader.buffer_position > reader.buffer_length
            _refill_mesh_text_buffer!(reader) || break
        end
        newline = 0
        @inbounds for index in reader.buffer_position:reader.buffer_length
            if reader.buffer[index] == UInt8('\n')
                newline = index
                break
            end
        end
        segment_end = iszero(newline) ? reader.buffer_length : newline - 1
        segment_length = max(0, segment_end - reader.buffer_position + 1)
        if iszero(newline)
            resulting_length = try
                Base.Checked.checked_add(length(line), segment_length)
            catch err
                err isa OverflowError || rethrow()
                throw(ArgumentError(
                    "Mesh input $(reader.path) line $(reader.line_number) " *
                    "length overflows the supported Int range"))
            end
            # A CR at the end of a refill can be the first byte of a CRLF
            # terminator split across buffers. Defer counting only that one
            # byte until the next refill proves whether LF follows it.
            pending_cr = segment_length > 0 &&
                reader.buffer[segment_end] == UInt8('\r')
            (resulting_length <= reader.line_limit ||
             (pending_cr && resulting_length - 1 <= reader.line_limit)) ||
                throw(ArgumentError(
                    "Mesh input $(reader.path) line $(reader.line_number) " *
                    "exceeds max_line_bytes=$(reader.line_limit) while parsing"))
            if segment_length > 0
                old_length = length(line)
                resize!(line, resulting_length)
                copyto!(
                    line, old_length + 1, reader.buffer,
                    reader.buffer_position, segment_length)
            end
            reader.buffer_position = reader.buffer_length + 1
            continue
        end

        trailing_cr_in_segment = segment_length > 0 &&
            reader.buffer[segment_end] == UInt8('\r')
        trailing_cr_in_line = segment_length == 0 && !isempty(line) &&
            line[end] == UInt8('\r')
        # `max_line_bytes` bounds parsed line content, independent of whether
        # CRLF happens to straddle the 64 KiB refill boundary.
        prefix_content_length = length(line) - Int(trailing_cr_in_line)
        segment_content_length = segment_length - Int(trailing_cr_in_segment)
        resulting_content_length = try
            Base.Checked.checked_add(
                prefix_content_length, segment_content_length)
        catch err
            err isa OverflowError || rethrow()
            throw(ArgumentError(
                "Mesh input $(reader.path) line $(reader.line_number) " *
                "length overflows the supported Int range"))
        end
        resulting_content_length <= reader.line_limit ||
            throw(ArgumentError(
                "Mesh input $(reader.path) line $(reader.line_number) " *
                "exceeds max_line_bytes=$(reader.line_limit) while parsing"))

        if !keep && trailing_cr_in_line
            pop!(line)
        end
        bytes_to_copy = keep ? segment_length : segment_content_length
        if bytes_to_copy > 0
            old_length = length(line)
            resize!(line, old_length + bytes_to_copy)
            copyto!(
                line, old_length + 1, reader.buffer,
                reader.buffer_position, bytes_to_copy)
        end
        if !iszero(newline)
            reader.buffer_position = newline + 1
            reader.line_number = Base.Checked.checked_add(
                reader.line_number, 1)
            if keep
                push!(line, UInt8('\n'))
            end
            return String(line)
        end
    end
    length(line) <= reader.line_limit ||
        throw(ArgumentError(
            "Mesh input $(reader.path) line $(reader.line_number) " *
            "exceeds max_line_bytes=$(reader.line_limit) while parsing"))
    return String(line)
end

@inline function _mesh_raw_payload_bytes(nvertices::Int, ntriangles::Int)
    try
        values = Base.Checked.checked_add(nvertices, ntriangles)
        return Base.Checked.checked_mul(24, values)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "Mesh raw coordinate/connectivity payload overflows the " *
            "supported Int byte-count range for $nvertices vertices and " *
            "$ntriangles triangles"))
    end
end

@inline function _validate_mesh_resource_request(
        nvertices::Int, ntriangles::Int, context::AbstractString;
        max_vertices::Integer=_DEFAULT_MESH_MAX_VERTICES,
        max_triangles::Integer=_DEFAULT_MESH_MAX_TRIANGLES,
        max_raw_bytes::Integer=_DEFAULT_MESH_MAX_RAW_BYTES)
    vertex_limit = _validated_mesh_resource_limit("max_vertices", max_vertices)
    triangle_limit = _validated_mesh_resource_limit("max_triangles", max_triangles)
    byte_limit = _validated_mesh_resource_limit("max_raw_bytes", max_raw_bytes)
    nvertices >= 0 && ntriangles >= 0 ||
        throw(ArgumentError(
            "$context requires nonnegative mesh counts, got " *
            "$nvertices vertices and $ntriangles triangles"))
    nvertices <= vertex_limit ||
        throw(ArgumentError(
            "$context requires $nvertices vertices, exceeding " *
            "max_vertices=$vertex_limit"))
    ntriangles <= triangle_limit ||
        throw(ArgumentError(
            "$context requires $ntriangles triangles, exceeding " *
            "max_triangles=$triangle_limit"))
    payload_bytes = _mesh_raw_payload_bytes(nvertices, ntriangles)
    payload_bytes <= byte_limit ||
        throw(ArgumentError(
            "$context requires at least $payload_bytes bytes for raw " *
            "coordinates and connectivity, exceeding max_raw_bytes=$byte_limit"))
    return payload_bytes
end

@inline function _positive_finite_length(name::AbstractString, value::Real)
    converted = Float64(value)
    (isfinite(converted) && converted > 0.0) ||
        throw(ArgumentError("$name must be finite and positive, got $value"))
    return converted
end

@inline function _positive_subdivision(name::AbstractString, value::Int;
                                       minimum::Int=1)
    value >= minimum ||
        throw(ArgumentError("$name must be at least $minimum, got $value"))
    return value
end

function _rect_mesh_counts(Nx::Int, Ny::Int)
    try
        Nv = Base.Checked.checked_mul(
            Base.Checked.checked_add(Nx, 1),
            Base.Checked.checked_add(Ny, 1),
        )
        Nt = Base.Checked.checked_mul(Base.Checked.checked_mul(2, Nx), Ny)
        return Nv, Nt
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("Requested rectangular mesh dimensions overflow Int: Nx=$Nx, Ny=$Ny"))
    end
end

function _radial_mesh_counts(Nr::Int, Nphi::Int)
    try
        Nv = Base.Checked.checked_add(1, Base.Checked.checked_mul(Nr, Nphi))
        Nt = Base.Checked.checked_mul(
            Nphi,
            Base.Checked.checked_sub(Base.Checked.checked_mul(2, Nr), 1),
        )
        return Nv, Nt
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("Requested radial mesh dimensions overflow Int: Nr=$Nr, Nphi=$Nphi"))
    end
end

@inline function _require_finite_coordinates(xyz::AbstractMatrix{<:Real},
                                             generator::AbstractString)
    all(isfinite, xyz) ||
        throw(ArgumentError("$generator produced non-finite coordinates; check the requested dimensions"))
    return nothing
end

"""
    make_rect_plate(Lx, Ly, Nx, Ny; resource_limits...)

Generate a triangulated rectangular plate in the xy-plane, centered at the
origin. Returns a `TriMesh` with `(Nx+1)*(Ny+1)` vertices and `2*Nx*Ny`
triangles.

Output is rejected before mesh-matrix allocation when its vertex count,
triangle count, or raw `xyz`/`tri` payload exceeds the configured limits.
"""
function make_rect_plate(Lx::Real, Ly::Real, Nx::Int, Ny::Int;
                         max_vertices::Integer=_DEFAULT_MESH_MAX_VERTICES,
                         max_triangles::Integer=_DEFAULT_MESH_MAX_TRIANGLES,
                         max_raw_bytes::Integer=_DEFAULT_MESH_MAX_RAW_BYTES)
    Lx_f = _positive_finite_length("Lx", Lx)
    Ly_f = _positive_finite_length("Ly", Ly)
    _positive_subdivision("Nx", Nx)
    _positive_subdivision("Ny", Ny)
    Nv, Nt = _rect_mesh_counts(Nx, Ny)
    _validate_mesh_resource_request(
        Nv, Nt, "make_rect_plate";
        max_vertices, max_triangles, max_raw_bytes)

    # Validate representable cell spacing before allocating the output.
    dx = Lx_f / Nx
    dy = Ly_f / Ny
    (dx > 0.0 && dy > 0.0) ||
        throw(ArgumentError("Plate dimensions are too small for the requested Float64 subdivisions"))
    if !_rectangular_triangle_area_is_representable(dx, dy)
        # Extremely large cells may have unrepresentable triangle areas while
        # still being useful to low-level periodic/RWG range tests. Reject the
        # opposite boundary: a triangle whose exact Float64 area rounds to 0.
        area_fraction_x, area_exponent_x = frexp(dx)
        area_fraction_y, area_exponent_y = frexp(dy)
        triangle_area = ldexp(
            0.5 * area_fraction_x * area_fraction_y,
            area_exponent_x + area_exponent_y)
        triangle_area == 0.0 &&
        !_rectangular_triangle_area_is_representable_big(dx, dy) &&
            throw(ArgumentError(
                "Plate triangle area underflows to zero for the requested " *
                "dimensions and subdivisions"))
    end
    half_x = Lx_f / 2
    half_y = Ly_f / 2
    half_x > 0.0 && half_y > 0.0 ||
        throw(ArgumentError(
            "Plate half-extents are not representable as positive Float64 values"))
    2 * half_x == Lx_f && 2 * half_y == Ly_f ||
        throw(ArgumentError(
            "Plate dimensions cannot be represented exactly by a " *
            "Float64 grid symmetric about the origin"))
    (-half_x + dx) > -half_x && (-half_y + dy) > -half_y ||
        throw(ArgumentError(
            "Plate cell spacing does not advance the first Float64 grid point"))

    xyz = zeros(3, Nv)
    tri = zeros(Int, 3, Nt)

    # Vertex grid
    @inline axis_coordinate(half_extent, cells, index) =
        index == cells ? half_extent :
        index == 0 ? -half_extent :
        half_extent * (2 * (index / cells) - 1)
    idx = 0
    for jy in 0:Ny
        for jx in 0:Nx
            idx += 1
            xyz[1, idx] = axis_coordinate(half_x, Nx, jx)
            xyz[2, idx] = axis_coordinate(half_y, Ny, jy)
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

    _require_finite_coordinates(xyz, "make_rect_plate")
    return TriMesh(xyz, tri)
end

@inline function _rectangular_triangle_area_is_representable(
        dx::Float64, dy::Float64)
    dx_fraction, dx_exponent = frexp(dx)
    dy_fraction, dy_exponent = frexp(dy)
    area = ldexp(
        0.5 * dx_fraction * dy_fraction,
        dx_exponent + dy_exponent)
    if area == 0.0
        return _rectangular_triangle_area_is_representable_big(dx, dy)
    end
    return isfinite(area) && area > 0.0
end

@noinline function _rectangular_triangle_area_is_representable_big(
        dx::Float64, dy::Float64)
    return setprecision(BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        area = Float64(BigFloat(dx) * BigFloat(dy) / 2)
        isfinite(area) && area > 0.0
    end
end

"""
    make_circular_plate(radius, Nr, Nphi; resource_limits...)

Generate a triangulated circular plate (disk) in the xy-plane, centered at
the origin. Uses radial rings with azimuthal subdivision.

Returns a `TriMesh` with approximately `Nr*Nphi + 1` vertices.
Output resource limits are checked before mesh-matrix allocation.
"""
function make_circular_plate(radius::Real, Nr::Int, Nphi::Int;
                             max_vertices::Integer=_DEFAULT_MESH_MAX_VERTICES,
                             max_triangles::Integer=_DEFAULT_MESH_MAX_TRIANGLES,
                             max_raw_bytes::Integer=_DEFAULT_MESH_MAX_RAW_BYTES)
    radius_f = _positive_finite_length("radius", radius)
    _positive_subdivision("Nr", Nr)
    _positive_subdivision("Nphi", Nphi; minimum=3)
    Nv, Nt_expected = _radial_mesh_counts(Nr, Nphi)
    _validate_mesh_resource_request(
        Nv, Nt_expected, "make_circular_plate";
        max_vertices, max_triangles, max_raw_bytes)
    dr = radius_f / Nr
    dr > 0.0 ||
        throw(ArgumentError("radius is too small for the requested Float64 radial subdivisions"))
    _radial_projected_area_is_representable(dr, Nphi) ||
        throw(ArgumentError(
            "Circular-plate innermost triangles do not have a representable " *
            "positive Float64 area for the requested radius and subdivisions"))

    # Vertices: center + Nr rings × Nphi points each
    verts = zeros(3, Nv)

    # Center vertex. Scalar stores avoid a temporary column vector.
    verts[1, 1] = 0.0
    verts[2, 1] = 0.0
    verts[3, 1] = 0.0

    # Ring vertices
    idx = 1
    for ir in 1:Nr
        r = radius_f * (ir / Nr)
        for ip in 1:Nphi
            phi = 2π * (ip - 1) / Nphi
            idx += 1
            verts[1, idx] = r * cos(phi)
            verts[2, idx] = r * sin(phi)
        end
    end

    # The topology count is known exactly. Allocate the returned connectivity
    # once instead of growing a flat vector and retaining its spare capacity.
    tri = Matrix{Int}(undef, 3, Nt_expected)
    tid = 0

    # Inner ring: triangles from center to first ring
    for ip in 1:Nphi
        v1 = 1   # center
        v2 = 1 + ip
        v3 = 1 + mod(ip, Nphi) + 1  # wraps: ip=Nphi -> next is 1+1=2
        # Fix: the next vertex in the ring
        v3 = ip < Nphi ? 1 + ip + 1 : 1 + 1
        tid += 1
        tri[1, tid] = v1
        tri[2, tid] = v2
        tri[3, tid] = v3
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
            tid += 1
            tri[1, tid] = v1
            tri[2, tid] = v2
            tri[3, tid] = v3
            tid += 1
            tri[1, tid] = v1
            tri[2, tid] = v3
            tri[3, tid] = v4
        end
    end
    tid == Nt_expected || error(
        "Internal circular-mesh topology count mismatch: wrote $tid " *
        "triangles, expected $Nt_expected")

    _require_finite_coordinates(verts, "make_circular_plate")
    return TriMesh(verts, tri)
end

@inline function _radial_projected_area_is_representable(
        first_radius::Float64, azimuthal_subdivisions::Int)
    origin = Vec3(0.0, 0.0, 0.0)
    first = Vec3(first_radius, 0.0, 0.0)
    angle = 2π / azimuthal_subdivisions
    second = Vec3(
        first_radius * cos(angle), first_radius * sin(angle), 0.0)
    fast, _, _, _, scale1_exponent, scale2_exponent, cross_norm =
        _scaled_triangle_cross(origin, first, second)
    if fast && cross_norm > 0.0
        area = _scaled_triangle_area(
            cross_norm, scale1_exponent, scale2_exponent)
        isfinite(area) && area > 0.0 && return true
    end
    return _triangle_area_big(origin, first, second) > 0.0
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
    tanh_g = tanh(grading_factor)
    half_L = L / 2
    use_linear_limit = grading_factor < sqrt(eps(Float64))
    for j in 0:N
        u = j / N                           # uniform parameter [0, 1]
        s = 2u - 1                          # map to [-1, 1]
        g = use_linear_limit ? s : tanh(grading_factor * s) / tanh_g
        coords[j + 1] = half_L * g          # physical coordinate
    end
    @inbounds for j in 2:length(coords)
        coords[j] > coords[j - 1] ||
            throw(ArgumentError("grading_factor=$grading_factor collapses adjacent Float64 mesh coordinates"))
    end
    return coords
end

"""
    make_rect_plate_graded(Lx, Ly, Nx, Ny;
                           grading_factor=3.0, resource_limits...)

Generate a triangulated rectangular plate in the xy-plane with graded mesh
density near the edges.  Same topology as `make_rect_plate` but vertex
positions are redistributed using a tanh grading function.

`grading_factor` controls edge clustering:
- `1.0`: nearly uniform
- `3.0` (default): ~5:1 edge-to-center density ratio
- `5.0`: ~10:1 ratio

Practical range: 1.0–5.0.  Values above 6 may create highly skewed
center elements.

Output resource limits are checked before coordinate-vector or mesh-matrix
allocation.
"""
function make_rect_plate_graded(Lx::Real, Ly::Real, Nx::Int, Ny::Int;
                                 grading_factor::Real=3.0,
                                 max_vertices::Integer=_DEFAULT_MESH_MAX_VERTICES,
                                 max_triangles::Integer=_DEFAULT_MESH_MAX_TRIANGLES,
                                 max_raw_bytes::Integer=_DEFAULT_MESH_MAX_RAW_BYTES)
    Lx_f = _positive_finite_length("Lx", Lx)
    Ly_f = _positive_finite_length("Ly", Ly)
    grading_factor_f = _positive_finite_length("grading_factor", grading_factor)
    _positive_subdivision("Nx", Nx)
    _positive_subdivision("Ny", Ny)
    Nv, Nt = _rect_mesh_counts(Nx, Ny)
    _validate_mesh_resource_request(
        Nv, Nt, "make_rect_plate_graded";
        max_vertices, max_triangles, max_raw_bytes)

    half_x = Lx_f / 2
    half_y = Ly_f / 2
    half_x > 0.0 && half_y > 0.0 &&
    2 * half_x == Lx_f && 2 * half_y == Ly_f ||
        throw(ArgumentError(
            "Graded-plate dimensions cannot be represented exactly by a " *
            "Float64 grid symmetric about the origin"))

    xs = _grade_1d(Nx, Lx_f, grading_factor_f)
    ys = _grade_1d(Ny, Ly_f, grading_factor_f)
    @inbounds for ix in 1:Nx, iy in 1:Ny
        dx = xs[ix + 1] - xs[ix]
        dy = ys[iy + 1] - ys[iy]
        _rectangular_triangle_area_is_representable(dx, dy) ||
            throw(ArgumentError(
                "Graded-plate triangle area is outside the positive " *
                "representable Float64 range near cell ($ix, $iy)"))
    end

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

    _require_finite_coordinates(xyz, "make_rect_plate_graded")
    return TriMesh(xyz, tri)
end

"""
    make_parabolic_reflector(D, f, Nr, Nphi;
                             center=Vec3(0,0,0), resource_limits...)

Generate a triangulated open parabolic reflector with aperture diameter `D`
and focal length `f`, aligned with +z:

`z = (x² + y²)/(4f)`, for `x² + y² ≤ (D/2)²`.

The mesh uses `Nr` radial rings and `Nphi` azimuth samples per ring.
Returns a `TriMesh` suitable for open-surface EFIE runs (`allow_boundary=true`).
Output resource limits are checked before mesh-matrix allocation.
"""
function make_parabolic_reflector(D::Real, f::Real, Nr::Int, Nphi::Int;
                                  center::Vec3=Vec3(0.0, 0.0, 0.0),
                                  max_vertices::Integer=_DEFAULT_MESH_MAX_VERTICES,
                                  max_triangles::Integer=_DEFAULT_MESH_MAX_TRIANGLES,
                                  max_raw_bytes::Integer=_DEFAULT_MESH_MAX_RAW_BYTES)
    D_f = _positive_finite_length("Reflector diameter D", D)
    f_f = _positive_finite_length("Reflector focal length f", f)
    _positive_subdivision("Nr", Nr; minimum=2)
    _positive_subdivision("Nphi", Nphi; minimum=3)
    all(isfinite, center) ||
        throw(ArgumentError("Reflector center must contain only finite coordinates, got $center"))
    Nv, Nt = _radial_mesh_counts(Nr, Nphi)
    _validate_mesh_resource_request(
        Nv, Nt, "make_parabolic_reflector";
        max_vertices, max_triangles, max_raw_bytes)

    R = D_f / 2
    R > 0.0 ||
        throw(ArgumentError(
            "Reflector diameter D is too small to represent a positive " *
            "Float64 aperture radius"))
    first_radius = R / Nr
    first_radius > 0.0 ||
        throw(ArgumentError(
            "Reflector radial spacing is not representable for Nr=$Nr"))
    _validate_parabolic_reflector_extent(
        center, R, D_f, f_f, Nr, Nphi)

    xyz = zeros(3, Nv)
    tri = zeros(Int, 3, Nt)

    # Vertex 1: apex. Scalar stores avoid a temporary column vector.
    xyz[1, 1] = center[1]
    xyz[2, 1] = center[2]
    xyz[3, 1] = center[3]

    @inline vid(ir, j) = 2 + (ir - 1) * Nphi + (j - 1)  # ir=1:Nr, j=1:Nphi
    @inline jnext(j) = (j == Nphi) ? 1 : (j + 1)

    # Ring vertices
    for ir in 1:Nr
        r = R * (ir / Nr)
        z = _parabolic_reflector_height(r, f_f)
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
        tri[1, tid] = 1
        tri[2, tid] = vid(1, j)
        tri[3, tid] = vid(1, jnext(j))
    end

    # Ring-to-ring quads split into 2 triangles
    for ir in 1:(Nr - 1)
        for j in 1:Nphi
            v00 = vid(ir, j)
            v01 = vid(ir, jnext(j))
            v10 = vid(ir + 1, j)
            v11 = vid(ir + 1, jnext(j))

            tid += 1
            tri[1, tid] = v00
            tri[2, tid] = v10
            tri[3, tid] = v11
            tid += 1
            tri[1, tid] = v00
            tri[2, tid] = v11
            tri[3, tid] = v01
        end
    end

    _require_finite_coordinates(xyz, "make_parabolic_reflector")
    return TriMesh(xyz, tri)
end

@inline function _parabolic_reflector_height(radius::Float64,
                                              focal_length::Float64)
    radius == 0.0 && return 0.0
    radius_fraction, radius_exponent = frexp(radius)
    focal_fraction, focal_exponent = frexp(focal_length)
    fraction = (radius_fraction * radius_fraction) / (4 * focal_fraction)
    height = ldexp(fraction, 2 * radius_exponent - focal_exponent)
    if height == 0.0 || !isfinite(height)
        return _parabolic_reflector_height_big(radius, focal_length)
    end
    isfinite(height) ||
        throw(OverflowError(
            "Parabolic-reflector height is outside the representable Float64 range"))
    height > 0.0 ||
        throw(ArgumentError(
            "Positive parabolic-reflector height underflows Float64 for " *
            "radius=$radius and focal_length=$focal_length"))
    return height
end

@noinline function _parabolic_reflector_height_big(
        radius::Float64, focal_length::Float64)
    return setprecision(BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        height = Float64(
            BigFloat(radius) * BigFloat(radius) / (4 * BigFloat(focal_length)))
        isfinite(height) ||
            throw(OverflowError(
                "Parabolic-reflector height is outside the representable " *
                "Float64 range"))
        height > 0.0 ||
            throw(ArgumentError(
                "Positive parabolic-reflector height underflows Float64 for " *
                "radius=$radius and focal_length=$focal_length"))
        return height
    end
end

@inline function _parabolic_reflector_point(
        center::Vec3, radius::Float64, height::Float64,
        sample::Int, azimuthal_subdivisions::Int)
    angle = 2π * (sample - 1) / azimuthal_subdivisions
    return Vec3(
        center[1] + radius * cos(angle),
        center[2] + radius * sin(angle),
        center[3] + height,
    )
end

@inline function _parabolic_triangle_area_representable(
        first::Vec3, second::Vec3, third::Vec3)
    all(isfinite, first) && all(isfinite, second) && all(isfinite, third) ||
        return 0.0
    fast, _, _, _, scale1_exponent, scale2_exponent, cross_norm =
        _scaled_triangle_cross(first, second, third)
    if fast
        cross_norm > 0.0 || return 0.0
        area = _scaled_triangle_area(
            cross_norm, scale1_exponent, scale2_exponent)
        if isfinite(area) && area > floatmin(Float64) &&
           area < 0.5 * floatmax(Float64)
            return area
        end
    end
    return _parabolic_triangle_area_big(first, second, third)
end

@noinline function _parabolic_triangle_area_big(
        first::Vec3, second::Vec3, third::Vec3)
    return setprecision(BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        e1x = BigFloat(second[1]) - BigFloat(first[1])
        e1y = BigFloat(second[2]) - BigFloat(first[2])
        e1z = BigFloat(second[3]) - BigFloat(first[3])
        e2x = BigFloat(third[1]) - BigFloat(first[1])
        e2y = BigFloat(third[2]) - BigFloat(first[2])
        e2z = BigFloat(third[3]) - BigFloat(first[3])
        cx = e1y * e2z - e1z * e2y
        cy = e1z * e2x - e1x * e2z
        cz = e1x * e2y - e1y * e2x
        cross_norm = hypot(hypot(cx, cy), cz)
        area = Float64(cross_norm / 2)
        return isfinite(area) && area > 0.0 ? area : 0.0
    end
end

@noinline function _validate_parabolic_reflector_extent(
        center::Vec3, radius::Float64, diameter::Float64,
        focal_length::Float64,
        radial_subdivisions::Int, azimuthal_subdivisions::Int)
    apex = center
    previous_ring = Vector{Vec3}(undef, azimuthal_subdivisions)
    current_ring = similar(previous_ring)
    xmin = xmax = center[1]
    ymin = ymax = center[2]
    zmin = zmax = center[3]
    minimum_area = Inf

    for ring in 1:radial_subdivisions
        radial_fraction = ring / radial_subdivisions
        ring_radius = ring == radial_subdivisions ? radius :
                      radius * radial_fraction
        ring_radius > 0.0 ||
            throw(ArgumentError(
                "Parabolic-reflector ring $ring radius is not representable"))
        ring_height = _parabolic_reflector_height(
            ring_radius, focal_length)
        for sample in 1:azimuthal_subdivisions
            point = _parabolic_reflector_point(
                center, ring_radius, ring_height,
                sample, azimuthal_subdivisions)
            all(isfinite, point) ||
                throw(OverflowError(
                    "Parabolic-reflector sampled coordinates are outside " *
                    "the representable Float64 range for D=$diameter, " *
                    "f=$focal_length"))
            point[3] > center[3] ||
                throw(ArgumentError(
                    "Parabolic-reflector ring $ring height is not " *
                    "representable relative to the requested center"))
            projected = (point[1], point[2])
            projected != (center[1], center[2]) ||
                throw(ArgumentError(
                    "Parabolic-reflector ring $ring sample $sample " *
                    "collapses onto the apex in Float64"))
            if ring > 1
                previous_projected = (
                    previous_ring[sample][1], previous_ring[sample][2])
                projected != previous_projected ||
                    throw(ArgumentError(
                        "Parabolic-reflector projected samples do not " *
                        "advance between rings $(ring - 1) and $ring"))
            end
            current_ring[sample] = point
            xmin = min(xmin, point[1]); xmax = max(xmax, point[1])
            ymin = min(ymin, point[2]); ymax = max(ymax, point[2])
            zmin = min(zmin, point[3]); zmax = max(zmax, point[3])
        end

        # Sample order follows one traversal of the ring. A repeated projected
        # point can be detected without an O(Nphi) hash table: any repetition
        # makes at least one consecutive polar angle fail to advance.
        previous_angle = -Inf
        for sample in 1:azimuthal_subdivisions
            point = current_ring[sample]
            angle = atan(point[2] - center[2], point[1] - center[1])
            angle < 0.0 && (angle += 2π)
            if sample > 1
                angle > previous_angle ||
                    throw(ArgumentError(
                        "Parabolic-reflector ring $ring projected samples " *
                        "are not unique and ordered in Float64"))
            end
            previous_angle = angle
        end

        current_scale = hypot(
            hypot(xmax - xmin, ymax - ymin), zmax - zmin)
        isfinite(current_scale) ||
            throw(ArgumentError(
                "Parabolic-reflector sampled extent is outside the " *
                "supported Float64 quality range"))
        current_tolerance = _area_tolerance(current_scale, 1.0e-12)
        minimum_area > current_tolerance ||
            throw(ArgumentError(
                "Parabolic-reflector sampled Float64 geometry contains a " *
                "triangle with area $minimum_area at or below the solver " *
                "quality tolerance $current_tolerance"))

        for sample in 1:azimuthal_subdivisions
            following = sample == azimuthal_subdivisions ? 1 : sample + 1
            if ring == 1
                area = _parabolic_triangle_area_representable(
                    apex, current_ring[sample], current_ring[following])
                area > 0.0 ||
                    throw(ArgumentError(
                        "Parabolic-reflector innermost triangle $sample " *
                        "does not have a representable positive Float64 area"))
                area > current_tolerance ||
                    throw(ArgumentError(
                        "Parabolic-reflector sampled Float64 geometry " *
                        "contains a triangle with area $area at or below " *
                        "the solver quality tolerance $current_tolerance"))
                minimum_area = min(minimum_area, area)
            else
                first_area = _parabolic_triangle_area_representable(
                    previous_ring[sample], current_ring[sample],
                    current_ring[following])
                first_area > 0.0 ||
                    throw(ArgumentError(
                        "Parabolic-reflector ring-cell triangle at ring " *
                        "$ring, sample $sample does not have a representable " *
                        "positive Float64 area"))
                first_area > current_tolerance ||
                    throw(ArgumentError(
                        "Parabolic-reflector sampled Float64 geometry " *
                        "contains a triangle with area $first_area at or " *
                        "below the solver quality tolerance " *
                        "$current_tolerance"))
                second_area = _parabolic_triangle_area_representable(
                    previous_ring[sample], current_ring[following],
                    previous_ring[following])
                second_area > 0.0 ||
                    throw(ArgumentError(
                        "Parabolic-reflector ring-cell triangle at ring " *
                        "$ring, sample $sample does not have a representable " *
                        "positive Float64 area"))
                second_area > current_tolerance ||
                    throw(ArgumentError(
                        "Parabolic-reflector sampled Float64 geometry " *
                        "contains a triangle with area $second_area at or " *
                        "below the solver quality tolerance " *
                        "$current_tolerance"))
                minimum_area = min(minimum_area, first_area, second_area)
            end
        end
        previous_ring, current_ring = current_ring, previous_ring
    end
    return nothing
end

function _mesh_coordinate_diagnostics(mesh::TriMesh)
    Nv = nvertices(mesh)
    finite_vertex = trues(Nv)
    invalid_vertices = Int[]

    @inbounds for i in 1:Nv
        x = mesh.xyz[1, i]
        y = mesh.xyz[2, i]
        z = mesh.xyz[3, i]
        if !(isfinite(x) && isfinite(y) && isfinite(z))
            finite_vertex[i] = false
            push!(invalid_vertices, i)
        end
    end

    xmin = Inf
    ymin = Inf
    zmin = Inf
    xmax = -Inf
    ymax = -Inf
    zmax = -Inf
    has_finite_reference = false

    # Area tolerances describe retained geometry, not orphan vertices or faces
    # that are unconditionally invalid/zero-area. Only distinct, in-range,
    # finite, positive-area faces contribute to the bounding box.
    @inbounds for t in 1:ntriangles(mesh)
        i1 = mesh.tri[1, t]
        i2 = mesh.tri[2, t]
        i3 = mesh.tri[3, t]
        valid_idx = (1 <= i1 <= Nv) && (1 <= i2 <= Nv) && (1 <= i3 <= Nv)
        distinct = (i1 != i2) && (i2 != i3) && (i3 != i1)
        valid_idx && distinct || continue
        finite_vertex[i1] && finite_vertex[i2] && finite_vertex[i3] ||
            continue
        area = triangle_area(mesh, t)
        isfinite(area) && area > 0.0 || continue
        for i in (i1, i2, i3)
            has_finite_reference = true
            x = mesh.xyz[1, i]
            y = mesh.xyz[2, i]
            z = mesh.xyz[3, i]
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            zmin = min(zmin, z)
            xmax = max(xmax, x)
            ymax = max(ymax, y)
            zmax = max(zmax, z)
        end
    end

    scale = has_finite_reference ?
            norm(Vec3(xmax - xmin, ymax - ymin, zmax - zmin)) : 0.0
    isfinite(scale) ||
        throw(ArgumentError("Mesh coordinate extent is too large for finite Float64 geometry"))
    return scale, finite_vertex, invalid_vertices
end

function _bbox_diagonal(mesh::TriMesh)
    size(mesh.xyz, 1) == 3 ||
        throw(DimensionMismatch("Mesh xyz must have size (3, Nv), got $(size(mesh.xyz))"))

    Nv = nvertices(mesh)
    Nv == 0 && return 0.0
    xmin = Inf
    ymin = Inf
    zmin = Inf
    xmax = -Inf
    ymax = -Inf
    zmax = -Inf
    @inbounds for i in 1:Nv
        x = mesh.xyz[1, i]
        y = mesh.xyz[2, i]
        z = mesh.xyz[3, i]
        (isfinite(x) && isfinite(y) && isfinite(z)) ||
            throw(ArgumentError("Mesh contains non-finite vertex coordinates at vertex $i"))
        xmin = min(xmin, x)
        ymin = min(ymin, y)
        zmin = min(zmin, z)
        xmax = max(xmax, x)
        ymax = max(ymax, y)
        zmax = max(zmax, z)
    end

    scale = norm(Vec3(xmax - xmin, ymax - ymin, zmax - zmin))
    isfinite(scale) ||
        throw(ArgumentError("Mesh coordinate extent is too large for finite Float64 geometry"))
    return scale
end

@inline function _area_tolerance(scale::Float64, area_tol_rel::Float64)
    (isfinite(area_tol_rel) && area_tol_rel >= 0.0) ||
        throw(ArgumentError("area_tol_rel must be finite and nonnegative, got $area_tol_rel"))
    area_tol_rel == 0.0 && return 0.0

    scaled = sqrt(area_tol_rel) * scale
    area_tol_abs = scaled * scaled
    isfinite(area_tol_abs) ||
        throw(ArgumentError("Mesh scale and area_tol_rel produce a non-finite area tolerance"))
    return area_tol_abs
end

@inline function _triangle_face_key(i1::Int, i2::Int, i3::Int)
    first, second = minmax(i1, i2)
    if i3 < first
        return (i3, first, second)
    elseif i3 > second
        return (first, second, i3)
    end
    return (first, i3, second)
end

"""
    mesh_quality_report(mesh; area_tol_rel=1e-12, check_orientation=true)

Compute mesh-quality diagnostics for a triangle surface mesh.
The report includes:
- non-finite vertex coordinates,
- invalid triangles (index out of bounds or repeated vertices),
- degenerate triangles (area below tolerance),
- duplicate triangles (the same three vertex indices in any winding),
- boundary-edge count,
- non-manifold-edge count (>2 incident triangles),
- orientation-conflict count on interior edges.
"""
function mesh_quality_report(mesh::TriMesh; area_tol_rel::Float64=1e-12, check_orientation::Bool=true)
    Nv = nvertices(mesh)
    Nt = ntriangles(mesh)

    size(mesh.xyz, 1) == 3 || error("Mesh xyz must have size (3, Nv)")
    size(mesh.tri, 1) == 3 || error("Mesh tri must have size (3, Nt)")

    scale, finite_vertex, invalid_vertices = _mesh_coordinate_diagnostics(mesh)
    area_tol_abs = _area_tolerance(scale, area_tol_rel)

    invalid_triangles = Int[]
    degenerate_triangles = Int[]
    duplicate_triangles = Int[]
    seen_faces = Set{NTuple{3,Int}}()

    # edge_map[(i,j)] = directions of edge traversal in each incident triangle
    # direction +1 means (i->j) where i<j, -1 means (j->i)
    edge_map = Dict{Tuple{Int,Int}, Vector{Int8}}()

    for t in 1:Nt
        i1 = mesh.tri[1, t]
        i2 = mesh.tri[2, t]
        i3 = mesh.tri[3, t]

        valid_idx = (1 <= i1 <= Nv) && (1 <= i2 <= Nv) && (1 <= i3 <= Nv)
        distinct = (i1 != i2) && (i2 != i3) && (i3 != i1)
        finite_coords = valid_idx &&
                        finite_vertex[i1] && finite_vertex[i2] && finite_vertex[i3]
        if !(valid_idx && distinct && finite_coords)
            push!(invalid_triangles, t)
            continue
        end

        face_key = _triangle_face_key(i1, i2, i3)
        is_duplicate = face_key in seen_faces
        if is_duplicate
            push!(duplicate_triangles, t)
        else
            push!(seen_faces, face_key)
        end

        area = triangle_area(mesh, t)
        if !isfinite(area)
            push!(invalid_triangles, t)
            continue
        elseif area <= area_tol_abs
            push!(degenerate_triangles, t)
        end

        # Count topology once per unoriented face. Including duplicates here
        # can make a coincident double surface look closed and manifold.
        is_duplicate && continue

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
        n_invalid_vertices = length(invalid_vertices),
        n_invalid_triangles = length(invalid_triangles),
        n_degenerate_triangles = length(degenerate_triangles),
        n_duplicate_triangles = length(duplicate_triangles),
        invalid_vertices = invalid_vertices,
        invalid_triangles = invalid_triangles,
        degenerate_triangles = degenerate_triangles,
        duplicate_triangles = duplicate_triangles,
        mesh_scale = scale,
        area_tol_abs = area_tol_abs,
    )
end

"""
    mesh_quality_ok(report; allow_boundary=true, require_closed=false)

Return `true` if a mesh-quality report passes hard checks:
- at least three vertices and one triangle,
- no non-finite vertices,
- no invalid triangles,
- no degenerate triangles,
- no duplicate triangles,
- no non-manifold edges,
- no orientation conflicts,
- boundary edges allowed unless `allow_boundary=false` or `require_closed=true`.
"""
function mesh_quality_ok(report; allow_boundary::Bool=true, require_closed::Bool=false)
    if report.n_vertices < 3 || report.n_triangles < 1
        return false
    end
    if report.n_invalid_vertices > 0
        return false
    end
    if report.n_invalid_triangles > 0
        return false
    end
    if report.n_degenerate_triangles > 0
        return false
    end
    if report.n_duplicate_triangles > 0
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
    return _assert_mesh_quality_report(
        report;
        allow_boundary=allow_boundary,
        require_closed=require_closed,
    )
end

function _assert_mesh_quality_report(report;
                                     allow_boundary::Bool=true,
                                     require_closed::Bool=false)
    problems = String[]

    if report.n_vertices < 3 || report.n_triangles < 1
        push!(problems, "mesh must contain at least 3 vertices and 1 triangle (got $(report.n_vertices) vertices, $(report.n_triangles) triangles)")
    end
    if report.n_invalid_vertices > 0
        sample = join(report.invalid_vertices[1:min(end, 5)], ", ")
        push!(problems, "vertices with non-finite coordinates: $(report.n_invalid_vertices) (sample: $sample)")
    end
    if report.n_invalid_triangles > 0
        sample = join(report.invalid_triangles[1:min(end, 5)], ", ")
        push!(problems, "invalid triangles: $(report.n_invalid_triangles) (sample: $sample)")
    end
    if report.n_degenerate_triangles > 0
        sample = join(report.degenerate_triangles[1:min(end, 5)], ", ")
        push!(problems, "degenerate triangles: $(report.n_degenerate_triangles) (sample: $sample), area_tol_abs=$(report.area_tol_abs)")
    end
    if report.n_duplicate_triangles > 0
        sample = join(report.duplicate_triangles[1:min(end, 5)], ", ")
        push!(problems,
              "duplicate triangles: $(report.n_duplicate_triangles) " *
              "(later coincident face indices; sample: $sample)")
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

The file is scanned once to determine the exact matrix sizes and a second
time to fill them, avoiding per-field split vectors and intermediate
vertex/face collections.
"""
@inline function _obj_field_bounds(line::AbstractString, position::Int)
    last = lastindex(line)
    while position <= last && isspace(line[position])
        position = nextind(line, position)
    end
    position > last && return (0, 0, position)

    first = position
    while position <= last && !isspace(line[position])
        position = nextind(line, position)
    end
    return (first, prevind(line, position), position)
end

@inline function _obj_record_is(line::AbstractString, first::Int, last::Int,
                                record::Char)
    return first == last && line[first] == record
end

@inline function _required_obj_field(line::AbstractString, position::Int,
                                     path::AbstractString, line_number::Int,
                                     context::AbstractString)
    first, last, next_position = _obj_field_bounds(line, position)
    if iszero(first) || line[first] == '#'
        error("Invalid OBJ $context at $path:$line_number: $line")
    end
    return SubString(line, first, last), next_position
end

function _count_obj_mesh(
        io::IO, path::AbstractString;
        max_vertices::Integer=_DEFAULT_MESH_MAX_VERTICES,
        max_triangles::Integer=_DEFAULT_MESH_MAX_TRIANGLES,
        max_raw_bytes::Integer=_DEFAULT_MESH_MAX_RAW_BYTES)
    vertex_limit = _validated_mesh_resource_limit("max_vertices", max_vertices)
    triangle_limit = _validated_mesh_resource_limit("max_triangles", max_triangles)
    byte_limit = _validated_mesh_resource_limit("max_raw_bytes", max_raw_bytes)
    n_vertices = 0
    n_triangles = 0

    for (line_number, line) in enumerate(eachline(io))
        record_first, record_last, position =
            _obj_field_bounds(line, firstindex(line))
        iszero(record_first) && continue
        line[record_first] == '#' && continue

        if _obj_record_is(line, record_first, record_last, 'v')
            x_field, next_position = _required_obj_field(
                line, position, path, line_number, "vertex")
            y_field, next_position = _required_obj_field(
                line, next_position, path, line_number, "vertex")
            z_field, next_position = _required_obj_field(
                line, next_position, path, line_number, "vertex")
            coord = (
                parse(Float64, x_field),
                parse(Float64, y_field),
                parse(Float64, z_field),
            )
            isfinite(coord[1]) && isfinite(coord[2]) && isfinite(coord[3]) ||
                error(
                    "OBJ vertex coordinates must be finite at " *
                    "$path:$line_number: $line")
            n_fields = 3
            position = next_position
            while true
                first, _, next_position = _obj_field_bounds(line, position)
                iszero(first) && break
                line[first] == '#' && break
                n_fields = Base.checked_add(n_fields, 1)
                position = next_position
            end
            n_fields >= 3 ||
                error("Invalid OBJ vertex at $path:$line_number: $line")
            n_vertices = Base.checked_add(n_vertices, 1)
            n_vertices <= vertex_limit ||
                throw(ArgumentError(
                    "OBJ mesh exceeds max_vertices=$vertex_limit at " *
                    "$path:$line_number"))
            _mesh_raw_payload_bytes(n_vertices, n_triangles) <= byte_limit ||
                throw(ArgumentError(
                    "OBJ mesh exceeds max_raw_bytes=$byte_limit at " *
                    "$path:$line_number"))
        elseif _obj_record_is(line, record_first, record_last, 'f')
            n_face_vertices = 0
            while true
                first, last, next_position = _obj_field_bounds(line, position)
                iszero(first) && break
                line[first] == '#' && break
                _parse_obj_vertex_index(
                    line, first, last, n_vertices, path, line_number)
                n_face_vertices = Base.checked_add(n_face_vertices, 1)
                position = next_position
            end
            n_face_vertices >= 3 ||
                error("Invalid OBJ face at $path:$line_number: $line")
            n_triangles = Base.checked_add(n_triangles, n_face_vertices - 2)
            n_triangles <= triangle_limit ||
                throw(ArgumentError(
                    "OBJ mesh exceeds max_triangles=$triangle_limit at " *
                    "$path:$line_number"))
            _mesh_raw_payload_bytes(n_vertices, n_triangles) <= byte_limit ||
                throw(ArgumentError(
                    "OBJ mesh exceeds max_raw_bytes=$byte_limit at " *
                    "$path:$line_number"))
        end
    end

    return n_vertices, n_triangles
end

@inline function _parse_obj_vertex_index(line::AbstractString,
                                         first::Int, last::Int,
                                         n_vertices::Int,
                                         path::AbstractString,
                                         line_number::Int)
    index_last = last
    position = first
    while position <= last
        if line[position] == '/'
            index_last = prevind(line, position)
            break
        end
        position = nextind(line, position)
    end
    index_last >= first ||
        error("Invalid OBJ face token at $path:$line_number: $line")

    raw_index = parse(Int, SubString(line, first, index_last))
    raw_index != 0 ||
        error("OBJ vertex index zero is invalid at $path:$line_number: $line")
    index = if raw_index > 0
        raw_index
    else
        Base.checked_add(Base.checked_add(n_vertices, raw_index), 1)
    end
    1 <= index <= n_vertices ||
        error("OBJ face index out of range at $path:$line_number: $line")
    return index
end

function _fill_obj_mesh!(io::IO, path::AbstractString,
                         xyz::Matrix{Float64}, tri::Matrix{Int})
    vertices_written = 0
    triangles_written = 0

    for (line_number, line) in enumerate(eachline(io))
        record_first, record_last, position =
            _obj_field_bounds(line, firstindex(line))
        iszero(record_first) && continue
        line[record_first] == '#' && continue

        if _obj_record_is(line, record_first, record_last, 'v')
            x_field, position = _required_obj_field(
                line, position, path, line_number, "vertex")
            y_field, position = _required_obj_field(
                line, position, path, line_number, "vertex")
            z_field, _ = _required_obj_field(
                line, position, path, line_number, "vertex")
            coord = (
                parse(Float64, x_field),
                parse(Float64, y_field),
                parse(Float64, z_field),
            )
            isfinite(coord[1]) && isfinite(coord[2]) && isfinite(coord[3]) ||
                error("OBJ vertex coordinates must be finite at $path:$line_number: $line")

            vertices_written < size(xyz, 2) ||
                error("OBJ file changed while reading vertices: $path")
            vertices_written += 1
            @inbounds begin
                xyz[1, vertices_written] = coord[1]
                xyz[2, vertices_written] = coord[2]
                xyz[3, vertices_written] = coord[3]
            end
        elseif _obj_record_is(line, record_first, record_last, 'f')
            n_face_vertices = 0
            first_vertex = 0
            previous_vertex = 0

            while true
                first, last, next_position = _obj_field_bounds(line, position)
                iszero(first) && break
                line[first] == '#' && break
                vertex = _parse_obj_vertex_index(
                    line, first, last, vertices_written, path, line_number)
                n_face_vertices += 1

                if n_face_vertices == 1
                    first_vertex = vertex
                elseif n_face_vertices == 2
                    previous_vertex = vertex
                else
                    triangles_written < size(tri, 2) ||
                        error("OBJ file changed while reading faces: $path")
                    triangles_written += 1
                    @inbounds begin
                        tri[1, triangles_written] = first_vertex
                        tri[2, triangles_written] = previous_vertex
                        tri[3, triangles_written] = vertex
                    end
                    previous_vertex = vertex
                end
                position = next_position
            end

            n_face_vertices >= 3 ||
                error("Invalid OBJ face at $path:$line_number: $line")
        end
    end

    vertices_written == size(xyz, 2) ||
        error("OBJ file changed while reading vertices: $path")
    triangles_written == size(tri, 2) ||
        error("OBJ file changed while reading faces: $path")
    return nothing
end

"""
    read_obj_mesh(path; resource_limits...)

Read a Wavefront OBJ mesh. Polygon faces are fan-triangulated. Vertex,
triangle, raw output-payload, input-file, and text-line limits are enforced
before the corresponding large allocation or line parse.
"""
function read_obj_mesh(path::AbstractString;
                       max_vertices::Integer=_DEFAULT_MESH_MAX_VERTICES,
                       max_triangles::Integer=_DEFAULT_MESH_MAX_TRIANGLES,
                       max_raw_bytes::Integer=_DEFAULT_MESH_MAX_RAW_BYTES,
                       max_input_bytes::Integer=_DEFAULT_MESH_MAX_INPUT_BYTES,
                       max_line_bytes::Integer=_DEFAULT_MESH_MAX_LINE_BYTES)
    return open(path, "r") do io
        _mesh_input_size(io, path; max_input_bytes)
        reader = _BoundedMeshTextIO(
            io, path; max_input_bytes, max_line_bytes)
        n_vertices, n_triangles = _count_obj_mesh(
            reader, path; max_vertices, max_triangles, max_raw_bytes)
        n_vertices > 0 || error("OBJ mesh has no vertices: $path")
        n_triangles > 0 || error("OBJ mesh has no faces: $path")
        _validate_mesh_resource_request(
            n_vertices, n_triangles, "OBJ mesh $path";
            max_vertices, max_triangles, max_raw_bytes)

        xyz = Matrix{Float64}(undef, 3, n_vertices)
        tri = Matrix{Int}(undef, 3, n_triangles)
        seekstart(reader)
        _fill_obj_mesh!(reader, path, xyz, tri)
        return TriMesh(xyz, tri)
    end
end

"""
    write_obj_mesh(path, mesh; header="...")

Write a `TriMesh` to a Wavefront OBJ file using triangle faces.
"""
function _validate_text_mesh_header(header::AbstractString, format::AbstractString)
    if occursin('\n', header) || occursin('\r', header)
        throw(ArgumentError("$format header must be a single line."))
    end
    return nothing
end

function _validate_obj_mesh_for_write(mesh::TriMesh)
    size(mesh.xyz, 1) == 3 ||
        throw(DimensionMismatch(
            "OBJ vertex coordinates must have size (3, Nv), got $(size(mesh.xyz))."))
    size(mesh.tri, 1) == 3 ||
        throw(DimensionMismatch(
            "OBJ triangle connectivity must have size (3, Nt), got $(size(mesh.tri))."))

    nv = nvertices(mesh)
    nt = ntriangles(mesh)
    nv > 0 || throw(ArgumentError("Cannot write an OBJ mesh with 0 vertices."))
    nt > 0 || throw(ArgumentError("Cannot write an OBJ mesh with 0 triangles."))

    @inbounds for vertex in 1:nv
        x = mesh.xyz[1, vertex]
        y = mesh.xyz[2, vertex]
        z = mesh.xyz[3, vertex]
        (isfinite(x) && isfinite(y) && isfinite(z)) ||
            throw(ArgumentError(
                "OBJ vertex $vertex has non-finite coordinates: ($x, $y, $z)."))
    end
    @inbounds for triangle in 1:nt
        for local_vertex in 1:3
            vertex = mesh.tri[local_vertex, triangle]
            1 <= vertex <= nv ||
                throw(ArgumentError(
                    "OBJ triangle $triangle references vertex $vertex outside 1:$nv."))
        end
    end
    return nothing
end

function write_obj_mesh(path::AbstractString, mesh::TriMesh; header::AbstractString="Exported by DiffMoM")
    _validate_text_mesh_header(header, "OBJ")
    _validate_obj_mesh_for_write(mesh)
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
function estimate_dense_matrix_gib(N::Integer)
    N >= 0 || throw(ArgumentError("Matrix dimension N must be nonnegative, got $N"))
    return 16.0 * float(N) * float(N) / 1024.0^3
end

@inline function _sorted_triangle_vertices(a::Int, b::Int, c::Int)
    if a > b
        a, b = b, a
    end
    if b > c
        b, c = c, b
    end
    if a > b
        a, b = b, a
    end
    return (a, b, c)
end

const _DEFAULT_CLUSTER_MAX_EXACT_CELL_INDICES = 10_000
const _DEFAULT_CLUSTER_MAX_EXACT_AREA_CHECKS = 1_000
const _CLUSTER_FLOAT_EXPONENT_MIN = -1074
const _CLUSTER_FLOAT_EXPONENT_MAX = 971
const _CLUSTER_FLOAT_EXPONENT_COUNT =
    _CLUSTER_FLOAT_EXPONENT_MAX - _CLUSTER_FLOAT_EXPONENT_MIN + 1

@noinline function _cluster_cell_index_exact(value::Float64,
                                              origin::Float64,
                                              h::Float64)
    value == origin && return BigInt(0)
    value_significand, value_exponent, value_sign = Base.decompose(value)
    origin_significand, origin_exponent, origin_sign = Base.decompose(origin)
    common_exponent = if iszero(value)
        origin_exponent
    elseif iszero(origin)
        value_exponent
    else
        min(value_exponent, origin_exponent)
    end
    numerator = BigInt(value_sign) * BigInt(value_significand)
    numerator <<= value_exponent - common_exponent
    origin_integer = BigInt(origin_sign) * BigInt(origin_significand)
    origin_integer <<= origin_exponent - common_exponent
    numerator -= origin_integer
    numerator >= 0 ||
        throw(ArgumentError(
            "cluster_mesh_vertices: value=$value is below origin=$origin"))

    h_significand, h_exponent, _ = Base.decompose(h)
    if h_exponent >= common_exponent
        denominator = BigInt(h_significand)
        denominator <<= h_exponent - common_exponent
        return fld(numerator, denominator)
    end
    numerator <<= common_exponent - h_exponent
    return fld(numerator, BigInt(h_significand))
end

@inline function _cluster_cell_index(value::Float64,
                                     origin::Float64,
                                     h::Float64,
                                     exact_indices::Base.RefValue{Int},
                                     exact_index_limit::Int)
    (isfinite(value) && isfinite(origin)) ||
        throw(ArgumentError(
            "cluster_mesh_vertices: vertex coordinates must be finite"))
    value == origin && return 0
    delta = value - origin
    if isfinite(delta) && delta >= 0.0
        scaled = delta / h
        if isfinite(scaled) && 0.0 <= scaled < Float64(typemax(Int))
            subtraction_error = _two_difference_error(
                value, origin, delta)
            scaled_error = subtraction_error / h
            uncertainty = if isfinite(scaled_error)
                abs(scaled_error) +
                8 * eps(Float64) * max(1.0, abs(scaled))
            else
                Inf
            end
            distance_to_boundary = abs(scaled - round(scaled))
            if distance_to_boundary > uncertainty
                return floor(Int, scaled)
            end
        end
    end

    exact_indices[] < exact_index_limit ||
        throw(ArgumentError(
            "cluster_mesh_vertices exceeded max_exact_cell_indices=" *
            "$exact_index_limit; too many distinct coordinates require " *
            "exact voxel classification"))
    exact_indices[] += 1
    exact_index = _cluster_cell_index_exact(value, origin, h)
    return exact_index <= typemax(Int) ? Int(exact_index) : exact_index
end

@inline function _cluster_cell_index_cached(
        value::Float64, origin::Float64, h::Float64,
        cache::Dict{UInt64,Union{Int,BigInt}},
        exact_indices::Base.RefValue{Int}, exact_index_limit::Int)
    key = reinterpret(UInt64, iszero(value) ? 0.0 : value)
    haskey(cache, key) && return cache[key]
    index = _cluster_cell_index(
        value, origin, h, exact_indices, exact_index_limit)
    cache[key] = index
    return index
end

@noinline function _cluster_exact_mean(
        xyz::Matrix{Float64}, component::Int, first_vertex::Int,
        next_vertex::Vector{Int}, count::Int,
        exponent_bins::Vector{Int128})
    fill!(exponent_bins, Int128(0))
    vertex = first_vertex
    while !iszero(vertex)
        value = xyz[component, vertex]
        if !iszero(value)
            significand, exponent, sign = Base.decompose(value)
            index = exponent - _CLUSTER_FLOAT_EXPONENT_MIN + 1
            exponent_bins[index] += Int128(sign) * Int128(significand)
        end
        vertex = next_vertex[vertex]
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

    # Collapse a narrow dyadic span exactly in Int128. Ordinary voxel clusters
    # share only a few exponents, so this keeps merged centers allocation-free.
    occupied_span = last_nonzero - first_nonzero
    if occupied_span <= 72
        exact_integer = Int128(exponent_bins[last_nonzero])
        overflowed = false
        @inbounds for index in (last_nonzero - 1):-1:first_nonzero
            if exact_integer > (typemax(Int128) >> 1) ||
               exact_integer < (typemin(Int128) >> 1)
                overflowed = true
                break
            end
            exact_integer <<= 1
            addend = exponent_bins[index]
            if (addend > 0 && exact_integer > typemax(Int128) - addend) ||
               (addend < 0 && exact_integer < typemin(Int128) - addend)
                overflowed = true
                break
            end
            exact_integer += addend
        end
        if !overflowed
            exact_exponent = _CLUSTER_FLOAT_EXPONENT_MIN + first_nonzero - 1
            divisor = Int128(count)
            common_factor = gcd(abs(exact_integer), divisor)
            reduced_integer = exact_integer ÷ common_factor
            reduced_divisor = divisor ÷ common_factor
            result = if ispow2(reduced_divisor)
                ldexp(
                    Float64(reduced_integer),
                    exact_exponent - trailing_zeros(reduced_divisor))
            elseif abs(exact_integer) <= Int128(1) << 53
                ldexp(
                    Float64(exact_integer) / count,
                    exact_exponent)
            else
                NaN
            end
            # Subnormal rounding can depend on bits discarded by the first
            # conversion. Settle those boundary cases in the exact path below.
            isfinite(result) && abs(result) >= floatmin(Float64) &&
                return result
        end
    end

    exact_significand = BigInt(exponent_bins[last_nonzero])
    @inbounds for index in (last_nonzero - 1):-1:first_nonzero
        exact_significand <<= 1
        exact_significand += exponent_bins[index]
    end
    exact_exponent = _CLUSTER_FLOAT_EXPONENT_MIN + first_nonzero - 1
    return setprecision(BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        result = Float64(ldexp(
            BigFloat(exact_significand) / BigFloat(count), exact_exponent))
        isfinite(result) ||
            throw(OverflowError(
                "cluster_mesh_vertices: cluster centroid is outside the " *
                "representable Float64 range"))
        result
    end
end

function _cluster_exact_centers(
        mesh::TriMesh, first_vertices::Vector{Int},
        next_vertex::Vector{Int}, counts::Vector{Int})
    nclusters = length(first_vertices)
    xyz = Matrix{Float64}(undef, 3, nclusters)
    exponent_bins = zeros(Int128, _CLUSTER_FLOAT_EXPONENT_COUNT)
    @inbounds for cluster in 1:nclusters
        first_vertex = first_vertices[cluster]
        count = counts[cluster]
        if count == 1
            xyz[1, cluster] = mesh.xyz[1, first_vertex]
            xyz[2, cluster] = mesh.xyz[2, first_vertex]
            xyz[3, cluster] = mesh.xyz[3, first_vertex]
            continue
        end
        for component in 1:3
            xyz[component, cluster] = _cluster_exact_mean(
                mesh.xyz, component, first_vertex, next_vertex, count,
                exponent_bins)
        end
    end
    return xyz
end

"""
    cluster_mesh_vertices(mesh, h; max_exact_cell_indices=10_000,
                          max_exact_area_checks=1_000)

Voxel-cluster a mesh using cubic cell size `h`, replacing all vertices in each
cell by their centroid and remapping triangles. Degenerate and duplicate
triangles created by remapping are removed.
"""
function cluster_mesh_vertices(
        mesh::TriMesh, h::Float64;
        max_exact_cell_indices::Integer=
            _DEFAULT_CLUSTER_MAX_EXACT_CELL_INDICES,
        max_exact_area_checks::Integer=
            _DEFAULT_CLUSTER_MAX_EXACT_AREA_CHECKS)
    (isfinite(h) && h > 0.0) ||
        throw(ArgumentError("cluster_mesh_vertices: h must be finite and positive, got $h"))
    exact_index_limit = _validated_mesh_resource_limit(
        "max_exact_cell_indices", max_exact_cell_indices)
    exact_area_limit = _validated_mesh_resource_limit(
        "max_exact_area_checks", max_exact_area_checks)

    size(mesh.xyz, 1) == 3 ||
        throw(DimensionMismatch(
            "cluster_mesh_vertices requires a 3×Nv coordinate matrix"))
    size(mesh.tri, 1) == 3 ||
        throw(DimensionMismatch(
            "cluster_mesh_vertices requires a 3×Nt connectivity matrix"))
    all(isfinite, mesh.xyz) ||
        throw(ArgumentError(
            "cluster_mesh_vertices: vertex coordinates must be finite"))

    nv = nvertices(mesh)
    nv >= 1 ||
        throw(ArgumentError(
            "cluster_mesh_vertices requires at least one vertex"))
    @inbounds for vertex in mesh.tri
        1 <= vertex <= nv ||
            throw(ArgumentError(
                "cluster_mesh_vertices: vertex index $vertex is outside 1:$nv"))
    end
    mins = (
        minimum(@view mesh.xyz[1, :]),
        minimum(@view mesh.xyz[2, :]),
        minimum(@view mesh.xyz[3, :]),
    )

    small_key_to_id = Dict{NTuple{3,Int},Int}()
    large_key_to_id = Dict{NTuple{3,BigInt},Int}()
    x_index_cache = Dict{UInt64,Union{Int,BigInt}}()
    y_index_cache = Dict{UInt64,Union{Int,BigInt}}()
    z_index_cache = Dict{UInt64,Union{Int,BigInt}}()
    exact_indices = Ref(0)
    vmap = Vector{Int}(undef, nv)
    first_vertices = Int[]
    cluster_counts = Int[]
    next_vertex = zeros(Int, nv)

    for i in 1:nv
        x = mesh.xyz[1, i]
        y = mesh.xyz[2, i]
        z = mesh.xyz[3, i]
        ix = _cluster_cell_index_cached(
            x, mins[1], h, x_index_cache, exact_indices, exact_index_limit)
        iy = _cluster_cell_index_cached(
            y, mins[2], h, y_index_cache, exact_indices, exact_index_limit)
        iz = _cluster_cell_index_cached(
            z, mins[3], h, z_index_cache, exact_indices, exact_index_limit)
        small_key = ix isa Int && iy isa Int && iz isa Int
        id = if small_key
            get(small_key_to_id, (ix::Int, iy::Int, iz::Int), 0)
        else
            large_key = (
                ix isa BigInt ? ix : BigInt(ix),
                iy isa BigInt ? iy : BigInt(iy),
                iz isa BigInt ? iz : BigInt(iz),
            )
            get(large_key_to_id, large_key, 0)
        end
        if iszero(id)
            push!(first_vertices, i)
            push!(cluster_counts, 1)
            id = length(first_vertices)
            if small_key
                small_key_to_id[(ix::Int, iy::Int, iz::Int)] = id
            else
                large_key = (
                    ix isa BigInt ? ix : BigInt(ix),
                    iy isa BigInt ? iy : BigInt(iy),
                    iz isa BigInt ? iz : BigInt(iz),
                )
                large_key_to_id[large_key] = id
            end
        else
            cluster_counts[id] = Base.checked_add(cluster_counts[id], 1)
            next_vertex[i] = first_vertices[id]
            first_vertices[id] = i
        end
        vmap[i] = id
    end

    xyz_new = _cluster_exact_centers(
        mesh, first_vertices, next_vertex, cluster_counts)

    tri_vec = Int[]
    seen = Set{NTuple{3,Int}}()
    exact_area_checks = 0
    for t in 1:ntriangles(mesh)
        a = vmap[mesh.tri[1, t]]
        b = vmap[mesh.tri[2, t]]
        c = vmap[mesh.tri[3, t]]
        if a == b || b == c || c == a
            continue
        end
        key = _sorted_triangle_vertices(a, b, c)
        key in seen && continue
        push!(seen, key)
        first = Vec3(xyz_new[1, a], xyz_new[2, a], xyz_new[3, a])
        second = Vec3(xyz_new[1, b], xyz_new[2, b], xyz_new[3, b])
        third = Vec3(xyz_new[1, c], xyz_new[2, c], xyz_new[3, c])
        fast, _, _, _, first_exponent, second_exponent, cross_norm =
            _scaled_triangle_cross(first, second, third)
        has_positive_area = if fast && cross_norm > 0.0
            area = _scaled_triangle_area(
                cross_norm, first_exponent, second_exponent)
            if area > 0.0
                true
            else
                exact_area_checks += 1
                exact_area_checks <= exact_area_limit ||
                    throw(ArgumentError(
                        "cluster_mesh_vertices exceeded " *
                        "max_exact_area_checks=$exact_area_limit"))
                _triangle_area_rounds_positive_big(first, second, third)
            end
        elseif fast
            false
        else
            exact_area_checks += 1
            exact_area_checks <= exact_area_limit ||
                throw(ArgumentError(
                    "cluster_mesh_vertices exceeded " *
                    "max_exact_area_checks=$exact_area_limit"))
            _triangle_area_rounds_positive_big(first, second, third)
        end
        has_positive_area || continue
        push!(tri_vec, a, b, c)
    end

    isempty(tri_vec) &&
        throw(ArgumentError(
            "cluster_mesh_vertices: clustering removed all triangles"))
    tri_new = reshape(tri_vec, 3, :)
    return TriMesh(xyz_new, tri_new)
end

"""
    drop_nonmanifold_triangles(mesh; max_passes=8)

Remove all triangles attached to non-manifold edges (edges with more than two
incident triangles) in one simultaneous pass. Returns a mesh with only
manifold/boundary edges. `max_passes` is retained for API compatibility.
"""
function drop_nonmanifold_triangles(mesh::TriMesh; max_passes::Int=8)
    max_passes >= 1 ||
        throw(ArgumentError("drop_nonmanifold_triangles: max_passes must be at least 1, got $max_passes"))
    size(mesh.xyz, 1) == 3 ||
        throw(DimensionMismatch(
            "drop_nonmanifold_triangles requires a 3×Nv coordinate matrix"))
    size(mesh.tri, 1) == 3 ||
        throw(DimensionMismatch(
            "drop_nonmanifold_triangles requires a 3×Nt connectivity matrix"))

    nt = ntriangles(mesh)
    edge_counts = Dict{Tuple{Int,Int},UInt8}()
    @inbounds for triangle in 1:nt
        i1 = mesh.tri[1, triangle]
        i2 = mesh.tri[2, triangle]
        i3 = mesh.tri[3, triangle]
        for (first, second) in ((i1, i2), (i2, i3), (i3, i1))
            edge = first < second ? (first, second) : (second, first)
            count = get(edge_counts, edge, UInt8(0))
            edge_counts[edge] = min(UInt8(3), count + UInt8(1))
        end
    end

    keep = trues(nt)
    @inbounds for triangle in 1:nt
        i1 = mesh.tri[1, triangle]
        i2 = mesh.tri[2, triangle]
        i3 = mesh.tri[3, triangle]
        for (first, second) in ((i1, i2), (i2, i3), (i3, i1))
            edge = first < second ? (first, second) : (second, first)
            if edge_counts[edge] == UInt8(3)
                keep[triangle] = false
                break
            end
        end
    end
    tri_new = copy(mesh.tri[:, keep])
    isempty(tri_new) && error("drop_nonmanifold_triangles: empty mesh after cleanup.")
    return TriMesh(copy(mesh.xyz), tri_new)
end

"""
    coarsen_mesh_to_target_rwg(mesh, target_rwg; kwargs...)

Auto-coarsen a mesh by voxel clustering to approach a target RWG count.
Each candidate mesh is non-manifold cleaned and repaired before RWG counting.
An input mesh already no more than 15% above the target is returned unchanged.

Returns a named tuple:
`(mesh, rwg_count, target_rwg, best_gap, iterations)`.
"""
function coarsen_mesh_to_target_rwg(mesh::TriMesh, target_rwg::Int;
                                    max_iters::Int=10,
                                    allow_boundary::Bool=true,
                                    require_closed::Bool=false,
                                    area_tol_rel::Float64=1e-12,
                                    strict_nonmanifold::Bool=true)
    target_rwg > 0 ||
        throw(ArgumentError("coarsen_mesh_to_target_rwg: target_rwg must be positive, got $target_rwg"))
    max_iters >= 1 ||
        throw(ArgumentError("coarsen_mesh_to_target_rwg: max_iters must be at least 1, got $max_iters"))

    normalized_input = _normalize_mesh_for_coarsening(mesh)
    _assert_coarsening_mesh_quality(
        mesh, normalized_input;
        allow_boundary=allow_boundary,
        require_closed=require_closed,
        area_tol_rel=area_tol_rel,
    )
    _validate_coarsening_candidate_area_range(mesh)
    best_rwg = _count_rwg_edges(mesh)
    best_mesh = mesh
    best_gap = abs(best_rwg - target_rwg)
    niter = 0

    initial_ratio = best_rwg / target_rwg
    if initial_ratio <= 1.15
        return (mesh=best_mesh, rwg_count=best_rwg, target_rwg=target_rwg,
                best_gap=best_gap, iterations=niter)
    end

    working_mesh, _, _ = _compact_mesh_vertices(
        mesh.xyz, copy(mesh.tri))
    target_vertices = max(80, Int(round(target_rwg / 3)))
    surface_area = _mesh_surface_area_for_coarsening(working_mesh)
    h = _coarsening_initial_cell_size(surface_area, target_vertices)
    invalid_upper = Inf
    valid_lower = 0.0

    for iter in 1:max_iters
        candidate = try
            cand = cluster_mesh_vertices(working_mesh, h)
            cand = drop_nonmanifold_triangles(cand)
            normalized_cand = _normalize_mesh_for_coarsening(cand)
            normalized_scale =
                _coarsening_referenced_bbox_scale(normalized_cand)
            normalized_tolerance = normalized_scale === nothing ? Inf :
                _area_tolerance(normalized_scale, area_tol_rel)
            physical_scale = _coarsening_referenced_bbox_scale(cand)
            physical_tolerance = physical_scale === nothing ? Inf :
                _area_tolerance(physical_scale, area_tol_rel)
            physical_rounding_can_change = normalized_scale === nothing ||
                _coarsening_physical_rounding_can_change(
                    cand, normalized_cand,
                    normalized_tolerance, area_tol_rel) ||
                _coarsening_area_near_tolerance(
                    normalized_cand, normalized_tolerance, false)
            repair_input = physical_rounding_can_change ?
                           cand : normalized_cand
            repair_area_tol_rel = physical_tolerance == 0.0 ?
                                  0.0 : area_tol_rel
            cand_rep = repair_mesh_for_simulation(
                repair_input;
                allow_boundary=allow_boundary,
                require_closed=require_closed,
                area_tol_rel=repair_area_tol_rel,
                drop_invalid=true,
                drop_degenerate=true,
                fix_orientation=true,
                strict_nonmanifold=strict_nonmanifold,
            )
            cand_mesh = physical_rounding_can_change ?
                        cand_rep.mesh :
                        _restore_coarsened_mesh(cand, cand_rep)
            _assert_coarsening_mesh_quality(
                cand_mesh, cand_rep.mesh;
                allow_boundary=allow_boundary,
                require_closed=require_closed,
                area_tol_rel=area_tol_rel,
            )
            _validate_coarsening_candidate_area_range(cand_mesh)
            nrwg = _count_rwg_edges(cand_mesh)
            (mesh=cand_mesh, rwg=nrwg)
        catch err
            _recoverable_coarsening_candidate_error(err) || rethrow()
            nothing
        end
        niter = iter

        if candidate === nothing
            invalid_upper = min(invalid_upper, h)
            if valid_lower > 0.0
                h = _geometric_midpoint(valid_lower, invalid_upper)
            else
                h *= 0.5
            end
            h > 0.0 && isfinite(h) || break
            continue
        end

        cand_mesh = candidate.mesh
        nrwg = candidate.rwg

        gap = abs(nrwg - target_rwg)
        if gap < best_gap
            best_gap = gap
            best_mesh = cand_mesh
            best_rwg = nrwg
        end
        ratio = nrwg / max(target_rwg, 1)
        if 0.85 <= ratio <= 1.15
            return (mesh=best_mesh, rwg_count=best_rwg, target_rwg=target_rwg,
                    best_gap=best_gap, iterations=iter)
        end

        if nrwg > target_rwg
            valid_lower = max(valid_lower, h)
            proposed = _scale_positive_float(h, sqrt(ratio))
            h = isfinite(invalid_upper) && proposed >= invalid_upper ?
                _geometric_midpoint(valid_lower, invalid_upper) : proposed
        else
            invalid_upper = min(invalid_upper, h)
            h = valid_lower > 0.0 ?
                _geometric_midpoint(valid_lower, invalid_upper) : h * 0.5
        end
        h > 0.0 && isfinite(h) || break
    end

    return (mesh=best_mesh, rwg_count=best_rwg, target_rwg=target_rwg, best_gap=best_gap, iterations=niter)
end

function _normalize_mesh_for_coarsening(mesh::TriMesh)
    size(mesh.xyz, 1) == 3 ||
        throw(DimensionMismatch(
            "coarsening requires a 3×Nv coordinate matrix"))
    size(mesh.tri, 1) == 3 ||
        throw(DimensionMismatch(
            "coarsening requires a 3×Nt connectivity matrix"))
    nvertices(mesh) > 0 ||
        throw(ArgumentError("cannot normalize an empty coarsening candidate"))
    all(isfinite, mesh.xyz) ||
        throw(ArgumentError(
            "coarsening candidate contains non-finite vertex coordinates"))
    referenced = falses(nvertices(mesh))
    for vertex in mesh.tri
        1 <= vertex <= nvertices(mesh) ||
            throw(ArgumentError(
                "coarsening candidate contains vertex index $vertex " *
                "outside 1:$(nvertices(mesh))"))
        referenced[vertex] = true
    end
    findfirst(referenced) === nothing &&
        throw(ArgumentError("cannot normalize a coarsening candidate without faces"))
    maximum_exponent = typemin(Int)
    @inbounds for vertex in 1:nvertices(mesh)
        referenced[vertex] || continue
        for component in 1:3
            value = mesh.xyz[component, vertex]
            iszero(value) && continue
            _, exponent = frexp(abs(value))
            maximum_exponent = max(maximum_exponent, exponent)
        end
    end
    maximum_exponent != typemin(Int) ||
        throw(ArgumentError(
            "coarsening candidate has no representable coordinate extent"))

    # A common power-of-two scale preserves the stored Float64 geometry and
    # its orientation; unlike translation, it cannot discard a subtraction
    # tail at a borderline triangle. Keep ordinary coordinates unchanged and
    # move only extreme magnitudes into a range where areas are normal.
    shift = maximum_exponent > 450 ? 450 - maximum_exponent :
            maximum_exponent < -450 ? -450 - maximum_exponent : 0

    # Some valid meshes contain referenced components spanning the complete
    # Float64 exponent range. If one common scale cannot preserve every stored
    # component exactly, retain the physical scale and let the certified cold
    # geometry paths settle it instead of rejecting valid input.
    if !iszero(shift)
        exact_scale = true
        @inbounds for vertex in 1:nvertices(mesh)
            referenced[vertex] || continue
            for component in 1:3
                value = mesh.xyz[component, vertex]
                scaled = ldexp(value, shift)
                if !isfinite(scaled) || ldexp(scaled, -shift) != value
                    exact_scale = false
                    break
                end
            end
            exact_scale || break
        end
        exact_scale || (shift = 0)
    end

    xyz = Matrix{Float64}(undef, 3, nvertices(mesh))
    @inbounds for vertex in 1:nvertices(mesh)
        if !referenced[vertex]
            xyz[1, vertex] = 0.0
            xyz[2, vertex] = 0.0
            xyz[3, vertex] = 0.0
            continue
        end
        for component in 1:3
            value = mesh.xyz[component, vertex]
            normalized = ldexp(value, shift)
            isfinite(normalized) ||
                throw(OverflowError(
                    "coarsening candidate normalization produced a non-finite coordinate"))
            xyz[component, vertex] = normalized
        end
    end
    return TriMesh(xyz, copy(mesh.tri))
end

@inline function _coarsening_area_near_tolerance(
        mesh::TriMesh, tolerance::Float64, physical_scale_required::Bool)
    tolerance == 0.0 && return false
    physical_scale_required && return true
    @inbounds for triangle in 1:ntriangles(mesh)
        area = triangle_area(mesh, triangle)
        separation = abs(area - tolerance)
        uncertainty = 256 * eps(Float64) * max(area, tolerance)
        separation <= uncertainty && return true
    end
    return false
end

function _assert_coarsening_mesh_quality(
        physical_mesh::TriMesh, normalized_mesh::TriMesh;
        allow_boundary::Bool, require_closed::Bool,
        area_tol_rel::Float64)
    report = try
        assert_mesh_quality(
            normalized_mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    catch err
        # A globally scaled copy can still push a very small secondary feature
        # through a representability boundary. Settle only that cold failure
        # against the stored endpoints.
        err isa ErrorException &&
        startswith(sprint(showerror, err), "Mesh quality precheck failed:") ||
            rethrow()
        return assert_mesh_quality(
            physical_mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end
    physical_scale_required = _coarsening_physical_rounding_can_change(
        physical_mesh, normalized_mesh, report.area_tol_abs, area_tol_rel)
    if normalized_mesh !== physical_mesh &&
       _coarsening_area_near_tolerance(
           normalized_mesh, report.area_tol_abs, physical_scale_required)
        return assert_mesh_quality(
            physical_mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end
    return report
end

function _coarsening_physical_rounding_can_change(
        physical_mesh::TriMesh, normalized_mesh::TriMesh,
        normalized_tolerance::Float64, area_tol_rel::Float64)
    normalized_mesh.xyz == physical_mesh.xyz && return false
    shift = nothing
    @inbounds for index in eachindex(physical_mesh.xyz)
        physical = physical_mesh.xyz[index]
        normalized = normalized_mesh.xyz[index]
        if !iszero(physical)
            physical_fraction, physical_exponent = frexp(physical)
            normalized_fraction, normalized_exponent = frexp(normalized)
            physical_fraction == normalized_fraction || return true
            shift = normalized_exponent - physical_exponent
            break
        end
    end
    shift === nothing && return false

    physical_scale = _coarsening_referenced_bbox_scale(physical_mesh)
    physical_scale === nothing && return true
    physical_tolerance = _area_tolerance(physical_scale, area_tol_rel)
    physical_tolerance == 0.0 && return false
    physical_tolerance <= floatmin(Float64) && return true

    # If both the rounded area and rounded tolerance stay normal after the
    # inverse scale, their ordering is exactly invariant under the power of
    # two. Only a representability boundary requires the physical cold oracle.
    tolerance_fraction, tolerance_exponent = frexp(normalized_tolerance)
    physical_tolerance_exponent = tolerance_exponent - 2shift
    if normalized_tolerance > 0.0 &&
       (physical_tolerance_exponent <= -1022 ||
        physical_tolerance_exponent >= 1024)
        return true
    end
    @inbounds for triangle in 1:ntriangles(normalized_mesh)
        area = triangle_area(normalized_mesh, triangle)
        area_fraction, area_exponent = frexp(area)
        physical_area_exponent = area_exponent - 2shift
        if area > 0.0 &&
           (physical_area_exponent <= -1022 ||
            physical_area_exponent >= 1024)
            return true
        end
    end
    return false
end

function _coarsening_referenced_bbox_scale(mesh::TriMesh)
    xmin = Inf
    ymin = Inf
    zmin = Inf
    xmax = -Inf
    ymax = -Inf
    zmax = -Inf
    @inbounds for triangle in 1:ntriangles(mesh)
        for row in 1:3
            vertex = mesh.tri[row, triangle]
            x = mesh.xyz[1, vertex]
            y = mesh.xyz[2, vertex]
            z = mesh.xyz[3, vertex]
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            zmin = min(zmin, z)
            xmax = max(xmax, x)
            ymax = max(ymax, y)
            zmax = max(zmax, z)
        end
    end
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    isfinite(dx) && isfinite(dy) && isfinite(dz) || return nothing
    # Match the public mesh-quality oracle bit for bit: its independently
    # rounded tolerance is part of the API contract at subnormal boundaries.
    scale = norm(Vec3(dx, dy, dz))
    isfinite(scale) || return nothing
    return scale
end

function _restore_coarsened_mesh(mesh::TriMesh, repair)
    mapping = repair.vertex_old_to_new
    xyz = Matrix{Float64}(undef, 3, nvertices(repair.mesh))
    @inbounds for old_vertex in eachindex(mapping)
        new_vertex = mapping[old_vertex]
        iszero(new_vertex) && continue
        xyz[1, new_vertex] = mesh.xyz[1, old_vertex]
        xyz[2, new_vertex] = mesh.xyz[2, old_vertex]
        xyz[3, new_vertex] = mesh.xyz[3, old_vertex]
    end
    return TriMesh(xyz, copy(repair.mesh.tri))
end

function _count_rwg_edges(mesh::TriMesh)
    counts = Dict{Tuple{Int,Int},UInt8}()
    @inbounds for triangle in 1:ntriangles(mesh)
        first = mesh.tri[1, triangle]
        second = mesh.tri[2, triangle]
        third = mesh.tri[3, triangle]
        for (left, right) in
                ((first, second), (second, third), (third, first))
            edge = left < right ? (left, right) : (right, left)
            count = get(counts, edge, UInt8(0))
            counts[edge] = min(UInt8(3), count + UInt8(1))
        end
    end
    result = 0
    @inbounds for count in values(counts)
        count == UInt8(2) && (result += 1)
    end
    return result
end

@inline function _coarsening_candidate_edge_length(
    first::Vec3,
    second::Vec3,
)
    dx = second[1] - first[1]
    dy = second[2] - first[2]
    dz = second[3] - first[3]
    if isfinite(dx) && isfinite(dy) && isfinite(dz)
        length_float = hypot(hypot(dx, dy), dz)
        if isfinite(length_float) &&
           length_float < 0.5 * floatmax(Float64)
            return length_float
        end
    end
    return _mesh_edge_length_big(first, second)
end

function _validate_coarsening_candidate_area_range(mesh::TriMesh)
    @inbounds for triangle in 1:ntriangles(mesh)
        first = _mesh_vertex(mesh, mesh.tri[1, triangle])
        second = _mesh_vertex(mesh, mesh.tri[2, triangle])
        third = _mesh_vertex(mesh, mesh.tri[3, triangle])
        fast, _, _, _, first_exponent, second_exponent, cross_norm =
            _scaled_triangle_cross(first, second, third)
        area = if fast && cross_norm > 0.0
            cross_fraction, cross_exponent = frexp(cross_norm)
            approximate = ldexp(
                cross_fraction,
                cross_exponent + first_exponent + second_exponent - 1)
            if !isfinite(approximate) ||
               approximate <= floatmin(Float64) ||
               approximate >= 0.5 * floatmax(Float64)
                triangle_area(mesh, triangle)
            else
                approximate
            end
        else
            triangle_area(mesh, triangle)
        end
        isfinite(area) && area > 0.0 ||
            throw(OverflowError(
                "triangle area is outside the representable Float64 range"))
        _coarsening_candidate_edge_length(first, second)
        _coarsening_candidate_edge_length(second, third)
        _coarsening_candidate_edge_length(third, first)
    end
    return nothing
end

@inline function _recoverable_coarsening_candidate_error(err)
    message = sprint(showerror, err)
    if err isa OverflowError
        return message in (
            "OverflowError: triangle area is outside the representable Float64 range",
            "OverflowError: mesh edge length is outside the representable Float64 range",
        )
    end
    if err isa ArgumentError || err isa DomainError
        return startswith(
                   message,
                   "ArgumentError: cluster_mesh_vertices: clustering removed all triangles") ||
               startswith(
                   message,
                   "ArgumentError: repair_mesh_for_simulation: empty mesh after cleanup")
    end
    return err isa ErrorException &&
           (startswith(
                message,
                "drop_nonmanifold_triangles: empty mesh after cleanup.") ||
            startswith(message, "Mesh quality precheck failed:"))
end

function _mesh_surface_area_for_coarsening(mesh::TriMesh)
    scale_exponent = typemin(Int)
    normalized_sum = 0.0
    compensation = 0.0
    @inbounds for triangle in 1:ntriangles(mesh)
        first = _mesh_vertex(mesh, mesh.tri[1, triangle])
        second = _mesh_vertex(mesh, mesh.tri[2, triangle])
        third = _mesh_vertex(mesh, mesh.tri[3, triangle])
        fast, _, _, _, scale1_exponent, scale2_exponent, cross_norm =
            _scaled_triangle_cross(first, second, third)
        area_fraction, area_exponent = if fast && cross_norm > 0.0
            cross_fraction, cross_exponent = frexp(cross_norm)
            (cross_fraction,
             cross_exponent + scale1_exponent + scale2_exponent - 1)
        else
            area = try
                triangle_area(mesh, triangle)
            catch err
                err isa OverflowError || rethrow()
                throw(ArgumentError(
                    "coarsen_mesh_to_target_rwg requires representable " *
                    "positive input-triangle areas"))
            end
            area > 0.0 ||
                throw(ArgumentError(
                    "coarsen_mesh_to_target_rwg requires positive triangle areas"))
            frexp(area)
        end
        if scale_exponent == typemin(Int)
            scale_exponent = area_exponent
        elseif area_exponent > scale_exponent
            ratio = ldexp(1.0, scale_exponent - area_exponent)
            normalized_sum *= ratio
            compensation *= ratio
            scale_exponent = area_exponent
        end
        normalized = ldexp(area_fraction, area_exponent - scale_exponent)
        previous = normalized_sum
        updated = previous + normalized
        correction = abs(previous) >= abs(normalized) ?
            (previous - updated) + normalized :
            (normalized - updated) + previous
        normalized_sum = updated
        compensation += correction
    end
    scale_exponent == typemin(Int) &&
        throw(ArgumentError(
            "coarsen_mesh_to_target_rwg requires positive surface area"))
    normalized = normalized_sum + compensation
    isfinite(normalized) && normalized > 0.0 ||
        throw(ArgumentError(
            "coarsen_mesh_to_target_rwg requires positive surface area"))
    return (normalized=normalized, exponent=scale_exponent)
end

@inline function _coarsening_initial_cell_size(
        surface_area, target_vertices::Int)
    sum_fraction, sum_exponent = frexp(surface_area.normalized)
    target_fraction, target_exponent = frexp(Float64(target_vertices))
    quotient_fraction = sum_fraction / target_fraction
    quotient_exponent =
        surface_area.exponent + sum_exponent - target_exponent
    quotient_fraction, normalization_exponent = frexp(quotient_fraction)
    quotient_exponent += normalization_exponent
    if isodd(quotient_exponent)
        quotient_fraction *= 2
        quotient_exponent -= 1
    end
    h = ldexp(sqrt(quotient_fraction), quotient_exponent ÷ 2)
    isfinite(h) && h > 0.0 ||
        throw(ArgumentError(
            "coarsening cell size is outside the positive finite " *
            "Float64 range"))
    return h
end

@inline function _scale_positive_float(value::Float64, factor::Float64)
    value_fraction, value_exponent = frexp(value)
    factor_fraction, factor_exponent = frexp(factor)
    return ldexp(
        value_fraction * factor_fraction,
        value_exponent + factor_exponent)
end

@inline function _geometric_midpoint(lower::Float64, upper::Float64)
    lower > 0.0 && upper >= lower && isfinite(upper) ||
        throw(ArgumentError(
            "invalid coarsening bracket [$lower, $upper]"))
    ratio = upper / lower
    if isfinite(ratio)
        return _scale_positive_float(lower, sqrt(ratio))
    end
    lower_fraction, lower_exponent = frexp(lower)
    upper_fraction, upper_exponent = frexp(upper)
    exponent_sum = lower_exponent + upper_exponent
    fraction = sqrt(lower_fraction * upper_fraction)
    if isodd(exponent_sum)
        fraction *= sqrt(2.0)
        exponent_sum -= 1
    end
    return ldexp(fraction, exponent_sum ÷ 2)
end

function _compact_mesh_vertices(
        xyz::Matrix{Float64}, tri::Matrix{Int})
    nv = size(xyz, 2)
    referenced = falses(nv)
    @inbounds for vertex in tri
        referenced[vertex] = true
    end

    old_to_new = zeros(Int, nv)
    removed_vertices = Int[]
    nretained = 0
    @inbounds for old_vertex in 1:nv
        if referenced[old_vertex]
            nretained += 1
            old_to_new[old_vertex] = nretained
        else
            push!(removed_vertices, old_vertex)
        end
    end

    xyz_clean = Matrix{Float64}(undef, 3, nretained)
    @inbounds for old_vertex in 1:nv
        new_vertex = old_to_new[old_vertex]
        iszero(new_vertex) && continue
        xyz_clean[1, new_vertex] = xyz[1, old_vertex]
        xyz_clean[2, new_vertex] = xyz[2, old_vertex]
        xyz_clean[3, new_vertex] = xyz[3, old_vertex]
    end

    @inbounds for index in eachindex(tri)
        tri[index] = old_to_new[tri[index]]
    end
    return TriMesh(xyz_clean, tri), removed_vertices, old_to_new
end

function _clean_mesh_triangles(mesh::TriMesh;
                               drop_invalid::Bool=true,
                               drop_degenerate::Bool=true,
                               area_tol_rel::Float64=1e-12)
    nv = nvertices(mesh)
    nt = ntriangles(mesh)
    tri = mesh.tri
    xyz = mesh.xyz

    scale, finite_vertex, invalid_vertices = _mesh_coordinate_diagnostics(mesh)
    area_tol_abs = _area_tolerance(scale, area_tol_rel)
    !drop_invalid && !isempty(invalid_vertices) &&
        error("Mesh contains $(length(invalid_vertices)) non-finite " *
              "vertices and `drop_invalid=false`.")

    keep_triangle = trues(nt)
    removed_invalid = Int[]
    removed_degenerate = Int[]
    removed_duplicate = Int[]
    retained_faces = Set{NTuple{3,Int}}()

    for t in 1:nt
        i1 = tri[1, t]
        i2 = tri[2, t]
        i3 = tri[3, t]

        valid_idx = (1 <= i1 <= nv) && (1 <= i2 <= nv) && (1 <= i3 <= nv)
        distinct = (i1 != i2) && (i2 != i3) && (i3 != i1)

        finite_coords = valid_idx &&
                        finite_vertex[i1] && finite_vertex[i2] && finite_vertex[i3]
        if !(valid_idx && distinct && finite_coords)
            if drop_invalid
                keep_triangle[t] = false
                push!(removed_invalid, t)
                continue
            else
                error("Triangle $t is invalid and `drop_invalid=false`.")
            end
        end

        area = triangle_area(mesh, t)

        if !isfinite(area) || area <= area_tol_abs
            if drop_degenerate
                keep_triangle[t] = false
                push!(removed_degenerate, t)
                continue
            else
                error("Triangle $t is degenerate (area=$area <= $area_tol_abs) and `drop_degenerate=false`.")
            end
        end

        face_key = _triangle_face_key(i1, i2, i3)
        if face_key in retained_faces
            keep_triangle[t] = false
            push!(removed_duplicate, t)
            continue
        end
        push!(retained_faces, face_key)
    end

    tri_retained = tri[:, keep_triangle]
    cleaned_mesh, removed_vertices, vertex_old_to_new =
        _compact_mesh_vertices(xyz, tri_retained)
    return (
        cleaned_mesh,
        removed_invalid,
        removed_degenerate,
        removed_duplicate,
        removed_vertices,
        vertex_old_to_new,
        area_tol_abs,
    )
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
- remove duplicate faces independent of winding,
- compact vertices that are no longer referenced,
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

    cleaned_mesh,
    removed_invalid,
    removed_degenerate,
    removed_duplicate,
    removed_vertices,
    vertex_old_to_new,
    area_tol_abs = _clean_mesh_triangles(
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
        cleaned_mesh, _, cleaned_to_final =
            _compact_mesh_vertices(mesh_nm.xyz, mesh_nm.tri)
        @inbounds for old_vertex in eachindex(vertex_old_to_new)
            cleaned_vertex = vertex_old_to_new[old_vertex]
            iszero(cleaned_vertex) && continue
            vertex_old_to_new[old_vertex] =
                cleaned_to_final[cleaned_vertex]
        end
        empty!(removed_vertices)
        @inbounds for old_vertex in eachindex(vertex_old_to_new)
            iszero(vertex_old_to_new[old_vertex]) &&
                push!(removed_vertices, old_vertex)
        end
        report_cleaned = mesh_quality_report(cleaned_mesh; area_tol_rel=area_tol_rel, check_orientation=true)
    end

    if strict_nonmanifold && report_cleaned.n_nonmanifold_edges > 0
        error("Mesh repair cannot continue with non-manifold edges ($(report_cleaned.n_nonmanifold_edges)).")
    end

    repaired_mesh = cleaned_mesh
    flipped_triangles = Int[]
    report_after = report_cleaned
    if fix_orientation && report_cleaned.n_orientation_conflicts > 0
        flip_flag = _compute_orientation_flips(cleaned_mesh)
        repaired_mesh, flipped_triangles = _apply_orientation_flips(cleaned_mesh, flip_flag)
        report_after = mesh_quality_report(
            repaired_mesh;
            area_tol_rel=area_tol_rel,
            check_orientation=true,
        )
    end

    _assert_mesh_quality_report(
        report_after;
        allow_boundary=allow_boundary,
        require_closed=require_closed,
    )

    return (
        mesh = repaired_mesh,
        before = report_before,
        cleaned = report_cleaned,
        after = report_after,
        removed_invalid = removed_invalid,
        removed_degenerate = removed_degenerate,
        removed_duplicate = removed_duplicate,
        removed_vertices = removed_vertices,
        vertex_old_to_new = vertex_old_to_new,
        removed_nonmanifold = removed_nonmanifold,
        flipped_triangles = flipped_triangles,
        area_tol_abs = area_tol_abs,
    )
end

"""
    repair_obj_mesh(input_path, output_path; reader_kwargs=NamedTuple(), kwargs...)

Read an OBJ mesh, repair it for solver prechecks, and write a repaired OBJ.
Returns the same metadata as `repair_mesh_for_simulation`, plus `output_path`.
OBJ resource limits may be passed in the `reader_kwargs` named tuple.
"""
function repair_obj_mesh(
        input_path::AbstractString, output_path::AbstractString;
        reader_kwargs::NamedTuple=NamedTuple(), kwargs...)
    mesh = read_obj_mesh(input_path; reader_kwargs...)
    result = repair_mesh_for_simulation(mesh; kwargs...)
    write_obj_mesh(output_path, result.mesh; header="Repaired from $input_path by DiffMoM")
    return (; result..., output_path=output_path)
end

@inline _mesh_vertex(mesh::TriMesh, i::Int) = Vec3(mesh.xyz[1, i], mesh.xyz[2, i], mesh.xyz[3, i])

# Exact differences between two Float64 endpoints can span 2,099 binary
# places. A cross product can double that span, so 4,352 bits cover the exact
# endpoint differences and determinant products with a guard margin. These
# fallbacks are cold: ordinary finite geometry stays allocation-free.
const _TRIANGLE_GEOMETRY_FALLBACK_PRECISION = 4352

@inline function _finite_triangle_vertices(v1::Vec3, v2::Vec3, v3::Vec3)
    return isfinite(v1[1]) && isfinite(v1[2]) && isfinite(v1[3]) &&
           isfinite(v2[1]) && isfinite(v2[2]) && isfinite(v2[3]) &&
           isfinite(v3[1]) && isfinite(v3[2]) && isfinite(v3[3])
end

@inline function _compensated_determinant(a::Float64, b::Float64,
                                          c::Float64, d::Float64)
    first_product = a * b
    second_product = c * d
    leading = first_product - second_product
    correction = fma(a, b, -first_product) - fma(c, d, -second_product)
    return leading + correction
end

@inline function _two_difference_error(first::Float64, second::Float64,
                                       difference::Float64)
    virtual_second = first - difference
    virtual_first = difference + virtual_second
    second_roundoff = virtual_second - second
    first_roundoff = first - virtual_first
    return first_roundoff + second_roundoff
end

@inline function _scaled_component_is_well_resolved(value::Float64)
    # Products of two admitted components stay above 2^-900. Their complete
    # 106-bit product expansions therefore remain normal Float64 numbers, so
    # a compensated zero determinant is a proof of exact collinearity rather
    # than an underflowed nonzero determinant.
    return value == 0.0 || abs(value) >= ldexp(1.0, -450)
end

@inline function _scaled_triangle_cross(v1::Vec3, v2::Vec3, v3::Vec3)
    e1x = v2[1] - v1[1]
    e1y = v2[2] - v1[2]
    e1z = v2[3] - v1[3]
    e2x = v3[1] - v1[1]
    e2y = v3[2] - v1[2]
    e2z = v3[3] - v1[3]

    if !(isfinite(e1x) && isfinite(e1y) && isfinite(e1z) &&
         isfinite(e2x) && isfinite(e2y) && isfinite(e2z))
        return false, 0.0, 0.0, 0.0, 0, 0, 0.0
    end

    scale1 = max(abs(e1x), abs(e1y), abs(e1z))
    scale2 = max(abs(e2x), abs(e2y), abs(e2z))
    if !(scale1 > 0.0 && scale2 > 0.0)
        return false, 0.0, 0.0, 0.0, 0, 0, 0.0
    end

    _, scale1_exponent = frexp(scale1)
    _, scale2_exponent = frexp(scale2)
    a1 = ldexp(e1x, -scale1_exponent)
    a2 = ldexp(e1y, -scale1_exponent)
    a3 = ldexp(e1z, -scale1_exponent)
    b1 = ldexp(e2x, -scale2_exponent)
    b2 = ldexp(e2y, -scale2_exponent)
    b3 = ldexp(e2z, -scale2_exponent)

    e1x_error = _two_difference_error(v2[1], v1[1], e1x)
    e1y_error = _two_difference_error(v2[2], v1[2], e1y)
    e1z_error = _two_difference_error(v2[3], v1[3], e1z)
    e2x_error = _two_difference_error(v3[1], v1[1], e2x)
    e2y_error = _two_difference_error(v3[2], v1[2], e2y)
    e2z_error = _two_difference_error(v3[3], v1[3], e2z)
    a1_error = ldexp(e1x_error, -scale1_exponent)
    a2_error = ldexp(e1y_error, -scale1_exponent)
    a3_error = ldexp(e1z_error, -scale1_exponent)
    b1_error = ldexp(e2x_error, -scale2_exponent)
    b2_error = ldexp(e2y_error, -scale2_exponent)
    b3_error = ldexp(e2z_error, -scale2_exponent)

    # A nonzero component lost while scaling can later be rescued by the
    # product of the two edge scales. Defer that exceptional case instead of
    # silently treating it as zero.
    if ((e1x != 0.0 && a1 == 0.0) ||
        (e1y != 0.0 && a2 == 0.0) ||
        (e1z != 0.0 && a3 == 0.0) ||
        (e2x != 0.0 && b1 == 0.0) ||
        (e2y != 0.0 && b2 == 0.0) ||
        (e2z != 0.0 && b3 == 0.0) ||
        (e1x_error != 0.0 && a1_error == 0.0) ||
        (e1y_error != 0.0 && a2_error == 0.0) ||
        (e1z_error != 0.0 && a3_error == 0.0) ||
        (e2x_error != 0.0 && b1_error == 0.0) ||
        (e2y_error != 0.0 && b2_error == 0.0) ||
        (e2z_error != 0.0 && b3_error == 0.0))
        return false, 0.0, 0.0, 0.0,
               scale1_exponent, scale2_exponent, 0.0
    end

    cx = _compensated_determinant(a2, b3, a3, b2)
    cy = _compensated_determinant(a3, b1, a1, b3)
    cz = _compensated_determinant(a1, b2, a2, b1)
    cross_norm = hypot(hypot(cx, cy), cz)
    edge1_norm = hypot(hypot(a1, a2), a3)
    edge2_norm = hypot(hypot(b1, b2), b3)
    edge1_error_norm = hypot(hypot(a1_error, a2_error), a3_error)
    edge2_error_norm = hypot(hypot(b1_error, b2_error), b3_error)
    subtraction_uncertainty =
        edge1_error_norm * edge2_norm +
        edge1_norm * edge2_error_norm +
        edge1_error_norm * edge2_error_norm
    arithmetic_uncertainty =
        32 * eps(Float64)^2 * edge1_norm * edge2_norm
    uncertainty = subtraction_uncertainty + arithmetic_uncertainty
    # Near collinearity amplifies both endpoint-subtraction residuals and the
    # compensated determinant's remaining second-order roundoff. Require a
    # 20-bit separation from that estimate; otherwise the exact fallback
    # keeps thin translated triangles from inheriting amplified Float64
    # endpoint-subtraction error.
    cancellation_floor = max(
        128 * eps(Float64) * edge1_norm * edge2_norm,
        1_048_576 * uncertainty,
    )
    if cross_norm == 0.0 &&
       e1x_error == 0.0 && e1y_error == 0.0 && e1z_error == 0.0 &&
       e2x_error == 0.0 && e2y_error == 0.0 && e2z_error == 0.0 &&
       _scaled_component_is_well_resolved(a1) &&
       _scaled_component_is_well_resolved(a2) &&
       _scaled_component_is_well_resolved(a3) &&
       _scaled_component_is_well_resolved(b1) &&
       _scaled_component_is_well_resolved(b2) &&
       _scaled_component_is_well_resolved(b3)
        return true, cx, cy, cz,
               scale1_exponent, scale2_exponent, cross_norm
    end
    if !(isfinite(cross_norm) && cross_norm > cancellation_floor)
        return false, cx, cy, cz,
               scale1_exponent, scale2_exponent, cross_norm
    end
    return true, cx, cy, cz,
           scale1_exponent, scale2_exponent, cross_norm
end

@inline function _scaled_triangle_area(cross_norm::Float64,
                                       scale1_exponent::Int,
                                       scale2_exponent::Int)
    cross_norm == 0.0 && return 0.0
    norm_fraction, norm_exponent = frexp(cross_norm)
    area = ldexp(
        0.5 * norm_fraction,
        norm_exponent + scale1_exponent + scale2_exponent,
    )
    return area
end

@noinline function _triangle_area_big(v1::Vec3, v2::Vec3, v3::Vec3)
    return setprecision(BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        e1x = BigFloat(v2[1]) - BigFloat(v1[1])
        e1y = BigFloat(v2[2]) - BigFloat(v1[2])
        e1z = BigFloat(v2[3]) - BigFloat(v1[3])
        e2x = BigFloat(v3[1]) - BigFloat(v1[1])
        e2y = BigFloat(v3[2]) - BigFloat(v1[2])
        e2z = BigFloat(v3[3]) - BigFloat(v1[3])
        cx = e1y * e2z - e1z * e2y
        cy = e1z * e2x - e1x * e2z
        cz = e1x * e2y - e1y * e2x
        area = Float64(hypot(hypot(cx, cy), cz) / 2)
        isfinite(area) ||
            throw(OverflowError("triangle area is outside the representable Float64 range"))
        return area
    end
end

@noinline function _triangle_area_rounds_positive_big(
        v1::Vec3, v2::Vec3, v3::Vec3)
    return setprecision(BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        e1x = BigFloat(v2[1]) - BigFloat(v1[1])
        e1y = BigFloat(v2[2]) - BigFloat(v1[2])
        e1z = BigFloat(v2[3]) - BigFloat(v1[3])
        e2x = BigFloat(v3[1]) - BigFloat(v1[1])
        e2y = BigFloat(v3[2]) - BigFloat(v1[2])
        e2z = BigFloat(v3[3]) - BigFloat(v1[3])
        cx = e1y * e2z - e1z * e2y
        cy = e1z * e2x - e1x * e2z
        cz = e1x * e2y - e1y * e2x
        area = hypot(hypot(cx, cy), cz) / 2
        return Float64(area) > 0.0
    end
end

@noinline function _triangle_normal_big(v1::Vec3, v2::Vec3, v3::Vec3,
                                        triangle_index::Int)
    return setprecision(BigFloat, _TRIANGLE_GEOMETRY_FALLBACK_PRECISION) do
        e1x = BigFloat(v2[1]) - BigFloat(v1[1])
        e1y = BigFloat(v2[2]) - BigFloat(v1[2])
        e1z = BigFloat(v2[3]) - BigFloat(v1[3])
        e2x = BigFloat(v3[1]) - BigFloat(v1[1])
        e2y = BigFloat(v3[2]) - BigFloat(v1[2])
        e2z = BigFloat(v3[3]) - BigFloat(v1[3])
        cx = e1y * e2z - e1z * e2y
        cy = e1z * e2x - e1x * e2z
        cz = e1x * e2y - e1y * e2x
        cross_norm = hypot(hypot(cx, cy), cz)
        cross_norm > 0 ||
            throw(DomainError(
                triangle_index,
                "triangle $triangle_index is degenerate; cannot compute a unit normal"))
        normal = Vec3(
            Float64(cx / cross_norm),
            Float64(cy / cross_norm),
            Float64(cz / cross_norm),
        )
        all(isfinite, normal) ||
            throw(OverflowError(
                "triangle $triangle_index unit normal is outside the representable Float64 range"))
        return normal
    end
end

@inline function _two_sum(first::Float64, second::Float64)
    total = first + second
    virtual_second = total - first
    error = (first - (total - virtual_second)) + (second - virtual_second)
    return total, error
end

@noinline function _mean3_big(first::Float64, second::Float64, third::Float64)
    # Preserve a subnormal addend across cancellation of opposite maximum
    # finite values, independent of vertex order.
    return setprecision(BigFloat, 2304) do
        Float64((BigFloat(first) + BigFloat(second) + BigFloat(third)) / 3)
    end
end

@inline function _mean3_finite(first::Float64, second::Float64, third::Float64)
    max_component = max(abs(first), abs(second), abs(third))
    max_component == 0.0 && return 0.0

    _, scale_exponent = frexp(max_component)
    first_scaled = ldexp(first, -scale_exponent)
    second_scaled = ldexp(second, -scale_exponent)
    third_scaled = ldexp(third, -scale_exponent)
    if ((first != 0.0 && first_scaled == 0.0) ||
        (second != 0.0 && second_scaled == 0.0) ||
        (third != 0.0 && third_scaled == 0.0))
        return _mean3_big(first, second, third)
    end

    leading12, error12 = _two_sum(first_scaled, second_scaled)
    leading123, error3 = _two_sum(leading12, third_scaled)
    correction_leading, correction_error = _two_sum(error12, error3)
    sum_leading, sum_error = _two_sum(leading123, correction_leading)
    tail = sum_error + correction_error
    corrected = sum_leading + tail

    # A nonzero expansion that rounds all the way to zero can still be rescued
    # by the power-of-two scale. That exceptional case needs BigFloat;
    # ordinary same-scale cancellation stays allocation-free. The restored
    # subnormal boundary is settled separately below.
    if corrected == 0.0 && (sum_leading != 0.0 || tail != 0.0)
        return _mean3_big(first, second, third)
    end
    corrected == 0.0 && return 0.0

    mean_fraction, mean_exponent = frexp(corrected)
    mean = ldexp(mean_fraction / 3, mean_exponent + scale_exponent)
    # Dividing the normalized significand before restoring the exponent can
    # choose the wrong side of a subnormal rounding tie. Re-evaluate that cold
    # boundary from the stored coordinates.
    return abs(mean) <= floatmin(Float64) ?
           _mean3_big(first, second, third) : mean
end

"""
    triangle_area(mesh, t)

Compute the area of triangle `t` in the mesh.
"""
function triangle_area(mesh::TriMesh, t::Int)
    v1 = _mesh_vertex(mesh, mesh.tri[1, t])
    v2 = _mesh_vertex(mesh, mesh.tri[2, t])
    v3 = _mesh_vertex(mesh, mesh.tri[3, t])
    _finite_triangle_vertices(v1, v2, v3) ||
        throw(DomainError(t, "triangle $t has non-finite coordinates"))
    fast, _, _, _, scale1_exponent, scale2_exponent, cross_norm =
        _scaled_triangle_cross(v1, v2, v3)
    fast || return _triangle_area_big(v1, v2, v3)
    cross_norm == 0.0 && return 0.0

    area = _scaled_triangle_area(
        cross_norm, scale1_exponent, scale2_exponent)
    # Endpoint-subtraction residuals can move a rounded result across either
    # representability boundary even when the triangle is well-conditioned.
    # Settle subnormal and upper-half-range results from the stored endpoints.
    if !isfinite(area) ||
       area <= floatmin(Float64) ||
       area >= 0.5 * floatmax(Float64)
        return _triangle_area_big(v1, v2, v3)
    end
    return area
end

"""
    triangle_center(mesh, t)

Compute the centroid of triangle `t`.
"""
function triangle_center(mesh::TriMesh, t::Int)
    v1 = _mesh_vertex(mesh, mesh.tri[1, t])
    v2 = _mesh_vertex(mesh, mesh.tri[2, t])
    v3 = _mesh_vertex(mesh, mesh.tri[3, t])
    _finite_triangle_vertices(v1, v2, v3) ||
        throw(DomainError(t, "triangle $t has non-finite coordinates"))
    return Vec3(
        _mean3_finite(v1[1], v2[1], v3[1]),
        _mean3_finite(v1[2], v2[2], v3[2]),
        _mean3_finite(v1[3], v2[3], v3[3]),
    )
end

"""
    triangle_normal(mesh, t)

Compute the outward unit normal of triangle `t`.
"""
function triangle_normal(mesh::TriMesh, t::Int)
    v1 = _mesh_vertex(mesh, mesh.tri[1, t])
    v2 = _mesh_vertex(mesh, mesh.tri[2, t])
    v3 = _mesh_vertex(mesh, mesh.tri[3, t])
    _finite_triangle_vertices(v1, v2, v3) ||
        throw(DomainError(
            t,
            "triangle $t has non-finite coordinates; cannot compute a unit normal"))
    fast, cx, cy, cz, _, _, cross_norm =
        _scaled_triangle_cross(v1, v2, v3)
    if fast
        cross_norm > 0.0 ||
            throw(DomainError(t, "triangle $t is degenerate; cannot compute a unit normal"))
        return Vec3(cx / cross_norm, cy / cross_norm, cz / cross_norm)
    end
    return _triangle_normal_big(v1, v2, v3, t)
end

"""
    mesh_unique_edges(mesh; max_edge_records=30_000_000)

Return the unique undirected edges of a triangle mesh as a vector of
`(i, j)` vertex-index pairs with `i < j`.
"""
function mesh_unique_edges(
        mesh::TriMesh;
        max_edge_records::Integer=_DEFAULT_MAX_MESH_EDGE_RECORDS)
    record_limit = _validated_resource_limit(
        "max_edge_records", max_edge_records)
    record_count = try
        Base.Checked.checked_mul(3, ntriangles(mesh))
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("mesh edge-record count overflows Int"))
    end
    record_count <= record_limit ||
        throw(ArgumentError(
            "mesh edge extraction requires $record_count triangle-edge " *
            "records, exceeding max_edge_records=$record_limit"))
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
    mesh_wireframe_segments(mesh;
        max_edge_records=30_000_000,
        max_output_bytes=536_870_912)

Build line-segment arrays for lightweight 3D wireframe visualization.
Returns a named tuple `(x, y, z, n_edges)` where each edge contributes
`(p1, p2, NaN)` to each coordinate vector, suitable for `Plots.path3d`.
"""
function mesh_wireframe_segments(
        mesh::TriMesh;
        max_edge_records::Integer=_DEFAULT_MAX_MESH_EDGE_RECORDS,
        max_output_bytes::Integer=_DEFAULT_MAX_MESH_EDGE_OUTPUT_BYTES)
    edges = mesh_unique_edges(mesh; max_edge_records=max_edge_records)
    n_edges = length(edges)
    coordinate_count = try
        Base.Checked.checked_mul(3, n_edges)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("mesh wireframe coordinate count overflows Int"))
    end
    output_bytes = _checked_array_payload_bytes(
        Float64, coordinate_count, 3; label="mesh wireframe coordinates")
    _enforce_payload_limit(
        output_bytes, max_output_bytes,
        "mesh wireframe coordinates", "max_output_bytes")
    x = Vector{Float64}(undef, coordinate_count)
    y = Vector{Float64}(undef, coordinate_count)
    z = Vector{Float64}(undef, coordinate_count)

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
        first = _mesh_vertex(mesh, i)
        second = _mesh_vertex(mesh, j)
        (all(isfinite, first) && all(isfinite, second)) ||
            throw(DomainError((i, j), "mesh edge has non-finite coordinates"))
        # Rounded endpoint differences can hide an exact distance just across
        # either side of Float64's upper boundary. The shared cold exact path
        # settles the entire upper half-range, including a rounded `Inf`.
        lens[k] = _coarsening_candidate_edge_length(first, second)
    end
    return lens
end

@noinline function _mesh_edge_length_big(first::Vec3, second::Vec3)
    return setprecision(BigFloat, 2304) do
        dx = BigFloat(second[1]) - BigFloat(first[1])
        dy = BigFloat(second[2]) - BigFloat(first[2])
        dz = BigFloat(second[3]) - BigFloat(first[3])
        length_float = Float64(hypot(hypot(dx, dy), dz))
        isfinite(length_float) ||
            throw(OverflowError(
                "mesh edge length is outside the representable Float64 range"))
        return length_float
    end
end

function _finite_nonnegative_mean(values::Vector{Float64})
    isempty(values) && return 0.0
    maximum_value = maximum(values)
    (isfinite(maximum_value) && maximum_value >= 0.0) ||
        throw(ArgumentError("edge lengths must be finite and nonnegative"))
    maximum_value == 0.0 && return 0.0

    scaled_sum = 0.0
    compensation = 0.0
    @inbounds for value in values
        (isfinite(value) && value >= 0.0) ||
            throw(ArgumentError("edge lengths must be finite and nonnegative"))
        scaled = value / maximum_value
        corrected = scaled - compensation
        updated = scaled_sum + corrected
        compensation = (updated - scaled_sum) - corrected
        scaled_sum = updated
    end
    return maximum_value * (scaled_sum / length(values))
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
the maximum unique edge length. Comparisons admit at most 64 floating-point
steps of roundoff at the boundary so an analytically exact target is not
rejected after coordinate transforms and Euclidean-norm evaluation.
"""
function mesh_resolution_report(mesh::TriMesh, freq_hz::Real;
                                points_per_wavelength::Real=10.0,
                                c0::Real=299792458.0)
    freq_hz_f = _positive_finite_length("mesh_resolution_report: freq_hz", freq_hz)
    points_per_wavelength_f = _positive_finite_length(
        "mesh_resolution_report: points_per_wavelength", points_per_wavelength)
    c0_f = _positive_finite_length("mesh_resolution_report: c0", c0)

    λ = c0_f / freq_hz_f
    target_h = λ / points_per_wavelength_f
    (isfinite(λ) && λ > 0.0 && isfinite(target_h) && target_h > 0.0) ||
        throw(ArgumentError(
            "mesh_resolution_report: frequency, c0, and points_per_wavelength " *
            "must produce finite positive wavelength and edge target"))

    lens = _mesh_edge_lengths(mesh)
    isempty(lens) && error("mesh_resolution_report: mesh has no edges")
    lens_sorted = sort(lens)

    h_min = lens_sorted[1]
    h_med = _percentile_from_sorted(lens_sorted, 0.50)
    h_p95 = _percentile_from_sorted(lens_sorted, 0.95)
    h_mean = _finite_nonnegative_mean(lens_sorted)
    h_max = lens_sorted[end]

    meets = _mesh_resolution_at_or_below(h_max, target_h)

    return (
        freq_hz = freq_hz_f,
        wavelength_m = λ,
        points_per_wavelength = points_per_wavelength_f,
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

@inline function _mesh_resolution_at_or_below(value::Float64, limit::Float64)
    value <= limit && return true
    return value - limit <= 64 * eps(max(value, limit))
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
        return _mesh_resolution_at_or_below(
            report.edge_max_m, report.target_max_edge_m)
    elseif criterion == :p95
        return _mesh_resolution_at_or_below(
            report.edge_p95_m, report.target_max_edge_m)
    elseif criterion == :median
        return _mesh_resolution_at_or_below(
            report.edge_median_m, report.target_max_edge_m)
    end
    error("mesh_resolution_ok: unknown criterion=$(criterion). Use :max, :p95, or :median.")
end

const _DEFAULT_REFINEMENT_MAX_OUTPUT_BYTES = 512 * 1024 * 1024

@noinline function _midpoint_big(first::Float64, second::Float64)
    return setprecision(BigFloat, 2304) do
        Float64((BigFloat(first) + BigFloat(second)) / 2)
    end
end

@inline function _safe_midpoint_component(first::Float64, second::Float64)
    # Canonical numeric order makes the rounded midpoint independent of the
    # edge's vertex labels. `isless` also gives signed zeros a stable order.
    lower, upper = isless(second, first) ? (second, first) : (first, second)
    midpoint = if signbit(lower) == signbit(upper)
        lower + (upper - lower) / 2
    else
        (lower + upper) / 2
    end
    if abs(midpoint) <= floatmin(Float64)
        leading, error = _two_sum(lower, upper)
        leading == 0.0 && error == 0.0 && return midpoint
        return _midpoint_big(lower, upper)
    end
    return midpoint
end

@inline function _safe_edge_midpoint(first::Vec3, second::Vec3)
    return Vec3(
        _safe_midpoint_component(first[1], second[1]),
        _safe_midpoint_component(first[2], second[2]),
        _safe_midpoint_component(first[3], second[3]),
    )
end

function _checked_mesh_payload_bytes(vertex_count::Int, triangle_count::Int)
    try
        vertex_bytes = Base.Checked.checked_mul(3 * sizeof(Float64), vertex_count)
        triangle_bytes = Base.Checked.checked_mul(3 * sizeof(Int), triangle_count)
        return Base.Checked.checked_add(vertex_bytes, triangle_bytes)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "refined mesh storage size overflows Int: vertices=$vertex_count, triangles=$triangle_count"))
    end
end

function _midpoint_refine_once(mesh::TriMesh, max_output_bytes::Int)
    Nv = nvertices(mesh)
    Nt = ntriangles(mesh)
    output_triangles = try
        Base.Checked.checked_mul(4, Nt)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("midpoint refinement triangle count overflows Int for $Nt input triangles"))
    end

    # Reject only from a guaranteed lower bound before counting unique edges.
    # The exact payload check below must decide borderline accepted cases.
    minimum_output_payload_bytes =
        _checked_mesh_payload_bytes(Nv, output_triangles)
    minimum_output_payload_bytes <= max_output_bytes ||
        return (mesh=nothing, stop_reason=:max_output_bytes,
                output_payload_bytes=minimum_output_payload_bytes)

    edges = mesh_unique_edges(mesh)
    output_vertices = try
        Base.Checked.checked_add(Nv, length(edges))
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "midpoint refinement vertex count overflows Int for $Nv vertices and $(length(edges)) edges"))
    end
    output_payload_bytes = _checked_mesh_payload_bytes(output_vertices, output_triangles)
    output_payload_bytes <= max_output_bytes ||
        return (mesh=nothing, stop_reason=:max_output_bytes,
                output_payload_bytes=output_payload_bytes)

    edge_mid = Dict{Tuple{Int,Int},Int}()
    sizehint!(edge_mid, length(edges))
    midpoints = Vector{Vec3}(undef, length(edges))
    @inbounds for (edge_index, (i, j)) in enumerate(edges)
        first = _mesh_vertex(mesh, i)
        second = _mesh_vertex(mesh, j)
        (all(isfinite, first) && all(isfinite, second)) ||
            throw(DomainError((i, j), "cannot refine an edge with non-finite coordinates"))
        midpoint = _safe_edge_midpoint(first, second)
        all(isfinite, midpoint) ||
            throw(OverflowError("midpoint refinement produced a non-finite coordinate"))
        if first != second && (midpoint == first || midpoint == second)
            return (mesh=nothing, stop_reason=:coordinate_resolution,
                    output_payload_bytes=output_payload_bytes)
        end
        midpoint == first && midpoint == second &&
            return (mesh=nothing, stop_reason=:coordinate_resolution,
                    output_payload_bytes=output_payload_bytes)
        midpoints[edge_index] = midpoint
        edge_mid[(i, j)] = Nv + edge_index
    end

    xyz_new = Matrix{Float64}(undef, 3, output_vertices)
    copyto!(view(xyz_new, :, 1:Nv), mesh.xyz)
    @inbounds for edge_index in eachindex(midpoints)
        midpoint = midpoints[edge_index]
        output_index = Nv + edge_index
        xyz_new[1, output_index] = midpoint[1]
        xyz_new[2, output_index] = midpoint[2]
        xyz_new[3, output_index] = midpoint[3]
    end

    tri_new = Matrix{Int}(undef, 3, output_triangles)
    tid = 0
    @inbounds for t in 1:Nt
        a = mesh.tri[1, t]
        b = mesh.tri[2, t]
        c = mesh.tri[3, t]

        mab = edge_mid[minmax(a, b)]
        mbc = edge_mid[minmax(b, c)]
        mca = edge_mid[minmax(c, a)]

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

    return (mesh=TriMesh(xyz_new, tri_new), stop_reason=:refined,
            output_payload_bytes=output_payload_bytes)
end

"""
    refine_mesh_to_target_edge(mesh, target_max_edge_m;
        max_iters=8, max_triangles=2_000_000,
        max_output_bytes=536_870_912)

Uniformly refine a triangle mesh via midpoint subdivision until
`edge_max_m <= target_max_edge_m` or limits are reached.
"""
function refine_mesh_to_target_edge(mesh::TriMesh, target_max_edge_m::Real;
                                    max_iters::Int=8,
                                    max_triangles::Int=2_000_000,
                                    max_output_bytes::Int=_DEFAULT_REFINEMENT_MAX_OUTPUT_BYTES)
    target_max_edge_m_f = _positive_finite_length(
        "refine_mesh_to_target_edge: target_max_edge_m", target_max_edge_m)
    max_iters >= 0 ||
        throw(ArgumentError("refine_mesh_to_target_edge: max_iters must be nonnegative, got $max_iters"))
    max_triangles > 0 ||
        throw(ArgumentError("refine_mesh_to_target_edge: max_triangles must be positive, got $max_triangles"))
    max_output_bytes > 0 ||
        throw(ArgumentError(
            "refine_mesh_to_target_edge: max_output_bytes must be positive, got $max_output_bytes"))

    mesh_cur = mesh
    before_lens = _mesh_edge_lengths(mesh_cur)
    isempty(before_lens) && error("refine_mesh_to_target_edge: mesh has no edges")
    edge_max_before = maximum(before_lens)

    hist_edge_max = Float64[edge_max_before]
    hist_triangles = Int[ntriangles(mesh_cur)]

    converged = edge_max_before <= target_max_edge_m_f
    iters = 0
    stop_reason = converged ? :target_reached : :max_iterations

    while !converged && iters < max_iters
        ntriangles(mesh_cur) <= max_triangles ÷ 4 || begin
            stop_reason = :max_triangles
            break
        end
        refined = _midpoint_refine_once(mesh_cur, max_output_bytes)
        if refined.mesh === nothing
            stop_reason = refined.stop_reason
            break
        end
        mesh_cur = refined.mesh
        iters += 1

        lens = _mesh_edge_lengths(mesh_cur)
        edge_max = maximum(lens)
        push!(hist_edge_max, edge_max)
        push!(hist_triangles, ntriangles(mesh_cur))
        converged = edge_max <= target_max_edge_m_f
        stop_reason = converged ? :target_reached : :max_iterations
    end

    return (
        mesh = mesh_cur,
        iterations = iters,
        converged = converged,
        stop_reason = stop_reason,
        target_max_edge_m = target_max_edge_m_f,
        edge_max_before_m = edge_max_before,
        edge_max_after_m = hist_edge_max[end],
        triangles_before = hist_triangles[1],
        triangles_after = hist_triangles[end],
        history_edge_max_m = hist_edge_max,
        history_triangles = hist_triangles,
    )
end

"""
    refine_mesh_for_mom(mesh, freq_hz;
        points_per_wavelength=10.0, max_iters=8,
        max_triangles=2_000_000, max_output_bytes=536_870_912)

Refine a mesh to satisfy a frequency-based MoM edge-length target:
`target_max_edge_m = λ / points_per_wavelength`.
"""
function refine_mesh_for_mom(mesh::TriMesh, freq_hz::Real;
                             points_per_wavelength::Real=10.0,
                             max_iters::Int=8,
                             max_triangles::Int=2_000_000,
                             max_output_bytes::Int=_DEFAULT_REFINEMENT_MAX_OUTPUT_BYTES,
                             c0::Real=299792458.0)
    report_before = mesh_resolution_report(mesh, freq_hz;
                                           points_per_wavelength=points_per_wavelength,
                                           c0=c0)
    result = refine_mesh_to_target_edge(mesh, report_before.target_max_edge_m;
                                        max_iters=max_iters,
                                        max_triangles=max_triangles,
                                        max_output_bytes=max_output_bytes)
    report_after = mesh_resolution_report(result.mesh, freq_hz;
                                          points_per_wavelength=points_per_wavelength,
                                          c0=c0)
    return (; result..., report_before=report_before, report_after=report_after)
end
