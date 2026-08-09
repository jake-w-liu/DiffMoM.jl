# MeshIO.jl — Multi-format mesh I/O: STL, Gmsh MSH, unified dispatcher, CAD conversion

export read_stl_mesh, write_stl_mesh
export read_msh_mesh
export read_mesh, write_mesh
export convert_cad_to_mesh

# ───────────────────────────────────────────────────────────────
# STL: Binary and ASCII reader/writer
# ───────────────────────────────────────────────────────────────

@inline function _ascii_field_bounds(line::AbstractString, position::Int)
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

@inline function _required_ascii_field(line::AbstractString, position::Int,
                                       context::AbstractString)
    first, last, next_position = _ascii_field_bounds(line, position)
    iszero(first) && error("Invalid $context: $line")
    return SubString(line, first, last), next_position
end

@inline function _required_ascii_field(line::AbstractString, position::Int,
                                       context::AbstractString,
                                       path::AbstractString)
    first, last, next_position = _ascii_field_bounds(line, position)
    iszero(first) && error("Invalid $context in $path: $line")
    return SubString(line, first, last), next_position
end

abstract type _STLVertexMerger end

struct _ExactSTLVertexMerger <: _STLVertexMerger
    vertex_map::Dict{NTuple{3,UInt64},Int}
    xyz::Vector{NTuple{3,Float64}}
end

struct _ToleranceSTLVertexMerger <: _STLVertexMerger
    bucket_heads::Dict{NTuple{3,Int64},Int}
    next_in_bucket::Vector{Int}
    xyz::Vector{NTuple{3,Float64}}
    tolerance::Float64
end

@inline function _validated_stl_merge_tol(merge_tol::Float64)
    isfinite(merge_tol) && merge_tol >= 0 ||
        throw(ArgumentError("merge_tol must be finite and nonnegative, got $merge_tol."))
    return merge_tol
end

@inline function _new_stl_vertex_merger(merge_tol::Float64)
    if iszero(merge_tol)
        return _ExactSTLVertexMerger(
            Dict{NTuple{3,UInt64},Int}(), NTuple{3,Float64}[])
    end
    return _ToleranceSTLVertexMerger(
        Dict{NTuple{3,Int64},Int}(), Int[], NTuple{3,Float64}[], merge_tol)
end

@inline function _validated_stl_coordinate(coord::NTuple{3,Float64})
    isfinite(coord[1]) && isfinite(coord[2]) && isfinite(coord[3]) ||
        throw(ArgumentError("STL vertex coordinates must be finite, got $coord."))
    return coord
end

@inline function _merge_stl_vertex!(merger::_ExactSTLVertexMerger,
                                    coord::NTuple{3,Float64})
    _validated_stl_coordinate(coord)
    key = (reinterpret(UInt64, coord[1]),
           reinterpret(UInt64, coord[2]),
           reinterpret(UInt64, coord[3]))
    id = get(merger.vertex_map, key, 0)
    if iszero(id)
        push!(merger.xyz, coord)
        id = length(merger.xyz)
        merger.vertex_map[key] = id
    end
    return id
end

function _stl_merge_cell_coordinate(x::Float64, tolerance::Float64)
    scaled = x / tolerance
    isfinite(scaled) ||
        throw(ArgumentError(
            "merge_tol=$tolerance is too small for STL coordinate $x."))
    cell = try
        floor(Int64, scaled)
    catch err
        err isa InexactError || rethrow()
        throw(ArgumentError(
            "merge_tol=$tolerance is too small for STL coordinate $x."))
    end
    typemin(Int64) < cell < typemax(Int64) ||
        throw(ArgumentError(
            "merge_tol=$tolerance is too small for STL coordinate $x."))
    return cell
end

function _merge_stl_vertex!(merger::_ToleranceSTLVertexMerger,
                            coord::NTuple{3,Float64})
    _validated_stl_coordinate(coord)
    tolerance = merger.tolerance
    cell = (
        _stl_merge_cell_coordinate(coord[1], tolerance),
        _stl_merge_cell_coordinate(coord[2], tolerance),
        _stl_merge_cell_coordinate(coord[3], tolerance),
    )

    best_id = 0
    best_distance = Inf
    @inbounds for dz in -1:1, dy in -1:1, dx in -1:1
        neighbor_cell = (cell[1] + dx, cell[2] + dy, cell[3] + dz)
        candidate_id = get(merger.bucket_heads, neighbor_cell, 0)
        while !iszero(candidate_id)
            candidate = merger.xyz[candidate_id]
            distance = hypot(
                hypot(coord[1] - candidate[1], coord[2] - candidate[2]),
                coord[3] - candidate[3],
            )
            if distance <= tolerance &&
                    (distance < best_distance ||
                     (distance == best_distance &&
                      (iszero(best_id) || candidate_id < best_id)))
                best_id = candidate_id
                best_distance = distance
            end
            candidate_id = merger.next_in_bucket[candidate_id]
        end
    end
    !iszero(best_id) && return best_id

    push!(merger.xyz, coord)
    id = length(merger.xyz)
    head = get(merger.bucket_heads, cell, 0)
    push!(merger.next_in_bucket, head)
    merger.bucket_heads[cell] = id
    return id
end

function _stl_mesh_from_merger(merger::_STLVertexMerger, tri::Matrix{Int})
    xyz_list = merger.xyz
    xyz = Matrix{Float64}(undef, 3, length(xyz_list))
    @inbounds for i in eachindex(xyz_list)
        xyz[1, i] = xyz_list[i][1]
        xyz[2, i] = xyz_list[i][2]
        xyz[3, i] = xyz_list[i][3]
    end
    return TriMesh(xyz, tri)
end

@inline function _stl_uint32_le(bytes::AbstractVector{UInt8}, i::Int)
    @inbounds return UInt32(bytes[i]) |
                     (UInt32(bytes[i + 1]) << 8) |
                     (UInt32(bytes[i + 2]) << 16) |
                     (UInt32(bytes[i + 3]) << 24)
end

@inline _stl_float32_le(bytes::AbstractVector{UInt8}, i::Int) =
    reinterpret(Float32, _stl_uint32_le(bytes, i))

function _stl_binary_triangle_count(header::AbstractVector{UInt8},
                                    file_size::Integer)
    length(header) >= 84 || return nothing
    ntri = Int(_stl_uint32_le(header, 81))
    expected = Base.checked_add(84, Base.checked_mul(50, ntri))
    return expected == file_size ? ntri : nothing
end

"""
    read_stl_mesh(path; merge_tol=0.0)

Read a triangle mesh from an STL file (binary or ASCII, auto-detected).

STL stores three vertices per facet with no shared-vertex topology, so
duplicate vertices are merged. With the default `merge_tol=0.0`, vertices
are merged when their Float64 coordinates are bitwise identical (suitable
for most STL files). Set `merge_tol` to a small positive value (e.g.
`1e-10 * bbox_diagonal`) if your exporter introduces tiny floating-point
noise between shared vertices. A positive tolerance merges vertices whose
Euclidean distance is no greater than `merge_tol`.

Returns a `TriMesh`.
"""
function read_stl_mesh(path::AbstractString; merge_tol::Float64=0.0)
    tolerance = _validated_stl_merge_tol(merge_tol)
    return open(path, "r") do io
        seekend(io)
        file_size = position(io)
        seekstart(io)

        if file_size >= 84
            header = Vector{UInt8}(undef, 84)
            read!(io, header)
            ntri = _stl_binary_triangle_count(header, file_size)
            if !isnothing(ntri)
                return _read_stl_binary(io, ntri; merge_tol=tolerance)
            end
            seekstart(io)
        end
        return _read_stl_ascii(io, path; merge_tol=tolerance)
    end
end

function _stl_is_binary(data::Vector{UInt8})
    return !isnothing(_stl_binary_triangle_count(data, length(data)))
end

function _read_stl_binary(io::IO, ntri::Int; merge_tol::Float64=0.0)
    ntri > 0 || error("STL binary file has 0 triangles.")
    tolerance = _validated_stl_merge_tol(merge_tol)

    merger = _new_stl_vertex_merger(tolerance)
    tri = Matrix{Int}(undef, 3, ntri)
    record = Vector{UInt8}(undef, 50)
    for t in 1:ntri
        read!(io, record)
        for v in 1:3
            offset = 13 + 12 * (v - 1)
            coord = (
                Float64(_stl_float32_le(record, offset)),
                Float64(_stl_float32_le(record, offset + 4)),
                Float64(_stl_float32_le(record, offset + 8)),
            )
            tri[v, t] = _merge_stl_vertex!(merger, coord)
        end
    end
    return _stl_mesh_from_merger(merger, tri)
end

function _read_stl_binary(data::Vector{UInt8}; merge_tol::Float64=0.0)
    tolerance = _validated_stl_merge_tol(merge_tol)
    ntri = _stl_binary_triangle_count(data, length(data))
    isnothing(ntri) && error("Invalid or truncated binary STL data.")
    io = IOBuffer(data)
    seek(io, 84)
    return _read_stl_binary(io, ntri; merge_tol=tolerance)
end

function _parse_stl_ascii_vertex(s::AbstractString)
    position = firstindex(s)
    _, position = _required_ascii_field(s, position, "STL vertex line")
    x_field, position = _required_ascii_field(s, position, "STL vertex line")
    y_field, position = _required_ascii_field(s, position, "STL vertex line")
    z_field, _ = _required_ascii_field(s, position, "STL vertex line")

    return (
        parse(Float64, x_field),
        parse(Float64, y_field),
        parse(Float64, z_field),
    )
end

function _read_stl_ascii(io::IO, path::AbstractString; merge_tol::Float64=0.0)
    tolerance = _validated_stl_merge_tol(merge_tol)
    merger = _new_stl_vertex_merger(tolerance)
    tri_ids = Int[]
    ntri = 0

    for line in eachline(io)
        s = strip(line)
        if startswith(s, "facet normal")
            ntri += 1
        elseif startswith(s, "vertex")
            coord = _parse_stl_ascii_vertex(s)
            push!(tri_ids, _merge_stl_vertex!(merger, coord))
        end
    end

    expected_vertices = Base.checked_mul(3, ntri)
    length(tri_ids) == expected_vertices ||
        error("STL ASCII: expected $expected_vertices vertices for $ntri facets, got $(length(tri_ids)).")
    ntri > 0 || error("STL ASCII file has 0 facets: $path")

    # Copy out of the geometrically grown vector so the returned mesh does not
    # retain unused capacity from ASCII parsing.
    tri = Matrix{Int}(undef, 3, ntri)
    copyto!(tri, tri_ids)
    return _stl_mesh_from_merger(merger, tri)
end

function _read_stl_ascii(path::AbstractString; merge_tol::Float64=0.0)
    tolerance = _validated_stl_merge_tol(merge_tol)
    return open(path, "r") do io
        _read_stl_ascii(io, path; merge_tol=tolerance)
    end
end

function _merge_stl_vertices(raw_verts::Vector{NTuple{3,Float64}}, ntri::Int;
                              merge_tol::Float64=0.0)
    ntri >= 0 || throw(ArgumentError("ntri must be nonnegative, got $ntri."))
    tolerance = _validated_stl_merge_tol(merge_tol)
    expected_vertices = Base.checked_mul(3, ntri)
    length(raw_verts) == expected_vertices ||
        throw(DimensionMismatch(
            "expected $expected_vertices STL vertices, got $(length(raw_verts))"))

    merger = _new_stl_vertex_merger(tolerance)
    tri = Matrix{Int}(undef, 3, ntri)
    @inbounds for t in 1:ntri
        for v in 1:3
            coord = raw_verts[3 * (t - 1) + v]
            tri[v, t] = _merge_stl_vertex!(merger, coord)
        end
    end
    return _stl_mesh_from_merger(merger, tri)
end

"""
    write_stl_mesh(path, mesh; header="Exported by DiffMoM", ascii=false)

Write a `TriMesh` to an STL file. Default is binary STL (compact, fast).
Set `ascii=true` for a human-readable ASCII STL.
"""
function write_stl_mesh(path::AbstractString, mesh::TriMesh;
                         header::AbstractString="Exported by DiffMoM",
                         ascii::Bool=false)
    if ascii
        return _write_stl_ascii(path, mesh; header=header)
    else
        return _write_stl_binary(path, mesh; header=header)
    end
end

@inline _write_stl_uint32_le(io::IO, value::UInt32) =
    write(io, htol(value))
@inline _write_stl_uint16_le(io::IO, value::UInt16) =
    write(io, htol(value))
@inline _write_stl_float32_le(io::IO, value) =
    _write_stl_uint32_le(io, reinterpret(UInt32, Float32(value)))

function _validate_stl_mesh_for_write(mesh::TriMesh; binary::Bool)
    size(mesh.xyz, 1) == 3 ||
        throw(DimensionMismatch(
            "STL vertex coordinates must have size (3, Nv), got $(size(mesh.xyz))."))
    size(mesh.tri, 1) == 3 ||
        throw(DimensionMismatch(
            "STL triangle connectivity must have size (3, Nt), got $(size(mesh.tri))."))

    nt = ntriangles(mesh)
    nt > 0 || throw(ArgumentError("Cannot write an STL mesh with 0 triangles."))
    if binary
        nt <= typemax(UInt32) ||
            throw(ArgumentError(
                "Binary STL supports at most $(typemax(UInt32)) triangles."))
    end

    nv = nvertices(mesh)
    @inbounds for t in 1:nt
        for local_vertex in 1:3
            vertex = mesh.tri[local_vertex, t]
            1 <= vertex <= nv ||
                throw(ArgumentError(
                    "STL triangle $t references vertex $vertex outside 1:$nv."))
            coord = (mesh.xyz[1, vertex], mesh.xyz[2, vertex], mesh.xyz[3, vertex])
            _validated_stl_coordinate(coord)
            if binary
                coord32 = (Float32(coord[1]), Float32(coord[2]), Float32(coord[3]))
                isfinite(coord32[1]) && isfinite(coord32[2]) && isfinite(coord32[3]) ||
                    throw(ArgumentError(
                        "STL vertex coordinates at triangle $t, local vertex $local_vertex " *
                        "are outside the finite Float32 range required by binary STL: $coord."))
            end
        end
        # Validate degeneracy before opening the destination.  Both STL writers
        # require a finite unit normal for every facet.
        triangle_normal(mesh, t)
    end
    return nothing
end

function _write_stl_binary(path::AbstractString, mesh::TriMesh; header::AbstractString="")
    _validate_stl_mesh_for_write(mesh; binary=true)
    nt = ntriangles(mesh)
    open(path, "w") do io
        # 80-byte header (padded with zeros)
        hdr = zeros(UInt8, 80)
        header_bytes = codeunits(header)
        copyto!(hdr, 1, header_bytes, 1, min(length(header_bytes), length(hdr)))
        write(io, hdr)
        _write_stl_uint32_le(io, UInt32(nt))
        for t in 1:nt
            n = triangle_normal(mesh, t)
            _write_stl_float32_le(io, n[1])
            _write_stl_float32_le(io, n[2])
            _write_stl_float32_le(io, n[3])
            for vi in 1:3
                idx = mesh.tri[vi, t]
                _write_stl_float32_le(io, mesh.xyz[1, idx])
                _write_stl_float32_le(io, mesh.xyz[2, idx])
                _write_stl_float32_le(io, mesh.xyz[3, idx])
            end
            _write_stl_uint16_le(io, UInt16(0))  # attribute byte count
        end
    end
    return path
end

function _write_stl_ascii(path::AbstractString, mesh::TriMesh; header::AbstractString="")
    _validate_stl_mesh_for_write(mesh; binary=false)
    _validate_text_mesh_header(header, "ASCII STL")
    nt = ntriangles(mesh)
    open(path, "w") do io
        println(io, "solid ", header)
        for t in 1:nt
            n = triangle_normal(mesh, t)
            println(io, "  facet normal $(n[1]) $(n[2]) $(n[3])")
            println(io, "    outer loop")
            for vi in 1:3
                idx = mesh.tri[vi, t]
                x = mesh.xyz[1, idx]
                y = mesh.xyz[2, idx]
                z = mesh.xyz[3, idx]
                println(io, "      vertex $x $y $z")
            end
            println(io, "    endloop")
            println(io, "  endfacet")
        end
        println(io, "endsolid ", header)
    end
    return path
end

# ───────────────────────────────────────────────────────────────
# Gmsh MSH: v2 and v4 ASCII reader
# ───────────────────────────────────────────────────────────────

"""
    read_msh_mesh(path)

Read a triangle surface mesh from a Gmsh MSH file (ASCII v2.x or v4.x).
The parser streams each section and rejects binary or unsupported versions.

Only 3-node triangle elements (Gmsh type 2) are extracted; all other
element types (lines, quads, tetrahedra, etc.) are silently ignored.
Node IDs are remapped to 1-based contiguous indices. Section counts,
end markers, tag bounds, duplicate node tags, and missing references are
validated.

Returns a `TriMesh`.
"""
function read_msh_mesh(path::AbstractString)
    return open(path, "r") do io
        version = _read_msh_format(io, path)
        seekstart(io)
        if 2.0 <= version < 3.0
            return _read_msh_v2(io, path)
        elseif 4.0 <= version < 5.0
            return _read_msh_v4(io, path)
        end
        error("Unsupported MSH version $version in $path; supported ASCII versions are 2.x and 4.x.")
    end
end

function _required_msh_line(io::IO, path::AbstractString, context::AbstractString)
    while !eof(io)
        line = strip(readline(io))
        isempty(line) || return line
    end
    error("Unexpected end of MSH file $path while reading $context.")
end

function _parse_msh_format_line(line::AbstractString, path::AbstractString)
    position = firstindex(line)
    version_field, position =
        _required_ascii_field(line, position, "MSH format line", path)
    file_type_field, position =
        _required_ascii_field(line, position, "MSH format line", path)
    data_size_field, _ =
        _required_ascii_field(line, position, "MSH format line", path)

    version = parse(Float64, version_field)
    file_type = parse(Int, file_type_field)
    data_size = parse(Int, data_size_field)
    isfinite(version) && version > 0 ||
        error("Invalid MSH version $version in $path.")
    file_type == 0 ||
        error("Binary MSH files are not supported: $path.")
    data_size > 0 ||
        error("Invalid MSH data-size field $data_size in $path.")
    return version
end

function _read_msh_format(io::IO, path::AbstractString)
    while !eof(io)
        line = strip(readline(io))
        line == "\$MeshFormat" || continue
        format_line = _required_msh_line(io, path, "\$MeshFormat")
        version = _parse_msh_format_line(format_line, path)
        end_marker = _required_msh_line(io, path, "\$EndMeshFormat")
        end_marker == "\$EndMeshFormat" ||
            error("MSH file $path is missing \$EndMeshFormat.")
        return version
    end
    error("MSH file missing \$MeshFormat section: $path")
end

function _parse_msh_int(line::AbstractString, path::AbstractString,
                        context::AbstractString)
    value_field, _ = _required_ascii_field(
        line, firstindex(line), context, path)
    return parse(Int, value_field)
end

function _parse_msh_int4(line::AbstractString, path::AbstractString,
                         context::AbstractString)
    position = firstindex(line)
    field1, position = _required_ascii_field(
        line, position, context, path)
    field2, position = _required_ascii_field(
        line, position, context, path)
    field3, position = _required_ascii_field(
        line, position, context, path)
    field4, _ = _required_ascii_field(
        line, position, context, path)
    return (
        parse(Int, field1),
        parse(Int, field2),
        parse(Int, field3),
        parse(Int, field4),
    )
end

function _parse_msh_v2_node(line::AbstractString, path::AbstractString)
    position = firstindex(line)
    tag_field, position =
        _required_ascii_field(line, position, "MSH v2 node", path)
    x_field, position =
        _required_ascii_field(line, position, "MSH v2 node", path)
    y_field, position =
        _required_ascii_field(line, position, "MSH v2 node", path)
    z_field, _ =
        _required_ascii_field(line, position, "MSH v2 node", path)
    coord = (
        parse(Float64, x_field),
        parse(Float64, y_field),
        parse(Float64, z_field),
    )
    isfinite(coord[1]) && isfinite(coord[2]) && isfinite(coord[3]) ||
        error("MSH node coordinates must be finite in $path: $line")
    return parse(Int, tag_field), coord
end

function _parse_msh_v2_element(line::AbstractString, path::AbstractString)
    position = firstindex(line)
    element_tag_field, position =
        _required_ascii_field(line, position, "MSH v2 element", path)
    type_field, position =
        _required_ascii_field(line, position, "MSH v2 element", path)
    ntags_field, position =
        _required_ascii_field(line, position, "MSH v2 element", path)

    element_tag = parse(Int, element_tag_field)
    element_tag > 0 || error("MSH element tags must be positive in $path.")
    element_type = parse(Int, type_field)
    ntags = parse(Int, ntags_field)
    ntags >= 0 || error("Negative MSH v2 element tag count in $path: $line")
    for _ in 1:ntags
        _, position = _required_ascii_field(
            line, position, "MSH v2 element tags", path)
    end
    element_type == 2 || return nothing

    n1_field, position =
        _required_ascii_field(line, position, "MSH v2 triangle", path)
    n2_field, position =
        _required_ascii_field(line, position, "MSH v2 triangle", path)
    n3_field, _ =
        _required_ascii_field(line, position, "MSH v2 triangle", path)
    return (parse(Int, n1_field), parse(Int, n2_field), parse(Int, n3_field))
end

function _read_msh_v2(io::IO, path::AbstractString)
    nodes = Dict{Int, NTuple{3,Float64}}()
    triangles = NTuple{3,Int}[]

    while !eof(io)
        section = strip(readline(io))
        if section == "\$Nodes"
            count_line = _required_msh_line(io, path, "MSH v2 node count")
            n_nodes = _parse_msh_int(count_line, path, "MSH v2 node count")
            n_nodes >= 0 || error("Negative MSH v2 node count in $path.")
            sizehint!(nodes, Base.checked_add(length(nodes), n_nodes))
            for _ in 1:n_nodes
                line = _required_msh_line(io, path, "MSH v2 node")
                node_tag, coord = _parse_msh_v2_node(line, path)
                node_tag > 0 || error("MSH node tags must be positive in $path.")
                haskey(nodes, node_tag) &&
                    error("Duplicate MSH node tag $node_tag in $path.")
                nodes[node_tag] = coord
            end
            end_marker = _required_msh_line(io, path, "\$EndNodes")
            end_marker == "\$EndNodes" ||
                error("MSH file $path is missing \$EndNodes.")
        elseif section == "\$Elements"
            count_line = _required_msh_line(io, path, "MSH v2 element count")
            n_elems = _parse_msh_int(count_line, path, "MSH v2 element count")
            n_elems >= 0 || error("Negative MSH v2 element count in $path.")
            for _ in 1:n_elems
                line = _required_msh_line(io, path, "MSH v2 element")
                triangle = _parse_msh_v2_element(line, path)
                isnothing(triangle) || push!(triangles, triangle)
            end
            end_marker = _required_msh_line(io, path, "\$EndElements")
            end_marker == "\$EndElements" ||
                error("MSH file $path is missing \$EndElements.")
        end
    end

    isempty(nodes) && error("MSH v2 file has no nodes: $path")
    isempty(triangles) && @warn "MSH v2 file has no triangle elements: $path"

    return _build_trimesh_from_msh(nodes, triangles)
end

function _read_msh_integer_block(io::IO, path::AbstractString, count::Int,
                                 context::AbstractString)
    values = Vector{Int}(undef, count)
    filled = 0
    while filled < count
        line = _required_msh_line(io, path, context)
        position = firstindex(line)
        while true
            first, last, next_position = _ascii_field_bounds(line, position)
            iszero(first) && break
            filled < count ||
                error("Too many integer values while reading $context in $path.")
            filled += 1
            values[filled] = parse(Int, SubString(line, first, last))
            position = next_position
        end
    end
    return values
end

function _parse_msh_xyz(line::AbstractString, path::AbstractString)
    position = firstindex(line)
    x_field, position =
        _required_ascii_field(line, position, "MSH node coordinates", path)
    y_field, position =
        _required_ascii_field(line, position, "MSH node coordinates", path)
    z_field, _ =
        _required_ascii_field(line, position, "MSH node coordinates", path)
    coord = (
        parse(Float64, x_field),
        parse(Float64, y_field),
        parse(Float64, z_field),
    )
    isfinite(coord[1]) && isfinite(coord[2]) && isfinite(coord[3]) ||
        error("MSH node coordinates must be finite in $path: $line")
    return coord
end

function _parse_msh_v4_element(line::AbstractString, path::AbstractString,
                               is_triangle::Bool)
    position = firstindex(line)
    tag_field, position =
        _required_ascii_field(line, position, "MSH v4 element", path)
    element_tag = parse(Int, tag_field)
    is_triangle || return element_tag, nothing

    n1_field, position =
        _required_ascii_field(line, position, "MSH v4 triangle", path)
    n2_field, position =
        _required_ascii_field(line, position, "MSH v4 triangle", path)
    n3_field, _ =
        _required_ascii_field(line, position, "MSH v4 triangle", path)
    return element_tag, (
        parse(Int, n1_field),
        parse(Int, n2_field),
        parse(Int, n3_field),
    )
end

function _read_msh_v4(io::IO, path::AbstractString)
    nodes = Dict{Int, NTuple{3,Float64}}()
    triangles = NTuple{3,Int}[]

    while !eof(io)
        section = strip(readline(io))
        if section == "\$Nodes"
            header_line = _required_msh_line(io, path, "MSH v4 node header")
            n_entity_blocks, total_nodes, min_node_tag, max_node_tag =
                _parse_msh_int4(header_line, path, "MSH v4 node header")
            n_entity_blocks >= 0 && total_nodes >= 0 ||
                error("Negative MSH v4 node count in $path.")
            sizehint!(nodes, Base.checked_add(length(nodes), total_nodes))

            nodes_read = 0
            actual_min_tag = typemax(Int)
            actual_max_tag = typemin(Int)
            for _ in 1:n_entity_blocks
                block_line = _required_msh_line(io, path, "MSH v4 node block header")
                entity_dim, _, parametric, n_nodes_in_block =
                    _parse_msh_int4(block_line, path, "MSH v4 node block header")
                0 <= entity_dim <= 3 ||
                    error("Invalid MSH entity dimension $entity_dim in $path.")
                parametric in (0, 1) ||
                    error("Invalid MSH parametric flag $parametric in $path.")
                n_nodes_in_block >= 0 ||
                    error("Negative MSH v4 node-block count in $path.")

                node_tags = _read_msh_integer_block(
                    io, path, n_nodes_in_block, "MSH v4 node tags")
                for node_tag in node_tags
                    node_tag > 0 || error("MSH node tags must be positive in $path.")
                    haskey(nodes, node_tag) &&
                        error("Duplicate MSH node tag $node_tag in $path.")
                    line = _required_msh_line(io, path, "MSH v4 node coordinates")
                    nodes[node_tag] = _parse_msh_xyz(line, path)
                    actual_min_tag = min(actual_min_tag, node_tag)
                    actual_max_tag = max(actual_max_tag, node_tag)
                end
                nodes_read = Base.checked_add(nodes_read, n_nodes_in_block)
            end
            nodes_read == total_nodes ||
                error("MSH v4 node header declares $total_nodes nodes, read $nodes_read in $path.")
            if total_nodes > 0
                actual_min_tag == min_node_tag && actual_max_tag == max_node_tag ||
                    error("MSH v4 node-tag bounds do not match the header in $path.")
            end
            end_marker = _required_msh_line(io, path, "\$EndNodes")
            end_marker == "\$EndNodes" ||
                error("MSH file $path is missing \$EndNodes.")
        elseif section == "\$Elements"
            header_line = _required_msh_line(io, path, "MSH v4 element header")
            n_entity_blocks, total_elements, min_element_tag, max_element_tag =
                _parse_msh_int4(header_line, path, "MSH v4 element header")
            n_entity_blocks >= 0 && total_elements >= 0 ||
                error("Negative MSH v4 element count in $path.")

            elements_read = 0
            actual_min_tag = typemax(Int)
            actual_max_tag = typemin(Int)
            for _ in 1:n_entity_blocks
                block_line = _required_msh_line(io, path, "MSH v4 element block header")
                _, _, element_type, n_elems_in_block =
                    _parse_msh_int4(block_line, path, "MSH v4 element block header")
                n_elems_in_block >= 0 ||
                    error("Negative MSH v4 element-block count in $path.")
                for _ in 1:n_elems_in_block
                    line = _required_msh_line(io, path, "MSH v4 element")
                    element_tag, triangle =
                        _parse_msh_v4_element(line, path, element_type == 2)
                    element_tag > 0 ||
                        error("MSH element tags must be positive in $path.")
                    actual_min_tag = min(actual_min_tag, element_tag)
                    actual_max_tag = max(actual_max_tag, element_tag)
                    isnothing(triangle) || push!(triangles, triangle)
                end
                elements_read = Base.checked_add(elements_read, n_elems_in_block)
            end
            elements_read == total_elements ||
                error("MSH v4 element header declares $total_elements elements, read $elements_read in $path.")
            if total_elements > 0
                actual_min_tag == min_element_tag &&
                    actual_max_tag == max_element_tag ||
                    error("MSH v4 element-tag bounds do not match the header in $path.")
            end
            end_marker = _required_msh_line(io, path, "\$EndElements")
            end_marker == "\$EndElements" ||
                error("MSH file $path is missing \$EndElements.")
        end
    end

    isempty(nodes) && error("MSH v4 file has no nodes: $path")
    isempty(triangles) && @warn "MSH v4 file has no triangle elements: $path"

    return _build_trimesh_from_msh(nodes, triangles)
end

function _build_trimesh_from_msh(nodes::Dict{Int, NTuple{3,Float64}},
                                  triangles::Vector{NTuple{3,Int}})
    # Remap node IDs to 1-based contiguous
    sorted_ids = sort!(collect(keys(nodes)))

    nv = length(sorted_ids)
    xyz = Matrix{Float64}(undef, 3, nv)
    @inbounds for (new_id, old_id) in enumerate(sorted_ids)
        c = nodes[old_id]
        xyz[1, new_id] = c[1]
        xyz[2, new_id] = c[2]
        xyz[3, new_id] = c[3]
    end

    nt = length(triangles)
    tri = Matrix{Int}(undef, 3, nt)
    contiguous_ids = true
    @inbounds for i in eachindex(sorted_ids)
        if sorted_ids[i] != i
            contiguous_ids = false
            break
        end
    end
    if contiguous_ids
        @inbounds for t in 1:nt
            for v in 1:3
                node_tag = triangles[t][v]
                1 <= node_tag <= nv ||
                    error("MSH triangle references missing node tag $node_tag.")
                tri[v, t] = node_tag
            end
        end
    else
        id_map = Dict{Int,Int}()
        sizehint!(id_map, nv)
        @inbounds for (new_id, old_id) in enumerate(sorted_ids)
            id_map[old_id] = new_id
        end
        @inbounds for t in 1:nt
            for v in 1:3
                node_tag = triangles[t][v]
                new_id = get(id_map, node_tag, 0)
                !iszero(new_id) ||
                    error("MSH triangle references missing node tag $node_tag.")
                tri[v, t] = new_id
            end
        end
    end

    return TriMesh(xyz, tri)
end

# ───────────────────────────────────────────────────────────────
# Unified dispatchers
# ───────────────────────────────────────────────────────────────

"""
    read_mesh(path)

Read a triangle mesh from a file, dispatching by file extension:
- `.obj` → `read_obj_mesh`
- `.stl` → `read_stl_mesh`
- `.msh` → `read_msh_mesh`
"""
function read_mesh(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    if ext == ".obj"
        return read_obj_mesh(path)
    elseif ext == ".stl"
        return read_stl_mesh(path)
    elseif ext == ".msh"
        return read_msh_mesh(path)
    else
        error("Unsupported mesh format '$(ext)'. Supported: .obj, .stl, .msh")
    end
end

"""
    write_mesh(path, mesh; kwargs...)

Write a `TriMesh` to a file, dispatching by file extension:
- `.obj` → `write_obj_mesh`
- `.stl` → `write_stl_mesh`
"""
function write_mesh(path::AbstractString, mesh::TriMesh; kwargs...)
    ext = lowercase(splitext(path)[2])
    if ext == ".obj"
        return write_obj_mesh(path, mesh; kwargs...)
    elseif ext == ".stl"
        return write_stl_mesh(path, mesh; kwargs...)
    else
        error("Unsupported mesh write format '$(ext)'. Supported: .obj, .stl")
    end
end

# ───────────────────────────────────────────────────────────────
# CAD conversion via external gmsh CLI
# ───────────────────────────────────────────────────────────────

"""
    convert_cad_to_mesh(cad_path, output_path; mesh_size=0.0, gmsh_exe="gmsh")

Convert a CAD file (STEP, IGES, BREP) to a triangle surface mesh by
calling the Gmsh CLI. Gmsh must be installed and accessible from PATH
(or provide the full path via `gmsh_exe`).

If `mesh_size > 0`, it is passed as `-clmax` to control the maximum
element size. Otherwise Gmsh uses its default sizing.

Returns the imported `TriMesh`.

**Example:**
```julia
mesh = convert_cad_to_mesh("model.step", "model.msh"; mesh_size=0.01)
```
"""
function convert_cad_to_mesh(cad_path::AbstractString, output_path::AbstractString;
                              mesh_size::Float64=0.0,
                              gmsh_exe::AbstractString="gmsh")
    isfile(cad_path) || error("CAD file not found: $cad_path")

    cad_ext = lowercase(splitext(cad_path)[2])
    cad_ext in (".step", ".stp", ".iges", ".igs", ".brep") ||
        error("Unsupported CAD format '$(cad_ext)'. Supported: .step, .stp, .iges, .igs, .brep")

    out_ext = lowercase(splitext(output_path)[2])
    out_ext in (".msh", ".stl", ".obj") ||
        error("Unsupported output format '$(out_ext)'. Supported: .msh, .stl, .obj")

    # Check gmsh availability
    gmsh_found = try
        success(`$(gmsh_exe) --version`)
    catch
        false
    end
    gmsh_found || error(
        "Gmsh not found at '$(gmsh_exe)'. Install Gmsh (https://gmsh.info) and " *
        "ensure it is on your PATH, or pass the full path via gmsh_exe."
    )

    # Build command
    cmd = [gmsh_exe, "-2", cad_path, "-o", output_path, "-format",
           out_ext == ".msh" ? "msh2" : (out_ext == ".stl" ? "stl" : "obj")]
    if mesh_size > 0
        push!(cmd, "-clmax")
        push!(cmd, string(mesh_size))
    end

    # ignorestatus so a non-zero exit yields our explicit message (with the code)
    # rather than a bare ProcessFailedException thrown by run() before the check.
    result = run(ignorestatus(Cmd(cmd)); wait=true)
    result.exitcode == 0 || error("Gmsh conversion failed (exit code $(result.exitcode)).")
    isfile(output_path) || error("Gmsh did not produce output file: $output_path")

    return read_mesh(output_path)
end
