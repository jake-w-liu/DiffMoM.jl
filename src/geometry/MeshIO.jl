# MeshIO.jl — Multi-format mesh I/O: STL, Gmsh MSH, unified dispatcher, CAD conversion

export read_stl_mesh, write_stl_mesh
export read_msh_mesh
export read_mesh, write_mesh
export convert_cad_to_mesh

# ───────────────────────────────────────────────────────────────
# STL: Binary and ASCII reader/writer
# ───────────────────────────────────────────────────────────────

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
    fields = eachsplit(s)
    item = iterate(fields)
    isnothing(item) && error("Invalid STL vertex line: $s")
    _, state = item

    item = iterate(fields, state)
    isnothing(item) && error("Invalid STL vertex line: $s")
    x_field, state = item
    item = iterate(fields, state)
    isnothing(item) && error("Invalid STL vertex line: $s")
    y_field, state = item
    item = iterate(fields, state)
    isnothing(item) && error("Invalid STL vertex line: $s")
    z_field, _ = item

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

function _write_stl_binary(path::AbstractString, mesh::TriMesh; header::AbstractString="")
    nt = ntriangles(mesh)
    nt <= typemax(UInt32) ||
        throw(ArgumentError("Binary STL supports at most $(typemax(UInt32)) triangles."))
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

Read a triangle surface mesh from a Gmsh MSH file (v2 or v4 ASCII).

Only 3-node triangle elements (Gmsh type 2) are extracted; all other
element types (lines, quads, tetrahedra, etc.) are silently ignored.
Node IDs are remapped to 1-based contiguous indices.

Returns a `TriMesh`.
"""
function read_msh_mesh(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("MSH file is empty: $path")

    # Detect version from $MeshFormat
    version = 0.0
    for (i, l) in enumerate(lines)
        if strip(l) == "\$MeshFormat"
            parts = split(strip(lines[i+1]))
            version = parse(Float64, parts[1])
            break
        end
    end
    version > 0 || error("MSH file missing \$MeshFormat section: $path")

    if version >= 4.0
        return _read_msh_v4(lines, path)
    else
        return _read_msh_v2(lines, path)
    end
end

function _read_msh_v2(lines::Vector{String}, path::AbstractString)
    nodes = Dict{Int, NTuple{3,Float64}}()
    triangles = NTuple{3,Int}[]

    i = 1
    while i <= length(lines)
        s = strip(lines[i])

        if s == "\$Nodes"
            i += 1
            n_nodes = parse(Int, strip(lines[i]))
            for _ in 1:n_nodes
                i += 1
                parts = split(strip(lines[i]))
                nid = parse(Int, parts[1])
                x = parse(Float64, parts[2])
                y = parse(Float64, parts[3])
                z = parse(Float64, parts[4])
                nodes[nid] = (x, y, z)
            end

        elseif s == "\$Elements"
            i += 1
            n_elems = parse(Int, strip(lines[i]))
            for _ in 1:n_elems
                i += 1
                parts = split(strip(lines[i]))
                etype = parse(Int, parts[2])
                if etype == 2  # 3-node triangle
                    ntags = parse(Int, parts[3])
                    offset = 3 + ntags
                    n1 = parse(Int, parts[offset + 1])
                    n2 = parse(Int, parts[offset + 2])
                    n3 = parse(Int, parts[offset + 3])
                    push!(triangles, (n1, n2, n3))
                end
            end
        end
        i += 1
    end

    isempty(nodes) && error("MSH v2 file has no nodes: $path")
    isempty(triangles) && @warn "MSH v2 file has no triangle elements: $path"

    return _build_trimesh_from_msh(nodes, triangles)
end

function _read_msh_v4(lines::Vector{String}, path::AbstractString)
    nodes = Dict{Int, NTuple{3,Float64}}()
    triangles = NTuple{3,Int}[]

    i = 1
    while i <= length(lines)
        s = strip(lines[i])

        if s == "\$Nodes"
            i += 1
            header = split(strip(lines[i]))
            n_entity_blocks = parse(Int, header[1])
            # header[2] = total nodes (not needed for parsing)
            for _ in 1:n_entity_blocks
                i += 1
                block_header = split(strip(lines[i]))
                n_nodes_in_block = parse(Int, block_header[4])
                # Read node tags
                node_tags = Vector{Int}(undef, n_nodes_in_block)
                for k in 1:n_nodes_in_block
                    i += 1
                    node_tags[k] = parse(Int, strip(lines[i]))
                end
                # Read node coordinates
                for k in 1:n_nodes_in_block
                    i += 1
                    parts = split(strip(lines[i]))
                    x = parse(Float64, parts[1])
                    y = parse(Float64, parts[2])
                    z = parse(Float64, parts[3])
                    nodes[node_tags[k]] = (x, y, z)
                end
            end

        elseif s == "\$Elements"
            i += 1
            header = split(strip(lines[i]))
            n_entity_blocks = parse(Int, header[1])
            for _ in 1:n_entity_blocks
                i += 1
                block_header = split(strip(lines[i]))
                etype = parse(Int, block_header[3])
                n_elems_in_block = parse(Int, block_header[4])
                for _ in 1:n_elems_in_block
                    i += 1
                    if etype == 2  # 3-node triangle
                        parts = split(strip(lines[i]))
                        n1 = parse(Int, parts[2])
                        n2 = parse(Int, parts[3])
                        n3 = parse(Int, parts[4])
                        push!(triangles, (n1, n2, n3))
                    end
                end
            end
        end
        i += 1
    end

    isempty(nodes) && error("MSH v4 file has no nodes: $path")
    isempty(triangles) && @warn "MSH v4 file has no triangle elements: $path"

    return _build_trimesh_from_msh(nodes, triangles)
end

function _build_trimesh_from_msh(nodes::Dict{Int, NTuple{3,Float64}},
                                  triangles::Vector{NTuple{3,Int}})
    # Remap node IDs to 1-based contiguous
    sorted_ids = sort(collect(keys(nodes)))
    id_map = Dict{Int,Int}()
    for (new_id, old_id) in enumerate(sorted_ids)
        id_map[old_id] = new_id
    end

    nv = length(sorted_ids)
    xyz = Matrix{Float64}(undef, 3, nv)
    for (old_id, new_id) in id_map
        c = nodes[old_id]
        xyz[1, new_id] = c[1]
        xyz[2, new_id] = c[2]
        xyz[3, new_id] = c[3]
    end

    nt = length(triangles)
    tri = Matrix{Int}(undef, 3, nt)
    for t in 1:nt
        tri[1, t] = id_map[triangles[t][1]]
        tri[2, t] = id_map[triangles[t][2]]
        tri[3, t] = id_map[triangles[t][3]]
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
