# MeshIO.jl — Multi-format mesh I/O: STL, Gmsh MSH, unified dispatcher, CAD conversion

export read_stl_mesh, write_stl_mesh
export read_msh_mesh
export read_mesh, write_mesh
export convert_cad_to_mesh

# ───────────────────────────────────────────────────────────────
# STL: Binary and ASCII reader/writer
# ───────────────────────────────────────────────────────────────

"""
    read_stl_mesh(path; merge_tol=0.0)

Read a triangle mesh from an STL file (binary or ASCII, auto-detected).

STL stores three vertices per facet with no shared-vertex topology, so
duplicate vertices are merged. With the default `merge_tol=0.0`, vertices
are merged when their Float64 coordinates are bitwise identical (suitable
for most STL files). Set `merge_tol` to a small positive value (e.g.
`1e-10 * bbox_diagonal`) if your exporter introduces tiny floating-point
noise between shared vertices.

Returns a `TriMesh`.
"""
function read_stl_mesh(path::AbstractString; merge_tol::Float64=0.0)
    data = read(path)
    if _stl_is_binary(data)
        return _read_stl_binary(data; merge_tol=merge_tol)
    else
        return _read_stl_ascii(path; merge_tol=merge_tol)
    end
end

function _stl_is_binary(data::Vector{UInt8})
    length(data) < 84 && return false
    ntri = reinterpret(UInt32, data[81:84])[1]
    expected = 84 + 50 * Int(ntri)
    return length(data) == expected
end

function _read_stl_binary(data::Vector{UInt8}; merge_tol::Float64=0.0)
    ntri = Int(reinterpret(UInt32, data[81:84])[1])
    ntri > 0 || error("STL binary file has 0 triangles.")

    raw_verts = Vector{NTuple{3,Float64}}(undef, 3 * ntri)
    offset = 84
    for t in 1:ntri
        # Skip 12 bytes of normal (3 × Float32)
        offset += 12
        for v in 1:3
            fx = reinterpret(Float32, data[offset+1:offset+4])[1]
            fy = reinterpret(Float32, data[offset+5:offset+8])[1]
            fz = reinterpret(Float32, data[offset+9:offset+12])[1]
            raw_verts[3*(t-1) + v] = (Float64(fx), Float64(fy), Float64(fz))
            offset += 12
        end
        offset += 2  # attribute byte count
    end

    return _merge_stl_vertices(raw_verts, ntri; merge_tol=merge_tol)
end

function _read_stl_ascii(path::AbstractString; merge_tol::Float64=0.0)
    raw_verts = NTuple{3,Float64}[]
    ntri = 0

    open(path, "r") do io
        for line in eachline(io)
            s = strip(line)
            if startswith(s, "facet normal")
                ntri += 1
            elseif startswith(s, "vertex")
                fields = split(s)
                length(fields) >= 4 || error("Invalid STL vertex line: $s")
                x = parse(Float64, fields[2])
                y = parse(Float64, fields[3])
                z = parse(Float64, fields[4])
                push!(raw_verts, (x, y, z))
            end
        end
    end

    length(raw_verts) == 3 * ntri ||
        error("STL ASCII: expected $(3*ntri) vertices for $ntri facets, got $(length(raw_verts)).")
    ntri > 0 || error("STL ASCII file has 0 facets: $path")

    return _merge_stl_vertices(raw_verts, ntri; merge_tol=merge_tol)
end

function _merge_stl_vertices(raw_verts::Vector{NTuple{3,Float64}}, ntri::Int;
                              merge_tol::Float64=0.0)
    vertex_map = Dict{NTuple{3,Int64},Int}()   # quantized key → unique vertex id
    xyz_list = NTuple{3,Float64}[]
    tri = Matrix{Int}(undef, 3, ntri)

    if merge_tol <= 0.0
        # Exact merge: use bitwise representation
        for t in 1:ntri
            for v in 1:3
                coord = raw_verts[3*(t-1) + v]
                key = (reinterpret(Int64, coord[1]),
                       reinterpret(Int64, coord[2]),
                       reinterpret(Int64, coord[3]))
                id = get!(vertex_map, key) do
                    push!(xyz_list, coord)
                    length(xyz_list)
                end
                tri[v, t] = id
            end
        end
    else
        # Tolerance-based merge: quantize to grid
        inv_tol = 1.0 / merge_tol
        for t in 1:ntri
            for v in 1:3
                coord = raw_verts[3*(t-1) + v]
                key = (round(Int64, coord[1] * inv_tol),
                       round(Int64, coord[2] * inv_tol),
                       round(Int64, coord[3] * inv_tol))
                id = get!(vertex_map, key) do
                    push!(xyz_list, coord)
                    length(xyz_list)
                end
                tri[v, t] = id
            end
        end
    end

    nv = length(xyz_list)
    xyz = Matrix{Float64}(undef, 3, nv)
    for i in 1:nv
        xyz[1, i] = xyz_list[i][1]
        xyz[2, i] = xyz_list[i][2]
        xyz[3, i] = xyz_list[i][3]
    end

    return TriMesh(xyz, tri)
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

function _write_stl_binary(path::AbstractString, mesh::TriMesh; header::AbstractString="")
    nt = ntriangles(mesh)
    open(path, "w") do io
        # 80-byte header (padded with zeros)
        hdr = Vector{UInt8}(codeunits(header))
        resize!(hdr, 80)
        hdr[length(codeunits(header))+1:end] .= 0x00
        write(io, hdr)
        # Triangle count
        write(io, UInt32(nt))
        for t in 1:nt
            n = triangle_normal(mesh, t)
            write(io, Float32(n[1]), Float32(n[2]), Float32(n[3]))
            for vi in 1:3
                idx = mesh.tri[vi, t]
                write(io, Float32(mesh.xyz[1, idx]),
                          Float32(mesh.xyz[2, idx]),
                          Float32(mesh.xyz[3, idx]))
            end
            write(io, UInt16(0))  # attribute byte count
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
