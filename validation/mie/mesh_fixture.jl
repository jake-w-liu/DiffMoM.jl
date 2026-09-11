using DiffMoM: TriMesh, Vec3
using LinearAlgebra: norm

# Shared projected-icosphere fixture for the Mie and finite-space validations.
function make_icosphere(radius::Float64; subdivisions::Int=2,
                        max_triangles::Int=1_000_000)
    isfinite(radius) && radius > 0 || throw(ArgumentError("radius must be finite and positive"))
    subdivisions >= 0 || throw(ArgumentError("subdivisions must be nonnegative"))
    max_triangles >= 20 || throw(ArgumentError("max_triangles must permit the 20 initial faces"))
    # Bound all refinement work before allocating a mesh, including huge orders.
    face_count = 20
    for _ in 1:subdivisions
        face_count <= div(max_triangles, 4) ||
            throw(ArgumentError("icosphere exceeds max_triangles=$max_triangles"))
        face_count *= 4
    end
    phi_gold = (1 + sqrt(5.0)) / 2
    verts0 = [
        (-1.0,  phi_gold, 0.0), ( 1.0,  phi_gold, 0.0),
        (-1.0, -phi_gold, 0.0), ( 1.0, -phi_gold, 0.0),
        ( 0.0, -1.0, phi_gold), ( 0.0,  1.0, phi_gold),
        ( 0.0, -1.0,-phi_gold), ( 0.0,  1.0,-phi_gold),
        ( phi_gold, 0.0, -1.0), ( phi_gold, 0.0,  1.0),
        (-phi_gold, 0.0, -1.0), (-phi_gold, 0.0,  1.0),
    ]
    faces = [
        (1,12,6), (1,6,2), (1,2,8), (1,8,11), (1,11,12),
        (2,6,10), (6,12,5), (12,11,3), (11,8,7), (8,2,9),
        (4,10,5), (4,5,3), (4,3,7), (4,7,9), (4,9,10),
        (5,10,6), (3,5,12), (7,3,11), (9,7,8), (10,9,2),
    ]
    verts = [Vec3(v...) / norm(Vec3(v...)) for v in verts0]

    for _ in 1:subdivisions
        edge_mid = Dict{Tuple{Int,Int},Int}()
        new_faces = NTuple{3,Int}[]
        function midpoint_index(i::Int, j::Int)
            key = i < j ? (i, j) : (j, i)
            haskey(edge_mid, key) && return edge_mid[key]
            vmid = (verts[i] + verts[j]) / 2
            vmid /= norm(vmid)
            push!(verts, vmid)
            edge_mid[key] = length(verts)
            return length(verts)
        end
        for (i, j, k) in faces
            a = midpoint_index(i, j)
            b = midpoint_index(j, k)
            c = midpoint_index(k, i)
            push!(new_faces, (i, a, c))
            push!(new_faces, (j, b, a))
            push!(new_faces, (k, c, b))
            push!(new_faces, (a, b, c))
        end
        faces = new_faces
    end

    Nv = length(verts)
    Nt = length(faces)
    xyz = zeros(3, Nv)
    tri = zeros(Int, 3, Nt)
    for i in 1:Nv
        xyz[:, i] = radius .* verts[i]
    end
    for t in 1:Nt
        tri[1, t] = faces[t][1]
        tri[2, t] = faces[t][2]
        tri[3, t] = faces[t][3]
    end
    return TriMesh(xyz, tri)
end
