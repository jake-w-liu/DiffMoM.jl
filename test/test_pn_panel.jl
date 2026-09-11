using Test
using LinearAlgebra
using DiffMoM

@testset "Bent/slotted panel topology and geometric invariants" begin
    for angle in (pi/3, pi/2, 2pi/3, pi)
        mesh = make_bent_slotted_panel(bend_angle=angle, max_edge=0.06)
        @test isapprox(sum(triangle_area(mesh, t) for t in 1:ntriangles(mesh)),
                       0.4 * 0.3 + 0.15 * 0.3 - 0.12 * 0.02; rtol=1e-13)
        @test Set(vec(mesh.tri)) == Set(1:nvertices(mesh))
        adjacency = DiffMoM._build_edge_triangle_map(mesh)
        @test all(length(triangles) in (1, 2) for triangles in values(adjacency))
        @test nvertices(mesh) - length(adjacency) + ntriangles(mesh) == 0
        boundary = Dict{Int,Vector{Int}}()
        for (edge, triangles) in adjacency
            length(triangles) == 1 || continue
            push!(get!(boundary, edge[1], Int[]), edge[2])
            push!(get!(boundary, edge[2], Int[]), edge[1])
        end
        @test all(length(neighbors) == 2 for neighbors in values(boundary))
        unseen = Set(keys(boundary))
        components = 0
        while !isempty(unseen)
            components += 1
            pending = [first(unseen)]
            while !isempty(pending)
                vertex = pop!(pending)
                vertex in unseen || continue
                delete!(unseen, vertex)
                append!(pending, boundary[vertex])
            end
        end
        @test components == 2
        report = mesh_resolution_report(mesh, 3e9)
        @test report.edge_max_m <= 0.06 * (1 + 64eps(Float64))
        if angle == pi
            @test iszero(maximum(abs, mesh.xyz[3, :]))
            @test isapprox(maximum(mesh.xyz[1, :]), 0.35; atol=eps(Float64))
        else
            @test maximum(mesh.xyz[3, :]) > 0
            @test minimum(mesh.xyz[3, :]) == 0
        end
    end
    @test_throws ArgumentError make_bent_slotted_panel(slot_length=0.4)
    @test_throws ArgumentError make_bent_slotted_panel(slot_width=0.3)
    @test_throws ArgumentError make_bent_slotted_panel(bend_angle=0)
    @test_throws ArgumentError make_bent_slotted_panel(bend_angle=NaN)
    @test_throws ArgumentError make_bent_slotted_panel(bend_angle=nextfloat(0.0))
    @test_throws ArgumentError make_bent_slotted_panel(max_work_bytes=1)
    @test_throws ArgumentError make_bent_slotted_panel(max_vertices=10)
    @test_throws ArgumentError make_bent_slotted_panel(max_triangles=10)
    @test_throws ArgumentError make_bent_slotted_panel(max_edge=nextfloat(0.0))

    # Closed fixed-facet nesting is a separate case from the open-panel tests.
    tetrahedron = TriMesh(
        [1.0 -1 -1 1; 1 -1 1 -1; 1 1 -1 -1] / 20,
        [1 1 1 2; 3 2 4 3; 2 4 3 4])
    pair = build_nested_rwg_pair(tetrahedron)
    coefficients = ComplexF64.(1:pair.coarse_rwg.nedges)
    injected = pair.P * coefficients
    for triangle in 1:ntriangles(pair.fine_mesh)
        parent = pair.parent_triangles[triangle]
        point = triangle_center(pair.fine_mesh, triangle)
        coarse_field = sum(coefficients[e] * eval_rwg(pair.coarse_rwg, e, point, parent)
                           for e in 1:pair.coarse_rwg.nedges)
        fine_field = sum(injected[e] * eval_rwg(pair.fine_rwg, e, point, triangle)
                         for e in 1:pair.fine_rwg.nedges)
        @test isapprox(coarse_field, fine_field; rtol=1e-12, atol=1e-12)
    end
end
