using Test
include(joinpath(@__DIR__, "..", "pn3d", "export_reference_cases.jl"))

@testset "Independent-reference geometry and resonance contracts" begin
    for side in (0.02, 0.04)
        mesh = reference_cube_mesh(side)
        for level in 0:2
            quality = mesh_quality_report(mesh)
            @test ntriangles(mesh) == 12 * 4^level
            @test build_rwg(mesh).nedges == 18 * 4^level
            @test quality.n_vertices - quality.n_edges_total + quality.n_triangles == 2
            @test quality.n_boundary_edges == 0
            @test quality.n_nonmanifold_edges == 0
            @test quality.n_orientation_conflicts == 0
            @test quality.n_duplicate_triangles == 0
            area = 0.0
            volume = 0.0
            for triangle in eachcol(mesh.tri)
                a, b, c = (mesh.xyz[:, vertex] for vertex in triangle)
                normal = cross(b - a, c - a)
                @test dot(normal, (a + b + c) / 3) > 0
                area += norm(normal) / 2
                volume += dot(a, cross(b, c)) / 6
            end
            # Roundoff budgets scale with the independent sum's operation count.
            @test isapprox(area, 6side^2; rtol=8ntriangles(mesh) * eps())
            @test isapprox(volume, side^3; rtol=16ntriangles(mesh) * eps())
            level < 2 && (mesh = reference_refine_once(mesh))
        end
    end

    radius, sectors = 0.04, 24
    disk = make_circular_plate(radius, 2, sectors)
    polygon_area = sectors * radius^2 * sin(2pi / sectors) / 2
    for level in 0:1
        quality = mesh_quality_report(disk)
        @test ntriangles(disk) == 72 * 4^level
        @test quality.n_boundary_edges == sectors * 2^level
        @test quality.n_vertices - quality.n_edges_total + quality.n_triangles == 1
        @test quality.n_nonmanifold_edges == 0
        @test quality.n_orientation_conflicts == 0
        @test all(iszero, disk.xyz[3, :])
        area = sum(norm(cross(disk.xyz[:, t[2]] - disk.xyz[:, t[1]],
                             disk.xyz[:, t[3]] - disk.xyz[:, t[1]])) / 2
                   for t in eachcol(disk.tri))
        @test isapprox(area, polygon_area; rtol=8ntriangles(disk) * eps())
        level == 0 && (disk = reference_refine_once(disk))
    end

    # Independent mode enumeration rejects a missing mode-index or 2pi factor.
    modes = [SPEED * sqrt(m^2 + n^2 + p^2) / (2 * 0.04)
             for m in 0:3 for n in 0:3 for p in 0:3
             if count(>(0), (m, n, p)) >= 2]
    guard = cube_resonance_guard(0.04, FREQUENCY, SPEED)
    @test isapprox(guard.first_cavity_frequency_hz, minimum(modes); rtol=4eps())
    @test 0 < guard.frequency_ratio < 1
    @test cube_resonance_guard(0.08, FREQUENCY / 2, SPEED).frequency_ratio ==
          guard.frequency_ratio
    @test_throws ArgumentError cube_resonance_guard(
        0.04, guard.first_cavity_frequency_hz, SPEED)
    @test_throws ArgumentError cube_resonance_guard(
        0.04, 1.01guard.first_cavity_frequency_hz, SPEED)
    for invalid in (0.0, -0.04, Inf, NaN)
        @test_throws ArgumentError reference_cube_mesh(invalid)
        @test_throws ArgumentError cube_resonance_guard(invalid, FREQUENCY, SPEED)
        @test_throws ArgumentError cube_resonance_guard(0.04, invalid, SPEED)
        @test_throws ArgumentError cube_resonance_guard(0.04, FREQUENCY, invalid)
    end
    @test_throws ArgumentError reference_refine_once(
        reference_cube_mesh(0.04); max_output_bytes=1)
    mktempdir() do directory
        for selection in ("unknown", "disk_coarse,disk_coarse", "")
            withenv("PN_REF_OUTPUT" => directory, "PN_REF_CASES" => selection) do
                @test_throws ArgumentError reference_export_main()
                @test isempty(readdir(directory))
            end
        end
    end
end
