using Test
include(joinpath(@__DIR__, "..", "pn3d", "generate_population.jl"))

@testset "Specified population registration" begin
    mktempdir() do temporary
        first_dir, second_dir = joinpath(temporary, "first"), joinpath(temporary, "second")
        first_manifest = generate_panel_population(first_dir)
        second_manifest = generate_panel_population(second_dir)
        @test first_manifest.seed == second_manifest.seed == 20260907
        @test first_manifest.artifact_sha256 == second_manifest.artifact_sha256
        @test (first_manifest.training_cases, first_manifest.calibration_cases,
               first_manifest.test_cases) == (60, 199, 300)
        rows = collect(CSV.File(joinpath(first_dir, "parameters.csv")))
        @test length(rows) == length(unique(row.case_id for row in rows)) == 559
        @test count(row -> row.split == "train", rows) == 60
        @test count(row -> row.split == "calibration", rows) == 199
        @test count(row -> row.split == "test", rows) == 300
        oracle = MersenneTwister(20260907)
        for row in rows
            u_angle, u_length, u_width = rand(oracle, 3)
            @test row.bend_angle_rad == (60 + 60u_angle) * pi / 180
            @test row.slot_length_m == 0.08 + 0.08u_length
            @test row.slot_width_m == 0.01 + 0.02u_width
        end
        @test length(collect(CSV.File(joinpath(first_dir, "looks.csv")))) == 12
        masks = collect(CSV.File(joinpath(first_dir, "masks.csv")))
        @test length(masks) == 21 * 12
        @test length(unique(row.mask_id for row in masks)) == 21
        @test_throws Base.IOError generate_panel_population(first_dir)
        for (name, digest) in first_manifest.artifact_sha256
            @test bytes2hex(sha256(read(joinpath(first_dir, name)))) == digest
        end
    end
end
