include("export_validation_table.jl")
using Test

length(ARGS) in (1, 2) || error("usage: test_validation_table.jl REFERENCE_DIR [EXPANDED_REFERENCE_DIR]")
reference_dir = abspath(ARGS[1])
@testset "independent table publication and provenance failures" begin
    mktempdir() do temp
        paper = joinpath(temp, "paper")
        export_validation_table(reference_dir, paper)
        published = joinpath(paper, "data")
        csv_file = joinpath(published, "independent-fields.csv")
        table_file = joinpath(published, "independent-fields-table.tex")
        manifest = JSON.parsefile(joinpath(published, "independent-fields-provenance.json"))
        data = CSV.read(csv_file, DataFrame)
        text = read(table_file, String)
        @test manifest["rows"] == nrow(data) == 9
        @test manifest["source_csv_sha256"] == bytes2hex(sha256(read(csv_file)))
        @test manifest["published_table_sha256"] == bytes2hex(sha256(text))
        @test all(x -> occursin(@sprintf("%.3f", 100x), text), data.relative_field_difference)
        @test_throws ErrorException export_validation_table(reference_dir, paper)
        @test read(table_file, String) == text
        copied = joinpath(temp, "references")
        mkpath(copied)
        cp(joinpath(reference_dir, "composite"), joinpath(copied, "composite"))
        for id in unique(data.case_id)
            name = "bempp-current-$(id)-julia28-q6s8.json"
            cp(joinpath(reference_dir, name), joinpath(copied, name))
        end
        table_csv = joinpath(copied, "quadrature-comparison.csv")
        CSV.write(table_csv, vcat(data, data[1:1, :]))
        @test_throws ErrorException export_validation_table(copied, joinpath(temp, "duplicate"))
        @test !ispath(joinpath(temp, "duplicate"))
        broken = copy(data)
        broken.relative_field_difference[1] = 0.0
        CSV.write(table_csv, broken)
        @test_throws ErrorException export_validation_table(copied, joinpath(temp, "ratio"))
        broken = copy(data)
        broken.absolute_field_difference[1] *= 2
        broken.relative_field_difference[1] *= 2
        CSV.write(table_csv, broken)
        @test_throws ErrorException export_validation_table(copied, joinpath(temp, "consistent-tamper"))
        @test !ispath(joinpath(temp, "consistent-tamper"))
        broken = copy(data)
        broken.dof_count .+= 1
        CSV.write(table_csv, broken)
        @test_throws ErrorException export_validation_table(copied, joinpath(temp, "degrees"))
        broken = copy(data)
        broken.julia_source_sha256 .= repeat("a", 64)
        CSV.write(table_csv, broken)
        @test_throws ErrorException export_validation_table(copied, joinpath(temp, "source"))
        CSV.write(table_csv, data)
        input = joinpath(copied, "composite", "plate_2x2_q7.json")
        original_input = read(input, String)
        open(input, "a") do io
            write(io, "\n")
        end
        @test_throws ErrorException export_validation_table(copied, joinpath(temp, "changed-input"))
        @test !ispath(joinpath(temp, "changed-input"))
        input_row = findfirst((data.case_id .== "plate_2x2") .& (data.quadrature_points .== 7))
        for defect in (:components, :geometry)
            record = JSON.parse(original_input)
            if defect == :components
                record["field_real"][1] = [0.0]
            else
                record["vertices_m"][1][1] += 0.001
            end
            write(input, JSON.json(record))
            broken = copy(data)
            broken.input_sha256[input_row] = bytes2hex(sha256(read(input)))
            CSV.write(table_csv, broken)
            @test_throws ErrorException export_validation_table(copied, joinpath(temp, string(defect)))
            @test !ispath(joinpath(temp, string(defect)))
        end
        write(input, original_input)
        report_path = joinpath(copied, "bempp-current-plate_2x2-julia28-q6s8.json")
        original_report = read(report_path, String)
        for defect in (:failed_report, :wrong_input_norm)
            report = JSON.parse(original_report)
            if defect == :failed_report
                report["passed"] = false
            else
                report["reference_field_norm"] = norm(reference_complex_fields(report,
                    "julia_convention_field_real", "julia_convention_field_imag"))
            end
            write(report_path, JSON.json(report))
            broken = copy(data)
            broken.independent_report_sha256[broken.case_id .== "plate_2x2"] .=
                bytes2hex(sha256(read(report_path)))
            CSV.write(table_csv, broken)
            @test_throws ErrorException export_validation_table(copied, joinpath(temp, string(defect)))
            @test !ispath(joinpath(temp, string(defect)))
        end
        report = JSON.parse(original_report)
        for key in ("bempp_field_real", "bempp_field_imag",
                    "julia_convention_field_real", "julia_convention_field_imag")
            report[key] = [0.5 .* components for components in report[key]]
        end
        anchor_path = joinpath(copied, "composite", "plate_2x2_q28.json")
        anchor = reference_complex_fields(JSON.parsefile(anchor_path), "field_real", "field_imag")
        independent = reference_complex_fields(report,
            "julia_convention_field_real", "julia_convention_field_imag")
        report["field_difference_norm"] = norm(anchor - independent)
        report["relative_field_tolerance"] = 1.0
        write(report_path, JSON.json(report))
        @test_throws ErrorException reference_comparison_row(anchor_path, report_path)
        CSV.write(table_csv, data[1:0, :])
        @test_throws ErrorException export_validation_table(copied, joinpath(temp, "empty"))
    end
end

if length(ARGS) == 2
    expanded = abspath(ARGS[2])
    @testset "Seven-geometry publication against raw independent fields" begin
        mktempdir() do paper
            export_validation_table(expanded, paper)
            data = CSV.read(joinpath(paper, "data", "independent-fields.csv"), DataFrame)
            @test nrow(data) == 21
            @test Set(data.case_id) == Set(keys(REFERENCE_CASE_LABELS))
            for row in eachrow(data)
                trial = JSON.parsefile(joinpath(expanded, "composite",
                    "$(row.case_id)_q$(row.quadrature_points).json"))
                report = JSON.parsefile(joinpath(expanded,
                    "bempp-current-$(row.case_id)-julia28-q6s8.json"))
                field = complex.(vcat(trial["field_real"]...), vcat(trial["field_imag"]...))
                reference = conj.(complex.(vcat(report["bempp_field_real"]...),
                                           vcat(report["bempp_field_imag"]...)))
                expected = sqrt(sum(abs2, field - reference) / sum(abs2, reference))
                @test isapprox(row.relative_field_difference, expected; rtol=32eps(Float64))
            end
        end
    end
end
