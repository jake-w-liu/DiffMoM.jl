using JSON, CSV, SHA, LinearAlgebra

function reference_field_matrix(data, real_key, imag_key)
    real_rows, imag_rows = data[real_key], data[imag_key]
    length(real_rows) == length(imag_rows) > 0 ||
        throw(DimensionMismatch("complex field arrays have different or empty look counts"))
    all(row -> length(row) == 2, real_rows) &&
        all(row -> length(row) == 2, imag_rows) ||
        throw(DimensionMismatch("each look must have two transverse field components"))
    values = [ComplexF64(real_rows[j][c], imag_rows[j][c])
              for c in 1:2, j in eachindex(real_rows)]
    all(isfinite, values) || throw(ArgumentError("reference fields must be finite"))
    return values
end

function reference_comparison_main(root, output)
    physical_keys = ("convention", "vertices_m", "triangles_zero_based", "looks_unit",
        "theta_rad", "phi_rad", "frequency_hz", "c0_m_s", "incident_amplitude_v_m",
        "incident_wavevector_rad_m", "incident_polarization", "dof_count")
    rows = NamedTuple[]
    for case_id in ("plate_2x2", "plate_4x4", "slotted_panel")
        reference_path = joinpath(root, "bempp-current-" * case_id * "-julia28-q6s8.json")
        reference = JSON.parsefile(reference_path)
        reference["case_id"] == case_id || error("independent reference case ID differs")
        base_path = joinpath(root, "composite", case_id * "_q28.json")
        bytes2hex(sha256(read(base_path))) == reference["input_sha256"] ||
            error("independent reference input fingerprint differs")
        base = JSON.parsefile(base_path)
        expected = reference_field_matrix(reference,
            "julia_convention_field_real", "julia_convention_field_imag")
        reference_norm = norm(expected)
        reference_norm > 0 || error("a zero reference field needs an absolute-error study")
        for points in (7, 28, 112)
            path = joinpath(root, "composite", case_id * "_q" * string(points) * ".json")
            trial = JSON.parsefile(path)
            all(key -> trial[key] == base[key], physical_keys) ||
                error("physical problem changed for $case_id at $points points")
            trial["julia_source_sha256"] == base["julia_source_sha256"] ||
                error("Julia source changed across the quadrature comparison")
            actual = reference_field_matrix(trial, "field_real", "field_imag")
            size(actual) == size(expected) || throw(DimensionMismatch("look counts differ"))
            difference = norm(actual - expected)
            row = (; case_id, dof_count=Int(trial["dof_count"]), quadrature_points=points,
                absolute_field_difference=difference,
                relative_field_difference=difference / reference_norm,
                reference_field_norm=reference_norm,
                independent_linear_residual=Float64(reference["relative_linear_residual"]),
                julia_source_sha256=String(trial["julia_source_sha256"]),
                input_sha256=bytes2hex(sha256(read(path))),
                independent_report_sha256=bytes2hex(sha256(read(reference_path))))
            push!(rows, row)
            println(case_id, ", ", points, " points: ", row.relative_field_difference)
        end
    end
    mkpath(dirname(output))
    CSV.write(output, rows)
    return rows
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 2 || error("usage: compare_reference_fields.jl REFERENCE_DIRECTORY OUTPUT_CSV")
    reference_comparison_main(ARGS[1], ARGS[2])
end
