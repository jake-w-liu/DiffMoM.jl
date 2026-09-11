using CSV
using DataFrames
using JSON
using LinearAlgebra
using Printf
using SHA

const REFERENCE_CASE_LABELS = Dict(
    "plate_2x2" => "Plate", "plate_4x4" => "Plate",
    "slotted_panel" => "Bent slotted panel",
    "disk_coarse" => "Disk", "disk_refined" => "Disk",
    "cube_coarse" => "Cube", "cube_refined" => "Cube")

function reference_complex_fields(record, real_key, imag_key)
    reals, imags = record[real_key], record[imag_key]
    reals isa AbstractVector && imags isa AbstractVector &&
        !isempty(reals) && length(reals) == length(imags) ||
        error("complex reference fields require matching nonempty component arrays")
    fields = Matrix{ComplexF64}(undef, 2, length(reals))
    for look in eachindex(reals, imags)
        reals[look] isa AbstractVector && imags[look] isa AbstractVector &&
            length(reals[look]) == length(imags[look]) == 2 ||
            error("each reference look must contain two transverse components")
        for component in 1:2
            x, y = reals[look][component], imags[look][component]
            all(v -> v isa Real && !(v isa Bool) && isfinite(v), (x, y)) ||
                error("reference field components must be finite numbers")
            fields[component, look] = complex(Float64(x), Float64(y))
        end
    end
    all(isfinite, fields) || error("reference field conversion is non-finite")
    return fields
end

"""Recompute a comparison from the immutable bytes of its field and report records."""
function reference_comparison_row(trial_path::AbstractString, report_path::AbstractString)
    trial_text, report_text = read(trial_path, String), read(report_path, String)
    trial, report = JSON.parse(trial_text), JSON.parse(report_text)
    id, order = trial["case_id"], trial["quadrature_order"]
    haskey(REFERENCE_CASE_LABELS, id) || error("unsupported reference geometry")
    order in (7, 28, 112) || error("unsupported reference quadrature")
    anchor_text = order == 28 ? trial_text :
        read(joinpath(dirname(trial_path), "$(id)_q28.json"), String)
    anchor = order == 28 ? trial : JSON.parse(anchor_text)
    anchor["quadrature_order"] == 28 || error("independent reference anchor is not q28")
    for key in ("case_id", "dof_count", "julia_source_sha256", "convention",
                "frequency_hz", "c0_m_s", "incident_amplitude_v_m",
                "incident_wavevector_rad_m", "incident_polarization", "vertices_m",
                "triangles_zero_based", "looks_unit", "theta_rad", "phi_rad")
        trial[key] == anchor[key] || error("reference $key changed across quadrature settings")
    end
    trial["convention"] == "exp(+i*omega*t)" || error("unsupported field convention")
    report["passed"] === true || error("independent comparison did not pass")
    report["input_sha256"] == bytes2hex(sha256(anchor_text)) ||
        error("independent report is not bound to the q28 input")
    for key in ("case_id", "dof_count", "julia_source_sha256")
        report[key] == trial[key] || error("independent report $key differs from its input")
    end
    reference = reference_complex_fields(report,
        "julia_convention_field_real", "julia_convention_field_imag")
    reference == conj.(reference_complex_fields(report,
        "bempp_field_real", "bempp_field_imag")) ||
        error("independent field convention conversion is inconsistent")
    field = reference_complex_fields(trial, "field_real", "field_imag")
    anchor_field = reference_complex_fields(anchor, "field_real", "field_imag")
    size(field) == size(anchor_field) == size(reference) || error("reference look counts differ")
    size(field, 2) == length(trial["looks_unit"]) == length(trial["theta_rad"]) ==
        length(trial["phi_rad"]) || error("fields and look metadata differ")
    reference_norm = norm(reference)
    isfinite(reference_norm) && reference_norm > 0 || error("invalid reference field norm")
    # The comparison CLI calls its supplied Julia q28 field the reference.
    # The manuscript metric instead uses the independently computed field norm.
    input_norm = norm(anchor_field)
    isfinite(input_norm) && input_norm > 0 || error("invalid comparison input field norm")
    anchor_difference = norm(anchor_field - reference)
    # Each norm reduces 24 finite physical components; 32eps covers reduction
    # order differences between the independent Python and Julia calculations.
    for (key, value) in (("reference_field_norm", input_norm),
                         ("field_difference_norm", anchor_difference))
        isapprox(report[key], value; rtol=32eps(Float64), atol=0) ||
            error("independent report $key differs from its complex fields")
    end
    residual = report["relative_linear_residual"]
    relative_limit, absolute_limit = report["relative_field_tolerance"],
                                     report["absolute_field_tolerance"]
    all(v -> v isa Real && !(v isa Bool) && isfinite(v) && v >= 0,
        (residual, relative_limit, absolute_limit)) || error("invalid independent comparison budgets")
    residual <= 1e-10 || error("independent report exceeds the linear-residual budget")
    anchor_difference <= absolute_limit + relative_limit * input_norm ||
        error("independent anchor fields fail their recorded comparison tolerance")
    # Recheck actual fields against the publication budget even when an older
    # producer requested a looser tolerance. Its passed flag is not the gate.
    anchor_difference <= 1e-12 + 0.02 * reference_norm ||
        error("independent anchor fields exceed the publication comparison budget")
    difference = norm(field - reference)
    isfinite(difference) || error("field difference is non-finite")
    return (; case_id=id, dof_count=trial["dof_count"], quadrature_points=order,
        absolute_field_difference=difference,
        relative_field_difference=difference / reference_norm,
        reference_field_norm=reference_norm, independent_linear_residual=residual,
        julia_source_sha256=trial["julia_source_sha256"],
        input_sha256=bytes2hex(sha256(trial_text)),
        independent_report_sha256=bytes2hex(sha256(report_text)))
end

"""Publish the independently checked field-comparison data without changing its provenance."""
function export_validation_table(reference_dir::AbstractString, paper_dir::AbstractString)
    input = joinpath(reference_dir, "quadrature-comparison.csv")
    data = CSV.read(input, DataFrame; strict=true, types=Dict(
        :case_id => String, :julia_source_sha256 => String,
        :input_sha256 => String, :independent_report_sha256 => String))
    required = [:case_id, :dof_count, :quadrature_points,
        :absolute_field_difference, :relative_field_difference,
        :reference_field_norm, :independent_linear_residual,
        :julia_source_sha256, :input_sha256, :independent_report_sha256]
    all(in(propertynames(data)), required) || error("comparison columns are missing")
    nrow(data) > 0 || error("comparison data are empty")
    any(ismissing, Matrix(data[:, required])) && error("comparison data contain missing values")
    length(unique(data.julia_source_sha256)) == 1 || error("mixed Julia source revisions")
    length(unique(zip(data.case_id, data.quadrature_points))) == nrow(data) ||
        error("duplicate case and quadrature setting")
    labels = REFERENCE_CASE_LABELS
    case_set = Set(data.case_id)
    case_set == Set(("plate_2x2", "plate_4x4", "slotted_panel")) ||
        case_set == Set(keys(labels)) || error("unexpected comparison case set")
    for row in eachrow(data)
        for column in (:absolute_field_difference, :relative_field_difference,
                       :reference_field_norm, :independent_linear_residual)
            value = row[column]
            value isa Real && isfinite(value) && value >= 0 ||
                error("invalid $column for $(row.case_id)")
        end
        row.reference_field_norm > 0 || error("zero reference norm")
        row.dof_count isa Integer && row.dof_count > 0 || error("invalid degree count")
        isapprox(row.relative_field_difference,
                 row.absolute_field_difference / row.reference_field_norm;
                 rtol=32eps(Float64), atol=0) || error("inconsistent relative difference")
        for column in (:julia_source_sha256, :input_sha256, :independent_report_sha256)
            occursin(r"^[0-9a-f]{64}$", row[column]) || error("invalid provenance digest")
        end
        trial = joinpath(reference_dir, "composite",
                         "$(row.case_id)_q$(row.quadrature_points).json")
        report = joinpath(reference_dir,
            "bempp-current-$(row.case_id)-julia28-q6s8.json")
        computed = reference_comparison_row(trial, report)
        for column in required
            actual = getproperty(computed, column)
            agrees = column in (:absolute_field_difference, :relative_field_difference,
                                 :reference_field_norm, :independent_linear_residual) ?
                isapprox(row[column], actual; rtol=32eps(Float64), atol=0) : row[column] == actual
            agrees || error("comparison $column differs from its raw fields or provenance")
        end
    end
    cases = sort(unique(data.case_id); by=id -> minimum(data.dof_count[data.case_id .== id]))
    table = IOBuffer()
    println(table, "\\begin{tabular}{lrrrr}")
    println(table, "\\toprule")
    println(table, "Geometry & RWG count & 7 points & 28 points & 112 points \\\\")
    println(table, "\\midrule")
    for id in cases
        subset = sort(data[data.case_id .== id, :], :quadrature_points)
        subset.quadrature_points == [7, 28, 112] || error("incomplete quadrature comparison")
        length(unique(subset.dof_count)) == 1 || error("mesh changed across quadrature settings")
        values = [@sprintf("%.3f", 100x) for x in subset.relative_field_difference]
        println(table, join([labels[id], string(first(subset.dof_count)), values...], " & "), " \\\\")
    end
    println(table, "\\bottomrule\n\\end{tabular}")
    table_text = String(take!(table))
    output = joinpath(paper_dir, "data")
    mkpath(output)
    names = ("independent-fields.csv", "independent-fields-table.tex",
             "independent-fields-provenance.json")
    all(name -> !ispath(joinpath(output, name)), names) ||
        error("refusing to replace an existing published comparison")
    mktempdir(output) do staging
        cp(input, joinpath(staging, names[1]))
        write(joinpath(staging, names[2]), table_text)
        provenance = (; schema_version=1, rows=nrow(data),
            source_csv_sha256=bytes2hex(sha256(read(input))),
            julia_source_sha256=only(unique(data.julia_source_sha256)),
            exporter_sha256=bytes2hex(sha256(read(@__FILE__))),
            metric="100 * norm(Julia field - independent field) / norm(independent field)",
            scope="Same-mesh discrete complex fields; not continuum error or coverage",
            independent_anchor_relative_tolerance=0.02,
            independent_anchor_absolute_tolerance=1e-12,
            independent_linear_residual_limit=1e-10,
            published_table_sha256=bytes2hex(sha256(table_text)))
        open(joinpath(staging, names[3]), "w") do io
            JSON.print(io, provenance, 2)
        end
        # Publish provenance last so an interrupted copy cannot look complete.
        for name in names
            mv(joinpath(staging, name), joinpath(output, name); force=false)
        end
    end
    println("Published $(nrow(data)) field comparisons with checked input and report digests.")
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 2 || error("usage: export_validation_table.jl REFERENCE_DIR PAPER_DIR")
    export_validation_table(abspath(ARGS[1]), abspath(ARGS[2]))
end
