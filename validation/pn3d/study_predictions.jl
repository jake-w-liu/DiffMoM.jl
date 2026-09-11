include("reference_population.jl")

const STUDY_LEVELS = (0, 1)
const STUDY_PROBE_COUNTS = (0, 2, 8, 24)
const STUDY_MAX_WORK_BYTES = 16_000_000_000
const STUDY_MAX_EXACT_TERMS = 64_000_000
const STUDY_ALGORITHM_VERSION = 1

function _study_main_rows(main_looks)
    return reduce(vcat, ([2look - 1, 2look] for look in main_looks))
end

function _study_field_record(field, look_count)
    values = reshape(Vector{ComplexF64}(field), 2, look_count)
    return (
        real=[real.(collect(column)) for column in eachcol(values)],
        imag=[imag.(collect(column)) for column in eachcol(values)],
    )
end

function _study_scale_record(square_root, look_count)
    size(square_root, 1) == 2look_count ||
        throw(DimensionMismatch("output factor must have two rows per look"))
    values = [norm(view(square_root, 2look - 1:2look, :)) for look in 1:look_count]
    all(value -> isfinite(value) && value >= 0, values) ||
        error("output scales must be finite and nonnegative")
    return values
end

function _study_same_mesh(first_mesh, second_mesh)
    return first_mesh.tri == second_mesh.tri && first_mesh.xyz == second_mesh.xyz
end

function _study_level_solve(mesh, row, source, level)
    method = level <= 1 ? :dense_direct : :aca_gmres
    return solve_scattering(
        mesh, row.frequency_hz, source; method,
        quad_order=7, c0=row.c0_m_s, aca_tol=1e-12, aca_max_rank=256,
        gmres_tol=1e-10, gmres_maxiter=1500, gmres_memory=80,
        true_residual_factor=10.0,
        max_true_residual_exact_terms=STUDY_MAX_EXACT_TERMS,
        max_aca_storage_bytes=STUDY_MAX_WORK_BYTES,
        max_dense_matrix_bytes=STUDY_MAX_WORK_BYTES,
        return_state=true, check_resolution=false, verbose=false,
    )
end

function _study_global_factor(system, output_map)
    coarse_map = output_map * system.pair.P
    adjoint_solution = solve_prepared_adjoint!(
        system.coarse, Matrix(adjoint(coarse_map));
        max_work_bytes=STUDY_MAX_WORK_BYTES)
    factor = norm(system.coarse.rhs) * Matrix(adjoint(adjoint_solution))
    all(isfinite, factor) || error("global-covariance output factor is non-finite")
    return factor
end

function _study_level_prediction(mesh, fine_mesh, state, fine_state, grid, main_rows,
                                 look_count, source_amplitude)
    pair, nesting_s = @timed build_nested_rwg_pair(
        mesh; max_work_bytes=STUDY_MAX_WORK_BYTES)
    _study_same_mesh(pair.fine_mesh, fine_mesh) ||
        error("candidate hierarchy differs from the registered midpoint refinement")
    system, restriction_s = @timed prepare_galerkin_error(
        state, pair, fine_state.operator, fine_state.rhs;
        max_work_bytes=STUDY_MAX_WORK_BYTES)
    k = DiffMoM._frequency_to_wavenumber(
        state.frequency_hz, state.c0, "panel study output map")
    wide_map, radiation_s = @timed rcs_output_map(
        pair.fine_mesh, pair.fine_rwg, grid, k; quad_order=7,
        max_work_bytes=STUDY_MAX_WORK_BYTES, max_exact_work=5_000_000_000)
    output_map = wide_map[main_rows, :]
    coarse_map = output_map * pair.P
    algebraic_coarse_v, algebraic_coarse_s = @timed _study_residual_output_change(
        state, coarse_map, look_count)
    algebraic_fine_v, algebraic_fine_s = @timed _study_residual_output_change(
        fine_state, output_map, look_count)
    restriction_v = _study_look_changes(
        coarse_map, system.coarse.I_coeffs - state.I_coeffs, look_count)
    prepared_outputs, output_rows_s = @timed prepare_error_outputs(
        system, output_map; row_batch_size=24,
        max_work_bytes=STUDY_MAX_WORK_BYTES)
    area = sum(triangle_area(mesh, triangle) for triangle in 1:ntriangles(mesh))
    tau = abs(source_amplitude) * sqrt(area) / IMPEDANCE
    predictions = NamedTuple[]
    for count in STUDY_PROBE_COUNTS
        selected = count == 0 ? Int[] :
            unique(round.(Int, range(1, size(output_map, 1); length=count)))
        length(selected) == count || error("probe schedule contains duplicate output rows")
        probes, probe_s = @timed count == 0 ?
            zeros(ComplexF64, 0, size(pair.Q, 2)) :
            output_residual_probes(
                prepared_outputs; rows=selected,
                max_work_bytes=STUDY_MAX_WORK_BYTES)
        model, conditioning_s = @timed condition_discretization_error(
            system, probes; tau, max_work_bytes=STUDY_MAX_WORK_BYTES)
        outputs, moments_s = @timed evaluate_error_outputs(
            model, prepared_outputs; row_batch_size=24,
            max_work_bytes=STUDY_MAX_WORK_BYTES)
        push!(predictions, (
            probe_count=count,
            effective_rank=model.conditioning.rank,
            mean=_study_field_record(outputs.mean, look_count),
            scales=_study_scale_record(outputs.square_root, look_count),
            observation_residual=model.conditioning.diagnostics.observation_residual,
            costs=(; probe_s, conditioning_s, moments_s),
        ))
    end
    global_factor, global_s = @timed _study_global_factor(system, output_map)
    residual_scale = norm(system.residual) / sqrt(length(system.residual))
    isfinite(residual_scale) && residual_scale >= 0 ||
        error("residual-only scale is invalid")
    return (
        coarse_unknowns=state.rwg.nedges,
        fine_unknowns=fine_state.rwg.nedges,
        unresolved_unknowns=size(pair.Q, 2),
        tau_a=tau,
        coarse_mean=_study_field_record(prepared_outputs.coarse_mean, look_count),
        fine_mean=_study_field_record(output_map * fine_state.I_coeffs, look_count),
        residual_scale_v=residual_scale,
        global_scales=_study_scale_record(global_factor, look_count),
        algebraic_coarse_v, algebraic_fine_v,
        restriction_field_change_v=restriction_v,
        proposed=predictions,
        restriction=system.restriction_report,
        costs=(; nesting_s, restriction_s, radiation_s, output_rows_s, global_s,
            algebraic_coarse_s, algebraic_fine_s),
    )
end

function _study_prediction_protocol(population_dir)
    manifest_path = joinpath(population_dir, "population_manifest.json")
    manifest = JSON.parsefile(manifest_path)
    manifest["seed"] == 20260907 || error("study population seed is not the registered seed")
    manifest["training_cases"] == 60 && manifest["calibration_cases"] == 199 &&
        manifest["test_cases"] == 300 || error("study split counts differ from the contract")
    for (name, digest) in manifest["artifact_sha256"]
        bytes2hex(sha256(read(joinpath(population_dir, name)))) == digest ||
            error("population artifact changed: $name")
    end
    driver_hashes = (
        study_predictions=bytes2hex(sha256(read(@__FILE__))),
        reference_population=bytes2hex(sha256(read(joinpath(@__DIR__, "reference_population.jl")))),
        pilot=bytes2hex(sha256(read(joinpath(@__DIR__, "pilot.jl")))),
    )
    population = (
        seed=manifest["seed"],
        distribution_id=manifest["distribution_id"],
        training_cases=manifest["training_cases"],
        calibration_cases=manifest["calibration_cases"],
        test_cases=manifest["test_cases"],
        look_count=manifest["look_count"],
        mask_count=manifest["mask_count"],
    )
    return (
        schema_version=STUDY_ALGORITHM_VERSION,
        source_sha256=source_digest(),
        driver_hashes,
        population_manifest_sha256=bytes2hex(sha256(read(manifest_path))),
        population,
        candidate_levels=STUDY_LEVELS,
        probe_counts=STUDY_PROBE_COUNTS,
        quadrature_order=7,
        aca_tolerance=1e-12,
        gmres_tolerance=1e-10,
        reference_target="registered level-3 numerical field",
    )
end

function _study_protocol_hash(protocol)
    return bytes2hex(sha256(JSON.json(protocol)))
end

function _study_validate_protocol(protocol, expected_hash)
    _study_protocol_hash(protocol) == expected_hash ||
        error("prediction protocol changed; use a new output directory")
    source_digest() == protocol.source_sha256 ||
        error("package source changed during panel predictions")
    paths = (
        study_predictions=@__FILE__,
        reference_population=joinpath(@__DIR__, "reference_population.jl"),
        pilot=joinpath(@__DIR__, "pilot.jl"),
    )
    for name in keys(paths)
        bytes2hex(sha256(read(getproperty(paths, name)))) ==
            getproperty(protocol.driver_hashes, name) ||
            error("prediction driver changed during execution: $name")
    end
    return nothing
end

function _study_predict_case(row, population_dir, protocol, protocol_hash)
    meshes = reference_case_meshes(row, 0.05, maximum(STUDY_LEVELS) + 1)
    _, source = reference_case_source(row)
    grid, main_looks = reference_observation_grid(population_dir)
    main_rows = _study_main_rows(main_looks)
    states = Any[]
    solves = NamedTuple[]
    for level in 0:(maximum(STUDY_LEVELS) + 1)
        prepared, elapsed_s = @timed _study_level_solve(meshes[level + 1], row, source, level)
        push!(states, prepared.state)
        push!(solves, (
            level,
            method=string(prepared.result.method),
            unknowns=prepared.state.rwg.nedges,
            elapsed_s,
            assembly_s=prepared.result.assembly_time_s,
            solve_s=prepared.result.solve_time_s,
            preconditioner_s=prepared.result.preconditioner_time_s,
        ))
    end
    levels = [_study_level_prediction(
        meshes[level + 1], meshes[level + 2], states[level + 1], states[level + 2],
        grid, main_rows, length(main_looks), row.incident_amplitude_v_m)
        for level in STUDY_LEVELS]
    _study_validate_protocol(protocol, protocol_hash)
    return (
        schema_version=STUDY_ALGORITHM_VERSION,
        case_id=String(row.case_id),
        split=String(row.split),
        protocol_hash,
        status="complete",
        source_unchanged=true,
        levels,
        solves,
    )
end

function study_predictions_main(population_dir, output_dir, split, first_case, last_case)
    split in ("train", "calibration", "test") || error("unknown study split")
    protocol = _study_prediction_protocol(population_dir)
    protocol_hash = _study_protocol_hash(protocol)
    mkpath(output_dir)
    protocol_path = joinpath(output_dir, "prediction_protocol.json")
    if isfile(protocol_path)
        stored = JSON.parsefile(protocol_path)
        stored["protocol_hash"] == protocol_hash ||
            error("stored prediction protocol differs")
    else
        open(protocol_path, "w") do io
            JSON.print(io, (; protocol_hash, protocol), 2)
        end
    end
    rows = [row for row in CSV.File(joinpath(population_dir, "parameters.csv"))
            if row.split == split]
    1 <= first_case <= last_case <= length(rows) || error("invalid case range")
    for index in first_case:last_case
        row = rows[index]
        output_path = joinpath(output_dir, String(row.case_id) * ".json")
        if isfile(output_path)
            previous = JSON.parsefile(output_path)
            previous["protocol_hash"] == protocol_hash ||
                error("cached prediction protocol differs")
            previous["status"] == "complete" && previous["source_unchanged"] && continue
        end
        started = time_ns()
        record = nothing
        try
            record = _study_predict_case(row, population_dir, protocol, protocol_hash)
        catch error
            record = (
                schema_version=STUDY_ALGORITHM_VERSION,
                case_id=String(row.case_id), split=String(row.split), protocol_hash,
                status="incomplete", source_unchanged=source_digest() == protocol.source_sha256,
                failure=(type=string(typeof(error)), message=sprint(showerror, error)),
                elapsed_s=(time_ns() - started) / 1e9,
            )
            temporary = output_path * ".partial"
            open(temporary, "w") do io
                JSON.print(io, record, 2)
            end
            mv(temporary, output_path; force=true)
            rethrow()
        end
        record = merge(record, (elapsed_s=(time_ns() - started) / 1e9,))
        temporary = output_path * ".partial"
        open(temporary, "w") do io
            JSON.print(io, record, 2)
        end
        mv(temporary, output_path; force=true)
        println(row.case_id, ": prediction complete")
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 5 ||
        error("usage: study_predictions.jl POPULATION OUTPUT SPLIT FIRST LAST")
    study_predictions_main(
        abspath(ARGS[1]), abspath(ARGS[2]), ARGS[3],
        parse(Int, ARGS[4]), parse(Int, ARGS[5]))
end
