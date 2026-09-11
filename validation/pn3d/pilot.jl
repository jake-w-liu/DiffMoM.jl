using DiffMoM
using LinearAlgebra
using Random
using JSON
using SHA

BLAS.set_num_threads(1)
const FREQUENCY = 3.0e9
const SPEED = 299792458.0
const IMPEDANCE = 376.730313668
const WORK_BYTES = parse(Int, get(ENV, "PN_WORK_BYTES", "8000000000"))
const OUTPUT_DIR = isempty(ARGS) ? joinpath(@__DIR__, "..", "..", "data", "pn3d_pilot") : abspath(ARGS[1])
const EDGE = parse(Float64, get(ENV, "PN_PILOT_EDGE", "0.05"))
const RADIATION_EXACT_WORK = parse(Int, get(ENV, "PN_RADIATION_EXACT_WORK", "64000000"))

function source_wave()
    direction = normalize(Vec3(-0.2, -0.3, -1.0))
    polarization = normalize(cross(direction, Vec3(0.0, 0.0, 1.0)))
    return make_plane_wave((2pi * FREQUENCY / SPEED) * direction, 1.0, polarization)
end

function spatial_probe_rows(pair, count)
    m = size(pair.Q, 2)
    count == 0 && return Int[]
    count <= m || throw(ArgumentError("pilot probe count exceeds the unresolved space"))
    function center(column)
        edge = rowvals(pair.Q)[first(nzrange(pair.Q, column))]
        first_vertex, second_vertex = pair.fine_rwg.evert[:, edge]
        return Tuple(DiffMoM._safe_edge_midpoint(
            Vec3(pair.fine_mesh.xyz[:, first_vertex]),
            Vec3(pair.fine_mesh.xyz[:, second_vertex])))
    end
    ordered = sortperm(1:m; by=center)
    return ordered[round.(Int, range(1, m; length=count))]
end

function source_digest()
    root = dirname(pathof(DiffMoM))
    paths = sort([joinpath(directory, file) for (directory, _, files) in walkdir(root)
                  for file in files if endswith(file, ".jl")])
    records = [relpath(path, root) * ":" * bytes2hex(sha256(read(path))) for path in paths]
    return bytes2hex(sha256(join(records, "\n")))
end

function run_case(mesh, source; counts)
    grid = make_sph_grid(3, 4)
    k = 2pi * FREQUENCY / SPEED
    started = time_ns()
    prepared = solve_scattering(
        mesh, FREQUENCY, source; method=:dense_direct, return_state=true,
        verbose=false, max_dense_matrix_bytes=WORK_BYTES)
    coarse_total_s = (time_ns() - started) / 1e9
    pair_time = @elapsed pair = build_nested_rwg_pair(mesh; max_work_bytes=WORK_BYTES)
    assembly_time = @elapsed fine_matrix = assemble_Z_efie(
        pair.fine_mesh, pair.fine_rwg, k; max_output_bytes=WORK_BYTES)
    rhs_time = @elapsed fine_rhs = assemble_excitation(pair.fine_mesh, pair.fine_rwg, source)
    output_map_time = @elapsed output_map = rcs_output_map(
        pair.fine_mesh, pair.fine_rwg, grid, k; max_work_bytes=WORK_BYTES,
        max_exact_work=RADIATION_EXACT_WORK)
    system = prepare_galerkin_error(
        prepared.state, pair, fine_matrix, fine_rhs; max_work_bytes=WORK_BYTES)
    prepared_outputs = prepare_error_outputs(
        system, output_map; max_work_bytes=WORK_BYTES)
    fine_factor_time = @elapsed fine_factor = DiffMoM._factor_dense_linear_system(
        fine_matrix, ComplexF64, "pilot fine factor")
    fine_solve_time = @elapsed fine_current = DiffMoM._solve_factored_linear_system(
        fine_factor, fine_matrix, fine_rhs, "pilot fine solution")
    fine_field = output_map * fine_current
    norm(fine_field) > 0 || error("pilot needs a nonzero reference field for relative-error reporting")
    area = sum(triangle_area(mesh, triangle) for triangle in 1:ntriangles(mesh))
    tau = sqrt(area) / IMPEDANCE
    records = NamedTuple[]
    for count in counts
        rows = spatial_probe_rows(pair, count)
        model = condition_discretization_error(
            system, rows; tau=tau, max_work_bytes=WORK_BYTES)
        outputs = evaluate_error_outputs(model, prepared_outputs; max_work_bytes=WORK_BYTES)
        push!(records, (
            rule="spatial", probes=count, rank=model.conditioning.rank,
            field_error=norm(outputs.mean - fine_field),
            relative_field_error=norm(outputs.mean - fine_field) / norm(fine_field),
            coarse_relative_field_error=norm(outputs.coarse_mean - fine_field) / norm(fine_field),
            output_std_norm=norm(outputs.square_root),
            conditioning_s=model.conditioning_s,
            output_moments_s=outputs.diagnostics.elapsed_s,
            observation_residual=model.conditioning.diagnostics.observation_residual))
        if count > 0
            selected = round.(Int, range(1, size(output_map, 1);
                                         length=min(count, size(output_map, 1))))
            probe_time = @elapsed probes = output_residual_probes(
                prepared_outputs; rows=selected, max_work_bytes=WORK_BYTES)
            informed = condition_discretization_error(
                system, probes; tau=tau, max_work_bytes=WORK_BYTES)
            informed_outputs = evaluate_error_outputs(
                informed, prepared_outputs; max_work_bytes=WORK_BYTES)
            push!(records, (
                rule="output_diagonal", probes=size(probes, 1), rank=informed.conditioning.rank,
                field_error=norm(informed_outputs.mean - fine_field),
                relative_field_error=norm(informed_outputs.mean - fine_field) / norm(fine_field),
                coarse_relative_field_error=norm(informed_outputs.coarse_mean - fine_field) / norm(fine_field),
                output_std_norm=norm(informed_outputs.square_root),
                conditioning_s=probe_time + informed.conditioning_s,
                output_moments_s=informed_outputs.diagnostics.elapsed_s,
                observation_residual=informed.conditioning.diagnostics.observation_residual))
        end
    end
    return (
        coarse_unknowns=prepared.result.N, fine_unknowns=pair.fine_rwg.nedges,
        unresolved_unknowns=size(pair.Q, 2), area_m2=area, frequency_hz=FREQUENCY,
        coarse_total_s=coarse_total_s, pair_s=pair_time,
        fine_assembly_s=assembly_time, fine_rhs_s=rhs_time,
        radiation_map_s=output_map_time, consistent_preparation_s=system.preparation_s,
        output_rows_s=prepared_outputs.preparation_s,
        fine_factor_s=fine_factor_time, fine_solve_s=fine_solve_time,
        restriction=system.restriction_report, records=records)
end

using SparseArrays: rowvals, nzrange
function main()
    initial_digest = source_digest()
    source = source_wave()
    # Compile the same method/type paths on a small physical mesh before
    # application timing. These pilot times are not publication benchmarks.
    run_case(make_rect_plate(0.05, 0.04, 1, 1), source; counts=(0, 2))
    mesh = make_bent_slotted_panel(max_edge=EDGE)
    result = run_case(mesh, source; counts=(0, 2, 8, 24))
    mkpath(OUTPUT_DIR)
    record = (
        kind="pn3d_pilot", julia_version=string(VERSION), threads=Threads.nthreads(),
        blas=string(BLAS.get_config()), blas_threads=BLAS.get_num_threads(),
        source_sha256=initial_digest, source_unchanged=initial_digest == source_digest(),
        driver_sha256=bytes2hex(sha256(read(@__FILE__))), max_work_bytes=WORK_BYTES,
        radiation_exact_work_limit=RADIATION_EXACT_WORK,
        max_edge_setting_m=EDGE, result=result)
    open(joinpath(OUTPUT_DIR, "pilot.json"), "w") do io
        JSON.print(io, record, 2)
    end
    println(JSON.json(record))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
