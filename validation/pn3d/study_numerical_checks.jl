include("reference_population.jl")

# Measured non-discretization output changes shared by every screened case.
# Quadrature is measured at level 1 with the dense backend: the quad-7 and
# quad-28 results differ in operator, excitation, and output map. Compression
# is measured at level 2: the ACA operator at the study tolerance against the
# dense operator at unchanged quadrature. Both records are empirical error
# budgets in volts at the 12 registered looks, not continuum-error bounds.

function _study_main_look_fields(mesh, rwg, coefficients, grid, k, main_looks;
        quad_order, max_work_bytes)
    mapping = rcs_output_map(mesh, rwg, grid, k;
        quad_order, max_work_bytes, max_exact_work=5_000_000_000)
    field = reshape(vec(DiffMoM._error_action_columns(
        mapping, reshape(coefficients, :, 1))), 2, :)
    return field[:, main_looks]
end

function _study_look_differences(first_field, second_field)
    size(first_field) == size(second_field) ||
        throw(DimensionMismatch("quadrature/compression fields differ in shape"))
    return [norm(view(first_field, :, look) - view(second_field, :, look))
            for look in axes(first_field, 2)]
end

function study_numerical_checks(population_dir, output_path, case_id)
    ispath(output_path) && error("use a new numerical-checks output path")
    manifest_path = joinpath(population_dir, "population_manifest.json")
    manifest = JSON.parsefile(manifest_path)
    for (name, digest) in manifest["artifact_sha256"]
        bytes2hex(sha256(read(joinpath(population_dir, name)))) == digest ||
            error("population artifact changed: $name")
    end
    cases = [row for row in CSV.File(joinpath(population_dir, "parameters.csv"))
             if row.case_id == case_id]
    length(cases) == 1 || error("check case identifier is missing or ambiguous")
    row = only(cases)
    row.split == "train" || error("numerical checks use training cases only")
    source_hash = source_digest()
    dependencies = [@__FILE__, joinpath(@__DIR__, "reference_population.jl"),
                    joinpath(@__DIR__, "pilot.jl")]
    driver_hashes = Dict(basename(path) => bytes2hex(sha256(read(path)))
                         for path in dependencies)
    grid, main_looks = reference_observation_grid(population_dir)
    controls = (; quadrature_level=1, quadrature_orders=(7, 28),
        compression_level=2, quad_order=7, aca_tol=1e-12, aca_max_rank=256,
        gmres_tol=1e-10, gmres_maxiter=1500, gmres_memory=80,
        true_residual_factor=10.0,
        max_true_residual_exact_terms=64_000_000,
        max_work_bytes=16_000_000_000, initial_edge_m=0.05)
    record = nothing
    failure = nothing
    completed = false
    costs = Dict{String,Float64}()
    try
        k = DiffMoM._frequency_to_wavenumber(
            row.frequency_hz, row.c0_m_s, "study numerical checks")
        _, source = reference_case_source(row)
        meshes, costs["mesh_s"] = @timed reference_case_meshes(
            row, controls.initial_edge_m, controls.compression_level)
        quad_mesh = meshes[controls.quadrature_level + 1]
        quad_rwg = build_rwg(quad_mesh)
        quad_solves = map(controls.quadrature_orders) do order
            prepared, elapsed = @timed solve_scattering(
                quad_mesh, row.frequency_hz, source;
                method=:dense_direct, quad_order=order, c0=row.c0_m_s,
                max_dense_matrix_bytes=controls.max_work_bytes,
                return_state=true, check_resolution=false, verbose=false)
            costs["quadrature_$(order)_solve_s"] = elapsed
            field, map_s = @timed _study_main_look_fields(
                quad_mesh, quad_rwg, prepared.state.I_coeffs, grid, k, main_looks;
                quad_order=order, max_work_bytes=controls.max_work_bytes)
            costs["quadrature_$(order)_map_s"] = map_s
            field
        end
        quadrature_v = _study_look_differences(quad_solves...)
        compression_mesh = meshes[controls.compression_level + 1]
        compression_rwg = build_rwg(compression_mesh)
        dense, costs["compression_dense_s"] = @timed solve_scattering(
            compression_mesh, row.frequency_hz, source;
            method=:dense_direct, quad_order=controls.quad_order,
            c0=row.c0_m_s, max_dense_matrix_bytes=controls.max_work_bytes,
            return_state=true, check_resolution=false, verbose=false)
        compressed, costs["compression_aca_s"] = @timed solve_scattering(
            compression_mesh, row.frequency_hz, source;
            method=:aca_gmres, quad_order=controls.quad_order, c0=row.c0_m_s,
            aca_tol=controls.aca_tol, aca_max_rank=controls.aca_max_rank,
            gmres_tol=controls.gmres_tol, gmres_maxiter=controls.gmres_maxiter,
            gmres_memory=controls.gmres_memory,
            true_residual_factor=controls.true_residual_factor,
            max_true_residual_exact_terms=controls.max_true_residual_exact_terms,
            max_aca_storage_bytes=controls.max_work_bytes,
            max_dense_matrix_bytes=controls.max_work_bytes,
            return_state=true, check_resolution=false, verbose=false)
        dense_field, costs["compression_map_s"] = @timed _study_main_look_fields(
            compression_mesh, compression_rwg, dense.state.I_coeffs,
            grid, k, main_looks;
            quad_order=controls.quad_order, max_work_bytes=controls.max_work_bytes)
        compressed_field = _study_main_look_fields(
            compression_mesh, compression_rwg, compressed.state.I_coeffs,
            grid, k, main_looks;
            quad_order=controls.quad_order, max_work_bytes=controls.max_work_bytes)
        compression_v = _study_look_differences(dense_field, compressed_field)
        record = (
            schema_version=1, case_id=String(row.case_id), controls,
            source_sha256=source_hash, driver_hashes,
            population_manifest_sha256=bytes2hex(sha256(read(manifest_path))),
            julia_version=string(VERSION), main_looks,
            quadrature_v, compression_v, costs,
        )
        completed = true
    catch error
        failure = sprint(showerror, error)
        rethrow()
    finally
        unchanged = source_digest() == source_hash && all(
            path -> driver_hashes[basename(path)] == bytes2hex(sha256(read(path))),
            dependencies)
        outcome = merge(
            record === nothing ? NamedTuple() : record,
            (status=completed && unchanged ? "complete" : "incomplete",
             source_unchanged=unchanged, failure))
        temporary = output_path * ".partial"
        open(temporary, "w") do io
            JSON.print(io, outcome, 2)
        end
        mv(temporary, output_path; force=true)
        unchanged || error("source changed during numerical checks")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 3 ||
        error("usage: study_numerical_checks.jl POPULATION NEW_OUTPUT_JSON TRAIN_CASE_ID")
    study_numerical_checks(abspath(ARGS[1]), abspath(ARGS[2]), ARGS[3])
end
