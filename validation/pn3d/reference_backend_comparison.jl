include("reference_population.jl")

function reference_backend_comparison(population_dir, output_dir, case_id;
        aca_tolerance::Float64=1e-9, mlfma_precisions::Tuple{Int,Int}=(3, 6))
    DiffMoM._validate_aca_options(aca_tolerance, 256)
    1 <= first(mlfma_precisions) < last(mlfma_precisions) ||
        throw(ArgumentError("MLFMA comparison precisions must be positive and increasing"))
    ispath(output_dir) && error("use a new backend-comparison output directory")
    manifest_path = joinpath(population_dir, "population_manifest.json")
    manifest = JSON.parsefile(manifest_path)
    for (name, digest) in manifest["artifact_sha256"]
        bytes2hex(sha256(read(joinpath(population_dir, name)))) == digest ||
            error("population artifact changed: $name")
    end
    cases = [row for row in CSV.File(joinpath(population_dir, "parameters.csv"))
             if row.case_id == case_id]
    length(cases) == 1 || error("case identifier is missing or ambiguous")
    row = only(cases)
    row.split == "train" || error("backend selection uses training cases only")
    source_hash = source_digest()
    dependencies = [@__FILE__, joinpath(@__DIR__, "reference_population.jl"),
                    joinpath(@__DIR__, "pilot.jl")]
    driver_hashes = Dict(basename(path) => bytes2hex(sha256(read(path))) for path in dependencies)
    # This gate compares operator approximations, not interval coverage or
    # continuum accuracy. Per-interval budgets still require separate checks.
    controls = (; level=1, initial_edge_m=0.05, quad_order=7,
        relative_field_tolerance=1e-4, absolute_field_tolerance=1e-10,
        gmres_memory=80, gmres_tol=1e-10, gmres_maxiter=1500,
        true_residual_factor=10.0, max_true_residual_exact_terms=64_000_000,
        max_work_bytes=16_000_000_000, max_workspace_bytes=512*1024*1024,
        max_exact_combine_work=64_000_000,
        aca_tol=aca_tolerance, aca_max_rank=256, mlfma_leaf_lambda=0.25,
        mlfma_precisions, action_probe_count=4, action_probe_seed=20260907)
    rows = NamedTuple[]
    reference = nothing
    failure = nothing
    passed = false
    active = "geometry"
    started = time_ns()
    mkpath(output_dir)
    function checkpoint()
        unchanged = source_hash == source_digest() && all(path ->
            driver_hashes[basename(path)] == bytes2hex(sha256(read(path))), dependencies)
        record = (; schema_version=1, case_id, controls, active,
            status=passed && unchanged ? "passed" : "incomplete",
            source_unchanged=unchanged, source_sha256=source_hash, driver_hashes,
            population_manifest_sha256=bytes2hex(sha256(read(manifest_path))),
            julia_version=string(VERSION), julia_threads=Threads.nthreads(),
            blas_threads=BLAS.get_num_threads(), reference, comparisons=rows, failure,
            elapsed_s=(time_ns()-started)/1e9)
        path = joinpath(output_dir, "comparison.json")
        open(path * ".partial", "w") do io
            JSON.print(io, record, 2)
        end
        mv(path * ".partial", path; force=true)
        unchanged || error("source changed during backend comparison")
    end
    try
        mesh = reference_case_meshes(row, controls.initial_edge_m, controls.level)[end]
        _, source = reference_case_source(row)
        k = DiffMoM._frequency_to_wavenumber(row.frequency_hz, row.c0_m_s, "backend comparison")
        grid, main_looks = reference_observation_grid(population_dir)
        active = "dense reference"
        checkpoint()
        prepared, dense_s = @timed solve_scattering(mesh, row.frequency_hz, source;
            method=:dense_direct, quad_order=controls.quad_order, c0=row.c0_m_s,
            max_true_residual_exact_terms=controls.max_true_residual_exact_terms,
            max_dense_matrix_bytes=controls.max_work_bytes, return_state=true,
            check_resolution=false, verbose=false)
        state = prepared.state
        dense_residual, dense_residual_s = @timed DiffMoM._true_residual_ratio(state.operator, state.I_coeffs,
            state.rhs, "dense backend reference";
            max_true_residual_exact_terms=controls.max_true_residual_exact_terms)
        dense_residual <= 1e-10 || error("dense reference residual exceeds its budget")
        mapping, radiation_s = @timed rcs_output_map(mesh, state.rwg, grid, k;
            quad_order=controls.quad_order, max_work_bytes=controls.max_work_bytes,
            max_exact_work=5_000_000_000)
        reference_field, dense_field_s = @timed vec(DiffMoM._error_action_columns(
            mapping, reshape(state.I_coeffs, :, 1)))
        reference_norm = norm(reference_field)
        isfinite(reference_norm) && reference_norm > 0 || error("invalid dense reference field")
        conditioning, condition_s = @timed condition_diagnostics(state.operator)
        isfinite(conditioning.cond) || error("dense reference condition number is non-finite")
        probes = randn(MersenneTwister(controls.action_probe_seed), ComplexF64,
            state.rwg.nedges, controls.action_probe_count)
        dense_actions = DiffMoM._error_action_columns(state.operator, probes)
        reference = (; dof_count=state.rwg.nedges, relative_linear_residual=dense_residual,
            operator_wavenumber=k, matrix_conditioning=conditioning,
            field_real=real.(reference_field), field_imag=imag.(reference_field),
            theta_rad=grid.theta, phi_rad=grid.phi, main_looks,
            costs=(; total_s=dense_s, assembly_s=prepared.result.assembly_time_s,
                solve_s=prepared.result.solve_time_s, radiation_s,
                residual_s=dense_residual_s, field_s=dense_field_s, condition_s))
        low_name = "mlfma_p$(first(mlfma_precisions))"
        high_name = "mlfma_p$(last(mlfma_precisions))"
        for (name, precision) in (("aca", 0), (low_name, first(mlfma_precisions)),
                                  (high_name, last(mlfma_precisions)))
            active = name * " assembly"
            checkpoint()
            operator, assembly_s = @timed if name == "aca"
                build_aca_operator(mesh, state.rwg, k; aca_tol=controls.aca_tol,
                    max_rank=controls.aca_max_rank, quad_order=controls.quad_order,
                    max_storage_bytes=controls.max_work_bytes)
            else
                build_mlfma_operator(mesh, state.rwg, k; precision,
                    leaf_lambda=controls.mlfma_leaf_lambda, quad_order=controls.quad_order,
                    max_setup_bytes=controls.max_work_bytes,
                    max_nearfield_bytes=controls.max_work_bytes,
                    max_exact_combine_work=controls.max_exact_combine_work)
            end
            preconditioner, preconditioner_s = @timed (
                name == "aca" ? build_nearfield_preconditioner(operator;
                    factorization=:lu, max_triplet_bytes=controls.max_work_bytes) :
                build_nearfield_preconditioner(operator.Z_near; factorization=:lu,
                    max_triplet_bytes=controls.max_work_bytes))
            action_error = NaN
            entry_error = nothing
            diagnostic_s = @elapsed begin
                actions = DiffMoM._error_action_columns(operator, probes)
                action_error = norm(actions - dense_actions) / norm(dense_actions)
                isfinite(action_error) || error("operator-action comparison is non-finite")
                if name == "aca"
                    # Sample exact ACA oracle entries separately from its compressed
                    # actions. This separates cache/assembly consistency from rank error.
                    indices = unique(round.(Int, range(1, state.rwg.nedges;
                        length=min(16, state.rwg.nedges))))
                    pairs = vcat([(i, i) for i in indices], collect(zip(indices, reverse(indices))))
                    entry_error = maximum(abs(efie_entry(operator, i, j) - state.operator[i, j])
                        for (i, j) in pairs)
                end
            end
            active = name * " solve"
            checkpoint()
            (coefficients, stats), solve_s = @timed solve_gmres(operator, state.rhs;
                preconditioner, memory=controls.gmres_memory, tol=controls.gmres_tol,
                maxiter=controls.gmres_maxiter, max_workspace_bytes=controls.max_workspace_bytes,
                true_residual_factor=controls.true_residual_factor,
                max_true_residual_exact_terms=controls.max_true_residual_exact_terms)
            residual, residual_s = @timed DiffMoM._true_residual_ratio(operator, coefficients, state.rhs,
                "backend comparison";
                max_true_residual_exact_terms=controls.max_true_residual_exact_terms)
            field, field_s = @timed vec(DiffMoM._error_action_columns(mapping, reshape(coefficients, :, 1)))
            difference = norm(field - reference_field)
            isfinite(difference) || error("backend field difference is non-finite")
            comparison_passed = difference <= controls.absolute_field_tolerance +
                controls.relative_field_tolerance * reference_norm
            push!(rows, (; backend=name, precision=name == "aca" ? nothing : precision, comparison_passed,
                relative_field_difference=difference/reference_norm,
                absolute_field_difference=difference, relative_linear_residual=residual,
                probed_action_relative_difference=action_error,
                sampled_exact_entry_absolute_difference=entry_error,
                iterations=stats.niter, field_real=real.(field), field_imag=imag.(field),
                costs=(; assembly_s, preconditioner_s, solve_s, residual_s, field_s, diagnostic_s)))
            println(name, ": relative field difference ", difference/reference_norm,
                "; comparison_passed=", comparison_passed)
            flush(stdout)
            operator = nothing
            preconditioner = nothing
            GC.gc()
        end
        # The lower precision is an explicit comparison, not a required success.
        # The higher-precision candidate and ACA must pass the unchanged field gate.
        passed = all(r -> r.comparison_passed, filter(r -> r.backend != low_name, rows))
        passed || error("required backend comparisons exceed the declared field tolerance")
    catch err
        failure = sprint(showerror, err)
        rethrow()
    finally
        checkpoint()
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 3 || error("usage: reference_backend_comparison.jl POPULATION OUTPUT TRAIN_CASE_ID")
    reference_backend_comparison(abspath(ARGS[1]), abspath(ARGS[2]), ARGS[3])
end
