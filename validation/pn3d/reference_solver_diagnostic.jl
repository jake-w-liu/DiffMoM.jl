include("reference_population.jl")

function reference_solver_diagnostic(population_dir, output_dir, case_id)
    BLAS.set_num_threads(1)
    ispath(output_dir) && error("use a new diagnostic output directory")
    population = JSON.parsefile(joinpath(population_dir, "population_manifest.json"))
    for (name, digest) in population["artifact_sha256"]
        bytes2hex(sha256(read(joinpath(population_dir, name)))) == digest ||
            error("population artifact changed: $name")
    end
    cases = [row for row in CSV.File(joinpath(population_dir, "parameters.csv")) if row.case_id == case_id]
    length(cases) == 1 || error("case identifier is missing or ambiguous")
    row = only(cases)
    source_hash = source_digest()
    driver_hash = bytes2hex(sha256(read(@__FILE__)))
    helper_path = joinpath(@__DIR__, "reference_population.jl")
    helper_hash = bytes2hex(sha256(read(helper_path)))
    pilot_path = joinpath(@__DIR__, "pilot.jl")
    pilot_hash = bytes2hex(sha256(read(pilot_path)))
    grid, main_looks = reference_observation_grid(population_dir)
    controls = (; level=3, initial_edge_m=0.05, quad_order=7,
        aca_tol=1e-9, aca_max_rank=256, max_storage_bytes=16_000_000_000,
        max_workspace_bytes=512*1024*1024,
        tolerance=1e-10, true_residual_factor=10.0,
        maxiter=1500, max_true_residual_exact_terms=64_000_000)
    attempts = NamedTuple[]
    costs = Dict{String,Float64}()
    failure = nothing
    completed = false
    active = "geometry"
    solution = nothing
    started = time_ns()
    mkpath(output_dir)
    function checkpoint()
        unchanged = source_digest() == source_hash &&
            bytes2hex(sha256(read(@__FILE__))) == driver_hash &&
            bytes2hex(sha256(read(helper_path))) == helper_hash &&
            bytes2hex(sha256(read(pilot_path))) == pilot_hash
        record = (; schema_version=1, case_id, controls, active,
            status=completed && unchanged ? "solved" : "incomplete",
            source_unchanged=unchanged, source_sha256=source_hash,
            driver_sha256=driver_hash, helper_sha256=helper_hash,
            pilot_sha256=pilot_hash,
            preconditioner="LU of ACA inadmissible blocks", attempts, costs,
            solution, failure, elapsed_s=(time_ns()-started)/1e9)
        path = joinpath(output_dir, "solver.json")
        temporary = path * ".partial"
        open(temporary, "w") do io
            JSON.print(io, record, 2)
        end
        mv(temporary, path; force=true)
        unchanged || error("source changed during the solver diagnostic")
    end
    try
        meshes, costs["mesh_s"] = @timed reference_case_meshes(
            row, controls.initial_edge_m, controls.level)
        mesh = meshes[end]
        rwg = build_rwg(mesh)
        k, source = reference_case_source(row)
        active = "ACA assembly"
        checkpoint()
        operator, costs["assembly_s"] = @timed build_aca_operator(mesh, rwg, k;
            aca_tol=controls.aca_tol, max_rank=controls.aca_max_rank,
            quad_order=controls.quad_order, max_storage_bytes=controls.max_storage_bytes)
        rhs, costs["rhs_s"] = @timed assemble_excitation(mesh, rwg, source; quad_order=controls.quad_order)
        active = "ACA-block preconditioner"
        checkpoint()
        preconditioner, costs["preconditioner_s"] = @timed build_nearfield_preconditioner(
            operator; factorization=:lu)
        # The high-level ACA path uses inadmissible blocks, not a distance
        # cutoff. These attempts change actual Krylov controls and reuse setup.
        for (memory, side) in ((80, :left), (80, :right), (160, :right))
            active = "GMRES memory=$memory side=$side"
            checkpoint()
            attempt_start = time_ns()
            try
                coefficients, stats = solve_gmres(operator, rhs;
                    preconditioner=preconditioner, precond_side=side,
                    tol=controls.tolerance, maxiter=controls.maxiter, memory=memory,
                    max_workspace_bytes=controls.max_workspace_bytes,
                    true_residual_factor=controls.true_residual_factor,
                    max_true_residual_exact_terms=controls.max_true_residual_exact_terms,
                    verbose=true)
                residual = DiffMoM._true_residual_ratio(operator, coefficients, rhs,
                    "reference solver diagnostic";
                    max_true_residual_exact_terms=controls.max_true_residual_exact_terms)
                push!(attempts, (; memory, side=string(side), status="passed",
                    iterations=stats.niter, relative_residual=residual,
                    seconds=(time_ns()-attempt_start)/1e9))
                mapping, costs["radiation_s"] = @timed rcs_output_map(mesh, rwg, grid, k;
                    quad_order=controls.quad_order, max_work_bytes=controls.max_storage_bytes,
                    max_exact_work=5_000_000_000)
                field = reshape(vec(DiffMoM._error_action_columns(mapping,
                    reshape(coefficients, :, 1))), 2, :)
                solution = (; dof_count=rwg.nedges, triangles=ntriangles(mesh),
                    preconditioner_nnz_ratio=preconditioner.nnz_ratio,
                    theta_rad=grid.theta, phi_rad=grid.phi, main_looks,
                    field_real=[real.(collect(v)) for v in eachcol(field)],
                    field_imag=[imag.(collect(v)) for v in eachcol(field)],
                    numerical_qualification="pending compression and mesh-convergence assessment")
                completed = true
                break
            catch err
                message = sprint(showerror, err)
                eligible = err isa ErrorException && (
                    startswith(message, "forward GMRES did not converge consistently") ||
                    startswith(message, "forward GMRES true residual too large"))
                eligible || rethrow()
                push!(attempts, (; memory, side=string(side), status="failed", message,
                    seconds=(time_ns()-attempt_start)/1e9))
                checkpoint()
            end
        end
        completed || error("all declared reference-solver attempts failed")
    catch err
        failure = sprint(showerror, err)
        rethrow()
    finally
        checkpoint()
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 3 || error("usage: reference_solver_diagnostic.jl POPULATION OUTPUT CASE_ID")
    reference_solver_diagnostic(abspath(ARGS[1]), abspath(ARGS[2]), ARGS[3])
end
