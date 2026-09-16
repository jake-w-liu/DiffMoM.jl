include("pilot.jl")
using CSV

function reference_case_meshes(row, initial_edge, max_level)
    mesh = make_bent_slotted_panel(; panel_length=row.main_length_m,
        panel_width=row.main_width_m, flange_width=row.flange_length_m,
        bend_angle=row.bend_angle_rad, slot_length=row.slot_length_m,
        slot_width=row.slot_width_m, max_edge=initial_edge)
    meshes = [mesh]
    base_triangles = ntriangles(mesh)
    for level in 1:max_level
        refinement = refine_mesh_to_target_edge(mesh, initial_edge / 2^level)
        refinement.converged || error("reference refinement stopped: $(refinement.stop_reason)")
        mesh = refinement.mesh
        ntriangles(mesh) == base_triangles * 4^level ||
            error("reference meshes do not form the declared midpoint hierarchy")
        push!(meshes, mesh)
    end
    return meshes
end

function reference_case_source(row)
    k = 2pi * row.frequency_hz / row.c0_m_s
    source = make_plane_wave(k * Vec3(row.incident_dx, row.incident_dy, row.incident_dz),
        row.incident_amplitude_v_m,
        Vec3(row.polarization_x, row.polarization_y, row.polarization_z))
    return k, source
end

"""
Solve `state.operator * correction = residual` without mutating the state.

The residual RHS is already as small as the forward solve's own tolerance, so
the iterative correction is verified against the ORIGINAL right-hand side
norm: the achieved true residual must stay within `factor * tol * |rhs|`,
matching the accuracy the forward solve established. Relative checks against
`|residual|` would demand an unattainable absolute accuracy.
"""
function _study_state_correction(state, residual, label)
    if state.factorization !== nothing
        return DiffMoM._solve_factored_linear_system(
            state.factorization, state.operator, residual, label)
    end
    residual_scale = norm(residual)
    rhs_scale = norm(state.rhs)
    residual_scale > 0 && rhs_scale > 0 ||
        return zeros(ComplexF64, length(residual))
    options = state.solver_options
    allowed = options.true_residual_factor * options.tol * rhs_scale /
              residual_scale
    correction, _ = solve_gmres(state.operator, residual;
        preconditioner=state.preconditioner, precond_side=:left,
        tol=min(0.1, 0.1allowed), maxiter=options.maxiter,
        memory=options.memory,
        check_gmres_convergence=options.check_gmres_convergence,
        check_true_residual=false,
        true_residual_factor=options.true_residual_factor,
        max_true_residual_exact_terms=options.max_true_residual_exact_terms,
        max_workspace_bytes=512 * 1024 * 1024)
    achieved = DiffMoM._true_residual_ratio(
        state.operator, correction, residual, label;
        max_true_residual_exact_terms=options.max_true_residual_exact_terms)
    isfinite(achieved) && achieved <= allowed ||
        error("$label correction residual exceeds the forward-solve budget: " *
              "relative_residual=$achieved, allowed=$allowed")
    return correction
end

"""
Measured per-look output change in volts induced by a coefficient change
`delta`, where `output_map` supplies two transverse rows per look.
"""
function _study_look_changes(output_map, delta, look_count)
    field = vec(DiffMoM._error_action_columns(output_map, reshape(delta, :, 1)))
    values = reshape(field, 2, look_count)
    changes = [norm(view(values, :, look)) for look in 1:look_count]
    all(value -> isfinite(value) && value >= 0, changes) ||
        error("algebraic output change is invalid")
    return changes
end

"""
Measured per-look output change in volts from the selected solve's algebraic
residual. The correction solve reuses the retained factor or preconditioner
without mutating the retained state.
"""
function _study_residual_output_change(state, output_map, look_count)
    residual = state.rhs - vec(DiffMoM._error_action_columns(
        state.operator, reshape(state.I_coeffs, :, 1)))
    correction = _study_state_correction(
        state, residual, "reference algebraic correction")
    return _study_look_changes(output_map, correction, look_count)
end

"""Return the 36-look grid and indices of the 12 exact registered observations."""
function reference_observation_grid(population_dir)
    small_grid = make_sph_grid(3, 4)
    grid = make_sph_grid(3, 12)
    main_looks = Int[]
    for j in axes(small_grid.rhat, 2)
        distances = [norm(grid.rhat[:, i] - small_grid.rhat[:, j]) for i in axes(grid.rhat, 2)]
        nearest = argmin(distances)
        distances[nearest] <= 1e-12 || error("wide look grid does not contain the main looks")
        push!(main_looks, nearest)
        # Preserve the registered coordinates exactly, not just the same angles
        # to within the rounding of two independently generated angular grids.
        grid.rhat[:, nearest] = small_grid.rhat[:, j]
        grid.theta[nearest] = small_grid.theta[j]
        grid.phi[nearest] = small_grid.phi[j]
    end
    length(unique(main_looks)) == 12 || error("main look mapping is not one-to-one")
    registered_looks = collect(CSV.File(joinpath(population_dir, "looks.csv")))
    length(registered_looks) == length(main_looks) || error("registered look count differs")
    for j in eachindex(registered_looks)
        row = registered_looks[j]
        row.look_id == j || error("registered look identifiers are not in order")
        small_grid.theta[j] == row.theta_rad && small_grid.phi[j] == row.phi_rad ||
            error("main look definition differs from the registered population")
        Tuple(small_grid.rhat[:, j]) == (row.direction_x, row.direction_y, row.direction_z) ||
            error("main look directions differ from the registered population")
    end
    return grid, main_looks
end

function population_reference_main(population_dir, output_dir, split, first_case, last_case)
    split in ("train", "calibration", "test") || error("unknown population split")
    manifest_path = joinpath(population_dir, "population_manifest.json")
    population = JSON.parsefile(manifest_path)
    for (name, expected_hash) in population["artifact_sha256"]
        bytes2hex(sha256(read(joinpath(population_dir, name)))) == expected_hash ||
            error("population artifact changed: $name")
    end
    cases = [row for row in CSV.File(joinpath(population_dir, "parameters.csv"))
             if row.split == split]
    1 <= first_case <= last_case <= length(cases) || error("invalid case range")
    code_hash = source_digest()
    driver_hash = bytes2hex(sha256(read(@__FILE__)))
    pilot_path = joinpath(@__DIR__, "pilot.jl")
    pilot_hash = bytes2hex(sha256(read(pilot_path)))
    controls = (; max_level=3, quad_order=7, gmres_tol=1e-10, gmres_maxiter=1500,
        gmres_memory=80, true_residual_factor=10.0, aca_tol=1e-9, aca_max_rank=256,
        max_true_residual_exact_terms=64_000_000,
        radiation_exact_work_limit=5_000_000_000, initial_edge_m=0.05,
        max_work_bytes=16_000_000_000, max_aca_storage_bytes=16_000_000_000,
        preconditioner_triplet_bytes=4_000_000_000)
    grid, main_looks = reference_observation_grid(population_dir)
    protocol = (; schema_version=1, code_hash, driver_hash, controls,
        helper_sha256=pilot_hash,
        julia_version=string(VERSION), main_looks,
        theta_rad=grid.theta, phi_rad=grid.phi,
        population_manifest_sha256=bytes2hex(sha256(read(manifest_path))))
    protocol_hash = bytes2hex(sha256(JSON.json(protocol)))
    mkpath(output_dir)
    protocol_path = joinpath(output_dir, "reference_protocol.json")
    if isfile(protocol_path)
        JSON.parsefile(protocol_path)["protocol_hash"] == protocol_hash ||
            error("reference protocol changed; use a new output directory")
    else
        open(protocol_path, "w") do io
            JSON.print(io, (; protocol_hash, protocol), 2)
        end
    end
    for index in first_case:last_case
        row = cases[index]
        case_id = String(row.case_id)
        output_path = joinpath(output_dir, case_id * ".json")
        if isfile(output_path)
            previous = JSON.parsefile(output_path)
            previous["protocol_hash"] == protocol_hash || error("cached case protocol differs")
            if previous["status"] == "complete"
                previous["source_unchanged"] && length(previous["levels"]) == controls.max_level + 1 ||
                    error("cached reference case is incomplete or invalidated")
                all(i -> previous["levels"][i]["level"] == i - 1,
                    eachindex(previous["levels"])) || error("cached reference levels are out of order")
                continue
            end
        end
        levels = NamedTuple[]
        started = time_ns()
        active_level = -1
        failure = nothing
        completed = false
        try
            meshes = reference_case_meshes(row, controls.initial_edge_m, controls.max_level)
            k, source = reference_case_source(row)
            for level in 0:controls.max_level
                active_level = level
                mesh = meshes[level + 1]
                method = level <= 1 ? :dense_direct : :aca_gmres
                (result, state), elapsed = @timed solve_scattering(mesh, row.frequency_hz, source;
                    method=method, quad_order=controls.quad_order, c0=row.c0_m_s,
                    aca_tol=controls.aca_tol, aca_max_rank=controls.aca_max_rank,
                    max_aca_storage_bytes=controls.max_aca_storage_bytes,
                    max_dense_matrix_bytes=controls.max_work_bytes,
                    gmres_tol=controls.gmres_tol, gmres_maxiter=controls.gmres_maxiter,
                    gmres_memory=controls.gmres_memory,
                    true_residual_factor=controls.true_residual_factor,
                    max_true_residual_exact_terms=controls.max_true_residual_exact_terms,
                    max_triplet_bytes=controls.preconditioner_triplet_bytes,
                    return_state=true,
                    check_resolution=false, verbose=false)
                residual = DiffMoM._true_residual_ratio(state.operator, state.I_coeffs, state.rhs,
                    "population reference";
                    max_true_residual_exact_terms=controls.max_true_residual_exact_terms)
                isfinite(residual) && residual <= controls.gmres_tol * controls.true_residual_factor ||
                    error("reference solve failed its selected-operator residual test")
                G, radiation_s = @timed rcs_output_map(mesh, state.rwg, grid, k;
                    quad_order=controls.quad_order, max_work_bytes=controls.max_work_bytes,
                    max_exact_work=controls.radiation_exact_work_limit)
                fields = reshape(vec(DiffMoM._error_action_columns(
                    G, reshape(state.I_coeffs, :, 1))), 2, :)
                algebraic_v, algebraic_s = @timed _study_residual_output_change(
                    state, G, size(fields, 2))
                push!(levels, (; level, triangles=ntriangles(mesh), vertices=nvertices(mesh),
                    dof_count=state.rwg.nedges, method=string(result.method),
                    relative_linear_residual=residual,
                    algebraic_field_change_v=algebraic_v,
                    field_real=[real.(collect(v)) for v in eachcol(fields)],
                    field_imag=[imag.(collect(v)) for v in eachcol(fields)],
                    costs=(; forward_total_s=elapsed, assembly_s=result.assembly_time_s,
                        solve_s=result.solve_time_s, preconditioner_s=result.preconditioner_time_s,
                        radiation_s, algebraic_s)))
                println(case_id, ": level ", level, ", ", state.rwg.nedges, " unknowns")
                flush(stdout)
                state = nothing
                G = nothing
                GC.gc()
            end
            completed = true
        catch error
            failure = (; type=string(typeof(error)), message=sprint(showerror, error), active_level)
            rethrow()
        finally
            unchanged = source_digest() == code_hash &&
                        bytes2hex(sha256(read(@__FILE__))) == driver_hash &&
                        bytes2hex(sha256(read(pilot_path))) == pilot_hash
            record = (; schema_version=1, case_id, protocol_hash,
                status=unchanged && completed ? "complete" : "incomplete", source_unchanged=unchanged,
                numerical_qualification="pending convergence and independent-reference assessment",
                levels, failure, total_elapsed_s=(time_ns() - started) / 1e9)
            temporary = output_path * ".partial"
            open(temporary, "w") do io
                JSON.print(io, record, 2)
            end
            mv(temporary, output_path; force=true)
            unchanged || error("source changed during reference generation")
        end
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 5 || error("usage: reference_population.jl POPULATION OUTPUT SPLIT FIRST LAST")
    population_reference_main(abspath(ARGS[1]), abspath(ARGS[2]), ARGS[3],
        parse(Int, ARGS[4]), parse(Int, ARGS[5]))
end
