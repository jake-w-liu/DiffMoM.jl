include("pilot.jl")

function reference_refine_once(mesh::TriMesh; max_output_bytes::Int=WORK_BYTES)
    result = DiffMoM._midpoint_refine_once(mesh, max_output_bytes)
    result.mesh === nothing && throw(ArgumentError(
        "reference midpoint refinement stopped: $(result.stop_reason)"))
    return result.mesh
end

function reference_cube_mesh(side_length::Real)
    side = DiffMoM._positive_finite_length("cube side length", side_length)
    # Eight corners and two outward-oriented triangles per planar cube face.
    xyz = (side / 2) .* Float64[
        -1  1  1 -1 -1  1  1 -1;
        -1 -1  1  1 -1 -1  1  1;
        -1 -1 -1 -1  1  1  1  1]
    tri = Int[
        1 1 5 5 1 1 4 4 1 1 2 2;
        3 4 6 7 2 6 8 7 5 8 3 7;
        2 3 7 8 6 5 7 3 8 4 7 6]
    mesh = TriMesh(xyz, tri)
    quality = mesh_quality_report(mesh)
    all(iszero, (quality.n_boundary_edges, quality.n_nonmanifold_edges,
        quality.n_orientation_conflicts, quality.n_invalid_vertices,
        quality.n_invalid_triangles, quality.n_degenerate_triangles,
        quality.n_duplicate_triangles)) || throw(ArgumentError(
            "reference cube must be a nondegenerate, consistently oriented closed mesh"))
    return mesh
end

function cube_resonance_guard(side_length::Real, frequency_hz::Real, c0_m_s::Real)
    side, frequency, speed = Float64.((side_length, frequency_hz, c0_m_s))
    all(x -> isfinite(x) && x > 0, (side, frequency, speed)) ||
        throw(ArgumentError("cube side, frequency, and wave speed must be finite and positive"))
    # A PEC cavity needs at least two nonzero mode indices. For a cube,
    # MIT 6.013, Chapter 9, Eq. (9.4.4) gives f_first = c / (sqrt(2) * side).
    first_frequency = (speed / side) / sqrt(2.0)
    isfinite(first_frequency) && first_frequency > 0 ||
        throw(ArgumentError("cube cavity frequency is not representable"))
    frequency < first_frequency || throw(ArgumentError(
        "closed-cube reference frequency must be below its first cavity resonance"))
    return (; criterion="below_first_cavity_resonance", side_length_m=side,
        first_cavity_frequency_hz=first_frequency,
        frequency_ratio=frequency / first_frequency,
        reference="https://ocw.mit.edu/courses/6-013-electromagnetics-and-applications-spring-2009/318e025511f2c95bb95b6322b30f9c06_MIT6_013S09_chap09.pdf#page=34")
end

function reference_export_main()
    BLAS.set_num_threads(1)
    quad_order = parse(Int, get(ENV, "PN_REF_QUAD_ORDER", "3"))
    tri_quad_rule(quad_order)
    exact_work = parse(Int, get(ENV, "PN_REF_MAX_EXACT_WORK", string(RADIATION_EXACT_WORK)))
    exact_work > 0 || throw(ArgumentError("PN_REF_MAX_EXACT_WORK must be positive"))
    output = get(ENV, "PN_REF_OUTPUT", joinpath(OUTPUT_DIR, "reference-cases"))
    mkpath(output)
    source = source_wave()
    grid = make_sph_grid(3, 4)
    digest = source_digest()
    driver_digest = bytes2hex(sha256(read(@__FILE__)))
    pilot_path = joinpath(@__DIR__, "pilot.jl")
    pilot_digest = bytes2hex(sha256(read(pilot_path)))
    disk = make_circular_plate(0.04, 2, 24)
    cube_side = 0.04
    cube = reference_refine_once(reference_cube_mesh(cube_side))
    cube_guard = cube_resonance_guard(cube_side, FREQUENCY, SPEED)
    cases = (
        ("plate_2x2", make_rect_plate(0.08, 0.06, 2, 2)),
        ("plate_4x4", make_rect_plate(0.08, 0.06, 4, 4)),
        ("slotted_panel", make_bent_slotted_panel(; max_edge=EDGE)),
        ("disk_coarse", disk),
        ("disk_refined", reference_refine_once(disk)),
        ("cube_coarse", cube),
        ("cube_refined", reference_refine_once(cube)),
    )
    # Preserve the existing default campaign; the additional cases are opt-in.
    requested = strip.(split(get(ENV, "PN_REF_CASES",
        "plate_2x2,plate_4x4,slotted_panel"), ","))
    !isempty(requested) && length(unique(requested)) == length(requested) &&
        all(name -> name in first.(cases), requested) ||
        throw(ArgumentError("PN_REF_CASES must select unique names from $(first.(cases))"))
    for (case_id, mesh) in cases
        case_id in requested || continue
        (result, state), solve_s = @timed solve_scattering(
            mesh, FREQUENCY, source; method=:dense_direct, quad_order=quad_order,
            c0=SPEED, check_resolution=false, return_state=true, verbose=false)
        residual, residual_s = @timed DiffMoM._true_residual_ratio(
            state.operator, state.I_coeffs, state.rhs, "independent reference export")
        # Match the predeclared selected-operator budget of the sphere campaign;
        # this checks algebraic accuracy, not the remaining spatial error.
        isfinite(residual) && residual <= 1e-10 ||
            error("reference export exceeds the relative linear-residual budget")
        resonance_guard = startswith(case_id, "cube_") ? cube_guard : nothing
        conditioning, conditioning_s = @timed (
            resonance_guard === nothing ? nothing : condition_diagnostics(state.operator))
        conditioning === nothing || isfinite(conditioning.cond) ||
            error("closed-object reference matrix is singular or numerically unresolved")
        G, radiation_s = @timed rcs_output_map(
            mesh, state.rwg, grid, 2pi * FREQUENCY / SPEED;
            quad_order=quad_order, eta0=IMPEDANCE, max_work_bytes=WORK_BYTES,
            max_exact_work=exact_work)
        values = reshape(vec(DiffMoM._error_action_columns(
            G, reshape(state.I_coeffs, :, 1))), 2, :)
        digest == source_digest() &&
            driver_digest == bytes2hex(sha256(read(@__FILE__))) &&
            pilot_digest == bytes2hex(sha256(read(pilot_path))) ||
            error("source or driver changed during reference export")
        record = (; schema_version=1, case_id,
            convention="exp(+i*omega*t)", frequency_hz=FREQUENCY, c0_m_s=SPEED,
            incident_amplitude_v_m=source.E0,
            incident_wavevector_rad_m=collect(source.k_vec),
            incident_polarization=collect(source.pol),
            vertices_m=[collect(v) for v in eachcol(mesh.xyz)],
            triangles_zero_based=[collect(t .- 1) for t in eachcol(mesh.tri)],
            looks_unit=[collect(v) for v in eachcol(grid.rhat)],
            theta_rad=grid.theta, phi_rad=grid.phi,
            field_real=[real.(collect(v)) for v in eachcol(values)],
            field_imag=[imag.(collect(v)) for v in eachcol(values)],
            dof_count=state.rwg.nedges, quadrature_order=state.quad_order,
            relative_linear_residual=residual, linear_residual_limit=1e-10,
            resonance_guard, matrix_conditioning=conditioning,
            radiation_exact_work_limit=exact_work,
            julia_source_sha256=digest, julia_driver_sha256=driver_digest,
            julia_pilot_sha256=pilot_digest,
            costs=(; solve_s, radiation_s, assembly_s=result.assembly_time_s,
                linear_solve_s=result.solve_time_s, residual_s, conditioning_s))
        path = joinpath(output, case_id * "_q" * string(quad_order) * ".json")
        temporary = path * ".partial"
        open(temporary, "w") do io
            JSON.print(io, record, 2)
        end
        mv(temporary, path; force=true)
        println(case_id, ": ", state.rwg.nedges, " unknowns; ", path)
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    reference_export_main()
end
