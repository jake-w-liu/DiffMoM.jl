include("pilot.jl")
include(joinpath(@__DIR__, "..", "mie", "mesh_fixture.jl"))
using Test

# With the negative PEC coefficients and outgoing h^(2) convention used by
# mie_s1s2_pec, the dimensional electric far field is i*S/k. The electric
# plus magnetic Rayleigh limit below fixes this sign before any MoM solve.
function pec_mie_field(k::Float64, radius::Float64, direction::Vec3,
                       polarization::Vec3, look::Vec3)
    mie_bistatic_rcs_pec(k, radius, direction, polarization, look)
    perpendicular = cross(direction, look)
    if norm(perpendicular) <= 64eps(Float64)
        perpendicular = cross(direction, polarization)
    end
    perpendicular /= norm(perpendicular)
    incident_parallel = cross(perpendicular, direction)
    scattered_parallel = cross(perpendicular, look)
    s1, s2 = mie_s1s2_pec(k * radius, clamp(dot(direction, look), -1.0, 1.0))
    return (1im / k) * (
        s1 * dot(polarization, perpendicular) * perpendicular +
        s2 * dot(polarization, incident_parallel) * scattered_parallel)
end

function sphere_reference(grid, k, radius, direction, polarization)
    field = zeros(ComplexF64, 2length(grid.w))
    for j in eachindex(grid.w)
        theta, phi = grid.theta[j], grid.phi[j]
        e_theta = Vec3(cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta))
        e_phi = Vec3(-sin(phi), cos(phi), 0.0)
        vector = pec_mie_field(k, radius, direction, polarization, Vec3(grid.rhat[:, j]))
        field[2j-1], field[2j] = dot(e_theta, vector), dot(e_phi, vector)
    end
    return field
end

function check_sphere_oracles()
    @testset "sphere mesh and absolute Mie field mapping" begin
        for level in 0:3
            mesh = make_icosphere(0.05; subdivisions=level)
            @test ntriangles(mesh) == 20 * 4^level
            @test nvertices(mesh) == 10 * 4^level + 2
            @test build_rwg(mesh; allow_boundary=false, require_closed=true).nedges == 30 * 4^level
            @test all(j -> isapprox(norm(mesh.xyz[:, j]), 0.05; rtol=8eps(Float64)),
                      axes(mesh.xyz, 2))
        end
        for radius in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError make_icosphere(radius)
        end
        @test_throws ArgumentError make_icosphere(1.0; subdivisions=-1)
        @test_throws ArgumentError make_icosphere(1.0; subdivisions=typemax(Int))
        @test_throws ArgumentError make_icosphere(1.0; max_triangles=19)
        @test_throws ArgumentError make_icosphere(1.0; subdivisions=2, max_triangles=319)
        grid = make_sph_grid(3, 4)
        d = normalize(Vec3(1.0, 2.0, -3.0))
        p = normalize(cross(d, Vec3(0.0, 0.0, 1.0)))
        k, a = 1.0, 1e-3
        for r in [Vec3(grid.rhat[:, j]) for j in eachindex(grid.w)]
            exact = pec_mie_field(k, a, d, p, r)
            rayleigh = k^2 * a^3 * (p - r * dot(r, p) + 0.5cross(r, cross(d, p)))
            @test norm(exact - rayleigh) <= 5e-6 * norm(rayleigh)
            @test isapprox(4pi * sum(abs2, exact), mie_bistatic_rcs_pec(k, a, d, p, r);
                           rtol=32eps(Float64), atol=0)
            @test abs(dot(r, exact)) <= 32eps(Float64) * norm(exact)
        end
        for r in (d, -d)
            exact = pec_mie_field(k, a, d, p, r)
            rayleigh = k^2 * a^3 * (p - r * dot(r, p) + 0.5cross(r, cross(d, p)))
            @test norm(exact - rayleigh) <= 5e-6 * norm(rayleigh)
        end
    end
end

function sphere_validation_main()
    BLAS.set_num_threads(1)
    check_sphere_oracles()
    get(ENV, "PN_SPHERE_ORACLES_ONLY", "0") == "1" && return nothing
    output = isempty(ARGS) ? joinpath(OUTPUT_DIR, "sphere-validation.json") : abspath(ARGS[1])
    ispath(output) && error("refusing to replace $output")
    mkpath(dirname(output))
    digest = source_digest()
    driver_digest = bytes2hex(sha256(read(@__FILE__)))
    helper_path = joinpath(@__DIR__, "..", "mie", "mesh_fixture.jl")
    helper_digest = bytes2hex(sha256(read(helper_path)))
    pilot_path = joinpath(@__DIR__, "pilot.jl")
    pilot_digest = bytes2hex(sha256(read(pilot_path)))
    rows = NamedTuple[]
    failure = nothing
    completed = false
    active = nothing
    started = time_ns()
    k = 2pi * FREQUENCY / SPEED
    direction, polarization = Vec3(0.0, 0.0, -1.0), Vec3(1.0, 0.0, 0.0)
    source = make_plane_wave(k * direction, 1.0, polarization)
    grid = make_sph_grid(3, 4)
    sizes = (0.2, 1.0, 2.0)
    # These acceptance limits are fixed before computing the sphere solutions.
    thresholds = (; finest_relative_complex_field=0.02,
        finest_relative_rcs=0.05, finest_max_component_phase_rad=0.03,
        selected_operator_residual=1e-10)
    function checkpoint()
        unchanged = source_digest() == digest &&
            bytes2hex(sha256(read(@__FILE__))) == driver_digest &&
            bytes2hex(sha256(read(helper_path))) == helper_digest &&
            bytes2hex(sha256(read(pilot_path))) == pilot_digest
        report = (; schema_version=1, status=completed ? "complete" :
            (isnothing(failure) ? "partial" : "failed"), source_unchanged=unchanged,
            source_sha256=digest, driver_sha256=driver_digest, helper_sha256=helper_digest,
            pilot_sha256=pilot_digest,
            frequency_hz=FREQUENCY, size_parameters=sizes, thresholds,
            incident_direction=collect(direction), incident_polarization=collect(polarization),
            incident_amplitude_v_m=1.0, phase_convention="exp(+i*omega*t)",
            max_work_bytes=WORK_BYTES, rows, active, failure,
            elapsed_s=(time_ns()-started)/1e9)
        temporary, io = mktemp(dirname(output))
        try
            JSON.print(io, report, 2)
            close(io)
            mv(temporary, output; force=true)
        finally
            isopen(io) && close(io)
            ispath(temporary) && rm(temporary)
        end
        unchanged || error("source changed during sphere validation")
    end
    function solve_case(mesh, ka, geometry, level, quadrature)
        active = (; ka, geometry, level, quadrature)
        checkpoint()
        (result, state), solve_s = @timed solve_scattering(
            mesh, FREQUENCY, source; method=:dense_direct, quad_order=quadrature,
            check_resolution=false, return_state=true, verbose=false,
            max_dense_matrix_bytes=WORK_BYTES, max_true_residual_exact_terms=16_000_000)
        mapping, radiation_s = @timed rcs_output_map(
            mesh, state.rwg, grid, k; eta0=IMPEDANCE,
            max_work_bytes=WORK_BYTES, max_exact_work=5_000_000_000)
        field = mapping * state.I_coeffs
        residual, residual_check_s = @timed DiffMoM._true_residual_ratio(
            state.operator, state.I_coeffs, state.rhs, "sphere validation";
            max_true_residual_exact_terms=16_000_000)
        reference = sphere_reference(grid, k, ka/k, direction, polarization)
        reference_norm = norm(reference)
        rcs = [4pi * sum(abs2, view(field, 2j-1:2j)) for j in eachindex(grid.w)]
        mie_rcs = [mie_bistatic_rcs_pec(k, ka/k, direction, polarization,
                     Vec3(grid.rhat[:, j])) for j in eachindex(grid.w)]
        # Component phase is omitted at near-zero reference components, where
        # it is undefined; the complete complex-field norm still includes them.
        phase_indices = findall(x -> abs(x) >= 1e-6reference_norm, reference)
        isempty(phase_indices) && error("no nonzero reference component for phase comparison")
        phase_error = maximum(abs(angle(field[j] / reference[j])) for j in phase_indices)
        row = (; ka, radius_m=ka/k, geometry, level, quadrature_points=quadrature,
            vertices=nvertices(mesh), triangles=ntriangles(mesh), rwg_unknowns=result.N,
            solve_s, radiation_s, assembly_s=result.assembly_time_s,
            linear_solve_s=result.solve_time_s, true_residual=residual, residual_check_s,
            relative_complex_field_error=norm(field-reference)/reference_norm,
            field_norm_ratio=norm(field)/reference_norm,
            global_field_phase_offset_rad=angle(dot(reference, field)),
            max_component_phase_error_rad=phase_error, phase_components=length(phase_indices),
            relative_rcs_error=norm(rcs-mie_rcs)/norm(mie_rcs),
            field_real=real.(field), field_imag=imag.(field),
            mie_field_real=real.(reference), mie_field_imag=imag.(reference), rcs_m2=rcs,
            mie_rcs_m2=mie_rcs)
        push!(rows, row)
        println("sphere ka=$ka $geometry level=$level q=$quadrature N=$(result.N): complex error=$(row.relative_complex_field_error)")
        flush(stdout)
        checkpoint()
        row.true_residual <= thresholds.selected_operator_residual ||
            error("sphere selected-operator residual exceeds its threshold")
        if geometry == "projected" && level == 3 && quadrature == 28
            row.relative_complex_field_error <= thresholds.finest_relative_complex_field ||
                error("finest projected sphere complex-field comparison failed")
            row.relative_rcs_error <= thresholds.finest_relative_rcs ||
                error("finest projected sphere RCS comparison failed")
            phase_error <= thresholds.finest_max_component_phase_rad ||
                error("finest projected sphere phase comparison failed")
        end
    end
    try
        for ka in sizes
            for level in 0:3
                mesh = make_icosphere(ka/k; subdivisions=level)
                solve_case(mesh, ka, "projected", level, 7)
                level == 3 && solve_case(mesh, ka, "projected", level, 28)
            end
            # These children stay on the 80 parent facets. Their errors against
            # Mie include geometry error and must not be called sphere convergence.
            mesh = make_icosphere(ka/k; subdivisions=1)
            for level in 2:3
                pair = build_nested_rwg_pair(mesh; max_work_bytes=WORK_BYTES)
                mesh = pair.fine_mesh
                solve_case(mesh, ka, "fixed_facets_from_level_1", level, 7)
            end
        end
        completed = true
    catch err
        failure = sprint(showerror, err)
        rethrow()
    finally
        checkpoint()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    sphere_validation_main()
end
