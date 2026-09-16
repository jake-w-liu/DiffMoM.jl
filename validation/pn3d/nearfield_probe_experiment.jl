include("pilot.jl")
using Test

# Exact noiseless conditioning makes the mean independent of a positive scalar
# prior scale. Use unit scale here; this is not a fitted uncertainty calibration.
const PROBE_TAU = 1.0

# This experiment selects rows from the operator and outputs only. The fine
# solution is used afterwards to measure the error, never to select a probe.
function experimental_nf_probes(prepared, preconditioner;
                                rows=collect(axes(prepared.rows, 1)))
    pair = prepared.system.pair
    nf, nc = size(pair.P)
    m = size(pair.Q, 2)
    selected = collect(Int, rows)
    length(unique(selected)) == length(selected) ||
        throw(ArgumentError("output rows must be unique"))
    all(i -> 1 <= i <= size(prepared.rows, 1), selected) ||
        throw(ArgumentError("output row is out of bounds"))
    inverse_adjoint = DiffMoM.NearFieldAdjointOperator(preconditioner)
    size(inverse_adjoint) == (nf, nf) ||
        throw(DimensionMismatch("fine preconditioner has the wrong size"))
    q = length(selected)
    work = BigInt(16) * (4BigInt(nc)^2 + (10BigInt(nf) + 8BigInt(m)) * q)
    work <= WORK_BYTES || throw(ArgumentError("probe work exceeds WORK_BYTES"))
    q == 0 && return zeros(ComplexF64, 0, m)

    pivots = pair.pivot_rows
    native = [rowvals(pair.Q)[only(nzrange(pair.Q, j))] for j in 1:m]
    Pi = Matrix{ComplexF64}(pair.P[pivots, :])
    Pj = pair.P[native, :]
    pivot_factor = DiffMoM._factor_dense_linear_system(
        Pi, ComplexF64, "nested pivot matrix")
    Hstar = Matrix(adjoint(prepared.rows[selected, :]))

    # L = T^(-H)[:, W] with T = [P Q]. Its native unresolved rows are I,
    # and its pivot rows are -P_I^(-H) P_J^H. Thus L^H M^(-H) L is the
    # adjoint inverse Schur complement of the near-field matrix M.
    lifted = zeros(ComplexF64, nf, q)
    lifted[native, :] = Hstar
    lifted[pivots, :] = -DiffMoM._solve_factored_linear_system(
        adjoint(pivot_factor), adjoint(Pi),
        DiffMoM._error_action_columns(adjoint(Pj), Hstar),
        "nested dual lift")
    applied = similar(lifted)
    for j in 1:q
        applied[:, j] = inverse_adjoint * view(lifted, :, j)
    end
    pivot_solution = DiffMoM._solve_factored_linear_system(
        pivot_factor, Pi, applied[pivots, :], "nested dual restriction")
    candidate = Matrix(adjoint(applied[native, :] -
        DiffMoM._error_action_columns(Pj, pivot_solution)))
    all(isfinite, candidate) || error("non-finite near-field probe")
    for i in axes(candidate, 1)
        scale = norm(view(candidate, i, :))
        iszero(scale) || (candidate[i, :] ./= scale)
    end
    decomposition = svd(candidate; full=false)
    threshold = 64eps(Float64) * max(size(candidate)...) * maximum(decomposition.S)
    numerical_rank = count(>(threshold), decomposition.S)
    return Matrix(decomposition.Vt[1:numerical_rank, :])
end

function probe_experiment_setup(mesh; fine_method=:dense_direct, quad_order=3,
                                radiation_exact_work=RADIATION_EXACT_WORK)
    residual_exact_terms = parse(Int, get(ENV, "PN_RESIDUAL_EXACT_TERMS", "5000000"))
    source = source_wave()
    (coarse_result, coarse), coarse_s = @timed solve_scattering(
        mesh, FREQUENCY, source; method=:dense_direct, quad_order=quad_order, c0=SPEED,
        max_true_residual_exact_terms=residual_exact_terms,
        check_resolution=false, return_state=true, verbose=false)
    pair, nesting_s = @timed build_nested_rwg_pair(mesh; max_work_bytes=WORK_BYTES)
    fine_controls = (; method=fine_method, quad_order=quad_order, c0=SPEED,
        aca_tol=1e-8, aca_max_rank=256, gmres_tol=1e-9, gmres_maxiter=1000,
        nf_cutoff_lambda=0.25, max_true_residual_exact_terms=residual_exact_terms,
        check_resolution=false, return_state=true, verbose=false)
    (fine_result, fine), fine_s = @timed solve_scattering(
        pair.fine_mesh, FREQUENCY, source; fine_controls...)
    system, restriction_s = @timed prepare_galerkin_error(
        coarse, pair, fine.operator, fine.rhs; max_work_bytes=WORK_BYTES)
    grid = make_sph_grid(3, 4)
    G, radiation_s = @timed rcs_output_map(
        pair.fine_mesh, pair.fine_rwg, grid, 2pi * FREQUENCY / SPEED;
        quad_order=quad_order, eta0=IMPEDANCE, max_work_bytes=WORK_BYTES,
        max_exact_work=radiation_exact_work)
    prepared, output_s = @timed prepare_error_outputs(
        system, G; max_work_bytes=WORK_BYTES)
    reference = vec(DiffMoM._error_action_columns(G, reshape(fine.I_coeffs, :, 1)))
    costs = (; coarse_s, nesting_s, fine_s, restriction_s, radiation_s, output_s,
        coarse_assembly_s=coarse_result.assembly_time_s,
        coarse_linear_solve_s=coarse_result.solve_time_s,
        coarse_preconditioner_s=coarse_result.preconditioner_time_s,
        fine_assembly_s=fine_result.assembly_time_s,
        fine_linear_solve_s=fine_result.solve_time_s,
        fine_preconditioner_s=fine_result.preconditioner_time_s)
    return (; pair, fine, prepared, reference, costs, fine_controls)
end

function check_full_nearfield_oracle()
    setup = probe_experiment_setup(make_rect_plate(0.08, 0.06, 2, 2))
    (; pair, fine, prepared, reference) = setup
    # Two metres exceeds the complete diameter of this 0.08 by 0.06 m mesh.
    nf = build_nearfield_preconditioner(
        fine.operator, pair.fine_mesh, pair.fine_rwg, 2.0)
    @testset "near-field inverse Schur probes" begin
        @test nf.nnz_ratio == 1.0
        @test size(experimental_nf_probes(prepared, nf; rows=Int[])) ==
              (0, size(pair.Q, 2))
        @test_throws ArgumentError experimental_nf_probes(prepared, nf; rows=[0])
        @test_throws ArgumentError experimental_nf_probes(prepared, nf; rows=[1, 1])
        T = Matrix(hcat(pair.P, pair.Q))
        transformed = adjoint(T) * fine.operator * T
        nc = size(pair.P, 2)
        A = transformed[1:nc, 1:nc]
        B = transformed[1:nc, nc+1:end]
        C = transformed[nc+1:end, 1:nc]
        D = transformed[nc+1:end, nc+1:end]
        S = D - C * (A \ B)
        for rows in ([1, 2], collect(1:8), collect(1:size(prepared.rows, 1)))
            R = experimental_nf_probes(prepared, nf; rows)
            exact_rows = prepared.rows[rows, :] / S
            # The SVD may rotate rows or discard true linear dependencies.
            @test norm(exact_rows - exact_rows * adjoint(R) * R) <=
                  2e-9 * norm(exact_rows)
            model = condition_discretization_error(
                prepared.system, R; tau=PROBE_TAU, max_work_bytes=WORK_BYTES)
            result = evaluate_error_outputs(model, prepared; max_work_bytes=WORK_BYTES)
            @test norm(result.mean[rows] - reference[rows]) <=
                  2e-8 * norm(reference[rows])
            @test norm(result.square_root[rows, :]) <=
                  2e-8 * norm(prepared.rows[rows, :])
        end
    end
end

function nearfield_experiment_main()
    BLAS.set_num_threads(1)
    digest = source_digest()
    driver_digest = bytes2hex(sha256(read(@__FILE__)))
    total_started = time_ns()
    check_full_nearfield_oracle()
    fine_method = Symbol(get(ENV, "PN_FINE_METHOD", "dense_direct"))
    quad_order = parse(Int, get(ENV, "PN_QUAD_ORDER", "3"))
    radiation_exact_work = parse(Int, get(ENV, "PN_RADIATION_EXACT_WORK", string(RADIATION_EXACT_WORK)))
    setup = probe_experiment_setup(make_bent_slotted_panel(; max_edge=EDGE);
        fine_method=fine_method, quad_order=quad_order, radiation_exact_work=radiation_exact_work)
    (; pair, fine, prepared, reference) = setup
    reference_norm = norm(reference)
    reference_norm > 0 || error("zero reference field cannot define relative error")
    rows = NamedTuple[]
    output = get(ENV, "PN_NF_OUTPUT", joinpath(OUTPUT_DIR, "nearfield-probes.json"))
    mkpath(dirname(output))
    active_config = (; cutoff_lambda=0.0, probe_rows=0, stage="setup_complete")
    active_started = time_ns()
    function write_report(status; failure=nothing)
        unchanged = digest == source_digest() &&
                    driver_digest == bytes2hex(sha256(read(@__FILE__)))
        report = (; status=unchanged ? status : "invalidated", source_unchanged=unchanged,
            source_sha256=digest, driver_sha256=driver_digest,
            frequency_hz=FREQUENCY, tau=PROBE_TAU, fine_method=string(fine.method),
            fine_controls=setup.fine_controls, requested_max_edge_m=EDGE,
            quadrature_order=quad_order, radiation_exact_work_limit=radiation_exact_work,
            max_work_bytes=WORK_BYTES,
            coarse_edges=size(pair.P, 2), fine_edges=size(pair.P, 1),
            coarse_relative_error=norm(prepared.coarse_mean - reference) / reference_norm,
            costs=setup.costs, active_config, failure, rows,
            total_elapsed_s=(time_ns() - total_started) / 1e9)
        open(output, "w") do io
            JSON.print(io, report, 2)
        end
        unchanged || error("source changed during the experiment")
    end
    write_report("partial")
    try
      for cutoff_lambda in (0.25, 0.5, 1.0, 2.0)
        active_config = (; cutoff_lambda, probe_rows=0, stage="nearfield_setup")
        write_report("partial")
        active_started = time_ns()
        nf, nf_s = @timed build_nearfield_preconditioner(
            fine.operator, pair.fine_mesh, pair.fine_rwg,
            cutoff_lambda * SPEED / FREQUENCY)
        for q in (2, 8, size(prepared.rows, 1))
            active_config = (; cutoff_lambda, probe_rows=q, stage="conditioning_and_outputs")
            write_report("partial")
            active_started = time_ns()
            probes, probe_s = @timed experimental_nf_probes(
                prepared, nf; rows=collect(1:q))
            model, condition_s = @timed condition_discretization_error(
                prepared.system, probes; tau=PROBE_TAU, max_work_bytes=WORK_BYTES)
            result, moments_s = @timed evaluate_error_outputs(
                model, prepared; max_work_bytes=WORK_BYTES)
            entry = (; cutoff_lambda, requested_rows=q, rank=size(probes, 1),
                nnz_ratio=nf.nnz_ratio, nf_s, probe_s, condition_s, moments_s,
                relative_mean_error=norm(result.mean - reference) / reference_norm,
                square_root_norm=norm(result.square_root))
            push!(rows, entry)
            println(entry)
            flush(stdout)
            write_report("partial")
        end
      end
    catch error
        write_report("failed"; failure=(; type=string(typeof(error)),
            message=sprint(showerror, error),
            failed_stage_elapsed_s=(time_ns() - active_started) / 1e9))
        rethrow()
    end
    write_report("complete")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    nearfield_experiment_main()
end
