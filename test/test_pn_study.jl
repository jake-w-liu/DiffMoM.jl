using Test
using LinearAlgebra
using JSON
using SHA
using DiffMoM

include(joinpath(@__DIR__, "..", "validation", "pn3d", "study_analysis.jl"))

@testset "PN3D study prediction contracts" begin
    grid = make_sph_grid(3, 4)
    main_looks = [2, 5]
    main_grid = DiffMoM.SphGrid(
        grid.rhat[:, main_looks], grid.theta[main_looks],
        grid.phi[main_looks], grid.w[main_looks])
    @test size(main_grid.rhat, 2) == 2
    @test main_grid.theta == grid.theta[main_looks]
    @test main_grid.phi == grid.phi[main_looks]
    @test main_grid.w == grid.w[main_looks]
    @test main_grid.rhat == grid.rhat[:, main_looks]
    field = _study_field_record(ComplexF64[1, 2, 3, 4], 2)
    @test field.real == [[1.0, 2.0], [3.0, 4.0]]
    @test field.imag == [[0.0, 0.0], [0.0, 0.0]]
    @test_throws DimensionMismatch _study_scale_record(zeros(3, 1), 2)

    mktempdir() do directory
        hashes = Dict{String,String}()
        for name in ("parameters.csv", "looks.csv", "masks.csv")
            path = joinpath(directory, name)
            write(path, "header\n")
            hashes[name] = bytes2hex(sha256(read(path)))
        end
        manifest = (
            schema_version=1,
            seed=20260907,
            distribution_id="bent-slotted-panel-uniform-v1",
            training_cases=60,
            calibration_cases=199,
            test_cases=300,
            look_count=12,
            mask_count=21,
            artifact_sha256=hashes,
        )
        open(joinpath(directory, "population_manifest.json"), "w") do io
            JSON.print(io, manifest)
        end
        protocol = _study_prediction_protocol(directory)
        @test _study_protocol_hash(protocol) ==
              _study_protocol_hash(_study_prediction_protocol(directory))
        write(joinpath(directory, "looks.csv"), "changed\n")
        @test_throws ErrorException _study_prediction_protocol(directory)
    end

    coarse_field = reshape(ComplexF64.(1:24), 2, 12)
    fine_field = 1.1 .* coarse_field
    encode(values) = JSON.parse(JSON.json(_study_field_record(vec(values), 12)))
    levels = Any[]
    for level in STUDY_LEVELS
        proposed = [Dict(
            "probe_count" => count,
            "effective_rank" => min(count, 7),
            "mean" => encode((1 + count / 100) .* coarse_field),
            "scales" => fill(0.1 + count / 100, 12),
        ) for count in STUDY_PROBE_COUNTS]
        push!(levels, Dict(
            "coarse_mean" => encode(coarse_field),
            "fine_mean" => encode(fine_field),
            "residual_scale_v" => 0.2,
            "global_scales" => fill(0.3, 12),
            "proposed" => proposed,
        ))
    end
    prediction = Dict(
        "status" => "complete", "source_unchanged" => true,
        "levels" => levels)
    proposed_means, proposed_scales = _study_configuration(prediction, "proposed-q8")
    @test proposed_means[:, :, 1] == 1.08 .* coarse_field
    @test proposed_scales[:, 1] == fill(0.18, 12)
    deterministic_means, deterministic_scales =
        _study_configuration(prediction, "deterministic-q8")
    @test deterministic_means == proposed_means
    @test deterministic_scales == fill(0.2, 12, 2)
    residual_means, _ = _study_configuration(prediction, "residual-only")
    @test residual_means[:, :, 1] == coarse_field
    global_means, global_scales = _study_configuration(prediction, "global-covariance")
    @test global_means == residual_means
    @test global_scales == fill(0.3, 12, 2)
    richardson_means, _ = _study_configuration(
        prediction, "richardson"; richardson_order=1.0)
    @test richardson_means[:, :, 1] ≈ 1.2 .* coarse_field
    fit = _study_fit_richardson_order([prediction], [1.2 .* coarse_field])
    @test fit.order == 1.0
    context = _study_context(repeat("a", 64), repeat("b", 64), "law", "proposed-q8")
    @test isfinite(_study_score(prediction, 1.2 .* coarse_field,
                                context, "proposed-q8"))
    @test_throws ArgumentError _study_configuration(prediction, "unknown")
    @test_throws ArgumentError _study_configuration(
        prediction, "richardson"; richardson_order=0.0)

    solves = [Dict("level" => level, "elapsed_s" => 10.0 + level,
                   "assembly_s" => 4.0 + level, "solve_s" => 5.0,
                   "preconditioner_s" => 1.0) for level in 0:2]
    level_costs = Dict(
        "nesting_s" => 1.0, "restriction_s" => 2.0, "radiation_s" => 3.0,
        "output_rows_s" => 4.0, "algebraic_coarse_s" => 0.5,
        "algebraic_fine_s" => 0.5, "global_s" => 7.0)
    proposed_costs = [Dict("probe_count" => count,
        "costs" => Dict("probe_s" => 1.0, "conditioning_s" => 2.0,
                        "moments_s" => 3.0)) for count in STUDY_PROBE_COUNTS]
    cost_prediction = Dict(
        "status" => "complete", "source_unchanged" => true,
        "solves" => solves,
        "levels" => [merge(Dict(l), Dict(
            "costs" => level_costs,
            "proposed" => proposed_costs,
            "algebraic_coarse_v" => fill(0.01, 12),
            "algebraic_fine_v" => fill(0.02, 12),
            "restriction_field_change_v" => fill(0.03, 12)))
            for l in levels])
    reference_record = Dict("levels" => [Dict(
        "level" => level, "field_real" => l["fine_mean"]["real"],
        "field_imag" => l["fine_mean"]["imag"],
        "algebraic_field_change_v" => fill(0.05, 36),
        "costs" => Dict("forward_total_s" => 100.0 + level,
                        "radiation_s" => 1.0, "algebraic_s" => 2.0))
        for (level, l) in enumerate(vcat(levels, levels[2:2], levels[2:2]))])
    for (index, l) in enumerate(reference_record["levels"])
        l["level"] = index - 1
    end
    checks = Dict("status" => "complete", "source_unchanged" => true,
        "case_id" => "train-0001",
        "quadrature_v" => fill(0.04, 12), "compression_v" => fill(0.05, 12))
    # Level-0 decision for proposed-q8: solve 0 fully, level-1 assembly only,
    # level-0 machinery including the q8 probe/conditioning/moment costs.
    expected = solves[1]["elapsed_s"] + solves[2]["assembly_s"] +
        1.0 + 2.0 + 3.0 + 4.0 + 0.5 + 0.5 + 1.0 + 2.0 + 3.0
    @test _study_decision_cost(
        cost_prediction, reference_record, "proposed-q8", 1, false) == expected
    # Escalation through both levels adds the level-1 solve (without its
    # assembly, already charged), the level-2 assembly, both machinery
    # instances, and the level-3 reference solve.
    expected_all = solves[1]["elapsed_s"] + solves[2]["assembly_s"] +
        (1.0 + 2.0 + 3.0 + 4.0 + 0.5 + 0.5 + 1.0 + 2.0 + 3.0) +
        solves[2]["solve_s"] + solves[2]["preconditioner_s"] +
        solves[3]["assembly_s"] +
        (1.0 + 2.0 + 3.0 + 4.0 + 0.5 + 0.5 + 1.0 + 2.0 + 3.0) +
        reference_record["levels"][4]["costs"]["forward_total_s"] +
        reference_record["levels"][4]["costs"]["radiation_s"] +
        reference_record["levels"][4]["costs"]["algebraic_s"]
    @test _study_decision_cost(
        cost_prediction, reference_record, "proposed-q8", nothing, true) ==
          expected_all
    @test _study_decision_cost(
        cost_prediction, reference_record, "uniform-final", 1, false) ==
          solves[3]["elapsed_s"] + level_costs["radiation_s"]
    @test _study_decision_cost(
        cost_prediction, reference_record, "richardson", 2, false) ==
          sum(s["elapsed_s"] for s in solves) + 2 * level_costs["radiation_s"]
    budget = _study_case_budget(
        cost_prediction, reference_record, checks, 1, collect(1:12))
    @test budget.changes.algebraic == fill(0.03, 12)
    @test budget.changes.geometry == 0.0
    @test all(budget.tolerance .>= budget.changes.algebraic)
    interval = _study_wilson_interval(95, 100)
    @test interval.lower < 0.95 < interval.upper
    @test _study_wilson_interval(0, 100).lower == 0.0
    @test _study_wilson_interval(100, 100).upper == 1.0
    @test_throws ArgumentError _study_wilson_interval(1, 0)
    @test_throws BoundsError _study_decision_cost(
        cost_prediction, reference_record, "residual-only", 4, false)

    k = 2pi * 3e9 / 299792458.0
    correction_mesh = make_rect_plate(0.05, 0.04, 2, 2)
    correction_rwg = build_rwg(correction_mesh)
    correction_source = make_plane_wave(Vec3(0, 0, -k), 1.0, Vec3(1, 0, 0))
    aca_state = solve_scattering(
        correction_mesh, 3e9, correction_source; method=:aca_gmres,
        quad_order=3, gmres_tol=1e-10, gmres_memory=40, gmres_maxiter=500,
        check_resolution=false, return_state=true, verbose=false).state
    aca_state.factorization === nothing ||
        error("ACA test state must use the iterative correction path")
    correction_map = rcs_output_map(
        correction_mesh, correction_rwg, make_sph_grid(3, 4), k; quad_order=3)
    aca_changes = _study_residual_output_change(aca_state, correction_map, 12)
    @test length(aca_changes) == 12
    @test all(change -> isfinite(change) && change >= 0, aca_changes)
    @test maximum(aca_changes) < 1e-6

    mesh = make_rect_plate(0.05, 0.04, 1, 1)
    pair = build_nested_rwg_pair(mesh)
    source = make_plane_wave(Vec3(0, 0, -k), 1.0, Vec3(1, 0, 0))
    coarse = solve_scattering(
        mesh, 3e9, source; method=:dense_direct, return_state=true,
        check_resolution=false, verbose=false).state
    fine = solve_scattering(
        pair.fine_mesh, 3e9, source; method=:dense_direct, return_state=true,
        check_resolution=false, verbose=false).state
    result = _study_level_prediction(
        mesh, pair.fine_mesh, coarse, fine, make_sph_grid(3, 4), 12, 1.0)
    @test result.coarse_unknowns == coarse.rwg.nedges
    @test result.fine_unknowns == fine.rwg.nedges
    @test result.unresolved_unknowns == size(pair.Q, 2)
    @test length(result.proposed) == length(STUDY_PROBE_COUNTS)
    @test all(length(entry.scales) == 12 for entry in result.proposed)
    @test all(all(isfinite, entry.scales) for entry in result.proposed)
    @test_throws ErrorException _study_level_prediction(
        mesh, make_rect_plate(0.05, 0.04, 1, 1), coarse, fine,
        make_sph_grid(3, 4), 12, 1.0)
end
