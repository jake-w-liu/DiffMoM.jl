using Test
using LinearAlgebra
using Random
using DiffMoM

function _pn_test_context(; algorithm_hash=repeat("a", 64), look_count=3,
                          levels=(0, 1), field_floor=1e-8)
    return RCSCalibrationContext(
        algorithm_hash=algorithm_hash, training_revision="frozen-training",
        distribution_id="synthetic-test-law", reference_protocol="complex-field-test",
        look_id="prescribed-test-looks", look_count=look_count, levels=levels,
        field_floor=field_floor, absolute_field_tolerance=1e-12)
end

@testset "Casewise augmented calibration ranks" begin
    context = _pn_test_context()
    ids = ["case-$i" for i in 1:199]
    profile = calibrate_rcs(
        Float64.(1:199), context; calibration_case_ids=ids,
        training_case_ids=["train-1"], epsilon=0.05)
    @test profile.quantile_index == 190
    @test profile.quantile == 190
    @test profile.epsilon == 1 // 20
    @test profile.context == _pn_test_context()
    @test hash(profile.context) == hash(_pn_test_context())
    for n in 1:99
        scores = Float64.(1:n)
        case_ids = ids[1:n]
        for numerator in 1:n
            # Exact rational levels put the requested rank on an integer.
            exact = calibrate_rcs(
                scores, context; calibration_case_ids=case_ids,
                epsilon=numerator // (n + 1))
            @test exact.quantile_index == n + 1 - numerator
            # Decimal percentages exercise floating-point boundary handling.
            decimal = calibrate_rcs(
                scores, context; calibration_case_ids=case_ids,
                epsilon=numerator / 100)
            expected = cld((n + 1) * (100 - numerator), 100)
            @test decimal.quantile_index == expected
            @test decimal.quantile == (expected > n ? Inf : expected)
        end
    end
    small = calibrate_rcs(zeros(18), context; calibration_case_ids=ids[1:18])
    @test isinf(small.quantile)
    ties = calibrate_rcs(fill(2.0, 19), context; calibration_case_ids=ids[1:19])
    @test ties.quantile == 2.0
    empty_profile = calibrate_rcs(Float64[], context; calibration_case_ids=String[])
    @test isinf(empty_profile.quantile)
    reordered = calibrate_rcs([3.0, 1.0, 2.0], context;
                             calibration_case_ids=["c", "a", "b"], epsilon=0.5)
    @test reordered.sorted_scores == [1.0, 2.0, 3.0]
    @test reordered.calibration_cases == ("a", "b", "c")
    @test_throws ArgumentError calibrate_rcs([NaN], context; calibration_case_ids=["x"])
    @test_throws ArgumentError calibrate_rcs([-1.0], context; calibration_case_ids=["x"])
    @test_throws ArgumentError calibrate_rcs([1.0, 2.0], context; calibration_case_ids=["x", "x"])
    @test_throws ArgumentError calibrate_rcs([1.0], context;
        calibration_case_ids=["x"], training_case_ids=["x"])
    @test_throws ArgumentError calibrate_rcs([1.0], context; calibration_case_ids=["x"], epsilon=0)
    @test_throws ArgumentError calibrate_rcs([1.0], context; calibration_case_ids=["x"], epsilon=1)
    @test_throws ArgumentError calibrate_rcs([1.0], context; calibration_case_ids=["x"], max_work_bytes=1)
    @test_throws ArgumentError _pn_test_context(algorithm_hash="unversioned")
    @test_throws ArgumentError _pn_test_context(levels=(0, 0))
    @test_throws ArgumentError _pn_test_context(field_floor=0.0)

    # Exhaust every holdout rank in one exchangeable set of distinct scores.
    covered = 0
    n = 19
    for held_out in 1:20
        calibration = [Float64(i) for i in 1:20 if i != held_out]
        ranks = calibrate_rcs(
            calibration, context; calibration_case_ids=ids[1:n], epsilon=1//5)
        covered += held_out <= ranks.quantile
    end
    @test covered == 16
end

@testset "Whole-case score includes all levels and looks" begin
    context = _pn_test_context()
    reference = zeros(ComplexF64, 2, 3)
    means = zeros(ComplexF64, 2, 3, 2)
    scales = ones(3, 2)
    means[1, 2, 2] = 3 + 4im
    @test rcs_case_score(reference, means, scales, context) == 5.0
    means[2, 1, 1] = 12
    @test rcs_case_score(reference, means, scales, context) == 12.0
    scales[1, 1] = 4
    @test rcs_case_score(reference, means, scales, context) == 5.0
    @test_throws DimensionMismatch rcs_case_score(reference, means[:, :, 1:1], scales[:, 1:1], context)
    @test_throws ArgumentError rcs_case_score(reference, means, zeros(3, 2), context)
    @test_throws ArgumentError rcs_case_score(reference, means, fill(1e-10, 3, 2), context)
    means[1] = NaN
    @test_throws ArgumentError rcs_case_score(reference, means, scales, context)

    large_reference = fill(complex(1e308, 0.0), 2, 1)
    large_mean = fill(complex(-1e308, 0.0), 2, 1, 1)
    large_context = _pn_test_context(look_count=1, levels=(0,))
    score = rcs_case_score(large_reference, large_mean, fill(1e308, 1, 1), large_context)
    @test score >= sqrt(8.0)
    @test score <= nextfloat(sqrt(8.0), 2)
end

@testset "Outward RCS balls and decision boundaries" begin
    rng = MersenneTwister(17130)
    # All sampled perturbations lie strictly within each ball; high-precision
    # RCS evaluation is independent of the rational square-root enclosures.
    for case_index in 1:100
        mean_field = randn(rng, ComplexF64, 2)
        radius = rand(rng)
        amplitude = 0.25 + rand(rng)
        bounds = rcs_ball_bounds(mean_field, radius; incident_amplitude=amplitude)
        for sample in 1:100
            direction = randn(rng, ComplexF64, 2)
            perturbation = (0.99 * radius * rand(rng) / norm(direction)) * direction
            field = Complex{BigFloat}.(mean_field + perturbation)
            actual = 4BigFloat(pi) * sum(abs2, field) / BigFloat(amplitude)^2
            @test BigFloat(bounds.lower) <= actual <= BigFloat(bounds.upper)
        end
    end
    zero = rcs_ball_bounds(ComplexF64[0, 0], 0.0)
    @test zero == (lower=0.0, upper=0.0)
    @test rcs_ball_bounds(ComplexF64[1, 0], Inf) == (lower=0.0, upper=Inf)
    extreme = rcs_ball_bounds(ComplexF64[1e308, 1e308], 0.0; incident_amplitude=1e308)
    @test extreme.lower <= 8BigFloat(pi) <= extreme.upper
    tiny = rcs_ball_bounds(ComplexF64[1e-300, 0], 0.0)
    @test tiny.lower == 0.0
    @test tiny.upper == nextfloat(0.0)
    @test_throws ArgumentError rcs_ball_bounds(ComplexF64[1, 0], -1.0)
    @test_throws ArgumentError rcs_ball_bounds(ComplexF64[NaN, 0], 1.0)
    @test_throws ArgumentError rcs_ball_bounds(ComplexF64[1, 0], 1.0; incident_amplitude=0)

    context = _pn_test_context()
    ids = ["cal-$i" for i in 1:19]
    profile = calibrate_rcs(zeros(19), context; calibration_case_ids=ids)
    means = ComplexF64[0.1 0.2 0.3; 0 0 0]
    scales = fill(context.field_floor, 3)
    budget = NumericalErrorBudget(
        tolerance=0.0, algebraic=0.0, quadrature=0.0, compression=0.0,
        restriction=0.0, reference=0.0, geometry=0.0)
    upper = [rcs_ball_bounds(means[:, j], 0).upper for j in 1:3]
    lower = [rcs_ball_bounds(means[:, j], 0).lower for j in 1:3]
    decide(mask; kwargs...) = screen_rcs_mask(
        means, scales, profile, mask; context=context, level=0,
        case_supported=true, numerical_budget=budget, kwargs...)
    @test decide(upper).decision == :pass
    @test decide(prevfloat.(lower)).decision == :fail
    @test decide(lower).decision == :unresolved
    missing_checks = screen_rcs_mask(
        means, scales, profile, upper; context=context, level=0, case_supported=true)
    @test missing_checks.decision == :unresolved
    @test :numerical_checks_missing in missing_checks.reasons
    unsupported = screen_rcs_mask(
        means, scales, profile, upper; context=context, level=0, numerical_budget=budget)
    @test unsupported.decision == :unresolved
    @test :out_of_support in unsupported.reasons
    stale = screen_rcs_mask(
        means, scales, profile, upper;
        context=_pn_test_context(algorithm_hash=repeat("b", 64)), level=0,
        case_supported=true, numerical_budget=budget)
    @test :incompatible_calibration in stale.reasons
    wrong_level = screen_rcs_mask(
        means, scales, profile, upper; context=context, level=3,
        case_supported=true, numerical_budget=budget)
    @test :unregistered_level in wrong_level.reasons
    partial_budget = NumericalErrorBudget(tolerance=0.0, algebraic=0.0)
    unchecked = screen_rcs_mask(
        means, scales, profile, upper; context=context, level=0,
        case_supported=true, numerical_budget=partial_budget)
    @test :unchecked_reference in unchecked.reasons
    @test unchecked.decision == :unresolved
    too_small = calibrate_rcs(zeros(18), context; calibration_case_ids=ids[1:18])
    infinite = screen_rcs_mask(
        means, scales, too_small, fill(1e100, 3); context=context, level=0,
        case_supported=true, numerical_budget=budget)
    @test infinite.decision == :unresolved
    @test :unbounded_calibration in infinite.reasons
    @test_throws ArgumentError NumericalErrorBudget(0.0, (algebraic=0.0,))
    @test_throws ArgumentError NumericalErrorBudget(tolerance=1e-6, reference=false)
    heterogeneous = calibrate_rcs(ones(19), context; calibration_case_ids=ids)
    varying_scales = [0.01, 0.1, 1.0]
    per_look_budget = NumericalErrorBudget(
        tolerance=[0.001, 0.01, 0.1], algebraic=0.0,
        quadrature=[0.0005, 0.005, 0.05], compression=0.0,
        restriction=0.0, reference=[0.0005, 0.005, 0.05], geometry=0.0)
    wide_masks = [rcs_ball_bounds(means[:, j], varying_scales[j]).upper for j in 1:3]
    per_look = screen_rcs_mask(
        means, varying_scales, heterogeneous, wide_masks; context=context, level=0,
        case_supported=true, numerical_budget=per_look_budget)
    @test per_look.decision == :pass
    @test_throws ArgumentError NumericalErrorBudget(tolerance=[0.1, NaN])
    @test_throws ArgumentError NumericalErrorBudget(tolerance=0.1, reference=[true, false])
    @test_throws ArgumentError decide([-1.0, 1.0, 1.0])
    @test_throws ArgumentError decide(upper; incident_amplitude=0.0)
    profile.sorted_scores[1] = 1.0
    @test_throws ArgumentError decide(upper)
end

@testset "Two-component radiation map" begin
    rng = MersenneTwister(17131)
    mesh = make_rect_plate(0.1, 0.08, 2, 2)
    rwg = build_rwg(mesh)
    grid = make_sph_grid(3, 4)
    k = 2pi
    cartesian = radiation_vectors(mesh, rwg, grid, k)
    transverse = rcs_output_map(mesh, rwg, grid, k)
    current = randn(rng, ComplexF64, rwg.nedges)
    cartesian_fields = reshape(cartesian * current, 3, :)
    transverse_fields = reshape(transverse * current, 2, :)
    @test isapprox(vec(sum(abs2, transverse_fields; dims=1)),
                   vec(sum(abs2, cartesian_fields; dims=1)); rtol=1e-12)
    @test_throws ArgumentError rcs_output_map(mesh, rwg, grid, k; max_work_bytes=1)
end
