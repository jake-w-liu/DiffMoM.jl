include("study_predictions.jl")

const STUDY_FIELD_FLOOR_V = 1.0e-10
const STUDY_EPSILON = 0.05
const STUDY_RICHARDSON_ORDERS = Tuple(0.5:0.25:3.0)

function _study_complex_field(record)
    real_values = record["real"]
    imag_values = record["imag"]
    length(real_values) == length(imag_values) ||
        throw(DimensionMismatch("real and imaginary field records differ"))
    !isempty(real_values) || throw(ArgumentError("field record must be nonempty"))
    all(length(row) == 2 for row in real_values) &&
        all(length(row) == 2 for row in imag_values) ||
        throw(DimensionMismatch("each look requires two transverse components"))
    result = Matrix{ComplexF64}(undef, 2, length(real_values))
    for look in eachindex(real_values), component in 1:2
        result[component, look] = complex(
            Float64(real_values[look][component]),
            Float64(imag_values[look][component]))
    end
    all(isfinite, result) || throw(ArgumentError("field record contains non-finite values"))
    return result
end

function _study_reference_field(reference, level, main_looks)
    reference["status"] == "complete" && reference["source_unchanged"] ||
        error("reference case is incomplete or invalidated")
    levels = reference["levels"]
    length(levels) >= level + 1 || error("reference case lacks level $level")
    record = levels[level + 1]
    record["level"] == level || error("reference levels are out of order")
    wide = _study_complex_field(Dict(
        "real" => record["field_real"],
        "imag" => record["field_imag"]))
    all(look -> 1 <= look <= size(wide, 2), main_looks) ||
        error("registered look is outside the reference field")
    return wide[:, main_looks]
end

function _study_prediction_levels(prediction)
    prediction["status"] == "complete" && prediction["source_unchanged"] ||
        error("prediction case is incomplete or invalidated")
    levels = prediction["levels"]
    length(levels) == length(STUDY_LEVELS) ||
        error("prediction case has the wrong number of candidate levels")
    return levels
end

function _study_proposed_entry(level, probe_count)
    entries = [entry for entry in level["proposed"]
               if entry["probe_count"] == probe_count]
    length(entries) == 1 || error("prediction lacks a unique probe configuration")
    return only(entries)
end

function _study_positive_scales(values, floor_value)
    scales = Float64.(values)
    all(value -> isfinite(value) && value >= 0, scales) ||
        throw(ArgumentError("prediction scale is invalid"))
    return max.(scales, floor_value)
end

function _study_field_difference_scales(first_field, second_field, floor_value)
    size(first_field) == size(second_field) ||
        throw(DimensionMismatch("field scale inputs differ"))
    return [max(norm(view(first_field - second_field, :, look)), floor_value)
            for look in axes(first_field, 2)]
end

function _study_configuration(prediction, name; richardson_order=1.0,
                              field_floor=STUDY_FIELD_FLOOR_V)
    isfinite(richardson_order) && richardson_order > 0 ||
        throw(ArgumentError("Richardson order must be finite and positive"))
    levels = _study_prediction_levels(prediction)
    look_count = length(levels[1]["global_scales"])
    means = Array{ComplexF64}(undef, 2, look_count, length(levels))
    scales = Matrix{Float64}(undef, look_count, length(levels))
    for (position, level) in enumerate(levels)
        coarse = _study_complex_field(level["coarse_mean"])
        fine = _study_complex_field(level["fine_mean"])
        size(coarse) == (2, look_count) && size(fine) == (2, look_count) ||
            throw(DimensionMismatch("prediction fields have inconsistent look counts"))
        if startswith(name, "proposed-q")
            count = parse(Int, split(name, 'q')[end])
            entry = _study_proposed_entry(level, count)
            means[:, :, position] .= _study_complex_field(entry["mean"])
            scales[:, position] .= _study_positive_scales(entry["scales"], field_floor)
        elseif startswith(name, "deterministic-q")
            count = parse(Int, split(name, 'q')[end])
            entry = _study_proposed_entry(level, count)
            means[:, :, position] .= _study_complex_field(entry["mean"])
            scales[:, position] .= max(Float64(level["residual_scale_v"]), field_floor)
        elseif name == "residual-only"
            means[:, :, position] .= coarse
            scales[:, position] .= max(Float64(level["residual_scale_v"]), field_floor)
        elseif name == "global-covariance"
            means[:, :, position] .= coarse
            scales[:, position] .= _study_positive_scales(level["global_scales"], field_floor)
        elseif name == "full-enrichment"
            means[:, :, position] .= fine
            scales[:, position] .= _study_field_difference_scales(coarse, fine, field_floor)
        elseif name == "uniform-final"
            final_field = _study_complex_field(levels[end]["fine_mean"])
            means[:, :, position] .= final_field
            scales[:, position] .= field_floor
        elseif name == "richardson"
            denominator = exp2(richardson_order) - 1
            means[:, :, position] .= fine + (fine - coarse) / denominator
            scales[:, position] .= _study_field_difference_scales(coarse, fine, field_floor)
        else
            throw(ArgumentError("unknown study configuration: $name"))
        end
    end
    all(isfinite, means) && all(isfinite, scales) ||
        error("study configuration produced non-finite values")
    return means, scales
end

function _study_configuration_names()
    proposed = ["proposed-q$count" for count in STUDY_PROBE_COUNTS]
    deterministic = ["deterministic-q$count" for count in STUDY_PROBE_COUNTS if count > 0]
    return vcat(proposed, deterministic,
                ["residual-only", "global-covariance", "full-enrichment",
                 "uniform-final", "richardson"])
end

function _study_relative_field_error(prediction, reference, name;
                                     richardson_order=1.0)
    means, _ = _study_configuration(
        prediction, name; richardson_order, field_floor=STUDY_FIELD_FLOOR_V)
    size(means, 1) == size(reference, 1) && size(means, 2) == size(reference, 2) ||
        throw(DimensionMismatch("prediction and reference fields differ"))
    reference_norm = norm(reference)
    reference_norm > 0 || error("reference field norm must be positive")
    return [norm(view(means, :, :, level) - reference) / reference_norm
            for level in axes(means, 3)]
end

function _study_fit_richardson_order(predictions, references)
    length(predictions) == length(references) && !isempty(predictions) ||
        throw(DimensionMismatch("training predictions and references must be nonempty and matched"))
    losses = Float64[]
    for order in STUDY_RICHARDSON_ORDERS
        values = Float64[]
        for (prediction, reference) in zip(predictions, references)
            errors = _study_relative_field_error(
                prediction, reference, "richardson"; richardson_order=order)
            push!(values, errors[end])
        end
        push!(losses, sum(values) / length(values))
    end
    index = argmin(losses)
    return (
        order=STUDY_RICHARDSON_ORDERS[index],
        mean_relative_error=losses[index],
        candidates=collect(STUDY_RICHARDSON_ORDERS),
        losses,
    )
end

function _study_context(algorithm_hash, reference_protocol_hash, distribution_id, name)
    digest = bytes2hex(sha256(join((algorithm_hash, reference_protocol_hash, name), ':')))
    return RCSCalibrationContext(
        algorithm_hash=digest,
        training_revision=algorithm_hash,
        distribution_id=distribution_id,
        reference_protocol=reference_protocol_hash,
        look_id="registered-12-look-grid",
        look_count=12,
        levels=STUDY_LEVELS,
        field_floor=STUDY_FIELD_FLOOR_V,
        absolute_field_tolerance=STUDY_FIELD_FLOOR_V,
        numerical_fraction=0.1,
    )
end

function _study_score(prediction, reference, context, name; richardson_order=1.0)
    means, scales = _study_configuration(
        prediction, name; richardson_order, field_floor=context.field_floor)
    return rcs_case_score(reference, means, scales, context)
end

function _study_complete_record(directory, case_id, protocol_hash, split)
    path = joinpath(directory, case_id * ".json")
    isfile(path) || error("missing prediction record: $case_id")
    record = JSON.parsefile(path)
    record["case_id"] == case_id && record["split"] == split ||
        error("prediction identity or split differs: $case_id")
    record["protocol_hash"] == protocol_hash ||
        error("prediction protocol differs: $case_id")
    _study_prediction_levels(record)
    return record
end

function _study_complete_reference(directory, case_id, protocol_hash)
    path = joinpath(directory, case_id * ".json")
    isfile(path) || error("missing reference record: $case_id")
    record = JSON.parsefile(path)
    record["case_id"] == case_id || error("reference identity differs: $case_id")
    record["protocol_hash"] == protocol_hash || error("reference protocol differs: $case_id")
    record["status"] == "complete" && record["source_unchanged"] ||
        error("reference record is incomplete: $case_id")
    length(record["levels"]) == 4 || error("reference record lacks four levels: $case_id")
    return record
end

function _study_profile_record(profile)
    return (
        epsilon=string(profile.epsilon),
        sorted_scores=profile.sorted_scores,
        quantile_index=profile.quantile_index,
        quantile=profile.quantile,
        calibration_cases=collect(profile.calibration_cases),
        training_cases=collect(profile.training_cases),
    )
end

function study_calibration_main(population_dir, prediction_dir, reference_dir, output_path)
    population = JSON.parsefile(joinpath(population_dir, "population_manifest.json"))
    prediction_protocol_record = JSON.parsefile(joinpath(prediction_dir, "prediction_protocol.json"))
    prediction_protocol_hash = prediction_protocol_record["protocol_hash"]
    reference_protocol_record = JSON.parsefile(joinpath(reference_dir, "reference_protocol.json"))
    reference_protocol_hash = reference_protocol_record["protocol_hash"]
    main_looks = Int.(reference_protocol_record["protocol"]["main_looks"])
    rows = collect(CSV.File(joinpath(population_dir, "parameters.csv")))
    training_rows = [row for row in rows if row.split == "train"]
    calibration_rows = [row for row in rows if row.split == "calibration"]
    length(training_rows) == 60 && length(calibration_rows) == 199 ||
        error("training or calibration split count differs")
    training_predictions = Any[]
    training_references = Matrix{ComplexF64}[]
    training_ids = String[]
    reference_changes = Float64[]
    for row in training_rows
        case_id = String(row.case_id)
        prediction = _study_complete_record(
            prediction_dir, case_id, prediction_protocol_hash, "train")
        reference = _study_complete_reference(reference_dir, case_id, reference_protocol_hash)
        push!(training_predictions, prediction)
        push!(training_references, _study_reference_field(reference, 3, main_looks))
        reference_norm = norm(training_references[end])
        reference_norm > 0 || error("training reference field norm must be positive")
        level2 = _study_reference_field(reference, 2, main_looks)
        push!(reference_changes, norm(level2 - training_references[end]) / reference_norm)
        push!(training_ids, case_id)
    end
    richardson = _study_fit_richardson_order(training_predictions, training_references)
    algorithm_hash = bytes2hex(sha256(
        prediction_protocol_hash * JSON.json(richardson)))
    contexts = Dict{String,RCSCalibrationContext}()
    profiles = Dict{String,Any}()
    score_records = Dict{String,Vector{Float64}}()
    for name in _study_configuration_names()
        context = _study_context(
            algorithm_hash, reference_protocol_hash,
            population["distribution_id"], name)
        scores = Float64[]
        ids = String[]
        for row in calibration_rows
            case_id = String(row.case_id)
            prediction = _study_complete_record(
                prediction_dir, case_id, prediction_protocol_hash, "calibration")
            reference_record = _study_complete_reference(
                reference_dir, case_id, reference_protocol_hash)
            reference = _study_reference_field(reference_record, 3, main_looks)
            push!(scores, _study_score(
                prediction, reference, context, name;
                richardson_order=richardson.order))
            push!(ids, case_id)
        end
        profile = calibrate_rcs(
            scores, context; calibration_case_ids=ids,
            training_case_ids=training_ids, epsilon=STUDY_EPSILON,
            max_work_bytes=STUDY_MAX_WORK_BYTES)
        contexts[name] = context
        profiles[name] = _study_profile_record(profile)
        score_records[name] = scores
    end
    result = (
        schema_version=1,
        prediction_protocol_hash,
        reference_protocol_hash,
        population_manifest_sha256=bytes2hex(sha256(read(joinpath(
            population_dir, "population_manifest.json")))),
        source_sha256=source_digest(),
        richardson,
        field_floor_v=STUDY_FIELD_FLOOR_V,
        training_case_ids=training_ids,
        training_reference_relative_changes=reference_changes,
        maximum_training_reference_relative_change=maximum(reference_changes),
        contexts=Dict(name => (
            algorithm_hash=context.algorithm_hash,
            training_revision=context.training_revision,
            distribution_id=context.distribution_id,
            reference_protocol=context.reference_protocol,
            look_id=context.look_id,
            look_count=context.look_count,
            levels=collect(context.levels),
            field_floor=context.field_floor,
            absolute_field_tolerance=context.absolute_field_tolerance,
            numerical_fraction=context.numerical_fraction,
        ) for (name, context) in contexts),
        profiles,
        calibration_scores=score_records,
    )
    mkpath(dirname(output_path))
    ispath(output_path) && error("use a new calibration output path")
    open(output_path, "w") do io
        JSON.print(io, result, 2)
    end
    println("Calibrated ", length(profiles), " configurations on 199 whole cases.")
end

function _study_masks(population_dir, look_count)
    rows = collect(CSV.File(joinpath(population_dir, "masks.csv")))
    grouped = Dict{String,Vector{Tuple{Int,Float64}}}()
    for row in rows
        push!(get!(grouped, String(row.mask_id), Tuple{Int,Float64}[]),
              (Int(row.look_id), Float64(row.threshold_m2)))
    end
    masks = Dict{String,Vector{Float64}}()
    for (id, entries) in grouped
        sort!(entries; by=first)
        [first(entry) for entry in entries] == collect(1:look_count) ||
            error("mask $id does not list every registered look once")
        masks[id] = [last(entry) for entry in entries]
    end
    return masks
end

function _study_solve_record(prediction, level)
    entries = [entry for entry in prediction["solves"] if entry["level"] == level]
    length(entries) == 1 ||
        error("prediction lacks a unique solve record for level $level")
    return only(entries)
end

function _study_machinery_cost(level_record, name)
    costs = level_record["costs"]
    total = costs["nesting_s"] + costs["restriction_s"] + costs["radiation_s"] +
            costs["output_rows_s"] + costs["algebraic_coarse_s"] +
            costs["algebraic_fine_s"]
    if name == "global-covariance"
        total += costs["global_s"]
    elseif startswith(name, "proposed-q") || startswith(name, "deterministic-q")
        count = parse(Int, split(name, 'q')[end])
        entry = _study_proposed_entry(level_record, count)
        total += entry["costs"]["probe_s"] + entry["costs"]["conditioning_s"] +
                 entry["costs"]["moments_s"]
    end
    return total
end

"""
Deployment cost in seconds for one case/configuration. The adaptive rule
inspects candidate levels in order; an unresolved level-0 outcome pays for the
level-1 coarse solve plus the level-2 fine operator, and a still-unresolved
case escalates to the level-3 reference solve recorded for that case.
"""
function _study_decision_cost(prediction, reference, name, decided_level_index,
                              escalated)
    solves = prediction["solves"]
    levels = prediction["levels"]
    cost = 0.0
    if name == "uniform-final"
        final = _study_solve_record(prediction, STUDY_LEVELS[end] + 1)
        cost += final["elapsed_s"] + levels[end]["costs"]["radiation_s"]
    elseif name in ("full-enrichment", "richardson")
        last_solve = decided_level_index === nothing ?
            STUDY_LEVELS[end] + 1 : STUDY_LEVELS[decided_level_index] + 1
        for level in 0:last_solve
            cost += _study_solve_record(prediction, level)["elapsed_s"]
        end
        last_level = decided_level_index === nothing ?
            length(STUDY_LEVELS) : decided_level_index
        for position in 1:last_level
            cost += levels[position]["costs"]["radiation_s"]
        end
    else
        # Calibrated predictors pay each coarse solve, each fine operator
        # assembly (not its solve), and the level's error machinery.
        last_level = decided_level_index === nothing ?
            length(STUDY_LEVELS) : decided_level_index
        for position in 1:last_level
            level = STUDY_LEVELS[position]
            coarse = _study_solve_record(prediction, level)
            fine = _study_solve_record(prediction, level + 1)
            cost += position == 1 ? coarse["elapsed_s"] :
                    coarse["solve_s"] + coarse["preconditioner_s"]
            cost += fine["assembly_s"] + _study_machinery_cost(
                levels[position], name)
        end
    end
    if escalated
        final = reference["levels"][end]
        final["level"] == 3 || error("reference record lacks the escalation level")
        cost += final["costs"]["forward_total_s"] + final["costs"]["radiation_s"] +
                get(final["costs"], "algebraic_s", 0.0)
    end
    return cost
end

function _study_case_budget(prediction, reference_record, checks, level_position,
                            main_looks)
    level = prediction["levels"][level_position]
    algebraic = Float64.(level["algebraic_coarse_v"]) .+
                Float64.(level["algebraic_fine_v"])
    restriction = Float64.(level["restriction_field_change_v"])
    quadrature = Float64.(checks["quadrature_v"])
    compression = Float64.(checks["compression_v"])
    reference_levels = reference_record["levels"]
    fine_field = _study_complex_field(Dict(
        "real" => reference_levels[end - 1]["field_real"],
        "imag" => reference_levels[end - 1]["field_imag"]))[:, main_looks]
    reference_field = _study_complex_field(Dict(
        "real" => reference_levels[end]["field_real"],
        "imag" => reference_levels[end]["field_imag"]))[:, main_looks]
    reference_change = _study_field_difference_scales(
        reference_field, fine_field, 0.0) .+
        Float64.(reference_levels[end]["algebraic_field_change_v"])[main_looks]
    tolerance = algebraic .+ restriction .+ quadrature .+ compression .+
                reference_change
    return NumericalErrorBudget(; tolerance, algebraic, quadrature, compression,
        restriction, reference=reference_change, geometry=0.0)
end

function _study_wilson_interval(successes, total; z=1.959963984540054)
    total >= 1 || throw(ArgumentError("an interval needs at least one case"))
    0 <= successes <= total ||
        throw(ArgumentError("success count must lie between zero and the total"))
    p = successes / total
    denominator = 1 + z^2 / total
    center = (p + z^2 / (2total)) / denominator
    width = z * sqrt(p * (1 - p) / total + z^2 / (4total^2)) / denominator
    lower = successes == 0 ? 0.0 : max(0.0, center - width)
    upper = successes == total ? 1.0 : min(1.0, center + width)
    return (lower=lower, upper=upper)
end

function study_evaluation_main(population_dir, prediction_dir, reference_dir,
                               calibration_path, checks_path, output_path)
    ispath(output_path) && error("use a new evaluation output path")
    calibration = JSON.parsefile(calibration_path)
    checks = JSON.parsefile(checks_path)
    checks["status"] == "complete" && checks["source_unchanged"] ||
        error("numerical checks are incomplete or invalidated")
    calibration["source_sha256"] == source_digest() ||
        error("package source changed since calibration; recalibrate first")
    prediction_protocol_hash = calibration["prediction_protocol_hash"]
    reference_protocol_hash = calibration["reference_protocol_hash"]
    reference_protocol_record = JSON.parsefile(joinpath(
        reference_dir, "reference_protocol.json"))
    reference_protocol_record["protocol_hash"] == reference_protocol_hash ||
        error("reference protocol differs from the calibrated protocol")
    main_looks = Int.(reference_protocol_record["protocol"]["main_looks"])
    population = JSON.parsefile(joinpath(population_dir, "population_manifest.json"))
    manifest_digest = bytes2hex(sha256(read(joinpath(
        population_dir, "population_manifest.json"))))
    calibration["population_manifest_sha256"] == manifest_digest ||
        error("calibration and evaluation populations differ")
    bytes2hex(sha256(read(joinpath(
        population_dir, "masks.csv")))) ==
        population["artifact_sha256"]["masks.csv"] ||
        error("mask artifact differs from the registered population")
    _, derived_looks = reference_observation_grid(population_dir)
    derived_looks == main_looks ||
        error("registered looks differ from the calibrated protocol")
    rows = collect(CSV.File(joinpath(population_dir, "parameters.csv")))
    calibration_ids = [String(row.case_id) for row in rows if row.split == "calibration"]
    test_rows = [row for row in rows if row.split == "test"]
    length(test_rows) == 300 || error("test split count differs from the contract")
    training_ids = String.(calibration["training_case_ids"])
    richardson_order = Float64(calibration["richardson"]["order"])
    masks = _study_masks(population_dir, 12)
    names = _study_configuration_names()
    contexts = Dict{String,RCSCalibrationContext}()
    profiles = Dict{String,RCSCalibrationProfile}()
    for name in names
        record = calibration["contexts"][name]
        context = RCSCalibrationContext(
            algorithm_hash=record["algorithm_hash"],
            training_revision=record["training_revision"],
            distribution_id=record["distribution_id"],
            reference_protocol=record["reference_protocol"],
            look_id=record["look_id"], look_count=record["look_count"],
            levels=record["levels"], field_floor=record["field_floor"],
            absolute_field_tolerance=record["absolute_field_tolerance"],
            numerical_fraction=record["numerical_fraction"])
        scores = Float64.(calibration["calibration_scores"][name])
        profile = calibrate_rcs(scores, context;
            calibration_case_ids=calibration_ids, training_case_ids=training_ids,
            epsilon=STUDY_EPSILON, max_work_bytes=STUDY_MAX_WORK_BYTES)
        stored = calibration["profiles"][name]
        profile.quantile == stored["quantile"] &&
            profile.sorted_scores == Float64.(stored["sorted_scores"]) ||
            error("rebuilt calibration profile differs for $name")
        contexts[name] = context
        profiles[name] = profile
    end
    mask_ids = sort(collect(keys(masks)))
    aggregates = Dict(name => Dict(
        "covered" => 0, "total" => 0,
        "masks" => Dict(id => Dict(
            "pass" => 0, "fail" => 0, "decided" => 0,
            "false_pass" => 0, "false_fail" => 0,
            "escalated" => 0, "unresolved" => 0,
            "truth_pass" => 0, "truth_fail" => 0, "costs" => Float64[])
            for id in mask_ids))
        for name in names)
    offline_reference_s = 0.0
    offline_prediction_s = 0.0
    for row in vcat(
            [r for r in rows if r.split == "train"],
            [r for r in rows if r.split == "calibration"])
        case_id = String(row.case_id)
        offline_reference_s += JSON.parsefile(joinpath(
            reference_dir, case_id * ".json"))["total_elapsed_s"]
        offline_prediction_s += JSON.parsefile(joinpath(
            prediction_dir, case_id * ".json"))["elapsed_s"]
    end
    case_records = Any[]
    for row in test_rows
        case_id = String(row.case_id)
        prediction = _study_complete_record(
            prediction_dir, case_id, prediction_protocol_hash, "test")
        reference_record = _study_complete_reference(
            reference_dir, case_id, reference_protocol_hash)
        reference = _study_reference_field(reference_record, 3, main_looks)
        amplitude = abs(Float64(row.incident_amplitude_v_m))
        amplitude > 0 || error("test case has zero incident amplitude")
        truth_rcs = [4pi * norm(view(reference, :, look))^2 / amplitude^2
                     for look in axes(reference, 2)]
        for name in names
            context = contexts[name]
            profile = profiles[name]
            means, scales = _study_configuration(
                prediction, name; richardson_order=richardson_order,
                field_floor=context.field_floor)
            score = _study_score(prediction, reference, context, name;
                richardson_order=richardson_order)
            covered = score <= profile.quantile
            aggregates[name]["covered"] += covered
            aggregates[name]["total"] += 1
            for mask_id in mask_ids
                mask = masks[mask_id]
                truth = all(truth_rcs .<= mask) ? :pass : :fail
                decision = :unresolved
                decided_level = nothing
                reasons = Symbol[]
                for (position, level) in enumerate(STUDY_LEVELS)
                    budget = _study_case_budget(
                        prediction, reference_record, checks, position, main_looks)
                    report = screen_rcs_mask(
                        means[:, :, position], scales[:, position], profile, mask;
                        context, level,
                        incident_amplitude=Float64(row.incident_amplitude_v_m),
                        case_supported=true, numerical_budget=budget)
                    if report.decision != :unresolved
                        decision = report.decision
                        decided_level = position
                        break
                    end
                    append!(reasons, report.reasons)
                end
                escalated = decision == :unresolved
                escalated && (decision = truth)
                cost = _study_decision_cost(
                    prediction, reference_record, name, decided_level, escalated)
                bucket = aggregates[name]["masks"][mask_id]
                bucket["truth_pass"] += truth == :pass
                bucket["truth_fail"] += truth == :fail
                bucket["false_pass"] += decision == :pass && truth == :fail
                bucket["false_fail"] += decision == :fail && truth == :pass
                bucket["escalated"] += escalated
                bucket["unresolved"] += escalated
                bucket["decided"] += !escalated
                bucket[String(decision)] += 1
                push!(bucket["costs"], cost)
                push!(case_records, (
                    case_id, configuration=name, mask_id, truth=string(truth),
                    decision=string(decision), escalated,
                    decided_level=decided_level === nothing ? nothing :
                        STUDY_LEVELS[decided_level],
                    reasons=string.(unique(reasons)), covered, cost_s=cost))
            end
        end
    end
    summary = Dict{String,Any}()
    for name in names
        data = aggregates[name]
        interval = _study_wilson_interval(data["covered"], data["total"])
        mask_summary = Dict{String,Any}()
        for mask_id in mask_ids
            bucket = data["masks"][mask_id]
            mask_summary[mask_id] = (
                threshold_m2=masks[mask_id][1],
                conclusive=bucket["decided"], escalated=bucket["escalated"],
                false_pass=bucket["false_pass"], false_fail=bucket["false_fail"],
                truth_pass=bucket["truth_pass"], truth_fail=bucket["truth_fail"],
                mean_cost_s=sum(bucket["costs"]) / length(bucket["costs"]))
        end
        summary[name] = (
            coverage=(cases=data["covered"], total=data["total"],
                fraction=data["covered"] / data["total"], wilson95=interval),
            masks=mask_summary)
    end
    result = (
        schema_version=1,
        prediction_protocol_hash, reference_protocol_hash,
        population_manifest_sha256=manifest_digest,
        checks_case_id=checks["case_id"],
        checks_sha256=bytes2hex(sha256(read(checks_path))),
        calibration_sha256=bytes2hex(sha256(read(calibration_path))),
        source_sha256=source_digest(),
        epsilon=STUDY_EPSILON, richardson_order,
        offline_reference_s, offline_prediction_s,
        summary, cases=case_records,
    )
    open(output_path, "w") do io
        JSON.print(io, result, 2)
    end
    println("Evaluated ", length(test_rows), " held-out cases over ",
        length(mask_ids), " masks and ", length(names), " configurations.")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 4 || length(ARGS) == 6 ||
        error("usage: study_analysis.jl POPULATION PREDICTIONS REFERENCES NEW_OUTPUT_JSON [CALIBRATION_JSON CHECKS_JSON]")
    if length(ARGS) == 4
        study_calibration_main(abspath.(ARGS)...)
    else
        study_evaluation_main(
            abspath(ARGS[1]), abspath(ARGS[2]), abspath(ARGS[3]),
            abspath(ARGS[4]), abspath(ARGS[5]), abspath(ARGS[6]))
    end
end
