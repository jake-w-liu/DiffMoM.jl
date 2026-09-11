export RCSCalibrationContext, RCSCalibrationProfile, NumericalErrorBudget,
       RCSDecisionReport, rcs_field_scales, rcs_case_score, calibrate_rcs,
       rcs_ball_bounds, screen_rcs_mask

# Exact rational arithmetic keeps Float64 extreme/null cases and inclusive
# mask comparisons independent of overflow and global MPFR rounding state.
function _rational_sqrt_interval(value::Rational{BigInt})
    value >= 0 || throw(ArgumentError("squared field norm must be nonnegative"))
    iszero(value) && return (value, value)
    numerator_value, denominator_value = numerator(value), denominator(value)
    exponent = ndigits(numerator_value; base=2) - ndigits(denominator_value; base=2)
    below = exponent >= 0 ?
        numerator_value < (denominator_value << exponent) :
        (numerator_value << (-exponent)) < denominator_value
    below && (exponent -= 1)
    # 128 binary fraction bits exceed twice the Float64 significand width.
    shift = 128 - fld(exponent, 2)
    scaled_integer = shift >= 0 ?
        div(numerator_value << (2shift), denominator_value) :
        div(numerator_value, denominator_value << (-2shift))
    root = isqrt(scaled_integer)
    unit = shift >= 0 ? BigInt(1) // (BigInt(1) << shift) :
                        (BigInt(1) << (-shift)) // BigInt(1)
    lower = root * unit
    upper = lower^2 == value ? lower : (root + 1) * unit
    return lower, upper
end

function _round_nonnegative_rational(value::Rational{BigInt}, upward::Bool)
    value >= 0 || throw(ArgumentError("rounded quantity must be nonnegative"))
    result = Float64(value)
    if isinf(result)
        return upward ? Inf : floatmax(Float64)
    end
    exact_result = Rational{BigInt}(result)
    if upward && exact_result < value
        return nextfloat(result)
    elseif !upward && exact_result > value
        return prevfloat(result)
    end
    return result
end

function _exact_squared_field_norm(values)
    total = BigInt(0) // BigInt(1)
    for value in values
        total += Rational{BigInt}(real(value))^2 + Rational{BigInt}(imag(value))^2
    end
    return total
end

"""
    rcs_ball_bounds(mean_field, radius; incident_amplitude=1.0)

Outward-rounded lower/upper RCS bounds in square meters for a complex-field
Euclidean ball. Include both transverse components. Radius may be infinity,
which returns an unbounded interval. Zero incident amplitude is invalid.
Exact rational norm enclosures prevent overflow and near-null cancellation
from turning a field ball into a false finite RCS bound.
"""
function rcs_ball_bounds(mean_field::AbstractVector{<:Number}, radius::Real;
                         incident_amplitude::Real=1.0)
    !isempty(mean_field) && all(isfinite, mean_field) ||
        throw(ArgumentError("field mean must be nonempty and finite"))
    radius >= 0 && !isnan(radius) ||
        throw(ArgumentError("field radius must be nonnegative"))
    amplitude = incident_amplitude
    isfinite(amplitude) && !iszero(amplitude) ||
        throw(ArgumentError("incident amplitude must be finite and nonzero"))
    isinf(radius) && return (lower=0.0, upper=Inf)
    amplitude_exact = abs(Rational{BigInt}(amplitude))
    squared = _exact_squared_field_norm(mean_field) / amplitude_exact^2
    norm_lower, norm_upper = _rational_sqrt_interval(squared)
    normalized_radius = Rational{BigInt}(radius) / amplitude_exact
    lower_amplitude = max(BigInt(0) // BigInt(1), norm_lower - normalized_radius)
    upper_amplitude = norm_upper + normalized_radius
    # Adjacent Float64 values enclose the correctly rounded Julia pi constant.
    pi_lower = Rational{BigInt}(prevfloat(Float64(pi)))
    pi_upper = Rational{BigInt}(nextfloat(Float64(pi)))
    return (
        lower=_round_nonnegative_rational(4pi_lower * lower_amplitude^2, false),
        upper=_round_nonnegative_rational(4pi_upper * upper_amplitude^2, true))
end

"""
    RCSCalibrationContext(; algorithm_hash, training_revision, distribution_id,
        reference_protocol, look_id, look_count, levels, field_floor,
        absolute_field_tolerance, numerical_fraction=0.1)

Frozen prediction/calibration context. The algorithm hash must identify the
source and prediction settings (40- or 64-digit hexadecimal digest).
The caller supplies provenance from the executed implementation. Whole-case
exchangeability is a study assumption, not something a metadata label proves.
"""
struct RCSCalibrationContext
    algorithm_hash::String
    training_revision::String
    distribution_id::String
    reference_protocol::String
    look_id::String
    look_count::Int
    levels::Tuple{Vararg{Int}}
    field_floor::Float64
    absolute_field_tolerance::Float64
    numerical_fraction::Float64
end

function RCSCalibrationContext(;
        algorithm_hash::AbstractString, training_revision::AbstractString,
        distribution_id::AbstractString, reference_protocol::AbstractString,
        look_id::AbstractString, look_count::Integer, levels,
        field_floor::Real, absolute_field_tolerance::Real,
        numerical_fraction::Real=0.1)
    occursin(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$", algorithm_hash) ||
        throw(ArgumentError("algorithm_hash must be a hexadecimal source/settings digest"))
    all(value -> !isempty(strip(value)),
        (training_revision, distribution_id, reference_protocol, look_id)) ||
        throw(ArgumentError("calibration provenance fields must be nonempty"))
    count = _validated_resource_limit("look_count", look_count)
    level_values = Tuple(Int(level) for level in levels)
    !isempty(level_values) && all(level -> level >= 0, level_values) &&
        allunique(level_values) ||
        throw(ArgumentError("candidate levels must be distinct nonnegative integers"))
    floor_value, tolerance, fraction =
        Float64(field_floor), Float64(absolute_field_tolerance), Float64(numerical_fraction)
    all(value -> isfinite(value) && value > 0, (floor_value, tolerance)) ||
        throw(ArgumentError("field floor and absolute tolerance must be finite and positive"))
    isfinite(fraction) && 0 < fraction < 1 ||
        throw(ArgumentError("numerical_fraction must lie strictly between zero and one"))
    return RCSCalibrationContext(
        lowercase(String(algorithm_hash)), String(training_revision),
        String(distribution_id), String(reference_protocol), String(look_id),
        count, level_values, floor_value, tolerance, fraction)
end

_calibration_context_key(c::RCSCalibrationContext) =
    (c.algorithm_hash, c.training_revision, c.distribution_id, c.reference_protocol,
     c.look_id, c.look_count, c.levels, c.field_floor,
     c.absolute_field_tolerance, c.numerical_fraction)
Base.:(==)(a::RCSCalibrationContext, b::RCSCalibrationContext) =
    _calibration_context_key(a) == _calibration_context_key(b)
Base.hash(context::RCSCalibrationContext, seed::UInt) =
    hash(_calibration_context_key(context), seed)

"""
    rcs_field_scales(outputs, context)

Return sqrt(trace(Sigma_j)) plus the frozen positive field floor at each look.
The shared output square-root factor supplies the trace without overflowing
a sum of large variances.
"""
function rcs_field_scales(outputs::ErrorOutputs, context::RCSCalibrationContext)
    p = 2context.look_count
    length(outputs.mean) == p && size(outputs.square_root, 1) == p &&
        size(outputs.covariance) == (p, p) ||
        throw(DimensionMismatch("RCS outputs require two components at each prescribed look"))
    all(isfinite, outputs.mean) && all(isfinite, outputs.square_root) &&
        all(isfinite, outputs.covariance) ||
        throw(ArgumentError("RCS means and factors must be finite"))
    scales = [norm(view(outputs.square_root, (2j-1):(2j), :)) + context.field_floor
              for j in 1:context.look_count]
    all(value -> isfinite(value) && value > 0, scales) ||
        throw(ArgumentError("RCS field scales are outside the supported numerical range"))
    return scales
end

"""
    rcs_case_score(reference, means, scales, context)

Maximum normalized complex-field error across every prescribed look AND
candidate level of one case. Reference has shape (2, looks), means has
shape (2, looks, levels), and scales has shape (looks, levels). The caller
must freeze the scale rule before calibration. Scores round upward.
"""
function rcs_case_score(
        reference::AbstractMatrix{<:Number}, means::AbstractArray{<:Number,3},
        scales::AbstractMatrix{<:Real}, context::RCSCalibrationContext)
    looks, levels = context.look_count, length(context.levels)
    size(reference) == (2, looks) && size(means) == (2, looks, levels) &&
        size(scales) == (looks, levels) ||
        throw(DimensionMismatch("case score requires all prescribed looks and levels"))
    all(isfinite, reference) && all(isfinite, means) &&
        all(value -> isfinite(value) && value >= context.field_floor, scales) ||
        throw(ArgumentError("case fields must be finite and scales at least the frozen field floor"))
    result = 0.0
    for level in 1:levels, look in 1:looks
        squared = BigInt(0) // BigInt(1)
        for component in 1:2
            real_error = Rational{BigInt}(real(reference[component, look])) -
                         Rational{BigInt}(real(means[component, look, level]))
            imag_error = Rational{BigInt}(imag(reference[component, look])) -
                         Rational{BigInt}(imag(means[component, look, level]))
            squared += real_error^2 + imag_error^2
        end
        _, upper = _rational_sqrt_interval(
            squared / Rational{BigInt}(scales[look, level])^2)
        result = max(result, _round_nonnegative_rational(upper, true))
    end
    return result
end

function _calibration_fraction(epsilon::Real)
    if epsilon isa Rational
        result = Rational{BigInt}(epsilon)
    else
        value = Float64(epsilon)
        isfinite(value) && 0 < value < 1 ||
            throw(ArgumentError("epsilon must lie strictly between zero and one"))
        # Treat the displayed decimal error level exactly, avoiding a rounded
        # floating product crossing an integral order-statistic boundary.
        parts = split(lowercase(string(value)), 'e')
        mantissa = split(parts[1], '.')
        decimals = length(mantissa) == 2 ? length(mantissa[2]) : 0
        digits = join(mantissa)
        exponent = length(parts) == 2 ? parse(Int, parts[2]) : 0
        shift = exponent - decimals
        numerator_value = parse(BigInt, digits)
        result = shift >= 0 ? numerator_value * big(10)^shift // BigInt(1) :
                              numerator_value // big(10)^(-shift)
    end
    0 < result < 1 || throw(ArgumentError("epsilon must lie strictly between zero and one"))
    return result
end

"""
    RCSCalibrationProfile

Whole-case split-calibration scores, provenance, augmented order statistic,
and explicit training/calibration case IDs. An infinite quantile requires
abstention. Coverage is marginal over exchangeable whole cases and refers to
the recorded numerical reference, not exact continuum or conditional risk.
"""
struct RCSCalibrationProfile
    context::RCSCalibrationContext
    epsilon::Rational{BigInt}
    sorted_scores::Vector{Float64}
    quantile_index::Int
    quantile::Float64
    calibration_cases::Tuple{Vararg{String}}
    training_cases::Tuple{Vararg{String}}
    signature::UInt
end

function _calibration_profile_signature(context, epsilon, scores, index, quantile, cases, training)
    return _content_fingerprint(
        (_calibration_context_key(context), epsilon, scores, index, quantile, cases, training),
        UInt(0))
end

function _validate_calibration_profile(profile::RCSCalibrationProfile)
    profile.signature == _calibration_profile_signature(
        profile.context, profile.epsilon, profile.sorted_scores, profile.quantile_index,
        profile.quantile, profile.calibration_cases, profile.training_cases) ||
        throw(ArgumentError("calibration data changed; rebuild the profile"))
    return nothing
end

"""
    calibrate_rcs(case_scores, context; calibration_case_ids,
        training_case_ids=String[], epsilon=0.05,
        max_work_bytes=2_000_000_000)

Use ceil((n+1)*(1-epsilon)) in the scores augmented by infinity. Each score
must come from a whole case through the same frozen prediction algorithm.
Reject repeated calibration IDs and overlap with training IDs.
"""
function calibrate_rcs(scores::AbstractVector{<:Real}, context::RCSCalibrationContext;
        calibration_case_ids::AbstractVector{<:AbstractString},
        training_case_ids::AbstractVector{<:AbstractString}=String[],
        epsilon::Real=0.05, max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    n = length(scores)
    length(calibration_case_ids) == n ||
        throw(DimensionMismatch("each calibration score needs one whole-case ID"))
    basic_bytes = _checked_payload_sum(
        "case calibration", _checked_array_payload_bytes(Float64, 4, n),
        _checked_array_payload_bytes(UInt8, 256, n + length(training_case_ids)))
    _enforce_payload_limit(basic_bytes, max_work_bytes, "case calibration", "max_work_bytes")
    _enforce_payload_limit(
        _checked_payload_sum(
            "case calibration", Base.summarysize((scores, calibration_case_ids, training_case_ids)),
            basic_bytes),
        max_work_bytes, "case calibration", "max_work_bytes")
    all(value -> !isnan(value) && value >= 0, scores) ||
        throw(ArgumentError("calibration scores must be nonnegative and not NaN"))
    calibration_ids, training_ids = String.(calibration_case_ids), String.(training_case_ids)
    all(id -> !isempty(strip(id)), calibration_ids) &&
        all(id -> !isempty(strip(id)), training_ids) ||
        throw(ArgumentError("case IDs must be nonempty"))
    allunique(calibration_ids) && allunique(training_ids) ||
        throw(ArgumentError("training and calibration IDs must each be unique"))
    isempty(intersect(Set(calibration_ids), Set(training_ids))) ||
        throw(ArgumentError("training and calibration cases must be disjoint"))
    fraction = _calibration_fraction(epsilon)
    index = Int(cld((BigInt(n) + 1) * (denominator(fraction) - numerator(fraction)),
                    denominator(fraction)))
    converted = [isinf(score) ? Inf :
        _round_nonnegative_rational(Rational{BigInt}(score), true) for score in scores]
    order = sortperm(converted)
    ordered = converted[order]
    quantile = index > n ? Inf : ordered[index]
    ordered_cases, training = Tuple(calibration_ids[order]), Tuple(training_ids)
    return RCSCalibrationProfile(
        context, fraction, ordered, index, quantile, ordered_cases, training,
        _calibration_profile_signature(context, fraction, ordered, index, quantile,
                                       ordered_cases, training))
end

"""
    NumericalErrorBudget(; tolerance, algebraic=nothing, quadrature=nothing,
        compression=nothing, restriction=nothing, reference=nothing,
        geometry=nothing)

Measured field changes in volts for six separate numerical checks. Tolerance
and measurements may be scalars applying to every look, or vectors with one
entry per look. Missing measurements are unchecked, not zero. These are
empirical numerical budgets, not a theorem bounding continuum error.
"""
struct NumericalErrorBudget
    tolerance::Union{Float64,Vector{Float64}}
    changes::NamedTuple
    function NumericalErrorBudget(tolerance, changes::NamedTuple)
        keys(changes) == (:algebraic, :quadrature, :compression, :restriction, :reference, :geometry) ||
            throw(ArgumentError("a numerical budget must record all six checks"))
        limit = _numerical_measurement(tolerance)
        limit === nothing && throw(ArgumentError("numerical tolerance must be specified"))
        return new(limit, map(_numerical_measurement, changes))
    end
end

function _numerical_measurement(value)
    value === nothing && return nothing
    if value isa Real && !(value isa Bool)
        converted = Float64(value)
        isfinite(converted) && converted >= 0 ||
            throw(ArgumentError("numerical measurements must be finite and nonnegative"))
        return converted
    elseif value isa AbstractVector
        !isempty(value) && all(x -> x isa Real && !(x isa Bool), value) ||
            throw(ArgumentError("numerical measurement vectors must contain real values"))
        converted = Float64.(value)
        all(x -> isfinite(x) && x >= 0, converted) ||
            throw(ArgumentError("numerical measurements must be finite and nonnegative"))
        return converted
    end
    throw(ArgumentError("numerical measurements must be real scalars, vectors, or nothing"))
end

function _numerical_budget_values(value, looks)
    if value isa Vector
        length(value) == looks ||
            throw(DimensionMismatch("numerical measurements need one entry per look"))
        all(x -> isfinite(x) && x >= 0, value) ||
            throw(ArgumentError("numerical budget contains an invalid measurement"))
        return value
    end
    return fill(value, looks)
end

function NumericalErrorBudget(; tolerance, algebraic=nothing, quadrature=nothing,
        compression=nothing, restriction=nothing, reference=nothing, geometry=nothing)
    inputs = (algebraic=algebraic, quadrature=quadrature, compression=compression,
              restriction=restriction, reference=reference, geometry=geometry)
    return NumericalErrorBudget(tolerance, inputs)
end

"""
    RCSDecisionReport

Finite-look pass/fail/unresolved decision and its linear-unit bounds.
Cost is the caller's measured breakdown, or nothing when not supplied.
An unresolved report never supplies a fabricated compliance probability.
"""
struct RCSDecisionReport
    decision::Symbol
    lower::Vector{Float64}
    upper::Vector{Float64}
    radii::Vector{Float64}
    scales::Vector{Float64}
    reasons::Vector{Symbol}
    level::Int
    reference_protocol::String
    cost::Union{Nothing,NamedTuple}
end

"""
    screen_rcs_mask(outputs, calibration, mask; context, level,
        incident_amplitude=1.0, case_supported=false,
        numerical_budget=nothing, cost=nothing)

Pass only if every upper RCS bound is at most its mask. Fail only if at least
one lower bound is strictly above its mask. Otherwise abstain. Missing
numerical checks, unsupported cases, stale contexts, unregistered levels,
and infinite calibration quantiles produce unresolved decisions.
"""
function screen_rcs_mask(
        mean_field::AbstractMatrix{<:Number}, supplied_scales::AbstractVector{<:Real},
        profile::RCSCalibrationProfile, mask::AbstractVector{<:Real};
        context::RCSCalibrationContext, level::Integer,
        incident_amplitude::Real=1.0, case_supported::Bool=false,
        numerical_budget::Union{Nothing,NumericalErrorBudget}=nothing,
        cost::Union{Nothing,NamedTuple}=nothing)
    looks = context.look_count
    _validate_calibration_profile(profile)
    size(mean_field) == (2, looks) && length(supplied_scales) == looks ||
        throw(DimensionMismatch("screening requires two field components and one scale per look"))
    all(isfinite, mean_field) &&
        all(value -> isfinite(value) && value >= context.field_floor, supplied_scales) ||
        throw(ArgumentError("field means must be finite and scales at least the frozen floor"))
    length(mask) == looks ||
        throw(DimensionMismatch("mask must have one value per prescribed look"))
    all(value -> isfinite(value) && value >= 0, mask) ||
        throw(ArgumentError("RCS masks must be finite and nonnegative"))
    amplitude = incident_amplitude
    isfinite(amplitude) && !iszero(amplitude) ||
        throw(ArgumentError("incident amplitude must be finite and nonzero"))
    cost === nothing || all(value -> value isa Real && isfinite(value) && value >= 0, cost) ||
        throw(ArgumentError("cost entries must be measured nonnegative finite numbers"))
    scales = Float64.(supplied_scales)
    all(isfinite, scales) ||
        throw(ArgumentError("field scales must be representable finite Float64 values"))
    radii = [isinf(profile.quantile) ? Inf :
             _round_nonnegative_rational(
                 Rational{BigInt}(profile.quantile) * Rational{BigInt}(scale), true)
             for scale in scales]
    reasons = Symbol[]
    context == profile.context || push!(reasons, :incompatible_calibration)
    level in context.levels || push!(reasons, :unregistered_level)
    case_supported || push!(reasons, :out_of_support)
    isfinite(profile.quantile) || push!(reasons, :unbounded_calibration)
    if numerical_budget === nothing
        push!(reasons, :numerical_checks_missing)
    else
        allowed = max.(context.absolute_field_tolerance, context.numerical_fraction .* radii)
        tolerances = _numerical_budget_values(numerical_budget.tolerance, looks)
        all(tolerances .<= allowed) ||
            push!(reasons, :numerical_tolerance_exceeds_radius_budget)
        for (name, value) in pairs(numerical_budget.changes)
            if value === nothing
                push!(reasons, Symbol("unchecked_", name))
            elseif !all(_numerical_budget_values(value, looks) .<= tolerances)
                push!(reasons, Symbol("exceeded_", name))
            end
        end
    end
    lower, upper = zeros(looks), fill(Inf, looks)
    if !isempty(reasons)
        return RCSDecisionReport(
            :unresolved, lower, upper, radii, scales, reasons, Int(level),
            profile.context.reference_protocol, cost)
    end
    for look in 1:looks
        bounds = rcs_ball_bounds(
            view(mean_field, :, look), radii[look];
            incident_amplitude=amplitude)
        lower[look], upper[look] = bounds.lower, bounds.upper
    end
    decision = all(upper .<= mask) ? :pass : any(lower .> mask) ? :fail : :unresolved
    decision === :unresolved && push!(reasons, :mask_intersection)
    return RCSDecisionReport(
        decision, lower, upper, radii, scales, reasons, Int(level),
        profile.context.reference_protocol, cost)
end

function screen_rcs_mask(
        outputs::ErrorOutputs, profile::RCSCalibrationProfile, mask::AbstractVector{<:Real};
        context::RCSCalibrationContext, kwargs...)
    scales = rcs_field_scales(outputs, context)
    return screen_rcs_mask(
        reshape(outputs.mean, 2, context.look_count), scales, profile, mask;
        context=context, kwargs...)
end
