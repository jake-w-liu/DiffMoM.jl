# Diagnostics.jl — Energy conservation, condition number, and convergence utilities

export radiated_power, projected_power, input_power, energy_ratio, condition_diagnostics
export bistatic_rcs, backscatter_rcs

const _PROJECTED_POWER_FALLBACK_PRECISION = 65_536
const _PROJECTED_POWER_MIN_FALLBACK_PRECISION = 256

function _projected_power_component_bounds(values)
    minimum_low_bit = typemax(Int)
    maximum_high_bit = typemin(Int)
    @inbounds for value in values
        for component in (real(value), imag(value))
            isfinite(component) || return nothing
            iszero(component) && continue
            if component isa BigFloat
                rational_component = Rational{BigInt}(component)
                numerator_bits = abs(numerator(rational_component))
                denominator_bits = denominator(rational_component)
                denominator_exponent = ndigits(denominator_bits; base=2) - 1
                high_bit = ndigits(numerator_bits; base=2) - 1 -
                           denominator_exponent
                low_bit = trailing_zeros(numerator_bits) -
                          denominator_exponent
            elseif component isa AbstractFloat
                high_bit = exponent(abs(component))
                component_precision = precision(component)
                low_bit = high_bit - component_precision + 1
            else
                converted = try
                    Float64(component)
                catch err
                    @debug(
                        "projected-power bound conversion failed; using " *
                        "the conservative fallback precision",
                        component_type=typeof(component),
                        exception=(err, catch_backtrace()),
                    )
                    return nothing
                end
                isfinite(converted) || return nothing
                high_bit = exponent(abs(converted))
                component_precision = precision(converted)
                low_bit = high_bit - component_precision + 1
            end
            minimum_low_bit = min(minimum_low_bit, low_bit)
            maximum_high_bit = max(maximum_high_bit, high_bit)
        end
    end
    maximum_high_bit == typemin(Int) && return :zero
    return minimum_low_bit, maximum_high_bit
end

function _projected_power_fallback_precision(
    E_ff::Matrix{<:Number},
    grid::SphGrid,
    pol::AbstractMatrix{<:Complex},
)
    weight_bounds = _projected_power_component_bounds(grid.w)
    field_bounds = _projected_power_component_bounds(E_ff)
    polarization_bounds = _projected_power_component_bounds(pol)
    any(isnothing, (weight_bounds, field_bounds, polarization_bounds)) &&
        return _PROJECTED_POWER_FALLBACK_PRECISION
    any(==(:zero), (weight_bounds, field_bounds, polarization_bounds)) &&
        return _PROJECTED_POWER_MIN_FALLBACK_PRECISION

    minimum_bit = weight_bounds[1] +
                  2 * field_bounds[1] +
                  2 * polarization_bounds[1]
    maximum_bit = weight_bounds[2] +
                  2 * field_bounds[2] +
                  2 * polarization_bounds[2] + 4
    # Five significands contribute at most four carry bits. The remaining 70
    # bits cover both complex square expansions and every Int-addressable
    # sample, so accumulation is exact at the selected precision.
    required_precision = maximum_bit - minimum_bit + 70
    required_precision <= _PROJECTED_POWER_FALLBACK_PRECISION ||
        throw(ArgumentError(
            "projected-power exact reduction requires $required_precision " *
            "bits, exceeding the supported limit " *
            "$_PROJECTED_POWER_FALLBACK_PRECISION"))
    return max(required_precision, _PROJECTED_POWER_MIN_FALLBACK_PRECISION)
end

function _validate_farfield_samples(E_ff::Matrix{<:Number}, grid::SphGrid)
    NΩ = _validate_sph_grid(grid)
    size(E_ff) == (3, NΩ) ||
        throw(DimensionMismatch(
            "E_ff has size $(size(E_ff)), expected (3, $NΩ)"))
    all(isfinite, E_ff) ||
        throw(ArgumentError("E_ff must contain only finite values"))
    return NΩ
end

@inline function _farfield_intensity(E_ff::Matrix{<:Number}, q::Int)
    intensity =
        abs2(E_ff[1, q]) + abs2(E_ff[2, q]) + abs2(E_ff[3, q])
    isfinite(intensity) ||
        throw(OverflowError(
            "far-field intensity overflowed at sample $q"))
    return intensity
end

@inline function _add_nonnegative_compensated(
    total::Float64,
    correction::Float64,
    term::Float64,
)
    updated = total + term
    roundoff = total >= term ?
        (total - updated) + term :
        (term - updated) + total
    return updated, correction + roundoff
end

function _validated_rcs_amplitude(E0::Real)
    E0_float = try
        Float64(E0)
    catch err
        throw(ArgumentError(
            "E0 must be representable as Float64: $(sprint(showerror, err))"))
    end
    isfinite(E0_float) && !iszero(E0_float) ||
        throw(ArgumentError("E0 must be finite and nonzero, got $E0"))
    return abs(E0_float)
end

@noinline function _rcs_sample_bigfloat(
    E_ff::Matrix{<:Number},
    q::Int,
    E0_abs::Float64,
)::Float64
    return setprecision(BigFloat, 256) do
        intensity = zero(BigFloat)
        @inbounds for component in axes(E_ff, 1)
            value = E_ff[component, q]
            value_real = BigFloat(real(value))
            value_imag = BigFloat(imag(value))
            intensity += value_real * value_real + value_imag * value_imag
        end
        denominator = BigFloat(E0_abs)^2
        sigma_big = 4 * BigFloat(π) * intensity / denominator
        sigma = Float64(sigma_big)
        isfinite(sigma) ||
            throw(OverflowError("RCS overflowed at sample $q"))
        return sigma
    end
end

@inline function _rcs_sample(
    E_ff::Matrix{<:Number},
    q::Int,
    E0_abs::Float64,
)::Float64
    if !_ieee_bilinear_supported_eltype(eltype(E_ff), Float64)
        return _rcs_sample_bigfloat(E_ff, q, E0_abs)
    end

    value_1 = E_ff[1, q]
    value_2 = E_ff[2, q]
    value_3 = E_ff[3, q]
    if _ieee_dense_extreme_factor(E0_abs, Float64) ||
       _ieee_dense_extreme_factor(value_1, Float64) ||
       _ieee_dense_extreme_factor(value_2, Float64) ||
       _ieee_dense_extreme_factor(value_3, Float64)
        return _rcs_sample_bigfloat(E_ff, q, E0_abs)
    end
    magnitude_1 = hypot(Float64(real(value_1)), Float64(imag(value_1)))
    magnitude_2 = hypot(Float64(real(value_2)), Float64(imag(value_2)))
    magnitude_3 = hypot(Float64(real(value_3)), Float64(imag(value_3)))
    field_scale = max(magnitude_1, magnitude_2, magnitude_3)
    iszero(field_scale) && return 0.0

    scaled_intensity = abs2(magnitude_1 / field_scale) +
                       abs2(magnitude_2 / field_scale) +
                       abs2(magnitude_3 / field_scale)
    amplitude_ratio = field_scale / E0_abs
    if !isfinite(amplitude_ratio) ||
       amplitude_ratio < floatmin(Float64)
        return _rcs_sample_bigfloat(E_ff, q, E0_abs)
    end
    root_sigma = sqrt(4π * scaled_intensity) * amplitude_ratio
    sigma = root_sigma * root_sigma
    if !isfinite(sigma) || sigma < floatmin(Float64)
        return _rcs_sample_bigfloat(E_ff, q, E0_abs)
    end
    return sigma
end

@noinline function _radiated_power_exact(
    E_ff::Matrix{<:Number},
    grid::SphGrid,
    eta0::Float64,
)
    if _ieee_bilinear_supported_eltype(eltype(E_ff), Float64)
        accumulator = _IEEEBilinearAccumulator(Float64)
        @inbounds for q in eachindex(grid.w)
            weight = grid.w[q]
            for component in axes(E_ff, 1)
                value = E_ff[component, q]
                value_real = Float64(real(value))
                value_imag = Float64(imag(value))
                _ieee_bilinear_add_triple!(
                    accumulator, weight, value_real, value_real, 1)
                _ieee_bilinear_add_triple!(
                    accumulator, weight, value_imag, value_imag, 1)
            end
        end
        return _ieee_bilinear_finish(
            accumulator, "radiated power", 0.5, eta0)
    end

    return setprecision(BigFloat, _IEEE_BILINEAR_FALLBACK_PRECISION) do
        total = zero(BigFloat)
        @inbounds for q in eachindex(grid.w)
            intensity = zero(BigFloat)
            for component in axes(E_ff, 1)
                value = E_ff[component, q]
                value_real = BigFloat(real(value))
                value_imag = BigFloat(imag(value))
                intensity += value_real * value_real + value_imag * value_imag
            end
            total += BigFloat(grid.w[q]) * intensity
        end
        power = Float64(total / (2 * BigFloat(eta0)))
        isfinite(power) ||
            throw(OverflowError("radiated power is non-finite"))
        return power
    end
end

@noinline function _projected_power_exact(
    E_ff::Matrix{<:Number},
    grid::SphGrid,
    pol::AbstractMatrix{<:Complex},
    mask,
)
    precision_bits = _projected_power_fallback_precision(E_ff, grid, pol)
    return setprecision(BigFloat, precision_bits) do
        total = zero(BigFloat)
        @inbounds for q in eachindex(grid.w)
            if mask !== nothing && !mask[q]
                continue
            end
            projection_real = zero(BigFloat)
            projection_imag = zero(BigFloat)
            for component in axes(E_ff, 1)
                polarization = pol[component, q]
                field = E_ff[component, q]
                polarization_real = BigFloat(real(polarization))
                polarization_imag = BigFloat(imag(polarization))
                field_real = BigFloat(real(field))
                field_imag = BigFloat(imag(field))
                projection_real +=
                    polarization_real * field_real +
                    polarization_imag * field_imag
                projection_imag +=
                    polarization_real * field_imag -
                    polarization_imag * field_real
            end
            total += BigFloat(grid.w[q]) *
                     (projection_real * projection_real +
                      projection_imag * projection_imag)
        end
        power = Float64(total)
        isfinite(power) ||
            throw(OverflowError("projected power is non-finite"))
        return power
    end
end

"""
    radiated_power(E_ff, grid; eta0=376.730313668)

Compute total radiated power from far-field pattern:
  P_rad = (1/(2η₀)) ∫ |E∞(r̂)|² dΩ ≈ (1/(2η₀)) Σ_q w_q |E∞(r̂_q)|²

The 1/(2η₀) factor converts the far-field electric field intensity to
time-averaged Poynting flux (watts).
"""
function radiated_power(E_ff::Matrix{<:Number}, grid::SphGrid;
                        eta0::Float64=376.730313668)
    NΩ = _validate_farfield_samples(E_ff, grid)
    isfinite(eta0) && eta0 > 0 ||
        throw(ArgumentError("eta0 must be finite and positive, got $eta0"))
    needs_fallback = _ieee_bilinear_values_require_fallback(
        E_ff, Float64) ||
        _ieee_bilinear_values_require_fallback(grid.w, Float64) ||
        _ieee_dense_extreme_factor(eta0, Float64)
    needs_fallback && return _radiated_power_exact(E_ff, grid, eta0)
    P = 0.0
    correction = 0.0
    @inbounds for q in 1:NΩ
        term = grid.w[q] * _farfield_intensity(E_ff, q)
        P, correction =
            _add_nonnegative_compensated(P, correction, term)
        (isfinite(P) && isfinite(correction)) ||
            return _radiated_power_exact(E_ff, grid, eta0)
    end
    P += correction
    (isfinite(P) && P >= 0.0) ||
        return _radiated_power_exact(E_ff, grid, eta0)
    power = (P / eta0) / 2
    isfinite(power) || return _radiated_power_exact(E_ff, grid, eta0)
    return power
end

"""
    projected_power(E_ff, grid, pol; mask=nothing)

Compute polarization-projected angular power:
  P = Σ_q w_q |p_q^† E∞(r̂_q)|²

When `mask` is provided, only selected angular samples are included.
This is the discrete quantity represented by `I† Q I` when `Q` is
constructed with the same `pol` and `mask`.
"""
function projected_power(E_ff::Matrix{<:Number}, grid::SphGrid,
                         pol::AbstractMatrix{<:Complex}; mask=nothing)
    NΩ = _validate_farfield_samples(E_ff, grid)
    size(pol) == (3, NΩ) ||
        throw(DimensionMismatch(
            "pol has size $(size(pol)), expected (3, $NΩ)"))
    all(isfinite, pol) ||
        throw(ArgumentError("pol must contain only finite values"))
    if mask !== nothing
        mask isa AbstractVector{Bool} ||
            throw(ArgumentError(
                "mask must be an AbstractVector{Bool}, got $(typeof(mask))"))
        length(mask) == NΩ ||
            throw(DimensionMismatch(
                "mask length $(length(mask)) != $NΩ"))
    end

    needs_fallback =
        !_ieee_bilinear_supported_eltype(eltype(E_ff), Float64) ||
        !_ieee_bilinear_supported_eltype(eltype(pol), Float64) ||
        _ieee_bilinear_values_require_fallback(E_ff, Float64) ||
        _ieee_bilinear_values_require_fallback(pol, Float64) ||
        _ieee_bilinear_values_require_fallback(grid.w, Float64)
    needs_fallback &&
        return _projected_power_exact(E_ff, grid, pol, mask)

    P = 0.0
    correction = 0.0
    projection_error_factor =
        _ieee_product_error_factor(Float64, 2, 3)
    @inbounds for q in 1:NΩ
        if mask !== nothing && !mask[q]
            continue
        end
        projection_real = 0.0
        projection_imag = 0.0
        real_magnitude = 0.0
        imag_magnitude = 0.0
        for component in 1:3
            polarization = pol[component, q]
            field = E_ff[component, q]
            polarization_real = Float64(real(polarization))
            polarization_imag = Float64(imag(polarization))
            field_real = Float64(real(field))
            field_imag = Float64(imag(field))
            real_real = polarization_real * field_real
            imag_imag = polarization_imag * field_imag
            real_imag = polarization_real * field_imag
            imag_real = polarization_imag * field_real
            projection_real += real_real + imag_imag
            projection_imag += real_imag - imag_real
            real_magnitude += abs(real_real) + abs(imag_imag)
            imag_magnitude += abs(real_imag) + abs(imag_real)
        end
        if _ieee_product_component_is_suspicious(
                projection_real, real_magnitude,
                projection_error_factor) ||
           _ieee_product_component_is_suspicious(
                projection_imag, imag_magnitude,
                projection_error_factor)
            return _projected_power_exact(E_ff, grid, pol, mask)
        end
        yq = ComplexF64(projection_real, projection_imag)
        term = grid.w[q] * abs2(yq)
        P, correction =
            _add_nonnegative_compensated(P, correction, term)
        (isfinite(P) && isfinite(correction)) ||
            return _projected_power_exact(E_ff, grid, pol, mask)
    end
    P += correction
    (isfinite(P) && P >= 0.0) ||
        return _projected_power_exact(E_ff, grid, pol, mask)
    return P
end

"""
    input_power(I, v)

Compute the power delivered to the structure:
  P_in = -½ Re(I† v)

For a PEC scatterer with Z I = v, this is the power extracted from the
incident field by the induced currents.
"""
function input_power(I::Vector{<:Number}, v::Vector{<:Number})
    length(I) == length(v) ||
        throw(DimensionMismatch(
            "I length $(length(I)) != v length $(length(v))"))
    !isempty(I) ||
        throw(ArgumentError("I and v must contain at least one entry"))
    all(isfinite, I) ||
        throw(ArgumentError("I must contain only finite values"))
    all(isfinite, v) ||
        throw(ArgumentError("v must contain only finite values"))
    return _finite_scaled_dot_component(
        I, v, Val(:real), -0.5, "input power")
end

"""
    energy_ratio(I, v, E_ff, grid; eta0=376.730313668)

Compute the ratio P_rad / P_in as an energy conservation diagnostic.
For a lossless PEC structure, this should be ≈ 1.
For an impedance sheet with Re(Z_s) > 0, P_rad/P_in < 1 (absorbed power).
"""
function energy_ratio(I::Vector{<:Number}, v::Vector{<:Number},
                      E_ff::Matrix{<:Number}, grid::SphGrid;
                      eta0::Float64=376.730313668)
    P_in  = input_power(I, v)
    P_rad = radiated_power(E_ff, grid; eta0=eta0)
    !iszero(P_in) ||
        throw(DomainError(
            P_in, "energy ratio is undefined for zero input power"))
    ratio = P_rad / P_in
    isfinite(ratio) ||
        throw(OverflowError("energy ratio is non-finite"))
    return ratio
end

"""
    condition_diagnostics(Z)

Return condition number and singular value extremes of the MoM matrix.
"""
function condition_diagnostics(Z::Matrix{<:Number})
    (size(Z, 1) > 0 && size(Z, 2) > 0) ||
        throw(ArgumentError(
            "Z must have at least one row and one column, got size $(size(Z))"))
    all(isfinite, Z) ||
        throw(ArgumentError("Z must contain only finite values"))
    svs = svdvals(Z)
    all(isfinite, svs) ||
        throw(OverflowError(
            "singular-value computation produced non-finite values"))
    sv_max = first(svs)
    sv_min = last(svs)
    κ = iszero(sv_min) ? Inf : sv_max / sv_min
    return (cond=κ, sv_max=sv_max, sv_min=sv_min)
end

"""
    bistatic_rcs(E_ff; E0=1.0)

Compute bistatic radar cross section samples from far-field amplitudes:
  σ(r̂_q) = 4π |E∞(r̂_q)|² / |E0|²

Returns a real vector of length `NΩ` in linear units (m²).
"""
function bistatic_rcs(E_ff::Matrix{<:Number}; E0::Real=1.0)
    size(E_ff, 1) == 3 ||
        throw(DimensionMismatch(
            "E_ff has $(size(E_ff, 1)) rows, expected 3"))
    all(isfinite, E_ff) ||
        throw(ArgumentError("E_ff must contain only finite values"))
    NΩ = size(E_ff, 2)
    E0_abs = _validated_rcs_amplitude(E0)
    σ = zeros(Float64, NΩ)
    @inbounds for q in 1:NΩ
        σ[q] = _rcs_sample(E_ff, q, E0_abs)
    end
    return σ
end

"""
    backscatter_rcs(E_ff, grid, k_inc_hat; E0=1.0)

Return monostatic/backscatter RCS for a plane-wave incidence direction
`k_inc_hat` (unit propagation direction). The backscatter direction is
`-k_inc_hat`, mapped to the nearest sample on `grid`.

Returns a named tuple:
`(sigma, index, theta, phi, angular_error_deg)`.
"""
function backscatter_rcs(E_ff::Matrix{<:Number}, grid::SphGrid,
                         k_inc_hat::Vec3; E0::Real=1.0)
    NΩ = _validate_farfield_samples(E_ff, grid)
    all(isfinite, k_inc_hat) ||
        throw(ArgumentError("k_inc_hat components must be finite"))
    E0_abs = _validated_rcs_amplitude(E0)

    khat = _validated_farfield_direction(k_inc_hat)
    r_back = -khat

    best_idx = 1
    best_dot = -Inf
    @inbounds for q in 1:NΩ
        d = r_back[1] * grid.rhat[1, q] +
            r_back[2] * grid.rhat[2, q] +
            r_back[3] * grid.rhat[3, q]
        if d > best_dot
            best_dot = d
            best_idx = q
        end
    end

    ang_err = acos(clamp(best_dot, -1.0, 1.0)) * 180 / π
    sigma = _rcs_sample(E_ff, best_idx, E0_abs)

    return (
        sigma = sigma,
        index = best_idx,
        theta = grid.theta[best_idx],
        phi = grid.phi[best_idx],
        angular_error_deg = ang_err,
    )
end
