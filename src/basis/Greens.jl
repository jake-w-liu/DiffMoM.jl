# Greens.jl — Free-space scalar Green's function and derivatives
#
# Convention: exp(+iωt), so G(r,r') = exp(-ik|r-r'|) / (4π|r-r'|)

export greens, greens_smooth, grad_greens

const _GREEN_FALLBACK_PRECISION = 2304
const _GREEN_NORMAL_EXPONENT_LIMIT = 128
const _GREEN_PHASE_FAST_EXPONENT_LIMIT = 40
const _GREEN_EXPREL_TERMS = 18
const _INV_FOUR_PI = inv(4.0 * Float64(π))
const _GreenFloatWavenumber = Union{Float64,ComplexF64}
const _GreenMachineFloat = Union{Float16,Float32,Float64}

@inline function _validate_green_arguments(r::SVector{3},
                                           rp::SVector{3},
                                           k)
    all(isfinite, r) ||
        throw(ArgumentError(
            "Green-function observation point must be finite"))
    all(isfinite, rp) ||
        throw(ArgumentError(
            "Green-function source point must be finite"))
    k isa Number ||
        throw(ArgumentError(
            "Green-function wavenumber must be numeric, got $(typeof(k))"))
    isfinite(k) ||
        throw(ArgumentError(
            "Green-function wavenumber must be finite, got $k"))
    return nothing
end

@inline function _green_promoted_real_type(
        r::SVector{3}, rp::SVector{3}, k)
    eltype(r) <: Real && eltype(rp) <: Real ||
        throw(ArgumentError(
            "Green-function points must have real coordinates"))
    return promote_type(
        Float64, eltype(r), eltype(rp), typeof(real(k)))
end

@inline function _green_promoted_geometry(
        r::SVector{3}, rp::SVector{3}, k)
    real_type = _green_promoted_real_type(r, rp, k)
    dx = real_type(r[1]) - real_type(rp[1])
    dy = real_type(r[2]) - real_type(rp[2])
    dz = real_type(r[3]) - real_type(rp[3])
    distance = hypot(hypot(dx, dy), dz)
    wavenumber = Complex{real_type}(k)
    return dx, dy, dz, distance, wavenumber
end

@inline function _green_has_exact_float64_conversions(
        r::SVector{3}, rp::SVector{3}, k)
    return eltype(r) <: _GreenMachineFloat &&
           eltype(rp) <: _GreenMachineFloat &&
           typeof(real(k)) <: _GreenMachineFloat
end

@inline function _greens_unchecked(r::SVector{3}, rp::SVector{3}, k)
    real_type = _green_promoted_real_type(r, rp, k)
    if real_type === Float64
        _green_has_exact_float64_conversions(r, rp, k) &&
            return _greens_unchecked(
                Vec3(r), Vec3(rp), ComplexF64(k))
        return _greens_exact(r, rp, k)
    end
    _, _, _, distance, wavenumber =
        _green_promoted_geometry(r, rp, k)
    iszero(distance) && return zero(wavenumber)
    phase_rate = -Complex{real_type}(0, 1) * wavenumber
    return exp(phase_rate * distance) /
           (4 * real_type(π) * distance)
end

@inline function _greens_smooth_unchecked(r::SVector{3}, rp::SVector{3}, k)
    real_type = _green_promoted_real_type(r, rp, k)
    if real_type === Float64
        _green_has_exact_float64_conversions(r, rp, k) &&
            return _greens_smooth_unchecked(
                Vec3(r), Vec3(rp), ComplexF64(k))
        return _greens_smooth_exact(r, rp, k)
    end
    _, _, _, distance, wavenumber =
        _green_promoted_geometry(r, rp, k)
    phase_rate = -Complex{real_type}(0, 1) * wavenumber
    iszero(distance) &&
        return phase_rate / (4 * real_type(π))
    return expm1(phase_rate * distance) /
           (4 * real_type(π) * distance)
end

@inline function _grad_greens_unchecked(r::SVector{3}, rp::SVector{3}, k)
    real_type = _green_promoted_real_type(r, rp, k)
    if real_type === Float64
        _green_has_exact_float64_conversions(r, rp, k) &&
            return _grad_greens_unchecked(
                Vec3(r), Vec3(rp), ComplexF64(k))
        return _grad_greens_exact(r, rp, k)
    end
    dx, dy, dz, distance, wavenumber =
        _green_promoted_geometry(r, rp, k)
    if iszero(distance)
        value = zero(wavenumber)
        return SVector{3}(value, value, value)
    end
    phase_rate = -Complex{real_type}(0, 1) * wavenumber
    G = exp(phase_rate * distance) /
        (4 * real_type(π) * distance)
    dGdR = (phase_rate - inv(distance)) * G
    return dGdR * SVector(dx, dy, dz) / distance
end

@inline function _green_float_geometry(r::Vec3, rp::Vec3)
    dx = r[1] - rp[1]
    dy = r[2] - rp[2]
    dz = r[3] - rp[3]
    return dx, dy, dz, hypot(hypot(dx, dy), dz)
end

@inline function _green_float_component_requires_fallback(value::Float64)
    isfinite(value) || return true
    iszero(value) && return false
    value_exponent = exponent(value)
    return value_exponent < -_GREEN_NORMAL_EXPONENT_LIMIT ||
           value_exponent > _GREEN_NORMAL_EXPONENT_LIMIT
end

@inline function _green_float_phase_requires_fallback(
        component::Float64, distance::Float64)
    (iszero(component) || iszero(distance)) && return false
    return exponent(abs(component)) + exponent(distance) + 1 >
           _GREEN_PHASE_FAST_EXPONENT_LIMIT
end

@inline function _green_float_requires_fallback(
    r::Vec3,
    rp::Vec3,
    dx::Float64,
    dy::Float64,
    dz::Float64,
    distance::Float64,
    k::_GreenFloatWavenumber,
    phase_argument::ComplexF64,
)
    return _green_float_component_requires_fallback(r[1]) ||
           _green_float_component_requires_fallback(r[2]) ||
           _green_float_component_requires_fallback(r[3]) ||
           _green_float_component_requires_fallback(rp[1]) ||
           _green_float_component_requires_fallback(rp[2]) ||
           _green_float_component_requires_fallback(rp[3]) ||
           _green_float_component_requires_fallback(dx) ||
           _green_float_component_requires_fallback(dy) ||
           _green_float_component_requires_fallback(dz) ||
           _green_float_component_requires_fallback(distance) ||
           _green_float_component_requires_fallback(Float64(real(k))) ||
           _green_float_component_requires_fallback(Float64(imag(k))) ||
           _green_float_phase_requires_fallback(Float64(real(k)), distance) ||
           _green_float_phase_requires_fallback(Float64(imag(k)), distance) ||
           _green_float_component_requires_fallback(real(phase_argument)) ||
           _green_float_component_requires_fallback(imag(phase_argument)) ||
           abs(real(phase_argument)) > 128.0
end

@inline function _green_exprel(phase_argument::ComplexF64)
    term = one(phase_argument)
    total = term
    correction = zero(phase_argument)
    @inbounds for order in 1:_GREEN_EXPREL_TERMS
        term *= phase_argument / (order + 1)
        corrected_term = term - correction
        updated_total = total + corrected_term
        correction = (updated_total - total) - corrected_term
        total = updated_total
    end
    return total
end

@inline function _green_radial_exponential_factor(
    phase_argument::ComplexF64)
    if abs(phase_argument) > 0.5
        return exp(phase_argument) *
               (phase_argument - one(phase_argument))
    end

    total = -one(phase_argument)
    correction = zero(phase_argument)
    term = phase_argument * phase_argument / 2
    @inbounds for order in 2:_GREEN_EXPREL_TERMS
        corrected_term = term - correction
        updated_total = total + corrected_term
        correction = (updated_total - total) - corrected_term
        total = updated_total
        order == _GREEN_EXPREL_TERMS && break
        term *= phase_argument * order / ((order + 1) * (order - 1))
    end
    return total
end

@noinline function _greens_exact(
    r::SVector{3,<:Real}, rp::SVector{3,<:Real}, k::Number)
    return setprecision(BigFloat, _GREEN_FALLBACK_PRECISION) do
        dx = BigFloat(r[1]) - BigFloat(rp[1])
        dy = BigFloat(r[2]) - BigFloat(rp[2])
        dz = BigFloat(r[3]) - BigFloat(rp[3])
        distance = sqrt(dx * dx + dy * dy + dz * dz)
        iszero(distance) && return 0.0 + 0.0im

        q = Complex{BigFloat}(BigFloat(imag(k)), -BigFloat(real(k)))
        phase_argument = q * distance
        inv_four_pi = inv(4 * BigFloat(π))
        return ComplexF64(exp(phase_argument) * (inv_four_pi / distance))
    end
end

@noinline function _greens_smooth_exact(
    r::SVector{3,<:Real}, rp::SVector{3,<:Real}, k::Number)
    return setprecision(BigFloat, _GREEN_FALLBACK_PRECISION) do
        dx = BigFloat(r[1]) - BigFloat(rp[1])
        dy = BigFloat(r[2]) - BigFloat(rp[2])
        dz = BigFloat(r[3]) - BigFloat(rp[3])
        distance = sqrt(dx * dx + dy * dy + dz * dz)
        q = Complex{BigFloat}(BigFloat(imag(k)), -BigFloat(real(k)))
        inv_four_pi = inv(4 * BigFloat(π))
        iszero(distance) && return ComplexF64(q * inv_four_pi)

        phase_argument = q * distance
        return ComplexF64(
            expm1(phase_argument) * (inv_four_pi / distance))
    end
end

@noinline function _grad_greens_exact(
    r::SVector{3,<:Real}, rp::SVector{3,<:Real}, k::Number)
    return setprecision(BigFloat, _GREEN_FALLBACK_PRECISION) do
        dx = BigFloat(r[1]) - BigFloat(rp[1])
        dy = BigFloat(r[2]) - BigFloat(rp[2])
        dz = BigFloat(r[3]) - BigFloat(rp[3])
        distance = sqrt(dx * dx + dy * dy + dz * dz)
        if iszero(distance)
            return CVec3(0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im)
        end

        q = Complex{BigFloat}(BigFloat(imag(k)), -BigFloat(real(k)))
        phase_argument = q * distance
        inv_four_pi = inv(4 * BigFloat(π))
        radial_factor = exp(phase_argument) * inv_four_pi *
                        (phase_argument - one(phase_argument))
        zero_value = zero(phase_argument)

        gx = iszero(dx) ? zero_value :
             ((radial_factor * (dx / distance)) / distance) / distance
        gy = iszero(dy) ? zero_value :
             ((radial_factor * (dy / distance)) / distance) / distance
        gz = iszero(dz) ? zero_value :
             ((radial_factor * (dz / distance)) / distance) / distance
        return CVec3(ComplexF64(gx), ComplexF64(gy), ComplexF64(gz))
    end
end

@inline function _green_gradient_component(
    delta::Float64,
    distance::Float64,
    radial_exponential_factor::ComplexF64,
)
    iszero(delta) && return 0.0 + 0.0im
    direction = delta / distance
    if distance >= 1.0
        radial_scale = (radial_exponential_factor / distance) / distance
        return _INV_FOUR_PI * (radial_scale * direction)
    end
    return ((_INV_FOUR_PI * radial_exponential_factor * direction) /
            distance) / distance
end

@inline function _greens_unchecked(
    r::Vec3, rp::Vec3, k::_GreenFloatWavenumber)
    dx, dy, dz, distance = _green_float_geometry(r, rp)
    iszero(distance) && return 0.0 + 0.0im

    q = ComplexF64(imag(k), -real(k))
    phase_argument = q * distance
    if _green_float_requires_fallback(
        r, rp, dx, dy, dz, distance, k, phase_argument)
        return _greens_exact(r, rp, k)
    end

    value = exp(phase_argument) * (_INV_FOUR_PI / distance)
    isfinite(value) && return value
    return _greens_exact(r, rp, k)
end

@inline function _greens_smooth_unchecked(
    r::Vec3, rp::Vec3, k::_GreenFloatWavenumber)
    dx, dy, dz, distance = _green_float_geometry(r, rp)
    q = ComplexF64(imag(k), -real(k))
    iszero(distance) && return q * _INV_FOUR_PI

    phase_argument = q * distance
    if _green_float_requires_fallback(
        r, rp, dx, dy, dz, distance, k, phase_argument)
        return _greens_smooth_exact(r, rp, k)
    end

    value = if abs(phase_argument) <= 0.5
        (q * _INV_FOUR_PI) * _green_exprel(phase_argument)
    else
        expm1(phase_argument) * (_INV_FOUR_PI / distance)
    end
    isfinite(value) && return value
    return _greens_smooth_exact(r, rp, k)
end

@inline function _grad_greens_unchecked(
    r::Vec3, rp::Vec3, k::_GreenFloatWavenumber)
    dx, dy, dz, distance = _green_float_geometry(r, rp)
    if iszero(distance)
        return CVec3(0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im)
    end

    q = ComplexF64(imag(k), -real(k))
    phase_argument = q * distance
    if _green_float_requires_fallback(
        r, rp, dx, dy, dz, distance, k, phase_argument)
        return _grad_greens_exact(r, rp, k)
    end

    radial_exponential_factor =
        _green_radial_exponential_factor(phase_argument)
    value = CVec3(
        _green_gradient_component(
            dx, distance, radial_exponential_factor),
        _green_gradient_component(
            dy, distance, radial_exponential_factor),
        _green_gradient_component(
            dz, distance, radial_exponential_factor),
    )
    all(isfinite, value) && return value
    return _grad_greens_exact(r, rp, k)
end

"""
    greens(r, rp, k)

Scalar free-space Green's function G(r,r') = exp(-ik R) / (4π R)
where R = |r - r'|.
Works with complex `k` for complex-step differentiation.
"""
function greens(r::SVector{3}, rp::SVector{3}, k)
    _validate_green_arguments(r, rp, k)
    value = _greens_unchecked(r, rp, k)
    isfinite(value) ||
        throw(OverflowError("Green function is non-finite"))
    return value
end

"""
    greens_smooth(r, rp, k)

Smooth part of the Green's function after singularity extraction:
  G_smooth(r,r') = [exp(-ikR) - 1] / (4πR)
with limit -ik/(4π) as R → 0.

Used for self-cell integration with singularity subtraction.
"""
function greens_smooth(r::SVector{3}, rp::SVector{3}, k)
    _validate_green_arguments(r, rp, k)
    value = _greens_smooth_unchecked(r, rp, k)
    isfinite(value) ||
        throw(OverflowError("smooth Green function is non-finite"))
    return value
end

"""
    grad_greens(r, rp, k)

Gradient of G with respect to r: ∇G = dG/dR * R̂ = [(-ik - 1/R) G] * R̂
where R̂ = (r - r') / |r - r'|.
Returns a 3-vector (SVector{3}).
"""
function grad_greens(r::SVector{3}, rp::SVector{3}, k)
    _validate_green_arguments(r, rp, k)
    value = _grad_greens_unchecked(r, rp, k)
    all(isfinite, value) ||
        throw(OverflowError("Green-function gradient is non-finite"))
    return value
end
