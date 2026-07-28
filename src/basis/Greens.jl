# Greens.jl — Free-space scalar Green's function and derivatives
#
# Convention: exp(+iωt), so G(r,r') = exp(-ik|r-r'|) / (4π|r-r'|)

export greens, greens_smooth, grad_greens

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
    R_vec = r - rp
    R = sqrt(dot(R_vec, R_vec))
    isfinite(R) ||
        throw(OverflowError(
            "Green-function point separation is non-finite"))
    return nothing
end

@inline function _greens_unchecked(r::SVector{3}, rp::SVector{3}, k)
    R_vec = r - rp
    R = sqrt(dot(R_vec, R_vec))
    if abs(R) < 1e-30
        return zero(complex(typeof(real(k))))
    end
    return exp(-im * k * R) / (4π * R)
end

@inline function _greens_smooth_unchecked(r::SVector{3}, rp::SVector{3}, k)
    R_vec = r - rp
    R = sqrt(dot(R_vec, R_vec))
    if abs(R) < 1e-14
        return -im * k / (4π)
    end
    return expm1(-im * k * R) / (4π * R)
end

@inline function _grad_greens_unchecked(r::SVector{3}, rp::SVector{3}, k)
    R_vec = r - rp
    R = sqrt(dot(R_vec, R_vec))
    if abs(R) < 1e-30
        value = zero(complex(typeof(real(k))))
        return SVector{3}(value, value, value)
    end
    G = exp(-im * k * R) / (4π * R)
    dGdR = (-im * k - 1 / R) * G
    return dGdR * (R_vec / R)
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
