# Greens2D.jl — 2D scalar Green's function for Helmholtz equation
#
# Convention: exp(+iωt)
# G₂D(r,r') = (-i/4) H₀⁽²⁾(k|r-r'|)
# Satisfies: (∇² + k²) G = -δ(r-r')

export greens_2d, self_cell_integral_2d

const _SELF_CELL_SERIES_CUTOFF_2D = 0.5
const _GREENS_SERIES_CUTOFF_2D = 0.5
const _EULER_GAMMA_2D = Float64(Base.MathConstants.eulergamma)
const _SELF_CELL_FALLBACK_PRECISION_2D = 256
const _GREENS_EXACT_PHASE_THRESHOLD_2D = 128.0

@inline function _greens_phase_precision_2d(
        k::Float64, r::Vec2, rp::Vec2)
    return _source_radial_phase_precision(
        k,
        Vec3(r[1], r[2], 0.0),
        Vec3(rp[1], rp[2], 0.0),
    )
end

@noinline function _greens_2d_exact(r::Vec2, rp::Vec2, k::Float64)
    precision = _greens_phase_precision_2d(k, r, rp)
    return setprecision(BigFloat, precision) do
        dx = BigFloat(r[1]) - BigFloat(rp[1])
        dy = BigFloat(r[2]) - BigFloat(rp[2])
        distance = hypot(dx, dy)
        iszero(distance) && return zero(ComplexF64)
        argument = BigFloat(k) * distance

        value = if argument <= BigFloat(_GREENS_SERIES_CUTOFF_2D)
            argument2_over_4 = (argument / 2)^2
            term = one(BigFloat)
            j0_series = one(BigFloat)
            harmonic = zero(BigFloat)
            y0_correction = zero(BigFloat)
            for order in 1:128
                harmonic += inv(BigFloat(order))
                term *= -argument2_over_4 / BigFloat(order)^2
                j0_series += term
                y0_correction -= harmonic * term
                abs(term) <= eps(BigFloat) * abs(j0_series) && break
            end
            logarithmic_factor = log(argument) - log(BigFloat(2)) +
                                 BigFloat(Base.MathConstants.eulergamma)
            Complex{BigFloat}(
                -(logarithmic_factor * j0_series + y0_correction) /
                 (2 * BigFloat(pi)),
                -j0_series / 4,
            )
        elseif argument <= BigFloat(_GREENS_EXACT_PHASE_THRESHOLD_2D)
            converted_argument = Float64(argument)
            Complex{BigFloat}(
                (-im / 4) * besselh(0, 2, converted_argument))
        else
            # H_0^(2)(z) asymptotic expansion.  At z>128 the omitted term
            # after convergence is far below a Float64 ulp, while the phase is
            # reduced from the exact supplied coordinates and wavenumber.
            term = one(Complex{BigFloat})
            series = term
            for order in 1:24
                odd = BigFloat(2order - 1)
                term *= Complex{BigFloat}(0, -1) * (-odd^2) /
                        (BigFloat(order) * 8 * argument)
                series += term
                abs(term) <= eps(Float64) * abs(series) && break
            end
            phase = exp(Complex{BigFloat}(
                0, -(argument - BigFloat(pi) / 4)))
            (-Complex{BigFloat}(0, 1) / 4) *
            sqrt(2 / (BigFloat(pi) * argument)) * phase * series
        end

        converted = ComplexF64(value)
        isfinite(converted) ||
            throw(OverflowError(
                "greens_2d value is outside the representable ComplexF64 range."))
        return converted
    end
end

function _small_greens_2d(k::Float64, distance::Float64, phase::Float64)
    phase2_over_4 = (phase / 2)^2
    term = 1.0
    j0_series = 1.0
    harmonic = 0.0
    y0_correction = 0.0

    for order in 1:32
        harmonic += inv(Float64(order))
        term *= -phase2_over_4 / (order * order)
        j0_series += term
        y0_correction -= harmonic * term
        abs(term) <= eps(Float64) * abs(j0_series) && break
    end

    log_phase = iszero(phase) ? log(k) + log(distance) : log(phase)
    logarithmic_factor = log_phase - log(2.0) + _EULER_GAMMA_2D
    return ComplexF64(
        -(logarithmic_factor * j0_series + y0_correction) / (2π),
        -j0_series / 4,
    )
end

@inline function _scale_by_positive_square_2d(value::Float64, scale::Float64)
    iszero(value) && return value
    value_fraction, value_exponent = frexp(value)
    scale_fraction, scale_exponent = frexp(scale)
    return ldexp(
        value_fraction * scale_fraction * scale_fraction,
        value_exponent + 2 * scale_exponent,
    )
end

function _small_self_cell_normalized_2d(
        k::Float64, a_eq::Float64, ka::Float64)
    # Work with the finite ratios J₁(z)/z and
    # (zY₁(z) + 2/π)/z².  Evaluating zH₁⁽²⁾(z) - 2i/π directly loses the
    # logarithmic real part once z² approaches machine precision.
    z2_over_4 = (ka / 2)^2
    term = 1.0
    j_series = 1.0
    harmonic = 0.0
    y_series = 1.0 - 2 * _EULER_GAMMA_2D

    for order in 1:32
        harmonic += inv(Float64(order))
        term *= -z2_over_4 / (order * (order + 1))
        j_series += term
        digamma_sum = 2 * harmonic + inv(Float64(order + 1)) -
                       2 * _EULER_GAMMA_2D
        y_series += digamma_sum * term
        abs(term) <= eps(Float64) * abs(j_series) && break
    end

    j_ratio = j_series / 2
    log_ka = iszero(ka) ? log(k) + log(a_eq) : log(ka)
    y_ratio = (2 / π) * (log_ka - log(2.0)) * j_ratio -
              y_series / (2π)
    normalized_real = -(π / 2) * y_ratio
    normalized_imag = -(π / 2) * j_ratio
    return ComplexF64(normalized_real, normalized_imag)
end

function _small_self_cell_integral_2d(k::Float64, a_eq::Float64, ka::Float64)
    normalized = _small_self_cell_normalized_2d(k, a_eq, ka)
    value = ComplexF64(
        _scale_by_positive_square_2d(real(normalized), a_eq),
        _scale_by_positive_square_2d(imag(normalized), a_eq),
    )
    isfinite(value) ||
        throw(OverflowError(
            "self-cell integral is outside the representable ComplexF64 range."))
    return value
end

@noinline function _self_cell_integral_big_2d(
        k::Float64, ka::Float64, H1::ComplexF64)
    return setprecision(BigFloat, _SELF_CELL_FALLBACK_PRECISION_2D) do
        k_big = BigFloat(k)
        ka_big = BigFloat(ka)
        pi_big = BigFloat(π)
        imaginary_unit = Complex{BigFloat}(zero(BigFloat), one(BigFloat))
        value = (-imaginary_unit * pi_big / (2 * k_big^2)) *
                (ka_big * Complex{BigFloat}(H1) -
                 2 * imaginary_unit / pi_big)
        converted = ComplexF64(value)
        isfinite(converted) ||
            throw(OverflowError(
                "self-cell integral is outside the representable ComplexF64 range."))
        return converted
    end
end

@noinline function _self_cell_integral_large_exact_2d(
        k::Float64, a_eq::Float64)
    precision = _greens_phase_precision_2d(
        k, Vec2(a_eq, 0.0), Vec2(0.0, 0.0))
    return setprecision(BigFloat, precision) do
        k_big = BigFloat(k)
        argument = k_big * BigFloat(a_eq)
        term = one(Complex{BigFloat})
        series = term
        order_parameter = BigFloat(4)
        for order in 1:24
            odd = BigFloat(2order - 1)
            term *= Complex{BigFloat}(0, -1) *
                    (order_parameter - odd^2) /
                    (BigFloat(order) * 8 * argument)
            series += term
            abs(term) <= eps(Float64) * abs(series) && break
        end
        hankel = sqrt(2 / (BigFloat(pi) * argument)) *
                 exp(Complex{BigFloat}(
                     0, -(argument - 3BigFloat(pi) / 4))) * series
        value = (-Complex{BigFloat}(0, 1) * BigFloat(pi) /
                 (2 * k_big^2)) *
                (argument * hankel -
                 2Complex{BigFloat}(0, 1) / BigFloat(pi))
        converted = ComplexF64(value)
        isfinite(converted) ||
            throw(OverflowError(
                "self-cell integral is outside the representable ComplexF64 range."))
        return converted
    end
end

"""
    greens_2d(r, rp, k)

2D scalar free-space Green's function:
  G(r,r') = (-i/4) H₀⁽²⁾(k|r-r'|)

Uses exp(+iωt) convention with outgoing H₀⁽²⁾.
"""
function greens_2d(r::Vec2, rp::Vec2, k::Float64)
    _validate_finite_vec2_2d(r, "greens_2d observation point")
    _validate_finite_vec2_2d(rp, "greens_2d source point")
    _validate_positive_finite_2d(k, "greens_2d wavenumber")
    return _greens_2d_unchecked(r, rp, k)
end

@inline function _greens_2d_unchecked(r::Vec2, rp::Vec2, k::Float64)
    R_vec = r - rp
    R = hypot(R_vec[1], R_vec[2])
    isfinite(R) || return _greens_2d_exact(r, rp, k)
    if iszero(R)
        return zero(ComplexF64)
    end
    kR = k * R
    (!isfinite(kR) || kR > _GREENS_EXACT_PHASE_THRESHOLD_2D) &&
        return _greens_2d_exact(r, rp, k)
    if kR <= _GREENS_SERIES_CUTOFF_2D
        return _small_greens_2d(k, R, kR)
    end
    value = (-im / 4) * besselh(0, 2, kR)
    isfinite(value) ||
        error("greens_2d produced a non-finite Green's function value.")
    return value
end

"""
    self_cell_integral_2d(k, a_eq)

Analytical integral of G₂D over a circular cell of radius `a_eq`:

  ∫_{|r'|≤a_eq} G₂D(0, r') dA' = (-iπ/(2k²)) [k a_eq H₁⁽²⁾(k a_eq) - 2i/π]

Derived from: d/du[u H₁⁽²⁾(u)] = u H₀⁽²⁾(u).
"""
function self_cell_integral_2d(k::Float64, a_eq::Float64)
    _validate_positive_finite_2d(k, "self-cell wavenumber")
    _validate_positive_finite_2d(a_eq, "self-cell equivalent radius")
    ka = k * a_eq
    (!isfinite(ka) || ka > _GREENS_EXACT_PHASE_THRESHOLD_2D) &&
        return _self_cell_integral_large_exact_2d(k, a_eq)
    if ka <= _SELF_CELL_SERIES_CUTOFF_2D
        return _small_self_cell_integral_2d(k, a_eq, ka)
    end
    H1 = besselh(1, 2, ka)
    isfinite(H1) ||
        error("self_cell_integral_2d produced a non-finite Hankel value.")
    k_squared = k^2
    if isfinite(k_squared) && !iszero(k_squared)
        value = (-im * π / (2 * k_squared)) * (ka * H1 - 2im / π)
        if isfinite(value) &&
           max(abs(real(value)), abs(imag(value))) >= floatmin(Float64)
            return value
        end
    end
    return _self_cell_integral_big_2d(k, ka, H1)
end

"""
    assemble_D_matrix(mesh::Mesh2D, k;
                      max_output_bytes=2_000_000_000)

Assemble the Green's function integral matrix D where:
  D[m,n] = ∫_{cell_n} G₂D(r_m, r') dA'

For m ≠ n: midpoint approximation D[m,n] ≈ G₂D(r_m, r_n) × A_n
For m = n: analytical self-cell integral with equivalent circular cell.
"""
function assemble_D_matrix(
        mesh::Mesh2D, k::Float64;
        max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    _validate_mesh_2d(mesh)
    _validate_positive_finite_2d(k, "assemble_D_matrix wavenumber")
    payload_bytes = _checked_array_payload_bytes(
        ComplexF64, mesh.ncells, mesh.ncells;
        label="assemble_D_matrix output")
    _enforce_payload_limit(
        payload_bytes, max_output_bytes,
        "assemble_D_matrix output", "max_output_bytes")
    return _assemble_D_matrix_unchecked(mesh, k)
end

function _assemble_D_matrix_unchecked(mesh::Mesh2D, k::Float64)
    N = mesh.ncells
    A = mesh.cell_area
    a_eq = _equivalent_radius_unchecked(mesh)
    D_self = self_cell_integral_2d(k, a_eq)

    D = Matrix{ComplexF64}(undef, N, N)

    @inbounds for n in 1:N
        for m in 1:N
            if m == n
                D[m, n] = D_self
            else
                D[m, n] =
                    _greens_2d_unchecked(mesh.centers[m], mesh.centers[n], k) * A
            end
        end
    end

    all(isfinite, D) ||
        error("assemble_D_matrix produced non-finite matrix entries.")
    return D
end
