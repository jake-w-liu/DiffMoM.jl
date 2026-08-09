# Greens2D.jl — 2D scalar Green's function for Helmholtz equation
#
# Convention: exp(+iωt)
# G₂D(r,r') = (-i/4) H₀⁽²⁾(k|r-r'|)
# Satisfies: (∇² + k²) G = -δ(r-r')

export greens_2d, self_cell_integral_2d

const _SELF_CELL_SERIES_CUTOFF_2D = 0.5
const _EULER_GAMMA_2D = Float64(Base.MathConstants.eulergamma)

@inline function _scale_by_positive_square_2d(value::Float64, scale::Float64)
    iszero(value) && return value
    value_fraction, value_exponent = frexp(value)
    scale_fraction, scale_exponent = frexp(scale)
    return ldexp(
        value_fraction * scale_fraction * scale_fraction,
        value_exponent + 2 * scale_exponent,
    )
end

function _small_self_cell_integral_2d(k::Float64, a_eq::Float64, ka::Float64)
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
    value = ComplexF64(
        _scale_by_positive_square_2d(normalized_real, a_eq),
        _scale_by_positive_square_2d(normalized_imag, a_eq),
    )
    isfinite(value) ||
        throw(OverflowError(
            "self-cell integral is outside the representable ComplexF64 range."))
    return value
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
    isfinite(R) ||
        throw(ArgumentError(
            "greens_2d point separation must be finite, got $R."))
    if iszero(R)
        return zero(ComplexF64)
    end
    kR = k * R
    isfinite(kR) ||
        throw(ArgumentError(
            "greens_2d phase argument k*R must be finite, got $kR."))
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
    isfinite(ka) ||
        throw(ArgumentError(
            "self-cell size parameter k*a_eq must be finite, got $ka."))
    if ka <= _SELF_CELL_SERIES_CUTOFF_2D
        return _small_self_cell_integral_2d(k, a_eq, ka)
    end
    H1 = besselh(1, 2, ka)
    value = (-im * π / (2 * k^2)) * (ka * H1 - 2im / π)
    isfinite(value) ||
        error("self_cell_integral_2d produced a non-finite value.")
    return value
end

"""
    assemble_D_matrix(mesh::Mesh2D, k)

Assemble the Green's function integral matrix D where:
  D[m,n] = ∫_{cell_n} G₂D(r_m, r') dA'

For m ≠ n: midpoint approximation D[m,n] ≈ G₂D(r_m, r_n) × A_n
For m = n: analytical self-cell integral with equivalent circular cell.
"""
function assemble_D_matrix(mesh::Mesh2D, k::Float64)
    _validate_mesh_2d(mesh)
    _validate_positive_finite_2d(k, "assemble_D_matrix wavenumber")
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
