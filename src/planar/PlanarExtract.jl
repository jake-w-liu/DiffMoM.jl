# PlanarExtract.jl — circuit-element extraction from planar port data
#
# Helpers that turn the solver's port-admittance matrices into circuit
# quantities: per-unit-length RLGC for a uniform line section, the
# pi-equivalent of a 2-port, and inductor series-branch parameters.
# Line constants reuse the canonical `_line_params_of_abcd` from
# PlanarDeembed.jl:
#
#     gamma = gl/len,  z = gamma*zc = R + i w L,  y = gamma/zc = G + i w C
#
# All Y matrices use the solver's port convention (port current = current
# into the network); e^{+i wt} time convention; frequencies in Hz.

export RLGC, PiModel, InductorParams
export planar_line_params, planar_rlgc, planar_pi_model, planar_inductor

"""Per-unit-length line constants extracted from a uniform line section:
`z` series impedance [Ohm/m] and `y` shunt admittance [S/m] with
`r = Re(z)`, `l = Im(z)/omega` [H/m], `g = Re(y)`, `c = Im(y)/omega`
[F/m]."""
struct RLGC
    z::ComplexF64
    y::ComplexF64
    r::Float64
    l::Float64
    g::Float64
    c::Float64
end

"""Pi-equivalent admittances of a reciprocal 2-port: `ys` series branch,
`ya`/`yb` shunt branches at port 1/2 (branch impedances are `1 ./ y`)."""
struct PiModel
    ys::ComplexF64
    ya::ComplexF64
    yb::ComplexF64
end

"""Series-branch parameters of an inductor: `z` branch impedance [Ohm],
`r = Re(z)` [Ohm], `l = Im(z)/omega` [H], `q = Im(z)/Re(z)` (negative `q`
flags a nonphysical negative-resistance branch)."""
struct InductorParams
    z::ComplexF64
    r::Float64
    l::Float64
    q::Float64
end

_check_square(Y::AbstractMatrix, n::Int, what::AbstractString) =
    size(Y) == (n, n) ||
    throw(DimensionMismatch("$what needs a $(n)x$(n) matrix, got $(size(Y))"))

_check_finite(Y::AbstractMatrix, what::AbstractString) =
    all(isfinite, Y) ||
    throw(ArgumentError("$what contains non-finite entries"))

"""    planar_line_params(Y) -> NamedTuple(zc, gl)

Extract the characteristic impedance `zc` [Ohm] and complex electrical
length `gl = gamma*len` of a uniform line section from its 2x2
port-admittance matrix `Y` (e.g. a de-embedded thru).  Fails closed if
the section is not a symmetric transmission line."""
function planar_line_params(Y::AbstractMatrix)
    _check_square(Y, 2, "planar_line_params")
    _check_finite(Y, "planar_line_params")
    zc, gl = _line_params_of_abcd(_abcd_of_y(Y))
    return (zc=zc, gl=gl)
end

"""    planar_rlgc(Y, len, f) -> RLGC

Per-unit-length R, L, G, C of the uniform line section whose 2x2
port-admittance matrix is `Y`, of physical length `len` [m], evaluated
at frequency `f` [Hz].  `planar_rlgc(cal, f)` uses the line parameters
of a `DoubleDelayCal` directly."""
function planar_rlgc(Y::AbstractMatrix, len::Real, f::Real)
    (isfinite(len) && len > 0) || throw(ArgumentError(
        "line length must be finite and > 0, got $len"))
    zc, gl = planar_line_params(Y)
    return _rlgc_of(zc, gl / len, f)
end

planar_rlgc(cal::DoubleDelayCal, f::Real) =
    _rlgc_of(cal.zc, cal.gamma_l / cal.len, f)

function _rlgc_of(zc::Number, gam::Number, f::Real)
    (isfinite(f) && f > 0) || throw(ArgumentError(
        "frequency must be finite and > 0, got $f"))
    omega = 2pi * f
    z = ComplexF64(gam) * zc          # R + i w L  [Ohm/m]
    y = ComplexF64(gam) / zc          # G + i w C  [S/m]
    all(isfinite, (z, y)) || throw(ArgumentError(
        "extracted line constants are non-finite"))
    return RLGC(z, y, real(z), imag(z) / omega, real(y), imag(y) / omega)
end

"""    planar_pi_model(Y) -> PiModel

Pi-equivalent of a reciprocal 2-port from its port-admittance matrix:
series branch `ys = -Y12`, shunts `ya = Y11 + Y12`, `yb = Y22 + Y12`
(branch impedances are `1 ./ y`)."""
function planar_pi_model(Y::AbstractMatrix)
    _check_square(Y, 2, "planar_pi_model")
    _check_finite(Y, "planar_pi_model")
    ys = -Y[2, 1]
    return PiModel(ys, Y[1, 1] + Y[2, 1], Y[2, 2] + Y[2, 1])
end

"""    planar_inductor(Y, f) -> InductorParams

Inductor series-branch parameters at frequency `f` [Hz].  For a 1x1
`Y` the branch is the port impedance `1/Y`; for a 2x2 `Y` it is the
pi-model series branch `-1/Y12`."""
function planar_inductor(Y::AbstractMatrix, f::Real)
    n = size(Y, 1)
    (n == 1 || n == 2) && size(Y, 2) == n || throw(DimensionMismatch(
        "planar_inductor needs a 1x1 or 2x2 matrix, got $(size(Y))"))
    _check_finite(Y, "planar_inductor")
    (isfinite(f) && f > 0) || throw(ArgumentError(
        "frequency must be finite and > 0, got $f"))
    y = n == 1 ? Y[1, 1] : -Y[2, 1]
    abs2(y) > eps(real(abs(y)))^2 || throw(ArgumentError(
        "branch admittance is zero: no current path to extract"))
    z = inv(y)
    isfinite(z) || throw(ArgumentError(
        "extracted branch impedance is non-finite"))
    return InductorParams(z, real(z), imag(z) / (2pi * f),
        iszero(real(z)) ? sign(imag(z)) * Inf : imag(z) / real(z))
end
