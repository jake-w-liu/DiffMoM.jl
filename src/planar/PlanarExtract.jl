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

export RLGC, MulticonductorRLGC, PiModel, InductorParams
export planar_line_params, planar_rlgc, planar_rlgc_sweep, planar_pi_model, planar_inductor

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

function _check_reciprocal(Y::AbstractMatrix, what::AbstractString)
    abs(Y[1, 2] - Y[2, 1]) <=
        1e-8 * max(abs(Y[1, 2]), abs(Y[2, 1]), floatmin(Float64)) ||
        throw(ArgumentError("$what requires a reciprocal two-port (Y12 != Y21)"))
    return nothing
end

"""Multiconductor per-unit-length line constants. The matrix fields
`z,y,r,l,g,c` use the same units as [`RLGC`](@ref), and `gammas` contains
the modal propagation constants [1/m]. Ports are ordered `[left; right]`
with one port per conductor at each end and a common return conductor."""
struct MulticonductorRLGC
    z::Matrix{ComplexF64}
    y::Matrix{ComplexF64}
    r::Matrix{Float64}
    l::Matrix{Float64}
    g::Matrix{Float64}
    c::Matrix{Float64}
    gammas::Vector{ComplexF64}
end

"""    planar_line_params(Y) -> NamedTuple(zc, gl)

Extract the characteristic impedance `zc` [Ohm] and complex electrical
length `gl = gamma*len` of a uniform line section from its 2x2
port-admittance matrix `Y` (e.g. a de-embedded thru).  Fails closed if
the section is not a symmetric transmission line."""
function planar_line_params(Y::AbstractMatrix; phase_hint::Union{Nothing,Real}=nothing)
    _check_square(Y, 2, "planar_line_params")
    _check_finite(Y, "planar_line_params")
    zc, gl = _line_params_of_abcd(_abcd_of_y(Y))
    if phase_hint !== nothing
        isfinite(phase_hint) || throw(ArgumentError("phase_hint must be finite"))
        gl += 2pi * 1im * round((phase_hint - imag(gl)) / (2pi))
    end
    return (zc=zc, gl=gl)
end

"""    planar_rlgc(Y, len, f) -> RLGC

Per-unit-length R, L, G, C of the uniform line section whose 2x2
port-admittance matrix is `Y`, of physical length `len` [m], evaluated
at frequency `f` [Hz].  `planar_rlgc(cal, f)` uses the line parameters
of a `DoubleDelayCal` directly."""
function planar_rlgc(Y::AbstractMatrix, len::Real, f::Real;
        phase_hint::Union{Nothing,Real}=nothing,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    (isfinite(len) && len > 0) || throw(ArgumentError(
        "line length must be finite and > 0, got $len"))
    n = size(Y, 1)
    if n > 2 && iseven(n) && size(Y, 2) == n
        phase_hint === nothing || throw(ArgumentError(
            "phase_hint is scalar and applies only to a single-conductor line"))
        return _planar_multiconductor_rlgc(Y, Float64(len), f; max_bytes=max_bytes)
    end
    zc, gl = planar_line_params(Y; phase_hint=phase_hint)
    return _rlgc_of(zc, gl / len, f)
end

function _planar_multiconductor_rlgc(Y::AbstractMatrix, len::Float64, f::Real;
        max_bytes::Integer)
    isfinite(f) && f > 0 || throw(ArgumentError("frequency must be finite and positive"))
    _check_finite(Y, "multiconductor RLGC")
    n2 = size(Y, 1)
    n = n2 ÷ 2
    _enforce_payload_limit(
        _checked_array_payload_bytes(ComplexF64, 24, n2, n2;
            label="multiconductor RLGC workspace"),
        max_bytes, "multiconductor RLGC", "max_bytes")
    Ym = Matrix{ComplexF64}(Y)
    maximum(abs, Ym - transpose(Ym)) <=
        1e-8 * max(maximum(abs, Ym), floatmin(Float64)) ||
        throw(ArgumentError("multiconductor RLGC requires a reciprocal line"))
    left, right = 1:n, (n + 1):n2
    Y11, Y12 = Ym[left,left], Ym[left,right]
    Y21, Y22 = Ym[right,left], Ym[right,right]
    F = lu(Y21; check=false)
    issuccess(F) || throw(ArgumentError("line has a singular through-admittance block"))
    A = -(F \ Y22)
    B = -(F \ Matrix{ComplexF64}(I,n,n))
    C = Y12 + Y11 * A
    D = Y11 * B
    # Normalize voltage/current units before the matrix logarithm so its
    # Schur decomposition does not mix hundreds of ohms with millisiemens.
    zref = 50.0
    M = [A B / zref; C * zref D]
    H = log(M) / len
    all(isfinite, H) || throw(ArgumentError("line matrix logarithm is non-finite"))
    scale = max(maximum(abs, H), floatmin(Float64))
    max(maximum(abs, view(H,left,left)), maximum(abs, view(H,right,right))) <=
        1e-7 * scale || throw(ArgumentError(
            "network is not a uniform multiconductor transmission-line section"))
    z = Matrix{ComplexF64}(H[left,right] * zref)
    y = Matrix{ComplexF64}(H[right,left] / zref)
    omega = 2pi * f
    l, c = imag.(z) / omega, imag.(y) / omega
    # Principal matrix-log extraction is unambiguous below half a
    # wavelength in every mode. Reject phase aliases yielding negative
    # stored energy instead of returning a plausible negative inductance.
    for (value, name) in ((l,"inductance"),(c,"capacitance"))
        eigs = eigvals(LinearAlgebra.Symmetric((value + transpose(value)) / 2))
        minimum(eigs) >= -1e-8 * max(maximum(abs,eigs),floatmin(Float64)) ||
            throw(ArgumentError("extracted $name is not positive semidefinite; " *
                "use a shorter section (every mode must be below half a wavelength)"))
    end
    gammas = ComplexF64[sqrt(complex(v)) for v in eigvals(z * y)]
    return MulticonductorRLGC(z,y,real.(z),l,real.(y),c,gammas)
end

"""`planar_rlgc_sweep(Ys, len, freqs; phase_hint=nothing)` extracts a
single-conductor line over strictly increasing positive frequencies and
continues its electrical-length branch. The first point must be shorter
than half a wavelength or have an explicit `phase_hint` [rad]. Subsequent
phase changes must be smaller than π relative to the previous point's
frequency-scaled phase estimate. This resolves the 2π ambiguity of long
lines without independently aliasing each frequency."""
function planar_rlgc_sweep(Ys::AbstractVector{<:AbstractMatrix}, len::Real,
        freqs::AbstractVector{<:Real}; phase_hint::Union{Nothing,Real}=nothing)
    length(Ys) == length(freqs) && !isempty(Ys) || throw(DimensionMismatch(
        "RLGC sweep requires matching nonempty matrices and frequencies"))
    isfinite(len) && len > 0 || throw(ArgumentError("line length must be finite and positive"))
    all(f -> isfinite(f) && f > 0, freqs) && issorted(freqs) &&
        all(diff(freqs) .> 0) || throw(ArgumentError(
            "RLGC frequencies must be finite, positive and strictly increasing"))
    result = Vector{RLGC}(undef,length(Ys))
    previous_phase = phase_hint
    for k in eachindex(Ys)
        hint = k == 1 ? previous_phase : previous_phase * freqs[k] / freqs[k-1]
        zc, gl = planar_line_params(Ys[k]; phase_hint=hint)
        result[k] = _rlgc_of(zc, gl / len, freqs[k])
        previous_phase = imag(gl)
    end
    return result
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
    _check_reciprocal(Y, "planar_pi_model")
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
    n == 2 && _check_reciprocal(Y, "planar_inductor")
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
