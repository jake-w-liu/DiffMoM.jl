# PlanarDeembed.jl — port de-embedding for planar (gap-port) analysis
#
# A gap port in the shielded planar solver presents a port discontinuity
# that is, to leading order, a pure shunt admittance yd at the port plane
# followed by a length of the port's connecting transmission line.  The
# measured Y therefore differs from the device Y by a per-port two-port
# parasitic chain M_p (ABCD from the EM reference plane toward the
# device):
#
#     [V_A,p; I_A,p] = M_p * [V_B,p; I_B,p],   I_A = Y_A V_A, I_B = Y_B V_B
#
# With diagonal A,B,C,D built from the per-port chains,
#     Y_A = (C + D Y_B) (A + B Y_B)^{-1}
#     Y_B = (Y_A B - D)^{-1} (C - Y_A A).
#
# This file provides:
#   * planar_line_abcd(zc, gl): ABCD of a line section (gl = gamma*len).
#   * deembed_ports(Y, chains): remove per-port ABCD chains (length-0
#     chains skipped; also usable for port extension with any known line).
#   * DoubleDelayCal / deembed_double_delay_calibrate(Y_l, Y_2l):
#     Rautio's double-delay extraction — T_l * T_2l^{-1} * T_l isolates
#     the doubled shunt discontinuity (self-diagnostic on its form),
#     then the connecting-line ABCD gives the electrical length and the
#     TEM-equivalent Zc.
#   * deembed_double_delay_apply(Y, cal): per-port removal of Md*L(ell)
#     (reference plane moves the standard length into the DUT) or of Md
#     alone (plane stays at the port).
#
# All Y matrices use the solver's port convention (port current = current
# into the network); ABCD uses I_B positive INTO the device, which makes
# the diagonal-matrix derivation above exact for uncoupled port chains.

export DoubleDelayCal
export planar_line_abcd, deembed_ports, deembed_port_extension
export deembed_double_delay_calibrate, deembed_double_delay_apply

# ---------------- 2x2 ABCD <-> Y ----------------

"""ABCD (cascade) matrix of a transmission-line section: `zc` complex
characteristic impedance [Ohm], `gl = gamma * len` complex electrical
length.  e^{+i wt} convention."""
function planar_line_abcd(zc::Number, gl::Number)
    c, s = cosh(ComplexF64(gl)), sinh(ComplexF64(gl))
    z = ComplexF64(zc)
    return ComplexF64[c z * s; s / z c]
end

# Y -> ABCD for a reciprocal 2-port (currents into the network)
function _abcd_of_y(Y::AbstractMatrix)
    size(Y) == (2, 2) || throw(DimensionMismatch(
        "2-port conversion needs a 2x2 matrix, got $(size(Y))"))
    y21 = Y[2, 1]
    abs2(y21) > eps(real(abs(y21)))^2 || throw(ArgumentError(
        "Y21 is zero: no through path, cannot form ABCD parameters"))
    a = -Y[2, 2] / y21
    b = -inv(y21)
    c = -(Y[1, 1] * Y[2, 2] - Y[1, 2] * y21) / y21
    d = -Y[1, 1] / y21
    return ComplexF64[a b; c d]
end

# ABCD -> Y for a reciprocal 2-port
function _y_of_abcd(T::AbstractMatrix)
    b = T[1, 2]
    abs2(b) > eps(real(abs(b)))^2 || throw(ArgumentError(
        "ABCD B term is zero: cannot form Y parameters"))
    return ComplexF64[T[2, 2] / b (T[2, 1] * b - T[1, 1] * T[2, 2]) / b;
        -inv(b) T[1, 1] / b]
end

# ---------------- per-port ABCD-chain de-embedding ----------------

"""    deembed_ports(Y, chains) -> Matrix{ComplexF64}

Remove per-port two-port parasitic chains from the port-admittance matrix
`Y`.  `chains[p]` is the ABCD matrix mapping the device plane to the EM
reference plane at port `p` (e.g. `planar_line_abcd(zc, gl)` for a feed
line, or a shunt `[1 0; yd 1]`, or their cascade).  A chain equal to the
identity leaves that port untouched.  With `chains[p]` mapping
(V_B, I_B) -> (V_A, I_A), the result satisfies
`Y_B = (Y_A*B - D)^{-1} (C - Y_A*A)` where A,B,C,D are the diagonal
per-port ABCD entries.  `deembed_port_extension(Y, zc, gl)` is the
special case removing the same line section `planar_line_abcd(zc, gl)`
at every port.
"""
function deembed_ports(Y::AbstractMatrix,
        chains::AbstractVector{<:AbstractMatrix})
    n = size(Y, 1)
    (size(Y, 2) == n && length(chains) == n) || throw(DimensionMismatch(
        "Y must be square with one ABCD chain per port: " *
        "size(Y)=$(size(Y)), chains=$(length(chains))"))
    all(isfinite, Y) || throw(ArgumentError(
        "Y contains non-finite entries"))
    A = zeros(ComplexF64, n)
    B = zeros(ComplexF64, n)
    C = zeros(ComplexF64, n)
    D = zeros(ComplexF64, n)
    @inbounds for p in 1:n
        m = chains[p]
        size(m) == (2, 2) || throw(DimensionMismatch(
            "chain[$p] must be 2x2 ABCD, got $(size(m))"))
        all(isfinite, m) || throw(ArgumentError(
            "chain[$p] contains non-finite entries"))
        A[p], B[p], C[p], D[p] = m[1, 1], m[1, 2], m[2, 1], m[2, 2]
    end
    YA = ComplexF64.(Y)
    W = YA .* transpose(B) - Diagonal(D)
    iszero(W) && throw(ArgumentError(
        "de-embedding solve is singular (Y_A*B - D = 0)"))
    f = lu(W)
    issuccess(f) || throw(ArgumentError(
        "de-embedding solve is singular: check the per-port chains"))
    return ldiv!(f, ComplexF64.(Diagonal(C)) - YA .* transpose(A))
end

"""    deembed_port_extension(Y, zc, gl) -> Matrix{ComplexF64}

Shift every port's reference plane by removing a connecting-line section
of characteristic impedance `zc` and electrical length `gl = gamma*len`
(`gl` may be complex for loss and negative to add length).  Equivalent to
`deembed_ports(Y, fill(planar_line_abcd(zc, gl), n))`.
"""
deembed_port_extension(Y::AbstractMatrix, zc::Number, gl::Number) =
    deembed_ports(Y, fill(planar_line_abcd(zc, gl), size(Y, 1)))

# ---------------- double-delay calibration ----------------

"""Result of `deembed_double_delay_calibrate`: the naked port
discontinuity `yd` (shunt admittance at the EM port plane), the
connecting-line propagation `gamma_l` and TEM-equivalent `zc`, the
standard length `len`, and a `residual` measuring how well the extracted
double discontinuity matches the pure-shunt form (zero = perfect)."""
struct DoubleDelayCal
    yd::ComplexF64
    zc::ComplexF64
    gamma_l::ComplexF64
    len::Float64
    residual::Float64
end

"""    deembed_double_delay_calibrate(Y_l, Y_2l; len, tol=0.25) -> DoubleDelayCal

Extract the naked port discontinuity and the connecting-line parameters
from two simulated thru standards of length `len` and `2*len` sharing the
DUT's port geometry (Rautio double-delay).  `P = T_l T_2l^{-1} T_l` must
be the doubled pure-shunt discontinuity `[1 0; 2*yd 1]`; `tol` is the
allowed relative deviation of P's diagonal/zero entries (fail-closed
self-diagnostic)."""
function deembed_double_delay_calibrate(Yl::AbstractMatrix,
        Y2l::AbstractMatrix; len::Real, tol::Real=0.25)
    (isfinite(len) && len > 0) || throw(ArgumentError(
        "standard length must be finite and > 0, got $len"))
    (isfinite(tol) && tol > 0) || throw(ArgumentError(
        "tol must be finite and > 0, got $tol"))
    Tl = _abcd_of_y(Yl)
    T2 = _abcd_of_y(Y2l)
    all(isfinite, Tl) && all(isfinite, T2) || throw(ArgumentError(
        "calibration standard ABCD is non-finite (degenerate Y21)"))
    f2 = lu(T2)
    issuccess(f2) || throw(ArgumentError(
        "2*len standard ABCD is singular"))
    P = Tl * (f2 \ Tl)
    all(isfinite, P) || throw(ArgumentError(
        "double-delay extraction produced non-finite values"))
    nrm = max(abs(P[2, 1]), 1.0)
    resid = max(abs(P[1, 1] - 1), abs(P[2, 2] - 1), abs(P[1, 2])) / nrm
    resid <= tol || throw(ArgumentError(
        "double-delay standards fail the pure-shunt check " *
        "(rel residual $(resid) > tol $tol): the port discontinuity is " *
        "not a shunt admittance or the standards are inconsistent"))
    yd = P[2, 1] / 2
    Md = ComplexF64[1 0; yd 1]
    Mi = ComplexF64[1 0; -yd 1]
    Tline = (Mi * Tl) * Mi
    all(isfinite, Tline) || throw(ArgumentError(
        "de-embedded line ABCD is non-finite"))
    c = (Tline[1, 1] + Tline[2, 2]) / 2
    s2 = Tline[1, 2] * Tline[2, 1]
    abs2(s2) > eps(real(abs(s2)))^2 || throw(ArgumentError(
        "de-embedded line has zero B*C: no propagating mode"))
    s = sqrt(s2)
    zc = Tline[1, 2] / s
    # (s, Zc) and (-s, -Zc) both satisfy B = Zc*s: resolve to Re(Zc) > 0
    real(zc) < 0 && (s = -s; zc = -zc)
    (isfinite(zc) && isfinite(yd)) || throw(ArgumentError(
        "extracted port parameters are non-finite"))
    # consistency: the two sinh products must agree
    abs(Tline[2, 1] - s / zc) <=
        0.05 * max(abs(s / zc), abs(Tline[2, 1])) ||
        throw(ArgumentError(
            "de-embedded line ABCD is not a symmetric line section " *
            "(B/Zc != C*Zc within 5%)"))
    # acosh has a +/- branch; pick the one whose sinh matches the
    # extracted line so planar_line_abcd(zc, gamma_l) rebuilds it
    gl = acosh(c)
    abs(sinh(-gl) - s) < abs(sinh(gl) - s) && (gl = -gl)
    abs(sinh(gl) - s) <= 0.05 * max(abs(s), 1.0) || throw(ArgumentError(
        "reconstructed line does not match the de-embedded section"))
    return DoubleDelayCal(yd, zc, gl, Float64(len), Float64(resid))
end

"""    deembed_double_delay_apply(Y, cal; line=true) -> Matrix{ComplexF64}

Remove the calibrated port parasitic chain from `Y`.  With `line=true`
the chain is `Md * L(len)` per port — the reference plane moves `cal.len`
into the DUT; with `line=false` only the naked discontinuity is removed.
"""
function deembed_double_delay_apply(Y::AbstractMatrix, cal::DoubleDelayCal;
        line::Bool=true)
    Md = ComplexF64[1 0; cal.yd 1]
    chain = line ? Md * planar_line_abcd(cal.zc, cal.gamma_l) : Md
    return deembed_ports(Y, fill(chain, size(Y, 1)))
end
