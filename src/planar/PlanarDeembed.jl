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

# Keep the normal scalar contraction allocation-free. Exceptional products
# may overflow before cancellation even though the result is representable.
function _deembed_product_sum(a::ComplexF64,b::ComplexF64,c::ComplexF64)
    result=c+a*b
    isfinite(result) && return result
    bits=_planar_terminal_product_precision(2)
    re,im,left,right,product=ntuple(_->BigFloat(0.;precision=bits),5)
    _planar_wide_set!(re,real(c));_planar_wide_set!(im,imag(c))
    for (x,y,target,subtract) in ((real(a),real(b),re,false),
            (imag(a),imag(b),re,true),(real(a),imag(b),im,false),
            (imag(a),real(b),im,false))
        _planar_wide_set!(left,x);_planar_wide_set!(right,y)
        _planar_wide_mul!(product,left,right)
        subtract ? _planar_wide_sub!(target,target,product) : _planar_wide_add!(target,target,product)
    end
    return complex(Float64(re),Float64(im))
end

"""ABCD (cascade) matrix of a transmission-line section: `zc` complex
characteristic impedance [Ohm], `gl = gamma * len` complex electrical
length.  e^{+i wt} convention."""
function planar_line_abcd(zc::Number, gl::Number)
    isfinite(zc) && !iszero(zc) && isfinite(gl) || throw(ArgumentError(
        "line impedance must be finite and nonzero, and electrical length finite"))
    c, s = cosh(ComplexF64(gl)), sinh(ComplexF64(gl))
    z = ComplexF64(zc)
    result = ComplexF64[c z * s; s / z c]
    all(isfinite, result) || throw(ArgumentError(
        "line ABCD overflows at this electrical length"))
    return result
end

# Y -> ABCD for a reciprocal 2-port (currents into the network)
function _abcd_of_y(Y::AbstractMatrix)
    size(Y) == (2, 2) || throw(DimensionMismatch(
        "2-port conversion needs a 2x2 matrix, got $(size(Y))"))
    y11, y12 = _planar_stored_phasor(Y[1, 1]), _planar_stored_phasor(Y[1, 2])
    y21, y22 = _planar_stored_phasor(Y[2, 1]), _planar_stored_phasor(Y[2, 2])
    !iszero(y21) || throw(ArgumentError(
        "Y21 is zero: no through path, cannot form ABCD parameters"))
    a = -y22 / y21
    b = -inv(y21)
    # Divide before multiplying: the determinant products can underflow
    # or overflow even when each transfer coefficient is representable.
    c = _deembed_product_sum(y11,a,y12)
    d = -y11 / y21
    result = ComplexF64[a b; c d]
    all(isfinite, result) || throw(ArgumentError("Y conversion exceeds finite ABCD range"))
    return result
end

# ABCD -> Y for a reciprocal 2-port
function _y_of_abcd(T::AbstractMatrix)
    size(T) == (2, 2) || throw(DimensionMismatch(
        "ABCD conversion needs a 2x2 matrix, got $(size(T))"))
    a, b = _planar_stored_phasor(T[1, 1]), _planar_stored_phasor(T[1, 2])
    c, d = _planar_stored_phasor(T[2, 1]), _planar_stored_phasor(T[2, 2])
    !iszero(b) || throw(ArgumentError(
        "ABCD B term is zero: cannot form Y parameters"))
    a_over_b = a / b
    result = ComplexF64[d / b _deembed_product_sum(-d,a_over_b,c);
        -inv(b) a_over_b]
    all(isfinite, result) || throw(ArgumentError("ABCD conversion exceeds finite Y range"))
    return result
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
    f = lu(W; check=false)
    issuccess(f) || throw(ArgumentError(
        "de-embedding solve is singular: check the per-port chains"))
    result = ldiv!(f, ComplexF64.(Diagonal(C)) - YA .* transpose(A))
    all(isfinite, result) || throw(ArgumentError(
        "de-embedding produced non-finite admittances"))
    return result
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
    len = _circuit_stored_real(len, "calibration standard length")
    (isfinite(tol) && tol > 0) || throw(ArgumentError(
        "tol must be finite and > 0, got $tol"))
    Tl = _abcd_of_y(Yl)
    T2 = _abcd_of_y(Y2l)
    all(isfinite, Tl) && all(isfinite, T2) || throw(ArgumentError(
        "calibration standard ABCD is non-finite (degenerate Y21)"))
    f2 = lu(T2; check=false)
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
    zc, gl = _line_params_of_abcd(Tline)
    isfinite(yd) || throw(ArgumentError(
        "extracted port discontinuity is non-finite"))
    return DoubleDelayCal(yd, zc, gl, Float64(len), Float64(resid))
end

# Recover (zc, gamma*len) from a symmetric line-section ABCD:
# A = D = cosh(gl), B = zc*sinh(gl), C = sinh(gl)/zc.  The (s, zc) ->
# (-s, -zc) ambiguity in B = zc*s is resolved to Re(zc) > 0, and the
# inverse-hyperbolic branch is resolved so that planar_line_abcd(zc, gl)
# reconstructs the section (fail-closed on non-line sections).
function _line_params_of_abcd(Tline::AbstractMatrix)
    size(Tline) == (2, 2) || throw(DimensionMismatch(
        "line extraction needs a 2x2 ABCD matrix"))
    all(isfinite, Tline) || throw(ArgumentError("line ABCD contains non-finite entries"))
    # A reciprocal uniform section has identical end impedances and unit
    # determinant. Checking only B/C cannot establish either invariant.
    abs(Tline[1, 1] - Tline[2, 2]) <=
        0.05 * max(abs(Tline[1, 1]), abs(Tline[2, 2]), 1.0) ||
        throw(ArgumentError("ABCD is not a symmetric line section (A != D)"))
    determinant = Tline[1, 1] * Tline[2, 2] - Tline[1, 2] * Tline[2, 1]
    abs(determinant - 1) <= 0.05 || throw(ArgumentError(
        "ABCD is not a reciprocal line section (determinant != 1)"))
    c = Tline[1, 1]/2 + Tline[2, 2]/2
    !iszero(Tline[1, 2]) && !iszero(Tline[2, 1]) || throw(ArgumentError(
        "line section has zero B*C: no propagating mode"))
    # B*C can underflow even when sinh(gl) and both stored off-diagonal
    # entries are representable. Separate square roots retain that scale.
    s = sqrt(ComplexF64(Tline[1, 2])) * sqrt(ComplexF64(Tline[2, 1]))
    zc = Tline[1, 2] / s
    real(zc) < 0 && (s = -s; zc = -zc)
    isfinite(zc) || throw(ArgumentError(
        "extracted characteristic impedance is non-finite"))
    # consistency: the two sinh products must agree
    abs(Tline[2, 1] - s / zc) <=
        0.05 * max(abs(s / zc), abs(Tline[2, 1])) ||
        throw(ArgumentError(
            "ABCD is not a symmetric line section " *
            "(B/Zc != C*Zc within 5%)"))
    # Short sections retain their first-order length in B/C after cosh(gl)
    # has rounded to one. Invert sinh, then let cosh select the half-plane
    # branch; both candidates have the retained sinh and differ in cosh's
    # sign. The phase remains on the principal [-pi,pi] branch.
    gl = asinh(s)
    reconstructed_c=cosh(gl)
    if abs(reconstructed_c-c)>abs(reconstructed_c+c)
        gl = complex(0.,imag(s)<0 ? -pi : pi)-gl
        reconstructed_c=-reconstructed_c
    end
    abs(reconstructed_c-c) <= 0.05*max(abs(c),1.0) || throw(ArgumentError(
        "reconstructed line does not match the given section"))
    abs(sinh(gl) - s) <= 0.05 * max(abs(s), 1.0) || throw(ArgumentError(
        "reconstructed line does not match the given section"))
    return zc, gl
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
