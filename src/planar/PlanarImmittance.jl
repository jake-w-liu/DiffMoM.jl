# PlanarImmittance.jl — Transmission-line cascade for the layered Green kernel
#
# For each box mode (kx, ky) and each polarization (TE to z / TM to z) the
# z-direction collapses to a transmission-line problem: every layer is a
# section with propagation constant gamma_l = sqrt(kc^2 - k_l^2) (Re >= 0 by
# the principal root, convention e^{+i w t}) and characteristic impedance
#   Zc_TE = i*omega*mu_l / gamma_l,   Zc_TM = gamma_l / (i*omega*eps_l).
# A layer may be uniaxial with optic axis z (epsr_z, mur_z distinct from
# the transverse epsr, mur); then each polarization sees its own decay
# constant,
#   gamma_TE^2 = (mu_t/mu_z)*kc^2 - k_t^2,
#   gamma_TM^2 = (eps_t/eps_z)*kc^2 - k_t^2,   k_t^2 = w^2 mu_t eps_t,
# while the characteristic impedances keep the transverse constants above
# (TE sees mu_t, TM sees eps_t).
# Interface i (i = 0..L) is the top face of layer i.  Looking-down impedances
# Zdn[i] terminate in `stack.bottom`, looking-up Zup[i] in `stack.top`.
# A unit modal sheet current on source interface s sees the parallel
# combination V_s = Zdn[s] || Zup[s], and the modal voltage at field
# interface f follows from the cascade transfer factors.  Everything uses
# e1 = exp(-gamma*d) and e2 = exp(-2*gamma*d) forms (|e2| <= 1 for
# Re gamma >= 0) so deeply evanescent modes cannot overflow and tanh poles
# of lossless propagating modes never appear as Inf/Inf.
#
# Near axial cutoff, scaled ABCD coefficients use an even-power series
# in gamma^2 instead of separating singular characteristic impedances.
# All arithmetic runs in the scalar type T carried by the stackup, so
# complex-step / dual-number perturbations of layer parameters propagate
# straight through.

export PlanarPol, TE_POL, TM_POL, PlanarCascade
export planar_mode_cascade, planar_modal_voltage

# a non-finite impedance (Inf or NaN from a 0/0 complex divide) is an
# open-circuit / decoupled load in the cascade
@inline _planarisinf(z::Number) = !isfinite(real(z)) || !isfinite(imag(z))

"""Modal polarization tag: `TE_POL` / `TM_POL` (TE-to-z / TM-to-z)."""
@enum PlanarPol::UInt8 begin
    TE_POL
    TM_POL
end
@doc "TE-to-z polarization (transverse E, modal impedance i*w*mu/gamma)." TE_POL
@doc "TM-to-z polarization (transverse H, modal impedance gamma/(i*w*eps))." TM_POL

"""Per-mode transmission-line state of a `PlanarStackup`: looking-down and
looking-up impedances at every interface plus the reciprocal transfer
factors across each layer, all in the scalar type `T`."""
struct PlanarCascade{T<:Number}
    # index i+1 holds interface i = 0..L
    zdn::Vector{T}
    zup::Vector{T}
    # reciprocal transfer factors across layer l (index l):
    #   inv_tau_up[l] = V(bottom of l)/V(top of l) = cosh(gd) + u*sinh(gd)
    #   inv_tau_dn[l] = V(top of l)/V(bottom of l)
    # so modal voltages accumulate *divisions* by these factors.
    inv_tau_up::Vector{T}
    inv_tau_dn::Vector{T}
    inv_current_up::Vector{T}
    inv_current_dn::Vector{T}
end

# Preserve the original four-vector workspace constructor.  Current
# transfers are filled alongside voltage transfers by the cascade.
PlanarCascade(zdn::Vector{T}, zup::Vector{T}, up::Vector{T}, dn::Vector{T}) where {T<:Number} =
    PlanarCascade(zdn, zup, up, dn, Vector{T}(undef, length(up)),
        Vector{T}(undef, length(dn)))

@inline function _planar_zchar(pol::PlanarPol, omega::Number,
        eps::T, mu::T, gamma::T) where {T<:Number}
    return pol === TE_POL ? (1im * omega * mu) / gamma :
                            gamma / (1im * omega * eps)
end

@inline function _planar_gamma(kc2::Float64, k2::T) where {T<:Number}
    return sqrt(ComplexF64(kc2) - k2)
end

@inline _planar_k2(omega::Number, epsr::Number, mur::Number) =
    planar_k2_layer(omega, epsr, mur)

# Uniaxial layer (optic axis z): the axial decay constant differs per
# polarization —
#   gamma_TE^2 = (mur/mur_z)*kc2 - k2_t  (ordinary wave)
#   gamma_TM^2 = (epsr/epsr_z)*kc2 - k2_t (extraordinary wave)
# with k2_t = (omega/c0)^2 * epsr * mur the transverse product, which also
# enters Zc_TE = i*w*mu_t/gamma and Zc_TM = gamma/(i*w*eps_t).  Isotropic
# (epsr_z = epsr, mur_z = mur) recovers sqrt(kc2 - k2).
@inline function _planar_gamma_layer(pol::PlanarPol, kc2::Float64,
        omega::Number, layer::PlanarLayer{T}) where {T<:Number}
    return sqrt(_planar_gamma2_layer(pol, kc2, omega, layer))
end

@inline function _planar_gamma2_layer(pol::PlanarPol, kc2::Float64,
        omega::Number, layer::PlanarLayer{T}) where {T<:Number}
    k2 = planar_k2_layer(omega, layer.epsr, layer.mur)
    alpha = pol === TE_POL ? layer.mur / layer.mur_z :
                             layer.epsr / layer.epsr_z
    return alpha * ComplexF64(kc2) - k2
end

# Scaled ABCD coefficients of one layer.  Forming Zc*sinh(gamma*h)
# and sinh(gamma*h)/Zc separately loses the finite TE series impedance
# / TM shunt admittance at gamma == 0.  The even-power small-argument
# series is analytic in gamma^2, including parameter derivatives there.
@inline function _planar_layer_abcd(pol::PlanarPol, omega::Number,
        kc2::Float64, layer::PlanarLayer)
    g2 = _planar_gamma2_layer(pol, kc2, omega, layer)
    h = layer.thickness
    q = g2 * h * h
    if abs2(q) <= one(abs2(q))
        # The same exact cosh/sinh even series used by conductor transfer.
        # Coefficients and termination retain the working scalar precision.
        a, series = _planar_conductor_even_series(q)
        sh = h * series
        scale = one(a)
    else
        x = sqrt(g2) * h
        e1 = exp(-x)
        e2 = e1 * e1
        a = 1 + e2
        sh = h * (1 - e2) / x
        scale = 2 * e1
    end
    ze = 1im * omega * layer.mur * _MU0
    ym = 1im * omega * layer.epsr * _EPS0
    b, c = pol === TE_POL ? (ze * sh, g2 * sh / ze) :
                            (g2 * sh / ym, ym * sh)
    return a, b, c, scale
end

@inline function _planar_input_abcd(a::T, b::T, c::T,
        zl::T) where {T<:Number}
    if _planarisinf(zl)
        return iszero(c) ? T(Inf) : a / c
    end
    den = a + c * zl
    return iszero(den) ? T(Inf) : (a * zl + b) / den
end

@inline function _planar_inv_abcd(a::T, b::T, scale::T,
        zl::T) where {T<:Number}
    iszero(scale) && return T(Inf)
    _planarisinf(zl) && return a / scale
    iszero(zl) && return _planar_inv_zero_load_limit(a, b, scale, zl)
    return (a + b / zl) / scale
end

@inline _planar_inv_zero_load_limit(a::T, b::T, scale::T,
        zl::T) where {T<:Number} = iszero(b) ? a / scale : T(Inf)

@inline function _planar_inv_current_abcd(a::T, c::T, scale::T,
        zl::T) where {T<:Number}
    iszero(scale) && return T(Inf)
    _planarisinf(zl) && return iszero(c) ? a / scale : T(Inf)
    return (a + c * zl) / scale
end

"""Terminator impedance seen by a mode with transverse cutoff kc2."""
@inline function _planar_term_impedance(term::PlanarTerminator{T},
        pol::PlanarPol, omega::Number, kc2::Float64) where {T<:Number}
    if term.kind === TERM_PEC
        return zero(T)
    elseif term.kind === TERM_PMC
        return T(Inf)
    elseif term.kind === TERM_SURFACE
        return term.zs
    else # TERM_OPEN: characteristic impedance of the exterior half-space
        gamma0 = _planar_gamma(kc2, _planar_k2(omega, term.epsr, term.mur))
        return _planar_zchar(pol, omega,
            term.epsr * _EPS0, term.mur * _MU0, gamma0)
    end
end

"""
Input impedance of a layer section terminated by zl, in exp(-2*gamma*d)
form:  Zin = Zc*(zl*(1+e2) + Zc*(1-e2)) / (Zc*(1+e2) + zl*(1-e2)).
A zero denominator is a physical anti-resonance and returns Inf.
"""
@inline function _planar_input_impedance(zc::T, e2::T,
        zl::T) where {T<:Number}
    # e2 = 1 (gamma*d = 0, mode at cutoff, or a half/full-wave layer): the
    # section is transparent and the formula is 0/0 with a degenerate Zc
    isone(e2) && return _planar_e2_one_limit(zc, e2, zl)
    if _planarisinf(zl)
        num = one(T) + e2
        den = one(T) - e2
        iszero(den) && return T(Inf)
        return zc * num / den
    end
    num = zc * (zl * (one(T) + e2) + zc * (one(T) - e2))
    den = zc * (one(T) + e2) + zl * (one(T) - e2)
    iszero(den) && return T(Inf)
    return num / den
end

# value-level limit at e2 = 1 (transparent section); PlanarAdjoint.jl adds
# the _PlanarDual method carrying d(Zin) = dzl - (Zc - zl^2/Zc)*e2'/2
@inline _planar_e2_one_limit(zc::T, e2::T, zl::T) where {T<:Number} = zl

"""
    planar_mode_cascade(stack, omega, kc2, pol) -> PlanarCascade

Per-mode transmission-line state for transverse cutoff `kc2 = kx^2 + ky^2`
[m^-2] at angular frequency `omega` and polarization `pol`.
"""
function planar_mode_cascade(stack::PlanarStackup{T}, omega::Number,
        kc2::Float64, pol::PlanarPol) where {T<:Number}
    planar_validate(stack)
    isfinite(omega) && real(omega) > 0 ||
        throw(ArgumentError("omega must be finite with Re > 0"))
    isfinite(kc2) && kc2 >= 0 ||
        throw(ArgumentError("kc2 must be finite and nonnegative"))
    L = length(stack.layers)
    # working scalar carries omega's perturbation type too (complex-step)
    S = promote_type(T, typeof(complex(omega)))
    ws = PlanarCascade(Vector{S}(undef, L + 1), Vector{S}(undef, L + 1),
        Vector{S}(undef, L), Vector{S}(undef, L))
    scratch = (Vector{S}(undef, L), Vector{S}(undef, L),
               Vector{S}(undef, L))
    return planar_mode_cascade!(ws, stack, omega, kc2, pol, scratch)
end

# fill the four cascade vectors in place; scratch space owned by the
# caller comes in through `scratch` (3 x Vector{S} of length L).
function planar_mode_cascade!(ws::PlanarCascade{S},
        stack::PlanarStackup{T}, omega::Number, kc2::Float64,
        pol::PlanarPol, scratch::NTuple{3,Vector{S}}) where {T<:Number,S<:Number}
    L = length(stack.layers)
    zdn, zup, inv_tau_up, inv_tau_dn =
        ws.zdn, ws.zup, ws.inv_tau_up, ws.inv_tau_dn
    amat, bmat, cmat = scratch
    length(zdn) == L + 1 && length(zup) == L + 1 &&
        length(inv_tau_up) == L && length(inv_tau_dn) == L &&
        length(ws.inv_current_up) == L && length(ws.inv_current_dn) == L &&
        all(v -> length(v) == L, scratch) ||
        throw(ArgumentError("cascade workspace sized for $(length(zdn)-1) " *
                            "layers, stackup has $L"))

    @inbounds for l in 1:L
        layer = stack.layers[l]
        a, b, c, _ = _planar_layer_abcd(pol, omega, kc2, layer)
        amat[l], bmat[l], cmat[l] = a, b, c
    end

    zdn[1] = _planar_term_impedance(stack.bottom, pol, omega, kc2)
    @inbounds for l in 1:L
        zdn[l + 1] = _planar_input_abcd(amat[l], bmat[l], cmat[l], zdn[l])
    end
    zup[L + 1] = _planar_term_impedance(stack.top, pol, omega, kc2)
    # looking up from interface l-1 (bottom face of layer l):
    # layer-l section terminated by zup[l+1] (looking up at interface l)
    @inbounds for l in L:-1:1
        zup[l] = _planar_input_abcd(amat[l], bmat[l], cmat[l], zup[l + 1])
    end

    @inbounds for l in 1:L
        _, _, _, scale = _planar_layer_abcd(pol, omega, kc2, stack.layers[l])
        inv_tau_up[l] = _planar_inv_abcd(amat[l], bmat[l], scale, zup[l + 1])
        inv_tau_dn[l] = _planar_inv_abcd(amat[l], bmat[l], scale, zdn[l])
        ws.inv_current_up[l] = _planar_inv_current_abcd(amat[l], cmat[l], scale, zup[l+1])
        ws.inv_current_dn[l] = _planar_inv_current_abcd(amat[l], cmat[l], scale, zdn[l])
    end
    return ws
end

# reciprocal transfer factor: ((1+u) + e2*(1-u)) / (2*e1).
# u = Inf (short-circuit load at the far face) -> factor Inf (no transfer).
# e1 = 0 (gamma*d underflow, deeply evanescent) -> the layer decouples
# completely; return Inf rather than finite/0 = NaN in complex division.
@inline function _planar_inv_tau(u::T, e2::T, e1::T) where {T<:Number}
    # e1 = 1 (gamma*d = 0 or a full-wave layer): no z-drop across the
    # layer even though u is degenerate (Zc is 0 or Inf at cutoff)
    isone(e1) && return _planar_e1_one_limit(u, e1)
    (_planarisinf(u) || iszero(e1)) && return T(Inf)
    return ((one(T) + u) + e2 * (one(T) - u)) / (2 * e1)
end

# value-level limit at e1 = 1; PlanarAdjoint.jl adds the _PlanarDual
# method carrying d(inv_tau) = -u * d(e1)/dtheta (from d/dx = u at x=0)
@inline _planar_e1_one_limit(u::T, e1::T) where {T<:Number} = one(T)

"""
    planar_modal_voltage(cascade, f, s) -> T

Modal voltage at field interface `f` produced by a unit modal sheet current
on source interface `s` (interfaces 0..L).  Infinite results are physical
box resonances and propagate to the caller.
"""
function planar_modal_voltage(c::PlanarCascade{T}, f::Integer,
        s::Integer) where {T<:Number}
    L = length(c.inv_tau_up)
    (0 <= f <= L && 0 <= s <= L) ||
        throw(ArgumentError("interfaces must be in 0:$L, got f=$f s=$s"))
    zd, zu = c.zdn[s + 1], c.zup[s + 1]
    vs = if iszero(zd) && iszero(zu)
        _planar_zero_parallel_limit(zd, zu)
    elseif _planarisinf(zd) && _planarisinf(zu)
        T(Inf)
    elseif _planarisinf(zd)
        zu
    elseif _planarisinf(zu)
        zd
    else
        den = zd + zu
        iszero(den) ? T(Inf) : zd * zu / den
    end
    f == s && return vs
    v = vs
    if f > s
        @inbounds for l in (s + 1):f
            v /= c.inv_tau_up[l]
        end
    else
        @inbounds for l in (f + 1):s
            v /= c.inv_tau_dn[l]
        end
    end
    return v
end

@inline _planar_zero_parallel_limit(zd::T, zu::T) where {T<:Number} = zero(T)

# Modal line current associated with the positive sheet-voltage Green
# function.  Its source jump is H(up)-H(down)=+1.  Current transfers
# remain regular when voltage is zero at a PEC-loaded axial cutoff.
function _planar_modal_current(c::PlanarCascade{T}, f::Int, s::Int,
        side::Symbol) where {T<:Number}
    L = length(c.inv_tau_up)
    0 <= f <= L && 0 <= s <= L || throw(ArgumentError("invalid current interface"))
    zd, zu = c.zdn[s+1], c.zup[s+1]
    dinf, uinf = _planarisinf(zd), _planarisinf(zu)
    hup, hdn = if dinf && uinf
        (T(Inf), T(Inf))
    elseif dinf
        (one(T), zero(T))
    elseif uinf
        (zero(T), -one(T))
    else
        den = zd + zu
        iszero(den) ? (T(Inf), T(Inf)) : (zd / den, -zu / den)
    end
    if f > s
        v = hup
        for l in s+1:f
            v /= c.inv_current_up[l]
        end
        return v
    elseif f < s
        v = hdn
        for l in f+1:s
            v /= c.inv_current_dn[l]
        end
        return v
    end
    return side === :up ? hup : hdn
end
