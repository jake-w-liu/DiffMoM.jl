# PlanarImmittance.jl — Transmission-line cascade for the layered Green kernel
#
# For each box mode (kx, ky) and each polarization (TE to z / TM to z) the
# z-direction collapses to a transmission-line problem: every layer is a
# section with propagation constant gamma_l = sqrt(kc^2 - k_l^2) (Re >= 0 by
# the principal root, convention e^{+i w t}) and characteristic impedance
#   Zc_TE = i*omega*mu_l / gamma_l,   Zc_TM = gamma_l / (i*omega*eps_l).
# Interface i (i = 0..L) is the top face of layer i.  Looking-down impedances
# Zdn[i] terminate in `stack.bottom`, looking-up Zup[i] in `stack.top`.
# A unit modal sheet current on source interface s sees the parallel
# combination V_s = Zdn[s] || Zup[s], and the modal voltage at field
# interface f follows from the cascade transfer factors.  Everything uses
# e1 = exp(-gamma*d) and e2 = exp(-2*gamma*d) forms (|e2| <= 1 for
# Re gamma >= 0) so deeply evanescent modes cannot overflow and tanh poles
# of lossless propagating modes never appear as Inf/Inf.
#
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
end

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
    zchar, e1, e2 = scratch
    length(zdn) == L + 1 && length(zup) == L + 1 &&
        length(inv_tau_up) == L && length(inv_tau_dn) == L &&
        all(v -> length(v) == L, scratch) ||
        throw(ArgumentError("cascade workspace sized for $(length(zdn)-1) " *
                            "layers, stackup has $L"))

    @inbounds for l in 1:L
        layer = stack.layers[l]
        gamma = _planar_gamma(kc2, _planar_k2(omega, layer.epsr, layer.mur))
        zchar[l] = _planar_zchar(pol, omega,
            layer.epsr * _EPS0, layer.mur * _MU0, gamma)
        gd = gamma * layer.thickness
        e1[l] = exp(-gd)
        e2[l] = e1[l] * e1[l]
    end

    zdn[1] = _planar_term_impedance(stack.bottom, pol, omega, kc2)
    @inbounds for l in 1:L
        zdn[l + 1] = _planar_input_impedance(zchar[l], e2[l], zdn[l])
    end
    zup[L + 1] = _planar_term_impedance(stack.top, pol, omega, kc2)
    # looking up from interface l-1 (bottom face of layer l):
    # layer-l section terminated by zup[l+1] (looking up at interface l)
    @inbounds for l in L:-1:1
        zup[l] = _planar_input_impedance(zchar[l], e2[l], zup[l + 1])
    end

    @inbounds for l in 1:L
        # cosh(gd) + u*sinh(gd) = ((1+u) + e2*(1-u)) / (2*e1)
        inv_tau_up[l] = _planar_inv_tau(zchar[l] / zup[l + 1],
                                        e2[l], e1[l])
        inv_tau_dn[l] = _planar_inv_tau(zchar[l] / zdn[l],
                                        e2[l], e1[l])
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
    vs = if _planarisinf(zd) && _planarisinf(zu)
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
