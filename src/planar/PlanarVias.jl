# PlanarVias.jl — z-directed via-column reaction kernels on the modal cascade
#
# A via basis is a z-directed volume current on one cell footprint,
# J_z(x,y,z) = jz_c * w(u) * g3(x,y)/N3, spanning layer j from interface
# j-1 (bottom face) to interface j (top face).  The axial profile w(u)
# is constant (uniform) or the up-taper u/h (0 at the bottom face, max at
# the top); a down-taper is the linear combination uniform - taper.
#
# Vias couple only to TM modes (TE has no E_z).  Two analytic identities
# reduce every reaction integral to cascade endpoint quantities:
#
# 1) Field side — for any TM line voltage V(u) that is smooth through
#    layer j,
#      M_w[V] = int_0^h w(u) V'(u) du
#             = a_t*V(j) + a_b*V(j-1) + k*wbar*(V(j-1) + V(j))
#    where wbar = tanh(gh/2)/(gh) is the mean of the smooth layer profile
#    relative to the endpoint average and (a_t, a_b, k) = (1,-1,0) for the
#    uniform profile, (1,0,-1) for the up-taper.  The z-directed modal
#    field is E_z = (N3/N_TM)*beta*V' with beta = 1 + k_t^2/gamma^2
#    (= kc^2*eps_t/(gamma^2*eps_z) for a uniaxial layer), so a via's
#    reaction to a sheet source telescopes to its face voltages.
#
# 2) Source side — a via injects the series voltage density
#      s(u) = i*N_TM*jz(u)/(omega*eps_z*N3)
#    in the modal TM line of its layer.  Solving the line against the
#    layer's zdn/zup loads gives closed-form face voltages per unit
#    transform Tb:
#      Vhat_t =  i*N_TM*zup*Jd[w]/(omega*eps_z*N3^2*Dz)
#      Vhat_b = -i*N_TM*zdn*Ju[w]/(omega*eps_z*N3^2*Dz)
#      <Vhat> =  i*N_TM*Zc*(Ju[w]-Jd[w])/(omega*eps_z*N3^2*Dz*gh)
#    with Dz = Zc*(zdn+zup)*cosh(gh) + (Zc^2+zdn*zup)*sinh(gh) and the
#    analytic moments
#      Jd[w] = int w(u) [Zc cosh(g u) + zdn sinh(g u)] du
#      Ju[w] = int w(u) [Zc cosh(g(h-u)) + zup sinh(g(h-u))] du.
#    Inside its own layer the field via additionally sees the filament
#    contact i*(1-beta)*jz/(omega*eps_z), giving the kernel term
#    i*(1-beta)*int(wa*wb)du/(omega*eps_z*N3^2).
#
# All moments are evaluated in 2*e1 = 2*exp(-gh) scaled form so deeply
# evanescent modes stay finite, with x = gh -> 0 series branches where
# the closed forms degenerate.  Non-finite endpoint loads (open-circuit
# faces) take their Inf limits coefficientwise.  All arithmetic is
# scalar-generic so dual perturbations flow through the adjoint path.
#
# Reference: IEEE TMTT 3026376 (2020), shielded volume-current MoM; the
# endpoint telescoping used here is equivalent to the paper's reaction
# integrals (eqs. 54-67) expressed through interface quantities.

# ---------------- element axis ----------------
#
# The level-pair machinery keys on an integer element id per basis:
#   elem >= 0 : sheet basis on interface `elem`
#   elem <  0 : via element; t = -elem encodes (layer, kind) with
#               layer = (t+1)>>1 and kind = uniform iff isodd(t).
#               PlanarVolumes.jl adds a second negative-id family for
#               transverse volume elements below -_VOL_ELEM_SHIFT.

@inline _is_via_elem(e::Int) = e < 0 && e >= -_VOL_ELEM_SHIFT
@inline _via_elem(layer::Int, kind::UInt8) =
    kind == _BASIS_VIA_U ? 1 - 2 * layer : -2 * layer
@inline _via_elem_kind(e::Int) =
    isodd(-e) ? _BASIS_VIA_U : _BASIS_VIA_T
@inline _via_elem_layer(e::Int) = (1 - e) >> 1


# M_w[V] = a_t V(top) + a_b V(bottom) + k * <V>: uniform (1,-1,0),
# up-taper (1,0,-1)
@inline _via_mcoeffs(kind::UInt8) = kind == _BASIS_VIA_U ?
    (1.0, -1.0, 0.0) : (1.0, 0.0, -1.0)

# int_0^h w_a(u) w_b(u) du: UxU = h, UxT = h/2, TxT = h/3
@inline function _via_overlap(ka::UInt8, kb::UInt8, h)
    return h * (ka == _BASIS_VIA_U && kb == _BASIS_VIA_U ? 1.0 :
                ka == _BASIS_VIA_T && kb == _BASIS_VIA_T ? 1 / 3 : 0.5)
end

# ---------------- 2e1-scaled axial moment coefficients ----------------
#
# With x = g*h, e1 = exp(-x), e2 = e1^2 the J moments factor as
#   Jd[w] = h * (Zc*cA[w] + zdn*cB[w])
#   Ju[w] = h * (Zc*cA[wrev] + zup*cB[wrev])   (u <-> h-u reversal)
# where profile codes are 1 = uniform, 2 = up-taper, 3 = down-taper and
# the cA/cB below are already 2e1-scaled.  Each closed form has an
# x -> 0 series branch (analytic continuation through 2*e1*(poly in x^2)).

@inline function _via_ca(x::T, e1::T, e2::T, prof::Int) where {T<:Number}
    x2 = x * x
    if abs2(x2) < 1e-8
        s = prof == 1 ? (1 + x2 / 6 + x2 * x2 / 120) :
            prof == 2 ? (1 / 2 + x2 / 8 + x2 * x2 / 180) :
                        (1 / 2 + x2 / 24 + x2 * x2 / 720)
        return 2 * e1 * s
    elseif prof == 1
        return (1 - e2) / x
    elseif prof == 2
        return (1 - e2) / x - (1 + e2 - 2 * e1) / x2
    else
        return (1 + e2 - 2 * e1) / x2
    end
end

@inline function _via_cb(x::T, e1::T, e2::T, prof::Int) where {T<:Number}
    x2 = x * x
    if abs2(x2) < 1e-8
        s = prof == 1 ? (1 / 2 + x2 / 24) :
            prof == 2 ? (1 / 3 + x2 / 30) : (1 / 6 + x2 / 120)
        return 2 * e1 * x * s
    elseif prof == 1
        return (1 + e2 - 2 * e1) / x
    elseif prof == 2
        return ((x - 1) + e2 * (x + 1)) / x2
    else
        return ((1 - e2) - 2 * x * e1) / x2
    end
end

# (Ju[T]-Jd[T])/(h*x) coefficient forms in the same 2e1 scaling — the
# mean-voltage moment for the up-tapered source profile:
#   (Ju-Jd)/x = h*(Zc*q1 + zup*q2 - zdn*q3)
@inline function _via_q1(x::T, e1::T, e2::T) where {T<:Number}
    x2 = x * x
    abs2(x2) < 1e-8 && return -2 * e1 * x * (1 / 12 + x2 / 180)
    return (2 * (1 + e2 - 2 * e1) - x * (1 - e2)) / (x2 * x)
end
@inline function _via_q2(x::T, e1::T, e2::T) where {T<:Number}
    x2 = x * x
    abs2(x2) < 1e-8 && return 2 * e1 * (1 / 6 + x2 / 120)
    return ((1 - e2) - 2 * x * e1) / (x2 * x)
end
@inline function _via_q3(x::T, e1::T, e2::T) where {T<:Number}
    x2 = x * x
    abs2(x2) < 1e-8 && return 2 * e1 * (1 / 3 + x2 / 30)
    return ((x - 1) + e2 * (x + 1)) / (x2 * x)
end

# cB_U/x for the uniform-profile mean-voltage moment
@inline function _via_cbx_u(x::T, e1::T, e2::T) where {T<:Number}
    x2 = x * x
    abs2(x2) < 1e-8 && return 2 * e1 * (1 / 2 + x2 / 24)
    return (1 + e2 - 2 * e1) / x2
end

# tanh(x/2)/x — mean of the smooth layer profile relative to the endpoint
# average; 1/2 at x = 0
@inline function _via_wbar(x::T, e1::T) where {T<:Number}
    x2 = x * x
    abs2(x2) < 1e-8 && return 1 / 2 - x2 / 24 + x2 * x2 / 240
    return (1 - e1) / ((1 + e1) * x)
end

# ---------------- per-mode via-layer state ----------------

"""TM cascade quantities for one via-bearing layer at one box mode:
`coef` = beta = 1 + k_t^2/gamma^2 multiplies int w V' on the field side,
`wbar` is the smooth-profile mean weight, `om_eps_z` = omega*eps_z of the
layer, `h` its thickness, and `phi_t/phi_b/phi_m` the normalized via
drive moments per kind (unit suffix _u uniform, _t up-taper) such that
the via's modal face voltages per unit transform are
Vhat_t = i*N_TM*phi_t/(om_eps_z*N3^2), Vhat_b = -i*N_TM*phi_b/(...) and
mean <Vhat> = i*N_TM*phi_m/(...)."""
struct _ViaLayerState{T<:Number}
    coef::T
    wbar::T
    om_eps_z::T
    h::T
    phit_u::T
    phib_u::T
    phim_u::T
    phit_t::T
    phib_t::T
    phim_t::T
end

"""Fill the via state for layer `j` (1..L) at mode cutoff `kc2`: reuses
the cascade's endpoint loads zdn[j], zup[j+1] and recomputes the layer
axial constants in the same scalar type (dual-safe)."""
function _via_layer_state(stack::PlanarStackup, casc::PlanarCascade{T},
        omega::Number, kc2::Float64, j::Int) where {T<:Number}
    layer = stack.layers[j]
    h = layer.thickness
    gamma = _planar_gamma_layer(TM_POL, kc2, omega, layer)
    zc = _planar_zchar(TM_POL, omega,
        layer.epsr * _EPS0, layer.mur * _MU0, gamma)
    x = gamma * h
    e1 = exp(-x)
    e2 = e1 * e1
    zdn = casc.zdn[j]
    zup = casc.zup[j + 1]
    iszero(gamma) && throw(DomainError(gamma,
        "via kernel singular: TM box mode at exact axial cutoff " *
        "(gamma == 0); detune the frequency or add loss"))
    phit_u, phib_u, phim_u = _via_drives(zc, zdn, zup, x, e1, e2, h, 1)
    phit_t, phib_t, phim_t = _via_drives(zc, zdn, zup, x, e1, e2, h, 2)
    k2t = planar_k2_layer(omega, layer.epsr, layer.mur)
    return _ViaLayerState(1 + k2t / (gamma * gamma), _via_wbar(x, e1),
        omega * layer.epsr_z * _EPS0, h,
        phit_u, phib_u, phim_u, phit_t, phib_t, phim_t)
end

# per-kind normalized drive moments (prof: 1 = uniform, 2 = up-taper).
# dz is the 2e1-scaled Dz; non-finite loads drop to the Inf limit.
@inline function _via_drives(zc::T, zdn::T, zup::T, x::T, e1::T,
        e2::T, h, prof::Int) where {T}
    rev = prof == 2 ? 3 : prof
    ca_d, cb_d = _via_ca(x, e1, e2, prof), _via_cb(x, e1, e2, prof)
    ca_u, cb_u = _via_ca(x, e1, e2, rev), _via_cb(x, e1, e2, rev)
    dinf, uinf = _planarisinf(zdn), _planarisinf(zup)
    if dinf && uinf
        om = 1 - e2
        return (h * cb_d / om, h * cb_u / om, zero(T))
    elseif dinf
        dz = zc * (1 + e2) + zup * (1 - e2)
        phit = zup * h * cb_d / dz
        phib = h * (zc * ca_u + zup * cb_u) / dz
        phim = prof == 1 ? -zc * h * _via_cbx_u(x, e1, e2) / dz :
                           -zc * h * _via_q3(x, e1, e2) / dz
        return phit, phib, phim
    elseif uinf
        dz = zc * (1 + e2) + zdn * (1 - e2)
        phit = h * (zc * ca_d + zdn * cb_d) / dz
        phib = zdn * h * cb_u / dz
        phim = prof == 1 ? zc * h * _via_cbx_u(x, e1, e2) / dz :
                           zc * h * _via_q2(x, e1, e2) / dz
        return phit, phib, phim
    end
    dz = zc * (zdn + zup) * (1 + e2) + (zc * zc + zdn * zup) * (1 - e2)
    jd = h * (zc * ca_d + zdn * cb_d)
    ju = h * (zc * ca_u + zup * cb_u)
    phit = zup * jd / dz
    phib = zdn * ju / dz
    phim = prof == 1 ?
        zc * h * (zup - zdn) * _via_cbx_u(x, e1, e2) / dz :
        zc * h * (zc * _via_q1(x, e1, e2) + zup * _via_q2(x, e1, e2) -
            zdn * _via_q3(x, e1, e2)) / dz
    return phit, phib, phim
end

# ---------------- pair kernels ----------------
#
# vtm[c, pi] values: the modal kernel multiplying Wf*Ws.  Via weights are
# the raw transforms Ta = fx*fy; sheet weights keep their kx/ky factors.

# voltage at interface f propagated from interface s carrying value v
@inline function _via_prop_up(c::PlanarCascade{T}, v::T, s::Int,
        f::Int) where {T}
    f == s && return v
    @inbounds for l in (s + 1):f
        v /= c.inv_tau_up[l]
    end
    return v
end
@inline function _via_prop_dn(c::PlanarCascade{T}, v::T, s::Int,
        f::Int) where {T}
    f == s && return v
    @inbounds for l in (f + 1):s
        v /= c.inv_tau_dn[l]
    end
    return v
end

# M_wa[V] of a *smooth* (sheet-driven or propagated) voltage on layer ja
@inline function _via_moment_smooth(vt::T, vb::T, ka::UInt8,
        st::_ViaLayerState) where {T}
    at, ab, k = _via_mcoeffs(ka)
    return at * vt + ab * vb + k * st.wbar * (vt + vb)
end

# kernel for via field element (ja,ka) <- sheet source on interface s
@inline function _via_sheet_kern(ja::Int, ka::UInt8, s::Int,
        casc::PlanarCascade{T}, vst::_ViaLayerState{T},
        ntm2::Float64) where {T}
    vt = planar_modal_voltage(casc, ja, s)
    vb = planar_modal_voltage(casc, ja - 1, s)
    return vst.coef * _via_moment_smooth(vt, vb, ka, vst) / ntm2
end

# kernel for via field (ja,ka) <- via source (jb,kb)
@inline function _via_via_kern(ja::Int, ka::UInt8, jb::Int, kb::UInt8,
        casc::PlanarCascade{T}, sta::_ViaLayerState{T},
        stb::_ViaLayerState{T}, n3_2::Float64) where {T}
    at, ab, k = _via_mcoeffs(ka)
    if kb == _BASIS_VIA_U
        phit, phib, phim = stb.phit_u, stb.phib_u, stb.phim_u
    else
        phit, phib, phim = stb.phit_t, stb.phib_t, stb.phim_t
    end
    if ja == jb
        # source face voltages carry i*N_TM/(om_eps_z*N3^2); the bottom
        # face voltage is -i*N_TM*phib/(...), folded into -ab*phib
        core = at * phit - ab * phib + k * phim
        iab = _via_overlap(ka, kb, stb.h)
        return 1im * (sta.coef * core +
            (1 - sta.coef) * iab) / (stb.om_eps_z * n3_2)
    end
    # propagate the source's signed face drive to the field layer faces:
    # Vhat top = +i N phi_t/(om_eps_z*N3^2), bottom = -i N phi_b/(...);
    # the common i/(om_eps_z*N3^2) factor is applied at the end so the
    # bottom drive enters propagation as -phib
    if ja > jb
        vt = _via_prop_up(casc, phit, jb, ja)
        vb = _via_prop_up(casc, phit, jb, ja - 1)
    else
        vt = _via_prop_dn(casc, -phib, jb - 1, ja)
        vb = _via_prop_dn(casc, -phib, jb - 1, ja - 1)
    end
    m = _via_moment_smooth(vt, vb, ka, sta)
    return 1im * sta.coef * m / (stb.om_eps_z * n3_2)
end

