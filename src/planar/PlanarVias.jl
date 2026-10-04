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
#   elem <  0 : three interleaved ids per layer: uniform via, tapered
#               via, and transverse volume.  This avoids a fixed layer
#               count at which via ids collide with volume ids.

@inline _is_via_elem(e::Int) = e < 0 && rem(e, 3) != 0
@inline _via_elem(layer::Int, kind::UInt8) =
    kind == _BASIS_VIA_U ? 2 - 3 * layer : 1 - 3 * layer
@inline _via_elem_kind(e::Int) =
    rem(-e, 3) == 1 ? _BASIS_VIA_U : _BASIS_VIA_T
@inline _via_elem_layer(e::Int) = (2 - e) ÷ 3


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
            prof == 2 ? (1 / 2 + x2 / 8 + x2 * x2 / 144) :
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
    amp_h::T          # beta*V' = i*kc^2/(omega*eps_z)*H off source
    yline::T          # i*omega*eps_t
    ht_u::T
    hb_u::T
    ht_t::T
    hb_t::T
    hm_uu::T          # int H dz, uniform source
    hm_tu::T          # int (z/h)*H dz, uniform source
    hm_ut::T          # int H dz, tapered source
    hm_tt::T          # int (z/h)*H dz, tapered source
    gamma2_h2::T      # even expansion coordinate for cutoff moments
end

"""Fill the via state for layer `j` (1..L) at mode cutoff `kc2`: reuses
the cascade's endpoint loads zdn[j], zup[j+1] and recomputes the layer
axial constants in the same scalar type (dual-safe)."""
function _via_layer_state(stack::PlanarStackup, casc::PlanarCascade{T},
        omega::Number, kc2::Float64, j::Int) where {T<:Number}
    layer = stack.layers[j]
    h = convert(T, layer.thickness)
    g2 = convert(T, _planar_gamma2_layer(TM_POL, kc2, omega, layer))
    q = g2 * h * h
    om_eps_z = convert(T, omega * layer.epsr_z * _EPS0)
    yh = convert(T, 1im * omega * layer.epsr * _EPS0)
    zdn = casc.zdn[j]
    zup = casc.zup[j + 1]
    if abs2(q) < 1e-8
        # The finite H-based reaction avoids both beta's 1/gamma^2
        # singularity and cancellation against the filament contact.
        # coef==0 tags this analytic near-cutoff representation.
        u = _via_current_drives(g2, yh, zdn, zup, h, 1)
        t = _via_current_drives(g2, yh, zdn, zup, h, 2)
        return _ViaLayerState(zero(T), 1 / 2 - q / 24 + q*q / 240,
            om_eps_z, h, u[1], u[2], u[3], t[1], t[2], t[3],
            1im * kc2 / om_eps_z, yh, u[4], u[5], t[4], t[5],
            u[6], u[7], t[6], t[7], q)
    end
    gamma = sqrt(g2)
    zc = _planar_zchar(TM_POL, omega,
        layer.epsr * _EPS0, layer.mur * _MU0, gamma)
    x = gamma * h
    e1 = exp(-x)
    e2 = e1 * e1
    phit_u, phib_u, phim_u = _via_drives(zc, zdn, zup, x, e1, e2, h, 1)
    phit_t, phib_t, phim_t = _via_drives(zc, zdn, zup, x, e1, e2, h, 2)
    ht_u, hb_u = _via_drive_currents(zc, zdn, zup, x, e1, e2, h, 1)
    ht_t, hb_t = _via_drive_currents(zc, zdn, zup, x, e1, e2, h, 2)
    k2t = planar_k2_layer(omega, layer.epsr, layer.mur)
    return _ViaLayerState(1 + k2t / (gamma * gamma), _via_wbar(x, e1),
        om_eps_z, h,
        phit_u, phib_u, phim_u, phit_t, phib_t, phim_t,
        1im * kc2 / om_eps_z, yh, ht_u, hb_u, ht_t, hb_t,
        zero(T), zero(T), zero(T), zero(T), q)
end

# Driven line with V' = -g^2/Y*H + w and H' = -Y*V.
# Even-power moments make values and derivatives regular at g==0.
function _via_current_drives(g2::T, Y::T, zd::T, zu::T, h::T,
        profile::Int) where {T<:Number}
    q = g2*h*h
    a = 1 + q/2 + q*q/24 + q*q*q/720
    su = 1 + q/6 + q*q/120 + q*q*q/5040
    sc = 1/2 + q/24 + q*q/720 + q*q*q/40320
    st = 1/6 + q/120 + q*q/5040
    b, c = g2*h*su/Y, Y*h*su
    p = h * (profile == 1 ? su : sc)
    Q = h*h * (profile == 1 ? sc : st)
    dinf, uinf = _planarisinf(zd), _planarisinf(zu)
    vb, hb = if dinf && uinf
        (-Y*Q/c, zero(T))
    elseif dinf
        den = a + zu*c
        iszero(den) && throw(DomainError(den, "via modal resonance"))
        (-(p+zu*Y*Q)/den, zero(T))
    elseif uinf
        den = a + zd*c
        iszero(den) && throw(DomainError(den, "via modal resonance"))
        hv = Y*Q/den
        (-zd*hv, hv)
    else
        den = a*(zd+zu)+b+c*zd*zu
        iszero(den) && throw(DomainError(den, "via modal resonance at axial cutoff"))
        hv = (p+zu*Y*Q)/den
        (-zd*hv, hv)
    end
    vt = a*vb-b*hb+p
    ht = -c*vb+a*hb-Y*Q
    vm = vb*su-g2*hb*h*sc/Y+h*(profile==1 ? sc : st)
    hc_u = h*su
    hc_t = h*(1/2+q/8+q*q/144)
    hs_u = h*h*sc
    hs_t = h*h*(1/3+q/30+q*q/840)
    hp_u = h*h*h*(profile==1 ? (1/6+q/120+q*q/5040) :
                                             (1/24+q/720+q*q/40320))
    hp_t = h*h*h*(profile==1 ? (1/8+q/144+q*q/5760) :
                                             (1/30+q/840+q*q/45360))
    hu = hb*hc_u-Y*vb*hs_u-Y*hp_u
    htaper = hb*hc_t-Y*vb*hs_t-Y*hp_t
    return vt, -vb, vm, ht, hb, hu, htaper
end

@inline function _via_drive_currents(zc::T, zd::T, zu::T, x::T,
        e1::T, e2::T, h, profile::Int) where {T}
    reverse = profile == 2 ? 3 : profile
    cd, sd = _via_ca(x,e1,e2,profile), _via_cb(x,e1,e2,profile)
    cu, su = _via_ca(x,e1,e2,reverse), _via_cb(x,e1,e2,reverse)
    dinf, uinf = _planarisinf(zd), _planarisinf(zu)
    dinf && uinf && return zero(T), zero(T)
    if dinf
        den = zc*(1+e2)+zu*(1-e2)
        return h*sd/den, zero(T)
    elseif uinf
        den = zc*(1+e2)+zd*(1-e2)
        return zero(T), h*su/den
    end
    den = zc*(zd+zu)*(1+e2)+(zc*zc+zd*zu)*(1-e2)
    return h*(zc*cd+zd*sd)/den, h*(zc*cu+zu*su)/den
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
    if iszero(vst.coef)
        ht = _planar_modal_current(casc, ja, s, :down)
        hb = _planar_modal_current(casc, ja-1, s, :up)
        return vst.amp_h * _via_current_moment_smooth(ht,hb,ka,vst) / ntm2
    end
    vt = planar_modal_voltage(casc, ja, s)
    vb = planar_modal_voltage(casc, ja - 1, s)
    return vst.coef * _via_moment_smooth(vt, vb, ka, vst) / ntm2
end

@inline function _via_current_moment_smooth(ht::T, hb::T, kind::UInt8,
        st::_ViaLayerState{T}) where {T}
    # This method is used in the near-cutoff representation, with a
    # second-order even expansion for the tapered smooth-H moment.
    q = st.gamma2_h2
    return kind == _BASIS_VIA_U ? st.h*st.wbar*(ht+hb) :
        st.h*((1/6-7q/360+31q*q/15120)*hb+
              (1/3-q/45+2q*q/945)*ht)
end

# kernel for via field (ja,ka) <- via source (jb,kb)
# The driven TM line reconstructs the actual longitudinal electric
# field with +i/(omega*eps_z).  Unlike the sheet modal voltage, it has
# not yet been negated.  Return its negative because the common modal
# assembler subtracts every kernel.  This sign makes endpoint charges
# cancel against connected sheet-current divergence in a galvanic via.
@inline function _via_via_kern(ja::Int, ka::UInt8, jb::Int, kb::UInt8,
        casc::PlanarCascade{T}, sta::_ViaLayerState{T},
        stb::_ViaLayerState{T}, n3_2::Float64) where {T}
    if iszero(sta.coef)
        moment = if ja == jb
            kb == _BASIS_VIA_U ? (ka == _BASIS_VIA_U ? stb.hm_uu : stb.hm_tu) :
                (ka == _BASIS_VIA_U ? stb.hm_ut : stb.hm_tt)
        else
            ht_source = kb == _BASIS_VIA_U ? stb.ht_u : stb.ht_t
            hb_source = kb == _BASIS_VIA_U ? stb.hb_u : stb.hb_t
            ht, hb = if ja > jb
                (_via_current_prop_up(casc,ht_source,jb,ja),
                 _via_current_prop_up(casc,ht_source,jb,ja-1))
            else
                (_via_current_prop_dn(casc,hb_source,jb-1,ja),
                 _via_current_prop_dn(casc,hb_source,jb-1,ja-1))
            end
            _via_current_moment_smooth(ht,hb,ka,sta)
        end
        overlap = ja == jb ? _via_overlap(ka,kb,stb.h) : zero(T)
        return -1im*(sta.amp_h*moment+overlap)/(stb.om_eps_z*n3_2)
    end
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
        return -1im * (sta.coef * core +
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
    return -1im * sta.coef * m / (stb.om_eps_z * n3_2)
end

@inline function _via_current_prop_up(c::PlanarCascade{T}, v::T,
        s::Int, f::Int) where {T}
    for l in s+1:f
        v /= c.inv_current_up[l]
    end
    return v
end
@inline function _via_current_prop_dn(c::PlanarCascade{T}, v::T,
        s::Int, f::Int) where {T}
    for l in f+1:s
        v /= c.inv_current_dn[l]
    end
    return v
end
