# PlanarVolumes.jl — transverse volume-rooftop (thick metal) kernels
#
# A volume rooftop is an x- or y-directed rooftop on a cell edge
# extruded uniformly through the thickness of one stackup layer:
#   J_p(x,y,z) = T_p(x,y) * w_z(z),   w_z = 1/h on layer j
# (the Rautio-Thelen "volume rooftop" family, TMTT 3026376).  Its
# transverse transform is the sheet rooftop's; the z-profile is
# uniform so the basis carries the same total current as a sheet.
#
# Per mode the layer forms a transmission-line section with cutoff
# gamma and impedance Zc.  The axial Green's function for a transverse
# source distributed in z is the verified BVP form
#   G(z,u) = Zc * phi_d(z_<) * phi_u(z_>) / Dz
#   phi_d(u) = zdn*cosh(gu) + Zc*sinh(gu)
#   phi_u(u) = zup*cosh(g(h-u)) + Zc*sinh(g(h-u))
#   Dz = Zc*(zdn+zup)*cosh(gh) + (Zc^2+zdn*zup)*sinh(gh)
# equivalent to the paper's gT*gB/D form (eqs. 22-27, sin/cos with
# transformed load coefficients) and verified against a direct 1-D
# finite-difference BVP solve to ~9 digits.
#
# For a unit uniform source j(u) = 1/h the analytic moments are
#   V_t = Zc*zup*(zdn*sS + Zc*sC) / dz      (voltage at interface j)
#   V_b = Zc*zdn*(zup*sS + Zc*sC) / dz      (voltage at interface j-1)
#   M   = 2*Zc*Bx / dz                     (in-layer mean of V)
# with x = gh, dz the 2e1-scaled Dz, and the scaled moment forms
#   sS = sinch(x)*(2e1),  sC = ((cosh x - 1)/x)*(2e1)
#   Bx = [zdn*(zup*p2 + Zc*p4) + Zc*(zup*p1 + Zc*p2 - iu)]/x
#   p1 = e1*(cosh+sinch) = ((1+e2) + sS)/2
#   p2 = e1*sinh = x*sS/2
#   p4 = e1*(cosh-sinch) = ((1+e2) - sS)/2
#   iu = zup*sS + Zc*sC
# Each has an x -> 0 series branch; at x = 0 everything reduces to the
# parallel impedance zdn||zup (the sheet limit).  Infinite endpoint
# loads (open-circuit faces) take Inf limits coefficientwise, matching
# _via_drives.  As h -> 0, V_t = V_b = M -> zdn||zup, recovering the
# interface sheet kernel.
#
# Field-side moments: a volume element in layer a measures
# (1/h) int V dz = mean V.  For a smooth V this is wbar*(Vt+Vb) with
# wbar = tanh(x/2)/x (the _via_wbar identity); for the source's own
# layer it is M above.  Sheet elements measure V at their interface —
# so every pair kernel reduces to cascade endpoint quantities exactly
# like PlanarVias.jl.  Volume elements couple to BOTH polarizations
# (unlike vias, TM only); via<->vol pairs are TM only.
#
# ---------------- element axis ----------------
#
# elem >= 0          : sheet basis on interface `elem`
# -SHIFT < elem < 0  : via element (PlanarVias.jl encoding)
# elem <= -SHIFT-1   : volume element; layer = -elem - SHIFT
#
const _VOL_ELEM_SHIFT = 1 << 14   # via encoding occupies -1 .. -16384

@inline _is_vol_elem(e::Int) = e < -_VOL_ELEM_SHIFT
@inline _vol_elem(layer::Int) = -(_VOL_ELEM_SHIFT + layer)
@inline _vol_elem_layer(e::Int) = -e - _VOL_ELEM_SHIFT
@inline _is_inlayer_elem(e::Int) = e < 0   # via or volume element

# element id for basis p (vol kinds index `vols`, via kinds index
# `vias`, sheets index `sheets`)
@inline function _basis_elem(basis::PlanarBasisSet, p::Int,
        sheets::Vector{SheetLevel}, vias::Vector{ViaLevel},
        vols::Vector{VolLevel}=VolLevel[])
    kind = basis.kind[p]
    return _is_vol_kind(kind) ? _vol_elem(vols[basis.level[p]].layer) :
        _is_via_kind(kind) ?
        _via_elem(vias[basis.level[p]].layer, kind) :
        sheets[basis.level[p]].interface
end

# ---------------- scaled axial moments ----------------

# sS = sinch(x)*2e1 = (1-e2)/x ; sC = chh(x)*2e1 = (1+e2-2e1)/x
@inline function _vol_ss(x::T, e1::T, e2::T) where {T<:Number}
    x2 = x * x
    abs2(x2) < 1e-8 && return 2 * e1 * (1 + x2 / 6 + x2 * x2 / 120)
    return (1 - e2) / x
end
@inline function _vol_sc(x::T, e1::T, e2::T) where {T<:Number}
    x2 = x * x
    abs2(x2) < 1e-8 &&
        return 2 * e1 * x * (1 / 2 + x2 / 24 + x2 * x2 / 720)
    return (1 + e2 - 2 * e1) / x
end

# Bx = B/x, the mean-voltage bracket divided by x.  Direct form cancels
# to O(x) at small x, so |x| < ~1e-3 uses the series
#   Bx = 2e1*(B1 + x*B2 + x^2*B3 + x^3*B4 + O(x^4))
#   B1 = zdn*zup/2,  B2 = Zc*(zdn+zup)/6,
#   B3 = zdn*zup/12 + Zc^2/24,  B4 = Zc*(zdn+zup)/60
@inline function _vol_bx(zc::T, zdn::T, zup::T, x::T, e1::T,
        e2::T, sS::T, sC::T) where {T<:Number}
    x2 = x * x
    if abs2(x2) < 1e-6
        b1 = zdn * zup / 2
        b2 = zc * (zdn + zup) / 6
        b3 = zdn * zup / 12 + zc * zc / 24
        b4 = zc * (zdn + zup) / 60
        return 2 * e1 * (b1 + x * (b2 + x * (b3 + x * b4)))
    end
    p1 = ((1 + e2) + sS) / 2
    p2 = x * sS / 2
    p4 = ((1 + e2) - sS) / 2
    iu = zup * sS + zc * sC
    return (zdn * (zup * p2 + zc * p4) +
            zc * (zup * p1 + zc * p2 - iu)) / x
end

# zdn||zup with the same Inf handling as planar_modal_voltage
@inline function _vol_parallel(zdn::T, zup::T) where {T<:Number}
    dinf, uinf = _planarisinf(zdn), _planarisinf(zup)
    dinf && uinf && return T(Inf)
    dinf && return zup
    uinf && return zdn
    den = zdn + zup
    return iszero(den) ? T(Inf) : zdn * zup / den
end

# ---------------- per-mode volume-layer state ----------------

"""Cascade quantities for one volume-bearing layer at one box mode:
`wbar` = smooth-profile mean weight; `vt`/`vb` = modal voltages at the
layer's top/bottom interfaces per unit uniform-z source; `mself` = the
in-layer mean of the source's own voltage profile (field moment of a
co-layer volume basis)."""
struct _VolLayerState{T<:Number}
    wbar::T
    vt::T
    vb::T
    mself::T
end

"""Fill the volume state for layer `j` (1..L) at mode cutoff `kc2` and
polarization `pol`: reuses the cascade's endpoint loads zdn[j],
zup[j+1] and recomputes the layer axial constants in the same scalar
type (dual-safe)."""
function _vol_layer_state(stack::PlanarStackup, casc::PlanarCascade{T},
        omega::Number, kc2::Float64, j::Int, pol::PlanarPol) where {T<:Number}
    layer = stack.layers[j]
    h = layer.thickness
    gamma = _planar_gamma_layer(pol, kc2, omega, layer)
    zc = _planar_zchar(pol, omega,
        layer.epsr * _EPS0, layer.mur * _MU0, gamma)
    x = gamma * h
    e1 = exp(-x)
    e2 = e1 * e1
    zdn = casc.zdn[j]
    zup = casc.zup[j + 1]
    wbar = _via_wbar(x, e1)
    # exact axial cutoff (x == 0): the layer is transparent and all
    # endpoint quantities collapse to the parallel sheet impedance
    iszero(x) && return _VolLayerState(wbar,
        _vol_parallel(zdn, zup), _vol_parallel(zdn, zup),
        _vol_parallel(zdn, zup))
    sS = _vol_ss(x, e1, e2)
    sC = _vol_sc(x, e1, e2)
    dinf, uinf = _planarisinf(zdn), _planarisinf(zup)
    if dinf && uinf
        # dz ~ zdn*zup*(1-e2); Vt = Vb = M = Zc*sS/(1-e2) = Zc/x
        v = zc * sS / (1 - e2)
        return _VolLayerState(wbar, v, v, v)
    elseif dinf
        dz = zc * (1 + e2) + zup * (1 - e2)
        vt = zc * zup * sS / dz
        vb = zc * (zup * sS + zc * sC) / dz
        ms = 2 * zc * (zup * (sS / 2) + zc * _vol_p4x(x, e1, e2, sS)) / dz
        return _VolLayerState(wbar, vt, vb, ms)
    elseif uinf
        dz = zc * (1 + e2) + zdn * (1 - e2)
        vt = zc * (zdn * sS + zc * sC) / dz
        vb = zc * zdn * sS / dz
        ms = 2 * zc * (zdn * (sS / 2) + zc * _vol_p4x(x, e1, e2, sS)) / dz
        return _VolLayerState(wbar, vt, vb, ms)
    end
    dz = zc * (zdn + zup) * (1 + e2) + (zc * zc + zdn * zup) * (1 - e2)
    vt = zc * zup * (zdn * sS + zc * sC) / dz
    vb = zc * zdn * (zup * sS + zc * sC) / dz
    ms = 2 * zc * _vol_bx(zc, zdn, zup, x, e1, e2, sS, sC) / dz
    return _VolLayerState(wbar, vt, vb, ms)
end

# p4/x = e1*(cosh x - sinch x)/x — used by the Inf-load mself limits;
# series e1*x*(1/3 + x^2/30 + x^4/840) avoids the 0/0 at x = 0
@inline function _vol_p4x(x::T, e1::T, e2::T, sS::T) where {T<:Number}
    x2 = x * x
    abs2(x2) < 1e-8 &&
        return e1 * x * (1 / 3 + x2 / 30 + x2 * x2 / 840)
    return ((1 + e2) - sS) / (2 * x)
end

# ---------------- pair kernels ----------------
#
# All kernels return the modal factor multiplying Wf*Ws; the 1/N_pol^2
# norm is applied via `norm2` (ntm2 or nte2).  `vsts`/`volsts` are the
# per-layer TM/volume states built by the caller for this mode.

# via element (ja,ka) field <- volume element jb source (TM only).
# The via reads E_z ~ coef*V': uniform profile sees the endpoint
# difference Vt-Vb, the up-taper sees Vt - <V> in-layer; off-layer the
# voltage is smooth and the standard smooth moment applies.
@inline function _vol_via_kern(ja::Int, ka::UInt8, jb::Int,
        casc::PlanarCascade{T}, sta::_ViaLayerState{T},
        stb::_VolLayerState{T}, ntm2::Float64) where {T}
    if ja == jb
        m = ka == _BASIS_VIA_U ? stb.vt - stb.vb :
                                 stb.vt - stb.mself
        return sta.coef * m / ntm2
    end
    at, ab, k = _via_mcoeffs(ka)
    if ja > jb
        vt = _via_prop_up(casc, stb.vt, jb, ja)
        vb = _via_prop_up(casc, stb.vt, jb, ja - 1)
    else
        vt = _via_prop_dn(casc, stb.vb, jb - 1, ja)
        vb = _via_prop_dn(casc, stb.vb, jb - 1, ja - 1)
    end
    m = at * vt + ab * vb + k * sta.wbar * (vt + vb)
    return sta.coef * m / ntm2
end

# volume element ja field <- volume element jb source
@inline function _vol_vol_kern(ja::Int, jb::Int,
        casc::PlanarCascade{T}, sta::_VolLayerState{T},
        stb::_VolLayerState{T}, norm2::Float64) where {T}
    ja == jb && return sta.mself / norm2
    if ja > jb
        vt = _via_prop_up(casc, stb.vt, jb, ja)
        vb = _via_prop_up(casc, stb.vt, jb, ja - 1)
    else
        vt = _via_prop_dn(casc, stb.vb, jb - 1, ja)
        vb = _via_prop_dn(casc, stb.vb, jb - 1, ja - 1)
    end
    return sta.wbar * (vt + vb) / norm2
end

# dispatch one pair to its kernel for one polarization's cascade.
# `vsts` is unused for TE pairs (vias couple TM only); `volsts` holds
# per-layer states for the active polarization's cascade.
@inline function _elem_pair_pol(f::Int, s::Int, casc::PlanarCascade{T},
        vsts, volsts, norm2::Float64, n3_2::Float64) where {T}
    if f >= 0
        if s >= 0
            return planar_modal_voltage(casc, f, s) / norm2
        elseif _is_vol_elem(s)
            # sheet field <- vol source = vol moment of the sheet's
            # modal voltage on the source layer's faces (reciprocity)
            jb = _vol_elem_layer(s)
            st = volsts[jb]
            return st.wbar * (planar_modal_voltage(casc, jb, f) +
                planar_modal_voltage(casc, jb - 1, f)) / norm2
        else
            # sheet field <- via source (TM only)
            jb, kb = _via_elem_layer(s), _via_elem_kind(s)
            return _via_sheet_kern(jb, kb, f, casc, vsts[jb], norm2)
        end
    elseif _is_vol_elem(f)
        ja = _vol_elem_layer(f)
        sta = volsts[ja]
        if s >= 0
            # vol field <- sheet source: smooth-V moment on layer ja
            return sta.wbar * (planar_modal_voltage(casc, ja, s) +
                planar_modal_voltage(casc, ja - 1, s)) / norm2
        elseif _is_vol_elem(s)
            return _vol_vol_kern(ja, _vol_elem_layer(s), casc,
                sta, volsts[_vol_elem_layer(s)], norm2)
        else
            # vol field <- via source: reciprocity of via<-vol
            jb, kb = _via_elem_layer(s), _via_elem_kind(s)
            return _vol_via_kern(jb, kb, ja, casc, vsts[jb], sta, norm2)
        end
    else
        ja, ka = _via_elem_layer(f), _via_elem_kind(f)
        if s >= 0
            return _via_sheet_kern(ja, ka, s, casc, vsts[ja], norm2)
        elseif _is_vol_elem(s)
            return _vol_via_kern(ja, ka, _vol_elem_layer(s), casc,
                vsts[ja], volsts[_vol_elem_layer(s)], norm2)
        else
            jb, kb = _via_elem_layer(s), _via_elem_kind(s)
            return _via_via_kern(ja, ka, jb, kb, casc, vsts[ja],
                vsts[jb], n3_2)
        end
    end
end
