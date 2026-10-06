# PlanarSurface.jl — conductor surface-impedance models
#
# The `surface_zs` keyword of assemble_planar_z / solve_planar adds a
# surface-impedance Gram loss term Z -= Zs * Gram on every sheet.  This
# file produces that value from bulk conductor properties and published
# roughness correction factors:
#
#   smooth good conductor (e^{+i wt}):  Zs = Rs * (1 + i),
#     Rs = sqrt(omega * mu / (2 sigma)),  delta = sqrt(2/(omega*mu*sigma))
#
#   Hammerstad-Jensen (and modified form, Shlepnev's RF parameter):
#     K = 1 + (rf - 1) * (2/pi) * atan(1.4 * (rms/delta)^2)
#   K saturates at rf for rms >> delta (classical rf = 2).
#
#   Huray "snowball" (N spherical nodules of radius r per flat area A):
#     K = 1 + (3/2) * (N*4*pi*r^2/A) / (1 + delta/r + delta^2/(2 r^2))
#   the magnetic-screening term vanishes for delta << r (large nodules
#   behave flat) and saturates for delta >> r.
#
# Both factors correct the dissipative power; applying them to the full
# complex Zs (default) also scales the internal inductance, matching the
# common solver convention.  loss_only=true restricts the correction to
# Re(Zs), matching the models' literal attenuation-constant statement.

export PlanarRoughness, HammerstadRoughness, HurayRoughness
export planar_skin_depth, roughness_factor, planar_surface_zs

"""Abstract conductor-roughness model for `roughness_factor`."""
abstract type PlanarRoughness end

"""Hammerstad-Jensen roughness: `rms` profile height [m], `rf` asymptotic
loss multiplier (classical model: `rf = 2`)."""
struct HammerstadRoughness <: PlanarRoughness
    rms::Float64
    rf::Float64
    function HammerstadRoughness(rms::Real; rf::Real=2.0)
        rms = Float64(rms); rf = Float64(rf)
        (isfinite(rms) && rms >= 0) || throw(ArgumentError(
            "roughness rms must be finite and >= 0, got $rms"))
        (isfinite(rf) && rf >= 1) || throw(ArgumentError(
            "roughness rf must be finite and >= 1, got $rf"))
        return new(rms, rf)
    end
end

"""Huray "snowball" roughness: spherical nodules of `radius` [m] at
`density` nodules per unit conductor area [m^-2]."""
struct HurayRoughness <: PlanarRoughness
    radius::Float64
    density::Float64
    function HurayRoughness(radius::Real, density::Real)
        radius = Float64(radius); density = Float64(density)
        (isfinite(radius) && radius > 0) || throw(ArgumentError(
            "nodule radius must be finite and > 0, got $radius"))
        (isfinite(density) && density >= 0) || throw(ArgumentError(
            "nodule density must be finite and >= 0, got $density"))
        return new(radius, density)
    end
end

"""Skin depth delta = 1/sqrt(pi * f * mu0 * mur * sigma) [m]."""
function planar_skin_depth(f::Real, sigma::Real; mur::Real=1.0)
    (isfinite(f) && f > 0) || throw(ArgumentError(
        "frequency must be finite and > 0, got $f"))
    (isfinite(sigma) && sigma > 0) || throw(ArgumentError(
        "conductivity must be finite and > 0, got $sigma"))
    (isfinite(mur) && mur > 0) || throw(ArgumentError(
        "mur must be finite and > 0, got $mur"))
    product=pi*f*_MU0*mur*sigma
    if !isfinite(product) || iszero(product) || _planar_metal_subnormal(product) ||
            _planar_metal_subnormal(pi*f*_MU0) || _planar_metal_subnormal(pi*f*_MU0*mur)
        if f isa Float64 && sigma isa Float64 && mur isa Float64
            m,e=_planar_metal_sqrt_parts(_planar_metal_product_parts((pi*_MU0,f,mur,sigma)))
            return ldexp(inv(m),-e)
        end
        return _planar_surface_wide(f,sigma,mur,:depth,nothing)
    end
    return inv(sqrt(product))
end

"""Roughness correction factor K >= 1 at `f` [Hz] for conductor `sigma`
[S/m] (relative permeability `mur`)."""
roughness_factor

function roughness_factor(m::HammerstadRoughness, f::Real,
        sigma::Real; mur::Real=1.0)
    d = planar_skin_depth(f, sigma; mur=mur)
    (iszero(m.rms) || m.rf==1) && return one(promote_type(Float64,typeof(f),typeof(sigma),typeof(mur)))
    q=1.4*(m.rms/d)^2
    if !isfinite(d) || iszero(d) || !isfinite(q) || iszero(q) || _planar_metal_subnormal(q)
        return _planar_surface_wide(f,sigma,mur,:hammerstad,m)
    end
    return 1 + (m.rf - 1) * (2 / pi) * atan(q)
end

function roughness_factor(m::HurayRoughness, f::Real,
        sigma::Real; mur::Real=1.0)
    d = planar_skin_depth(f, sigma; mur=mur)
    iszero(m.density) && return one(promote_type(Float64,typeof(f),typeof(sigma),typeof(mur)))
    x = d / m.radius
    area=4pi*m.radius^2*m.density
    denominator=1+x+x^2/2
    if !isfinite(area) || iszero(area) || _planar_metal_subnormal(area) ||
            !isfinite(denominator) || !isfinite(m.radius^2) || iszero(m.radius^2) ||
            _planar_metal_subnormal(m.radius^2)
        return _planar_surface_wide(f,sigma,mur,:huray,m)
    end
    return 1 + 1.5 * area / denominator
end

"""Complex surface impedance [Ohm] of a conductor with bulk `sigma` [S/m]
at `f` [Hz] under e^{+i wt}:  Zs = Rs*(1 + i) with
Rs = sqrt(omega*mu0*mur / (2*sigma)).  `roughness` (a `PlanarRoughness`
model) scales Zs by its published factor; `loss_only=true` applies the
factor to Re(Zs) only (the models' literal attenuation statement)."""
function planar_surface_zs(f::Real, sigma::Real; mur::Real=1.0,
        roughness::Union{Nothing,PlanarRoughness}=nothing,
        loss_only::Bool=false)
    d = planar_skin_depth(f, sigma; mur=mur)
    rs = 0.5 * _MU0 * mur * 2pi * f * d   # = sqrt(omega*mu / (2*sigma))
    product=pi*f*_MU0*mur*sigma
    if !isfinite(product) || iszero(product) || _planar_metal_subnormal(product) ||
            _planar_metal_subnormal(pi*f*_MU0) || _planar_metal_subnormal(pi*f*_MU0*mur) || !isfinite(rs) || iszero(rs)
        if roughness!==nothing
            return _planar_surface_wide(f,sigma,mur,:impedance,(roughness,loss_only))
        elseif f isa Float64 && sigma isa Float64 && mur isa Float64
            m,e=_planar_metal_sqrt_parts(_planar_metal_product_parts((pi*_MU0,f,mur),sigma))
            rs=ldexp(m,e)
        else
            return _planar_surface_wide(f,sigma,mur,:impedance,(roughness,loss_only))
        end
    end
    k = roughness === nothing ? 1.0 :
        roughness_factor(roughness, f, sigma; mur=mur)
    result=loss_only ? complex(k*rs,rs) : k*rs*(1+1im)
    if !(f isa BigFloat && sigma isa BigFloat && mur isa BigFloat) &&
            (!isfinite(result) || (roughness!==nothing && _planar_metal_subnormal(rs)))
        return _planar_surface_wide(f,sigma,mur,:impedance,(roughness,loss_only))
    end
    return result
end

# Rare positive material equations use scoped precision; ordinary paths
# keep their original arithmetic. The exponent helpers are shared with
# the independently qualified conductor film recovery.
function _planar_surface_wide(f,sigma,mur,kind,model)
    result_type=promote_type(Float64,typeof(f),typeof(sigma),typeof(mur))
    kind===:impedance && (result_type=Complex{result_type})
    input_precision(x)=x isa BigFloat ? precision(x) : 0
    working_precision=max(4096,precision(BigFloat),input_precision(f),input_precision(sigma),input_precision(mur))
    result=setprecision(BigFloat,working_precision) do
        factor=BigFloat(pi)*BigFloat(_MU0)*BigFloat(f)*BigFloat(mur)
        conductivity=BigFloat(sigma)
        depth=inv(sqrt(factor*conductivity))
        if kind===:impedance
            planar_surface_zs(BigFloat(f),conductivity;mur=BigFloat(mur),
                roughness=model[1],loss_only=model[2])
        elseif kind===:depth
            depth
        elseif kind===:resistance
            sqrt(factor/conductivity)
        elseif kind===:hammerstad
            1+(BigFloat(model.rf)-1)*(2/BigFloat(pi))*atan(BigFloat(1.4)*(BigFloat(model.rms)/depth)^2)
        else
            radius=BigFloat(model.radius);x=depth/radius
            1+BigFloat(1.5)*4*BigFloat(pi)*radius^2*BigFloat(model.density)/(1+x+x^2/2)
        end
    end
    return convert(result_type,result)
end
