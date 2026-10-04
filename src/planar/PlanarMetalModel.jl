# Stratified conductor transmission lines and coupled two-face metal loss.

export PlanarConductorLayer, planar_layered_surface_zs, planar_two_sheet_zs

"""Finite conductor layer for a surface-impedance stack, with conductivity
`sigma` [S/m], `thickness` [m], and relative permeability `mur`.
Layer vectors are ordered from the exposed surface toward the substrate."""
struct PlanarConductorLayer{T<:Number}
    sigma::T
    thickness::T
    mur::T
    function PlanarConductorLayer{T}(sigma::T,t::T,mur::T) where {T<:Number}
        isfinite(sigma) && real(sigma)>0 || throw(ArgumentError("conductor sigma must be finite with Re > 0"))
        isfinite(t) && real(t)>0 || throw(ArgumentError("conductor thickness must be finite with Re > 0"))
        isfinite(mur) && real(mur)>0 || throw(ArgumentError("conductor mur must be finite with Re > 0"))
        return new{T}(sigma,t,mur)
    end
end

function PlanarConductorLayer(sigma::Number,t::Number;mur::Number=1.0)
    s,h,m=promote(sigma,t*1.0,mur)
    return PlanarConductorLayer{typeof(s)}(s,h,m)
end

@inline function _planar_conductor_line(f::Number,sigma::Number,mur::Number)
    omega=2pi*f
    gamma2=1im*omega*_MU0*mur*sigma
    gamma=sqrt(gamma2)
    zc=1im*omega*_MU0*mur/gamma
    return gamma2,gamma,zc
end

function _planar_roughen_metal(z,f,sigma,mur,roughness,loss_only)
    roughness===nothing && return z
    k=roughness_factor(roughness,real(f),real(sigma);mur=real(mur))
    return loss_only ? complex(k*real(z),imag(z)) : k*z
end

"""Exact conductor slab relation between face E and exterior face H.
This material relation is not a sheet-current-jump boundary condition:
the exterior/background field correction is required before assembly."""
function _planar_conductor_face_zs(f::Number,sigma::Number,thickness::Number;
        mur::Number=1.0,roughness=nothing,loss_only::Bool=false)
    isfinite(f) && real(f)>0 || throw(ArgumentError("frequency must be finite with Re > 0"))
    layer=PlanarConductorLayer(sigma,thickness;mur)
    g2,g,zc=_planar_conductor_line(f,layer.sigma,layer.mur)
    q=g2*layer.thickness^2
    if abs2(q)<1e-8
        dc=inv(layer.sigma*layer.thickness)
        diagonal=dc*(1+q/3-q*q/45+2q*q*q/945)
        coupling=dc*(1-q/6+7q*q/360-31q*q*q/15120)
    else
        e=exp(-g*layer.thickness)
        den=1-e*e
        diagonal=zc*(1+e*e)/den
        coupling=zc*(2*e)/den
    end
    diagonal=_planar_roughen_metal(diagonal,f,layer.sigma,layer.mur,roughness,loss_only)
    coupling=_planar_roughen_metal(coupling,f,layer.sigma,layer.mur,roughness,loss_only)
    return [diagonal coupling;coupling diagonal]
end

"""`planar_two_sheet_zs(f,sigma,thickness;mur=1,roughness=nothing)`
returns the diagonal 2×2 impedance for the Rautio–Demir two-sheet thick
metal approximation. Place identical metal footprints at the physical
top and bottom faces, separated by `thickness`; each face receives the
open-back finite-film impedance for half that thickness. Their mutual
electromagnetic coupling is supplied by the layered Green function.
Use the matrix as `sheet_coupling_zs`, with `surface_zs=0`, or use its
diagonal entries as the two `surface_zs` values. Leave the intervening
host dielectric free of conductivity and volume-current metal loss.
The approximation omits lateral side currents; resolve those with volume
layers when they materially affect the result."""
function planar_two_sheet_zs(f::Number,sigma::Number,thickness::Number;
        mur::Number=1.0,roughness=nothing,loss_only::Bool=false)
    layer=PlanarConductorLayer(sigma,thickness;mur)
    z=planar_layered_surface_zs(f,[PlanarConductorLayer(layer.sigma,
        layer.thickness/2;mur=layer.mur)];roughness,loss_only)
    return [z zero(z);zero(z) z]
end

"""`planar_layered_surface_zs(f,layers; substrate_sigma=nothing,load=:open)`
cascades finite conductor layers from the surface inward.  Supply
`substrate_sigma` for a bulk conductor beneath a plating stack (for
example, Au/Ni over copper).  Otherwise `load=:open` imposes zero H at
the back face of a finite film, `load=:pec` imposes zero E, or a numeric
load specifies its E/H impedance in ohms.  `roughness` applies the
chosen correction to the resulting exposed-surface impedance.
This is a one-face input impedance; the published two-sheet approximation
uses it independently on each half of the conductor thickness."""
function planar_layered_surface_zs(f::Number,layers::AbstractVector{<:PlanarConductorLayer};
        substrate_sigma=nothing,substrate_mur::Number=1.0,load=:open,
        roughness=nothing,loss_only::Bool=false)
    isfinite(f) && real(f)>0 || throw(ArgumentError("frequency must be finite with Re > 0"))
    isempty(layers) && substrate_sigma===nothing &&
        throw(ArgumentError("a conductor layer or bulk substrate is required"))
    z=if substrate_sigma!==nothing
        load===:open || throw(ArgumentError("specify substrate_sigma or an explicit load"))
        isfinite(substrate_sigma) && real(substrate_sigma)>0 &&
            isfinite(substrate_mur) && real(substrate_mur)>0 ||
            throw(ArgumentError("bulk substrate parameters must be finite with Re > 0"))
        _planar_conductor_line(f,substrate_sigma,substrate_mur)[3]
    elseif load===:open
        ComplexF64(Inf)
    elseif load===:pec
        0.0im
    elseif load isa Number && isfinite(load)
        complex(load)
    else
        throw(ArgumentError("load must be :open, :pec, or a finite impedance"))
    end
    for layer in Iterators.reverse(layers)
        g2,g,zc=_planar_conductor_line(f,layer.sigma,layer.mur)
        q=g2*layer.thickness^2
        if abs2(q)<1e-8
            a=1+q/2+q*q/24+q*q*q/720
            sh=layer.thickness*(1+q/6+q*q/120+q*q*q/5040)
            b=1im*2pi*f*_MU0*layer.mur*sh
            c=layer.sigma*sh
        else
            e=exp(-g*layer.thickness)
            e2=e*e
            a=1+e2
            b=zc*(1-e2)
            c=(1-e2)/zc
        end
        T=promote_type(typeof(a),typeof(b),typeof(c),typeof(z))
        z=_planar_input_abcd(T(a),T(b),T(c),T(z))
    end
    sigma=isempty(layers) ? substrate_sigma : first(layers).sigma
    mur=isempty(layers) ? substrate_mur : first(layers).mur
    return _planar_roughen_metal(z,f,sigma,mur,roughness,loss_only)
end
