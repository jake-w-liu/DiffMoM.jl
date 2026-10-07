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
    half=layer.thickness/2
    z=if half*2!=layer.thickness
        # Halving can lose a subnormal component while E/H stays finite.
        _planar_layered_surface_wide(f,[layer],nothing,1.0,:open,roughness,loss_only;
            thickness_divisor=2)
    else
        planar_layered_surface_zs(f,[PlanarConductorLayer(layer.sigma,half;mur=layer.mur)];roughness,loss_only)
    end
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
        characteristic=_planar_conductor_line(f,substrate_sigma,substrate_mur)[3]
        if !(f isa Complex{BigFloat} && substrate_sigma isa Complex{BigFloat} &&
                substrate_mur isa Complex{BigFloat}) && (!isfinite(characteristic) || iszero(characteristic))
            return _planar_layered_surface_wide(f,layers,substrate_sigma,substrate_mur,load,roughness,loss_only)
        end
        characteristic
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
        wide_parameters=f isa Complex{BigFloat} && layer isa PlanarConductorLayer{Complex{BigFloat}}
        if !wide_parameters &&
                (!isfinite(g2) || iszero(g2) || !isfinite(layer.thickness^2) || iszero(layer.thickness^2) ||
                 !isfinite(q) || iszero(q) || !isfinite(zc) || iszero(zc) ||
                 _planar_metal_subnormal(g2) || _planar_metal_subnormal(layer.thickness^2) ||
                 _planar_metal_subnormal(q) || _planar_metal_subnormal(zc))
            return _planar_layered_surface_wide(f,layers,substrate_sigma,substrate_mur,load,roughness,loss_only)
        end
        if abs2(q)<1e-8
            a=1+q/2+q*q/24+q*q*q/720
            series=1+q/6+q*q/120+q*q*q/5040
            sh=layer.thickness*series
            b=1im*2pi*f*_MU0*layer.mur*sh
            c=layer.sigma*sh
            if !wide_parameters && (_planar_metal_product_lost(layer.thickness,series) ||
                    _planar_metal_product_lost(layer.sigma,sh) || _planar_metal_subnormal(b))
                return _planar_layered_surface_wide(f,layers,substrate_sigma,substrate_mur,load,roughness,loss_only)
            end
        else
            x=g*layer.thickness
            if real(x)<0
                # Factor the growing exponential out of every ABCD entry.
                e=exp(x);e2=e*e;a=1+e2
                b=-zc*(1-e2);c=-(1-e2)/zc
            else
                e=exp(-x);e2=e*e;a=1+e2
                b=zc*(1-e2);c=(1-e2)/zc
            end
        end
        T=promote_type(typeof(a),typeof(b),typeof(c),typeof(z))
        z=_planar_input_abcd(T(a),T(b),T(c),T(z))
        if !wide_parameters && !isfinite(z)
            return _planar_layered_surface_wide(f,layers,substrate_sigma,substrate_mur,load,roughness,loss_only)
        end
    end
    sigma=isempty(layers) ? substrate_sigma : first(layers).sigma
    mur=isempty(layers) ? substrate_mur : first(layers).mur
    result=_planar_roughen_metal(z,f,sigma,mur,roughness,loss_only)
    fully_wide=f isa Complex{BigFloat} && all(layer->layer isa PlanarConductorLayer{Complex{BigFloat}},layers) &&
        (substrate_sigma===nothing || (substrate_sigma isa Complex{BigFloat} && substrate_mur isa Complex{BigFloat}))
    if !fully_wide && (!isfinite(result) || (roughness!==nothing &&
            (_planar_metal_subnormal(z) || iszero(real(z)) || iszero(imag(z)))))
        return _planar_layered_surface_wide(f,layers,substrate_sigma,substrate_mur,load,roughness,loss_only)
    end
    return result
end


# The public do-block precision API is scoped on supported Julia versions.
# Escalate only after a Float64 intermediate loses range; ordinary paths
# retain their arithmetic and avoid arbitrary-precision allocations.
function _planar_layered_surface_wide(f,layers,substrate_sigma,substrate_mur,load,roughness,loss_only; thickness_divisor=1)
    if thickness_divisor==1 && f isa Float64 && length(layers)==1 && first(layers) isa PlanarConductorLayer{Float64} &&
            substrate_sigma===nothing && load===:open && roughness===nothing
        layer=first(layers)
        z=_planar_scaled_open_real_slab(f,layer.sigma,layer.thickness,layer.mur)
        return _planar_roughen_metal(z,f,layer.sigma,layer.mur,roughness,loss_only)
    end
    scalar_type=promote_type(ComplexF64,typeof(complex(f)),typeof(complex(substrate_mur)))
    substrate_sigma!==nothing && (scalar_type=promote_type(scalar_type,typeof(complex(substrate_sigma))))
    load isa Number && (scalar_type=promote_type(scalar_type,typeof(complex(load))))
    for layer in layers
        scalar_type=promote_type(scalar_type,typeof(complex(layer.sigma)),
            typeof(complex(layer.thickness)),typeof(complex(layer.mur)))
    end
    input_precision(x)=x isa BigFloat ? precision(x) :
        x isa Complex{BigFloat} ? max(precision(real(x)),precision(imag(x))) : 0
    working_precision=max(16384,precision(BigFloat),input_precision(f),input_precision(substrate_sigma),
        input_precision(substrate_mur),input_precision(load))
    for layer in layers
        working_precision=max(working_precision,input_precision(layer.sigma),
            input_precision(layer.thickness),input_precision(layer.mur))
    end
    widened=setprecision(BigFloat,working_precision) do
        converted=PlanarConductorLayer{Complex{BigFloat}}[PlanarConductorLayer(Complex{BigFloat}(layer.sigma),
            Complex{BigFloat}(layer.thickness)/thickness_divisor;mur=Complex{BigFloat}(layer.mur)) for layer in layers]
        substrate=substrate_sigma===nothing ? nothing : Complex{BigFloat}(substrate_sigma)
        terminal=load isa Number ? Complex{BigFloat}(load) : load
        planar_layered_surface_zs(Complex{BigFloat}(f),converted;substrate_sigma=substrate,
            substrate_mur=Complex{BigFloat}(substrate_mur),load=terminal,roughness,loss_only)
    end
    result=convert(scalar_type,widened)
    isfinite(result) && (!iszero(result) || iszero(widened)) ||
        throw(ArgumentError("conductor surface impedance is not representable in its result type"))
    return result
end

# Detect component range loss before the ABCD quotient can hide it.
@inline _planar_metal_subnormal(x::Union{Float16,Float32,Float64})=issubnormal(x)
@inline _planar_metal_subnormal(x::Complex)=_planar_metal_subnormal(real(x)) || _planar_metal_subnormal(imag(x))
@inline _planar_metal_subnormal(x::Number)=false
@inline function _planar_metal_scalar_product_lost(x,y)
    (iszero(x) || iszero(y)) && return false
    value=x*y
    return !isfinite(value) || iszero(value) || _planar_metal_subnormal(value)
end
@inline function _planar_metal_product_lost(x,y)
    return _planar_metal_scalar_product_lost(real(x),real(y)) ||
        _planar_metal_scalar_product_lost(real(x),imag(y)) ||
        _planar_metal_scalar_product_lost(imag(x),real(y)) ||
        _planar_metal_scalar_product_lost(imag(x),imag(y))
end

# Exponent-balanced recovery for the positive Float64 open-film case.
@inline function _planar_metal_product_parts(values,divisor=1.0)
    m=1.0;e=0
    for value in values
        part,power=frexp(value);m*=part;e+=power
        m,power=frexp(m);e+=power
    end
    part,power=frexp(divisor);m/=part;e-=power
    m,power=frexp(m)
    return m,e+power
end
@inline _planar_metal_part_value(parts)=ldexp(parts[1],parts[2])
@inline function _planar_metal_sqrt_parts(parts)
    m,e=parts
    if isodd(e);m*=2;e-=1;end
    return sqrt(m),e÷2
end
function _planar_scaled_open_real_slab(f::Float64,sigma::Float64,h::Float64,mur::Float64=1.)
    all(x->isfinite(x) && x>0,(f,sigma,h,mur)) || throw(ArgumentError("finite positive real slab parameters required"))
    constant=2pi*_MU0
    qparts=_planar_metal_product_parts((constant,f,mur,sigma,h,h))
    q=_planar_metal_part_value(qparts)
    z=if q<1e-4
        # Invert the exponent before rounding sigma*h into Float64.
        sm,se=_planar_metal_product_parts((sigma,h));dc=ldexp(inv(sm),-se)
        realpart=dc*(1+q*q/45)
        imaginary=_planar_metal_part_value(_planar_metal_product_parts((constant,f,mur,h,1/3-2q*q/945)))
        complex(realpart,imaginary)
    else
        xm,xe=_planar_metal_sqrt_parts((qparts[1]/2,qparts[2]));x=ldexp(xm,xe)
        vparts=_planar_metal_sqrt_parts(_planar_metal_product_parts((constant,f,mur,.5),sigma))
        # An infinite propagation product has the exact thick-film limit;
        # finite products use the exponential relation and IEEE underflow.
        factor=if isinf(x)
            complex(1.,1.)
        else
            decay=exp(-complex(x,x));complex(1.,1.)*(1+decay^2)/(1-decay^2)
        end
        complex(ldexp(vparts[1]*real(factor),vparts[2]),ldexp(vparts[1]*imag(factor),vparts[2]))
    end
    isfinite(z) && !iszero(z) || throw(ArgumentError("slab impedance is outside finite nonzero Float64 range"))
    return z
end
