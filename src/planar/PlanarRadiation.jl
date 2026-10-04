export PlanarRadiationPattern, planar_farfield, planar_radiated_power
using PlotlySupply: scatterpolar
export planar_radiation_stack, write_planar_radiation_csv
export planar_radiation_metrics, plot_planar_radiation

"""Continuous-angle fields `r*E*exp(+ik*r)` [V], radiation intensity
[W/sr], and accepted peak-phasor power [W]. Matrices are indexed
`[theta,phi]`; angles are radians. The source stack is extended infinitely
in x/y for this current-based radiation calculation. Its radiation model
is distinct from the finite-sidewall model that produced the currents."""
struct PlanarRadiationPattern
    theta::Vector{Float64}
    phi::Vector{Float64}
    etheta::Matrix{ComplexF64}
    ephi::Matrix{ComplexF64}
    intensity::Matrix{Float64}
    accepted_power::Union{Nothing,Float64}
    frequency::Float64
end

function _planar_radiation_power(value,label="accepted power")
    value===nothing && return nothing
    value isa Real && isfinite(value) && value>0 ||
        throw(ArgumentError("$label must be finite and positive"))
    stored=Float64(value)
    isfinite(stored) && stored>0 ||
        throw(ArgumentError("$label must fit a finite positive Float64"))
    return stored
end

"""Choose the infinite-layer radiation environment independently of the
current solve. PEC terminations represent infinite ground planes. Open
terminations specify lossless exterior media. A native resistive
`FREESPACE` cover can explicitly be replaced with `top=TERM_SPACE` (or
`bottom=TERM_SPACE`) for current-based far-field postprocessing."""
function planar_radiation_stack(stack::PlanarStackup;bottom=stack.bottom,top=stack.top)
    result=PlanarStackup(stack.layers,bottom,top,stack.a,stack.b)
    planar_validate(result)
    for term in (result.bottom,result.top)
        term.kind in (TERM_PEC,TERM_OPEN) || throw(ArgumentError(
            "radiation terminations must be an infinite PEC plane or an open exterior medium"))
        term.kind==TERM_OPEN && (!isreal(term.epsr) || !isreal(term.mur)) &&
            throw(ArgumentError("radiation exterior media must be lossless"))
    end
    return result
end

@inline function _planar_rooftop_fourier(prob::PlanarProblem,p,kx,ky)
    basis,grid=prob.basis,prob.grid
    kind=_sheet_kind(basis.kind[p]);i,j=basis.ei[p],basis.ej[p]
    dx,dy=grid.dx,grid.dy
    if _is_via_kind(basis.kind[p])
        return dx*dy*_sinc0(kx*dx/2)*_sinc0(ky*dy/2)*
            cis(kx*(i-.5)*dx+ky*(j-.5)*dy)
    elseif kind==_BASIS_X_FULL
        return dx*dy*_sinc0sq(kx*dx/2)*_sinc0(ky*dy/2)*
            cis(kx*i*dx+ky*(j-.5)*dy)
    elseif kind in (_BASIS_X_LO,_BASIS_X_HI)
        sign=kind==_BASIS_X_LO ? 1. : -1.
        return dy*_sinc0(ky*dy/2)*cis(kx*i*dx+ky*(j-.5)*dy)*
            complex(_ramp_cos(kx,dx),sign*_ramp_sin(kx,dx))
    elseif kind==_BASIS_Y_FULL
        return dx*dy*_sinc0(kx*dx/2)*_sinc0sq(ky*dy/2)*
            cis(kx*(i-.5)*dx+ky*j*dy)
    else
        sign=kind==_BASIS_Y_LO ? 1. : -1.
        return dx*_sinc0(kx*dx/2)*cis(kx*(i-.5)*dx+ky*j*dy)*
            complex(_ramp_cos(ky,dy),sign*_ramp_sin(ky,dy))
    end
end

# Fields of a unit plane wave incident from the observation medium.
# Reciprocity converts its exact overlap with each current basis into
# the outgoing field. V is transverse E; H follows V'=-Zseries*H.
function _planar_receive_fields(stack,omega,kc2,pol,vinc,zext)
    casc=planar_mode_cascade(stack,omega,kc2,pol)
    L=length(stack.layers)
    V=zeros(ComplexF64,L+1);H=similar(V)
    zd=casc.zdn[end]
    if _planarisinf(zd)
        V[end]=2vinc;H[end]=0
    else
        den=zd+zext
        iszero(den) && throw(DomainError(den,"radiation receive-field resonance"))
        V[end]=2vinc*zd/den;H[end]=-2vinc/den
    end
    for j in L:-1:1
        a,b,c,scale=_planar_layer_abcd(pol,omega,kc2,stack.layers[j])
        load=casc.zdn[j]
        if _planarisinf(load)
            H[j]=0
            V[j]=abs(a)>=abs(c)*abs(zext) ? scale*V[j+1]/a : -scale*H[j+1]/c
        else
            dv,dh=a*load+b,c*load+a
            # The H equation also resolves a voltage node at a quarter
            # wave; choosing the finite equation avoids division by zero.
            current=abs(dv)>=abs(dh)*abs(zext) ? scale*V[j+1]/dv : -scale*H[j+1]/dh
            H[j]=-current;V[j]=load*current
        end
    end
    all(isfinite,V) && all(isfinite,H) || throw(DomainError(kc2,"nonfinite radiation receive fields"))
    return V,H
end

# The uniform-medium solution also supplies the exact horizon limit;
# subtracting k² from kc² at grazing incidence would lose its digits.
function _planar_receive_direction(stack,omega,k,kc2,c,pol,vinc,zext)
    term=stack.top
    matched=all(l->l.epsr==term.epsr && l.mur==term.mur &&
        l.epsr_z==l.epsr && l.mur_z==l.mur,stack.layers)
    lower=stack.bottom
    if matched && (lower.kind==TERM_PEC || (lower.kind==TERM_OPEN && lower.epsr==term.epsr && lower.mur==term.mur))
        heights=real.(planar_interfaces(stack));height=heights[end];kz=k*c
        V=Vector{ComplexF64}(undef,length(heights));H=similar(V)
        for j in eachindex(heights)
            down=cis(kz*(heights[j]-height))
            up=lower.kind==TERM_PEC ? -cis(-kz*(heights[j]+height)) : 0.0im
            V[j]=vinc*(down+up);H[j]=vinc*(-down+up)/zext
        end
        return V,H
    end
    return _planar_receive_fields(stack,omega,kc2,pol,vinc,zext)
end

@inline function _planar_receive_moments(layer,omega,kc2,pol,vb,vt,hb,ht)
    h=real(layer.thickness);g2=_planar_gamma2_layer(pol,kc2,omega,layer);x=sqrt(g2)*h
    zs=pol==TE_POL ? 1im*omega*_MU0*layer.mur : g2/(1im*omega*_EPS0*layer.epsr)
    ys=pol==TM_POL ? 1im*omega*_EPS0*layer.epsr : g2/(1im*omega*_MU0*layer.mur)
    if real(x)>20
        e=exp(-x);e2=e*e
        w=(1-e)/(x*(1+e))
        vm=w*(vb+vt);hm=h*w*(hb+ht)
        cb=inv(x*x)-2e/(x*(1-e2))
        ct=(1+e2)/(x*(1-e2))-inv(x*x)
        return vm,hm,h*(cb*hb+ct*ht)
    end
    q=x*x
    s,c2,c3=if abs(q)<1e-5
        (1+q/6+q*q/120+q^3/5040,
            .5+q/24+q*q/720+q^3/40320,
            1/3+q/30+q*q/840+q^3/45360)
    else
        (sinh(x)/x,(cosh(x)-1)/q,(cosh(x)-sinh(x)/x)/q)
    end
    return vb*s-zs*h*hb*c2,
        h*hb*s-ys*h*h*vb*c2,
        h*hb*(s-c2)-ys*h*h*vb*c3
end

function _planar_radiation_overlap(prob::PlanarProblem,coeff,stack,omega,kx,ky,pol,V,H,side)
    cp,sp=if iszero(kx) && iszero(ky)
        # Caller rotates this normal-incidence basis through its phi.
        (1.,0.)
    else
        kc=hypot(kx,ky);(kx/kc,ky/kc)
    end
    overlap=0.0im;L=length(stack.layers);kc2=kx*kx+ky*ky
    for p in eachindex(coeff)
        iszero(coeff[p]) && continue
        kind=prob.basis.kind[p];elem=_basis_elem(prob.basis,p,prob.sheets,prob.vias,prob.vols)
        field=if _is_via_kind(kind)
            pol==TE_POL && continue
            origlayer=prob.vias[prob.basis.level[p]].layer
            j=side==1 ? origlayer : L-origlayer+1
            layer=stack.layers[j]
            _,hu,ht=_planar_receive_moments(layer,omega,kc2,TM_POL,V[j],V[j+1],H[j],H[j+1])
            side*sqrt(kc2)/(omega*_EPS0*layer.epsr_z)*(kind==_BASIS_VIA_U ? hu : side==1 ? ht : hu-ht)
        else
            value=if _is_vol_kind(kind)
                origlayer=prob.vols[prob.basis.level[p]].layer
                j=side==1 ? origlayer : L-origlayer+1
                first(_planar_receive_moments(stack.layers[j],omega,kc2,pol,V[j],V[j+1],H[j],H[j+1]))
            else
                V[(side==1 ? elem : L-elem)+1]
            end
            directional=pol==TE_POL ? (_is_xdir(kind) ? -sp : cp) : (_is_xdir(kind) ? cp : sp)
            value*directional
        end
        overlap+=coeff[p]*_planar_rooftop_fourier(prob,p,kx,ky)*field
    end
    return overlap
end

function _planar_radiation_overlap(prob::PlanarConformalProblem,coeff,stack,omega,kx,ky,pol,V,H,side)
    kc=hypot(kx,ky);cp,sp=iszero(kc) ? (1.,0.) : (kx/kc,ky/kc)
    result=0.0im;L=length(stack.layers)
    for b in eachindex(coeff)
        fx,fy=_planar_conformal_fourier(prob,b,kx,ky)
        level=prob.basis.interfaces[b]
        v=V[(side==1 ? level : L-level)+1]
        result+=coeff[b]*v*(pol==TE_POL ? -sp*fx+cp*fy : cp*fx+sp*fy)
    end
    return result
end
function _planar_radiation_overlap(prob::PlanarHybridProblem,coeff,stack,omega,kx,ky,pol,V,H,side)
    nc=length(prob.conformal.basis.width)
    return _planar_radiation_overlap(prob.conformal,view(coeff,1:nc),stack,omega,kx,ky,pol,V,H,side)+
        _planar_radiation_overlap(prob.bulk,view(coeff,nc+1:length(coeff)),stack,omega,kx,ky,pol,V,H,side)
end
_planar_radiation_basis_count(prob::PlanarProblem)=planar_basis_count(prob.basis)
_planar_radiation_basis_count(prob::PlanarConformalProblem)=length(prob.basis.width)
_planar_radiation_basis_count(prob::PlanarHybridProblem)=
    length(prob.conformal.basis.width)+planar_basis_count(prob.bulk.basis)

function _planar_farfield_direction(prob,coeff,stack,frequency,theta,phi)
    side=theta<=pi/2 ? 1 : -1
    term=side==1 ? stack.top : stack.bottom
    term.kind==TERM_PEC && return 0.0im,0.0im,0.
    omega=2pi*frequency;k=omega/_C0*sqrt(real(term.epsr*term.mur))
    eta=sqrt(_MU0*real(term.mur)/(_EPS0*real(term.epsr)))
    # Grazing incidence is the continuous one-sided limit. Keeping a
    # small nonzero axial component resolves the TE/TM cutoff degeneracy.
    c=max(abs(cos(theta)),1e-7);s=sqrt(max(0.,1-c*c))
    # Normal incidence uses a tiny azimuth-preserving transverse k, so
    # polarization rotates with phi without dividing by zero.
    s=max(s,1e-12)
    kx,ky=k*s*cos(phi),k*s*sin(phi);kc2=kx*kx+ky*ky
    receiving=side==1 ? stack : PlanarStackup(reverse(stack.layers),stack.top,stack.bottom,stack.a,stack.b)
    vtm=side*c
    Vtm,Htm=_planar_receive_direction(receiving,omega,k,kc2,c,TM_POL,vtm,eta*c)
    Vte,Hte=_planar_receive_direction(receiving,omega,k,kc2,c,TE_POL,1.,eta/c)
    tm=_planar_radiation_overlap(prob,coeff,receiving,omega,kx,ky,TM_POL,Vtm,Htm,side)
    te=_planar_radiation_overlap(prob,coeff,receiving,omega,kx,ky,TE_POL,Vte,Hte,side)
    # V/H are referenced at the exterior interface. Restore its physical
    # origin to obtain the absolute phase of the outgoing spherical wave.
    zface=side==1 ? sum(real(l.thickness) for l in stack.layers) : 0.
    factor=-1im*omega*_MU0*real(term.mur)/(4pi)*cis(side*k*c*zface)
    et,ep=factor*tm,factor*te
    return et,ep,(abs2(et)+abs2(ep))/(2eta)
end

"""Compute polarized far fields by Lorentz reciprocity in the infinite
layered substrate. Exact analytic Fourier integrals are used for sheet,
via, volume and conformal triangle currents. `coefficients` are solved
peak-phasor basis amplitudes. Angles are radians; azimuth is measured from
+x toward +y. Grazing angles use a one-sided axial-cosine limit of 1e-7.
This postprocessing does not remove finite-box errors in the input currents;
box/cover convergence must be checked for radiating designs."""
function planar_farfield(prob::Union{PlanarProblem,PlanarConformalProblem,PlanarHybridProblem},coefficients::AbstractVector,freq::Real;
        theta=range(0.,pi;length=181),phi=[0.],radiation_stack=prob.stack,
        accepted_power=nothing,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    isfinite(freq) && freq>0 || throw(ArgumentError("radiation frequency must be finite and positive"))
    power=_planar_radiation_power(accepted_power)
    nb=_planar_radiation_basis_count(prob)
    length(coefficients)==nb && all(isfinite,coefficients) || throw(ArgumentError("radiation coefficients must match the basis and be finite"))
    all(t->isfinite(t) && 0<=t<=pi,theta) && all(isfinite,phi) && !isempty(theta) && !isempty(phi) ||
        throw(ArgumentError("theta must lie in [0,pi] and phi must be finite; neither angle list may be empty"))
    L=length(prob.stack.layers)
    payload=_checked_payload_sum("planar radiation",
        _checked_array_payload_bytes(ComplexF64,nb),
        _checked_array_payload_bytes(ComplexF64,2,length(theta),length(phi)),
        _checked_array_payload_bytes(Float64,length(theta),length(phi)),
        _checked_array_payload_bytes(Float64,length(theta)+length(phi)),
        _checked_array_payload_bytes(ComplexF64,32,L+1))
    _enforce_payload_limit(payload,max_bytes,"planar radiation","max_bytes")
    stack=planar_radiation_stack(radiation_stack)
    stack.layers==prob.stack.layers || throw(ArgumentError("radiation layers must match the source layers; only exterior terminations may change"))
    coeff=ComplexF64.(coefficients);all(isfinite,coeff) || throw(ArgumentError("radiation coefficients must fit finite ComplexF64"))
    ts,ps=Float64.(theta),Float64.(phi)
    all(isfinite,ts) && all(isfinite,ps) && isfinite(Float64(freq)) || throw(ArgumentError("radiation angles and frequency must fit finite Float64"))
    et=Matrix{ComplexF64}(undef,length(ts),length(ps));ep=similar(et);u=Matrix{Float64}(undef,size(et))
    for j in eachindex(ps),i in eachindex(ts)
        et[i,j],ep[i,j],u[i,j]=_planar_farfield_direction(prob,coeff,stack,Float64(freq),ts[i],ps[j])
    end
    return PlanarRadiationPattern(ts,ps,et,ep,u,power,Float64(freq))
end

_planar_radiation_problem(result::Union{PlanarContractedResult,PlanarCalibratedResult})=_planar_radiation_problem(result.raw)
_planar_radiation_problem(result)=result.problem

"""Far field of a solved network. The default sets `port` to one volt and
shorts the other gap voltages. `incident_waves` instead specifies incident
power-wave amplitudes with matched port terminations; accepted power
includes the reflected-wave correction. Calibrated/contracted results
retain their physical source coefficient mapping."""
function planar_farfield(result::Union{PlanarResult,PlanarUFFTResult,PlanarSourceResult,PlanarContractedResult,PlanarCalibratedResult,PlanarConformalResult,PlanarConformalUFFTResult,PlanarConformalDefectResult,PlanarHybridResult,PlanarHybridUFFTResult};
        port::Integer=1,voltages=nothing,incident_waves=nothing,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kw...)
    n=size(result.y,1)
    source=_planar_radiation_problem(result)
    direct=result isa Union{PlanarConformalResult,PlanarConformalUFFTResult,PlanarConformalDefectResult,PlanarHybridResult,PlanarHybridUFFTResult}
    nb=_planar_radiation_basis_count(source)
    extra=result isa PlanarCalibratedResult ? _checked_array_payload_bytes(ComplexF64,nb,n) : 0
    reserve=_checked_payload_sum("radiation excitation",_checked_array_payload_bytes(ComplexF64,nb),
        _checked_array_payload_bytes(ComplexF64,8,n),extra)
    _enforce_payload_limit(reserve,max_bytes,"radiation excitation","max_bytes")
    refs=_planar_reference_values(hasproperty(result,:z0) ? result.z0 :
        [p.z0 for p in result.problem.ports],n;freq=real(result.freq))
    voltages===nothing || incident_waves===nothing || throw(ArgumentError("provide voltages or incident_waves"))
    input=voltages===nothing ? incident_waves : voltages
    if input===nothing
        1<=port<=n || throw(ArgumentError("port outside solved network"))
        v=zeros(ComplexF64,n);v[port]=1
    else
        input isa AbstractVector && length(input)==n && all(isfinite,input) || throw(ArgumentError("excitation must match solved ports and be finite"))
        v=ComplexF64.(input)
        if incident_waves!==nothing
            v=_planar_wave_voltage(result.s,v,refs)
        end
    end
    X=direct ? result.currents : _planar_coefficient_columns(result)
    current=result.y*v;pin=.5real(dot(v,current))
    tolerance=100eps(Float64)*norm(v)*norm(current)
    pin>=-tolerance || throw(ArgumentError("radiation gain requires nonnegative accepted power"))
    return planar_farfield(source,X*v,real(result.freq);accepted_power=pin>tolerance ? pin : nothing,
        max_bytes=Int(BigInt(_validated_resource_limit("max_bytes",max_bytes))-reserve),kw...)
end

"""Integrate the radiated power over the sphere using Gauss-Legendre
quadrature in cos(theta) and a periodic azimuth rule, with no duplicated
poles or 0/2pi samples. Returns power and the independently doubled-grid
change when `refine=true`; the change is a quadrature estimate, not a
certificate of current-solve accuracy."""
function planar_radiated_power(prob::Union{PlanarProblem,PlanarConformalProblem,PlanarHybridProblem},coefficients::AbstractVector,freq::Real;
        ntheta::Integer=24,nphi::Integer=48,refine::Bool=true,radiation_stack=prob.stack,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    ntheta>=2 && nphi>=4 || throw(ArgumentError("radiation power needs ntheta>=2 and nphi>=4"))
    _checked_array_payload_bytes(Float64,refine ? 2BigInt(ntheta) : BigInt(ntheta),
        refine ? 2BigInt(ntheta) : BigInt(ntheta);label="radiation quadrature rule")
    nphi<=typemax(Int)÷(refine ? 2 : 1) || throw(ArgumentError("radiation azimuth count overflows Int"))
    nt=Int(ntheta);np=Int(nphi)
    # The symmetric tridiagonal eigenvectors used by gauss_legendre are
    # dense; account for both the coarse and refined rules before work.
    rule=refine ? 2nt : nt
    reserve=_checked_payload_sum("radiation quadrature",_checked_array_payload_bytes(Float64,rule,rule),
        _checked_array_payload_bytes(Float64,6,rule+np*(refine ? 2 : 1)),
        _checked_array_payload_bytes(Float64,3,rule,rule))
    _enforce_payload_limit(reserve,max_bytes,"radiation quadrature","max_bytes")
    budget=Int(BigInt(_validated_resource_limit("max_bytes",max_bytes))-reserve)
    function integrate(nt,np)
        x,w=gauss_legendre(nt)
        pattern=planar_farfield(prob,coefficients,freq;theta=acos.(x),phi=[2pi*j/np for j in 0:np-1],radiation_stack,max_bytes=budget)
        return sum(w[i]*pattern.intensity[i,j] for i in 1:nt,j in 1:np)*2pi/np
    end
    coarse=integrate(nt,np)
    fine=refine ? integrate(2nt,2np) : coarse
    return (power=fine,coarse_power=coarse,relative_change=refine ? abs(fine-coarse)/max(fine,coarse,eps(Float64)) : nothing,
        ntheta=refine ? 2nt : nt,nphi=refine ? 2np : np)
end

"""Write complex polarized fields, intensity and accepted-power gain
(when available) to a deterministic CSV. Angles are radians and fields
are spherical-wave amplitudes [V], not fields at an arbitrary distance."""
function write_planar_radiation_csv(path::AbstractString,p::PlanarRadiationPattern)
    size(p.etheta)==size(p.ephi)==size(p.intensity)==(length(p.theta),length(p.phi)) || throw(ArgumentError("radiation data dimensions disagree"))
    all(isfinite,p.etheta) && all(isfinite,p.ephi) && all(x->isfinite(x) && x>=0,p.intensity) || throw(ArgumentError("radiation data must be finite"))
    all(t->isfinite(t) && 0<=t<=pi,p.theta) && all(isfinite,p.phi) && isfinite(p.frequency) && p.frequency>0 ||
        throw(ArgumentError("radiation frequency and angle metadata must be valid"))
    p.accepted_power===nothing || (isfinite(p.accepted_power) && p.accepted_power>0) || throw(ArgumentError("accepted power must be positive"))
    open(path,"w") do io
        println(io,"frequency_hz,theta_rad,phi_rad,Etheta_real_V,Etheta_imag_V,Ephi_real_V,Ephi_imag_V,intensity_W_per_sr,gain")
        for j in eachindex(p.phi),i in eachindex(p.theta)
            gain=p.accepted_power===nothing ? "" : string(4pi*p.intensity[i,j]/p.accepted_power)
            println(io,join((p.frequency,p.theta[i],p.phi[j],real(p.etheta[i,j]),imag(p.etheta[i,j]),real(p.ephi[i,j]),imag(p.ephi[i,j]),p.intensity[i,j],gain),','))
        end
    end
    return path
end

"""Polarization and power-normalized radiation curves. `radiated_power`
is an independently integrated full-sphere power, required for directivity
and efficiency. Gain uses accepted port power. Efficiencies above one are
preserved as a finite-box/current-normalization diagnostic. Circular
components are explicitly `(Etheta ± i*Ephi)/sqrt(2)`; `axial_ratio` is
the major/minor polarization-ellipse amplitude ratio (infinite for linear
polarization and undefined for a zero field)."""
function planar_radiation_metrics(p::PlanarRadiationPattern;radiated_power=nothing,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    accepted=_planar_radiation_power(p.accepted_power)
    radiated=_planar_radiation_power(radiated_power,"radiated power")
    shape=size(p.etheta)
    size(p.ephi)==size(p.intensity)==shape || throw(DimensionMismatch("radiation data dimensions disagree"))
    all(isfinite,p.etheta) && all(isfinite,p.ephi) && all(u->isfinite(u) && u>=0,p.intensity) || throw(ArgumentError("radiation data must be finite"))
    _enforce_payload_limit(_checked_array_payload_bytes(Float64,24,shape...),max_bytes,"radiation metrics","max_bytes")
    intensity=abs2.(p.etheta).+abs2.(p.ephi)
    q=abs2.(p.etheta).-abs2.(p.ephi);u=2real.(p.etheta.*conj.(p.ephi))
    spread=hypot.(q,u)
    minor=max.(0.,(intensity.-spread)./2);major=(intensity.+spread)./2
    ratio=[iszero(intensity[i]) ? NaN : iszero(minor[i]) ? Inf : sqrt(major[i]/minor[i]) for i in eachindex(intensity)]
    return (circular_plus=(p.etheta.+im.*p.ephi)./sqrt(2),
        circular_minus=(p.etheta.-im.*p.ephi)./sqrt(2),
        axial_ratio=reshape(ratio,shape),
        gain=accepted===nothing ? nothing : 4pi.*p.intensity./accepted,
        directivity=radiated===nothing ? nothing : 4pi.*p.intensity./radiated,
        efficiency=radiated===nothing || accepted===nothing ? nothing : radiated/accepted)
end

"""Interactive angular radiation cuts. `quantity` is :intensity,
:gain, :directivity, :etheta or :ephi. Power quantities use 10log10 and
field quantities use 20log10 when `db=true`. One trace per azimuth retains
the actual sampled polar angles, without smoothing or clipping nulls."""
function plot_planar_radiation(p::PlanarRadiationPattern;quantity::Symbol=:intensity,
        radiated_power=nothing,db::Bool=true,polar::Bool=true,
        title::AbstractString="Radiation pattern",width::Integer=750,height::Integer=650)
    quantity in (:intensity,:gain,:directivity,:etheta,:ephi) || throw(ArgumentError("unknown radiation plot quantity"))
    accepted=quantity==:gain ? _planar_radiation_power(p.accepted_power) : nothing
    radiated=quantity==:directivity ? _planar_radiation_power(radiated_power,"radiated power") : nothing
    quantity==:gain && accepted===nothing && throw(ArgumentError("gain requires accepted power"))
    quantity==:directivity && radiated===nothing && throw(ArgumentError("directivity requires integrated radiated power"))
    values=quantity==:etheta ? abs.(p.etheta) : quantity==:ephi ? abs.(p.ephi) :
        quantity==:gain ? 4pi.*p.intensity./accepted : quantity==:directivity ? 4pi.*p.intensity./radiated : p.intensity
    plotted=db ? (quantity in (:etheta,:ephi) ? 20log10.(values) : 10log10.(values)) : values
    sf=polar ? subplots(1,1;sync=false,show=false,title=String(title),width=Int(width),height=Int(height),
        specs=reshape([Spec(kind="polar")],1,1)) : _planar_plot_canvas(title,width,height)
    for j in eachindex(p.phi)
        trace=polar ? scatterpolar(theta=rad2deg.(p.theta),r=_planar_plot_finite(plotted[:,j]),mode="lines+markers",name="φ=$(rad2deg(p.phi[j]))°") :
            scatter(x=rad2deg.(p.theta),y=_planar_plot_finite(plotted[:,j]),mode="lines+markers",name="φ=$(rad2deg(p.phi[j]))°")
        addtraces!(sf,trace;row=1,col=1)
    end
    polar || relayout!(sf.plot,xaxis=attr(title="Theta [deg]"),yaxis=attr(title="$(String(quantity))$(db ? " [dB]" : "")"))
    return sf.plot
end
