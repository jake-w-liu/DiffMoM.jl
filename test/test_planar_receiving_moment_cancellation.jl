module ReceivingMomentCancellationTests
using DiffMoM,LinearAlgebra,Test,ForwardDiff

function centered_cosh_integral(q)
    result=zero(q);n=1
    while true
        term=-2n*q^n/((2n+1)*(2n+2)*factorial(big(2n)))
        next=result+term
        next==result && return result
        result=next;n+=1
    end
end

function run_tests()
    # The former1e-5 boundary is retained as a regression witness. Grid,
    # frequency, thickness, original40-point quadrature and field tolerance
    # come from the existing physical radiation tests. The weights1,-2
    # describe a zero-mean linear current, not a tuned cancellation.
    a=b=.002;grid=CellGrid(a,b,4,4);h=.001;frequency=3e9
    phi=0.;omega=2pi*frequency;k=omega/DiffMoM._C0
    c,s=cos(Float64(pi/4)),sin(Float64(pi/4));kx=k*s;kc2=kx*kx;k2=(omega/DiffMoM._C0)^2
    eta=sqrt(DiffMoM._MU0/DiffMoM._EPS0);zext=eta*c
    boundary=1e-5
    magnitudes=(eps(Float64),sqrt(eps(Float64)),prevfloat(boundary),boundary,nextfloat(boundary),abs(kc2-k2)*h*h/2)
    caller_bits=precision(BigFloat)
    nodes,weights=DiffMoM.gauss_legendre(40)
    for side in (1,-1),magnitude in magnitudes,direction in (1,-1,im,-im)
        theta=side==1 ? Float64(pi/4) : Float64(3pi/4)
        c,s=abs(cos(theta)),sin(theta);kx=k*s;kc2=kx*kx;zext=eta*c
        qtarget=complex(magnitude*direction)
        epsr=complex(kc2/k2-real(qtarget)/(h*h*k2),-imag(qtarget)/(h*h*k2))
        receiving_layers=[PlanarLayer(epsr,1.,h),PlanarLayer(1.,1.,h)]
        stack=side==1 ? PlanarStackup(receiving_layers,TERM_GND,TERM_SPACE,a,b) :
            PlanarStackup(reverse(receiving_layers),TERM_SPACE,TERM_GND,a,b)
        sheet=sheet_level(1,4,4);rasterize_rect!(sheet,grid,0,a,0,b)
        vias=[via_level(side==1 ? 1 : 2,4,4)];vias[1].uni[2,2]=true;vias[1].tap[2,2]=true
        prob=build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:x,2,2:2,50.)];vias)
        for amplitude in (1.,inv(grid.dx*grid.dy*h))
            coeff=zeros(ComplexF64,length(prob.basis.kind))
            coeff[findfirst(==(DiffMoM._BASIS_VIA_U),prob.basis.kind)]=amplitude
            coeff[findfirst(==(DiffMoM._BASIS_VIA_T),prob.basis.kind)]=-2amplitude
            pattern=planar_farfield(prob,coeff,frequency;theta=[theta],phi=[phi])
            # Independent full Maxwell transfer with a PEC V(0)=0 and
            # exterior V(top)-Zext*H(top)=2*Vinc. Dispersion coefficients
            # are frozen from the stated Float64 model before the reference
            # calculation, isolating moment loss from cutoff subtraction.
            g2s=[complex(kc2)-((omega/DiffMoM._C0)^2*l.epsr*l.mur) for l in receiving_layers]
            actual_q=g2s[1]*h*h
            expected,quadrature_expected=setprecision(BigFloat,2precision(Float64)) do
                M=Matrix{Complex{BigFloat}}(I,2,2)
                for (layer,g2) in zip(receiving_layers,g2s)
                    hh=BigFloat(real(layer.thickness));gg=Complex{BigFloat}(g2)
                    x=sqrt(gg)*hh;sh=iszero(x) ? hh : hh*sinh(x)/x
                    zs=gg/(im*BigFloat(omega)*BigFloat(DiffMoM._EPS0)*Complex{BigFloat}(layer.epsr))
                    ys=im*BigFloat(omega)*BigFloat(DiffMoM._EPS0)*Complex{BigFloat}(layer.epsr)
                    section=Complex{BigFloat}[cosh(x) -zs*sh;-ys*sh cosh(x)]
                    M=section*M
                end
                factor=Complex{BigFloat}(-im*omega*DiffMoM._MU0/(4pi)*cis(side*k*c*(side==1 ? 2h : 0.)))
                vinc=BigFloat(side*c)*factor
                hb=2vinc/(M[1,2]-BigFloat(zext)*M[2,2])
                axial=BigFloat(side*h)*hb*centered_cosh_integral(Complex{BigFloat}(g2s[1])*BigFloat(h)^2)
                sinc0(x)=iszero(x) ? one(x) : sin(x)/x
                xx=BigFloat(kx);dx=BigFloat(grid.dx);dy=BigFloat(grid.dy)
                lateral=dx*dy*sinc0(xx*dx/2)*exp(im*xx*BigFloat(1.5grid.dx))
                prefactor=BigFloat(amplitude)*lateral*BigFloat(side*sqrt(kc2))/(BigFloat(omega)*BigFloat(DiffMoM._EPS0)*Complex{BigFloat}(epsr))
                x=sqrt(Complex{BigFloat}(g2s[1]))*BigFloat(h)
                integral=sum(BigFloat(weights[i])/2*(1-2u)*(cosh(x*u)-1)
                    for i in eachindex(nodes) for u in ((BigFloat(nodes[i])+1)/2,))
                (prefactor*axial,prefactor*BigFloat(side*h)*hb*integral)
            end
            actual=pattern.etheta[1]
            relative=abs((Complex{BigFloat}(actual)-expected)/expected)
            @test isapprox(actual,expected;rtol=5e-11,atol=0)
            @test isapprox(quadrature_expected,expected;rtol=5e-11,atol=0)
            @test pattern.ephi[1]==0
            @test isfinite(pattern.intensity[1]) && pattern.intensity[1]>=0
        end
    end

    # Check the working-type approximation contract with Julia's own
    # BigFloat isapprox default, against doubled-caller-precision exact
    # field integrals. The old fixed cubic loses far more than that default.
    R=BigFloat;hh=R(h);ww=R(omega);layer=PlanarLayer(one(R),one(R),hh)
    for direction in (1,-1,im,-im),pol in (DiffMoM.TE_POL,DiffMoM.TM_POL),field in (:electric,:magnetic)
        qq=complex(R(boundary)/2*direction);g2=qq/(hh*hh);x=sqrt(g2)*hh
        vb=field===:electric ? complex(one(R)) : complex(zero(R))
        hb=field===:magnetic ? complex(one(R)) : complex(zero(R))
        zs=pol==DiffMoM.TE_POL ? im*ww*DiffMoM._MU0 : g2/(im*ww*DiffMoM._EPS0)
        ys=pol==DiffMoM.TM_POL ? im*ww*DiffMoM._EPS0 : g2/(im*ww*DiffMoM._MU0)
        sh=sinh(x)/x;vt=vb*cosh(x)-zs*hh*hb*sh;ht=hb*cosh(x)-ys*hh*vb*sh
        actual=DiffMoM._planar_receive_moments(layer,ww,0.,pol,vb,vt,hb,ht,g2)
        expected=setprecision(BigFloat,2caller_bits) do
            h2=BigFloat(hh);xx=sqrt(Complex{BigFloat}(g2))*h2;q2=xx*xx
            s2=sinh(xx)/xx;c2=2sinh(xx/2)^2/q2;c3=(cosh(xx)-s2)/q2
            v2,b2=Complex{BigFloat}(vb),Complex{BigFloat}(hb)
            zs2,ys2=Complex{BigFloat}(zs),Complex{BigFloat}(ys)
            (v2*s2-zs2*h2*b2*c2,h2*b2*s2-ys2*h2*h2*v2*c2,
             h2*b2*(s2-c2)-ys2*h2*h2*v2*c3)
        end
        for (value,want) in zip(actual,expected)
            @test isapprox(value,want)
        end
    end
    # At exact cutoff the fields are analytic in gamma squared. These
    # slopes are the differentiated Maxwell field integrals at q=0.
    layer0=PlanarLayer(1.,1.,h)
    for pol in (DiffMoM.TE_POL,DiffMoM.TM_POL),field in (:electric,:magnetic),quantity in 1:3,component in (:real,:imag)
        vb=field===:electric ? 1.0+0im : 0.0+0im
        hb=field===:magnetic ? 1.0+0im : 0.0+0im
        part=component===:real ? real : imag
        function sample(t)
            result=DiffMoM._planar_receive_moments(layer0,omega,0.,pol,vb,vb,hb,hb,complex(t,zero(t)))
            part(result[quantity])
        end
        actual=@inferred ForwardDiff.derivative(sample,0.)
        expected=setprecision(BigFloat,2precision(Float64)) do
            hh=BigFloat(h);ww=BigFloat(omega);vv=Complex{BigFloat}(vb);bb=Complex{BigFloat}(hb)
            ze=im*ww*BigFloat(DiffMoM._MU0);ym=im*ww*BigFloat(DiffMoM._EPS0)
            slopes=pol==DiffMoM.TE_POL ?
                (vv*hh^2/6-ze*bb*hh^3/24,bb*hh^3/6-vv*hh^2/(2ze),bb*hh^3/8-vv*hh^2/(3ze)) :
                (vv*hh^2/6-bb*hh/(2ym),bb*hh^3/6-ym*vv*hh^4/24,bb*hh^3/8-ym*vv*hh^4/30)
            part(slopes[quantity])
        end
        @test isfinite(actual)
        @test isapprox(actual,expected)
    end
    @test precision(BigFloat)==caller_bits
end

@testset "Receiving moments retain zero-mean currents and working precision" begin
    run_tests()
end
end
