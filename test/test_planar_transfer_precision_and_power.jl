module TransferPrecisionAndPowerTests
using DiffMoM,LinearAlgebra,Test

@testset "Convergent transfer preserves layer precision, conditioned moments and positive power" begin
    caller_bits=precision(BigFloat)
    @testset "Decaying Maxwell integrals before the former representation switch" begin
        layer=PlanarLayer(1.0,1.0,0.001);h=real(layer.thickness);omega=2pi*3e9
        boundary=20.0
        for requested in (2.0,4.0,8.0,16.0,prevfloat(boundary),boundary,nextfloat(boundary),32.0),
            pol in (DiffMoM.TE_POL,DiffMoM.TM_POL)
            g2=complex((requested/h)^2)
            zs=pol==DiffMoM.TE_POL ? im*omega*DiffMoM._MU0 : g2/(im*omega*DiffMoM._EPS0)
            hb=1.0+0im;vb=zs*h/requested;ht=exp(-requested)*hb;vt=exp(-requested)*vb
            actual=DiffMoM._planar_receive_moments(layer,omega,0.0,pol,vb,vt,hb,ht,g2)
            expected=setprecision(BigFloat,2precision(Float64)) do
                x=sqrt(Complex{BigFloat}(g2))*BigFloat(h);factor=-expm1(-x)/x
                (Complex{BigFloat}(vb)*factor,BigFloat(h)*factor,
                 BigFloat(h)*(-expm1(-x)-x*exp(-x))/(x*x))
            end
            for (value,want) in zip(actual,expected)
                @test isapprox(value,ComplexF64(want);rtol=5e-11,atol=0)
            end
        end
    end
    @testset "Public grounded cascade retains working precision and unit boundary" begin
        for R in (Float64,BigFloat),target in (R(eps(Float64)),R(sqrt(eps(Float64))),prevfloat(one(R)),one(R),nextfloat(one(R)),R(4),R(16)),
            direction in (:evanescent,:propagating),pol in (DiffMoM.TE_POL,DiffMoM.TM_POL)
            h=R(0.001);layer=PlanarLayer(one(R),one(R),h)
            omega=direction===:propagating ? sqrt(target)*R(DiffMoM._C0)/h : R(2pi)*R(3e9)
            kc2=direction===:propagating ? 0.0 : Float64((omega/R(DiffMoM._C0))^2+target/(h*h))
            stack=PlanarStackup([layer],TERM_GND,TERM_SPACE,0.002,0.002)
            actual=planar_mode_cascade(stack,omega,kc2,pol).zdn[2]
            g2=DiffMoM._planar_gamma2_layer(pol,kc2,omega,layer)
            expected=setprecision(BigFloat,2precision(R)) do
                gg=Complex{BigFloat}(g2);hh=BigFloat(h);ww=BigFloat(omega);x=sqrt(gg)*hh
                zs=pol==DiffMoM.TE_POL ? im*ww*BigFloat(DiffMoM._MU0) : gg/(im*ww*BigFloat(DiffMoM._EPS0))
                zs*hh*tanh(x)/x
            end
            @test isapprox(actual,Complex{R}(expected))
        end
        omega=2pi*3e9;h=0.001;k2=(omega/DiffMoM._C0)^2
        for pol in (DiffMoM.TE_POL,DiffMoM.TM_POL)
            epsilon=DiffMoM._PlanarDual(1.0+0im,1.0+0im)
            stack=PlanarStackup([PlanarLayer(epsilon,1.0,h)],TERM_GND,DiffMoM.PlanarTerminator(TERM_PMC),0.002,0.002)
            value=planar_mode_cascade(stack,omega,k2,pol).zdn[2]
            expected=pol==DiffMoM.TE_POL ? im*omega*DiffMoM._MU0*h^3*k2/3 : -k2*h/(im*omega*DiffMoM._EPS0)
            @test isfinite(value.d)
            @test isapprox(value.d,expected)
        end
    end
    @testset "Caller-retained admittance keeps exactly positive accepted power" begin
        a=b=0.002;grid=CellGrid(a,b,4,4)
        stack=PlanarStackup([PlanarLayer(1.0,1.0,0.001),PlanarLayer(1.0,1.0,0.001)],TERM_SPACE,TERM_SPACE,a,b)
        sheet=sheet_level(1,4,4);rasterize_rect!(sheet,grid,0,a,0,b)
        prob=build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:x,2,2:2,50.0)])
        nb=planar_basis_count(prob.basis);frequency=3e9
        pbs=DiffMoM._port_basis_indices(prob.basis,1)
        sgn=[DiffMoM._planar_port_sign(prob.ports[1])]
        rhs=DiffMoM._planar_dense_source_rhs!(zeros(ComplexF64,nb,1),prob)
        for g in (sqrt(eps(Float64)),eps(Float64),eps(Float64)^2)
            # Coherent caller-retained linear model and actual port extraction.
            # The fixture is not an assembled electromagnetic solution.
            Z=Matrix{ComplexF64}(I,nb,nb)
            for p in pbs[1];Z[p,p]=rhs[p,1]/complex(g,1.0);end
            F=lu(Z);X=F\rhs;Y=DiffMoM._planar_dense_port_y(prob,pbs,sgn,X)
            S=planar_y_to_s(Y,[50.0]);wanted=real(Y[1])/2
            @test norm(Z*X-rhs)/norm(rhs)<=1e-9
            @test wanted>0
            result=PlanarResult(prob,complex(2pi*frequency),complex(frequency),Z,F,X,Y,S)
            pattern=planar_farfield(result;theta=[Float64(pi/4)],phi=[0.0])
            direct=planar_farfield(prob,X[:,1],frequency;theta=[Float64(pi/4)],phi=[0.0],accepted_power=wanted)
            @test pattern.accepted_power===wanted
            @test pattern.etheta==direct.etheta && pattern.ephi==direct.ephi
        end
    end
    @test precision(BigFloat)==caller_bits
end
end
