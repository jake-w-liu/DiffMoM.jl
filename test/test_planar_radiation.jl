using DiffMoM,Test,LinearAlgebra

function radiation_fixture(;bottom=TERM_SPACE,top=TERM_SPACE,via=false,volume=false)
    a=b=.002;grid=CellGrid(a,b,4,4)
    stack=PlanarStackup([PlanarLayer(1.,1.,.001),PlanarLayer(1.,1.,.001)],bottom,top,a,b)
    sheet=sheet_level(1,4,4);rasterize_rect!(sheet,grid,0,a,0,b)
    vias=via ? [via_level(1,4,4)] : ViaLevel[]
    via && (vias[1].uni[2,2]=true;vias[1].tap[2,2]=true)
    vols=volume ? [vol_level(1,4,4)] : VolLevel[]
    volume && rasterize_rect!(vols[1],grid,0,a,0,b)
    return build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:x,2,2:2,50.)];vias,vols)
end

@testset "radiation power metadata fits finite positive Float64" begin
    prob=radiation_fixture();coeff=zeros(ComplexF64,planar_basis_count(prob.basis));coeff[1]=1
    for value in (0.,-1.,Inf,NaN,big"1e500",big"1e-500",1.0im)
        @test_throws ArgumentError planar_farfield(prob,coeff,1e9;theta=[.4],phi=[.2],accepted_power=value)
    end
    valid=planar_farfield(prob,coeff,1e9;theta=[.4],phi=[.2],accepted_power=big"0.5")
    @test valid.accepted_power===.5
    @test eltype(planar_radiation_metrics(valid;radiated_power=big"0.25").directivity)===Float64
    for power in (0.,-1.,Inf,NaN)
        malformed=PlanarRadiationPattern(valid.theta,valid.phi,valid.etheta,valid.ephi,valid.intensity,power,valid.frequency)
        @test_throws ArgumentError planar_radiation_metrics(malformed)
        @test_throws ArgumentError plot_planar_radiation(malformed;quantity=:gain)
    end
    for value in (big"1e500",big"1e-500")
        @test_throws ArgumentError planar_radiation_metrics(valid;radiated_power=value)
        @test_throws ArgumentError plot_planar_radiation(valid;quantity=:directivity,radiated_power=value)
    end
    reject_power()=try
        planar_farfield(prob,coeff,1e9;theta=range(0.,pi;length=100001),phi=[0.],accepted_power=big"1e500")
    catch err
        err isa ArgumentError || rethrow()
        nothing
    end
    reject_power()
    @test (@allocated reject_power())<16000
end

@testset "radiation metrics, plotting, conformal result integration" begin
    # Independent linear/circular polarization ellipse and normalization.
    p=PlanarRadiationPattern([.1,.5,.9],[0.],reshape(ComplexF64[1,1,0],3,1),
        reshape(ComplexF64[0,-im,0],3,1),reshape([.1,.2,0.],3,1),.5,1e9)
    metrics=planar_radiation_metrics(p;radiated_power=.25)
    @test isinf(metrics.axial_ratio[1])
    @test metrics.axial_ratio[2]≈1
    @test isnan(metrics.axial_ratio[3])
    @test metrics.circular_plus[2]≈sqrt(2)
    @test metrics.circular_minus[2]==0
    @test metrics.gain≈4pi.*p.intensity./.5
    @test metrics.directivity≈4pi.*p.intensity./.25
    @test metrics.efficiency==.5
    @test_throws ArgumentError planar_radiation_metrics(p;max_bytes=1)
    @test_throws ArgumentError planar_radiation_metrics(p;radiated_power=0.)
    polar=plot_planar_radiation(p;quantity=:gain)
    @test length(polar.data)==1 && polar.data[1][:theta]≈rad2deg.(p.theta)
    cart=plot_planar_radiation(p;quantity=:etheta,polar=false,db=false)
    @test cart.data[1][:y]==[1.,1.,0.]
    @test_throws ArgumentError plot_planar_radiation(p;quantity=:directivity)
    mktempdir() do dir
        file=joinpath(dir,"radiation.html");save_planar_plot(file,polar)
        @test isfile(file) && occursin("plotly",read(file,String))
        csv=joinpath(dir,"sentinel.csv");write(csv,"keep")
        p.theta[1]=-1
        @test_throws ArgumentError write_planar_radiation_csv(csv,p)
        @test read(csv,String)=="keep"
    end
    # Both genuine triangle solve result types preserve source amplitudes.
    a=b=.002
    stack=PlanarStackup([PlanarLayer(1.,1.,.001),PlanarLayer(1.,1.,.001)],TERM_GND,TERM_SPACE,a,b)
    mesh=PlanarConformalMesh([0. a a 0.;0. 0. b b],[1 1;2 3;3 4];interfaces=1)
    cp=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(0.,b)),PlanarConformalPort(1,:east,(0.,b))])
    dense=solve_planar_conformal(cp,1e9;mx=12,my=12,surface_zs=.1)
    fast=solve_planar_conformal(cp,1e9;method=:ufft,nx=1,ny=1,mx=12,my=12,surface_zs=.1,rtol=1e-9,memory=20)
    pd=planar_farfield(dense;theta=[.4,.7],phi=[.2])
    pf=planar_farfield(fast;theta=[.4,.7],phi=[.2])
    @test pd.etheta≈pf.etheta rtol=1e-8
    @test pd.ephi≈pf.ephi rtol=1e-8
end

@testset "independent analytic current radiation and layer reciprocity" begin
    prob=radiation_fixture();basis=prob.basis;grid=prob.grid
    p=findfirst(b->basis.kind[b]==DiffMoM._BASIS_X_FULL && basis.ei[b]==2 && basis.ej[b]==2,eachindex(basis.kind))
    X=zeros(ComplexF64,planar_basis_count(basis));X[p]=1.2+.4im
    f=2e9;omega=2pi*f;k=omega/299792458.;mu=4pi*1e-7
    thetas=[0.,.3,1.,pi/2,2.,pi];phis=[0.,.7,pi/2]
    pat=planar_farfield(prob,X,f;theta=thetas,phi=phis)
    # Independent triangle and rectangle transform, homogeneous dyadic
    # Green far field; no TL helper or production Fourier routine used.
    sinc0(x)=iszero(x) ? 1. : sin(x)/x
    for (i,t) in enumerate(thetas),(j,ph) in enumerate(phis)
        kx=k*sin(t)*cos(ph);ky=k*sin(t)*sin(ph);kz=k*cos(t)
        jhat=X[p]*grid.dx*grid.dy*sinc0(kx*grid.dx/2)^2*sinc0(ky*grid.dy/2)*
            cis(kx*2grid.dx+ky*1.5grid.dy+kz*.001)
        factor=-im*omega*mu/(4pi)
        @test pat.etheta[i,j]≈factor*cos(t)*cos(ph)*jhat atol=5e-11 rtol=5e-7
        @test pat.ephi[i,j]≈-factor*sin(ph)*jhat atol=2e-11 rtol=5e-7
    end
    # Electrically short dipole power is η*k²*|I*l|²/(12π).
    lowf=1e6;klow=2pi*lowf/299792458.;eta=sqrt(DiffMoM._MU0/DiffMoM._EPS0)
    power=planar_radiated_power(prob,X,lowf;ntheta=8,nphi=16)
    expected=eta*klow^2*abs2(X[p]*grid.dx*grid.dy)/(12pi)
    @test power.power≈expected rtol=2e-10
    @test power.relative_change<1e-12
    @test power.ntheta==16 && power.nphi==32
    # Horizontal electric-current image is antiparallel at the infinite PEC.
    ground=radiation_fixture(bottom=TERM_GND)
    gp=planar_farfield(ground,X,f;theta=[.4,2.],phi=[.2])
    free=planar_farfield(prob,X,f;theta=[.4],phi=[.2])
    image=1-cis(-2k*cos(.4)*.001)
    @test gp.etheta[1]≈free.etheta[1]*image rtol=2e-12
    @test gp.ephi[1]≈free.ephi[1]*image rtol=2e-12
    @test gp.intensity[2]==0 && gp.etheta[2]==0 && gp.ephi[2]==0
    # A two-layer receive problem solved independently by one 4x4 wave
    # coefficient boundary system: Et and H continuity at the interface.
    slab=PlanarStackup([PlanarLayer(3.2-.03im,1.,.003),PlanarLayer(1.6-.01im,1.,.002)],
        TERM_GND,TERM_SPACE,.01,.01)
    kc2=(k*sin(.6))^2
    for pol in (TE_POL,TM_POL)
        gs=[sqrt(complex(kc2-(omega/299792458.)^2*l.epsr*l.mur)) for l in slab.layers]
        zs=[pol==TE_POL ? im*omega*DiffMoM._MU0/gs[j] : gs[j]/(im*omega*DiffMoM._EPS0*slab.layers[j].epsr) for j in 1:2]
        zext=pol==TE_POL ? eta/cos(.6) : eta*cos(.6)
        A=zeros(ComplexF64,4,4);rhs=zeros(ComplexF64,4)
        # Layer fields V(z)=u*exp(-g*z)+d*exp(+g*z),
        # H(z)=(u*exp(-g*z)-d*exp(+g*z))/Z.
        A[1,1:2].=1
        e1=exp(-gs[1]*.003);e2=exp(-gs[2]*.002)
        A[2,:]=[e1,inv(e1),-1,-1]
        A[3,:]=[e1/zs[1],-inv(e1)/zs[1],-1/zs[2],1/zs[2]]
        A[4,3:4]=[e2*(1-zext/zs[2]),inv(e2)*(1+zext/zs[2])];rhs[4]=2
        c=A\rhs
        V=[c[1]+c[2],c[3]+c[4],c[3]*e2+c[4]/e2]
        H=[(c[1]-c[2])/zs[1],(c[3]-c[4])/zs[2],(c[3]*e2-c[4]/e2)/zs[2]]
        vp,hp=DiffMoM._planar_receive_fields(slab,omega,kc2,pol,1.,zext)
        @test vp≈V rtol=2e-12
        @test hp≈H rtol=2e-12
    end
    @test_throws ArgumentError planar_farfield(prob,X,f;theta=[-1.])
    @test_throws ArgumentError planar_farfield(prob,X,f;phi=[Inf])
    @test_throws ArgumentError planar_farfield(prob,X,f;max_bytes=1)
    @test_throws ArgumentError planar_farfield(prob,X,0.)
    @test_throws ArgumentError planar_radiation_stack(prob.stack;top=PlanarTerminator(TERM_SURFACE))
    @test_throws ArgumentError planar_radiation_stack(prob.stack;top=PlanarTerminator{ComplexF64}(TERM_OPEN,0.,1-.1im,1.))
    mktempdir() do dir
        file=joinpath(dir,"radiation.csv");write_planar_radiation_csv(file,pat)
        @test length(readlines(file))==length(thetas)*length(phis)+1
    end
end

@testset "vertical and volume currents and solved power-wave mapping" begin
    prob=radiation_fixture(via=true,volume=true);basis=prob.basis
    f=3e9;omega=2pi*f;k=omega/299792458.;t=.7;ph=.2
    z0,z1=0.,.001;dx=prob.grid.dx;dy=prob.grid.dy
    sincz(x)=iszero(x) ? 1. : sin(x)/x
    for kind in (DiffMoM._BASIS_VIA_U,DiffMoM._BASIS_VIA_T,DiffMoM._BASIS_VX_FULL)
        p=findfirst(==(kind),basis.kind);X=zeros(ComplexF64,length(basis.kind));X[p]=.8+.2im
        pattern=planar_farfield(prob,X,f;theta=[t,pi-t],phi=[ph])
        # High-order independent axial integration of the actual source.
        q,w=DiffMoM.gauss_legendre(40)
        for (i,angle) in enumerate((t,pi-t))
            kx=k*sin(angle)*cos(ph);ky=k*sin(angle)*sin(ph);kz=k*cos(angle)
            axial=sum(w[j]/2*cis(kz*(q[j]+1)*.001/2)*
                (kind==DiffMoM._BASIS_VIA_T ? (q[j]+1)/2 : 1.) for j in eachindex(q))
            if kind in (DiffMoM._BASIS_VIA_U,DiffMoM._BASIS_VIA_T)
                lateral=dx*dy*sincz(kx*dx/2)*sincz(ky*dy/2)*cis(kx*1.5dx+ky*1.5dy)
                integral=X[p]*lateral*.001*axial*(-sin(angle))
            else
                lateral=dx*dy*sincz(kx*dx/2)^2*sincz(ky*dy/2)*cis(kx*basis.ei[p]*dx+ky*(basis.ej[p]-.5)*dy)
                integral=X[p]*lateral*axial*cos(angle)*cos(ph)
            end
            @test pattern.etheta[i]≈-im*omega*DiffMoM._MU0/(4pi)*integral rtol=5e-11
            kind in (DiffMoM._BASIS_VIA_U,DiffMoM._BASIS_VIA_T) && @test pattern.ephi[i]==0
        end
    end
    source=radiation_fixture(bottom=TERM_GND)
    raw=solve_planar(source,1e9;mx=16,my=16,surface_zs=.1)
    a=[1. + .2im];v=sqrt(50.)*(a+raw.s*a)
    fromwaves=planar_farfield(raw;incident_waves=a,theta=[.4,.8],phi=[.2])
    explicit=planar_farfield(source,raw.currents*v,1e9;theta=[.4,.8],phi=[.2])
    @test fromwaves.etheta≈explicit.etheta rtol=1e-13
    @test fromwaves.accepted_power≈.5*(sum(abs2,a)-sum(abs2,raw.s*a)) rtol=2e-12
    @test_throws ArgumentError planar_farfield(raw;max_bytes=1)
end
