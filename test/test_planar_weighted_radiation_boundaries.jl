module WeightedRadiationBoundaryTests
using DiffMoM,Test
function fixture(;bottom=TERM_SPACE,bulk=:via)
    a=b=.002;grid=CellGrid(a,b,4,4)
    stack=PlanarStackup([PlanarLayer(1.,1.,.001),PlanarLayer(1.,1.,.001)],bottom,TERM_SPACE,a,b)
    sheet=sheet_level(1,4,4);rasterize_rect!(sheet,grid,0,a,0,b)
    if bulk==:via
        vias=[via_level(1,4,4)];vias[1].uni[2,2]=true;vias[1].tap[2,2]=true
        return build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:x,2,2:2,50.)];vias)
    elseif bulk==:volume
        vols=[vol_level(1,4,4)];rasterize_rect!(vols[1],grid,0,a,0,b)
        return build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:x,2,2:2,50.)];vols)
    end
    return build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:x,2,2:2,50.)])
end
sinc0(x)=iszero(x) ? 1. : sin(x)/x
function run_tests()
    # Reuse the original physical radiation fixture, independent 40-point
    # axial integration, and its existing field tolerances. Amplitudes and
    # boundary angles follow basis volume and Float64 representation.
    prob=fixture();grid=prob.grid;f=3e9;omega=2pi*f;k=omega/299792458.
    q,w=DiffMoM.gauss_legendre(40)
    angles=(nextfloat(0.),floatmin(Float64),sqrt(nextfloat(0.))/k/2)
    amplitudes=(inv(grid.dx*grid.dy*.001),inv(grid.dx*grid.dy*.001)/sqrt(eps(Float64)),inv(sqrt(floatmin(Float64))))
    @testset "Weighted axial fields retain amplified representable boundary values" begin
        for amplitude in amplitudes,kind in (DiffMoM._BASIS_VIA_U,DiffMoM._BASIS_VIA_T),theta in angles,phi in (0.,.2,pi/2)
            coeff=zeros(ComplexF64,length(prob.basis.kind));coeff[findfirst(==(kind),prob.basis.kind)]=amplitude*(.8+.2im)
            pattern=planar_farfield(prob,coeff,f;theta=[theta],phi=[phi])
            kx=k*sin(theta)*cos(phi);ky=k*sin(theta)*sin(phi);kz=k*cos(theta)
            axial=sum(w[j]/2*cis(kz*(q[j]+1)*.001/2)*(kind==DiffMoM._BASIS_VIA_T ? (q[j]+1)/2 : 1.) for j in eachindex(q))
            lateral=grid.dx*grid.dy*sinc0(kx*grid.dx/2)*sinc0(ky*grid.dy/2)*cis(kx*1.5grid.dx+ky*1.5grid.dy)
            expected=-im*omega*DiffMoM._MU0/(4pi)*(amplitude*(.8+.2im))*lateral*.001*axial*(-sin(theta))
            @test pattern.etheta[1]≈expected rtol=5e-11 atol=0
            @test pattern.ephi[1]==0
            @test isfinite(pattern.intensity[1]) && pattern.intensity[1]>=0
        end
    end
    @testset "Grazing observation preserves the physical sheet-current angle" begin
        prob=fixture(;bulk=:sheet);grid=prob.grid;basis=prob.basis;f=2e9;omega=2pi*f;k=omega/299792458.
        angles=(prevfloat(Float64(pi/2)),Float64(pi/2),nextfloat(Float64(pi/2)),pi/2-sqrt(eps(Float64)),pi/2+sqrt(eps(Float64)))
        for amplitude in (1.,inv(sqrt(eps(Float64)))),theta in angles
            p=findfirst(b->basis.kind[b]==DiffMoM._BASIS_X_FULL && basis.ei[b]==2 && basis.ej[b]==2,eachindex(basis.kind))
            coeff=zeros(ComplexF64,length(basis.kind));coeff[p]=amplitude*(1.2+.4im)
            pattern=planar_farfield(prob,coeff,f;theta=[theta],phi=[0.]);kx=k*sin(theta);kz=k*cos(theta)
            jhat=coeff[p]*grid.dx*grid.dy*sinc0(kx*grid.dx/2)^2*cis(kx*2grid.dx+kz*.001)
            expected=-im*omega*DiffMoM._MU0/(4pi)*cos(theta)*jhat
            @test pattern.etheta[1]≈expected rtol=5e-7 atol=5e-11
        end
    end
    @testset "PEC volume image retains matched axial dispersion near grazing" begin
        prob=fixture(;bottom=TERM_GND,bulk=:volume);grid=prob.grid;basis=prob.basis
        p=findfirst(==(DiffMoM._BASIS_VX_FULL),basis.kind);coeff=zeros(ComplexF64,length(basis.kind));coeff[p]=.8+.2im
        f=3e9;omega=2pi*f;k=omega/299792458.
        for theta in (.7,pi/2-sqrt(eps(Float64)),prevfloat(Float64(pi/2)),Float64(pi/2)),phi in (0.,.2)
            pattern=planar_farfield(prob,coeff,f;theta=[theta],phi=[phi]);kx=k*sin(theta)*cos(phi);ky=k*sin(theta)*sin(phi);kz=k*cos(theta)
            axial=sum(w[j]/2*(2im*sin(kz*(q[j]+1)*.001/2)) for j in eachindex(q))
            lateral=grid.dx*grid.dy*sinc0(kx*grid.dx/2)^2*sinc0(ky*grid.dy/2)*cis(kx*basis.ei[p]*grid.dx+ky*(basis.ej[p]-.5)*grid.dy)
            expected=-im*omega*DiffMoM._MU0/(4pi)*coeff[p]*lateral*axial*cos(theta)*cos(phi)
            @test pattern.etheta[1]≈expected rtol=5e-11 atol=0
        end
    end
end
run_tests()
end
