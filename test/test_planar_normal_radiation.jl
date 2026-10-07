using DiffMoM, Test, LinearAlgebra

let
function _source15_normal_fixture(;bottom=TERM_SPACE,top=TERM_SPACE)
 a=b=.002;grid=CellGrid(a,b,4,4)
 stack=PlanarStackup([PlanarLayer(1.,1.,.001),PlanarLayer(1.,1.,.001)],bottom,top,a,b)
 sheet=sheet_level(1,4,4);rasterize_rect!(sheet,grid,0,a,0,b)
 vias=[via_level(1,4,4)];vias[1].uni[2,2]=true;vias[1].tap[2,2]=true
 build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:x,2,2:2,50.)];vias)
end
 @testset "Exact axial radiation preserves physical nulls near-axis sine and polarization" begin
  for bottom in (TERM_SPACE,TERM_GND),kind in (DiffMoM._BASIS_VIA_U,DiffMoM._BASIS_VIA_T)
   prob=_source15_normal_fixture(;bottom);coeff=zeros(ComplexF64,length(prob.basis.kind));coeff[findfirst(==(kind),prob.basis.kind)]=.8+.2im
   pattern=planar_farfield(prob,coeff,3e9;theta=[0.,pi],phi=[0.,.2,pi/2])
   @test all(iszero,pattern.etheta)
   @test all(iszero,pattern.ephi)
   @test all(iszero,pattern.intensity)
  end
  prob=_source15_normal_fixture();f=3e9;omega=2pi*f;k=omega/299792458.;q,w=DiffMoM.gauss_legendre(40)
  sinc0(x)=iszero(x) ? 1. : sin(x)/x
  for kind in (DiffMoM._BASIS_VIA_U,DiffMoM._BASIS_VIA_T)
   coeff=zeros(ComplexF64,length(prob.basis.kind));coeff[findfirst(==(kind),prob.basis.kind)]=.8+.2im
   # The additional angles straddle the smallest representable square,
   # where a linear projection must use the norm before squaring.
   for theta in (eps(Float64)^2,eps(Float64),sqrt(eps(Float64)),prevfloat(Float64(pi)),
           sqrt(nextfloat(0.))/k/2,sqrt(nextfloat(0.))/k,sqrt(floatmin(Float64))/k),phi in (0.,.2,pi/2)
    pattern=planar_farfield(prob,coeff,f;theta=[theta],phi=[phi]);kx=k*sin(theta)*cos(phi);ky=k*sin(theta)*sin(phi);kz=k*cos(theta)
    axial=sum(w[j]/2*cis(kz*(q[j]+1)*.001/2)*(kind==DiffMoM._BASIS_VIA_T ? (q[j]+1)/2 : 1.) for j in eachindex(q))
    lateral=prob.grid.dx*prob.grid.dy*sinc0(kx*prob.grid.dx/2)*sinc0(ky*prob.grid.dy/2)*cis(kx*1.5prob.grid.dx+ky*1.5prob.grid.dy)
    expected=-im*omega*DiffMoM._MU0/(4pi)*(.8+.2im)*lateral*.001*axial*(-sin(theta))
    @test pattern.etheta[1]≈expected rtol=5e-11 atol=0
    @test pattern.ephi[1]==0
   end
  end
  for kind in (DiffMoM._BASIS_X_FULL,DiffMoM._BASIS_Y_FULL)
   coeff=zeros(ComplexF64,length(prob.basis.kind));coeff[findfirst(==(kind),prob.basis.kind)]=.8+.2im
   for theta in (0.,pi)
    phis=[0.,.2,pi/2];pattern=planar_farfield(prob,coeff,f;theta=[theta],phi=phis);et0,ep0=pattern.etheta[1,1],pattern.ephi[1,1];side=iszero(theta) ? 1 : -1
    for (j,phi) in enumerate(phis)
     @test pattern.etheta[1,j]≈et0*cos(phi)+side*ep0*sin(phi) rtol=5e-11 atol=0
     @test pattern.ephi[1,j]≈-side*et0*sin(phi)+ep0*cos(phi) rtol=5e-11 atol=0
    end
   end
  end
 end
end
