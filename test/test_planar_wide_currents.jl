module PlanarWideCurrentTests
using DiffMoM,Test,LinearAlgebra
@test unsafe_string(ccall((:mpfr_print_rnd_mode,DiffMoM._planar_mpfr_library),Cstring,(Cint,),DiffMoM._planar_mpfr_nearest))=="MPFR_RNDN"
const FIXTURE=joinpath(@__DIR__,"fixtures","native_volume_sheet_selector","triangle_hollow")
project()=read_sonnet_project(joinpath(FIXTURE,"project.son"))
function residuals(raw,X=raw.currents)
 basis=raw.problem.basis;rhs=zeros(ComplexF64,size(X))
 weights=[basis.width[b]*(basis.kind[b]==DiffMoM._BASIS_VIA_T ? .5 : 1.) for b in axes(X,1)]
 for b in axes(rhs,1)
  q=basis.port[b];q==0 && continue
  port=raw.problem.ports[q]
  sign=port.polarity*(port.wall in (:east,:north,:volume_east,:volume_north,:terminal_x_hi,:terminal_y_hi) ? -1. : 1.)
  rhs[b,q]=-sign*weights[b]
 end
 setprecision(BigFloat,256) do
  a=Complex{BigFloat}.(raw.z_mom);x=Complex{BigFloat}.(X);source=Complex{BigFloat}.(rhs);w=BigFloat.(weights)
  [Float64(norm((a*x[:,q]-source[:,q])./w)/norm(source[:,q]./w)) for q in axes(X,2)]
 end
end
@testset "wide dense currents retain native low-frequency physical reactions" begin
 reference=planar_read_touchstone(joinpath(FIXTURE,"native_raw.s2p"))
 for method in (:dense,:dense_fft),frequency in (1e6,1e7,1e8)
  result=solve_sonnet_project(project(),frequency;raw=true,grid=(20,20),mx=40,my=40,method)
  raw=result.raw
  compact=solve_sonnet_project(project(),frequency;raw=true,grid=(20,20),mx=40,my=40,method,retain_matrix=false).raw
  native=reference.s[only(findall(==(frequency),reference.frequencies))]
  @test maximum(abs,result.s-native)<=.06
  @test abs(real(100result.s[1,1]/result.s[2,1])/real(100native[1,1]/native[2,1])-1)<=.05
  @test maximum(residuals(raw))<=1e-9
  @test maximum(residuals(raw,compact.currents))<=1e-9
  @test maximum(raw.relative_residuals)<=1e-9
  @test maximum(compact.relative_residuals)<=1e-9
  @test maximum(svdvals(result.s))<=1+1e-9
  @test maximum(abs,result.s-transpose(result.s))<=1e-9
  @test compact.z_mom===nothing
  @test raw.z_mom!==raw.lu_fact.factors
  @test compact.currents ≈ raw.currents rtol=2e-14
  @test compact.s ≈ raw.s rtol=2e-14
  @test eltype(raw.currents)==Complex{BigFloat}
  @test all(x->precision(real(x))==128 && precision(imag(x))==128,raw.currents)
 end
end
@testset "wide result constructors and current-map cancellation" begin
 r=solve_sonnet_project(project(),1e6;raw=true,grid=(20,20),mx=40,my=40,method=:dense).raw
 saved=deepcopy(r.currents);nb,np=size(r.currents)
 realx=real.(ComplexF64.(r.currents))
 ctor=PlanarResult(r.problem,r.omega,r.freq,r.z_mom,r.lu_fact,realx,r.y,r.s,r.z0)
 @test ctor.currents==ComplexF64.(realx)
 ctor=PlanarResult(r.problem,r.omega,r.freq,r.z_mom,r.lu_fact,view(r.currents,:,:),r.y,r.s,r.z0)
 @test ctor.currents==r.currents && eltype(ctor.currents)==Complex{BigFloat}
 @test ctor.currents!==r.currents
 artificial=setprecision(BigFloat,256) do
  z=zeros(Complex{BigFloat},nb,np);a=BigFloat(2)^100;z[1,1]=a+1;z[1,2]=-a;z
 end
 witness=PlanarResult(r.problem,r.omega,r.freq,r.z_mom,r.lu_fact,artificial,r.y,r.s,r.z0)
 maps=planar_current_maps(witness;voltages=ones(np))
 unit=zeros(ComplexF64,nb);unit[1]=1
 expected=DiffMoM._planar_current_maps_from_coefficients(r.problem,unit)
 @test all(all(getfield(a,k)==getfield(b,k) for k in (:jx,:jy,:jz)) for (a,b) in zip(maps,expected))
 values=Float64[]
 for bits in (32,256,8192),mode in (RoundNearest,RoundUp,RoundDown)
  before=(precision(BigFloat),rounding(BigFloat))
  value=setprecision(BigFloat,bits) do
   setrounding(BigFloat,mode) do
    maps=planar_current_maps(r;voltages=[1+im,2-im])
    @test precision(BigFloat)==bits && rounding(BigFloat)==mode
    sum(sum(abs,m.jx)+sum(abs,m.jy)+sum(abs,m.jz) for m in maps)
   end
  end
  @test (precision(BigFloat),rounding(BigFloat))==before
  push!(values,value)
 end
 @test all(==(first(values)),values)
 @test r.currents==saved
 prob=r.problem;nl=length(prob.sheets)+length(prob.vols)+length(prob.vias);grid=prob.grid
 scalar=DiffMoM._planar_wide_scalar_payload(128)
 # Sizes follow the coefficient/map representations and their explicit
 # four-scalar product or shared-zero/weight/product operation workspaces.
 complex_bytes=2scalar+sizeof(Complex{BigFloat})
 inner=DiffMoM._planar_current_reconstruction_payload(prob)+sizeof(ComplexF64)*nb+complex_bytes*3nl*grid.nx*grid.ny+4scalar
 budget=inner+10sizeof(ComplexF64)*np+complex_bytes*nb+4scalar
 @test_throws ArgumentError planar_current_maps(r;max_bytes=budget-1)
 @test length(planar_current_maps(r;max_bytes=budget))==nl
 @test_throws ArgumentError planar_current_maps(r;max_bytes=1)
 rad=planar_radiation_stack(prob.stack;top=TERM_SPACE)
 expectedfield=planar_farfield(prob,unit,1e6;theta=[.31],phi=[.49],radiation_stack=rad)
 for bits in (32,256,8192)
  field=setprecision(BigFloat,bits) do
   planar_farfield(witness;voltages=ones(np),theta=[.31],phi=[.49],radiation_stack=rad)
  end
  @test field.etheta ≈ expectedfield.etheta rtol=2e-14
  @test field.ephi ≈ expectedfield.ephi rtol=2e-14
 end
end
@testset "wide port contraction precision, ownership and concurrent calls" begin
 r=solve_sonnet_project(project(),1e6;raw=true,grid=(20,20),mx=40,my=40,method=:dense).raw
 nb,np=size(r.currents);C=[1. .25;-.5 1.25];saved=deepcopy(r.currents)
 scalar=DiffMoM._planar_wide_scalar_payload(128)
 budget=DiffMoM._planar_contract_payload(nb,np,np)+(2scalar+sizeof(Complex{BigFloat}))*nb*np+4scalar
 @test_throws ArgumentError planar_contract_ports(r,C;max_bytes=budget-1)
 @test_throws ArgumentError planar_contract_ports(r,ones(2,2))
 @test_throws ArgumentError planar_contract_ports(r,[1. NaN;0. 1.])
 expected=setprecision(BigFloat,512) do
  Complex{BigFloat}.(r.currents)*BigFloat.(C)
 end
 for bits in (32,256,8192),mode in (RoundNearest,RoundUp,RoundDown)
  setprecision(BigFloat,bits) do
   setrounding(BigFloat,mode) do
    a=planar_contract_ports(r,C;z0=r.z0,max_bytes=budget)
    @test precision(BigFloat)==bits && rounding(BigFloat)==mode
    @test eltype(a.currents)==Complex{BigFloat}
    @test a.y ≈ transpose(C)*r.y*C rtol=2e-14
    @test all(x->precision(real(x))==128 && precision(imag(x))==128,a.currents)
    @test Float64(norm(a.currents-expected)/norm(expected))<1e-35
   end
  end
  @test r.currents==saved
 end
 identity=Matrix{Float64}(I,np,np)
 for original in (r,planar_reference_planes(r))
  a=planar_contract_ports(original,identity;z0=r.z0)
  @test eltype(a.currents)==Complex{BigFloat}
  @test maximum(residuals(r,a.currents))<=1e-9
 end
 tasks=[Threads.@spawn setprecision(BigFloat,bits) do
  setrounding(BigFloat,mode) do
   observations=Tuple{Bool,Bool}[]
   for _ in 1:8
    a=planar_contract_ports(r,C;max_bytes=budget)
    accurate=Float64(norm(a.currents-expected)/norm(expected))<1e-35
    context=precision(BigFloat)==bits && rounding(BigFloat)==mode
    push!(observations,(accurate,context))
    yield()
   end
   observations
  end
 end for (bits,mode) in ((32,RoundUp),(8192,RoundDown),(256,RoundNearest),(64,RoundUp))]
 results=map(fetch,tasks)
 @test length(results)==4
 @test all(length(a)==8 for a in results)
 @test all(all(all(t) for t in a) for a in results)
 @test r.currents==saved
end
@testset "wide factor coarse native fixture and gradient precision" begin
 path=joinpath(@__DIR__,"fixtures","native_box_port_attachment","cases","attachment__baseline","project.son")
 model=sonnet_planar_problem(read_sonnet_project(path);freq=1e9,grid=(8,8),_materials=true,_details=true)
 prob=model.problem;kw=(mx=8,my=8,surface_zs=model.sheet_zs,via_sigma=model.via_sigma)
 raw=solve_planar(prob,1e9;kw...)
 compact=solve_planar(prob,1e9;kw...,retain_matrix=false)
 @test eltype(raw.lu_fact.factors)==Complex{BigFloat}
 @test eltype(raw.currents)==Complex{BigFloat}
 @test all(x->precision(real(x))==128 && precision(imag(x))==128,raw.lu_fact.factors)
 @test raw.z_mom!==raw.lu_fact.factors
 @test compact.z_mom===nothing
 @test compact.currents ≈ raw.currents rtol=2e-14
 @test compact.s ≈ raw.s rtol=2e-14
 @test maximum(residuals(raw))<=1e-9
 @test maximum(raw.relative_residuals)<=1e-9
 params=[PlanarParam(1,:epsr,:re)]
 objective=Y->imag(Y[1,1]);gY=Y->ComplexF64[-im 0;0 0]
 baseline=planar_objective_gradient(prob,1e9,objective;params,gY,kw...)
 @test all(isfinite,baseline[2])
 for (bits,mode) in ((32,RoundUp),(256,RoundNearest),(8192,RoundDown))
  before=(precision(BigFloat),rounding(BigFloat))
  result=setprecision(BigFloat,bits) do
   setrounding(BigFloat,mode) do
    planar_objective_gradient(prob,1e9,objective;params,gY,kw...)
   end
  end
  @test result[1]==baseline[1]
  @test result[2]==baseline[2]
  @test before==(precision(BigFloat),rounding(BigFloat))
 end
end
@testset "wide adjoints preserve the original transposed equation" begin
 for method in (:dense,:dense_fft),retain_matrix in (true,false),bits in (32,256)
  model=sonnet_planar_problem(project();freq=1e6,grid=(20,20),_materials=true,_details=true)
  prob=model.problem;kw=(mx=40,my=40,method,retain_matrix,surface_zs=model.sheet_zs,via_sigma=model.via_sigma)
  raw=solve_planar(prob,1e6;kw...)
  Z=retain_matrix ? raw.z_mom : solve_planar(prob,1e6;kw...,retain_matrix=true).z_mom
  pbs=DiffMoM._port_basis_indices(prob.basis,length(prob.ports));sgn=[DiffMoM._planar_port_sign(p) for p in prob.ports]
  before=(precision(BigFloat),rounding(BigFloat))
  adj=setprecision(BigFloat,bits) do
   setrounding(BigFloat,RoundUp) do
    DiffMoM._planar_gradient_checked_adjoint(raw,prob,pbs,sgn,0,128,2^30,kw)
   end
  end
  rhs=DiffMoM._planar_gradient_adjoint_rhs(prob,pbs,sgn)
  residual=setprecision(BigFloat,512) do
   A=transpose(Complex{BigFloat}.(Z));X=Complex{BigFloat}.(adj);B=Complex{BigFloat}.(rhs)
   weights=BigFloat.([DiffMoM._planar_port_weight(prob.basis,b) for b in axes(rhs,1)])
   maximum(Float64(norm((A*X[:,q]-B[:,q])./weights)/norm(B[:,q]./weights)) for q in axes(X,2))
  end
  @test residual<=1e-9
  @test all(z->precision(real(z))==128 && precision(imag(z))==128,adj)
  @test before==(precision(BigFloat),rounding(BigFloat))
 end
 model=sonnet_planar_problem(project();freq=1e6,grid=(20,20),_materials=true,_details=true)
 prob=model.problem;calls=Ref(0);objective=Y->(calls[]+=1;imag(Y[1,1]))
 @test_throws ArgumentError planar_objective_gradient(prob,1e6,objective;params=[PlanarParam(1,:epsr,:re)],max_bytes=1,mx=40,my=40,surface_zs=model.sheet_zs,via_sigma=model.via_sigma)
 @test calls[]==0
end
@testset "wide gradient public allocation and callback preflight" begin
 model=sonnet_planar_problem(project();freq=1e6,grid=(20,20),_materials=true,_details=true)
 prob=model.problem;params=[PlanarParam(1,:epsr,:re)]
 calls=Ref(0);objective=Y->(calls[]+=1;imag(Y[1,1]));gY=Y->ComplexF64[-im 0;0 0]
 kw=(params,gY,mx=40,my=40,surface_zs=model.sheet_zs,via_sigma=model.via_sigma)
 # One physical MoM matrix alone is a required subset of this operation.
 # A budget limited to that subset must reject before either callback.
 mandatory_matrix=sizeof(ComplexF64)*planar_basis_count(prob.basis)^2
 @test_throws ArgumentError planar_objective_gradient(prob,1e6,objective;kw...,max_bytes=mandatory_matrix)
 @test calls[]==0
 evaluate()=planar_objective_gradient(prob,1e6,objective;kw...,max_bytes=2^24)
 smaller=planar_objective_gradient(prob,1e6,objective;kw...,max_bytes=2^23)
 @test all(isfinite,smaller[2])
 @test calls[]==1
 warm=evaluate();measured=@timed evaluate()
 @test measured.value==warm
 @test measured.value==smaller
 @test all(isfinite,measured.value[2])
 # Matched pre-fix warmed runs allocate 4.559 GB; current measured runs
 # allocate 10.4 MB on Julia 1.12/1.13. Keep a portable 32 MiB bound.
 @test measured.bytes<=2^25
end
end

module PlanarProjectWideRadiationTests
using DiffMoM,LinearAlgebra,Test
@testset "project radiation retains owned currents and measured staging bounds" begin
    project=load_planar_project(joinpath(@__DIR__,"..","examples","planar_project_line.toml"))
    original=solve_planar_project(project,1e9;mx=16,my=12)
    prob=DiffMoM._planar_radiation_problem(original.em)
    nb=planar_basis_count(prob.basis);np=length(original.z0)
    wide=setprecision(BigFloat,256) do
        X=zeros(Complex{BigFloat},nb,np)
        # 101 significant bits exceed Float64's53 and fit the owned256.
        huge=BigFloat(2)^100;X[1,1]=huge+1;X[1,2]=-huge;X
    end
    saved=deepcopy(wide)
    em=PlanarResult(prob,ComplexF64(2pi*1e9),ComplexF64(1e9),nothing,
        lu(Matrix{ComplexF64}(I,nb,nb)),wide,zeros(ComplexF64,np,np),zeros(ComplexF64,np,np),original.z0)
    witness=PlanarProjectResult(project,original.model,em,nothing,1e9,
        original.port_names,original.z0,zeros(ComplexF64,np,np),zeros(ComplexF64,np,np))
    unit=zeros(ComplexF64,nb);unit[1]=1
    radiation=planar_radiation_stack(prob.stack;top=TERM_SPACE)
    expected=planar_farfield(prob,unit,1e9;theta=[.31],phi=[.49],radiation_stack=radiation)
    @test norm(expected.etheta)+norm(expected.ephi)>0
    for bits in (32,256,8192)
        before=(precision(BigFloat),rounding(BigFloat))
        setprecision(BigFloat,bits) do
            excitation=DiffMoM._project_radiation_excitation(witness;voltages=ones(np),max_bytes=2^30)
            @test excitation.coefficients[1]==1
            @test all(iszero,excitation.coefficients[2:end])
            @test all(x->precision(real(x))==256 && precision(imag(x))==256,excitation.coefficients)
            output_bytes=Base.summarysize(excitation.coefficients)
            @test_throws ArgumentError DiffMoM._project_radiation_excitation(witness;
                voltages=ones(np),max_bytes=output_bytes-sizeof(excitation.coefficients))
            actual=planar_farfield(witness;voltages=ones(np),theta=[.31],phi=[.49],radiation_stack=radiation)
            @test actual.etheta==expected.etheta && actual.ephi==expected.ephi
        end
        @test before==(precision(BigFloat),rounding(BigFloat))
    end
    @test witness.em.currents==saved
end
end

module PlanarOwnedWideFactorTests
using DiffMoM,LinearAlgebra,Test
@testset "owned wide factors preserve exact rational equations and bounded arithmetic" begin
Q=Rational{BigInt};exact(A)=Complex{Q}.(A)
squarednorm(A)=sum(abs2,A)
A0=ComplexF64[0 2+im -3;1+2im -4 2;3-im 1 1+im]
scaled=ComplexF64[0 0 ldexp(1.,-500);ldexp(1.,500) 0 0;0 1 0]
B0=ComplexF64[1+im -2;2-im 1;3 1+2im]
for (name,original) in (("nonsymmetric-complex-pivots",A0),("permuted-extreme-dyadic-scales",scaled)),bits in (128,256),caller in (32,8192)
    A,B=setprecision(BigFloat,bits) do
        setrounding(BigFloat,RoundNearest) do
            Complex{BigFloat}.(original),Complex{BigFloat}.(B0)
        end
    end
    scratch=DiffMoM._planar_wide_factor_scratch(bits)
    before=(precision(BigFloat),rounding(BigFloat))
    factor,solution,transposed=setprecision(BigFloat,caller) do
        setrounding(BigFloat,RoundUp) do
            factor=DiffMoM._planar_owned_wide_lu!(A,scratch)
            solution=DiffMoM._planar_owned_wide_factor_solve!(factor,deepcopy(B),scratch)
            transposed=DiffMoM._planar_owned_wide_factor_solve!(factor,deepcopy(B),scratch;transposed=true)
            factor,solution,transposed
        end
    end
    @test before==(precision(BigFloat),rounding(BigFloat))
    @test any(factor.ipiv.!=collect(eachindex(factor.ipiv)))
    weights=[maximum(abs,original[i,:]) for i in axes(original,1)]
    E=exact(original);rhs=exact(B0);w=Q.(weights)
    residual=(E*exact(solution)-rhs)./w
    relative2=squarednorm(residual)/squarednorm(rhs./w)
    transpose_residual=transpose(E)*exact(transposed)-rhs
    transpose_relative2=squarednorm(transpose_residual)/squarednorm(rhs)
    tolerance=Q(DiffMoM._PLANAR_DENSE_VOLTAGE_RTOL)
    @test relative2<=tolerance^2
    @test transpose_relative2<=tolerance^2
    reconstruction=exact(original[factor.p,:])-exact(Matrix(factor.L))*exact(Matrix(factor.U))
    factor_relative2=squarednorm(reconstruction)/squarednorm(E)
    @test factor_relative2<=Q(eps(Float64))^2
    @test all(x->precision(real(x))==bits && precision(imag(x))==bits,solution)
    @test all(x->precision(real(x))==bits && precision(imag(x))==bits,transposed)
    @test B==Complex{BigFloat}.(B0)
    # Warm in-place solve measurements own only the explicit workspace.
    work=deepcopy(B);DiffMoM._planar_owned_wide_factor_solve!(factor,work,scratch)
    work=deepcopy(B);bytes=@allocated DiffMoM._planar_owned_wide_factor_solve!(factor,work,scratch)
    generic=deepcopy(B)
    setprecision(BigFloat,bits) do;ldiv!(factor,generic);end
    generic=deepcopy(B)
    oldbytes=setprecision(BigFloat,bits) do;@allocated ldiv!(factor,generic);end
    @test bytes<oldbytes

end
singular=setprecision(BigFloat,128) do;Complex{BigFloat}.([1 2;2 4]);end
@test_throws SingularException DiffMoM._planar_owned_wide_lu!(singular,DiffMoM._planar_wide_factor_scratch(128))
@test_throws DimensionMismatch DiffMoM._planar_owned_wide_lu!(zeros(Complex{BigFloat},2,3),DiffMoM._planar_wide_factor_scratch(128))
end
end
