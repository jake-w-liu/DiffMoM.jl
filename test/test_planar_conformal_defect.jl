using DiffMoM, Test, LinearAlgebra, SparseArrays, Random

@testset "conformal: charge cancellation preserves product and sum roundoff" begin
    D=sparse([1,1,1,2,2],[1,2,3,1,2],[1.,1.,1.,1.1,-1.],2,3)
    X=ComplexF64[1e16+10im 10+1e16im;1+11im 11+1im;-1e16 0-1e16im]
    exact=setprecision(256) do
        ComplexF64.(BigFloat.(Matrix(D))*Complex{BigFloat}.(X))
    end
    ordinary=D*X
    @test ordinary[1,1]!=exact[1,1]
    @test ordinary[2,2]!=exact[2,2]
    out=similar(exact);scratch=similar(exact)
    @test DiffMoM._planar_projection_charge_mul!(out,D,X,scratch)===out
    @test out==exact
    for p in axes(X,2)
        DiffMoM._planar_projection_charge_mul!(view(out,:,p),D,view(X,:,p),view(scratch,:,p))
        @test out[:,p]==exact[:,p]
    end
    @test (@allocated DiffMoM._planar_projection_charge_mul!(out,D,X,scratch))==0
    @test_throws DimensionMismatch DiffMoM._planar_projection_charge_mul!(out,D,X[:,1],scratch)
end

function _defect_conformal_problem(;loaded=false,z0=(50.0,75.0))
    vertices=[0. 1e-3 1e-3 0.;.375e-3 .25e-3 .75e-3 .625e-3]
    mesh=planar_conformal_mesh(vertices;interface=loaded ? 2 : 1,
        edge_size=.0003,interior_size=.00036)
    layers=loaded ? [PlanarLayer(2.5-.005im,1.,.3e-3),PlanarLayer(4.0-.04im,1.2,.2e-3),
        PlanarLayer(1.,1.,.3e-3),PlanarLayer(2.0-.002im,1.1,.2e-3)] :
        [PlanarLayer(1.,1.,.5e-3),PlanarLayer(1.,1.,.5e-3)]
    stack=PlanarStackup(layers,TERM_GND,TERM_GND,1e-3,1e-3)
    level=loaded ? 2 : 1
    PlanarConformalProblem(stack,mesh,
        [PlanarConformalPort(level,:west,(.375e-3,.625e-3);z0=z0[1]),
         PlanarConformalPort(level,:east,(.25e-3,.75e-3);z0=z0[2])])
end

# Independent degree-two triangle quadrature, constructed directly from each
# physical RWG edge/free vertex rather than the production Gram emitter.
function _defect_sheet_gram_quadrature(prob,zs)
    nb=length(prob.basis.width);G=zeros(ComplexF64,nb,nb)
    barycentric=((2/3,1/6,1/6),(1/6,2/3,1/6),(1/6,1/6,2/3))
    for t in eachindex(prob.mesh.interfaces),q in barycentric
        ids=prob.mesh.triangles[:,t];v=prob.mesh.vertices[:,ids]
        point=v*collect(q);F=zeros(2,nb)
        for b in 1:nb,half in 1:2
            prob.basis.triangles[half,b]==t || continue
            edge=prob.basis.edges[:,b];free=only(i for i in ids if !(i in edge))
            direction=half==2 || prob.basis.triangles[2,b]==0 ? -1. : 1.
            F[:,b].+=direction*prob.basis.width[b]/(2prob.mesh.areas[t])*(point-prob.mesh.vertices[:,free])
        end
        impedance=zs isa Number ? zs : zs[t]
        G.-=impedance*prob.mesh.areas[t]/3*(transpose(F)*F)
    end
    G
end

@testset "conformal: exact local material and original power equation" begin
    prob=_defect_conformal_problem(;z0=(50+12im,75-9im))
    nt=length(prob.mesh.interfaces);nb=length(prob.basis.width)
    spatial=ComplexF64[sum(prob.mesh.vertices[1,prob.mesh.triangles[:,t]])/3<.5e-3 ?
        2+im : 5-2im for t in 1:nt]
    for zs in (2.,2+3im,5im,spatial)
        r=solve_planar_conformal_defect(prob,10e9;modes=64,nx=32,surface_zs=zs)
        dense=solve_planar_conformal(prob,10e9;mx=64,my=64,surface_zs=zs)
        A=r.preconditioner;G=_defect_sheet_gram_quadrature(prob,zs)
        @test maximum(r.relative_residuals)<=1e-10
        @test minimum(r.diagnostics.initial_original_relative_residuals)>1e-8
        @test maximum(r.approximate_relative_residuals)>1e-8
        @test r.y≈dense.y rtol=1e-10 atol=1e-12
        @test r.s≈dense.s rtol=1e-10 atol=1e-12
        @test r.currents≈dense.currents rtol=1e-10 atol=1e-12
        @test Matrix(A.loss)≈G rtol=2e-14 atol=1e-20
        @test issymmetric(A.loss)
        @test nnz(A.loss)<=9nt
        @test r.s≈transpose(r.s) rtol=1e-10 atol=1e-12
        @test minimum(eigvals(Hermitian(I-r.s'*r.s)))>=-1e-10
        @test r.diagnostics.surface_impedances==(zs isa Number ? fill(zs,nt) : zs)
        @test r.diagnostics.owned_preconditioner_payload==DiffMoM._planar_projection_fft_owned_bytes(A)
        @test r.diagnostics.owned_preconditioner_payload<r.diagnostics.constructor_payload_bound
        @test r.diagnostics.total_payload_bound==r.diagnostics.constructor_payload_bound+r.diagnostics.solve_payload_bound
        @test r.diagnostics.total_payload_bound<=128_000_000
        @test DiffMoM._planar_projection_sparse_bytes(A.loss)<=r.diagnostics.material_constructor_payload_bound
        for voltage in (ComplexF64[1,0],ComplexF64[1+im,.3-.2im])
            current=r.currents*voltage
            accepted=.5real(dot(voltage,r.y*voltage))
            dissipated=-.5real(dot(current,G*current))
            waves=planar_power_waves(voltage,r.y*voltage;z0=r.z0)
            @test accepted≈dissipated rtol=1e-9 atol=1e-12
            @test dissipated>=-1e-12
            @test waves.b≈r.s*waves.a rtol=1e-10 atol=1e-12
            @test .5(sum(abs2,waves.a)-sum(abs2,waves.b))≈dissipated rtol=1e-9 atol=1e-12
        end
        zs==5im && @test r.s'*r.s≈Matrix{ComplexF64}(I,2,2) rtol=1e-10 atol=1e-12
        x=randn(MersenneTwister(97235),ComplexF64,nb);expected=A*x;aliased=copy(x)
        mul!(aliased,A,aliased)
        @test aliased≈expected rtol=2e-13 atol=1e-12
        @test_throws ArgumentError mul!(similar(x),A,A.projection.output)
        @test_throws ArgumentError mul!(similar(x),A,view(vec(A.projection.lattice),1:nb))
        @test_throws DimensionMismatch mul!(similar(x),A,A.projection.charge)
        @test_throws ArgumentError mul!(similar(x),A,view(A.loss.nzval,1:nb))
        @test_throws ArgumentError mul!(view(A.loss.nzval,1:nb),A,x)
        @test_throws DimensionMismatch mul!(zeros(ComplexF64,nb-1),A,x)
        @test size(A,3)==1
        @test_throws ArgumentError size(A,0)
        incident=ComplexF64[.2+.1im,-.4+.3im]
        for (map,oracle) in zip(planar_current_maps(r;incident_waves=incident),
                planar_current_maps(dense;incident_waves=incident))
            @test map.current≈oracle.current rtol=1e-10 atol=1e-12
            @test map.integrated_current≈oracle.integrated_current rtol=1e-10 atol=1e-15
        end
        angles=(theta=[.3,.7],phi=[0.,.8])
        radiation=planar_farfield(r;incident_waves=incident,angles...)
        oracle=planar_farfield(dense;incident_waves=incident,angles...)
        @test radiation.etheta≈oracle.etheta rtol=1e-10 atol=1e-12
        @test radiation.ephi≈oracle.ephi rtol=1e-10 atol=1e-12
    end
    @test_throws ErrorException solve_planar_conformal_defect(prob,10e9;modes=64,nx=32,surface_zs=2.,max_outer=0)
    @test_throws ErrorException solve_planar_conformal_defect(prob,10e9;modes=64,nx=32,surface_zs=2.,maxiter=1)
    loaded=_defect_conformal_problem(;loaded=true,z0=(50+12im,75-9im))
    r=solve_planar_conformal_defect(loaded,10e9;modes=48,nx=24,surface_zs=spatial)
    dense=solve_planar_conformal(loaded,10e9;mx=48,my=48,surface_zs=spatial)
    @test maximum(r.relative_residuals)<=1e-10
    @test r.y≈dense.y rtol=1e-10 atol=1e-12
    @test r.s≈dense.s rtol=1e-10 atol=1e-12
    @test r.currents≈dense.currents rtol=1e-10 atol=1e-12
    @test opnorm(r.s)<=1+1e-10
    voltage=ComplexF64[1+im,.3-.2im];current=r.currents*voltage
    sheet_power=-.5real(dot(current,_defect_sheet_gram_quadrature(loaded,spatial)*current))
    accepted=.5real(dot(voltage,r.y*voltage))
    @test accepted>sheet_power>0 # The loaded dielectric also absorbs power.
end

@testset "conformal: material providers and bounded preflight" begin
    material_calls=Ref(0);reference_calls=Ref(0)
    ref=f->begin reference_calls[]+=1;50+12im end
    prob=_defect_conformal_problem(;z0=(ref,ref));nt=length(prob.mesh.interfaces)
    values=fill(2+3im,nt)
    provider=f->begin material_calls[]+=1;@test f==10e9;values end
    vertices=hcat(prob.mesh.vertices,prob.mesh.vertices)
    triangles=hcat(prob.mesh.triangles,prob.mesh.triangles.+size(prob.mesh.vertices,2))
    mesh=PlanarConformalMesh(vertices,triangles;interfaces=vcat(fill(1,nt),fill(2,nt)))
    stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3) for _ in 1:3],TERM_GND,TERM_GND,1e-3,1e-3)
    multilevel=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(.375e-3,.625e-3);z0=ref),
        PlanarConformalPort(2,:east,(.25e-3,.75e-3);z0=ref)])
    @test_throws ArgumentError solve_planar_conformal_defect(multilevel,10e9;surface_zs=provider,max_bytes=1)
    @test material_calls[]==0 && reference_calls[]==0
    for kw in ((;max_bytes=1),(;max_pairs=0),(;max_visits=0),(;max_patch_visits=0),
            (;max_stencil_entries=0),(;max_modal_terms=1),(;max_pair_mode_products=1),
            (;rtol=0.),(;memory=0))
        @test_throws ArgumentError solve_planar_conformal_defect(prob,10e9;
            modes=32,nx=16,surface_zs=provider,kw...)
        @test material_calls[]==0 && reference_calls[]==0
    end
    for invalid in (NaN,Inf,1+Inf*im,big"1e1000",fill(0.,nt-1),fill(NaN,nt),
            fill("metal",nt),ones(2,2),nothing)
        @test_throws ArgumentError solve_planar_conformal_defect(prob,10e9;
            modes=16,nx=8,surface_zs=invalid)
        @test reference_calls[]==0
    end
    for invalid in (NaN,fill(0.,nt-1),fill("metal",nt),ones(2,2),nothing)
        invalid_provider=f->begin material_calls[]+=1;invalid end
        before=material_calls[]
        @test_throws ArgumentError solve_planar_conformal_defect(prob,10e9;
            modes=8,nx=8,surface_zs=invalid_provider)
        @test material_calls[]==before+1 && reference_calls[]==0
    end
    material_calls[]=0
    r=solve_planar_conformal_defect(prob,10e9;modes=48,nx=24,surface_zs=provider)
    @test material_calls[]==1 && reference_calls[]==2
    @test_throws ArgumentError solve_planar_conformal_defect(prob,10e9;modes=48,nx=24,
        surface_zs=provider,max_bytes=r.diagnostics.total_payload_bound-1)
    @test material_calls[]==1 && reference_calls[]==2
    @test r.diagnostics.surface_impedances==values
    @test r.diagnostics.surface_impedances!==values
    fill!(values,99)
    @test all(==(2+3im),r.diagnostics.surface_impedances)
    planar_current_maps(r;incident_waves=[1.,0.])
    planar_farfield(r;incident_waves=[1.,0.],theta=[.3],phi=[0.])
    @test material_calls[]==1 && reference_calls[]==2
    zero_calls=Ref(0);zero_provider=f->begin zero_calls[]+=1;0. end
    z=solve_planar_conformal_defect(prob,10e9;modes=32,nx=16,surface_zs=zero_provider)
    @test zero_calls[]==1 && reference_calls[]==4
    @test z.preconditioner isa DiffMoM._PlanarConformalProjection
    @test all(iszero,z.diagnostics.surface_impedances)
    @test z.diagnostics.material_constructor_payload_bound>0 # Unknown provider was reserved.
    bare=solve_planar_conformal_defect(_defect_conformal_problem(),10e9;modes=32,nx=16)
    @test bare.diagnostics.material_constructor_payload_bound==0
    @test all(iszero,bare.diagnostics.surface_impedances)
    scalar_calls=Ref(0);scalar_provider=f->begin scalar_calls[]+=1;2+3im end
    scalar=solve_planar_conformal_defect(_defect_conformal_problem(;z0=(50+12im,50+12im)),10e9;
        modes=48,nx=24,surface_zs=scalar_provider)
    @test scalar_calls[]==1
    @test scalar.currents≈r.currents rtol=1e-10 atol=1e-12
    @test_throws ArgumentError DiffMoM._planar_projection_material_gram(prob,fill(2.,nt);max_bytes=1)
end

@testset "conformal: resistive rectangular sheet DC oracle" begin
    a,b=1e-3,.5e-3;resistance=2.
    stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3),PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,a,b)
    mesh=PlanarConformalMesh([0. a a 0.;0. 0. b b],[1 1;2 3;3 4])
    prob=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(0.,b)),
        PlanarConformalPort(1,:east,(0.,b))])
    r=solve_planar_conformal_defect(prob,10e6;modes=32,nx=16,ny=8,surface_zs=resistance)
    @test maximum(r.relative_residuals)<=1e-10
    impedance=2/(r.y[1,1]-r.y[1,2])
    @test real(impedance)≈resistance*a/b rtol=2e-6
    @test real.(r.y)/real(r.y[1,1])≈[1. -1.;-1. 1.] rtol=1e-9
    current=planar_current_maps(r;voltages=[1.,0.])
    for triangle in current
        @test real.(triangle.current[1,:])≈fill(real(r.y[1,1])/b,3) rtol=2e-6
        @test maximum(abs,real.(triangle.current[2,:]))<1e-6/(resistance*a)
    end
    # Recomputed voltage residuals used to fail after all three Arnoldi
    # directions, despite a converged recurrence. Both physical gates stay.
    for (frequency,rs) in ((1e6,2.),(20e6,.2))
        low=solve_planar_conformal_defect(prob,frequency;modes=32,nx=16,ny=8,surface_zs=rs)
        @test maximum(low.diagnostics.initial_projected_relative_residuals)<=1e-10
        @test maximum(low.relative_residuals)<=1e-10
        @test real(2/(low.y[1,1]-low.y[1,2]))≈rs*a/b rtol=2e-6
        @test all(i->i<=10000,low.iterations)
    end
end

Base.@noinline function _defect_folded_kernel_alloc(prob,workspace)
    DiffMoM._planar_folded_projection_kernels(prob,2,2,10e9,workspace.modes;workspace)
end

@testset "conformal: bounded original-equation defect solve" begin
    prob=_defect_conformal_problem()
    r=solve_planar_conformal_defect(prob,10e9;modes=64,nx=32,block=31)
    dense=solve_planar_conformal(prob,10e9;mx=64,my=64)
    @test r isa PlanarConformalDefectResult
    @test r.problem===prob
    @test r.freq==10e9 && r.omega==2pi*10e9
    @test maximum(r.relative_residuals)<=1e-10
    @test minimum(r.diagnostics.initial_original_relative_residuals)>1e-8
    @test maximum(r.diagnostics.initial_projected_relative_residuals)<=1e-10
    @test maximum(r.approximate_relative_residuals)>1e-8
    @test all(!isempty,r.diagnostics.correction_iterations)
    @test r.y≈dense.y rtol=1e-10 atol=1e-12
    @test r.s≈dense.s rtol=1e-10 atol=1e-12
    @test r.currents≈dense.currents rtol=1e-10 atol=1e-12
    @test r.s≈transpose(r.s) rtol=1e-10 atol=1e-12
    @test r.s'*r.s≈Matrix{ComplexF64}(I,2,2) rtol=1e-10 atol=1e-12
    @test r.diagnostics.owned_preconditioner_payload==DiffMoM._planar_projection_fft_owned_bytes(r.preconditioner)
    @test r.diagnostics.owned_preconditioner_payload<r.diagnostics.constructor_payload_bound
    @test r.diagnostics.total_payload_bound==r.diagnostics.constructor_payload_bound+r.diagnostics.solve_payload_bound
    @test r.diagnostics.total_payload_bound<=512_000_000
    @test_throws ErrorException solve_planar_conformal_defect(prob,10e9;modes=64,nx=32,max_outer=0)
    @test_throws ErrorException solve_planar_conformal_defect(prob,10e9;modes=64,nx=32,maxiter=1)

    # Independent public modal matrix, with arbitrary physical source columns.
    nb=length(prob.basis.width);nt=length(prob.mesh.interfaces)
    w=DiffMoM._planar_projection_exact_workspace(prob,10e9,24,13)
    D=DiffMoM._planar_physical_charge_incidence(prob)
    X=randn(MersenneTwister(89713),ComplexF64,nb,3);out=similar(X)
    work=DiffMoM._planar_projection_original_work(nb,nt,3;max_bytes=1_000_000)
    DiffMoM._planar_projection_original_mul!(out,w,D,X,work)
    Z=assemble_planar_conformal_z(prob,10e9;mx=24,my=24)
    @test out≈Z*X rtol=2e-13 atol=1e-12
    @test_throws ArgumentError DiffMoM._planar_projection_original_mul!(X,w,D,X,work)
    @test_throws ArgumentError DiffMoM._planar_projection_original_mul!(work.ev,w,D,X,work)
    @test_throws DimensionMismatch DiffMoM._planar_projection_original_mul!(zeros(ComplexF64,nb,2),w,D,X,work)
    A=r.preconditioner;x=X[:,1];expected=A*x;aliased=copy(x)
    mul!(aliased,A,aliased)
    @test aliased≈expected rtol=2e-13 atol=1e-12
    @test_throws ArgumentError mul!(similar(x),A,A.output)
    @test_throws DimensionMismatch mul!(zeros(ComplexF64,nb-1),A,x)
    @test size(A,3)==1
    @test_throws ArgumentError size(A,0)
    # Folding reuses the reserved original modal grid/cascades. Its three
    # small lattice kernels allocate less than even one extra mode vector;
    # allocating a second complete modal workspace would violate this gate.
    shared=DiffMoM._planar_projection_exact_workspace(prob,10e9,256,1)
    _defect_folded_kernel_alloc(prob,shared)
    @test minimum(@allocated(_defect_folded_kernel_alloc(prob,shared)) for _ in 1:3)<sizeof(shared.grid.kx)
    @test_throws ArgumentError DiffMoM._planar_folded_projection_kernels(prob,2,2,10e9,255;workspace=shared)

    # Physical loops have exactly zero pulse charge before scalar correction.
    interior=first(v for v in axes(prob.mesh.vertices,2) if
        0<prob.mesh.vertices[1,v]<prob.stack.a && 0<prob.mesh.vertices[2,v]<prob.stack.b)
    loop=zeros(nb)
    for b in 1:nb
        e1,e2=prob.basis.edges[:,b]
        interior in (e1,e2) || continue
        triangle=prob.basis.triangles[1,b]
        triangle==0 && continue
        ids=prob.mesh.triangles[:,triangle]
        at=findfirst(==(interior),ids);other=e1==interior ? e2 : e1
        loop[b]=(ids[mod1(at+1,3)]==other ? 1. : -1.)/prob.basis.width[b]
    end
    @test norm(loop)>0
    @test norm(D*loop)/(norm(D)*norm(loop))<1e-14
end

@testset "conformal: loaded physical responses and providers" begin
    calls=[Ref(0),Ref(0)];refs=ComplexF64[50+12im,75-9im]
    providers=[f->begin calls[p][]+=1;@test f==10e9;refs[p] end for p in 1:2]
    prob=_defect_conformal_problem(;loaded=true,z0=providers)
    for kw in ((;max_bytes=1),(;memory=0),(;maxiter=0),(;max_outer=-1),(;max_outer=9),
        (;rtol=0.),(;rtol=NaN),(;rtol=1.),(;rtol=nextfloat(0.)),(;modes=0),(;nx=0),(;ny=0),
        (;block=0),(;order=2),(;sigma=NaN),(;radius=Inf),(;max_pairs=0),
        (;max_references=0),(;max_visits=0),(;max_patch_visits=0),(;max_stencil_entries=0),
        (;max_modal_terms=1),(;max_pair_mode_products=1),(;modes=typemax(Int)),
        (;nx=typemax(Int)),(;memory=BigInt(typemax(Int))+1))
        @test_throws ArgumentError solve_planar_conformal_defect(prob,10e9;modes=48,nx=24,kw...)
        @test getindex.(calls)==[0,0]
    end
    for f in (0.,-1.,NaN,Inf,1+im,big"1e1000",big"1e-1000",floatmax(Float64))
        @test_throws ArgumentError solve_planar_conformal_defect(prob,f;modes=48,nx=24)
        @test getindex.(calls)==[0,0]
    end
    r=solve_planar_conformal_defect(prob,10e9;modes=48,nx=24)
    @test getindex.(calls)==[1,1]
    @test r.z0==refs
    dense=solve_planar_conformal(prob,10e9;mx=48,my=48)
    @test getindex.(calls)==[2,2]
    @test maximum(r.relative_residuals)<=1e-10
    @test r.s≈dense.s rtol=1e-10 atol=1e-12
    @test r.y≈dense.y rtol=1e-10 atol=1e-12
    @test minimum(eigvals(Hermitian(I-r.s'*r.s)))>=-1e-10
    for voltage in (ComplexF64[1,0],ComplexF64[1+im,.2-.3im])
        @test real(dot(voltage,r.y*voltage))>=-1e-10*norm(voltage)*norm(r.y*voltage)
    end
    incident=ComplexF64[.2+.1im,-.4+.3im]
    maps=planar_current_maps(r;incident_waves=incident)
    reference=planar_conformal_current_maps(dense;incident_waves=incident)
    @test getindex.(calls)==[2,2] # Consumers use retained evaluated references.
    for (a,b) in zip(maps,reference)
        @test a.vertices==b.vertices && a.interface==b.interface
        @test a.current≈b.current rtol=1e-10 atol=1e-12
        @test a.integrated_current≈b.integrated_current rtol=1e-10 atol=1e-15
    end
    @test_throws ArgumentError planar_current_maps(r;max_bytes=1)
    @test_throws ArgumentError planar_current_maps(r;voltages=[1,0],incident_waves=incident)
    angles=(theta=[.3,.7,1.1],phi=[0.,.8])
    pattern=planar_farfield(r;incident_waves=incident,angles...)
    oracle=planar_farfield(dense;incident_waves=incident,angles...)
    @test pattern.etheta≈oracle.etheta rtol=1e-10 atol=1e-12
    @test pattern.ephi≈oracle.ephi rtol=1e-10 atol=1e-12
    @test pattern.intensity≈oracle.intensity rtol=1e-10 atol=1e-12
    @test getindex.(calls)==[2,2]
    @test_throws ArgumentError planar_farfield(r;max_bytes=1,angles...)

    vertices=vcat(reshape(prob.stack.a.-prob.mesh.vertices[2,:],1,:),reshape(prob.mesh.vertices[1,:],1,:))
    rotated=PlanarConformalProblem(prob.stack,PlanarConformalMesh(vertices,prob.mesh.triangles;interfaces=2),
        [PlanarConformalPort(2,:south,(.375e-3,.625e-3);z0=refs[1]),
         PlanarConformalPort(2,:north,(.25e-3,.75e-3);z0=refs[2])])
    turned=solve_planar_conformal_defect(rotated,10e9;modes=48,nx=24)
    @test maximum(turned.relative_residuals)<=1e-10
    @test turned.s≈r.s rtol=1e-10 atol=1e-12
end

@testset "conformal: bounded neighbors and unsupported scope" begin
    rng=MersenneTwister(37191);n=37;a,b=1.,.71;radius=.093
    lo=zeros(2,n);hi=similar(lo)
    for i in 1:n,d in 1:2
        extent=d==1 ? a : b;x=rand(rng)*extent
        lo[d,i]=max(0.,x-rand(rng)*.06);hi[d,i]=min(extent,x+rand(rng)*.06)
    end
    lo[:,1].=(0.,0.);hi[:,1].=(.001,.001)
    lo[:,2].=(a-.001,b-.001);hi[:,2].=(a,b)
    pairs=DiffMoM._planar_projection_local_pairs(lo,hi,a,b,radius)
    actual=Set(zip(pairs.row,pairs.column))
    expected=Set((r,q) for q in 1:n for r in q:n if DiffMoM._planar_projection_box_distance(lo,hi,q,r)<=radius)
    @test actual==expected
    for q in 1:n,r in q:n,sx in (-1,1),sy in (-1,1),ix in -1:1,iy in -1:1
        xl,xh=sx==1 ? (lo[1,r]+2ix*a,hi[1,r]+2ix*a) : (-hi[1,r]+2ix*a,-lo[1,r]+2ix*a)
        yl,yh=sy==1 ? (lo[2,r]+2iy*b,hi[2,r]+2iy*b) : (-hi[2,r]+2iy*b,-lo[2,r]+2iy*b)
        distance=hypot(max(0.,lo[1,q]-xh,xl-hi[1,q]),max(0.,lo[2,q]-yh,yl-hi[2,q]))
        distance<=radius && @test (r,q) in actual
    end
    for kw in ((;max_pairs=length(actual)-1),(;max_references=pairs.references-1),
        (;max_visits=pairs.visits-1),(;max_bytes=1))
        @test_throws ArgumentError DiffMoM._planar_projection_local_pairs(lo,hi,a,b,radius;kw...)
    end
    for (x,y) in ((0.,b),(a,NaN),(-1.,b),(Inf,b))
        @test_throws ArgumentError DiffMoM._planar_projection_local_pairs(zeros(2,0),zeros(2,0),x,y,radius)
    end
    prob=_defect_conformal_problem()
    openstack=PlanarStackup(prob.stack.layers,TERM_GND,TERM_SPACE,prob.stack.a,prob.stack.b)
    unsupported=PlanarConformalProblem(openstack,prob.mesh,prob.ports)
    @test_throws ArgumentError solve_planar_conformal_defect(unsupported,10e9;modes=8,nx=8)
    anisotropic=PlanarStackup([PlanarLayer(2.,1.,.5e-3;epsr_z=3.),PlanarLayer(1.,1.,.5e-3)],
        TERM_GND,TERM_GND,prob.stack.a,prob.stack.b)
    @test_throws ArgumentError solve_planar_conformal_defect(PlanarConformalProblem(anisotropic,prob.mesh,prob.ports),10e9;modes=8,nx=8)
end
