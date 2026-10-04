using DiffMoM, Test, LinearAlgebra, SparseArrays, Random

function _multilevel_defect_test_build(prob,f;kw...)
    defaults=(;modes=48,nx=32,block=31,max_bytes=32_000_000)
    DiffMoM._planar_bounded_multi_charge_projection_fft(prob,f;merge(defaults,(;kw...))...)
end

@testset "conformal: three interfaces, original workspace and lossless power" begin
    a=b=1e-3
    stack=PlanarStackup([PlanarLayer(e,1.,.25e-3) for e in (2.5,7.5,1.,3.)],
        TERM_GND,TERM_GND,a,b)
    base=planar_conformal_mesh([0. a a 0.;.375b .25b .75b .625b];
        edge_size=.3e-3,interior_size=.36e-3)
    nv=size(base.vertices,2);nt=length(base.interfaces)
    mesh=PlanarConformalMesh(hcat(base.vertices,base.vertices,base.vertices),
        hcat(base.triangles,base.triangles.+nv,base.triangles.+2nv);
        interfaces=vcat(fill(1,nt),fill(2,nt),fill(3,nt)))
    prob=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(.375b,.625b);z0=50+12im),
        PlanarConformalPort(2,:east,(.25b,.75b);z0=75-9im),
        PlanarConformalPort(3,:west,(.375b,.625b);z0=60+4im)])
    nb=length(prob.basis.width)
    built=DiffMoM._planar_bounded_multi_charge_projection_fft(prob,10e9;
        modes=48,nx=32,block=31,max_bytes=64_000_000)
    @test size(built.operator.kernels)==(3,3)
    X=randn(MersenneTwister(143),ComplexF64,nb,3);output=similar(X)
    work=DiffMoM._planar_projection_original_work(nb,length(mesh.interfaces),3;
        nlevels=3,max_bytes=64_000_000)
    DiffMoM._planar_projection_original_mul!(output,built.workspace,built.operator.incidence,X,work)
    original=assemble_planar_conformal_z(prob,10e9;mx=48,my=48)
    @test output≈original*X rtol=4e-13 atol=1e-12
    @test original≈transpose(original) rtol=3e-14 atol=1e-12
    @test_throws ArgumentError DiffMoM._planar_projection_original_mul!(X,
        built.workspace,built.operator.incidence,X,work)
    @test_throws ArgumentError DiffMoM._planar_projection_original_mul!(work.ev,
        built.workspace,built.operator.incidence,X,work)
    @test_throws DimensionMismatch DiffMoM._planar_projection_original_mul!(output,
        built.workspace,built.operator.incidence,X,DiffMoM._planar_projection_original_work(nb,length(mesh.interfaces),3;
            nlevels=2,max_bytes=64_000_000))
    @test_throws ArgumentError DiffMoM._planar_projection_original_work(nb,length(mesh.interfaces),3;
        nlevels=3,max_bytes=work.payload-1)
    zs=5im
    result=solve_planar_conformal_defect(prob,10e9;modes=48,nx=32,block=31,
        surface_zs=zs,max_bytes=64_000_000)
    dense=solve_planar_conformal(prob,10e9;mx=48,my=48,surface_zs=zs)
    @test result.diagnostics.interface_count==3
    @test maximum(result.relative_residuals)<=1e-10
    @test result.currents≈dense.currents rtol=2e-10 atol=1e-12
    @test result.s≈dense.s rtol=2e-10 atol=1e-12
    @test result.s'*result.s≈Matrix{ComplexF64}(I,3,3) rtol=2e-10 atol=1e-12
    voltage=ComplexF64[1+im,.3-.2im,.7+.1im]
    @test abs(.5real(dot(voltage,result.y*voltage)))<1e-11
end
function _multilevel_defect_test_solve(prob,f;kw...)
    defaults=(;modes=48,nx=32,block=31,max_bytes=64_000_000)
    solve_planar_conformal_defect(prob,f;merge(defaults,(;kw...))...)
end
function _multilevel_defect_test_original(prob,f,X;modes=48)
    w=DiffMoM._planar_multi_exact_workspace(prob,f,modes,31;max_bytes=16_000_000)
    D=DiffMoM._planar_physical_charge_incidence(prob)
    work=DiffMoM._planar_projection_original_work(size(X,1),size(D,1),size(X,2);
        nlevels=length(w.levels),max_bytes=16_000_000)
    output=similar(X)
    DiffMoM._planar_projection_original_mul!(output,w,D,X,work)
    (;output)
end
Base.@noinline _multilevel_defect_apply!(out,A,x)=mul!(out,A,x)

@testset "conformal: multilevel original equations and physical consumers" begin
    a=b=1e-3;stack=PlanarStackup([PlanarLayer(2.5-.025im,1.1,.3e-3),
        PlanarLayer(7.5-.075im,1.,.2e-3),PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,a,b)
    base=planar_conformal_mesh([0. a a 0.;.375b .25b .75b .625b];edge_size=.3e-3,interior_size=.36e-3)
    nt=length(base.interfaces)
    mesh=PlanarConformalMesh(hcat(base.vertices,base.vertices),hcat(base.triangles,
        base.triangles.+size(base.vertices,2));interfaces=vcat(fill(1,nt),fill(2,nt)))
    prob=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(.375b,.625b)),
        PlanarConformalPort(2,:east,(.25b,.75b))])
    nb=length(prob.basis.width);built=_multilevel_defect_test_build(prob,10e9);A=built.operator
    @test built.constructionbound<=32_000_000
    rng=MersenneTwister(7724);x=randn(rng,ComplexF64,nb);y=randn(rng,ComplexF64,nb)
    modal=DiffMoM._planar_multi_exact_workspace(prob,10e9,48,31;max_bytes=16_000_000)
    modal_arrays=(modal.triangles,modal.frees,modal.amplitudes,modal.wx,modal.wy,modal.wc,
        modal.kernels,modal.plus,modal.minus,modal.levels,modal.blevel,modal.tlevel)
    @test sum(sizeof,modal_arrays)<modal.payload
    @test_throws ArgumentError DiffMoM._planar_multi_exact_workspace(prob,10e9,48,31;max_bytes=modal.payload-1)
    independent_dense=assemble_planar_conformal_z(prob,10e9;mx=48,my=48)
    random_columns=randn(rng,ComplexF64,nb,3)
    exact_action=_multilevel_defect_test_original(prob,10e9,random_columns).output
    @test exact_action≈independent_dense*random_columns rtol=4e-13 atol=1e-12
    @test transpose(y)*(A*x)≈transpose(x)*(A*y) rtol=2e-12 atol=1e-12
    alias=copy(x);expected=A*x;mul!(alias,A,alias)
    @test alias≈expected rtol=2e-13 atol=1e-12
    @test_throws ArgumentError _multilevel_defect_test_build(prob,10e9;max_bytes=1)
    @test_throws ArgumentError _multilevel_defect_test_build(prob,10e9;max_pairs=1)
    @test_throws ArgumentError _multilevel_defect_test_build(prob,10e9;max_pair_mode_products=1)
    # Independent source/current solve with true original residual acceptance.
    rhs=zeros(ComplexF64,nb,2)
    for b in 1:nb
        p=prob.basis.port[b];p>0 && (rhs[b,p]=-prob.basis.port_sign[b]*prob.basis.width[b])
    end
    weights=prob.basis.width;M=Diagonal(inv.(weights));N=Diagonal(weights./built.diagonal)
    X=similar(rhs);initial=zeros(2);original_res=zeros(2);counts=zeros(Int,2)
    for p in 1:2
        solution,_=DiffMoM.Krylov.gmres(A,view(rhs,:,p);M,N,rtol=1e-11,atol=0.,memory=nb,itmax=10000,restart=true,reorthogonalization=true)
        X[:,p].=solution
    end
    for outer in 0:4
        residual=_multilevel_defect_test_original(prob,10e9,X).output-rhs
        for p in 1:2
            original_res[p]=norm(residual[:,p]./weights)/sqrt(count(==(p),prob.basis.port))
            outer==0 && (initial[p]=original_res[p])
        end
        maximum(original_res)<=1e-10 && break
        outer==4 && error("validation multilevel original residual $original_res")
        for p in 1:2
            original_res[p]<=1e-10 && continue
            correction,_=DiffMoM.Krylov.gmres(A,-view(residual,:,p);M,N,rtol=1e-11,atol=0.,memory=nb,itmax=10000,restart=true,reorthogonalization=true)
            X[:,p].+=correction;counts[p]+=1
        end
    end
    dense=solve_planar_conformal(prob,10e9;mx=48,my=48)
    @test maximum(original_res)<=1e-10
    @test X≈dense.currents rtol=2e-10 atol=1e-12
    println("multilevel bounded construction=",built.constructionbound," pairs=",length(built.vpairs.row),"/",length(built.cpairs.row),
        " initialoriginal=",initial," finaloriginal=",original_res," corrections=",counts,
        " densecurrentrelative=",norm(X-dense.currents)/norm(dense.currents));flush(stdout)
    # Simultaneous dielectric/sheet loss, purely reactive sheets and explicit
    # provider/cap/consumer checks on the same physical triangle geometry.
    spatial=ComplexF64[mesh.interfaces[t]==1 ? 2+im : 5-2im for t in eachindex(mesh.interfaces)]
    for zs in (0.,5im,spatial)
        r=_multilevel_defect_test_solve(prob,10e9;surface_zs=zs)
        oracle=solve_planar_conformal(prob,10e9;mx=48,my=48,surface_zs=zs)
        @test maximum(r.relative_residuals)<=1e-10
        @test r.y≈oracle.y rtol=2e-10 atol=1e-12
        @test r.s≈oracle.s rtol=2e-10 atol=1e-12
        @test r.currents≈oracle.currents rtol=2e-10 atol=1e-12
        @test r.s≈transpose(r.s) rtol=2e-10 atol=1e-12
        @test opnorm(r.s)<=1+1e-10
        @test r.diagnostics.owned_preconditioner_payload<=r.diagnostics.constructor_payload_bound
        @test r.diagnostics.total_payload_bound<=64_000_000
        for v in (ComplexF64[1,0],ComplexF64[1+im,.3-.2im])
            accepted=.5real(dot(v,r.y*v));current=r.currents*v
            loss=zs==0 ? spzeros(ComplexF64,nb,nb) : DiffMoM._planar_projection_material_gram(prob,
                r.diagnostics.surface_impedances;max_bytes=100_000_000)
            joule=-.5real(dot(current,loss*current))
            @test accepted>=joule-1e-12 && joule>=-1e-12
            waves=planar_power_waves(v,r.y*v;z0=r.z0)
            @test waves.b≈r.s*waves.a rtol=2e-10 atol=1e-12
        end
        incident=ComplexF64[.2+.1im,-.4+.3im]
        for (actual,reference) in zip(planar_current_maps(r;incident_waves=incident),
                planar_current_maps(oracle;incident_waves=incident))
            @test actual.current≈reference.current rtol=2e-10 atol=1e-12
            @test actual.integrated_current≈reference.integrated_current rtol=2e-10 atol=1e-15
        end
        angles=(theta=[.3,.7],phi=[0.,.8]);radiation=planar_farfield(r;incident_waves=incident,angles...)
        reference=planar_farfield(oracle;incident_waves=incident,angles...)
        @test radiation.etheta≈reference.etheta rtol=2e-10 atol=1e-12
        @test radiation.ephi≈reference.ephi rtol=2e-10 atol=1e-12
        println("multilevel zs=",zs," original=",r.relative_residuals," Sdelta=",maximum(abs,r.s-oracle.s),
            " owned=",r.diagnostics.owned_preconditioner_payload," bound=",r.diagnostics.total_payload_bound);flush(stdout)
    end
    materialcalls=Ref(0);referencecalls=Ref(0)
    referenceprovider=f->begin referencecalls[]+=1;50+12im end
    p=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(.375b,.625b);z0=referenceprovider),
        PlanarConformalPort(2,:east,(.25b,.75b);z0=referenceprovider)])
    provider=f->begin materialcalls[]+=1;spatial end
    for kw in ((;max_bytes=1),(;max_pairs=1),(;max_pair_mode_products=1),(;modes=0),
            (;nx=0),(;order=2),(;sigma=Inf),(;radius=NaN),(;memory=0),(;rtol=0.),
            (;max_references=0),(;max_visits=0),(;max_patch_visits=0),
            (;max_stencil_entries=0),(;max_modal_terms=1),(;block=0),
            (;modes=big(typemax(Int))+1),(;nx=big(typemax(Int))+1))
        @test_throws ArgumentError _multilevel_defect_test_solve(p,10e9;surface_zs=provider,kw...)
        @test materialcalls[]==0 && referencecalls[]==0
    end
    providerresult=_multilevel_defect_test_solve(p,10e9;surface_zs=provider)
    @test materialcalls[]==1 && referencecalls[]==2
    planar_current_maps(providerresult;incident_waves=[1.,0.])
    planar_farfield(providerresult;incident_waves=[1.,0.],theta=[.3],phi=[0.])
    @test materialcalls[]==1 && referencecalls[]==2
    rotatedvertices=vcat(reshape(a.-mesh.vertices[2,:],1,:),reshape(mesh.vertices[1,:],1,:))
    rotated=PlanarConformalProblem(stack,PlanarConformalMesh(rotatedvertices,mesh.triangles;interfaces=mesh.interfaces),
        [PlanarConformalPort(1,:south,(.375b,.625b)),PlanarConformalPort(2,:north,(.25b,.75b))])
    r=_multilevel_defect_test_solve(prob,10e9;surface_zs=spatial)
    rotation=_multilevel_defect_test_solve(rotated,10e9;surface_zs=spatial)
    @test maximum(rotation.relative_residuals)<=1e-10
    @test rotation.s≈r.s rtol=2e-10 atol=1e-12
    println("rotation fullS delta=",maximum(abs,rotation.s-r.s));flush(stdout)
end

# Independent quadratic Gaussian integration on the physical restrictions.
function _multilevel_sheet_gram_quadrature(prob,zs)
    nb=length(prob.basis.width);out=zeros(ComplexF64,nb,nb)
    for t in eachindex(prob.mesh.interfaces),q in ((2/3,1/6,1/6),(1/6,2/3,1/6),(1/6,1/6,2/3))
        ids=prob.mesh.triangles[:,t];point=prob.mesh.vertices[:,ids]*collect(q);F=zeros(2,nb)
        for b in 1:nb,h in 1:2
            prob.basis.triangles[h,b]==t || continue
            free=only(i for i in ids if !(i in prob.basis.edges[:,b]))
            sign=h==2 || prob.basis.triangles[2,b]==0 ? -1. : 1.
            F[:,b].+=sign*prob.basis.width[b]/(2prob.mesh.areas[t])*(point-prob.mesh.vertices[:,free])
        end
        impedance=zs isa Number ? zs : zs[t]
        out.-=impedance*prob.mesh.areas[t]/3*(transpose(F)*F)
    end
    out
end

@testset "multilevel independent Gram, charge, image and resource readiness" begin
    a=b=1e-3;stack=PlanarStackup([PlanarLayer(2.5-.025im,1.1,.3e-3),
        PlanarLayer(7.5-.075im,1.,.2e-3),PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,a,b)
    base=planar_conformal_mesh([0. a a 0.;.375b .25b .75b .625b];edge_size=.3e-3,interior_size=.36e-3)
    nt=length(base.interfaces);mesh=PlanarConformalMesh(hcat(base.vertices,base.vertices),
        hcat(base.triangles,base.triangles.+size(base.vertices,2));interfaces=vcat(fill(1,nt),fill(2,nt)))
    prob=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(.375b,.625b)),
        PlanarConformalPort(2,:east,(.25b,.75b))]);nb=length(prob.basis.width)
    zs=ComplexF64[mesh.interfaces[t]==1 ? 2+im : 5-2im for t in eachindex(mesh.interfaces)]
    quad=_multilevel_sheet_gram_quadrature(prob,zs)
    exact=DiffMoM._planar_projection_material_gram(prob,zs;max_bytes=100_000_000)
    @test Matrix(exact)≈quad rtol=3e-14 atol=1e-20
    @test issymmetric(exact)
    @test minimum(eigvals(Hermitian(-quad)))>0
    r=_multilevel_defect_test_solve(prob,10e9;surface_zs=zs)
    for v in (ComplexF64[1,0],ComplexF64[1+im,.3-.2im])
        current=r.currents*v;accepted=.5real(dot(v,r.y*v));joule=-.5real(dot(current,quad*current))
        @test accepted>joule>0
        @test abs(joule+.5real(dot(current,exact*current)))<1e-14
    end
    built=_multilevel_defect_test_build(prob,10e9);raw=built.operator;A=raw
    rng=MersenneTwister(14281);x=randn(rng,ComplexF64,nb);out=similar(x)
    _multilevel_defect_apply!(out,A,x)
    allocation=minimum(@allocated(_multilevel_defect_apply!(out,A,x)) for _ in 1:3)
    @test allocation<sizeof(x) # No per-action full basis work-vector allocation.
    alias=copy(x);mul!(alias,A,alias)
    @test alias≈out rtol=2e-13 atol=1e-12
    following=randn(rng,ComplexF64,nb);next_expected=A*following
    for scratch in (raw.lattice,raw.modes,raw.source,raw.target,raw.charge,
            raw.field,raw.vector_second,raw.output)
        length(scratch)>=nb || continue
        destination=view(vec(scratch),1:nb)
        mul!(destination,A,x)
        @test destination≈out rtol=2e-13 atol=1e-12
        @test A*following≈next_expected rtol=2e-13 atol=1e-12
        @test A*x≈out rtol=2e-13 atol=1e-12
    end
    for coefficient in (view(vec(raw.kernels[1,1][1]),1:nb),view(raw.vector_correction.nzval,1:nb))
        @test_throws ArgumentError mul!(out,A,coefficient)
        @test_throws ArgumentError mul!(coefficient,A,x)
    end
    @test_throws ArgumentError mul!(out,A,view(raw.source,1:nb))
    @test_throws DimensionMismatch mul!(zeros(ComplexF64,nb-1),A,x)
    @test size(A,3)==1
    @test_throws ArgumentError size(A,0)
    D=raw.incidence
    for level in (1,2)
        candidates=unique(vec(mesh.triangles[:,findall(==(level),mesh.interfaces)]))
        vertex=first(v for v in candidates if 0<mesh.vertices[1,v]<a && 0<mesh.vertices[2,v]<b)
        loop=zeros(nb)
        for basis in 1:nb
            firstvertex,lastvertex=prob.basis.edges[:,basis]
            vertex in (firstvertex,lastvertex) || continue
            triangle=prob.basis.triangles[1,basis];triangle==0 && continue
            ids=mesh.triangles[:,triangle];position=findfirst(==(vertex),ids)
            other=firstvertex==vertex ? lastvertex : firstvertex
            loop[basis]=(ids[mod1(position+1,3)]==other ? 1. : -1.)/prob.basis.width[basis]
        end
        @test norm(loop)>0
        @test norm(D*loop)/(norm(D)*norm(loop))<1e-14
    end
    lo,hi=DiffMoM._planar_projection_support_boxes(prob);radius=6*.5a/32
    pair=DiffMoM._planar_projection_local_pairs(lo,hi,a,b,radius)
    actual=Set(zip(pair.row,pair.column))
    brute=Set((r,q) for q in 1:nb for r in q:nb if DiffMoM._planar_projection_box_distance(lo,hi,q,r)<=radius)
    @test actual==brute
    image_checks=0
    for q in 1:nb,r in q:nb,sx in (-1,1),sy in (-1,1),ix in -1:1,iy in -1:1
        xl,xh=sx==1 ? (lo[1,r]+2ix*a,hi[1,r]+2ix*a) : (-hi[1,r]+2ix*a,-lo[1,r]+2ix*a)
        yl,yh=sy==1 ? (lo[2,r]+2iy*b,hi[2,r]+2iy*b) : (-hi[2,r]+2iy*b,-lo[2,r]+2iy*b)
        distance=hypot(max(0.,lo[1,q]-xh,xl-hi[1,q]),max(0.,lo[2,q]-yh,yl-hi[2,q]))
        if distance<=radius
            @test (r,q) in actual;image_checks+=1
        end
    end
    materialcalls=Ref(0);refcalls=Ref(0)
    ref=f->begin refcalls[]+=1;50+12im end
    provider=f->begin materialcalls[]+=1;zs end
    p=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(.375b,.625b);z0=ref),
        PlanarConformalPort(2,:east,(.25b,.75b);z0=ref)])
    accepted=_multilevel_defect_test_solve(p,10e9;surface_zs=provider)
    @test materialcalls[]==1 && refcalls[]==2
    @test_throws ArgumentError _multilevel_defect_test_solve(p,10e9;surface_zs=provider,
        max_bytes=accepted.diagnostics.total_payload_bound-1)
    @test materialcalls[]==1 && refcalls[]==2
    @test_throws ErrorException _multilevel_defect_test_solve(p,10e9;surface_zs=provider,max_outer=0)
    @test materialcalls[]==2 && refcalls[]==4
    println("readiness allocated bytes/mul=",allocation," exact image guards=",image_checks,
        " independent Gram relative=",norm(Matrix(exact)-quad)/norm(quad),
        " original=",r.relative_residuals," owned=",r.diagnostics.owned_preconditioner_payload,
        " aggregate=",r.diagnostics.total_payload_bound);flush(stdout)
end
