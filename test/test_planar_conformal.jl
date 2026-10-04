using DiffMoM, Test, LinearAlgebra
isdefined(DiffMoM,:_planar_triangle_affine_fourier) || Base.include(DiffMoM,joinpath(@__DIR__,"../src/planar/PlanarTriangleTransforms.jl"))
isdefined(DiffMoM,:PlanarConformalMesh) || Base.include(DiffMoM,joinpath(@__DIR__,"../src/planar/PlanarConformal.jl"))

function conformal_line_mesh(a,b,nx,ny;interface=1,nonuniform=false)
    xs=nonuniform ? a.*[0.,.1,.3,.7,1.] : collect(range(0.,a;length=nx+1))
    ys=collect(range(b/3,2b/3;length=ny+1))
    vertices=hcat(([x,y] for y in ys for x in xs)...)
    ids(i,j)=i+1+(nx+1)*j
    triangles=hcat(([ids(i,j),ids(i+1,j),ids(i,j+1)] for j in 0:ny-1 for i in 0:nx-1)...,
        ([ids(i+1,j+1),ids(i,j+1),ids(i+1,j)] for j in 0:ny-1 for i in 0:nx-1)...)
    PlanarConformalMesh(vertices,triangles;interfaces=interface)
end

@testset "conformal polygon meshing and independent edge/interior sizing" begin
    polygon=[0. 1. 1. 0.;.375 .25 .75 .625].*1e-3
    mesh=planar_conformal_mesh(polygon;edge_size=.08e-3,interior_size=.3e-3,edge_band=.04e-3)
    @test sum(mesh.areas)≈.375e-6 rtol=2e-14
    @test all(any(mesh.vertices[:,i]==polygon[:,j] for i in axes(mesh.vertices,2)) for j in 1:4)
    counts=Dict{Tuple{Int,Int},Int}()
    for t in axes(mesh.triangles,2),j in 1:3
        a,b=mesh.triangles[j,t],mesh.triangles[mod1(j+1,3),t]
        key=minmax(a,b);counts[key]=get(counts,key,0)+1
    end
    boundary=Float64[];interior=Float64[]
    for ((a,b),count) in counts
        length=norm(mesh.vertices[:,a]-mesh.vertices[:,b])
        if count==1
            push!(boundary,length)
            for vertex in (a,b)
                @test minimum(DiffMoM._planar_conformal_segment_distance(mesh.vertices[1,vertex],mesh.vertices[2,vertex],
                    polygon[:,j],polygon[:,mod1(j+1,4)]) for j in 1:4)<1e-17
            end
        else
            push!(interior,length)
        end
    end
    @test maximum(boundary)<=.08e-3*(1+1e-13)
    @test maximum(interior)<=.3e-3*(1+1e-13)
    @test maximum(interior)>1.5maximum(boundary)
    # Independent shoelace oracle for a concave polygon.
    concave=[0. 1. 1. .4 .4 0.;0. 0. .4 .4 1. 1.].*1e-3
    cm=planar_conformal_mesh(concave;edge_size=.2e-3,interior_size=.4e-3)
    @test sum(cm.areas)≈.64e-6 rtol=1e-14
    clockwise=planar_conformal_mesh(concave[:,end:-1:1];edge_size=.2e-3,interior_size=.4e-3)
    @test sum(clockwise.areas)≈sum(cm.areas) rtol=1e-14
    @test_throws ArgumentError planar_conformal_mesh([0. 1. 0. 1.;0. 1. 1. 0.];edge_size=.1,interior_size=.3)
    @test_throws ArgumentError planar_conformal_mesh(polygon;edge_size=.1e-3,interior_size=.05e-3)
    @test_throws ArgumentError planar_conformal_mesh(polygon;edge_size=.01e-3,interior_size=.1e-3,max_triangles=3)
    @test_throws ArgumentError planar_conformal_mesh(polygon;edge_size=.1e-3,interior_size=.3e-3,max_bytes=1)
end

function conformal_gauss_weights(prob,kx,ky;n=36)
    beta=[j/sqrt(4j*j-1) for j in 1:n-1]
    q=eigen(SymTridiagonal(zeros(n),beta));nodes=(q.values.+1)./2;weights=q.vectors[1,:].^2
    nb=length(prob.basis.width);te=zeros(nb);tm=zeros(nb)
    for b in 1:nb,half in 1:2
        prob.basis.triangles[half,b]==0 && continue
        t,xvalues,yvalues=DiffMoM._planar_conformal_half_values(prob,b,half)
        vertices=prob.mesh.vertices[:,prob.mesh.triangles[:,t]]
        for (i,u) in enumerate(nodes),(j,v) in enumerate(nodes)
            lambda=((1-u)*(1-v),u,(1-u)*v)
            x=sum(lambda[k]*vertices[1,k] for k in 1:3);y=sum(lambda[k]*vertices[2,k] for k in 1:3)
            fx=sum(lambda[k]*xvalues[k] for k in 1:3);fy=sum(lambda[k]*yvalues[k] for k in 1:3)
            px,py=prob.sidewalls===WALL_PEC ?
                (fx*cos(kx*x)*sin(ky*y),fy*sin(kx*x)*cos(ky*y)) :
                (fx*sin(kx*x)*cos(ky*y),fy*cos(kx*x)*sin(ky*y))
            measure=2prob.mesh.areas[t]*(1-u)*weights[i]*weights[j]
            te[b]+=measure*(ky*px-kx*py);tm[b]+=measure*(kx*px+ky*py)
        end
    end
    return te,tm
end

function conformal_constant_x(prob)
    coefficients=zeros(ComplexF64,length(prob.basis.width))
    for b in eachindex(coefficients)
        t=prob.basis.triangles[1,b];a,c=prob.basis.edges[:,b];ids=prob.mesh.triangles[:,t]
        # Counterclockwise triangle edges have their outward normal on
        # the right. The plus RWG half has unit outward normal current.
        j=only(j for j in 1:3 if Set((ids[j],ids[mod1(j+1,3)]))==Set((a,c)))
        p,q=ids[j],ids[mod1(j+1,3)]
        normalx=(prob.mesh.vertices[2,q]-prob.mesh.vertices[2,p])/prob.basis.width[b]
        coefficients[b]=(prob.basis.triangles[2,b]==0 ? -1 : 1)*normalx
    end
    coefficients
end

@testset "genuine conformal geometry and normal continuity" begin
    a,b=4e-3,3e-3;mesh=conformal_line_mesh(a,b,4,2;nonuniform=true)
    stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3),PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,a,b)
    ports=[PlanarConformalPort(1,:west,(b/3,2b/3)),PlanarConformalPort(1,:east,(b/3,2b/3))]
    prob=PlanarConformalProblem(stack,mesh,ports);G=DiffMoM
    @test sum(mesh.areas)≈a*b/3 rtol=1e-14
    @test length(unique(mesh.areas))>=3
    @test all(>(0),mesh.areas)
    @test count(>(0),prob.basis.port)==4
    coefficients=conformal_constant_x(prob)
    values=zeros(ComplexF64,2,3,length(mesh.interfaces))
    for bb in eachindex(coefficients),half in 1:2
        prob.basis.triangles[half,bb]==0 && continue
        t,x,y=G._planar_conformal_half_values(prob,bb,half)
        values[1,:,t].+=coefficients[bb].*x;values[2,:,t].+=coefficients[bb].*y
    end
    @test values[1,:,:]≈ones(3,length(mesh.interfaces)) atol=2e-15
    @test norm(values[2,:,:])<2e-15
    # A single shared-edge function has opposite outgoing normals and
    # equal physical normal current across both triangular faces.
    for bb in eachindex(coefficients)
        prob.basis.triangles[2,bb]==0 && continue
        a1,c1=prob.basis.edges[:,bb];mid=(mesh.vertices[:,a1]+mesh.vertices[:,c1])/2
        tangent=(mesh.vertices[:,c1]-mesh.vertices[:,a1])/prob.basis.width[bb]
        normal=[tangent[2],-tangent[1]];normalvalues=Float64[]
        for half in 1:2
            t,x,y=G._planar_conformal_half_values(prob,bb,half);ids=mesh.triangles[:,t]
            ia=findfirst(==(a1),ids);ic=findfirst(==(c1),ids)
            push!(normalvalues,dot(normal,[(x[ia]+x[ic])/2,(y[ia]+y[ic])/2]))
        end
        @test normalvalues[1]≈normalvalues[2] atol=1e-14
        @test abs(normalvalues[1])≈1 atol=1e-14
    end
    @test_throws ArgumentError PlanarConformalMesh([0. 1. 2.;0. 0. 0.],reshape([1,2,3],3,1))
    @test_throws ArgumentError PlanarConformalMesh([0. 1. 0.;0. 0. 1.],[1 1;2 3;3 2])
    @test_throws ArgumentError PlanarConformalMesh([0. 1. 0. .5 .5 .8;0. 0. 1. 0. .5 .5],[1 4;2 5;3 6])
    @test_throws ArgumentError PlanarConformalMesh([0. 1. 0.;0. 0. 1.],reshape([1,2,3],3,1);max_bytes=1)
    @test_throws ArgumentError PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(b/3+.01b,2b/3))])
    @test_throws ArgumentError PlanarConformalPort(1,:west,(0.,1.);z0=-1.)
end

@testset "genuine triangle vector modal integrals and weighted Gram" begin
    a,b=4e-3,3e-3;mesh=conformal_line_mesh(a,b,4,2;nonuniform=true)
    stack=PlanarStackup([PlanarLayer(2.,1.,.4e-3),PlanarLayer(1.,1.,1e-3)],TERM_GND,TERM_GND,a,b)
    ports=[PlanarConformalPort(1,:west,(b/3,2b/3)),PlanarConformalPort(1,:east,(b/3,2b/3))]
    for wall in (WALL_PEC,WALL_PMC)
        prob=PlanarConformalProblem(stack,mesh,ports;sidewalls=wall)
        for (m,n) in ((0,0),(1,0),(0,1),(1,2),(17,13))
            te,tm=DiffMoM._planar_conformal_weights(prob,m*pi/a,n*pi/b)
            teg,tmg=conformal_gauss_weights(prob,m*pi/a,n*pi/b)
            @test te≈teg rtol=2e-11 atol=1e-13
            @test tm≈tmg rtol=2e-11 atol=1e-13
        end
        nb=length(prob.basis.width);zs=collect(range(.1,1.;length=length(mesh.interfaces)))
        loss=zeros(ComplexF64,nb,nb)
        DiffMoM._planar_conformal_loss_entries((p,q,value)->(loss[p,q]+=value),prob,zs)
        @test loss≈transpose(loss) atol=1e-20
        @test minimum(eigvals(Hermitian(-loss)))>0
        x=conformal_constant_x(prob)
        @test real(transpose(x)*loss*x)≈-sum(mesh.areas.*zs) rtol=1e-14
        # Aggregate RWG transforms of constant x current agree with an
        # independently integrated exact rectangular footprint.
        kx,ky=3pi/a,4pi/b;te,tm=DiffMoM._planar_conformal_weights(prob,kx,ky)
        integralx=wall===WALL_PEC ? sin(kx*a)/kx : (1-cos(kx*a))/kx
        integraly=wall===WALL_PEC ? (cos(ky*b/3)-cos(ky*2b/3))/ky : (sin(ky*2b/3)-sin(ky*b/3))/ky
        @test sum(te.*x)≈ky*integralx*integraly atol=3e-18
        @test sum(tm.*x)≈kx*integralx*integraly atol=3e-18
    end
end

@testset "conformal lossy line physical DC and reciprocal response" begin
    a,b=4e-3,3e-3;stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3),PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,a,b)
    ports=[PlanarConformalPort(1,:west,(b/3,2b/3)),PlanarConformalPort(1,:east,(b/3,2b/3))]
    responses=[]
    for variable in (false,true)
        mesh=conformal_line_mesh(a,b,4,2;nonuniform=variable);prob=PlanarConformalProblem(stack,mesh,ports)
        result=solve_planar_conformal(prob,1e6;mx=28,my=24,surface_zs=.1)
        # Differential impedance tends to rho_sheet*length/width in DC.
        impedance=2/(result.y[1,1]-result.y[1,2])
        @test real(impedance)≈.1*a/(b/3) rtol=2e-5
        @test result.y≈transpose(result.y) rtol=1e-9
        @test opnorm(result.s)<=1+1e-9
        @test maximum(result.relative_residuals)<1e-8
        maps=planar_conformal_current_maps(result;voltages=[1.,-1.])
        @test length(maps)==length(mesh.interfaces)
        @test sum(m.integrated_current[1] for m in maps)≈a*(result.y*[1.,-1.])[1] rtol=1e-5
        push!(responses,result.y)
        @test_throws ArgumentError solve_planar_conformal(prob,1e6;mx=28,my=24,max_bytes=1)
    end
    @test responses[1]≈responses[2] rtol=2e-5
end

@testset "physical interior and diagonal shared-edge voltage sources" begin
    a,b=2e-3,1e-3;rho=.1
    stack=PlanarStackup([PlanarLayer(1.,1.,.4e-3),PlanarLayer(1.,1.,.6e-3)],TERM_GND,TERM_GND,a,b)
    vertices=[0. a a 0.;0. 0. b b]
    mesh=PlanarConformalMesh(vertices,[1 1;2 3;3 4])
    wallports=[PlanarConformalPort(1,:west,(0.,b)),PlanarConformalPort(1,:east,(0.,b))]
    cut=((0.,0.),(a,b))
    for polarity in (-1,1)
        internal=PlanarConformalPort(1,:internal,cut;polarity)
        prob=PlanarConformalProblem(stack,mesh,vcat(wallports,[internal]))
        @test count(==(3),prob.basis.port)==1
        p=only(findall(==(3),prob.basis.port))
        @test prob.basis.triangles[2,p]!=0
        @test prob.basis.width[p]≈hypot(a,b) rtol=1e-14
        result=solve_planar_conformal(prob,1e6;mx=28,my=24,surface_zs=rho)
        # Independent resistive path with an ideal differential voltage
        # source inserted across the oriented diagonal cut. Uniform x
        # current is exactly representable by the genuine triangle basis.
        incidence=[1.,-1.,-Float64(polarity)]
        @test real(inv(result.y[1,1]))≈rho*a/b rtol=2e-6
        @test real.(result.y)/real(result.y[1,1])≈incidence*transpose(incidence) rtol=1e-9
        @test result.y≈transpose(result.y) rtol=1e-11
        @test opnorm(result.s)<=1+1e-12
        fast=solve_planar_conformal(prob,1e9;method=:ufft,nx=2,ny=1,
            mx=19,my=21,surface_zs=rho,memory=20,rtol=1e-10)
        dense=solve_planar_conformal(prob,1e9;mx=19,my=21,surface_zs=rho)
        @test fast.y≈dense.y rtol=2e-9
        @test maximum(fast.relative_residuals)<=1e-10
    end
    reversed=PlanarConformalPort(1,:internal,reverse(cut))
    reverseprob=PlanarConformalProblem(stack,mesh,vcat(wallports,[reversed]))
    rp=solve_planar_conformal(reverseprob,1e6;mx=28,my=24,surface_zs=rho)
    incidence=[1.,-1.,1.]
    @test real(inv(rp.y[1,1]))≈rho*a/b rtol=2e-6
    @test real.(rp.y)/real(rp.y[1,1])≈incidence*transpose(incidence) rtol=1e-9
    @test_throws ArgumentError PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:internal,((0.,0.),(a,0.)))])
    @test_throws ArgumentError PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:internal,((0.,0.),(a/2,b/2)))])
    @test_throws ArgumentError PlanarConformalPort(1,:internal,((0.,0.),(0.,0.)))
end
