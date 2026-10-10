module SonnetEndpointCurrentTests
using DiffMoM,Test,LinearAlgebra,SHA,JSON

const FIXTURE=joinpath(@__DIR__,"fixtures","native_via_endpoints")
project(name)=read_sonnet_project(joinpath(FIXTURE,name,"project.son"))
function native_s(name,frequency)
    reference=planar_read_touchstone(joinpath(FIXTURE,name,"native_raw.s2p"))
    reference.s[only(findall(==(frequency),reference.frequencies))]
end
chain_b(s)=50*((1+s[1,1])*(1+s[2,2])-s[1,2]*s[2,1])/(2s[2,1])
function physical_residuals(raw)
    basis=raw.problem.basis
    weights=[basis.width[b]*(basis.kind[b]==DiffMoM._BASIS_VIA_T ? .5 : 1.) for b in axes(raw.currents,1)]
    rhs=zeros(ComplexF64,size(raw.currents))
    for b in axes(rhs,1)
        q=basis.port[b];q==0 && continue
        port=raw.problem.ports[q]
        sign=port.polarity*(port.wall in (:east,:north,:volume_east,:volume_north,:terminal_x_hi,:terminal_y_hi) ? -1 : 1)
        rhs[b,q]=-sign*weights[b]
    end
    [norm((raw.z_mom*raw.currents[:,q]-rhs[:,q])./weights)/norm(rhs[:,q]./weights) for q in axes(rhs,2)]
end
function rf_gates(actual,native;pec=false,conformal=false)
    s=Matrix(actual.s)
    @test maximum(abs,s-native)<=.06
    if pec
        # Full ABCD B is valid for asymmetric reciprocal networks. The
        # symmetric100*S11/S21 proxy can falsely report loss for PEC.
        @test abs(real(chain_b(s))-real(chain_b(native)))<=1e-8
    else
        @test abs(real(100s[1,1]/s[2,1])/real(100native[1,1]/native[2,1])-1)<=.05
    end
    residuals=conformal ? actual.raw.relative_residuals : physical_residuals(actual.raw)
    @test maximum(residuals)<=1e-9
    @test maximum(svdvals(s))<=1+1e-9
    @test maximum(abs,s-transpose(s))<=1e-9
end

@testset "endpoint original native provenance and schema controls" begin
    manifest=JSON.parsefile(joinpath(FIXTURE,"manifest.json"))
    @test manifest["engine_version"]=="18.53-Lite (64-bit Windows)"
    @test manifest["engine_sha256"]=="c25fd7deb8d71b4b3df154088193b3ee6d1a963279d3dc8c95eab82e7f1c91ac"
    for file in manifest["files"]
        @test bytes2hex(sha256(read(joinpath(FIXTURE,file["path"]))))==file["sha256"]
    end
    for name in manifest["cases"]
        capture=JSON.parsefile(joinpath(FIXTURE,name,"capture.json"))
        @test capture["exit_code"]==0 && capture["completion_marker"] && capture["zero_error_marker"]
        @test capture["source_sha256_before"]==capture["source_sha256_after"]==bytes2hex(sha256(read(joinpath(FIXTURE,name,"project.son"))))
    end
    for pads in ("no_pads","native_pads")
        a=planar_read_touchstone(joinpath(FIXTURE,"rpv_inactive_wall_"*pads,"native_raw.s2p"))
        b=planar_read_touchstone(joinpath(FIXTURE,"rpv_solid_"*pads,"native_raw.s2p"))
        @test a.frequencies==b.frequencies && a.s==b.s
    end
    a=planar_read_touchstone(joinpath(FIXTURE,"pec_native_pads","native_raw.s2p"))
    b=planar_read_touchstone(joinpath(FIXTURE,"pec_explicit_pec_pads","native_raw.s2p"))
    @test a.s==b.s
    # A hollow rectangular wall saturates at half the shorter side: native
    # treats wall>=min(width,breadth)/2 as the complete solid volume fill.
    saturated=planar_read_touchstone(joinpath(FIXTURE,"volume_hollow_saturated_native_pads","native_raw.s2p"))
    solid=planar_read_touchstone(joinpath(FIXTURE,"volume_solid_native_pads","native_raw.s2p"))
    thin=planar_read_touchstone(joinpath(FIXTURE,"volume_hollow_native_pads","native_raw.s2p"))
    @test saturated.frequencies==solid.frequencies && saturated.s==solid.s
    @test maximum(abs,saturated.s[1]-thin.s[1])>1e-3
end

@testset "endpoint partial contacts and finite native pads at RF" begin
    names=("volume_solid_no_pads","volume_hollow_no_pads","pec_no_pads","rpv_solid_no_pads",
        "volume_solid_native_pads","volume_hollow_native_pads","rpv_solid_native_pads","pec_native_pads",
        "volume_hollow_saturated_native_pads")
    for name in names,frequency in (1e9,1e10)
        p=project(name);coordinates=[copy(q.vertices) for q in p.polygons]
        result=solve_sonnet_project(p,frequency;raw=true,grid=(20,20),mx=160,my=160,method=:dense_fft)
        rf_gates(result,native_s(name,frequency);pec=startswith(name,"pec_"))
        @test all(q.vertices==v for (q,v) in zip(p.polygons,coordinates))
    end
    # Real triangle consumers exercise the endpoint geometry and providers.
    for name in ("volume_solid_no_pads","volume_hollow_native_pads","rpv_solid_no_pads","pec_no_pads")
        result=solve_sonnet_conformal(project(name),1e9;raw=true,grid=(20,20),edge_size=50e-6,
            interior_size=100e-6,max_triangles=10000,mx=160,my=160)
        rf_gates(result,native_s(name,1e9);pec=startswith(name,"pec_"),conformal=true)
    end
end

@testset "shared endpoint film order and grounded dissipation" begin
    results=[]
    for name in ("stacked_forward","stacked_reverse")
        result=solve_sonnet_project(project(name),1e9;raw=true,grid=(20,20),mx=160,my=160,method=:dense_fft)
        native=native_s(name,1e9);yn=planar_s_to_y(native,50.)
        gn=(yn+yn')/2;ga=(result.y+result.y')/2
        @test maximum(abs,result.s-native)<=.06
        @test norm(ga-gn)/norm(gn)<=.05
        @test maximum(physical_residuals(result.raw))<=1e-9
        @test maximum(svdvals(result.s))<=1+1e-9
        @test maximum(abs,result.s-transpose(result.s))<=1e-9
        push!(results,result)
    end
    @test maximum(abs,results[1].s-results[2].s)<=1e-9
    # Compare matching native40actual cells with independently sized sheets.
    result=solve_sonnet_conformal(project("stacked_halfcells_80"),1e9;raw=true,grid=(40,40),
        edge_size=25e-6,interior_size=50e-6,max_triangles=10000,mx=320,my=320)
    native=native_s("stacked_halfcells_80",1e9);yn=planar_s_to_y(native,50.)
    gn=(yn+yn')/2;ga=(result.y+result.y')/2
    @test maximum(abs,result.s-native)<=.06
    @test norm(ga-gn)/norm(gn)<=.05
    @test maximum(result.raw.relative_residuals)<=1e-9
    @test maximum(svdvals(result.s))<=1+1e-9
    @test maximum(abs,result.s-transpose(result.s))<=1e-9
end

@testset "Float64 geometry preserves caller precision and source" begin
    vertices=[DiffMoM._P2(0.,0.),DiffMoM._P2(.001,.00071),DiffMoM._P2(.00023,.001)]
    v,b=planar_normalize_polygon(vertices);polygons=[PlanarPolygon("triangle",1,"pec","",v,b)]
    outside=precision(BigFloat);meshes=[]
    for bits in (32,256,8192)
        setprecision(BigFloat,bits) do
            mesh=planar_conformal_mesh(polygons;edge_size=.002,interior_size=.002,max_bytes=1024^2)
            push!(meshes,mesh)
            @test DiffMoM._planar_orient2d(0.,0.,.001,.001,.001,nextfloat(.001))>0
            @test precision(BigFloat)==bits
        end
    end
    @test meshes[1].vertices==meshes[2].vertices==meshes[3].vertices
    @test polygons[1].vertices==vertices && precision(BigFloat)==outside
end

function rejection_allocation(backgrounds)
    function attempt()
        try
            DiffMoM._sonnet_endpoint_overlay(backgrounds,PlanarPolygon[],1024)
            error("insufficient overlay budget accepted")
        catch e
            e isa ArgumentError || rethrow()
        end
    end
    attempt()
    @allocated attempt()
end
@testset "endpoint budget checks precede owned staging" begin
    grid=CellGrid(.001,.001,4,2000);mask=trues(4,2000)
    @test length(DiffMoM._sonnet_endpoint_rectangles(mask,grid,144000))==2000
    @test_throws ArgumentError DiffMoM._sonnet_endpoint_rectangles(mask,grid,143999)
    p=project("volume_solid_no_pads");poly=only(filter(q->q.kind===:via,p.polygons))
    vertices=[DiffMoM._P2(poly.vertices[1,k],poly.vertices[2,k]) for k in 1:4]
    backgrounds=NamedTuple[(;poly,level=1,vertices) for _ in 1:10000]
    @test rejection_allocation(backgrounds)<65536
end

@testset "translated native physical area and material components" begin
original=read_sonnet_project(joinpath(FIXTURE,"volume_solid_no_pads","project.son"))
original_coordinates=[copy(q.vertices) for q in original.polygons];outside=precision(BigFloat)
for offset in (0.,.001,.01), loss in (".25","1e-308","1e-320")
    box=copy(original.box);extent=.001+offset;box[2]=box[3]=string(extent/original.length_scale)
    polygons=SonnetPolygon[]
    for q in original.polygons
        v=q.vertices.+offset
        if q.kind===:sheet
            for k in axes(v,2)
                q.vertices[1,k]==0. && (v[1,k]=0.)
                q.vertices[1,k]==.001 && (v[1,k]=extent)
            end
        end
        push!(polygons,SonnetPolygon(q.kind,q.level,q.material,q.id,v,q.target,q.technology,q.flags))
    end
    record=only(original.metals);metal=[record[1],record[2],"VOL",loss,"SOLID","0","SRVY"]
    p=SonnetProject(original.source,original.units,original.length_scale,original.frequency_scale,box,original.layers,[metal],original.top,original.bottom,polygons,original.ports,original.variables,original.components,original.sweeps,original.records)
    grid=(round(Int,extent/50e-6),round(Int,extent/50e-6));before=[copy(q.vertices) for q in p.polygons]
        model=sonnet_planar_problem(p;freq=1e9,grid,_materials=true,_details=true)
        via=only(filter(q->q.kind===:via,p.polygons))
        sigma,reference,film=setprecision(BigFloat,32768) do
            v=BigFloat.(via.vertices);n=size(v,2)-1
            area=abs(sum(v[1,k]*v[2,mod1(k+1,n)]-v[1,mod1(k+1,n)]*v[2,k] for k in 1:n))/2
            width=maximum(v[1,:])-minimum(v[1,:]);breadth=maximum(v[2,:])-minimum(v[2,:])
            depth=area/(width+breadth+sqrt((width+breadth)^2-4area));sigma=inv(BigFloat(parse(Float64,loss))*depth)
            gr=model.problem.grid;mask=only(model.problem.vias).uni
            mesh=BigFloat(count(mask))*BigFloat(gr.dx)*BigFloat(gr.dy)
            q=depth*sqrt(BigFloat(pi)*BigFloat(DiffMoM._MU0)*BigFloat(1e9)*sigma);u=complex(q,q)
            effective=sigma*area/(mesh*(u/tanh(u)))
            gamma=sqrt(complex(zero(BigFloat),2BigFloat(pi)*BigFloat(1e9)*BigFloat(DiffMoM._MU0)*sigma))
            zs=gamma/(sigma*tanh(gamma*BigFloat(.0001)/2))
            Float64(sigma),ComplexF64(effective),ComplexF64(zs)
        end
        actual=only(model.via_sigma)
        axial_errors=[abs(real(actual)/real(reference)-1),abs(imag(actual)/imag(reference)-1)]
        films=unique(z for matrix in model.sheet_zs for z in matrix if !iszero(z))
        @assert length(films)==1
        film_errors=[abs(real(only(films))/real(film)-1),abs(imag(only(films))/imag(film)-1)]
        @test axial_errors[1]<=2e-12
        @test axial_errors[2]<=2e-12
        @test film_errors[1]<=2e-12
        @test film_errors[2]<=2e-12
        @test all(q.vertices==v for (q,v) in zip(p.polygons,before))
end
@test precision(BigFloat)==outside
@test all(q.vertices==v for (q,v) in zip(original.polygons,original_coordinates))
end

@testset "rectangular material depth retains exact bounding fill" begin
p=read_sonnet_project(joinpath(FIXTURE,"volume_solid_no_pads","project.son"))
original=only(filter(q->q.kind===:via,p.polygons));before=copy(original.vertices);outside=precision(BigFloat)
metal=copy(p.metals[original.material+1]);metal[4]="0.25"
length(metal)==6 ? push!(metal,"SRVY") : (metal[end]="SRVY")
for offset in (0.,.001,.01,.1,1.,10.,100.,1000.)
    vertices=original.vertices.+offset
    poly=SonnetPolygon(original.kind,original.level,original.material,original.id,vertices,original.target,original.technology,original.flags)
    expected=setprecision(BigFloat,32768) do
        v=BigFloat.(vertices);n=size(v,2)
        area=abs(sum(v[1,k]*v[2,mod1(k+1,n)]-v[1,mod1(k+1,n)]*v[2,k] for k in 1:n))/2
        width=maximum(v[1,:])-minimum(v[1,:]);breadth=maximum(v[2,:])-minimum(v[2,:])
        depth=area/(width+breadth+sqrt((width+breadth)^2-4area))
        Float64(inv(BigFloat(.25)*depth))
    end
    actual=DiffMoM._sonnet_via_endpoint_sigma(p,poly,metal,.25,"SRVY")
    @test isfinite(actual) && abs(actual/expected-1)<=2e-12
end
@assert original.vertices==before && precision(BigFloat)==outside
end

function fine_ground_import(p)
    sonnet_conformal_layout(p;freq=1e9,grid=(40,40),edge_size=25e-6,
        interior_size=50e-6,max_triangles=10000)
end
@testset "genuine refinement avoids captured coordinate boxing" begin
    p=project("stacked_halfcells_80");before=[copy(q.vertices) for q in p.polygons]
    fine_ground_import(p)
    measured=@timed fine_ground_import(p)
    @test measured.bytes<3*1024^3
    mesh=measured.value.layout.problem.conformal.mesh
    # Original pre-fix geometry, matching both qualified Julia versions.
    @test bytes2hex(sha256(reinterpret(UInt8,vec(mesh.interfaces))))=="67a6faed14c52a018ad93c07d9144a3d3ec71fca19ec622fb5f2b8a9472d60c3"
    @test bytes2hex(sha256(reinterpret(UInt8,vec(mesh.triangles))))=="02710c660861b79909d2737f7f831ba07ba84e1336f0f9d3255bff34736b1b7d"
    @test bytes2hex(sha256(reinterpret(UInt8,vec(mesh.vertices))))=="3e5cc164495404b4196ffdeb4f15e64ac0e7e865cebc377669792d2b4d166e3e"
    @test bytes2hex(sha256(reinterpret(UInt8,vec(mesh.areas))))=="70685987547787d6e1dd2b2c84c134c429ecbc28d282fb4ffb8f88e53c531fbf"
    @test all(q.vertices==v for (q,v) in zip(p.polygons,before))
end
end
