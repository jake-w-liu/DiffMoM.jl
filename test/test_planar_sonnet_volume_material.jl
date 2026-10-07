module SonnetVolumeMaterialTests
using DiffMoM, Test, LinearAlgebra, SHA, JSON

const FIXTURE=joinpath(@__DIR__,"fixtures","native_volume_sheet_selector")
cross2(a,b)=a[1]*b[2]-a[2]*b[1]

# Independent inward shifted-line construction, distinct from the production
# perimeter/corner formula. Closed and open vertex lists describe the same wall.
function wall_area(vertices,thickness)
    v=BigFloat.(vertices)
    v[:,1]==v[:,end] && (v=v[:,1:end-1])
    n=size(v,2)
    twice=sum(cross2(v[:,k],v[:,mod1(k+1,n)]) for k in 1:n)
    orientation=sign(twice);inner=similar(v)
    for k in 1:n
        previous=mod1(k-1,n);following=mod1(k+1,n)
        e1=v[:,k]-v[:,previous];e2=v[:,following]-v[:,k]
        normal1=orientation*[-e1[2],e1[1]]/norm(e1)
        normal2=orientation*[-e2[2],e2[1]]/norm(e2)
        point1=v[:,k]+thickness*normal1;point2=v[:,k]+thickness*normal2
        determinant=cross2(e1,e2)
        inner[:,k]=iszero(determinant) ? point1 :
            point1+cross2(point2-point1,e2)/determinant*e1
    end
    inside=abs(sum(cross2(inner[:,k],inner[:,mod1(k+1,n)]) for k in 1:n))/2
    return abs(twice)/2-inside
end

function material_project(original,loss,wall,solid;selector="SRVY")
    record=only(original.metals)
    material=solid ? [record[1],record[2],"VOL",loss,"SOLID",wall,selector] :
        [record[1],record[2],"VOL",loss,wall,selector]
    return SonnetProject(original.source,original.units,original.length_scale,original.frequency_scale,
        original.box,original.layers,[material],original.top,original.bottom,original.polygons,
        original.ports,original.variables,original.components,original.sweeps,original.records)
end

function reference_sigma(p,model,frequency,loss,wall,solid;selector="SRVY")
    setprecision(BigFloat,2400) do
        polygon=only(filter(q->q.kind===:via,p.polygons));v=BigFloat.(polygon.vertices)
        n=size(v,2);v[:,1]==v[:,end] && (n-=1)
        area=abs(sum(cross2(v[:,k],v[:,mod1(k+1,n)]) for k in 1:n))/2
        width=maximum(v[1,:])-minimum(v[1,:]);breadth=maximum(v[2,:])-minimum(v[2,:])
        declared=BigFloat(parse(Float64,wall))*BigFloat(p.length_scale)
        depth=solid ? area/(width+breadth+sqrt((width+breadth)^2-4area)) : declared
        conductivity=selector=="SRVY" ? inv(BigFloat(parse(Float64,loss))*depth) :
            selector=="RSVY" ? 100/BigFloat(parse(Float64,loss)) : BigFloat(parse(Float64,loss))
        metal_area=solid ? area : wall_area(polygon.vertices,declared)
        via=only(model.problem.vias)
        mesh=BigFloat(count(via.uni))*BigFloat(model.problem.grid.dx)*BigFloat(model.problem.grid.dy)
        q=depth*sqrt(BigFloat(pi)*BigFloat(DiffMoM._MU0)*BigFloat(frequency)*conductivity)
        u=complex(q,q);transition=q>100 ? u : u/tanh(u)
        return conductivity*metal_area/(mesh*transition)
    end
end

function component_errors(actual,reference)
    setprecision(BigFloat,2400) do
        return abs((BigFloat(real(actual))-real(reference))/real(reference)),
            abs((BigFloat(imag(actual))-imag(reference))/imag(reference))
    end
end

@testset "volume selector captured native provenance" begin
    manifest=JSON.parsefile(joinpath(FIXTURE,"manifest.json"))
    @test manifest["engine_version"]=="18.53-Lite (64-bit Windows)"
    for file in manifest["files"]
        @test bytes2hex(sha256(read(joinpath(FIXTURE,file["path"]))))==file["sha256"]
    end
    for name in ("triangle","ell","diamond")
        hollow=read_sonnet_project(joinpath(FIXTURE,name*"_hollow","project.son"))
        @test only(hollow.metals)[end]=="SRVY"
        sheet=planar_read_touchstone(joinpath(FIXTURE,name*"_solid_SRVY","native_raw.s2p"))
        conductivity=planar_read_touchstone(joinpath(FIXTURE,name*"_solid_CDVY","native_raw.s2p"))
        @test sheet.frequencies==conductivity.frequencies
        @test all(maximum(abs,a-b)<=1e-9 for (a,b) in zip(sheet.s,conductivity.s))
    end
    controls=[planar_read_touchstone(joinpath(FIXTURE,"triangle_solid_wall"*string(w),"native_raw.s2p")) for w in (0,10,20)]
    @test controls[1].s==controls[2].s==controls[3].s
end

@testset "volume sheet selector independent law, range and ownership" begin
    cases=((".1",".001",1e9),(".1",".001",1e300),
        ("1e-320",".001",1e9),("1e-308","1e-317",1e300),("1e-308","5e-324",1e300))
    for name in ("triangle","ell","diamond"),solid in (false,true)
        original=read_sonnet_project(joinpath(FIXTURE,name*"_hollow","project.son"))
        for (loss,wall,frequency) in cases
            p=material_project(original,loss,wall,solid)
            coordinates=copy(only(filter(q->q.kind===:via,p.polygons)).vertices)
            metals=deepcopy(p.metals);bits=precision(BigFloat)
            model=sonnet_planar_problem(p;freq=frequency,grid=(20,20),_materials=true,_details=true)
            expected=reference_sigma(p,model,frequency,loss,wall,solid)
            er,ei=component_errors(only(model.via_sigma),expected)
            @test er<=2e-12
            @test ei<=2e-12
            @test p.metals==metals && precision(BigFloat)==bits &&
                only(filter(q->q.kind===:via,p.polygons)).vertices==coordinates
        end
        for wall in (".001","1e-317")
            p=material_project(original,"0",wall,solid)
            @test only(sonnet_planar_problem(p;freq=1e9,grid=(20,20),_materials=true,_details=true).via_sigma)==Inf
        end
        for (loss,selector) in (("10000","CDVY"),(".01","RSVY"))
            p=material_project(original,loss,".001",solid;selector)
            model=sonnet_planar_problem(p;freq=1e9,grid=(20,20),_materials=true,_details=true)
            expected=reference_sigma(p,model,1e9,loss,".001",solid;selector)
            er,ei=component_errors(only(model.via_sigma),expected)
            @test er<=2e-12
            @test ei<=2e-12
            @test eltype(model.via_sigma)==ComplexF64
        end
    end
end

@testset "volume material precision and import workspace" begin
    limit=1_000_000;outside=precision(BigFloat)
    # SOLID/rectangle recovery owns scalar MPFR buffers without wide vertex
    # arrays. All four material paths must include that scalar workspace.
    for (name,solid) in (("triangle_hollow",false),("triangle_solid_SRVY",true),
            ("rectangle_hollow",false),("rectangle_solid",true))
        original=read_sonnet_project(joinpath(FIXTURE,name,"project.son"))
        p=material_project(original,"1e-320",".01",solid)
        baseline=sonnet_planar_problem(p;freq=1e9,grid=(20,20),_materials=true,_details=true,max_bytes=limit)
        reference=reference_sigma(p,baseline,1e9,"1e-320",".01",solid)
        for method in (:raster,:conformal),bits in (32,8192,1_048_576)
            setprecision(BigFloat,bits) do
                import_model()=method===:raster ?
                    sonnet_planar_problem(p;freq=1e9,grid=(20,20),_materials=true,_details=true,max_bytes=limit) :
                    sonnet_conformal_layout(p;freq=1e9,grid=(20,20),edge_size=1e-4,interior_size=2e-4,max_bytes=limit)
                if bits==1_048_576
                    # Seven converted inputs plus the percent/100 result:
                    # MPFR limbs alone exceed1MB, without headers orgeometry.
                    @test 8*8cld(bits,64)>limit
                    @test_throws ArgumentError import_model()
                else
                    er,ei=component_errors(only(import_model().via_sigma),reference)
                    @test er<=2e-12
                    @test ei<=2e-12
                end
                @test precision(BigFloat)==bits
            end
        end
    end
    @test precision(BigFloat)==outside
end

# The outer dumbbell is simple and all local offset edges keep their direction;
# its narrow neck creates nonlocal inset crossings. The thin-wall law must not
# silently accept that geometry. Thick/topology support remains unfinished.
@testset "volume hollow rejects closing nonlocal neck" begin
    p=read_sonnet_project(joinpath(FIXTURE,"triangle_hollow","project.son"))
    model=sonnet_planar_problem(p;freq=1e10,grid=(20,20),_materials=true,_details=true)
    original=only(filter(q->q.kind===:via,p.polygons))
    outer=1e-5*[0. 4 4 6 6 10 10 6 6 4 4 0;0. 0 1.8 1.8 0 0 4 4 2.2 2.2 4 4]
    polygon=SonnetPolygon(original.kind,original.level,original.material,original.id,
        outer,original.target,original.technology,original.flags)
    before=copy(outer);bits=precision(BigFloat)
    error=try
        DiffMoM._sonnet_volume_polygon_sigma(p,polygon,model.problem.grid,model.problem.stack,
            only(model.problem.vias).uni,1e10,1e6,3e-6/p.length_scale,"CDVY")
        nothing
    catch exception
        exception
    end
    @test error isa ArgumentError
    @test occursin("topology adapter",sprint(showerror,error))
    @test outer==before && precision(BigFloat)==bits
end

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
    return [norm((raw.z_mom*raw.currents[:,q]-rhs[:,q])./weights)/norm(rhs[:,q]./weights) for q in axes(rhs,2)]
end

@testset "volume material native S, loss and physical equation at RF" begin
    names=vcat([name*"_hollow" for name in ("triangle","ell","diamond")],
        ["triangle_solid_wall"*string(wall) for wall in (0,10,20)],
        [shape*"_solid_"*selector for shape in ("triangle","ell","diamond") for selector in ("SRVY","CDVY")])
    for name in names
        directory=joinpath(FIXTURE,name);p=read_sonnet_project(joinpath(directory,"project.son"))
        reference=planar_read_touchstone(joinpath(directory,"native_raw.s2p"))
        for frequency in (1e9,1e10)
            native=reference.s[only(findall(==(frequency),reference.frequencies))]
            actual=solve_sonnet_project(p,frequency;raw=true,grid=(20,20),mx=160,my=160,method=:dense_fft)
            @test maximum(abs,actual.s-native)<=.06
            @test abs(real(100actual.s[1,1]/actual.s[2,1])/real(100native[1,1]/native[2,1])-1)<=.05
            @test maximum(physical_residuals(actual.raw))<=1e-9
            @test maximum(svdvals(actual.s))<=1+1e-6
            @test maximum(abs,actual.s-transpose(actual.s))<=1e-10
        end
    end
end

@testset "volume sheet selector genuine triangle consumer" begin
    for name in ("triangle_hollow","triangle_solid_SRVY")
        directory=joinpath(FIXTURE,name);p=read_sonnet_project(joinpath(directory,"project.son"))
        reference=planar_read_touchstone(joinpath(directory,"native_raw.s2p"));frequency=1e9
        native=reference.s[only(findall(==(frequency),reference.frequencies))]
        actual=solve_sonnet_conformal(p,frequency;raw=true,grid=(20,20),edge_size=50e-6,
            interior_size=100e-6,max_triangles=10_000,mx=160,my=160)
        @test maximum(abs,actual.s-native)<=.06
        @test abs(real(100actual.s[1,1]/actual.s[2,1])/real(100native[1,1]/native[2,1])-1)<=.05
        @test maximum(actual.raw.relative_residuals)<=1e-9
        @test maximum(svdvals(actual.s))^2<=1+1e-9
        @test maximum(abs,actual.s-transpose(actual.s))<=1e-9
    end
end
end
