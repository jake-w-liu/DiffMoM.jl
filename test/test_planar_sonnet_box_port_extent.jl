module NativeSonnetBoxPortExtentTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
const fixture=joinpath(@__DIR__,"fixtures/native_plain_box_port_extent")
const observations=NamedTuple[]
function changed_project(p;polygons=p.polygons,ports=p.ports)
    SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,p.box,p.layers,
        p.metals,p.top,p.bottom,polygons,ports,p.variables,p.components,p.sweeps,p.records)
end
function port_variant(p,kind,number)
    ports=[SonnetPortSpec(kind,ps.polygon,ps.edge,i==1 ? number : ps.number,
        ps.values,ps.records) for (i,ps) in enumerate(p.ports)]
    changed_project(p;ports)
end
function error_from(f)
    try
        f()
        nothing
    catch err
        err
    end
end
@testset "Native literal box-port edges stay within the physical box" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    files=Set(replace(relpath(joinpath(d,f),fixture),'\\'=>'/')
        for (d,_,fs) in walkdir(fixture) for f in fs if f!="sha256.toml")
    @test Set(keys(hashes))==files
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    before=TOML.parsefile(joinpath(fixture,"original_before_comparison.toml"))
    followup=TOML.parsefile(joinpath(fixture,"conformal_followup_comparison.toml"))
    @test before["source_unchanged"] && followup["source_unchanged"]
    @test bytes2hex(sha256(read(joinpath(fixture,"source_before/PlanarSonnetIO.jl"))))==
        before["source_before"]["src\\planar\\PlanarSonnetIO.jl"]
    cases=TOML.parsefile(joinpath(fixture,"index.toml"))["cases"]
    @test length(cases)==6
    for row in cases
        name=row["name"];folder=joinpath(fixture,"cases",name)
        p=read_sonnet_project(joinpath(folder,"project.son"))
        original=deepcopy(p);valid=row["native_status"]=="ACCEPT"
        metadata=TOML.parsefile(joinpath(folder,"native/metadata.toml"))
        @test metadata["source_sha256"]==bytes2hex(sha256(read(p.source)))
        @test metadata["process_success"]==valid
        @test only(filter(c->c["case"]==name,before["cases"]))["raster_status"]=="ACCEPT"
        @test only(filter(c->c["case"]==name,followup["cases"]))["conformal_status"]==
            (valid ? "ACCEPT" : "REJECT")
        if valid
            @test metadata["touchstone_selected_log_checks"]["native_raw.s2p"]["status"]=="PASS"
            problem=sonnet_planar_problem(p;freq=1e9)
            @test length(problem.ports)==2
            native=only(planar_read_touchstone(joinpath(folder,"native/native_raw.s2p")).s)
            # Use independent direct modal assembly for these corner contacts;
            # FFT assembly's rounding was separately measured near the residual gate.
            result=solve_sonnet_project(p,1e9;raw=true,method=:dense,
                mx=128,my=128,retain_matrix=true)
            raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
            for i in eachindex(raw.problem.basis.port)
                port=raw.problem.basis.port[i];iszero(port) && continue
                rhs[i,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[i]
            end
            error=maximum(abs,result.s-native);residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
            @test error<=.005 && residual<=1e-9
            @test maximum(abs,result.s-transpose(result.s))<=1e-10 && opnorm(result.s)<=1+1e-9
            push!(observations,(case=name,full_s_error=error,original_voltage_residual=residual))
            for kind in (:box,:std)
                @test length(sonnet_planar_problem(port_variant(p,kind,0);freq=1e9).ports)==1
                @test length(sonnet_planar_problem(port_variant(p,kind,1);freq=1e9).ports)==2
            end
        else
            @test occursin("partially or entirely outside of the box",
                read(joinpath(folder,"native/engine_stderr.log"),String))
            for kind in (:box,:std),number in (0,1)
                variant=port_variant(p,kind,number)
                for route in (q->sonnet_planar_problem(q;freq=1e9),
                    q->solve_sonnet_project(q,1e9;raw=true))
                    err=error_from(()->route(variant))
                    @test err isa ArgumentError && occursin("box-port edge",sprint(showerror,err))
                end
            end
        end
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,original.polygons))
        @test all(a.values==b.values for (a,b) in zip(p.ports,original.ports))
    end
end
function extent_batch(vertices,edges,a,b)
    count=0
    for edge in edges
        count+=DiffMoM._sonnet_box_port_edge_inside(vertices,edge,a,b,32,32)
    end
    count
end
@testset "Box-port extent boundaries, open polygons and allocation" begin
    for scale in (1.,1e-3,1e100,1e-100)
        vertices=[0. scale scale 0. 0.;0. 0. scale scale 0.]
        for v in (vertices,vertices[:,1:4]),edge in 0:3
            @test DiffMoM._sonnet_box_port_edge_inside(v,edge,scale,scale,32,32)
            for coordinate in 1:2,value in (-scale/32,scale+scale/32,NaN,Inf,-Inf)
                outside=copy(v);outside[coordinate,edge+1]=value
                @test !DiffMoM._sonnet_box_port_edge_inside(outside,edge,scale,scale,32,32)
            end
            for coordinate in 1:2,value in (-scale/128,scale+scale/128)
                near=copy(v);near[coordinate,edge+1]=value
                @test DiffMoM._sonnet_box_port_edge_inside(near,edge,scale,scale,32,32)
            end
        end
    end
    vertices=[0. 1. 1. 0. 0.;0. 0. 1. 1. 0.];edges=collect(0:3)
    extent_batch(vertices,edges,1.,1.)
    @test (@allocated extent_batch(vertices,edges,1.,1.))==0
    # Clipping metal without a port on its outside edge remains supported.
    p=read_sonnet_project(joinpath(fixture,"cases/y_inside/project.son"))
    poly=only(p.polygons)
    # Use a separate polygon so the native referenced edges remain unchanged.
    extension=SonnetPolygon(:sheet,poly.level,poly.material,2,
        [0. 1e-3 1e-3 0. 0.;-.0625e-3 -.0625e-3 .0625e-3 .0625e-3 -.0625e-3],
        poly.target,poly.technology,poly.flags)
    @test length(sonnet_planar_problem(changed_project(p;polygons=[poly,extension]);freq=1e9).ports)==2
end
end
