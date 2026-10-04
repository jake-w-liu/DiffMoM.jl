module NativeSonnetPortAttachmentTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
const fixture=joinpath(@__DIR__,"fixtures/native_box_port_attachment")
const observations=NamedTuple[]
project(name)=read_sonnet_project(joinpath(fixture,"cases",name,"project.son"))
native_s(name,n=2)=only(planar_read_touchstone(joinpath(fixture,"cases",name,"native","native_raw.s$(n)p")).s)
function changed_ports(p,ports)
    SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,p.box,p.layers,p.metals,p.top,p.bottom,
        p.polygons,ports,p.variables,p.components,p.sweeps,p.records)
end
function wall_projection_allocation(p)
    DiffMoM._sonnet_raster_wall_project(p,.001,.001,32,32)
    @allocated DiffMoM._sonnet_raster_wall_project(p,.001,.001,32,32)
end
@testset "Native port attachment uses logical edges and retains annotations" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    files=Set(replace(relpath(joinpath(d,f),fixture),'\\'=>'/')
        for (d,_,fs) in walkdir(fixture) for f in fs if f!="sha256.toml")
    @test Set(keys(hashes))==files
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    for group in ("attachment","variants","unported","clipped")
        before=TOML.parsefile(joinpath(fixture,group,"original_before_comparison.toml"))
        @test before["source_unchanged"]
        for f in ("PlanarSonnetIO.jl","PlanarSonnetGeometryVariables.jl","PlanarSonnetConformal.jl")
            @test bytes2hex(sha256(read(joinpath(fixture,"source_before",f))))==
                before["source_before"]["src\\planar\\"*f]
        end
    end
    cases=TOML.parsefile(joinpath(fixture,"index.toml"))["cases"]
    @test length(cases)==20
    for row in cases
        name=row["name"];folder=joinpath(fixture,"cases",name)
        metadata=TOML.parsefile(joinpath(folder,"native/metadata.toml"))
        valid=row["native_status"]=="ACCEPT"
        @test metadata["process_success"]==valid
        @test metadata["source_sha256"]==bytes2hex(sha256(read(joinpath(folder,"project.son"))))
        if valid
            p=project(name);saved=deepcopy(p)
            for grid in (nothing,(16,16),(64,64))
                @test length(sonnet_planar_problem(p;freq=1e9,grid).ports)==row["nports"]
            end
            @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved.polygons))
            @test all(a.values==b.values for (a,b) in zip(p.ports,saved.ports))
        else
            @test_throws ArgumentError sonnet_planar_problem(project(name);freq=1e9)
        end
    end
    baseline=sonnet_planar_problem(project("attachment__baseline"))
    sizes=(edge_size=.125e-3,interior_size=.25e-3,edge_band=.05e-3)
    for case in ("baseline","off_edge_position","opposite_wall_position")
        name="attachment__"*case;p=project(name);problem=sonnet_planar_problem(p)
        @test native_s(name)==native_s("attachment__baseline")
        @test [q.wall for q in problem.ports]==[:west,:east]
        @test [q.cells for q in problem.ports]==[q.cells for q in baseline.ports]
        c=sonnet_conformal_layout(p;sizes...)
        @test [q.span for q in c.layout.problem.ports]==[(.375e-3,.625e-3),(.375e-3,.625e-3)]
        @test c.geometry_project.ports===p.ports
    end
    p=project("attachment__baseline");ps=first(p.ports)
    invalid=SonnetPortSpec(ps.kind,ps.polygon,4,ps.number,ps.values,ps.records)
    @test_throws ArgumentError sonnet_planar_problem(changed_ports(p,[invalid,last(p.ports)]))
    @test_throws ArgumentError sonnet_conformal_layout(changed_ports(p,[invalid,last(p.ports)]);sizes...)
    for case in ("collinear_upper","collinear_lower","normal_inside","normal_near_limit")
        name="variants__"*case;problem=sonnet_planar_problem(project(name))
        @test native_s(name)==native_s("variants__plain_baseline")
        @test first(problem.ports).cells==13:20
        if startswith(case,"collinear")
            c=sonnet_conformal_layout(project(name);sizes...)
            @test first(c.layout.problem.ports).span==(.375e-3,.625e-3)
        end
    end
    before=TOML.parsefile(joinpath(fixture,"variants/original_before_comparison.toml"))
    for case in ("active_baseline","active_nonmid_position","active_offedge_position","active_opposite_wall_position")
        p=project("variants__"*case);saved=deepcopy(p)
        effective=DiffMoM._sonnet_geometry_project(p,1e9)
        expected=only(filter(r->r["case"]==case,before["cases"]))["normalized_port_fields"]
        @test [parse.(Float64,q.values[6:7]) for q in effective.ports]≈[parse.(Float64,q) for q in expected]
        @test native_s("variants__"*case)==native_s("variants__active_baseline")
        @test all(a.values==b.values for (a,b) in zip(p.ports,saved.ports))
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved.polygons))
    end
end
@testset "Native undriven wall contacts retain short/open physics" begin
    for name in ("unported__exact","unported__inside","unported__near_limit","unported__outside_limit","clipped__outside_negative")
        p=project(name);expected=native_s(name,1)
        for grid in (nothing,(16,16),(64,64))
            problem=sonnet_planar_problem(p;grid)
            @test any(only(problem.sheets).connect_west)==(name!="unported__outside_limit")
        end
        result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
        raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
        for i in eachindex(raw.problem.basis.port)
            iszero(raw.problem.basis.port[i]) && continue
            rhs[i,1]=raw.problem.basis.width[i]
        end
        error=maximum(abs,result.s-expected);residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
        @test error<=.005 && residual<=1e-9
        @test opnorm(result.s)<=1+1e-9
        push!(observations,(case=name,full_s_error=error,original_voltage_residual=residual))
    end
    p=project("unported__exact")
    DiffMoM._sonnet_raster_wall_project(p,.001,.001,32,32)
    @test DiffMoM._sonnet_raster_wall_project(p,.001,.001,32,32)===p
    @test wall_projection_allocation(p)==0
end
end
