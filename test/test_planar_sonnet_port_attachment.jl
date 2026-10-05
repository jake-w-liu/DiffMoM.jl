module NativeSonnetPortAttachmentTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
include(joinpath(@__DIR__,"../validation/sonnet_stripline/sonnet_reference.jl"))
const fixture=joinpath(@__DIR__,"fixtures/native_box_port_attachment")
const observations=NamedTuple[]
const conformal_observations=NamedTuple[]
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
function wall_budget_rejection(p,max_bytes=1)
    try
        sonnet_conformal_layout(p;max_bytes,edge_size=.125e-3,interior_size=.25e-3)
    catch err
        label=max_bytes==1 ? "native stack layer workspace" : "native conformal metadata"
        err isa ArgumentError && occursin(label,sprint(showerror,err)) || rethrow()
        return
    end
    error("missing conformal metadata budget rejection")
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
        p=project(name);saved=deepcopy(p)
        for grid in (nothing,(16,16),(64,64))
            c=sonnet_conformal_layout(p;grid,sizes...)
            @test first(c.layout.problem.ports).span==(.375e-3,.625e-3)
            @test c.geometry_project.ports===p.ports
        end
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved.polygons))
        if startswith(case,"normal")
            result=solve_sonnet_conformal(p,1e9;raw=true,sizes...,mx=128,my=128)
            error=maximum(abs,result.s-native_s(name))
            @test error<=.005 && maximum(result.raw.relative_residuals)<=1e-9
            push!(conformal_observations,(case=name,full_s_error=error,
                original_voltage_residual=maximum(result.raw.relative_residuals)))
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
        saved=deepcopy(p)
        conformal=solve_sonnet_conformal(p,1e9;raw=true,edge_size=.125e-3,
            interior_size=.25e-3,edge_band=.05e-3,mx=128,my=128)
        raw=conformal.raw;rhs=zeros(ComplexF64,size(raw.currents))
        for i in eachindex(raw.problem.basis.port)
            port=raw.problem.basis.port[i];iszero(port) && continue
            rhs[i,port]=-raw.problem.basis.port_sign[i]*raw.problem.basis.width[i]
        end
        error=maximum(abs,conformal.s-expected)
        residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
        @test error<=.005 && residual<=1e-9
        @test opnorm(conformal.s)<=1+1e-9
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved.polygons))
        @test all(a.values==b.values for (a,b) in zip(p.ports,saved.ports))
        push!(conformal_observations,(case=name,full_s_error=error,
            original_voltage_residual=residual))
    end
    p=project("unported__exact")
    DiffMoM._sonnet_raster_wall_project(p,.001,.001,32,32)
    @test DiffMoM._sonnet_raster_wall_project(p,.001,.001,32,32)===p
    @test wall_projection_allocation(p)==0
end
@testset "Conformal native wall failures retain exact source provenance" begin
    folder=joinpath(@__DIR__,"fixtures/native_conformal_wall_ownership")
    hashes=TOML.parsefile(joinpath(folder,"sha256.toml"))["sha256"]
    files=Set(replace(relpath(joinpath(d,f),folder),'\\'=>'/')
        for (d,_,fs) in walkdir(folder) for f in fs if f!="sha256.toml")
    @test Set(keys(hashes))==files
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(folder,path))))==digest
    end
    before=TOML.parsefile(joinpath(folder,"original_before_comparison.toml"))
    @test before["source_unchanged"]
    @test bytes2hex(sha256(read(joinpath(folder,"source_before/PlanarSonnetConformal.jl"))))==
        before["source_before"]["src\\planar\\PlanarSonnetConformal.jl"]
    @test length(before["cases"])==5
    for row in before["cases"]
        if startswith(row["case"],"variants")
            @test row["conformal_status"]=="REJECT"
        elseif row["case"]=="unported__outside_limit"
            @test row["full_complex_s_error"]<=.005
        else
            @test row["full_complex_s_error"]>1.9
        end
    end
end
@testset "Conformal metadata budgets preflight wall normalization" begin
    mktempdir() do dir
        p=project("variants__normal_inside");n=1000
        corners=[(.001,.375),(1.,.375),(1.,.625),(.001,.625)]
        points=[((1-t)*corners[k][1]+t*corners[mod1(k+1,4)][1],
            (1-t)*corners[k][2]+t*corners[mod1(k+1,4)][2])
            for k in 1:4 for t in (i/n for i in 0:n-1)]
        push!(points,first(points))
        vertices=p.length_scale.*hcat(([q[1],q[2]] for q in points)...)
        old=only(p.polygons)
        poly=SonnetPolygon(old.kind,old.level,old.material,old.id,vertices,old.target,old.technology,old.flags)
        ports=[SonnetPortSpec(q.kind,q.polygon,k==1 ? 3n : n,q.number,q.values,q.records)
            for (k,q) in enumerate(p.ports)]
        literal=SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,p.box,p.layers,
            p.metals,p.top,p.bottom,[poly],ports,p.variables,p.components,p.sweeps,p.records)
        source=joinpath(dir,"large_wall.son")
        SonnetReference.write_native_fixture(source,literal;frequency_native=1.)
        large=read_sonnet_project(source);saved=deepcopy(large)
        @test size(only(large.polygons).vertices,2)==4001
        wall_budget_rejection(large)
        bytes=minimum(@allocated(wall_budget_rejection(large)) for _ in 1:3)
        # The budget must reject before even one complete vertex copy.
        @test bytes<sizeof(only(large.polygons).vertices)
        # A budget admitting the stack must still reject before wall copies.
        wall_budget_rejection(large,384)
        metadata_bytes=minimum(@allocated(wall_budget_rejection(large,384)) for _ in 1:3)
        @test metadata_bytes<sizeof(only(large.polygons).vertices)
        @test only(large.polygons).vertices==only(saved.polygons).vertices
        @test all(a.values==b.values for (a,b) in zip(large.ports,saved.ports))
    end
end
end
