module NativeSonnetBoxPortBoundaryAllowanceTests
using Test,DiffMoM,SHA,TOML
const fixture=joinpath(@__DIR__,"fixtures/native_box_port_boundary_allowance")
function rejection(f)
    try;f();nothing;catch err;err;end
end
@testset "Native box-port boundary allowance, units, grids and aspect ratio" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    files=Set(replace(relpath(joinpath(d,f),fixture),'\\'=>'/')
        for (d,_,fs) in walkdir(fixture) for f in fs if f!="sha256.toml")
    @test Set(keys(hashes))==files
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    for group in ("threshold","tiny","aspect")
        before=TOML.parsefile(joinpath(fixture,group,"original_before_comparison.toml"))
        @test before["source_unchanged"]
        for file in ("PlanarSonnetIO.jl","PlanarSonnetGeometryVariables.jl")
            @test bytes2hex(sha256(read(joinpath(fixture,"source_before",file))))==
                before["source_before"]["src\\planar\\"*file]
        end
    end
    cases=TOML.parsefile(joinpath(fixture,"index.toml"))["cases"]
    @test length(cases)==68
    accepted=0
    for row in cases
        folder=joinpath(fixture,"cases",row["name"])
        p=read_sonnet_project(joinpath(folder,"project.son"));original=deepcopy(p)
        valid=row["native_status"]=="ACCEPT";accepted+=valid
        metadata=TOML.parsefile(joinpath(folder,"native/metadata.toml"))
        @test metadata["process_success"]==valid
        @test metadata["source_sha256"]==bytes2hex(sha256(read(p.source)))
        if valid
            @test metadata["touchstone_selected_log_checks"]["native_raw.s2p"]["status"]=="PASS"
            for grid in (nothing,(16,16),(64,64))
                @test length(sonnet_planar_problem(p;freq=1e9,grid).ports)==2
            end
        else
            @test occursin("partially or entirely outside of the box",
                read(joinpath(folder,"native/engine_stderr.log"),String))
            for grid in (nothing,(16,16),(64,64))
                err=rejection(()->sonnet_planar_problem(p;freq=1e9,grid))
                @test err isa ArgumentError && occursin("box-port edge",sprint(showerror,err))
            end
        end
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,original.polygons))
        @test all(a.values==b.values for (a,b) in zip(p.ports,original.ports))
    end
    @test accepted==28
end
@testset "Active ANC port-edge boundary matches native literal controls" begin
    folder=joinpath(@__DIR__,"fixtures/native_moved_box_port_boundary_allowance")
    for (path,digest) in TOML.parsefile(joinpath(folder,"sha256.toml"))["sha256"]
        @test bytes2hex(sha256(read(joinpath(folder,path))))==digest
    end
    @test TOML.parsefile(joinpath(folder,"comparison.toml"))["source_unchanged"]
    for row in TOML.parsefile(joinpath(folder,"index.toml"))["cases"]
        path=joinpath(folder,"cases",row["name"])
        p=read_sonnet_project(joinpath(path,"parameter.son"));saved=deepcopy(p)
        for suffix in ("parameter","literal")
            metadata=TOML.parsefile(joinpath(path,suffix,"metadata.toml"))
            @test metadata["source_sha256"]==bytes2hex(sha256(read(joinpath(path,suffix*".son"))))
            @test metadata["process_success"]==(row["native_status"]=="ACCEPT")
        end
        if row["native_status"]=="ACCEPT"
            q=read_sonnet_project(joinpath(path,"literal.son"))
            @test planar_read_touchstone(joinpath(path,"parameter/native_raw.s2p")).s==
                planar_read_touchstone(joinpath(path,"literal/native_raw.s2p")).s
            effective=DiffMoM._sonnet_geometry_project(p,1e9)
            @test only(effective.polygons).vertices≈only(q.polygons).vertices rtol=8eps(Float64) atol=0
            @test only(sonnet_planar_problem(p;freq=1e9).sheets).mask==
                only(sonnet_planar_problem(q;freq=1e9).sheets).mask
        else
            for suffix in ("parameter","literal")
                @test occursin("partially or entirely outside of the box",
                    read(joinpath(path,suffix,"engine_stderr.log"),String))
            end
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9)
            @test_throws ArgumentError sonnet_planar_problem(p;freq=1e9)
        end
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved.polygons))
        @test all(a.values==b.values for (a,b) in zip(p.ports,saved.ports))
    end
end
end
