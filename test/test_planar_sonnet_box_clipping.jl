module NativeSonnetBoxClippingTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
const fixture=joinpath(@__DIR__,"fixtures/native_conformal_box_clipping")
const observations=NamedTuple[]
const sizes=(edge_size=.125e-3,interior_size=.25e-3,edge_band=.05e-3)
record_values(records)=[(r.line,r.tokens) for r in records]
@testset "Native exact box clipping retains geometry, contacts and sources" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    files=Set(replace(relpath(joinpath(d,f),fixture),'\\'=>'/')
        for (d,_,fs) in walkdir(fixture) for f in fs if f!="sha256.toml")
    @test Set(keys(hashes))==files
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    before=TOML.parsefile(joinpath(fixture,"original_before_comparison.toml"))
    @test before["source_unchanged"] && length(before["cases"])==10
    for f in ("PlanarSonnetConformal.jl","PlanarConformalLayout.jl")
        @test bytes2hex(sha256(read(joinpath(fixture,"source_before",f))))==
            before["source_before"]["src\\planar\\"*f]
    end
    for row in before["cases"]
        name=row["case"];folder=joinpath(fixture,"cases",name)
        metadata=TOML.parsefile(joinpath(folder,"native/metadata.toml"))
        valid=row["native_status"]=="ACCEPT"
        @test row["conformal_status"]=="REJECT"
        @test metadata["process_success"]==valid
        @test metadata["source_sha256"]==row["source_sha256"]==
            bytes2hex(sha256(read(joinpath(folder,"project.son"))))
        p=read_sonnet_project(joinpath(folder,"project.son"));saved=deepcopy(p)
        if valid
            @test metadata["actual_cell_counts"]==[32,32]
            @test metadata["response_nports"]==1 && !metadata["deembedded"]
            logcheck=metadata["touchstone_selected_log_checks"]["native_raw.s2p"]
            @test logcheck["status"]=="PASS" && logcheck["max_printed_bound_ratio"]<=1
            @test logcheck["output_sha256"]==bytes2hex(sha256(read(joinpath(folder,"native/native_raw.s2p"))))
            expected=only(planar_read_touchstone(joinpath(folder,"native/native_raw.s2p");nports=1).s)
            @test vec(real.(expected))==row["native_s_real"] && vec(imag.(expected))==row["native_s_imag"]
            for grid in (nothing,(16,16),(64,64))
                c=sonnet_conformal_layout(p;grid,sizes...)
                mesh=c.layout.problem.mesh
                @test all(x->0<=x<=.001,mesh.vertices[1,:]) && all(y->0<=y<=.001,mesh.vertices[2,:])
                @test c.geometry_project.ports===p.ports
                if name=="concave_split"
                    @test sum(mesh.areas)≈2.5e-7 rtol=1e-13
                    @test all(t->maximum(mesh.vertices[2,mesh.triangles[:,t]])<=.375e-3 ||
                        minimum(mesh.vertices[2,mesh.triangles[:,t]])>=.625e-3,eachindex(mesh.areas))
                end
            end
            result=solve_sonnet_conformal(p,1e9;raw=true,sizes...,mx=128,my=128)
            raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
            for i in eachindex(raw.problem.basis.port)
                port=raw.problem.basis.port[i];iszero(port) && continue
                rhs[i,port]=-raw.problem.basis.port_sign[i]*raw.problem.basis.width[i]
            end
            error=maximum(abs,result.s-expected)
            residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
            @test error<=.005 && residual<=1e-9
            @test opnorm(result.s)<=1+1e-9
            push!(observations,(case=name,full_s_error=error,original_voltage_residual=residual))
        else
            @test occursin("partially or entirely outside",read(joinpath(folder,"native/engine_stderr.log"),String))
            @test_throws ArgumentError sonnet_conformal_layout(p;sizes...)
        end
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved.polygons))
        @test length(p.ports)==length(saved.ports) &&
            all(a.values==b.values && record_values(a.records)==record_values(b.records)
                for (a,b) in zip(p.ports,saved.ports))
        @test record_values(p.records)==record_values(saved.records)
    end
    p=read_sonnet_project(joinpath(fixture,"cases/concave_split/project.son"))
    @test_throws ArgumentError sonnet_conformal_layout(p;max_bytes=1,sizes...)
    @test_throws ArgumentError sonnet_conformal_layout(p;max_triangles=1,sizes...)
end
end
