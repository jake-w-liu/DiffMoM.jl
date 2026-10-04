module NativeSonnetInternalAttachmentTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
const fixture=joinpath(@__DIR__,"fixtures/native_internal_port_attachment")
const observations=NamedTuple[]
project(name)=read_sonnet_project(joinpath(fixture,"cases",name,"project.son"))
native_s(name)=only(planar_read_touchstone(joinpath(fixture,"cases",name,"native/native_raw.s1p")).s)
@testset "Native physical wall/shared attachment and annotation independence" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    files=Set(replace(relpath(joinpath(d,f),fixture),'\\'=>'/') for (d,_,fs) in walkdir(fixture) for f in fs if f!="sha256.toml")
    @test Set(keys(hashes))==files
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    for group in ("contract","guards","kinds")
        before=TOML.parsefile(joinpath(fixture,group,"original_before_comparison.toml"))
        @test before["source_unchanged"]
        for f in ("PlanarSonnetIO.jl","PlanarSonnetConformal.jl")
            @test bytes2hex(sha256(read(joinpath(fixture,"source_before",f))))==before["source_before"]["src\\planar\\"*f]
        end
    end
    cases=TOML.parsefile(joinpath(fixture,"index.toml"))["cases"]
    @test length(cases)==13
    sizes=(edge_size=.125e-3,interior_size=.25e-3,edge_band=.05e-3)
    for row in cases
        name=row["name"];p=project(name);saved=deepcopy(p)
        metadata=TOML.parsefile(joinpath(fixture,"cases",name,"native/metadata.toml"))
        @test metadata["source_sha256"]==bytes2hex(sha256(read(p.source)))
        @test metadata["process_success"]==(row["native_status"]=="ACCEPT")
        if row["native_status"]=="ACCEPT"
            expected_wall=name=="kinds__gap_wall" ? :west : :internal_x
            expected_cells=occursin("partial",name) ? (15:18) : (13:20)
            for grid in (nothing,(16,16),(64,64))
                problem=sonnet_planar_problem(p;grid)
                @test only(problem.ports).wall==expected_wall
                grid===nothing && @test only(problem.ports).cells==expected_cells
            end
            c=sonnet_conformal_layout(p;sizes...)
            @test only(c.layout.problem.ports).wall==(expected_wall===:west ? :west : :internal)
            if expected_wall===:internal_x
                @test only(c.layout.problem.ports).span[2]≈(occursin("partial",name) ? .125e-3 : .25e-3)
            end
            result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
            raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
            for i in eachindex(raw.problem.basis.port)
                iszero(raw.problem.basis.port[i]) && continue
                rhs[i,1]=-raw.problem.ports[1].polarity*raw.problem.basis.width[i]
            end
            error=maximum(abs,result.s-native_s(name));residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
            @test error<=.005 && residual<=1e-9
            @test opnorm(result.s)<=1+1e-9
            push!(observations,(case=name,full_s_error=error,original_voltage_residual=residual))
            if name in ("contract__std_baseline","guards__std_partial","kinds__gap_wall")
                physical=solve_sonnet_conformal(p,1e9;raw=true,sizes...,mx=128,my=128)
                error=maximum(abs,physical.s-native_s(name));residual=maximum(physical.raw.relative_residuals)
                @test error<=.005 && residual<=1e-9
                push!(observations,(case=name*"__conformal",full_s_error=error,original_voltage_residual=residual))
            end
        else
            @test_throws ArgumentError sonnet_planar_problem(p)
            @test_throws ArgumentError sonnet_conformal_layout(p;sizes...)
        end
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved.polygons))
        @test all(a.values==b.values for (a,b) in zip(p.ports,saved.ports))
    end
    baseline=native_s("contract__std_baseline")
    for name in ("contract__std_offedge","contract__std_opposite_wall","contract__gap_baseline","contract__gap_offedge","kinds__box_internal")
        @test native_s(name)==baseline
    end
    @test native_s("guards__std_partial")==native_s("kinds__box_partial")
end
@testset "Native internal two-port phase is independent of polygon reference" begin
    folder=joinpath(@__DIR__,"fixtures/native_internal_port_orientation")
    for (path,digest) in TOML.parsefile(joinpath(folder,"sha256.toml"))["sha256"]
        @test bytes2hex(sha256(read(joinpath(folder,path))))==digest
    end
    before=TOML.parsefile(joinpath(folder,"original_before_comparison.toml"))
    @test before["source_unchanged"]
    @test first(before["cases"])["conformal_full_s_error"]>1
    @test bytes2hex(sha256(read(joinpath(folder,"source_before/PlanarSonnetConformal.jl"))))==
        before["source_before"]["src\\planar\\PlanarSonnetConformal.jl"]
    matrices=Matrix{ComplexF64}[]
    yfolder=joinpath(@__DIR__,"fixtures/native_internal_y_port_orientation")
    for (path,digest) in TOML.parsefile(joinpath(yfolder,"sha256.toml"))["sha256"]
        @test bytes2hex(sha256(read(joinpath(yfolder,path))))==digest
    end
    @test TOML.parsefile(joinpath(yfolder,"original_before_comparison.toml"))["source_unchanged"]
    for folder in (folder,yfolder),row in TOML.parsefile(joinpath(folder,"index.toml"))["cases"]
        name=row["name"];path=joinpath(folder,"cases",name);p=read_sonnet_project(joinpath(path,"project.son"))
        expected=only(planar_read_touchstone(joinpath(path,"native/native_raw.s2p")).s)
        push!(matrices,expected)
        raster=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128)
        conformal=solve_sonnet_conformal(p,1e9;raw=true,edge_size=.125e-3,interior_size=.25e-3,edge_band=.05e-3,mx=128,my=128)
        @test maximum(abs,raster.s-expected)<=.005
        error=maximum(abs,conformal.s-expected);residual=maximum(conformal.raw.relative_residuals)
        @test error<=.005 && residual<=1e-9
        @test real(conformal.s[1,2])<-.99
        push!(observations,(case=name,full_s_error=error,original_voltage_residual=residual))
    end
    @test all(maximum(abs,m-matrices[1])<=1e-10 for m in matrices)
end
@testset "Native diagonal internal sources reject before conformal lowering" begin
    folder=joinpath(@__DIR__,"fixtures/native_internal_diagonal_rejection")
    hashes=TOML.parsefile(joinpath(folder,"sha256.toml"))["sha256"]
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(folder,path))))==digest
    end
    before=TOML.parsefile(joinpath(folder,"original_before_comparison.toml"))
    @test before["source_unchanged"]
    @test bytes2hex(sha256(read(joinpath(folder,"source_before/PlanarSonnetConformal.jl"))))==before["source_before"]["src\\planar\\PlanarSonnetConformal.jl"]
    cases=TOML.parsefile(joinpath(folder,"index.toml"))["cases"]
    @test length(cases)==4
    for row in cases
        path=joinpath(folder,"cases",row["name"]);p=read_sonnet_project(joinpath(path,"project.son"))
        saved=deepcopy(p);metadata=TOML.parsefile(joinpath(path,"native/metadata.toml"))
        @test !metadata["process_success"]
        @test metadata["source_sha256"]==bytes2hex(sha256(read(p.source)))
        @test occursin("Port 1 is diagonal",read(joinpath(path,"native/engine_stderr.log"),String))
        for route in (q->sonnet_planar_problem(q),
            q->sonnet_conformal_layout(q;edge_size=.125e-3,interior_size=.25e-3))
            err=try;route(p);nothing;catch err;err;end
            @test err isa ArgumentError && occursin("diagonal",sprint(showerror,err))
        end
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved.polygons))
        @test all(a.values==b.values for (a,b) in zip(p.ports,saved.ports))
    end
end
end
