module ODBConcaveNativeReferenceTests
using DiffMoM,Test,SHA,TOML,LinearAlgebra
@testset "Concave polygon line stroke vs archived actual Sonnet full matrices" begin
    root=normpath(joinpath(@__DIR__,"..","validation","sonnet_stripline","odb_concave_stroke_reference"))
    hashes=TOML.parsefile(joinpath(root,"sha256.toml"))["sha256"]
    @test length(hashes)==241
    for (path,wanted) in hashes
        @test open(io->bytes2hex(sha256(io)),joinpath(root,path))==wanted
    end
    evidence=TOML.parsefile(joinpath(root,"comparison.toml"))
    @test evidence["source_unchanged"]
    @test evidence["source_sha256_before"]==evidence["source_sha256_after"]
    @test length(evidence["runs"])==9
    repeat=TOML.parsefile(joinpath(root,"final_snapshot_repeat","comparison.toml"))
    @test repeat["source_unchanged"]
    @test repeat["source_sha256_before"]==repeat["source_sha256_after"]
    @test length(repeat["runs"])==9
    @test all(r->r["status"]=="PASS" && r["full_complex_s_error"]<=.06 &&
        r["original_voltage_residual"]<=1e-9,repeat["runs"])
    # The independently declared original aperture and native rectangular
    # union are reproduced literally; no solved mask defines this oracle.
    vertices=[(-.25,-.25),(.25,-.25),(.25,.25),(.125,.25),(.125,-.125),
        (-.125,-.125),(-.125,.25),(-.25,.25)]
    @test TOML.parsefile(joinpath(root,"custom_u.toml"))["vertices"]==[collect(v) for v in vertices]
    polygon=DiffMoM._artwork_polygon([(x*1e-3,y*1e-3) for (x,y) in vertices])
    artwork=read_odb_features(joinpath(root,"concave_stroke.features");layer="metal",
        symbol_resolver=name->name=="custom_u" ? polygon : error("unexpected symbol"))
    rectangles=[(.25,.5,.375,.625),(.625,.875,.375,.625),
        (0.,1.,.25,.375),(0.,1.,.625,.75)]
    for cells in (16,32,64)
        grid=CellGrid(.001,.001,cells,cells)
        stack=PlanarStackup([PlanarLayer(1.,1.,.0001),PlanarLayer(1.,1.,.0001)],TERM_GND,TERM_GND,.001,.001)
        lower=cells÷4+1:3cells÷8;upper=5cells÷8+1:3cells÷4
        ports=[PlanarPort(1,:west,lower,50.),PlanarPort(1,:east,lower,50.),
            PlanarPort(1,:west,upper,50.),PlanarPort(1,:east,upper,50.)]
        expected=BitMatrix([any(x0<=(i-.5)/cells<=x1 && y0<=(j-.5)/cells<=y1
            for (x0,x1,y0,y1) in rectangles) for i in 1:cells,j in 1:cells])
        prob=artwork_planar_problem(artwork,stack,grid,Dict("metal"=>(;kind=:sheet,interface=1)),ports;max_bytes=2_000_000_000)
        @test only(prob.sheets).mask==expected
        for frequency in (1e9,5e9,1e10)
            tag="native_$(cells)_$(Int(frequency))";folder=joinpath(root,tag)
            metadata=TOML.parsefile(joinpath(folder,"metadata.toml"))
            @test startswith(metadata["engine_version"],"18.53-Lite")
            @test metadata["process_success"]
            @test metadata["deembedded"]==false
            @test metadata["response_nports"]==4
            @test metadata["source_sha256"]==open(io->bytes2hex(sha256(io)),joinpath(folder,tag*".son"))
            guard=metadata["touchstone_selected_log_checks"]["native_raw.s4p"]
            @test guard["status"]=="PASS"
            @test guard["intended_log_section"]=="raw"
            @test guard["source_options"]==String[]
            @test guard["max_printed_bound_ratio"]<=1
            file=joinpath(folder,"native_raw.s4p")
            @test guard["output_sha256"]==open(io->bytes2hex(sha256(io)),file)
            native=planar_read_touchstone(file)
            @test native.frequencies==[frequency]
            @test native.z0==fill(50.,4)
            result=solve_planar_contracted(prob,frequency,Matrix{Float64}(I,4,4);z0=fill(50.,4),
                method=:dense_fft,mx=4cells,my=4cells,max_bytes=2_000_000_000,retain_matrix=true)
            @test maximum(abs,result.s-only(native.s))<=.06
            @test maximum(result.raw.relative_residuals)<=1e-9
            @test maximum(abs,result.s-transpose(result.s))<1e-10
            @test eigmax(Hermitian(result.s'*result.s))<=1+1e-9
            result=nothing;GC.gc()
        end
    end
end
end
