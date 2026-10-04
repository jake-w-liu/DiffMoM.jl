module NativeSonnetComplexArgumentsTests
using Test,DiffMoM,LinearAlgebra,TOML,SHA,JSON

function crc(bytes)
    value=typemax(UInt32)
    for byte in bytes
        value⊻=UInt32(byte)
        for _ in 1:8;value=isodd(value) ? (value>>1)⊻0xedb88320 : value>>1;end
    end
    return ~value
end

@testset "Actual native complex atan2, conversion operands and CSV keys" begin
    fixtures=("native_sonnet_complex_atan2_units_keys","native_sonnet_complex_function_corrected")
    controls=Dict{Float64,Matrix{ComplexF64}}()
    cases=Tuple{String,Dict{String,Any}}[]
    for (fixture,jobs,matrices) in zip(fixtures,(105,16),(103,16))
        root=joinpath(@__DIR__,"fixtures",fixture)
        for file in JSON.parsefile(joinpath(root,"manifest.json"))["files"]
            bytes=read(joinpath(root,file["path"]))
            @test length(bytes)==file["bytes"]
            @test bytes2hex(sha256(bytes))==file["sha256"]
            @test string(crc(bytes);base=16,pad=8)==file["crc32"]
        end
        provenance=JSON.parsefile(joinpath(root,"provenance.json"))
        @test provenance["jobs"]==jobs && provenance["matrices"]==matrices
        for artifact in provenance["artifacts"]
            report=TOML.parsefile(joinpath(root,artifact,"comparison.toml"))
            @test report["module_before"]==report["module_after"]
            for row in report["runs"]
                dir=joinpath(root,artifact,row["tag"])
                if startswith(row["tag"],"literal")
                    controls[row["hypothesized_real_ohm_square"]]=only(planar_read_touchstone(joinpath(dir,"native.s2p")).s)
                end
                push!(cases,(dir,row))
            end
        end
    end
    # These corrections retain the original false hypotheses in the archived
    # reports. Fresh independent literal jobs prove the missing positive
    # constants bit-exact; negative physical material values remain rejected.
    corrections=Dict(
        "h2p_complex_imag"=>0.,"p2h_complex_imag"=>0.,
        "m2p_complex_imag"=>0.,"p2m_complex_imag"=>0.,"m2p_complex_real"=>.003,
        "atan2_complex_y_zero_x_imag"=>1.,"atan2_zero_x_positive_imag"=>2.,
        "atan2_zero_x_negative_real"=>2+pi/2,"atan2_zero_x_negative_imag"=>-2.,
        "atan2_zero_x_pure_imaginary_real"=>pi/2,"atan2_zero_x_pure_imaginary_imag"=>0.,
        "atan2_zero_x_negative_pure_imaginary_real"=>pi/2,
        "atan2_zero_x_negative_pure_imaginary_imag"=>4.,"atan2_negative_zero_x"=>1.,
        "table2_complex"=>38.,"table2_pure_imaginary"=>42.,"table2_negative_complex"=>24.,
        "atan2_real_positive_zero_imag"=>2.,"atan2_real_negative_zero_real"=>2+pi/2,
        "atan2_real_negative_zero_imag"=>1.,"atan2_complex_real_negative_zero_real"=>2+pi/2,
        "atan2_complex_real_negative_zero_imag"=>1.)
    for (dir,row) in cases
        tag=row["tag"];source=joinpath(dir,tag*".son")
        p=read_sonnet_project(source);metadata=TOML.parsefile(joinpath(dir,"metadata.toml"))
        @test bytes2hex(sha256(read(source)))==row["source_sha256"]==metadata["source_sha256"]
        @test startswith(metadata["engine_version"],"18.53-Lite (64-bit Windows)")
        @test metadata["actual_cell_counts"]==[20,20]
        @test !metadata["deembedded"]
        for (name,digest) in metadata["dependency_sha256"]
            @test bytes2hex(sha256(read(joinpath(dir,name))))==digest
        end
        expression=only(filter(metal->metal[1]=="Sheet",p.metals))[4]
        if tag in ("atan2_zero_both_real","atan2_zero_both_imag")
            @test row["status"]=="UNVERIFIED"
            @test occursin("Loss = nan",read(joinpath(dir,"engine_stdout.log"),String))
            @test isempty(read(joinpath(dir,"native.s2p")))
            @test_throws ArgumentError sonnet_variable_value(p,expression)
            @test_throws ArgumentError solve_sonnet_project(p,1e9;raw=true,grid=(typemax(Int),typemax(Int)))
            continue
        end
        @test row["status"]=="SIMULATED"
        @test isempty(read(joinpath(dir,"engine_stderr.log"),String))
        @test metadata["touchstone_selected_log_checks"]["native.s2p"]["status"]=="PASS"
        value=get(corrections,tag,row["hypothesized_real_ohm_square"])
        original=deepcopy(p.variables)
        @test sonnet_variable_value(p,expression)≈value rtol=3e-15 atol=2e-15
        snapshot=sonnet_scalar_files(p)
        @test sonnet_variable_value(snapshot,expression)≈value rtol=3e-15 atol=2e-15
        if value<0
            @test_throws ArgumentError solve_sonnet_project(p,1e9;raw=true,grid=(typemax(Int),typemax(Int)))
            continue
        end
        data=planar_read_touchstone(joinpath(dir,"native.s2p"));native=only(data.s)
        @test data.frequencies==[1e9]
        @test native==reshape(complex.(row["s_real"],row["s_imag"]),2,2)
        @test haskey(controls,value)
        @test maximum(abs,native-controls[value])<=2e-13
        actual=solve_sonnet_project(p,1e9;raw=true,mx=160,my=160,method=:dense_fft)
        retained=solve_sonnet_project(snapshot.project,1e9;raw=true,scalar_files=snapshot,mx=160,my=160,method=:dense_fft)
        @test maximum(abs,actual.s-native)<.005
        @test maximum(abs,actual.s-retained.s)<1e-11
        @test actual.raw.currents≈retained.raw.currents rtol=1e-9
        rhs=zeros(ComplexF64,size(actual.raw.currents))
        for b in eachindex(actual.raw.problem.basis.kind)
            q=actual.raw.problem.basis.port[b];q==0 && continue
            rhs[b,q]=(q==1 ? -1 : 1)*actual.raw.problem.basis.width[b]
        end
        @test norm(actual.raw.z_mom*actual.raw.currents-rhs)/norm(rhs)<1e-9
        @test opnorm(actual.s)<=1+1e-10
        a=only(planar_current_maps(actual;voltages=ComplexF64[.6+.2im,-.3+.4im]))
        b=only(planar_current_maps(retained;voltages=ComplexF64[.6+.2im,-.3+.4im]))
        @test hypot(norm(a.jx-b.jx),norm(a.jy-b.jy))<1e-8*hypot(norm(b.jx),norm(b.jy))
        @test p.variables==original
    end
end
end
