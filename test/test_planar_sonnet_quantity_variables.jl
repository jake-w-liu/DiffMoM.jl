module NativeSonnetQuantityVariablesTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
@testset "Named native quantity variables retain real projection at their boundary" begin
    directory=joinpath(@__DIR__,"fixtures/native_sonnet_quantity_variables/native")
    report=TOML.parsefile(joinpath(directory,"comparison.toml"))
    controls=Dict{String,Matrix{ComplexF64}}()
    for row in report["runs"]
        row["status"]=="SIMULATED" || continue
        tag=row["tag"];source=joinpath(directory,tag,tag*".son")
        @test bytes2hex(sha256(read(source)))==row["source_sha256"]
        metadata=TOML.parsefile(joinpath(directory,tag,"metadata.toml"))
        @test metadata["source_sha256"]==row["source_sha256"]
        @test startswith(metadata["engine_version"],"18.53-Lite (64-bit Windows)")
        @test isempty(row["native_stderr"])
        controls[tag]=only(planar_read_touchstone(joinpath(directory,tag,"native.s2p")).s)
        @test controls[tag]==reshape(complex.(row["s_real"],row["s_imag"]),2,2)
    end
    @test controls["sres"]==controls["literal3"]
    @test maximum(abs,controls["sres"]-controls["literal5"])>.05
    row=only(r for r in report["runs"] if r["tag"]=="sres")
    @test row["native_variable_lines"]==["      Inner = 3 Ohms/sq","      Loss = 3 Ohms/sq"]
    p=read_sonnet_project(joinpath(directory,"sres/sres.son"))
    @test sonnet_variable_value(p,"Inner")==3.
    @test sonnet_variable_value(p,"Loss")==3.
    @test sonnet_variable_value(p,"abs(cmplx(3,4))")==5.
    # The inline controls in test_planar_sonnet_functions establish the
    # same finite real projection for unnamed material quantities.
    @test sonnet_variable_value(p,"cmplx(3,4)")==3.
    @test sonnet_variable_value(p,"cmplx(3,4)")==sonnet_variable_value(p,"Inner")
    @test sonnet_variable_value(p,"abs(Inner)")==3.
    @test sonnet_variable_value(p,"imag(Inner)")==0.
    @test sonnet_variable_value(p,"imag(cmplx(3,4))")==4.
    @test_throws ArgumentError sonnet_variable_value(p,"Inner";variables=Dict("Inner"=>3+4im))
    result=solve_sonnet_project(p,1e9;raw=true,mx=160,my=160,method=:dense_fft)
    reference=solve_sonnet_project(read_sonnet_project(joinpath(directory,"literal3/literal3.son")),1e9;
        raw=true,mx=160,my=160,method=:dense_fft)
    @test result.s==reference.s
    @test result.raw.currents==reference.raw.currents
    @test maximum(abs,result.s-controls["sres"])<.005
    @test opnorm(result.s)<=1+1e-10
    p.variables["Inner"]="cmplx(3,-4)"
    @test sonnet_variable_value(p,"Loss")==3.
    p.variables["Inner"]="cmplx(3,exp(1000))"
    @test_throws ArgumentError sonnet_variable_value(p,"Inner")
end
end
