# Independent native check of named complex intermediates by variable unit.
using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference
function main()
    origin=joinpath(@__DIR__,"../../test/fixtures/native_sonnet_scalar_math/math/real_expression/real_expression.son")
    original=read(origin,String)
    expressionline=only(filter(x->startswith(x,"VALVAR Loss "),split(original,'\n')))
    evidence=evidence_directory("native_complex_variables");em=find_em()
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    rows=Dict{String,Any}[]
    for (tag,definition) in (("literal3","VALVAR Loss SRES \"3\" \"literal\""),
            ("literal5","VALVAR Loss SRES \"5\" \"literal\""),
            ("sres","VALVAR Inner SRES \"cmplx(3,4)\" \"intermediate\"\r\nVALVAR Loss SRES \"abs(Inner)\" \"outer\""),
            ("undef","VALVAR Inner UNDEF \"cmplx(3,4)\" \"intermediate\"\r\nVALVAR Loss SRES \"abs(Inner)\" \"outer\""))
        source=joinpath(evidence,tag*".son");write(source,replace(original,expressionline=>definition))
        row=Dict{String,Any}("tag"=>tag,"source_sha256"=>bytes2hex(sha256(read(source))))
        try
            p=read_sonnet_project(source)
            try row["current_value"]=sonnet_variable_value(p,"Loss")
            catch err;row["current_error"]=sprint(showerror,err);end
            native=reference_run(em,source;output_dir=joinpath(evidence,tag),deembedded=false)
            data=checked_native_touchstone(native,joinpath(native.output_dir,"native.s2p");deembedded=false,expected_z0=50.)
            row["native_stderr"]=read(joinpath(native.output_dir,"engine_stderr.log"),String)
            row["native_variable_lines"]=filter(x->occursin("Loss",x)||occursin("Inner",x),readlines(joinpath(native.output_dir,"engine_stdout.log")))
            row["s_real"]=vec(real.(only(data.s)));row["s_imag"]=vec(imag.(only(data.s)));row["status"]="SIMULATED"
        catch err
            row["error"]=sprint(showerror,err);row["status"]="UNVERIFIED"
        end
        push!(rows,row)
        open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,Dict("runs"=>rows,"artifact"=>evidence));end
        println(tag," ",row)
    end
    println(evidence)
end
main()
