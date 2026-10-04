# Validation-only two-dimensional CSV scalar interpolation oracle.
using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference

function main()
    origin=joinpath(@__DIR__,"..","..","data","sonnet_validation","native_normal_resistivity_mz8D3l")
    evidence=evidence_directory("native_scalar_table2")
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    csv=joinpath(evidence,"loss2.csv")
    write(csv,"! Nonseparable corner data in ohms/square\n,10,20\n1,2,4\n3,8,16 ! last row\n")
    em=find_em();em===nothing && error("actual native Sonnet required")
    original=read(joinpath(origin,"srvy_2","srvy_2.son"),String)
    original=join(filter(line->!startswith(line,"VAR Rho "),split(original,'\n')),'\n')
    material=only(filter(line->startswith(line,"MET \"Sheet\""),split(original,'\n')))
    rows=Dict{String,Any}[]
    report=Dict{String,Any}("artifact"=>evidence,
        "scope"=>"actual native table2 CSV nodes/non-affine midpoint/quarter versus separately literal sheet resistance; not STF XML interpolation",
        "dependency_sha256"=>bytes2hex(sha256(read(csv))),"runs"=>rows)
    for (tag,rowkey,colkey,loss) in (("node_first",1.,10.,2.),("node_last",3.,20.,16.),
            ("midpoint",2.,15.,7.5),("quarter",1.5,12.5,4.375))
        expected=Matrix{ComplexF64}[]
        for literal in (true,false)
            variant=tag*(literal ? "_literal" : "_table")
            expression="table2(\\\"loss2.csv\\\",$rowkey,$colkey)"
            text=replace(original,material=>literal ? "MET \"Sheet\" 0 NOR $loss .5 .001 SRVY\r" : "MET \"Sheet\" 0 NOR \"Loss\" .5 .001 SRVY\r")
            if !literal
                text=replace(text,"LORGN "=>"VALVAR Loss SRES \"$expression\" \"CSV 2D probe\"\r\nLORGN ")
            end
            source=joinpath(evidence,variant*".son");write(source,text)
            record=Dict{String,Any}("tag"=>variant,"row_key"=>rowkey,"column_key"=>colkey,
                "independent_literal_ohm_square"=>loss,"source_sha256"=>bytes2hex(sha256(read(source))),"status"=>"UNVERIFIED")
            try
                native=reference_run(em,source;output_dir=joinpath(evidence,variant),deembedded=false,dependencies=literal ? String[] : [csv])
                data=checked_native_touchstone(native,joinpath(native.output_dir,"native.s2p");deembedded=false,expected_z0=50.)
                data.frequencies==[1e9,1e10] || error("unexpected frequency")
                literal && append!(expected,data.s)
                if !literal
                    length(expected)==2 || error("no literal oracle")
                    record["native_literal_delta_s"]=[maximum(abs,a-b) for (a,b) in zip(data.s,expected)]
                end
                record["s_real"]=[vec(real.(s)) for s in data.s]
                record["s_imag"]=[vec(imag.(s)) for s in data.s]
                record["status"]="SIMULATED"
            catch err
                record["error"]=sprint(showerror,err)
            end
            push!(rows,record)
            open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,report);end
            println(variant," ",record)
        end
    end
    println(evidence)
end
main()
