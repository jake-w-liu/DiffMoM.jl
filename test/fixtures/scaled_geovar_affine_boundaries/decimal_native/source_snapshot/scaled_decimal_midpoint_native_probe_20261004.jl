using DiffMoM,SHA,TOML,LinearAlgebra
const producer=joinpath(@__DIR__,"two_axis_geovar_native_probe_20261004.jl")
let text=read(producer,String)
    first=findfirst("function main()",text)
    definitions=replace(text[1:first.start-1],"nominal=.25"=>"nominal=.2",
        "(0.,.375),(1.,.375),(1.,.4375),(.8,.5),(1.,.5625),(1.,.625),(0.,.625)"=>
        "(0.,.4),(1.,.4),(1.,.45),(.8,.5),(1.,.55),(1.,.6),(0.,.6)")
    include_string(Main,definitions,producer)
end

function main()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="scaled_decimal_midpoint_",cleanup=false)
    sources=[joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(repo,"src")) for file in files if endswith(file,".jl")]
    append!(sources,[@__FILE__,producer,template,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in sources)
    report=Dict{String,Any}("scope"=>"native decimal SCUNI symmetric centers with fixed .005 full-S and 1e-9 original voltage gates; active midpoint classification candidate","source_before"=>hashes(),"cases"=>Any[])
    try
        for sign in (1,-1)
            tag=sign==1 ? "sym_positive" : "sym_negative"
            row=Dict{String,Any}("case"=>tag,"reader_before"=>"UNVERIFIED","native_status"=>"UNVERIFIED")
            push!(report["cases"],row)
            matrices=Matrix{ComplexF64}[]
            for mode in ("parameter","literal")
                source=joinpath(directory,tag*"_"*mode*".son")
                text=replace(two_axis_source("SYM",sign,.3;literal=mode=="literal",axis_only=true),"SCXY"=>"SCUNI")
                write(source,text)
                if mode=="parameter"
                    try
                        DiffMoM._sonnet_geometry_project(read_sonnet_project(source),1e9)
                        row["reader_before"]="ACCEPTED"
                    catch exception
                        row["reader_before"]="REJECTED";row["reader_error"]=sprint(showerror,exception)
                    end
                end
                native=reference_run(find_em(),source;output_dir=joinpath(directory,tag*"_"*mode),deembedded=false)
                data=checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
                push!(matrices,only(data.s))
                if mode=="literal"
                    p=read_sonnet_project(source)
                    result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
                    raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
                    for b in eachindex(raw.problem.basis.port)
                        port=raw.problem.basis.port[b];iszero(port) && continue
                        rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
                    end
                    row["literal_public_full_s_error"]=maximum(abs,result.s-matrices[end])
                    row["literal_original_voltage_residual"]=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
                end
            end
            row["bit_identical_s"]=matrices[1]==matrices[2]
            row["native_full_s_error"]=maximum(abs,matrices[1]-matrices[2])
            row["native_status"]=row["bit_identical_s"] ? "PASS" : "FAIL"
            println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained evidence: ",directory)
    end
    @assert report["source_unchanged"] && all(row->row["native_status"]=="PASS",report["cases"])
end
main()
