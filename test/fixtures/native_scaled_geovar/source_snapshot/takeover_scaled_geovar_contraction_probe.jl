using DiffMoM,SHA,TOML
const producer=joinpath(@__DIR__,"takeover_scaled_geovar_x_probe.jl")
text=read(producer,String)
first=findfirst("function main_x()",text)
definitions=replace(text[1:first.start-1],"width=.375;nominal=.25"=>"width=.125;nominal=.25")
include_string(Main,definitions,producer)

function contraction_probe()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="scaled_geovar_contraction_",cleanup=false)
    sources=[joinpath(folder,file) for (folder,_,files) in walkdir(joinpath(repo,"src")) for file in files if endswith(file,".jl")]
    append!(sources,[producer,@__FILE__,template,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in sources)
    report=Dict{String,Any}("scope"=>"fresh native SCUNI contraction controls: independent signed ANC/SYM XDIR/YDIR notched polygons, dimension .125 versus NOM .25","source_before"=>hashes(),"cases"=>Any[])
    try
        for axis in ("XDIR","YDIR"),kind in ("ANC","SYM"),sign in (1,-1)
            tag=lowercase(axis)*"_"*lowercase(kind)*(sign==1 ? "_positive" : "_negative")
            matrices=Matrix{ComplexF64}[]
            row=Dict{String,Any}("tag"=>tag,"axis"=>axis,"kind"=>kind,"direction"=>sign,"status"=>"UNVERIFIED")
            push!(report["cases"],row)
            for mode in ("parameter","proportional_literal")
                source=joinpath(directory,tag*"_"*mode*".son")
                text=scaled_source(kind,sign;literal=mode!="parameter")
                write(source,axis=="XDIR" ? transpose_source(text) : text)
                native=reference_run(find_em(),source;output_dir=joinpath(directory,tag*"_"*mode),deembedded=false)
                response=checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
                push!(matrices,only(response.s))
            end
            row["full_s_error"]=maximum(abs,matrices[1]-matrices[2])
            row["bit_identical_s"]=matrices[1]==matrices[2]
            row["status"]=row["full_s_error"]<=1e-12 ? "PASS" : "FAIL"
            println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained evidence: ",directory)
    end
    @assert report["source_unchanged"] && all(row->row["status"]=="PASS",report["cases"])
end
contraction_probe()
