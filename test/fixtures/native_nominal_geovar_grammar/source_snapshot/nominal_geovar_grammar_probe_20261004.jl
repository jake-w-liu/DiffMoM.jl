using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference

function main()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="nominal_geovar_grammar_",cleanup=false)
    input=joinpath(repo,"test/fixtures/native_scaled_geovar/native/anc_positive_parameter.son")
    nominal=replace(read(input,String),"VALVAR Width LNG 0.375"=>"VALVAR Width LNG 0.25")
    changes=(("valid","POS .5 .5","POS .5 .5"),
        ("display_nonnumeric","POS .5 .5","POS .5 nope"),
        ("unknown_polygon","REF1 POLY 1 1","REF1 POLY 9999 1"),
        ("unknown_point","REF2 POLY 1 1\n6","REF2 POLY 1 1\n99"),
        ("negative_group_count","PS2 1","PS2 -1"))
    sources=[joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(repo,"src")) for file in files if endswith(file,".jl")]
    append!(sources,[input,@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in sources)
    report=Dict{String,Any}("scope"=>"independent actual native acceptance of nominal saved GEOVAR grammar; no edits","cases"=>Any[],"source_before"=>hashes())
    try
        for (name,from,to) in changes
            source=joinpath(directory,name*".son");write(source,replace(nominal,from=>to))
            row=Dict{String,Any}("case"=>name,"source_sha256"=>bytes2hex(sha256(read(source))),
                "reader_acceptance"=>"UNVERIFIED","native_acceptance"=>"UNVERIFIED")
            push!(report["cases"],row)
            try
                p=read_sonnet_project(source)
                row["reader_acceptance"]=DiffMoM._sonnet_geometry_project(p,1e9)===p ? "NOMINAL_PASSTHROUGH" : "ADAPTED"
            catch exception
                row["reader_acceptance"]="REJECTED";row["reader_error"]=sprint(showerror,exception)
            end
            try
                native=reference_run(find_em(),source;output_dir=joinpath(directory,name),deembedded=false)
                checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
                row["native_acceptance"]="ACCEPTED"
            catch exception
                row["native_acceptance"]="REJECTED";row["native_error"]=sprint(showerror,exception)
                row["native_stderr"]=read(joinpath(directory,name,"engine_stderr.log"),String)
            end
            println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained evidence: ",directory)
    end
    @assert report["source_unchanged"]
end
main()
