using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
    original=joinpath(repo,"data/sonnet_validation/geovar_zero_nominal_axes_GAt2Wu")
    output=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="zero_nominal_y_literal_mesh_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    sources=[joinpath(original,"YDIR_dir_-1_target_$(target)_literal.son") for target in (.0625,.125)]
    paths=vcat([@__FILE__],sources,[joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(repo,"src")) for file in files])
    hashes()=Dict(path=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"Fresh -j ASCII native subsection/current diagnostic for one failed and one passing independent zero-offset literal; unchanged S identity gate1e-12, no physical acceptance replacement.","source_before"=>hashes(),"cases"=>Any[])
    try
        for source in sources
            tag=splitext(basename(source))[1]
            text=read(source,String);@assert occursin("OPTIONS\n",text)
            file=joinpath(output,tag*".son");write(file,replace(text,"OPTIONS\n"=>"OPTIONS -j\n"))
            native=reference_run(find_em(),file;output_dir=joinpath(output,tag),deembedded=false)
            matrix=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
            previous=only(planar_read_touchstone(joinpath(original,tag,"native_raw.s2p")).s)
            currents=[joinpath(dir,name) for (dir,_,files) in walkdir(native.output_dir) for name in files if startswith(name,"current_") && endswith(name,".sid")]
            row=Dict("case"=>tag,"full_s_error"=>maximum(abs,matrix-previous),"bit_identical_s"=>matrix==previous,"current_files"=>currents)
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained native current mesh: ",output)
    end
    @assert report["source_unchanged"] && all(row["bit_identical_s"] && !isempty(row["current_files"]) for row in report["cases"])
end
main()
