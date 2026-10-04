using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
function main()
    original=joinpath(repo,"data/sonnet_validation/geovar_zero_nominal_axes_GAt2Wu")
    output=mktempdir(joinpath(repo,"data/planar_audit");prefix="zero_nominal_literal_public_masks_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    paths=vcat([@__FILE__],[joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(repo,"src")) for file in files])
    hashes()=Dict(path=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"Actual public literal mask export for comparison with independently captured native ASCII subsection support; no physical acceptance claim.","source_before"=>hashes(),"cases"=>Any[])
    try
        for target in (.0625,.125)
            tag="XDIR_dir_-1_target_$(target)_literal"
            source=joinpath(original,tag*".son")
            p=read_sonnet_project(source);prob=sonnet_planar_problem(p;freq=1e9)
            mask=only(prob.sheets).mask
            file=joinpath(output,tag*"_mask.txt")
            open(file,"w") do io
                for j in axes(mask,2)
                    for i in axes(mask,1);print(io,mask[i,j] ? '1' : '0');end
                    println(io)
                end
            end
            push!(report["cases"],Dict("case"=>tag,"occupied_cells"=>count(mask),"shape"=>collect(size(mask)),"source_sha256"=>bytes2hex(sha256(read(source))),"mask_sha256"=>bytes2hex(sha256(read(file)))))
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained public literal masks: ",output)
    end
    @assert report["source_unchanged"]
end
main()
