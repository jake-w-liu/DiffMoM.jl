using DiffMoM,SHA,TOML,Dates
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
const sources=Dict("NSCD"=>"native_sonnet_geometry_variables/accepted/single_ydir_anc_positive_parameter.son",
    "SCUNI"=>"native_sonnet_geometry_variables/accepted/single_ydir_anc_positive_parameter.son",
    "SCXY"=>"native_two_axis_geovar/native/ydir_anc_positive_expand_parameter.son",
    "RAD"=>"native_radial_geovar/native/expand_nscd_ydir_positive.son")

function duplicate_source(mode,kind;literal="")
    source=read(joinpath(repo,"test/fixtures",sources[mode]),String)
    if mode in ("NSCD","SCUNI")
        source=replace(source,"VALVAR Width LNG 0.375"=>"VALVAR Width LNG 0.3125"," NSCD"=>" $mode")
        selected=[2];firstref=0;secondref=3;nominal=.25;target=.3125
    elseif mode=="SCXY"
        source=replace(source,"VALVAR Width LNG 0.375"=>"VALVAR Width LNG 0.3125")
        selected=[2,3,4];firstref=1;secondref=5;nominal=.25;target=.3125
    else
        selected=[4];firstref=2;secondref=3;nominal=.3125;target=.375
    end
    extra=kind=="reference" ? secondref : first(selected)
    moving=vcat(secondref,selected,extra)
    firstblock=findfirst("PS2",source);lastblock=findnext("END\nEND\n",source,last(firstblock))
    newset=vcat(selected,extra)
    source=source[1:first(firstblock)-1]*"PS2 1\nPOLY 1 $(length(newset))\n"*join(newset,'\n')*"\n"*source[first(lastblock):end]
    isempty(literal) && return source
    lines=split(source,'\n');num=findfirst(line->startswith(line,"NUM 1"),lines)
    stop=findnext(==("END"),lines,num+2)
    points=[Tuple(parse.(Float64,split(lines[i]))) for i in num+2:stop-1]
    old=copy(points);anchor=old[firstref+1]
    for index in (literal=="multiplicity" ? moving : unique(moving))
        x,y=points[index+1];ax,ay=anchor
        points[index+1]=if mode=="RAD"
            dx=x-ax;dy=y-ay;r=hypot(dx,dy);delta=target-nominal
            (x+dx/r*delta,y+dy/r*delta)
        elseif mode=="NSCD"
            (x,y+target-nominal)
        else
            (mode=="SCXY" ? ax+(x-ax)*(target/nominal) : x,ay+(y-ay)*(target/nominal))
        end
    end
    points[end]=points[1]
    for i in eachindex(points);lines[num+1+i]="$(points[i][1]) $(points[i][2])";end
    for i in eachindex(lines)
        lines[i]=="POR1 BOX" || continue
        edge=parse(Int,lines[i+2]);fields=split(lines[i+3]);a=old[edge+1];b=old[edge+2]
        axis=abs(b[1]-a[1])>=abs(b[2]-a[2]) ? 1 : 2
        fraction=(parse(Float64,fields[5+axis])-a[axis])/(b[axis]-a[axis])
        for component in 1:2
            fields[5+component]=string((1-fraction)*points[edge+1][component]+fraction*points[edge+2][component])
        end
        lines[i+3]=join(fields,' ')
    end
    source=join(lines,'\n');beginvar=findfirst("VALVAR",source);ports=findfirst("POR1 BOX",source)
    return source[1:first(beginvar)-1]*source[first(ports):end]
end

function main()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="geovar_point_multiplicity_",cleanup=false)
    paths=[@__FILE__,joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl")]
    append!(paths,[joinpath(repo,"test/fixtures",p) for p in values(sources)])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"actual native explicit reference and repeated adjustable-point identities versus sequential and deduplicated independent literals", "started_utc"=>string(now(UTC)),"source_before"=>hashes(),"cases"=>Any[])
    try
        for mode in ("NSCD","SCUNI","SCXY","RAD"),kind in ("reference","ordinary")
            tag=lowercase(mode)*"_"*kind;matrices=Dict{String,Matrix{ComplexF64}}()
            row=Dict{String,Any}("tag"=>tag,"mode"=>mode,"kind"=>kind);push!(report["cases"],row)
            for input in ("parameter","multiplicity","deduplicated")
                source=joinpath(directory,tag*"_"*input*".son");write(source,duplicate_source(mode,kind;literal=input=="parameter" ? "" : input))
                try
                    native=reference_run(find_em(),source;output_dir=joinpath(directory,tag*"_"*input),deembedded=false)
                    matrices[input]=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
                    row[input*"_status"]="ACCEPT"
                catch err
                    row[input*"_status"]="REJECT";row[input*"_error"]=sprint(showerror,err)
                end
            end
            for input in ("multiplicity","deduplicated")
                haskey(matrices,"parameter") && haskey(matrices,input) || continue
                row[input*"_full_s_error"]=maximum(abs,matrices["parameter"]-matrices[input])
                row[input*"_bit_identical"]=matrices["parameter"]==matrices[input]
            end
            println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained point multiplicity evidence: ",directory)
    end
end
main()
