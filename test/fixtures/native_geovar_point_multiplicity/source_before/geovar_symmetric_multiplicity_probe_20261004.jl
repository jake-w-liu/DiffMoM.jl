using DiffMoM,SHA,TOML,Dates
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference

function symmetric_duplicate_source(mode,side;literal=false)
    sourcefile=mode=="SCXY" ? "native_two_axis_geovar/native/ydir_sym_positive_expand_parameter.son" :
        "native_sonnet_geometry_variables/accepted/single_ydir_sym_positive_parameter.son"
    source=read(joinpath(repo,"test/fixtures",sourcefile),String)
    source=replace(source,"VALVAR Width LNG 0.375"=>"VALVAR Width LNG 0.3125"," NSCD"=>" $mode")
    a,b,firstref,secondref=mode=="SCXY" ? ([2,3],[4],1,5) : ([1],[2],0,3)
    side==1 ? push!(a,first(a)) : push!(b,first(b))
    set(tag,ids)="$tag 1\nPOLY 1 $(length(ids))\n"*join(ids,'\n')*"\nEND\n"
    start=findfirst("PS1",source);stop=findfirst("POR1 BOX",source)
    source=source[1:first(start)-1]*set("PS1",a)*set("PS2",b)*"END\n"*source[first(stop):end]
    literal || return source
    rows=split(source,'\n');num=findfirst(row->startswith(row,"NUM 1"),rows);lastrow=findnext(==("END"),rows,num+2)
    points=[Tuple(parse.(Float64,split(rows[i]))) for i in num+2:lastrow-1];old=copy(points)
    anchor=((old[firstref+1][1]+old[secondref+1][1])/2,(old[firstref+1][2]+old[secondref+1][2])/2)
    for (ids,shift) in ((vcat(a,firstref),-.03125),(vcat(b,secondref),.03125)),index in ids
        x,y=points[index+1]
        points[index+1]=mode=="NSCD" ? (x,y+shift) :
            (mode=="SCXY" ? anchor[1]+(x-anchor[1])*1.25 : x,anchor[2]+(y-anchor[2])*1.25)
    end
    points[end]=points[1]
    for i in eachindex(points);rows[num+1+i]="$(points[i][1]) $(points[i][2])";end
    for i in eachindex(rows)
        rows[i]=="POR1 BOX" || continue
        edge=parse(Int,rows[i+2]);fields=split(rows[i+3]);p=old[edge+1];q=old[edge+2]
        axis=abs(q[1]-p[1])>=abs(q[2]-p[2]) ? 1 : 2
        fraction=(parse(Float64,fields[5+axis])-p[axis])/(q[axis]-p[axis])
        for component in 1:2
            fields[5+component]=string((1-fraction)*points[edge+1][component]+fraction*points[edge+2][component])
        end
        rows[i+3]=join(fields,' ')
    end
    source=join(rows,'\n');start=findfirst("VALVAR",source);stop=findfirst("POR1 BOX",source)
    source[1:first(start)-1]*source[first(stop):end]
end

function main()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="geovar_symmetric_multiplicity_",cleanup=false)
    paths=[@__FILE__,joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl")]
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"native repeated ordinary vertices in either symmetric set; independent sequential literal", "source_before"=>hashes(),"cases"=>Any[])
    try
        for mode in ("NSCD","SCUNI","SCXY"),side in (1,2)
            tag=lowercase(mode)*"_ps"*string(side);matrices=Matrix{ComplexF64}[]
            for literal in (false,true)
                label=tag*(literal ? "_literal" : "_parameter");source=joinpath(directory,label*".son")
                write(source,symmetric_duplicate_source(mode,side;literal))
                native=reference_run(find_em(),source;output_dir=joinpath(directory,label),deembedded=false)
                push!(matrices,only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s))
            end
            row=Dict("tag"=>tag,"mode"=>mode,"side"=>side,"full_s_error"=>maximum(abs,matrices[1]-matrices[2]),"bit_identical_s"=>matrices[1]==matrices[2])
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained symmetric multiplicity evidence: ",directory)
    end
    @assert all(row->row["bit_identical_s"],report["cases"])
end
main()
