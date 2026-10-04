using DiffMoM,SHA,TOML,LinearAlgebra
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
const fixture=joinpath(repo,"test/fixtures/native_geovar_point_multiplicity/anchored_radial")
function inputs(mode,sign,repeats)
    text=read(joinpath(fixture,mode*"_reference_parameter.son"),String)
    radial=mode=="rad";nominal=radial ? .3125 : .25;target=nominal+sign*.03125
    anchorindex=radial ? 2 : 0;referenceindex=3;selected=radial ? 4 : 2
    text=replace(text,r"VALVAR (\w+) LNG [^\s]+"=>s"VALVAR \1 LNG TARGET")
    text=replace(text,"TARGET"=>string(target))
    start=findfirst("PS2 1",text);stop=findnext("END\nEND\n",text,last(start))
    list=vcat(selected,fill(referenceindex,repeats))
    text=text[1:first(start)-1]*"PS2 1\nPOLY 1 $(length(list))\n"*join(list,'\n')*"\n"*text[first(stop):end]
    rows=split(text,'\n');num=findfirst(==("NUM 1"),rows);stop=findnext(==("END"),rows,num+2)
    points=[Tuple(parse.(Float64,split(rows[i]))) for i in num+2:stop-1];old=copy(points);anchor=old[anchorindex+1]
    for index in vcat(referenceindex,list)
        x,y=points[index+1];ax,ay=anchor;delta=target-nominal
        if radial
            dx=x-ax;dy=y-ay;radius=hypot(dx,dy)
            points[index+1]=(x+dx/radius*delta,y+dy/radius*delta)
        else
            points[index+1]=(x,y+delta)
        end
    end
    points[end]=points[1]
    for i in eachindex(points);rows[num+1+i]="$(points[i][1]) $(points[i][2])";end
    for i in eachindex(rows)
        rows[i]=="POR1 BOX" || continue
        edge=parse(Int,rows[i+2]);fields=split(rows[i+3]);a=old[edge+1];b=old[edge+2]
        axis=abs(b[1]-a[1])>=abs(b[2]-a[2]) ? 1 : 2
        fraction=(parse(Float64,fields[5+axis])-a[axis])/(b[axis]-a[axis])
        for component in 1:2
            fields[5+component]=string((1-fraction)*points[edge+1][component]+fraction*points[edge+2][component])
        end
        rows[i+3]=join(fields,' ')
    end
    literal=join(rows,'\n');start=findfirst("VALVAR",literal);ports=findfirst("POR1 BOX",literal)
    return text,literal[1:first(start)-1]*literal[first(ports):end]
end
function main()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="geovar_reference_translation_variants_v2_",cleanup=false)
    paths=[@__FILE__,joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl"),joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")]
    append!(paths,[joinpath(fixture,mode*"_reference_parameter.son") for mode in ("nscd","rad")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"fresh native NSCD ANC/RAD explicit moving-reference repetitions1/2, contraction/expansion; independent literals 1e-12 native full-S identity, .005 public full-S and 1e-9 original voltage residual gates at mx=my128", "source_before"=>hashes(),"cases"=>Any[])
    try
        for mode in ("nscd","rad"),sign in (-1,1),repeats in (1,2)
            tag=mode*"_"*(sign<0 ? "contract" : "expand")*"_"*string(repeats)
            pair=inputs(mode,sign,repeats);matrices=Dict{String,Matrix{ComplexF64}}();projects=SonnetProject[];errors=Dict{String,String}()
            for (suffix,text) in zip(("parameter","literal"),pair)
                source=joinpath(directory,tag*"_"*suffix*".son");write(source,text)
                try
                    native=reference_run(find_em(),source;output_dir=joinpath(directory,tag*"_"*suffix),deembedded=false)
                    matrices[suffix]=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
                catch err
                    errors[suffix]=sprint(showerror,err)
                end
                push!(projects,read_sonnet_project(source))
            end
            row=Dict{String,Any}("tag"=>tag,"native_errors"=>errors)
            if length(matrices)==2
                row["native_full_s_error"]=maximum(abs,matrices["parameter"]-matrices["literal"]);row["bit_identical_native"]=matrices["parameter"]==matrices["literal"]
            end
            push!(report["cases"],row)
            row["geometry_error"]=maximum(abs,only(DiffMoM._sonnet_geometry_project(projects[1],1e9).polygons).vertices-only(projects[2].polygons).vertices)
            for (suffix,project) in zip(("parameter","literal"),projects)
                result=solve_sonnet_project(project,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
                raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
                for b in eachindex(raw.problem.basis.port)
                    port=raw.problem.basis.port[b];iszero(port) && continue
                    rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
                end
                row[suffix*"_full_s_error"]=maximum(abs,result.s-get(matrices,"parameter",matrices["literal"]))
                row[suffix*"_voltage_residual"]=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
            end
            row["status"]=get(row,"native_full_s_error",Inf)<=1e-12 && all(row[s*"_full_s_error"]<=.005 && row[s*"_voltage_residual"]<=1e-9 for s in ("parameter","literal")) ? "PASS" : "FAIL"
            println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained variants: ",directory)
    end
    @assert report["source_unchanged"] && length(report["cases"])==8
end
main()
