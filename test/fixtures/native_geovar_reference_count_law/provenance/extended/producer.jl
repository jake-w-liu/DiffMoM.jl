using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function parameter_text(mode,repeats,change)
    path=joinpath(repo,"test/fixtures/native_geovar_reference_variants/native",mode*"_expand_2_parameter.son")
    text=replace(read(path,String),"\r\n"=>"\n");nominal=mode=="rad" ? .3125 : .25
    text=replace(text,r"VALVAR (\w+) LNG \S+"=>s"VALVAR \1 LNG TARGET")
    text=replace(text,"TARGET"=>string(nominal+change),"OPTIONS\n"=>"OPTIONS -j\n")
    start=findfirst("PS2 1",text);stop=findnext("END\nEND\n",text,last(start))
    entries=vcat(mode=="rad" ? 4 : 2,fill(3,repeats))
    return text[1:first(start)-1]*"PS2 1\nPOLY 1 $(length(entries))\n"*join(entries,'\n')*"\n"*text[first(stop):end]
end
function literal_text(text,factor)
    rows=split(text,'\n');num=findfirst(==("NUM 1"),rows);stop=findnext(==("END"),rows,num+2)
    old=[Tuple(parse.(Float64,split(rows[i]))) for i in num+2:stop-1];points=copy(old)
    radial=occursin(" RAD ",text);anchor=old[radial ? 3 : 1]
    nominal=parse(Float64,match(r"(?m)^NOM (\S+)",text)[1])
    target=parse(Float64,match(r"VALVAR \w+ LNG (\S+)",text)[1]);header=split(match(r"(?m)^GEOVAR (.*)",text)[1]);axis=header[3]=="XDIR" ? 1 : 2
    delta=factor*(target-nominal)*(radial ? 1 : parse(Int,header[4]))
    ps=findfirst(row->startswith(row,"PS2 "),rows);groups=parse(Int,split(rows[ps])[2]);cursor=ps+1;entries=[4]
    for _ in 1:groups
        count=parse(Int,split(rows[cursor])[3]);append!(entries,[parse(Int,rows[cursor+q])+1 for q in 1:count]);cursor+=1+count
    end
    for index in entries
        x,y=points[index]
        points[index]=if radial
            dx=x-anchor[1];dy=y-anchor[2];radius=hypot(dx,dy)
            (x+dx/radius*delta,y+dy/radius*delta)
        else
            axis==1 ? (x+delta,y) : (x,y+delta)
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
    text=join(rows,'\n');start=findfirst("VALVAR",text);ports=findfirst("POR1 BOX",text)
    return text[1:first(start)-1]*text[first(ports):end]
end
function rotate_text(text)
    rows=split(text,'\n');num=findfirst(==("NUM 1"),rows);stop=findnext(==("END"),rows,num+2)
    for i in num+2:stop-1;xy=split(rows[i]);rows[i]=join(reverse(xy),' ');end
    for i in eachindex(rows)
        rows[i]=="POR1 BOX" || continue
        fields=split(rows[i+3]);fields[6],fields[7]=fields[7],fields[6];rows[i+3]=join(fields,' ')
    end
    replace(join(rows,'\n'),"YDIR"=>"XDIR")
end
function main()
    directory=mktempdir(joinpath(repo,"data/planar_audit");prefix="geovar_reference_count_law_extended_",cleanup=false)
    cp(@__FILE__,joinpath(directory,"producer.jl"))
    fixture=joinpath(repo,"test/fixtures/native_geovar_reference_variants/native")
    paths=vcat([@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")],
        [joinpath(fixture,mode*"_expand_2_parameter.son") for mode in ("nscd","rad")],
        [joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Fresh native r(r-1)+1 common-shift hypothesis: explicit moving-reference counts5,8,16,32 both directions in ANC NSCD/RAD; count4 orders/groups/ordinary duplicates, X/Y axes, direction/scaling header controls. Independent literal geometry, unchanged1e-12 full complex S gate; no production edits.","source_before"=>hashes(),"cases"=>Any[])
    cases=Tuple{String,String,Int}[]
    for mode in ("nscd","rad"),r in (5,8,16,32),sign in (-1,1)
        change=sign*(r==5 ? 2.0^-10 : r==8 ? 2.0^-12 : r==16 ? 2.0^-15 : 2.0^-18)
        push!(cases,("$(mode)_$(r)_$(sign)",parameter_text(mode,r,change),r))
    end
    for mode in ("nscd","rad"),variant in ("order","groups","duplicates","xnegative")
        text=parameter_text(mode,4,-2.0^-9);ordinary=mode=="rad" ? 4 : 2
        if variant=="order"
            text=replace(text,"$(ordinary)\n3\n3\n3\n3\nEND"=>"3\n3\n$(ordinary)\n3\n3\nEND")
        elseif variant=="groups"
            text=replace(text,"PS2 1\nPOLY 1 5\n$(ordinary)\n3\n3\n3\n3\nEND"=>
                "PS2 2\nPOLY 1 2\n3\n$(ordinary)\nPOLY 1 3\n3\n3\n3\nEND")
        elseif variant=="duplicates"
            text=replace(text,"POLY 1 5\n$(ordinary)\n"=>"POLY 1 6\n$(ordinary)\n$(ordinary)\n")
        else
            text=replace(rotate_text(text),"XDIR 1"=>"XDIR -1")
        end
        push!(cases,("$(mode)_4_$(variant)",text,4))
    end
    for header in ("SCUNI","SCXY")
        push!(cases,("rad_4_$(header)",replace(parameter_text("rad",4,-2.0^-9),"YDIR 1 NSCD"=>"XDIR -1 $(header)"),4))
    end
    try
        for (tag,text,r) in cases
            factor=r*(r-1)+1
            # Read every explicit entry, including entries across POLY groups,
            # before authoring the literal. Its movement helper is independent
            # of the production geometry adapter.
            source=joinpath(directory,tag*"_parameter.son");write(source,text)
            native=reference_run(find_em(),source;output_dir=joinpath(directory,tag*"_parameter"),deembedded=false)
            expected=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
            literal=literal_text(text,factor)
            source=joinpath(directory,tag*"_literal.son");write(source,literal)
            native=reference_run(find_em(),source;output_dir=joinpath(directory,tag*"_literal"),deembedded=false)
            actual=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
            error=maximum(abs,actual-expected)
            row=Dict("case"=>tag,"explicit_reference_count"=>r,"factor"=>factor,"full_s_error"=>error,
                "bit_identical"=>actual==expected,"status"=>error<=1e-12 ? "PASS" : "FAIL")
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained extended count law: ",directory)
    end
    @assert report["source_unchanged"] && length(report["cases"])==26
    @assert all(r->r["status"]=="PASS",report["cases"])
end
main()
