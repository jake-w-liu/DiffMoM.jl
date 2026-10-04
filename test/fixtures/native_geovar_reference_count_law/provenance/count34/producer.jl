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
    target=parse(Float64,match(r"VALVAR \w+ LNG (\S+)",text)[1]);delta=factor*(target-nominal)
    ps=findfirst(==("PS2 1"),rows);count=parse(Int,split(rows[ps+1])[3])
    entries=vcat(4,[parse(Int,rows[ps+1+q])+1 for q in 1:count])
    for index in entries
        x,y=points[index]
        points[index]=if radial
            dx=x-anchor[1];dy=y-anchor[2];radius=hypot(dx,dy)
            (x+dx/radius*delta,y+dy/radius*delta)
        else
            (x,y+delta)
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
function main()
    directory=mktempdir(joinpath(repo,"data/planar_audit");prefix="geovar_reference_count_law_probe_",cleanup=false)
    cp(@__FILE__,joinpath(directory,"producer.jl"))
    fixture=joinpath(repo,"test/fixtures/native_geovar_reference_variants/native")
    paths=vcat([@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")],
        [joinpath(fixture,mode*"_expand_2_parameter.son") for mode in ("nscd","rad")],
        [joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Fresh native explicit moving-reference counts3/4: expansion/contraction in ANC NSCD and RAD. Linear, triangular, quadratic and exponential common-shift hypotheses tested against independently authored literal geometries at unchanged1e-12 full complex S gate. No production edits.","source_before"=>hashes(),"cases"=>Any[])
    try
        for mode in ("nscd","rad"),repeats in (3,4),sign in (-1,1)
            change=sign*(repeats==3 ? .0078125 : .001953125)
            text=parameter_text(mode,repeats,change);tag="$(mode)_$(repeats)_$(sign)"
            source=joinpath(directory,tag*"_parameter.son");write(source,text)
            native=reference_run(find_em(),source;output_dir=joinpath(directory,tag*"_parameter"),deembedded=false)
            expected=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
            factors=repeats==3 ? (5,6,7) : (7,10,13,15)
            outcomes=Any[]
            for factor in factors
                source=joinpath(directory,tag*"_factor_$(factor).son");write(source,literal_text(text,factor))
                native=reference_run(find_em(),source;output_dir=joinpath(directory,tag*"_factor_$(factor)"),deembedded=false)
                actual=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
                push!(outcomes,Dict("factor"=>factor,"full_s_error"=>maximum(abs,actual-expected),"bit_identical"=>actual==expected))
            end
            row=Dict("case"=>tag,"explicit_reference_count"=>repeats,"outcomes"=>outcomes,
                "matching_factors"=>[r["factor"] for r in outcomes if r["full_s_error"]<=1e-12])
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained count law hypotheses: ",directory)
    end
    @assert report["source_unchanged"] && length(report["cases"])==8
end
main()
