using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function independent_literal(text,factor)
    rows=split(text,'\n');start=findfirst(row->startswith(row,"NUM "),rows)
    npoly=parse(Int,split(rows[start])[2]);cursor=start+1
    old=Dict{Int,Vector{Tuple{Float64,Float64}}}();positions=Dict{Int,Int}()
    for _ in 1:npoly
        header=split(rows[cursor]);count=parse(Int,header[2]);id=parse(Int,header[5]);positions[id]=cursor
        old[id]=[Tuple(parse.(Float64,split(rows[cursor+i]))) for i in 1:count];cursor+=count+2
    end
    points=deepcopy(old)
    firstrow=findfirst(row->startswith(row,"REF1 "),rows);secondrow=findfirst(row->startswith(row,"REF2 "),rows)
    fixedref=(parse(Int,split(rows[firstrow])[3]),parse(Int,rows[firstrow+1])+1)
    movingref=(parse(Int,split(rows[secondrow])[3]),parse(Int,rows[secondrow+1])+1)
    anchor=old[fixedref[1]][fixedref[2]];reference=old[movingref[1]][movingref[2]]
    header=split(match(r"(?m)^GEOVAR (.*)",text)[1]);radial=header[2]=="RAD";axis=header[3]=="XDIR" ? 1 : 2
    direction=anchor[axis]==reference[axis] ? parse(Int,header[4]) : sign(reference[axis]-anchor[axis])
    nominal=parse(Float64,match(r"(?m)^NOM (\S+)",text)[1]);target=parse(Float64,match(r"VALVAR \w+ LNG (\S+)",text)[1])
    delta=factor*(target-nominal)*(radial ? 1 : direction)
    ps=findfirst(row->startswith(row,"PS2 "),rows);groups=parse(Int,split(rows[ps])[2]);cursor=ps+1;entries=[movingref]
    for _ in 1:groups
        tokens=split(rows[cursor]);id=parse(Int,tokens[2]);count=parse(Int,tokens[3])
        if iszero(count)
            append!(entries,[(id,index) for index in 1:length(old[id])-1]);cursor+=1
        else
            append!(entries,[(id,parse(Int,rows[cursor+q])+1) for q in 1:count]);cursor+=1+count
        end
    end
    for (id,index) in entries
        x,y=points[id][index]
        points[id][index]=if radial
            dx=x-anchor[1];dy=y-anchor[2];radius=hypot(dx,dy)
            (x+dx/radius*delta,y+dy/radius*delta)
        else
            axis==1 ? (x+delta,y) : (x,y+delta)
        end
    end
    for (id,vertices) in points
        vertices[end]=vertices[1];cursor=positions[id]
        for i in eachindex(vertices);rows[cursor+i]="$(vertices[i][1]) $(vertices[i][2])";end
    end
    for i in eachindex(rows)
        rows[i]=="POR1 BOX" || continue
        id=parse(Int,split(rows[i+1])[2]);edge=parse(Int,rows[i+2]);fields=split(rows[i+3]);a=old[id][edge+1];b=old[id][edge+2]
        along=abs(b[1]-a[1])>=abs(b[2]-a[2]) ? 1 : 2
        fraction=(parse(Float64,fields[5+along])-a[along])/(b[along]-a[along])
        for component in 1:2
            fields[5+component]=string((1-fraction)*points[id][edge+1][component]+fraction*points[id][edge+2][component])
        end
        rows[i+3]=join(fields,' ')
    end
    text=join(rows,'\n');start=findfirst("VALVAR",text);ports=findfirst("POR1 BOX",text)
    text[1:first(start)-1]*text[first(ports):end]
end
function main()
    output=mktempdir(joinpath(repo,"data/planar_audit");prefix="geovar_reference_repeated_whole_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    inputs=[joinpath(repo,"test/fixtures/native_rad_reference_headers/rad_ydir_positive_expand_scxy.son")]
    append!(inputs,[joinpath(repo,"test/fixtures/native_geovar_whole_polygon/variants/nscd_$(lowercase(axis))_$(dir)_expand_parameter.son")
        for axis in ("XDIR","YDIR"),dir in ("positive","negative")])
    paths=vcat([@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")],inputs,
        [joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Native repeated whole-polygon selectors, moving-reference multiplicities2/3; RAD all scaling headers and ANC NSCD both axes/directions. Independent literal geometry, unchanged1e-12 complete native S gate; no production edits.","source_before"=>hashes(),"cases"=>Any[])
    cases=Tuple{String,String,Int}[]
    for header in ("NSCD","SCUNI","SCXY"),r in (2,3),sign in (-1,1)
        text=replace(read(first(inputs),String),"\r\n"=>"\n","LNG 0.1875"=>"LNG $(.125+sign*(r==2 ? 2.0^-7 : 2.0^-9))",
            "OPTIONS\n"=>"OPTIONS -j\n","YDIR 1 SCXY"=>"YDIR 1 $(header)")
        text=replace(text,"PS2 1\nPOLY 2 0"=>"PS2 $(r)\n"*join(fill("POLY 2 0",r),'\n'))
        push!(cases,("rad_$(header)_r$(r)_delta$(sign)",text,r))
    end
    for input in inputs[2:end],sign in (-1,1)
        text=replace(read(input,String),"\r\n"=>"\n","LNG 0.1875"=>"LNG $(.125+sign*2.0^-7)",
            "OPTIONS\n"=>"OPTIONS -j\n","PS2 1\nPOLY 2 0"=>"PS2 2\nPOLY 2 0\nPOLY 2 0")
        push!(cases,("$(splitext(basename(input))[1])_delta$(sign)",text,2))
    end
    try
        for (tag,text,r) in cases
            source=joinpath(output,tag*"_parameter.son");write(source,text)
            native=reference_run(find_em(),source;output_dir=joinpath(output,tag*"_parameter"),deembedded=false)
            expected=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
            source=joinpath(output,tag*"_literal.son");write(source,independent_literal(text,1+r*(r-1)))
            native=reference_run(find_em(),source;output_dir=joinpath(output,tag*"_literal"),deembedded=false)
            actual=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
            error=maximum(abs,actual-expected);row=Dict("case"=>tag,"r"=>r,"full_s_error"=>error,"bit_identical"=>actual==expected,"status"=>error<=1e-12 ? "PASS" : "FAIL")
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained repeated whole polygons: ",output)
    end
    @assert report["source_unchanged"] && length(report["cases"])==20 && all(r->r["status"]=="PASS",report["cases"])
end
main()
