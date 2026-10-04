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
    radial=occursin(" RAD ",text);ref1=findfirst(row->startswith(row,"REF1 "),rows);ref2=findfirst(row->startswith(row,"REF2 "),rows)
    firstindex=parse(Int,rows[ref1+1])+1;secondindex=parse(Int,rows[ref2+1])+1;anchor=old[firstindex]
    nominal=parse(Float64,match(r"(?m)^NOM (\S+)",text)[1])
    target=parse(Float64,match(r"VALVAR \w+ LNG (\S+)",text)[1]);header=split(match(r"(?m)^GEOVAR (.*)",text)[1]);axis=header[3]=="XDIR" ? 1 : 2
    delta=factor*(target-nominal)*(radial ? 1 : parse(Int,header[4]))
    ps=findfirst(row->startswith(row,"PS2 "),rows);groups=parse(Int,split(rows[ps])[2]);cursor=ps+1;entries=[secondindex]
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
function reverse_refs(text,r)
    text=replace(text,"REF1 POLY 1 1\n0\nREF2 POLY 1 1\n3"=>"REF1 POLY 1 1\n3\nREF2 POLY 1 1\n0")
    start=findfirst("PS2 1",text);stop=findnext("END\nEND\n",text,last(start))
    entries=vcat(1,fill(0,r))
    text[1:first(start)-1]*"PS2 1\nPOLY 1 $(length(entries))\n"*join(entries,'\n')*"\n"*text[first(stop):end]
end
function main()
    directory=mktempdir(joinpath(repo,"data/planar_audit");prefix="geovar_reference_orientation_",cleanup=false)
    cp(@__FILE__,joinpath(directory,"producer.jl"))
    zero=joinpath(repo,"test/fixtures/native_zero_nominal_geovar/native/YDIR_dir_1_target_0.125_parameter.son")
    paths=vcat([@__FILE__,zero,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"),
        joinpath(repo,"test/fixtures/native_geovar_reference_variants/native/nscd_expand_2_parameter.son")],
        [joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Native ANC declared direction versus reference-coordinate orientation, explicit counts0,1,2,4 and zero saved offsets. Original full-S1e-12 gate; independent signed literal hypotheses; no production edits.","source_before"=>hashes(),"cases"=>Any[])
    cases=Tuple{String,String,Int,Int}[]
    for r in (0,1,2,4),change_sign in (-1,1),direction in (-1,1)
        text=parameter_text("nscd",r,change_sign*(r<=2 ? 2.0^-7 : 2.0^-9))
        if direction<0;text=replace(text,"YDIR 1"=>"YDIR -1");else;text=reverse_refs(text,r);end
        push!(cases,("r$(r)_delta$(change_sign)_dir$(direction)_opposed",text,r,-1))
    end
    for r in (2,4),direction in (-1,1),axis in ("XDIR","YDIR")
        text=replace(read(zero,String),"\r\n"=>"\n","LNG 0.125"=>"LNG 0.0009765625","OPTIONS\n"=>"OPTIONS -j\n")
        start=findfirst("PS2 1",text);stop=findnext("END\nEND\n",text,last(start));entries=vcat(3,fill(4,r))
        text=text[1:first(start)-1]*"PS2 1\nPOLY 1 $(length(entries))\n"*join(entries,'\n')*"\n"*text[first(stop):end]
        text=replace(text,"YDIR 1"=>"YDIR $(direction)")
        axis=="XDIR" && (text=rotate_text(text))
        push!(cases,("zero_r$(r)_dir$(direction)_$(axis)",text,r,1))
    end
    try
        for (tag,text,r,expected_sign) in cases
            source=joinpath(directory,tag*"_parameter.son");write(source,text)
            native=reference_run(find_em(),source;output_dir=joinpath(directory,tag*"_parameter"),deembedded=false)
            expected=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
            outcomes=Any[];factor=1+r*(r-1)
            for sign in (-1,1)
                source=joinpath(directory,tag*"_factor$(sign*factor).son");write(source,literal_text(text,sign*factor))
                native=reference_run(find_em(),source;output_dir=joinpath(directory,tag*"_factor$(sign*factor)"),deembedded=false)
                actual=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
                push!(outcomes,Dict("factor"=>sign*factor,"full_s_error"=>maximum(abs,actual-expected),"bit_identical"=>actual==expected))
            end
            row=Dict("case"=>tag,"r"=>r,"outcomes"=>outcomes,"proposed_sign"=>expected_sign,
                "matching_factors"=>[x["factor"] for x in outcomes if x["full_s_error"]<=1e-12])
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained reference orientation: ",directory)
    end
    @assert report["source_unchanged"] && length(report["cases"])==24
end
main()
