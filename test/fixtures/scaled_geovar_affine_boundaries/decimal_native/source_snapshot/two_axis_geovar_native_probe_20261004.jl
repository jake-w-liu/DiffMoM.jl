using DiffMoM,SHA,TOML,Dates
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
const template=joinpath(repo,"test/fixtures/native_scaled_geovar/native/anc_positive_parameter.son")

function two_axis_source(kind,sign,width;literal=false,axis_only=false)
    nominal=.25
    points=[(0.,.375),(1.,.375),(1.,.4375),(.8,.5),(1.,.5625),(1.,.625),(0.,.625)]
    firstref,secondref=sign==1 ? (1,5) : (5,1)
    anchor=(1.,kind=="SYM" ? .5 : sign==1 ? .375 : .625)
    a=kind=="ANC" ? Int[] : sign==1 ? [2,3] : [4,3]
    b=kind=="ANC" ? [2,3,4] : sign==1 ? [4] : [2]
    if literal
        selected=kind=="ANC" ? vcat(b,secondref) : vcat(a,b,firstref,secondref)
        for i in selected
            x,y=points[i+1]
            x=axis_only ? x : anchor[1]+(x-anchor[1])*(width/nominal)
            y=anchor[2]+(y-anchor[2])*(width/nominal)
            points[i+1]=(x,y)
        end
    end
    block=""
    if !literal
        pointset(tag,ids)=isempty(ids) ? "$tag 0\nEND\n" :
            "$tag 1\nPOLY 1 $(length(ids))\n"*join(ids,'\n')*"\nEND\n"
        block="VALVAR Width LNG $width \"two-axis notch control\"\n"*
            "GEOVAR Width $kind YDIR $sign SCXY\nPOS .5 .5\nNOM $nominal\n"*
            "REF1 POLY 1 1\n$firstref\nREF2 POLY 1 1\n$secondref\n"*
            pointset("PS1",a)*pointset("PS2",b)*"END\n"
    end
    right=(points[2][2]+points[3][2])/2
    ports="POR1 BOX\nPOLY 1 1\n6\n1 50 0 0 0 0 .5\n"*
        "POR1 BOX\nPOLY 1 1\n1\n2 50 0 0 0 1 $right\n"
    polygon="NUM 1\n0 8 -1 N 1 1 1 100 100 0 0 0 Y\n"*
        join(("$x $y" for (x,y) in vcat(points,points[1:1])),'\n')*"\nEND\n"
    text=read(template,String);first=findfirst("VALVAR Width",text);last=findfirst("END GEO",text)
    return text[1:first.start-1]*block*ports*polygon*text[last.start:end]
end

function transpose_source(text)
    rows=split(replace(text,"YDIR"=>"XDIR"),'\n');inside=false;expected=false
    for i in eachindex(rows)
        t=split(rows[i])
        if startswith(rows[i],"NUM 1");inside=true;expected=true;continue;end
        if inside && expected;expected=false;continue;end
        if inside && rows[i]=="END";inside=false;continue;end
        if inside && length(t)==2
            rows[i]=t[2]*" "*t[1]
        elseif length(t)==7 && t[1] in ("1","2") && t[2]=="50"
            rows[i]=join(vcat(t[1:5],t[7],t[6])," ")
        end
    end
    join(rows,'\n')
end

function main()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="two_axis_geovar_",cleanup=false)
    sources=[joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(repo,"src")) for file in files if endswith(file,".jl")]
    append!(sources,[template,@__FILE__,joinpath(repo,"Project.toml"),joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in sources)
    report=Dict{String,Any}("scope"=>"fresh native SCXY ANC/SYM signed XDIR/YDIR expansion/contraction; independent proportional literal and retained axis-only hypotheses; attached wall ports",
        "started_utc"=>string(now(UTC)),"source_before"=>hashes(),"cases"=>Any[])
    try
        for axis in ("XDIR","YDIR"),kind in ("ANC","SYM"),sign in (1,-1),width in (.375,.125)
            tag=lowercase(axis)*"_"*lowercase(kind)*(sign==1 ? "_positive" : "_negative")*(width>.25 ? "_expand" : "_contract")
            matrices=Matrix{ComplexF64}[]
            row=Dict{String,Any}("tag"=>tag,"axis"=>axis,"kind"=>kind,"direction"=>sign,"nominal"=>.25,"target"=>width,"status"=>"UNVERIFIED")
            push!(report["cases"],row)
            modes=axis=="YDIR" && width>.25 ? ("parameter","proportional_literal","axis_only_hypothesis") : ("parameter","proportional_literal")
            for mode in modes
                source=joinpath(directory,tag*"_"*mode*".son")
                text=two_axis_source(kind,sign,width;literal=mode!="parameter",axis_only=mode=="axis_only_hypothesis")
                write(source,axis=="XDIR" ? transpose_source(text) : text)
                native=reference_run(find_em(),source;output_dir=joinpath(directory,tag*"_"*mode),deembedded=false)
                data=checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
                push!(matrices,only(data.s))
            end
            row["full_s_error"]=maximum(abs,matrices[1]-matrices[2]);row["bit_identical_s"]=matrices[1]==matrices[2]
            length(matrices)==3 && (row["axis_only_full_s_error"]=maximum(abs,matrices[1]-matrices[3]))
            row["status"]=row["full_s_error"]<=1e-12 ? "PASS" : "FAIL"
            println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained evidence: ",directory)
    end
    @assert report["source_unchanged"] && all(row->row["status"]=="PASS",report["cases"])
end
main()
