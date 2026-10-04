using DiffMoM, SHA, TOML, Dates
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference

# Copy only the historical literal-project constructor into this new probe.
# Its archived file remains byte-exact and is never executed as a script.
const template=joinpath(repo,"data/prior_worker_validation_20261004/validation/planar_audit/fresh_geovar_20261004/probe.jl")
let text=read(template,String)
    first=findfirst("function geovar_source(",text)
    last=findfirst("function main()",text)
    include_string(Main,text[first.start:last.start-1],"retained_geovar_project_constructor")
end

function scaled_source(kind,sign;literal=false,translation=false)
    width=.375;nominal=.25
    points=[(0.,.375),(1.,.375),(1.,.4375),(.8,.5),
        (1.,.5625),(1.,.625),(0.,.625)]
    anchor=kind=="SYM" ? .5 : sign==1 ? .375 : .625
    firstref,secondref=sign==1 ? (0,6) : (6,0)
    a=kind=="ANC" ? Int[] : sign==1 ? [1,2,3] : [4,5]
    b=kind=="ANC" ? collect(1:5) : sign==1 ? [4,5] : [1,2,3]
    if literal
        selected=kind=="ANC" ? vcat(b,secondref) : vcat(a,b,firstref,secondref)
        for i in selected
            x,y=points[i+1]
            if translation
                side=i in vcat(a,firstref) ? 1 : 2
                delta=sign*(width-nominal)
                y+=kind=="ANC" ? delta : side==1 ? -delta/2 : delta/2
            else
                y=anchor+(y-anchor)*(width/nominal)
            end
            points[i+1]=(x,y)
        end
    end
    block=""
    if !literal
        set(tag,ids)=isempty(ids) ? "$tag 0\nEND\n" :
            "$tag 1\nPOLY 1 $(length(ids))\n"*join(ids,'\n')*"\nEND\n"
        block="VALVAR Width LNG $width \"scaled notch control\"\n"*
            "GEOVAR Width $kind YDIR $sign SCUNI\nPOS .5 .5\nNOM $nominal\n"*
            "REF1 POLY 1 1\n$firstref\nREF2 POLY 1 1\n$secondref\n"*
            set("PS1",a)*set("PS2",b)*"END\n"
    end
    text=geovar_source(kind,sign,nominal;literal=true)
    text=replace(text,"POR1 BOX\nPOLY 1 1\n3"=>block*"POR1 BOX\nPOLY 1 1\n6")
    left=literal ? (points[1][2]+points[7][2])/2 : .5
    right=literal ? (points[2][2]+points[3][2])/2 : .40625
    text=replace(text,"1 50 0 0 0 0 .5"=>"1 50 0 0 0 0 $left",
        "2 50 0 0 0 1 .5"=>"2 50 0 0 0 1 $right")
    first=findfirst("NUM 1\n",text);last=findfirst("END GEO",text)
    polygon="NUM 1\n0 8 -1 N 1 1 1 100 100 0 0 0 Y\n"*
        join(("$x $y" for (x,y) in vcat(points,points[1:1])),'\n')*"\nEND\n"
    return text[1:first.start-1]*polygon*text[last.start:end]
end

function main()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="scaled_geovar_",cleanup=false)
    em=find_em();em===nothing && error("installed native engine unavailable")
    sources=vcat([joinpath(folder,file) for (folder,_,files) in walkdir(joinpath(repo,"src"))
        for file in files if endswith(file,".jl")],
        [joinpath(repo,"Project.toml"),joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"),template,@__FILE__])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in sources)
    before=hashes()
    report=Dict{String,Any}("scope"=>"fresh native SCUNI one-axis scaled ANC/SYM notched polygons versus independent proportional and rejected translation hypotheses; no production edit",
        "directory"=>directory,"started_utc"=>string(now(UTC)),"source_before"=>before,"cases"=>Any[])
    try
        for kind in ("ANC","SYM"),sign in (1,-1)
            tag=lowercase(kind)*(sign==1 ? "_positive" : "_negative")
            matrices=Matrix{ComplexF64}[]
            row=Dict{String,Any}("kind"=>kind,"direction"=>sign,"status"=>"UNVERIFIED")
            push!(report["cases"],row)
            for mode in ("parameter","proportional_literal","translation_hypothesis")
                name=tag*"_"*mode
                source=joinpath(directory,name*".son")
                write(source,scaled_source(kind,sign;literal=mode!="parameter",translation=mode=="translation_hypothesis"))
                native=reference_run(em,source;output_dir=joinpath(directory,name),deembedded=false)
                data=checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
                push!(matrices,only(data.s))
                row[mode*"_s_real"]=vec(real(matrices[end]))
                row[mode*"_s_imag"]=vec(imag(matrices[end]))
                row[mode*"_source_sha256"]=bytes2hex(sha256(read(source)))
            end
            row["proportional_full_s_error"]=maximum(abs,matrices[1]-matrices[2])
            row["translation_full_s_error"]=maximum(abs,matrices[1]-matrices[3])
            row["bit_identical_proportional"]=matrices[1]==matrices[2]
            row["status"]=row["proportional_full_s_error"]<=1e-12 ? "PASS" : "FAIL"
            println(tag," ",row["status"]," proportional=",row["proportional_full_s_error"]," translation=",row["translation_full_s_error"])
            flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Evidence retained: ",directory)
    end
    report["source_unchanged"] || error("probe source changed")
    all(row->row["status"]=="PASS",report["cases"]) || error("scaled parameter hypothesis did not match native")
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
function main_x()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="scaled_geovar_x_",cleanup=false)
    sources=[joinpath(folder,file) for (folder,_,files) in walkdir(joinpath(repo,"src")) for file in files if endswith(file,".jl")]
    push!(sources,@__FILE__)
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in sources)
    report=Dict{String,Any}("scope"=>"fresh native XDIR SCUNI controls; coordinate-transposed independently declared notched polygons","cases"=>Any[],"source_before"=>hashes())
    try
        for kind in ("ANC","SYM"),sign in (1,-1)
            tag=lowercase(kind)*(sign==1 ? "_positive" : "_negative");matrices=Matrix{ComplexF64}[]
            row=Dict{String,Any}("kind"=>kind,"direction"=>sign,"axis"=>"XDIR","status"=>"UNVERIFIED");push!(report["cases"],row)
            for mode in ("parameter","proportional_literal")
                source=joinpath(directory,tag*"_"*mode*".son")
                write(source,transpose_source(scaled_source(kind,sign;literal=mode!="parameter")))
                native=reference_run(find_em(),source;output_dir=joinpath(directory,tag*"_"*mode),deembedded=false)
                data=checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
                push!(matrices,only(data.s))
            end
            row["full_s_error"]=maximum(abs,matrices[1]-matrices[2]);row["bit_identical_s"]=matrices[1]==matrices[2]
            row["status"]=row["full_s_error"]<=1e-12 ? "PASS" : "FAIL"
            println(tag," ",row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained evidence: ",directory)
    end
    @assert report["source_unchanged"] && all(row->row["status"]=="PASS",report["cases"])
end
main_x()
