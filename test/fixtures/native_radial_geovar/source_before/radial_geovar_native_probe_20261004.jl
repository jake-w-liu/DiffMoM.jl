using DiffMoM, SHA, TOML, Dates
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
const template=joinpath(repo,"test/fixtures/native_scaled_geovar/native/anc_positive_parameter.son")

function radial_source(target;literal=false,scale=false,axis="YDIR",sign=1,mode="NSCD")
    points=[(0.,.375),(1.,.375),(1.,.4375),(.75,.625),(1.,.6875),(1.,.8125),(0.,.8125)]
    firstref=2;secondref=3;nominal=.3125
    anchor=points[firstref+1]
    if literal
        for i in (secondref,4)
            x,y=points[i+1];dx=x-anchor[1];dy=y-anchor[2];distance=hypot(dx,dy)
            factor=scale ? target/nominal : 1+(target-nominal)/distance
            points[i+1]=(anchor[1]+dx*factor,anchor[2]+dy*factor)
        end
    end
    block=literal ? "" : "VALVAR Radius LNG $target \"radial point movement\"\n"*
        "GEOVAR Radius RAD $axis $sign $mode\nPOS .5 .5\nNOM $nominal\n"*
        "REF1 POLY 1 1\n$firstref\nREF2 POLY 1 1\n$secondref\n"*
        "PS1 0\nEND\nPS2 1\nPOLY 1 1\n4\nEND\nEND\n"
    ports="POR1 BOX\nPOLY 1 1\n6\n1 50 0 0 0 0 .59375\n"*
        "POR1 BOX\nPOLY 1 1\n1\n2 50 0 0 0 1 .40625\n"
    polygon="NUM 1\n0 8 -1 N 1 1 1 100 100 0 0 0 Y\n"*
        join(("$x $y" for (x,y) in vcat(points,points[1:1])),'\n')*"\nEND\n"
    text=read(template,String);first=findfirst("VALVAR Width",text);last=findfirst("END GEO",text)
    return text[1:first.start-1]*block*ports*polygon*text[last.start:end]
end

function main()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="radial_geovar_",cleanup=false)
    paths=[joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(repo,"src")) for file in files if endswith(file,".jl")]
    append!(paths,[template,@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"native radial movement compared with independent same-distance and proportional literal geometries", "started_utc"=>string(now(UTC)),"source_before"=>hashes(),"cases"=>Any[])
    try
        for target in (.375,.25)
            matrices=Dict{String,Matrix{ComplexF64}}()
            for mode in ("NSCD","SCUNI","SCXY"),axis in ("XDIR","YDIR"),sign in (1,-1)
                tag=(target>.3125 ? "expand" : "contract")*"_"*lowercase(mode)*"_"*lowercase(axis)*(sign==1 ? "_positive" : "_negative")
                source=joinpath(directory,tag*".son");write(source,radial_source(target;axis,sign,mode))
                row=Dict{String,Any}("tag"=>tag,"target"=>target,"mode"=>mode,"axis"=>axis,"direction"=>sign,"status"=>"UNVERIFIED");push!(report["cases"],row)
                try
                    native=reference_run(find_em(),source;output_dir=joinpath(directory,tag),deembedded=false)
                    data=checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
                    matrices[tag]=only(data.s);row["status"]="ACCEPT"
                catch err
                    row["status"]="REJECT";row["error"]=sprint(showerror,err)
                end
                println(row);flush(stdout)
            end
            for scale in (false,true)
                tag=(target>.3125 ? "expand" : "contract")*(scale ? "_proportional_hypothesis" : "_radial_literal")
                source=joinpath(directory,tag*".son");write(source,radial_source(target;literal=true,scale))
                native=reference_run(find_em(),source;output_dir=joinpath(directory,tag),deembedded=false)
                data=checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
                matrices[tag]=only(data.s)
            end
            prefix=target>.3125 ? "expand" : "contract"
            for row in report["cases"]
                row["target"]==target && row["status"]=="ACCEPT" || continue
                parameter=matrices[row["tag"]]
                row["radial_full_s_error"]=maximum(abs,parameter-matrices[prefix*"_radial_literal"])
                row["radial_bit_identical"]=parameter==matrices[prefix*"_radial_literal"]
                row["proportional_full_s_error"]=maximum(abs,parameter-matrices[prefix*"_proportional_hypothesis"])
                println(row);flush(stdout)
            end
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained evidence: ",directory)
    end
end
main()
