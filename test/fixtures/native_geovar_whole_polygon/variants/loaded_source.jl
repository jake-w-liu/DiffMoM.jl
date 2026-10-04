using DiffMoM,SHA,TOML,Dates
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
const template=joinpath(repo,"test/fixtures/native_scaled_geovar/native/anc_positive_parameter.son")
function whole_source(mode,hypothesis="parameter";target=.1875)
    main=[(0.,.375),(.5,.375),(.5,.75),(0.,.75)]
    variable=[(.5,.4375),(1.,.4375),(1.,.5625),(.5,.5625)]
    pad=[(.875,.25),(1.,.25),(1.,.3125),(.875,.3125)]
    nominal=.125;anchor=(1.,.3125)
    transform(point)=if mode=="RAD"
        x,y=point;dx=x-anchor[1];dy=y-anchor[2];r=hypot(dx,dy)
        (x+dx/r*(target-nominal),y+dy/r*(target-nominal))
    elseif mode=="NSCD"
        (point[1],point[2]+target-nominal)
    else
        (mode=="SCXY" ? anchor[1]+(point[1]-anchor[1])*(target/nominal) : point[1],
            anchor[2]+(point[2]-anchor[2])*(target/nominal))
    end
    if hypothesis!="parameter"
        for i in (hypothesis=="reference_only" ? (2,) : eachindex(variable))
            variable[i]=transform(variable[i])
        end
        hypothesis=="double_reference" && (variable[2]=transform(variable[2]))
    end
    block=hypothesis=="parameter" ? "VALVAR Move LNG $target \"whole polygon control\"\n"*
        "GEOVAR Move $(mode=="RAD" ? "RAD" : "ANC") YDIR 1 $(mode=="RAD" ? "NSCD" : mode)\nPOS .5 .5\nNOM $nominal\n"*
        "REF1 POLY 3 1\n2\nREF2 POLY 2 1\n1\nPS1 0\nEND\nPS2 1\nPOLY 2 0\nEND\nEND\n" : ""
    porty=(variable[2][2]+variable[3][2])/2
    ports="POR1 BOX\nPOLY 1 1\n3\n1 50 0 0 0 0 .5625\n"*
        "POR1 BOX\nPOLY 2 1\n1\n2 50 0 0 0 1 $porty\n"
    polygons="NUM 3\n"
    for (id,points) in enumerate((main,variable,pad))
        polygons*="0 5 -1 N $id 1 1 100 100 0 0 0 Y\n"*
            join(("$x $y" for (x,y) in vcat(points,points[1:1])),'\n')*"\nEND\n"
    end
    text=read(template,String);first=findfirst("VALVAR Width",text);last=findfirst("END GEO",text)
    return text[1:first.start-1]*block*ports*polygons*text[last.start:end]
end
function main()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="geovar_whole_polygon_",cleanup=false)
    paths=[@__FILE__,template,joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl"),joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")]
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"native POLY count0 group with separate anchor polygon, variable rectangle and two wall ports; whole-once, reference-only and double-reference literal hypotheses", "source_before"=>hashes(),"cases"=>Any[])
    try
        for mode in ("NSCD","SCUNI","SCXY","RAD")
            matrices=Dict{String,Matrix{ComplexF64}}();row=Dict{String,Any}("mode"=>mode);push!(report["cases"],row)
            for hypothesis in ("parameter","whole_once","reference_only","double_reference")
                tag=lowercase(mode)*"_"*hypothesis;source=joinpath(directory,tag*".son");write(source,whole_source(mode,hypothesis))
                try
                    native=reference_run(find_em(),source;output_dir=joinpath(directory,tag),deembedded=false)
                    matrices[hypothesis]=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
                    row[hypothesis*"_status"]="ACCEPT"
                catch err
                    row[hypothesis*"_status"]="REJECT";row[hypothesis*"_error"]=sprint(showerror,err)
                end
            end
            for hypothesis in ("whole_once","reference_only","double_reference")
                haskey(matrices,"parameter") && haskey(matrices,hypothesis) || continue
                row[hypothesis*"_full_s_error"]=maximum(abs,matrices["parameter"]-matrices[hypothesis])
                row[hypothesis*"_bit_identical"]=matrices["parameter"]==matrices[hypothesis]
            end
            println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained whole-polygon evidence: ",directory)
    end
end
