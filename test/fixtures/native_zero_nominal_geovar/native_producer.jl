using DiffMoM, SHA, TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
const template=joinpath(repo,"test/fixtures/native_scaled_geovar/native/anc_positive_parameter.son")

function zero_source(axis,direction,target,literal)
    points=[(0.,.375),(1.,.375),(1.,.4375),(.8,.5),(1.,.5625),(1.,.625),(0.,.625)]
    geometry="VALVAR Offset LNG $target \"zero saved X offset\"\n"*
        "GEOVAR Offset ANC $axis $direction NSCD\nPOS .5 .5\nNOM 0\n"*
        "REF1 POLY 1 1\n2\nREF2 POLY 1 1\n4\nPS1 0\nEND\n"*
        "PS2 1\nPOLY 1 1\n3\nEND\nEND\n"
    if literal
        for index in (4,5)
            points[index]=(points[index][1]+direction*target,points[index][2])
        end
        geometry=""
    end
    ports="POR1 BOX\nPOLY 1 1\n6\n1 50 0 0 0 0 .5\n"*
        "POR1 BOX\nPOLY 1 1\n1\n2 50 0 0 0 1 .40625\n"
    if axis=="YDIR"
        points=[(y,x) for (x,y) in points]
        ports="POR1 BOX\nPOLY 1 1\n6\n1 50 0 0 0 .5 0\n"*
            "POR1 BOX\nPOLY 1 1\n1\n2 50 0 0 0 .40625 1\n"
    end
    polygon="NUM 1\n0 8 -1 N 1 1 1 100 100 0 0 0 Y\n"*
        join(("$x $y" for (x,y) in vcat(points,points[1:1])),'\n')*"\nEND\n"
    text=read(template,String);first=findfirst("VALVAR Width",text);last=findfirst("END GEO",text)
    return text[1:first.start-1]*geometry*ports*polygon*text[last.start:end]
end

function main()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="geovar_zero_nominal_axes_",cleanup=false)
    cp(@__FILE__,joinpath(directory,"producer.jl"))
    paths=vcat([@__FILE__,template,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")],
        [joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(repo,"src")) for file in files])
    hashes()=Dict(path=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"Fresh zero saved X/Y dimension ANC/NSCD; reference coordinates coincide on selected axis. Both directions and two target sizes; independent literal controls; ports unchanged.","source_before"=>hashes(),"cases"=>Any[])
    try
        for axis in ("XDIR","YDIR"),direction in (-1,1),target in (.0625,.125)
            matrices=Dict{String,Matrix{ComplexF64}}()
            row=Dict{String,Any}("axis"=>axis,"direction"=>direction,"target"=>target)
            for kind in ("parameter","literal")
                tag="$(axis)_dir_$(direction)_target_$(target)_$kind"
                source=joinpath(directory,tag*".son");write(source,zero_source(axis,direction,target,kind=="literal"))
                try
                    native=reference_run(find_em(),source;output_dir=joinpath(directory,tag),deembedded=false)
                    matrix=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
                    matrices[kind]=matrix;row[kind*"_status"]="ACCEPT"
                catch err
                    row[kind*"_status"]="REJECT";row[kind*"_error"]=sprint(showerror,err)
                end
                if kind=="parameter"
                    p=read_sonnet_project(source)
                    try
                        effective=DiffMoM._sonnet_geometry_project(p,1e9)
                        row["public_geometry_status"]="ACCEPT"
                    catch err
                        row["public_geometry_status"]="REJECT";row["public_geometry_error"]=sprint(showerror,err)
                    end
                end
            end
            if length(matrices)==2
                row["bit_identical_s"]=matrices["parameter"]==matrices["literal"]
                row["full_s_error"]=maximum(abs,matrices["parameter"]-matrices["literal"])
            end
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained zero nominal native evidence: ",directory)
    end
    @assert report["source_unchanged"] && length(report["cases"])==8
    @assert all(get(row,"bit_identical_s",false) for row in report["cases"])
end
main()
