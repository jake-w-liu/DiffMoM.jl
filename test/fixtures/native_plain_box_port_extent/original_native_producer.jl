using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function project_text(axis,lo,hi)
    mid=(lo+hi)/2
    points=axis==:y ? [(0.,lo),(1.,lo),(1.,hi),(0.,hi),(0.,lo)] : [(lo,0.),(hi,0.),(hi,1.),(lo,1.),(lo,0.)]
    ports=axis==:y ? [(3,0.,mid),(1,1.,mid)] : [(0,mid,0.),(2,mid,1.)]
    porttext=join(["POR1 BOX\nPOLY 1 1\n$(edge)\n$(i) 50 0 0 0 $(x) $(y)" for (i,(edge,x,y)) in enumerate(ports)],'\n')
    polygon=join(["$(x) $(y)" for (x,y) in points],'\n')
    """
FTYP SONPROJ 19
VER "18.53"
DIM
ANG DEG
CAP PF
CON /OH
FREQ GHZ
IND NH
LNG MM
RES OH
END DIM
CONTROL
VARSWP
OPTIONS -j
SPEED 0
SUBSPLAM N 100
END CONTROL
GEO
TMET "PEC" 0 SUP 0 0 0 0
BMET "PEC" 0 SUP 0 0 0 0
BOX 1 1 1 64 64 100 0
.1 1 1 0 0 0 2 "Air"
.1 4 1 0 0 0 2 "Substrate"
$(porttext)
NUM 1
0 5 -1 N 1 1 1 100 100 0 0 0 Y
$(polygon)
END
END GEO
VARSWP
ENABLED Y
FREQ Y AN SWEEP 1
END
END VARSWP
FILEOUT
TOUCH ND Y native_raw.s2p IC 15 S RI R 50
FOLDER .
END FILEOUT
"""
end
function main()
    output=mktempdir(joinpath(repo,"data/planar_audit");prefix="plain_box_port_extent_",cleanup=false);cp(@__FILE__,joinpath(output,"producer.jl"))
    paths=vcat([@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Fresh native and public raster/conformal entry-point contract: ordinary literal rectangles with box-port edges wholly inside versus partly outside each axis end. No geometry variables. No production edits; native failures retain exact logs, no S accuracy claim for rejected circuits.","source_before"=>hashes(),"cases"=>Any[])
    try
        for axis in (:x,:y),(name,lo,hi) in (("inside",0.,.1875),("negative",-.0625,.1875),("positive",.8125,1.0625))
            tag="$(axis)_$(name)";source=joinpath(output,tag*".son");write(source,project_text(axis,lo,hi));row=Dict{String,Any}("case"=>tag,"axis"=>string(axis),"lo"=>lo,"hi"=>hi)
            try
                native=reference_run(find_em(),source;output_dir=joinpath(output,tag),deembedded=false)
                checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false);row["native_status"]="ACCEPT"
            catch err
                row["native_status"]="REJECT";row["native_error"]=sprint(showerror,err)
                row["native_stderr"]=read(joinpath(output,tag,"engine_stderr.log"),String)
            end
            p=read_sonnet_project(source)
            for route in (:raster,:conformal)
                try
                    route==:raster ? sonnet_planar_problem(p;freq=1e9) : sonnet_conformal_layout(p;freq=1e9)
                    row[string(route)*"_status"]="ACCEPT"
                catch err
                    row[string(route)*"_status"]="REJECT";row[string(route)*"_error"]=sprint(showerror,err)
                end
            end
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained plain-port audit: ",output)
    end
    @assert report["source_unchanged"] && length(report["cases"])==6
    @assert all(c->c["native_status"]==(c["lo"]>=0 && c["hi"]<=1 ? "ACCEPT" : "REJECT"),report["cases"])
end
main()
