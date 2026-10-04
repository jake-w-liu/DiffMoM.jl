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
function snap_main()
    output=mktempdir(joinpath(repo,"data/planar_audit");prefix="plain_box_port_native_boundary_allowance_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    paths=vcat([@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")],
        [joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Independent native port-edge boundary versus declared half-cell grid; no tolerance or quantization law presumed.",
        "source_before"=>hashes(),"cases"=>Any[])
    try
        for unit in ("MM","IN"),halfcells in (32,64,128),axis in (:x,:y),side in (:low,:high),fraction in (1.0001999,1.0002,1.0002001)
            unit=="IN" && halfcells!=64 && continue
            delta=fraction/halfcells;tag="$(unit)_h$(halfcells)_$(axis)_$(side)_$(fraction)"
            source=joinpath(output,tag*".son")
            text=replace(project_text(axis,side==:low ? -delta : .8125,side==:low ? .1875 : 1+delta),"BOX 1 1 1 64 64 100 0"=>"BOX 1 1 1 $(halfcells) $(halfcells) 100 0")
            text=replace(text,"LNG MM"=>"LNG $(unit)")
            write(source,text);row=Dict{String,Any}("case"=>tag,"unit"=>unit,"axis"=>string(axis),"side"=>string(side),"halfcells"=>halfcells,"fraction"=>fraction,"delta"=>delta)
            try
                native=reference_run(find_em(),source;output_dir=joinpath(output,tag),deembedded=false)
                checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
                row["native_status"]="ACCEPT"
            catch err
                row["native_status"]="REJECT";row["native_error"]=sprint(showerror,err)
                row["native_stderr"]=read(joinpath(output,tag,"engine_stderr.log"),String)
            end
            try
                sonnet_planar_problem(read_sonnet_project(source);freq=1e9);row["raster_before_status"]="ACCEPT"
            catch err
                row["raster_before_status"]="REJECT";row["raster_before_error"]=sprint(showerror,err)
            end
            push!(report["cases"],row);println(tag," ",row["native_status"]);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained native snap audit: ",output)
    end
    @assert report["source_unchanged"] && length(report["cases"])==48
end
snap_main()
