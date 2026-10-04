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
function attachment_main()
 output=mktempdir(joinpath(repo,"data/planar_audit");prefix="plain_box_port_attachment_contract_",cleanup=false)
 cp(@__FILE__,joinpath(output,"producer.jl"))
 paths=vcat([@__FILE__],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
 hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
 report=Dict{String,Any}("scope"=>"Independent native box-port attachment contract: referenced logical edge versus stored display position, unchanged centered strip geometry.","source_before"=>hashes(),"cases"=>Any[])
 baseline=project_text(:y,.375,.625)
 variants=(("baseline",baseline),("off_edge_position",replace(baseline,"1 50 0 0 0 0.0 0.5"=>"1 50 0 0 0 0.0 0.9")),
  ("opposite_wall_position",replace(baseline,"1 50 0 0 0 0.0 0.5"=>"1 50 0 0 0 1.0 0.5")),
  ("interior_attached_edge",replace(baseline,"POLY 1 1\n3\n"=>"POLY 1 1\n0\n")),
  ("duplicate_closing_edge",replace(baseline,"POLY 1 1\n3\n"=>"POLY 1 1\n4\n")))
 try
  for (tag,text) in variants
   source=joinpath(output,tag*".son");write(source,text);row=Dict{String,Any}("case"=>tag)
   try
    native=reference_run(find_em(),source;output_dir=joinpath(output,tag),deembedded=false)
    checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
    row["native_status"]="ACCEPT"
   catch err
    row["native_status"]="REJECT";row["native_error"]=sprint(showerror,err)
    row["native_stderr"]=read(joinpath(output,tag,"engine_stderr.log"),String)
   end
   try
    problem=sonnet_planar_problem(read_sonnet_project(source);freq=1e9)
    row["raster_status"]="ACCEPT";row["raster_port_walls"]=string.([p.wall for p in problem.ports])
   catch err
    row["raster_status"]="REJECT";row["raster_error"]=sprint(showerror,err)
   end
   push!(report["cases"],row);println(row);flush(stdout)
  end
 finally
  report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
  open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
  println("Retained attachment contract audit: ",output)
 end
 @assert report["source_unchanged"] && length(report["cases"])==5
end
attachment_main()
