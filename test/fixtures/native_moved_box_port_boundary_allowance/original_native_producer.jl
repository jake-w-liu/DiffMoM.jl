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
function moved_main()
 output=mktempdir(joinpath(repo,"data/planar_audit");prefix="moved_box_port_native_allowance_",cleanup=false)
 cp(@__FILE__,joinpath(output,"producer.jl"));rows=Any[]
 paths=vcat([@__FILE__],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
 hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
 report=Dict{String,Any}("source_before"=>hashes(),"cases"=>rows)
 base=read(joinpath(repo,"test/fixtures/native_sonnet_geometry_variables/accepted/single_ydir_anc_positive_parameter.son"),String)
 try
  for fraction in (1.0001999,1.0002,1.0002001)
   delta=fraction/64;target=.625+delta;tag="fraction$(fraction)"
   text=replace(base,"VALVAR Width LNG 0.375"=>"VALVAR Width LNG $(target)","OPTIONS\n"=>"OPTIONS -j\n")
   row=Dict{String,Any}("case"=>tag,"fraction"=>fraction,"target"=>target)
   for (suffix,contents) in (("parameter",text),("literal",project_text(:y,.375,1+delta)))
    source=joinpath(output,tag*"_"*suffix*".son");write(source,contents)
    try
     native=reference_run(find_em(),source;output_dir=joinpath(output,tag,suffix),deembedded=false)
     checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
     row[suffix*"_native_status"]="ACCEPT"
    catch err
     row[suffix*"_native_status"]="REJECT";row[suffix*"_native_stderr"]=read(joinpath(output,tag,suffix,"engine_stderr.log"),String)
    end
   end
   p=read_sonnet_project(joinpath(output,tag*"_parameter.son"))
   try
    resolved=DiffMoM._sonnet_geometry_project(p,1e9);sonnet_planar_problem(p;freq=1e9)
    row["geometry_status"]="ACCEPT";row["upper_y"]=maximum(only(resolved.polygons).vertices[2,:])
   catch err
    row["geometry_status"]="REJECT";row["geometry_error"]=sprint(showerror,err)
   end
   push!(rows,row);println(row);flush(stdout)
  end
 finally
  report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
  open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
  println("Retained moved allowance audit: ",output)
 end
 @assert report["source_unchanged"] && length(rows)==3
end
moved_main()
