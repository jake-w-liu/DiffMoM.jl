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
 output=mktempdir(joinpath(repo,"data/planar_audit");prefix="native_unported_wall_contact_v2_",cleanup=false)
 cp(@__FILE__,joinpath(output,"producer.jl"))
 paths=vcat([@__FILE__],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
 hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
 report=Dict{String,Any}("scope"=>"Independent native one-port strip with an undriven west edge at and near a native grid wall; complete native and public S.","source_before"=>hashes(),"cases"=>Any[])
 base=replace(project_text(:y,.375,.625),"POR1 BOX\nPOLY 1 1\n3\n1 50 0 0 0 0.0 0.5\n"=>"","2 50 0 0 0 1.0 0.5"=>"1 50 0 0 0 1.0 0.5","native_raw.s2p"=>"native_raw.s1p")
 try
  for (name,x) in (("exact",0.),("inside",.001),("near_limit",.0156281234375),("outside_limit",.02))
   source=joinpath(output,name*".son");text=replace(base,"0.0 0.375"=>"$(x) 0.375","0.0 0.625"=>"$(x) 0.625");write(source,text)
   native=reference_run(find_em(),source;output_dir=joinpath(output,name),deembedded=false,nports=1)
   ns=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s1p");deembedded=false).s)
   p=read_sonnet_project(source);result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
   row=Dict("case"=>name,"west_x_mm"=>x,"native_s_real"=>real.(vec(ns)),"native_s_imag"=>imag.(vec(ns)),
    "public_s_real"=>real.(vec(result.s)),"public_s_imag"=>imag.(vec(result.s)),"full_s_error"=>maximum(abs,result.s-ns),
    "west_connections"=>count(only(result.raw.problem.sheets).connect_west))
   push!(report["cases"],row);println(row);flush(stdout)
  end
 finally
  report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
  open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
  println("Retained unported wall-contact audit: ",output)
 end
 @assert report["source_unchanged"] && length(report["cases"])==4
end
main()
