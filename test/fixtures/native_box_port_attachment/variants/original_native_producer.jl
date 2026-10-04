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
 output=mktempdir(joinpath(repo,"data/planar_audit");prefix="port_attachment_variants_v3_",cleanup=false)
 cp(@__FILE__,joinpath(output,"producer.jl"))
 paths=vcat([@__FILE__],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
 hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
 report=Dict{String,Any}("scope"=>"Independent native edge spans, wall normal offsets, active ANC annotations, and native ReadWrite annotation updates; raw normalizer exports remain ignored and only selected port fields are recorded.","source_before"=>hashes(),"cases"=>Any[])
 plain=project_text(:y,.375,.625)
 split_project=replace(plain,"0 5 -1 N"=>"0 6 -1 N","0.0 0.625\n0.0 0.375"=>"0.0 0.625\n0.0 0.5\n0.0 0.375")
 active=replace(read(joinpath(repo,"test/fixtures/native_sonnet_geometry_variables/accepted/single_ydir_anc_positive_parameter.son"),String),r"OPTIONS\r?\n"=>"OPTIONS -j\n")
 variants=Pair{String,String}["plain_baseline"=>plain,
  "collinear_upper"=>replace(split_project,"1 50 0 0 0 0.0 0.5"=>"1 50 0 0 0 0.0 0.5625"),
  "collinear_lower"=>replace(split_project,"POLY 1 1\n3\n"=>"POLY 1 1\n4\n","1 50 0 0 0 0.0 0.5"=>"1 50 0 0 0 0.0 0.4375"),
  "normal_inside"=>replace(plain,"0.0 0.375"=>"0.001 0.375","0.0 0.625"=>"0.001 0.625"),
  "normal_near_limit"=>replace(plain,"0.0 0.375"=>"0.0156281234375 0.375","0.0 0.625"=>"0.0156281234375 0.625"),
  "normal_outside_limit"=>replace(plain,"0.0 0.375"=>"0.0156281265625 0.375","0.0 0.625"=>"0.0156281265625 0.625"),
  "active_baseline"=>active,
  "active_nonmid_position"=>replace(active,"1 50 0 0 0 0 .5"=>"1 50 0 0 0 0 .4"),
  "active_offedge_position"=>replace(active,"1 50 0 0 0 0 .5"=>"1 50 0 0 0 0 .9"),
  "active_opposite_wall_position"=>replace(active,"1 50 0 0 0 0 .5"=>"1 50 0 0 0 1 .5")]
 try
  for (name,text) in variants
   source=joinpath(output,name*".son");write(source,text);row=Dict{String,Any}("case"=>name)
   try
    native=reference_run(find_em(),source;output_dir=joinpath(output,name),deembedded=false)
    checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
    row["native_status"]="ACCEPT"
   catch err
    row["native_status"]="REJECT";row["native_error"]=sprint(showerror,err)
    row["native_stderr"]=read(joinpath(output,name,"engine_stderr.log"),String)
   end
   p=read_sonnet_project(source)
   try
    effective=DiffMoM._sonnet_geometry_project(p,1e9);problem=sonnet_planar_problem(p;freq=1e9)
    row["raster_status"]="ACCEPT";row["raster_port_walls"]=string.([q.wall for q in problem.ports]);row["raster_port_cells"]=[collect(q.cells) for q in problem.ports]
    row["effective_port_fields"]=[q.values[6:7] for q in effective.ports]
   catch err
    row["raster_status"]="REJECT";row["raster_error"]=sprint(showerror,err)
   end
   if row["native_status"]=="ACCEPT"
    normalized=joinpath(output,name*"_normalized.son")
    command=`$(joinpath(dirname(find_em()),"soncmd.exe")) -ReadWrite $source $normalized`
    process=run(pipeline(ignorestatus(command),stdout=joinpath(output,name*"_normalizer_stdout.log"),stderr=joinpath(output,name*"_normalizer_stderr.log")))
    row["normalizer_success"]=success(process)
    if success(process)
     lines=readlines(normalized);ports=Any[]
     for i in eachindex(lines)
      startswith(lines[i],"POR1") || continue
      j=findnext(line->startswith(line,"POLY"),lines,i+1)
      fields=split(lines[j+2]);push!(ports,fields[6:7])
     end
     row["normalized_port_fields"]=ports
     normalized_project=read_sonnet_project(normalized)
     row["normalized_vertices_si"]=[collect(col) for col in eachcol(only(normalized_project.polygons).vertices)]
    end
   end
   push!(report["cases"],row);println(row);flush(stdout)
  end
 finally
  report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
  open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
  println("Retained attachment variants: ",output)
 end
 @assert report["source_unchanged"] && length(report["cases"])==10
end
main()
