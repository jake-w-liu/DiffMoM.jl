using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
 output=mktempdir(joinpath(repo,"data/planar_audit");prefix="native_interior_edge_guards_",cleanup=false)
 cp(@__FILE__,joinpath(output,"producer.jl"))
 paths=vcat([@__FILE__],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
 hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
 report=Dict{String,Any}("scope"=>"Independent native internal attachment syntax and annotation controls on two adjacent rectangular sheets; initial parser/lowering and complete native matrices, no new production edits.","source_before"=>hashes(),"cases"=>Any[])
 base=read(joinpath(repo,"test/fixtures/native_box_port_attachment/cases/attachment__baseline/project.son"),String)
 first=findfirst("POR1 BOX",base);last=findfirst("NUM 1",base)
 prefix=base[1:first.start-1];suffix=base[last.start:end]
 suffix=replace(suffix,"NUM 1"=>"NUM 2","1.0 0.375"=>"0.5 0.375","1.0 0.625"=>"0.5 0.625")
 second="0 5 -1 N 2 1 1 100 100 0 0 0 Y\n0.5 0.375\n1.0 0.375\n1.0 0.625\n0.5 0.625\n0.5 0.375\nEND\n"
 suffix=replace(suffix,"END GEO"=>second*"END GEO","native_raw.s2p"=>"native_raw.s1p")
 try
  for (name,kind,x,y) in (("std_isolated","STD",.5,.5),("gap_isolated","GAP",.5,.5),("std_partial","STD",.5,.5))
   case_suffix=occursin("isolated",name) ? replace(suffix,"NUM 2"=>"NUM 1",second=>"") : replace(suffix,second=>replace(second,"0.375"=>"0.4375","0.625"=>"0.5625"))
   text=prefix*"POR1 $(kind)\nPOLY 1 1\n1\n1 50 0 0 0 $(x) $(y)\n"*case_suffix
   source=joinpath(output,name*".son");write(source,text);row=Dict{String,Any}("case"=>name)
   try
    native=reference_run(find_em(),source;output_dir=joinpath(output,name),deembedded=false,nports=1)
    checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s1p");deembedded=false)
    row["native_status"]="ACCEPT"
   catch err
    row["native_status"]="REJECT";row["native_error"]=sprint(showerror,err)
    row["native_stderr"]=read(joinpath(output,name,"engine_stderr.log"),String)
   end
   try
    p=read_sonnet_project(source);problem=sonnet_planar_problem(p;freq=1e9)
    row["raster_status"]="ACCEPT";row["raster_port_walls"]=string.([q.wall for q in problem.ports]);row["raster_port_cells"]=[collect(q.cells) for q in problem.ports]
   catch err
    row["raster_status"]="REJECT";row["raster_error"]=sprint(showerror,err)
   end
   push!(report["cases"],row);println(row);flush(stdout)
  end
 finally
  report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
  open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
  println("Retained native interior attachment audit: ",output)
 end
 @assert report["source_unchanged"] && length(report["cases"])==3
end
main()
