using DiffMoM,TOML,SHA,LinearAlgebra
const repo=normpath(joinpath(@__DIR__,".."));const fixture=joinpath(repo,"test/fixtures/native_box_port_attachment")
function main()
 output=mktempdir(joinpath(repo,"data/planar_audit");prefix="conformal_wall_ownership_before_",cleanup=false);cp(@__FILE__,joinpath(output,"producer.jl"))
 names=["variants__normal_inside","variants__normal_near_limit","unported__inside","unported__near_limit","unported__outside_limit"]
 paths=vcat([@__FILE__],[joinpath(fixture,"cases",name,"project.son") for name in names],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs]);hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
 report=Dict{String,Any}("scope"=>"Current conformal backend handling of independently retained native half-cell wall attachments and undriven contacts; prior fixture metadata/hash manifests contain independent native acceptance/matrices. Before any proposed edit.","source_before"=>hashes(),"cases"=>Any[])
 try
  for name in names
   p=read_sonnet_project(joinpath(fixture,"cases",name,"project.son"));row=Dict{String,Any}("case"=>name)
   try
    model=sonnet_conformal_layout(p;freq=1e9,edge_size=.125e-3,interior_size=.25e-3,edge_band=.05e-3)
    row["conformal_status"]="ACCEPT";row["port_walls"]=[string(q.wall) for q in model.layout.problem.ports]
    if startswith(name,"unported")
     result=solve_sonnet_conformal(p,1e9;raw=true,edge_size=.125e-3,interior_size=.25e-3,edge_band=.05e-3,mx=128,my=128)
     expected=only(planar_read_touchstone(joinpath(fixture,"cases",name,"native/native_raw.s1p")).s)
     row["full_complex_s_error"]=maximum(abs,result.s-expected)
    end
   catch err
    row["conformal_status"]="REJECT";row["error"]=sprint(showerror,err)
   end
   push!(report["cases"],row);println(row);flush(stdout)
  end
 finally
  report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
  open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
  println("Retained conformal wall ownership before: ",output)
 end
 @assert report["source_unchanged"] && length(report["cases"])==length(names)
end
main()
