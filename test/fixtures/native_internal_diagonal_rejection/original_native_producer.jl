using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
 output=mktempdir(joinpath(repo,"data/planar_audit");prefix="native_internal_diagonal_contract_",cleanup=false)
 cp(@__FILE__,joinpath(output,"producer.jl"))
 paths=vcat([@__FILE__],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
 hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
 report=Dict{String,Any}("scope"=>"Independent native two-port internal attachment orientation: same seam referenced from left/right polygon along a diagonal seam, plus a west wall port; complete genuine conformal matrices.","source_before"=>hashes(),"cases"=>Any[])
 base=read(joinpath(repo,"test/fixtures/native_internal_port_attachment/cases/contract__std_baseline/project.son"),String)
 second="POR1 BOX\nPOLY 1 1\n3\n2 50 0 0 0 0 .5\n"
 base=replace(base,"NUM 2"=>second*"NUM 2","native_raw.s1p"=>"native_raw.s2p",
    "0.0 0.375\n0.5 0.375\n0.5 0.625\n0.0 0.625"=>"0.0 0.375\n0.25 0.375\n0.75 0.625\n0.0 0.625",
    "0.5 0.375\n1.0 0.375\n1.0 0.625\n0.5 0.625\n0.5 0.375"=>"0.25 0.375\n1.0 0.375\n1.0 0.625\n0.75 0.625\n0.25 0.375")
 try
  for name in ("std_left","std_right","box_left","gap_left")
   text=name=="std_right" ? replace(base,"POR1 STD\nPOLY 1 1\n1\n"=>"POR1 STD\nPOLY 2 1\n3\n") : name=="box_left" ? replace(base,"POR1 STD"=>"POR1 BOX") : name=="gap_left" ? replace(base,"POR1 STD"=>"POR1 GAP") : base
   source=joinpath(output,name*".son");write(source,text)
   row=Dict{String,Any}("case"=>name)
   try
    native=reference_run(find_em(),source;output_dir=joinpath(output,name),deembedded=false,nports=2)
    checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
    row["native_status"]="ACCEPT"
   catch err
    row["native_status"]="REJECT";row["native_error"]=sprint(showerror,err)
    row["native_stderr"]=read(joinpath(output,name,"engine_stderr.log"),String)
   end
   p=read_sonnet_project(source)
   try
    model=sonnet_conformal_layout(p;edge_size=.125e-3,interior_size=.25e-3,edge_band=.05e-3)
    row["conformal_status"]="ACCEPT";row["conformal_walls"]=string.([q.wall for q in model.layout.problem.ports])
   catch err
    row["conformal_status"]="REJECT";row["conformal_error"]=sprint(showerror,err)
   end
   push!(report["cases"],row);println(row);flush(stdout)
  end
 finally
  report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
  open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
  println("Retained internal orientation audit: ",output)
 end
 @assert report["source_unchanged"] && length(report["cases"])==4
end
main()
