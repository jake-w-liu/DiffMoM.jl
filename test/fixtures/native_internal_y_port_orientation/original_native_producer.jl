using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
 output=mktempdir(joinpath(repo,"data/planar_audit");prefix="native_internal_y_port_orientation_",cleanup=false)
 cp(@__FILE__,joinpath(output,"producer.jl"))
 paths=vcat([@__FILE__],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs])
 hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
 report=Dict{String,Any}("scope"=>"Independent native two-port internal attachment orientation: same seam referenced from left/right polygon, plus a west wall port; complete public raster and genuine conformal matrices.","source_before"=>hashes(),"cases"=>Any[])
 base=read(joinpath(repo,"test/fixtures/native_internal_port_attachment/cases/contract__std_baseline/project.son"),String)
 second="POR1 BOX\nPOLY 1 1\n3\n2 50 0 0 0 0 .5\n"
 base=replace(base,"NUM 2"=>second*"NUM 2","native_raw.s1p"=>"native_raw.s2p")
 try
  for name in ("left_reference","right_reference")
   text=name=="left_reference" ? base : replace(base,"POR1 STD\nPOLY 1 1\n1\n"=>"POR1 STD\nPOLY 2 1\n3\n")
   # Swap only polygon and annotation coordinates; the BOX is square.
   rows=split(text,'\n');inpolygon=false;remaining=0
   for i in eachindex(rows)
    fields=split(rows[i])
    if length(fields)==13 && fields[4]=="N"
     remaining=parse(Int,fields[2]);inpolygon=true
    elseif inpolygon && remaining>0 && length(fields)==2
     rows[i]=fields[2]*" "*fields[1];remaining-=1
     remaining==0 && (inpolygon=false)
    elseif length(fields)==7 && fields[2]=="50"
     fields[6],fields[7]=fields[7],fields[6];rows[i]=join(fields,' ')
    end
   end
   text=join(rows,'\n')
   source=joinpath(output,name*".son");write(source,text)
   native=reference_run(find_em(),source;output_dir=joinpath(output,name),deembedded=false,nports=2)
   ns=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
   p=read_sonnet_project(source);r=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
   c=solve_sonnet_conformal(p,1e9;raw=true,edge_size=.125e-3,interior_size=.25e-3,edge_band=.05e-3,mx=128,my=128)
   row=Dict("case"=>name,"native_status"=>"ACCEPT","native_s_real"=>real.(vec(ns)),"native_s_imag"=>imag.(vec(ns)),
    "raster_full_s_error"=>maximum(abs,r.s-ns),"raster_s_real"=>real.(vec(r.s)),"raster_s_imag"=>imag.(vec(r.s)),
    "conformal_full_s_error"=>maximum(abs,c.s-ns),"conformal_s_real"=>real.(vec(c.s)),"conformal_s_imag"=>imag.(vec(c.s)))
   push!(report["cases"],row);println(row);flush(stdout)
  end
 finally
  report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
  open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
  println("Retained internal orientation audit: ",output)
 end
 @assert report["source_unchanged"] && length(report["cases"])==2
end
main()
