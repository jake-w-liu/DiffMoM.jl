using DiffMoM,SHA,TOML,LinearAlgebra
const repo=normpath(joinpath(@__DIR__,".."));include(joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl"));using .SonnetReference
function main()
 output=mktempdir(joinpath(repo,"data/planar_audit");prefix="native_conformal_clipping_before_",cleanup=false);cp(@__FILE__,joinpath(output,"producer.jl"));mkpath(joinpath(output,"source_before"))
 for name in ("PlanarSonnetConformal.jl","PlanarConformalLayout.jl");cp(joinpath(repo,"src/planar",name),joinpath(output,"source_before",name));end
 p=read_sonnet_project(joinpath(repo,"test/fixtures/native_box_port_attachment/cases/clipped__outside_negative/project.son"))
 base=[(-.0625,.375),(1.,.375),(1.,.625),(-.0625,.625),(-.0625,.375)]
 cases=[(name="wall_$(r)",points=[foldl((q,_)->(1-q[2],q[1]),1:r;init=q) for q in base],edge=1,extra=[],internal=false) for r in 0:3]
 append!(cases,[(name="concave_split",points=[(-.125,.25),(1.,.25),(1.,.375),(-.0625,.375),(-.0625,.625),(1.,.625),(1.,.75),(-.125,.75),(-.125,.25)],edge=1,extra=[],internal=false),
 (name="sloped",points=[(-.0625,.375),(1.,.375),(1.,.625),(-.125,.5),(-.0625,.375)],edge=1,extra=[],internal=false),
 (name="corner",points=[(-.125,-.125),(.5,-.125),(1.,.25),(1.,.5),(-.125,.5),(-.125,-.125)],edge=2,extra=[],internal=false),
 (name="outside_undriven",points=[(0.,.375),(1.,.375),(1.,.625),(0.,.625),(0.,.375)],edge=1,extra=[(-.25,.375),(-.125,.375),(-.125,.625),(-.25,.625),(-.25,.375)],internal=false),
 (name="outside_driven_edge",points=[(-.125,.375),(1.125,.375),(1.125,.625),(-.125,.625),(-.125,.375)],edge=1,extra=[],internal=false),
 (name="internal_shared",points=[(-.125,.375),(.5,.375),(.5,.625),(-.125,.625),(-.125,.375)],edge=1,extra=[(.5,.375),(1.,.375),(1.,.625),(.5,.625),(.5,.375)],internal=true)])
 paths=vcat([@__FILE__],[joinpath(d,f) for (d,_,fs) in walkdir(joinpath(repo,"src")) for f in fs]);hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
 report=Dict{String,Any}("scope"=>"Independent native clipping outcomes/matrices on four walls, concave disconnected intersection, oblique/corner cuts, undriven outside metal, invalid driven edge, and shared internal source. Original conformal outcomes retained before any production edit.","head"=>readchomp(`git -C $repo rev-parse HEAD`),"source_before"=>hashes(),"cases"=>Any[])
 try
  for c in cases
   folder=joinpath(output,c.name);mkpath(folder);v=p.length_scale.*hcat(([q[1],q[2]] for q in c.points)...)
   polys=[SonnetPolygon(:sheet,0,-1,1,v,"","",String[])]
   isempty(c.extra) || push!(polys,SonnetPolygon(:sheet,0,-1,2,p.length_scale.*hcat(([q[1],q[2]] for q in c.extra)...),"","",String[]))
   i=c.edge+1;j=mod1(i+1,length(c.points)-1);a,b=c.points[i],c.points[j]
   port=SonnetPortSpec(:box,1,c.edge,1,["1","50","0","0","0",string((a[1]+b[1])/2),string((a[2]+b[2])/2)],SonnetRecord[])
   literal=SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,p.box,p.layers,p.metals,p.top,p.bottom,polys,[port],p.variables,p.components,p.sweeps,p.records)
   source=joinpath(folder,"project.son");write_native_fixture(source,literal;frequency_native=1.,high_precision=true)
   row=Dict{String,Any}("case"=>c.name,"source_sha256"=>bytes2hex(sha256(read(source))))
   try
    native=reference_run(find_em(),source;output_dir=joinpath(folder,"native"),deembedded=false,nports=1)
    s=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
    row["native_status"]="ACCEPT";row["native_s_real"]=real.(vec(s));row["native_s_imag"]=imag.(vec(s))
   catch err
    row["native_status"]="REJECT";row["native_error"]=sprint(showerror,err);row["native_stderr"]=read(joinpath(folder,"native/engine_stderr.log"),String)
   end
   try
    sonnet_conformal_layout(read_sonnet_project(source);freq=1e9,edge_size=.125e-3,interior_size=.25e-3,edge_band=.05e-3)
    row["conformal_status"]="ACCEPT"
   catch err;row["conformal_status"]="REJECT";row["conformal_error"]=sprint(showerror,err)
   end
   push!(report["cases"],row);println(row);flush(stdout)
  end
 finally
  report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
  open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end;println("Retained clipping baseline: ",output)
 end
 @assert report["source_unchanged"] && length(report["cases"])==10
end
main()
