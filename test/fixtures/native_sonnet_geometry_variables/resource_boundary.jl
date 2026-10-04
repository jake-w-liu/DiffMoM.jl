using DiffMoM,TOML,SHA
function bench_membership(n)
 a=[(1,i) for i in 1:n];b=[(1,i) for i in n+1:2n];bs=Set(b)
 any(point->point in bs,a)
 old=@elapsed old_result=any(point->point in b,a)
 new=@elapsed new_result=any(point->point in bs,a)
 return Dict("per_set_points"=>n,"old_seconds"=>old,"set_seconds"=>new,"same_false_result"=>!old_result && !new_result)
end
p=read_sonnet_project("test/fixtures/native_sonnet_geometry_variables/accepted/single_ydir_sym_positive_parameter.son")
function stress(p,n)
 rows=SonnetRecord[SonnetRecord(1,["GEO"]),SonnetRecord(2,["VALVAR","Width","LNG","0.375"]),
   SonnetRecord(3,["GEOVAR","Width","SYM","YDIR","1","NSCD"]),
   SonnetRecord(4,["POS","0","0"]),SonnetRecord(5,["NOM","0.25"]),
   SonnetRecord(6,["REF1","POLY","1","1"]),SonnetRecord(7,["0"]),
   SonnetRecord(8,["REF2","POLY","1","1"]),SonnetRecord(9,[string(n)])]
 for (tag,start) in (("PS1",0),("PS2",n))
   push!(rows,SonnetRecord(length(rows)+1,[tag,"1"]),SonnetRecord(length(rows)+2,["POLY","1",string(n)]))
   for i in start:start+n-1;push!(rows,SonnetRecord(length(rows)+1,[string(i)]));end
   push!(rows,SonnetRecord(length(rows)+1,["END"]))
 end
 push!(rows,SonnetRecord(length(rows)+1,["END"]),SonnetRecord(length(rows)+2,["END","GEO"]))
 q=only(p.polygons);v=zeros(2,2n+1);v[1,:]=collect(0.:2n);v[2,:].=1.
 polygon=SonnetPolygon(q.kind,q.level,q.material,q.id,v,q.target,q.technology,q.flags)
 project=SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,p.box,p.layers,p.metals,p.top,p.bottom,
   [polygon],p.ports,p.variables,p.components,p.sweeps,rows)
 limit=2n+2
 elapsed=@elapsed result=DiffMoM._sonnet_geovar_parameters(project,1,limit)
 exceeded=false
 try;DiffMoM._sonnet_geovar_parameters(project,1,limit-1);catch e;exceeded=e isa ArgumentError;end
 return Dict("point_budget"=>limit,"emitted_point_count"=>sum(length,only(result).points),
   "seconds"=>elapsed,"implicit_reference_budget_rejection"=>exceeded)
end
bench_membership(100);stress(p,50)
report=Dict("scope"=>"adjustable-set parser and overlap check measured independently of polygon normalization/EM solver; single Julia process, no universal performance claim",
 "membership"=>[bench_membership(5000),bench_membership(10000)],"parser"=>[stress(p,4999),stress(p,49999)])
open("validation/planar_audit/fresh_geovar_20261004/resource_boundary.toml","w") do io;TOML.print(io,report);end
TOML.print(stdout,report)
